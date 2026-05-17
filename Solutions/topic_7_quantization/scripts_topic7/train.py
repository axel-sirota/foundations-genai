"""
train.py -- QAT + LoRA complaint classifier for SageMaker GPU job.

Base model : distilbert-base-uncased (66M params)
Task       : 4-class Barclays complaint routing
             0 = fraud and security
             1 = billing and charges
             2 = account access
             3 = general enquiry
Techniques : Quantization-Aware Training (INT8, fbgemm backend) + PEFT LoRA adapters
Instance   : ml.g4dn.xlarge (NVIDIA T4, 16 GB VRAM)
Container  : HuggingFace estimator, transformers 4.56.2, pytorch 2.8.0, py312

SageMaker toolkit auto-installs requirements.txt before running this script.
Hyperparameters are passed as CLI args by the HuggingFace estimator.

Key rules encoded here:
  - eval_strategy='epoch' (NOT evaluation_strategy, removed in transformers 4.41+)
  - NO evaluate library, use inline numpy for metrics
  - Save to /opt/ml/model/ so SageMaker can package artifacts
  - QAT observers only on Linear layers; Embedding layers must opt out
  - QAT prepare/convert cycle: prepare on CPU, train on GPU, convert back on CPU
"""

import argparse
import os
import numpy as np
import torch
import torch.ao.quantization
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",               type=int,   default=3)
    parser.add_argument("--batch_size",           type=int,   default=16)
    parser.add_argument("--lr",                   type=float, default=2e-4)
    parser.add_argument("--quantization_backend", type=str,   default="fbgemm",
                        help="fbgemm (x86 CPU) or qnnpack (ARM/mobile)")
    parser.add_argument("--lora_r",               type=int,   default=8)
    parser.add_argument("--lora_alpha",           type=int,   default=16)
    parser.add_argument("--max_length",           type=int,   default=128)
    parser.add_argument("--warmup_ratio",         type=float, default=0.1)
    parser.add_argument("--model_dir",
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_data_dir",
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    return parser.parse_args()


LABEL_MAP = {
    0: "fraud and security",
    1: "billing and charges",
    2: "account access",
    3: "general enquiry",
}
NUM_LABELS = len(LABEL_MAP)

LABEL_NAMES = [LABEL_MAP[i] for i in range(NUM_LABELS)]


def barclays_routing_examples():
    """
    A compact 4-class Barclays complaint-routing dataset.

    Categories:
      0 = fraud and security
      1 = billing and charges
      2 = account access
      3 = general enquiry
    """
    fraud = [
        "There is an unauthorised payment on my account I did not make.",
        "Someone used my card details to buy things abroad.",
        "I think my account has been hacked, please block it.",
        "A suspicious transfer left my account overnight.",
        "I received a phishing text pretending to be Barclays.",
        "My contactless card was cloned and used at a shop.",
        "Fraudulent charges keep appearing every few days.",
        "My identity may have been stolen and used for a loan.",
        "An unknown direct debit was set up without my consent.",
        "I want to report a scam that took money from my account.",
        "My card was used while it was still in my wallet.",
        "Please investigate these payments, they are not mine.",
        "Someone tried to log in to my account from another country.",
        "I got an alert about a payment I never authorised.",
    ]
    billing = [
        "You charged me an overdraft fee I did not expect.",
        "My monthly account fee went up without warning.",
        "I was billed twice for the same standing order.",
        "There is an interest charge on my statement I do not understand.",
        "Why was I charged a fee for using my card overseas?",
        "The late payment charge is unfair, my payment was on time.",
        "I see a duplicate charge for the same transaction.",
        "My statement shows a fee that should have been waived.",
        "I was charged for a service I never signed up for.",
        "The exchange rate fee on my purchase looks wrong.",
        "Please explain the charges added to my account this month.",
        "I want a refund for an incorrect billing amount.",
        "My credit card minimum payment was miscalculated.",
        "There is an unexpected annual fee on my account.",
    ]
    access = [
        "I cannot log in to the mobile banking app.",
        "My online banking password reset link does not work.",
        "I am locked out of my account after too many attempts.",
        "The app does not recognise my fingerprint anymore.",
        "I did not receive the one-time passcode to sign in.",
        "My account is showing as suspended and I cannot access it.",
        "I forgot my memorable word and cannot get back in.",
        "The website keeps logging me out immediately.",
        "I need to reset my online banking credentials.",
        "My card reader is not letting me authorise a login.",
        "I cannot access my statements online.",
        "The app crashes every time I try to open it.",
        "I changed my phone and cannot set up the app again.",
        "My digital banking access has stopped working.",
    ]
    general = [
        "What are your branch opening hours this weekend?",
        "How do I order a new chequebook?",
        "Can you tell me which documents I need to open an account?",
        "I would like to update my mailing address.",
        "How long does an international transfer usually take?",
        "What is the interest rate on your savings accounts?",
        "Can I book an appointment with a mortgage adviser?",
        "How do I add a second cardholder to my account?",
        "Where can I find the nearest cash machine?",
        "How do I close an account I no longer use?",
        "What is the daily withdrawal limit on my debit card?",
        "Can you explain how to set up a standing order?",
        "I want to know more about your travel insurance options.",
        "How do I register for paperless statements?",
    ]
    sentences, labels = [], []
    for label, group in enumerate((fraud, billing, access, general)):
        for text in group:
            sentences.append(text)
            labels.append(label)
    return {"sentence": sentences, "labels": labels}


def load_and_prepare_dataset(tokenizer, max_length):
    """Build the 4-class Barclays routing dataset and tokenize it."""
    from datasets import Dataset, DatasetDict

    full_ds = Dataset.from_dict(barclays_routing_examples())
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({"train": split["train"], "test": split["test"]})

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names
         if c not in ["input_ids", "attention_mask", "labels"]]
    )
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def insert_qat_observers(model, backend="fbgemm"):
    """
    Insert fake-quantization observers into all Linear layers for QAT.

    Rules:
      - Embedding and LayerNorm layers MUST opt out (qconfig=None)
      - QAT prepare/convert must happen on CPU; training can use GPU after prepare
      - backend 'fbgemm' targets x86 CPUs (SageMaker inference endpoints)
    """
    torch.backends.quantized.engine = backend
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            module.qconfig = None

    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model):
    """Convert fake-quantization ops to real INT8 quantized ops on CPU."""
    model.eval()
    quantized = torch.ao.quantization.convert(model, inplace=False)
    return quantized


def apply_lora(model, lora_r, lora_alpha):
    """Wrap the model with PEFT LoRA adapters targeting attention projections."""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_lin", "k_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args: {args}")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )

    model = apply_lora(model, args.lora_r, args.lora_alpha)

    print("Inserting QAT observers...")
    model = insert_qat_observers(model, backend=args.quantization_backend)

    model.to(device)

    print("Loading dataset...")
    tokenized = load_and_prepare_dataset(tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting QAT training...")
    trainer.train()

    print("Converting to INT8 quantized model (must be on CPU)...")
    model.to("cpu")
    quantized_model = convert_to_quantized(model)
    print("Conversion complete.")

    os.makedirs(args.model_dir, exist_ok=True)
    tokenizer.save_pretrained(args.model_dir)
    torch.save(quantized_model.state_dict(), os.path.join(args.model_dir, "quantized_model.pt"))
    model.config.save_pretrained(args.model_dir)
    print(f"Quantized model saved to {args.model_dir}")

    results = trainer.evaluate()
    print(f"Final eval results: {results}")


if __name__ == "__main__":
    main()
