"""
train.py - PEFT fine-tuning of DistilBERT for the Barclays complaint
routing task (4 classes).
Supports LoRA and QLoRA via --peft_method argument.

Task: route a customer complaint into one of 4 Barclays categories:
  0 = fraud and security
  1 = billing and charges
  2 = account access
  3 = general enquiry

Dataset: a compact 4-class Barclays complaint-routing dataset defined
in this file (the course spiral normally feeds a labelled dataset from
Topic 3 via S3; the local set keeps this script self-contained).
Instance: ml.g4dn.xlarge (NVIDIA T4, 16GB VRAM).
SageMaker toolkit auto-installs requirements.txt before this script runs.

Arguments:
  --peft_method   lora | qlora  (default: lora)
  --lora_r        LoRA rank (default: 8)
  --lora_alpha    LoRA alpha (default: 16)
  --epochs        training epochs (default: 3)
  --batch_size    per-device batch size (default: 16)
  --lr            learning rate (default: 2e-4)
  --model_dir     SageMaker model artifact dir
  --output_dir    SageMaker output data dir
"""

import argparse
import os
import json
import numpy as np

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_method", type=str, default="lora",
                        choices=["lora", "qlora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    return parser.parse_args()


def compute_metrics(eval_pred):
    # Inline numpy accuracy. Do NOT use the evaluate library (incompatible with datasets 4.x).
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


LABEL_MAP = {
    0: "fraud and security",
    1: "billing and charges",
    2: "account access",
    3: "general enquiry",
}
NUM_LABELS = len(LABEL_MAP)


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
    return {"sentence": sentences, "label": labels}


def load_and_tokenise(tokenizer, max_length=128):
    """Build the 4-class Barclays routing dataset and tokenise it."""
    full_ds = Dataset.from_dict(barclays_routing_examples())
    # The local dataset is small; build an 80/20 train/eval split.
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    def tokenise(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = train_ds.map(tokenise, batched=True, remove_columns=["sentence"])
    eval_ds = eval_ds.map(tokenise, batched=True, remove_columns=["sentence"])

    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")

    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    return train_ds, eval_ds


def build_model(peft_method, lora_r, lora_alpha, num_labels=NUM_LABELS):
    """Build DistilBERT with LoRA or QLoRA adapters for 4-class routing."""
    model_name = "distilbert-base-uncased"
    id2label = LABEL_MAP
    label2id = {v: k for k, v in LABEL_MAP.items()}

    if peft_method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier", "pre_classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"PEFT method : {args.peft_method}")
    print(f"LoRA rank   : {args.lora_r}")
    print(f"LoRA alpha  : {args.lora_alpha}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds, eval_ds = load_and_tokenise(tokenizer)

    model = build_model(args.peft_method, args.lora_r, args.lora_alpha)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        # eval_strategy not evaluation_strategy (transformers 4.41+).
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save PEFT adapters (not the full model, much smaller).
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    metrics = {
        "peft_method": args.peft_method,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "eval_accuracy": eval_results.get("eval_accuracy", 0.0),
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Artifacts saved to", args.model_dir)


if __name__ == "__main__":
    main()
