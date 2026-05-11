"""
train.py -- QAT + LoRA complaint classifier for SageMaker GPU job.

Base model : distilbert-base-uncased (66M params)
Task       : 5-class complaint classification (Barclays customer service)
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
from datasets import load_dataset
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


CATEGORY_MAP = {
    "card_arrival":           0,
    "card_linking":           0,
    "card_payment_fee_charged": 0,
    "card_swallowed":         0,
    "declined_card_payment":  0,
    "card_not_working":       0,
    "wrong_amount_of_cash_received": 1,
    "transaction_charged_twice": 1,
    "transfer_not_received_by_recipient": 1,
    "beneficiary_not_allowed": 1,
    "balance_not_updated_after_bank_transfer": 1,
    "failed_transfer":        1,
    "exchange_rate":          2,
    "transfer_fee_charged":   2,
    "getting_spare_card":     3,
    "getting_virtual_card":   3,
    "lost_or_stolen_card":    3,
    "lost_or_stolen_phone":   3,
    "pending_card_payment":   3,
    "pending_cash_withdrawal": 3,
}
DEFAULT_LABEL = 4

LABEL_NAMES = [
    "Card and Account Issues",
    "Transaction Disputes",
    "FX and International",
    "General Queries",
    "Other",
]


def remap_label(example):
    """Map banking77 fine-grained intents to 5 Barclays complaint categories."""
    intent = example.get("intent", "")
    example["labels"] = CATEGORY_MAP.get(intent, DEFAULT_LABEL)
    return example


def load_and_prepare_dataset(tokenizer, max_length):
    """Load banking77, remap to 5 classes, tokenize."""
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    dataset = dataset.map(remap_label)
    dataset = dataset.rename_column("text", "sentence")

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
        num_labels=5,
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
