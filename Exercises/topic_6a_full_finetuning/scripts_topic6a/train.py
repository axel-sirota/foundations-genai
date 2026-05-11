"""
train.py - Full fine-tuning of distilbert-base-uncased for complaint sentiment
classification. Runs as a SageMaker HuggingFace estimator GPU job on ml.g4dn.xlarge.

Task: binary classification (0 = negative/complaint, 1 = positive/resolved).
Dataset: synthetic Barclays complaint data (generated inline).
Target: approximately 15 min on ml.g4dn.xlarge (NVIDIA T4 16 GB).

SageMaker toolkit auto-installs requirements.txt before running this script.
Hyperparameters passed as CLI args by the HuggingFace estimator.

Hard rules applied (from CORE_TECHNOLOGIES_AND_DECISIONS.md):
  - eval_strategy="epoch" (NOT evaluation_strategy, removed in 4.41+)
  - NO evaluate library; inline numpy for accuracy
  - Save to /opt/ml/model/
  - numpy<2 pinned in requirements.txt
"""

import argparse
import os
import json
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


POSITIVE_TEXTS = [
    "My issue was resolved quickly by the support team, very happy.",
    "The agent was helpful and processed my refund without delay.",
    "Great service, they fixed the unauthorised charge immediately.",
    "Quick resolution to my dispute, I am satisfied with the outcome.",
    "The fraud alert was handled professionally and my account is safe.",
    "Excellent communication throughout the complaint process.",
    "My account was restored promptly after the error was identified.",
    "Very pleased with how the branch manager handled my concern.",
    "The team followed up as promised and my issue is fully resolved.",
    "Smooth experience from start to finish, no further problems.",
    "Agent was courteous and resolved the billing error in one call.",
    "I received my refund within 3 business days as stated.",
    "Password reset was straightforward and staff were helpful.",
    "The escalation team handled my case with professionalism.",
    "Transaction dispute was resolved in my favour, very thankful.",
    "Online banking issue was fixed after one chat session.",
    "Customer service exceeded my expectations on this occasion.",
    "Fraud team acted swiftly and reimbursed the stolen funds.",
    "My mortgage query was answered thoroughly and clearly.",
    "Very impressed with the speed of response to my complaint.",
]

NEGATIVE_TEXTS = [
    "Unauthorised charge appeared on my account and no one is helping.",
    "I have been waiting 3 weeks for a refund with no update.",
    "My card was blocked without warning and I cannot access my funds.",
    "Spoke to 4 different agents and still no resolution to my dispute.",
    "Fraud on my account went undetected for weeks, very disappointed.",
    "The branch told me one thing and online support said another.",
    "My complaint has been escalated twice and nothing has changed.",
    "Charged twice for the same transaction and I want my money back.",
    "Online banking keeps logging me out and losing my transfers.",
    "Nobody follows up when they say they will, terrible service.",
    "I am furious about the hidden fee that was never disclosed.",
    "Three weeks and my account is still frozen, this is unacceptable.",
    "The complaint team is impossible to reach and never returns calls.",
    "My direct debit was cancelled without any notification from Barclays.",
    "I was transferred to 5 departments and nobody could help me.",
    "Still waiting for my replacement card after reporting it stolen.",
    "The interest charge was applied incorrectly and it has not been fixed.",
    "I feel ignored and my issue has been dragging on for months.",
    "Terrible experience at the branch, staff were dismissive.",
    "My savings account was closed without any explanation.",
]


def make_dataset(n_train=800, n_val=200, seed=42):
    rng = random.Random(seed)
    all_texts = POSITIVE_TEXTS * 20 + NEGATIVE_TEXTS * 20
    all_labels = [1] * (len(POSITIVE_TEXTS) * 20) + [0] * (len(NEGATIVE_TEXTS) * 20)
    combined = list(zip(all_texts, all_labels))
    rng.shuffle(combined)
    train_data = combined[:n_train]
    val_data = combined[n_train: n_train + n_val]
    train_dataset = Dataset.from_dict({
        "text": [x[0] for x in train_data],
        "label": [x[1] for x in train_data],
    })
    val_dataset = Dataset.from_dict({
        "text": [x[0] for x in val_data],
        "label": [x[1] for x in val_data],
    })
    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """Compute accuracy without the evaluate library."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()
    return {"accuracy": accuracy}


def main():
    args = parse_args()
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Args:   {vars(args)}")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )

    print("Generating synthetic complaint dataset ...")
    train_dataset, val_dataset = make_dataset(seed=args.seed)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # CRITICAL: eval_strategy NOT evaluation_strategy (removed in transformers 4.41+)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting full fine-tuning ...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final validation metrics: {metrics}")

    model_output_dir = args.model_dir
    os.makedirs(model_output_dir, exist_ok=True)
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    metrics_path = os.path.join(model_output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model and tokenizer saved to {model_output_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
