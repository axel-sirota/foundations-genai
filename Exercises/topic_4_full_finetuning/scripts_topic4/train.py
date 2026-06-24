"""
train.py - Full fine-tuning of distilbert-base-uncased for Barclays complaint
routing. Runs as a SageMaker HuggingFace estimator GPU job on ml.g4dn.xlarge.

Task: 4-class routing classification. The four Barclays routing categories are
0 = fraud and security, 1 = billing and charges, 2 = account access,
3 = general enquiry. This matches the labelled_dataset.json produced in Topic 3.
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
    parser.add_argument("--num_labels", type=int, default=4)
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


# Four Barclays routing categories, matching Topic 3's label_map:
#   0 = fraud and security, 1 = billing and charges,
#   2 = account access,     3 = general enquiry
FRAUD_TEXTS = [
    "Unauthorised charge appeared on my account and no one is helping.",
    "I think my card has been cloned, there are payments I never made.",
    "Someone used my account details to set up a payment to a stranger.",
    "There is a suspicious transaction from another country on my statement.",
    "My card was used for a purchase I did not authorise.",
    "I received a scam text pretending to be Barclays asking for my PIN.",
    "Money was taken from my account by a merchant I have never dealt with.",
    "I reported fraud last week and the stolen funds are still not back.",
    "A direct debit I never agreed to is taking money from my account.",
    "My online banking shows logins from a device that is not mine.",
]

BILLING_TEXTS = [
    "You charged me an overdraft fee I did not expect this month.",
    "I was charged twice for the same transaction and want a refund.",
    "The interest charge on my statement was applied incorrectly.",
    "There is a hidden fee on my account that was never disclosed.",
    "My monthly account fee went up without any notification.",
    "I was billed a late payment charge even though I paid on time.",
    "The foreign transaction fee seems much higher than advertised.",
    "A service charge appeared that I do not understand at all.",
    "I am being charged for a packaged account I asked to close.",
    "The refund I was promised has not been credited to my balance.",
]

ACCESS_TEXTS = [
    "I cannot log in to the mobile app, it keeps rejecting my password.",
    "My card was blocked without warning and I cannot access my funds.",
    "Online banking keeps logging me out before I can finish a transfer.",
    "I am locked out of my account and the reset link does not work.",
    "The app will not accept my new passcode after I changed it.",
    "I cannot access my account because the security questions failed.",
    "My replacement card has not arrived and I cannot withdraw money.",
    "Two-factor authentication is not sending me the verification code.",
    "I forgot my membership number and cannot get into online banking.",
    "The website says my account is suspended but I do not know why.",
]

GENERAL_TEXTS = [
    "What are your branch opening hours over the bank holiday weekend?",
    "Can you tell me how to set up a standing order from my account?",
    "I would like to know the current interest rate on a savings account.",
    "How do I update the postal address on my account?",
    "What documents do I need to open a joint account with my partner?",
    "Is there a mobile app feature to freeze my card temporarily?",
    "How long does an international transfer usually take to arrive?",
    "Can I order a paper statement instead of the digital one?",
    "What is the daily limit for cash withdrawals at an ATM?",
    "How do I add a new payee to my online banking?",
]

CATEGORY_TEXTS = [FRAUD_TEXTS, BILLING_TEXTS, ACCESS_TEXTS, GENERAL_TEXTS]


def make_dataset(n_train=800, n_val=200, seed=42):
    rng = random.Random(seed)
    all_texts = []
    all_labels = []
    for label, texts in enumerate(CATEGORY_TEXTS):
        all_texts.extend(texts * 20)
        all_labels.extend([label] * (len(texts) * 20))
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
    # YOUR CODE: load a sequence-classification model from args.model_name
    # with the right number of output labels (args.num_labels) for 4-class routing.
    model = None  # YOUR CODE

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

    # YOUR CODE: build the TrainingArguments for full fine-tuning.
    # Fill in the four blanks below. CRITICAL: use eval_strategy, NOT
    # evaluation_strategy (removed in transformers 4.41+). Evaluate and save
    # once per epoch so load_best_model_at_end can pick the best checkpoint.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=None,  # YOUR CODE: how many passes over the data (args.epochs)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=None,  # YOUR CODE: the step size (args.lr)
        eval_strategy=None,  # YOUR CODE: when to run evaluation ("epoch")
        save_strategy=None,  # YOUR CODE: when to checkpoint ("epoch")
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
    # YOUR CODE: persist the fine-tuned model and its tokenizer to model_output_dir
    # so the artifact can be loaded back for inference. (2 lines)
    None  # YOUR CODE
    None  # YOUR CODE

    metrics_path = os.path.join(model_output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model and tokenizer saved to {model_output_dir}")
    print("Training complete.")


if __name__ == "__main__":
    main()
