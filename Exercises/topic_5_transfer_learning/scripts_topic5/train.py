"""
train.py - Transfer Learning with DistilBERT on the Barclays complaint
routing task (4 classes).
SageMaker PyTorch estimator entry point (CPU: ml.m5.xlarge)

Task: route a customer complaint into one of 4 Barclays categories:
  0 = fraud and security
  1 = billing and charges
  2 = account access
  3 = general enquiry

Strategy: freeze all DistilBERT encoder layers, train only the
pre_classifier + classifier head. This is transfer learning, not
full fine-tuning.

Installed via requirements.txt before this script runs:
  transformers==4.40.0
  datasets==2.18.0
  numpy<2
"""

import argparse
import os
import json
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import concatenate_datasets


def compute_metrics(eval_pred):
    """Inline numpy accuracy metric. No evaluate library (incompatible with datasets 4.x)."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freeze_encoder", type=int, default=1,
                        help="1 = freeze DistilBERT encoder (transfer learning), 0 = full fine-tune")
    parser.add_argument("--model_name", type=str,
                        default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_data_dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return parser.parse_args()


def freeze_encoder(model):
    """
    Freeze all parameters in the DistilBERT encoder.
    Only the pre_classifier and classifier layers remain trainable.

    DistilBertForSequenceClassification architecture:
      model.distilbert     -> encoder (6 transformer layers) - FROZEN
      model.pre_classifier -> Linear(768, 768)               - TRAINABLE
      model.classifier     -> Linear(768, num_labels)        - TRAINABLE
      model.dropout        -> Dropout                        - TRAINABLE (no params)
    """
    frozen_count = 0
    # YOUR CODE: freeze the encoder. Loop over model.distilbert.parameters()
    # and mark each one as non-trainable, then add param.numel() to frozen_count.
    # The pre_classifier and classifier heads must stay trainable (do not touch them).
    for param in []:  # YOUR CODE: iterate the encoder parameters
        pass  # YOUR CODE: set this param to non-trainable, then count it
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters:    {frozen_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Trainable ratio: {100.0 * trainable_count / (frozen_count + trainable_count):.2f}%")


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

    Used when no labelled dataset is supplied. The course spiral normally
    feeds a real labelled dataset from Topic 3 via S3; this local set keeps
    the training script self-contained and reproducible.

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


def build_dataset(tokenizer, max_length, num_train=2000, num_eval=500):
    """
    Build the 4-class Barclays routing dataset, tokenize, and return
    train/eval splits.

    The local dataset is small, so it is repeated to reach the requested
    training size; this keeps CPU training time predictable while still
    teaching transfer learning on the real 4-class routing task.

    Label mapping: 0 = fraud and security, 1 = billing and charges,
    2 = account access, 3 = general enquiry.
    Column 'sentence' contains the complaint text.
    """
    from datasets import Dataset

    base = barclays_routing_examples()
    base_ds = Dataset.from_dict(base)

    repeats = max(1, (num_train + num_eval) // len(base_ds) + 1)
    pool = concatenate_datasets([base_ds] * repeats).shuffle(seed=42)
    train_data = pool.select(range(min(num_train, len(pool))))
    eval_data = base_ds.shuffle(seed=7).select(range(min(num_eval, len(base_ds))))

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    train_tok = train_data.map(tokenize_fn, batched=True,
                               remove_columns=["sentence"])
    eval_tok = eval_data.map(tokenize_fn, batched=True,
                             remove_columns=["sentence"])
    train_tok = train_tok.rename_column("label", "labels")
    eval_tok = eval_tok.rename_column("label", "labels")
    train_tok.set_format("torch")
    eval_tok.set_format("torch")
    return train_tok, eval_tok


def main():
    args = parse_args()
    set_seed(42)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Freeze encoder: {bool(args.freeze_encoder)}")

    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # YOUR CODE: load a sequence-classification model from args.model_name with
    # NUM_LABELS outputs. Pass id2label=LABEL_MAP and label2id so predictions
    # carry human-readable category names.
    model = None  # YOUR CODE

    if args.freeze_encoder:
        print("\nApplying transfer learning: freezing encoder layers")
        freeze_encoder(model)
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\nFull fine-tuning mode: all {total:,} parameters trainable")

    print("\nBuilding and tokenizing the 4-class Barclays routing dataset...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset, eval_dataset = build_dataset(
        tokenizer, args.max_length
    )
    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # YOUR CODE: build TrainingArguments. Fill the three blanks. CRITICAL: use
    # eval_strategy, NOT evaluation_strategy (removed in transformers 4.41+).
    # Evaluate and checkpoint once per epoch so the best model can be reloaded.
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=None,  # YOUR CODE: the step size (args.lr)
        eval_strategy=None,  # YOUR CODE: when to run evaluation ("epoch")
        save_strategy=None,  # YOUR CODE: when to checkpoint ("epoch")
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=42,
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    train_result = trainer.train()
    print(f"\nTraining complete. Loss: {train_result.training_loss:.4f}")

    print("\nRunning final evaluation...")
    eval_result = trainer.evaluate()
    print(f"Final accuracy: {eval_result['eval_accuracy']:.4f}")

    print(f"\nSaving model to {args.model_dir}")
    # YOUR CODE: save the trained model and tokenizer to args.model_dir. (2 lines)
    None  # YOUR CODE
    None  # YOUR CODE

    metrics = {
        "training_loss": round(train_result.training_loss, 4),
        "eval_accuracy": round(eval_result["eval_accuracy"], 4),
        "eval_loss": round(eval_result["eval_loss"], 4),
        "freeze_encoder": bool(args.freeze_encoder),
        "epochs": args.epochs,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
