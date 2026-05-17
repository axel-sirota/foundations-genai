"""
train.py - Transfer Learning with DistilBERT on SST-2 sentiment data
SageMaker PyTorch estimator entry point (CPU: ml.m5.xlarge)

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
from datasets import load_dataset


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
    for param in model.distilbert.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters:    {frozen_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Trainable ratio: {100.0 * trainable_count / (frozen_count + trainable_count):.2f}%")


def build_dataset(dataset_name, tokenizer, max_length, num_train=2000, num_eval=500):
    """
    Load SST-2 from HuggingFace Hub, subsample for CPU speed,
    tokenize, and return train/eval splits.

    SST-2 label mapping: 0 = negative, 1 = positive
    Column 'sentence' contains the review text.
    """
    raw = load_dataset(dataset_name)

    train_data = raw["train"].shuffle(seed=42).select(range(num_train))
    eval_data = raw["validation"].select(range(min(num_eval, len(raw["validation"]))))

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    train_tok = train_data.map(tokenize_fn, batched=True,
                               remove_columns=["sentence", "idx"])
    eval_tok = eval_data.map(tokenize_fn, batched=True,
                             remove_columns=["sentence", "idx"])
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
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    if args.freeze_encoder:
        print("\nApplying transfer learning: freezing encoder layers")
        freeze_encoder(model)
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\nFull fine-tuning mode: all {total:,} parameters trainable")

    print("\nLoading and tokenizing SST-2 dataset (subsampled for CPU)...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset, eval_dataset = build_dataset(
        "stanfordnlp/sst2", tokenizer, args.max_length
    )
    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # eval_strategy (NOT evaluation_strategy - removed in transformers 4.41+)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
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
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

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
