"""
train.py - PEFT fine-tuning of DistilBERT for complaint classification.
Supports LoRA and QLoRA via --peft_method argument.

Dataset: financial_phrasebank (sentiment) as complaint proxy.
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
from datasets import load_dataset
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


def load_and_tokenise(tokenizer, max_length=128):
    """Load financial_phrasebank and tokenise for sequence classification."""
    ds = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=True,
    )
    # financial_phrasebank has only a train split, build an 80/20 split.
    split = ds["train"].train_test_split(test_size=0.2, seed=42)
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


def build_model(peft_method, lora_r, lora_alpha, num_labels=3):
    """Build DistilBERT with LoRA or QLoRA adapters."""
    model_name = "distilbert-base-uncased"

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
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
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
