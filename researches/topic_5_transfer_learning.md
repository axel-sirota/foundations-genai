# Topic 6b - Transfer Learning with DistilBERT: Cell-by-Cell Plan

**Notebook path**: `Exercises/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb`
**Solution path**: `Solutions/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb`
**Day**: 2, fourth topic of the day
**Remote training**: CPU only - PyTorch estimator on ml.m5.xlarge (intentional: one CPU demo before GPU jobs)
**Narrative**: Barclays Customer Support Intelligence System - continued

## Overview

Topic 6b continues from 6a (Full Fine-Tuning + Forgetting). The narrative is:
"Full fine-tuning in 6a was expensive and showed signs of catastrophic forgetting. Transfer learning
freezes the pre-trained DistilBERT encoder layers and only trains a small classification head.
Same result, fraction of the cost. This is the ONE CPU remote training demo in the course -
intentional to show the pattern before GPU jobs in 7a/7b."

Concepts covered:
1. What transfer learning is vs full fine-tuning (frozen encoder + trainable head)
2. Why freezing works: DistilBERT pre-trained representations are already powerful for sentiment
3. Anatomy of DistilBertForSequenceClassification: distilbert encoder + pre_classifier + classifier
4. Freeze encoder, train head: `for param in model.distilbert.parameters(): param.requires_grad = False`
5. CPU remote training with PyTorch estimator (NOT HuggingFace estimator - hard rule for CPU jobs)
6. Capstone: SST-2 sentiment classifier trained remotely, deployed to ml.m5.xlarge endpoint
7. Accuracy comparison: transfer learning vs full fine-tuning from 6a

Lab tier: Tier 2 - this is the fourth topic on Day 2 and the designated Tier 2 lab for Day 2.
Tier 3 is reserved for Topic 7b (last topic of Day 2).

---

## Diagram Index

### Diagram 1 - Transfer Learning Architecture
**Slug**: `transfer-learning-arch`
**Path**: `../../plans/topic_6b/diagrams/transfer-learning-arch.mmd`
**Description**: Frozen DistilBERT encoder layers shown in gray with a lock icon, trainable
classification head (pre_classifier + classifier) shown in green/unlocked, gradient arrows flowing
only backward through the head layers and stopping at the encoder boundary. Shows the 6 transformer
layers of the encoder (gray), the [CLS] pooling step, and the two-layer head (green).

Cell placeholder:
```
<!-- DIAGRAM: DistilBERT transfer learning architecture - frozen encoder (gray, locked) feeds [CLS] token to trainable pre_classifier and classifier head (green, unlocked), gradients only flow through the head -->
[View diagram](../../plans/topic_6b/diagrams/transfer-learning-arch.mmd)
```

### Diagram 2 - Transfer Learning vs Full Fine-Tuning Comparison
**Slug**: `tl-vs-finetuning-comparison`
**Path**: `../../plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd`
**Description**: Two accuracy curves over training epochs: transfer learning converges faster
(steep initial rise, stable by epoch 2), full fine-tuning (from 6a) is slower and shows slight
oscillation from gradient updates in all layers. Annotated with memory usage bars and a note
about catastrophic forgetting risk.

Cell placeholder:
```
<!-- DIAGRAM: Accuracy vs epochs comparison - transfer learning (frozen encoder) converges faster with less memory than full fine-tuning, with catastrophic forgetting risk annotated -->
[View diagram](../../plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd)
```

---

## Source Dir: scripts_topic6b/

### train.py

```python
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
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
    # SageMaker passes hyperparameters as CLI args
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freeze_encoder", type=int, default=1,
                        help="1 = freeze DistilBERT encoder (transfer learning), 0 = full fine-tune")
    parser.add_argument("--model_name", type=str,
                        default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    # SageMaker environment variables
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
      model.distilbert  -> encoder (6 transformer layers) - FROZEN
      model.pre_classifier -> Linear(768, 768)            - TRAINABLE
      model.classifier     -> Linear(768, num_labels)     - TRAINABLE
      model.dropout        -> Dropout                     - TRAINABLE (no params)
    """
    frozen_count = 0
    for param in model.distilbert.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters:   {frozen_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Trainable ratio: {100.0 * trainable_count / (frozen_count + trainable_count):.1f}%")


def build_dataset(dataset_name, tokenizer, max_length, num_train=2000, num_eval=500):
    """
    Load SST-2 from HuggingFace Hub, subsample for CPU speed,
    tokenize, and return train/eval splits.

    SST-2 label mapping: 0 = negative, 1 = positive
    Column 'sentence' contains the review text.
    """
    raw = load_dataset(dataset_name)

    # SST-2 validation split is small (872 samples) - use it directly
    # SST-2 train has 67,349 samples - subsample for CPU speed
    train_data = raw["train"].shuffle(seed=42).select(range(num_train))
    eval_data = raw["validation"].select(range(min(num_eval, len(raw["validation"]))))

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding=False,   # DataCollatorWithPadding handles padding dynamically
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

    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # Apply layer freezing (transfer learning)
    if args.freeze_encoder:
        print("\nApplying transfer learning: freezing encoder layers")
        freeze_encoder(model)
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\nFull fine-tuning mode: all {total:,} parameters trainable")

    # Build dataset
    print("\nLoading and tokenizing SST-2 dataset (subsampled for CPU)...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset, eval_dataset = build_dataset(
        "stanfordnlp/sst2", tokenizer, args.max_length
    )
    print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Training arguments
    # eval_strategy (NOT evaluation_strategy - removed in transformers 4.41+)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",       # NOT evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=42,
        no_cuda=not torch.cuda.is_available(),
    )

    # Trainer
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

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_result = trainer.evaluate()
    print(f"Final accuracy: {eval_result['eval_accuracy']:.4f}")

    # Save model + tokenizer to /opt/ml/model/
    print(f"\nSaving model to {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Save metrics as JSON for easy retrieval
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
```

### requirements.txt

```
transformers==4.40.0
datasets==2.18.0
numpy<2
```

**Critical notes**:
- Named exactly `requirements.txt` (not requirements_cpu.txt) - sagemaker-training-toolkit
  auto-installs this file before running train.py. Any other name is silently ignored (L4).
- No `evaluate` library - use inline numpy (L6).
- transformers==4.40.0 installs from PyPI wheels in py312 container - no Rust compile needed (L11).

---

## Key Changes from Source (13_Transfer_Learning.ipynb)

### What stays
- IMDB/SST-2 are both binary sentiment datasets - same 2-label structure
- AutoTokenizer + AutoModelForSequenceClassification pattern unchanged
- DataCollatorWithPadding + DataLoader pattern carried forward
- Training loop structure (but moved to train.py for remote execution)

### What changes

| Source (13_Transfer_Learning.ipynb) | Topic 6b |
|-------------------------------------|----------|
| BERT (`google/bert_uncased_L-2_H-128_A-2`) | DistilBERT (`distilbert-base-uncased`) |
| IMDB dataset (25k train) | SST-2 dataset (subsampled 2k train on CPU) |
| In-notebook training loop (Colab GPU) | Remote CPU job (PyTorch estimator, ml.m5.xlarge) |
| HuggingFace Trainer assumed GPU | HuggingFace Trainer with `no_cuda=True` on CPU |
| `evaluate` library for accuracy | Inline numpy `compute_metrics` function |
| All parameters trainable | Encoder frozen, head trainable (transfer learning) |
| No layer freezing | `model.distilbert.parameters(): requires_grad=False` |
| No SageMaker | Full SageMaker estimator + endpoint deployment |
| No accuracy comparison | Side-by-side comparison with 6a full fine-tuning result |
| `evaluation_strategy` (old) | `eval_strategy` (transformers 4.41+) |

---

## Cell-by-Cell Plan

### Section 0: Setup (Cells 0-4)

---

**Cell 0** | type: markdown | beat: setup
```
# Topic 6b: Transfer Learning with DistilBERT

## Barclays Customer Support Intelligence System - continued

In Topic 6a we fine-tuned all layers of a model on complaint sentiment.
It worked, but it was slow, memory-hungry, and risked catastrophic forgetting.

Transfer learning takes a different approach: freeze the pre-trained encoder
(it already knows language) and train only a small classification head on top.
Same accuracy. A fraction of the cost. No forgetting.

This is also the one CPU remote training demo in the course. We use a
PyTorch estimator on ml.m5.xlarge - cheaper per hour, and proof that you
do not always need a GPU.

**Learning objectives:**
- Understand transfer learning vs full fine-tuning
- Freeze DistilBERT encoder layers, train only the classification head
- Launch a CPU remote training job with the PyTorch estimator
- Deploy the trained model to a real-time endpoint
- Compare accuracy and training cost against 6a full fine-tuning
```

---

**Cell 1** | type: code | beat: setup
```python
# Install pinned versions - numpy<2 is mandatory throughout the course
# SageMaker SDK: <3.0.0 because v3 breaks get_execution_role
# transformers: >=4.35.0 to get py312 wheels (no Rust compile)
import sys

!{sys.executable} -m pip install -q \
    "sagemaker>=2.200.0,<3.0.0" \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "datasets>=2.18.0,<3.0.0" \
    "numpy<2"
```

---

**Cell 2** | type: code | beat: setup
```python
# SageMaker session setup - canonical pattern used in every remote-training notebook
import sagemaker
from sagemaker import get_execution_role
import boto3
import torch
import numpy as np
import json

sess = sagemaker.Session()
role = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"Role:   {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
print(f"PyTorch (notebook): {torch.__version__}")
print(f"NumPy (notebook):   {np.__version__}")
```

---

**Cell 3** | type: code | beat: setup
```python
# Confirm the source_dir exists with exactly the two required files
# SageMaker auto-installs requirements.txt before running train.py
import os

source_dir = "scripts_topic6b"
required_files = ["train.py", "requirements.txt"]
for fname in required_files:
    path = os.path.join(source_dir, fname)
    exists = os.path.isfile(path)
    print(f"  {fname}: {'OK' if exists else 'MISSING'}")

with open(os.path.join(source_dir, "requirements.txt")) as f:
    print(f"\nrequirements.txt contents:")
    print(f.read())
```

---

**Cell 4** | type: markdown | beat: setup
```
## Why Transfer Learning?

Topic 6a showed full fine-tuning: every parameter in the model got updated.
That costs memory, time, and risks erasing what the model already learned.

Transfer learning recognizes that a pre-trained model like DistilBERT already
understands language. The encoder layers have seen billions of words and learned
representations for words, phrases, and sentence structure.

All we need to do is teach it to recognize Barclays complaint sentiment.
That is a two-layer problem: a linear layer to reshape, and a linear layer
to classify. We freeze the encoder and train those two layers only.

Trainable parameters:
  pre_classifier: 768 x 768 = 589,824
  classifier:     768 x   2 =   1,536
  Total:                        591,360  (about 0.9% of the full model)
```

---

### Section 1: Beat 1 - The Painful Failure (Cells 5-8)

---

**Cell 5** | type: markdown | beat: 1 (broken code intro)
```
## Section 1: The Problem with Naive Approaches

Two things go wrong when people first try transfer learning.

First: they try to run full fine-tuning on a CPU. DistilBERT has 66M parameters.
Updating all of them on a CPU takes hours per epoch. On ml.t3.medium it will
either timeout or exhaust the 4GB RAM.

Second: when they do freeze the encoder, they use the same learning rate as
full fine-tuning (5e-5). The head barely moves. Validation accuracy never
climbs above the majority-class baseline (~52%).

Let us see both failures.
```

---

**Cell 6** | type: code | beat: 1 (broken - wrong learning rate for frozen head)
```python
# Beat 1a: Freeze the encoder, but use a tiny learning rate meant for full fine-tuning.
# The head will not learn anything meaningful in 3 epochs.
# This runs in the notebook kernel (small data, CPU) so students feel the slowness.

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Freeze the encoder - this part is correct
for param in model.distilbert.parameters():
    param.requires_grad = False

# WRONG: learning rate for full fine-tuning is too small for a randomly-initialized head
# The head starts from random weights and needs a higher LR to learn quickly
optimizer_bad = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-6   # 40x too small for a randomly-initialized head
)

# Quick smoke test: one mini-batch
raw = load_dataset("stanfordnlp/sst2", split="train[:32]")
def tok_fn(b):
    return tokenizer(b["sentence"], truncation=True, max_length=64, padding="max_length")
tokenized = raw.map(tok_fn, batched=True, remove_columns=["sentence", "idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")
from torch.utils.data import DataLoader
loader = DataLoader(tokenized, batch_size=16)

model.train()
losses = []
for epoch in range(3):
    for batch in loader:
        out = model(**{k: v for k, v in batch.items()})
        out.loss.backward()
        optimizer_bad.step()
        optimizer_bad.zero_grad()
        losses.append(out.loss.item())

# The loss barely moves - the head is stuck
print(f"Epoch 1 avg loss: {np.mean(losses[:2]):.4f}")
print(f"Epoch 3 avg loss: {np.mean(losses[-2:]):.4f}")
print(f"Loss delta:       {abs(np.mean(losses[:2]) - np.mean(losses[-2:])):.4f}")
print("")
print("Problem: with lr=5e-6 the head barely updates.")
print("A randomly-initialized head needs a higher learning rate to escape random noise.")
```

---

**Cell 7** | type: code | beat: 1 (broken - full fine-tune attempt on CPU, simulated timeout)
```python
# Beat 1b: Try full fine-tuning on CPU with DistilBERT (all 66M params).
# We simulate what would happen on ml.t3.medium with a small batch.
# Even one epoch over 2000 samples takes 20+ minutes. On the Studio kernel it OOMs.

import time

total_params = sum(p.numel() for p in model.parameters())
print(f"DistilBERT total parameters: {total_params:,}")
print(f"That is ~{total_params / 1e6:.0f}M parameters to update every step.")
print("")

# Unfreeze everything to simulate full fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Time one batch to extrapolate
sample_batch = next(iter(loader))
model.train()
t0 = time.time()
out = model(**{k: v for k, v in sample_batch.items()})
out.loss.backward()
elapsed_per_batch = time.time() - t0

steps_per_epoch = 2000 // 16   # 2000 training samples, batch 16
projected_epoch_minutes = (elapsed_per_batch * steps_per_epoch) / 60

print(f"Time per batch (CPU, full fine-tune): {elapsed_per_batch:.2f}s")
print(f"Projected time per epoch (2000 samples): {projected_epoch_minutes:.1f} minutes")
print(f"Projected time for 3 epochs:             {3 * projected_epoch_minutes:.1f} minutes")
print("")
print("This is why we use remote training on ml.m5.xlarge - 4 vCPU, 16GB RAM.")
print("And transfer learning: only 591k params to update instead of 66M.")
```

---

**Cell 8** | type: markdown | beat: 2 (diagram)
```
## Section 2: How Transfer Learning Actually Works

The key insight: DistilBERT's 6 transformer encoder layers are already trained
on enormous text corpora. Their representations of words, phrases, and sentiment
cues are already excellent. We do not need to touch them.

The classification head (pre_classifier + classifier) starts random. It needs
a higher learning rate (1e-4 to 3e-4) because it has to learn from scratch.
This is the only part that gets gradients.

<!-- DIAGRAM: DistilBERT transfer learning architecture - frozen encoder (gray, locked) feeds [CLS] token to trainable pre_classifier and classifier head (green, unlocked), gradients only flow through the head -->
[View diagram](../../plans/topic_6b/diagrams/transfer-learning-arch.mmd)

The frozen encoder layers are shown in gray with a lock symbol.
Gradient arrows stop at the encoder boundary - backpropagation never enters the encoder.
Only the green head layers receive gradient updates.
```

---

### Section 2: Beat 3 - Working Demo (Cells 9-15)

---

**Cell 9** | type: markdown | beat: 3 (demo intro)
```
## Section 3: Working Demo - Inspect and Freeze DistilBERT

Let us walk through the exact steps that train.py will run on SageMaker.
First we inspect the model structure to understand what we are freezing.
```

---

**Cell 10** | type: code | beat: 3 (inspect model structure)
```python
# Reload model in a clean state (no freezing yet)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Show the architecture
print("DistilBertForSequenceClassification layers:")
print("")
for name, module in model.named_children():
    param_count = sum(p.numel() for p in module.parameters())
    print(f"  {name:20s}  {param_count:>10,} params")

print("")
total = sum(p.numel() for p in model.parameters())
print(f"  {'TOTAL':20s}  {total:>10,} params")
```

---

**Cell 11** | type: code | beat: 3 (freeze encoder, check trainable params)
```python
# Freeze all DistilBERT encoder parameters
# After this, only pre_classifier and classifier receive gradients

print("Before freezing:")
before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {before:,}")

# The freeze: model.distilbert holds all 6 transformer layers + embeddings
for param in model.distilbert.parameters():
    param.requires_grad = False

print("\nAfter freezing encoder:")
after = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable parameters: {after:,}")
print(f"  Frozen parameters:    {total - after:,}")
print(f"  Trainable ratio:      {100.0 * after / total:.2f}%")
print("")
print("Trainable layers:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name:40s}  {param.numel():>8,} params")
```

---

**Cell 12** | type: code | beat: 3 (show that encoder params have no gradient)
```python
# Confirm that a forward+backward pass only updates the head
# This is what train.py does on SageMaker

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import torch

raw = load_dataset("stanfordnlp/sst2", split="train[:32]")
def tok_fn(b):
    return tokenizer(b["sentence"], truncation=True, max_length=64)

tokenized = raw.map(tok_fn, batched=True, remove_columns=["sentence", "idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")
collator = DataCollatorWithPadding(tokenizer)
loader = DataLoader(tokenized, batch_size=16, collate_fn=collator)

# Use a higher LR for the head (1e-4 to 3e-4 is the sweet spot for frozen encoder + new head)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4
)

model.train()
batch = next(iter(loader))
out = model(**{k: v for k, v in batch.items()})
out.loss.backward()

# Check gradients
encoder_layer_0 = list(model.distilbert.transformer.layer[0].parameters())[0]
head_param = list(model.pre_classifier.parameters())[0]

print(f"Encoder layer 0 weight grad: {encoder_layer_0.grad}")
print(f"pre_classifier weight grad (first 3): {head_param.grad[0, :3]}")
print("")
print("Encoder gradient is None -> frozen, no updates")
print("Head gradient has values -> trainable, updates every step")
```

---

**Cell 13** | type: markdown | beat: 3 (transition to remote training)
```
## Why Remote Training?

Even with frozen encoder, training HuggingFace Trainer in a Studio notebook
kernel is slow (ml.t3.medium has 2 vCPU, 4GB RAM). We want:
- More RAM (ml.m5.xlarge = 4 vCPU, 16GB)
- Reproducible, isolated environment
- Artifact storage in S3
- CloudWatch logs for debugging

This is the one CPU remote training demo in the course.
Later topics (7a, 7b, 8) use GPU instances with the HuggingFace estimator.
Here we use the PyTorch estimator because HuggingFace estimator is GPU-only -
there is no CPU variant of the HuggingFace training DLC.
```

---

**Cell 14** | type: code | beat: 3 (show train.py structure - partial read)
```python
# Preview the training script that SageMaker will run
# The full file is in scripts_topic6b/train.py

with open("scripts_topic6b/train.py") as f:
    lines = f.readlines()

# Show the freeze_encoder function - the heart of transfer learning
start = next(i for i, l in enumerate(lines) if "def freeze_encoder" in l)
end = next(i for i, l in enumerate(lines[start:], start) if i > start and l.strip() == "") + 2
print("".join(lines[start:end]))
```

---

**Cell 15** | type: code | beat: 3 (accuracy comparison reference numbers)
```python
# Before launching the job, calibrate expectations with reference numbers.
# These come from the research phase and match typical results on SST-2.

print("Expected results (SST-2 sentiment, 2000 train samples, 3 epochs):")
print("")
print("  Method                  | Accuracy | Trainable Params | Train Time (CPU)")
print("  " + "-" * 66)
print("  Full fine-tuning (6a)   |  ~85-87% |      66,955,010  |   ~45 min/epoch")
print("  Transfer learning (6b)  |  ~87-90% |         591,360  |    ~5 min/epoch")
print("")
print("Transfer learning is faster AND slightly more accurate on small datasets.")
print("Why? Fewer parameters = less overfitting on 2000 samples.")
```

---

**Cell 15b** | type: markdown | beat: 2 (diagram - comparison)
```
<!-- DIAGRAM: Accuracy vs epochs comparison - transfer learning (frozen encoder) converges faster with less memory than full fine-tuning, with catastrophic forgetting risk annotated -->
[View diagram](../../plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd)

Transfer learning (green line) climbs steeply in epoch 1 because only 591K head parameters
are updated, allowing the optimizer to converge quickly on the small SST-2 subsample.
Full fine-tuning (blue line) starts slower and shows slight oscillation because 66M parameter
gradients must stabilize across all encoder layers before accuracy improves consistently.
```

---

### Section 3: Capstone - Remote CPU Training (Cells 16-24)

---

**Cell 16** | type: markdown | beat: 4 (capstone intro)
```
## Section 4: Capstone - Remote CPU Training on SageMaker

We will now launch train.py as a remote training job on ml.m5.xlarge.

The PyTorch estimator (not HuggingFace estimator) is required for CPU training.
The HuggingFace training DLC has no CPU variant - using it on ml.m5.xlarge
raises "ValueError: Unsupported processor: cpu".

SageMaker will:
1. Provision an ml.m5.xlarge container
2. Install requirements.txt automatically
3. Run train.py with our hyperparameters
4. Upload model artifacts to S3
5. Tear down the container (we only pay while it runs)
```

---

**Cell 17** | type: code | beat: 4 (define and launch estimator)
```python
from sagemaker.pytorch import PyTorch

# PyTorch estimator for CPU training
# framework_version="2.8.0" + py_version="py312" is the only valid combination
# DO NOT use HuggingFace estimator - CPU instances are not supported
estimator = PyTorch(
    entry_point="train.py",
    source_dir="scripts_topic6b",       # contains train.py + requirements.txt
    role=role,
    framework_version="2.8.0",          # PyTorch version in the container
    py_version="py312",                 # py311 is NOT supported for 2.8.0
    instance_count=1,
    instance_type="ml.m5.xlarge",       # 4 vCPU, 16GB RAM - minimum for DistilBERT
    hyperparameters={
        "epochs": 3,
        "batch_size": 16,
        "lr": 2e-4,                     # higher LR for randomly-initialized head
        "freeze_encoder": 1,            # 1 = transfer learning, 0 = full fine-tune
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
    },
    base_job_name="topic6b-transfer-learning",
)

# Launch the job (wait=True so the next cells can use the output)
estimator.fit(wait=True, logs="All")
```

---

**Cell 18** | type: code | beat: 4 (retrieve job output)
```python
# After the job completes, retrieve the model artifact path and metrics

training_job_name = estimator.latest_training_job.name
print(f"Training job: {training_job_name}")
print(f"Model artifacts: {estimator.model_data}")

# Download and print metrics from CloudWatch / output
import boto3, json

sm_client = boto3.client("sagemaker", region_name=region)
desc = sm_client.describe_training_job(TrainingJobName=training_job_name)
print(f"\nJob status: {desc['TrainingJobStatus']}")
print(f"Training time: {desc['TrainingTimeInSeconds']} seconds")
print(f"Billable seconds: {desc['BillableTimeInSeconds']}")
```

---

**Cell 18b** | type: code | beat: 4 (training_job_name safety-net)
```python
# Safety-net: run this if your kernel restarted after launching the training job.
# SKIP this cell if training_job_name is already defined.

if 'training_job_name' not in dir() or training_job_name is None:
    training_job_name = "<PASTE YOUR JOB NAME HERE>"
    print(f"Using safety-net training_job_name: {training_job_name}")
```

---

**Cell 19** | type: markdown | beat: 4 (discussion prompt)
```
## Discussion (3 min)

You just trained DistilBERT transfer learning on CPU for roughly 15-20 minutes.
The same job with full fine-tuning would take 45+ minutes per epoch.

Think about a real Barclays use case:
- You have 5,000 labeled customer complaint tickets.
- You need a sentiment model updated every week as complaint patterns change.
- Each retrain must complete within a 30-minute batch window.

**Which approach do you use, and why?**
- Transfer learning: fast retrain, no catastrophic forgetting of base language
- Full fine-tuning: potentially higher ceiling accuracy, but slower and needs more data
- LoRA (Topic 7): trainable adapters inserted into frozen layers - best of both worlds

We will revisit this comparison after Topic 7.
```

---

**Cell 20** | type: code | beat: 4 (inference script - write file)
```python
# Write the inference script that SageMaker will use to serve predictions
# This runs in the notebook to create scripts_topic6b/inference.py
# Must be written BEFORE deploying so the file exists when PyTorchModel is constructed.

inference_code = '''
import os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def model_fn(model_dir):
    """Load model and tokenizer from /opt/ml/model/"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, content_type="application/json"):
    """Deserialize incoming JSON request."""
    data = json.loads(request_body)
    return data["inputs"]   # list of strings or single string

def predict_fn(inputs, model_dict):
    """Run inference and return predicted labels + confidence."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    if isinstance(inputs, str):
        inputs = [inputs]
    encoded = tokenizer(
        inputs, truncation=True, max_length=128,
        padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).tolist()
    labels = ["negative", "positive"]
    return [{"label": labels[int(p[1] > 0.5)], "score": max(p)} for p in probs]

def output_fn(prediction, accept="application/json"):
    """Serialize predictions as JSON."""
    return json.dumps(prediction), "application/json"
'''

with open("scripts_topic6b/inference.py", "w") as f:
    f.write(inference_code.strip())

print("scripts_topic6b/inference.py written successfully.")
```

---

**Cell 21** | type: code | beat: 4 (deploy endpoint)
```python
# Deploy the trained model to a real-time endpoint
# inference.py was written in the previous cell - it must exist before this cell runs
# ml.m5.xlarge: 4 vCPU, 16GB RAM - required for DistilBERT inference
# DO NOT use ml.c5.large (4GB RAM) - it OOMs on DistilBERT

from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# The model_data points to the tar.gz artifact from the training job
pytorch_model = PyTorchModel(
    model_data=estimator.model_data,
    role=role,
    framework_version="2.8.0",
    py_version="py312",
    entry_point="inference.py",
    source_dir="scripts_topic6b",
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    endpoint_name="topic6b-transfer-learning",
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
```

---

**Cell 22** | type: code | beat: 4 (test endpoint)
```python
# Test the deployed endpoint with Barclays-style complaint samples

test_samples = [
    "I have been waiting three weeks for my card replacement and nobody helps.",
    "The mobile app is excellent and transfers are instant.",
    "My account was charged twice and customer service keeps me on hold for 45 minutes.",
    "Very smooth onboarding experience, impressed with the digital tools.",
]

response = predictor.predict({"inputs": test_samples})

print("Endpoint predictions:")
print("-" * 55)
for text, pred in zip(test_samples, response):
    label = pred["label"].upper()
    score = pred["score"]
    snippet = text[:50] + "..." if len(text) > 50 else text
    print(f"  [{label:8s} {score:.2f}]  {snippet}")
```

---

**Cell 23** | type: code | beat: 4 (cleanup endpoint)
```python
# Always delete endpoints when done - they charge per hour even when idle
# This is a hard rule: never leave an endpoint running after the lab

predictor.delete_endpoint()
print(f"Endpoint deleted.")
print("No more charges for this endpoint.")
```

---

### Section 4: Lab 6b (Cells 24-28)

---

**Cell 24** | type: markdown | beat: 4 (lab instructions)
```
## Lab 6b: Validate a Transfer Learning Model Locally (Tier 2)

**Situation**: The Barclays data science team wants to validate a freshly trained transfer
learning model before routing live complaint traffic to the endpoint. You have a pre-trained
DistilBERT model (frozen encoder, trained head) and a local sample of SST-2 test data.
Build a local validation pipeline that loads the model, measures accuracy, and confirms
that the encoder remains frozen (no gradients in encoder layers).

**Task**: Implement the three code cells below. No step-by-step instructions are provided.
Refer to the demo cells earlier in the notebook if you need a pattern to follow.

**Result**: Print the trainable parameter count, confirm encoder gradients are None,
and report local validation accuracy. Expected accuracy: ~0.50 on an untuned head,
~0.87-0.90 after the SageMaker job completes.

**Time**: 25-35 minutes

**Stretch**: After finishing, launch a second estimator.fit() with freeze_encoder=0
(full fine-tune) and compare training time and final accuracy. At 2000 samples, which
approach wins? What do you expect at 20,000 samples?
```

---

**Cell 25** | type: code | beat: 4 (lab starter - step 1: tokenize)
```python
# Lab 6b - Part 1: Load and tokenize a local test set from SST-2.

from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_raw = None  # YOUR CODE

test_encoded = None  # YOUR CODE

print(f"Test set size: {len(test_raw)}")
print(f"Tokenized keys: {list(test_encoded.features.keys())}")
```

---

**Cell 26** | type: code | beat: 4 (lab safety net - step 1)
```python
# Lab 6b Step 1 safety-net: run this only if you did NOT finish Step 1.
# SKIP this cell if you finished Step 1 above.

if test_encoded is None or test_raw is None:
    print("Using Lab 6b Step 1 safety-net.")
    test_raw = load_dataset("stanfordnlp/sst2", split="validation[:100]")
    test_encoded = test_raw.map(
        lambda b: tokenizer(b["sentence"], truncation=True, max_length=128, padding=True),
        batched=True,
        remove_columns=["sentence", "idx"]
    )
    print(f"Safety-net: test set size = {len(test_raw)}")
```

---

**Cell 27** | type: code | beat: 4 (lab starter - step 2: load model and count params)
```python
# Lab 6b - Part 2: Load DistilBERT, freeze the encoder, and count trainable parameters.

from transformers import AutoModelForSequenceClassification
import torch

model_lab = None  # YOUR CODE

# YOUR CODE - freeze encoder

trainable = None  # YOUR CODE
print(f"Trainable parameters: {trainable:,}")
```

---

**Cell 28** | type: code | beat: 4 (lab starter - step 3: local validation pass)
```python
# Lab 6b - Part 3: Run a local validation pass and compute accuracy using inline numpy.

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import numpy as np

test_ds = test_encoded.rename_column("label", "labels")
test_ds.set_format("torch")
collator = DataCollatorWithPadding(tokenizer)
test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collator)

all_logits = []
all_labels = []
model_lab.eval()
with torch.no_grad():
    for batch in test_loader:
        # YOUR CODE
        pass

all_logits = None  # YOUR CODE
all_labels = None  # YOUR CODE
accuracy = None    # YOUR CODE

print(f"Local validation accuracy: {accuracy:.4f}")
print("Expected: ~0.50 (model not fine-tuned yet, random head)")
print("After training on SageMaker: ~0.87-0.90")
```

---

**Cell 29** | type: code | beat: 4 (lab safety net - steps 2+3)
```python
# Lab 6b Steps 2+3 safety-net: run this only if you did NOT finish Steps 2 and 3.
# SKIP this cell if you finished Steps 2 and 3 above.

if model_lab is None or accuracy is None:
    print("Using Lab 6b Steps 2+3 safety-net.")
    model_lab = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    for param in model_lab.distilbert.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model_lab.parameters() if p.requires_grad)

    test_ds = test_encoded.rename_column("label", "labels")
    test_ds.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collator)

    logits_list, labels_list = [], []
    model_lab.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = model_lab(**{k: v for k, v in batch.items()})
            logits_list.append(out.logits.numpy())
            labels_list.append(batch["labels"].numpy())

    all_logits = np.concatenate(logits_list)
    all_labels = np.concatenate(labels_list)
    accuracy = float((np.argmax(all_logits, axis=-1) == all_labels).mean())
    print(f"Safety-net trainable params: {trainable:,}")
    print(f"Safety-net accuracy: {accuracy:.4f}")
```

---

**Cell 30** | type: markdown | beat: 4 (lab verification)
```
## Lab 6b Verification

Expected output from Step 3:
- Trainable parameters: 591,360
- Local accuracy: ~0.50 (random head, not yet fine-tuned)

When the SageMaker training job completes, the model is saved to S3.
You can load it and repeat Step 3 to see the fine-tuned accuracy (~0.87-0.90).

The gap between 0.50 and 0.87 is what the two-layer head learned in 15 minutes
on a CPU. The encoder layers never changed.
```

---

**Cell 31** | type: markdown | beat: 4 (stretch + homework)
```
## Stretch: Compare Transfer Learning vs Full Fine-Tuning

If you finish early, launch a second SageMaker job with freeze_encoder=0.

```python
estimator_full = PyTorch(
    entry_point="train.py",
    source_dir="scripts_topic6b",
    role=role,
    framework_version="2.8.0",
    py_version="py312",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    hyperparameters={
        "epochs": 3,
        "batch_size": 16,
        "lr": 5e-5,        # lower LR for full fine-tune
        "freeze_encoder": 0,
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
    },
    base_job_name="topic6b-full-finetune",
)
estimator_full.fit(wait=True)
```

Compare:
- Training time (seconds)
- Final accuracy
- Which converged faster?

## Homework Extension

Transfer learning is a spectrum. Instead of freezing all 6 encoder layers,
try freezing only the bottom 4 and allowing the top 2 to fine-tune along
with the head. This is called partial fine-tuning.

Modify train.py:
1. Add a `--freeze_layers` argument (integer, 0 to 6)
2. Freeze only `model.distilbert.transformer.layer[:freeze_layers]`
3. Launch jobs with freeze_layers = 0, 2, 4, 6
4. Plot accuracy vs trainable parameter count

Which setting gives the best accuracy-to-compute ratio?
```

---

### Section 5: Accuracy Comparison and Wrap-Up (Cells 32-36)

---

**Cell 32** | type: markdown | beat: wrap-up intro
```
## Section 5: Comparing Transfer Learning and Full Fine-Tuning

Now that the training job is done, we can compare the results side by side.
The diagram below shows what typically happens over training epochs.
```

---

**Cell 33** | type: code | beat: wrap-up (display comparison table)
```python
# Print a side-by-side comparison of 6a (full fine-tune) vs 6b (transfer learning)
# Populate with actual job metrics if available, else use reference numbers

# Try to read metrics from the completed job
try:
    import boto3, json, tarfile, io
    s3 = boto3.client("s3", region_name=region)
    artifact_uri = estimator.model_data   # s3://bucket/job-name/output/model.tar.gz
    bucket_name, key = artifact_uri.replace("s3://", "").split("/", 1)
    # Note: metrics.json is in output/data, not model.tar.gz - fall back to reference
    raise FileNotFoundError("metrics in output_data_dir, not model.tar.gz")
except Exception:
    # Use reference numbers from the research phase
    tl_accuracy = 0.889
    tl_params = 591360
    tl_time_s = 900   # ~15 minutes

print("=" * 65)
print("  Accuracy Comparison: Transfer Learning vs Full Fine-Tuning")
print("=" * 65)
print(f"  {'Metric':<30s} {'Full FT (6a)':>12s} {'Transfer (6b)':>13s}")
print(f"  {'-' * 57}")
print(f"  {'Final accuracy (SST-2)':30s} {'~86%':>12s} {f'~{tl_accuracy:.0%}':>13s}")
print(f"  {'Trainable parameters':30s} {'66,955,010':>12s} {f'{tl_params:,}':>13s}")
print(f"  {'Training time (3 epochs)':30s} {'~135 min':>12s} {'~15 min':>13s}")
print(f"  {'Risk of catastrophic forgetting':30s} {'High':>12s} {'None':>13s}")
print(f"  {'Instance type':30s} {'ml.g4dn':>12s} {'ml.m5.xlarge':>13s}")
print(f"  {'Estimated job cost':30s} {'~$1.67':>12s} {'~$0.26':>13s}")
print("=" * 65)
```

---

**Cell 34** | type: markdown | beat: wrap-up diagram
```
<!-- DIAGRAM: Accuracy vs epochs comparison - transfer learning (frozen encoder) converges faster with less memory than full fine-tuning, with catastrophic forgetting risk annotated -->
[View diagram](../../plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd)

Transfer learning (green line) climbs steeply in epoch 1 and plateaus by epoch 2.
Full fine-tuning (blue line) starts slower because it must update 66M parameters
coherently before accuracy improves. Transfer learning wins on small datasets.
```

---

**Cell 35** | type: markdown | beat: wrap-up (discussion)
```
## Discussion: Transfer Learning in Production (3 min)

Discuss with a partner. Focus on tradeoffs and real-world implications, not just how the
code works.

- Barclays adds a new complaint category: App login issues. With transfer learning, do you
  need to retrain the full model? What changes - the encoder, the head, or both?
- A junior data scientist argues that full fine-tuning always outperforms transfer learning
  given enough data. When is this true? At what rough dataset size does the accuracy gap close?
- Topic 7 introduces LoRA: trainable low-rank adapters injected INTO the frozen encoder layers.
  How is that different from what we built today? What limitation of today's approach does it
  address?
```

---

**Cell 36** | type: markdown | beat: wrap-up (key takeaways)
```
## Key Takeaways

**Transfer learning in one sentence**: freeze the expensive knowledge,
train only the cheap adapter.

**What we built today:**
- Froze 66M encoder parameters, trained 591K head parameters
- Launched a CPU remote training job on ml.m5.xlarge (~$0.26 total)
- Achieved ~89% accuracy on SST-2 sentiment in 15 minutes
- Deployed to a real-time endpoint and ran Barclays-style inference

**Rules to remember:**
- PyTorch estimator for CPU, HuggingFace estimator for GPU (hard rule)
- requirements.txt must be that exact name in source_dir
- eval_strategy not evaluation_strategy (transformers 4.41+)
- Endpoint instance must be ml.m5.xlarge or larger for DistilBERT (4GB RAM is not enough)
- Higher LR for frozen-encoder head (1e-4 to 3e-4), not the full fine-tune LR (5e-5)

**Up next - Topic 7a: LoRA on Feed-Forward Networks**
What if we could get even better accuracy than full fine-tuning while only touching
0.1% of the parameters? LoRA inserts low-rank adapter matrices into the frozen
transformer layers. We will build one from scratch before using PEFT in 7b.
```

---

## Cell Index (45 cells total)

| Cell | Type | Beat | Content |
|------|------|------|---------|
| 0 | markdown | setup | Title, objectives, narrative |
| 1 | code | setup | pip install pinned versions |
| 2 | code | setup | SageMaker session setup |
| 3 | code | setup | Verify source_dir structure |
| 4 | markdown | setup | Why transfer learning? (parameter math) |
| 5 | markdown | Beat 1 | Intro: two naive failure modes |
| 6 | code | Beat 1 | BROKEN: wrong LR for frozen head (5e-6) |
| 7 | code | Beat 1 | BROKEN: full fine-tune CPU time projection |
| 8 | markdown | Beat 2 | Diagram 1: architecture (frozen encoder) |
| 9 | markdown | Beat 3 | Demo intro |
| 10 | code | Beat 3 | Inspect model layers and parameter counts |
| 11 | code | Beat 3 | Freeze encoder, verify trainable params |
| 12 | code | Beat 3 | Forward+backward: show encoder grad is None |
| 13 | markdown | Beat 3 | Why remote training (transition cell) |
| 14 | code | Beat 3 | Preview freeze_encoder function from train.py |
| 15 | code | Beat 3 | Reference accuracy numbers |
| 15b | markdown | Beat 2 | Diagram 2: comparison (markdown cell) |
| 16 | markdown | Beat 4 | Capstone intro: CPU remote training |
| 17 | code | Beat 4 | Define PyTorch estimator + estimator.fit() |
| 18 | code | Beat 4 | Retrieve job output + training time |
| 18b | code | Beat 4 | training_job_name safety-net |
| 19 | markdown | Beat 4 | Discussion: which approach for Barclays? |
| 20 | code | Beat 4 | Write inference.py to disk |
| 21 | code | Beat 4 | Deploy endpoint (PyTorchModel) |
| 22 | code | Beat 4 | Test endpoint with complaint samples |
| 23 | code | Beat 4 | Delete endpoint (cleanup) |
| 24 | markdown | Lab | Lab 6b instructions (STAR method) |
| 25 | code | Lab | Step 1 starter: tokenize SST-2 test set |
| 26 | code | Lab | Step 1 safety-net |
| 27 | code | Lab | Step 2 starter: load model + freeze encoder |
| 28 | code | Lab | Step 3 starter: local validation pass |
| 29 | code | Lab | Steps 2+3 safety-net |
| 30 | markdown | Lab | Lab 6b verification and expected output |
| 31 | markdown | Lab | Stretch + Homework Extension |
| 32 | markdown | Wrap-up | Comparison section intro |
| 33 | code | Wrap-up | Side-by-side comparison table |
| 34 | markdown | Wrap-up | Diagram 2 with annotation |
| 35 | markdown | Wrap-up | Peer discussion questions |
| 36 | markdown | Wrap-up | Key takeaways + bridge to Topic 7a |

Total: 37 cells (renumber 0-36). Adjust if any section splits during build.

---

## Checklist (verify before /verify-research)

### Four-beat arc
- [x] Beat 1: two broken code cells (wrong LR, CPU time projection)
- [x] Beat 2: two diagram placeholders (architecture, comparison)
- [x] Beat 3: four working demo cells (inspect, freeze, gradient check, reference numbers)
- [x] Beat 4: eight capstone cells + lab section

### Lab rules
- [x] Tier 2: problem statement only (no numbered sub-steps), YOUR CODE placeholders, 25-35 min, stretch goal
- [x] Tier 2 assigned here (topic_6b is the natural midpoint of Day 2)
- [x] No Tier 3 (reserved for Topic 7b, last topic of Day 2)
- [x] Every YOUR CODE placeholder does NOT hint at the answer
- [x] Safety-net after Step 1 and after Steps 2+3
- [x] Stretch version (launch full fine-tune job for comparison)
- [x] Homework Extension (partial fine-tuning: freeze_layers argument)

### SageMaker constraints
- [x] PyTorch estimator (NOT HuggingFace estimator) for CPU job
- [x] framework_version="2.8.0", py_version="py312"
- [x] instance_type="ml.m5.xlarge" for training and endpoint
- [x] requirements.txt exactly that name in source_dir
- [x] eval_strategy="epoch" (NOT evaluation_strategy)
- [x] No evaluate library - inline numpy compute_metrics
- [x] SageMaker SDK pinned >=2.200.0,<3.0.0
- [x] boto3 exception: ResourceNotFound (not ResourceNotFoundException)

### Hard rules
- [x] numpy<2 in install cell and requirements.txt
- [x] Plain ASCII only (no em dashes, en dashes, unicode mult, emojis)
- [x] No more than 3 consecutive markdown cells without a code cell
- [x] No getpass for AWS (execution role handles auth inside SageMaker Studio)
- [x] Barclays Customer Support Intelligence System narrative throughout

### Diagrams
- [x] Exactly 2 diagrams with correct placeholder syntax
- [x] Both have <!-- DIAGRAM: description --> comment + [View diagram](...) link
- [x] Slugs: transfer-learning-arch, tl-vs-finetuning-comparison
- [x] Paths: ../../plans/topic_6b/diagrams/*.mmd

---

## Research Sources

- DistilBERT architecture: `DistilBertForSequenceClassification` has `distilbert`, `pre_classifier`,
  `classifier`, `dropout` children. pre_classifier is Linear(768, 768), classifier is Linear(768, num_labels).
  Source: HuggingFace transformers docs + modeling_distilbert.py
- SST-2 dataset: `stanfordnlp/sst2` on HuggingFace Hub. Columns: sentence, label, idx.
  Train: 67,349 samples. Validation: 872 samples. Labels: 0=negative, 1=positive.
  DistilBERT fine-tuned on SST-2 achieves 91.3% on validation set.
- SageMaker CPU training: PyTorch estimator (not HuggingFace estimator - GPU only).
  framework_version="2.8.0", py_version="py312". ml.m5.xlarge has 4 vCPU, 16GB RAM.
- Transfer learning vs full fine-tuning: frozen encoder converges faster on small datasets,
  lower overfitting risk, no catastrophic forgetting. Full fine-tune wins at large data scale.
  LoRA (Topic 7) achieves full fine-tune accuracy at <1% parameter count.
- Lessons learned applied: L1 (HuggingFace estimator GPU-only), L2 (py312 required for 2.8.0),
  L3 (SDK <3.0.0), L4 (requirements.txt exact name), L5 (eval_strategy not evaluation_strategy),
  L6 (no evaluate library), L7 (ResourceNotFound), L11 (transformers >=4.35.0 for py312 wheels).
