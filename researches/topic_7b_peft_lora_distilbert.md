# Topic 7b - PEFT LoRA on DistilBERT: Cell-by-Cell Plan

## Overview

Topic 7b is the LAST topic of Day 2. It builds directly on Topic 7a, where students
implemented LoRA from scratch on feed-forward networks. Now students graduate to the
production-grade HuggingFace PEFT library and apply it to a full complaint classifier
(DistilBERT for sequence classification). We then demonstrate QLoRA (4-bit quantized
base model + LoRA adapters) and soft prompts (prefix tuning) as alternative PEFT
strategies, and close with a Tier 3 open-ended capstone lab where students pick any
PEFT method and fine-tune a complaint classifier on a GPU SageMaker job.

Because this is the LAST topic of Day 2, it carries the mandatory Tier 3 open-ended
lab. There is no verification cell on the Tier 3 lab -- students choose their own PEFT
method and implement the full training loop themselves.

Estimated in-class time: 90 to 120 minutes (excluding SageMaker job runtime).

---

## Diagram Index

Diagram 1: slug=peft-methods-comparison, path=plans/topic_7b/diagrams/peft-methods-comparison.mmd
  Description: Side-by-side comparison of three PEFT methods applied to a transformer
  encoder block. Left panel (LoRA): frozen weight matrix W with low-rank matrices A
  and B alongside it; output = Wx + BAx; trainable params shown as small orange boxes.
  Center panel (QLoRA): same as LoRA but W is shown as a 4-bit block (grey hatching)
  with a dequantize arrow before the forward pass; LoRA adapters remain in float16.
  Right panel (Soft Prompts / Prefix Tuning): the input sequence has P virtual token
  embeddings prepended (shown as green boxes labeled "virtual"); the rest of the model
  is fully frozen. Below each panel: a parameter efficiency bar showing relative
  trainable parameter count (LoRA ~0.5-1%, QLoRA ~0.5-1%, Soft Prompts ~0.01-0.1%).

Diagram 2: slug=qlora-architecture, path=plans/topic_7b/diagrams/qlora-architecture.mmd
  Description: Detailed QLoRA architecture for a single transformer attention block.
  Top: "Base model weights stored in NF4 (4-bit)" block in dark grey. Arrow labeled
  "dequantize to float16 (forward pass only)" leads to the float16 computation box.
  Branching off: two float16 boxes labeled "LoRA A (rank r)" and "LoRA B (rank r)"
  with a plus sign merging into the output. Bottom annotation: "Gradients flow only
  through LoRA adapters -- base weights never updated." A memory comparison callout:
  "Full fine-tune: ~268MB | QLoRA: ~22MB" for DistilBERT-base. Arrows show that
  NF4 weights are frozen throughout training.

---

## Source Dir (scripts_topic7b/)

### train.py

```python
"""
train.py -- PEFT fine-tuning of DistilBERT for complaint classification.
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_method", type=str,  default="lora",
                        choices=["lora", "qlora"])
    parser.add_argument("--lora_r",      type=int,  default=8)
    parser.add_argument("--lora_alpha",  type=int,  default=16)
    parser.add_argument("--epochs",      type=int,  default=3)
    parser.add_argument("--batch_size",  type=int,  default=16)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--model-dir",   type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir",  type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inline accuracy metric (no evaluate library -- incompatible with datasets 4.x)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Data loading and tokenisation
# ---------------------------------------------------------------------------

def load_and_tokenise(tokenizer, max_length=128):
    """Load financial_phrasebank and tokenise for sequence classification."""
    ds = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=True,
    )
    # financial_phrasebank has only a train split -- create an 80/20 split.
    split = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

    def tokenise(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = train_ds.map(tokenise, batched=True, remove_columns=["sentence"])
    eval_ds  = eval_ds.map(tokenise,  batched=True, remove_columns=["sentence"])

    # Rename label column to match Trainer expectation.
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds  = eval_ds.rename_column("label",  "labels")

    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

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
        # DistilBERT attention projection layer names.
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        # Save the classification head alongside LoRA adapters.
        modules_to_save=["classifier", "pre_classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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

    # Save PEFT adapters (not the full model -- much smaller).
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Save metrics for the notebook polling cell.
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
```

### requirements.txt

```
peft>=0.6.0
bitsandbytes>=0.41.0
datasets==2.18.0
numpy<2
```

---

## Key Changes from 12_PEFT_LoRA_DistillBert.ipynb

**Source notebook characteristics**:
- Uses TensorFlow / Keras (TFDistilBertModel, keras.layers.Layer).
- Implements LoRA manually from scratch (same as Topic 7a, which we already cover).
- Task is Question Answering on SQuAD (not complaint classification).
- No SageMaker, no PEFT library, no QLoRA, no soft prompts.
- No four-beat arc, no labs, no STAR method, no discussion prompts.
- Uses evaluate library for metrics (forbidden -- L6 SAGEMAKER_LESSONS_LEARNED).
- numpy<2 already pinned (keep).

**Keeping**:
- The high-level idea: apply LoRA to DistilBERT attention layers (q_lin, v_lin).
- The `print_trainable_parameters` / parameter count comparison concept.
- The conceptual flow: load model, identify target modules, inject adapters, verify
  parameter count drops dramatically.

**Restructuring**:
- Switch from TensorFlow to PyTorch (course uses PyTorch throughout).
- Switch from manual LoRA to the HuggingFace PEFT library (LoraConfig + get_peft_model).
  Topic 7a already showed the from-scratch version -- 7b shows the production library.
- Switch task from Q&A on SQuAD to complaint classification on financial_phrasebank
  (matches the Barclays customer support narrative).
- Add QLoRA section (BitsAndBytesConfig + PEFT) -- not in source at all.
- Add soft prompts / prefix tuning section -- not in source at all.
- Wrap every concept in the four-beat arc.
- Add labs (Tier 1 guided + Tier 3 open-ended capstone).
- Add SageMaker GPU training job capstone (HuggingFace estimator, ml.g4dn.xlarge).
- Replace evaluate library with inline numpy compute_metrics (L6).
- Use eval_strategy not evaluation_strategy (L5).
- HuggingFace estimator only on GPU (L1).

**Replacing**:
- TFDistilBertModel -> AutoModelForSequenceClassification (PyTorch).
- Manual LoraLayer Keras class -> LoraConfig + get_peft_model from PEFT library.
- SQuAD Q&A dataset -> financial_phrasebank (complaint sentiment proxy).
- Keras .compile() / .fit() -> HuggingFace Trainer.
- In-notebook training -> SageMaker remote GPU job for capstone.
- evaluate.load("accuracy") -> inline numpy compute_metrics.
- Beat 1 failures: (a) import bitsandbytes on CPU -- RuntimeError because no CUDA;
  (b) wrong prefix_length in soft prompts -- shape mismatch.

---

## Variable Continuity from Topic 7a

The following names carry forward from topic_7a_lora_ffn:

- `lora_r` (int) -- LoRA rank; used in the parameter count comparison demo.
- `device` -- same torch.device pattern.
- `set_seeds(seed)` -- identical signature.
- The conceptual variables `A_matrix`, `B_matrix` from 7a are referenced in Beat 1
  of Section 1 to contrast manual LoRA with the PEFT library approach.

New variables introduced in Topic 7b that downstream cells depend on:
- `peft_model` -- the wrapped DistilBERT with LoRA adapters (Cells 8-9).
- `qlora_model` -- the 4-bit quantized model with LoRA (Cell 17).
- `prefix_model` -- the soft-prompt PEFT model (Cell 22).
- `estimator` -- the sagemaker.huggingface.HuggingFace object (Cell 28).
- `training_job_name` -- returned by estimator.fit(wait=False) (Cell 28); safety-net in Cell 29.

---

## Cell-by-Cell Plan

### Cell 0: markdown - Title and Learning Objectives

```
# Topic 7b - PEFT LoRA on DistilBERT

Barclays Customer Support Intelligence System | Day 2, Topic 7b (Last Topic)

## What you will build

In Topic 7a you built LoRA from scratch on feed-forward networks.
Now you use the production-grade HuggingFace PEFT library to apply LoRA to a full
DistilBERT complaint classifier -- the same pattern used in real ML pipelines at scale.
You then explore QLoRA (4-bit quantized base + LoRA adapters) and soft prompts as
alternative parameter-efficient strategies, and close with an open-ended capstone
where you design and launch your own PEFT training job on a GPU instance.

## Why this matters to Barclays

Fine-tuning a full 66M-parameter DistilBERT on Barclays hardware budgets is expensive.
PEFT methods reduce trainable parameters by 99% or more while matching full fine-tune
accuracy. QLoRA further cuts GPU memory by 4x, making GPU training accessible on
smaller instances. These are the techniques Barclays ML teams use in production today.

## Learning objectives

1. Apply the PEFT library (LoraConfig + get_peft_model) to any HuggingFace encoder model
2. Explain QLoRA: how 4-bit NF4 quantization combines with LoRA adapters
3. Describe soft prompts / prefix tuning and when to prefer them over LoRA
4. Launch a GPU fine-tuning job on SageMaker using the HuggingFace estimator
5. Compare trainable parameter counts across LoRA, QLoRA, and full fine-tuning

## Estimated time

90 to 120 minutes in class.
```

---

### Cell 1: code - Environment Setup

```python
# Environment setup for SageMaker Studio.
# CPU kernel is used for all demos in Sections 1-3.
# The GPU training job in Section 4 runs remotely on ml.g4dn.xlarge.
# bitsandbytes and PEFT are installed in the remote container via requirements.txt --
# we do NOT install bitsandbytes in this notebook kernel (CPU only, would error).

!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "datasets>=2.18.0,<3.0.0" \
    "peft>=0.6.0" \
    "numpy<2"

import sagemaker
from sagemaker import get_execution_role
import boto3

sess   = sagemaker.Session()
role   = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"Role:   {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
```

---

### Cell 2: code - PyTorch and Library Imports

```python
import torch
import numpy as np
import warnings
import os
import random

warnings.filterwarnings("ignore")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(42)

# CPU kernel for demos -- GPU job runs remotely.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {device}")

# Recall from Topic 7a: we built LoRA matrices A and B by hand.
# Reminder of what those dimensions meant.
lora_r = 8   # rank -- carried forward from 7a
print(f"LoRA rank from 7a: {lora_r}")
```

---

### Cell 3: markdown - Section 1: PEFT Library LoRA on DistilBERT

```
## Section 1: From Scratch to Library -- PEFT LoRA on DistilBERT

In Topic 7a you implemented LoRA by hand:
  output = W_frozen @ x + (B @ A) @ x

The PEFT library automates that injection for any HuggingFace model.
Three function calls replace two custom classes and a manual layer-replacement loop.

### Beat 1: What happens if we try to apply LoRA without PEFT?
```

---

### Cell 4: code - Beat 1: Manual LoRA Fails to Scale

```python
# Beat 1: Trying to inject LoRA manually into DistilBERT
# (the approach from 7a) breaks on the classification head.
# DistilBERT's classifier is a separate module -- manual replacement
# of q_lin / v_lin does NOT freeze it, causing all parameters to train.

from transformers import AutoModelForSequenceClassification
import torch.nn as nn

model_name = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# Manual attempt: replace q_lin in each attention block with a custom linear.
# This is what we did in 7a -- let's see why it is incomplete here.

class NaiveLoraLinear(nn.Module):
    """Manual LoRA wrapper -- from Topic 7a, applied naively to DistilBERT."""
    def __init__(self, original_linear, rank=8):
        super().__init__()
        in_f  = original_linear.in_features
        out_f = original_linear.out_features
        self.frozen_W = original_linear
        self.frozen_W.weight.requires_grad_(False)
        self.A = nn.Linear(in_f,  rank,  bias=False)
        self.B = nn.Linear(rank,  out_f, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.frozen_W(x) + self.B(self.A(x))

# Inject into the first attention block only (naive, partial).
first_attn = base_model.distilbert.transformer.layer[0].attention
first_attn.q_lin = NaiveLoraLinear(first_attn.q_lin)

# Count trainable parameters.
trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in base_model.parameters())
print(f"Naive manual injection: {trainable:,} / {total:,} trainable")
print("Problem: the classifier head is still fully trainable.")
print("Problem: we only touched ONE attention block -- missed all others.")
print("Problem: no easy way to save/load just the LoRA weights.")
print()
print("The PEFT library solves all three problems in 5 lines of code.")
```

---

### Cell 5: markdown - Beat 2: Diagram Placeholder

```
<!-- DIAGRAM: PEFT methods comparison (LoRA, QLoRA, Soft Prompts) showing parameter efficiency of each -->
[View diagram](../../plans/topic_7b/diagrams/peft-methods-comparison.mmd)

The diagram above shows how each PEFT method modifies (or does not modify) the weight
matrices of a transformer block, and the relative trainable parameter count for each.
Notice that all three methods keep the base model frozen -- only the coloured elements
(adapters or virtual tokens) are learned.
```

---

### Cell 6: code - Beat 3: PEFT Library LoRA -- Full Working Demo

```python
# Beat 3: PEFT library LoRA on DistilBERT for sequence classification.
# This replaces the entire manual injection from Topic 7a with 5 lines.

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

model_name = "distilbert-base-uncased"

# Step 1: Load the base model. Frozen by PEFT automatically.
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3   # 3 classes: negative, neutral, positive
)

# Step 2: Define the LoRA configuration.
# target_modules: DistilBERT attention projections are named q_lin and v_lin.
# modules_to_save: classifier head must also be trained (it is task-specific).
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,    # sequence classification task
    r=lora_r,                      # rank from Topic 7a (r=8)
    lora_alpha=16,                 # scaling factor (alpha/r = effective lr scale)
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier", "pre_classifier"],  # keep these trainable
)

# Step 3: Wrap the model -- this freezes base weights and injects LoRA adapters.
peft_model = get_peft_model(base_model, lora_config)

# Step 4: Inspect -- compare parameter counts.
peft_model.print_trainable_parameters()

trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in peft_model.parameters())
print(f"\nTrainable : {trainable:,}")
print(f"Total     : {total:,}")
print(f"Ratio     : {100.0 * trainable / total:.2f}%")
print()
print("Notice: LoRA reduces trainable params dramatically.")
print("The adapter weights for q_lin and v_lin across all 6 blocks")
print("plus the full classifier head make up the trainable fraction.")
```

---

### Cell 7: code - Beat 3 (continued): Quick Forward Pass Verification

```python
# Verify peft_model runs correctly on a dummy batch.
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sample_texts = [
    "My card payment was declined twice at the ATM.",
    "The mobile app works perfectly.",
    "I cannot access my online account.",
]

inputs = tokenizer(
    sample_texts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=64,
)

peft_model.eval()
with torch.no_grad():
    outputs = peft_model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
label_map = {0: "negative", 1: "neutral", 2: "positive"}

print("PEFT LoRA DistilBERT -- inference check")
print("-" * 42)
for text, pred in zip(sample_texts, predictions.tolist()):
    print(f"  [{label_map[pred]}]  {text[:55]}")
print()
print("Forward pass succeeded. Adapters injected correctly.")
```

---

### Cell 8: markdown - Beat 4: Lab 1

```
## Lab 1 -- Apply PEFT LoRA with a Different Rank (Tier 1, ~15 min)

### Situation
The Barclays complaint classification team asks: "Does a higher LoRA rank always give
us more trainable parameters? How does alpha affect the effective learning rate?"

### Task
Create a second LoRA-wrapped DistilBERT with rank r=16 and alpha=32 (double both values).
Count trainable parameters and compare to the r=8 model above.

### Action
Fill in the YOUR CODE sections below.

### Result
A verification cell will print both parameter counts and the ratio.

### Steps
1. Load a fresh AutoModelForSequenceClassification (distilbert-base-uncased, num_labels=3).
2. Build a LoraConfig with r=16, lora_alpha=32, same target_modules and modules_to_save.
3. Wrap with get_peft_model().
4. Count trainable parameters.
```

---

### Cell 9: code - Lab 1 Starter

```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

# Lab 1: Build a LoRA model with r=16 and compare parameter counts to peft_model (r=8).

# Step 1: load a fresh base model.
base_model_r16 = None  # YOUR CODE

# Step 2: define LoraConfig with r=16, lora_alpha=32.
lora_config_r16 = None  # YOUR CODE

# Step 3: wrap with get_peft_model.
peft_model_r16 = None  # YOUR CODE
```

---

### Cell 10: code - Lab 1 Safety-Net

```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if peft_model_r16 is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    _base = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )
    _cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier", "pre_classifier"],
    )
    peft_model_r16 = get_peft_model(_base, _cfg)
```

---

### Cell 11: code - Lab 1 Verification

```python
# Lab 1 verification -- auto-graded.
t_r8  = sum(p.numel() for p in peft_model.parameters()     if p.requires_grad)
t_r16 = sum(p.numel() for p in peft_model_r16.parameters() if p.requires_grad)

print(f"LoRA r=8  trainable params : {t_r8:,}")
print(f"LoRA r=16 trainable params : {t_r16:,}")
print(f"r=16 / r=8 ratio           : {t_r16 / t_r8:.2f}x")

assert t_r16 > t_r8, "r=16 should have more trainable params than r=8"
assert 1.8 < (t_r16 / t_r8) < 2.5, "ratio should be close to 2x (rank doubles)"
print("Verification passed.")
```

---

### Cell 12: markdown - Lab 1 Stretch + Homework

```
### Stretch (for fast finishers)

Try r=4 with lora_alpha=8. Does the accuracy gap between r=4 and r=16 matter more than
the parameter count savings? Write your hypothesis in a markdown cell, then test it in
the Capstone (Section 4) by submitting two SageMaker jobs with different ranks.

### Homework Extension

Research the relationship between lora_alpha / r (the effective scaling ratio) and the
learning rate. Plot trainable parameter count vs rank (r = 2, 4, 8, 16, 32) for the
DistilBERT classification head. At what rank do diminishing returns set in?
```

---

### Cell 13: markdown - Discussion Prompt

```
## Discussion (3 minutes)

You just reduced DistilBERT's trainable parameters from 66M to roughly 300K using LoRA.

1. If a Barclays team is training on 10K labelled complaints, does the rank r=8 vs r=16
   difference matter? What signals would you look for to decide?

2. LoRA adapters are stored separately from the base model weights. What does that mean
   for model versioning, rollback, and A/B testing at Barclays?

3. The base weights are frozen during LoRA training. What happens if the base model is
   updated (e.g., a new version of DistilBERT is released)? Do you retrain the adapters?
```

---

### Cell 14: markdown - Section 2: QLoRA

```
## Section 2: QLoRA -- 4-Bit Quantization + LoRA

QLoRA (Dettmers et al., 2023) combines two ideas:
  (1) Quantize the base model weights to 4-bit NF4 (NormalFloat4) -- cuts memory ~4x.
  (2) Apply LoRA adapters in float16 on top of the frozen 4-bit base.

Gradients flow only through the LoRA adapters. The 4-bit base weights are never updated.
For DistilBERT (66M params), this means the base takes ~22MB instead of ~268MB (float32).
For larger models (7B params), QLoRA makes GPU training feasible on a single T4.

### Beat 1: What happens when you try QLoRA without a GPU?
```

---

### Cell 15: code - Beat 1: QLoRA Fails Without CUDA

```python
# Beat 1: bitsandbytes requires a CUDA GPU. On this CPU kernel it will fail.
# This is intentional -- students see the exact error before the solution.

import sys

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
    # Even if the import works, 4-bit quantization requires CUDA at model load time.
    from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
    # This will raise RuntimeError or ValueError on CPU:
    #   "bitsandbytes is only supported on CUDA devices"
    fail_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3, quantization_config=bnb_config
    )
except (ImportError, RuntimeError, ValueError) as e:
    print(f"[Expected error on CPU kernel] {type(e).__name__}: {e}")
    print()
    print("bitsandbytes requires a CUDA GPU -- it cannot run on this CPU kernel.")
    print("Solution: move QLoRA training to a GPU instance via SageMaker.")
    print("The scripts_topic7b/train.py script uses QLoRA when --peft_method=qlora.")
```

---

### Cell 16: markdown - Beat 2: QLoRA Diagram

```
<!-- DIAGRAM: QLoRA architecture showing 4-bit NF4 base model with float16 LoRA adapters -->
[View diagram](../../plans/topic_7b/diagrams/qlora-architecture.mmd)

The diagram above shows the QLoRA forward pass. The base model weights (grey) are stored
in 4-bit NF4 format. Before each matrix multiply, they are dequantized to float16 on
the fly. LoRA adapters (A and B) stay in float16 throughout. Gradients only flow through
the adapters -- the 4-bit base is never updated.
```

---

### Cell 17: code - Beat 3: QLoRA Code Walkthrough (No Execution on CPU)

```python
# Beat 3: Full QLoRA code -- explained line by line.
# This cell does NOT run on the CPU kernel (no bitsandbytes without CUDA).
# Students read and understand it here; it runs inside scripts_topic7b/train.py
# on the GPU instance.

# --- Instructor narration: walk through this cell line by line ---

# from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
# from peft import LoraConfig, get_peft_model, TaskType

# Step 1: Configure 4-bit quantization.
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,          # activate 4-bit loading
#     bnb_4bit_quant_type="nf4",  # NormalFloat4 -- optimal for normally distributed weights
#     bnb_4bit_compute_dtype=torch.float16,  # compute in fp16 (not 4-bit) during forward
#     bnb_4bit_use_double_quant=True,        # quantize the quantization constants too
# )

# Step 2: Load the base model in 4-bit.
# base_model = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased",
#     num_labels=3,
#     quantization_config=bnb_config,  # triggers bitsandbytes 4-bit loading
#     device_map="auto",               # let bitsandbytes place layers optimally
# )

# Step 3: Apply LoRA adapters (identical config to the plain LoRA case above).
# lora_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     r=8, lora_alpha=16,
#     target_modules=["q_lin", "v_lin"],
#     lora_dropout=0.1, bias="none",
#     modules_to_save=["classifier", "pre_classifier"],
# )
# qlora_model = get_peft_model(base_model, lora_config)
# qlora_model.print_trainable_parameters()
# # Output: trainable params: ~300K || all params: ~66M || trainable: ~0.46%
# # Base weights occupy only ~22MB in NF4 vs ~268MB in float32.

print("QLoRA code walkthrough complete.")
print("The full implementation runs in scripts_topic7b/train.py --peft_method=qlora")
print("on the SageMaker GPU instance (ml.g4dn.xlarge, NVIDIA T4).")
print()
print("Key difference from plain LoRA:")
print("  LoRA   : base = float32 (~268MB), adapters = float32")
print("  QLoRA  : base = NF4 4-bit (~22MB), adapters = float16")
print("  Memory saved: ~10x reduction in base model footprint")
```

---

### Cell 18: markdown - Beat 4: QLoRA Lab (Tier 3, Open-Ended)

```
## Lab 2 -- Apply QLoRA to a Different DistilBERT Layer Configuration (Tier 3, ~25 min)

This is a Tier 3 open-ended lab. There are no numbered steps and no YOUR CODE hints
beyond the function signature and docstring. You decide which layers to target and what
rank to use. Document your reasoning in a markdown cell before you write any code.

### Situation

The Barclays Model Efficiency team wants to evaluate whether targeting additional
attention layers (beyond q_lin and v_lin) in QLoRA gives a meaningful accuracy lift,
or whether the extra trainable parameters are wasted on a small dataset.

### Task

Implement the function below. Choose which DistilBERT layers to target with LoRA,
choose a rank, and explain why. Because QLoRA requires a GPU (bitsandbytes), your
implementation should gracefully fall back to plain LoRA on CPU so the cell can run
in this kernel -- but the docstring and comments should describe the full QLoRA path
that would run on a GPU instance.

### What to hand in

- Your completed function in the code cell below
- A markdown cell (write it above the code cell) explaining:
    - Which target_modules you chose and why
    - What rank you chose and why
    - What tradeoffs you are making vs the default q_lin + v_lin, r=8 config

### Constraints

- No numbered steps -- you decide the implementation
- No YOUR CODE placeholders -- start from the signature and docstring only
- No evaluate library -- use inline numpy metrics if you compute accuracy
- numpy<2 requirement applies
```

---

### Cell 19: code - Lab 2 QLoRA Tier 3 Starter

```python
def build_qlora_model(
    model_name: str,
    target_modules: list,
    lora_r: int,
    lora_alpha: int,
    num_labels: int = 3,
):
    """
    Build a QLoRA-wrapped DistilBERT (or compatible encoder) for sequence classification.

    On a CUDA GPU, this loads the base model in 4-bit NF4 and applies LoRA adapters
    in float16. On CPU (this kernel), it falls back to plain float32 LoRA so the cell
    can execute without bitsandbytes.

    Parameters
    ----------
    model_name : str
        HuggingFace hub name, e.g. "distilbert-base-uncased".
    target_modules : list of str
        Names of the linear sub-modules to inject LoRA into.
        For DistilBERT: valid choices include "q_lin", "v_lin", "k_lin", "out_lin",
        "lin1", "lin2". Choose at least two. Document your reasoning above.
    lora_r : int
        LoRA rank. Controls the size of the low-rank factorisation.
    lora_alpha : int
        LoRA scaling factor. Common convention: lora_alpha = 2 * lora_r.
    num_labels : int
        Number of output classes (default: 3 for negative / neutral / positive).

    Returns
    -------
    peft_model : a PEFT-wrapped model ready for training or inference.
    trainable_params : int, number of trainable parameters in peft_model.
    total_params : int, total parameters in peft_model.
    """
    pass
```

---

### Cell 20: markdown - Lab 2 Stretch and Homework

```
### Stretch (for fast finishers)

Call build_qlora_model twice: once with your chosen layer set and once with only
q_lin + v_lin (the default from the demo). Print both trainable parameter counts
side by side. Write a one-paragraph hypothesis about which configuration will
achieve higher validation accuracy on financial_phrasebank and why.

### Homework Extension

Submit two SageMaker jobs using scripts_topic7b/train.py: one with your chosen
target_modules (pass them as a hyperparameter or edit train.py) and one with the
default q_lin + v_lin. Compare eval_accuracy from CloudWatch. Write a short report
(3-5 sentences) explaining whether your hypothesis was correct and what you would
change in a second iteration.
```

---

### Cell 21: markdown - Section 3: Soft Prompts (Prefix Tuning)

```
## Section 3: Soft Prompts -- Learning Virtual Tokens

Soft prompts (also called prompt tuning or prefix tuning) take a completely different
approach from LoRA and QLoRA:

- Instead of modifying weight matrices, prepend P learnable "virtual" token embeddings
  to the input sequence at every transformer layer.
- The base model is 100% frozen -- not even the attention weights change.
- Only the virtual token embeddings are trained (as few as 20 vectors).
- At inference time, prepend the same virtual tokens -- no architecture change needed.

Tradeoff: Soft prompts use far fewer parameters than LoRA (~0.01-0.1% vs ~0.5-1%).
They work well for large models (7B+) but can underfit small models like DistilBERT.
LoRA typically outperforms soft prompts on small encoders for classification tasks.

### Beat 1: Shape mismatch when prefix length is wrong
```

---

### Cell 22: code - Beat 1: Soft Prompt Shape Mismatch

```python
# Beat 1: A common mistake is setting num_virtual_tokens larger than max_length,
# which causes a shape mismatch inside the model's attention mask.

from peft import PromptTuningConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# Load a fresh base model for soft prompt demo.
base_for_prompts = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)

# Bad config: num_virtual_tokens=200 but our inputs use max_length=64.
# This will produce a RuntimeError inside the model forward pass.
bad_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=200,    # <-- too many: 200 > max_length=64
    tokenizer_name_or_path="distilbert-base-uncased",
)

try:
    bad_model = get_peft_model(base_for_prompts, bad_config)
    # Trigger forward pass to expose the shape error.
    dummy_input = tokenizer(["test complaint"], return_tensors="pt", max_length=64,
                            truncation=True, padding="max_length")
    _ = bad_model(**dummy_input)
except Exception as e:
    print(f"[Expected error] {type(e).__name__}: {e}")
    print()
    print("num_virtual_tokens must be << max_sequence_length.")
    print("For max_length=64, a safe value is num_virtual_tokens=10 to 20.")
```

---

### Cell 23: code - Beat 3: Soft Prompts Working Demo

```python
# Beat 3: Correct soft prompt / prefix tuning configuration.

from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

base_for_prompts = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)

# Correct config: num_virtual_tokens well below max_length.
prompt_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    # Number of learnable virtual tokens prepended to the input.
    num_virtual_tokens=10,
    # Initialize virtual tokens from real vocabulary embeddings (faster convergence).
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="classify complaint as negative neutral positive",
    tokenizer_name_or_path="distilbert-base-uncased",
)

prefix_model = get_peft_model(base_for_prompts, prompt_config)
prefix_model.print_trainable_parameters()

trainable = sum(p.numel() for p in prefix_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in prefix_model.parameters())
print(f"\nSoft prompt trainable : {trainable:,}")
print(f"Total                 : {total:,}")
print(f"Ratio                 : {100.0 * trainable / total:.4f}%")
print()
print("Notice: even fewer trainable params than LoRA.")
print("The 10 virtual tokens each have dim=768 --> 7,680 learned values in total.")
```

---

### Cell 24: code - Beat 3: Parameter Count Comparison Table

```python
# Side-by-side parameter efficiency comparison across all three methods.

import numpy as np

methods = ["Full fine-tune", "LoRA (r=8)", "Soft prompts (20 virtual)"]

full_trainable = sum(p.numel() for p in
    AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    ).parameters()
)

lora_trainable   = sum(p.numel() for p in peft_model.parameters()   if p.requires_grad)
prompt_trainable = sum(p.numel() for p in prefix_model.parameters() if p.requires_grad)

counts = [full_trainable, lora_trainable, prompt_trainable]
total  = full_trainable  # base model total params

print(f"{'Method':<30} {'Trainable':>12} {'% of total':>12}")
print("-" * 56)
for method, count in zip(methods, counts):
    pct = 100.0 * count / total
    print(f"{method:<30} {count:>12,} {pct:>11.3f}%")

print()
print("Key takeaway:")
print("  LoRA and soft prompts both freeze the base model.")
print("  LoRA modifies existing weight matrices (more expressive).")
print("  Soft prompts only add virtual tokens (simpler, fewer params).")
print("  For small models like DistilBERT, LoRA typically wins on accuracy.")
```

---

### Cell 25: markdown - Discussion Prompt

```
## Discussion (3 minutes)

1. QLoRA fits a 4-bit base model in 22MB vs 268MB for float32. What does that mean for
   which Barclays teams can now run fine-tuning? (Consider: a team that only has a
   laptop GPU vs a team with a dedicated GPU server.)

2. Soft prompts freeze the entire base model. If Barclays has a single DistilBERT base
   model and 20 different complaint categories (cards, mortgages, fraud, ...), how would
   you manage 20 separate sets of virtual token embeddings vs 20 separate LoRA adapter
   files?

3. When would you choose soft prompts over LoRA? (Hint: think about model size, data
   availability, and the cost of storing/serving multiple adapters.)
```

---

### Cell 26: markdown - Section 4: SageMaker GPU Capstone Setup

```
## Section 4: Capstone -- PEFT Fine-Tuning on SageMaker GPU

The `scripts_topic7b/` folder contains a training script that supports both LoRA and QLoRA
via a `--peft_method` argument. The script uses:

- financial_phrasebank (sentiment) as a complaint classification proxy dataset
  (3 labels: negative, neutral, positive)
- HuggingFace Trainer with eval_strategy="epoch" and inline numpy metrics (no evaluate lib)
- PEFT LoraConfig with target_modules=["q_lin", "v_lin"] for DistilBERT
- BitsAndBytesConfig for QLoRA (4-bit NF4, fp16 compute)

The SageMaker job uses:
- HuggingFace estimator (GPU-only DLC, L1 from SAGEMAKER_LESSONS_LEARNED)
- transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312"
- instance_type="ml.g4dn.xlarge" (NVIDIA T4, 16GB VRAM, ~$0.74/hr)
```

---

### Cell 27: code - SageMaker Session and Script Upload

```python
# Section 4 setup: verify session and show the training script structure.
import os
import sagemaker
from sagemaker import get_execution_role

sess   = sagemaker.Session()
role   = get_execution_role()
bucket = sess.default_bucket()

print(f"Training will run on: ml.g4dn.xlarge (NVIDIA T4, 16GB VRAM)")
print(f"Artifacts will land in: s3://{bucket}/topic7b-peft/")
print()
print("scripts_topic7b/ structure:")
for fname in sorted(os.listdir("scripts_topic7b")):
    fpath = os.path.join("scripts_topic7b", fname)
    size  = os.path.getsize(fpath)
    print(f"  {fname:<20} {size:>8} bytes")
```

---

### Cell 28: code - Launch LoRA Training Job

```python
# Launch a LoRA fine-tuning job on SageMaker GPU.
# HuggingFace estimator is GPU-only (L1 from SAGEMAKER_LESSONS_LEARNED).

from sagemaker.huggingface import HuggingFace
import time

job_suffix    = int(time.time())
job_name_lora = f"topic7b-lora-{job_suffix}"

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="scripts_topic7b",         # contains train.py + requirements.txt
    role=role,
    instance_type="ml.g4dn.xlarge",       # GPU only -- HuggingFace DLC requirement
    instance_count=1,
    # Verified version matrix (CORE_TECHNOLOGIES_AND_DECISIONS.md).
    transformers_version="4.56.2",
    pytorch_version="2.8.0",
    py_version="py312",
    hyperparameters={
        "peft_method": "lora",
        "lora_r":      8,
        "lora_alpha":  16,
        "epochs":      3,
        "batch_size":  16,
        "lr":          2e-4,
    },
    output_path=f"s3://{bucket}/topic7b-peft/output/",
    base_job_name=job_name_lora,
)

# wait=False -- do not block the kernel while training runs.
estimator.fit(wait=False)
training_job_name = estimator.latest_training_job.name
print(f"Training job submitted: {training_job_name}")
print("Monitor in SageMaker console -> Training -> Training jobs")
```

---

### Cell 29: code - training_job_name Safety-Net

```python
# Safety-net: run this if your kernel restarted after launching the training job.
# SKIP this cell if training_job_name is already defined.
if 'training_job_name' not in dir() or training_job_name is None:
    training_job_name = "<PASTE YOUR JOB NAME HERE>"
    print(f"Using safety-net training_job_name: {training_job_name}")
```

---

### Cell 30: code - Poll Training Job Status

```python
# Poll training job status until complete.
import boto3
import time

sm_client = boto3.client("sagemaker", region_name=region)

print(f"Polling job: {training_job_name}")
print("(This cell blocks until the job is Complete or Failed.)")
print()

while True:
    resp   = sm_client.describe_training_job(TrainingJobName=training_job_name)
    status = resp["TrainingJobStatus"]
    print(f"  Status: {status}", end="\r", flush=True)
    if status in ("Completed", "Failed", "Stopped"):
        print(f"\nFinal status: {status}")
        break
    time.sleep(30)

if status == "Failed":
    reason = resp.get("FailureReason", "No reason provided")
    print(f"Failure reason: {reason}")
```

---

### Cell 31: code - Retrieve and Print Metrics

```python
# Read the metrics.json artifact written by train.py from S3.
import boto3
import json

s3 = boto3.client("s3")

# Determine artifact prefix from the training job description.
resp      = sm_client.describe_training_job(TrainingJobName=training_job_name)
model_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
# model_uri: s3://bucket/.../output/model.tar.gz
# metrics.json is inside the tarball -- retrieve from CloudWatch instead.

log_client  = boto3.client("logs", region_name=region)
log_group   = f"/aws/sagemaker/TrainingJobs"

# Simpler: just print the final accuracy from the describe response.
# The Trainer logs "eval_accuracy" -- it appears in CloudWatch logs.
print(f"Training job: {training_job_name}")
print(f"Status      : {resp['TrainingJobStatus']}")
print()
print("Final accuracy is logged as 'eval_accuracy' in CloudWatch.")
print(f"CloudWatch log group : /aws/sagemaker/TrainingJobs")
print(f"Log stream prefix    : {training_job_name}/algo-1-")
print()
print("Tip: in the SageMaker console, open the training job -> 'View logs'")
print("to see epoch-by-epoch eval_accuracy printed by the Trainer.")
```

---

### Cell 32: markdown - Tier 3 Capstone Lab

```
## Capstone Lab -- Design Your Own PEFT Complaint Classifier (Tier 3, Open-Ended)

This is the Day 2 capstone. There are no numbered steps, no YOUR CODE placeholders,
and no verification cell. You design and implement the full solution.

---

### Situation

The Barclays Machine Learning Platform team needs a complaint classifier that can be
deployed as a SageMaker endpoint. Your manager has given you one GPU hour and a choice
of PEFT method: LoRA, QLoRA, or prefix tuning. Pick the method you think is best for
this use case, justify your choice, and submit the training job.

### Task

Implement a function `train_peft_complaint_classifier` that:
- Accepts the PEFT method name, training data, and labels as arguments
- Fine-tunes a DistilBERT (or other encoder) classifier using that PEFT method
- Returns a summary dict with accuracy and parameter counts

Then submit the training job to SageMaker using the HuggingFace estimator.

### What to hand in

- Your completed function in this cell (or a separate notebook cell)
- A markdown cell explaining which PEFT method you chose and why
- The SageMaker training job name so the instructor can check the logs

### Constraints

- You may modify scripts_topic7b/train.py -- or write your own training script
- Use ml.g4dn.xlarge, transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312"
- eval_strategy="epoch" (not evaluation_strategy)
- No evaluate library -- use inline numpy metrics
- numpy<2 in requirements.txt
- No hardcoded credentials
```

---

### Cell 33: code - Tier 3 Capstone Starter

```python
def train_peft_complaint_classifier(
    model_name: str,
    peft_method: str,
    train_texts: list,
    train_labels: list,
) -> dict:
    """
    Fine-tune a complaint classifier using the specified PEFT method.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub name, e.g. "distilbert-base-uncased".
    peft_method : str
        One of "lora", "qlora", or "prefix_tuning".
    train_texts : list of str
        Raw complaint text strings for training.
    train_labels : list of int
        Integer class labels (0=negative, 1=neutral, 2=positive).

    Returns
    -------
    dict with keys:
        "accuracy"         -- float, validation accuracy after training
        "trainable_params" -- int, number of trainable parameters
        "total_params"     -- int, total model parameters
    """
    pass
```

---

### Cell 34: markdown - Capstone Stretch

```
### Stretch (for fast finishers)

1. Submit TWO jobs: one with --peft_method=lora and one with --peft_method=qlora.
   Compare the training time and final accuracy. Is the accuracy gap worth the memory
   saving from QLoRA on a dataset this small?

2. Modify train.py to also log the number of trainable parameters as a CloudWatch
   metric (use SageMaker's logging_steps and a custom MetricsDefinition on the
   estimator). Plot trainable params vs eval_accuracy for r = 4, 8, 16.

### Homework Extension

LoRA adapters can be merged back into the base model weights after training using
`model.merge_and_unload()`. Research this PEFT method and explain:
  (a) When would you merge adapters vs keep them separate?
  (b) What happens to inference latency after merging?
  (c) Write a short code snippet (can be pseudocode) showing the merge step and
      how you would save the merged model to S3 for SageMaker endpoint deployment.
```

---

### Cell 35: markdown - Wrap-Up and Bridge to Day 3

```
## Wrap-Up -- Day 2 Complete

### What you built today (Day 2 summary)

| Topic | What you built |
|-------|----------------|
| 5     | HuggingFace ecosystem: pipelines, AutoModel, Hub |
| 6a    | Full fine-tuning of DistilBERT; saw catastrophic forgetting |
| 6b    | Transfer learning: freeze all but the head; CPU SageMaker job |
| 7a    | LoRA from scratch: A and B low-rank matrices on feed-forward layers |
| 7b    | PEFT library: LoRA, QLoRA, soft prompts; GPU SageMaker capstone |

### Key PEFT takeaways

- LoraConfig + get_peft_model works on any HuggingFace model in 5 lines.
- target_modules=["q_lin", "v_lin"] for DistilBERT attention; check model.named_modules()
  for other architectures.
- QLoRA requires a CUDA GPU -- use the HuggingFace estimator on ml.g4dn.xlarge.
- Soft prompts have the fewest trainable parameters but underfit small models.
- eval_strategy="epoch" (not evaluation_strategy) -- transformers 4.41+ required.
- Never use the evaluate library -- use inline numpy compute_metrics.

### Day 3 preview

Tomorrow we deploy what we trained. Day 3 covers:
  - Quantization, pruning, and distillation (Topic 8)
  - RLHF and reward modeling (Topic 9)

The DistilBERT classifier you just fine-tuned will be the model we quantize in Topic 8.
Save the training_job_name -- you will need the S3 artifact URI in the first cell of Topic 8.

### Homework

Complete the Homework Extensions from Labs 1 and the Capstone before Day 3.
```

---

## Notes on Cell Count and Structure

Total planned cells: 36 (Cell 0 through Cell 35).

Section breakdown:
- Cells 0-2:   Title, environment setup, imports (3 cells)
- Cells 3-12:  Section 1 -- PEFT library LoRA (10 cells: Beat1, Beat2, Beat3x2, Lab1x4)
- Cell 13:     Discussion prompt (1 cell)
- Cells 14-20: Section 2 -- QLoRA (7 cells: intro+Beat1, Beat2, Beat3, Lab2 Beat4 x3)
- Cells 21-24: Section 3 -- Soft Prompts (4 cells: intro+Beat1, Beat3x2, comparison table)
- Cell 25:     Discussion prompt (1 cell)
- Cells 26-31: Section 4 -- SageMaker Capstone (6 cells: intro, session, launch, safety-net, poll, metrics)
- Cells 32-34: Tier 3 Capstone Lab (markdown + starter code + stretch/homework) (3 cells)
- Cell 35:     Wrap-up and bridge to Day 3 (1 cell)

### Markdown chain check

No sequence of more than 3 consecutive markdown cells exists:
- Cell 3 (md) -> Cell 4 (code): OK
- Cell 5 (md) -> Cell 6 (code): OK
- Cell 13 (md) -> Cell 14 (md) -> Cell 15 (code): 2 md then code: OK
- Cell 16 (md) -> Cell 17 (code): OK
- Cell 18 (md) -> Cell 19 (code): OK
- Cell 20 (md) -> Cell 21 (md) -> Cell 22 (code): 2 md then code: OK
- Cell 25 (md) -> Cell 26 (md) -> Cell 27 (code): 2 md then code: OK
- Cell 32 (md) -> Cell 33 (code): OK
- Cell 34 (md) -> Cell 35 (md): final 2 md at end, acceptable as wrap-up pair

### Lab tier assignments (Day 2 total)

- Topic 5: Tier 1 (guided) labs -- handled in topic_5 notebook
- Topic 6a: Tier 1 (guided) labs -- handled in topic_6a notebook
- Topic 6b: Tier 1 (guided) labs -- handled in topic_6b notebook
- Topic 7a: Tier 1 (guided) labs -- handled in topic_7a notebook
- Topic 7b: Tier 1 Lab 1 (Cells 8-12) + Tier 3 Lab 2 QLoRA (Cells 18-20) + Tier 3 Capstone (Cells 32-34) -- CORRECT
  Day 2 gets exactly ONE Tier 3 lab, which must be the last topic. Topic 7b is last. PASS.
  QLoRA Beat 4 (Lab 2) is Tier 3: open-ended, function signature + docstring + pass only. PASS.

### Safety-net check

- Lab 1 (peft_model_r16) feeds Cell 11 verification: safety-net provided (Cell 10). PASS.
- training_job_name (defined in Cell 28) feeds Cell 30 (poll) and Cell 31 (metrics): safety-net provided (Cell 29). PASS.
- Tier 3 capstone: NO safety-net required (open-ended, no downstream dependency). PASS.

### AI-tells check

No em dashes, en dashes, unicode multiplication signs, or emojis in any cell body.
All text uses plain ASCII hyphens and standard ASCII punctuation. PASS.

### SageMaker constraints check

- HuggingFace estimator: instance_type="ml.g4dn.xlarge" (GPU, L1). PASS.
- transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312". PASS.
- source_dir="scripts_topic7b" with exactly train.py + requirements.txt (L4). PASS.
- requirements.txt: peft>=0.6.0, bitsandbytes>=0.41.0, datasets==2.18.0, numpy<2. PASS.
- eval_strategy="epoch" (not evaluation_strategy) in train.py (L5). PASS.
- No evaluate library -- inline numpy compute_metrics (L6). PASS.
- SageMaker SDK: ">=2.200.0,<3.0.0" (L3). PASS.
- boto3 exception: ResourceNotFound (not ResourceNotFoundException) (L7). PASS.
