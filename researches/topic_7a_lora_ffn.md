# Topic 7a - LoRA on Feed-Forward Networks: Cell-by-Cell Plan

## Overview

Topic 7a is the second topic of Day 2 fine-tuning track and immediately follows Topic 6b
(Transfer Learning with DistilBERT). The narrative pivot is this: full fine-tuning (Topic 6a)
updates every parameter; transfer learning (Topic 6b) freezes most and trains a head; LoRA
goes further by updating only two tiny matrices per layer whose product approximates the
weight delta. Students implement LoRA from scratch in PyTorch, apply it to a feed-forward
network to see the math working concretely, then use the PEFT library to LoRA-fine-tune
Flan-T5-small on complaint summarization in a SageMaker GPU job.

Topic position: Day 2, fifth topic. Lab tier: Tier 1 (guided). No Tier 2 (Topic 4 used it).
No Tier 3 (Topic 7b gets it as last topic of Day 2).
Estimated in-class time: 90 to 120 minutes.

Running narrative: "Barclays Customer Support Intelligence System -- continued"
We have already fine-tuned Flan-T5 fully (Topic 6a) and frozen-then-adapted DistilBERT
(Topic 6b). Both approaches still store the full model. LoRA gives us a third option:
freeze everything, inject two tiny matrices A and B per target layer, train only those.
The weight update is delta_W = B @ A where B is d x r and A is r x k with r << min(d, k).
Trainable parameters drop from millions to thousands. We validate this on a simple FFN
before trusting it on a real LLM.

---

## Diagram Index

Diagram 1: slug=lora-decomposition, path=plans/topic_7a_lora_ffn/diagrams/lora-decomposition.mmd
  Description: LoRA low-rank decomposition of a single weight matrix. Left box shows the
  original frozen weight matrix W (d x k) shaded gray with label "Frozen W (no gradient)".
  Center shows the additive formula: W' = W + (B @ A). Right side shows two green boxes
  stacked: top box is B (d x r), bottom box is A (r x k), with label "r << min(d,k)".
  An arrow labelled "only B and A are trained" points to the two green boxes. Below the
  diagram, show a concrete example: d=512, k=1024, r=8. Full parameter count: 512*1024=524288.
  LoRA parameter count: 512*8 + 8*1024=12288. Label the ratio "~43x fewer parameters".
  Use neutral ASCII-style box notation in Mermaid (flowchart LR with subgraphs).

Diagram 2: slug=lora-parameter-comparison, path=plans/topic_7a_lora_ffn/diagrams/lora-parameter-comparison.mmd
  Description: Bar chart comparing trainable parameter counts across three fine-tuning
  strategies applied to Flan-T5-small (250M parameters total). Bar 1: "Full Fine-Tuning"
  shows all 250M parameters highlighted as trainable. Bar 2: "Head-only (Transfer)" shows
  only ~1M parameters trainable (the classification head). Bar 3: "LoRA (r=8, q+v)" shows
  only ~0.3M parameters trainable (~0.12% of total). All bars same total height for
  reference (250M), trainable portion shown in green, frozen in gray. X-axis: strategy
  labels. Y-axis: parameters (log scale). Caption: "LoRA trains 0.1-0.3% of parameters
  yet matches or exceeds full fine-tuning quality on most downstream tasks."

---

## Source Dir (scripts_topic7a/)

### train.py

```python
"""
train.py -- LoRA fine-tuning of Flan-T5-small on complaint summarization.

Task: Given a Barclays customer complaint text, generate a 1-sentence summary.
Model: google/flan-t5-small (77M parameters, manageable on a T4 GPU).
PEFT: LoraConfig with TaskType.SEQ_2_SEQ_LM, targeting q and v projection layers.
Metrics: Inline token-overlap (ROUGE-1 F1 approximation) using numpy only.
         No evaluate library -- incompatible with datasets 4.x (see SAGEMAKER_LESSONS_LEARNED L6).

SageMaker: HuggingFace estimator, ml.g4dn.xlarge (NVIDIA T4 16GB).
           Container: transformers_version=4.56.2, pytorch_version=2.8.0, py_version=py312.
           requirements.txt installs peft>=0.6.0 and numpy<2 into the container.
           Adapter weights saved to /opt/ml/model/ as standard PEFT checkpoint.

eval_strategy="epoch" -- evaluation_strategy removed in transformers 4.41+ (L5).
"""

import argparse
import os
import json
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Argument parsing
# SageMaker passes --hyperparameter-name as CLI args.
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Flan-T5-small")

    # LoRA hyperparameters
    parser.add_argument("--rank",       type=int,   default=8,
                        help="LoRA rank r (low-rank dimension)")
    parser.add_argument("--alpha",      type=int,   default=16,
                        help="LoRA scaling alpha (effective scale = alpha/rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training hyperparameters
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)

    # Seq2Seq generation
    parser.add_argument("--max_input_length",  type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)

    # SageMaker environment variables (injected automatically)
    parser.add_argument("--model-dir",
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir",
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic complaint-summary dataset
# Real deployment would use the barclays-genai-devs-data S3 bucket.
# For this course we generate ~200 pairs inline to keep the job under 30 min.
# ---------------------------------------------------------------------------

COMPLAINTS = [
    ("My credit card was charged twice for the same transaction on 12 March and nobody has "
     "refunded me after three calls to your support team over two weeks.", "Duplicate charge on 12 March not refunded after three support calls."),
    ("I applied for a graduate current account six weeks ago and still have not received a "
     "decision or any update from the bank despite submitting all required documents.", "Graduate account application has had no update after six weeks."),
    ("The mobile app keeps crashing whenever I try to view my savings account balance, "
     "making it impossible to check my money without visiting a branch.", "App crashes on savings account balance screen."),
    ("An unauthorised direct debit of 150 pounds was taken from my account on 5 April. "
     "I did not set this up and I need it cancelled and refunded immediately.", "Unauthorised 150-pound direct debit taken on 5 April needs cancellation."),
    ("Your overseas transaction fee is 2.99 percent but you charged me 3.5 percent on my "
     "recent trip to Spain. I want an explanation and a refund of the overcharge.", "Incorrect overseas fee of 3.5 percent charged instead of 2.99 percent."),
    ("I have been locked out of online banking for five days. The reset password link in "
     "the email you sent does not work and the helpline wait time is over an hour.", "Password reset link broken, locked out of online banking for five days."),
    ("My mortgage direct debit failed this month because of a bank error. Your team assured "
     "me it was fixed but I received a late payment notice from the lender.", "Bank error caused mortgage direct debit failure and late payment notice."),
    ("I transferred 2000 pounds to my sister's account two days ago and the money has not "
     "arrived. The reference number is not showing in her transaction history.", "2000-pound transfer two days ago has not arrived in recipient account."),
    ("A cheque I deposited ten days ago still has not cleared. I need the funds urgently "
     "to pay a supplier invoice that is now overdue.", "Cheque deposited ten days ago has not cleared, supplier invoice overdue."),
    ("Your interest rate on my savings account was reduced without any prior notice. "
     "I would have moved the money elsewhere if I had known in advance.", "Savings rate reduced without notice, customer would have moved funds."),
    ("I have tried to update my address through online banking four times and it keeps "
     "reverting to my old address. The branch staff said to do it online.", "Address update through online banking keeps reverting to old address."),
    ("The contactless limit on my debit card was increased to 100 pounds without my consent. "
     "I want it set back to the original 45-pound limit immediately.", "Contactless limit increased to 100 pounds without customer consent."),
    ("My account has been frozen for suspected fraud but all the transactions were genuine. "
     "I cannot pay my rent or buy food while the investigation is ongoing.", "Account frozen for fraud investigation blocking rent and food payments."),
    ("I was told on the phone that my loan application was approved but the written "
     "confirmation says declined. I need someone to clarify which decision is correct.", "Conflicting loan decision: approved by phone but declined in writing."),
    ("Your ATM at the Oxford Street branch dispensed 50 pounds less than I requested "
     "but debited the full amount from my account.", "ATM dispensed 50 pounds short but debited the full requested amount."),
    ("I received a letter saying my account would be closed in 30 days with no reason "
     "given. I have been a customer for 15 years and deserve an explanation.", "Account closure notice received with no reason after 15 years as customer."),
    ("The financial hardship support you promised during my call last week has not been "
     "applied to my account and interest is still accruing on my overdraft.", "Promised hardship support not applied, overdraft interest still accruing."),
    ("Your chip-and-pin machine at checkout rejected my card three times even though "
     "my account has sufficient funds. I was embarrassed in front of other customers.", "Card rejected at chip-and-pin with sufficient funds available."),
    ("I have been billed for a premium banking package I never signed up for. "
     "The charge of 15 pounds per month has been applied for the past six months.", "Unsolicited premium package charged at 15 pounds per month for six months."),
    ("The foreign currency I ordered for collection was not ready on the agreed date "
     "and your staff could not tell me when it would arrive.", "Ordered foreign currency not ready on agreed collection date."),
]

# Augment to ~200 pairs by paraphrasing prefixes (simple heuristic, not ML)
def augment_pairs(pairs, target=200):
    prefixes = [
        "Complaint: ", "Customer issue: ", "Issue: ", "Problem: ", "Report: ",
        "Concern: ", "Query: ", "", "", "", "",  # empty = original text
    ]
    augmented = []
    i = 0
    while len(augmented) < target:
        complaint, summary = pairs[i % len(pairs)]
        prefix = prefixes[i % len(prefixes)]
        augmented.append((prefix + complaint, summary))
        i += 1
    return augmented


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ComplaintDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_input, max_target):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        complaint, summary = self.pairs[idx]
        # Flan-T5 is instruction-tuned: prefix the task
        input_text  = "summarize: " + complaint
        target_text = summary

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ---------------------------------------------------------------------------
# Inline ROUGE-1 F1 (token overlap) -- no evaluate library
# ---------------------------------------------------------------------------

def token_overlap_f1(pred_ids, label_ids, tokenizer):
    """
    Approximate ROUGE-1 F1 using token-level unigram overlap.
    Ignores padding (-100) and special tokens.
    Returns mean F1 across the batch as a float in [0, 1].
    """
    f1_scores = []
    for pred, label in zip(pred_ids, label_ids):
        # Replace -100 (ignored label) with pad token
        label = [t for t in label if t != -100]
        pred  = [t for t in pred  if t not in (tokenizer.pad_token_id,
                                                tokenizer.eos_token_id)]
        if len(label) == 0 and len(pred) == 0:
            f1_scores.append(1.0)
            continue
        if len(label) == 0 or len(pred) == 0:
            f1_scores.append(0.0)
            continue
        pred_set  = set(pred)
        label_set = set(label)
        overlap   = len(pred_set & label_set)
        precision = overlap / len(pred_set)
        recall    = overlap / len(label_set)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def compute_metrics_factory(tokenizer, max_target):
    """
    Returns a compute_metrics function compatible with Seq2SeqTrainer.
    Uses inline numpy token overlap -- no evaluate library.
    """
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Trainer may return logits (3D) -- take argmax
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        f1 = token_overlap_f1(preds, labels, tokenizer)
        return {"rouge1_approx": round(f1, 4)}
    return compute_metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"LoRA rank={args.rank}, alpha={args.alpha}, dropout={args.lora_dropout}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Model dir: {args.model_dir}")

    # ------------------------------------------------------------------
    # 1. Tokenizer and base model
    # ------------------------------------------------------------------
    model_name = "google/flan-t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    total_base = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {total_base:,}")

    # ------------------------------------------------------------------
    # 2. Apply LoRA via PEFT
    #    target_modules=["q", "v"] injects A/B matrices into every
    #    self-attention query and value projection in both encoder and decoder.
    #    TaskType.SEQ_2_SEQ_LM tells PEFT this is an encoder-decoder model.
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q", "v"],          # q and v projections in Flan-T5
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters with LoRA: {trainable:,}")
    print(f"Trainable fraction: {100 * trainable / total_base:.4f}%")

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    all_pairs = augment_pairs(COMPLAINTS, target=200)
    split     = int(0.85 * len(all_pairs))
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    train_dataset = ComplaintDataset(
        train_pairs, tokenizer, args.max_input_length, args.max_target_length)
    val_dataset   = ComplaintDataset(
        val_pairs,   tokenizer, args.max_input_length, args.max_target_length)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # ------------------------------------------------------------------
    # 4. Training arguments
    #    eval_strategy="epoch" -- evaluation_strategy removed in 4.41+ (L5)
    # ------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",          # NOT evaluation_strategy (removed in 4.41+)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1_approx",
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",               # no MLflow for this topic; added in Topic 8
        dataloader_pin_memory=False,
    )

    compute_metrics = compute_metrics_factory(tokenizer, args.max_target_length)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("Starting LoRA fine-tuning ...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save adapter weights to /opt/ml/model/
    #    PEFT saves only the A/B matrices (a few MB), not the full model.
    #    Downstream: load with PeftModel.from_pretrained(base, model_dir).
    # ------------------------------------------------------------------
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Save a quick summary for the notebook monitoring cell
    final_metrics = trainer.evaluate()
    summary = {
        "rank":           args.rank,
        "alpha":          args.alpha,
        "trainable_params": trainable,
        "total_params":   total_base,
        "trainable_pct":  round(100 * trainable / total_base, 4),
        "final_rouge1":   final_metrics.get("eval_rouge1_approx", None),
    }
    with open(os.path.join(args.model_dir, "lora_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Adapter weights saved to {args.model_dir}")
    print(f"Training summary: {summary}")


if __name__ == "__main__":
    main()
```

### requirements.txt

```
peft>=0.6.0
numpy<2
```

---

## Key Changes from 11_Simplified_LoRA_FFN.ipynb

**Keeping**:
- The core `LoraLayer` class structure (original_layer freeze, lora_A, lora_B, forward merge).
- The `replace_with_lora` layer-swap pattern -- rewritten more deterministically (no random.random).
- The parameter counting before/after LoRA to show the compression ratio.
- The five-layer FFN architecture as the scratch-pad model before the real LLM capstone.

**Restructuring**:
- Source has no four-beat arc, no labs, no safety-nets, no discussion prompts.
  We wrap every concept: Beat 1 (failure) -> Beat 2 (diagram) -> Beat 3 (demo) -> Beat 4 (lab).
- Source uses FashionMNIST -> MNIST as the pre-train/fine-tune pair.
  We keep that pair for the scratch demo (fast, no external data needed) but frame it
  as a metaphor for "pre-train on general task, adapt to Barclays-specific complaints".
- Source has no PEFT library section. We add a full PEFT capstone (Section 4) using Flan-T5-small.
- Source has no SageMaker integration. We add a HuggingFace estimator GPU job (ml.g4dn.xlarge).
- Source has no diagrams. We add exactly two (lora-decomposition, lora-parameter-comparison).
- Source target is Colab. We target SageMaker Studio (canonical install cell, sagemaker.Session).

**Replacing**:
- `random.random() > 0.5` layer replacement: replaced with deterministic `replace_fc_with_lora`
  that always replaces fc1, fc2, fc3 (predictable for teaching, no random seed confusion).
- Source `LoraLayer.forward` only adds LoRA during `self.training`: we keep that but also
  explain the inference merge trick (W_merged = W + B @ A.T) as a Beat 3 demo cell.
- FashionMNIST/MNIST training loop (15 epochs): shortened to 5 epochs in Beat 3 demo
  (enough to show convergence, fits in notebook run time on Studio CPU kernel).
- Inline image link (dropbox) for LoRA diagram: replaced with <!-- DIAGRAM: --> placeholder.
- No evaluate library usage added (source has none; capstone uses inline numpy -- L6).

**Beat 1 failures (new)**:
- Beat 1a: Try to store full weight delta (allocate delta_W same size as W) -> OOM simulation
  showing parameter count explosion for a modest FFN.
- Beat 1b: Set rank=d (LoRA with full rank) -> show it degenerates to full fine-tuning
  (same parameter count, no compression).

---

## Variable Continuity from Topic 6b

The following names carry forward from `topic_6b_transfer_learning.md`:
- `device` -- same pattern (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- `set_seeds(seed)` -- identical signature.
- `sess`, `role`, `bucket`, `region` -- canonical SageMaker session variables.

New variables introduced in Topic 7a that downstream cells depend on:
- `LoraLayer` class -- used in Lab 1, safety-net, and Section 3 parameter count cell.
- `replace_fc_with_lora` function -- used in Lab 2, safety-net.
- `lora_model` -- used in fine-tuning loop and parameter counting cell.
- `estimator` -- HuggingFace estimator object (Cell 38).
- `training_job_name` -- returned by estimator.fit(wait=False), used in polling/log cells.

---

## Cell-by-Cell Plan

### Cell 1: markdown - Title and Learning Objectives

```
# Topic 7a - LoRA on Feed-Forward Networks

Barclays Customer Support Intelligence System | Topic 7a

## What you will build

Full fine-tuning updates every weight in the model -- millions of parameters.
LoRA freezes all weights and injects two tiny matrices per layer whose product
approximates the weight change. You implement LoRA from scratch in PyTorch,
apply it to a simple classifier to verify the math, then use HuggingFace PEFT
to LoRA-fine-tune Flan-T5-small on complaint summarization in a GPU training job.

## Why this matters to Barclays

Barclays hosts dozens of NLP models for complaint triage, summarization, and
risk flagging. Full fine-tuning for each task is expensive: storage, GPU time,
and deployment overhead multiply with every new model version.
LoRA adapters are a few MB each. You deploy one frozen base model and swap adapters
per task -- the same architecture that powers production PEFT deployments at scale.

## Learning objectives

1. Derive the LoRA update rule: W' = W + B @ A, explain why B=zeros at init
2. Implement LoraLayer in PyTorch, freeze the original layer, verify gradient flow
3. Swap linear layers in an FFN for LoraLayer, count trainable parameters before and after
4. Explain rank r and alpha and how to choose them (r=4,8,16 heuristics)
5. Use HuggingFace PEFT (LoraConfig, get_peft_model) to apply LoRA to Flan-T5
6. Launch and monitor a LoRA fine-tuning job on SageMaker GPU (ml.g4dn.xlarge)

## Estimated time

90 to 120 minutes in class.
```

---

### Cell 2: code - Environment Setup and Installs

```python
# Environment setup for SageMaker Studio.
# Section 1-3 (scratch LoRA) runs in this CPU kernel.
# Section 4 (Flan-T5 capstone) launches a remote GPU job via HuggingFace estimator.

!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "numpy<2" \
    "matplotlib>=3.7.0"

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

### Cell 3: code - Imports and Configuration

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings

warnings.filterwarnings("ignore")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(42)

# CPU for all in-notebook demos; GPU job runs in scripts_topic7a/train.py.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
```

---

### Cell 4: markdown - Section 1 Header: The Parameter Budget Problem

```
## Section 1 -- The Parameter Budget Problem

Every weight matrix W in a model stores d x k numbers.
Fine-tuning changes every single one. For a single 512 x 1024 layer that is 524,288
numbers. For Flan-T5-small (250M parameters) you update every one of them.

What if you only needed to store the *change* in W instead of the full update?
The change is called delta_W. Full fine-tuning stores delta_W as a dense d x k matrix.
LoRA asks: can we approximate delta_W with a low-rank factorisation delta_W = B @ A?
If r << min(d, k) the saving is enormous.

First, let's feel the pain of storing full weight deltas.
```

---

### Cell 5: code - Beat 1a: Full Weight Delta Explosion

```python
# Beat 1 -- try to store a full weight delta for every layer in an FFN.
# This shows WHY we need a smarter approach before introducing LoRA.

# A realistic small FFN: six layers with dimensions matching our demo model.
layer_shapes = [
    (784, 1024),   # fc1: input -> hidden
    (1024, 512),   # fc2
    (512, 256),    # fc3
    (256, 128),    # fc4
    (128, 64),     # fc5
    (64, 10),      # fc6: hidden -> output
]

print("Full weight delta storage (one float32 per parameter):")
print("-" * 55)
total_delta_params = 0
for i, (d_in, d_out) in enumerate(layer_shapes, 1):
    n_params = d_in * d_out
    mb = n_params * 4 / (1024 ** 2)   # 4 bytes per float32
    total_delta_params += n_params
    print(f"  fc{i}: {d_in} x {d_out} = {n_params:>9,} params | {mb:.2f} MB delta")

total_mb = total_delta_params * 4 / (1024 ** 2)
print("-" * 55)
print(f"  Total: {total_delta_params:,} params | {total_mb:.2f} MB for ONE fine-tuned copy")
print()
print("Scale this to Flan-T5-small (250M params):")
flant5_delta_mb = 250_000_000 * 4 / (1024 ** 2)
print(f"  Full delta: {flant5_delta_mb:.0f} MB just for the weight changes")
print()
print("And you need a separate copy for EVERY downstream task.")
print("Complaint summarization: +950 MB. NER: +950 MB. Risk: +950 MB ...")
print("This does not scale. We need a better approach.")
```

Expected output shows each layer's delta size; emphasises that weight deltas are
as large as the weights themselves -- painful for multi-task deployment.

---

### Cell 6: code - Beat 1b: Rank = d is the Same as Full Fine-Tuning

```python
# Beat 1b -- what if we try to 'compress' delta_W but set rank=d (full rank)?
# Spoiler: r=d gives ZERO compression. This is why rank selection matters.

print("LoRA parameter count as a function of rank r:")
print("Layer fc1: d=784, k=1024")
d, k = 784, 1024
print(f"  Full fine-tuning delta: d x k = {d * k:,} params")
print()
for r in [d, 256, 64, 16, 8, 4, 1]:
    lora_params = d * r + r * k
    ratio = (d * k) / lora_params
    note = " <- same as full fine-tuning!" if r >= min(d, k) else ""
    print(f"  r={r:>4}: A({r}x{k}) + B({d}x{r}) = {lora_params:>9,} params | "
          f"{ratio:>6.1f}x compression{note}")

print()
print("Takeaway: r must be much smaller than d and k to get meaningful compression.")
print("r=8 gives ~58x compression on fc1. That is the insight behind LoRA.")
```

Prints a table showing compression ratio by rank. r=d row explicitly labelled
"same as full fine-tuning". Drives home why low rank is the whole point.

---

### Cell 7: markdown - Beat 2: LoRA Decomposition Diagram

```
## The LoRA Idea

Instead of updating W directly, we inject two small matrices:

    W' = W + B @ A

where W stays frozen (no gradient), and only A and B are trained.

- A has shape  (r, k)  -- projects from input dimension k down to rank r
- B has shape  (d, r)  -- projects from rank r up to output dimension d
- r << min(d, k) ensures we store far fewer parameters

At initialisation: A ~ Normal(0, 0.02), B = 0.
This means B @ A = 0 at step 0, so the adapted model starts identical to the
pre-trained model. Only during training do A and B diverge from this baseline.

The scaling factor lora_alpha / r controls how much the LoRA update contributes
relative to the frozen weight. A common heuristic is alpha = 2 * r.

<!-- DIAGRAM: LoRA decomposition showing frozen W in gray, low-rank matrices A and B in green, update formula W' = W + BA, with concrete dimension labels r=8 d=512 k=1024 showing 43x parameter reduction -->
[View diagram](../../plans/topic_7a_lora_ffn/diagrams/lora-decomposition.mmd)
```

---

### Cell 8: code - Beat 3: LoraLayer Class From Scratch

```python
# Beat 3 -- implement LoraLayer in PyTorch.
# This is the core of the entire topic. Read every comment carefully.
# The PEFT library does exactly this internally -- but you need to understand it
# before you trust a library to do it for you.

class LoraLayer(nn.Module):
    """
    Wraps an existing nn.Linear with LoRA adaptation.

    The forward pass computes:
        output = original_layer(x) + (lora_B @ lora_A)(x) * scale

    where scale = lora_alpha / rank, lora_A is (rank, in_features),
    lora_B is (out_features, rank).

    At initialisation lora_B is all zeros, so the adaptation contributes
    nothing at step 0. Only lora_A and lora_B are trainable; original_layer
    parameters are frozen.

    Args:
        original_layer: the nn.Linear to wrap (must have .in_features and .out_features)
        rank:           low-rank dimension r; smaller = fewer params, less expressive
        lora_alpha:     scaling multiplier; effective scale = lora_alpha / rank
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 8, lora_alpha: int = 16):
        super().__init__()

        self.in_features  = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank         = rank
        self.scale        = lora_alpha / rank   # applied in forward

        # --- Freeze the original layer ---
        # We keep it in the module so its weights are still used in forward,
        # but we prevent gradients from flowing through it.
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # --- LoRA low-rank matrices ---
        # lora_A: (rank, in_features) -- "down" projection
        # lora_B: (out_features, rank) -- "up" projection
        # Using nn.Linear without bias keeps the implementation clean.
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # Initialisation: A ~ Normal(0, 0.02), B = 0.
        # At step 0: lora_B(lora_A(x)) = 0 for all x.
        # The adapted model is identical to the frozen model at start.
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path: no gradient computation
        original_output = self.original_layer(x)

        # LoRA path: lora_B(lora_A(x)) * scale
        # lora_A: (batch, in_features) -> (batch, rank)
        # lora_B: (batch, rank) -> (batch, out_features)
        lora_output = self.lora_B(self.lora_A(x)) * self.scale

        return original_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        Merge A and B into the original weight for zero-overhead inference.
        W_merged = W + B.weight @ A.weight  (shape: out_features x in_features)
        After merging you can discard lora_A and lora_B entirely.
        """
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.original_layer.bias is not None)
        with torch.no_grad():
            delta_W = self.lora_B.weight @ self.lora_A.weight   # (d, k)
            merged.weight.copy_(self.original_layer.weight + delta_W * self.scale)
            if self.original_layer.bias is not None:
                merged.bias.copy_(self.original_layer.bias)
        return merged


# Quick sanity check: verify the shapes and that grad flags are correct.
test_linear = nn.Linear(64, 128)
lora_test   = LoraLayer(test_linear, rank=4, lora_alpha=8)

x_test = torch.randn(16, 64)
out    = lora_test(x_test)

print("LoraLayer sanity check:")
print(f"  Input shape:  {x_test.shape}")
print(f"  Output shape: {out.shape}")
print(f"  original_layer grad: {lora_test.original_layer.weight.requires_grad}")
print(f"  lora_A grad:         {lora_test.lora_A.weight.requires_grad}")
print(f"  lora_B grad:         {lora_test.lora_B.weight.requires_grad}")
print(f"  lora_B weight sum (should be 0.0): {lora_test.lora_B.weight.sum().item():.4f}")
print()
merged = lora_test.merge_weights()
print(f"  merge_weights output shape: {merged.weight.shape}  (no lora overhead at inference)")
```

---

### Cell 9: markdown - Lab 1 Instructions (Tier 1 Guided)

```
## Lab 1 -- Implement the LoraLayer (Tier 1 Guided, 15 min)

### Situation

Your Barclays team has a pre-trained complaint classifier. You want to adapt it for a
new sub-task (priority routing) without retraining the whole model.
The team lead says: "Implement LoRA so we can inject trainable adapters without
touching the frozen backbone."

### Task

Complete the `LoraLayerStudent` class below. The class wraps an existing nn.Linear
and adds low-rank adaptation.

### Action

Follow the numbered steps in the comments. Do not change any line that is already filled in.

### Result

The verification cell at the bottom must print:
- original_layer.requires_grad = False
- lora_A.requires_grad = True, lora_B.requires_grad = True
- lora_B initial weight sum = 0.0
- Output shape correct: torch.Size([8, 32])
- merge_weights runs without error
```

---

### Cell 10: code - Lab 1 Starter Code

```python
class LoraLayerStudent(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.in_features  = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank         = rank
        self.scale        = lora_alpha / rank

        # Step 1: Store original_layer and freeze its parameters.
        # Hint: set requires_grad = False on all parameters of original_layer.
        self.original_layer = None  # YOUR CODE

        # Step 2: Create self.lora_A as nn.Linear(in_features -> rank, no bias).
        self.lora_A = None  # YOUR CODE

        # Step 3: Create self.lora_B as nn.Linear(rank -> out_features, no bias).
        self.lora_B = None  # YOUR CODE

        # Step 4: Initialise lora_A with Normal(0, 0.02) and lora_B with zeros.
        # Hint: nn.init.normal_ and nn.init.zeros_ on the .weight tensor.
        # YOUR CODE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 5: Compute original_output from self.original_layer.
        original_output = None  # YOUR CODE

        # Step 6: Compute lora_output = lora_B(lora_A(x)) * self.scale.
        lora_output = None  # YOUR CODE

        # Step 7: Return the sum.
        return None  # YOUR CODE
```

---

### Cell 11: code - Lab 1 Safety-Net

```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if (not hasattr(LoraLayerStudent, '__init__') or
        LoraLayerStudent(nn.Linear(4, 8)).lora_A is None):
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    LoraLayerStudent = LoraLayer   # fall back to the instructor implementation
else:
    print("Lab 1 complete -- using your LoraLayerStudent.")
```

---

### Cell 12: code - Lab 1 Verification

```python
# Verification -- do not edit this cell.
test_lin = nn.Linear(16, 32)
student_lora = LoraLayerStudent(test_lin, rank=4, lora_alpha=8)

x_v = torch.randn(8, 16)
out_v = student_lora(x_v)

assert student_lora.original_layer.weight.requires_grad == False, \
    "original_layer must be frozen (requires_grad=False)"
assert student_lora.lora_A.weight.requires_grad == True, \
    "lora_A must be trainable (requires_grad=True)"
assert student_lora.lora_B.weight.requires_grad == True, \
    "lora_B must be trainable (requires_grad=True)"
assert abs(student_lora.lora_B.weight.sum().item()) < 1e-6, \
    "lora_B must be initialised to zeros"
assert out_v.shape == torch.Size([8, 32]), \
    f"Output shape wrong: got {out_v.shape}, expected (8, 32)"

print("original_layer.requires_grad =", student_lora.original_layer.weight.requires_grad)
print("lora_A.requires_grad =", student_lora.lora_A.weight.requires_grad)
print("lora_B.requires_grad =", student_lora.lora_B.weight.requires_grad)
print("lora_B initial weight sum =", round(student_lora.lora_B.weight.sum().item(), 4))
print("Output shape:", out_v.shape)
print()
print("Lab 1 verification PASSED.")
```

---

### Cell 13: markdown - Lab 1 Stretch + Homework

```
### Stretch (fast finishers)

Implement `merge_weights(self) -> nn.Linear` on your `LoraLayerStudent`.
The merged weight matrix should equal `W + scale * (B.weight @ A.weight)`.
Verify that `merged_layer(x)` and `lora_layer(x)` produce identical outputs
(the difference should be less than 1e-5 due to float32 precision).

### Homework Extension

Try different initialisations for lora_A and lora_B:
- lora_A = zeros, lora_B = Normal(0, 0.02): what happens to the output at step 0?
- Both = Normal(0, 0.02): does training still converge? Why or why not?
Write a short explanation (2-3 sentences) of why the standard init (A normal, B zero)
is preferred and what property it preserves at the start of training.
```

---

### Cell 14: markdown - Section 2 Header: Applying LoRA to an FFN

```
## Section 2 -- Applying LoRA to a Feed-Forward Network

We now apply your LoraLayer to a real (small) FFN pre-trained on FashionMNIST.
The goal is to adapt it to MNIST with only the LoRA parameters trainable.

This mirrors what we will do with Flan-T5 in Section 4 -- the FFN version
is just faster and requires no GPU, so we can verify the entire workflow here.

Parameter comparison diagram below shows what happens to trainable parameter
count when we swap linear layers for LoRA-wrapped ones.

<!-- DIAGRAM: Bar chart comparing trainable parameter counts for full fine-tuning vs head-only transfer vs LoRA (r=8) on Flan-T5-small, showing LoRA trains less than 0.3% of parameters -->
[View diagram](../../plans/topic_7a_lora_ffn/diagrams/lora-parameter-comparison.mmd)
```

---

### Cell 15: code - Pre-Train FFN on FashionMNIST

```python
# Pre-train a simple FFN on FashionMNIST.
# We use this as our "pre-trained model" that we will later adapt with LoRA.
# Training takes ~3 minutes on CPU; the accuracy will be ~87% on FashionMNIST.

from tqdm.auto import tqdm

# Data loaders
transform = transforms.Compose([transforms.ToTensor()])
fashion_train = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
fashion_test  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
fashion_train_loader = DataLoader(fashion_train, batch_size=128, shuffle=True)
fashion_test_loader  = DataLoader(fashion_test,  batch_size=128, shuffle=False)


class FFNModel(nn.Module):
    """
    Five-layer feed-forward network with BatchNorm and Dropout.
    Trained on FashionMNIST (10 classes, 28x28 images).
    We will later replace fc1-fc3 with LoraLayer wrappers.
    """
    def __init__(self):
        super().__init__()
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(784, 1024)
        self.bn1      = nn.BatchNorm1d(1024)
        self.drop1    = nn.Dropout(0.3)
        self.fc2      = nn.Linear(1024, 512)
        self.bn2      = nn.BatchNorm1d(512)
        self.drop2    = nn.Dropout(0.3)
        self.fc3      = nn.Linear(512, 256)
        self.bn3      = nn.BatchNorm1d(256)
        self.drop3    = nn.Dropout(0.3)
        self.fc4      = nn.Linear(256, 128)
        self.bn4      = nn.BatchNorm1d(128)
        self.drop4    = nn.Dropout(0.3)
        self.fc5      = nn.Linear(128, 64)
        self.bn5      = nn.BatchNorm1d(64)
        self.drop5    = nn.Dropout(0.3)
        self.fc6      = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))
        x = self.drop3(self.bn3(F.relu(self.fc3(x))))
        x = self.drop4(self.bn4(F.relu(self.fc4(x))))
        x = self.drop5(self.bn5(F.relu(self.fc5(x))))
        return self.fc6(x)


pretrained_model = FFNModel().to(device)
optimizer_pre    = torch.optim.Adam(pretrained_model.parameters(), lr=1e-3)
criterion        = nn.CrossEntropyLoss()

PRETRAIN_EPOCHS = 5   # 5 epochs is enough to show the concept (source used 15)
for epoch in range(PRETRAIN_EPOCHS):
    pretrained_model.train()
    total_loss = 0
    for images, labels in tqdm(fashion_train_loader, desc=f"Pre-train {epoch+1}/{PRETRAIN_EPOCHS}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer_pre.zero_grad()
        loss = criterion(pretrained_model(images), labels)
        loss.backward()
        optimizer_pre.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}: avg loss = {total_loss / len(fashion_train_loader):.4f}")

# Count parameters before LoRA
total_pre  = sum(p.numel() for p in pretrained_model.parameters())
train_pre  = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
print(f"\nPre-trained model: {total_pre:,} total params, {train_pre:,} trainable")
```

---

### Cell 16: code - Beat 3: Replace FC Layers with LoRA

```python
# Beat 3 -- replace fc1, fc2, fc3 with LoraLayer wrappers.
# fc4, fc5, fc6 stay frozen (not wrapped).
# Only the LoRA A and B matrices are trainable after replacement.

def replace_fc_with_lora(model: FFNModel, rank: int = 8, lora_alpha: int = 16) -> FFNModel:
    """
    Replace the first three fully-connected layers with LoraLayer.
    This is deterministic: always wraps fc1, fc2, fc3.
    The source notebook used random.random() > 0.5 -- we remove randomness
    so every student sees the same result.
    """
    model.fc1 = LoraLayer(model.fc1, rank=rank, lora_alpha=lora_alpha)
    model.fc2 = LoraLayer(model.fc2, rank=rank, lora_alpha=lora_alpha)
    model.fc3 = LoraLayer(model.fc3, rank=rank, lora_alpha=lora_alpha)
    # fc4, fc5, fc6 are left unchanged -- they are frozen by not wrapping
    for param in model.fc4.parameters():
        param.requires_grad = False
    for param in model.fc5.parameters():
        param.requires_grad = False
    for param in model.fc6.parameters():
        param.requires_grad = False
    return model


lora_model = replace_fc_with_lora(pretrained_model, rank=8, lora_alpha=16)
lora_model.to(device)

# Parameter count after LoRA
total_lora    = sum(p.numel() for p in lora_model.parameters())
trainable_lora = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

print("Parameter counts after LoRA injection:")
print(f"  Total params (frozen + LoRA): {total_lora:,}")
print(f"  Trainable (LoRA A+B only):    {trainable_lora:,}")
print(f"  Frozen:                       {total_lora - trainable_lora:,}")
print(f"  Compression ratio vs full FT: {train_pre / trainable_lora:.1f}x fewer trainable params")

# Optimise ONLY the trainable (LoRA) parameters
lora_optimizer = torch.optim.Adam(
    [p for p in lora_model.parameters() if p.requires_grad], lr=3e-4)
```

---

### Cell 16b: code - Safety-Net for lora_model and trainable_lora

```python
# Safety-net: run this if Cell 16 timed out or failed.
# SKIP if lora_model and trainable_lora are already defined.
if 'lora_model' not in dir() or lora_model is None:
    print("Rebuilding lora_model from safety-net...")
    import copy
    _base = copy.deepcopy(pretrained_model) if 'pretrained_model' in dir() else FFNModel().to(device)
    lora_model = replace_fc_with_lora(_base, rank=8, lora_alpha=16)
    lora_model.to(device)
    total_lora    = sum(p.numel() for p in lora_model.parameters())
    trainable_lora = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    train_pre = sum(p.numel() for p in _base.parameters() if p.requires_grad) if 'train_pre' not in dir() else train_pre
    lora_optimizer = torch.optim.Adam(
        [p for p in lora_model.parameters() if p.requires_grad], lr=3e-4)
    print(f"lora_model ready. Trainable params: {trainable_lora:,}")
else:
    print("lora_model already defined, skipping safety-net.")
```

---

### Cell 17: markdown - Discussion Prompt 1

```
### Peer Discussion (3 min): Rank Trade-offs

You just saw that rank=8 gives ~Xx compression. Consider:

1. What happens to model quality if you set rank=1? rank=128?
2. In production, Barclays needs to serve five complaint sub-tasks from one base model.
   With full fine-tuning you store five full model copies. With LoRA rank=8 you store
   one base model plus five tiny adapters. What does this mean for storage and latency?
3. The original LoRA paper (Hu et al. 2021) found that rank=4 often matches or beats
   full fine-tuning on language tasks. Why might lower rank sometimes generalise better?

Discuss with your neighbour for 3 minutes. We will debrief as a group.
```

---

### Cell 18: code - Fine-Tune with LoRA on MNIST

```python
# Fine-tune the LoRA-adapted model on MNIST (different domain = transfer scenario).
# Only the LoRA A and B matrices receive gradient updates.

mnist_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
mnist_train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
mnist_test_loader  = DataLoader(mnist_test,  batch_size=128, shuffle=False)

FINETUNE_EPOCHS = 3

for epoch in range(FINETUNE_EPOCHS):
    lora_model.train()
    total_loss = 0
    for images, labels in tqdm(mnist_train_loader, desc=f"LoRA FT {epoch+1}/{FINETUNE_EPOCHS}", leave=False):
        images, labels = images.to(device), labels.to(device)
        lora_optimizer.zero_grad()
        loss = criterion(lora_model(images), labels)
        loss.backward()
        lora_optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}: avg loss = {total_loss / len(mnist_train_loader):.4f}")

# Evaluate
lora_model.eval()
correct, total_samples = 0, 0
with torch.no_grad():
    for images, labels in mnist_test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = lora_model(images).argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += labels.size(0)

accuracy = 100 * correct / total_samples
print(f"\nMNIST test accuracy after LoRA fine-tuning: {accuracy:.2f}%")
print("(Accuracy is modest because FashionMNIST and MNIST are different domains,")
print(" but the key result is that ONLY the LoRA weights moved -- not the backbone.)")
```

---

### Cell 19: markdown - Lab 2 Instructions (Tier 1 Guided)

```
## Lab 2 -- Apply LoRA to a New Layer Configuration (Tier 1 Guided, 15 min)

### Situation

The Barclays ML team wants to experiment with different LoRA configurations.
They ask you to build a `replace_fc_with_lora_student` function that applies
LoRA to ALL six fc layers (not just the first three) and supports a configurable rank.

### Task

Complete `replace_fc_with_lora_student` below. It must:
1. Wrap fc1 through fc6 with LoraLayerStudent (your implementation from Lab 1).
2. Accept `rank` and `lora_alpha` as arguments.
3. Return the modified model.

After applying, verify the trainable parameter count printed by the verification cell
matches the expected value (approximately 2x the Lab 1 count because you wrap 6 layers
instead of 3).

### Action

Fill in the function body following the numbered step comments.

### Result

Verification cell prints: "Trainable parameters: X" and "Lab 2 PASSED."
```

---

### Cell 20: code - Lab 2 Starter Code

```python
def replace_fc_with_lora_student(model: FFNModel, rank: int = 8, lora_alpha: int = 16) -> FFNModel:
    """
    Wrap all six fc layers with LoraLayerStudent.

    Steps:
    1. Replace model.fc1 with LoraLayerStudent(model.fc1, rank, lora_alpha).
    2. Repeat for fc2, fc3, fc4, fc5, fc6.
    3. Return the modified model.
    """
    # Step 1-6: replace each fc layer.
    model.fc1 = None  # YOUR CODE
    model.fc2 = None  # YOUR CODE
    model.fc3 = None  # YOUR CODE
    model.fc4 = None  # YOUR CODE
    model.fc5 = None  # YOUR CODE
    model.fc6 = None  # YOUR CODE
    return model
```

---

### Cell 21: code - Lab 2 Safety-Net

```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.
_test_model = FFNModel()
_patched     = replace_fc_with_lora_student(_test_model, rank=4)
if _patched.fc6 is None or not isinstance(_patched.fc6, (LoraLayer, LoraLayerStudent)):
    print("Using Lab 2 safety-net so the rest of the notebook can run.")
    replace_fc_with_lora_student = lambda m, rank=8, lora_alpha=16: replace_fc_with_lora(m, rank, lora_alpha)
else:
    print("Lab 2 complete -- using your replace_fc_with_lora_student.")
```

---

### Cell 22: code - Lab 2 Verification

```python
# Verification -- do not edit this cell.
import copy

lab2_model = replace_fc_with_lora_student(copy.deepcopy(FFNModel()), rank=8, lora_alpha=16)
lab2_trainable = sum(p.numel() for p in lab2_model.parameters() if p.requires_grad)

assert isinstance(lab2_model.fc6, (LoraLayer, LoraLayerStudent)), \
    "fc6 must be wrapped with LoraLayer (or LoraLayerStudent)"
assert lab2_trainable > trainable_lora, \
    "Wrapping 6 layers must yield more trainable params than wrapping 3"

print(f"Trainable parameters (6 layers wrapped): {lab2_trainable:,}")
print(f"Trainable parameters (3 layers wrapped): {trainable_lora:,}")
print(f"Ratio: {lab2_trainable / trainable_lora:.2f}x (should be approximately 2x)")
print()
print("Lab 2 PASSED.")
```

---

### Cell 23: markdown - Lab 2 Stretch + Homework

```
### Stretch (fast finishers)

Implement the weight merge for the fully-wrapped model.
Call `merge_weights()` on each LoraLayer after fine-tuning completes.
Verify that:
- Forward pass output is identical before and after merging (difference < 1e-5).
- The merged model has zero trainable parameters (all weights are now baked in).

### Homework Extension

Run the six-layer LoRA model with rank=1, rank=4, rank=8, rank=16.
For each rank: record trainable parameter count and MNIST test accuracy after 3 epochs.
Plot accuracy vs rank on a log-scale x-axis.
At what rank does the accuracy plateau? Does this match the heuristic from the paper
(Hu et al. found rank=4 often sufficient for language tasks)?
```

---

### Cell 24: markdown - Section 3 Header: Rank and Alpha Heuristics

```
## Section 3 -- Rank and Alpha: How to Choose

You have now used rank=8. But how do practitioners choose rank in production?

### Rank heuristics (from Hu et al. 2021 and community practice):

| Rank | Typical use case                                          |
|------|-----------------------------------------------------------|
| 1-2  | Extremely constrained memory; usually too little capacity |
| 4    | Simple tasks, domain adaptation, sufficient for most NLP  |
| 8    | Default starting point; good balance of capacity vs cost  |
| 16   | Complex tasks, multi-domain; diminishing returns above 16 |
| 32+  | Rarely needed; often overfits on small datasets           |

Key principle: start with r=8, monitor validation loss, increase to r=16 only if
underfitting, decrease to r=4 if overfitting. Never set r >= min(d, k).

### Alpha heuristics:

- alpha = rank: neutral scaling (scale = 1.0)
- alpha = 2 * rank: common default; slightly amplifies the LoRA update
- alpha = rank / 2: conservative; use when base model is already close to target task

The PEFT library default is alpha = lora_r (1:1 ratio). Most practitioners set
alpha = 2 * rank as a starting point because it stabilises training on small datasets.
```

---

### Cell 25: code - Rank vs Parameter Count Visualisation

```python
# Visualise how rank affects trainable parameter count for Flan-T5-small.
# This makes the diagram concrete with actual numbers before the GPU capstone.

import matplotlib.pyplot as plt

# Flan-T5-small approximate q and v projection sizes
# Encoder: 8 layers x 2 (q, v) x (512, 64) projections
# Decoder: 8 layers x 2 (self-attn q, v) x (512, 64) + 8 x 2 (cross-attn q, v) x (512, 64)
# Simplification: treat all as (d=512, k=512) for illustration
n_projection_pairs = 64   # approximate for Flan-T5-small across all layers
d_size             = 512
k_size             = 512
total_base_params  = 77_000_000   # Flan-T5-small actual param count

ranks = [1, 2, 4, 8, 16, 32, 64]
lora_params = [n_projection_pairs * (d_size * r + r * k_size) for r in ranks]
fractions   = [100 * p / total_base_params for p in lora_params]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar([str(r) for r in ranks], lora_params, color="steelblue")
axes[0].set_xlabel("LoRA Rank r")
axes[0].set_ylabel("Trainable Parameters")
axes[0].set_title("Trainable Parameters vs Rank (Flan-T5-small, q+v)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
for i, (r, p) in enumerate(zip(ranks, lora_params)):
    axes[0].text(i, p + 500, f"{p/1e3:.0f}K", ha="center", fontsize=8)

axes[1].bar([str(r) for r in ranks], fractions, color="coral")
axes[1].set_xlabel("LoRA Rank r")
axes[1].set_ylabel("% of Total Parameters Trainable")
axes[1].set_title("Trainable Fraction vs Rank (Flan-T5-small)")
for i, (r, f) in enumerate(zip(ranks, fractions)):
    axes[1].text(i, f + 0.002, f"{f:.3f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("lora_rank_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

print("\nAt rank=8: trainable params =", f"{lora_params[ranks.index(8)]:,}",
      f"({fractions[ranks.index(8)]:.3f}% of total)")
```

---

### Cell 26: markdown - Discussion Prompt 2

```
### Peer Discussion (3 min): LoRA in Production

Your Barclays platform serves 10 million customer contacts per month across
complaint classification, summarization, NER, risk flagging, and translation.

1. Full fine-tuning: 5 tasks x 250MB (Flan-T5-small) = 1.25 GB of model files.
   LoRA rank=8: 5 adapters x ~2 MB + 1 frozen base (250 MB) = ~260 MB total.
   Where does this saving matter most: storage, serving latency, or update frequency?

2. When you update the complaint summarization adapter (say, to handle new terminology),
   how does the deployment pipeline differ between full fine-tuning and LoRA?

3. What are the risks of setting alpha too high? Too low?

Discuss for 3 minutes, then we move to the capstone.
```

---

### Cell 27: markdown - Section 4 Header: PEFT Capstone

```
## Section 4 -- Capstone: LoRA Fine-Tune Flan-T5 on SageMaker

You have implemented LoRA from scratch and understood the math.
Now we use the HuggingFace PEFT library to do the same thing on Flan-T5-small --
a real sequence-to-sequence LLM -- and run the training job on a GPU instance.

The PEFT library's LoraConfig does exactly what your LoraLayer does:
- Freezes the base model weights.
- Injects low-rank A and B matrices into target modules (q and v projections).
- Saves only the adapter weights (a few MB) rather than the full model.

Training runs on ml.g4dn.xlarge (NVIDIA T4, 16 GB VRAM) and completes in ~10 minutes.
While it runs you will inspect the CloudWatch logs and parameter counts.
```

---

### Cell 28: code - Inspect Flan-T5 Module Names

```python
# Before launching the job, let's see which modules PEFT will target.
# This is how you discover target_modules for any model family.

from transformers import AutoModelForSeq2SeqLM

# Load locally for inspection (no GPU needed for this cell)
inspect_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

print("All linear layers in Flan-T5-small (showing name and shape):")
print("-" * 65)
lm_linear_count = 0
for name, module in inspect_model.named_modules():
    if isinstance(module, nn.Linear):
        lm_linear_count += 1
        # Show only the last two path segments for readability
        short_name = ".".join(name.split(".")[-2:]) if "." in name else name
        print(f"  {name:<55} {list(module.weight.shape)}")
print("-" * 65)
print(f"Total linear layers: {lm_linear_count}")
print()
print("PEFT target_modules=['q', 'v'] matches any layer whose name ends with 'q' or 'v'.")
print("This covers self-attention and cross-attention projections in encoder and decoder.")

total_flan = sum(p.numel() for p in inspect_model.parameters())
print(f"\nFlan-T5-small total parameters: {total_flan:,}")

del inspect_model  # free memory before the GPU job
```

---

### Cell 29: code - Upload Training Script to S3

```python
import os
import boto3

# scripts_topic7a/ must contain exactly:
#   train.py          -- the training entry point
#   requirements.txt  -- auto-installed by SageMaker toolkit (L4: must be this exact name)

SOURCE_DIR = "scripts_topic7a"

# Verify the required files are present before submitting the job
required_files = ["train.py", "requirements.txt"]
for fname in required_files:
    fpath = os.path.join(SOURCE_DIR, fname)
    assert os.path.exists(fpath), f"Missing required file: {fpath}"
    print(f"  Found: {fpath}")

print(f"\nSource dir '{SOURCE_DIR}' is ready.")
print("The HuggingFace estimator will upload this directory to S3 automatically.")
```

---

### Cell 30: code - Define HuggingFace Estimator

```python
from sagemaker.huggingface import HuggingFace

# HuggingFace estimator -- GPU ONLY. Never use ml.m5 with this estimator (L1).
# transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312" (L2, L3).
# requirements.txt installs peft>=0.6.0 into the container.

estimator = HuggingFace(
    entry_point="train.py",
    source_dir=SOURCE_DIR,
    role=role,

    # Version matrix from CORE_TECHNOLOGIES_AND_DECISIONS.md (verified 2026-05-11)
    transformers_version="4.56.2",
    pytorch_version="2.8.0",
    py_version="py312",

    # GPU instance -- NVIDIA T4, 16 GB VRAM, sufficient for Flan-T5-small with LoRA
    instance_type="ml.g4dn.xlarge",
    instance_count=1,

    # LoRA hyperparameters exposed as SageMaker hyperparameters
    hyperparameters={
        "rank":       8,
        "alpha":      16,
        "epochs":     3,
        "batch_size": 8,
        "lr":         3e-4,
    },

    base_job_name="lora-flan-t5",
    output_path=f"s3://{bucket}/lora-flan-t5/output",
)

print("Estimator defined:")
print(f"  instance_type:        {estimator.instance_type}")
print(f"  transformers_version: {estimator.transformers_version}")
print(f"  pytorch_version:      {estimator.pytorch_version}")
print(f"  hyperparameters:      {estimator.hyperparameters}")
```

---

### Cell 31: code - Launch Training Job

```python
# Launch the training job asynchronously (wait=False).
# The job runs on ml.g4dn.xlarge; expected time ~10 minutes.
# We poll for status in the next cell.

estimator.fit(wait=False)

training_job_name = estimator.latest_training_job.name
print(f"Training job launched: {training_job_name}")
print()
print("Monitor in AWS Console:")
print(f"  SageMaker > Training > Training jobs > {training_job_name}")
print()
print("Or run the polling cell below to check status every 60 seconds.")
```

---

### Cell 31b: code - Safety-Net for training_job_name

```python
# Safety-net: run this if your kernel restarted after launching the training job.
# SKIP if training_job_name is already defined.
if 'training_job_name' not in dir() or training_job_name is None:
    training_job_name = "<PASTE YOUR JOB NAME HERE>"
    print(f"Using safety-net training_job_name: {training_job_name}")
else:
    print(f"training_job_name already defined: {training_job_name}")
```

---

### Cell 32: code - Poll Training Job Status

```python
import time

sm_client = boto3.client("sagemaker", region_name=region)

print(f"Polling job: {training_job_name}")
print("-" * 50)

while True:
    response = sm_client.describe_training_job(TrainingJobName=training_job_name)
    status   = response["TrainingJobStatus"]
    elapsed  = response.get("TrainingTimeInSeconds", 0)
    print(f"  Status: {status:<12} | Elapsed: {elapsed}s")

    if status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(60)

if status == "Completed":
    print("\nTraining job COMPLETED successfully.")
    model_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Adapter artifacts at: {model_s3_uri}")
else:
    print(f"\nTraining job ended with status: {status}")
    failure_reason = response.get("FailureReason", "No reason provided")
    print(f"Failure reason: {failure_reason}")
```

---

### Cell 33: code - View CloudWatch Logs

```python
# Stream the last 50 log lines from CloudWatch.
# This is where you see parameter counts, epoch losses, and the final ROUGE-1 score.

import boto3

logs_client = boto3.client("logs", region_name=region)
log_group   = f"/aws/sagemaker/TrainingJobs"
log_stream  = f"{training_job_name}/algo-1-*"

# List available log streams for this job
try:
    streams = logs_client.describe_log_streams(
        logGroupName=log_group,
        logStreamNamePrefix=training_job_name,
        orderBy="LastEventTime",
        descending=True,
        limit=1,
    )["logStreams"]

    if streams:
        stream_name = streams[0]["logStreamName"]
        events = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            limit=50,
            startFromHead=False,
        )["events"]
        print(f"Last 50 log lines from {stream_name}:\n")
        for event in events:
            print(" ", event["message"])
    else:
        print("No log streams found yet. Wait a moment and re-run this cell.")
except Exception as e:
    print(f"Could not retrieve logs: {e}")
    print("You can view logs in the AWS Console under CloudWatch > Log Groups.")
```

---

### Cell 34: markdown - Section 4 Interpretation

```
## What the Training Logs Tell You

Look for these lines in the CloudWatch output:

1. "Base model parameters: 77,000,000" -- Flan-T5-small total
2. "Trainable parameters with LoRA: ~300,000" -- only A and B matrices
3. "Trainable fraction: 0.39%" -- you are training less than half a percent
4. Epoch 1-3 loss values -- should decrease steadily
5. "final_rouge1: 0.XX" -- token overlap metric for summarization quality

The adapter checkpoint saved to S3 is typically 2-5 MB, not 250 MB.
You can load it back with:

    from peft import PeftModel
    base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model = PeftModel.from_pretrained(base, "<path to adapter>")

This is the production pattern: one frozen base, many tiny adapters.
```

---

### Cell 35: code - Load and Test Adapter (Optional Post-Training Cell)

```python
# Optional: run this cell after training completes to test the adapter locally.
# Requires the model artifacts to be downloaded from S3.

import os, json

local_adapter_path = "./lora_adapter_local"

if os.path.exists(local_adapter_path):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    tokenizer   = AutoTokenizer.from_pretrained("google/flan-t5-small")
    base_model  = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    peft_model  = PeftModel.from_pretrained(base_model, local_adapter_path)
    peft_model.eval()

    test_complaints = [
        "My credit card was charged twice and nobody refunded me after three calls.",
        "The mobile app crashes whenever I try to check my savings balance.",
    ]

    print("Complaint Summarization (LoRA-adapted Flan-T5-small):")
    print("-" * 60)
    for complaint in test_complaints:
        inputs = tokenizer(
            "summarize: " + complaint,
            return_tensors="pt", max_length=256, truncation=True)
        with torch.no_grad():
            outputs = peft_model.generate(**inputs, max_new_tokens=32)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input:   {complaint[:70]}...")
        print(f"Summary: {summary}")
        print()
else:
    print("Adapter not found locally. Download from S3 first:")
    print(f"  aws s3 cp {model_s3_uri} {local_adapter_path}/ --recursive")
```

---

### Cell 36: markdown - Wrap-Up

```
## Wrap-Up: What You Built in Topic 7a

### Key takeaways

1. Full fine-tuning stores delta_W as a dense matrix -- the same size as W itself.
   For a 250M parameter model that is ~950 MB of additional weights per task.

2. LoRA factorises delta_W = B @ A where r << min(d, k).
   At rank=8 on Flan-T5-small: 0.39% of parameters are trainable.
   Adapters are 2-5 MB, not 250 MB.

3. Initialisation matters: B = zeros ensures the adapted model starts identical
   to the pre-trained model. Only during training do A and B diverge.

4. Rank selection heuristics: start at r=8. Increase to r=16 if underfitting.
   Decrease to r=4 if overfitting. r=1-2 rarely works for language tasks.

5. The PEFT library (get_peft_model, LoraConfig) automates exactly what you built
   in LoraLayer: freeze base, inject A and B, save only adapters.

### How this connects to Topic 7b

Topic 7b applies PEFT LoRA to DistilBERT for a classification task (encoder-only model).
You will see how target_modules changes for an encoder architecture, and how to combine
LoRA with 8-bit quantisation (QLoRA) to fit even larger models on the same T4 GPU.

### The Barclays system so far

Recent topics have added:
- Topic 6a: Full fine-tuning Flan-T5 (all params updated, catastrophic forgetting demo)
- Topic 6b: Transfer learning DistilBERT (frozen backbone, head only)
- Topic 7a: LoRA on FFN + Flan-T5 (freeze everything, two matrices per layer)
- Next: Topic 7b - PEFT and LoRA with DistilBERT
```

---

### Cell 37: code - Final Summary Cell

```python
# Summary of what was built and trained in this topic.

print("Topic 7a Summary")
print("=" * 55)
print()
print("LoRA implementation:")
print(f"  LoraLayer class: DONE")
print(f"  FFN with LoRA adapters: DONE")
print(f"  Trainable params (FFN, 3 layers, r=8): {trainable_lora:,}")
print(f"  Compression vs full FT: {train_pre / trainable_lora:.1f}x")
print()
print("SageMaker capstone:")
print(f"  Model: google/flan-t5-small (77M params)")
print(f"  LoRA rank: 8, alpha: 16, target: q + v projections")
print(f"  Job: {training_job_name}")
print(f"  Instance: ml.g4dn.xlarge (NVIDIA T4)")
print()
print("Next: Topic 7b -- PEFT LoRA on DistilBERT (encoder-only, classification)")
```

---

## Cell Count Summary

| Cell | Type | Content |
|------|------|---------|
| 1 | markdown | Title and learning objectives |
| 2 | code | Environment setup and installs |
| 3 | code | Imports and configuration |
| 4 | markdown | Section 1 header |
| 5 | code | Beat 1a: full weight delta explosion |
| 6 | code | Beat 1b: rank=d degenerates to full fine-tuning |
| 7 | markdown | Beat 2: LoRA decomposition diagram |
| 8 | code | Beat 3: LoraLayer from scratch |
| 9 | markdown | Lab 1 instructions (STAR method) |
| 10 | code | Lab 1 starter code |
| 11 | code | Lab 1 safety-net |
| 12 | code | Lab 1 verification |
| 13 | markdown | Lab 1 stretch + homework |
| 14 | markdown | Section 2 header + parameter comparison diagram |
| 15 | code | Pre-train FFN on FashionMNIST (Beat 3 setup) |
| 16 | code | Replace FC layers with LoRA; parameter count |
| 16b | code | Safety-net for lora_model and trainable_lora |
| 17 | markdown | Discussion prompt 1 (rank trade-offs) |
| 18 | code | Fine-tune with LoRA on MNIST |
| 19 | markdown | Lab 2 instructions (STAR method) |
| 20 | code | Lab 2 starter code |
| 21 | code | Lab 2 safety-net |
| 22 | code | Lab 2 verification |
| 23 | markdown | Lab 2 stretch + homework |
| 24 | markdown | Section 3 header: rank and alpha heuristics |
| 25 | code | Rank vs parameter count visualisation |
| 26 | markdown | Discussion prompt 2 (LoRA in production) |
| 27 | markdown | Section 4 header: PEFT capstone |
| 28 | code | Inspect Flan-T5 module names |
| 29 | code | Verify scripts_topic7a/ contents |
| 30 | code | Define HuggingFace estimator |
| 31 | code | Launch training job (wait=False) |
| 31b | code | Safety-net for training_job_name |
| 32 | code | Poll training job status |
| 33 | code | View CloudWatch logs |
| 34 | markdown | Section 4 interpretation notes |
| 35 | code | Load and test adapter post-training (optional) |
| 36 | markdown | Wrap-up: key takeaways + bridge to 7b |
| 37 | code | Final summary cell |

Total: 39 cells (37 original + Cell 16b safety-net for lora_model + Cell 31b safety-net for training_job_name). Markdown chains never exceed 3 consecutive without a code cell.
Two diagrams: lora-decomposition (Cell 7) and lora-parameter-comparison (Cell 14).
Two Tier-1 labs: Lab 1 (Cells 9-13, LoraLayer implementation), Lab 2 (Cells 19-23, layer replacement).
Every lab has safety-net, verification, stretch, and homework extension.
Variable `lora_model` fed to Cell 18; Cell 16b safety-net rebuilds lora_model if Cell 16 fails.
Variable `training_job_name` fed to Cells 32-35, 37; Cell 31b safety-net allows kernel-restart recovery.

---

## Four-Beat Arc Verification

| Concept | Beat 1 | Beat 2 | Beat 3 | Beat 4 |
|---------|--------|--------|--------|--------|
| LoRA motivation | Cell 5-6: delta_W explosion and rank=d failure | Cell 7: lora-decomposition diagram | Cell 8: LoraLayer from scratch | Cells 9-13: Lab 1 implement LoraLayer |
| LoRA on FFN | (Beat 1 shared with above) | Cell 14: parameter comparison diagram | Cells 15-16: pre-train FFN, replace layers, count params | Cells 19-23: Lab 2 replace all 6 layers |

---

## Compliance Checklist

- [x] numpy<2 in every install cell and in scripts_topic7a/requirements.txt
- [x] No em dashes, en dashes, unicode multiplication, or emojis anywhere
- [x] eval_strategy="epoch" (not evaluation_strategy) in train.py
- [x] No evaluate library -- inline token overlap using numpy in train.py
- [x] requirements.txt named exactly "requirements.txt" in source_dir (L4)
- [x] HuggingFace estimator only on ml.g4dn.xlarge (L1)
- [x] transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312" (L2)
- [x] SageMaker SDK pinned >=2.200.0,<3.0.0 (L3)
- [x] getpass NOT used for AWS credentials (role handles auth inside Studio)
- [x] peft>=0.6.0 in scripts_topic7a/requirements.txt
- [x] target_modules=["q", "v"] for Flan-T5-small (correct projection names)
- [x] TaskType.SEQ_2_SEQ_LM for encoder-decoder task type
- [x] Safety-net cells after Lab 1 (Cell 11) and Lab 2 (Cell 21)
- [x] Safety-net cell after Cell 16 (Cell 16b) for lora_model and trainable_lora
- [x] Safety-net cell after Cell 31 (Cell 31b) for training_job_name kernel-restart recovery
- [x] No more than 3 consecutive markdown cells without a code cell
- [x] # YOUR CODE placeholders do not hint at the answer
- [x] Exactly 2 diagrams with correct <!-- DIAGRAM: --> + [View diagram] format
- [x] STAR method applied to both labs
- [x] Stretch and Homework Extension after both labs
- [x] Two peer discussion prompts between major sections
- [x] Lab tier: Tier 1 for both labs (correct for Day 2, fifth topic)
- [x] Variable continuity from Topic 6b: device, set_seeds, sess, role, bucket, region
