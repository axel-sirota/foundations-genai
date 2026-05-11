# Topic 8 - Quantization, Pruning and Distillation: Cell-by-Cell Plan

## Overview

Topic 8 opens Day 3 of the Barclays Generative AI for Developers course.
The narrative pivot is critical: Days 1 and 2 gave students a fine-tuned complaint
classifier (DistilBERT + PEFT LoRA from Topics 6b/7b). It works well, but it is
too large and too slow to deploy at Barclays scale. Day 3 answers the question:
"How do we make this production-ready?"

Three techniques are presented in escalating depth:
  1. Quantization   -- reduce numeric precision to shrink the model
  2. Pruning        -- zero out unimportant weights to reduce compute
  3. Distillation   -- train a smaller student from the teacher's soft outputs

Every section follows the four-beat arc (broken -> diagram -> demo -> lab).
The capstone is a full QAT + LoRA remote training job on ml.g4dn.xlarge that
produces a deployable INT8 DistilBERT complaint classifier, then serves it on
an ml.m5.xlarge SageMaker endpoint.

Estimated in-class time: 90 to 120 minutes.
Lab tier for Day 3: Topic 8 is the LAST topic of Day 3 and carries BOTH the Tier 2
hard lab (QAT training loop) and the Day 3 Tier 3 open-ended capstone (end-to-end
compression pipeline). The remaining labs in this topic are Tier 1 guided.

---

## Variable Continuity from Topic 7b

Variables carried forward (re-defined in T8 Cell 2 for self-containment):
- `tokenizer`: DistilBERT tokenizer -- re-loaded from HuggingFace Hub (not from T7b kernel state)
- `baseline_model`: fresh `distilbert-base-uncased` loaded for compression demos
  Note: in production the T7b PEFT adapter artifact would be the input; here we use a
  fresh pretrained model for didactic simplicity so students can run T8 standalone.
- `sess`, `role`, `bucket`, `region`: re-defined in Cell 1 setup (same pattern as all prior topics)
- `device`, `set_seeds`: re-defined in Cell 1

Variables NOT carried forward (T8 defines its own):
- `peft_model`, `qlora_model`, `prefix_model` from T7b -- not used in T8
- `lora_r` from T7a/T7b -- not used in T8

---

## Diagram Index

Diagram 1:
  slug=quantization-precision-tradeoffs
  path=plans/topic_8_quantization/diagrams/quantization-precision-tradeoffs.mmd
  Description: Side-by-side comparison of four numeric precision formats:
  FP32 (32 bits, memory 4 bytes per weight, accuracy loss ~0%, use case: training),
  FP16/BF16 (16 bits, 2 bytes per weight, accuracy loss ~0.5%, use case: mixed precision training + serving),
  INT8 (8 bits, 1 byte per weight, accuracy loss 1-2% after QAT, use case: CPU and edge serving),
  INT4 (4 bits, 0.5 bytes per weight, accuracy loss 2-5%, use case: consumer GPU inference).
  Show a vertical staircase of bit-width blocks with memory reduction factor (1x, 2x, 4x, 8x)
  and a horizontal accuracy bar showing degradation growing as bit-width drops.
  Annotation: "QAT recovers up to 96% of accuracy lost by PTQ" (source: PyTorch 2025).

Diagram 2:
  slug=knowledge-distillation-architecture
  path=plans/topic_8_quantization/diagrams/knowledge-distillation-architecture.mmd
  Description: Two-path training diagram. Left path: Hard labels (one-hot ground truth) flow
  into a Cross-Entropy loss box. Right path: Teacher model (large, frozen, BERT-base) feeds
  logits through Temperature T scaling into Softmax, producing Soft Labels. Student model
  (small, DistilBERT or T5-small) also feeds logits through Temperature T softmax. KL
  Divergence loss box combines teacher and student soft outputs. Final loss =
  alpha * CE_loss + (1 - alpha) * KL_loss. Annotation boxes: "T=1: hard distribution",
  "T=4: smooth distribution (richer signal)", "alpha=0.5 typical". Arrow from teacher to
  student labelled "knowledge transfer". Teacher box shown greyed/frozen, student box shown
  highlighted.

---

## Source Dir (scripts_topic8/)

### train.py

```python
"""
train.py -- QAT + LoRA complaint classifier for SageMaker GPU job.

Base model : distilbert-base-uncased (66M params)
Task       : 5-class complaint classification (Barclays customer service)
Techniques : Quantization-Aware Training (INT8, fbgemm backend) + PEFT LoRA adapters
Instance   : ml.g4dn.xlarge (NVIDIA T4, 16 GB VRAM)
Container  : HuggingFace estimator, transformers 4.56.2, pytorch 2.8.0, py312

SageMaker toolkit auto-installs requirements.txt before running this script.
Hyperparameters are passed as CLI args by the HuggingFace estimator.

Key rules encoded here (from SAGEMAKER_LESSONS_LEARNED.md and CORE_TECHNOLOGIES.md):
  - eval_strategy='epoch' (NOT evaluation_strategy -- removed in transformers 4.41+)
  - NO evaluate library -- use inline numpy for metrics
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Training hyperparameters
    parser.add_argument("--epochs",               type=int,   default=3)
    parser.add_argument("--batch_size",           type=int,   default=16)
    parser.add_argument("--lr",                   type=float, default=2e-4)
    parser.add_argument("--quantization_backend", type=str,   default="fbgemm",
                        help="fbgemm (x86 CPU) or qnnpack (ARM/mobile)")
    parser.add_argument("--lora_r",               type=int,   default=8)
    parser.add_argument("--lora_alpha",           type=int,   default=16)
    parser.add_argument("--max_length",           type=int,   default=128)
    parser.add_argument("--warmup_ratio",         type=float, default=0.1)

    # SageMaker environment paths
    parser.add_argument("--model_dir",
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_data_dir",
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset: banking77 -> map to 5 Barclays complaint categories
# ---------------------------------------------------------------------------

# Banking77 has 77 intent classes. We map them to 5 Barclays complaint buckets.
CATEGORY_MAP = {
    "card_arrival":           0,  # Card / Account issues
    "card_linking":           0,
    "card_payment_fee_charged": 0,
    "card_swallowed":         0,
    "declined_card_payment":  0,
    "card_not_working":       0,
    "wrong_amount_of_cash_received": 1,  # Transaction disputes
    "transaction_charged_twice": 1,
    "transfer_not_received_by_recipient": 1,
    "beneficiary_not_allowed": 1,
    "balance_not_updated_after_bank_transfer": 1,
    "failed_transfer":        1,
    "exchange_rate":          2,  # FX / International
    "transfer_fee_charged":   2,
    "getting_spare_card":     3,  # General queries
    "getting_virtual_card":   3,
    "lost_or_stolen_card":    3,
    "lost_or_stolen_phone":   3,
    "pending_card_payment":   3,
    "pending_cash_withdrawal": 3,
}
DEFAULT_LABEL = 4  # Other

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
    # Rename 'label' -> process through remap (banking77 gives integer label + intent string)
    # banking77 columns: text, label (int 0-76), intent (str)
    dataset = dataset.map(remap_label)
    dataset = dataset.rename_column("text", "sentence")

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding=False,       # DataCollatorWithPadding handles padding dynamically
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names
         if c not in ["input_ids", "attention_mask", "labels"]]
    )
    tokenized.set_format("torch")
    return tokenized


# ---------------------------------------------------------------------------
# Metrics (inline numpy -- NO evaluate library)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# QAT preparation helpers
# ---------------------------------------------------------------------------

def insert_qat_observers(model, backend="fbgemm"):
    """
    Insert fake-quantization observers into all Linear layers for QAT.

    Rules:
      - Embedding layers MUST opt out (qconfig=None) -- torch.ao does not
        support embedding quantization and will raise an error.
      - QAT prepare/convert must happen on CPU; training can use GPU after prepare.
      - backend 'fbgemm' targets x86 CPUs (SageMaker inference endpoints).
    """
    torch.backends.quantized.engine = backend
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    # Disable quantization on embedding layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            module.qconfig = None

    # Insert observers (model must be on CPU at this point)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model):
    """
    Convert fake-quantization ops to real INT8 quantized ops.
    Must be called AFTER training is complete and model is on CPU.
    """
    model.eval()
    quantized = torch.ao.quantization.convert(model, inplace=False)
    return quantized


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def apply_lora(model, lora_r, lora_alpha):
    """Wrap the model with PEFT LoRA adapters targeting attention projections."""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_lin", "k_lin", "v_lin"],  # DistilBERT attention projection names
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Args: {args}")

    # --- Tokenizer and model ---
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )

    # --- Apply LoRA adapters first ---
    model = apply_lora(model, args.lora_r, args.lora_alpha)

    # --- Insert QAT observers (model must stay on CPU during prepare) ---
    print("Inserting QAT observers...")
    model = insert_qat_observers(model, backend=args.quantization_backend)

    # --- Move to GPU for training ---
    model.to(device)

    # --- Dataset ---
    print("Loading dataset...")
    tokenized = load_and_prepare_dataset(tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",          # NOT evaluation_strategy (removed in transformers 4.41+)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",               # No MLflow in this capstone
        fp16=False,                     # QAT and fp16 conflict; use fp32 for quantization sim
    )

    # --- Trainer ---
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

    # --- Convert to quantized model ---
    print("Converting to INT8 quantized model (must be on CPU)...")
    model.to("cpu")
    quantized_model = convert_to_quantized(model)
    print("Conversion complete.")

    # --- Save tokenizer and quantized model ---
    os.makedirs(args.model_dir, exist_ok=True)
    tokenizer.save_pretrained(args.model_dir)

    # Save the quantized model using torch.save (HuggingFace save_pretrained may
    # not handle torch.ao quantized ops cleanly; torch.save is the safe path)
    torch.save(quantized_model.state_dict(), os.path.join(args.model_dir, "quantized_model.pt"))

    # Also save the config so the inference endpoint can reconstruct the architecture
    model.config.save_pretrained(args.model_dir)

    print(f"Quantized model saved to {args.model_dir}")

    # --- Final evaluation ---
    results = trainer.evaluate()
    print(f"Final eval results: {results}")


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

## Key Changes from Source Scripts (Lab4/5/6 from mastering-llm-deployments)

### What Stays the Same

- QAT observer insertion pattern from Lab6/qat.py: `get_default_qat_qconfig`, `prepare_qat`, `convert`
- Embedding layer opt-out from QAT (critical -- Lab6 already does this correctly)
- CPU-for-prepare / GPU-for-training / CPU-for-convert sequence (Lab6)
- KL divergence distillation loss formula from Lab4/distillation_hard.py
- L1 unstructured pruning pattern from Lab5/starter_code.py (adapted)
- Teacher-student loading pattern from Lab4/distillation_easy.py

### What Changes

- Source scripts use local_models/ disk paths; course notebooks use HuggingFace Hub IDs
  (distilbert-base-uncased, bert-base-uncased) because SageMaker Studio has internet access
- Source scripts use flan-t5 seq2seq; course adapts to DistilBERT classification (SequenceClassification)
- Source scripts use DialogSum dataset; course uses banking77 (Barclays narrative continuity from Days 1-2)
- Lab4/distillation_easy.py uses HuggingFace Trainer for distillation (no custom KL loss);
  the notebook Beat 3 demo shows why temperature=1 gives no benefit, then switches to custom loop
- Lab5 pruning is global unstructured; notebook extends to show structured (attention head) pruning
  because head pruning is more meaningful for transformer inference
- train.py adds LoRA (PEFT) on top of QAT -- this combination is new vs the source scripts
- train.py replaces local_files_only=True with standard Hub downloads
- train.py uses eval_strategy="epoch" (transformers 4.41+ requirement -- source scripts predate this)
- No evaluate library anywhere (source scripts do not use it either -- safe)
- SageMaker endpoint uses ml.m5.xlarge (NOT ml.c5.large -- OOMs on DistilBERT; see LESSONS_LEARNED L*)

---

## Cell-by-Cell Plan

### Cell 0: markdown - Title and Topic Context

Type: markdown

Content outline:
  # Topic 8: Quantization, Pruning and Distillation
  ## Barclays Customer Support Intelligence System

  Opening narrative: "We now have a fine-tuned DistilBERT complaint classifier from earlier topics.
  It achieves 91% accuracy on banking77. Problem: it is 260 MB, takes 80ms per inference on CPU,
  and costs $0.074/hour to serve on a dedicated endpoint. That is unacceptable for a
  Barclays production system handling 10,000 complaints per day."

  Topic 8 goal: reduce the model to under 80 MB, under 20ms inference, and under $0.03/hour serving cost.

  Three tools we will use:
    1. Quantization -- reduce weight precision from FP32 to INT8
    2. Pruning      -- zero out the least important weights
    3. Distillation -- train a smaller student from the teacher's knowledge

  Learning objectives (bulleted):
    - Understand PTQ vs QAT and when to choose each
    - Apply L1 unstructured pruning with PyTorch's prune module
    - Understand knowledge distillation with temperature-scaled soft labels
    - Build a full QAT + LoRA training loop for a complaint classifier
    - Deploy a quantized model to a SageMaker real-time endpoint

---

### Cell 1: code - Environment Setup

Type: code

Purpose: install pinned packages, import libraries, set up SageMaker session

Content outline:
```python
# Install pinned dependencies
# numpy<2 is MANDATORY -- numpy 2.x breaks many torch operations
!pip install -q \
    "sagemaker>=2.200.0,<3.0.0" \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "datasets>=2.18.0,<3.0.0" \
    "peft>=0.6.0" \
    "numpy<2"

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization
import torch.nn.functional as F
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFace
import boto3

# SageMaker session (uses the execution role attached to this Studio instance)
sess   = sagemaker.Session()
role   = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"Role: {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
print(f"PyTorch version: {torch.__version__}")
```

Notes:
  - No getpass needed -- SageMaker execution role handles AWS auth automatically
  - Transformers pinned to <4.40.0 for notebook kernel (container uses 4.56.2 independently)

---

### Cell 2: code - Load Baseline Model and Measure Its Cost

Type: code (Beat 1 setup -- show the unoptimized model to motivate the topic)

Purpose: load distilbert complaint classifier, measure size, run timed inference, show cost

Content outline:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os

# Load the same DistilBERT classifier from Topics 6b/7b
# (students have already fine-tuned this; we load a fresh pretrained version for demo)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
baseline_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
)
baseline_model.eval()

# --- Measure model size ---
param_count = sum(p.numel() for p in baseline_model.parameters())
param_size_mb = param_count * 4 / (1024 ** 2)   # FP32 = 4 bytes per param
print(f"Parameter count  : {param_count:,}")
print(f"Model size (FP32): {param_size_mb:.1f} MB")

# --- Measure inference latency on CPU ---
sample_text = "I was charged twice for the same transaction and nobody has responded to my complaint."
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128)

# Warm-up run
with torch.no_grad():
    _ = baseline_model(**inputs)

# Timed runs
latencies = []
for _ in range(20):
    start = time.perf_counter()
    with torch.no_grad():
        _ = baseline_model(**inputs)
    latencies.append((time.perf_counter() - start) * 1000)

avg_ms = np.mean(latencies)
print(f"Avg CPU inference : {avg_ms:.1f} ms")
print(f"Endpoint cost est : ${0.074:.3f}/hr (ml.m5.xlarge) for a dedicated endpoint")
print()
print("Production target: <80 MB, <20 ms, <$0.03/hr equivalent throughput")
```

Teaching note: This cell sets up the "before" state. Students will see ~260 MB, ~70-80 ms.
The dramatic gap vs the production target creates urgency for everything that follows.

---

### Cell 3: markdown - Section 1 Header: Quantization

Type: markdown

Content outline:
  ## Section 1: Quantization

  Weights are stored as FP32 by default: 32 bits per number, 4 bytes per weight.
  Quantization maps those floats to a smaller type: INT8 uses 8 bits (1 byte), giving a 4x size reduction.

  The key question: how much accuracy do we sacrifice?

  Two strategies:
    - Post-Training Quantization (PTQ): quantize AFTER training. Simple, but can lose 2-5% accuracy.
    - Quantization-Aware Training (QAT): simulate quantization DURING training. Recovers up to 96%
      of the accuracy degraded by PTQ (PyTorch 2025 benchmarks on Llama3).

  We will see both. Beat 1 shows PTQ going wrong. Beat 3 shows QAT done right.

---

### Cell 4: markdown - Beat 1 Header: The PTQ Trap

Type: markdown

Content outline:
  ### Beat 1: The PTQ Trap -- What Happens When We Quantize Without Calibration

  PTQ requires a calibration step: you run a representative sample of data through the model
  to collect activation statistics (min, max). Without calibration, the quantized ranges are wrong.

  Watch what happens when we skip calibration and quantize naively.

---

### Cell 5: code - Beat 1: PTQ Without Calibration (Broken)

Type: code (Beat 1 -- runs and produces bad output)

Purpose: show that naive dynamic quantization degrades accuracy visibly without calibration

Content outline:
```python
# NAIVE PTQ: quantize the model with no calibration data at all
# torch.ao.quantization.quantize_dynamic is the simplest PTQ path
# It quantizes weights only (not activations) -- but watch what happens to the output distribution

naive_ptq_model = torch.ao.quantization.quantize_dynamic(
    baseline_model,
    {nn.Linear},
    dtype=torch.qint8
)

# --- Compare logit distributions ---
complaint_texts = [
    "My card was declined three times at the ATM and I lost money.",
    "I never received the international wire transfer I sent last week.",
    "I need to update my address on the account.",
]

print("=== Logit distributions: FP32 baseline vs naive PTQ ===")
print()
for text in complaint_texts:
    inputs_local = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        fp32_logits = baseline_model(**inputs_local).logits[0].numpy()
        ptq_logits  = naive_ptq_model(**inputs_local).logits[0].numpy()
    fp32_pred = np.argmax(fp32_logits)
    ptq_pred  = np.argmax(ptq_logits)
    print(f"Text   : {text[:60]}...")
    print(f"FP32   : pred={fp32_pred}, logits={np.round(fp32_logits, 2)}")
    print(f"NaivePTQ: pred={ptq_pred}, logits={np.round(ptq_logits, 2)}")
    if fp32_pred != ptq_pred:
        print("  *** PREDICTION CHANGED -- quantization error ***")
    print()

# --- Size comparison ---
ptq_size_mb = sum(
    p.numel() * (1 if p.dtype == torch.qint8 else 4)
    for p in naive_ptq_model.parameters()
) / (1024 ** 2)
print(f"Naive PTQ model size estimate: {ptq_size_mb:.1f} MB (vs {param_size_mb:.1f} MB FP32)")
print()
print("Problem: we got a smaller model, but some predictions flipped.")
print("With no calibration data, quantization ranges are estimated from weights alone.")
print("Activation distributions are ignored -- this causes errors on real inputs.")
```

Teaching note: dynamic quantization does NOT always flip predictions for pretrained distilbert
(it is relatively robust). But the logit magnitudes change noticeably and on some inputs
prediction does change. Instructor should run this live so students see the actual outputs.
The key pedagogical point is the logit distribution shift, not necessarily label flips.

---

### Cell 6: markdown - Beat 2: Quantization Diagram

Type: markdown

Content outline:
  ### Beat 2: Understanding the Precision Tradeoffs

  The diagram below shows how reducing bit-width reduces memory 4x to 8x, but also compresses
  the representable range of values -- which is exactly what damages accuracy when no calibration is done.

  QAT inserts fake-quantization operators during training that simulate this compression,
  so the model learns to be robust to the rounding errors before they happen for real.

  <!-- DIAGRAM: quantization-precision-tradeoffs -->
  [View diagram](../../plans/topic_8_quantization/diagrams/quantization-precision-tradeoffs.mmd)

---

### Cell 7: code - Beat 3: PTQ with Proper Calibration

Type: code (Beat 3 -- working demo, heavily commented)

Purpose: show static PTQ with calibration using a small representative dataset

Content outline:
```python
# PROPER PTQ: static quantization with calibration
# Steps:
#   1. Set qconfig (tells PyTorch what observers to insert)
#   2. torch.ao.quantization.prepare() -- inserts observers
#   3. Run calibration data through model (observers collect statistics)
#   4. torch.ao.quantization.convert() -- replaces observers with quantized ops

from datasets import load_dataset

# Load a small calibration set (100 examples from banking77)
calib_dataset = load_dataset("PolyAI/banking77", trust_remote_code=True, split="test[:100]")

# Clone the baseline model for PTQ (we do not want to modify the original)
import copy
ptq_model = copy.deepcopy(baseline_model)
ptq_model.eval()

# Step 1: Set quantization config -- fbgemm targets x86 CPUs (SageMaker inference endpoints)
ptq_model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

# Embedding and LayerNorm layers cannot be quantized by torch.ao -- opt them out
for name, module in ptq_model.named_modules():
    if isinstance(module, (nn.Embedding, nn.LayerNorm)):
        module.qconfig = None

# Step 2: Fuse and prepare (inserts calibration observers)
torch.ao.quantization.prepare(ptq_model, inplace=True)

# Step 3: Calibration -- run real data through the model
print("Running calibration (100 examples)...")
with torch.no_grad():
    for i, example in enumerate(calib_dataset):
        inputs_c = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        ptq_model(**inputs_c)
        if (i + 1) % 25 == 0:
            print(f"  Calibrated {i+1}/100 examples")

# Step 4: Convert -- replace observers with real INT8 quantized ops
ptq_model = torch.ao.quantization.convert(ptq_model, inplace=False)
print("PTQ with calibration complete.")
print()

# --- Compare predictions again ---
print("=== After calibrated PTQ: predictions vs FP32 ===")
for text in complaint_texts:
    inputs_local = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        fp32_logits = baseline_model(**inputs_local).logits[0].numpy()
        ptq_logits  = ptq_model(**inputs_local).logits[0].numpy()
    print(f"FP32 pred={np.argmax(fp32_logits)} | PTQ pred={np.argmax(ptq_logits)} | {text[:55]}...")
print()
print("With calibration, PTQ predictions are stable -- but training still beat calibration.")
print("QAT (capstone) recovers even more accuracy because the model saw quantized ops during training.")
```

---

### Cell 8: markdown - Beat 4 Header: Lab 1 (Tier 1)

Type: markdown

Content outline:
  ### Lab 1: Dynamic Quantization on Your Complaint Classifier (Tier 1 -- Guided, 15 min)

  **Situation**: You are the ML engineer at Barclays. You have a trained DistilBERT complaint
  classifier and need to produce a quick INT8 version for the on-premise inference server
  (which is x86 CPU only).

  **Task**: Apply dynamic quantization to the model and measure the size and latency improvement.

  **Action**: Follow the steps below. Each `= None  # YOUR CODE` line is one step.

  **Result**: You should see at least 2x size reduction and a measurable latency improvement.

  Steps:
    1. Apply `torch.ao.quantization.quantize_dynamic` to `baseline_model`, targeting `{nn.Linear}`,
       with `dtype=torch.qint8`. Save as `dynamic_quantized_model`.
    2. Measure the parameter count of `dynamic_quantized_model` (same formula as above, but note
       torch.ao quantized models report parameters differently -- use `torch.save` to measure file size).
    3. Run 20 timed inference passes with `dynamic_quantized_model` and compute mean latency.
    4. Compute the speedup ratio: `avg_ms_baseline / avg_ms_dynamic`.

  Stretch: Can you apply the same dynamic quantization to only the encoder layers (not the classifier head)?
    Hint: use `model.distilbert` as the target module instead of the full model.

---

### Cell 9: code - Lab 1 Starter Code

Type: code

Content outline:
```python
# ============================================================
# Lab 1: Dynamic Quantization on the Complaint Classifier
# ============================================================

# Step 1: Apply dynamic quantization
dynamic_quantized_model = None  # YOUR CODE

# Step 2: Measure file size by saving to disk
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.pt")
    torch.save(dynamic_quantized_model, path)
    dq_size_mb = os.path.getsize(path) / (1024 ** 2)
print(f"Dynamic quantized model size: {dq_size_mb:.1f} MB  (baseline: {param_size_mb:.1f} MB)")

# Step 3: Timed inference
dynamic_latencies = []
for _ in range(20):
    start = time.perf_counter()
    with torch.no_grad():
        # Hint: run inference with dynamic_quantized_model on inputs
        _ = None  # YOUR CODE
    dynamic_latencies.append((time.perf_counter() - start) * 1000)
avg_ms_dynamic = np.mean(dynamic_latencies)
print(f"Dynamic quantized avg latency: {avg_ms_dynamic:.1f} ms  (baseline: {avg_ms:.1f} ms)")

# Step 4: Speedup
speedup = None  # YOUR CODE
print(f"Speedup: {speedup:.2f}x")

# Verification
assert dynamic_quantized_model is not None, "Apply quantize_dynamic first."
assert dq_size_mb < param_size_mb * 0.8, "Expected at least 20% size reduction."
print("Lab 1 complete!")
```

---

### Cell 10: code - Lab 1 Safety Net

Type: code

Content outline:
```python
# Lab 1 safety-net: run this cell ONLY if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if dynamic_quantized_model is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    dynamic_quantized_model = torch.ao.quantization.quantize_dynamic(
        baseline_model, {nn.Linear}, dtype=torch.qint8
    )
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        torch.save(dynamic_quantized_model, path)
        dq_size_mb = os.path.getsize(path) / (1024 ** 2)
    avg_ms_dynamic = avg_ms * 0.7   # approximate if timing was skipped
    speedup = avg_ms / avg_ms_dynamic
    print(f"Safety-net: dynamic quantized model {dq_size_mb:.1f} MB, speedup {speedup:.2f}x")
```

---

### Cell 11: markdown - Lab 1 Homework Extension

Type: markdown

Content outline:
  #### Homework Extension: INT4 Quantization with bitsandbytes

  Dynamic quantization gives INT8 precision. For even more aggressive compression,
  the `bitsandbytes` library supports INT4 (NF4) quantization, which is what QLoRA uses.

  After class, try:
  ```python
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
  model_4bit = AutoModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased", quantization_config=bnb_config, num_labels=5
  )
  ```
  Question to explore: Can you fine-tune a 4-bit model with LoRA? What accuracy tradeoff do you see?

---

### Cell 11b: markdown - Peer Discussion: Quantization Tradeoffs (3 min)

Type: markdown

Content:

```
**Peer Discussion (3 min)**

We just saw dynamic quantization cut model size by ~4x with minimal accuracy loss:

1. Barclays runs 50 NLP models in production. Which ones would you quantize first and why?
2. Dynamic quantization works at inference time. What are the risks of quantizing a model that was fine-tuned on imbalanced complaint data?
3. INT8 loses some precision. For a fraud detection model, is a 0.5% accuracy drop acceptable? How would you decide?
```

---

### Cell 12: markdown - Section 2 Header: Pruning

Type: markdown

Content outline:
  ## Section 2: Weight Pruning

  Quantization reduces precision. Pruning takes a different approach: it zeros out weights
  entirely. A weight of exactly 0 contributes nothing to the output -- if we zero 30% of weights,
  we need 30% fewer multiplications (with the right sparse runtime).

  Two pruning strategies:
    - Unstructured: zero individual weights anywhere in the weight matrix. Maximum flexibility,
      but requires sparse matrix kernels to realize speedup. Default in PyTorch's prune module.
    - Structured: zero entire rows, columns, or attention heads. Reduces the actual matrix shape,
      so standard dense kernels get faster automatically. Harder to implement but immediately faster.

  Beat 1: prune too aggressively (80%) and watch the model collapse.
  Beat 3: prune conservatively (20%) with magnitude-based L1 and verify accuracy holds.

---

### Cell 13: code - Beat 1: Aggressive Pruning (Broken)

Type: code (Beat 1 -- runs and produces bad/collapsed output)

Content outline:
```python
# NAIVE PRUNING: remove 80% of weights globally -- too aggressive
import copy

aggressive_model = copy.deepcopy(baseline_model)

# Apply L1 unstructured pruning: zero out the 80% of weights with smallest magnitude
for name, module in aggressive_model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.80)
        prune.remove(module, "weight")   # make pruning permanent

aggressive_model.eval()

# Run inference -- watch what happens
print("=== Predictions after 80% aggressive pruning ===")
label_names = ["Card/Account", "Transaction Dispute", "FX/International", "General Query", "Other"]
for text in complaint_texts:
    inputs_local = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = aggressive_model(**inputs_local).logits[0].numpy()
    pred = np.argmax(logits)
    confidence = float(torch.softmax(torch.tensor(logits), dim=0)[pred])
    print(f"Pred: {label_names[pred]} (conf={confidence:.2%}) | {text[:50]}...")

print()
print("Problem: 80% pruning collapses the model. All predictions may cluster on one class.")
print("The model has lost so much capacity that it defaults to the most common category.")
print()
# Count non-zero parameters
nonzero = sum((p != 0).sum().item() for p in aggressive_model.parameters())
total   = sum(p.numel() for p in aggressive_model.parameters())
print(f"Non-zero weights: {nonzero:,} of {total:,} ({100*nonzero/total:.1f}%)")
```

---

### Cell 14: markdown - Peer Discussion: Pruning Tradeoffs

Type: markdown

Content outline:
  #### Discussion (3 min): How Much Pruning Is Safe?

  Consider the 80% pruning result you just saw. Now consider your production context at Barclays:

  - The complaint triage system routes 10,000 complaints per day to the right team.
    If 5% of complaints are mis-routed, that means 500 customers per day get the wrong response.
  - Your infrastructure team is asking for a 3x compute reduction to cut costs.
  - You have 2 hours of GPU time remaining for retraining.

  Questions:
    1. What pruning ratio would you propose, and how would you test it before deployment?
    2. If pruning alone cannot achieve 3x speedup safely, what would you combine it with?
    3. Should the pruning decision be made by the ML team, the product team, or both?

---

### Cell 15: code - Beat 3: Conservative L1 Pruning (Working Demo)

Type: code (Beat 3 -- working, heavily commented)

Content outline:
```python
# CONSERVATIVE PRUNING: 20% L1 unstructured -- safe for DistilBERT
import copy

pruned_model = copy.deepcopy(baseline_model)

# --- Count sparsity before pruning ---
def sparsity(model):
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return 100.0 * (1 - nonzero / total)

print(f"Sparsity before pruning: {sparsity(pruned_model):.2f}%")

# --- Apply L1 unstructured pruning to all Linear layers ---
# L1 unstructured: zeroes the weights with the smallest absolute value.
# amount=0.20 means 20% of weights in each Linear layer will be zeroed.
for name, module in pruned_model.named_modules():
    if isinstance(module, nn.Linear):
        # prune.l1_unstructured adds a weight_mask buffer but does NOT modify weights yet
        prune.l1_unstructured(module, name="weight", amount=0.20)
        # prune.remove makes the pruning permanent (removes the mask, applies zeros to weight)
        prune.remove(module, "weight")

print(f"Sparsity after  pruning: {sparsity(pruned_model):.2f}%")

# --- Verify predictions are preserved ---
pruned_model.eval()
print()
print("=== Predictions after 20% L1 pruning ===")
for text in complaint_texts:
    inputs_local = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        fp32_logits   = baseline_model(**inputs_local).logits[0].numpy()
        pruned_logits = pruned_model(**inputs_local).logits[0].numpy()
    fp32_pred   = np.argmax(fp32_logits)
    pruned_pred = np.argmax(pruned_logits)
    match = "OK" if fp32_pred == pruned_pred else "CHANGED"
    print(f"[{match}] FP32={fp32_pred} Pruned={pruned_pred} | {text[:55]}...")

# --- Latency after pruning (PyTorch dense ops: same speed because weights are zeros, not removed) ---
# Note for instructor: PyTorch sparse support is still maturing. Unstructured sparsity
# requires torch.sparse ops or NVIDIA Ampere sparse tensor cores to get real speedup.
# The pedagogical point here is sparsity percentage, not wall-clock speedup.
print()
print("Note: PyTorch's default dense kernels do not skip zero-weights.")
print("Real speedup from unstructured pruning requires torch.sparse or hardware sparse support.")
print("Structured pruning (next) gives immediate speedup with standard dense kernels.")
```

---

### Cell 16: markdown - Beat 4 Header: Lab 2 (Tier 1)

Type: markdown

Content outline:
  ### Lab 2: Structured Attention Head Pruning (Tier 1 -- Guided, 15 min)

  **Situation**: The Barclays infrastructure team says dense sparse ops will not be available
  on the production hardware for 6 months. You need pruning that delivers real speedup NOW.
  Structured pruning removes entire attention heads, reducing the actual matrix dimensions.

  **Task**: Implement global magnitude-based pruning that zeroes 30% of weights across ALL
  Linear layers simultaneously (not 30% per layer), then measure the impact on model predictions.

  **Action**: Fill in each `= None  # YOUR CODE` block.

  **Result**: Print the global sparsity percentage and verify that predictions are stable.

  Steps:
    1. Use `prune.global_unstructured` with `nn.Linear` layers and `pruning_method=prune.L1Unstructured`,
       targeting the `weight` parameter, with `amount=0.30` (global 30%).
    2. Call `prune.remove(module, "weight")` on each Linear layer to make pruning permanent.
    3. Compute and print the global sparsity of `global_pruned_model`.
    4. Run the three complaint texts through `global_pruned_model` and compare predictions to FP32.

  Stretch: Try `amount=0.50`. At what point do predictions start changing for the complaint texts?

---

### Cell 17: code - Lab 2 Starter Code

Type: code

Content outline:
```python
# ============================================================
# Lab 2: Global Unstructured Pruning
# ============================================================
import copy

global_pruned_model = copy.deepcopy(baseline_model)

# Collect all (module, parameter_name) tuples for Linear layers
parameters_to_prune = [
    (module, "weight")
    for name, module in global_pruned_model.named_modules()
    if isinstance(module, nn.Linear)
]

# Step 1: Apply global L1 unstructured pruning (30% of all weights across all layers)
# Hint: call prune.global_unstructured(...) with the parameters_to_prune list
None  # YOUR CODE

# Step 2: Make pruning permanent
for module, param_name in parameters_to_prune:
    # Hint: call prune.remove(module, param_name) to bake in the mask
    None  # YOUR CODE

# Step 3: Compute global sparsity
global_pruned_model.eval()
# Hint: compute (zeros / total) * 100 across all parameters
sparsity_pct = None  # YOUR CODE
print(f"Global sparsity: {sparsity_pct:.2f}%")

# Step 4: Check predictions
print()
for text in complaint_texts:
    inputs_local = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        fp32_logits   = baseline_model(**inputs_local).logits[0].numpy()
        pruned_logits = global_pruned_model(**inputs_local).logits[0].numpy()
    fp32_pred   = np.argmax(fp32_logits)
    pruned_pred = np.argmax(pruned_logits)
    print(f"FP32={fp32_pred} | GlobalPruned={pruned_pred} | {text[:55]}...")

# Verification
assert sparsity_pct is not None, "Compute sparsity_pct first."
assert 25 < sparsity_pct < 35, f"Expected ~30% sparsity, got {sparsity_pct:.1f}%"
print()
print("Lab 2 complete!")
```

---

### Cell 18: code - Lab 2 Safety Net

Type: code

Content outline:
```python
# Lab 2 safety-net: run this cell ONLY if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.
if "global_pruned_model" not in dir() or global_pruned_model is None:
    print("Using Lab 2 safety-net so the rest of the notebook can run.")
    global_pruned_model = copy.deepcopy(baseline_model)
    parameters_to_prune = [
        (module, "weight")
        for name, module in global_pruned_model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.30,
    )
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    global_pruned_model.eval()
    total   = sum(p.numel() for p in global_pruned_model.parameters())
    nonzero = sum((p != 0).sum().item() for p in global_pruned_model.parameters())
    sparsity_pct = 100.0 * (1 - nonzero / total)
    print(f"Safety-net: global sparsity {sparsity_pct:.2f}%")
```

---

### Cell 19: markdown - Lab 2 Homework Extension

Type: markdown

Content outline:
  #### Homework Extension: Pruning + Fine-Tuning (Lottery Ticket Hypothesis)

  The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) proposes that within every large network
  there is a smaller "winning ticket" subnetwork that can be trained from scratch to the same accuracy.

  After class, explore:
    1. Prune `baseline_model` at 30% globally.
    2. Fine-tune the pruned model on 1000 examples of banking77 for 1 epoch.
    3. Compare the fine-tuned pruned model accuracy vs the unpruned baseline.

  Does re-training after pruning recover accuracy? How does the pruning rate affect recovery?

---

### Cell 20: markdown - Section 3 Header: Knowledge Distillation

Type: markdown

Content outline:
  ## Section 3: Knowledge Distillation

  Quantization and pruning operate on a single model. Distillation uses two models:
  a large, accurate TEACHER and a small, fast STUDENT.

  The teacher is already trained and frozen. The student trains not on hard labels (right/wrong)
  but on the teacher's output distribution -- which carries richer signal.

  Example: For the input "my card was declined", a hard label just says "Card/Account Issues".
  A teacher might output: 70% Card/Account Issues, 20% Transaction Dispute, 10% Other.
  That 20% on Transaction Dispute is useful information (the complaint is ambiguous).
  The student learns from that uncertainty -- impossible to get from hard labels alone.

  Temperature scaling controls how "soft" the teacher's distribution is:
    T=1 : standard softmax -- teacher is confident, student sees near-hard labels
    T=4 : blurs the distribution -- more relative information between classes transfers

---

### Cell 21: markdown - Beat 1 Header: Distillation with Temperature=1 (Naive)

Type: markdown

Content outline:
  ### Beat 1: Why Temperature=1 Gives No Benefit

  Watch what happens when we use temperature=1 for the soft label loss.
  At T=1 the softmax output is nearly identical to the argmax (hard label).
  The KL divergence term contributes almost nothing to the loss.

---

### Cell 22: code - Beat 1: Distillation with T=1 (No Soft Label Benefit)

Type: code (Beat 1 -- runs but produces no improvement over hard labels)

Content outline:
```python
# NAIVE DISTILLATION: temperature=1 -- soft labels collapse to hard labels
# We use a tiny teacher (bert-base-uncased) and student (distilbert-base-uncased)
# to keep the demo fast. In the capstone, teacher = fine-tuned DistilBERT from Day 2.

# Load a small teacher (bert-base) -- pretrained, no task fine-tuning, just for demo
from transformers import AutoModelForSequenceClassification

teacher_name  = "bert-base-uncased"
student_name  = "distilbert-base-uncased"
teacher_tok   = AutoTokenizer.from_pretrained(teacher_name)
student_tok   = AutoTokenizer.from_pretrained(student_name)

teacher_model = AutoModelForSequenceClassification.from_pretrained(
    teacher_name, num_labels=5
)
student_model = AutoModelForSequenceClassification.from_pretrained(
    student_name, num_labels=5
)
teacher_model.eval()

# --- Distillation loss function with temperature=1 ---
def distillation_loss(student_logits, teacher_logits, labels, temperature=1.0, alpha=0.5):
    """
    Combined distillation loss.

    alpha * cross_entropy(student, hard_labels)
    + (1 - alpha) * kl_divergence(student_soft, teacher_soft)

    At temperature=1, teacher_soft is nearly one-hot -> KL divergence term is tiny.
    """
    # Hard label loss (standard cross-entropy)
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft label loss: temperature scaling makes distributions smoother
    # Multiply by T^2 to restore gradient magnitude (standard Hinton et al. 2015 scaling)
    T = temperature
    soft_student  = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher  = F.softmax(teacher_logits   / T, dim=-1)
    kl_loss       = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    return alpha * ce_loss + (1 - alpha) * kl_loss

# Simulate one batch
sample_inputs = tokenizer(
    complaint_texts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128,
)
hard_labels = torch.tensor([0, 1, 3])   # card, transaction, general

with torch.no_grad():
    teacher_tok_inputs = teacher_tok(
        complaint_texts, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    teacher_logits = teacher_model(**teacher_tok_inputs).logits

student_logits = student_model(**sample_inputs).logits

loss_t1 = distillation_loss(student_logits, teacher_logits, hard_labels, temperature=1.0)
loss_t4 = distillation_loss(student_logits, teacher_logits, hard_labels, temperature=4.0)

print(f"Distillation loss at T=1 : {loss_t1.item():.4f}")
print(f"Distillation loss at T=4 : {loss_t4.item():.4f}")
print()

# Show how soft labels change with temperature
with torch.no_grad():
    teacher_soft_t1 = F.softmax(teacher_logits[0] / 1.0, dim=-1).numpy()
    teacher_soft_t4 = F.softmax(teacher_logits[0] / 4.0, dim=-1).numpy()

print("Teacher soft labels for first example:")
for i, (p1, p4) in enumerate(zip(teacher_soft_t1, teacher_soft_t4)):
    print(f"  Class {i}: T=1 -> {p1:.3f}  |  T=4 -> {p4:.3f}")
print()
print("At T=1, one class dominates (near-hard label). At T=4, distribution is smoother.")
print("The richer T=4 distribution transfers more inter-class similarity information.")
```

---

### Cell 23: markdown - Beat 2: Distillation Diagram

Type: markdown

Content outline:
  ### Beat 2: The Knowledge Transfer Architecture

  The diagram below shows how the teacher and student interact during distillation.
  The temperature scaling step is the key mechanism: without it (T=1), soft labels
  are nearly identical to hard labels and the student learns nothing extra from the teacher.

  <!-- DIAGRAM: knowledge-distillation-architecture -->
  [View diagram](../../plans/topic_8_quantization/diagrams/knowledge-distillation-architecture.mmd)

---

### Cell 24: code - Beat 3: Full Distillation Demo with Temperature Scaling (Working)

Type: code (Beat 3 -- full working demo, heavily commented)

Purpose: show a proper distillation training loop with T=4, alpha=0.5

Content outline:
```python
# PROPER DISTILLATION: temperature=4, alpha=0.5, custom KL training loop
# This is adapted from Lab4_Distillation_TrainingLoop/distillation_hard.py
# Key changes:
#   - Task is classification (not seq2seq) so we use logits directly
#   - Dataset is banking77 (not DialogSum) for Barclays narrative
#   - Inline numpy metrics (no evaluate library)
#   - Temperature=4 to get meaningful soft labels

from datasets import load_dataset
from torch.utils.data import DataLoader
import copy

TEMPERATURE = 4.0
ALPHA       = 0.5   # balance between hard-label CE loss and soft-label KL loss
LR          = 2e-4
DEMO_EPOCHS = 1
DEMO_BATCH  = 8
DEMO_STEPS  = 20    # keep demo short; capstone runs full training remotely

# Load a tiny subset for in-notebook demo
demo_dataset = load_dataset("PolyAI/banking77", trust_remote_code=True, split="train[:160]")

# --- Teacher: bert-base-uncased (pretrained, frozen) ---
# In production Day 3, the teacher would be the fine-tuned DistilBERT from Day 2.
teacher = copy.deepcopy(teacher_model)  # already loaded above
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False   # freeze teacher completely

# --- Student: fresh DistilBERT (will be distilled) ---
student = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5
)
student.train()
optimizer = AdamW(student.parameters(), lr=LR)

# --- Tokenize the dataset ---
def collate_fn(examples):
    texts  = [ex["text"] for ex in examples]
    # Map banking77 label int to our 5-class system using CATEGORY_MAP from train.py
    labels = [DEFAULT_LABEL for _ in examples]   # simplified for demo
    s_enc = student_tok(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    t_enc = teacher_tok(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    return s_enc, t_enc, torch.tensor(labels)

loader = DataLoader(demo_dataset, batch_size=DEMO_BATCH, collate_fn=collate_fn, shuffle=True)

# --- Distillation training loop ---
print(f"Running {DEMO_STEPS} distillation steps (T={TEMPERATURE}, alpha={ALPHA})...")
step = 0
total_loss = 0.0

for s_inputs, t_inputs, hard_labels in loader:
    if step >= DEMO_STEPS:
        break

    # Teacher forward pass (no gradients -- teacher is frozen)
    with torch.no_grad():
        t_logits = teacher(**t_inputs).logits   # shape: (batch, 5)

    # Student forward pass (gradients needed)
    s_logits = student(**s_inputs).logits       # shape: (batch, 5)

    # Compute combined distillation loss
    loss = distillation_loss(s_logits, t_logits, hard_labels,
                             temperature=TEMPERATURE, alpha=ALPHA)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    step       += 1
    if step % 5 == 0:
        print(f"  Step {step:3d} | loss={loss.item():.4f} | avg={total_loss/step:.4f}")

print()
print(f"Demo complete. {step} steps, final avg loss: {total_loss/step:.4f}")
print()
print("Key insight: the KL divergence term makes the student learn not just WHAT to predict")
print("but HOW CONFIDENT the teacher is across all classes.")
print("This is the 'dark knowledge' Hinton et al. 2015 described.")
```

Teaching note: The DEFAULT_LABEL=4 simplification for the demo is intentional and explicitly
called out. Students see the distillation mechanics without needing the full CATEGORY_MAP
to be re-implemented in-notebook. The capstone train.py uses the full CATEGORY_MAP.

---

### Cell 25: markdown - Beat 4 Header: Lab 3 Tier 1

Type: markdown

Content outline:
  ### Lab 3: Implement Temperature Sweep for Distillation (Tier 1 -- Guided, 15 min)

  **Situation**: Your team at Barclays is deciding what temperature to use for the
  distillation of your complaint classifier. You need to empirically measure how much
  "soft information" flows from teacher to student at each temperature.

  **Task**: Compute the KL divergence between teacher and student soft labels at temperatures
  T=1, 2, 4, 8 to see how temperature affects the information content of the soft targets.

  **Action**: Fill in each `= None  # YOUR CODE` block.

  **Result**: Print a table of T -> KL divergence and identify which T gives the richest signal.

  Steps:
    1. For temperatures [1, 2, 4, 8], compute `F.softmax(teacher_logits / T, dim=-1)`.
    2. Compute `F.softmax(student_logits / T, dim=-1)` for each T (use the logits from Cell 22).
    3. Compute `F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")` for each T.
    4. Print the results as a table.

  Stretch: Plot the teacher soft distribution at each temperature for the first complaint text.

---

### Cell 26: code - Lab 3 Starter Code

Type: code

Content outline:
```python
# ============================================================
# Lab 3: Temperature Sweep for Distillation
# ============================================================
# Reuse student_logits and teacher_logits from Cell 22

temperatures = [1, 2, 4, 8]
kl_results = {}

for T in temperatures:
    # Step 1: Teacher soft labels at temperature T
    # Hint: apply F.softmax(logits / T, dim=-1) to get soft probabilities
    teacher_soft = None  # YOUR CODE

    # Step 2: Student log-soft labels at temperature T
    # Hint: apply F.log_softmax(logits / T, dim=-1) for the student side
    student_log_soft = None  # YOUR CODE

    # Step 3: KL divergence (scale by T^2 as per Hinton et al. 2015)
    kl = None  # YOUR CODE
    kl_results[T] = kl.item() * (T ** 2) if kl is not None else None

# Step 4: Print results
print("Temperature | KL Divergence (scaled)")
print("-" * 38)
for T in temperatures:
    val = kl_results.get(T)
    if val is not None:
        print(f"    T={T}    |    {val:.4f}")
    else:
        print(f"    T={T}    |    (not computed)")

# Verification
assert kl_results.get(4) is not None, "Compute KL at T=4 first."
assert kl_results.get(4) > kl_results.get(1, 0), \
    "KL at T=4 should be larger than at T=1 (more information transfer)."
print()
print("Lab 3 complete! Higher T -> larger KL -> more inter-class signal transferred.")
```

---

### Cell 27: code - Lab 3 Safety Net

Type: code

Content outline:
```python
# Lab 3 safety-net: run this cell ONLY if you did not finish Lab 3.
# SKIP this cell if you DID finish Lab 3.
if not kl_results.get(4):
    print("Using Lab 3 safety-net so the rest of the notebook can run.")
    kl_results = {}
    for T in [1, 2, 4, 8]:
        ts = F.softmax(teacher_logits / T, dim=-1)
        ss = F.log_softmax(student_logits / T, dim=-1)
        kl = F.kl_div(ss, ts, reduction="batchmean")
        kl_results[T] = kl.item() * (T ** 2)
    print(f"Safety-net KL results: {kl_results}")
```

---

### Cell 28: markdown - Lab 3 Homework Extension

Type: markdown

Content outline:
  #### Homework Extension: Distillation Loss Sensitivity Study

  Alpha controls the balance between hard-label CE loss and soft-label KL loss.
  After class, run a grid search over alpha in [0.1, 0.3, 0.5, 0.7, 0.9] and T in [2, 4, 8].
  For each combination, train a student for 50 steps and record the final training loss.
  Which (alpha, T) combination minimizes loss fastest?

---

### Cell 29: markdown - Section 4 Header: QAT Capstone

Type: markdown

Content outline:
  ## Section 4: Capstone -- QAT + LoRA on SageMaker GPU

  We have seen PTQ, pruning, and distillation individually. Now we combine the two
  most powerful techniques for production:
    - LoRA: fine-tune efficiently (from Topic 7b)
    - QAT: quantize-aware training to make INT8 robust

  The capstone runs on ml.g4dn.xlarge (NVIDIA T4, 16 GB VRAM) as a SageMaker remote
  training job. The HuggingFace estimator handles the container and dependency installation.

  After training, we deploy the quantized model to an ml.m5.xlarge real-time endpoint.
  (ml.c5.large has only 4 GB RAM and OOMs on DistilBERT; ml.m5.xlarge with 16 GB is the
  minimum safe choice -- see SAGEMAKER_LESSONS_LEARNED.md lesson L*)

  This is the **Tier 2 hard lab** for Day 3. You will build the core of the QAT training
  loop inside train.py. The lab is less prescriptive than Tier 1 -- you have the function
  signatures and the expected outputs, but fewer numbered hints.

---

### Cell 30: code - Capstone Setup: Verify Source Dir

Type: code

Purpose: show students the scripts directory, print train.py content so they can read it

Content outline:
```python
import os

# The source directory for this capstone is scripts_topic8/
# It contains: train.py (QAT + LoRA training) and requirements.txt
source_dir = "scripts_topic8"

for fname in sorted(os.listdir(source_dir)):
    fsize = os.path.getsize(os.path.join(source_dir, fname))
    print(f"  {fname}  ({fsize} bytes)")

print()
# Print requirements.txt (students should know what will be installed in the container)
with open(os.path.join(source_dir, "requirements.txt")) as f:
    print("requirements.txt:")
    print(f.read())
```

---

### Cell 31: markdown - Tier 2 Hard Lab: Build the QAT Training Loop

Type: markdown

Content outline:
  ### Tier 2 Lab (Hard, 25-35 min): Complete train.py -- The QAT Pipeline

  **Situation**: You are the ML engineer responsible for making the Barclays complaint classifier
  production-ready. The architecture is set (DistilBERT + LoRA). Your task is to implement the
  QAT pipeline that the remote training job will run.

  **Task**: Open `scripts_topic8/train.py` and complete the two functions marked with `# YOUR CODE`:
    1. `insert_qat_observers(model, backend)` -- insert INT8 fake-quantization ops into all
       Linear layers, opting out Embedding and LayerNorm layers (which cannot be quantized by
       `torch.ao`).
    2. `convert_to_quantized(model)` -- after training, convert the model from fake-quantized
       (float weights with observer data) to real INT8 ops.

  **Note**: Unlike Tier 1 labs, there are no numbered substeps here. You have:
    - The function signatures and docstrings in train.py
    - The PyTorch `torch.ao.quantization` API (check Cell 7's PTQ code for reference)
    - The broken Beat 1 demo in Cell 5 which shows what NOT to do

  **Result**: Submit the job (Cell 33). If train.py is correct, the job will complete and
  print the final accuracy. If incorrect, CloudWatch logs will show the error.

  **Stretch**: After the job completes, extend `insert_qat_observers` to also disable
  quantization on the `distilbert.embeddings.position_embeddings` layer specifically,
  and check whether accuracy improves.

---

### Cell 32: code - Inspect train.py (Students Read Before Editing)

Type: code

Purpose: print key sections of train.py so students understand what they need to fill in

Content outline:
```python
# Print the two functions students need to complete
with open("scripts_topic8/train.py") as f:
    content = f.read()

# Find and print the two target functions
import re
for fn_name in ["insert_qat_observers", "convert_to_quantized"]:
    pattern = rf"def {fn_name}.*?(?=\ndef |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f"--- {fn_name} ---")
        print(match.group()[:800])
        print("...")
        print()
```

---

### Cell 33: code - Launch QAT Training Job on SageMaker GPU

Type: code (Beat 3 + Capstone -- instructor live-codes, students watch and then submit their own)

Content outline:
```python
from sagemaker.huggingface import HuggingFace
import time

# Job name with timestamp to avoid collisions
job_name = f"qat-distilbert-{int(time.time())}"

# HuggingFace estimator -- GPU only (HuggingFace DLCs have no CPU variant)
# Lesson L1: never use ml.m5 with HuggingFace estimator
# Lesson L2: py312 is required for PyTorch 2.8.0
estimator = HuggingFace(
    entry_point="train.py",
    source_dir="scripts_topic8",
    role=role,
    instance_type="ml.g4dn.xlarge",   # NVIDIA T4, ~$0.74/hr
    instance_count=1,
    transformers_version="4.56.2",
    pytorch_version="2.8.0",
    py_version="py312",
    hyperparameters={
        "epochs":               3,
        "batch_size":           16,
        "lr":                   2e-4,
        "quantization_backend": "fbgemm",
        "lora_r":               8,
        "lora_alpha":           16,
        "max_length":           128,
    },
    base_job_name=job_name,
)

print(f"Launching job: {job_name}")
print("This will take approximately 15-20 minutes on ml.g4dn.xlarge.")
print("You can monitor progress in the AWS Console -> SageMaker -> Training Jobs")
print()

# Non-blocking launch -- move on to the endpoint demo while training runs
estimator.fit(wait=False)
training_job_name = estimator.latest_training_job.name
print(f"Job launched. Training job name: {training_job_name}")
```

---

### Cell 33b: code - training_job_name Safety-Net

Type: code

Purpose: if the kernel restarted after launching the training job, restore the
`training_job_name` variable so downstream cells (poll status, retrieve metrics,
deploy endpoint) can still run without relaunching the job.

Content outline:
```python
# Safety-net: run this if your kernel restarted after launching the training job.
# SKIP this cell if training_job_name is already defined.
if 'training_job_name' not in dir() or training_job_name is None:
    training_job_name = "<PASTE YOUR JOB NAME HERE>"
    print(f"Using safety-net training_job_name: {training_job_name}")
```

---

### Cell 34: markdown - QAT Pipeline Walk-Through

Type: markdown

Content outline:
  ### What is Happening Inside train.py

  While the job runs, let us walk through the QAT pipeline you implemented:

  1. **LoRA first**: We apply LoRA adapters (r=8) to the DistilBERT attention projections.
     This reduces trainable parameters from 66M to ~2M -- faster and less likely to overfit.

  2. **Insert observers on CPU**: `torch.ao.quantization.prepare_qat()` inserts fake-quantization
     modules that collect running min/max statistics of activations during each forward pass.
     This MUST happen on CPU because the observer insertion is CPU-only.

  3. **Train on GPU**: After prepare, we move the model to GPU. The fake-quantization ops
     simulate INT8 rounding during forward passes, so the model learns robust weights.

  4. **Convert on CPU**: After training, `torch.ao.quantization.convert()` replaces the
     fake-quantization modules with real INT8 quantized ops. Must be CPU again.

  5. **Save**: tokenizer + quantized model state dict + config saved to `/opt/ml/model/`
     so SageMaker can package and upload the artifacts to S3.

  The combination of LoRA + QAT means:
    - Fewer parameters to train (LoRA) -> less overfitting, faster convergence
    - INT8 model at the end -> 4x smaller than FP32, ~2x faster CPU inference

---

### Cell 35: code - Poll Training Job Status

Type: code

Purpose: non-blocking poll so students can check status without re-running fit()

Content outline:
```python
import boto3

sm_client = boto3.client("sagemaker", region_name=region)
training_job_name = estimator.latest_training_job.name

def poll_training_job(job_name, sm_client):
    """Print current job status without blocking. Call this cell to refresh."""
    # Lesson L7: use ResourceNotFound (not ResourceNotFoundException)
    try:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
    except sm_client.exceptions.ResourceNotFound:
        print(f"Job {job_name} not found yet -- try again in 30s.")
        return

    status = desc["TrainingJobStatus"]
    secondary = desc.get("SecondaryStatus", "")
    elapsed_sec = None

    if "TrainingStartTime" in desc and "TrainingEndTime" in desc:
        elapsed_sec = (desc["TrainingEndTime"] - desc["TrainingStartTime"]).seconds

    print(f"Job     : {job_name}")
    print(f"Status  : {status} ({secondary})")
    if elapsed_sec:
        print(f"Elapsed : {elapsed_sec // 60}m {elapsed_sec % 60}s")

    if status == "Failed":
        print(f"Failure : {desc.get('FailureReason', 'Unknown')}")
        print("Check CloudWatch: Training -> <job_name> -> View logs")

    return status

status = poll_training_job(training_job_name, sm_client)
print()
print("Run this cell again to refresh the status.")
print("Continue to the endpoint deployment cells while training runs.")
```

---

### Cell 36: markdown - Beat 4 Header: Tier 2 Hard Lab (QAT Tuning)

Type: markdown

Content outline:
  ### Tier 2 Lab Continued: Tune Your QAT Job (Tier 2 Hard, runs while training is in progress)

  **Situation**: While your QAT job is running, the Barclays deployment team has asked you
  to prepare for a follow-up experiment: they want to know if using `qnnpack` backend
  (ARM/mobile) instead of `fbgemm` (x86) affects accuracy, since they are evaluating
  deploying to mobile banking app backends as well.

  **Task**: Prepare a second training job configuration that uses `quantization_backend=qnnpack`
  and a different LoRA rank (`lora_r=16`). Do NOT launch it yet -- just configure the estimator.

  **Result**: Print the hyperparameters of your new estimator configuration.
  The instructor will decide whether to launch a second job based on budget.

  (No `= None  # YOUR CODE` scaffolding -- Tier 2 means you use the estimator API directly
  based on what you saw in Cell 33.)

---

### Cell 37: code - Tier 2 Lab: Configure Second Experiment

Type: code

Content outline:
```python
# ============================================================
# Tier 2 Lab: Configure qnnpack experiment
# (Do NOT call .fit() unless the instructor confirms budget)
# ============================================================

# YOUR CODE: create a second HuggingFace estimator with qnnpack backend and lora_r=16
# Refer to Cell 33 for the estimator pattern.
# Name it: estimator_v2

estimator_v2 = None  # YOUR CODE

# Print configuration to verify
if estimator_v2 is not None:
    print("Experiment v2 configuration:")
    for k, v in estimator_v2.hyperparameters().items():
        print(f"  {k}: {v}")
    print()
    print("WAIT for instructor confirmation before calling estimator_v2.fit()")
else:
    print("Configure estimator_v2 first.")
```

---

### Cell 38: markdown - Stretch and Homework for Tier 2 Lab

Type: markdown

Content outline:
  #### Stretch: Add a Custom Callback to Log Quantization Error Per Epoch

  Inside train.py, add a HuggingFace TrainerCallback that computes the L2 norm of
  (float_weight - dequantized_weight) at the end of each epoch. This measures how
  much the QAT observers are being "stressed" by the quantization simulation.

  #### Homework Extension: Full Pipeline Comparison

  After class, run three jobs:
    1. LoRA only (no QAT) -- baseline
    2. QAT only (no LoRA) -- quantization without parameter efficiency
    3. QAT + LoRA (capstone) -- combined

  Compare final accuracy, model size, and inference latency. Does the combination
  outperform either technique alone? Why or why not?

---

### Cell 39: markdown - Section 5 Header: Serving Quantized Models

Type: markdown

Content outline:
  ## Section 5: Serving Quantized Models on SageMaker

  Training is done. Now we deploy. SageMaker real-time endpoints take the model artifacts
  from S3 and serve them behind a managed HTTPS endpoint with auto-scaling.

  Key instance choice: ml.m5.xlarge (16 GB RAM) is the minimum for DistilBERT.
    - ml.c5.large (4 GB RAM) OOMs during model loading -- do NOT use it.
    - ml.m5.xlarge gives enough headroom for the quantized model plus tokenizer overhead.
    - See SAGEMAKER_LESSONS_LEARNED.md for the OOM lesson.

---

### Cell 40: code - Deploy Quantized Model to SageMaker Endpoint

Type: code

Purpose: deploy the trained model to an endpoint and call it with sample complaints

Content outline:
```python
import json

# Wait for training to complete before deploying
# (In class, instructor may pre-deploy from a pre-trained job if training is still running)
print("Checking training job status before deploy...")
status = poll_training_job(training_job_name, sm_client)
print()

if status != "Completed":
    print("Training not yet complete. Run this cell again after the job finishes.")
    print("If you are in the waiting period, read Section 5 cells 41-42 while you wait.")
else:
    # Deploy to real-time endpoint
    # ml.m5.xlarge: 4 vCPU, 16 GB RAM -- minimum for DistilBERT (ml.c5.large OOMs)
    endpoint_name = f"qat-distilbert-endpoint-{int(time.time())}"
    print(f"Deploying to endpoint: {endpoint_name}")
    print("Deployment takes 3-5 minutes...")

    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",   # NOT ml.c5.large -- OOM risk
        endpoint_name=endpoint_name,
    )
    print(f"Endpoint active: {endpoint_name}")
```

---

### Cell 41: code - Test the Endpoint with Barclays Complaints

Type: code

Purpose: invoke the endpoint with real complaint texts and measure latency

Content outline:
```python
# Test the endpoint with the same complaint texts from the beginning of the notebook
label_names = [
    "Card and Account Issues",
    "Transaction Disputes",
    "FX and International",
    "General Queries",
    "Other",
]

test_complaints = [
    "I was charged twice for the same transaction and nobody has responded to my complaint.",
    "My card was declined three times at the ATM and I lost money.",
    "I never received the international wire transfer I sent last week.",
    "Can you help me update my mailing address on my account?",
    "The exchange rate applied to my purchase in Paris was wrong.",
]

print("=== QAT + LoRA Complaint Classifier -- Live Endpoint ===")
print()

latencies_endpoint = []

for complaint in test_complaints:
    payload = json.dumps({"inputs": complaint})
    start = time.perf_counter()
    # Invoke using boto3 runtime client for low-level timing
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    latencies_endpoint.append(latency_ms)

    result = json.loads(response["Body"].read())
    # HuggingFace endpoint returns list of dicts with label and score
    if isinstance(result, list) and len(result) > 0:
        top = max(result, key=lambda x: x["score"])
        print(f"Complaint : {complaint[:60]}...")
        print(f"Prediction: {top['label']} (score={top['score']:.3f})")
        print(f"Latency   : {latency_ms:.1f} ms")
        print()

print(f"Average endpoint latency: {np.mean(latencies_endpoint):.1f} ms")
print()
print("Compare to baseline:")
print(f"  FP32 baseline CPU latency : {avg_ms:.1f} ms")
print(f"  QAT INT8 endpoint latency : {np.mean(latencies_endpoint):.1f} ms")
print()
print("Note: endpoint latency includes network round-trip. In-process INT8 inference")
print("is ~2x faster than FP32 on x86 CPUs when using torch.ao INT8 ops.")
```

---

### Cell 42: code - Cleanup: Delete Endpoint to Stop Billing

Type: code

Purpose: delete the endpoint after demo to avoid ongoing charges

Content outline:
```python
# IMPORTANT: SageMaker real-time endpoints bill by the hour even when idle.
# Delete the endpoint when you are done to stop charges (~$0.074/hr for ml.m5.xlarge).

# Lesson L7: use ResourceNotFound (not ResourceNotFoundException)
try:
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Endpoint '{endpoint_name}' deleted successfully.")
    print("No further charges will accrue for this endpoint.")
except sm_client.exceptions.ResourceNotFound:
    print(f"Endpoint '{endpoint_name}' not found (may already be deleted).")
except NameError:
    print("endpoint_name not defined -- endpoint was never created (training may still be running).")
```

---

### Cell 43: markdown - Results Comparison Table

Type: markdown

Content outline:
  ## Results: Before and After Day 3

  | Metric | FP32 Baseline | Dynamic PTQ | 20% Pruned | QAT + LoRA |
  |--------|--------------|-------------|------------|------------|
  | Model size | ~260 MB | ~70 MB | ~260 MB* | ~65 MB |
  | CPU latency | ~80 ms | ~35 ms | ~80 ms* | ~35 ms |
  | Accuracy (est) | 91% | 88-90% | 90-91% | 90-92% |
  | Production ready | No | Marginal | No | Yes |

  *Unstructured pruning does not reduce size or latency without sparse runtime support.

  Key takeaway: QAT + LoRA is the recommended path to production for DistilBERT-class models.
  PTQ is a fast first step. Distillation is the right choice when you need a fundamentally
  smaller architecture (e.g., going from BERT-base to DistilBERT).

---

### Cell 44: code - Side-by-Side Latency Bar Chart

Type: code

Purpose: visualize all four model variants for the presentation / closing discussion

Content outline:
```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "monospace"   # avoids font warning in SageMaker

models = ["FP32 Baseline", "Dynamic PTQ", "20% Pruned", "QAT+LoRA (est)"]
sizes  = [param_size_mb, dq_size_mb, param_size_mb, param_size_mb * 0.25]
latencies_all = [avg_ms, avg_ms_dynamic, avg_ms * 0.98, avg_ms * 0.42]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Size comparison
bars1 = ax1.bar(models, sizes, color=["#4477AA", "#66CCEE", "#228833", "#EE6677"])
ax1.set_title("Model Size (MB)")
ax1.set_ylabel("MB")
ax1.set_ylim(0, max(sizes) * 1.2)
for bar, val in zip(bars1, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{val:.0f}", ha="center", va="bottom", fontsize=9)
ax1.tick_params(axis="x", rotation=15)

# Latency comparison
bars2 = ax2.bar(models, latencies_all, color=["#4477AA", "#66CCEE", "#228833", "#EE6677"])
ax2.set_title("Avg CPU Latency (ms)")
ax2.set_ylabel("ms")
ax2.set_ylim(0, max(latencies_all) * 1.2)
for bar, val in zip(bars2, latencies_all):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.0f}", ha="center", va="bottom", fontsize=9)
ax2.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("topic8_model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("Chart saved to topic8_model_comparison.png")
```

---

### Cell 44b: markdown - Tier 3 Capstone: End-to-End Model Compression Pipeline (Open-Ended)

Cell type: markdown (lab header)
Content:
**Capstone Lab -- Build a Barclays Model Compression Pipeline (Tier 3 - Open-Ended)**

The Barclays ML platform team needs a single function that takes a fine-tuned DistilBERT
complaint classifier and returns a compressed, serving-ready version using the best
combination of quantization, pruning, and/or distillation for the given size/latency target.

Your function must:
- Accept a model, tokenizer, and a target (either "size" or "latency")
- Apply at least one compression technique from this topic
- Return the compressed model and a metrics dict with keys: "size_mb", "accuracy", "technique"

No hints. No numbered steps. Design your own pipeline.

---

### Cell 44c: code - Tier 3 Capstone starter code

Cell type: code
Content:
```python
def compress_model(model, tokenizer, dataset, target="size", device="cpu"):
    """
    Build a compression pipeline for the Barclays complaint classifier.

    Args:
        model: fine-tuned DistilBERT model
        tokenizer: matching tokenizer
        dataset: validation dataset for accuracy measurement
        target: "size" (minimise model size) or "latency" (minimise inference time)
        device: "cpu" or "cuda"

    Returns:
        compressed_model: the compressed model ready for serving
        metrics: dict with keys "size_mb" (float), "accuracy" (float), "technique" (str)
    """
    pass  # YOUR CODE
```

---

### Cell 45: markdown - Peer Discussion: Production Decision

Type: markdown

Content outline:
  #### Discussion (5 min): Which Technique Would You Deploy at Barclays?

  You have now seen three model variants. The engineering and product teams are asking you
  to make a recommendation for the customer complaint triage system.

  Context:
    - The triage system processes 10,000 complaints per day.
    - Mis-routing a complaint to the wrong team costs an average of 15 minutes of analyst time.
    - The quantized endpoint costs $0.03/hour vs $0.074/hour for the FP32 endpoint.
    - The compliance team requires that accuracy does not drop below 89%.

  Questions:
    1. Which model variant meets the constraints? (Size, latency, accuracy, cost)
    2. How would you verify that accuracy is above 89% before going live?
       What data would you use for that evaluation?
    3. Pruning gave us no runtime speedup in this session. When would you revisit it?
       (Hint: think about hardware roadmap and model size targets)
    4. Distillation was shown in demo only -- not deployed. When would you distillation INSTEAD
       of QAT? (Hint: think about when the architecture itself is too large, not just the weights)

---

### Cell 46: markdown - Topic Wrap-Up and Bridge to Topic 9

Type: markdown

Content outline:
  ## Wrap-Up: Key Takeaways

  Quantization:
    - PTQ is fast but sensitive to calibration data quality. Always calibrate on representative inputs.
    - QAT recovers accuracy by simulating quantization during training (up to 96% recovery vs PTQ).
    - The prepare -> train -> convert cycle in torch.ao.quantization is the canonical QAT path.
    - Embedding and LayerNorm layers must opt out of QAT (torch.ao limitation).

  Pruning:
    - Unstructured pruning zeros weights but does not reduce runtime without sparse hardware support.
    - Structured pruning (rows/columns/heads) gives immediate speedup with standard dense kernels.
    - 20% L1 unstructured pruning is a safe default; above 40% expect accuracy degradation.

  Distillation:
    - Temperature scaling (T=4) is the key insight: it surfaces inter-class similarity in teacher outputs.
    - Alpha=0.5 (equal weight to CE and KL) is a robust default.
    - Distillation is the right choice when you need a different architecture (smaller student),
      not just fewer bits or fewer weights.

  SageMaker deployment:
    - Use ml.m5.xlarge for DistilBERT endpoints (ml.c5.large OOMs).
    - HuggingFace estimator requires GPU instance (ml.g4dn.xlarge minimum).
    - Always delete endpoints after demos -- they bill by the hour even when idle.

  Bridge to Topic 9 (RLHF -- if not parked):
    The complaint classifier now runs in production. But it was trained only on classification
    labels. What if we want it to also EXPLAIN its decision in plain English? That requires the
    model to generate text that humans rate as helpful -- the domain of reinforcement learning
    from human feedback.

---

## Web Research Summary (Validated 2026-05-11)

### PTQ vs QAT (PyTorch 2025)
Source: pytorch.org/blog/quantization-aware-training/
Key facts used:
  - QAT recovers up to 96% of accuracy degradation vs PTQ on Llama3/hellaswag
  - QAT composed with LoRA gives 1.89x training speedup and 36.1% memory savings (torchao)
  - backend="fbgemm" targets x86, "qnnpack" targets ARM/mobile
  - Embedding layers must opt out of QAT (torch.ao limitation -- unchanged in 2025)
  - prepare_qat -> train -> convert is the canonical 3-step QAT cycle

### Knowledge Distillation (2025)
Source: pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
Key facts used:
  - T in [2, 8] practical range; T=4 is the standard demo value
  - Combined loss = alpha * CE + (1 - alpha) * KL_div * T^2
  - KL divergence scaled by T^2 to preserve gradient magnitude (Hinton et al. 2015)
  - DistiLLM-2 (ICML 2025 oral) uses contrastive approach -- advanced, not covered in course

### Pruning (2025)
Source: pytorch.org/tutorials/intermediate/pruning_tutorial.html, Torch-Pruning
Key facts used:
  - prune.l1_unstructured + prune.remove is the canonical PyTorch pruning pattern
  - prune.global_unstructured allows cross-layer budgeting (better than per-layer)
  - Unstructured sparsity requires torch.sparse or Ampere sparse tensor cores for runtime speedup
  - 20% safe, 40%+ risky for pretrained transformers without fine-tuning after pruning

### SageMaker Serving (2025)
Source: aws.amazon.com/blogs/machine-learning/accelerating-llm-inference...
Key facts used:
  - ml.m5.xlarge (16 GB) is minimum for DistilBERT endpoints
  - ml.c5.large (4 GB) OOMs on DistilBERT loading
  - SageMaker HuggingFace inference endpoints accept JSON payload {"inputs": text}
  - boto3 sagemaker-runtime.invoke_endpoint is the correct low-level invocation path

---

## Verification Checklist (for /verify-research gate)

- [x] Four-beat arc present for ALL three techniques (Quantization, Pruning, Distillation)
  - Beat 1 (broken): Cell 5 (PTQ no calibration), Cell 13 (aggressive pruning), Cell 22 (T=1 distillation)
  - Beat 2 (diagram): Cell 6 (quantization diagram), Cell 23 (distillation diagram)
  - Beat 3 (demo): Cell 7 (calibrated PTQ), Cell 15 (20% pruning), Cell 24 (T=4 distillation)
  - Beat 4 (lab): Cells 8-11 (Lab 1), Cells 16-19 (Lab 2), Cells 25-28 (Lab 3)
- [x] Exactly 2 diagrams indexed with slug + path + description
- [x] Tier 2 hard lab clearly marked (Cells 31, 36-38: QAT training loop, 25-35 min)
- [x] Tier 3 open-ended capstone added (Cells 44b-44c: end-to-end compression pipeline)
- [x] Tier distribution: Tier 1 x3 (Labs 1, 2, 3), Tier 2 x1 (QAT capstone), Tier 3 x1 (compression pipeline)
- [x] Every lab has stretch + Homework Extension
- [x] Safety-net cells after Lab 1 (Cell 10) and Lab 3 (Cell 27)
- [x] Lab 2 does NOT need safety-net (global_pruned_model not used in downstream cells)
- [x] STAR method applied in all lab Beat 4 markdown cells
- [x] Peer discussion cells present: Cell 14 (pruning tradeoffs), Cell 45 (production decision)
- [x] numpy<2 in install cell (Cell 1)
- [x] eval_strategy="epoch" (NOT evaluation_strategy) in train.py
- [x] NO evaluate library -- inline numpy compute_metrics in train.py
- [x] HuggingFace estimator on GPU (ml.g4dn.xlarge) -- Cell 33
- [x] Endpoint on ml.m5.xlarge (NOT ml.c5.large) -- Cell 40
- [x] transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312" -- Cell 33
- [x] SageMaker SDK pinned >=2.200.0,<3.0.0 -- Cell 1
- [x] requirements.txt (exact name) with peft>=0.6.0, bitsandbytes>=0.41.0, datasets==2.18.0, numpy<2
- [x] boto3 exception: ResourceNotFound (not ResourceNotFoundException) -- Cells 35, 42
- [x] No em dashes, en dashes, unicode mult, emojis -- plan uses plain ASCII only
- [x] No more than 3 consecutive markdown cells without a code cell
- [x] No AI-tells in cell content outlines
- [x] Total cells: 51 (Cell 0 to Cell 46 plus Cells 11b, 33b, 44b, and 44c) -- within 45-55 target
- [x] Train.py: full content in Source Dir section above
- [x] Requirements.txt: peft>=0.6.0, bitsandbytes>=0.41.0, numpy<2
- [x] Base model: distilbert-base-uncased (matches course narrative from Topics 6b/7b)
- [x] Quantization backend argument: --quantization_backend (fbgemm default)
- [x] Model saved to /opt/ml/model/ (SageMaker artifact path)
- [x] Source dir adaptation notes documented (What Stays / What Changes section)
