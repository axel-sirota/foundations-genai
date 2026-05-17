# Rework Design Doc: topic_optional_lora_ffn (standalone optional deep-dive)

## Purpose of this document

This is a cell-by-cell rework plan for the notebook pair:

- `Exercises/topic_optional_lora_ffn/topic_optional_lora_ffn.ipynb`
- `Solutions/topic_optional_lora_ffn/topic_optional_lora_ffn.ipynb`

A notebook-building agent must be able to implement this rework WITHOUT re-reading
the original notebook. Every cell of the reworked notebook is specified below: cell
number, type, purpose, and either full content or a precise change description
(KEEP / EDIT / NEW / DELETE / MERGE), with old text quoted and new text given.

This doc follows the structure of the three sibling optional design docs
(`optional_attention_python.md`, `optional_attention_pytorch.md`,
`optional_transformers.md`) - same KEEP/EDIT/NEW/DELETE/MERGE vocabulary, same
Codex-findings-resolved table, same Solutions-twin notes.

## Why the rework

The notebook (formerly "Topic 7a - LoRA from Scratch") is being DEMOTED from a
required sequential Day-2 topic to a STANDALONE OPTIONAL deep-dive. It is the only
one of the four optionals that never received a design doc; this is it.

The current notebook assumes the linear course path. It must become fully
self-contained: it cannot assume any earlier topic ran, it cannot chain forward to
"Topic 7b", and it is NOT part of the S3 handoff chain (optionals never are - see
MEGAPLAN.md section 4).

Per the MEGAPLAN required/optional split: the LoRA CONCEPT (rank, alpha, dropout,
why it works) is taught as a required mini-lesson inside required Topic 6
(`topic_6_peft_lora_distilbert`, the R8 LoRA mini-lesson). THIS notebook is the
OPTIONAL from-scratch BUILD: a learner implements `LoraLayer` by hand, wraps an
FFN, and runs a real PEFT capstone. Wording throughout must reflect that split:
this notebook is never described as required, and never as the only place LoRA is
explained. It is the deep-dive build for learners who want to open the box.

### What stays the same

- The four-beat teaching arc (Beat 1 broken/naive code, Beat 2 diagram, Beat 3
  working demo, Beat 4 lab).
- Lab 1 (Tier 1 guided, `LoraLayerStudent`) and Lab 2 (Tier 1 guided,
  `replace_fc_with_lora_student`), their STAR framing, Stretch blocks, and
  Homework Extensions.
- The from-scratch `LoraLayer` build, the FashionMNIST pre-train, the
  MNIST LoRA fine-tune, the rank/alpha treatment.
- The remote GPU training-job capstone using
  `scripts_optional_lora_ffn/train.py` on `ml.g4dn.xlarge` (HuggingFace
  estimator, GPU-only - see SAGEMAKER_LESSONS_LEARNED L1).
- All architecture code, all verification cells, all four safety-net cells in the
  Exercise.
- Plain ASCII only. No em-dashes, en-dashes, Unicode multiplication signs, emojis.
- The two diagrams (`lora-decomposition.mmd`,
  `lora-parameter-comparison.mmd`) and their `<!-- DIAGRAM: -->` placeholders.
- `numpy<2` pinned in the install cell.

### What changes (summary)

1. NEW first markdown cell (cell 0): OPTIONAL / SUPPLEMENTARY banner. States this
   is an optional from-scratch deep-dive, that the LoRA concept is taught in the
   required path (Topic 6 mini-lesson), who it is for, that it is self-contained
   and runs cold from a fresh kernel, and that the required path does not need it.
2. NEW second markdown cell (cell 1): a motivation opening - "Why would a model
   USER want to understand LoRA internals?" - placed before any math.
3. EDIT the title cell: drop "Topic 7a", drop the "| Topic 7a" subtitle, drop the
   "Estimated time: 90 to 120 minutes in class" line (a standalone deep-dive has
   no class slot), and reframe "Why this matters to Barclays" so it does not imply
   a course position.
4. REPLACE the "Day 2 System Overview" / "YOU ARE HERE" course-progression table
   cell in place with a short static "how this notebook is organised" note (no
   progression table, no T4/T5/T6a/T6b/T7a/T7b rows).
5. EDIT the Section 4 capstone intro: keep it, just confirm it has no course
   chaining (it does not - only edited if a stray reference is found at build
   time).
6. EDIT the Wrap-Up cell: remove the "How this connects to Topic 7b" section, the
   "The Barclays system so far" topic list, and "Next session: Topic 7b". Reframe
   as standalone optional-track closure.
7. EDIT the final summary code cell: drop "Topic 7a Summary" and the trailing
   "Next: Topic 7b ..." print line.
8. Normalize all cell ids (several code cells have `id: null` in the source).
9. NEW: an AWS / SageMaker prerequisite note + a credentials/bucket GUARD code
   cell placed immediately BEFORE the capstone job-submission cell, matching the
   pattern used in `optional_transformers.md` (Codex R4 family). A standalone
   learner with no AWS credentials gets a clear, actionable message and a
   `capstone_ready` boolean instead of a raw `NoCredentialsError`. The submit cell
   then checks `capstone_ready`.

### Why a guard cell is added here (and how it stays minimal)

The three sibling optionals split on this: the two attention optionals run fully
offline and need no guard; `optional_transformers` HAS a remote GPU capstone and
therefore got a prerequisite note + guard cell. This notebook also has a remote
GPU capstone (the Flan-T5 PEFT job), so it follows the `optional_transformers`
precedent: one prerequisite markdown cell and one guard code cell, both placed
immediately before the job-submission cell. The guard never raises; it sets
`capstone_ready`. This is the only structural addition beyond the banner and
motivation cells.

### Standalone / cold-run audit

This notebook already runs cold from a fresh kernel. The audit:

- It defines `LoraLayer`, `LoraLayerStudent`, `FFNModel`, `replace_fc_with_lora`,
  `replace_fc_with_lora_student`, `set_seeds`, `device` all in-notebook. No symbol
  is imported from another notebook.
- It is NOT in the S3 handoff chain. It has NO S3 LOAD cell and writes no S3
  artifact that any required notebook reads. The capstone WRITES
  `s3://<default-bucket>/lora-flan-t5/output/model.tar.gz` (the trained adapter),
  consumed by nobody else; `scripts_optional_lora_ffn/train.py` +
  `requirements.txt` are consumed only by the SageMaker job this same notebook
  submits. No required notebook depends on either. This is the R6-style
  "writes-only-its-own-artifacts" guarantee; the doc states it so the builder does
  not add a dependency.
- Downloaded assets: the FashionMNIST/MNIST datasets via `torchvision.datasets`
  (with `download=True`), and `google/flan-t5-small` via `AutoModelForSeq2SeqLM`.
  See "Offline behaviour" below.
- The capstone needs AWS credentials; the new guard cell handles a credential-less
  environment gracefully.

### Offline behaviour

Two of the notebook cells download assets:

- The FashionMNIST/MNIST pre-train and fine-tune cells (`datasets.FashionMNIST`,
  `datasets.MNIST` with `download=True`). On the SageMaker Studio image this works.
  These are tiny standard datasets cached after the first run.
- The Flan-T5 inspection cell (`AutoModelForSeq2SeqLM.from_pretrained(
  "google/flan-t5-small")`) - a public, ungated model, no HF token needed.

These two notebook cells are KEEP cells with no offline fallback in the source.
Adding a synthetic fallback for image datasets and for a 77M-parameter seq2seq
model is out of scope for this rework and would change the teaching content
materially (the FashionMNIST pre-train must produce a real ~87%-accurate model for
the LoRA transfer demo to be meaningful). The pragmatic, low-risk decision: the
banner is explicit that the notebook needs network access for two specific
downloads (the small image datasets and the public Flan-T5-small model), the same
way `optional_transformers` is explicit that its capstone needs AWS. This is the
honest contract: the from-scratch LoRA math (Sections 1-3 core) is the deep-dive,
and the two download cells are clearly flagged. The builder must NOT add a
`nltk.download`-style hidden network call; the only downloads are the two already
present, and the banner names them.

### Cell count

- Source Exercises notebook: 49 cells (indices 0..48).
- Reworked Exercises notebook: 53 cells (indices 0..52).
- Net change: +4 NEW cells (banner, motivation, AWS prerequisite note, AWS guard).
  One markdown cell (the "YOU ARE HERE" table) is REPLACED in place. No cells are
  deleted.

### Solutions twin

The source Solutions notebook is ALSO 49 cells and currently differs from the
Exercises notebook in EXACTLY four cells - the lab/homework cells 14, 18, 29, 33
(verified by a cell-by-cell diff). Critically, the source Solutions notebook STILL
CONTAINS all four safety-net cells (source cells 15, 25, 30, 42). Per CLAUDE.md and
the sibling optional docs, lab safety-net cells must be REMOVED from the Solutions
twin (the filled lab cell IS the solution there). The training-job safety-net is an
operational recovery cell and is KEPT.

Reworked Solutions notebook = reworked Exercises notebook with these differences:

1. Lab cells filled (already filled in the source for the lab content - carry the
   same fills): `LoraLayerStudent` (reworked cell 17), the Homework Extension 1
   starter (reworked cell 21), `replace_fc_with_lora_student` (reworked cell 32),
   the Homework Extension 2 rank-sweep starter (reworked cell 36). The exact
   solution bodies are the ones already in the source Solutions notebook (quoted
   in the cell sections below).
2. The two LAB safety-net cells are DELETED from the Solutions notebook:
   - reworked cell 18 (Lab 1 safety-net, `LoraLayerStudent = LoraLayer` fallback)
   - reworked cell 33 (Lab 2 safety-net, `replace_fc_with_lora_student` fallback)
3. The Section-2 LoRA-model safety-net (reworked cell 28, "Rebuilding lora_model
   from safety-net") is an OPERATIONAL recovery cell, not a lab safety-net - it
   recovers `lora_model` if the heavy training cell before it failed or was
   skipped. It is KEPT in the Solutions notebook unchanged, the same way
   `optional_transformers` keeps its training-job safety-net.
4. The training-job safety-net (reworked cell 47, "run this if your kernel
   restarted after launching the training job") is an OPERATIONAL recovery cell -
   KEPT in the Solutions notebook unchanged.
5. The AWS guard cell (reworked cell 45) is a runtime guard, not a lab - KEPT
   unchanged in the Solutions notebook.

Result: Solutions notebook has 51 cells (53 minus the 2 deleted lab safety-nets).

Build order: build the Exercises notebook fully first, then `cp` it to Solutions
and apply: fill the 4 lab/homework cells, delete the 2 lab safety-net cells.

---

## Cell-by-cell plan (Exercises notebook)

Reworked numbering runs 0..52. "Maps to original cell N" refers to the verified
49-cell source notebook (indices 0..48).

---

### Cell 0 - NEW - markdown - Optional/Supplementary banner

Action: NEW. Becomes the very first cell. Full content:

```markdown
# Optional Deep-Dive: LoRA From Scratch

> **This is an optional supplementary notebook.** The main course path does not
> require it. The LoRA concept itself - rank, alpha, why frozen weights plus two
> small matrices work - is taught as a short mini-lesson in the required topics.
> This notebook is the from-scratch BUILD, for learners who want to see the
> internals. You can complete every required topic without opening it.

### Who this is for

This deep-dive is for developers who want to see what is actually inside a LoRA
adapter. You will implement `LoraLayer` by hand in PyTorch, freeze a base model,
wrap a feed-forward network with your own adapters, and then run the same idea at
scale with the HuggingFace PEFT library on a real seq2seq model. If you are happy
treating PEFT as a black box you can safely skip this. If you are the kind of
engineer who wants to open the box, read on.

### Is it self-contained?

Yes. This notebook does not depend on any other notebook in the course and is not
part of any topic-to-topic data handoff. You can open it cold, with a fresh
kernel, and run it start to finish. Every class and helper it uses is defined
inside this notebook.

### What it needs

- A notebook kernel for Sections 1 to 3 (the from-scratch LoRA build). CPU is fine.
- Network access for two downloads: the small FashionMNIST and MNIST image
  datasets (used to pre-train and then LoRA-adapt a tiny feed-forward network),
  and the public `google/flan-t5-small` model. Both are standard, ungated, and
  small. No HuggingFace token is required.
- For the capstone only: working AWS / SageMaker credentials, because the capstone
  submits one short remote GPU training job (`ml.g4dn.xlarge`). That step is
  flagged with its own prerequisite note and a guard cell. The capstone is
  optional within this optional notebook: the from-scratch LoRA build is complete
  and verified before any remote job is submitted.
```

---

### Cell 1 - NEW - markdown - Motivation: why understand LoRA internals

Action: NEW. Placed before any math, before the title/objectives cell. Full
content:

```markdown
## Why would a model USER want to understand LoRA internals?

Most of the time you will USE LoRA, not build it. You call `get_peft_model` with a
`LoraConfig`, train, and ship a small adapter file. So why spend time implementing
the adapter by hand?

Because the internals explain the decisions and the failures you will meet in
practice:

- **Why the adapter is tiny.** A LoRA adapter is a few megabytes while the base
  model is hundreds. Once you have built the two low-rank matrices yourself, that
  size difference stops being a marketing number and becomes an obvious
  consequence of `r` being much smaller than the layer dimensions.
- **Why `rank` and `alpha` are the knobs that matter.** When a PEFT run underfits
  or overfits, `rank` and `alpha` are the first things you change. You will tune
  them with real intuition once you have seen exactly where they enter the
  computation.
- **Why the adapted model starts identical to the base model.** LoRA initialises
  one matrix to zero so training begins from the pre-trained behaviour, not from
  noise. You will see why that single choice makes LoRA training stable.
- **Which layers to target.** PEFT lets you pick `target_modules`. Knowing what a
  wrapped layer actually does tells you why people target attention projections
  first and what happens if you pick the wrong set.
- **Merge for zero-overhead inference.** A LoRA adapter can be folded back into the
  base weights so there is no extra latency at serving time. You will implement
  that merge and see it is just one matrix addition.

You do not need this notebook to use PEFT. You need it to use PEFT well, and to
know what to try when a fine-tune misbehaves. That is the whole reason this
optional deep-dive exists.
```

---

### Cell 2 - EDIT - markdown - Title and "What you will build"

Maps to original cell 0.

Old text:

```
# Topic 7a - LoRA from Scratch

Barclays Customer Support Intelligence System | Topic 7a

## What you will build

Full fine-tuning updates every weight in the model: millions of parameters.
LoRA freezes all weights and injects two tiny matrices per layer whose product
approximates the weight change. You implement LoRA from scratch in PyTorch,
apply it to a simple classifier to verify the math, then use HuggingFace PEFT
to LoRA-fine-tune Flan-T5-small on complaint summarization in a GPU training job.

## Why this matters to Barclays

Barclays hosts dozens of NLP models for complaint triage, summarization, and
risk flagging. Full fine-tuning for each task is expensive: storage, GPU time,
and deployment overhead multiply with every new model version.
LoRA adapters are a few MB each. You deploy one frozen base model and swap adapters
per task: the same architecture that powers production PEFT deployments at scale.

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

New text:

```markdown
## LoRA, Built From Scratch

A self-contained build of Low-Rank Adaptation, framed around a Barclays customer
support scenario.

## What you will build

Full fine-tuning updates every weight in the model: millions of parameters. LoRA
freezes all weights and injects two tiny matrices per layer whose product
approximates the weight change. In this notebook you will implement LoRA from
scratch in PyTorch, apply it to a simple classifier to verify the math, then use
the HuggingFace PEFT library to LoRA-fine-tune Flan-T5-small on complaint
summarization in a remote GPU training job.

## Why this matters in practice

An organisation that hosts many NLP models - complaint triage, summarization, risk
flagging - pays for full fine-tuning many times over: storage, GPU time, and
deployment overhead multiply with every new model version. LoRA adapters are a few
megabytes each. You deploy one frozen base model and swap adapters per task. That
is the architecture behind production PEFT deployments at scale, and this notebook
shows you exactly how it works underneath.

## Learning objectives

1. Derive the LoRA update rule: W' = W + B @ A, explain why B=zeros at init
2. Implement LoraLayer in PyTorch, freeze the original layer, verify gradient flow
3. Swap linear layers in an FFN for LoraLayer, count trainable parameters before and after
4. Explain rank r and alpha and how to choose them (r=4,8,16 heuristics)
5. Use HuggingFace PEFT (LoraConfig, get_peft_model) to apply LoRA to Flan-T5
6. Launch and monitor a LoRA fine-tuning job on SageMaker GPU (ml.g4dn.xlarge)
```

---

### Cell 3 - KEEP - code - TensorFlow backend disable

Maps to original cell 1. KEEP unchanged.

```python
# Disable TensorFlow backend in transformers (SageMaker image compatibility).
# Must run before any transformers import.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

---

### Cell 4 - EDIT - code - pip install

Maps to original cell 2. The leading comment references "Section 1-3" and
"Section 4 capstone" which is fine (in-notebook section names, not topic chaining).
Only a light comment touch-up is needed; the `!pip install` block and the
`print(...)` line are KEPT verbatim.

Old comment lines:

```
# Environment setup for SageMaker Studio.
# Section 1-3 (scratch LoRA) runs in this CPU kernel.
# Section 4 (Flan-T5 capstone) launches a remote GPU job via HuggingFace estimator.
```

New comment lines:

```
# Environment setup. This is a self-contained, optional deep-dive.
# Sections 1 to 3 (the from-scratch LoRA build) run in this notebook kernel; CPU
# is fine. Section 4 (the Flan-T5 capstone) launches a remote GPU job via the
# HuggingFace estimator and needs AWS credentials - there is a guard cell for it.
```

The rest of the cell stays identical:

```python
!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "transformers>=4.53,<4.54" \
    "accelerate>=1.0.0" \
    "tokenizers>=0.21,<0.22" \
    "numpy<2" \
    "matplotlib>=3.7.0"

print("RESTART KERNEL before continuing -- environment packages were installed/upgraded.")
```

---

### Cell 5 - KEEP - code - SageMaker session

Maps to original cell 3. KEEP unchanged.

```python
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

Builder note: this cell calls `get_execution_role()` / `sess.default_bucket()`,
which succeed on a SageMaker notebook and may raise elsewhere. The mandatory fix
for a credential-less environment is the guard cell (reworked cell 45). The builder
MAY optionally wrap the four `sess`/`role`/`bucket`/`region` lines in a try/except
that prints `"SageMaker session not available - the from-scratch LoRA build still
runs; the capstone in Section 4 needs AWS credentials."` and sets the four names to
`None`. This is optional hardening; cells 44 and 45 are the required fix.

---

### Cell 6 - REPLACE - markdown - "How this notebook is organised" (was "YOU ARE HERE" table)

Maps to original cell 4. Action: DELETE the old content, REPLACE in place with a
short static note. The course-progression table with T4/T5/T6a/T6b/T7a/T7b and the
"YOU ARE HERE" marker is removed entirely.

Old text (DELETE):

```
## Day 2 System Overview

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | T4 Transformers | Build the architecture from scratch |
| 2 | T5 HuggingFace | Load pre-trained models from the Hub |
| 3 | T6a Full Fine-Tuning | Adapt a model to Barclays complaints |
| 4 | T6b Transfer Learning | Freeze the encoder, train only the head |
| 5 | T7a LoRA from Scratch | Implement parameter-efficient adaptation (YOU ARE HERE) |
| 6 | T7b PEFT + LoRA | Apply PEFT library to a full classifier |

By end of Day 2 you will have a fine-tuned, PEFT-adapted DistilBERT complaint classifier
running as a SageMaker endpoint.
```

New text:

```markdown
## How this notebook is organised

This is a standalone deep-dive, so it does not sit at a fixed point in any course
sequence. It runs front to back on its own:

- Section 1 - the parameter-budget problem, the LoRA idea, and a from-scratch
  `LoraLayer`, plus Lab 1
- Section 2 - applying your `LoraLayer` to a feed-forward network, plus Lab 2
- Section 3 - choosing rank and alpha
- Section 4 - the capstone: LoRA-fine-tuning Flan-T5 on a remote GPU with the
  HuggingFace PEFT library

Run the cells in order. Each section restates what it needs from the section
before it.
```

---

### Cell 7 - KEEP - code - imports, seeds, device

Maps to original cell 5. KEEP unchanged.

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

# CPU for all in-notebook demos; GPU job runs in scripts_optional_lora_ffn/train.py.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
```

---

### Cell 8 - KEEP - markdown - CUDA health check intro

Maps to original cell 6. KEEP unchanged. No course references.

---

### Cell 9 - KEEP - code - CUDA health check + safety-net

Maps to original cell 7. KEEP unchanged. Operational cell, no course references.

---

### Cell 10 - KEEP - markdown - Beat 1 Section 1: parameter budget problem

Maps to original cell 8. KEEP unchanged. No course references.

---

### Cell 11 - KEEP - code - Beat 1: full weight delta storage

Maps to original cell 9. KEEP unchanged. No course references.

---

### Cell 12 - KEEP - code - Beat 1b: LoRA param count vs rank

Maps to original cell 10. KEEP unchanged. No course references.

---

### Cell 13 - KEEP - markdown - Beat 2: the LoRA idea + diagram

Maps to original cell 11. KEEP unchanged. Contains the
`<!-- DIAGRAM: LoRA low-rank decomposition -->` placeholder and the
`[View diagram](../../plans/topic_optional_lora_ffn/diagrams/lora-decomposition.mmd)`
link. The diagram path already uses the `topic_optional_lora_ffn` slug, so no path
edit is needed. No course references.

---

### Cell 14 - KEEP - code - Beat 3: LoraLayer implementation

Maps to original cell 12. KEEP unchanged. Defines `LoraLayer`. No course
references.

---

### Cell 15 - KEEP - markdown - Beat 4: Lab 1 instructions (Tier 1 guided)

Maps to original cell 13. KEEP unchanged. STAR-framed Lab 1 instructions. No course
references.

---

### Cell 16 - KEEP - code - Lab 1 starter (EXERCISES) / FILL (SOLUTIONS)

Maps to original cell 14.

Exercises: KEEP the starter exactly as-is, with the `None  # YOUR CODE` stubs.

Solutions: replace the stub lines with the working implementation. The exact
Solution body is the one already present in the source Solutions notebook:

```python
class LoraLayerStudent(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.in_features  = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank         = rank
        self.scale        = lora_alpha / rank

        # Step 1: Store original_layer and freeze its parameters.
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Step 2: Create self.lora_A as a Linear layer from in_features to rank with no bias.
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)

        # Step 3: Create self.lora_B as a Linear layer from rank to out_features with no bias.
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # Step 4: Initialise lora_A.weight with Normal(0, 0.02) and lora_B.weight with zeros.
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 5: Compute the original (frozen) output.
        original_output = self.original_layer(x)

        # Step 6: Compute the LoRA output scaled by self.scale.
        lora_output = self.lora_B(self.lora_A(x)) * self.scale

        # Step 7: Return the sum of the two outputs.
        return original_output + lora_output
```

---

### Cell 17 - KEEP (Exercises) / DELETE (Solutions) - code - Lab 1 safety-net

Maps to original cell 15.

Exercises: KEEP unchanged. Probes `LoraLayerStudent` and falls back to
`LoraLayerStudent = LoraLayer` so downstream cells run if Lab 1 is incomplete.

Solutions: DELETE this cell. The Lab 1 cell is the working implementation, so the
safety-net is redundant. After deletion the Solutions numbering shifts down by one.

---

### Cell 18 - KEEP - code - Lab 1 verification

Maps to original cell 16. KEEP unchanged in both notebooks.

---

### Cell 19 - KEEP - markdown - Lab 1 Stretch + Homework Extension

Maps to original cell 17. KEEP unchanged. No course references.

---

### Cell 20 - KEEP - code - Homework Extension 1 starter (EXERCISES) / FILL (SOLUTIONS)

Maps to original cell 18.

Exercises: KEEP the starter exactly as-is, with the `# YOUR CODE` placeholder.

Solutions: replace with the worked Homework Extension. The exact Solution body is
the one already present in the source Solutions notebook:

```python
# Homework Extension: try alternative initialisations and compare initial outputs.

alt_lora = LoraLayer(nn.Linear(16, 32), rank=4, lora_alpha=8)

# Alternative init: A zeros, B normal -- output is still zero at step 0 (symmetric to standard init).
nn.init.zeros_(alt_lora.lora_A.weight)
nn.init.normal_(alt_lora.lora_B.weight, std=0.02)

x_hw = torch.randn(8, 16)
lora_contribution = alt_lora.lora_B(alt_lora.lora_A(x_hw)) * alt_lora.scale
print("LoRA contribution norm (A=0, B~Normal):", lora_contribution.norm().item())
print("Initial total output norm:", alt_lora(x_hw).norm().item())

# Symmetric init: both Normal -- output is non-zero at step 0, base model is perturbed immediately.
alt2 = LoraLayer(nn.Linear(16, 32), rank=4, lora_alpha=8)
nn.init.normal_(alt2.lora_A.weight, std=0.02)
nn.init.normal_(alt2.lora_B.weight, std=0.02)
lora_contribution2 = alt2.lora_B(alt2.lora_A(x_hw)) * alt2.scale
print("LoRA contribution norm (both Normal):", lora_contribution2.norm().item())
print("Standard init (A normal, B zero) is preferred because it preserves the pre-trained")
print("model exactly at step 0; training then learns the delta from zero.")
```

Note: Homework Extension starters are async take-home work. Unlike the sibling
attention docs (which leave homework stubs unsolved in BOTH notebooks), the source
Solutions notebook for THIS notebook already ships a worked body for this cell.
Preserve that: Exercise keeps the stub, Solution carries the worked body above.
This keeps pair parity with the source intent.

---

### Cell 21 - KEEP - markdown - Peer Discussion: choosing LoRA

Maps to original cell 19. KEEP unchanged. No course references.

---

### Cell 22 - KEEP - markdown - Beat 3 Section 2: applying LoRA to an FFN

Maps to original cell 20. KEEP unchanged. The phrase "what we will do with Flan-T5
in Section 4" is an in-notebook section reference, not a topic reference. No course
references.

---

### Cell 23 - KEEP - code - Beat 1: naive layer-by-layer inspection

Maps to original cell 21. KEEP unchanged. The print line "Next: a single helper
replace_fc_with_lora" is in-notebook narrative, not a topic reference. No course
references.

---

### Cell 24 - KEEP - markdown - Beat 2: parameter comparison diagram

Maps to original cell 22. KEEP unchanged. Contains the
`<!-- DIAGRAM: LoRA parameter count comparison -->` placeholder and the
`[View diagram](../../plans/topic_optional_lora_ffn/diagrams/lora-parameter-comparison.mmd)`
link. Diagram path already uses the correct slug. No course references.

---

### Cell 25 - KEEP - code - pre-train FFN on FashionMNIST

Maps to original cell 23. KEEP unchanged. Defines `FFNModel`, `pretrained_model`,
`train_pre`. Downloads FashionMNIST (flagged in the banner). No course references.

---

### Cell 26 - KEEP - code - Beat 3: replace fc layers with LoRA

Maps to original cell 24. KEEP unchanged. Defines `replace_fc_with_lora`,
`lora_model`, `trainable_lora`, `lora_optimizer`. No course references.

---

### Cell 27 - KEEP - code - Section 2 lora_model safety-net (operational recovery)

Maps to original cell 25. KEEP unchanged in BOTH notebooks. This is an operational
recovery cell - it rebuilds `lora_model` / `trainable_lora` / `lora_optimizer` if
the previous heavy cell failed or was skipped - NOT a lab safety-net. It stays in
the Solutions notebook, the same way `optional_transformers` keeps its training-job
safety-net.

---

### Cell 28 - KEEP - markdown - Peer Discussion: rank trade-offs

Maps to original cell 26. KEEP unchanged. No course references.

---

### Cell 29 - KEEP - code - LoRA fine-tune on MNIST

Maps to original cell 27. KEEP unchanged. Downloads MNIST (flagged in the banner).
No course references.

---

### Cell 30 - KEEP - markdown - Lab 2 instructions (Tier 1 guided)

Maps to original cell 28. KEEP unchanged. STAR-framed Lab 2 instructions. No course
references.

---

### Cell 31 - KEEP - code - Lab 2 starter (EXERCISES) / FILL (SOLUTIONS)

Maps to original cell 29.

Exercises: KEEP the starter exactly as-is, with the six `None  # YOUR CODE` stubs.

Solutions: replace the stubs with the working implementation. The exact Solution
body is the one already present in the source Solutions notebook:

```python
def replace_fc_with_lora_student(model: FFNModel, rank: int = 8, lora_alpha: int = 16) -> FFNModel:
    """
    Wrap all six fc layers with LoraLayerStudent.
    """
    # Step 1-6: replace each fc layer.
    model.fc1 = LoraLayerStudent(model.fc1, rank=rank, lora_alpha=lora_alpha)
    model.fc2 = LoraLayerStudent(model.fc2, rank=rank, lora_alpha=lora_alpha)
    model.fc3 = LoraLayerStudent(model.fc3, rank=rank, lora_alpha=lora_alpha)
    model.fc4 = LoraLayerStudent(model.fc4, rank=rank, lora_alpha=lora_alpha)
    model.fc5 = LoraLayerStudent(model.fc5, rank=rank, lora_alpha=lora_alpha)
    model.fc6 = LoraLayerStudent(model.fc6, rank=rank, lora_alpha=lora_alpha)

    # Step 7: return the modified model.
    return model
```

---

### Cell 32 - KEEP (Exercises) / DELETE (Solutions) - code - Lab 2 safety-net

Maps to original cell 30.

Exercises: KEEP unchanged. Probes `replace_fc_with_lora_student` and installs a
working fallback definition if Lab 2 is incomplete.

Solutions: DELETE this cell. The Lab 2 cell is the working implementation.

---

### Cell 33 - KEEP - code - Lab 2 verification

Maps to original cell 31. KEEP unchanged in both notebooks.

---

### Cell 34 - KEEP - markdown - Lab 2 Stretch + Homework Extension

Maps to original cell 32. KEEP unchanged. No course references.

---

### Cell 35 - KEEP - code - Homework Extension 2 starter (EXERCISES) / FILL (SOLUTIONS)

Maps to original cell 33.

Exercises: KEEP the starter exactly as-is, with the `# YOUR CODE` placeholder.

Solutions: replace with the worked Homework Extension. The exact Solution body is
the one already present in the source Solutions notebook:

```python
# Homework Extension: sweep over ranks and record param counts.
import copy as _copy_hw

ranks_to_try = [1, 4, 8, 16]
results = []
for r in ranks_to_try:
    m = replace_fc_with_lora_student(_copy_hw.deepcopy(FFNModel()), rank=r, lora_alpha=2 * r)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    results.append((r, n_train))
    print(f"rank={r:>3}: trainable params = {n_train:,}")

print()
print("Plot accuracy vs rank after running 3 epochs of MNIST fine-tuning per rank.")
print("Expected pattern: accuracy plateaus around r=4 to r=8 for this small task.")
```

---

### Cell 36 - KEEP - markdown - Section 3: rank and alpha

Maps to original cell 34. KEEP unchanged. No course references.

---

### Cell 37 - KEEP - code - rank vs trainable-param visualisation

Maps to original cell 35. KEEP unchanged. No course references. (It calls
`plt.savefig("lora_rank_comparison.png", ...)`, writing one PNG to the working
directory. This is a local plot export, not a course artifact, and no other
notebook reads it. It is left as-is - changing it is out of scope.)

---

### Cell 38 - KEEP - markdown - Peer Discussion: LoRA in production

Maps to original cell 36. KEEP unchanged. No course references.

---

### Cell 39 - KEEP - markdown - Section 4: capstone intro

Maps to original cell 37. KEEP unchanged. Verified: no course-chaining language
(it references only Flan-T5, PEFT, and the GPU instance). If the builder finds any
stray "Topic" / "Day" reference at build time, strip it; the verified source has
none.

---

### Cell 40 - KEEP - code - inspect Flan-T5 linear layers

Maps to original cell 38. KEEP unchanged. Downloads `google/flan-t5-small`
(flagged in the banner; public, ungated, no token). No course references.

---

### Cell 41 - KEEP - code - verify source_dir contents

Maps to original cell 39. KEEP unchanged. Checks `scripts_optional_lora_ffn/`
contains `train.py` and `requirements.txt`. No course references.

---

### Cell 42 - KEEP - code - define the HuggingFace estimator

Maps to original cell 40. KEEP unchanged. Defines `estimator` for the
`ml.g4dn.xlarge` GPU job with `output_path=f"s3://{bucket}/lora-flan-t5/output"`.
The S3 prefix `lora-flan-t5` is already topic-neutral - no rename needed. No course
references.

---

### Cell 43 - NEW - markdown - AWS / SageMaker prerequisite note

Action: NEW. Placed immediately before the credentials guard cell and the
job-submission cell. Mirrors the `optional_transformers.md` prerequisite note.
Full content:

```markdown
## Prerequisite for the capstone: AWS / SageMaker credentials

Everything up to this point ran in this notebook kernel. The two downloads it
needed - the image datasets and the Flan-T5-small model - are public and need no
credentials. The capstone is different. It submits a real training job to Amazon
SageMaker, which requires:

- Valid AWS credentials reachable by `boto3` (an attached IAM role on a SageMaker
  notebook instance, or environment credentials elsewhere).
- A SageMaker execution role with permission to create training jobs.
- A writable default S3 bucket for the job output.

If you launched this notebook on a properly provisioned SageMaker notebook
instance, all three are already set up and you can run straight through. If you
are running it somewhere else (a laptop, a plain Jupyter server, an environment
where AWS was never configured), the next cells will fail with a credentials
error.

The cell directly below is a GUARD. It checks for credentials and bucket access
and prints exactly what to do if something is missing, instead of letting you hit
a raw `NoCredentialsError`. Run it first.

If you do not have AWS access, that is fine: the from-scratch LoRA build in
Sections 1 to 3 is the core of this deep-dive and is fully complete. You can read
the capstone cells without running them, then come back to the wrap-up.
```

---

### Cell 44 - NEW - code - AWS credentials / bucket guard cell

Action: NEW. A guard cell that runs BEFORE the job-submission cell, checks
credentials and bucket access, never raises, and sets `capstone_ready`. Mirrors the
`optional_transformers.md` guard cell. Full content:

```python
# Guard: verify AWS / SageMaker access before submitting the remote training job.
# This cell never raises - it reports status and sets `capstone_ready`.
# A standalone learner with no AWS credentials gets a clear message here instead
# of a raw NoCredentialsError deeper in the capstone.

capstone_ready = True
problems = []

# 0. Is the AWS SDK even installed?
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False
    capstone_ready = False
    problems.append("boto3 / botocore not installed (pip install boto3 sagemaker)")

# 1. Are AWS credentials reachable at all?
if _HAS_BOTO3:
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print(f"AWS credentials OK. Account: {identity['Account']}")
    except (NoCredentialsError, BotoCoreError, ClientError) as e:
        capstone_ready = False
        problems.append(f"No usable AWS credentials: {type(e).__name__}")

# 2. Is there a SageMaker execution role?
try:
    if role is None:
        raise ValueError("role is None")
    print(f"SageMaker execution role OK: {role}")
except Exception as e:
    capstone_ready = False
    problems.append(f"No SageMaker execution role: {e}")

# 3. Is the default bucket reachable?
if _HAS_BOTO3:
    try:
        if bucket is None:
            raise ValueError("bucket is None")
        boto3.client("s3").head_bucket(Bucket=bucket)
        print(f"Default S3 bucket OK: {bucket}")
    except Exception as e:
        capstone_ready = False
        problems.append(f"Default S3 bucket not reachable: {type(e).__name__}")

print()
if capstone_ready:
    print("Capstone prerequisites satisfied. You can run the cells below.")
else:
    print("CAPSTONE CANNOT RUN - missing prerequisites:")
    for p in problems:
        print(f"  - {p}")
    print()
    print("What to do:")
    print("  - On a SageMaker notebook instance: confirm the instance has an")
    print("    attached IAM role with SageMaker and S3 permissions.")
    print("  - Elsewhere: configure AWS credentials (aws configure, or set")
    print("    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION).")
    print("  - If you cannot get AWS access, skip the remaining capstone cells.")
    print("    The from-scratch LoRA build above is the core of this deep-dive")
    print("    and is already complete.")
```

Builder note: this guard reads `role` and `bucket` from cell 5. If the builder
applied the optional cell-5 hardening that sets them to `None` on failure, the
`if ... is None` checks above handle that path cleanly.

---

### Cell 45 - EDIT - code - launch the training job

Maps to original cell 41. Add a guard at the top of the cell body so the job is not
submitted when `capstone_ready` is False; everything else is KEPT verbatim.

Old text:

```python
# Launch the training job asynchronously (wait=False).
# The job runs on ml.g4dn.xlarge; expected time ~10 minutes.

estimator.fit(wait=False)

training_job_name = estimator.latest_training_job.name
print(f"Training job launched: {training_job_name}")
print()
print("Monitor in AWS Console:")
print(f"  SageMaker > Training > Training jobs > {training_job_name}")
print()
print("Or run the polling cell below to check status every 60 seconds.")
```

New text:

```python
# Launch the training job asynchronously (wait=False).
# The job runs on ml.g4dn.xlarge; expected time ~10 minutes.

if not capstone_ready:
    raise RuntimeError(
        "Capstone prerequisites not met - see the guard cell above. "
        "Fix AWS access or skip the remaining capstone cells."
    )

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

### Cell 46 - KEEP - code - training-job safety-net (kernel restart recovery)

Maps to original cell 42. KEEP unchanged in BOTH notebooks. This is an operational
recovery cell (re-defines `training_job_name` if the kernel restarted), not a lab
safety-net, so it is NOT removed from the Solutions notebook.

---

### Cell 47 - KEEP - code - poll training job status

Maps to original cell 43. KEEP unchanged. No course references.

---

### Cell 48 - KEEP - code - stream CloudWatch logs

Maps to original cell 44. KEEP unchanged. No course references.

---

### Cell 49 - KEEP - markdown - what the training logs tell you

Maps to original cell 45. KEEP unchanged. No course references.

---

### Cell 50 - KEEP - code - optional local adapter test

Maps to original cell 46. KEEP unchanged. No course references.

---

### Cell 51 - EDIT - markdown - Wrap-Up

Maps to original cell 47. Remove the "How this connects to Topic 7b" section, the
"The Barclays system so far" topic list, and the "Next session: Topic 7b" footer.
Reframe as standalone optional-track closure.

Old text:

```
## Wrap-Up: What You Built in Topic 7a

### Key takeaways

1. Full fine-tuning stores delta_W as a dense matrix: the same size as W itself.
   For a 250M parameter model that is ~950 MB of additional weights per task.

2. LoRA factorises delta_W = B @ A where r is much smaller than min(d, k).
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
- Next: Topic 7b: PEFT and LoRA with DistilBERT


Next session: Topic 7b -- PEFT Library.
You implemented LoRA by hand. Now you replace two custom classes and a manual
layer-replacement loop with three library calls. Same math, production-grade tooling.
```

New text:

```markdown
## Wrap-Up: What You Built

### Key takeaways

1. Full fine-tuning stores delta_W as a dense matrix: the same size as W itself.
   For a 250M parameter model that is ~950 MB of additional weights per task.

2. LoRA factorises delta_W = B @ A where r is much smaller than min(d, k).
   At rank=8 on Flan-T5-small: 0.39% of parameters are trainable.
   Adapters are 2-5 MB, not 250 MB.

3. Initialisation matters: B = zeros ensures the adapted model starts identical
   to the pre-trained model. Only during training do A and B diverge.

4. Rank selection heuristics: start at r=8. Increase to r=16 if underfitting.
   Decrease to r=4 if overfitting. r=1-2 rarely works for language tasks.

5. The PEFT library (get_peft_model, LoraConfig) automates exactly what you built
   in LoraLayer: freeze base, inject A and B, save only adapters.

### Why this matters when you USE PEFT

You did not just train one adapter. You built the mental model for every PEFT
fine-tune you will run:

- When PEFT prints "trainable params: 0.4%", you now know that number is
  `(d + k) * r` divided by `d * k`, and you know which knob (`r`) moves it.
- When a fine-tune underfits, you know to raise `rank`; when it overfits, you know
  to lower it - because you have seen exactly where `rank` enters the math.
- When you pick `target_modules`, you know you are choosing which `nn.Linear`
  layers get a `LoraLayer` wrapper, because you wrote that wrapper.
- When you want zero-overhead inference, you know the adapter merges into the base
  weights with a single matrix addition, because you implemented `merge_weights`.

That is the payoff of this optional deep-dive: not the FashionMNIST classifier or
the one Flan-T5 adapter, but the intuition to use, tune, and debug PEFT with
confidence.

### Where to go next (optional)

This was an optional notebook. The main course path covers the LoRA concept you
need as a user and continues independently of this deep-dive. If you want to keep
going inside the box, a natural next step is to read how the PEFT library applies
LoRA to an encoder-only classifier (where `target_modules` differs from the
seq2seq case here) and how LoRA combines with quantisation to fit larger models on
the same GPU.
```

---

### Cell 52 - EDIT - code - final summary

Maps to original cell 48. Drop "Topic 7a" from the header and drop the trailing
"Next: Topic 7b ..." print line.

Old text:

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
print("Next: Topic 7b: PEFT LoRA on DistilBERT (encoder-only, classification)")
```

New text:

```python
# Summary of what was built and trained in this deep-dive.

print("LoRA From Scratch - Summary")
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
```

Builder note: the `training_job_name` and `trainable_lora` / `train_pre` names this
cell prints are all defined earlier on a straight in-order run (cell 45 / cell 46
for `training_job_name`, cell 26 / cell 27 for the others). If the capstone was
skipped, `training_job_name` is still defined by the safety-net cell 46.

---

## Builder checklist

- [ ] Exercises notebook ends with 53 cells (0..52).
- [ ] All cell ids normalized (`nbformat.normalize`) - the source has several code
      cells with `id: null`.
- [ ] Cell 0 is the optional/supplementary banner; it states the LoRA concept is
      taught as a required mini-lesson elsewhere and this notebook is the optional
      build, and it names the two required downloads (image datasets, Flan-T5).
- [ ] Cell 1 is the motivation ("Why understand LoRA internals?"), before any math.
- [ ] Cell 6 is the "How this notebook is organised" note; the "YOU ARE HERE"
      / Day-2-System-Overview table is gone.
- [ ] Cell 43 is the AWS prerequisite note; cell 44 is the credentials guard cell;
      both sit BEFORE the job-submission cell.
- [ ] Cell 45 (launch) raises a clear RuntimeError if `capstone_ready` is False.
- [ ] No occurrence of "Topic 4", "Topic 5", "Topic 6", "Topic 6a", "Topic 6b",
      "Topic 7", "Topic 7a", "Topic 7b", "T4 Transformers", "T5 HuggingFace",
      "T6a", "T6b", "T7a", "T7b", "Day 2", "Day 3", "YOU ARE HERE",
      "Next session", "Estimated time", "in class" anywhere in any cell.
      (The model name "Flan-T5" and the install pin "transformers>=4.53" are NOT
      topic references and stay.)
- [ ] Plain ASCII only: no em-dash, en-dash, Unicode multiplication sign, emoji.
- [ ] The two `<!-- DIAGRAM: -->` placeholders (cells 13 and 24) and their
      `.mmd` links are intact; both `.mmd` files already exist under
      `plans/topic_optional_lora_ffn/diagrams/`.
- [ ] Gates: `nbformat.validate` passes; per-cell `ast.parse` passes;
      concatenated-pyflakes reports 0 new undefined names.
- [ ] Solutions notebook: lab cells 16, 20, 31, 35 carry the filled bodies given
      above; the two LAB safety-net cells (Exercises cells 17 and 32) are DELETED;
      the Section-2 lora_model safety-net (cell 27), the training-job safety-net
      (cell 46), and the AWS guard (cell 44) are KEPT. Solutions ends with 51
      cells.
- [ ] Pair parity: same cell-type sequence in both notebooks, Solutions = Exercises
      minus the 2 deleted lab safety-net code cells.

---

## Codex-style findings resolved

There is no Codex R1 review for this notebook (it never had a design doc). The
table below records the same classes of defect the sibling optionals' Codex
reviews flagged, and how this rework resolves each for this notebook.

| Finding class | Summary | Resolved by |
|---------------|---------|-------------|
| Standalone framing (sibling R12) | Notebook must not be described as required, and not as the only place LoRA is explained. | Cell 0 banner states the LoRA concept is a required mini-lesson elsewhere and THIS notebook is the optional from-scratch build. Cells 6, 51, 52 strip all required-path / linear-sequence framing. |
| Sequential chaining | "Topic 7a", "Day 2 System Overview", "YOU ARE HERE" table, "How this connects to Topic 7b", "Next session: Topic 7b". | Cell 2 drops "Topic 7a" and "Estimated time ... in class". Cell 6 replaces the progression table with a static organisation note. Cell 51 removes the Topic-7b section and the topic list. Cell 52 drops the "Next: Topic 7b" print line. |
| Cold-run safety (sibling R5) | The notebook must run from a fresh kernel with every symbol defined in-notebook. | Cold-run audit in the front matter: `LoraLayer`, `LoraLayerStudent`, `FFNModel`, `replace_fc_with_lora`, `replace_fc_with_lora_student`, `set_seeds`, `device`, `train_pre`, `trainable_lora`, `training_job_name` are all defined earlier in-notebook on a straight in-order run. No prior-notebook symbol is used. |
| No S3 chain (sibling R6) | Optionals are not in the S3 handoff chain and must not load/produce a chained artifact. | The notebook has NO S3 LOAD cell. It WRITES only `s3://<bucket>/lora-flan-t5/output/model.tar.gz` (the adapter, read by nobody) and the local `scripts_optional_lora_ffn/` files (consumed only by its own job). Stated explicitly so the builder adds no dependency. |
| AWS-credentials guard (sibling R4) | A remote-GPU capstone must not crash a credential-less standalone learner with a raw NoCredentialsError. | NEW cell 43 (prerequisite note) + NEW cell 44 (guard cell that checks STS credentials, the SageMaker role, the default bucket, never raises, sets `capstone_ready`). Cell 45 raises a clear RuntimeError if `capstone_ready` is False. |
| Offline honesty | The notebook downloads assets; the contract must be explicit. | The banner names the exact required downloads (FashionMNIST/MNIST datasets, public `google/flan-t5-small`). No hidden `nltk.download`-style call is added. The from-scratch LoRA math (the core of the deep-dive) is what runs without the capstone's AWS step. |

Note: the Solutions twin receives the identical structural changes - the four
lab/homework cells carry the worked bodies given inline above, the two lab
safety-net cells (Exercises cells 17 and 32) are deleted, and the operational
safety-nets (cells 27 and 46) plus the AWS guard (cell 44) are kept. Solutions
notebook ends with 51 cells.
