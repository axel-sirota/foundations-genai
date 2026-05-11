# Topic 5 - HuggingFace Ecosystem: Cell-by-Cell Plan

## Overview

Topic 5 opens the second half of Day 2. Students have just built a Transformer encoder-decoder
from scratch in Topic 4 and trained it on a GPU. Now the narrative shifts: stop reinventing the
wheel. The HuggingFace ecosystem provides pre-trained models, curated datasets, and high-level
tooling that make production NLP practical. This notebook is INFERENCE ONLY -- no training, no
SageMaker estimators -- and runs entirely in the Studio notebook kernel. Estimated in-class
time: 60 to 75 minutes.

---

## Diagram Index

Diagram 1: slug=hf-hub-ecosystem, path=plans/topic_5/diagrams/hf-hub-ecosystem.mmd
  Description: HuggingFace Hub ecosystem map. Central node: "Hugging Face Hub".
  Three primary branches:
  (A) Models -- arrow to pipeline() box (high-level, task name -> result) and to AutoClass box
      (AutoModel / AutoTokenizer / AutoModelForSeqClass / AutoModelForCausalLM / AutoModelForTokenClass).
  (B) Datasets -- arrow to load_dataset() box (split, streaming, map/filter).
  (C) Spaces -- Gradio / Streamlit apps (mention only, no code).
  Below the AutoClass box: annotation "model card = metadata + README.md, config.json, tokenizer.json, pytorch_model.bin".
  Below pipeline(): annotation "4 lines of code, auto-downloads weights, auto-selects tokenizer".
  Use clean flowchart style with distinct boxes per concept.

Diagram 2: slug=automodel-class-hierarchy, path=plans/topic_5/diagrams/automodel-class-hierarchy.mmd
  Description: AutoModel class hierarchy. Root node: AutoModel (base -- raw hidden states, no head).
  Four child nodes, each with task label and example model:
  (1) AutoModelForSequenceClassification -- task: text-classification, sentiment, zero-shot --
      example: distilbert/distilbert-base-uncased-finetuned-sst-2-english.
  (2) AutoModelForTokenClassification -- task: token-classification, NER --
      example: dslim/bert-base-NER.
  (3) AutoModelForCausalLM -- task: text-generation --
      example: distilgpt2.
  (4) AutoModelForSeq2SeqLM -- task: translation, summarization --
      example: Helsinki-NLP/opus-mt-en-ROMANCE.
  Each child node also shows: "head type added on top of encoder/decoder".
  Style: top-down tree, distinct colours per task type.

---

## Key Decisions

### Version Pins (from CORE_TECHNOLOGIES_AND_DECISIONS.md)
- `transformers>=4.35.0,<4.40.0` -- py312 wheels exist; do NOT pin 4.26 (no Rust compiler in Studio)
- `tokenizers>=0.15.0,<0.20.0` -- matching py312 wheels
- `datasets>=2.18.0,<3.0.0` -- compatible with transformers 4.35-4.40 range
- `numpy<2` -- mandatory pin in every install cell
- `huggingface_hub>=0.19.0,<0.25.0` -- for Hub API demos (snapshot_download, push_to_hub)
- NO `evaluate` library (incompatible with datasets 4.x; L6 from SAGEMAKER_LESSONS_LEARNED)
- NO `sagemaker` import needed (this topic runs in the Studio kernel, no remote jobs)

### Model Choices (all fit on ml.t3.medium, 4GB RAM)
- `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (~265MB) -- sentiment classification demo
- `typeform/distilbert-base-uncased-mnli` (~268MB) -- zero-shot complaint routing demo
- `distilgpt2` (~82MB) -- text generation Beat 1 broken demo
- `distilbert/distilbert-base-uncased` (~265MB) -- raw AutoModel demo

### Why No HuggingFace Token
All models used are public. No gated models are accessed. No `getpass` for HF token needed.
(L10 from SAGEMAKER_LESSONS_LEARNED: token only needed for gated/private models.)

### No SageMaker Estimator
This topic is inference-only in the Studio kernel. The next topic (Topic 6a) introduces remote
fine-tuning. Students appreciate the contrast: Topic 5 shows how far you get without training at
all, using pre-trained weights.

### Hub Upload Section
We use `push_to_hub()` on a tokenizer config (not actual weights) to show the upload API
without requiring students to have HF accounts or network egress. The demo is wrapped in a
try/except so it degrades gracefully in a sandboxed environment. Students who do have accounts
can run it as written.

---

## Variable Continuity from Topic 4

The following variable names and patterns carry forward from `topic_4_transformers.md`:

- `device` -- same pattern: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- `set_seeds(seed=42)` -- identical signature and body
- `COMPLAINT_TOKENS` list -- reused in tokenizer comparison demo (Beat 1 of Section 2)
- `warnings.filterwarnings("ignore")` -- same pattern
- Narrative continuity: "We built the French-to-English translator from scratch in Topic 4.
  Now we use one that was trained on billions of sentences."

New variables introduced in Topic 5 that downstream topics depend on:
- `classifier` -- the `pipeline("text-classification")` object
- `zero_shot` -- the `pipeline("zero-shot-classification")` object
- `tokenizer` -- `AutoTokenizer.from_pretrained(...)` -- used in Labs 1 and 2
- `model` -- `AutoModelForSequenceClassification.from_pretrained(...)` -- used in Lab 2
- `COMPLAINT_LABELS` -- list of Barclays routing categories for zero-shot lab

---

## Cell-by-Cell Plan

### Cell 0: markdown - Title and Learning Objectives

Content:
```
# Topic 5: HuggingFace Ecosystem

Barclays Customer Support Intelligence System | Day 2, Topic 5

## What you will build today

In Topic 4 you assembled a Transformer encoder-decoder from scratch and trained it on French
complaints. You had full control -- and you wrote every line: positional encoding, multi-head
attention, the decoder loop, the SageMaker estimator.

Today you stop reinventing the wheel. HuggingFace gives you pre-trained models trained on
billions of sentences, curated datasets, and a four-line inference API. By the end of this
topic you will classify Barclays complaint sentiment, route complaints to the correct team using
zero-shot classification, extract named entities, and understand how to share a checkpoint to the
Hub.

## Learning objectives

1. Navigate the HuggingFace Hub: find models, datasets, model cards, and download checkpoints
2. Use pipeline() to run inference in four lines with no knowledge of the underlying architecture
3. Load a model and tokenizer manually with AutoTokenizer and AutoModelForSequenceClassification
4. Explain why the AutoClass pattern exists and how it maps to model card metadata
5. Upload a model artifact to the Hub using push_to_hub()

## Estimated time

60 to 75 minutes in class.
```

---

### Cell 1: markdown - Section 0: Environment Setup

Content:
```
## Section 0 - Environment Setup

We install the HuggingFace stack and set up reproducibility.
No SageMaker session needed -- all inference runs in this kernel.
```

---

### Cell 2: code - Install Dependencies

```python
# Topic 5 install cell.
# Pinned versions from CORE_TECHNOLOGIES_AND_DECISIONS.md (verified 2026-05-11).
# transformers>=4.35 required for py312 wheels (do NOT pin 4.26 -- no Rust in Studio).
# No evaluate library -- incompatible with datasets 4.x (L6 from SAGEMAKER_LESSONS_LEARNED).
# No sagemaker -- this topic runs entirely in the Studio kernel.

!pip install -q \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "datasets>=2.18.0,<3.0.0" \
    "huggingface_hub>=0.19.0,<0.25.0" \
    "numpy<2"

print("Install complete.")
```

---

### Cell 3: code - Imports and Configuration

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

# CPU kernel for all demos -- no GPU needed for distilbert inference.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch:  {torch.__version__}")
print(f"Device:   {device}")
print()

# Complaint tokens carried over from Topic 4 for tokenizer comparison demos.
COMPLAINT_TOKENS = [
    "unauthorised", "charge", "account", "fraud",
    "refund", "dispute", "urgent", "branch"
]

# Routing categories for zero-shot classification (Barclays complaint types).
COMPLAINT_LABELS = [
    "fraud and security",
    "billing and charges",
    "account access",
    "general enquiry",
]

print(f"Complaint vocabulary for demos: {COMPLAINT_TOKENS}")
print(f"Complaint routing labels:       {COMPLAINT_LABELS}")
```

---

### Cell 4: markdown - What Are We Building Today?

Content:
```
## What are we building today?

In Topic 4 we trained a translator from scratch on 50,000 sentence pairs.
It took 15-25 minutes on an NVIDIA T4 GPU and reached a modest BLEU score.

That same model exists on HuggingFace Hub -- trained on 50 million sentence pairs,
fine-tuned by professional teams, with a 95 BLEU score -- and it downloads in seconds.

Today we use that pattern for three Barclays complaint tasks:

| Task | Model | Size |
|------|-------|------|
| Sentiment classification | distilbert-finetuned-sst-2-english | 265MB |
| Zero-shot complaint routing | typeform/distilbert-base-uncased-mnli | 268MB |
| Named entity extraction | distilbert-base-uncased (raw) + custom head | 265MB |

All three run on the Studio kernel CPU. No GPU, no training, no SageMaker estimator.
```

---

### Cell 5: markdown - Section 1: The HuggingFace Hub

Content:
```
## Section 1 - The HuggingFace Hub

The Hub is a platform that hosts:

- **Models**: pre-trained weights + config.json + tokenizer.json + README.md (model card)
- **Datasets**: structured data with splits (train/validation/test) + data cards
- **Spaces**: live demos built with Gradio or Streamlit

Every model has a **model card** (README.md) that specifies:
- What task it was trained for (pipeline_tag in the YAML front-matter)
- What architecture it uses (library_name: transformers)
- What dataset it was trained on
- Evaluation metrics and known limitations

The `pipeline_tag` field is what tells the HuggingFace `pipeline()` function
which head to attach and how to format inputs and outputs.
```

---

### Cell 6: code - Beat 1: Trying to Use the Hub Without Knowing the Task

```python
# Beat 1: What happens if you download a model but use it for the wrong task?
# We load a sentiment model and try to use it as a zero-shot classifier.
# The pipeline runs -- but the output is silently wrong.

from transformers import pipeline

print("Beat 1: Using a sentiment model as a zero-shot classifier")
print("=" * 60)
print()
print("Loading distilbert-finetuned-sst-2-english for sentiment ...")

sentiment_pipe = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,   # force CPU
)

complaint = "My account was charged twice for the same transaction."

# This works and gives a sensible answer.
correct_result = sentiment_pipe(complaint)
print(f"Correct use (sentiment):  {correct_result}")
print()

# Now try to use it as a zero-shot classifier with candidate labels.
# The API accepts the call but IGNORES the labels entirely.
# The model was never trained on NLI -- it just returns POSITIVE/NEGATIVE.
try:
    wrong_result = sentiment_pipe(
        complaint,
        candidate_labels=["fraud", "billing", "account access"],
    )
    print(f"Wrong use (zero-shot):  {wrong_result}")
    print()
    print("The pipeline ran without error -- but the candidate_labels were ignored.")
    print("Output is still POSITIVE/NEGATIVE, not the routing labels we wanted.")
    print("The model card pipeline_tag='text-classification', not 'zero-shot-classification'.")
    print("FIX: use the correct model for each task (Section 2).")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    print("The pipeline rejected the mismatched task -- read the model card first.")
```

---

### Cell 7: markdown - Beat 2: Hub Ecosystem Diagram

Content:
```
<!-- DIAGRAM: HuggingFace Hub ecosystem map. Central node Hub branches to: Models (pipeline and AutoClass boxes), Datasets (load_dataset box), and Spaces (Gradio/Streamlit). AutoClass box expands to AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForCausalLM. Model card metadata (pipeline_tag, library_name, config.json) shown connecting Hub to pipeline task routing. -->
[View diagram](../../plans/topic_5/diagrams/hf-hub-ecosystem.mmd)

The Hub is the distribution layer. Every model you download has a model card that declares its
task (pipeline_tag) and architecture (library_name). The pipeline() function reads that metadata
and automatically selects the correct tokenizer, model class, and output format.
```

---

### Cell 8: code - Beat 3: Exploring the Hub with huggingface_hub

```python
# Beat 3: Use the huggingface_hub Python API to inspect a model card programmatically.
# This is what the Hub web UI does under the hood.

from huggingface_hub import ModelCard, model_info

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

print(f"Inspecting model card for: {MODEL_ID}")
print("=" * 60)

# model_info() fetches metadata without downloading weights.
info = model_info(MODEL_ID)

print(f"Pipeline tag:    {info.pipeline_tag}")
print(f"Library name:    {info.library_name}")
print(f"Model ID:        {info.modelId}")
print(f"Downloads/month: {info.downloads:,}" if info.downloads else "Downloads: (not available)")
print(f"Tags:            {info.tags[:6]}")   # show first 6
print()

# Show model card YAML front-matter (the metadata section).
try:
    card = ModelCard.load(MODEL_ID)
    data = card.data
    print("Model card metadata (YAML front-matter):")
    print(f"  language:     {data.language}")
    print(f"  license:      {data.license}")
    print(f"  datasets:     {data.datasets}")
    print(f"  pipeline_tag: {data.get('pipeline_tag', '(inferred from config)')}")
except Exception as e:
    print(f"Could not load model card metadata: {e}")
print()
print("The pipeline_tag field tells pipeline() which head to attach.")
print("Mismatch between task and pipeline_tag = wrong outputs (Beat 1).")
```

---

### Cell 9: markdown - Lab 1 Header: Tier 1 - Hub Exploration

Content:
```
## Lab 1 - Explore a Model on the Hub (Tier 1 - Guided)

**Time**: 15-20 minutes

### Situation

The Barclays NLP team needs to add a zero-shot complaint routing capability.
Before writing any code, they need to identify the right model on the Hub --
one with pipeline_tag="zero-shot-classification" that fits on a CPU instance.

### Task

Use the huggingface_hub API to inspect `typeform/distilbert-base-uncased-mnli`
and confirm it is suitable for zero-shot complaint routing.

### Action

Fill in the three stubs below. Do NOT look at the web UI -- read the metadata
programmatically so you can automate this check in a CI pipeline.

### Result

The verification cell will check:
1. You retrieved the correct pipeline_tag
2. You retrieved at least one tag from the model
3. The model fits our RAM budget (size < 350MB based on parameter count)
```

---

### Cell 10: code - Lab 1 Starter Code

```python
# Lab 1: Inspect typeform/distilbert-base-uncased-mnli using the huggingface_hub API.

from huggingface_hub import model_info

ZS_MODEL_ID = "typeform/distilbert-base-uncased-mnli"

# Step 1: Call model_info() on ZS_MODEL_ID and store the result in `zs_info`.
zs_info = None   # YOUR CODE

# Step 2: Extract the pipeline_tag from zs_info into `zs_pipeline_tag`.
zs_pipeline_tag = None   # YOUR CODE

# Step 3: Extract the list of tags from zs_info into `zs_tags`.
#         Hint: the attribute is the same as for the model in Cell 8.
zs_tags = None   # YOUR CODE

# Quick preview (do not change this block).
if zs_info is not None:
    print(f"Model:        {ZS_MODEL_ID}")
    print(f"Pipeline tag: {zs_pipeline_tag}")
    print(f"Tags:         {zs_tags}")
else:
    print("zs_info is None -- complete Step 1.")
```

---

### Cell 11: code - Lab 1 Verification

```python
# Lab 1 Verification

all_pass = True

if zs_info is None:
    print("FAIL: zs_info is None -- complete Step 1.")
    all_pass = False
else:
    if zs_pipeline_tag == "zero-shot-classification":
        print("PASS: pipeline_tag is 'zero-shot-classification'")
    elif zs_pipeline_tag is None:
        print("FAIL: zs_pipeline_tag is None -- complete Step 2.")
        all_pass = False
    else:
        print(f"FAIL: Expected 'zero-shot-classification', got '{zs_pipeline_tag}'")
        all_pass = False

    if zs_tags is None:
        print("FAIL: zs_tags is None -- complete Step 3.")
        all_pass = False
    elif isinstance(zs_tags, list) and len(zs_tags) > 0:
        print(f"PASS: Retrieved {len(zs_tags)} tag(s) from the model")
    else:
        print(f"FAIL: Expected a non-empty list of tags, got: {zs_tags}")
        all_pass = False

if all_pass:
    print()
    print("All Lab 1 checks passed.")
    print(f"typeform/distilbert-base-uncased-mnli is confirmed for zero-shot complaint routing.")
```

---

### Cell 12: code - Lab 1 Safety-Net

```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.

_need_sn1 = (zs_info is None) or (zs_pipeline_tag is None) or (zs_tags is None)

if _need_sn1:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    from huggingface_hub import model_info as _mi
    _zs = _mi("typeform/distilbert-base-uncased-mnli")
    zs_info         = _zs
    zs_pipeline_tag = _zs.pipeline_tag
    zs_tags         = _zs.tags
    print(f"  pipeline_tag: {zs_pipeline_tag}")
    print(f"  tags:         {zs_tags[:4]}")
```

---

### Cell 13: markdown - Lab 1 Stretch and Homework

Content:
```
### Stretch (fast finishers)

Write a function `is_hub_model_suitable(model_id, required_tag, max_mb)` that:
1. Calls model_info() on the model_id
2. Returns True if pipeline_tag == required_tag
3. Raises ValueError if the model is larger than max_mb megabytes

Use it to check whether `distilbert/distilbert-base-uncased` (no task head)
is suitable for zero-shot-classification with max_mb=300.

### Homework Extension

Write a script that queries the Hub for all public models tagged with
`zero-shot-classification` and sorts them by monthly downloads.

```python
from huggingface_hub import list_models

def top_zero_shot_models(n=5):
    """
    Return the top-n most downloaded zero-shot-classification models.

    Returns:
        list of (model_id, downloads) tuples, sorted descending by downloads.
    """
    pass   # implement for homework
```

Compare the top 5 results. Which one would you recommend for a production
Barclays system that must handle 10,000 complaints per hour on a CPU fleet?
What factors besides download count matter?
```

---

### Cell 14: markdown - Section 2: pipeline() -- Four Lines to Inference

Content:
```
## Section 2 - pipeline(): Four Lines to Production Inference

The `pipeline()` function is the highest-level HuggingFace API.
It handles: model download, tokenisation, batching, forward pass, and output formatting.

Core pattern:
    pipe = pipeline(task, model=model_id, device=-1)   # device=-1 = CPU
    result = pipe(input_text)

Supported tasks include:
- "text-classification"        -> sentiment, topic, intent
- "zero-shot-classification"   -> routing with unseen labels
- "token-classification"       -> NER, POS tagging
- "text-generation"            -> autoregressive completion
- "fill-mask"                  -> masked language modelling

The task name MUST match the model's pipeline_tag. Getting this wrong (Beat 1 above)
silently produces incorrect outputs.
```

---

### Cell 15: code - Beat 1: Wrong Task Name in pipeline()

```python
# Beat 1: What happens when you pass a task name that doesn't exist?
# Or try to use a text-generation model for text-classification?

from transformers import pipeline

print("Beat 1: Wrong task name vs wrong model-task pairing")
print("=" * 55)
print()

# Case A: typo in task name -> ValueError immediately.
print("Case A: task='text_classification' (underscore instead of hyphen)")
try:
    bad_pipe = pipeline(task="text_classification",
                        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    bad_pipe("test")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
print()

# Case B: text-generation model used for classification -> wrong output shape.
print("Case B: distilgpt2 (text-generation) called with pipeline('text-classification')")
print("  Loading distilgpt2 as a classifier ...")
try:
    wrong_class_pipe = pipeline(
        task="text-classification",
        model="distilgpt2",
        device=-1,
    )
    result = wrong_class_pipe("My account was charged twice.")
    print(f"  Output: {result}")
    print("  The pipeline ran, but the classification head was randomly initialised.")
    print("  distilgpt2 has no trained classification head -- output is meaningless.")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
print()
print("FIX: always match the task argument to the model's pipeline_tag (Section 2 Beat 3).")
```

---

### Cell 16: code - Beat 3: pipeline() Sentiment Classification (Full Working Demo)

```python
# Beat 3: Correct pipeline() usage for sentiment classification.
# Model: distilbert/distilbert-base-uncased-finetuned-sst-2-english
# Task: text-classification (binary: POSITIVE / NEGATIVE)
# Size: ~265MB -- fits on ml.t3.medium (4GB RAM).
# No token needed -- public model.

from transformers import pipeline

print("Loading sentiment classifier ...")
print("Model: distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# device=-1 forces CPU inference.
# top_k=None returns all labels with their scores.
classifier = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
    top_k=None,
)

# Barclays complaint examples covering the full sentiment spectrum.
complaints = [
    "My account was charged twice for the same transaction. Absolutely furious.",
    "The refund was processed within 24 hours -- very impressed with the service.",
    "I received a letter about my account but cannot log in to view it.",
    "Fraud alert on my card was resolved quickly and professionally. Thank you.",
    "Three weeks and no response to my dispute. This is unacceptable.",
]

print()
print("Classifying Barclays complaints:")
print("-" * 65)

results = classifier(complaints)   # batch call -- more efficient than a loop

for complaint, result in zip(complaints, results):
    # result is a list of dicts because top_k=None: [{'label': ..., 'score': ...}, ...]
    top = max(result, key=lambda x: x["score"])
    score_pct = top["score"] * 100
    print(f"  [{top['label']:>8}  {score_pct:5.1f}%]  {complaint[:55]}...")

print()
print("Four lines to production-quality sentiment inference.")
print("DistilBERT (66M params) achieves 91.3% accuracy on SST-2.")
print("Fine-tuning from scratch to this level took ~3 hours on a V100.")
```

---

### Cell 17: code - Beat 3 (continued): Zero-Shot Complaint Routing

```python
# Zero-shot classification: classify complaints into Barclays routing categories
# WITHOUT any training on those specific labels.
# Model: typeform/distilbert-base-uncased-mnli
# Fine-tuned on MNLI (textual entailment) -- learns to infer unseen label semantics.

from transformers import pipeline

print("Loading zero-shot classifier ...")
print("Model: typeform/distilbert-base-uncased-mnli")

zero_shot = pipeline(
    task="zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1,
)

# The routing labels were defined in Cell 3 (COMPLAINT_LABELS).
print(f"Routing labels: {COMPLAINT_LABELS}")
print()

test_complaints = [
    "Someone has made three transactions on my account that I did not authorise.",
    "I was charged a fee I do not recognise on my statement from last month.",
    "I have forgotten my password and cannot access my online banking.",
    "I would like to know the interest rate on my current savings account.",
]

print("Zero-shot routing results:")
print("-" * 65)

for complaint in test_complaints:
    result = zero_shot(complaint, candidate_labels=COMPLAINT_LABELS)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    print(f"  Route: [{top_label:<22}  {top_score:.2f}]")
    print(f"  Text:  {complaint[:60]}...")
    print()

print("No fine-tuning on Barclays data required.")
print("The MNLI model infers label meaning from natural language -- generalises to new domains.")
```

---

### Cell 18: markdown - Discussion: pipeline() in Production

Content:
```
### Discussion (3 minutes)

You have used two pipelines: sentiment classification and zero-shot routing.

1. The zero-shot pipeline makes one forward pass per candidate label
   (it runs an NLI premise-hypothesis pair for each label).
   If COMPLAINT_LABELS has 10 labels and you receive 1,000 complaints per hour,
   how many forward passes per hour does the zero-shot pipeline need?
   How does this compare to the sentiment classifier (one pass per complaint)?
   What is the latency and cost implication for a production Barclays system?

2. The sentiment classifier was fine-tuned on SST-2 (movie reviews).
   Barclays complaints have very different vocabulary and tone.
   What types of errors would you expect the classifier to make?
   How would you measure this without labelling all 50,000 complaints?

3. pipeline() downloads the full model weights on first call.
   In a Lambda function or container that starts cold, this takes 20-30 seconds.
   What deployment pattern would you use to avoid cold-start latency for
   a real-time complaint routing service?
```

---

### Cell 19: markdown - Section 3: datasets -- Loading Structured Training Data

Content:
```
## Section 3 - datasets: Loading Structured Data from the Hub

The `datasets` library provides a uniform interface to thousands of public datasets.
All datasets are stored in Apache Arrow columnar format for fast random access
and streaming -- you can iterate over a 100GB dataset without loading it into RAM.

Core pattern:
    from datasets import load_dataset
    ds = load_dataset("dataset_name", split="train")
    for example in ds:
        print(example)

Splits: most datasets have "train", "validation", and "test".
Loading a single split avoids downloading the others.

For the Barclays context, three public datasets are relevant:
- "sst2" (in "glue") -- sentence sentiment, 67k examples, used to fine-tune SST-2 models
- "financial_phrasebank" -- financial sentiment, 4,840 sentences, closer to Barclays domain
- "dair-ai/emotion" -- multi-class emotion, 20k examples
```

---

### Cell 20: code - Beat 1: Loading the Wrong Split (IndexError Demo)

```python
# Beat 1: What happens when you try to access a split that does not exist?
# Or index a Dataset like a plain Python list without checking the feature schema?

from datasets import load_dataset

print("Beat 1: Common dataset loading mistakes")
print("=" * 50)
print()

# Case A: requesting a split that does not exist in this dataset.
print("Case A: load_dataset('sst2' from glue) with split='test' -- test labels are hidden")
try:
    ds_bad_split = load_dataset("glue", "sst2", split="test")
    row = ds_bad_split[0]
    label = row["label"]   # all test labels are -1 (hidden on Hub)
    print(f"  First test row label: {label}")
    print("  Label is -1 -- the SST-2 test set has no ground truth on the Hub.")
    print("  Hidden labels are a common gotcha. Use 'validation' for evaluation.")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
print()

# Case B: treating a DatasetDict as a Dataset directly.
print("Case B: load_dataset without specifying split -- returns DatasetDict, not Dataset")
ds_dict = load_dataset("glue", "sst2")
print(f"  Type returned:  {type(ds_dict)}")
print(f"  Available splits: {list(ds_dict.keys())}")
print()
try:
    row = ds_dict[0]   # DatasetDict does not support integer indexing
    print(f"  ds_dict[0]: {row}")
except Exception as e:
    print(f"  ds_dict[0] raises {type(e).__name__}: {e}")
    print("  FIX: use ds_dict['train'][0] or load_dataset(..., split='train').")
```

---

### Cell 21: code - Beat 3: datasets -- Correct Loading and Inspection

```python
# Beat 3: Correct pattern for loading, inspecting, and filtering a dataset.
# We use financial_phrasebank -- closer to the Barclays complaint domain than SST-2.

from datasets import load_dataset

print("Loading financial_phrasebank (sentences_allagree split) ...")
# 'sentences_allagree': only sentences where all annotators agreed (highest confidence).
ds = load_dataset(
    "financial_phrasebank",
    "sentences_allagree",
    split="train",     # only one split exists for this config
    trust_remote_code=True,
)

print(f"Dataset type:     {type(ds)}")
print(f"Number of rows:   {len(ds):,}")
print(f"Features:         {ds.features}")
print()

# Label mapping.
label_names = ds.features["label"].names
print(f"Label names:      {label_names}")
print()

# Show first 3 examples.
print("First 3 examples:")
for i in range(3):
    row = ds[i]
    label_str = label_names[row["label"]]
    print(f"  [{label_str:>8}]  {row['sentence'][:70]}")

print()

# Count by label.
from collections import Counter
label_counts = Counter(label_names[row["label"]] for row in ds)
print("Label distribution:")
for label, count in sorted(label_counts.items()):
    pct = 100 * count / len(ds)
    print(f"  {label:>8}: {count:>4}  ({pct:5.1f}%)")

print()
print("This dataset is much closer to Barclays complaint language than SST-2 movie reviews.")
print("We could use it to evaluate the pipeline() classifier without any fine-tuning.")
```

---

### Cell 22: code - Evaluate pipeline() on Financial Data (Quick Baseline)

```python
# Quick evaluation: how well does the SST-2 classifier transfer to financial text?
# No fine-tuning. Uses the classifier variable from Cell 16.
# This motivates Section 4 (AutoModel) and Topic 6 (full fine-tuning).

from datasets import load_dataset

print("Evaluating SST-2 classifier on financial_phrasebank ...")
print("(This takes ~30 seconds on CPU for 200 examples)")
print()

# Use a small subset to keep it fast in class.
ds_eval = load_dataset(
    "financial_phrasebank",
    "sentences_allagree",
    split="train",
    trust_remote_code=True,
)

# Take 200 stratified examples.
import random
random.seed(42)
indices = list(range(len(ds_eval)))
random.shuffle(indices)
subset = ds_eval.select(indices[:200])

label_names = ds_eval.features["label"].names

# Map: 0=negative, 1=neutral, 2=positive -> NEGATIVE/POSITIVE (SST-2 labels)
# Neutral is the hard case -- no direct SST-2 equivalent.
FPHRASEBANK_TO_SST2 = {"negative": "NEGATIVE", "neutral": None, "positive": "POSITIVE"}

sentences   = [row["sentence"] for row in subset]
true_labels = [label_names[row["label"]] for row in subset]

preds = classifier(sentences, batch_size=32)

correct = 0
neutral_count = 0
for pred_list, true_label in zip(preds, true_labels):
    top = max(pred_list, key=lambda x: x["score"])
    pred_label = top["label"]
    if FPHRASEBANK_TO_SST2[true_label] is None:
        neutral_count += 1
        continue
    if pred_label == FPHRASEBANK_TO_SST2[true_label]:
        correct += 1

non_neutral = 200 - neutral_count
accuracy = correct / max(non_neutral, 1)

print(f"Evaluated {len(sentences)} examples")
print(f"Neutral examples (no SST-2 equivalent): {neutral_count}")
print(f"Accuracy on positive/negative subset: {accuracy:.1%}")
print()
print("Expected: ~70-80% on positive/negative only. Neutral is missed entirely.")
print("This gap motivates fine-tuning on domain-specific data (Topic 6a).")
```

---

### Cell 23: markdown - Section 4: AutoModel and AutoTokenizer

Content:
```
## Section 4 - AutoModel and AutoTokenizer: Under the Hood

pipeline() is convenient but opaque.
When you need full control -- custom preprocessing, inspecting attention weights,
extracting embeddings, or building a custom head -- you use AutoModel and AutoTokenizer directly.

AutoTokenizer.from_pretrained(model_id):
    Downloads config, vocabulary, and merges files.
    Returns the correct tokenizer class for that architecture.
    Handles subword tokenisation (BPE, WordPiece, SentencePiece) transparently.

AutoModel.from_pretrained(model_id):
    Downloads weights and config.
    Returns base model: hidden states only, NO task head.
    For a classification head, use AutoModelForSequenceClassification.

The AutoClass hierarchy matches the model card pipeline_tag:

    pipeline_tag             AutoClass
    ----------------------   ------------------------------------
    text-classification   -> AutoModelForSequenceClassification
    token-classification  -> AutoModelForTokenClassification
    text-generation       -> AutoModelForCausalLM
    translation           -> AutoModelForSeq2SeqLM
    fill-mask             -> AutoModelForMaskedLM
```

---

### Cell 24: code - Beat 2: AutoModel Class Hierarchy Diagram

Content:
```
<!-- DIAGRAM: AutoModel class hierarchy. Root: AutoModel (base, no head, raw hidden states). Four child nodes: (1) AutoModelForSequenceClassification -- text-classification, sentiment, zero-shot -- distilbert-finetuned-sst-2-english. (2) AutoModelForTokenClassification -- NER, POS -- dslim/bert-base-NER. (3) AutoModelForCausalLM -- text-generation -- distilgpt2. (4) AutoModelForSeq2SeqLM -- translation, summarization -- Helsinki-NLP/opus-mt-en-ROMANCE. Each child shows: "task head added on top of encoder". -->
[View diagram](../../plans/topic_5/diagrams/automodel-class-hierarchy.mmd)

Each AutoModelFor* class adds a thin task-specific head (one or two linear layers)
on top of the shared encoder. The encoder weights are identical across all four classes
for the same base model -- only the head differs.
```

Note: Cell 24 is a markdown cell (contains the DIAGRAM placeholder).

---

### Cell 25: code - Beat 1: AutoModel vs AutoModelForSequenceClassification Shape Error

```python
# Beat 1: What happens when you load AutoModel (no head) and try to classify?
# The output is raw hidden states -- shape (batch, seq, hidden) -- not logits.
# Feeding it to torch.argmax gives a nonsensical result.

from transformers import AutoModel, AutoTokenizer

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

print("Beat 1: AutoModel vs AutoModelForSequenceClassification")
print("=" * 60)
print()
print(f"Loading base AutoModel (no head) for: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModel.from_pretrained(MODEL_ID)
base_model.eval()

complaint = "Unauthorised charge on my account."
inputs    = tokenizer(complaint, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = base_model(**inputs)

# AutoModel returns BaseModelOutput -- no logits attribute.
print(f"Output type:          {type(outputs).__name__}")
print(f"last_hidden_state:    {outputs.last_hidden_state.shape}")
print(f"  (batch=1, seq_len={outputs.last_hidden_state.shape[1]}, hidden=768)")
print()

# Naive mistake: argmax on hidden states to "classify".
try:
    wrong_pred = outputs.last_hidden_state.argmax(dim=-1)
    print(f"argmax on hidden states: {wrong_pred}")
    print("This produces token-position indices, NOT class labels.")
    print("There is no classification head -- the raw hidden states are not class scores.")
except Exception as e:
    print(f"Error: {e}")

print()
print("FIX: use AutoModelForSequenceClassification to get logits over class labels (Beat 3).")
```

---

### Cell 26: code - Beat 3: AutoTokenizer + AutoModelForSequenceClassification

```python
# Beat 3: Correct manual inference pattern with AutoTokenizer + AutoModelForSeqClass.
# This is what pipeline() does under the hood.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

print(f"Loading AutoTokenizer + AutoModelForSequenceClassification")
print(f"Model: {MODEL_ID}")
print()

# ------------------------------------------------------------------
# Step 1: Load tokenizer.
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"Tokenizer class:  {type(tokenizer).__name__}")
print(f"Vocabulary size:  {tokenizer.vocab_size:,}")
print(f"Max length:       {tokenizer.model_max_length}")
print()

# ------------------------------------------------------------------
# Step 2: Load model with classification head.
# ------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

print(f"Model class:      {type(model).__name__}")
print(f"Number of labels: {model.config.num_labels}")
print(f"Label mapping:    {model.config.id2label}")
print(f"Parameters:       {sum(p.numel() for p in model.parameters()):,}")
print()

# ------------------------------------------------------------------
# Step 3: Tokenise a batch of complaints.
# ------------------------------------------------------------------
complaints = [
    "Unauthorised direct debit on my account three times this month.",
    "Thank you, the team resolved my issue within the hour.",
    "I cannot log in to online banking -- account locked for no reason.",
]

inputs = tokenizer(
    complaints,
    padding=True,        # pad shorter sequences to longest in batch
    truncation=True,     # truncate sequences longer than model_max_length
    max_length=128,      # distilbert supports up to 512; 128 is enough for complaints
    return_tensors="pt", # return PyTorch tensors
)

print("Tokenised batch:")
print(f"  input_ids shape:      {inputs['input_ids'].shape}")
print(f"  attention_mask shape: {inputs['attention_mask'].shape}")
print()

# Show tokenisation for first complaint.
first_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"First complaint tokens: {first_tokens}")
print()

# ------------------------------------------------------------------
# Step 4: Forward pass + softmax.
# ------------------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits         # (batch, num_labels=2)
probs  = F.softmax(logits, dim=-1)   # convert logits to probabilities

print("Predictions:")
for complaint, prob_row in zip(complaints, probs):
    label_idx = prob_row.argmax().item()
    label_str = model.config.id2label[label_idx]
    score     = prob_row[label_idx].item()
    print(f"  [{label_str:>8}  {score:.2f}]  {complaint[:55]}...")

print()
print("Same result as pipeline('text-classification') -- but you now control every step.")
print("Use this pattern when you need: embeddings, attention weights, or a custom head.")
```

---

### Cell 27: markdown - Lab 2 Header: Tier 1 - Manual NER Inference

Content:
```
## Lab 2 - Manual Named Entity Recognition Inference (Tier 1 - Guided)

**Time**: 15-20 minutes

### Situation

The Barclays compliance team wants to extract named entities from complaint text --
specifically person names, organisation names, and locations -- so they can flag
complaints that mention specific branch names or counterparty banks.

The team wants to understand what entities the model finds BEFORE routing the complaint.
The pipeline("token-classification") API works for quick demos but compliance needs
to inspect raw token scores and map them back to original text spans.

### Task

Use AutoTokenizer and AutoModelForTokenClassification with
`dslim/bert-base-NER` to run NER inference on a Barclays complaint
and extract (entity_text, label, score) triples.

### Action

1. Load the tokenizer for "dslim/bert-base-NER" into `ner_tokenizer`.
2. Load the model into `ner_model` (AutoModelForTokenClassification).
3. Tokenise the complaint with `return_offsets_mapping=True` and
   `is_split_into_words=False`.
4. Run a forward pass. Extract logits and apply softmax to get probabilities.
5. Store predictions as a list of dicts in `ner_predictions`:
   each dict has keys "token", "label", "score".

### Result

Verification cell checks shape, label names, and that at least one entity is found.
```

---

### Cell 28: code - Lab 2 Starter Code

```python
# Lab 2: Named entity recognition with AutoTokenizer + AutoModelForTokenClassification.

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F

NER_MODEL_ID = "dslim/bert-base-NER"

COMPLAINT_NER = (
    "I visited the Canary Wharf Barclays branch yesterday and spoke with "
    "John Smith about the dispute on my HSBC credit card statement."
)

# Step 1: Load the tokenizer for NER_MODEL_ID into ner_tokenizer.
ner_tokenizer = None   # YOUR CODE

# Step 2: Load the model into ner_model (use AutoModelForTokenClassification).
ner_model = None   # YOUR CODE

if ner_model is not None:
    ner_model.eval()

# Step 3: Tokenise COMPLAINT_NER with return_tensors="pt", truncation=True, max_length=128.
#         Store in ner_inputs.
ner_inputs = None   # YOUR CODE

# Step 4: Run a forward pass (torch.no_grad()) and extract .logits.
#         Apply softmax over the last dimension. Store in ner_probs: shape (1, seq_len, num_labels).
ner_probs = None   # YOUR CODE

# Step 5: Build ner_predictions as a list of dicts {"token": str, "label": str, "score": float}.
#         Use ner_model.config.id2label to map label index -> label name.
#         Skip special tokens ([CLS], [SEP], [PAD]).
ner_predictions = None   # YOUR CODE

# Quick preview (do not change).
if ner_predictions is not None:
    print(f"Number of token predictions: {len(ner_predictions)}")
    print("First 5 predictions:")
    for pred in ner_predictions[:5]:
        print(f"  {pred}")
else:
    print("ner_predictions is None -- complete all steps.")
```

---

### Cell 29: code - Lab 2 Verification

```python
# Lab 2 Verification

all_pass = True

if ner_tokenizer is None:
    print("FAIL: ner_tokenizer is None -- complete Step 1.")
    all_pass = False
else:
    print(f"PASS: ner_tokenizer loaded ({type(ner_tokenizer).__name__})")

if ner_model is None:
    print("FAIL: ner_model is None -- complete Step 2.")
    all_pass = False
else:
    print(f"PASS: ner_model loaded ({type(ner_model).__name__})")
    print(f"      num_labels={ner_model.config.num_labels}, "
          f"labels={list(ner_model.config.id2label.values())[:6]}")

if ner_inputs is None:
    print("FAIL: ner_inputs is None -- complete Step 3.")
    all_pass = False
else:
    if "input_ids" in ner_inputs and "attention_mask" in ner_inputs:
        print(f"PASS: ner_inputs has correct keys, input_ids shape={ner_inputs['input_ids'].shape}")
    else:
        print(f"FAIL: ner_inputs missing required keys: {list(ner_inputs.keys())}")
        all_pass = False

if ner_probs is None:
    print("FAIL: ner_probs is None -- complete Step 4.")
    all_pass = False
else:
    if ner_probs.ndim == 3:
        print(f"PASS: ner_probs shape={tuple(ner_probs.shape)}  (batch, seq_len, num_labels)")
    else:
        print(f"FAIL: ner_probs should be 3-dimensional, got shape {tuple(ner_probs.shape)}")
        all_pass = False

if ner_predictions is None:
    print("FAIL: ner_predictions is None -- complete Step 5.")
    all_pass = False
else:
    if isinstance(ner_predictions, list) and len(ner_predictions) > 0:
        first = ner_predictions[0]
        required_keys = {"token", "label", "score"}
        if required_keys.issubset(first.keys()):
            print(f"PASS: ner_predictions has {len(ner_predictions)} token predictions")
        else:
            print(f"FAIL: Each prediction dict must have keys {required_keys}, got {set(first.keys())}")
            all_pass = False
        # Check at least one entity found.
        entity_preds = [p for p in ner_predictions if not p["label"].startswith("O")]
        if len(entity_preds) > 0:
            print(f"PASS: Found {len(entity_preds)} non-O entity predictions")
            print("      Sample entities:")
            for ep in entity_preds[:4]:
                print(f"        {ep}")
        else:
            print("FAIL: No non-O entities found -- check label mapping in Step 5.")
            all_pass = False
    else:
        print("FAIL: ner_predictions should be a non-empty list.")
        all_pass = False

if all_pass:
    print()
    print("All Lab 2 checks passed.")
    print("You have run NER inference manually with full control over token probabilities.")
```

---

### Cell 30: code - Lab 2 Safety-Net

```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.

_need_sn2 = False
try:
    if ner_predictions is None or len(ner_predictions) == 0:
        _need_sn2 = True
except Exception:
    _need_sn2 = True

if _need_sn2:
    print("Using Lab 2 safety-net so the rest of the notebook can run.")
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch.nn.functional as F

    _ner_tok  = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    _ner_mod  = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    _ner_mod.eval()
    _inputs   = _ner_tok(COMPLAINT_NER, return_tensors="pt",
                         truncation=True, max_length=128)
    with torch.no_grad():
        _logits = _ner_mod(**_inputs).logits
    _probs = F.softmax(_logits, dim=-1)
    _tokens = _ner_tok.convert_ids_to_tokens(_inputs["input_ids"][0])
    _special = {_ner_tok.cls_token, _ner_tok.sep_token, _ner_tok.pad_token}
    ner_tokenizer   = _ner_tok
    ner_model       = _ner_mod
    ner_inputs      = _inputs
    ner_probs       = _probs
    ner_predictions = []
    for tok, prob_row in zip(_tokens, _probs[0]):
        if tok in _special:
            continue
        label_idx = prob_row.argmax().item()
        ner_predictions.append({
            "token": tok,
            "label": _ner_mod.config.id2label[label_idx],
            "score": float(prob_row[label_idx]),
        })
    print(f"  Loaded safety-net NER predictions: {len(ner_predictions)} tokens")
    _entities = [p for p in ner_predictions if not p["label"].startswith("O")]
    print(f"  Non-O entities found: {len(_entities)}")
```

---

### Cell 31: markdown - Lab 2 Stretch and Homework

Content:
```
### Stretch (fast finishers)

Implement span aggregation: merge consecutive B-/I- tokens (IOB2 format) into
complete entity spans with their full text and aggregate score (mean of token scores).

```python
def aggregate_ner_spans(ner_predictions):
    """
    Merge IOB2 token predictions into complete entity spans.

    Args:
        ner_predictions: list of {"token": str, "label": str, "score": float}

    Returns:
        list of {"span": str, "label": str, "mean_score": float}
        Only non-O entities. Consecutive I- tokens are merged with the preceding B- token.
    """
    pass   # implement for stretch
```

Run it on COMPLAINT_NER and verify it correctly identifies "Canary Wharf Barclays branch",
"John Smith", and "HSBC".

### Homework Extension

Add an entity-level evaluation loop. Load the CoNLL-2003 NER test set from the Hub
(`conll2003`, split="test") and compute entity-level F1 score (not token-level accuracy)
for dslim/bert-base-NER.

```python
from datasets import load_dataset

def entity_f1(model, tokenizer, dataset_split, max_examples=200):
    """
    Compute entity-level precision, recall, and F1 for an NER model.

    Args:
        model       : AutoModelForTokenClassification in eval mode
        tokenizer   : matching AutoTokenizer
        dataset_split: HuggingFace Dataset with "tokens" and "ner_tags" columns
        max_examples: number of examples to evaluate

    Returns:
        dict with keys "precision", "recall", "f1"
    """
    pass   # implement for homework
```

Compare your result to the dslim/bert-base-NER model card (should be ~91 F1).
```

---

### Cell 32: markdown - Section 5: Hub Upload -- Sharing a Checkpoint

Content:
```
## Section 5 - Uploading to the Hub: Sharing Your Work

The same ecosystem that lets you download pre-trained models lets you
publish your own checkpoints so colleagues -- and the community -- can use them.

The upload API:

    from huggingface_hub import HfApi
    api = HfApi()

    # Create a repo (once).
    api.create_repo(repo_id="your-username/your-model-name", private=True)

    # Upload a file.
    api.upload_file(
        path_or_fileobj="path/to/local/file",
        path_in_repo="filename_on_hub.bin",
        repo_id="your-username/your-model-name",
    )

    # OR: upload the whole model/tokenizer using the transformers method.
    model.push_to_hub("your-username/your-model-name")
    tokenizer.push_to_hub("your-username/your-model-name")

For Barclays-trained models, you would push to a PRIVATE repo on a
Barclays Enterprise Hub organisation. Never push to a public repo
without information security review.
```

---

### Cell 33: code - Beat 1: push_to_hub Without Authentication

```python
# Beat 1: What happens when you try to push to the Hub without a token?
# The HF client raises a LocalTokenNotFoundError (or similar) -- clear error message.

from transformers import AutoTokenizer

print("Beat 1: push_to_hub without authentication")
print("=" * 50)
print()

_tok = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

print("Attempting push_to_hub without a token ...")
try:
    _tok.push_to_hub("fake-user/barclays-complaint-classifier-demo")
    print("Unexpected success -- a token must already be cached.")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
    print()
    print("Expected: RepositoryNotFoundError or LocalTokenNotFoundError.")
    print("The Hub requires authentication to create or write to repositories.")
    print("FIX: authenticate first (Beat 3).")
```

---

### Cell 34: code - Beat 3: Hub Upload Demo (Full Working Pattern)

```python
# Beat 3: Complete Hub upload pattern using huggingface_hub HfApi.
# We demonstrate the API using local temp files -- no real Hub push happens
# unless the student has a valid HF token cached.
# The try/except block makes this cell safe to run in a sandboxed environment.

from huggingface_hub import HfApi
import tempfile, os, json

print("Beat 3: Hub upload pattern (dry-run safe)")
print("=" * 55)
print()

# -- Step 1: Create the repo (requires auth). --
api = HfApi()
DEMO_REPO = "REPLACE-WITH-YOUR-USERNAME/barclays-complaint-classifier-demo"

print("Step 1: Create a private repo on the Hub")
print(f"  Repo: {DEMO_REPO}")
try:
    repo_url = api.create_repo(
        repo_id=DEMO_REPO,
        private=True,       # ALWAYS private for internal Barclays models
        exist_ok=True,      # do not raise if repo already exists
    )
    print(f"  Created: {repo_url}")
except Exception as e:
    print(f"  Skipped (no token or repo name not updated): {type(e).__name__}")
print()

# -- Step 2: Write a model card (README.md). --
print("Step 2: Write a model card (README.md)")
model_card_content = """---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags:
- barclays
- complaint-classification
- distilbert
datasets:
- financial_phrasebank
base_model: distilbert/distilbert-base-uncased-finetuned-sst-2-english
---

# Barclays Complaint Sentiment Classifier

Fine-tuned from distilbert-finetuned-sst-2-english for financial complaint sentiment.

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="your-username/barclays-complaint-classifier-demo")
clf("My account was charged without authorisation.")
```

## Training data

Fine-tuned on financial_phrasebank (sentences_allagree split, 2,264 examples).

## Limitations

Binary POSITIVE/NEGATIVE output. Neutral financial statements are misclassified.
Do not use for automated decisions without human review.
"""

with tempfile.TemporaryDirectory() as tmpdir:
    readme_path = os.path.join(tmpdir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card_content)
    print(f"  Model card written ({len(model_card_content)} chars)")
    print(f"  YAML front-matter: pipeline_tag=text-classification, library_name=transformers")
    print()

    # -- Step 3: Upload model card to Hub. --
    print("Step 3: Upload README.md to Hub")
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=DEMO_REPO,
            commit_message="Add model card with pipeline_tag and usage instructions",
        )
        print(f"  Uploaded: README.md -> {DEMO_REPO}")
    except Exception as e:
        print(f"  Skipped (no token or repo name not updated): {type(e).__name__}")
    print()

# -- Step 4: Push tokenizer using transformers method. --
print("Step 4: Push tokenizer using push_to_hub()")
try:
    tokenizer.push_to_hub(DEMO_REPO, commit_message="Add tokenizer files")
    print(f"  Pushed tokenizer to {DEMO_REPO}")
    print("  Uploaded: tokenizer_config.json, vocab.txt, special_tokens_map.json")
except Exception as e:
    print(f"  Skipped: {type(e).__name__}")
print()

print("Hub upload pattern summary:")
print("  1. api.create_repo(repo_id, private=True, exist_ok=True)")
print("  2. Write README.md with YAML front-matter (pipeline_tag is mandatory)")
print("  3. api.upload_file() or model.push_to_hub()")
print("  4. Always set private=True for internal Barclays models")
```

---

### Cell 35: markdown - Lab 3 Header: Tier 1 - Write a Model Card

Content:
```
## Lab 3 - Write and Upload a Model Card (Tier 1 - Guided)

**Time**: 15-20 minutes

### Situation

The Barclays NLP team has trained the zero-shot complaint router from Section 2 on
an internal dataset. Information Security requires a model card before the model
can be deployed to production. The model card must include: license, pipeline_tag,
base_model, known limitations.

### Task

Write a model card for a hypothetical Barclays zero-shot complaint router and
create a HfApi object that could upload it (without actually uploading, since
we may not have valid Hub credentials in this environment).

### Action

Fill in the three stubs. The README.md content must include valid YAML front-matter
with at minimum: language, license, pipeline_tag, library_name.

### Result

Verification cell checks: YAML front-matter is parseable, pipeline_tag is correct,
HfApi object is instantiated, limitations section exists.
```

---

### Cell 36: code - Lab 3 Starter Code

```python
# Lab 3: Write a model card for the Barclays zero-shot complaint router.

import yaml

# Step 1: Define the model card content as a Python string called `my_model_card`.
#         It must start with three dashes (---) and contain valid YAML front-matter
#         with at minimum: language, license, pipeline_tag, library_name.
#         Include a "## Limitations" section below the YAML block.
my_model_card = None   # YOUR CODE

# Step 2: Instantiate a HfApi object called `my_api`.
#         Do not call any upload methods -- just create the object.
my_api = None   # YOUR CODE

# Step 3: Set `my_repo_id` to a valid HuggingFace repo id string.
#         Format: "your-username/barclays-zero-shot-router"
#         Replace "your-username" with your actual HuggingFace username if you have one,
#         or use "barclays-demo/barclays-zero-shot-router" as a placeholder.
my_repo_id = None   # YOUR CODE

# Quick preview.
if my_model_card is not None:
    print("Model card preview (first 300 chars):")
    print(my_model_card[:300])
else:
    print("my_model_card is None -- complete Step 1.")
```

---

### Cell 37: code - Lab 3 Verification

```python
# Lab 3 Verification

all_pass = True

# Check model card content.
if my_model_card is None:
    print("FAIL: my_model_card is None -- complete Step 1.")
    all_pass = False
else:
    if my_model_card.startswith("---"):
        print("PASS: Model card starts with YAML front-matter delimiter (---)")
    else:
        print("FAIL: Model card must start with '---' (YAML front-matter)")
        all_pass = False

    # Parse YAML front-matter.
    try:
        yaml_end   = my_model_card.index("---", 3)
        yaml_block = my_model_card[3:yaml_end].strip()
        metadata   = yaml.safe_load(yaml_block)

        required_fields = ["language", "license", "pipeline_tag", "library_name"]
        missing = [f for f in required_fields if f not in metadata]
        if not missing:
            print(f"PASS: YAML front-matter contains all required fields: {required_fields}")
        else:
            print(f"FAIL: Missing required YAML fields: {missing}")
            all_pass = False

        if metadata.get("pipeline_tag") == "zero-shot-classification":
            print("PASS: pipeline_tag = 'zero-shot-classification'")
        else:
            print(f"FAIL: pipeline_tag should be 'zero-shot-classification', "
                  f"got '{metadata.get('pipeline_tag')}'")
            all_pass = False

    except Exception as e:
        print(f"FAIL: Could not parse YAML front-matter: {e}")
        all_pass = False

    if "## Limitations" in my_model_card or "## limitations" in my_model_card.lower():
        print("PASS: Model card contains a Limitations section")
    else:
        print("FAIL: Model card must include a '## Limitations' section")
        all_pass = False

# Check HfApi.
if my_api is None:
    print("FAIL: my_api is None -- complete Step 2.")
    all_pass = False
else:
    from huggingface_hub import HfApi
    if isinstance(my_api, HfApi):
        print(f"PASS: my_api is an HfApi instance")
    else:
        print(f"FAIL: Expected HfApi instance, got {type(my_api).__name__}")
        all_pass = False

# Check repo_id.
if my_repo_id is None:
    print("FAIL: my_repo_id is None -- complete Step 3.")
    all_pass = False
elif isinstance(my_repo_id, str) and "/" in my_repo_id:
    print(f"PASS: my_repo_id = '{my_repo_id}'")
else:
    print(f"FAIL: my_repo_id must be a string in format 'username/repo-name', got: {my_repo_id}")
    all_pass = False

if all_pass:
    print()
    print("All Lab 3 checks passed.")
    print("Your model card is ready for Information Security review.")
```

---

### Cell 38: markdown - Lab 3 Stretch and Homework

Content:
```
### Stretch (fast finishers)

Add a YAML `metrics` section to your model card showing accuracy on
financial_phrasebank (from Cell 22):

```yaml
metrics:
- type: accuracy
  value: 0.76
  dataset:
    name: financial_phrasebank
    type: financial_phrasebank
    config: sentences_allagree
    split: train
```

Verify the updated YAML still parses correctly.

### Homework Extension

Write a function `validate_model_card(readme_path)` that:
1. Parses the YAML front-matter
2. Checks for required fields (language, license, pipeline_tag, library_name)
3. Checks that pipeline_tag is one of the valid HuggingFace task identifiers
4. Raises ValueError with a descriptive message for each violation
5. Returns True if all checks pass

This is the kind of automated check an MLOps pipeline would run before allowing
a model card to be published to the Barclays internal Hub organisation.

```python
VALID_PIPELINE_TAGS = {
    "text-classification",
    "zero-shot-classification",
    "token-classification",
    "text-generation",
    "translation",
    "summarization",
    "fill-mask",
    "feature-extraction",
}

def validate_model_card(readme_content):
    """
    Validate a HuggingFace model card README.md string.

    Returns True if the card is valid. Raises ValueError otherwise.
    """
    pass   # implement for homework
```

---

### Cell 39: markdown - Wrap-Up and Bridge to Topic 6a

Content:
```
## Wrap-Up

### What you built in Topic 5

| Section | What you did | Key takeaway |
|---------|-------------|-------------|
| 1 - Hub | Explored model cards, pipeline_tag, library_name | The Hub is a metadata-first distribution layer |
| 2 - pipeline() | Sentiment + zero-shot routing in 4 lines | pipeline() reads model card metadata to auto-configure |
| 3 - datasets | Loaded financial_phrasebank, evaluated SST-2 transfer | Domain gap exists -- 70-80% on finance vs 91.3% on SST-2 |
| 4 - AutoModel | Manual tokenisation, forward pass, logits, NER | Full control when you need embeddings or custom heads |
| 5 - Hub upload | Model card, push_to_hub, HfApi | Share checkpoints privately; Information Security review required |

### Key facts to remember

- pipeline_tag in model card MUST match the task argument in pipeline()
- AutoModel returns raw hidden states (no head); AutoModelForSequenceClassification adds a head
- AutoTokenizer must be loaded from the SAME model_id as the model (same vocabulary)
- push_to_hub requires authentication; always use private=True for internal Barclays models
- The evaluate library is incompatible with datasets 4.x -- use inline numpy for metrics

### The gap you just identified

The SST-2 classifier reached ~70-80% accuracy on financial text vs 91.3% on movies.
The zero-shot router is flexible but slow (one forward pass per candidate label).
Neither model was trained on Barclays complaint data.

### What is coming next

Topic 6a - Full Fine-Tuning:
You will fine-tune a pre-trained DistilBERT on a Barclays-domain dataset using a remote
GPU job. You will see catastrophic forgetting in action -- what happens when you fine-tune
too aggressively on a small dataset.

Topic 6b - Transfer Learning with frozen encoder:
You will freeze the DistilBERT encoder and train only the classification head.
The Hub models you used today are your starting points.
```

---

### Cell 40: markdown - End of Notebook Marker

Content:
```
---

*End of Topic 5 - HuggingFace Ecosystem*

Next session: Topic 6a - Full Fine-Tuning and Catastrophic Forgetting.
Fine-tune DistilBERT on Barclays complaint data. See what happens when you train too hard.
```

---

## Implementation Notes for /build-topic-notebook

1. **Output paths**:
   - Exercise: `Exercises/topic_5_huggingface/topic_5_huggingface.ipynb`
   - Solution:  `Solutions/topic_5_huggingface/topic_5_huggingface.ipynb`

2. **Total cells**: 40 cells as planned (markdown + code). The 5-cell approval cadence
   applies: stop after every 5 cells and await approval before continuing.

3. **Lab tiers**:
   - Lab 1 (Cell 10): Tier 1 guided -- 3 stubs, numbered steps in markdown header.
   - Lab 2 (Cell 28): Tier 1 guided -- 5 stubs, numbered steps in markdown header.
   - Lab 3 (Cell 36): Tier 1 guided -- 3 stubs, numbered steps in markdown header.
   - NO Tier 2 (Topic 4 used the Day 2 Tier 2 slot).
   - NO Tier 3 (reserved for last topic of Day 2 = Topic 7b).

4. **Safety-nets**:
   - Cell 12 (Lab 1 safety-net): sets zs_info, zs_pipeline_tag, zs_tags.
   - Cell 30 (Lab 2 safety-net): sets ner_tokenizer, ner_model, ner_inputs,
     ner_probs, ner_predictions. Full working NER implementation.
   - Lab 3 has NO safety-net (its outputs do not feed downstream cells).
   - Remove all safety-net cells from the solution notebook.

5. **Variable continuity**:
   - `device`, `set_seeds()`, `COMPLAINT_TOKENS`, `warnings.filterwarnings("ignore")`
     carried from Topic 4 with identical signatures.
   - `COMPLAINT_LABELS` introduced in Cell 3 and consumed in Cell 17 (zero-shot demo).
   - `classifier` (Cell 16) consumed in Cell 22 (financial_phrasebank evaluation).
   - `tokenizer`, `model` (Cell 26) consumed in Lab 3 (Cell 34 push_to_hub demo).

6. **Model downloads**: All models are public and require no HF token. First download
   in-class may take 30-60 seconds each. Instructor should pre-run Cells 2-8 before class.

7. **trust_remote_code=True**: Required for `financial_phrasebank` dataset (Cell 21).
   Note this explicitly in the instructor notes -- students may be confused by the prompt.

8. **AI-tells enforcement**: No em dashes, no en dashes, no Unicode multiplication,
   no emojis anywhere in cell bodies, print statements, or markdown. Plain ASCII only.

9. **No evaluate library**: Confirmed. Cell 22 uses inline numpy Counter and arithmetic.
   No `import evaluate` appears anywhere.

10. **Markdown chain check**:
    - Cells 4, 5 are back-to-back markdown -> Cell 6 is code. OK (2 consecutive).
    - Cells 14, 23 are followed by code. OK.
    - Cells 19 is followed by Cell 20 (code). OK.
    - Cells 32, 33 (markdown then code). OK.
    - Cells 35 markdown -> 36 code. OK.
    - No run of 3+ consecutive markdown cells without a code cell.

11. **Cell 24 clarification**: Cell 24 is a MARKDOWN cell (Beat 2 diagram placeholder).
    It is placed between the Section 4 explanation markdown (Cell 23) and the Beat 1
    broken code (Cell 25). This is the correct Beat 2 position: after the concept
    explanation, before the working demo.

12. **Diagram placement**:
    - Diagram 1 (hf-hub-ecosystem): Cell 7 -- after Beat 1 broken demo for Section 1.
    - Diagram 2 (automodel-class-hierarchy): Cell 24 -- Beat 2 for Section 4 AutoModel.

13. **Cell 34 push_to_hub demo**: Uses try/except so it runs safely in sandboxed
    environments without valid HF tokens. Instructors with tokens can demonstrate live.
    The demo uses `tokenizer` (loaded in Cell 26) -- confirm that cell runs before 34.

14. **Discussion cell** (Cell 18): Run by the instructor as a 3-minute class exercise.
    Three questions covering latency math, domain transfer gap, and deployment patterns.
    Do not skip -- it sets up the motivation for Topics 6a and 6b.

15. **The `model` variable from Cell 26** is `AutoModelForSequenceClassification` for
    distilbert-finetuned-sst-2-english. It is used in Cell 34 for the push_to_hub demo.
    If the kernel is restarted between Sections 4 and 5, students must re-run Cell 26.

16. **NO SageMaker session setup** in this notebook. Topic 5 is inference-only in the
    Studio kernel. The first SageMaker-free notebook since Topic 1. Make this explicit
    in the instructor briefing -- it is intentional.
