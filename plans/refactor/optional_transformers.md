# Rework Design Doc: topic_optional_transformers

## Purpose of this document

This is a cell-by-cell rework plan for the notebook pair:

- `Exercises/topic_optional_transformers/topic_optional_transformers.ipynb`
- `Solutions/topic_optional_transformers/topic_optional_transformers.ipynb`

A separate notebook-building agent must be able to implement this rework WITHOUT
re-reading the original notebook. Every cell of the reworked notebook is specified
below: cell number, type, purpose, and either full content or a precise change
description (KEEP / EDIT / NEW / DELETE / MERGE).

### Why the rework

The notebook (formerly "Topic 4 - Transformers + Translator Capstone") is being
DEMOTED from a required sequential Day-2 topic to a STANDALONE OPTIONAL deep-dive.
The current notebook assumes the linear course path. It must become fully
self-contained: it cannot assume the old attention optionals (old Topics 3a/3b)
were done, and it cannot chain forward to old Topic 5 (Hugging Face).

Broader course feedback: attention/transformer topics were "too math-heavy;
students are USERS of models, not transformer authors." This notebook stays
OPTIONAL precisely because it is build-from-scratch internals. The rework keeps
the internals but reframes them: open with WHY a practitioner might want to
understand transformer internals BEFORE any math, and strip every sentence that
implies a fixed position in a linear course.

### What stays the same

- The four-beat teaching arc (Beat 1 broken/naive, Beat 2 diagram, Beat 3 working
  demo, Beat 4 lab).
- Lab 1 (Tier 1 guided, PositionalEmbedding) and Lab 2 (Tier 2 hard, DecoderLayer).
- The remote GPU training-job capstone using `scripts_optional_transformers/train.py`.
- All architecture code (PositionalEmbedding, EncoderLayer, Encoder, DecoderLayer,
  Decoder, Translator), all shape tests, all verification cells, all safety-net cells.
- Plain ASCII only. No em-dashes, no en-dashes, no Unicode multiplication signs,
  no emojis, anywhere.

### What changes (summary)

1. NEW first markdown cell: OPTIONAL / SUPPLEMENTARY banner.
2. Title cell loses "Topic 4" and "| Topic 4"; "What you will build" intro rewritten
   to drop "In Topics 3a and 3b you implemented...".
3. DELETE the "Day 2 System Overview" / "YOU ARE HERE" course-progression table cell,
   replace with a short static "this is an optional deep-dive" note.
4. NEW motivation markdown cell: "Why understand transformer internals?" placed
   before any math.
5. NEW self-contained recap markdown cell: scaled-dot-product attention restated
   inline so a student who skipped the attention optionals can follow.
6. EDIT every cell that says "In Topic 3b...", "In Topics 3a and 3b...",
   "In Topic 5...", "The DistilBERT you load in Topic 5...", "Day 2 capstone",
   "first remote training job in the course", "capstones in Days 2 and 3",
   "Next session: Topic 5" -> reframe as optional-track language.
7. Capstone instance: the prose must consistently say `ml.g4dn.xlarge` (GPU, NVIDIA
   T4). The current notebook is internally inconsistent (markdown and one comment say
   g4dn GPU; the estimator code and another comment say `ml.m5.xlarge` CPU). Resolve
   in favor of `ml.g4dn.xlarge` per the rework brief, and fix the estimator cell.

### Cell count

Original: 36 cells. Reworked: 39 cells (3 NEW markdown cells added; no deletions of
code; one markdown cell replaced in place).

### Solutions twin

The Solutions notebook is identical to the Exercises notebook EXCEPT:

- Every lab starter cell has its `= None  # YOUR CODE` stubs replaced with the full
  working implementation (see Lab 1 and Lab 2 solution code given inline below).
- The two safety-net cells (Exercises cells 19 and 29 in the new numbering, see below)
  are DELETED from the Solutions notebook, because the lab cell itself is the working
  implementation there. NOTE: this means the Solutions notebook has 37 cells, and the
  cell numbering after each removed safety-net shifts down by one. All other cells are
  byte-identical to the Exercises notebook.
- The training-job safety-net cell (Exercises cell 35, "run this if your kernel
  restarted") is KEPT in the Solutions notebook unchanged. It is an operational
  recovery cell, not a lab safety-net, so it stays.

Build order: build Exercises fully first, then `cp` to Solutions and apply the three
changes above (fill Lab 1, fill Lab 2, delete the two lab safety-net cells).

---

## Cell-by-cell plan (Exercises notebook)

New numbering runs 0..38. Each entry maps to the original where applicable.

---

### Cell 0 - NEW - markdown - Optional/Supplementary banner

Action: NEW. This becomes the very first cell. Full content:

```markdown
# Optional Deep-Dive: Transformers From Scratch

> **This is an optional supplementary notebook.** The main course path does not
> require it. You can complete every required topic without opening this notebook.

### Who this is for

This deep-dive is for developers who want to see what is actually inside a
Transformer. You will build the architecture behind BERT, GPT, and T5 module by
module in PyTorch, then train it on a remote GPU. If you are happy treating models
as black boxes you can safely skip this. If you are the kind of engineer who wants
to open the box, read on.

### Is it self-contained?

Yes. This notebook does not depend on any other notebook in the course. Comfort
with the idea of attention helps, but a short recap is included below, so you can
follow even if you have never seen scaled dot-product attention before.

### What you need

A SageMaker notebook kernel for the architecture demos (CPU is fine), and
permission to launch one short remote GPU training job (ml.g4dn.xlarge) for the
capstone at the end. The capstone is optional within this optional notebook: the
architecture is fully built and tested before you submit any remote job.
```

---

### Cell 1 - EDIT - markdown - Title and "What you will build"

Maps to original cell 0.

Old text:

```
# Topic 4 - Transformers + Translator Capstone

Barclays Customer Support Intelligence System | Topic 4

## What you will build
In Topics 3a and 3b you implemented scaled dot product attention from scratch and ported it to PyTorch.
In this notebook you will:
1. Understand WHY seq2seq with attention still struggled on long sequences
2. Build the full Transformer architecture from scratch in PyTorch
3. Combine your components into a Translator model
4. Submit your first remote GPU training job on SageMaker

## Learning objectives
1. Articulate the bottlenecks in seq2seq + attention that Transformers solve
2. Implement positional encoding and multi-head self-attention in PyTorch
3. Assemble encoder, decoder, and full Transformer from modular components
4. Submit a SageMaker PyTorch training job and retrieve model artifacts
```

New text:

```markdown
## Transformers and a Translator, Built From Scratch

A self-contained build of the Transformer architecture, framed around a Barclays
Customer Support translation scenario.

## What you will build
This notebook restates everything it needs as it goes, including a recap of
scaled dot-product attention. In it you will:
1. Understand WHY a sequence model that reads tokens one at a time struggles on
   long sequences
2. Build the full Transformer architecture from scratch in PyTorch
3. Combine your components into a Translator model
4. Submit a remote GPU training job on SageMaker

## Learning objectives
1. Articulate the bottlenecks that the Transformer architecture solves
2. Implement positional encoding and multi-head self-attention in PyTorch
3. Assemble encoder, decoder, and full Transformer from modular components
4. Submit a SageMaker PyTorch training job and retrieve model artifacts
```

---

### Cell 2 - NEW - markdown - Motivation: why understand the internals

Action: NEW. Placed before any math, before Section 0. Full content:

```markdown
## Why would a model USER want to understand the internals?

Most of the time you will use Transformers, not build them. You load a pre-trained
model and call it. So why spend an afternoon building one by hand?

Because the internals explain the behaviour you will debug in production:

- **Context windows and cost.** Attention compares every token to every other
  token. Once you have built that comparison yourself, the quadratic cost of long
  prompts stops being a mystery and becomes an obvious consequence of the design.
- **Why position matters.** Models can be sensitive to where information sits in a
  prompt. That is not a bug; it is positional encoding doing its job. You will see
  exactly why by removing it and watching the model go order-blind.
- **Encoder vs decoder vs encoder-decoder.** BERT-style, GPT-style, and T5-style
  models differ in which half of this architecture they keep. Knowing the halves
  tells you which model class fits which task.
- **Fine-tuning intuition.** When you later freeze layers, attach adapters, or pick
  a learning rate, you are reasoning about the exact blocks you are about to build.

You do not need this to use a model. You need it to use a model well, and to know
what to try when one misbehaves. That is the whole reason this optional notebook
exists.
```

---

### Cell 3 - KEEP - markdown - Section 0 header

Maps to original cell 1. KEEP unchanged.

```markdown
## Section 0 - Environment Setup
```

---

### Cell 4 - REPLACE - markdown - Optional deep-dive note (was "YOU ARE HERE" table)

Maps to original cell 2. Action: DELETE the old content, REPLACE in place with a
short static note. The course-progression table with T4/T5/T6a/T6b/T7a/T7b and the
"YOU ARE HERE" marker is removed entirely because it implies a linear position.

Old text (DELETE):

```
## Day 2 System Overview

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | T4 Transformers | Build the architecture from scratch (YOU ARE HERE) |
| 2 | T5 HuggingFace | Load pre-trained models from the Hub |
| 3 | T6a Full Fine-Tuning | Adapt a model to Barclays complaints |
| 4 | T6b Transfer Learning | Freeze the encoder, train only the head |
| 5 | T7a LoRA from Scratch | Implement parameter-efficient adaptation |
| 6 | T7b PEFT + LoRA | Apply PEFT library to a full classifier |

By end of Day 2 you will have a fine-tuned, PEFT-adapted DistilBERT complaint classifier
running as a SageMaker endpoint.
```

New text:

```markdown
## How this notebook is organised

This is a standalone deep-dive, so it does not sit at a fixed point in any course
sequence. It runs front to back on its own:

- Section 1 - why a token-at-a-time sequence model struggles, and a recap of
  scaled dot-product attention
- Section 2 - positional encoding, plus Lab 1
- Section 3 - building the Transformer block by block, plus Lab 2
- Section 4 - the capstone: training the Translator on a remote GPU

Run the cells in order. Each section restates what it needs from the section
before it.
```

---

### Cell 5 - KEEP - code - TensorFlow backend disable

Maps to original cell 3. KEEP unchanged.

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

### Cell 6 - EDIT - code - pip install

Maps to original cell 4. Only the leading comment changes (it said "Topic 4" and
"Day 2 capstone" framing). The pip install and restart print stay.

Old comment block:

```
# Topic 4 - Transformers + Translator Capstone
# All architecture demos run in this kernel (CPU, ml.t3.medium).
# The Capstone submits a remote GPU job on ml.g4dn.xlarge.
```

New comment block:

```
# Optional deep-dive: Transformers from scratch.
# All architecture demos run in this notebook kernel (CPU is fine).
# The capstone at the end submits a remote GPU job on ml.g4dn.xlarge.
```

Rest of cell unchanged:

```python
!pip install -q \
    "sagemaker>=2.200.0,<3.0.0" \
    "numpy<2" \
    "matplotlib>=3.7.0"

print("RESTART KERNEL before continuing -- environment packages were installed/upgraded.")
```

---

### Cell 7 - KEEP - code - imports, seeds, SageMaker session

Maps to original cell 5. KEEP unchanged (the comment "capstone trains on remote GPU"
is generic and fine). Content:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import warnings

import sagemaker
from sagemaker import get_execution_role
import boto3

warnings.filterwarnings("ignore")

# Reproducibility - set_seeds is defined here for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(42)

# Device check - training cells run on CPU here; capstone trains on remote GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device (notebook kernel): {device}")
print()

# SageMaker session - needed for the capstone remote training job
sess = sagemaker.Session()
role = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"SageMaker role: {role}")
print(f"Default bucket: {bucket}")
print(f"Region: {region}")
```

---

### Cell 8 - EDIT - markdown - "What are we building"

Maps to original cell 6. Remove the "In Topics 3a and 3b we added attention ON TOP
of an RNN" sequential reference; restate self-contained.

Old text:

```
## What are we building today?

In Topics 3a and 3b we added attention ON TOP of an RNN. That helped, but the RNN
was still the bottleneck.

Today we throw the RNN away entirely and use attention for everything.
The result is the Transformer - the architecture behind BERT, GPT, T5, and every
modern LLM you have heard of.

By the end of this notebook you will have:
1. A working Transformer built module by module in PyTorch
2. A translator that maps Spanish complaint tickets to English
3. Your first remote GPU training job running on SageMaker ml.g4dn.xlarge

The Barclays scenario: the complaints team receives tickets in Spanish from
Latin American customers. We will build a translator to route those tickets
to the English-speaking review team.
```

New text:

```markdown
## What are we building?

Earlier sequence models processed text with a recurrent network (an RNN): one
token at a time, each step feeding the next. Attention was bolted on top to let
the model focus on relevant earlier tokens, but the recurrent network was still
the bottleneck.

The Transformer throws the recurrent network away entirely and uses attention for
everything. The result is the architecture behind BERT, GPT, T5, and every modern
large language model you have heard of.

By the end of this notebook you will have:
1. A working Transformer built module by module in PyTorch
2. A translator that maps Spanish complaint tickets to English
3. A remote GPU training job running on SageMaker ml.g4dn.xlarge

The Barclays scenario: the complaints team receives tickets in Spanish from
Latin American customers. We will build a translator to route those tickets
to the English-speaking review team.
```

---

### Cell 9 - NEW - markdown - Recap: scaled dot-product attention

Action: NEW. Self-contained recap so a student who skipped the attention optionals
can follow. Placed before Section 1. Full content:

```markdown
## Recap: scaled dot-product attention in one screen

Everything in this notebook is built on one operation. If you have seen it before,
skim this. If you have not, this is all you need.

Attention answers: for a given token, which other tokens should it look at, and how
much? It works with three vectors per token:

- **Query (Q)** - what this token is looking for
- **Key (K)** - what each token offers, used for matching
- **Value (V)** - the actual information each token carries

The computation, for sequences packed into matrices:

```
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
```

Step by step:

1. `Q K^T` scores every query against every key. For a sequence of length T this
   is a T-by-T matrix: row i, column j is "how much should token i attend to
   token j".
2. Divide by `sqrt(d_k)` (the key dimension). Without this the scores grow large,
   softmax saturates, and gradients shrink. This is the "scaled" part.
3. `softmax` over each row turns the scores into weights that sum to 1.
4. Multiply by `V`: each token's output is a weighted blend of every token's value.

**Self-attention** is the case where Q, K, and V all come from the same sequence:
every token attends to every token in its own sequence. **Cross-attention** is when
Q comes from one sequence and K, V come from another; that is how a decoder reads
the encoder's output. **Multi-head attention** runs several of these in parallel on
different learned projections and concatenates the results, so the model can attend
in several ways at once.

That is the whole foundation. The rest of this notebook stacks this operation into
encoders, decoders, and a full Translator.
```

---

### Cell 10 - EDIT - markdown - Beat 1 Section 1 header

Maps to original cell 7. Remove "In Topic 3b we saw that attention helps...".

Old text:

```
## Beat 1 -- Section 1 - Why Transformers? The Problem with Seq2Seq + Attention


In Topic 3b we saw that attention helps the decoder focus on relevant encoder states.
But we kept the RNN around the outside. That caused two problems:

1. Sequential processing: the RNN had to process tokens one at a time.
   Token 50 could not be computed until token 49 was done.
   No parallelism - slow training on long sequences.

2. Long-range dependencies: even with attention, the hidden state that "carries"
   information from token 1 to token 50 can get diluted through 49 matrix multiplications.

Let us see the long-range problem concretely before we fix it.
```

New text:

```markdown
## Beat 1 -- Section 1 - Why Transformers? The problem with a recurrent sequence model

As the recap noted, attention helps a decoder focus on relevant earlier states.
But if you keep a recurrent network around the outside, two problems remain:

1. Sequential processing: the recurrent network processes tokens one at a time.
   Token 50 cannot be computed until token 49 is done.
   No parallelism, so training is slow on long sequences.

2. Long-range dependencies: even with attention, the hidden state that "carries"
   information from token 1 to token 50 can get diluted through 49 matrix
   multiplications.

Let us see the long-range problem concretely before we fix it.
```

---

### Cell 11 - KEEP - code - Beat 1 RNN gradient dilution demo

Maps to original cell 8. KEEP unchanged (no course references; "Transformers" in the
comment is fine). Content unchanged:

```python
# Beat 1: Hidden state dilution in a simple RNN.
# We show that the gradient of the final hidden state w.r.t. an early token
# decays exponentially with distance - even with tanh activations.
#
# This is the "vanishing gradient" problem that motivated Transformers.

set_seeds(42)

# Simulate a one-layer vanilla RNN over a complaint sequence.
# W_h: hidden-to-hidden weight matrix
# The recurrence is h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)

seq_len = 30
d_h = 64

W_h = torch.randn(d_h, d_h) * 0.5  # small init to avoid explosion
W_x = torch.randn(d_h, d_h) * 0.1

# Gradient of h_T w.r.t. each h_t decays as ||W_h||^(T-t).
# We track the spectral norm as a proxy.
spectral_radius = torch.linalg.norm(W_h).item()

print("Approximate gradient norm ||dh_T / dh_t|| for a vanilla RNN")
print(f"(seq_len={seq_len}, hidden_dim={d_h}, ||W_h||={spectral_radius:.3f})")
print()
print(f"{'Token t':>8}  {'Grad norm':>14}  {'Status':>20}")
print("-" * 50)

for t in [0, 5, 10, 15, 20, 25, 29]:
    g = spectral_radius ** (seq_len - 1 - t)
    status = "STRONG" if g > 0.1 else ("weak" if g > 0.001 else "VANISHED")
    print(f"{t:>8}  {g:>14.6e}  {status:>20}")

print()
print("Token 0 (first word) barely reaches the final hidden state.")
print("Attention alone on the hidden state does not fix the underlying gradient decay.")
print("FIX: process all positions in PARALLEL with attention - no recurrence at all.")
```

---

### Cell 12 - KEEP - markdown - Beat 2 diagram (full Transformer architecture)

Maps to original cell 9. KEEP unchanged. Contains the `<!-- DIAGRAM: ... -->`
placeholder and the inline Mermaid `graph TD` of the full encoder-decoder
Transformer plus the figure caption. No course references inside it. Keep verbatim.

---

### Cell 13 - KEEP - code - Beat 3 self-attention parallel demo

Maps to original cell 10. KEEP unchanged. The demo builds a single-head
`nn.MultiheadAttention`, runs self-attention on an 8-token sequence, prints the
attention weight matrix. No course references. Keep verbatim.

---

### Cell 14 - KEEP - markdown - Beat 4 observe the attention pattern

Maps to original cell 11. KEEP unchanged. Discussion prompt, 3 minutes, three
questions about the attention matrix. No course references. Keep verbatim.

---

### Cell 15 - KEEP - markdown - Section 2 Positional Encoding intro

Maps to original cell 12. KEEP unchanged. Content:

```markdown
## Section 2 - Positional Encoding

Attention is order-agnostic: if you shuffle the input tokens, the attention scores
change but there is no positional bias built in. We need to inject position information
before the first attention layer.

The original Transformer used sinusoidal positional encodings - a deterministic
function that maps each (position, dimension) pair to a unique value. Let us build it.
```

---

### Cell 16 - KEEP - code - Beat 1 order-blind Transformer demo

Maps to original cell 13. KEEP unchanged. Builds an `nn.TransformerEncoderLayer`
with no positional encoding, shows reversed sequences give the same multiset of
output norms. No course references. Keep verbatim.

---

### Cell 17 - KEEP - markdown - Beat 2 diagram (sinusoidal PE properties)

Maps to original cell 14. KEEP unchanged. `<!-- DIAGRAM: ... -->` placeholder plus
Mermaid `graph TD` of the PE formula and its properties, plus figure caption.
No course references. Keep verbatim.

---

### Cell 18 - KEEP - code - Beat 3 positional_encoding demo

Maps to original cell 15. KEEP unchanged. Defines `positional_encoding(max_len,
d_model)`, verifies smoothness via cosine similarity, plots the encoding heatmap.
No course references. Keep verbatim.

---

### Cell 19 - KEEP - markdown - Lab 1 instructions (Tier 1 guided)

Maps to original cell 16. KEEP unchanged. STAR-framed Lab 1 instructions for
`MyPositionalEmbedding`, including Stretch and Homework Extension. No course
references. Keep verbatim.

---

### Cell 20 - KEEP - code - Lab 1 starter (EXERCISES) / FILL (SOLUTIONS)

Maps to original cell 17.

Exercises: KEEP the starter exactly as-is, with the two `None  # YOUR CODE` stubs.

Solutions: replace the two stubs with the working implementation.

Stub 1 line:
```python
        self.embedding = None  # YOUR CODE
```
Solutions replacement:
```python
        self.embedding = nn.Embedding(vocab_size, d_model)
```

Stub 2 line:
```python
        embedded = None  # YOUR CODE
```
Solutions replacement:
```python
        embedded = self.embedding(x) * math.sqrt(self.d_model) + self.pe[:seq_len]
```

Everything else in the cell (the docstrings, the shape-check block at the bottom)
stays identical in both notebooks.

---

### Cell 21 - KEEP - code - Lab 1 verification

Maps to original cell 18. KEEP unchanged in both Exercises and Solutions. It builds
a `_RefPE` reference and compares the student layer against it. Keep verbatim.

---

### Cell 22 - KEEP (Exercises) / DELETE (Solutions) - code - Lab 1 safety-net

Maps to original cell 19.

Exercises: KEEP unchanged. It defines `_RefPositionalEmbedding` and `_lab1_working`,
and sets `pos_embed_layer` from either the student layer or the reference.

Solutions: DELETE this cell. The Lab 1 cell is already the working implementation,
so the safety-net is redundant. After deletion the Solutions numbering shifts down
by one from here on.

---

### Cell 23 - KEEP - markdown - Beat 3 Section 3 intro

Maps to original cell 20. KEEP unchanged. Content:

```markdown
## Beat 3 -- Section 3 - Building the Transformer Block by Block


Now we build the components. We will follow the architecture diagram:
1. Multi-head self-attention with add + layer norm
2. Feed-forward network with add + layer norm
3. Encoder layer (self-attention + FFN)
4. Causal (masked) self-attention for the decoder
5. Cross-attention (encoder-decoder)
6. Decoder layer (masked self-attention + cross-attention + FFN)
7. Full Encoder and Decoder stacks
8. Translator model (Encoder + Decoder + output linear)

We build each component, test its shape, then connect them together.
```

---

### Cell 24 - KEEP - code - Beat 1 naive translator demo

Maps to original cell 21. KEEP unchanged. Naive embed -> linear translator, shows
reversed input gives the same sorted predictions. No course references. Keep verbatim.

---

### Cell 25 - KEEP - markdown - Beat 2 what a proper translator needs

Maps to original cell 22. KEEP unchanged. Explains positional information and token
interactions. No course references. Keep verbatim.

---

### Cell 26 - KEEP - code - Beat 3 encoder components

Maps to original cell 23. KEEP unchanged. Defines `PositionalEmbedding`,
`EncoderLayer`, `Encoder`, and runs the encoder shape verification. No course
references. Keep verbatim.

---

### Cell 27 - KEEP - code - Beat 3 decoder components and Translator

Maps to original cell 24. KEEP unchanged. Defines `DecoderLayer`, `Decoder`,
`Translator`, runs the full-model shape test. No course references. Keep verbatim.

---

### Cell 28 - KEEP - markdown - Discussion (encoder-decoder vs decoder-only)

Maps to original cell 25. KEEP unchanged. Three-minute discussion: decoder-only vs
encoder-decoder, heads vs layers, real-time vs batch deployment. No course
references. Keep verbatim.

---

### Cell 29 - KEEP - markdown - Lab 2 instructions (Tier 2 hard)

Maps to original cell 26. KEEP unchanged. STAR-framed Lab 2 instructions for
`build_decoder_layer`, including Stretch and Homework Extension. No course
references. Keep verbatim.

---

### Cell 30 - KEEP - code - Lab 2 starter (EXERCISES) / FILL (SOLUTIONS)

Maps to original cell 27.

Exercises: KEEP the starter exactly as-is, with all seven `None  # YOUR CODE` stubs.

Solutions: replace the stubs with the working implementation. Replacements:

Step 1 stub:
```python
            self.self_attn = None  # YOUR CODE
```
->
```python
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
```

Step 2 stub:
```python
            self.cross_attn = None  # YOUR CODE
```
->
```python
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
```

Step 3 stub:
```python
            self.ffn = None  # YOUR CODE
```
->
```python
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
```

Step 4 stubs (four lines):
```python
            self.norm1 = None  # YOUR CODE
            self.norm2 = None  # YOUR CODE
            self.norm3 = None  # YOUR CODE
            self.dropout = None  # YOUR CODE
```
->
```python
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
```

Step 5 stub:
```python
            tgt = None  # YOUR CODE
```
(the self-attention one) ->
```python
            sa_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
            tgt = self.norm1(tgt + self.dropout(sa_out))
```

Step 6 stub:
```python
            tgt = None  # YOUR CODE
```
(the cross-attention one) ->
```python
            ca_out, _ = self.cross_attn(tgt, memory, memory,
                                        key_padding_mask=memory_key_padding_mask)
            tgt = self.norm2(tgt + self.dropout(ca_out))
```

Step 7 stub:
```python
            tgt = None  # YOUR CODE
```
(the feed-forward one) ->
```python
            ff_out = self.ffn(tgt)
            tgt = self.norm3(tgt + self.dropout(ff_out))
```

The docstrings, comments, and the bottom shape-check block stay identical in both
notebooks.

---

### Cell 31 - KEEP - code - Lab 2 verification

Maps to original cell 28. KEEP unchanged in both notebooks. Builds a reference
`DecoderLayer`, loads its state dict into the student layer, compares outputs.
Keep verbatim.

---

### Cell 32 - KEEP (Exercises) / DELETE (Solutions) - code - Lab 2 safety-net

Maps to original cell 29.

Exercises: KEEP unchanged. Defines `_lab2_working` and prints which implementation
is in use; the full `model` is already defined in cell 27 with the reference
`DecoderLayer`, so nothing downstream breaks.

Solutions: DELETE this cell. The Lab 2 cell is already the working implementation.

---

### Cell 33 - EDIT - markdown - Section 4 capstone intro

Maps to original cell 30. Remove "first remote GPU training job in the course" and
"Topics 6-9" forward chaining.

Old text:

```
## Section 4 - Capstone: Remote GPU Training on SageMaker

You have built the Transformer architecture in this notebook kernel (CPU, ml.t3.medium).
Now we train it for real on a remote GPU instance.

This is the first remote GPU training job in the course. The pattern you see here
is exactly the same pattern you will use for fine-tuning LLMs in Topics 6-9.

What we will do:
1. Write a training script to `scripts_optional_transformers/train.py`
2. Submit a SageMaker PyTorch estimator job (ml.g4dn.xlarge, NVIDIA T4)
3. Wait for the job to complete and retrieve the trained model

The training data is a small synthetic Spanish-English complaint vocabulary
(real translation datasets are too large for a classroom GPU budget).
Job time: approximately 8-12 minutes.
```

New text:

```markdown
## Section 4 - Capstone: Remote GPU Training on SageMaker

You have built the Transformer architecture in this notebook kernel. Now we train
it for real on a remote GPU instance.

This capstone shows the standard SageMaker training pattern: package a training
script, submit a PyTorch estimator job, and retrieve the trained model artifact.
It is the same pattern used for fine-tuning models anywhere on SageMaker, so it is
worth seeing once end to end.

What we will do:
1. Write a training script to `scripts_optional_transformers/train.py`
2. Submit a SageMaker PyTorch estimator job (ml.g4dn.xlarge, NVIDIA T4)
3. Wait for the job to complete and retrieve the trained model

The training data is a small synthetic Spanish-English complaint vocabulary
(real translation datasets are too large for a classroom GPU budget).
Job time: approximately 8-12 minutes.
```

---

### Cell 34 - KEEP - code - write train.py and requirements.txt

Maps to original cell 31. KEEP unchanged. It creates the
`scripts_optional_transformers/` directory, writes `requirements.txt` (`numpy<2`)
and the full `train.py` (the `train_script` string with PositionalEmbedding,
EncoderLayer, Encoder, DecoderLayer, Decoder, Translator, make_batch, train, and
the argparse main). No course references in the script. Keep verbatim.

---

### Cell 35 - EDIT - code - submit the SageMaker training job

Maps to original cell 32. The current cell is INCONSISTENT: the comment and the
estimator both use `ml.m5.xlarge` (CPU), but the section header and brief call for
`ml.g4dn.xlarge` (GPU, NVIDIA T4). The rework brief is explicit: keep the remote GPU
capstone on `ml.g4dn.xlarge`. Fix this cell to use the GPU instance and drop the
"FIRST remote training job in the course" / "all capstones in Days 2 and 3" framing.

Old comment block:

```
# Submit the Translator training job to ml.m5.xlarge (CPU instance).
# This is the FIRST remote training job in the course.
# The pattern: source_dir with train.py + requirements.txt -> PyTorch estimator -> .fit()
# You will use this exact pattern for all capstones in Days 2 and 3.
#
# wait=False: launch the job and return immediately so we can continue teaching.
# Re-run the status cell below to check progress.
```

New comment block:

```
# Submit the Translator training job to ml.g4dn.xlarge (GPU instance, NVIDIA T4).
# The pattern: source_dir with train.py + requirements.txt -> PyTorch estimator -> .fit()
# This is the standard SageMaker training pattern, reusable for any model.
#
# wait=False: launch the job and return immediately so you can keep reading.
# Re-run the status cell below to check progress.
```

Old estimator line:

```python
    instance_type="ml.m5.xlarge",      # CPU instance, ~$0.23/hr
```

New estimator line:

```python
    instance_type="ml.g4dn.xlarge",    # GPU instance, NVIDIA T4
```

Old print line:

```python
print(f"Instance: ml.m5.xlarge (CPU, ~$0.23/hr)")
```

New print line:

```python
print(f"Instance: ml.g4dn.xlarge (GPU, NVIDIA T4)")
```

Everything else in the cell stays identical: the `PyTorch` estimator construction,
`output_path`, `base_job_name`, `hyperparameters` dict, `job_name` build,
`estimator.fit(wait=False, ...)`, the launched-job prints and the monitor URL.

NOTE: the `output_path` is `f"s3://{bucket}/topic4-translator/output"`. Leave the
S3 prefix `topic4-translator` as-is to avoid breaking anything; it is just a bucket
key, not student-facing text. (Optional: a builder may rename it to
`optional-transformers-translator` for cleanliness, but this is not required.)

---

### Cell 36 - KEEP - code - training-job safety-net (kernel restart recovery)

Maps to original cell 33. KEEP unchanged in BOTH notebooks. This is an operational
recovery cell (re-defines `training_job_name` if the kernel restarted), not a lab
safety-net, so it is NOT removed from the Solutions notebook. Content:

```python
# Safety-net: run this if your kernel restarted after launching the training job.
# SKIP this cell if training_job_name is already defined.
if 'training_job_name' not in dir() or training_job_name is None:
    training_job_name = "<PASTE YOUR JOB NAME HERE>"
    print(f"Using safety-net training_job_name: {training_job_name}")
```

---

### Cell 37 - EDIT - code - check training job status

Maps to original cell 34. The functional code is correct and KEEPS. Only fix the
misleading comment: the code already catches a broad `Exception`-style path via
`sm_client.exceptions.ResourceNotFound`, but the comment line "L7: use
ResourceNotFound (not ResourceNotFoundException)" is a stale internal note. Replace
that comment line; everything else stays.

Old comment lines:

```python
# Check training job status. Re-run this cell to refresh.
# L7: use ResourceNotFound (not ResourceNotFound Exception).
```

New comment lines:

```python
# Check training job status. Re-run this cell to refresh.
# describe_training_job raises if the job name is not found yet.
```

The rest of the cell (the `sm_client`, `get_job_status`, the status branching for
InProgress / Completed / Failed) stays identical. NOTE for the builder: if
`sm_client.exceptions.ResourceNotFound` raises an AttributeError at runtime in the
target boto3 version, fall back to a broad `except Exception:` in `get_job_status`.
This is a known footgun; prefer `except Exception:` returning `("NotFound", "")` if
in any doubt. Keep the rest of the branching logic unchanged.

---

### Cell 38 - EDIT - markdown - Wrap-Up

Maps to original cell 35. Remove all forward chaining: "In Topic 5 (Hugging Face)
you will...", "capstones in Days 2 and 3", "The DistilBERT model you load in Topic
5...", "Flan-T5 model in Topics 6 and 7", "Next session: Topic 5". Reframe as
optional-track closure.

Old text:

```
## Wrap-Up - What We Built Today

### Key takeaways

1. The Transformer solves the RNN bottleneck by replacing sequential processing with
   parallel attention. Every token attends to every other token simultaneously.

2. Positional encoding injects order information into an otherwise position-agnostic
   attention mechanism. Sinusoidal encodings generalise to unseen sequence lengths.

3. The encoder-decoder structure naturally separates understanding (encoder) from
   generation (decoder). Cross-attention is the bridge between them.

4. Your first remote GPU training job ran on SageMaker ml.g4dn.xlarge.
   The pattern: write scripts_topicN/train.py + requirements.txt, create a PyTorch
   estimator, call .fit(). This is the same pattern for all capstones in Days 2 and 3.

### What comes next

In Topic 5 (Hugging Face) you will stop building transformers from scratch and start
USING the transformers library. You will see that `BertModel`, `T5ForConditionalGeneration`,
and `GPT2LMHeadModel` are all variations on the architecture you built today.

### Homework Extensions

1. Add label smoothing (use `torch.nn.CrossEntropyLoss(label_smoothing=0.1)`) to the
   training script and re-submit the job. Does the loss curve change?

2. Implement greedy decoding: given a trained model and a source sequence, generate
   tokens one at a time until you hit an EOS token or max_len. What challenge arises
   when the source vocabulary does not have real words?

3. Research learning rate warmup as used in the original Transformer paper.
   Implement the warmup schedule and add it to the training script.

### The architecture you built IS the models you will use next

The DistilBERT model you load in Topic 5 is six stacked encoder blocks -- exactly the
encoder half of what you built today. When HuggingFace loads distilbert-base-uncased,
it loads your architecture, pre-trained on 100 million sentences from Wikipedia and
BookCorpus. The Flan-T5 model in Topics 6 and 7 is the full encoder-decoder you built,
pre-trained on 1 trillion tokens.

You did not build a translator. You built the understanding of what every model
you will use for the rest of this course looks like on the inside.


Next session: Topic 5 -- HuggingFace Ecosystem.
You built this architecture. Now you will download it pre-trained and run it in four lines.
```

New text:

```markdown
## Wrap-Up - What We Built

### Key takeaways

1. The Transformer removes the recurrent-network bottleneck by replacing sequential
   processing with parallel attention. Every token attends to every other token
   simultaneously.

2. Positional encoding injects order information into an otherwise position-agnostic
   attention mechanism. Sinusoidal encodings generalise to unseen sequence lengths.

3. The encoder-decoder structure naturally separates understanding (encoder) from
   generation (decoder). Cross-attention is the bridge between them.

4. You ran a remote GPU training job on SageMaker ml.g4dn.xlarge. The pattern is
   reusable: write a train.py plus requirements.txt, create a PyTorch estimator,
   call .fit().

### Homework Extensions

1. Add label smoothing (use `torch.nn.CrossEntropyLoss(label_smoothing=0.1)`) to the
   training script and re-submit the job. Does the loss curve change?

2. Implement greedy decoding: given a trained model and a source sequence, generate
   tokens one at a time until you hit an EOS token or max_len. What challenge arises
   when the source vocabulary does not have real words?

3. Research learning rate warmup as used in the original Transformer paper.
   Implement the warmup schedule and add it to the training script.

### Why this matters when you USE pre-trained models

The architecture you just built is not a toy. The popular pre-trained models are
exactly these blocks:

- A DistilBERT model is a stack of encoder blocks: the encoder half of what you
  built, pre-trained on a large text corpus.
- A T5 or Flan-T5 model is the full encoder-decoder you built, pre-trained at much
  larger scale.
- A GPT-style model is the decoder half, used on its own.

When you load one of these with the transformers library and call it in a few
lines, you now know what is happening inside. That is the payoff of this optional
deep-dive: not a translator you will ship, but the mental model to use, debug, and
choose pre-trained Transformers with confidence.
```

---

## Builder checklist

- [ ] Exercises notebook ends with 39 cells (0..38).
- [ ] Cell 0 is the optional/supplementary banner.
- [ ] No occurrence of "Topic 3a", "Topic 3b", "Topic 4", "Topic 5", "Topics 6",
      "Topic 6", "Topic 7", "Day 2", "Day 3", "YOU ARE HERE", "Next session"
      anywhere in any cell.
- [ ] Capstone consistently references `ml.g4dn.xlarge` in prose, comments, the
      estimator `instance_type`, and the status print.
- [ ] Plain ASCII only: no em-dash, en-dash, Unicode multiplication sign, emoji.
- [ ] Solutions notebook: Lab 1 (cell 20) and Lab 2 (cell 30) stubs filled with the
      implementations given above; lab safety-net cells (Exercises cells 22 and 32)
      DELETED; training-job safety-net (cell 36) KEPT. Solutions ends with 37 cells.
- [ ] Run `/validate-notebooks` after each 5-cell batch.
