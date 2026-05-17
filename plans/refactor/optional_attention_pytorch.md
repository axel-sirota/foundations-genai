# Rework Design Doc (R2): Optional Attention in PyTorch (standalone deep-dive)

> This is the SECOND, corrected version of this design doc. It resolves the blocking
> defects found by the Codex (o3) adversarial review in `CODEX_FINDINGS_R1.md`.
> Findings relevant to this notebook: R1, R2, R3, R6, R12. See the resolution map at
> the end of this document.

## Scope and intent

Notebook: `Exercises/topic_optional_attention_pytorch/topic_optional_attention_pytorch.ipynb`
Solutions twin: `Solutions/topic_optional_attention_pytorch/topic_optional_attention_pytorch.ipynb`

This notebook was formerly "Topic 3b - Attention in PyTorch", a required capstone that
sat between an old Topic 3a (NumPy attention) and an old Topic 4 (Transformers). The
course is being restructured. This notebook is now a STANDALONE OPTIONAL deep-dive.

Per Codex finding R12, the framing is precise: the attention CONCEPT is now taught in
the required course path as a concept-level mini-lesson. This optional notebook is the
from-scratch PyTorch BUILD of that concept. It is therefore correct to say BOTH "you
can complete the course without this notebook" (the concept is covered elsewhere) AND
"attention is the mechanism behind every model you use" (this notebook builds it). The
wording throughout reflects: concept = required mini-lesson, from-scratch build = this
optional notebook.

Hard requirements for this rework:

- A student picks this notebook up cold, opens it FIRST, with a fresh kernel.
- It cannot assume old Topic 3a, old Topic 4, or any other notebook ran.
- It cannot assume any artifact (`.npy`, `.pt`, embedding cache) produced by another
  notebook exists on disk (Codex R6).
- It must run start to finish on a plain CPU kernel with no network access (Codex R3).
- Every function and class the notebook calls must be DEFINED inside this notebook
  before it is used (Codex R1, R2).
- Course feedback said attention/transformer material was too math-heavy and that
  students are USERS first. This notebook stays optional precisely because it IS the
  internals, but it must still open by motivating WHY before any math.

This doc is a complete cell-by-cell plan. The notebook-building agent must be able to
implement the rework from this doc alone, without re-reading the original.

### Applies to BOTH notebooks

- Build the Exercise notebook first, then the Solution twin.
- The Solution twin is identical EXCEPT:
  - Every `# YOUR CODE` lab cell is replaced with a complete working implementation.
  - All safety-net cells are REMOVED from the Solution (the filled lab cell IS the
    solution there).
- All other cells (markdown, demos, diagrams, verification) are identical in both.

### Global authoring rules (CLAUDE.md)

- Plain ASCII only. No em-dashes, en-dashes, Unicode multiplication signs, emojis.
- Four-beat arc per concept: Beat 1 broken/naive code -> Beat 2 diagram placeholder
  -> Beat 3 working demo -> Beat 4 lab.
- Diagram convention: `<!-- DIAGRAM: ... -->` HTML comment followed by a mermaid
  fenced block and an italic caption.
- STAR framing for labs: Situation / Task / Action / Result.
- `numpy<2` pinned in the install cell.
- Safety-net cell immediately after any lab whose variable feeds a downstream cell
  (Exercise notebook only).
- `# YOUR CODE` placeholder lines must not hint the answer.
- No more than 3 consecutive markdown cells without a code cell between them.

## Pre-flight findings against the live notebook (Codex defect triage)

Before the cell plan, here is exactly how each relevant Codex finding maps to the
CURRENT state of `topic_optional_attention_pytorch.ipynb` (32 cells, indices 0-31):

- **R1 (CRITICAL).** The notebook never calls a bare top-level
  `scaled_dot_product_attention()` function: it defines class-based modules
  (`DotProductAttention`, `ScaledDotProductAttentionReference`,
  `ScaledDotProductAttention`) and calls `torch.nn.functional.scaled_dot_product_attention`.
  HOWEVER old cell 5 only RESTATES a `scaled_dot_product_attention(Q, K, V)` function
  in a prose code block (text inside markdown, never executed). A student who opens
  this notebook first and tries to call that function (or reads the recap expecting it
  to exist) hits a NameError, and the recap leans on "you implemented this in NumPy"
  which is false for a cold reader. FIX: add a REAL, runnable code cell that fully
  DEFINES a PyTorch `scaled_dot_product_attention(query, key, value)` function in this
  notebook (new cell 8), so the function the recap discusses actually exists in the
  kernel and every later section can lean on it.

- **R2.** The original told students (Discussion / capstone framing) to compare against
  `nn.MultiheadAttention` they "saw in Topic 5" - false, Topic 5 is now transfer
  learning and never shows it. `nn.MultiheadAttention` IS used later (old cell 29) but
  only as a Section 4 reference, AFTER the Discussion (old cell 27, item 3) and the
  capstone safety-net (old cell 22) already reference it. FIX: add a real, runnable
  code cell that INTRODUCES `torch.nn.MultiheadAttention` with a small worked example
  EARLY (new cell 9), before any lab, discussion, or safety-net mentions it. Section 4
  (now new cell 30) then becomes a deeper reference that reuses the same correct API.

- **R3.** Codex R3 flags both attention optionals for re-running `nltk.download()` for
  `word2vec_sample`. THIS notebook does NOT download anything: all embeddings are
  `torch.randn` (see old cells 9, 18, 26). So there is no live R3 break here. FIX is
  preventive: the build agent must NOT introduce any `nltk.download()` or other
  network asset while reworking; the recap and demos stay fully synthetic and offline.
  This is recorded explicitly so a future edit does not regress. (No code cell needed.)

- **R6.** The notebook must not `load()` artifacts from other notebooks, and must
  document any artifact it itself writes. Current behaviour: this notebook saves
  NOTHING to disk - the heatmaps are shown with `plt.show()`, never `plt.savefig`, and
  there is no `torch.save` / `np.save`. FIX: keep it that way and state it explicitly
  in the environment-setup comment and the banner so the contract is visible.

- **R12.** Covered in "Scope and intent" above and reflected in the banner (cell 0),
  the motivation (cell 1), and the wrap-up (cell 32).

## Summary of structural changes

Original had 32 cells (0-31). Reworked Exercise notebook has 35 cells (0-34). Net
changes:

- NEW cell 0: optional/supplementary banner (Codex R12 framing, R6 no-artifacts note).
- NEW cell 1: "Why attention matters to a user" motivation before any math.
- NEW cell 8: REAL code cell defining the PyTorch `scaled_dot_product_attention`
  function (Codex R1). Placed right after the NumPy recap so the recap's function
  actually exists in the kernel.
- NEW cell 9: REAL code cell introducing `torch.nn.MultiheadAttention` with a small
  worked example (Codex R2). Placed before any lab, discussion, or safety-net that
  mentions it.
- Old cell 0 (title/objectives) becomes cell 2, fully rewritten: no "Topic 3b",
  no "Capstone", no "you implemented in Topic 3a".
- Old cell 5 "Bridging from Topic 3a" becomes cell 7, rewritten into a self-contained
  NumPy recap (restates scaled dot product attention inline so the PyTorch port makes
  sense with zero prior topic).
- All "Topic 3a", "Topic 3b", "Topic 3 capstone", "Topic 4", "Topic 5" references
  removed throughout.
- Capstone safety-net cells (old 22 and 24) are buggy: old cell 22 calls
  `ScaledDotProductAttention(embed_dim=8, num_heads=2)` and falls back to an
  `nn.MultiheadAttention` wrapper with a `forward(self, x)` signature; old cell 24
  replaces the class with `nn.MultiheadAttention` too. Both mismatch the actual
  `__init__(dropout_p=0.0)` / `forward(query, key, value) -> (output, weights)` API
  used everywhere else. MERGE into a single correct safety-net (new cell 27) and
  DELETE the duplicate. The correct safety-net falls back to
  `ScaledDotProductAttentionReference`, which has the matching API.
- Old cell 30 "Wrap-Up" rewritten: drop the "across Topic 3" table and the "coming in
  Topic 4" section; reframe as a self-contained recap plus an optional onward-pointer.
- Old cell 31 footer rewritten: remove "End of Topic 3b" and "Next session: Topic 4".
- Section 4 (multi-head, old cells 28-29) KEPT as a self-contained reference treatment,
  reworded so multi-head is "covered further in the optional transformers deep-dive",
  never a mandatory next topic. Its code cell now reuses the `nn.MultiheadAttention`
  API introduced in new cell 9, so the API is consistent across the notebook.

Cell-count map (old -> new). New cells are marked NEW; the rest carry an old index:

| Old | New | Action |
|-----|-----|--------|
| -   | 0   | NEW banner |
| -   | 1   | NEW motivation |
| 0   | 2   | EDIT (rewrite title/objectives) |
| 1   | 3   | KEEP (TF backend disable) |
| 2   | 4   | EDIT (install comment wording) |
| 3   | 5   | KEEP (SageMaker session) |
| 4   | 6   | KEEP (imports, seeds, COMPLAINT_TOKENS) |
| 5   | 7   | EDIT (rewrite into self-contained NumPy recap) |
| -   | 8   | NEW code: define PyTorch `scaled_dot_product_attention` (R1) |
| -   | 9   | NEW code: introduce `nn.MultiheadAttention` worked example (R2) |
| 6   | 10  | KEEP (Beat 1: unscaled saturates) |
| 7   | 11  | KEEP (Section 1 header) |
| 8   | 12  | EDIT (diagram caption wording) |
| 9   | 13  | KEEP (DotProductAttention demo) |
| 10  | 14  | KEEP (Lab 1 instructions) |
| 11  | 15  | LAB CELL (Lab 1 starter) |
| 12  | 16  | KEEP (Lab 1 verification) |
| 13  | 17  | SAFETY-NET (Lab 1 safety-net, Exercise only) |
| 14  | 18  | KEEP (Stretch + Homework) |
| 15  | 19  | EDIT (Section 2 header: drop "Tier 3 capstone") |
| 16  | 20  | KEEP (Beat 1 resolved: scaled vs unscaled) |
| 17  | 21  | KEEP (Beat 2 diagram: heatmap) |
| 18  | 22  | EDIT (reference scaled attention: tiny comment dechain) |
| 19  | 23  | EDIT (Discussion: items 2 and 3 dechain) |
| 20  | 24  | EDIT (rewrite open-ended lab framing) |
| 21  | 25  | LAB CELL (open-ended lab stub) |
| 22  | -   | DELETE (buggy duplicate safety-net) |
| 24  | 26  | (relocated) -> see cell 27 below; old 24 is the surviving safety-net base |
| -   | 27  | MERGE of old 22 + old 24 into ONE correct safety-net, placed BEFORE verification |
| 23  | 28  | KEEP (open-ended lab verification), now AFTER the safety-net |
| 25  | 29  | EDIT (Section 3 intro wording) |
| 26  | 30  | KEEP (applied complaint triage demo) |
| 27  | 31  | EDIT (Discussion: item 3 dechain) |
| 28  | 32  | EDIT (Section 4 intro: remove "Topic 4") |
| 29  | 33  | EDIT (multi-head reference demo: dechain, reuse cell 9 API) |
| 30  | 34  | EDIT (rewrite Wrap-Up) |
| 31  | 35  | EDIT (rewrite Homework footer) |

Ordering note: in the original, the capstone safety-net (old cell 24) sat AFTER the
verification (old cell 23). That is wrong: the verification needs a usable
`ScaledDotProductAttention` to run. In this rework the merged safety-net is new cell 27
and the verification is new cell 28, so the safety-net runs FIRST.

Final Exercise cell count: 36 cells, indices 0-35.
Final Solution cell count: 34 cells (the two safety-net cells, new 17 and new 27, are
removed; the filled lab cells are the solution).

---

## CELL-BY-CELL PLAN

### Cell 0 - markdown - NEW - Optional/supplementary banner

```markdown
# Optional Deep-Dive: Attention in PyTorch

> This is an optional, supplementary notebook. The main course path does not
> require it. The attention concept itself is taught as a short mini-lesson in the
> required notebooks. This deep-dive is where you BUILD that concept from scratch.
> You can complete the course without ever opening this notebook.

## Who this is for

This deep-dive is for developers who want to see what is actually happening inside
the attention mechanism that powers every modern language model. Most of the time you
will USE attention through a library call and never touch the internals. That is fine.
But if you are the kind of engineer who wants to open the box, this notebook builds
the core attention operation from scratch in PyTorch and lets you verify it against
PyTorch's own built-in function.

## This notebook is self-contained

You do not need to have done any other notebook first. Everything you need is recapped
here, including the math, with a short NumPy refresher before the PyTorch port. Run the
cells in order from top to bottom and it will work on a plain CPU kernel.

It needs no network access: every demo uses synthetic tensors, nothing is downloaded.
It also writes nothing to disk: all visualisations are shown inline, so running it
leaves no artifact files behind and depends on no artifact from any other notebook.

## What you get out of it

- A precise mental model of scaled dot product attention.
- A standalone `scaled_dot_product_attention` function and a reusable `nn.Module`
  you implement and verify yourself.
- A first look at PyTorch's own `nn.MultiheadAttention`.
- An interpretability heatmap that shows what attention "looks at".
```

### Cell 1 - markdown - NEW - Why attention matters to a user

```markdown
## Why attention matters (before any math)

You probably already use models built on attention every day: chat assistants,
autocomplete, translation, summarisation. You do not need to know how attention works
to use those tools. So why look inside?

Three practical reasons a working developer cares:

1. Cost and limits. Attention is the reason long inputs get expensive fast. When you
   see "context window" limits or rising token bills, attention is the mechanism
   underneath. Understanding it tells you why a 100-token prompt and a 10000-token
   prompt are not the same price.

2. Explainability. Attention produces a weight for every pair of tokens. In a
   regulated setting (for example, a bank flagging a complaint as high-severity) you
   may need to show WHY a model focused on certain words. Attention weights are one of
   the few windows into that.

3. Debugging and tuning. When a model "ignores" part of an input, or fixates on the
   wrong token, the attention pattern is where you look first.

The intuition in one sentence: attention lets every token decide how much to listen
to every other token, instead of being forced to compress everything into one fixed
summary. The rest of this notebook makes that sentence precise and runnable.

We use a running example throughout: a Barclays customer support system triaging
complaint messages. The tokens are complaint words like "unauthorised", "charge",
"fraud", "refund". You will watch attention decide which of those words listen to
which.
```

### Cell 2 - markdown - EDIT (was old cell 0) - Title and objectives

Replace the entire old cell. The old cell:

> # Topic 3b - Attention in PyTorch
> **Barclays Customer Support Intelligence System | Topic 3b (Capstone)**
> ## What you will build
> In Topic 3a you implemented scaled dot product attention from scratch in NumPy.
> In this notebook you will: ...
> ## Capstone lab (Tier 3 - open-ended)
> Implement `ScaledDotProductAttention(nn.Module)` - signature and docstring only.
> No step-by-step scaffold. You know the math from Topic 3a.
> ## Learning objectives ...

New content:

```markdown
## What you will build in this notebook

Running example: the Barclays Customer Support Intelligence System, triaging
complaint messages.

In this notebook you will:
1. Recap scaled dot product attention in plain NumPy (a short, self-contained refresher)
2. Define scaled dot product attention as a plain PyTorch function from the recap
3. Meet PyTorch's own `nn.MultiheadAttention` with a small worked example
4. Port dot product attention to a PyTorch `nn.Module` with autograd
5. Implement scaled dot product attention as an `nn.Module` from scratch
6. Verify your implementation against `torch.nn.functional.scaled_dot_product_attention`
7. Apply your attention module to a complaint triage task and visualise attention weights

## The two labs

- Lab 1 (Tier 1, guided): implement `DotProductAttention` as an `nn.Module` with
  numbered steps and a verification cell. 15-20 minutes.
- Open-ended lab (Tier 3): implement `ScaledDotProductAttention(nn.Module)` from
  scratch. You get the signature and docstring only, no step-by-step scaffold. The
  math is fully recapped earlier in this notebook, so this is self-contained even if
  this is the first attention material you have seen. 25-35 minutes.

## Learning objectives

1. Translate a NumPy attention implementation to PyTorch with gradient support
2. Understand how PyTorch handles batch dimensions and broadcasting automatically
3. Implement scaled dot product attention without scaffolding
4. Verify a custom PyTorch implementation against a reference library function
5. Read and use PyTorch's built-in `nn.MultiheadAttention`
6. Interpret attention weight heatmaps over complaint tokens
```

Rationale: removes "Topic 3b", "(Capstone)", "In Topic 3a you implemented", and
"You know the math from Topic 3a". Keeps the Tier 3 difficulty level but reframes it as
"the open-ended lab" rather than "the Topic 3 capstone".

### Cell 3 - code - KEEP (was old cell 1) - Disable TensorFlow backend

No change. Content:

```python
# Disable TensorFlow backend in transformers (SageMaker image compatibility).
# Must run before any transformers import.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

### Cell 4 - code - EDIT (was old cell 2) - Environment setup

Keep the pip install exactly. EDIT only the leading comment.

Old comment lines:
> # Environment setup for SageMaker Studio
> # All attention demos run in this kernel - no remote training jobs.

New content (the `!pip install` block and the final `print` line are unchanged):

```python
# Environment setup for SageMaker Studio.
# This is a self-contained, optional deep-dive: all attention demos run in this
# kernel on CPU. No remote training jobs. No network downloads. No prior-notebook
# artifacts are read, and this notebook writes none to disk.

!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "numpy<2" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0"

print("RESTART KERNEL before continuing -- environment packages were installed/upgraded.")
```

### Cell 5 - code - KEEP (was old cell 3) - SageMaker session

No change. Content:

```python
import sagemaker
from sagemaker import get_execution_role
import boto3

sess = sagemaker.Session()
role = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"Role: {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
```

### Cell 6 - code - KEEP (was old cell 4) - Imports, seeds, complaint vocabulary

No change. Defines `set_seeds`, `device`, and `COMPLAINT_TOKENS`. Content unchanged
from the original cell 4 (the imports block ending with the
`print(f"Complaint vocabulary for demos: {COMPLAINT_TOKENS}")` line).

### Cell 7 - markdown - EDIT (was old cell 5) - Self-contained NumPy recap

This is the central de-chaining change. The old cell said "In Topic 3a you implemented
this function in NumPy" and assumed prior work. Rewrite it into a standalone recap that
actually teaches the math inline. Note: the next cell (new cell 8) turns this recap
into a REAL runnable PyTorch function, so this markdown cell now ends by pointing at
that, not at a prior topic.

Old cell:

> ## Bridging from Topic 3a
> In Topic 3a you implemented this function in NumPy: ...
> In this notebook we port the SAME logic to PyTorch as an `nn.Module`. ...

New content:

```markdown
## A self-contained recap: scaled dot product attention

This notebook is standalone, so here is the whole idea in one place before we touch
PyTorch. If you have seen attention before, this is a quick refresher. If you have not,
read it slowly once and the PyTorch code afterwards will be obvious.

Attention takes three matrices:

- Q (queries): for each position, "what am I looking for?"
- K (keys): for each position, "what do I offer?"
- V (values): for each position, "what do I actually contribute if attended to?"

The computation, step by step:

1. Score: compare every query to every key with a dot product. A large dot product
   means the query and key are aligned, so that key is relevant.
2. Scale: divide the scores by `sqrt(d_k)`, where `d_k` is the key dimension. Without
   this, large dimensions push scores so far apart that the next step saturates. We
   demonstrate this failure concretely a few cells below.
3. Softmax: turn each row of scores into weights that are positive and sum to 1.
   Each query now has a probability distribution over all keys.
4. Weighted sum: multiply the weights by V. Each query position gets a blend of
   values, weighted by relevance.

The same idea in plain NumPy, so you can see it with no framework magic:

    import numpy as np

    def softmax_np(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def scaled_dot_product_attention_np(Q, K, V):
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
        attention_weights = softmax_np(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

That formula, `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`, is the entire
mechanism. The next cell turns exactly this into a real, runnable PyTorch function so
the rest of the notebook can call it. The equations do not change. What changes:

- `np.matmul` becomes `torch.matmul` (or the `@` operator)
- the NumPy softmax becomes `F.softmax(..., dim=-1)`
- autograd computes gradients automatically, so there is no backprop code to write
- GPU support is free: just move the tensors to a device

After that we start with plain dot product attention (the unscaled version) as a
warm-up, then build the scaled version up as an `nn.Module`.
```

### Cell 8 - code - NEW - Define the PyTorch `scaled_dot_product_attention` function (Codex R1)

This cell did not exist. It RESOLVES Codex R1. The original notebook only restated this
function as prose inside a markdown cell, and the recap referred to a NumPy version "you
implemented in Topic 3a". A student opening this notebook first therefore has no callable
`scaled_dot_product_attention` in the kernel. This cell defines a real, runnable PyTorch
version so the recap is backed by working code and the function name resolves.

Identical in BOTH Exercise and Solution notebooks (it is a demo, not a lab).

```python
# The recap above, now as a real runnable PyTorch function.
# This is the PyTorch port of the NumPy formula. Everything later in this
# notebook leans on this exact operation: softmax(Q K^T / sqrt(d_k)) V.
#
# Defining it here means a student who opens THIS notebook first has a working
# scaled_dot_product_attention in the kernel - no NameError, no prior notebook needed.

def scaled_dot_product_attention(query, key, value):
    """
    Scaled dot product attention, plain function form.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Args:
        query: (batch, T_q, d_k)
        key:   (batch, T_k, d_k)
        value: (batch, T_k, d_v)

    Returns:
        output:            (batch, T_q, d_v)
        attention_weights: (batch, T_q, T_k)  - each row sums to 1.0
    """
    d_k = query.shape[-1]
    # Step 1: raw dot product scores -> (batch, T_q, T_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    # Step 2: scale by sqrt(d_k) to keep softmax out of saturation
    scores = scores / (d_k ** 0.5)
    # Step 3: softmax over the key dimension
    attention_weights = F.softmax(scores, dim=-1)
    # Step 4: weighted sum of values
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# Quick check that the function is real and runnable.
set_seeds(42)
_demo_q = torch.randn(2, 4, 16)
_demo_out, _demo_w = scaled_dot_product_attention(_demo_q, _demo_q, _demo_q)
print("scaled_dot_product_attention is defined and runs.")
print(f"  output shape:            {_demo_out.shape}   -> (batch=2, T_q=4, d_v=16)")
print(f"  attention weights shape: {_demo_w.shape}   -> (batch=2, T_q=4, T_k=4)")
print(f"  row sums (must be 1.0):  {_demo_w[0].sum(dim=-1).detach().numpy().round(4)}")

# Sanity check against PyTorch's own built-in (PyTorch 2.0+), if available.
try:
    _builtin, _ = F.scaled_dot_product_attention(_demo_q, _demo_q, _demo_q), None
    print(f"  matches torch built-in:  "
          f"{torch.allclose(_demo_out, F.scaled_dot_product_attention(_demo_q, _demo_q, _demo_q), atol=1e-4)}")
except AttributeError:
    print("  torch built-in scaled_dot_product_attention not available (PyTorch < 2.0).")
```

### Cell 9 - code - NEW - Introduce `nn.MultiheadAttention` (Codex R2)

This cell did not exist. It RESOLVES Codex R2. The original notebook referenced
`nn.MultiheadAttention` in a Discussion item and in a buggy safety-net BEFORE it was
ever shown, and a removed line told students they "saw it in Topic 5" (false). This
cell introduces `torch.nn.MultiheadAttention` with a small worked example up front, so
every later mention (Discussion, Section 4, safety-net) has something concrete and
correct to point back to.

Identical in BOTH Exercise and Solution notebooks (it is a demo, not a lab).

```python
# Meet PyTorch's built-in nn.MultiheadAttention.
# You will build single-head attention from scratch later in this notebook.
# This small example shows the library module you will compare against, so the
# comparison later is grounded in something you have actually run here.
#
# nn.MultiheadAttention runs several scaled dot product attentions ("heads") in
# parallel, each with its own learned Q, K, V projection, then concatenates them.

set_seeds(42)

mha_intro_embed_dim = 16   # embedding size; must be divisible by num_heads
mha_intro_num_heads = 2    # 2 parallel heads, each of size 16 // 2 = 8

# batch_first=True means inputs are (batch, seq, embed_dim).
intro_mha = nn.MultiheadAttention(
    embed_dim=mha_intro_embed_dim,
    num_heads=mha_intro_num_heads,
    dropout=0.0,
    batch_first=True,
)

# A tiny self-attention example: 1 sequence, 4 tokens.
intro_x = torch.randn(1, 4, mha_intro_embed_dim)

# nn.MultiheadAttention is called as module(query, key, value).
# It returns (attn_output, attn_output_weights). need_weights=True returns the
# per-pair attention weights, averaged across heads.
intro_out, intro_weights = intro_mha(intro_x, intro_x, intro_x, need_weights=True)

print("nn.MultiheadAttention - introductory worked example")
print("=" * 52)
print(f"embed_dim={mha_intro_embed_dim}, num_heads={mha_intro_num_heads}, "
      f"d_per_head={mha_intro_embed_dim // mha_intro_num_heads}")
print(f"Input shape:             {intro_x.shape}      -> (batch=1, tokens=4, embed=16)")
print(f"Output shape:            {intro_out.shape}      -> same shape as input")
print(f"Attention weights shape: {intro_weights.shape}       -> (batch, T_q, T_k), averaged over heads")
print(f"Row sums of weights[0]:  {intro_weights[0].sum(dim=-1).detach().numpy().round(4)}")
print()
print("Call pattern to remember: module(query, key, value) -> (output, weights).")
print("This is the same (query, key, value) -> (output, weights) shape you will give")
print("your own ScaledDotProductAttention module later in this notebook.")
```

### Cell 10 - code - KEEP (was old cell 6) - Beat 1: unscaled attention saturates

No change. The `unscaled_attention_forward` function plus the d_k sweep loop printing
gradient norms. Content unchanged from original cell 6.

### Cell 11 - markdown - KEEP (was old cell 7) - Section 1 header

No change. Content:

```markdown
## Section 1 - Dot Product Attention in PyTorch

The warm-up: implement dot product attention as an `nn.Module`.
This is the unscaled version. We will add scaling in Section 2.

Notice how PyTorch's autograd handles the backward pass automatically -
no need to write gradient code by hand.
```

### Cell 12 - markdown - EDIT (was old cell 8) - Beat 2 diagram

Keep the `<!-- DIAGRAM: ... -->` comment and the full mermaid block unchanged. EDIT
only the italic caption, which references Topic 3a.

Old caption:
> *Figure: The diagram shows the same computation you implemented in Topic 3a, now with
> PyTorch tensor shapes. The `sqrt(d_k)` scaling (highlighted) is the only difference
> from plain dot product attention.*

New caption:

```markdown
*Figure: the diagram shows the same computation from the NumPy recap above, now drawn
with PyTorch tensor shapes. The `sqrt(d_k)` scaling (highlighted) is the only
difference from plain dot product attention.*
```

### Cell 13 - code - KEEP (was old cell 9) - Beat 3: DotProductAttention demo

No change. The complete `DotProductAttention(nn.Module)` class plus the complaint-domain
demo. Content unchanged from original cell 9.

### Cell 14 - markdown - KEEP (was old cell 10) - Lab 1 instructions

No change. STAR-framed Lab 1 (Tier 1 guided). Content unchanged from original cell 10.
(Contains no topic-chaining language; it references only "the demo above".)

### Cell 15 - code - LAB CELL (was old cell 11) - Lab 1 starter

EXERCISE notebook: KEEP unchanged. `MyDotProductAttention` with three
`scores = None  # YOUR CODE` style stubs and a sanity check.

SOLUTION notebook: replace the three stubs with the working implementation:

```python
        # Step 1: Compute raw dot product scores.
        # Shape: (batch, T_q, d_k) x (batch, d_k, T_k) -> (batch, T_q, T_k)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Step 2: Softmax over key positions.
        # Use F.softmax with the correct dim argument.
        attention_weights = F.softmax(scores, dim=-1)

        # Step 3: Weighted sum of values.
        # Shape: (batch, T_q, T_k) x (batch, T_k, d_v) -> (batch, T_q, d_v)
        context = torch.matmul(attention_weights, value)
```

The rest of the cell (class shell, docstring, sanity check) is identical in both.

### Cell 16 - code - KEEP (was old cell 12) - Lab 1 verification

No change in either notebook. The verification block comparing `MyDotProductAttention`
to `DotProductAttention`. Content unchanged from original cell 12.

### Cell 17 - code - SAFETY-NET (was old cell 13) - Lab 1 safety-net

EXERCISE notebook: KEEP unchanged. The `if 'MyDotProductAttention' not in dir(): ...`
fallback that aliases `MyDotProductAttention = DotProductAttention`.

SOLUTION notebook: DELETE this cell entirely (the filled Lab 1 cell is the solution).

### Cell 18 - markdown - KEEP (was old cell 14) - Stretch + Homework Extension

No change. The optional `mask` parameter stretch and the built-in comparison homework.
Content unchanged from original cell 14. (No topic-chaining language present.)

### Cell 19 - markdown - EDIT (was old cell 15) - Section 2 header

Old cell:
> ## Section 2 - Scaled Dot Product Attention in PyTorch
> You implemented the unscaled version. Now we add the `sqrt(d_k)` scaling.
> This is the EXACT operation at the core of every Transformer model.
> After this section you will implement it yourself as the Tier 3 capstone.

New content:

```markdown
## Section 2 - Scaled Dot Product Attention in PyTorch

You implemented the unscaled version. Now we add the `sqrt(d_k)` scaling.

This is the exact operation at the core of every Transformer model. You already saw it
as a plain function in the recap section. Here we build it up as an `nn.Module`, and
after this section you will implement that module yourself in the open-ended (Tier 3)
lab.
```

Rationale: replaces "the Tier 3 capstone" with "the open-ended (Tier 3) lab" so there
is no implied position in a topic sequence, and points back to the new cell 8 function.

### Cell 20 - code - KEEP (was old cell 16) - Beat 1 resolved: scaled vs unscaled

No change. The `scaled_attention_forward` function and the side-by-side gradient-norm
comparison at d_k=512. Content unchanged from original cell 16.

### Cell 21 - markdown - KEEP (was old cell 17) - Beat 2 diagram: attention heatmap

No change. The `<!-- DIAGRAM: ... -->` heatmap placeholder, mermaid block, and caption.
Content unchanged from original cell 17. (Caption mentions only "training", no topic
chaining.)

### Cell 22 - code - EDIT (was old cell 18) - Beat 3: reference scaled attention

Keep the entire `ScaledDotProductAttentionReference(nn.Module)` class, the complaint
self-attention demo, the built-in comparison, and the heatmap. EDIT only the leading
comment, which says "you will implement yourself in the Tier 3 capstone lab".

Old comment lines:
> # Beat 3: Reference implementation of scaled dot product attention.
> # This is what you will implement yourself in the Tier 3 capstone lab.
> # Study this carefully before starting the capstone.

New comment lines:

```python
# Beat 3: Reference implementation of scaled dot product attention as an nn.Module.
# This is what you will implement yourself in the open-ended (Tier 3) lab below.
# Study this carefully before starting that lab.
```

Everything else in the cell stays unchanged. The docstring line "the fundamental
operation of the Transformer (Vaswani et al., 2017)" is a factual citation, not a topic
reference - KEEP it.

### Cell 23 - markdown - EDIT (was old cell 19) - Discussion (3 minutes)

Keep discussion item 1 unchanged. EDIT items 2 and 3.

Old item 2:
> 2. In Topic 3a we used word2vec embeddings as the Q, K, V inputs directly.
>    In a real Transformer, Q, K, V are created by projecting the input embeddings
>    through three separate learned weight matrices (W_Q, W_K, W_V).
>    Why have separate projections? What would break if W_Q = W_K = W_V = the identity matrix?

New item 2:

```markdown
2. In the demos so far we fed embeddings in as Q, K, V directly. In a real Transformer,
   Q, K, V are created by projecting the input embeddings through three separate
   learned weight matrices (W_Q, W_K, W_V). This is exactly what `nn.MultiheadAttention`
   does internally - you saw its parameter count in the worked example earlier. Why
   have separate projections? What would break if W_Q = W_K = W_V = the identity matrix?
```

Old item 3 first sentence:
> 3. The capstone asks you to implement `ScaledDotProductAttention` from scratch.

New item 3 first sentence (the rest of item 3 is unchanged):

```markdown
3. The open-ended lab asks you to implement `ScaledDotProductAttention` from scratch.
```

### Cell 24 - markdown - EDIT (was old cell 20) - Open-ended lab instructions

Rewrite the framing. Old cell header and intro:
> ## Capstone Lab - Implement Scaled Dot Product Attention from Scratch (Tier 3 - Open-Ended)
> **Time**: 25-35 minutes | **Tier**: 3 (open-ended - function signature + docstring only)
> This is the Topic 3 capstone. You know the math from Topic 3a. You have studied the
> reference implementation above. Now implement it yourself with NO scaffold.

New header and intro (the STAR sub-sections Situation/Task/Action/Result and the
Stretch/Homework lines below them stay unchanged):

```markdown
## Open-Ended Lab - Implement Scaled Dot Product Attention from Scratch (Tier 3)

**Time**: 25-35 minutes | **Tier**: 3 (open-ended - function signature + docstring only)

This is the main exercise of the deep-dive. The math is fully recapped earlier in this
notebook (see the NumPy recap, the `scaled_dot_product_attention` function, and
Section 2), and you have just studied the reference module above. Now implement it
yourself as an `nn.Module` with no scaffold.
```

Keep the rest of the cell verbatim: the "### Situation", "### Task", "### Action",
"### Result" blocks, the "**Stretch**" paragraph, and the "**Homework Extension**"
paragraph. None of those contain topic-chaining language.

### Cell 25 - code - LAB CELL (was old cell 21) - Open-ended lab stub

EXERCISE notebook: KEEP unchanged. The `ScaledDotProductAttention(nn.Module)` class
with the full docstring and a bare `pass` body (no `__init__`, no `forward` - Tier 3).

SOLUTION notebook: replace the `pass` with a complete implementation. Keep the class
name and the entire docstring identical, then add:

```python
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
```

This matches `ScaledDotProductAttentionReference` (and the new cell 8 function) so the
verification cell passes.

### Cell 26 - DELETED - old cell 22 (buggy duplicate safety-net)

DELETE old cell 22 entirely. It called `ScaledDotProductAttention(embed_dim=8,
num_heads=2)` and fell back to an `nn.MultiheadAttention` wrapper with a
`forward(self, x)` signature - a different constructor and a different forward
signature than the real `__init__(dropout_p=0.0)` / `forward(query, key, value)` API.
That fallback would break every downstream cell. Its correct intent is absorbed into
the single merged safety-net (new cell 27).

### Cell 27 - code - SAFETY-NET (MERGE of old cells 22 and 24) - Open-ended lab safety-net

The original had TWO safety-net cells for this lab and both were buggy:

- Old cell 22 called `ScaledDotProductAttention(embed_dim=8, num_heads=2)` and fell
  back to an `nn.MultiheadAttention` wrapper with `forward(self, x)`.
- Old cell 24 fell back to `ScaledDotProductAttention = ScaledDotProductAttentionReference`
  (correct fallback target) but probed with `ScaledDotProductAttention(dropout_p=0.0)`
  and was placed AFTER the verification cell, so the verification could not use it.

MERGE into ONE correct safety-net. The fallback target is
`ScaledDotProductAttentionReference`, whose API is `__init__(dropout_p=0.0)` and
`forward(query, key, value) -> (output, weights)` - the exact API used everywhere else
in this notebook, including the new cell 9 `nn.MultiheadAttention` call pattern
`module(query, key, value) -> (output, weights)`. The `nn.MultiheadAttention` fallback
from old cell 22 is dropped because its signature does not match. Place this safety-net
BEFORE the verification cell so the verification can actually run.

EXERCISE notebook: single safety-net cell:

```python
# Open-ended lab safety-net: run this if you did not finish the lab above.
# SKIP this cell if your ScaledDotProductAttention class is complete and working.
#
# The fallback is ScaledDotProductAttentionReference, which has the same API your
# class is meant to have: __init__(dropout_p=0.0) and
# forward(query, key, value) -> (output, attention_weights).
_need_safety_net = False
try:
    _m = ScaledDotProductAttention(dropout_p=0.0)
    _q = torch.randn(1, 4, 16)
    _out, _w = _m(_q, _q, _q)
    if _out is None or _w is None:
        _need_safety_net = True
except Exception:
    _need_safety_net = True

if _need_safety_net:
    print("Using the open-ended lab safety-net so the rest of the notebook can run.")
    ScaledDotProductAttention = ScaledDotProductAttentionReference
else:
    print("ScaledDotProductAttention looks complete - safety-net not needed.")
```

SOLUTION notebook: DELETE this cell entirely.

### Cell 28 - code - KEEP (was old cell 23) - Open-ended lab verification

No change in either notebook. The five-check verification block (output shape, weights
shape, weights sum to 1, numerical match with reference, gradient flow, plus the
built-in comparison bonus). Content unchanged from original cell 23.

Placement note: in the final notebook this verification is cell 28, AFTER the merged
safety-net (cell 27). This fixes the original ordering bug where the safety-net sat
after the verification.

### Cell 29 - markdown - EDIT (was old cell 25) - Section 3 intro

Old cell:
> ## Section 3 - Applying Your Attention Module to Complaint Triage
> You have a working `ScaledDotProductAttention` module.
> Let us use it in a minimal complaint triage model and visualise the learned attention pattern.
> This is NOT a trained model - we use structured embeddings to simulate semantic proximity.
> But the architecture shows how attention would fit into a production complaints routing system.

New content (light cosmetic edit; no topic-chaining language was present):

```markdown
## Section 3 - Applying Your Attention Module to Complaint Triage

You have a working `ScaledDotProductAttention` module. Let us use it in a minimal
complaint triage model and visualise the attention pattern.

This is not a trained model: we use structured embeddings to simulate semantic
proximity. Even so, the architecture shows how attention would fit into a production
complaints routing system.
```

### Cell 30 - code - KEEP (was old cell 26) - Applied complaint triage demo

No change. `make_complaint_embeddings`, the self-attention call, the Reds heatmap, and
the intra-cluster vs cross-cluster check. Content unchanged from original cell 26.
Embeddings are synthetic `torch.randn`-based - no download, consistent with Codex R3.

### Cell 31 - markdown - EDIT (was old cell 27) - Discussion (3 minutes)

Keep discussion items 1 and 2 unchanged. EDIT item 3 only for consistency with the new
cell 9 introduction (no topic reference was present; this just makes the cross-ref real).

Old item 3:
> 3. `nn.MultiheadAttention` runs 4-8 attention heads in parallel. If each head
>    specialises in different relationships (one head for fraud keywords, another for
>    account action keywords), what does the final output capture that a single-head
>    model would miss?

New item 3:

```markdown
3. `nn.MultiheadAttention` (which you met in the worked example near the top of this
   notebook) runs several attention heads in parallel. If each head specialises in
   different relationships (one head for fraud keywords, another for account-action
   keywords), what does the final output capture that a single-head model would miss?
```

### Cell 32 - markdown - EDIT (was old cell 28) - Section 4 intro: multi-head

Old cell:
> ## Section 4 - Multi-Head Attention: Reference Only
> Multi-head attention runs several scaled dot product attentions in parallel,
> each with its own Q, K, V projections, then concatenates the results.
> You do not need to implement this today. The reference below shows how
> `ScaledDotProductAttention` composes into `nn.MultiheadAttention`.
> We will build the full multi-head attention module in Topic 4 (Transformers).

New content:

```markdown
## Section 4 - Multi-Head Attention: A Short Reference

You already met `nn.MultiheadAttention` in a small worked example at the top of this
notebook. This section is a slightly deeper look. Multi-head attention runs several
scaled dot product attentions in parallel, each with its own Q, K, V projections, then
concatenates the results. Intuitively, each head can specialise: one head might track
fraud keywords, another might track account-action keywords, and the concatenation
gives the model all of those views at once.

You do not need to implement multi-head attention to finish this notebook. The
reference cell below scales the same `nn.MultiheadAttention` module up and reads its
parameter count, so you can see how a single `ScaledDotProductAttention` head composes
into a full multi-head layer. Building a full multi-head layer from scratch is covered
in the optional transformers deep-dive; it is not a required next step here.
```

Rationale: removes "We will build the full multi-head attention module in Topic 4
(Transformers)". Reframes multi-head as covered in an optional transformers deep-dive,
explicitly "not a required next step". Cross-references the new cell 9 introduction so
the two `nn.MultiheadAttention` cells are coherent.

### Cell 33 - code - EDIT (was old cell 29) - nn.MultiheadAttention reference demo

Keep the entire demo. EDIT the leading comment and the final print so they no longer
reference Topic 4, and so the API matches the introduction in new cell 9 (same
`batch_first=True`, same `module(query, key, value)` call pattern - it already does).

Old comment lines near the top:
> # Reference: nn.MultiheadAttention using PyTorch's built-in module.
> # We will implement multi-head attention from scratch in Topic 4 (Transformers).
> # For now, verify it works and understand the parameter count.

New comment lines:

```python
# Reference: nn.MultiheadAttention using PyTorch's built-in module.
# Same module and same module(query, key, value) call pattern you met in the
# introductory example earlier, now with more heads so you can read the parameter
# count. Building multi-head attention from scratch is covered in the optional
# transformers deep-dive.
```

Old final print line:
> print(f"In Topic 4 you will build this from {num_heads} x ScaledDotProductAttention heads.")

New final print lines:

```python
print(f"This module is {num_heads} x ScaledDotProductAttention heads plus the Q, K, V")
print("and output projection matrices, all wrapped in one PyTorch module.")
```

Everything else in the cell (the `mha` construction with `embed_dim`, `num_heads`,
`batch_first=True`, the shapes, the parameter breakdown) stays unchanged. The construction
already uses the same keyword API as the new cell 9 introduction, so the two are
consistent.

### Cell 34 - markdown - EDIT (was old cell 30) - Wrap-Up

The old wrap-up has a "What you built across Topic 3" table (with old-3a rows) and a
"What is coming in Topic 4 - Transformers" section. Both must go. Rewrite as a
self-contained recap.

Old cell content (to be fully replaced):
> ## Wrap-Up
> ### What you built across Topic 3
> | Topic | What you implemented | ...
> ### Key principles to carry forward ...
> ### What is coming in Topic 4 - Transformers ...

New content:

```markdown
## Wrap-Up

### What you built in this deep-dive

| Step | What you implemented or saw |
|------|------------------------------|
| Recap | Scaled dot product attention math, then a runnable PyTorch function for it |
| Worked example | PyTorch's built-in `nn.MultiheadAttention` on a tiny sequence |
| Lab 1 | `DotProductAttention` as an `nn.Module` in PyTorch |
| Open-ended lab | `ScaledDotProductAttention` as an `nn.Module`, verified against PyTorch's built-in |
| Applied demo | Self-attention over complaint tokens with a heatmap visualisation |
| Reference | How a single head composes into a full `nn.MultiheadAttention` layer |

### Key principles to carry forward

1. Scaled dot product attention, `softmax(Q K^T / sqrt(d_k)) V`, is the core operation
   of every modern Transformer model. The scaling by `1/sqrt(d_k)` prevents gradient
   saturation at large embedding dimensions - you saw this fail and then get fixed.

2. Attention is largely parameter-free: the operation itself has no weights. The
   learned weights live in the Q, K, V projection matrices, which is what
   `nn.MultiheadAttention` adds on top.

3. Attention weights are interpretable: visualise them as a heatmap to see what the
   model focuses on. This matters for financial AI, where explainability is a
   requirement, not a nice-to-have.

4. Cost scales with the square of sequence length: the attention matrix is
   `T_q` by `T_k`. This is why long contexts get expensive - exactly the user-facing
   concern from the opening motivation.

### Where to go next (optional)

This was an optional deep-dive, so there is no required next notebook. The attention
concept itself is covered in the required course path; this notebook was the
from-scratch build. If you want to keep going inside the box, the optional transformers
deep-dive builds a full multi-head attention layer, positional encoding, and a complete
Transformer encoder on top of the `ScaledDotProductAttention` module you implemented
here.
```

### Cell 35 - markdown - EDIT (was old cell 31) - Homework Extensions

Keep Homework 1 and Homework 2 (both are self-contained code exercises with no topic
references). EDIT only the footer lines.

Old footer:
> *End of Topic 3b - Attention in PyTorch*
> Next session: Topic 4 - Transformers + Translator Capstone.

New footer:

```markdown
*End of the optional deep-dive: Attention in PyTorch.*

You have built and verified the core attention operation from scratch. The main course
path does not depend on this notebook, so you can return to wherever you were.
```

---

## Build checklist for the implementing agent

1. EXERCISE notebook: apply every cell above. Cells marked KEEP are byte-identical to
   the original (use the old-cell mapping in the change table). Cells marked EDIT use
   the new content shown. NEW cells (0, 1, 8, 9) are added with the content shown.
   DELETE old cell 22 (the buggy duplicate safety-net). The merged safety-net (new
   cell 27) is placed BEFORE the verification (new cell 28). Final Exercise cell count:
   36 (indices 0-35).
2. Validate the Exercise notebook (`/validate-notebooks`), confirm it runs top to
   bottom on CPU with NO network access, and that the verification cells print FAIL for
   the empty stubs (Lab 1 stubs return None, open-ended lab `pass` body) until labs are
   completed. Confirm new cell 8 prints that `scaled_dot_product_attention` is defined,
   and new cell 9 prints the `nn.MultiheadAttention` shapes.
3. SOLUTION notebook: copy the finished Exercise notebook, then:
   - Fill cell 15 (Lab 1) with the working forward body shown.
   - Fill cell 25 (open-ended lab) with the `__init__` and `forward` shown.
   - DELETE the two safety-net cells (Exercise cell 17 and Exercise cell 27).
   - Solution cell count is therefore 34, not 36.
4. Validate the Solution notebook: it must run top to bottom with every verification
   cell printing all PASS.
5. Grep both notebooks for: "Topic 3a", "Topic 3b", "Topic 3", "Topic 4", "Topic 5",
   "capstone", "Capstone", "Next session", "Bridging", "you saw in Topic". Zero matches
   expected (the Vaswani 2017 citation is fine and is not a topic reference).
6. Confirm no `nltk.download`, no other network call, no `torch.save` / `np.save` /
   `plt.savefig`, and no `load()` of an external artifact are present (Codex R3, R6).
7. Confirm plain ASCII only: no em-dashes, en-dashes, Unicode multiplication signs,
   emojis.

---

## Codex R1 findings resolved

| Finding | Resolution | Fixing cell(s) |
|---------|-----------|----------------|
| R1 - `scaled_dot_product_attention()` only existed as prose in the python attention notebook, so opening this notebook first risks a NameError and the recap leaned on a non-existent prior NumPy function | NEW code cell that fully DEFINES a real, runnable PyTorch `scaled_dot_product_attention(query, key, value)` function in THIS notebook, with a self-check and a built-in comparison; the recap markdown is rewritten to point at it instead of "Topic 3a" | NEW cell 8 (defines the function); cell 7 (recap rewritten to introduce it); cell 19, cell 24 (point back to it) |
| R2 - the notebook told students to compare against `nn.MultiheadAttention` "you saw in Topic 5" (false) and referenced it in a Discussion item and safety-net before it was ever shown | NEW code cell that INTRODUCES `torch.nn.MultiheadAttention` with a small runnable worked example (shapes, call pattern, weights) BEFORE any lab, discussion, or safety-net mentions it; Discussion item (cell 23 item 2, cell 31 item 3) and Section 4 (cells 32, 33) rewritten to cross-reference this introduction; the merged safety-net (cell 27) uses an API consistent with it | NEW cell 9 (introduces `nn.MultiheadAttention`); cells 23, 31, 32, 33 (consistent cross-references); cell 27 (consistent API) |
| R3 - both attention optionals re-run `nltk.download()` for `word2vec_sample`, breaking offline | Verified this notebook downloads NOTHING: all embeddings are synthetic `torch.randn`. Fix is preventive and documented: the build agent must not introduce any download; the env-setup comment and banner state the no-network contract; build checklist step 6 greps for `nltk.download` | cell 0 (banner no-network statement); cell 4 (env comment); cell 30 (synthetic embeddings kept); build checklist step 6 |
| R6 - optional notebooks save artifacts that required notebooks then `load()` | Verified this notebook saves NOTHING (no `torch.save` / `np.save` / `plt.savefig`) and loads no external artifact; this is now stated explicitly so the contract is visible and a future edit cannot regress it | cell 0 (banner no-artifact statement); cell 4 (env comment); build checklist step 6 |
| R12 - narrative tension between "transformers are everywhere" and "optional notebook" | Framed explicitly: the attention CONCEPT is taught as a required mini-lesson elsewhere; THIS notebook is the optional from-scratch BUILD. Both statements are now consistent | "Scope and intent"; cell 0 (banner); cell 1 (motivation); cell 34 (wrap-up "Where to go next") |
