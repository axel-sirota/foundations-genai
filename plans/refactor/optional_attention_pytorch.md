# Rework Design Doc: Optional Attention in PyTorch (standalone deep-dive)

## Scope and intent

Notebook: `Exercises/topic_optional_attention_pytorch/topic_optional_attention_pytorch.ipynb`
Solutions twin: `Solutions/topic_optional_attention_pytorch/topic_optional_attention_pytorch.ipynb`

This notebook was formerly "Topic 3b - Attention in PyTorch", a required capstone that
sat between an old Topic 3a (NumPy attention) and an old Topic 4 (Transformers). The
course is being restructured. This notebook is now a STANDALONE OPTIONAL deep-dive:

- A student picks it up cold. It cannot assume old-3a was done.
- It cannot chain forward into old-4.
- It must run start to finish on its own.
- Course feedback said attention/transformer material was too math-heavy and that
  students are USERS first. This notebook stays optional precisely because it IS the
  internals, but it must still open by motivating WHY before any math.

This doc is a complete cell-by-cell plan. A separate notebook-building agent must be
able to implement the rework from this doc alone, without re-reading the original.

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

## Summary of structural changes

Original had 32 cells (0-31). Reworked notebook has 33 cells (0-32). Net changes:

- NEW cell 0: optional/supplementary banner (was not present).
- NEW cell 1: "Why attention matters to a user" motivation before any math.
- Old cell 0 (title/objectives) becomes cell 2, fully rewritten: no "Topic 3b",
  no "Capstone", no "you implemented in Topic 3a".
- Old cell 5 "Bridging from Topic 3a" becomes cell 7, rewritten into a self-contained
  NumPy recap (restates scaled dot product attention inline so the PyTorch port makes
  sense with zero prior topic).
- All "Topic 3a", "Topic 3 capstone", "Topic 4" references removed throughout.
- Capstone safety-net cells (old 22 and 24) are buggy: old cell 22 calls
  `ScaledDotProductAttention(embed_dim=8, num_heads=2)` and old cell 24 replaces the
  class with `nn.MultiheadAttention`, both of which mismatch the actual
  `(query, key, value) -> (output, weights)` API used everywhere else. MERGE into a
  single correct safety-net (new cell 24) and DELETE the duplicate.
- Old cell 30 "Wrap-Up" rewritten: drop the "across Topic 3" table and the "coming in
  Topic 4" section; reframe as a self-contained recap plus an optional onward-pointer.
- Old cell 31 footer rewritten: remove "End of Topic 3b" and "Next session: Topic 4".
- Section 4 (multi-head, old cells 28-29) KEPT as a self-contained reference treatment,
  reworded so multi-head is "covered further in the optional transformers deep-dive",
  never a mandatory next topic.

Cell-count map (old -> new):

| Old | New | Action |
|-----|-----|--------|
| -   | 0   | NEW banner |
| -   | 1   | NEW motivation |
| 0   | 2   | EDIT (rewrite title/objectives) |
| 1   | 3   | KEEP |
| 2   | 4   | EDIT (install comment wording) |
| 3   | 5   | KEEP |
| 4   | 6   | KEEP |
| 5   | 7   | EDIT (rewrite into self-contained NumPy recap) |
| 6   | 8   | KEEP |
| 7   | 9   | KEEP |
| 8   | 10  | EDIT (caption wording) |
| 9   | 11  | KEEP |
| 10  | 12  | KEEP |
| 11  | 13  | KEEP |
| 12  | 14  | KEEP |
| 13  | 15  | KEEP (safety-net, Exercise only) |
| 14  | 16  | KEEP |
| 15  | 17  | EDIT (remove "Tier 3 capstone" forward-ref wording) |
| 16  | 18  | KEEP |
| 17  | 19  | KEEP |
| 18  | 20  | KEEP |
| 19  | 21  | EDIT (Discussion item 2 "In Topic 3a") |
| 20  | 22  | EDIT (rewrite capstone framing) |
| 21  | 23  | KEEP (capstone lab stub) |
| 22  | -   | DELETE (buggy duplicate safety-net) |
| 23  | 24b | KEEP as verification (renumber) |
| 24  | 24  | EDIT into single correct safety-net, placed BEFORE verification |
| 25  | 25  | EDIT (Section 3 intro wording) |
| 26  | 26  | KEEP |
| 27  | 27  | KEEP |
| 28  | 28  | EDIT (Section 4 intro: remove "Topic 4") |
| 29  | 29  | EDIT (remove "Topic 4" lines) |
| 30  | 30  | EDIT (rewrite Wrap-Up) |
| 31  | 31  | EDIT (rewrite Homework footer) |

Note on ordering: in the original the capstone safety-net (old 24) sits AFTER the
verification (old 23). That is wrong: the verification needs a usable
`ScaledDotProductAttention` to run. In the rework the safety-net (new cell 24) goes
BEFORE the verification (new cell 24b is actually placed as cell 25 in final order).
Final clean ordering for that block is given explicitly in the cell list below
(cells 23, 24, 25).

---

## CELL-BY-CELL PLAN

### Cell 0 - markdown - NEW - Optional/supplementary banner

```markdown
# Optional Deep-Dive: Attention in PyTorch

> **This is an optional, supplementary notebook.** The main course path does not
> require it. You can complete the course without ever opening this notebook.

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

## What you get out of it

- A precise mental model of scaled dot product attention.
- A reusable `nn.Module` you implement and verify yourself.
- An interpretability heatmap that shows what attention "looks at".
```

### Cell 1 - markdown - NEW - Why attention matters to a user

```markdown
## Why attention matters (before any math)

You probably already use models built on attention every day: chat assistants,
autocomplete, translation, summarisation. You do not need to know how attention works
to use those tools. So why look inside?

Three practical reasons a working developer cares:

1. **Cost and limits.** Attention is the reason long inputs get expensive fast. When
   you see "context window" limits or rising token bills, attention is the mechanism
   underneath. Understanding it tells you why a 100-token prompt and a 10000-token
   prompt are not the same price.

2. **Explainability.** Attention produces a weight for every pair of tokens. In a
   regulated setting (for example, a bank flagging a complaint as high-severity) you
   may need to show WHY a model focused on certain words. Attention weights are one of
   the few windows into that.

3. **Debugging and tuning.** When a model "ignores" part of an input, or fixates on
   the wrong token, the attention pattern is where you look first.

The intuition in one sentence: **attention lets every token decide how much to listen
to every other token, instead of being forced to compress everything into one fixed
summary.** The rest of this notebook makes that sentence precise and runnable.

We use a running example throughout: a Barclays customer support system triaging
complaint messages. The tokens are complaint words like "unauthorised", "charge",
"fraud", "refund". You will watch attention decide which of those words listen to
which.
```

### Cell 2 - markdown - EDIT (was old cell 0) - Title and objectives

Rewrite. The old cell:

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
2. Port dot product attention to a PyTorch `nn.Module` with autograd
3. Implement scaled dot product attention in PyTorch from scratch
4. Verify your implementation against `torch.nn.functional.scaled_dot_product_attention`
5. Apply your attention module to a complaint triage task and visualise attention weights

## The two labs

- **Lab 1 (Tier 1, guided)**: implement `DotProductAttention` as an `nn.Module` with
  numbered steps and a verification cell. 15-20 minutes.
- **Open-ended lab (Tier 3)**: implement `ScaledDotProductAttention(nn.Module)` from
  scratch. You get the signature and docstring only, no step-by-step scaffold. The
  math is fully recapped earlier in this notebook, so this is self-contained even if
  this is the first attention material you have seen. 25-35 minutes.

## Learning objectives

1. Translate a NumPy attention implementation to an `nn.Module` with gradient support
2. Understand how PyTorch handles batch dimensions and broadcasting automatically
3. Implement scaled dot product attention without scaffolding
4. Verify a custom PyTorch implementation against a reference library function
5. Interpret attention weight heatmaps over complaint tokens
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

Old comment line 2:
> # All attention demos run in this kernel - no remote training jobs.

New comment block:

```python
# Environment setup for SageMaker Studio.
# This is a self-contained, optional deep-dive: all attention demos run in this
# kernel on CPU. No remote training jobs, no prior-notebook artifacts required.

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
actually teaches the math inline.

Old cell:

> ## Bridging from Topic 3a
> In Topic 3a you implemented this function in NumPy: ...
> In this notebook we port the SAME logic to PyTorch as an `nn.Module`. ...

New content:

```markdown
## A self-contained recap: scaled dot product attention in NumPy

This notebook is standalone, so here is the whole idea in one place before we touch
PyTorch. If you have seen attention before, this is a quick refresher. If you have not,
read it slowly once and the PyTorch code afterwards will be obvious.

Attention takes three matrices:

- **Q (queries)**: for each position, "what am I looking for?"
- **K (keys)**: for each position, "what do I offer?"
- **V (values)**: for each position, "what do I actually contribute if attended to?"

The computation, step by step:

1. **Score**: compare every query to every key with a dot product. A large dot product
   means the query and key are aligned, so that key is relevant.
2. **Scale**: divide the scores by `sqrt(d_k)`, where `d_k` is the key dimension. Without
   this, large dimensions push scores so far apart that the next step saturates. We
   demonstrate this failure concretely in the next cell.
3. **Softmax**: turn each row of scores into weights that are positive and sum to 1.
   Each query now has a probability distribution over all keys.
4. **Weighted sum**: multiply the weights by V. Each query position gets a blend of
   values, weighted by relevance.

The whole thing in plain NumPy:

    import numpy as np

    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def scaled_dot_product_attention(Q, K, V):
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

That formula, `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`, is the entire
mechanism. The rest of this notebook ports it to PyTorch. The equations do not change.
What changes:

- `np.matmul` becomes `torch.matmul` (or the `@` operator)
- the NumPy softmax becomes `F.softmax(..., dim=-1)`
- autograd computes gradients automatically, so there is no backprop code to write
- GPU support is free: just move the tensors to a device

We start with plain dot product attention (the unscaled version) as a warm-up, then
add the `sqrt(d_k)` scaling.
```

### Cell 8 - code - KEEP (was old cell 6) - Beat 1: unscaled attention saturates

No change. The `unscaled_attention_forward` function plus the d_k sweep loop printing
gradient norms. Content unchanged from original cell 6.

### Cell 9 - markdown - KEEP (was old cell 7) - Section 1 header

No change. Content:

```markdown
## Section 1 - Dot Product Attention in PyTorch

The warm-up: implement dot product attention as an `nn.Module`.
This is the unscaled version. We will add scaling in Section 2.

Notice how PyTorch's autograd handles the backward pass automatically -
no need to write gradient code by hand.
```

### Cell 10 - markdown - EDIT (was old cell 8) - Beat 2 diagram

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

### Cell 11 - code - KEEP (was old cell 9) - Beat 3: DotProductAttention demo

No change. The complete `DotProductAttention(nn.Module)` class plus the complaint-domain
demo. Content unchanged from original cell 9.

### Cell 12 - markdown - KEEP (was old cell 10) - Lab 1 instructions

No change. STAR-framed Lab 1 (Tier 1 guided). Content unchanged from original cell 10.
(Contains no topic-chaining language; it references only "the demo above".)

### Cell 13 - code - LAB CELL (was old cell 11) - Lab 1 starter

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

### Cell 14 - code - KEEP (was old cell 12) - Lab 1 verification

No change in either notebook. The verification block comparing `MyDotProductAttention`
to `DotProductAttention`. Content unchanged from original cell 12.

### Cell 15 - code - SAFETY-NET (was old cell 13) - Lab 1 safety-net

EXERCISE notebook: KEEP unchanged. The `if 'MyDotProductAttention' not in dir(): ...`
fallback that aliases `MyDotProductAttention = DotProductAttention`.

SOLUTION notebook: DELETE this cell entirely (the filled Lab 1 cell is the solution).

### Cell 16 - markdown - KEEP (was old cell 14) - Stretch + Homework Extension

No change. The optional `mask` parameter stretch and the built-in comparison homework.
Content unchanged from original cell 14. (No topic-chaining language present.)

### Cell 17 - markdown - EDIT (was old cell 15) - Section 2 header

Old cell:
> ## Section 2 - Scaled Dot Product Attention in PyTorch
> You implemented the unscaled version. Now we add the `sqrt(d_k)` scaling.
> This is the EXACT operation at the core of every Transformer model.
> After this section you will implement it yourself as the Tier 3 capstone.

New content:

```markdown
## Section 2 - Scaled Dot Product Attention in PyTorch

You implemented the unscaled version. Now we add the `sqrt(d_k)` scaling.

This is the exact operation at the core of every Transformer model. After this section
you will implement it yourself in the open-ended (Tier 3) lab.
```

Rationale: replaces "the Tier 3 capstone" with "the open-ended (Tier 3) lab" so there
is no implied position in a topic sequence.

### Cell 18 - code - KEEP (was old cell 16) - Beat 1 resolved: scaled vs unscaled

No change. The `scaled_attention_forward` function and the side-by-side gradient-norm
comparison at d_k=512. Content unchanged from original cell 16.

### Cell 19 - markdown - KEEP (was old cell 17) - Beat 2 diagram: attention heatmap

No change. The `<!-- DIAGRAM: ... -->` heatmap placeholder, mermaid block, and caption.
Content unchanged from original cell 17. (Caption mentions only "training", no topic
chaining.)

### Cell 20 - code - KEEP (was old cell 18) - Beat 3: reference scaled attention

No change. `ScaledDotProductAttentionReference(nn.Module)` plus the complaint
self-attention demo, the built-in comparison, and the heatmap. Content unchanged from
original cell 18.

Note: the docstring contains "the fundamental operation of the Transformer (Vaswani et
al., 2017)" which is a factual citation, not a topic reference. KEEP it.

### Cell 21 - markdown - EDIT (was old cell 19) - Discussion (3 minutes)

Keep discussion items 1 and 3 unchanged. EDIT item 2, which says "In Topic 3a we used
word2vec embeddings".

Old item 2:
> 2. In Topic 3a we used word2vec embeddings as the Q, K, V inputs directly.
>    In a real Transformer, Q, K, V are created by projecting the input embeddings
>    through three separate learned weight matrices (W_Q, W_K, W_V).
>    Why have separate projections? What would break if W_Q = W_K = W_V = the identity matrix?

New item 2:

```markdown
2. In the demos so far we fed embeddings in as Q, K, V directly. In a real Transformer,
   Q, K, V are created by projecting the input embeddings through three separate
   learned weight matrices (W_Q, W_K, W_V). Why have separate projections? What would
   break if W_Q = W_K = W_V = the identity matrix?
```

Also in item 3, "The capstone asks you to implement" -> "The open-ended lab asks you
to implement". Old item 3 first sentence:
> 3. The capstone asks you to implement `ScaledDotProductAttention` from scratch.

New item 3 first sentence:

```markdown
3. The open-ended lab asks you to implement `ScaledDotProductAttention` from scratch.
```

### Cell 22 - markdown - EDIT (was old cell 20) - Open-ended lab instructions

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
notebook (see the NumPy recap and Section 2), and you have just studied the reference
implementation above. Now implement it yourself with no scaffold.
```

Keep the rest of the cell verbatim: the "### Situation", "### Task", "### Action",
"### Result" blocks, the "**Stretch**" paragraph, and the "**Homework Extension**"
paragraph. None of those contain topic-chaining language.

### Cell 23 - code - LAB CELL (was old cell 21) - Open-ended lab stub

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

This matches `ScaledDotProductAttentionReference` so the verification cell passes.

### Cell 24 - code - SAFETY-NET (MERGE of old cells 22 and 24) - Capstone safety-net

The original had TWO safety-net cells for this lab and both were buggy:

- Old cell 22 called `ScaledDotProductAttention(embed_dim=8, num_heads=2)` and fell
  back to an `nn.MultiheadAttention` wrapper with a `forward(self, x)` signature.
- Old cell 24 fell back to `ScaledDotProductAttention = ScaledDotProductAttentionReference`.

Both probed a different API than the real one (`dropout_p=0.0` constructor,
`forward(query, key, value)`). The `nn.MultiheadAttention` fallback in old cell 22
would break every downstream cell. MERGE into ONE correct safety-net and place it
BEFORE the verification cell so the verification can actually run.

EXERCISE notebook: single safety-net cell:

```python
# Open-ended lab safety-net: run this if you did not finish the lab above.
# SKIP this cell if your ScaledDotProductAttention class is complete and working.
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

### Cell 25 - code - KEEP (was old cell 23) - Open-ended lab verification

No change in either notebook. The five-check verification block (output shape, weights
shape, weights sum to 1, numerical match with reference, gradient flow, plus the
built-in comparison bonus). Content unchanged from original cell 23.

Placement note: in the final notebook this verification is cell 25, AFTER the merged
safety-net (cell 24). This fixes the original ordering bug where the safety-net sat
after the verification.

### Cell 26 - markdown - EDIT (was old cell 25) - Section 3 intro

Old cell:
> ## Section 3 - Applying Your Attention Module to Complaint Triage
> You have a working `ScaledDotProductAttention` module.
> Let us use it in a minimal complaint triage model and visualise the learned attention pattern.
> This is NOT a trained model - we use structured embeddings to simulate semantic proximity.
> But the architecture shows how attention would fit into a production complaints routing system.

New content (only the title number changes, no chaining language was present, so this
is a light EDIT for consistency):

```markdown
## Section 3 - Applying Your Attention Module to Complaint Triage

You have a working `ScaledDotProductAttention` module. Let us use it in a minimal
complaint triage model and visualise the attention pattern.

This is not a trained model: we use structured embeddings to simulate semantic
proximity. Even so, the architecture shows how attention would fit into a production
complaints routing system.
```

(If the build agent prefers, this cell may instead be marked KEEP verbatim - it
contains no topic-chaining. The edit above is purely cosmetic.)

### Cell 27 - code - KEEP (was old cell 26) - Applied complaint triage demo

No change. `make_complaint_embeddings`, the self-attention call, the Reds heatmap, and
the intra-cluster vs cross-cluster check. Content unchanged from original cell 26.

### Cell 28 - markdown - KEEP (was old cell 27) - Discussion (3 minutes)

No change. Three discussion items on sequence-length scaling, attention as explanation,
and multi-head specialisation. Content unchanged from original cell 27. (Item 3
mentions `nn.MultiheadAttention` generically, no topic reference.)

### Cell 29 - markdown - EDIT (was old cell 28) - Section 4 intro: multi-head

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

Multi-head attention runs several scaled dot product attentions in parallel, each with
its own Q, K, V projections, then concatenates the results. Intuitively, each head can
specialise: one head might track fraud keywords, another might track account-action
keywords, and the concatenation gives the model all of those views at once.

You do not need to implement multi-head attention to finish this notebook. The
reference cell below shows how the `ScaledDotProductAttention` you just built composes
into PyTorch's own `nn.MultiheadAttention`, and what it costs in parameters. Building a
full multi-head layer from scratch is covered in the optional transformers deep-dive;
it is not a required next step here.
```

Rationale: removes "We will build the full multi-head attention module in Topic 4
(Transformers)". Reframes multi-head as covered in an optional transformers deep-dive,
explicitly "not a required next step". Adds a one-sentence intuition so the section
still motivates WHY before the reference code.

### Cell 30 - code - EDIT (was old cell 29) - nn.MultiheadAttention reference demo

Keep the entire demo. EDIT only the two lines that reference Topic 4.

Old comment line near the top:
> # We will implement multi-head attention from scratch in Topic 4 (Transformers).

New:
```python
# Building multi-head attention from scratch is covered in the optional transformers
# deep-dive. Here we just verify PyTorch's built-in module and read its parameter count.
```

Old final print line:
> print(f"In Topic 4 you will build this from {num_heads} x ScaledDotProductAttention heads.")

New:
```python
print(f"This module is {num_heads} x ScaledDotProductAttention heads plus the Q, K, V")
print("and output projection matrices, all wrapped in one PyTorch module.")
```

Everything else in the cell (the `mha` construction, shapes, parameter breakdown)
stays unchanged.

### Cell 31 - markdown - EDIT (was old cell 30) - Wrap-Up

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

| Step | What you implemented |
|------|----------------------|
| Recap | Scaled dot product attention in NumPy (read-only refresher) |
| Lab 1 | `DotProductAttention` as an `nn.Module` in PyTorch |
| Open-ended lab | `ScaledDotProductAttention` as an `nn.Module`, verified against PyTorch's built-in |
| Applied demo | Self-attention over complaint tokens with a heatmap visualisation |
| Reference | How a single head composes into `nn.MultiheadAttention` |

### Key principles to carry forward

1. Scaled dot product attention, `softmax(Q K^T / sqrt(d_k)) V`, is the core operation
   of every modern Transformer model. The scaling by `1/sqrt(d_k)` prevents gradient
   saturation at large embedding dimensions - you saw this fail and then get fixed.

2. Attention is largely parameter-free: the module itself has no weights. The learned
   weights live in the Q, K, V projection matrices outside the module.

3. Attention weights are interpretable: visualise them as a heatmap to see what the
   model focuses on. This matters for financial AI, where explainability is a
   requirement, not a nice-to-have.

4. Cost scales with the square of sequence length: the attention matrix is
   `T_q` by `T_k`. This is why long contexts get expensive - exactly the user-facing
   concern from the opening motivation.

### Where to go next (optional)

This was an optional deep-dive, so there is no required next notebook. If you want to
keep going inside the box, the optional transformers deep-dive builds a full multi-head
attention layer, positional encoding, and a complete Transformer encoder on top of the
`ScaledDotProductAttention` module you implemented here.
```

### Cell 32 - markdown - EDIT (was old cell 31) - Homework Extensions

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
   the new content shown. NEW cells are added with the content shown. DELETE old cell
   22 (the buggy duplicate safety-net). The merged safety-net (new cell 24) is placed
   BEFORE the verification (new cell 25).
2. Validate the Exercise notebook (`/validate-notebooks`), confirm it runs top to
   bottom on CPU and that the verification cells print FAIL for the empty stubs (Lab 1
   stubs return None, open-ended lab `pass` body) until labs are completed.
3. SOLUTION notebook: copy the finished Exercise notebook, then:
   - Fill cell 13 (Lab 1) with the working forward body shown.
   - Fill cell 23 (open-ended lab) with the `__init__` and `forward` shown.
   - DELETE the two safety-net cells (Exercise cell 15 and Exercise cell 24).
   - Solution cell count is therefore 31, not 33.
4. Validate the Solution notebook: it must run top to bottom with every verification
   cell printing all PASS.
5. Grep both notebooks for: "Topic 3a", "Topic 3b", "Topic 3", "Topic 4", "capstone",
   "Capstone", "Next session", "Bridging". Zero matches expected (the Vaswani 2017
   citation is fine and is not a topic reference).
6. Confirm plain ASCII only: no em-dashes, en-dashes, Unicode multiplication signs,
   emojis.
```
