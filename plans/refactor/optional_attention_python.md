# Rework Design Doc (R2 - corrected): Optional Attention (Pure Python) Notebook

This is the SECOND, corrected version of the design doc. It resolves the blocking
defects found by the Codex (o3) adversarial review in
`plans/refactor/CODEX_FINDINGS_R1.md`. Relevant findings: R3, R5, R6, R12, plus
R1 context (the PyTorch sibling must be able to restate `scaled_dot_product_attention`
cleanly, so this notebook defines it as a single standalone function cell).

## Scope

Target notebook (Exercise): `Exercises/topic_optional_attention_python/topic_optional_attention_python.ipynb`
Solution twin: `Solutions/topic_optional_attention_python/topic_optional_attention_python.ipynb`

This notebook was formerly "Topic 3a - Seq2Seq and Bahdanau Attention", a required
sequential topic. It is being DEMOTED to a STANDALONE OPTIONAL deep-dive in the
optional/supplementary track.

### Required vs optional framing (Codex R12)

The attention CONCEPT belongs to the REQUIRED course path: required notebooks carry
a concept-level mini-lesson so a user can interpret attention-head visualisations
and CLS tokens. THIS notebook is the from-scratch BUILD / internals deep-dive. It is
genuinely optional. Nothing in the required path depends on it, and nothing in this
doc should imply a student must open it. Wording everywhere must reflect: concept =
required mini-lesson elsewhere; from-scratch build = this optional notebook.

### Applies-to-both note

Every change in this doc applies to BOTH the Exercise and the Solution notebooks,
with two differences in the Solution twin:

1. Solution has NO safety-net cells. Delete every cell whose purpose line says
   "Lab N safety-net" (reworked cells 24, 33, 44 below). Mark them Exercise-only.
2. Solution lab cells are fully filled in. Wherever a reworked cell is a lab starter
   with `= None  # YOUR CODE`, the Solution replaces those lines with the working
   implementation (shown in this doc under "SOLUTION FILL").

The Solution cell count is therefore Exercise count minus 3 (three safety-net cells).

## Why this rework is needed

1. The notebook currently assumes the linear course path. It says "carried forward
   from Topic 2", "We will find out in Topic 3b", "What is coming in Topic 3b",
   "Next: Topic 3b". A standalone optional notebook must not depend on or chain to
   other topics. A student picking it up cold must run it start to finish.
2. Course feedback: attention/transformer content was "too focused on the math;
   students are USERS and care about how they USE attention, not how it works
   internally." This notebook intentionally STAYS the math/internals deep-dive,
   which is exactly why it belongs in the optional track. But even an optional
   deep-dive must open by motivating WHY a user would care, before diving into math.
3. Codex R3: the notebook calls `nltk.download("word2vec_sample")` and then loads it.
   On a regulated, offline classroom image this raises an error and every demo cell
   that depends on the embeddings (and therefore on `found_tokens`) fails.
4. Codex R5: `found_tokens` is produced by the word2vec helper. If the download
   fails it is empty or wrong, and every later cell that iterates `found_tokens`
   breaks. The helper must always return a usable result so no cell hits a NameError
   or a shape mismatch on a cold in-order run.
5. Codex R6: this notebook must not assume artifacts produced by other notebooks
   exist, and any artifact it itself writes must be flagged so required notebooks
   are told never to load it.

## What the rework changes (high level)

- NEW cell 0: OPTIONAL/SUPPLEMENTARY banner (who it is for, self-contained, main
  course path does not require it; the concept is taught in the required path).
- NEW cell 1: short "why a user cares about attention" motivation, plain language,
  before any math.
- EDIT old cell 0 (title/intro): retitle, drop "Topic 3a", drop Topic-2 dependency
  language.
- EDIT old cell 4 (imports/helpers): wrap `nltk.download` in try/except and add a
  self-contained OFFLINE FALLBACK so `get_word2vec_embedding` always returns usable
  embeddings even with no network. This is the R3 + R5 fix.
- EDIT old cell 5 (COMPLAINT_TOKENS): becomes a self-contained setup cell that
  defines the vocabulary locally with a one-line standalone note.
- EDIT old cell 37 (Discussion): remove "We will find out in Topic 3b".
- EDIT old cell 43 (Wrap-Up): remove the "What is coming in Topic 3b" section,
  replace with optional-track framing that mentions the PyTorch companion as
  OPTIONAL, not as a mandatory next step.
- EDIT old cell 47 (end marker): remove "Next: Topic 3b - Attention in PyTorch".
- All four-beat arcs, lab tiers, STAR framing, diagram placeholders, and ASCII-only
  rule are preserved. No math is removed.

## Cell count

- Old Exercise: 48 cells (0-47).
- New Exercise: 50 cells (0-49). Two NEW cells added at the front (banner + user
  motivation). Everything else is renumbered +2 with edits as specified.
- New Solution: 47 cells (50 minus 3 safety-net cells: reworked 24, 33, 44).

## Global rules to honor (from CLAUDE.md)

- Plain ASCII only. No em dashes, en dashes, Unicode multiplication signs, emojis.
  Use `--` or `-` or "to" for ranges. Keep the existing plain-ASCII `x` for shapes
  (e.g. `T x d`).
- Four-beat arc preserved (Beat 1 broken/naive, Beat 2 diagram, Beat 3 demo,
  Beat 4 lab).
- Diagram convention: `<!-- DIAGRAM: ... -->` placeholder kept as-is.
- Safety-net cells in Exercise only.
- `# YOUR CODE` hygiene: placeholder must not hint the answer.
- `numpy<2` stays pinned in the install cell.

## Variable / artifact audit (Codex R5 and R6)

R5 - in-order cold-run variable audit. Every variable used by a cell is defined
earlier IN THIS NOTEBOOK on a straight top-to-bottom run:

| Variable | Defined in reworked cell | First used in reworked cell |
|----------|--------------------------|------------------------------|
| `softmax`, `get_word2vec_embedding`, `plot_attention_weight_matrix` | 6 | 9 |
| `_INLINE_EMBEDDINGS` (offline fallback table) | 6 | 6 (inside helper) |
| `COMPLAINT_TOKENS`, `COMPLAINT_LABELS` | 7 | 7 (print only; no later cell depends on them, kept for context) |
| `bahdanau_attention`, `W1`, `W2`, `v`, `encoder_states`, `decoder_state`, `found_tokens` | 18 | 19 |
| `alpha_mat` | 19 | 28 |
| `scaled_dot_product_attention` | 37 | 38, 42 |
| `Q_batch`, `K_batch`, `V_batch` | 37 | 42 |

`found_tokens` is the variable Codex R5 specifically flagged. It is produced by the
helper inside reworked cell 18 (`complaint_embeddings, found_tokens = get_word2vec_embedding(...)`)
and is padded to length `T_enc` immediately afterwards in the SAME cell. Reworked
cells 19, 28, 37, 38 all read `found_tokens` and all come AFTER cell 18, so an
in-order run never hits a NameError. Reworked cell 38 separately rebinds a local
name `found` (not `found_tokens`) and uses only that local; this is left as-is and
does not affect `found_tokens`.

The ONLY way `found_tokens` could be wrong on a cold run is if the word2vec helper
returns an empty array because the model is missing. The R3 fix (offline fallback in
cell 6) closes that hole: the helper ALWAYS returns a non-empty embedding array, so
`found_tokens` is always a usable, correctly shaped list.

R6 - artifacts. This notebook writes NO files to disk: it produces no `.npy`, `.pt`,
or checkpoint artifacts. It only renders matplotlib figures inline. Therefore no
required notebook can accidentally `load()` an artifact from this notebook. This is
recorded here explicitly so the continuity doc can tell required notebooks: "the
optional pure-Python attention notebook saves nothing; do not depend on it."

---

# CELL-BY-CELL PLAN (Exercise notebook)

Reworked cell numbers are absolute (0-49). "OLD" refers to the original index.

---

## Cell 0 - NEW (markdown) - Optional/supplementary banner

NEW. Insert as the very first cell.

Content:

```
# Optional Deep-Dive: How Attention Works (Pure Python)

> OPTIONAL SUPPLEMENTARY NOTEBOOK
>
> This is an optional deep-dive. The main course path does NOT require it.
> The required path teaches the attention CONCEPT as a short mini-lesson so you
> can read attention-head visualisations and understand CLS tokens. THIS notebook
> is the from-scratch BUILD: you implement the mechanism yourself in NumPy.
> You can complete the course without ever opening this notebook.

## Who this notebook is for

You should work through this notebook if you want to understand what is
happening INSIDE an attention layer, not just how to call one. It is aimed at
learners who are comfortable with NumPy and want to see attention built from
first principles.

If you only care about USING attention-based models (calling an LLM, fine-tuning
a Transformer, prompting), you can safely skip this notebook. The main course
covers the attention concept you need as a user.

## This notebook is self-contained

Everything you need is defined inside this notebook. It does not depend on any
other course notebook, and you do not need to have run any earlier topic. You
can open it cold and run it start to finish.

It also runs OFFLINE. The first demo would normally download a small word2vec
sample, but if there is no network the notebook falls back to a tiny set of
embedding vectors defined inline, so every cell still runs.

This notebook writes no files to disk. It produces only inline plots.

There is a companion optional notebook that ports these same ideas to PyTorch.
That companion is also optional. You may read it after this one if you want to
see the framework version, but it is not a required next step.

## What you will build

By the end you will have implemented, from scratch in NumPy:
- A simulation of seq2seq without attention, and seen exactly where it breaks down
- Bahdanau (additive) attention: alignment scores, softmax weights, context vector
- Dot product attention: a simpler variant of the same idea
- Scaled dot product attention: the version used in Transformers

## Learning objectives

1. Explain the fixed-size bottleneck problem in seq2seq models
2. Implement the Bahdanau attention score computation step by step
3. Compare additive vs dot product vs scaled dot product attention
4. Visualise attention weights as a heatmap over complaint tokens
```

Rationale: absorbs the "What you will build" and "Learning objectives" lists from
OLD cell 0 so OLD cell 0 can be trimmed (Cell 2). The banner now also states the
offline behavior (R3) and the no-artifacts fact (R6), and frames the notebook as
the optional BUILD versus the required CONCEPT (R12).

---

## Cell 1 - NEW (markdown) - Why a user cares about attention

NEW. Insert second, before any math.

Content:

```
## Before the math: why should a user care about attention?

Attention is the single idea that made modern language models work. Every model
you use as a developer -- the LLM behind a chat API, a Transformer you fine-tune,
a translation model -- is built out of attention layers.

As a user you mostly do not need to know the internals. But three things that you
DO notice every day come straight out of how attention works:

1. Context windows. A model can "look back" at any earlier token in the prompt
   because attention lets every position read every other position directly.
   The size of that window is bounded by how much attention you can afford to
   compute.

2. Long-input quality. Older sequence models compressed an entire input into one
   fixed-size vector, so details early in a long input got washed out. Attention
   removed that bottleneck. This is why a modern model can answer a question about
   page 1 of a long document you pasted on page 40.

3. Cost and latency. Attention compares every token to every other token, so its
   cost grows with the square of the input length. That is the real reason long
   prompts are slower and more expensive.

This notebook opens up that mechanism. We build it in plain NumPy on a small
customer-complaint example so you can see every number. You will not need this to
USE a model, but once you have seen it, the behaviour above stops being magic.
```

Rationale: directly answers the course feedback. Plain language, user-facing
consequences, ASCII only. Sits before Section 1 math.

---

## Cell 2 - EDIT of OLD cell 0 (markdown) - Context only

EDIT. The title, "What you will build", and "Learning objectives" moved into the
new banner (Cell 0). This cell is trimmed to just the running-example context, and
the heading drops "Topic 3a".

OLD text:

```
# Topic 3a - Seq2Seq and Bahdanau Attention

Barclays Customer Support Intelligence System

## What you will build

By the end of this notebook you will have implemented, from scratch in NumPy:
- A simulation of seq2seq without attention (and shown exactly where it breaks down)
- Bahdanau (additive) attention: alignment scores, softmax weights, context vector
- Dot product attention: a simpler variant of the same idea
- Scaled dot product attention: the version used in Transformers

## Learning objectives

1. Explain the fixed-size bottleneck problem in seq2seq models
2. Implement the Bahdanau attention score computation step by step
3. Compare additive vs dot product vs scaled dot product attention
4. Visualise attention weights as a heatmap over complaint tokens

## Context

Our customer support team receives thousands of messages a day. Long complaints (50+ tokens)
are the hardest to route automatically because they contain multiple issues.
A plain seq2seq model compresses the entire complaint into one fixed-size vector --
important details at the end of a long message get lost. Attention fixes this.
```

NEW text:

```
## The running example: a customer-complaint router

To keep every number concrete we use one running example throughout this
notebook: an imagined customer-support team that receives thousands of complaint
messages a day and wants to route them automatically.

Long complaints (50 or more tokens) are the hardest to route, because they
contain multiple issues. A plain seq2seq model compresses the entire complaint
into one fixed-size vector, so important details at the end of a long message get
lost. Attention fixes this.

You do not need any prior course notebook to follow along. The next cells install
the packages and define the small vocabulary this notebook uses.
```

Note: the em-dash-style `--` in the OLD "fixed-size vector --" line is replaced by
a plain `, so` clause (done above). The remaining `--` in cell 1 is a deliberate
ASCII double-hyphen, allowed.

---

## Cell 3 - KEEP of OLD cell 1 (code) - Disable TensorFlow backend

KEEP verbatim. No changes.

```
# Disable TensorFlow backend in transformers (SageMaker image compatibility).
# Must run before any transformers import.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

---

## Cell 4 - KEEP of OLD cell 2 (code) - Environment setup / pip install

KEEP verbatim. `numpy<2` already pinned. No changes.

---

## Cell 5 - KEEP of OLD cell 3 (code) - SageMaker session

KEEP verbatim. No changes. This cell only sets up a SageMaker session for the
hosted environment; it does not chain to another topic and writes no files.

---

## Cell 6 - EDIT of OLD cell 4 (code) - Imports, OFFLINE-SAFE helpers

EDIT. This is the core R3 + R5 fix. The original cell calls
`nltk.download("word2vec_sample", quiet=True)` at import time and
`get_word2vec_embedding` then loads the downloaded file. On an offline classroom
image the download fails and every embedding-dependent demo (and therefore
`found_tokens`) breaks.

The rework: wrap the download in try/except, define a small inline embedding table
covering every word this notebook ever embeds, and make `get_word2vec_embedding`
fall back to that table whenever the word2vec model is unavailable. The helper then
ALWAYS returns a non-empty `(embeddings, words_pass)` pair, so no downstream cell
ever sees an empty array or an undefined name.

OLD text (full cell):

```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import gensim
from nltk.data import find
import nltk

nltk.download("word2vec_sample", quiet=True)

def softmax(x, axis=-1):
    """Compute softmax along the specified axis. Numerically stable version."""
    # Subtract max for numerical stability before exp
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def get_word2vec_embedding(words):
    """
    Load the NLTK word2vec sample model and return embeddings for words found in vocabulary.
    Words not in vocabulary are silently skipped.
    Returns: (embeddings array of shape [N_found, 300], filtered word list)
    """
    word2vec_sample = str(find("models/word2vec_sample/pruned.word2vec.txt"))
    model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False
    )
    output = []
    words_pass = []
    for word in words:
        try:
            # get_vector replaces deprecated word_vec
            output.append(np.array(model.get_vector(word)))
            words_pass.append(word)
        except KeyError:
            pass
    embeddings = np.array(output)
    del model
    return embeddings, words_pass

def plot_attention_weight_matrix(weight_matrix, x_ticks, y_ticks, title="Attention weights"):
    """Plot a 2D attention weight matrix as a heatmap."""
    plt.figure(figsize=(max(8, len(x_ticks) * 1.2), max(5, len(y_ticks) * 0.8)))
    ax = sns.heatmap(weight_matrix, cmap="Blues", annot=True, fmt=".2f",
                     linewidths=0.5, linecolor="lightgray")
    plt.xticks(np.arange(weight_matrix.shape[1]) + 0.5, x_ticks, rotation=30, ha="right")
    plt.yticks(np.arange(weight_matrix.shape[0]) + 0.5, y_ticks, rotation=0)
    plt.title(title)
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.tight_layout()
    plt.show()

print("Imports and helpers loaded.")
```

NEW text (full cell):

```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# This notebook can run fully OFFLINE. We try to load the small NLTK word2vec
# sample for nicer real-word embeddings, but if there is no network (a common
# case on a locked-down classroom image) we fall back to a tiny inline table of
# random-but-fixed embedding vectors. Either way every demo below still runs.

EMBED_DIM = 300          # word2vec sample dimension; the inline fallback matches it
_WORD2VEC_MODEL = None   # populated lazily if the real model is available
_WORD2VEC_AVAILABLE = False

# Every word this notebook ever asks to embed. Keeping this list explicit means
# the offline fallback can guarantee an embedding for all of them.
_NOTEBOOK_VOCAB = [
    "charge", "unauthorised", "unauthorized", "account", "refund", "dispute",
    "urgent", "contacted", "branch", "twice", "no", "response", "still",
    "waiting", "three", "weeks", "unable", "resolve", "issue", "extremely",
    "frustrated", "customer", "fraud", "help", "transaction", "card", "blocked",
    "payment", "failed", "interest", "disputed", "transfer", "balance",
    "declined", "overdraft",
]

# Inline fallback: deterministic pseudo-embeddings, one per vocab word.
# A fixed seed makes the fallback reproducible run to run.
_fallback_rng = np.random.RandomState(20240517)
_INLINE_EMBEDDINGS = {
    word: _fallback_rng.randn(EMBED_DIM).astype(np.float64)
    for word in _NOTEBOOK_VOCAB
}

try:
    import nltk
    from nltk.data import find
    import gensim
    nltk.download("word2vec_sample", quiet=True)
    _word2vec_path = str(find("models/word2vec_sample/pruned.word2vec.txt"))
    _WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format(
        _word2vec_path, binary=False
    )
    _WORD2VEC_AVAILABLE = True
    print("word2vec sample loaded - using real pretrained embeddings.")
except Exception as exc:
    print("word2vec sample is not available (offline or download blocked).")
    print(f"Reason: {exc}")
    print("Falling back to the inline embedding table. Every demo still runs.")

def softmax(x, axis=-1):
    """Compute softmax along the specified axis. Numerically stable version."""
    # Subtract max for numerical stability before exp
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def get_word2vec_embedding(words):
    """
    Return embeddings for the given words.

    If the NLTK word2vec sample loaded successfully, real pretrained vectors are
    used. Otherwise the inline fallback table is used so the notebook runs
    offline. Words with no available vector are silently skipped.

    This function ALWAYS returns a usable result: as long as at least one input
    word is in the notebook vocabulary it returns a non-empty embedding array,
    so downstream cells never see an empty array or an undefined variable.

    Returns:
        embeddings: array of shape (N_found, EMBED_DIM)
        words_pass: the list of input words that were embedded, same order
    """
    output = []
    words_pass = []
    for word in words:
        vec = None
        if _WORD2VEC_AVAILABLE:
            try:
                vec = np.array(_WORD2VEC_MODEL.get_vector(word), dtype=np.float64)
            except KeyError:
                vec = None
        if vec is None:
            # Offline fallback (also covers words missing from word2vec).
            vec = _INLINE_EMBEDDINGS.get(word)
        if vec is not None:
            output.append(vec)
            words_pass.append(word)
    embeddings = np.array(output, dtype=np.float64)
    return embeddings, words_pass

def plot_attention_weight_matrix(weight_matrix, x_ticks, y_ticks, title="Attention weights"):
    """Plot a 2D attention weight matrix as a heatmap."""
    plt.figure(figsize=(max(8, len(x_ticks) * 1.2), max(5, len(y_ticks) * 0.8)))
    ax = sns.heatmap(weight_matrix, cmap="Blues", annot=True, fmt=".2f",
                     linewidths=0.5, linecolor="lightgray")
    plt.xticks(np.arange(weight_matrix.shape[1]) + 0.5, x_ticks, rotation=30, ha="right")
    plt.yticks(np.arange(weight_matrix.shape[0]) + 0.5, y_ticks, rotation=0)
    plt.title(title)
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.tight_layout()
    plt.show()

print("Imports and helpers loaded.")
```

Notes for the building agent:
- `_NOTEBOOK_VOCAB` MUST contain every word later passed to
  `get_word2vec_embedding`. Audited against the current notebook these are: the
  six `preview_tokens` (reworked cell 13), the eight `complaint_tokens` (reworked
  cell 18), the eight `complaint_words` (reworked cell 38), plus the words in
  `short_complaint` and `long_complaint` (reworked cell 9). The list above already
  covers all of them. If a later edit adds a new word to any demo, it must also be
  added here.
- The fallback uses British and American spellings (`unauthorised` and
  `unauthorized`) because the demos mix them.
- The fallback is deterministic (fixed `RandomState`), so attention heatmaps are
  reproducible offline.
- This cell still writes no files (R6).

---

## Cell 7 - EDIT of OLD cell 5 (code) - Self-contained complaint vocabulary

EDIT. Remove "carried forward from Topic 2"; make the cell self-contained with an
explicit one-line note.

OLD text:

```
# Barclays complaint token vocabulary -- carried forward from Topic 2
COMPLAINT_TOKENS = [
    "unauthorized", "transaction", "account", "card", "blocked",
    "payment", "failed", "interest", "charge", "disputed",
    "transfer", "balance", "fraud", "declined", "overdraft"
]
COMPLAINT_LABELS = ["fraud", "billing", "access", "payment", "general"]
print(f"Complaint vocabulary: {len(COMPLAINT_TOKENS)} tokens, {len(COMPLAINT_LABELS)} labels")
```

NEW text:

```
# Complaint token vocabulary for this notebook.
# This notebook is standalone, so we define the vocabulary right here.
# Other course notebooks use a similar list; we keep a local copy so this
# notebook runs on its own with no dependency on any earlier topic.
COMPLAINT_TOKENS = [
    "unauthorized", "transaction", "account", "card", "blocked",
    "payment", "failed", "interest", "charge", "disputed",
    "transfer", "balance", "fraud", "declined", "overdraft"
]
COMPLAINT_LABELS = ["fraud", "billing", "access", "payment", "general"]
print(f"Complaint vocabulary: {len(COMPLAINT_TOKENS)} tokens, {len(COMPLAINT_LABELS)} labels")
```

Only the comment block changed. The data is identical. (These two variables are
used only by this print line; no later cell reads them. They are kept for context
and to keep the renumbering 1:1 with the original.)

---

## Cell 8 - EDIT of OLD cell 6 (markdown) - Section 1 header

EDIT. Drop the "currently"/"Our" voice that implies an ongoing course project;
reframe as the running example. Math/structure unchanged.

OLD text:

```
## Section 1 - The Fixed-Size Bottleneck Problem

### The situation at Barclays

Our complaints routing pipeline currently uses a seq2seq model: an LSTM encoder reads
the full complaint message and compresses it into a single hidden state vector.
The decoder then generates a category label from that one vector.

This works for short messages. But watch what happens with a long complaint.
```

NEW text:

```
## Section 1 - The Fixed-Size Bottleneck Problem

### The situation in our running example

Imagine the complaints routing pipeline uses a seq2seq model: an LSTM encoder
reads the full complaint message and compresses it into a single hidden state
vector. The decoder then generates a category label from that one vector.

This works for short messages. But watch what happens with a long complaint.
```

---

## Cell 9 - KEEP of OLD cell 7 (code) - Beat 1: bottleneck in action

KEEP verbatim. Pure NumPy simulation. It calls `get_word2vec_embedding` on the
words of `short_complaint` and `long_complaint`; all those words are in
`_NOTEBOOK_VOCAB` so the embeddings are always non-empty even offline. No changes.

---

## Cell 10 - KEEP of OLD cell 8 (markdown) - Discussion (3 minutes)

KEEP verbatim. No topic chaining. No changes.

---

## Cell 11 - KEEP of OLD cell 9 (code) - Beat 2 diagram anchor

KEEP verbatim.

---

## Cell 12 - KEEP of OLD cell 10 (markdown) - Bottleneck vs attention diagram

KEEP verbatim. Mermaid diagram + `<!-- DIAGRAM: ... -->` placeholder. No changes.

---

## Cell 13 - KEEP of OLD cell 11 (code) - Beat 3: attention solves bottleneck

KEEP verbatim. Self-contained preview. Uses `preview_tokens` (all in
`_NOTEBOOK_VOCAB`). No changes.

---

## Cell 14 - KEEP of OLD cell 12 (markdown) - Quick Observation

KEEP verbatim. Mentions "Lab 1" within this notebook only, which is fine.

---

## Cell 15 - KEEP of OLD cell 13 (markdown) - Section 2: Bahdanau math

KEEP verbatim. No topic chaining.

---

## Cell 16 - KEEP of OLD cell 14 (code) - Beat 2 diagram anchor

KEEP verbatim.

---

## Cell 17 - KEEP of OLD cell 15 (markdown) - Bahdanau score diagram

KEEP verbatim.

---

## Cell 18 - KEEP of OLD cell 16 (code) - Beat 3: Bahdanau implementation

KEEP verbatim. This cell DEFINES `bahdanau_attention`, `W1`, `W2`, `v`,
`encoder_states`, `decoder_state`, and `found_tokens`. The R5 audit table above
confirms every later use of `found_tokens` (reworked cells 19, 28, 37, 38) comes
after this cell. The cell already pads `complaint_embeddings` and `found_tokens`
to length `T_enc`, so a short fallback embedding set still yields a correctly
shaped `(T_enc, d_h)` `encoder_states`. No changes.

---

## Cell 19 - KEEP of OLD cell 17 (code) - Beat 3 continued: vectorised + heatmap

KEEP verbatim. Defines `alpha_mat`, used later by reworked cell 28. Reads
`found_tokens` (defined in cell 18). No changes.

---

## Cell 20 - KEEP of OLD cell 18 (markdown) - Lab 1 instructions (Tier 1)

KEEP verbatim. STAR framing intact. No topic chaining.

---

## Cell 21 - KEEP of OLD cell 19 (code) - Lab 1 starter

KEEP verbatim in the Exercise. Three `= None  # YOUR CODE` stubs.

SOLUTION FILL (Solution twin only): replace the three stub lines so the body reads:

```
        enc_part = W1 @ h_i
        dec_part = W2 @ decoder_state
        energy[i] = v @ np.tanh(enc_part + dec_part)
```

and

```
    alpha = softmax(energy)
```

and

```
    context_vector = alpha @ encoder_states
```

(Matches `bahdanau_attention` from Cell 18.)

---

## Cell 22 - KEEP of OLD cell 20 (code) - Lab 1 verification

KEEP verbatim in both notebooks.

---

## Cell 23 - KEEP of OLD cell 21 (code) - Lab 1 safety-net

KEEP verbatim in the EXERCISE. DELETE from the SOLUTION.

---

## Cell 24 - KEEP of OLD cell 22 (markdown) - Lab 1 stretch + homework

KEEP verbatim. Self-contained, no topic chaining.

---

## Cell 25 - KEEP of OLD cell 23 (markdown) - Section 3: dot product intro

KEEP verbatim.

---

## Cell 26 - KEEP of OLD cell 24 (code) - Beat 1: naive dot product breaks

KEEP verbatim.

---

## Cell 27 - KEEP of OLD cell 25 (markdown) - Beat 2: how dot product works

KEEP verbatim. No topic chaining.

---

## Cell 28 - KEEP of OLD cell 26 (code) - Beat 3: working dot product attention

KEEP verbatim. Reads `encoder_states`, `decoder_state`, `found_tokens`, `alpha_mat`,
all defined in earlier reworked cells (18, 18, 18, 19). No changes.

---

## Cell 29 - KEEP of OLD cell 27 (markdown) - Lab 2 instructions (Tier 2)

KEEP verbatim. STAR framing intact.

---

## Cell 30 - KEEP of OLD cell 28 (code) - Lab 2 starter

KEEP verbatim in the Exercise. Two `= None  # YOUR CODE` stubs.

SOLUTION FILL (Solution twin only): replace the function body so it reads:

```
    scores = encoder_states @ decoder_state
    alpha = softmax(scores)
    context = alpha @ encoder_states
    return context, alpha
```

(Matches `dot_product_attention` from Cell 28 with no projection arguments.)

---

## Cell 31 - KEEP of OLD cell 29 (code) - Lab 2 verification

KEEP verbatim in both notebooks.

---

## Cell 32 - KEEP of OLD cell 30 (code) - Lab 2 safety-net

KEEP verbatim in the EXERCISE. DELETE from the SOLUTION.

---

## Cell 33 - KEEP of OLD cell 31 (markdown) - Lab 2 stretch + homework

KEEP verbatim.

---

## Cell 34 - KEEP of OLD cell 32 (markdown) - Section 4: scaled dot product intro

KEEP verbatim.

---

## Cell 35 - KEEP of OLD cell 33 (code) - Beat 1: softmax saturation

KEEP verbatim.

---

## Cell 36 - KEEP of OLD cell 34 (markdown) - Beat 2: why scaling fixes it

KEEP verbatim. No topic chaining.

---

## Cell 37 - KEEP of OLD cell 35 (code) - Beat 3: scaled dot product attention

KEEP verbatim. This cell DEFINES `scaled_dot_product_attention` as a single clean,
standalone function whose only dependency is `softmax` (defined in reworked cell 6)
and NumPy. This is intentional: per Codex R1, the optional PyTorch companion must
be able to restate or mirror this function on its own. Because the function here is
self-contained (no hidden globals, batched 3D signature `(batch, T, d)`, returns
`(output, attention_weights)`), the PyTorch notebook can define its own equivalent
without importing anything from this notebook. Do NOT make this function depend on
any notebook-level variable. No changes.

---

## Cell 38 - KEEP of OLD cell 36 (code) - Comparison of all three variants

KEEP verbatim. Calls `get_word2vec_embedding(complaint_words)`; all eight words are
in `_NOTEBOOK_VOCAB`, so this works offline. Rebinds local names `emb` and `found`
(distinct from `found_tokens`). No changes.

---

## Cell 39 - EDIT of OLD cell 37 (markdown) - Discussion (3 minutes)

EDIT. Remove the "(We will find out in Topic 3b.)" sequential chaining in
question 3. Reframe as an open question that does not depend on a later topic.

OLD text (question 3 only; questions 1 and 2 unchanged):

```
3. We implemented attention in pure NumPy. What would change if you did this in PyTorch?
   (We will find out in Topic 3b.)
```

NEW text (question 3):

```
3. We implemented attention in pure NumPy. What would change if you did this in
   a deep learning framework like PyTorch? Think about gradients, batching, and
   GPU execution. There is an optional companion notebook that ports this exact
   code to PyTorch if you want to check your answer.
```

The surrounding cell (header, questions 1 and 2, the three-variant summary list)
stays verbatim.

---

## Cell 40 - KEEP of OLD cell 38 (markdown) - Lab 3 instructions (Tier 1)

KEEP verbatim. STAR framing intact. The phrase "prototyping a Transformer-based
complaint summariser" is in-notebook narrative, not a topic reference.

---

## Cell 41 - KEEP of OLD cell 39 (code) - Lab 3 starter

KEEP verbatim in the Exercise. Four `= None  # YOUR CODE` stubs inside
`_lab3_template`, plus `my_scaled_dot_product_attention = None  # YOUR CODE`.

SOLUTION FILL (Solution twin only): replace the four step lines so the body of
`_lab3_template` reads:

```
    d_k = Q.shape[-1]
    raw_scores = np.matmul(Q, K.transpose(0, 2, 1))
    scaled_scores = raw_scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights
```

and change the top stub line from

```
my_scaled_dot_product_attention = None  # YOUR CODE
```

to (the assignment is placed AFTER the `_lab3_template` def so the name is bound to
a defined function):

```
my_scaled_dot_product_attention = _lab3_template
```

Keep the final `print(f"my_scaled_dot_product_attention is: ...")` line.

---

## Cell 42 - KEEP of OLD cell 40 (code) - Lab 3 verification

KEEP verbatim in both notebooks. Reads `Q_batch`, `K_batch`, `V_batch` and
`scaled_dot_product_attention`, all defined in reworked cell 37.

---

## Cell 43 - KEEP of OLD cell 41 (code) - Lab 3 safety-net

KEEP verbatim in the EXERCISE. DELETE from the SOLUTION.

---

## Cell 44 - KEEP of OLD cell 42 (markdown) - Lab 3 stretch + homework

KEEP verbatim.

---

## Cell 45 - EDIT of OLD cell 43 (markdown) - Wrap-Up

EDIT. Rename "Key takeaways from Topic 3a" to drop "Topic 3a". REPLACE the entire
"What is coming in Topic 3b" section with optional-track framing.

OLD text:

```
## Wrap-Up

### Key takeaways from Topic 3a

1. The seq2seq fixed bottleneck loses information from long inputs. The further a token
   is from the end of a sequence, the more it gets diluted in the context vector.

2. Bahdanau attention solves this by computing a DIFFERENT context vector at each
   decoder step, attending selectively to encoder hidden states via learned alignment weights.

3. Dot product attention is a simpler variant that requires the same dimensions for
   queries and keys. It is faster but less flexible than Bahdanau.

4. Scaled dot product attention divides scores by sqrt(d_k) to prevent softmax
   saturation and vanishing gradients at large key dimensions.

5. Attention weights are always non-negative and sum to 1 along the key dimension.
   You can visualise them as a heatmap to understand what the model focuses on.

### What is coming in Topic 3b

In Topic 3b we will port everything we just built to PyTorch. You will see:
- How PyTorch's nn.Module wraps the same math into a trainable class
- What torch.nn.functional.scaled_dot_product_attention looks like (and how your
  implementation compares)
- The capstone: implement scaled dot product attention from scratch in PyTorch,
  then apply it to a real complaint triage task

The math is the same. The framework handles gradients automatically.
```

NEW text:

```
## Wrap-Up

### Key takeaways

1. The seq2seq fixed bottleneck loses information from long inputs. The further a
   token is from the end of a sequence, the more it gets diluted in the context
   vector.

2. Bahdanau attention solves this by computing a DIFFERENT context vector at each
   decoder step, attending selectively to encoder hidden states via learned
   alignment weights.

3. Dot product attention is a simpler variant that requires the same dimensions
   for queries and keys. It is faster but less flexible than Bahdanau.

4. Scaled dot product attention divides scores by sqrt(d_k) to prevent softmax
   saturation and vanishing gradients at large key dimensions.

5. Attention weights are always non-negative and sum to 1 along the key dimension.
   You can visualise them as a heatmap to understand what the model focuses on.

### Connecting this back to using models

Remember the three user-facing facts from the start of this notebook: context
windows, long-input quality, and the square-with-length cost of long prompts.
You have now seen the mechanism behind all three. The "look back at any position"
ability is the weighted sum over encoder states. The cost growth is the dot
product of every query against every key.

### Where to go next (optional)

This notebook is the end of the pure-NumPy attention deep-dive. There is an
optional companion notebook that ports these exact mechanisms to PyTorch. It
shows how PyTorch's nn.Module wraps the same math into a trainable class, what
torch.nn.functional.scaled_dot_product_attention looks like, and how to apply
attention to a small triage task with automatic gradients.

That companion is optional, not a required next step. The attention concept you
need as a user is taught in the main course path; this notebook and its PyTorch
companion are the from-scratch builds for learners who want the internals. The
math is identical to what you built here; the framework only adds automatic
differentiation and GPU execution. Read it if you are curious; skip it if you
have what you came for.
```

---

## Cell 46 - KEEP of OLD cell 44 (markdown) - Homework Extensions header

KEEP verbatim.

---

## Cell 47 - KEEP of OLD cell 45 (code) - Homework Extension 1

KEEP verbatim in the Exercise (`pass  # YOUR IMPLEMENTATION` left as a stub by
design for homework).

SOLUTION FILL: leave as-is. Homework extension stubs are intentionally left
unsolved in both Exercise and Solution (async take-home work, not graded lab
cells). Do NOT fill these in the Solution.

---

## Cell 48 - KEEP of OLD cell 46 (code) - Homework Extension 2

KEEP verbatim in both notebooks (same reasoning as Cell 47).

---

## Cell 49 - EDIT of OLD cell 47 (markdown) - End marker

EDIT. Remove "Topic 3a" naming and the "Next: Topic 3b" chain.

OLD text:

```
*End of Topic 3a - Seq2Seq and Bahdanau Attention*

Next: Topic 3b - Attention in PyTorch
```

NEW text:

```
*End of the optional deep-dive: How Attention Works (Pure Python)*

This was an optional notebook. The main course path continues independently of
it. If you want the framework version of everything you built here, the optional
PyTorch companion notebook is available, but it is not required.
```

---

# Summary table of changes

| New cell | Old cell | Action | Note |
|----------|----------|--------|------|
| 0  | -  | NEW      | Optional/supplementary banner (offline + no-artifacts + R12 framing) |
| 1  | -  | NEW      | "Why a user cares about attention" motivation |
| 2  | 0  | EDIT     | Trim title/objectives (moved to banner), drop Topic 3a |
| 3  | 1  | KEEP     | |
| 4  | 2  | KEEP     | |
| 5  | 3  | KEEP     | |
| 6  | 4  | EDIT     | Offline-safe nltk.download + inline fallback embeddings (R3, R5) |
| 7  | 5  | EDIT     | Self-contained COMPLAINT_TOKENS, drop "from Topic 2" |
| 8  | 6  | EDIT     | Reframe "situation at Barclays" as running example |
| 9-14  | 7-12  | KEEP | |
| 15-19 | 13-17 | KEEP | |
| 20-22 | 18-20 | KEEP | Lab 1 |
| 23 | 21 | KEEP/DELETE | Safety-net: Exercise keeps, Solution deletes |
| 24-31 | 22-29 | KEEP | |
| 32 | 30 | KEEP/DELETE | Safety-net: Exercise keeps, Solution deletes |
| 33-38 | 31-36 | KEEP | |
| 39 | 37 | EDIT | Discussion Q3: drop "in Topic 3b" |
| 40-42 | 38-40 | KEEP | Lab 3 |
| 43 | 41 | KEEP/DELETE | Safety-net: Exercise keeps, Solution deletes |
| 44 | 42 | KEEP | |
| 45 | 43 | EDIT | Wrap-Up: replace "What is coming in Topic 3b" |
| 46-48 | 44-46 | KEEP | |
| 49 | 47 | EDIT | End marker: drop "Next: Topic 3b" |

Solution twin: identical except cells 23, 32, 43 (safety-nets) are deleted and
lab starters 21, 30, 41 carry the SOLUTION FILL bodies. Final Solution cell
count: 47.

---

# Codex R1 findings resolved

| Finding | What it requires | Cell(s) that fix it | How |
|---------|------------------|---------------------|-----|
| R3 | Wrap `nltk.download()` in try/except with a clear message AND a self-contained inline fallback so the first demo runs offline. | Cell 6 | The `nltk.download` + word2vec load is inside a `try/except`. On failure it prints a clear reason and sets `_WORD2VEC_AVAILABLE = False`. An inline `_INLINE_EMBEDDINGS` table (deterministic 300-dim vectors for every word the notebook embeds) is defined unconditionally. `get_word2vec_embedding` uses real vectors when available and the inline table otherwise, so the first demo (Cell 9) and all later demos run with no network. |
| R5 | Every helper-produced variable, especially `found_tokens`, must be defined earlier IN THIS NOTEBOOK so an in-order cold run never hits a NameError. | Cell 6 (helper), Cell 18 (defines `found_tokens`); audited in the "Variable / artifact audit" section | `found_tokens` is produced in reworked cell 18 and padded to `T_enc` in the same cell; every later use (cells 19, 28, 37, 38) is strictly after it. The remaining risk -- the helper returning an empty array offline, which would corrupt `found_tokens` -- is closed by the R3 fix: `get_word2vec_embedding` always returns a non-empty, correctly shaped array. Full variable table provided so the building agent can re-verify. |
| R6 | The notebook must not assume artifacts from other notebooks exist; any artifact it saves must be flagged. | Cell 0 banner; "Variable / artifact audit" section | The notebook reads no external artifacts (it defines its own `COMPLAINT_TOKENS` and embeddings) and writes NO files to disk -- only inline matplotlib plots. The banner states "This notebook writes no files to disk." The doc records this so the continuity doc can tell required notebooks never to load artifacts from this optional notebook. |
| R12 | Wording must reflect: attention CONCEPT = required mini-lesson; from-scratch BUILD = optional. Do not imply this notebook is required. | Cells 0, 45, 49 | The banner (Cell 0) explicitly says the required path teaches the attention concept as a mini-lesson and THIS notebook is the optional from-scratch build. The Wrap-Up (Cell 45) repeats "the attention concept you need as a user is taught in the main course path; this notebook and its PyTorch companion are the from-scratch builds." The end marker (Cell 49) states it was optional and the main path continues independently. |
| R1 (context) | The PyTorch optional must be able to restate/redefine `scaled_dot_product_attention` cleanly. | Cell 37 | The doc pins `scaled_dot_product_attention` as a single clean standalone function whose only dependencies are `softmax` and NumPy, with a batched 3D signature and an explicit "do NOT make this depend on any notebook-level variable" instruction. The PyTorch companion can therefore define its own mirror without importing from this notebook. |

---

# Verification checklist for the building agent

- [ ] grep the finished Exercise and Solution for "Topic 2", "Topic 3a",
      "Topic 3b": zero matches expected.
- [ ] grep for "carried forward": zero matches.
- [ ] grep for em dash, en dash, multiplication sign, emoji: zero matches.
- [ ] Cell 0 banner present and is the first cell; states offline + no-artifacts +
      optional-build-vs-required-concept framing.
- [ ] Cell 1 user-motivation present before Section 1.
- [ ] Cell 6 wraps `nltk.download` in try/except and defines `_INLINE_EMBEDDINGS`
      covering every word later passed to `get_word2vec_embedding`.
- [ ] `get_word2vec_embedding` returns a non-empty array with the network disabled.
- [ ] `COMPLAINT_TOKENS` defined locally with the standalone note.
- [ ] `found_tokens` defined before reworked cells 19, 28, 37, 38 use it.
- [ ] `scaled_dot_product_attention` (cell 37) depends only on `softmax` + NumPy.
- [ ] Notebook writes no files to disk.
- [ ] Exercise has 3 safety-net cells; Solution has 0.
- [ ] Solution lab cells (21, 30, 41) contain working code, no `# YOUR CODE`.
- [ ] Notebook runs top to bottom with no prior-topic state and no network.
