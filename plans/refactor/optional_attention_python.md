# Rework Design Doc: Optional Attention (Pure Python) Notebook

## Scope

Target notebook (Exercise): `Exercises/topic_optional_attention_python/topic_optional_attention_python.ipynb`
Solution twin: `Solutions/topic_optional_attention_python/topic_optional_attention_python.ipynb`

This notebook was formerly "Topic 3a - Seq2Seq and Bahdanau Attention", a required
sequential topic. It is being DEMOTED to a STANDALONE OPTIONAL deep-dive in the
optional/supplementary track.

### Applies-to-both note

Every change in this doc applies to BOTH the Exercise and the Solution notebooks,
with two differences in the Solution twin:

1. Solution has NO safety-net cells. Delete every cell whose purpose line says
   "Lab N safety-net" (current cells 21, 30, 41). In the reworked numbering these
   are reworked cells 24, 36, 49 below; mark them as Exercise-only.
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

## What the rework changes (high level)

- NEW cell 0: OPTIONAL/SUPPLEMENTARY banner (who it is for, self-contained, main
  course path does not require it).
- NEW cell 1: short "why a user cares about attention" motivation, plain language,
  before any math.
- EDIT old cell 0 (title/intro): retitle, drop "Topic 3a", drop Topic-2 dependency
  language.
- EDIT old cell 5 (COMPLAINT_TOKENS): becomes a self-contained setup cell that
  defines the vocabulary locally with a one-line note that other course notebooks
  use a similar list but this notebook defines its own copy so it runs alone.
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
- New Solution: 47 cells (50 minus 3 safety-net cells).

## Global rules to honor (from CLAUDE.md)

- Plain ASCII only. No em dashes, en dashes, Unicode multiplication signs, emojis.
  Use `--` or `-` or "to" for ranges. Use `x` or the word "by" for shapes already
  written as `T x d` in the original (keep that style, it is plain ASCII `x`).
- Four-beat arc preserved (Beat 1 broken/naive, Beat 2 diagram, Beat 3 demo,
  Beat 4 lab).
- Diagram convention: `<!-- DIAGRAM: ... -->` placeholder kept as-is.
- Safety-net cells in Exercise only.
- `# YOUR CODE` hygiene: placeholder must not hint the answer.
- `numpy<2` stays pinned in the install cell.

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
> You can complete the course without ever opening this notebook.

## Who this notebook is for

You should work through this notebook if you want to understand what is
happening INSIDE an attention layer, not just how to call one. It is aimed at
learners who are comfortable with NumPy and want to see attention built from
first principles.

If you only care about USING attention-based models (calling an LLM, fine-tuning
a Transformer, prompting), you can safely skip this notebook. The main course
covers everything you need as a user.

## This notebook is self-contained

Everything you need is defined inside this notebook. It does not depend on any
other course notebook, and you do not need to have run any earlier topic. You
can open it cold and run it start to finish.

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

Rationale: this absorbs the "What you will build" and "Learning objectives" lists
that were in OLD cell 0, so OLD cell 0 can be trimmed (see Cell 2).

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

EDIT. The title, "What you will build", and "Learning objectives" moved into
the new banner (Cell 0). This cell is trimmed to just the Barclays context, and
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

Note: replace the em-dash `--` style that was in OLD line "fixed-size vector --"
with a plain `, so` clause (done above).

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

KEEP verbatim. No changes. (This cell only sets up a SageMaker session for the
hosted environment; it does not chain to another topic.)

---

## Cell 6 - KEEP of OLD cell 4 (code) - Imports and helper functions

KEEP verbatim. Defines `softmax`, `get_word2vec_embedding`,
`plot_attention_weight_matrix`. No changes.

---

## Cell 7 - EDIT of OLD cell 5 (code) - Self-contained complaint vocabulary

EDIT. This is the key standalone fix. Remove "carried forward from Topic 2";
make the cell self-contained with an explicit one-line note.

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

Only the comment block changed. The data is identical, so all downstream cells
that use `COMPLAINT_TOKENS` / `COMPLAINT_LABELS` keep working unchanged.

---

## Cell 8 - EDIT of OLD cell 6 (markdown) - Section 1 header

EDIT. Minor: drop the word "currently" framing that implies an ongoing course
project, and remove the Barclays-internal-only "Our" voice to keep it
example-framed. Math/structure unchanged.

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

KEEP verbatim. Pure NumPy simulation, no topic dependency. No changes.

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

KEEP verbatim. Self-contained preview. No changes.

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

KEEP verbatim.

---

## Cell 19 - KEEP of OLD cell 17 (code) - Beat 3 continued: vectorised + heatmap

KEEP verbatim.

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

KEEP verbatim.

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

KEEP verbatim.

---

## Cell 38 - KEEP of OLD cell 36 (code) - Comparison of all three variants

KEEP verbatim.

---

## Cell 39 - EDIT of OLD cell 37 (markdown) - Discussion (3 minutes)

EDIT. Remove the parenthetical "(We will find out in Topic 3b.)" sequential
chaining in question 3. Reframe as an open question that does not depend on a
later topic.

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
complaint summariser" is fine; it is in-notebook narrative, not a topic reference.

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

to (placed AFTER the `_lab3_template` def so the name is bound):

```
my_scaled_dot_product_attention = _lab3_template
```

Keep the final `print(f"my_scaled_dot_product_attention is: ...")` line.

---

## Cell 42 - KEEP of OLD cell 40 (code) - Lab 3 verification

KEEP verbatim in both notebooks.

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

That companion is optional, not a required next step. The math is identical to
what you built here; the framework only adds automatic differentiation and GPU
execution. Read it if you are curious; skip it if you have what you came for.
```

---

## Cell 46 - KEEP of OLD cell 44 (markdown) - Homework Extensions header

KEEP verbatim.

---

## Cell 47 - KEEP of OLD cell 45 (code) - Homework Extension 1

KEEP verbatim in the Exercise (`pass  # YOUR IMPLEMENTATION` left as a stub by
design for homework).

SOLUTION FILL: leave as-is. Homework extension stubs are intentionally left
unsolved in both Exercise and Solution (they are async take-home work, not graded
lab cells). Do NOT fill these in the Solution.

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
| 0  | -  | NEW      | Optional/supplementary banner |
| 1  | -  | NEW      | "Why a user cares about attention" motivation |
| 2  | 0  | EDIT     | Trim title/objectives (moved to banner), drop Topic 3a |
| 3  | 1  | KEEP     | |
| 4  | 2  | KEEP     | |
| 5  | 3  | KEEP     | |
| 6  | 4  | KEEP     | |
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

# Verification checklist for the building agent

- [ ] grep the finished Exercise and Solution for "Topic 2", "Topic 3a",
      "Topic 3b": zero matches expected.
- [ ] grep for "carried forward": zero matches.
- [ ] grep for em dash, en dash, multiplication sign, emoji: zero matches.
- [ ] Cell 0 banner present and is the first cell.
- [ ] Cell 1 user-motivation present before Section 1.
- [ ] `COMPLAINT_TOKENS` defined locally with the standalone note.
- [ ] Exercise has 3 safety-net cells; Solution has 0.
- [ ] Solution lab cells (21, 30, 41) contain working code, no `# YOUR CODE`.
- [ ] Notebook runs top to bottom with no prior-topic state.
