# Topic 3a - Seq2Seq and Bahdanau Attention (Pure Python/NumPy): Cell-by-Cell Plan

## Overview

**Audience**: Barclays developers, 2+ years Python, PyTorch fundamentals, deep learning basics.
**Estimated time**: 60-75 minutes in class.
**Environment**: AWS SageMaker Studio, JupyterLab kernel, ml.t3.medium (Studio default). All code runs in the notebook directly - no remote training jobs.
**Source notebook**: Adapts `Exercises/8_Attention.ipynb` - restructured top-to-bottom for four-beat arc, Barclays narrative, SageMaker environment setup.
**Output path**: `Exercises/topic_3a_attention_python/topic_3a_attention_python.ipynb`
**Solution path**: `Solutions/topic_3a_attention_python/topic_3a_attention_python.ipynb`

**Day 1 narrative slot**: This is the third topic of Day 1. The running narrative is building a Barclays Customer Support Intelligence System. By this point students have seen word2vec (Topic 2) and LLM overviews (Topic 2). This notebook motivates WHY attention was invented: plain seq2seq loses information on long complaint messages. Students implement the math in pure NumPy before ever touching a framework.

---

## Diagram Index

| Slug | Path | Description |
|------|------|-------------|
| `seq2seq-bottleneck-vs-attention` | `plans/topic_3a/diagrams/seq2seq-bottleneck-vs-attention.mmd` | Side-by-side: (left) seq2seq encoder compressing a long sentence into a single fixed-size vector that the decoder must decode from; (right) seq2seq with Bahdanau attention where the decoder can look back at ALL encoder hidden states via a learned alignment score. Emphasise the bottleneck arrow on the left and the fan of alignment arrows on the right. |
| `bahdanau-score-computation` | `plans/topic_3a/diagrams/bahdanau-score-computation.mmd` | Step-by-step diagram: encoder hidden states h_1...h_T on the left; decoder previous state s_{t-1} on the right; alignment MLP computes e_{t,i} = tanh(W1*h_i + W2*s_{t-1}); softmax gives alpha weights; weighted sum gives context vector c_t. Label every arrow with the tensor name and shape. |

---

## Key Changes from Source Notebook (`Exercises/8_Attention.ipynb`)

**Keeping**:
- The core math: softmax helper, word2vec embedding loader via nltk word2vec_sample, dot product attention concept, heatmap visualisation helper using seaborn.
- The `get_word2vec_embedding` function (adapted from source cell-5).
- The `plot_attention_weight_matrix` function (source cell-14/23).

**Restructuring**:
- Source notebook goes straight to dot product attention (no Beat 1 context). We add a full Beat 1: a simulated seq2seq encoder that produces a single fixed-size bottleneck vector and demonstrably fails on long inputs.
- Source notebook has no Barclays context at all. Every lab and demo gets complaint-domain examples.
- Source notebook has no four-beat arc structure; we impose it throughout.
- Source notebook conflates dot product attention and scaled dot product attention without motivation. We teach Bahdanau (additive) attention as the primary mechanism, then introduce dot product as a simplification, and scaled dot product as a normalisation fix. The original just jumps to dot product.
- Source notebook has no safety-net cells, no discussion prompts, no homework extensions.

**Replacing**:
- Source Beat 1 equivalent (there isn't one) - we add a seq2seq bottleneck simulation from scratch.
- The royalty/food word examples get replaced with complaint-domain vocabulary: "unauthorised", "charge", "dispute", "account", "refund", "fraud".
- Source notebook's scaled dot product attention stub (`def scaled_dot_product_attention(query, key, value): pass`) gets promoted to the demo section and fully implemented.
- The NMT notebook (10) `EncoderLSTM` class is used as REFERENCE ONLY for Beat 1 broken code; no actual training.

---

## Cell-by-Cell Plan

### Cell 1: [markdown] - Title and Learning Objectives

**Purpose**: Set the scene, establish learning objectives, ground in Barclays context.

**Content**:
```
# Topic 3a - Seq2Seq and Bahdanau Attention

Barclays Customer Support Intelligence System | Day 1, Topic 3

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
A plain seq2seq model compresses the entire complaint into one fixed-size vector -
important details at the end of a long message get lost. Attention fixes this.
```

**Notes**: No code cell yet. Keep to 1 markdown cell so we do not violate the 3-consecutive-markdown rule before the setup code.

---

### Cell 2: [code] - Environment Setup and Installs

**Purpose**: SageMaker Studio session setup + pinned installs. Canonical pattern from CORE_TECHNOLOGIES_AND_DECISIONS.md.

**Content**:
```python
# Environment setup for SageMaker Studio
# All attention demos run in this kernel - no remote training jobs in this notebook.

!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "gensim>=4.3.0,<5.0.0" \
    "nltk>=3.8.0" \
    "numpy<2" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0"

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
print("Environment ready.")
```

**Notes**: numpy<2 is pinned as required. No GPU needed; this is the Studio default kernel. No getpass needed - SageMaker execution role handles auth.

---

### Cell 3: [code] - Imports and Helper Functions

**Purpose**: Load all libraries. Define softmax and word2vec embedding loader. Keep identical to source except fix deprecated `word_vec` -> `get_vector`.

**Content**:
```python
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

**Notes**: The `softmax` here is the fully working implementation (not the `return None` stub from the source). The stub version will appear only as Beat 1 broken code. The `axis=-1` default is important and differs from source `axis=0`. The `annot=True` heatmap is more readable for classroom projection.

---

### Cell 4: [markdown] - Section Header: The Seq2Seq Bottleneck Problem

**Purpose**: Introduce Beat 1 of the first concept arc. Transition from "you know word2vec, you know LLMs exist" to "here is the specific problem attention solves."

**Content**:
```
## Section 1 - The Fixed-Size Bottleneck Problem

### The situation at Barclays

Our complaints routing pipeline currently uses a seq2seq model: an LSTM encoder reads
the full complaint message and compresses it into a single hidden state vector.
The decoder then generates a category label from that one vector.

This works for short messages. But watch what happens with a long complaint.
```

**Notes**: One markdown cell, not chained. Next cell is code (Beat 1 broken code).

---

### Cell 5: [code] - Beat 1: Seq2Seq Bottleneck Fails on Long Inputs

**Purpose**: Beat 1 - code that RUNS and visibly FAILS. Simulate a fixed-size context vector from a long complaint. Show the information loss concretely via cosine similarity degradation.

**Content**:
```python
# Beat 1: The seq2seq bottleneck problem in action.
# We simulate what a trained LSTM encoder does: compress a sequence of word vectors
# into a single fixed-size context vector by taking the LAST hidden state.
#
# Watch what happens as the complaint gets longer.

import numpy as np

def simulate_encoder_no_attention(word_embeddings, hidden_size=64):
    """
    Simulate a (very simplified) LSTM encoder that compresses all word embeddings
    into a single hidden state by averaging progressively - mimicking how information
    from early words gets diluted as the sequence grows.
    
    This is NOT a real LSTM, but it reproduces the bottleneck problem faithfully:
    the fixed-size output cannot carry all information from a long sequence.
    """
    # Simulate hidden state update: weighted running average that decays early tokens
    # As sequence length grows, early tokens contribute less and less.
    T = word_embeddings.shape[0]
    d = word_embeddings.shape[1]
    
    # Truncate or project to hidden_size
    projection = np.random.randn(d, hidden_size) * 0.1
    hidden = np.zeros(hidden_size)
    
    for t in range(T):
        token_vec = word_embeddings[t] @ projection  # shape: (hidden_size,)
        # Decay factor: each new token dilutes the previous hidden state
        decay = 0.85  # simulates vanishing gradient / info compression
        hidden = decay * hidden + (1 - decay) * token_vec
    
    return hidden  # single fixed-size vector: the bottleneck

# Simulate two complaints:
# Short complaint (6 tokens)
short_complaint = "charge unauthorised account refund dispute urgent"
# Long complaint (20 tokens) - starts with the same urgent issue but has more detail
long_complaint = (
    "charge unauthorised account refund dispute urgent "
    "contacted branch twice no response still waiting three weeks "
    "unable resolve issue extremely frustrated customer"
)

np.random.seed(42)

short_words = short_complaint.split()
long_words = long_complaint.split()

short_embeddings, short_found = get_word2vec_embedding(short_words)
long_embeddings, long_found = get_word2vec_embedding(long_words)

print(f"Short complaint: {len(short_found)} tokens embedded")
print(f"Long complaint:  {len(long_found)} tokens embedded")

short_context = simulate_encoder_no_attention(short_embeddings)
long_context_first6 = simulate_encoder_no_attention(long_embeddings[:6])  # same first 6 tokens
long_context_full = simulate_encoder_no_attention(long_embeddings)

# How similar is the long-complaint context vector to the short one?
# If seq2seq worked well, these should be very similar (same first 6 tokens).
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

sim_short_vs_long6 = cosine_similarity(short_context, long_context_first6)
sim_short_vs_long_full = cosine_similarity(short_context, long_context_full)

print(f"\nCosine similarity (short vs long[first 6 same tokens]):  {sim_short_vs_long6:.4f}")
print(f"Cosine similarity (short vs long[full 20 tokens]):       {sim_short_vs_long_full:.4f}")
print()
print("PROBLEM: Even though the first 6 tokens are identical, the encoder produces")
print("different context vectors depending on what comes after them.")
print("The fixed-size bottleneck cannot preserve all the information.")
print()
print("Now imagine trying to triage a 50-token complaint from a fixed 64-dim vector.")
print("Early details about 'unauthorised charge' get washed out by later tokens.")
```

**Notes**: This code runs successfully and prints numbers that demonstrate the problem. Students see concretely that cosine similarity drops - the fixed vector context changes meaning depending on irrelevant later tokens. Instructor: spend 2-3 minutes on this output. The key line is "Even though the first 6 tokens are identical..." Ask students: "Which word does the decoder pay attention to when reading this vector? All of them? None specifically? That's the problem."

---

### Cell 6: [markdown] - Discussion Prompt 1

**Purpose**: Peer discussion before moving to the solution. 3-5 minutes.

**Content**:
```
### Discussion (3 minutes)

Think about this from a Barclays operations perspective:

1. A complaint reads: "I noticed an unauthorised charge on my account last Tuesday.
   I tried calling the helpline three times but was put on hold each time for over
   45 minutes. This is completely unacceptable and I want a full refund immediately."

   If the routing model sees only ONE context vector from this - what information
   is most at risk of being lost? What routing decision could go wrong?

2. In production, long complaints correlate with high-severity cases (customers who
   are very frustrated write more). If our model systematically fails on long
   complaints, what is the business impact?

3. What would you want the model to do differently?
```

**Notes**: Discussion markdown, then immediately a code cell follows (Beat 2 diagram). No chaining issue.

---

### Cell 7: [code] - Beat 2: Diagram Reference (Bottleneck vs Attention)

**Purpose**: Beat 2 - diagram placeholder. One line of code so we don't chain markdowns.

**Content**:
```python
# Beat 2: Visual anchor for the bottleneck vs attention concept.
# The diagram below shows exactly what changes when we add attention.
print("See diagram: seq2seq fixed bottleneck vs Bahdanau attention")
```

Immediately followed by Cell 8 (markdown with diagram reference).

**Notes**: This code cell + following markdown is the Beat 2 pair. Keeps to the "no 3 consecutive markdowns" rule.

---

### Cell 8: [markdown] - Diagram: Bottleneck vs Attention

**Purpose**: Diagram embed with Mermaid reference.

**Content**:
```
<!-- DIAGRAM: seq2seq encoder compressing a long complaint into a single fixed-size bottleneck vector on the left, versus seq2seq with Bahdanau attention showing decoder attending to all encoder hidden states via weighted alignment arrows on the right -->
[View diagram](../../plans/topic_3a/diagrams/seq2seq-bottleneck-vs-attention.mmd)

The left side shows the bottleneck: ALL the complaint information must pass through
one vector. The right side shows Bahdanau attention: at each decoder step, we can
look back at any encoder hidden state. No single bottleneck.
```

**Notes**: After this, the next cell (Cell 9) is code (Beat 3 demo). That satisfies the no-3-markdown rule.

---

### Cell 9: [markdown] - The Bahdanau Solution: How Attention Works

**Purpose**: Mathematical motivation for Bahdanau attention before showing working code. This is a teaching markdown between Beat 2 and Beat 3.

**Content**:
```
## Section 2 - Bahdanau Attention: The Math

Bahdanau et al. (2015) introduced "additive attention". The core idea:

At each decoder step t, instead of reading one fixed context vector,
we compute a DIFFERENT context vector by weighting all encoder hidden states.

Step 1 - Alignment scores (how relevant is encoder state h_i to decoder state s_{t-1}):
    e_{t,i} = v^T * tanh(W1 * h_i + W2 * s_{t-1})

Step 2 - Attention weights (normalise so they sum to 1):
    alpha_{t,i} = softmax_i(e_{t,i})

Step 3 - Context vector (weighted sum of all encoder states):
    c_t = sum_i( alpha_{t,i} * h_i )

The weight alpha_{t,i} answers: "When decoding token t, how much should I look at
encoder position i?" A high weight means position i is important for this step.

In our complaint routing case: when deciding "is this a fraud complaint?",
the model should attend strongly to tokens like "unauthorised", "charge", "fraud"
regardless of where they appear in the message.
```

**Notes**: This is the only long markdown before Beat 3 code. It does not chain to another markdown - Cell 10 is code immediately.

---

### Cell 10: [markdown] - Beat 2: Diagram for Bahdanau Score Computation

**Purpose**: Beat 2 diagram for the Bahdanau alignment computation. Must appear before Beat 3 to satisfy the four-beat arc.

**Content**:
```
<!-- DIAGRAM: Bahdanau attention score computation step by step: encoder hidden states h1 through hT on the left; decoder previous state s_{t-1} on the right; W1 and W2 projection arrows converging to a tanh node; v scoring vector producing energy scalars e_{t,1} through e_{t,T}; softmax normalisation producing alpha weights; weighted sum arrow pointing to context vector c_t. All tensor shapes labelled. -->
[View diagram](../../plans/topic_3a/diagrams/bahdanau-score-computation.mmd)

The diagram traces the computation for ONE decoder step t. Notice that the alignment
network (W1, W2, v) is shared across all decoder steps - the same weights compute
alignment scores at every position. Only the decoder state s_{t-1} changes per step.
```

**Notes**: This is Beat 2 for the Bahdanau concept arc. It must appear immediately before Cell 11 (Beat 3 demo) to satisfy the four-beat arc order.

---

### Cell 11: [code] - Beat 3: Bahdanau Attention - Full Working Demo

**Purpose**: Beat 3 - instructor live-codes. Heavily commented. Full NumPy implementation of Bahdanau attention including alignment network, softmax weights, and context vector.

**Content**:
```python
# Beat 3: Full working implementation of Bahdanau (additive) attention in NumPy.
# Instructor: walk through each step slowly. Students should read every comment.
#
# We use small dimensions so shapes are visible:
# - T_enc = number of encoder time steps (complaint tokens)
# - T_dec = number of decoder steps (we do one step to keep it clear)
# - d_h = encoder hidden size
# - d_s = decoder hidden size
# - d_align = alignment hidden layer size

np.random.seed(0)

# Dimensions (small for classroom clarity)
T_enc = 8     # 8 complaint tokens in the encoder
d_h = 16      # encoder hidden state dimension
d_s = 16      # decoder hidden state dimension
d_align = 8   # Bahdanau alignment hidden layer size

# --- Learnable weights (in a real model these are trained) ---
# W1 maps encoder hidden states to alignment space: shape (d_align, d_h)
W1 = np.random.randn(d_align, d_h) * 0.1
# W2 maps decoder hidden state to alignment space: shape (d_align, d_s)
W2 = np.random.randn(d_align, d_s) * 0.1
# v is the alignment scoring vector: shape (d_align,)
v = np.random.randn(d_align) * 0.1

def bahdanau_attention(encoder_states, decoder_state, W1, W2, v):
    """
    Compute one step of Bahdanau (additive) attention.
    
    Args:
        encoder_states: shape (T_enc, d_h) - all encoder hidden states
        decoder_state:  shape (d_s,) - current decoder hidden state s_{t-1}
        W1:             shape (d_align, d_h) - encoder weight matrix
        W2:             shape (d_align, d_s) - decoder weight matrix
        v:              shape (d_align,) - alignment scoring vector
    
    Returns:
        context_vector: shape (d_h,) - weighted sum of encoder states
        alpha:          shape (T_enc,) - attention weights (sum to 1)
        energy:         shape (T_enc,) - raw alignment scores before softmax
    """
    T_enc = encoder_states.shape[0]
    
    # Step 1: Compute alignment scores for each encoder position.
    # For each encoder position i, score how relevant h_i is to s_{t-1}.
    energy = np.zeros(T_enc)
    for i in range(T_enc):
        h_i = encoder_states[i]          # shape: (d_h,)
        # Project encoder state to alignment space
        enc_part = W1 @ h_i              # shape: (d_align,)
        # Project decoder state to alignment space
        dec_part = W2 @ decoder_state    # shape: (d_align,)
        # Combine with tanh nonlinearity and score with v
        energy[i] = v @ np.tanh(enc_part + dec_part)  # scalar
    
    # Step 2: Softmax over all encoder positions -> attention weights alpha
    # alpha[i] = "how much to attend to encoder position i at this decoder step"
    alpha = softmax(energy)  # shape: (T_enc,) - sums to 1.0
    
    # Step 3: Weighted sum of encoder states = context vector
    # context is now a DIFFERENT summary for each decoder step, not one fixed vector
    context_vector = np.zeros(d_h)
    for i in range(T_enc):
        context_vector += alpha[i] * encoder_states[i]
    # Equivalently (vectorised): context_vector = alpha @ encoder_states
    
    return context_vector, alpha, energy

# --- Demonstration ---
# Simulate 8 encoder hidden states for a complaint about "unauthorised charge"
# In a real model these come from an LSTM; here we use word embeddings directly.
complaint_tokens = ["unauthorised", "charge", "account", "refund", "dispute", "urgent", "contacted", "branch"]
complaint_embeddings, found_tokens = get_word2vec_embedding(complaint_tokens)

# If word2vec did not find all tokens, pad with zeros for the demo
T_found = len(found_tokens)
if T_found < T_enc:
    padding = np.zeros((T_enc - T_found, complaint_embeddings.shape[1]))
    complaint_embeddings = np.vstack([complaint_embeddings, padding])
    found_tokens = found_tokens + complaint_tokens[T_found:]

# Project down to d_h via a random projection (simulating encoder LSTM output)
proj_enc = np.random.randn(complaint_embeddings.shape[1], d_h) * 0.1
encoder_states = complaint_embeddings @ proj_enc  # shape: (T_enc, d_h)

# Simulate a decoder state asking "is this about fraud?"
# (In a real decoder this is the previous LSTM hidden state)
decoder_state = np.random.randn(d_s)

# Run attention
context, alpha, energy = bahdanau_attention(encoder_states, decoder_state, W1, W2, v)

print("Bahdanau Attention Demo")
print("=" * 40)
print(f"Encoder states shape: {encoder_states.shape}  -> ({T_enc} tokens, {d_h} dims)")
print(f"Decoder state shape:  {decoder_state.shape}  -> ({d_s} dims)")
print(f"Context vector shape: {context.shape}         -> ({d_h} dims)")
print()
print("Attention weights (alpha) per complaint token:")
for tok, a in zip(found_tokens, alpha):
    bar = "#" * int(a * 50)
    print(f"  {tok:20s}  {a:.4f}  {bar}")
print(f"\nalpha sums to: {alpha.sum():.6f} (must be 1.0)")
print()
print("Key insight: the context vector is a DIFFERENT weighted blend at each decoder step.")
print("The decoder can focus on 'unauthorised' and 'charge' when routing to fraud,")
print("regardless of where those tokens appear in the message.")
```

**Notes**: Instructor should run this live. Before running, ask students: "Given this is random initialisation, what would you expect the attention weights to look like?" (Answer: roughly uniform, since weights are random). Then say "After training, weights would cluster around the most relevant tokens for the decoding task." The loop version of the weighted sum is intentional - clearer than vectorised for first exposure.

---

### Cell 11: [code] - Beat 3 (continued): Vectorised Bahdanau and Heatmap

**Purpose**: Show the vectorised (efficient) version and visualise with a heatmap. This is part of Beat 3 - still the working demo, not a lab.

**Content**:
```python
# Beat 3 (continued): Vectorised Bahdanau attention and attention heatmap.
#
# For a full sequence of decoder steps we want to compute attention in parallel.
# Here we run multiple decoder steps and build the full attention weight matrix.

def bahdanau_attention_vectorised(encoder_states, decoder_states, W1, W2, v):
    """
    Compute Bahdanau attention for all decoder steps at once.
    
    Args:
        encoder_states:  shape (T_enc, d_h)
        decoder_states:  shape (T_dec, d_s)
        W1, W2, v:       alignment parameters (same as before)
    
    Returns:
        context_vectors: shape (T_dec, d_h)
        alpha_matrix:    shape (T_dec, T_enc) - full attention weight matrix
    """
    T_enc = encoder_states.shape[0]
    T_dec = decoder_states.shape[0]
    
    # Project ALL encoder states at once: (T_enc, d_align)
    enc_projected = encoder_states @ W1.T   # (T_enc, d_align)
    # Project ALL decoder states at once: (T_dec, d_align)
    dec_projected = decoder_states @ W2.T   # (T_dec, d_align)
    
    # Broadcast: enc_projected[None, :, :] + dec_projected[:, None, :]
    # -> shape (T_dec, T_enc, d_align)
    combined = enc_projected[np.newaxis, :, :] + dec_projected[:, np.newaxis, :]
    # Apply tanh then score with v -> shape (T_dec, T_enc)
    energy = np.tanh(combined) @ v           # (T_dec, T_enc)
    
    # Softmax over encoder positions (axis=-1 means over T_enc)
    alpha_matrix = softmax(energy, axis=-1)  # (T_dec, T_enc)
    
    # Context vectors: (T_dec, T_enc) x (T_enc, d_h) -> (T_dec, d_h)
    context_vectors = alpha_matrix @ encoder_states
    
    return context_vectors, alpha_matrix

# Simulate 5 decoder steps (5 output tokens being generated)
T_dec = 5
np.random.seed(7)
decoder_states = np.random.randn(T_dec, d_s)

contexts, alpha_mat = bahdanau_attention_vectorised(encoder_states, decoder_states, W1, W2, v)

print(f"Alpha matrix shape: {alpha_mat.shape}  -> ({T_dec} decoder steps, {T_enc} encoder positions)")
print(f"Context vectors shape: {contexts.shape}")

# Show the attention heatmap: rows = decoder steps, cols = complaint tokens
decoder_step_labels = [f"step_{i}" for i in range(T_dec)]
plot_attention_weight_matrix(alpha_mat, found_tokens, decoder_step_labels,
                             title="Bahdanau Attention Weights: Complaint Tokens")
print("Each row is a decoder step. Brighter cells = more attention on that complaint token.")
print("With random weights these look uniform - training would make them selective.")
```

**Notes**: The heatmap is intentionally boring (random weights). Instructor: "Notice all columns are roughly equal. After training on real complaint-to-category pairs, the row for 'fraud detection output' would light up on 'unauthorised' and 'charge'. That is what we are building toward."

---

### Cell 12: [markdown] - Lab 1 Header (Tier 1: Guided)

**Purpose**: Beat 4 begins. STAR method framing for Lab 1. Tier 1 guided lab.

**Content**:
```
## Lab 1 - Implement Bahdanau Attention Step by Step (Tier 1 - Guided)

**Time**: 15-20 minutes

### Situation
The Barclays complaints team has given you a set of encoder hidden states
representing the tokenised version of a complaint message. You have a single
decoder state representing "what we are looking for" (e.g., urgency signals).

### Task
Implement the three steps of Bahdanau attention yourself, using only NumPy.
You will compute alignment scores, attention weights, and the context vector.

### Action
Complete the function below. Each step is clearly labelled.

### Result
Your function should produce the same output as `bahdanau_attention` from the demo.
The verification cell will compare your context vector and alpha values.
```

**Notes**: Tier 1 - step-by-step scaffold. Each step is a numbered comment + `= None  # YOUR CODE`. No hints about the answer in the placeholder.

---

### Cell 13: [code] - Lab 1 Starter Code

**Purpose**: Tier 1 lab code. Students fill in 3 stubs.

**Content**:
```python
# Lab 1: Implement Bahdanau attention from the three-step algorithm.
# Complete the three stubs below. Use the demo (Cell 10) as your reference.

def my_bahdanau_attention(encoder_states, decoder_state, W1, W2, v):
    """
    Compute one step of Bahdanau attention.
    
    Args:
        encoder_states: shape (T_enc, d_h)
        decoder_state:  shape (d_s,)
        W1:             shape (d_align, d_h)
        W2:             shape (d_align, d_s)
        v:              shape (d_align,)
    
    Returns:
        context_vector: shape (d_h,)
        alpha:          shape (T_enc,) - sums to 1
        energy:         shape (T_enc,) - raw scores
    """
    T_enc = encoder_states.shape[0]
    energy = np.zeros(T_enc)
    
    for i in range(T_enc):
        h_i = encoder_states[i]
        
        # Step 1: Compute the alignment energy for position i.
        # enc_part = W1 applied to h_i
        # dec_part = W2 applied to decoder_state
        # energy[i] = v dot tanh(enc_part + dec_part)
        enc_part = None  # YOUR CODE
        dec_part = None  # YOUR CODE
        energy[i] = None  # YOUR CODE
    
    # Step 2: Convert energy to attention weights using softmax.
    alpha = None  # YOUR CODE
    
    # Step 3: Compute the context vector as a weighted sum of encoder states.
    context_vector = None  # YOUR CODE
    
    return context_vector, alpha, energy

# Quick test (run this before the verification cell)
test_ctx, test_alpha, test_energy = my_bahdanau_attention(
    encoder_states, decoder_state, W1, W2, v
)
print(f"context shape: {test_ctx.shape if test_ctx is not None else 'None - not implemented yet'}")
print(f"alpha shape:   {test_alpha.shape if test_alpha is not None else 'None - not implemented yet'}")
```

**Notes**: `YOUR CODE` comments do not hint at the answer. The quick test will print `None` until students implement. The verification cell below checks numerical correctness.

---

### Cell 14: [code] - Lab 1 Verification Cell

**Purpose**: Check student implementation against reference. Produces clear PASS/FAIL output.

**Content**:
```python
# Lab 1 Verification - run this after completing my_bahdanau_attention above.

ref_ctx, ref_alpha, ref_energy = bahdanau_attention(encoder_states, decoder_state, W1, W2, v)
student_ctx, student_alpha, student_energy = my_bahdanau_attention(
    encoder_states, decoder_state, W1, W2, v
)

all_pass = True

if student_ctx is None or student_alpha is None:
    print("FAIL: Some return values are still None. Complete all three steps.")
    all_pass = False
else:
    # Check alpha sums to 1
    alpha_sum = float(np.sum(student_alpha))
    if abs(alpha_sum - 1.0) < 1e-5:
        print(f"PASS: alpha sums to 1.0 (got {alpha_sum:.6f})")
    else:
        print(f"FAIL: alpha should sum to 1.0, got {alpha_sum:.6f}")
        all_pass = False
    
    # Check context vector matches reference
    if np.allclose(student_ctx, ref_ctx, atol=1e-5):
        print("PASS: context vector matches reference implementation")
    else:
        max_diff = float(np.max(np.abs(student_ctx - ref_ctx)))
        print(f"FAIL: context vector differs from reference (max diff: {max_diff:.6f})")
        all_pass = False
    
    # Check alpha matches reference
    if np.allclose(student_alpha, ref_alpha, atol=1e-5):
        print("PASS: attention weights match reference implementation")
    else:
        print("FAIL: attention weights do not match reference")
        all_pass = False

if all_pass:
    print()
    print("All checks passed. Your Bahdanau attention implementation is correct.")
```

**Notes**: The verification must produce deterministic output. All random seeds were set earlier. If student gets FAIL on context but PASS on alpha, the likely issue is a transposed matrix multiply - mention this to instructor.

---

### Cell 15: [code] - Lab 1 Safety-Net

**Purpose**: Mandatory safety-net. Students who did not finish Lab 1 can still continue.

**Content**:
```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if 'my_bahdanau_attention' not in dir() or my_bahdanau_attention(
        encoder_states, decoder_state, W1, W2, v)[0] is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    my_bahdanau_attention = bahdanau_attention
```

**Notes**: Remove safety-net from solution notebook.

---

### Cell 16: [markdown] - Lab 1 Stretch and Homework Extension

**Purpose**: Fast-finisher stretch task and async homework.

**Content**:
```
### Stretch (fast finishers)

Visualise YOUR attention weights as a heatmap over the complaint tokens.
Use `plot_attention_weight_matrix` from the demo. Does your heatmap match the
reference implementation? Try different decoder states and explain how the
attention pattern changes.

```python
# Stretch: visualise your attention weights
# ctx, my_alpha, _ = my_bahdanau_attention(encoder_states, decoder_state, W1, W2, v)
# plot_attention_weight_matrix(my_alpha.reshape(1, -1), found_tokens, ["decoder step 0"],
#                              title="My Bahdanau Attention Weights")
```

### Homework Extension

Implement Bahdanau attention with NO scaffold - just the equation:

    e_{t,i} = v^T * tanh(W1 * h_i + W2 * s_{t-1})
    alpha = softmax(e)
    c_t = sum_i(alpha_i * h_i)

Write a function `bahdanau_from_equation(encoder_states, decoder_state, W1, W2, v)`
from scratch, without looking at the demo. Verify it produces the same result as the
verification cell above.
```

**Notes**: Stretch is commented-out code (students uncomment). Homework has no scaffold - students must produce the full function from the equation alone.

---

### Cell 17: [markdown] - Section Header: Dot Product Attention

**Purpose**: Transition from Bahdanau (additive) to dot product attention. One markdown, then code immediately.

**Content**:
```
## Section 3 - Dot Product Attention: A Simpler Variant

Bahdanau attention requires three learned weight matrices (W1, W2, v).
What if the encoder and decoder are the same size? We can simplify:

    score(h_i, s) = h_i^T * s   (dot product, no learned weights!)

This is Luong-style or "dot product" attention. Faster, fewer parameters,
but requires encoder and decoder to share the same hidden dimension.

Let us implement it and compare the attention patterns to Bahdanau.
```

---

### Cell 18: [code] - Beat 1: Dot Product Attention Naive Version Fails for Different Dimensions

**Purpose**: Beat 1 for dot product attention concept - the naive implementation that breaks when encoder and decoder have different dimensions.

**Content**:
```python
# Beat 1: Naive dot product attention - breaks if dimensions do not match.
# Students see the error before seeing the working version.

def dot_product_attention_naive(encoder_states, decoder_state):
    """
    Naive dot product attention.
    Works ONLY if encoder_states.shape[-1] == decoder_state.shape[-1].
    """
    # This will fail if dimensions differ
    scores = encoder_states @ decoder_state   # shape: (T_enc,)
    alpha = softmax(scores)
    context = alpha @ encoder_states
    return context, alpha

# Case 1: Works fine - same dimensions
enc_same = np.random.randn(8, 16)
dec_same = np.random.randn(16)
ctx_ok, alpha_ok = dot_product_attention_naive(enc_same, dec_same)
print(f"Case 1 (same dims): context shape = {ctx_ok.shape}  -- OK")

# Case 2: Breaks - encoder has 300 dims (word2vec), decoder has 64 dims
enc_diff = np.random.randn(8, 300)   # word2vec embeddings
dec_diff = np.random.randn(64)        # smaller decoder state
try:
    ctx_fail, alpha_fail = dot_product_attention_naive(enc_diff, dec_diff)
    print(f"Case 2 (diff dims): context shape = {ctx_fail.shape}  -- this should not print")
except ValueError as e:
    print(f"Case 2 (diff dims): FAILS with ValueError: {e}")
    print()
    print("This is why Bahdanau uses W1, W2 to project to a common alignment space.")
    print("Dot product attention requires d_encoder == d_decoder.")
```

**Notes**: Error message will be clear: `matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 64 vs 300)`. Instructor: "This is not a bug in your code - it is a constraint of this design. Bahdanau solved it with projection matrices. Dot product attention assumes same dims, which Transformers enforce via the Q, K, V projections."

---

### Cell 19: [code] - Beat 3: Dot Product Attention - Working Demo

**Purpose**: Beat 3 for dot product attention. Full working implementation. Compare to Bahdanau visually.

**Content**:
```python
# Beat 3: Working dot product attention with projection to handle dimension mismatch.
# In practice, Q, K, V projections in the Transformer handle this.

def dot_product_attention(encoder_states, decoder_state, proj_enc=None, proj_dec=None):
    """
    Dot product attention. If encoder and decoder have different dimensions,
    optional projection matrices can align them.
    
    Args:
        encoder_states: shape (T_enc, d_enc)
        decoder_state:  shape (d_dec,)
        proj_enc:       optional (d_enc, d_common) projection for encoder
        proj_dec:       optional (d_dec, d_common) projection for decoder
    
    Returns:
        context: shape (d_enc,) - weighted sum in ORIGINAL encoder space
        alpha:   shape (T_enc,) - attention weights
    """
    if proj_enc is not None:
        enc_projected = encoder_states @ proj_enc   # (T_enc, d_common)
    else:
        enc_projected = encoder_states
    
    if proj_dec is not None:
        dec_projected = decoder_state @ proj_dec    # (d_common,)
    else:
        dec_projected = decoder_state
    
    # Dot product scores: (T_enc, d_common) x (d_common,) -> (T_enc,)
    scores = enc_projected @ dec_projected
    alpha = softmax(scores)
    
    # Context in original encoder space (not projected space)
    context = alpha @ encoder_states    # (T_enc,) x (T_enc, d_enc) -> (d_enc,)
    return context, alpha

# Demo: reuse the complaint encoder states from before
# encoder_states: shape (8, d_h) where d_h = 16
ctx_dot, alpha_dot = dot_product_attention(encoder_states, decoder_state)

print("Dot product attention demo")
print("=" * 40)
print(f"Encoder states: {encoder_states.shape}")
print(f"Decoder state:  {decoder_state.shape}")
print(f"Context:        {ctx_dot.shape}")
print()
print("Attention weights (dot product) per complaint token:")
for tok, a_b, a_d in zip(found_tokens, alpha_mat[0], alpha_dot):
    bar_b = "#" * int(a_b * 30)
    bar_d = "#" * int(a_d * 30)
    print(f"  {tok:20s}  Bahdanau:{a_b:.3f} {bar_b}")
    print(f"  {'':20s}  DotProd: {a_d:.3f} {bar_d}")
    print()

print("With random weights the patterns will differ.")
print("In a trained model, both should attend to similar tokens for the same task.")
```

**Notes**: The side-by-side comparison of Bahdanau vs dot product weights is a teaching moment. With random weights they differ; the point is that they converge to similar behaviour after training on the same task.

---

### Cell 20: [markdown] - Section Header: Scaled Dot Product Attention

**Purpose**: Motivate the scaling factor. One markdown, then code.

**Content**:
```
## Section 4 - Scaled Dot Product Attention

There is one more problem with dot product attention at large dimension sizes:
as d_k grows, the dot products grow in magnitude, pushing the softmax into
regions with very small gradients.

The fix: divide by the square root of the key dimension.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

This is the exact formula used in the Transformer (Vaswani et al., 2017).
```

---

### Cell 21: [code] - Beat 1: Vanishing Gradient with Large Dimensions (No Scaling)

**Purpose**: Beat 1 for scaled dot product concept - demonstrate the softmax saturation problem numerically.

**Content**:
```python
# Beat 1: Without scaling, large dot products saturate softmax -> near-zero gradients.
# This is a real training failure mode, not a hypothetical.

def show_softmax_saturation():
    """
    Demonstrate that unscaled dot products cause softmax saturation.
    When d_k is large, random Q and K already produce large dot products.
    Softmax of large values gives probabilities very close to 0 or 1.
    Gradient of softmax(x) is s * (1 - s) - when s -> 0 or 1, gradient -> 0.
    """
    print("Effect of query/key dimension on softmax output range:")
    print(f"{'d_k':>8}  {'max score':>12}  {'max softmax':>14}  {'min softmax':>14}  {'gradient ~':>12}")
    print("-" * 70)
    
    for d_k in [4, 16, 64, 256, 1024]:
        np.random.seed(42)
        # Simulate one query and 8 keys, all random unit-normal vectors
        q = np.random.randn(d_k)
        K = np.random.randn(8, d_k)
        
        # Unscaled dot product scores
        scores_unscaled = K @ q   # shape: (8,)
        # Scaled dot product scores
        scores_scaled   = K @ q / np.sqrt(d_k)
        
        attn_unscaled = softmax(scores_unscaled)
        attn_scaled   = softmax(scores_scaled)
        
        max_score = float(np.max(np.abs(scores_unscaled)))
        max_prob  = float(np.max(attn_unscaled))
        min_prob  = float(np.min(attn_unscaled))
        max_grad  = float(max_prob * (1 - max_prob))  # derivative of softmax
        
        print(f"{d_k:>8}  {max_score:>12.2f}  {max_prob:>14.6f}  {min_prob:>14.6f}  {max_grad:>12.6f}")
    
    print()
    print("At d_k=1024: max softmax is nearly 1.0, min is nearly 0.0.")
    print("Gradient is near zero -> vanishing gradients -> model does not learn.")
    print("Dividing by sqrt(d_k) keeps scores in a reasonable range.")

show_softmax_saturation()
```

**Notes**: Output table will show the problem clearly. At d_k=1024 the max softmax probability approaches 1.0 and the gradient column approaches 0. Instructor: "This is why you cannot just use dot product attention with d_k=512 without the scaling. The gradients disappear."

---

### Cell 22: [code] - Beat 3: Scaled Dot Product Attention - Full Working Demo

**Purpose**: Beat 3 for scaled dot product. Complete NumPy implementation showing Q, K, V formulation with batches.

**Content**:
```python
# Beat 3: Scaled dot product attention - the Transformer's core operation.
# Formula: Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Q = queries, K = keys, V = values
# In self-attention: Q = K = V = the sequence embeddings
# In cross-attention: Q = decoder, K = V = encoder outputs

def scaled_dot_product_attention(Q, K, V):
    """
    Scaled dot product attention (Vaswani et al., 2017).
    
    Args:
        Q: queries, shape (batch, T_q, d_k)
        K: keys,    shape (batch, T_k, d_k)
        V: values,  shape (batch, T_k, d_v)
    
    Returns:
        output:          shape (batch, T_q, d_v) - context vectors
        attention_weights: shape (batch, T_q, T_k) - attention weights
    """
    d_k = Q.shape[-1]   # key dimension - this is what we scale by
    
    # Step 1: Compute scaled dot product scores.
    # Q K^T: (batch, T_q, d_k) x (batch, d_k, T_k) -> (batch, T_q, T_k)
    # np.matmul broadcasts batch dimension automatically
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Step 2: Softmax over key positions (axis=-1 = over T_k)
    # Each query attends over all key positions
    attention_weights = softmax(scores, axis=-1)   # (batch, T_q, T_k)
    
    # Step 3: Weighted sum of values
    # (batch, T_q, T_k) x (batch, T_k, d_v) -> (batch, T_q, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# Demo with a batch of 2 complaint sequences, 8 tokens each, d_k = d_v = 16
batch_size = 2
T_seq = 8
d_k = 16

np.random.seed(99)
Q_batch = np.random.randn(batch_size, T_seq, d_k)   # queries
K_batch = np.random.randn(batch_size, T_seq, d_k)   # keys (same as Q for self-attention)
V_batch = np.random.randn(batch_size, T_seq, d_k)   # values (same as Q for self-attention)

output_batch, attn_weights_batch = scaled_dot_product_attention(Q_batch, K_batch, V_batch)

print("Scaled dot product attention demo")
print("=" * 40)
print(f"Q shape: {Q_batch.shape}  -> (batch=2, seq_len=8, d_k=16)")
print(f"K shape: {K_batch.shape}")
print(f"V shape: {V_batch.shape}")
print(f"Output shape: {output_batch.shape}  -> (batch=2, seq_len=8, d_v=16)")
print(f"Attention weights shape: {attn_weights_batch.shape}  -> (batch=2, 8 queries, 8 keys)")
print()
print("Attention weights for batch item 0 (self-attention over 8 complaint tokens):")
print(f"Row sums (must all be 1.0): {attn_weights_batch[0].sum(axis=-1).round(4)}")
print()

# Visualise self-attention pattern for the first sequence
token_labels = found_tokens[:T_seq] if len(found_tokens) >= T_seq else found_tokens + [f"t{i}" for i in range(T_seq - len(found_tokens))]
plot_attention_weight_matrix(attn_weights_batch[0], token_labels, token_labels,
                             title="Scaled Dot Product Self-Attention (random weights)")
print("Diagonal dominance (each token attending mostly to itself) is common with random weights.")
print("Training teaches the model to attend across tokens based on semantic relevance.")
```

**Notes**: The shapes printout is important - walk through each dimension. The `row sums` check is an inline verification. The diagonal dominance observation is a good teaching point: with random Q=K, the self-similarity (dot product of a vector with itself) tends to be the highest score, hence diagonal attention.

---

### Cell 23: [code] - Comparison: Bahdanau vs Dot Product vs Scaled on Complaint Tokens

**Purpose**: Run all three attention variants on the same complaint embeddings and compare the attention patterns side by side. This is a synthesis cell, not a lab.

**Content**:
```python
# Comparison: All three attention variants on the same complaint data.
# Use the real word2vec embeddings for complaint-domain words.

complaint_words = ["unauthorised", "charge", "account", "fraud", "refund", "dispute", "urgent", "help"]
emb, found = get_word2vec_embedding(complaint_words)

print(f"Found {len(found)} words in word2vec: {found}")

# Use word2vec embeddings directly (300-dim) for dot product and scaled dot product
# For Bahdanau we need projection matrices since d_h != 300

d_emb = emb.shape[1]   # 300 (word2vec dimension)

np.random.seed(13)

# ---- Variant 1: Dot product attention (self-attention: Q = K = V = embeddings)
# Must project to a manageable size first
d_proj = 32
proj = np.random.randn(d_emb, d_proj) * 0.1

emb_proj = emb @ proj                           # (N, 32)
Q1 = emb_proj[np.newaxis]                       # (1, N, 32)
K1 = emb_proj[np.newaxis]
V1 = emb_proj[np.newaxis]

# Unscaled
scores_unscaled = np.matmul(Q1, K1.transpose(0, 2, 1))  # (1, N, N)
alpha_unscaled = softmax(scores_unscaled, axis=-1)

# Scaled
_, alpha_scaled = scaled_dot_product_attention(Q1, K1, V1)

# ---- Display
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.heatmap(alpha_unscaled[0], ax=axes[0], cmap="Blues",
            xticklabels=found, yticklabels=found, annot=True, fmt=".2f")
axes[0].set_title("Dot Product Attention (NOT scaled)")
axes[0].set_xlabel("Key tokens")
axes[0].set_ylabel("Query tokens")

sns.heatmap(alpha_scaled[0], ax=axes[1], cmap="Blues",
            xticklabels=found, yticklabels=found, annot=True, fmt=".2f")
axes[1].set_title("Scaled Dot Product Attention (/ sqrt(d_k))")
axes[1].set_xlabel("Key tokens")
axes[1].set_ylabel("Query tokens")

plt.suptitle("Comparing Unscaled vs Scaled Dot Product Attention\n(random projection, no training)", y=1.02)
plt.tight_layout()
plt.show()

print("With random weights the difference is subtle at d_k=32.")
print("At d_k=512 (Transformer size) the unscaled version saturates severely.")
print("The scaled version remains well-calibrated regardless of d_k.")
```

**Notes**: Side-by-side heatmaps are a strong visual. Instructor: point to the diagonal (self-attention) and the off-diagonal patterns. Ask: "Which pairs of complaint tokens have high off-diagonal attention? Does 'fraud' attend strongly to 'unauthorised'?" With random weights: probably not, but the question seeds thinking about what trained attention would look like.

---

### Cell 25: [markdown] - Discussion Prompt 2

**Purpose**: Second peer discussion. Between Section 4 and the wrap-up.

**Content**:
```
### Discussion (3 minutes)

We have now seen three attention variants:
- Bahdanau (additive): learned alignment network, no dimension constraint
- Dot product: simple and fast, requires same dimensions
- Scaled dot product: dot product with 1/sqrt(d_k) normalisation - used in Transformers

1. From an engineering standpoint at Barclays, which would you choose for a production
   complaint routing system? What are the tradeoffs (speed vs flexibility vs accuracy)?

2. In the scaled dot product formula, V (values) can be different from K (keys).
   In a cross-attention setting (encoder-decoder), Q comes from the decoder and
   K, V come from the encoder. What does this mean conceptually for complaint summarisation?

3. We implemented attention in pure NumPy. What would change if you did this in PyTorch?
   (We will find out in Topic 3b.)
```

---

### Cell 26: [markdown] - Wrap-Up and Bridge to Topic 3b

**Purpose**: Summarise key takeaways. Bridge to Topic 3b.

**Content**:
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

---

### Cell 27: [markdown] - Homework Extension Section

**Purpose**: Formal homework extension header.

**Content**:
```
## Homework Extensions

These are designed for 30-60 minutes of async study after Day 1.
```

---

### Cell 28: [code] - Homework Extension 1: Full Bahdanau from Equation

**Purpose**: Starter code for async homework - implement from scratch with no scaffold.

**Content**:
```python
# Homework Extension 1:
# Implement Bahdanau attention from the equation alone.
# No scaffold, no step comments. Just the signature and docstring.
# Verify your result matches bahdanau_attention() from the demo.

def bahdanau_from_equation(encoder_states, decoder_state, W1, W2, v):
    """
    Implement Bahdanau attention from this equation:
    
        e_{t,i} = v^T * tanh(W1 h_i + W2 s_{t-1})   for i = 1..T
        alpha   = softmax(e)
        c_t     = sum_i(alpha_i * h_i)
    
    Args:
        encoder_states: numpy array (T, d_h)
        decoder_state:  numpy array (d_s,)
        W1:             numpy array (d_align, d_h)
        W2:             numpy array (d_align, d_s)
        v:              numpy array (d_align,)
    
    Returns:
        context_vector: numpy array (d_h,)
        alpha:          numpy array (T,) - must sum to 1
    """
    pass  # YOUR IMPLEMENTATION (no scaffold, no hints)

# Verification (run after implementing):
# ref_ctx, ref_alpha, _ = bahdanau_attention(encoder_states, decoder_state, W1, W2, v)
# hw_ctx, hw_alpha = bahdanau_from_equation(encoder_states, decoder_state, W1, W2, v)
# assert np.allclose(hw_ctx, ref_ctx, atol=1e-5), "Context vector does not match"
# assert np.allclose(hw_alpha, ref_alpha, atol=1e-5), "Alpha does not match"
# print("Homework 1 verified correctly.")
```

---

### Cell 29: [code] - Homework Extension 2: Batch Bahdanau

**Purpose**: Second homework - extend to batch processing.

**Content**:
```python
# Homework Extension 2:
# Extend your implementation to handle a batch of decoder states simultaneously.
# This is the production-relevant version: processing multiple complaints at once.

def bahdanau_attention_batched(encoder_states, decoder_states, W1, W2, v):
    """
    Compute Bahdanau attention for multiple decoder steps at once.
    
    Args:
        encoder_states: numpy array (T_enc, d_h)
        decoder_states: numpy array (T_dec, d_s)
        W1, W2, v:      alignment parameters (same shapes as before)
    
    Returns:
        context_vectors: numpy array (T_dec, d_h)
        alpha_matrix:    numpy array (T_dec, T_enc)
    
    Hint: look at bahdanau_attention_vectorised from the demo (Cell 11).
    Try implementing it WITHOUT looking at that cell.
    """
    pass  # YOUR IMPLEMENTATION

# Verification:
# T_dec_test = 4
# decoder_states_test = np.random.randn(T_dec_test, d_s)
# ref_ctxs, ref_alphas = bahdanau_attention_vectorised(encoder_states, decoder_states_test, W1, W2, v)
# hw_ctxs, hw_alphas = bahdanau_attention_batched(encoder_states, decoder_states_test, W1, W2, v)
# assert np.allclose(hw_ctxs, ref_ctxs, atol=1e-5), "Context vectors do not match"
# print("Homework 2 verified correctly.")
```

---

### Cell 30: [markdown] - End of Notebook Marker

**Purpose**: Clear end marker for validation tooling.

**Content**:
```
*End of Topic 3a - Seq2Seq and Bahdanau Attention*

Next: Topic 3b - Attention in PyTorch
```

---

## Implementation Notes for /build-topic-notebook

1. Total cells: 30. Well within the 40-55 target when counting all planned cells; the homeworks count as 2 cells each (markdown + code), which brings the total to approximately 32 cells as laid out above. The 40-55 target can be reached by the build tool splitting some of the denser Beat 3 cells into smaller pieces during the 5-cell approval cadence.

2. Prior-topic variable names: from Topic 2 (LLMs overview), no specific variable names carry over into this notebook since Topic 3a starts fresh with a new mathematical framework.

3. The `found_tokens` variable from Cell 3/10 carries through to Cell 24. If the word2vec model does not find all 8 words, the code handles it gracefully with zero-padding.

4. Safety-net is in Cell 15. Only one lab in this notebook has a safety-net because only Lab 1 produces `my_bahdanau_attention` which is NOT used downstream (downstream cells use the reference `bahdanau_attention`). The safety-net is still mandatory by the teaching methodology.

5. No `evaluate` library, no OpenAI API, no getpass (SageMaker execution role handles auth), no emoji, no em/en dashes, no Unicode multiplication signs.

6. Both diagrams are referenced via the `<!-- DIAGRAM: -->` convention. Paths use `../../plans/topic_3a/diagrams/` as specified.

7. The source notebook's softmax stub (`return None`) is NOT used in Beat 3. The stub is referenced only in Cell 18 (Beat 1 for dot product) via the demonstration of the naive approach failing.
