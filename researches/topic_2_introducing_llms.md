# Topic 2 - Introducing LLMs: Cell-by-Cell Plan

## Narrative and Audience

**Course context**: This is Day 1, Topic 2. Students have completed Topic 1 (Overview of Generative AI) and know conceptually what generative AI is and why Barclays wants an LLM for the customer complaint handling system. Topic 2 answers the natural next question: "We chose an LLM -- but which kind? And what does it actually do under the hood?"

**Audience**: Developers with 2+ years Python, PyTorch fundamentals, and basic deep learning. They are NOT beginners. They know what a neural network is but have likely never dissected a transformer in code.

**Narrative thread (Day 1)**: "Building a Barclays Customer Support Intelligence System." In this topic students learn what an LLM is architecturally so that in Topics 3 and 4 they can build attention and a transformer from scratch, and in Day 2 they can fine-tune one. The concluding bridge is: "We know what LLMs are. In Topics 3 and 4 we build the attention and transformer machinery from scratch. Then in Day 2 we train them on Barclays data."

**Estimated time**: 65-75 minutes in class.
- Setup + tokenization concept + Lab 1: ~20 min
- Embeddings concept + Lab 2 (Tier 2): ~30 min
- Transformer families diagram + inference demo: ~10 min
- GenAI lifecycle + wrap-up: ~10 min

---

## Diagram Index

| # | Slug | Path | Description |
|---|------|------|-------------|
| 1 | `transformer-families` | `plans/topic_2/diagrams/transformer-families.mmd` | Three transformer architecture families side-by-side: encoder-only (BERT style, tasks: classification, NER, embeddings), decoder-only (GPT style, tasks: generation, completion), encoder-decoder (T5 style, tasks: translation, summarization). Arrows show data flow through each. |
| 2 | `genai-lifecycle` | `plans/topic_2/diagrams/genai-lifecycle.mmd` | Two swim-lane comparison: Traditional ML lifecycle (data -> feature engineering -> train from scratch -> evaluate -> deploy) vs GenAI lifecycle (data -> select pretrained model -> prompt engineering / fine-tune / RAG -> evaluate -> deploy + monitor). Key difference labels: "no training from scratch", "model selection is a first-class step", "alignment and safety gate". |

---

## Cell-by-Cell Plan

---

### Cell 1: [type: markdown] - Title and Learning Objectives

**Purpose**: Orient students; state clearly what they will be able to do after this topic.

**Content**:
```markdown
# Topic 2 - Introducing LLMs

## What lies inside ChatGPT?

By the end of this topic you will be able to:
- Explain the three transformer architecture families (encoder, decoder, encoder-decoder) and
  choose the right one for a given task
- Tokenize a real Barclays complaint text, inspect token IDs, and decode them back
- Get dense embeddings from a pretrained model and measure semantic similarity between complaints
- Run inference with a small pretrained model (distilbert-base-uncased, distilgpt2) directly
  in the Studio notebook kernel
- Describe the Generative AI project lifecycle and how it differs from traditional ML

## Context: Our Barclays Complaint System

In Topic 1 we decided we want an LLM to handle customer complaints intelligently.
Now we need to answer: which kind of LLM? And what does "LLM" actually mean in code?

We will work with real-looking Barclays complaint text throughout this topic.
No training in this topic -- pure inference and exploration.
```

**Notes**: Read objectives aloud with students. Emphasize "no training today -- we are surgeons examining the model, not farmers growing one." Timing: 1 min.

---

### Cell 2: [type: code] - Environment Setup and Installs

**Purpose**: Pin all required libraries, import everything once. Fail fast here rather than mid-demo.

**Content**:
```python
# Environment setup -- run once at the start of every notebook.
# Do NOT skip this cell. Pinned versions are required for SageMaker Studio compatibility.

!pip install -q \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "datasets>=2.18.0,<3.0.0" \
    "numpy<2" \
    "scikit-learn>=1.3.0,<2.0.0"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline,
    DistilBertTokenizer,
    DistilBertModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from sklearn.metrics.pairwise import cosine_similarity

print("torch version:", torch.__version__)
print("numpy version:", np.__version__)
print("Setup complete.")
```

**Notes**: The install takes ~60 seconds on ml.t3.medium. Tell students to run it and move on while it completes. No SageMaker session needed for this topic -- all inference is local in the kernel. No getpass because all models are public on HuggingFace Hub. Timing: 2 min (run-and-forget).

---

### Cell 3: [type: markdown] - Section 1 Header: What Is a Token?

**Purpose**: Introduce the first concept with the narrative hook.

**Content**:
```markdown
## Section 1 -- What Is a Token?

Before any LLM sees text, it converts it to numbers.
That conversion is called **tokenization**.

The Barclays complaint system receives thousands of messages like:

> "I've been charged twice for the same transaction on my Barclaycard. This is unacceptable
>  and I need a refund immediately."

To an LLM, that sentence is just a sequence of integers. Let's see how.

### Beat 1 -- The Naive Approach (and Why It Fails)

Suppose we just split on spaces and look up words in a dictionary.
What happens with unknown words, contractions, or rare financial terms?
```

**Notes**: Read the complaint aloud. Ask students "how would YOU convert this to numbers?" before showing the naive code. Timing: 1 min.

---

### Cell 4: [type: code] - Beat 1: Naive Word-Split Tokenization (Broken)

**Purpose**: Students feel the pain of simple word splitting -- coverage gaps, OOV words, contractions.

**Content**:
```python
# Beat 1 -- Naive tokenization: split on spaces, assign sequential IDs.
# This is the WRONG approach. Run it and see what breaks.

complaint = (
    "I've been charged twice for the same transaction on my Barclaycard. "
    "This is unacceptable and I need a refund immediately."
)

# Build a tiny vocabulary from the complaint itself
words = complaint.split()
vocab = {word: idx for idx, word in enumerate(sorted(set(words)))}
print("Naive vocabulary size:", len(vocab))
print("Vocabulary:", vocab)

# Tokenize
naive_tokens = [vocab.get(word, -1) for word in words]
print("\nNaive token IDs:", naive_tokens)

# Problem 1: a new complaint uses words not in our tiny vocab
new_complaint = "My Barclaycard was fraudulently used at an ATM in Manchester."
new_tokens = [vocab.get(word, -1) for word in new_complaint.split()]
print("\nNew complaint token IDs:", new_tokens)
print("Number of UNKNOWN tokens (-1):", new_tokens.count(-1), "out of", len(new_tokens))

# Problem 2: contractions are not split -- "I've" is treated as one unit
print("\nDoes vocab contain 'I'?", "I" in vocab)
print("Does vocab contain \"I've\"?", "I've" in vocab)
print("Real models split I've into: ['I', \"'\", 've'] or ['I', \"'ve']")
```

**Expected output**: ~50% of new_complaint tokens are -1 (unknown). "I've" is one unit not two. Vocabulary is tiny and not reusable. Students see the problem clearly.

**Notes**: Let students read the output. Ask "what would happen if Barclays gets a complaint in Welsh or with a typo?" This motivates subword tokenization. Timing: 3 min.

---

### Cell 5: [type: markdown] - Beat 2: Diagram Placeholder

**Purpose**: Diagram anchor for transformer families (shown slightly early as a visual teaser; revisited in Section 3).

**Content**:
```markdown
### Beat 2 -- How Real LLMs Handle Tokenization

Real transformer models use **subword tokenization** (Byte-Pair Encoding for GPT-style,
WordPiece for BERT-style). Instead of a dictionary of whole words, they have a fixed
vocabulary of ~30,000 common subwords. Unknown words are split into known pieces.

"unacceptable" -> ["un", "##accept", "##able"] (BERT WordPiece)
"Barclaycard"  -> ["Bar", "clay", "card"] (GPT BPE)

This means the vocabulary is fixed and finite, but the model can handle ANY text --
including new financial product names -- by splitting them into known subpieces.

<!-- DIAGRAM: transformer-families -->
[View diagram](../../plans/topic_2/diagrams/transformer-families.mmd)

The three transformer families all share this subword tokenization front-end.
They differ in what happens AFTER the tokens become numbers.
```

**Notes**: Point to the diagram path. The diagram will be generated by /build-diagrams. Use the 30 seconds to let students copy the complaint text to think about it. Timing: 1 min.

---

### Cell 6: [type: code] - Beat 3: Real Tokenization with DistilBERT

**Purpose**: Full working demo of WordPiece tokenization; instructor live-codes commentary.

**Content**:
```python
# Beat 3 -- Real tokenization with DistilBERT (WordPiece, BERT-style).
# DistilBERT uses a ~30,000 word vocabulary of subwords.
# It is a smaller (66M param) distilled version of BERT -- perfect for demos.

print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print(f"Vocabulary size: {tokenizer.vocab_size:,} subwords")

# The same Barclays complaint from the naive demo
complaint = (
    "I've been charged twice for the same transaction on my Barclaycard. "
    "This is unacceptable and I need a refund immediately."
)

# Step 1: Tokenize -- converts text to a dict with input_ids and attention_mask
encoded = tokenizer(complaint, return_tensors="pt")
print("\n--- Encoded output ---")
print("Keys:", list(encoded.keys()))
print("input_ids shape:", encoded["input_ids"].shape)  # [1, num_tokens]
print("attention_mask shape:", encoded["attention_mask"].shape)

# Step 2: Inspect the token IDs
token_ids = encoded["input_ids"][0].tolist()
print("\nToken IDs:", token_ids)
print("Number of tokens:", len(token_ids))

# Step 3: Convert IDs back to human-readable subwords
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("\nSubwords:", tokens)
# Note: [CLS] at start, [SEP] at end -- special BERT bookend tokens
# ## prefix means "continuation of previous subword"

# Step 4: Decode back to original text
decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
print("\nDecoded back:", decoded)
print("Round-trip identical:", decoded.strip().lower() == complaint.strip().lower())

# Step 5: Show how an unknown financial term is split
rare_term = "Barclaycard fraudulently overdraft"
rare_encoded = tokenizer(rare_term)
rare_tokens = tokenizer.convert_ids_to_tokens(rare_encoded["input_ids"])
print("\nRare financial terms tokenized:", rare_tokens)
```

**Expected output**: ~28-32 tokens for the complaint. Students see [CLS] and [SEP] bookend tokens. "unacceptable" splits into ["un", "##acceptable"]. "Barclaycard" may split into ["bar", "##clay", "##card"]. Round-trip decode is identical. Rare terms are covered by subpieces.

**Notes**: Walk through each print step. Ask "why does [CLS] appear?" -- answer: BERT uses [CLS] representation for classification. "Why does ## appear?" -- continuation of previous piece. Timing: 5 min including discussion.

---

### Cell 7: [type: markdown] - Lab 1 Instructions (Tier 1, Guided)

**Purpose**: Tokenization lab with STAR framing.

**Content**:
```markdown
## Lab 1 -- Tokenization Explorer (Tier 1 Guided, ~15 min)

**Situation**: The Barclays complaint triage team wants to understand why certain messages
are flagged as "too long" by the LLM system. The system has a 512-token limit.
They need a simple function that counts tokens and warns when a complaint is near the limit.

**Task**: Build a complaint tokenizer that:
1. Tokenizes a complaint using distilbert-base-uncased
2. Returns the list of subword tokens
3. Returns the token count
4. Prints a warning if count is above 400 (leaves headroom before the 512 limit)

**Action**: Fill in the YOUR CODE sections below.

**Result**: Running the verification cell will show token counts for three test complaints
and flag the long one correctly.

### Steps
1. Call `tokenizer(complaint, return_tensors="pt")` to encode
2. Use `.convert_ids_to_tokens(...)` to get human-readable subwords
3. Count how many tokens were produced
4. Compare count to the 400-token threshold
```

**Notes**: Keep the lab framing concise. Students already saw all the building blocks in Beat 3. Timing cue for instructor: "You have 15 minutes. I will walk around. There is a stretch task if you finish early." Timing: 1 min to read.

---

### Cell 8: [type: code] - Lab 1 Starter Code

**Purpose**: Guided lab with YOUR CODE placeholders.

**Content**:
```python
# Lab 1 -- Tokenization Explorer
# The tokenizer is already loaded above as `tokenizer`.
# Do NOT reload it here.

def analyze_complaint_tokens(complaint_text, threshold=400):
    """
    Tokenize a complaint and return analysis results.

    Parameters
    ----------
    complaint_text : str
        Raw complaint text from a Barclays customer.
    threshold : int
        Token count above which to print a warning.

    Returns
    -------
    dict with keys: tokens (list[str]), count (int), over_limit (bool)
    """
    # Step 1: Encode the complaint
    encoded = None  # YOUR CODE

    # Step 2: Get the token IDs as a plain Python list
    token_ids = None  # YOUR CODE

    # Step 3: Convert token IDs to human-readable subwords
    tokens = None  # YOUR CODE

    # Step 4: Count the tokens
    count = None  # YOUR CODE

    # Step 5: Check if count exceeds threshold
    over_limit = None  # YOUR CODE

    if over_limit:
        print(f"WARNING: complaint has {count} tokens (limit is {threshold}). Consider truncating.")

    return {"tokens": tokens, "count": count, "over_limit": over_limit}


# Test complaints
short_complaint = "My card was declined at a supermarket."
medium_complaint = (
    "I have been a loyal Barclays customer for 15 years. Last Tuesday I attempted "
    "to make a payment of 2,500 pounds to my solicitor for a property purchase. "
    "The payment was blocked without any notification and I am now facing penalties. "
    "I need this resolved urgently or I will escalate to the Financial Ombudsman."
)
long_complaint = " ".join(["I am extremely dissatisfied with the service."] * 25)

result_short  = None  # YOUR CODE: call analyze_complaint_tokens on short_complaint
result_medium = None  # YOUR CODE: call analyze_complaint_tokens on medium_complaint
result_long   = None  # YOUR CODE: call analyze_complaint_tokens on long_complaint
```

**Notes**: The three `None  # YOUR CODE` at the bottom are intentionally open -- students must know to call the function. No hints about arguments.

---

### Cell 9: [type: code] - Lab 1 Verification

**Purpose**: Automated check of lab results.

**Content**:
```python
# Lab 1 verification -- run after completing the lab
assert result_short is not None, "result_short is still None -- did you call analyze_complaint_tokens?"
assert result_medium is not None, "result_medium is still None"
assert result_long is not None, "result_long is still None"

assert isinstance(result_short["tokens"], list), "tokens must be a list of strings"
assert result_short["count"] > 0, "count must be positive"
assert result_short["over_limit"] == False, "short complaint should not be over limit"
assert result_long["over_limit"] == True, "long complaint (25 repetitions) should be over limit"

print("Lab 1 passed!")
print(f"  short complaint:  {result_short['count']} tokens, over_limit={result_short['over_limit']}")
print(f"  medium complaint: {result_medium['count']} tokens, over_limit={result_medium['over_limit']}")
print(f"  long complaint:   {result_long['count']} tokens, over_limit={result_long['over_limit']}")
```

**Notes**: If students hit assertion errors, the message tells them exactly what is wrong. Timing: auto.

---

### Cell 10: [type: code] - Lab 1 Safety Net

**Purpose**: Students who did not finish Lab 1 can continue with the rest of the notebook.

**Content**:
```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if result_short is None or result_medium is None or result_long is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")

    def analyze_complaint_tokens(complaint_text, threshold=400):
        enc = tokenizer(complaint_text, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        count = len(tokens)
        over_limit = count > threshold
        if over_limit:
            print(f"WARNING: {count} tokens (limit {threshold}).")
        return {"tokens": tokens, "count": count, "over_limit": over_limit}

    result_short  = analyze_complaint_tokens(short_complaint)
    result_medium = analyze_complaint_tokens(medium_complaint)
    result_long   = analyze_complaint_tokens(long_complaint)
    print("Safety-net complete.")
```

**Notes**: Remove this cell from the solution notebook.

---

### Cell 11: [type: markdown] - Lab 1 Stretch and Homework Extension

**Purpose**: Keep fast finishers busy; set up async homework.

**Content**:
```markdown
### Lab 1 Stretch (for fast finishers)

Compare tokenization between DistilBERT (WordPiece) and DistilGPT2 (BPE) on the same complaint.
- Load `GPT2Tokenizer.from_pretrained("distilgpt2")`
- Tokenize the medium_complaint with both tokenizers
- Compare: which produces more tokens? Which subwords look different?
- Why does GPT2 tokenizer NOT add [CLS] and [SEP]?

### Homework Extension

Build a token budget checker for a batch of complaints loaded from a list.
Given 20 complaints, produce a summary table with columns:
  complaint_id | token_count | over_limit | first_10_tokens
Use `pandas.DataFrame` for the table.
Identify: what is the longest complaint in the batch? What is the average token count?
This is the kind of preprocessing check the Barclays data team would run before fine-tuning.
```

**Notes**: The stretch is doable in ~5 min with the pattern already shown. Homework is a real-world task. Timing: 0 min (read independently).

---

### Cell 12: [type: markdown] - Peer Discussion 1

**Purpose**: 3-minute discussion break; connect tokenization to production concerns.

**Content**:
```markdown
## Peer Discussion (3 min)

With the person next to you:

1. A customer pastes their complaint in Welsh. The model was trained mostly on English.
   What do you predict happens to the token count? What might happen to the output quality?

2. Barclays wants to process 10,000 complaints per hour through the LLM.
   Each complaint costs money proportional to its token count (input + output tokens).
   How would you use what you just built to estimate and control that cost?

3. Why does it matter that we can DECODE token IDs back to text?
   Think about debugging and compliance audit trails.
```

**Notes**: 3 minutes. Call on 1-2 pairs to share. Do not skip this -- it connects code to professional context which motivates the next section. Timing: 3 min.

---

### Cell 13: [type: markdown] - Section 2 Header: Embeddings and Semantic Similarity

**Purpose**: Transition from tokenization (IDs) to embeddings (vectors).

**Content**:
```markdown
## Section 2 -- From Tokens to Meaning: Embeddings

Token IDs are just integers -- 2345 does not "mean" anything on its own.
The transformer converts those integers into **dense vectors** (embeddings) that
encode semantic meaning.

The magic: two complaint sentences that mean the same thing will have
**similar embedding vectors**, even if they share no words.

"I was charged twice." and "A duplicate payment was taken." should be close together.
"I love the Barclays app." should be far away from both.

### Beat 1 -- Naive Text Comparison Without Embeddings (Broken)

Suppose we try to find "similar complaints" using string matching.
```

**Notes**: Ask "how do you currently find duplicate complaints in a system without an LLM?" Elicit answers (keyword search, regex). Then show why that fails. Timing: 1 min.

---

### Cell 14: [type: code] - Beat 1: Naive String Matching (Broken)

**Purpose**: Students see that exact-match and simple overlap fail for semantic similarity.

**Content**:
```python
# Beat 1 -- Naive similarity: exact string match and word-overlap (Jaccard).
# This is WRONG for semantic similarity. Run it and see the failures.

def jaccard_similarity(text_a, text_b):
    """Word overlap between two texts. Range [0, 1]."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0

# Three complaint pairs
pairs = [
    # Should be HIGH similarity (same meaning, different words)
    ("I was charged twice for the same payment.",
     "A duplicate transaction appeared on my account."),
    # Should be LOW similarity (different topics)
    ("I was charged twice for the same payment.",
     "My mortgage application has been delayed by three weeks."),
    # Should be MEDIUM similarity (same topic, very different wording)
    ("I've been a victim of card fraud.",
     "Someone made unauthorized purchases with my Barclaycard."),
]

print("Jaccard (word overlap) similarity scores:")
print("-" * 70)
for a, b in pairs:
    score = jaccard_similarity(a, b)
    print(f"Score: {score:.3f}")
    print(f"  A: {a}")
    print(f"  B: {b}")
    print()

# Expected: all three scores are LOW (around 0.05-0.15) because the words differ.
# Pair 1 SHOULD be high but is low -- Jaccard fails completely.
print("PROBLEM: Pair 1 should be highest (same meaning) but Jaccard scores it low.")
print("Jaccard only sees word overlap, not meaning.")
```

**Expected output**: All three Jaccard scores cluster around 0.05-0.20 regardless of semantic similarity. Pair 1 (duplicate payment, same meaning) scores similarly to pair 2 (different topic). Students see the failure clearly.

**Notes**: This is the "pain moment." Ask "what would happen if Barclays used this to route complaints?" -- answer: identical complaints would go to different queues. Timing: 3 min.

---

### Cell 15: [type: markdown] - Beat 2: Diagram Placeholder for GenAI Lifecycle

**Purpose**: Diagram anchor for GenAI lifecycle (second of two diagrams).

**Content**:
```markdown
### Beat 2 -- How Embeddings Capture Meaning

A transformer encoder processes a full sentence and produces a single dense vector
(typically 768 numbers for BERT-base, 384 for DistilBERT-base).

Words that appear in similar contexts during pretraining end up in similar regions
of this 768-dimensional space. Two sentences about duplicate charges will be
"near" each other even with different words.

<!-- DIAGRAM: genai-lifecycle -->
[View diagram](../../plans/topic_2/diagrams/genai-lifecycle.mmd)

The diagram shows the GenAI project lifecycle -- we will come back to this in Section 4.
For now, notice that "embeddings" and "semantic search" are standard production steps
in the GenAI workflow, not research novelties.
```

**Notes**: The diagram reference here is intentional -- we will discuss the lifecycle in detail in Section 4. This is a visual preview to connect the embedding work to the real product workflow. Timing: 1 min.

---

### Cell 16: [type: code] - Beat 3: Real Embeddings with DistilBERT

**Purpose**: Full working demo of mean-pooled sentence embeddings and cosine similarity.

**Content**:
```python
# Beat 3 -- Real sentence embeddings using DistilBERT encoder.
# DistilBERT is an encoder-only model: it reads the WHOLE sentence bidirectionally.
# We extract the [CLS] token embedding as the sentence-level representation.

print("Loading DistilBERT model (66M parameters)...")
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()  # inference mode -- no gradient tracking needed
print("Model loaded.")

def get_embedding(text):
    """
    Encode a sentence and return a 768-dim numpy embedding vector.
    Uses mean pooling across all token positions (more stable than CLS alone).
    """
    # Tokenize with padding/truncation
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # outputs.last_hidden_state shape: [1, seq_len, 768]
    # Mean pool across the sequence dimension (ignore padding via attention_mask)
    attention_mask = inputs["attention_mask"]  # [1, seq_len]
    token_embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
    # Expand mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze().numpy()
    return embedding  # shape: (768,)


# Get embeddings for the same three pairs
sentences = [
    "I was charged twice for the same payment.",
    "A duplicate transaction appeared on my account.",
    "My mortgage application has been delayed by three weeks.",
    "Someone made unauthorized purchases with my Barclaycard.",
    "I've been a victim of card fraud.",
    "I love the new Barclays mobile app design.",
]

print("\nGenerating embeddings (this takes a few seconds)...")
embeddings = [get_embedding(s) for s in sentences]
embeddings_matrix = np.stack(embeddings)  # shape: (6, 768)
print(f"Embeddings shape: {embeddings_matrix.shape}")

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(embeddings_matrix)

print("\nCosine similarity matrix (higher = more similar):")
print("Sentences:")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")
print()
print("Similarity matrix (rows x cols = sentence indices):")
print(np.round(sim_matrix, 3))

# Highlight the interesting pairs
print("\nKey pairs:")
print(f"  [0] vs [1] (duplicate charge -- SHOULD be high): {sim_matrix[0,1]:.3f}")
print(f"  [0] vs [2] (mortgage -- SHOULD be low):          {sim_matrix[0,2]:.3f}")
print(f"  [3] vs [4] (card fraud -- SHOULD be high):       {sim_matrix[3,4]:.3f}")
print(f"  [0] vs [5] (mobile app -- SHOULD be low):        {sim_matrix[0,5]:.3f}")
```

**Expected output**: Pair [0]-[1] (same meaning, duplicate charge) cosine ~0.85-0.93. Pair [0]-[2] (different topic) cosine ~0.45-0.60. Pair [3]-[4] (card fraud, same topic) cosine ~0.88-0.95. Pair [0]-[5] (app feedback) cosine ~0.50-0.65. Clear semantic clustering visible.

**Notes**: Walk through `get_embedding` line by line. Point out: "this is JUST forward-pass inference -- no training, no gradients." Key teaching moment: `with torch.no_grad()` is correct for inference. Explain mean pooling briefly. Timing: 6 min including walkthrough.

---

### Cell 17: [type: markdown] - Lab 2 Instructions (Tier 2, Hard)

**Purpose**: Embedding similarity lab -- the Tier 2 (hard) lab for Day 1.

**Content**:
```markdown
## Lab 2 -- Complaint Similarity Matrix (Tier 2 Hard, ~25 min)

**Situation**: The Barclays complaint routing team has a dataset of 10 complaints
across different categories: card fraud, duplicate charges, mortgage delays,
and app feedback. They want to automatically identify which complaints are
"the same issue described differently" so they can route them to the same specialist.

**Task**: Build a similarity search function that:
1. Takes a list of complaint texts
2. Computes DistilBERT embeddings for all of them
3. Builds a full pairwise cosine similarity matrix
4. Finds the single most similar pair in the dataset (excluding self-similarity on the diagonal)
5. Finds the single least similar pair

**Action**: This is a Tier 2 lab -- the steps are less prescriptive. Use the `get_embedding`
function defined above. Use `cosine_similarity` from sklearn (already imported).
NumPy operations for finding the best pair: `np.fill_diagonal`, `np.unravel_index`, `np.argmax`.

**Result**: Print the most similar pair and least similar pair with their cosine scores.
The most similar pair should be two complaints that describe card fraud in different words.
```

**Notes**: No numbered steps -- students must figure out the numpy index operations on their own. This distinguishes Tier 2 from Tier 1. Timing: 25 min. Walk around. Timing: 1 min to read instructions.

---

### Cell 18: [type: code] - Lab 2 Starter Code

**Purpose**: Tier 2 lab with minimal scaffolding.

**Content**:
```python
# Lab 2 -- Complaint Similarity Matrix (Tier 2)
# Use get_embedding() and cosine_similarity() from above.

complaints_dataset = [
    "I was charged twice for the same transaction.",
    "A duplicate payment was debited from my current account.",
    "Someone used my card without my permission.",
    "Fraudulent transactions appeared on my statement last week.",
    "My mortgage application has been stuck for six weeks.",
    "No one has updated me on my home loan status.",
    "The Barclays app keeps crashing when I try to log in.",
    "Your mobile app is completely broken on my Android phone.",
    "I need to increase my overdraft limit but no one is responding.",
    "I've been trying to raise my overdraft for two months.",
]

# Step 1: Compute embeddings for all complaints
# Hint: use a list comprehension with get_embedding
all_embeddings = None  # YOUR CODE

# Step 2: Stack into a matrix of shape (10, 768)
embeddings_matrix = None  # YOUR CODE

# Step 3: Compute pairwise cosine similarity matrix of shape (10, 10)
sim_matrix = None  # YOUR CODE

# Step 4: Zero out the diagonal (self-similarity = 1.0 should not count as "most similar pair")
# Hint: np.fill_diagonal modifies in place
# YOUR CODE

# Step 5: Find the index of the most similar pair
# Hint: np.argmax on the flattened matrix, then np.unravel_index to get (row, col)
most_similar_idx = None  # YOUR CODE

# Step 6: Find the index of the least similar pair
least_similar_idx = None  # YOUR CODE

# Step 7: Print results
if most_similar_idx is not None and least_similar_idx is not None:
    i, j = most_similar_idx
    print("Most similar pair:")
    print(f"  [{i}] {complaints_dataset[i]}")
    print(f"  [{j}] {complaints_dataset[j]}")
    print(f"  Cosine similarity: {sim_matrix[i, j]:.4f}")
    print()
    p, q = least_similar_idx
    print("Least similar pair:")
    print(f"  [{p}] {complaints_dataset[p]}")
    print(f"  [{q}] {complaints_dataset[q]}")
    print(f"  Cosine similarity: {sim_matrix[p, q]:.4f}")
```

**Notes**: `# YOUR CODE` blocks are at the right level of abstraction for Tier 2 -- they know WHAT to do but not HOW (which numpy calls to chain). No hints about unravel_index -- that is the discovery. Timing: 25 min for students.

---

### Cell 19: [type: code] - Lab 2 Verification

**Purpose**: Automated assertion check.

**Content**:
```python
# Lab 2 verification
assert all_embeddings is not None, "all_embeddings is still None"
assert embeddings_matrix is not None, "embeddings_matrix is still None"
assert sim_matrix is not None, "sim_matrix is still None"

assert len(all_embeddings) == 10, f"Expected 10 embeddings, got {len(all_embeddings)}"
assert embeddings_matrix.shape == (10, 768), f"Expected (10, 768), got {embeddings_matrix.shape}"
assert sim_matrix.shape == (10, 10), f"Expected (10, 10), got {sim_matrix.shape}"

# Diagonal should be 0.0 (zeroed out) not 1.0
assert abs(sim_matrix[0, 0]) < 0.01, "Diagonal was not zeroed out"

# Most similar pair should be within the same complaint category
i, j = most_similar_idx
max_score = sim_matrix[i, j]
assert max_score > 0.80, f"Most similar pair score {max_score:.3f} is suspiciously low. Check embeddings."

print("Lab 2 passed!")
print(f"Most similar pair score: {max_score:.4f}")
print(f"Complaint [{i}] and complaint [{j}] are semantically closest.")
```

**Notes**: The threshold 0.80 is conservative -- DistilBERT typically gives 0.88+ for paraphrase pairs. If students get 0.70 they likely forgot mean pooling or made a shape error.

---

### Cell 20: [type: code] - Lab 2 Safety Net

**Purpose**: Safety net for students who did not finish Lab 2.

**Content**:
```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.
if all_embeddings is None or sim_matrix is None or most_similar_idx is None:
    print("Using Lab 2 safety-net so the rest of the notebook can run.")

    all_embeddings = [get_embedding(c) for c in complaints_dataset]
    embeddings_matrix = np.stack(all_embeddings)
    sim_matrix = cosine_similarity(embeddings_matrix)
    np.fill_diagonal(sim_matrix, 0.0)

    flat_idx = np.argmax(sim_matrix)
    most_similar_idx = np.unravel_index(flat_idx, sim_matrix.shape)

    np.fill_diagonal(sim_matrix, 1.0)  # restore diagonal for least-similar search
    flat_idx_min = np.argmin(sim_matrix)
    least_similar_idx = np.unravel_index(flat_idx_min, sim_matrix.shape)
    np.fill_diagonal(sim_matrix, 0.0)

    i, j = most_similar_idx
    print(f"Most similar: [{i}] and [{j}], score={sim_matrix[i,j]:.4f}")
    print("Safety-net complete.")
```

**Notes**: Remove from solution notebook.

---

### Cell 21: [type: markdown] - Lab 2 Stretch and Homework Extension

**Purpose**: Fast-finisher stretch and async homework.

**Content**:
```markdown
### Lab 2 Stretch (for fast finishers)

Replace `DistilBertModel` with a sentence-transformers model:
```python
# pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("all-MiniLM-L6-v2")
st_embeddings = st_model.encode(complaints_dataset)
```
Compare the cosine similarity matrix from sentence-transformers vs DistilBERT mean pooling.
Which one produces higher similarity scores for paraphrase pairs?
Why are sentence-transformers models better for semantic similarity? (Hint: they are
trained with a contrastive loss designed for similarity, not masked language modeling.)

### Homework Extension

Build a complaint lookup function:
Given a new complaint text and a dataset of 50 historical complaints (you can repeat
the 10 above 5 times with slight variations), return the top 3 most similar complaints
with their cosine scores. This is the foundation of a retrieval-augmented support system:
find similar resolved complaints to generate a suggested resolution for the new one.

Deliverable: a function `find_similar_complaints(query, dataset, top_k=3) -> list[dict]`
where each dict has keys: text, score, rank.
```

**Notes**: Sentence-transformers is not in the pinned requirements; students must install it. This is intentional -- the stretch teaches dependency management too. Timing: 0 min.

---

### Cell 22: [type: markdown] - Peer Discussion 2

**Purpose**: Connect embeddings to production architecture before moving to transformer families.

**Content**:
```markdown
## Peer Discussion (3 min)

1. You just built the core of a "semantic search" system using embeddings.
   At Barclays scale (100,000 complaints per month), what are the computational
   bottlenecks? Where would you add caching? (Think: do embeddings change if the
   model does not change?)

2. DistilBERT is 66M parameters. GPT-3 is 175B parameters.
   Why might a SMALLER model sometimes be BETTER for production embedding search?
   Think about latency, cost, and batch throughput.

3. We used mean pooling to collapse a [1, seq_len, 768] tensor to [768].
   What information might we lose? What if two complaints are identical except
   one mentions "urgent" at the start -- will the embedding capture that?
```

**Notes**: Question 3 is a deliberate foreshadowing of positional encoding and attention, which students will build in Topics 3 and 4. Timing: 3 min.

---

### Cell 23: [type: markdown] - Section 3 Header: Transformer Families

**Purpose**: Transition to the taxonomy of transformer architectures.

**Content**:
```markdown
## Section 3 -- The Three Transformer Families

Now we understand tokenization and embeddings.
The deeper question: what happens inside the model between tokens-in and embeddings-out?

All transformers share the same tokenization front-end and the same self-attention
building block. They differ in HOW they wire those blocks together.

There are exactly three families that matter in practice:

| Family | Architecture | Trained to do | Famous models |
|--------|-------------|----------------|---------------|
| Encoder-only | Reads full sequence bidirectionally | Understand text | BERT, DistilBERT, RoBERTa |
| Decoder-only | Reads left-to-right, generates tokens | Generate text | GPT-2, GPT-4, LLaMA, Gemini |
| Encoder-decoder | Encodes input, decodes output | Transform text | T5, Flan-T5, BART |

For the Barclays complaint system:
- Classifying complaint category -> encoder-only (BERT-style)
- Generating a draft response -> decoder-only (GPT-style)
- Summarizing a long complaint -> encoder-decoder (T5-style)

### Beat 1 -- Using the Wrong Architecture for the Task (Broken)

What happens if we try to use a GPT-style (decoder-only) model for classification?
```

**Notes**: This table is the conceptual anchor of the entire topic. Have students copy it or point to it repeatedly in the rest of the session. Timing: 2 min.

---

### Cell 24: [type: code] - Beat 1: GPT Used for Classification (Broken)

**Purpose**: Students see decoder-only model produce nonsense when asked to classify.

**Content**:
```python
# Beat 1 -- Using distilgpt2 (decoder-only) for complaint classification.
# GPT models are trained to PREDICT THE NEXT TOKEN, not classify sequences.
# Watch what happens when we try to use it the wrong way.

print("Loading distilgpt2 (decoder-only GPT-style model)...")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # GPT2 has no pad token by default

# The "wrong" way: feed a complaint and ask for a classification label
complaint_to_classify = "I was charged twice for the same transaction. Please refund immediately."

# GPT will just continue the text -- it has no concept of "output a label"
gpt_inputs = gpt_tokenizer(complaint_to_classify, return_tensors="pt")

gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt_model.eval()

with torch.no_grad():
    output_ids = gpt_model.generate(
        gpt_inputs["input_ids"],
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=gpt_tokenizer.eos_token_id,
    )

generated_text = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\nInput complaint:")
print(f"  {complaint_to_classify}")
print("\nGPT-2 output (just continued the text):")
print(f"  {generated_text}")
print()
print("PROBLEM: GPT did not classify. It continued the sequence with more text.")
print("Decoder-only models are autoregressive text GENERATORS, not classifiers.")
print("For classification, we need an ENCODER-only model (BERT-style).")
```

**Expected output**: GPT will continue the complaint text with generic phrases like "I was charged twice for the same transaction. Please refund immediately. I have been a customer..." It does NOT output "billing dispute" or any category label.

**Notes**: This is the key architectural insight. The failure is clear and non-subtle. After running: "GPT doesn't know what 'classify' means. It only knows 'what comes next?' That's why we have three families." Timing: 4 min.

---

### Cell 25: [type: code] - Beat 3: Right Model for the Right Task (Working)

**Purpose**: Show the correct use of each architecture family via HuggingFace pipelines.

**Content**:
```python
# Beat 3 -- Right model for each task using HuggingFace pipelines.
# Pipelines abstract the tokenize -> model -> postprocess steps.

# Task 1: Classification (encoder-only model)
print("=== Task 1: Complaint Classification (encoder-only) ===")
# Using a zero-shot classifier backed by a cross-encoder (encoder-decoder, but behaves like encoder for classification)
# For pure encoder demo, we use the DistilBERT fill-mask as a proxy
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",  # encoder-only, fine-tuned on NLI
)
complaint = "I was charged twice for the same transaction and need a refund."
labels = ["billing dispute", "fraud", "mortgage", "app issue", "general inquiry"]
result = classifier(complaint, candidate_labels=labels)
print(f"Complaint: {complaint}")
print("Predicted labels (highest score = best match):")
for label, score in zip(result["labels"][:3], result["scores"][:3]):
    print(f"  {label}: {score:.3f}")

print()

# Task 2: Text Generation (decoder-only model)
print("=== Task 2: Draft Response Generation (decoder-only) ===")
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=40,
    do_sample=False,
    pad_token_id=50256,  # GPT2 EOS token ID
)
prompt = "Dear Barclays customer, thank you for contacting us regarding your duplicate charge."
gen_result = generator(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {gen_result[0]['generated_text']}")

print()

# Task 3: Summarization (encoder-decoder model)
print("=== Task 3: Complaint Summarization (encoder-decoder) ===")
# Note: T5 is too large for ml.t3.medium; use sshleifer/distilbart-cnn-6-6 (distilled BART)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    max_length=60,
    min_length=20,
)
long_complaint = (
    "I am writing to express my deep dissatisfaction with the handling of my mortgage "
    "application. I submitted all required documents over six weeks ago. Despite multiple "
    "phone calls and emails, I have received no update on the status. I am now at risk of "
    "losing my property purchase because the seller is withdrawing. This is causing me "
    "significant financial and emotional distress."
)
summary_result = summarizer(long_complaint)
print(f"Original ({len(long_complaint.split())} words):")
print(f"  {long_complaint}")
print(f"\nSummary ({len(summary_result[0]['summary_text'].split())} words):")
print(f"  {summary_result[0]['summary_text']}")
```

**Notes**: The `typeform/distilbert-base-uncased-mnli` model is ~268MB -- acceptable for ml.t3.medium. `distilbart-cnn-6-6` is ~306MB. Combined with DistilBERT already loaded, total memory is ~900MB, well within 4GB. Warn students that the first pipeline() call triggers a model download (~30 seconds). Timing: 8 min including download wait.

---

### Cell 26: [type: markdown] - Beat 4: Lab 3 Reference (No Separate Lab for Section 3)

**Purpose**: Lab 3 is intentionally skipped -- two labs are enough for Day 1 timing. Instead, provide a guided exploration cell.

**Content**:
```markdown
## Section 3 Exploration -- Architecture Decision Exercise

No separate lab for Section 3 (Day 1 time budget).
Instead, work through the decision table below mentally and verify with the code cell.

For each Barclays use case, which transformer family would you choose?

| Use Case | Encoder-only | Decoder-only | Encoder-decoder | Your choice |
|----------|-------------|-------------|----------------|-------------|
| Route complaint to correct department | ? | ? | ? | |
| Generate a polite acknowledgment email | ? | ? | ? | |
| Summarize a 500-word complaint to 3 sentences | ? | ? | ? | |
| Find complaints similar to a new one (embedding search) | ? | ? | ? | |
| Extract named entities (account numbers, dates) | ? | ? | ? | |
| Translate a complaint from Welsh to English | ? | ? | ? | |

Discuss with your neighbor. Answers are in the next code cell.
```

**Notes**: Do NOT reveal answers in the markdown. The next code cell shows them. This is a 2-min think-pair-share. Timing: 2 min.

---

### Cell 27: [type: code] - Architecture Decision Answers

**Purpose**: Confirm the decision table answers in code + comments.

**Content**:
```python
# Architecture Decision Reference -- answers for Section 3 exploration table.

architecture_decisions = {
    "Route complaint to correct department": {
        "choice": "Encoder-only (BERT-style)",
        "reason": "Classification over the whole input. BERT reads bidirectionally -- sees full context.",
    },
    "Generate a polite acknowledgment email": {
        "choice": "Decoder-only (GPT-style)",
        "reason": "Text generation: produce tokens autoregressively from a prompt.",
    },
    "Summarize a 500-word complaint to 3 sentences": {
        "choice": "Encoder-decoder (T5/BART-style)",
        "reason": "Seq2seq: encode the long input, decode a short output. Input and output are different.",
    },
    "Find complaints similar to a new one": {
        "choice": "Encoder-only (BERT-style)",
        "reason": "Produce a single embedding vector per sentence; cosine similarity search.",
    },
    "Extract named entities (account numbers, dates)": {
        "choice": "Encoder-only (BERT-style)",
        "reason": "Token classification: label each token independently. Needs full context.",
    },
    "Translate a complaint from Welsh to English": {
        "choice": "Encoder-decoder (T5/BART/Helsinki-NLP-style)",
        "reason": "Translation is the canonical seq2seq task. Different input and output languages.",
    },
}

for use_case, info in architecture_decisions.items():
    print(f"Use case: {use_case}")
    print(f"  --> {info['choice']}")
    print(f"  Why: {info['reason']}")
    print()
```

**Notes**: Read through with the class. Emphasize: "You will be making these decisions when choosing a model for your real Barclays system. The architecture choice determines EVERYTHING -- training data, fine-tuning strategy, inference cost." Timing: 3 min.

---

### Cell 28: [type: markdown] - Section 4 Header: Famous Models and the LLM Landscape

**Purpose**: Brief survey of the models students will hear about in the industry.

**Content**:
```markdown
## Section 4 -- Famous Transformers: The Landscape in 2026

There are thousands of transformer models on HuggingFace Hub.
Here are the ones that matter for a developer in 2026 -- categorized by family.

### Encoder-Only (BERT family)
| Model | Params | Best for |
|-------|--------|---------|
| BERT-base | 110M | Classification, NER, QA (the original 2018 Google model) |
| DistilBERT | 66M | Faster BERT; 97% of accuracy at 60% of size |
| RoBERTa | 125M | More robustly trained BERT; better benchmarks |
| DeBERTa-v3 | 184M | Best encoder for classification and NLU as of 2024 |

### Decoder-Only (GPT family)
| Model | Params | Best for |
|-------|--------|---------|
| GPT-2 | 1.5B | Open-source text generation baseline |
| LLaMA 4 (Meta, 2025) | 17B-400B+ | Open-weight frontier model; MoE architecture |
| GPT-4o / GPT-5 (OpenAI) | unknown | Frontier closed-source; multimodal |
| Claude Sonnet 4 (Anthropic) | unknown | Frontier closed-source; long context |
| Gemini 3 (Google) | unknown | Frontier closed-source; multimodal |
| DeepSeek-V3 | 671B MoE | Open-weight, competitive with GPT-4 class |

### Encoder-Decoder (T5 family)
| Model | Params | Best for |
|-------|--------|---------|
| T5-base | 220M | Any text-to-text task; flexible |
| Flan-T5 | 80M-11B | Instruction-tuned T5; great for summarization and QA |
| BART | 140M | Summarization, denoising, translation |

### Key Insight for Barclays

In Day 2 (fine-tuning), we will use **DistilBERT** for classification
and **Flan-T5** for instruction following. These are not the largest or newest models --
they are the RIGHT size for the tasks, the available infrastructure, and the budget.

The "which model" decision is a business decision as much as a technical one.
```

**Notes**: Do NOT drill into model internals here -- that is Topics 3 and 4. This is a practical taxonomy for decision-making. Timing: 3 min read + 2 min discussion.

---

### Cell 29: [type: code] - Quick Inference Check: All Three Families

**Purpose**: One-cell demo confirming all three pipeline types work.

**Content**:
```python
# Quick sanity check: verify all three architecture families run successfully.
# All three models are already loaded from Section 3. This cell just shows
# the pattern side by side.

sample_complaint = "I have been waiting two weeks for a callback about my blocked account."

print("Three transformer families, same complaint, different tasks:\n")

# Encoder-only: classify
clf_result = classifier(sample_complaint, candidate_labels=["account issue", "fraud", "billing"])
print("1. ENCODER-ONLY (classification):")
print(f"   Top label: {clf_result['labels'][0]} ({clf_result['scores'][0]:.3f})")

# Decoder-only: generate continuation
gen_result2 = generator(
    "We are sorry to hear that " + sample_complaint,
    max_new_tokens=25,
    do_sample=False,
)
print("\n2. DECODER-ONLY (generation):")
print(f"   {gen_result2[0]['generated_text']}")

# Encoder-decoder: summarize
summary_result2 = summarizer(
    sample_complaint + " The customer has called three times. No agent picked up.",
    max_length=30,
    min_length=10,
)
print("\n3. ENCODER-DECODER (summarization):")
print(f"   {summary_result2[0]['summary_text']}")
```

**Notes**: This cell runs fast because models are cached. Its purpose is to show the three families in direct comparison so students see the API pattern differences. Timing: 2 min.

---

### Cell 30: [type: markdown] - Section 5 Header: GenAI Project Lifecycle

**Purpose**: Transition to the final conceptual section.

**Content**:
```markdown
## Section 5 -- The Generative AI Project Lifecycle

Every developer in this room has shipped a traditional ML project.
The GenAI lifecycle looks similar on the surface but has critical differences.

Let us compare them directly, using our Barclays complaint system as the example.

### Traditional ML Lifecycle (what you know)

1. **Problem definition** -- classify complaint into 8 categories
2. **Data collection** -- 50,000 labeled complaints from the archives
3. **Feature engineering** -- TF-IDF, bag of words, embeddings
4. **Train from scratch** -- logistic regression, XGBoost, or an MLP
5. **Evaluate** -- accuracy, F1, confusion matrix
6. **Deploy** -- REST API, batch job
7. **Monitor** -- watch for data drift

### GenAI Lifecycle (what you are learning)

1. **Problem definition** -- same, but also: is this a generation, classification, or retrieval task?
2. **Model selection** -- WHICH pretrained model? Encoder? Decoder? Which vendor?
3. **Adapt the model** -- prompt engineering first (free), then fine-tuning (expensive), then RAG
4. **Evaluate with new metrics** -- not just accuracy; also: hallucination rate, toxicity, latency, cost/token
5. **Deploy** -- inference endpoint, not training job
6. **Alignment and safety gate** -- responsible AI review (new step that has no equivalent in classical ML)
7. **Monitor** -- watch for model drift AND prompt injection AND output quality AND cost

### The Critical Differences

- You almost NEVER train from scratch. Model selection IS a first-class engineering step.
- Evaluation is harder: how do you measure whether a generated response is "good"?
- Cost is token-denominated: every input and output token costs money.
- Safety is not optional: LLMs can generate harmful content that a logistic regression cannot.
```

**Notes**: Ask "which step do you spend the most time on in your current ML projects?" Then ask "which step do you think GenAI adds that costs the most?" (Answer: alignment and safety, and continuous prompt/output monitoring). Timing: 4 min.

---

### Cell 31: [type: code] - Lifecycle Decision Matrix (Interactive)

**Purpose**: Anchor the lifecycle with a simple code decision-tree that students can reuse.

**Content**:
```python
# Generative AI lifecycle decision helper.
# For a given Barclays use case, what lifecycle path do you follow?

def genai_lifecycle_path(
    task_type,      # "classify" | "generate" | "retrieve" | "summarize" | "extract"
    data_available, # "labeled_small" | "labeled_large" | "unlabeled" | "none"
    latency_sla_ms, # required latency in milliseconds for production
):
    """
    Return the recommended GenAI lifecycle path for a Barclays use case.
    This is a simplified heuristic -- real decisions involve more factors.
    """
    path = {}

    # Step 1: Model selection
    if task_type in ("classify", "retrieve", "extract"):
        path["model_family"] = "encoder-only (DistilBERT / DeBERTa)"
    elif task_type == "generate":
        path["model_family"] = "decoder-only (GPT-style or Flan-T5)"
    elif task_type == "summarize":
        path["model_family"] = "encoder-decoder (Flan-T5 / BART)"
    else:
        path["model_family"] = "unclear -- revisit task definition"

    # Step 2: Adaptation strategy
    if data_available == "none":
        path["adaptation"] = "prompt engineering only (zero-shot)"
    elif data_available == "labeled_small":
        path["adaptation"] = "few-shot prompting or LoRA fine-tuning on small dataset"
    elif data_available == "labeled_large":
        path["adaptation"] = "full fine-tuning (Day 2 capstone)"
    else:
        path["adaptation"] = "RAG with unlabeled data as retrieval corpus"

    # Step 3: Deployment
    if latency_sla_ms < 100:
        path["deployment"] = "small model + quantization (Topics 8/9)"
    elif latency_sla_ms < 1000:
        path["deployment"] = "DistilBERT or Flan-T5-small on ml.m5.xlarge"
    else:
        path["deployment"] = "larger model acceptable; batch inference"

    return path


# Barclays use cases
use_cases = [
    ("classify", "labeled_large", 200),
    ("generate", "none", 2000),
    ("summarize", "unlabeled", 500),
    ("retrieve", "none", 50),
]

for task, data, latency in use_cases:
    print(f"Task: {task} | Data: {data} | Latency SLA: {latency}ms")
    path = genai_lifecycle_path(task, data, latency)
    for k, v in path.items():
        print(f"  {k}: {v}")
    print()
```

**Notes**: Walk through each use case. The function is deliberately simple and opinionated -- it is a teaching tool, not a production system. Students can extend it as homework. Timing: 4 min.

---

### Cell 32: [type: markdown] - Peer Discussion 3

**Purpose**: Final discussion connecting lifecycle to the course journey.

**Content**:
```markdown
## Peer Discussion (3 min)

1. In the GenAI lifecycle we listed "alignment and safety gate" as a mandatory step.
   What could go wrong if the Barclays complaint response generator is deployed WITHOUT
   an alignment review? Give two specific examples.

2. The lifecycle says "you almost never train from scratch."
   But this course teaches you to build a transformer FROM SCRATCH in Topics 3 and 4.
   Why is that useful if you will never do it in production?

3. "Cost is token-denominated." You are given a budget of 10,000 GBP per month for
   LLM inference at Barclays. GPT-5 costs approximately $15 per million output tokens.
   How many complaints can you process per month if the average response is 200 output tokens?
   Is that enough? What would you do if it is not?
```

**Notes**: Question 2 is intentionally provocative -- the answer is: understanding the internals lets you debug, optimize, choose hyperparameters, and explain decisions to compliance. This is what distinguishes a developer who uses LLMs from one who UNDERSTANDS them. Timing: 3 min.

---

### Cell 33: [type: markdown] - Section 6 Header: Putting It All Together

**Purpose**: Synthesis section before wrap-up; shows the three concepts (tokenization, embeddings, inference) as one pipeline.

**Content**:
```markdown
## Section 6 -- The Complete LLM Pipeline

We have seen three pieces in isolation:
1. **Tokenization**: text -> integers
2. **Embeddings**: integers -> dense vectors (meaning)
3. **Inference**: vectors -> output (classification label, generated text, summary)

In a real transformer, these three steps are chained:

```
Raw text
   |
[Tokenizer]  (vocabulary: ~30,000 subwords; BPE or WordPiece)
   |
Token IDs    (e.g., [101, 1045, 2001, ...])
   |
[Embedding layer]  (lookup table: each ID -> 768-dim vector)
   |
   + [Positional encoding]  (adds ORDER information)
   |
[N x Transformer blocks]  (self-attention + FFN; this is Topics 3+4)
   |
[Task head]  (encoder: CLS -> label; decoder: predict next token; enc-dec: decode sequence)
   |
Output (label probabilities, generated token, summary token)
```

The self-attention blocks in the middle are what Topics 3 and 4 are about.
They are the part where "meaning" is enriched by context.
Today we have treated the model as a black box and used it through pipelines.
Tomorrow we build the box.
```

**Notes**: Draw or trace this pipeline on the whiteboard while students read. This is the conceptual bridge to Topics 3 and 4. Timing: 2 min.

---

### Cell 34: [type: code] - End-to-End Pipeline Demo

**Purpose**: Demonstrate the complete tokenize-embed-infer pipeline in one cell, showing internal shapes.

**Content**:
```python
# Complete pipeline: text -> token IDs -> embeddings -> task output
# We will trace the shapes at each step so you can see the data flowing.

complaint_text = "My Barclaycard statement shows a charge I did not make."

print("=" * 60)
print("Complete LLM Pipeline Trace")
print("=" * 60)

# Step 1: Tokenize
print("\n[Step 1] Tokenization")
inputs = bert_tokenizer(
    complaint_text, return_tensors="pt", truncation=True, max_length=512
)
token_ids = inputs["input_ids"]
print(f"  Input text: '{complaint_text}'")
print(f"  Token IDs shape: {token_ids.shape}")     # [1, num_tokens]
print(f"  Tokens: {bert_tokenizer.convert_ids_to_tokens(token_ids[0].tolist())}")

# Step 2: Embedding lookup (first layer of the transformer)
print("\n[Step 2] Token Embedding Lookup")
with torch.no_grad():
    raw_embeddings = bert_model.embeddings.word_embeddings(token_ids)
print(f"  Embedding matrix shape: {raw_embeddings.shape}")  # [1, num_tokens, 768]
print(f"  Each token -> a 768-dimensional vector")

# Step 3: Full forward pass (all transformer blocks)
print("\n[Step 3] Full Transformer Forward Pass (all attention blocks)")
with torch.no_grad():
    outputs = bert_model(**inputs)
contextual_embeddings = outputs.last_hidden_state
print(f"  Contextual embeddings shape: {contextual_embeddings.shape}")  # [1, num_tokens, 768]
print(f"  The SAME {token_ids.shape[1]} tokens, but now enriched by context.")

# Step 4: Pool to sentence embedding
print("\n[Step 4] Mean Pooling -> Sentence Embedding")
mask = inputs["attention_mask"].unsqueeze(-1).expand(contextual_embeddings.size()).float()
sentence_embedding = (contextual_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
sentence_embedding = sentence_embedding.squeeze().numpy()
print(f"  Sentence embedding shape: {sentence_embedding.shape}")  # (768,)
print(f"  First 5 values: {sentence_embedding[:5].round(4)}")
print(f"\nPipeline complete. One sentence -> one 768-dim vector capturing its meaning.")
```

**Notes**: Step 2 is optional advanced content -- `bert_model.embeddings.word_embeddings` is an internal layer access. Some students find this fascinating; others will skip it. Mark it clearly. Timing: 4 min.

---

### Cell 35: [type: markdown] - Section 7 Header: What We Did NOT Cover (and Why)

**Purpose**: Explicitly scope what comes next so students know what to expect.

**Content**:
```markdown
## What We Did NOT Cover Today -- And That Is Intentional

### The Self-Attention Mechanism
We said "N transformer blocks" and skipped straight to the output.
The inside of those blocks -- queries, keys, values, attention weights -- is the
mathematical heart of the transformer. That is **Topic 3 (Attention)**.
You will implement it from scratch in pure Python, then in PyTorch.

### Training and Fine-Tuning
Everything today was INFERENCE -- we used pretrained weights and did not update them.
In **Day 2 (Topics 6 and 7)** you will fine-tune DistilBERT and Flan-T5
on a Barclays complaint dataset on a remote GPU instance.

### Positional Encoding (the Details)
We mentioned it briefly in the pipeline. The math behind sinusoidal and learned
positional encodings is in **Topic 4 (Transformers)**.

### Retrieval-Augmented Generation (RAG)
The Lab 2 Homework Extension was the first seed of RAG.
Finding similar complaints and using them as context for generation is full RAG.
This is covered in the Day 3 deployment topics.

### What You CAN Do Right Now
- Choose the right transformer family for a task
- Tokenize any text and inspect the subword vocabulary
- Get embeddings and measure semantic similarity
- Run inference with a pretrained model through HuggingFace pipelines
- Describe the GenAI project lifecycle and its unique challenges
```

**Notes**: This section manages expectations and prevents the "I don't know everything yet" anxiety. Timing: 2 min (students read independently).

---

### Cell 36: [type: code] - Knowledge Check

**Purpose**: Quick true/false questions students answer in code to self-assess.

**Content**:
```python
# Knowledge check -- answer True or False for each statement.
# Run the cell to see if your answers are correct.

answers = {
    "GPT-style (decoder-only) models are best for text classification": None,  # YOUR CODE: True or False
    "DistilBERT has 66 million parameters, making it suitable for ml.t3.medium": None,  # YOUR CODE
    "Tokenization converts text to floating-point vectors": None,  # YOUR CODE
    "Cosine similarity of 0.0 means two sentences are identical": None,  # YOUR CODE
    "The GenAI lifecycle includes a safety/alignment gate that classical ML does not": None,  # YOUR CODE
    "You can use the same DistilBERT model for both classification and generation": None,  # YOUR CODE
}

correct_answers = {
    "GPT-style (decoder-only) models are best for text classification": False,
    "DistilBERT has 66 million parameters, making it suitable for ml.t3.medium": True,
    "Tokenization converts text to floating-point vectors": False,  # converts to integers (IDs)
    "Cosine similarity of 0.0 means two sentences are identical": False,  # 1.0 means identical
    "The GenAI lifecycle includes a safety/alignment gate that classical ML does not": True,
    "You can use the same DistilBERT model for both classification and generation": False,  # encoder-only cannot generate
}

none_count = sum(1 for v in answers.values() if v is None)
if none_count > 0:
    print(f"Answer {none_count} more question(s) before checking.")
else:
    score = 0
    for statement, your_answer in answers.items():
        correct = correct_answers[statement]
        mark = "CORRECT" if your_answer == correct else "WRONG"
        print(f"[{mark}] {statement}")
        if your_answer != correct:
            print(f"       Your answer: {your_answer}. Correct: {correct}.")
        score += 1 if your_answer == correct else 0
    print(f"\nScore: {score}/{len(answers)}")
```

**Notes**: Students fill in True/False. The self-assessment is low stakes but surfaces misconceptions immediately. Instructor can use show-of-hands "who got all 6?" Timing: 3 min.

---

### Cell 37: [type: markdown] - Wrap-Up and Bridge to Topic 3

**Purpose**: Close the topic with key takeaways and a bridge to the next topic.

**Content**:
```markdown
## Topic 2 Wrap-Up

### Key Takeaways

1. **LLMs are transformers**. All transformers share the same tokenization front-end
   and self-attention building blocks. They differ in how those blocks are wired.

2. **Three families, three use cases**: encoder-only for understanding (BERT),
   decoder-only for generation (GPT), encoder-decoder for transformation (T5).

3. **Tokenization is not word-splitting**. Subword tokenization (BPE/WordPiece) ensures
   any text -- including Welsh financial terms -- can be represented with a fixed vocabulary.

4. **Embeddings capture meaning, not just form**. Cosine similarity on DistilBERT embeddings
   clusters semantically similar complaints, enabling the routing and search system we built.

5. **The GenAI lifecycle differs from traditional ML** in three critical ways:
   you select a pretrained model, you adapt it (don't train from scratch), and
   you have a mandatory alignment/safety gate before deployment.

### Where We Are in the Course

```
Topic 1 (done): What is GenAI?
Topic 2 (done): What is an LLM? How does it tokenize and embed text?
Topic 3 (next): How does self-attention work? Build it from scratch in Python.
Topic 4:        Full transformer. Build the encoder-decoder from scratch in PyTorch.
Day 2:          Fine-tune DistilBERT and Flan-T5 on Barclays complaint data.
Day 3:          Deploy, quantize, align.
```

In Topic 3, we open the black box we used today.
You will implement the attention mechanism -- the core innovation that makes transformers
better than RNNs and LSTMs -- in pure Python, with no libraries.
```

**Notes**: End with energy -- "You now know what ChatGPT IS. In 90 minutes you will know how its attention mechanism works." Timing: 2 min.

---

### Cell 38: [type: code] - Cleanup (Optional)

**Purpose**: Free GPU/CPU memory before Topic 3 begins if running in the same kernel.

**Content**:
```python
# Optional cleanup -- free memory before moving to Topic 3.
# Run this if the kernel feels slow or you are getting OOM errors.
import gc

del bert_model, gpt_model, classifier, generator, summarizer
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Memory cleanup complete. Ready for Topic 3.")
```

**Notes**: ml.t3.medium has 4GB RAM. With 3 models loaded (~900MB total), there is headroom but cleanup is good practice. This is optional -- if Topic 3 runs in a new notebook, skip this. Timing: 1 min.

---

### Cell 39: [type: markdown] - Appendix: Model Size Reference

**Purpose**: Quick reference table students keep open during Day 2.

**Content**:
```markdown
## Appendix -- Model Size and Memory Reference

For ml.t3.medium (4GB RAM) and ml.g4dn.xlarge (16GB GPU VRAM):

| Model | Params | RAM (FP32) | RAM (FP16) | Fits in Studio kernel? |
|-------|--------|-----------|-----------|----------------------|
| distilgpt2 | 82M | ~330MB | ~165MB | YES |
| distilbert-base-uncased | 66M | ~265MB | ~133MB | YES |
| bert-base-uncased | 110M | ~440MB | ~220MB | YES |
| distilbart-cnn-6-6 | 306M | ~1.2GB | ~600MB | YES (marginal) |
| t5-small | 60M | ~240MB | ~120MB | YES |
| t5-base | 220M | ~880MB | ~440MB | YES |
| t5-large | 770M | ~3GB | ~1.5GB | NO (OOM on t3.medium) |
| flan-t5-base | 250M | ~1GB | ~500MB | YES |
| flan-t5-large | 780M | ~3.1GB | ~1.55GB | NO |
| llama-3-8b | 8B | ~32GB | ~16GB | YES on g4dn.xlarge FP16 |

Rule of thumb: model RAM in GB ~ params_in_billions x 4 (for FP32) or x 2 (for FP16).

In Day 2 remote training jobs, we run on ml.g4dn.xlarge (16GB VRAM) using FP16.
That is why we can use larger models for training but must use small ones for demos
in the Studio notebook kernel.
```

**Notes**: This appendix is a practical reference, not lecture content. Students refer back to it throughout Day 2. Timing: 0 min (reference).

---

### Cell 40: [type: code] - Appendix: Tokenization Side-by-Side GPT2 vs BERT

**Purpose**: Extended stretch content for the Lab 1 stretch task; shows BPE vs WordPiece differences.

**Content**:
```python
# Appendix -- BPE (GPT2) vs WordPiece (BERT) tokenization side-by-side.
# Run this as the Lab 1 stretch task.

print("Loading GPT2 tokenizer (BPE)...")
gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token

financial_terms = [
    "Barclaycard",
    "overdraft",
    "unacceptable",
    "I've been charged twice",
    "fraudulently",
    "unauthorised transaction",
]

print(f"\n{'Term':<30} {'BERT (WordPiece)':<40} {'GPT2 (BPE)'}")
print("-" * 100)

for term in financial_terms:
    bert_tokens = bert_tokenizer.tokenize(term)
    gpt2_tokens = gpt2_tok.tokenize(term)
    print(f"{term:<30} {str(bert_tokens):<40} {gpt2_tokens}")

print()
print("Observations:")
print("  - BERT uses ## prefix for continuation subwords")
print("  - GPT2 uses Ġ (visible space) prefix for subwords that follow a space")
print("  - Both can handle any word by splitting into known subpieces")
print("  - BERT adds [CLS] and [SEP]; GPT2 adds only <|endoftext|> as delimiter")
print("  - Token counts differ between tokenizers for the same text")
```

**Notes**: This cell is appendix/stretch only. Do not run it during the main session unless time permits. Timing: 0 min (student-driven).

---

### Cell 41: [type: code] - Appendix: Sentence-Transformers vs DistilBERT Comparison

**Purpose**: Extended stretch content for Lab 2 stretch; shows why sentence-transformers are better for similarity.

**Content**:
```python
# Appendix -- Sentence-transformers vs DistilBERT mean pooling for similarity.
# Requires: pip install -q sentence-transformers
# Run as the Lab 2 stretch task.

try:
    from sentence_transformers import SentenceTransformer
    st_available = True
except ImportError:
    st_available = False
    print("sentence-transformers not installed. Run: pip install -q sentence-transformers")

if st_available:
    print("Loading all-MiniLM-L6-v2 (sentence-transformers)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    paraphrase_pair = [
        "I was charged twice for the same transaction.",
        "A duplicate payment was debited from my account.",
    ]
    unrelated_pair = [
        "I was charged twice for the same transaction.",
        "The Barclays mobile app has a great new interface.",
    ]

    # DistilBERT mean pooling
    db_e1 = get_embedding(paraphrase_pair[0])
    db_e2 = get_embedding(paraphrase_pair[1])
    db_e3 = get_embedding(unrelated_pair[1])
    db_paraphrase_sim = cosine_similarity([db_e1], [db_e2])[0, 0]
    db_unrelated_sim  = cosine_similarity([db_e1], [db_e3])[0, 0]

    # Sentence-transformers
    st_embs = st_model.encode(paraphrase_pair + [unrelated_pair[1]])
    st_paraphrase_sim = cosine_similarity([st_embs[0]], [st_embs[1]])[0, 0]
    st_unrelated_sim  = cosine_similarity([st_embs[0]], [st_embs[2]])[0, 0]

    print("\n--- Paraphrase pair (should be HIGH) ---")
    print(f"  DistilBERT mean pooling: {db_paraphrase_sim:.4f}")
    print(f"  Sentence-transformers:   {st_paraphrase_sim:.4f}")

    print("\n--- Unrelated pair (should be LOW) ---")
    print(f"  DistilBERT mean pooling: {db_unrelated_sim:.4f}")
    print(f"  Sentence-transformers:   {st_unrelated_sim:.4f}")

    print("\nConclusion: sentence-transformers produces higher contrast (bigger gap between")
    print("paraphrase and unrelated) because it is trained with a contrastive similarity loss.")
    print("DistilBERT mean pooling works but produces less discriminative similarity scores.")
```

**Notes**: Appendix/stretch only. Sentence-transformers produces a visibly larger gap between paraphrase and unrelated pairs. This motivates the Day 2 content on fine-tuning for specific tasks. Timing: 0 min (student-driven).

---

### Cell 42: [type: markdown] - Appendix: GenAI Lifecycle Detailed Comparison Table

**Purpose**: Reference material for the lifecycle discussion.

**Content**:
```markdown
## Appendix -- GenAI vs Traditional ML Lifecycle Detailed Comparison

| Lifecycle Stage | Traditional ML | Generative AI |
|-----------------|---------------|--------------|
| Problem definition | Binary: predict X | Multi-mode: classify/generate/retrieve/summarize? |
| Data strategy | Collect + label everything | Pretrained model reduces label need; focus on evaluation data |
| Model selection | Algorithm choice (XGBoost vs NN) | Foundation model choice (size, family, vendor, license) |
| Training | Train from scratch on your data | Prompt engineering first, then fine-tuning if needed |
| Evaluation | Accuracy, F1, AUC | + BLEU, ROUGE, BERTScore, human eval, hallucination rate |
| Safety | Input validation | + output toxicity filters, PII detection, alignment review |
| Cost model | Compute cost during training | Token cost at inference (input + output, per request) |
| Deployment | Batch or real-time scoring | Chat interface, API, RAG pipeline, agent framework |
| Monitoring | Data drift, model drift | + prompt injection, output quality regression, token budget |
| Updates | Retrain on new data | Re-prompt, fine-tune delta, or swap foundation model |

### Why This Matters for Barclays

Barclays regulatory environment (FCA, PRA) adds an extra constraint: every model
decision must be auditable and explainable. The GenAI lifecycle must document:
- Which pretrained model was selected and why
- What fine-tuning data was used and whether it was reviewed for bias
- What safety filters are applied at inference time
- How human-in-the-loop review is triggered for edge cases

This is NOT a technical constraint -- it is a compliance constraint that shapes
the ENTIRE project lifecycle from day one.
```

**Notes**: The regulatory/compliance angle is specifically relevant to a Barclays audience. Instructors should invite students to share their own experience with ML model governance at Barclays. Timing: 0 min (reference).

---

### Cell 43: [type: code] - Appendix: Quick Model Download Test

**Purpose**: Troubleshooting cell for students whose Studio instance has network issues.

**Content**:
```python
# Appendix -- Network connectivity test for HuggingFace Hub.
# Run this if you get "Connection error" or "Timeout" when loading models.

import urllib.request

urls = [
    "https://huggingface.co",
    "https://cdn-lfs.huggingface.co",
]

for url in urls:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.getcode()
        print(f"[OK]   {url} returned {status}")
    except Exception as e:
        print(f"[FAIL] {url}: {e}")

print()
print("If any URL fails, ask the instructor -- Studio VPC may need an internet gateway.")
print("Public models (distilbert-base-uncased, distilgpt2) download from HuggingFace CDN.")
print("No HuggingFace token required.")
```

**Notes**: This cell is a support tool. SageMaker Studio in a VPC requires an internet gateway or VPC endpoint to reach HuggingFace Hub. The domain `barclays-training-v2` is configured with internet access, but students should run this if downloads fail. Timing: 0 min (troubleshooting).

---

### Cell 44: [type: markdown] - Final Summary Card

**Purpose**: Single-cell cheat sheet that students screenshot and keep.

**Content**:
```markdown
## Topic 2 Quick Reference Card

### Tokenization
```python
from transformers import DistilBertTokenizer
tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
ids = tok(text, return_tensors="pt")["input_ids"]
tokens = tok.convert_ids_to_tokens(ids[0].tolist())
decoded = tok.decode(ids[0].tolist(), skip_special_tokens=True)
```

### Embeddings
```python
from transformers import DistilBertModel
import torch, numpy as np
model = DistilBertModel.from_pretrained("distilbert-base-uncased"); model.eval()
def embed(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad(): out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    return ((out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).squeeze().numpy()
```

### Inference Pipelines
```python
from transformers import pipeline
classifier  = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
generator   = pipeline("text-generation", model="distilgpt2")
summarizer  = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
```

### Architecture Decisions
| Task | Family | Example model |
|------|--------|--------------|
| Classify / embed / NER | Encoder-only | DistilBERT, DeBERTa |
| Generate / chat | Decoder-only | distilgpt2, LLaMA 4 |
| Translate / summarize | Encoder-decoder | Flan-T5, BART |

### Versions (Studio ml.t3.medium)
- transformers>=4.35.0,<4.40.0
- tokenizers>=0.15.0,<0.20.0
- numpy<2
```

**Notes**: Intentionally code-heavy. Students keep this open during Day 2. The quick-reference card format is recognizable and screenshot-friendly. Timing: 0 min (reference).

---

## Lab Summary

| Lab | Tier | Time | Variables for downstream |
|-----|------|------|--------------------------|
| Lab 1: Tokenization Explorer | Tier 1 (guided) | 15 min | `result_short`, `result_medium`, `result_long`, `analyze_complaint_tokens` |
| Lab 2: Complaint Similarity Matrix | Tier 2 (hard) | 25 min | `all_embeddings`, `embeddings_matrix`, `sim_matrix`, `most_similar_idx`, `least_similar_idx` |

Safety-net cells: Cell 10 (Lab 1), Cell 20 (Lab 2).

---

## Timing Summary

| Section | Cells | Estimated Time |
|---------|-------|---------------|
| Setup + installs | 1-2 | 3 min |
| Section 1: Tokenization + Lab 1 | 3-11 | 22 min |
| Peer Discussion 1 | 12 | 3 min |
| Section 2: Embeddings + Lab 2 | 13-21 | 35 min |
| Peer Discussion 2 | 22 | 3 min |
| Section 3: Transformer Families | 23-27 | 12 min |
| Section 4: Model Landscape | 28-29 | 5 min |
| Section 5: GenAI Lifecycle | 30-32 | 10 min |
| Peer Discussion 3 | 32 | 3 min |
| Section 6: Pipeline + Wrap-up | 33-38 | 12 min |
| Knowledge check | 36 | 3 min |
| Appendix (reference/stretch) | 39-44 | 0 min in-class |
| **Total in-class** | | **~65-70 min** |

---

## Hard Rules Compliance Checklist

- [x] `numpy<2` pinned in install cell (Cell 2)
- [x] Only INFERENCE demos (no training anywhere in this topic)
- [x] Small models only: distilgpt2 (82M), distilbert-base-uncased (66M), distilbart-cnn-6-6 (306M) -- all fit in 4GB RAM
- [x] NO `evaluate` library
- [x] NO `getpass` (all models are public)
- [x] Plain ASCII only -- no em dashes, en dashes, Unicode multiplication, emojis
- [x] `# YOUR CODE` does NOT hint at the answer
- [x] No more than 3 consecutive markdown cells without a code cell
- [x] transformers `>=4.35.0,<4.40.0` and tokenizers `>=0.15.0,<0.20.0`
- [x] Exactly 2 diagram placeholders (Cell 5 and Cell 15)
- [x] Safety-net cells after every lab that feeds downstream (Cells 10, 20)
- [x] STAR method applied to Lab 1 and Lab 2
- [x] Peer discussion cells (Cells 12, 22, 32)
- [x] Homework extension for both labs (Cells 11, 21)
- [x] Lab tiers: Lab 1 = Tier 1, Lab 2 = Tier 2 (one Tier 2 per day, this is Day 1 second topic)
- [x] Stretch version for both labs
- [x] 40-44 cells in main notebook (44 total including appendix)
- [x] Four-beat arc applied to: tokenization (Cells 3-4-5-7), embedding similarity (Cells 13-14-15-16-17), transformer families (Cells 23-24-25-26)
