# Topic 1 - Overview of Generative AI: Cell-by-Cell Plan

## Narrative

**Scenario**: Barclays receives thousands of customer complaints per day across digital channels.
A team wants to automatically understand, classify, and route them to the right specialist.
This notebook opens that question: what kind of AI do you use? Students explore the landscape of
generative models (GANs, VAEs, Diffusion, Autoregressive), see hands-on why GANs fail for text,
and end with a working OpenAI API call that classifies a real Barclays-style complaint.

The punchline is delivered at the end of section 3: "For our complaint system, we need an
autoregressive LLM - and that is what the rest of this course builds."

**Audience**: Developers with 2+ years Python, PyTorch basics, deep learning fundamentals.
NOT beginners. Skip the equations - lean on intuition and code.

**Estimated time**: 50-60 minutes in class.
- Section 1 (What is GenAI + four families): 15 min
- Section 2 (Why not GANs/VAEs/Diffusion for text): 15 min
- Section 3 (Autoregressive = the right tool): 10 min
- Section 4 (LLM-as-a-Service, tokens, temperature, triage lab): 20 min (includes Tier 2 lab)

**Runs in**: SageMaker Studio notebook kernel, ml.t3.medium. No remote training. No estimators.

---

## Diagram Index

| # | Slug | Path | Description |
|---|------|------|-------------|
| 1 | autoregressive-loop | `../../plans/topic_1/diagrams/autoregressive-loop.mmd` | LLM token generation loop: context window box -> softmax over vocab -> sample token -> append to context -> repeat arrow back to start. Shows how temperature shifts the distribution. |
| 2 | openai-api-request-response-flow | `../../plans/topic_1_overview_genai/diagrams/openai-api-request-response-flow.mmd` | OpenAI API request/response cycle: system prompt and user message combined into HTTP request, model applies temperature to the output distribution, returns structured completion. Shows the role of the system prompt and how temperature affects the output. |

---

## Cell-by-Cell Plan

---

### Cell 1: [type: markdown] - Title and Learning Objectives

**Purpose**: Orient students. Establish the Barclays narrative. Set expectations for the 50-60 min session.

**Content**:
```
# Topic 1: Overview of Generative AI

## The Problem We Are Solving

Barclays processes thousands of customer complaints every day - fraud disputes, fee complaints,
transfer failures, account access issues. Today a human reads each one and decides:
- What category is this?
- How urgent is it?
- Which team handles it?

Your job as an engineering team: build an AI system that automates this triage.

## What You Will Learn Today

By the end of this notebook you will be able to:
1. Name the four major families of generative AI models and their trade-offs
2. Explain in one sentence why each family is (or is not) suited for text understanding
3. Call the OpenAI API with a system prompt, a user message, tokens, and temperature
4. Write a complaint triage prompt that classifies and routes a Barclays complaint

## How This Notebook Is Organised

- Section 1: The generative AI landscape - a map of the territory
- Section 2: Why the wrong model family fails (live demo)
- Section 3: Autoregressive LLMs - the right tool for text
- Section 4: LLM-as-a-Service - the OpenAI API in practice
```

**Notes**: Read aloud the problem statement. Ask the class: "How many of you have built a text
classifier before? How many have used an LLM API?" (quick show of hands - calibrates the room).

---

### Cell 2: [type: code] - Environment Setup and Imports

**Purpose**: Pin numpy<2, import everything needed for the notebook, print version info.
No SageMaker estimators needed - this is Day 1, all in-kernel.

**Content**:
```python
# Environment setup for Topic 1
# All code runs in the SageMaker Studio notebook kernel (no remote training jobs)

!pip install -q "numpy<2" "openai>=1.0.0"

import numpy as np
import random
import json
import getpass

# PyTorch for the GAN demo in Section 2
import torch
import torch.nn as nn
import torch.optim as optim

print("numpy version :", np.__version__)
print("torch version  :", torch.__version__)
print("Setup complete.")
```

**Notes**: The openai package install is lightweight. numpy<2 is pinned per hard rule.
If students already have the packages this cell is fast. Instructor: confirm everyone sees
"Setup complete" before moving on.

---

### Cell 3: [type: markdown] - Section 1 Header: The Generative AI Landscape

**Purpose**: Transition into content. Frame the question: "What does generative mean, and what are the families?"

**Content**:
```
## Section 1: The Generative AI Landscape

Before we pick a tool, we need a map.

**Discriminative AI** learns a boundary: given input X, predict label Y.
A spam classifier is discriminative. A fraud detector is discriminative.

**Generative AI** learns the distribution of data itself: it can *create* new X values
that look like they came from the training set. A model that writes new complaint text,
draws a new image, or synthesises a new audio clip - that is generative.

There are four main families. They all generate new data, but they do it in completely
different ways. Understanding *how* each works is what will let you choose the right one.
```

**Notes**: Pause after reading - ask "What other examples of generative AI have you used in the
last week?" Gets the class talking. One minute, then move on.

---

### Cell 4: [type: markdown] - Four Families Overview Table

**Purpose**: The taxonomy at a glance before the diagram. This is the "anchor" students will
return to throughout the course.

**Content**:
```
### The Four Families at a Glance

| Family | Core idea | Best at | Weakness |
|--------|-----------|---------|----------|
| GAN (Generative Adversarial Network) | Generator vs Discriminator game | Photorealistic images | Unstable training, mode collapse, bad for discrete text |
| VAE (Variational Autoencoder) | Encode to latent space, sample, decode | Smooth interpolation, anomaly detection | Blurry images, limited quality ceiling |
| Diffusion | Iterative denoising from pure noise | State-of-the-art image and audio quality | Very slow inference (hundreds of steps) |
| Autoregressive | Predict next token given all previous tokens | Text, code, structured sequences | Sequential - cannot parallelise generation |

The rest of Section 1 unpacks each row in turn.
```

**Notes**: No code cell yet - but we have not yet hit 3 consecutive markdown cells, so we are fine.
The next cell is the diagram placeholder (markdown), then we must have code immediately after.

---

### Cell 5: [type: markdown] - Taxonomy Summary and Section Bridge

**Purpose**: Brief commentary bridging the taxonomy table to the code demo. Keeps the markdown
chain from hitting 3 in a row (Cell 3, Cell 4, this cell) before Cell 6 code.

**Content**:
```
The table above is your map for this session. Three of the four families dominate *image*
generation. Autoregressive is the family that dominates *text*. That asymmetry is not an
accident - you will see exactly why in Section 2 when we try (and fail) to use a GAN for text.
```

**Notes**: This is the third consecutive markdown cell. Cell 6 MUST be code.

---

### Cell 6: [type: code] - Beat 1 (GAN Concept Demo - Broken for Text)

**Purpose**: Beat 1 of the GAN concept. Show a toy GAN generating continuous data (numbers),
then attempt to generate discrete text tokens - watch it produce garbage. Students feel the pain.

**Content**:
```python
# ------------------------------------------------------------------
# BEAT 1: A minimal GAN - works for numbers, breaks for text
# ------------------------------------------------------------------
# A GAN has two networks:
#   Generator (G): takes random noise -> produces fake data
#   Discriminator (D): takes data -> outputs P(real)
# They play a game. G tries to fool D; D tries to catch G.

# --- Part A: GAN working correctly on continuous data (numbers) ---

torch.manual_seed(42)

class TinyGenerator(nn.Module):
    def __init__(self, noise_dim=8, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 16), nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, z):
        return self.net(z)

class TinyDiscriminator(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = TinyGenerator()
D = TinyDiscriminator()
opt_G = optim.Adam(G.parameters(), lr=0.01)
opt_D = optim.Adam(D.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Train for 200 steps on "real data" = samples from N(5, 1)
for step in range(200):
    real = torch.randn(32, 1) + 5.0        # real data: mean=5
    noise = torch.randn(32, 8)
    fake = G(noise)

    # Discriminator step
    opt_D.zero_grad()
    d_real = D(real)
    d_fake = D(fake.detach())
    loss_D = loss_fn(d_real, torch.ones(32, 1)) + loss_fn(d_fake, torch.zeros(32, 1))
    loss_D.backward()
    opt_D.step()

    # Generator step
    opt_G.zero_grad()
    loss_G = loss_fn(D(G(noise)), torch.ones(32, 1))
    loss_G.backward()
    opt_G.step()

# Sample from the trained generator
noise = torch.randn(10, 8)
samples = G(noise).detach().numpy().flatten()
print("GAN on numbers - generated values (should be near 5.0):")
print([round(float(s), 2) for s in samples])
print()

# --- Part B: Now try to generate TEXT with the same approach ---
# Imagine our vocabulary is just 5 words:
vocab = ["complaint", "fraud", "account", "payment", "urgent"]
vocab_size = len(vocab)

# A naive GAN for text would output a one-hot vector over vocab
# and we pick argmax to get a "word". Let's try:
class TextGenerator(nn.Module):
    def __init__(self, noise_dim=8, vocab_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 16), nn.ReLU(),
            nn.Linear(16, vocab_size)
        )
    def forward(self, z):
        return torch.softmax(self.net(z), dim=-1)  # distribution over vocab

text_G = TextGenerator()
noise = torch.randn(1, 8)
output = text_G(noise)
word_index = output.argmax(dim=-1).item()
print("Naive GAN text output (before training):", vocab[word_index])
print()
print("--- THE PROBLEM ---")
print("To backpropagate through the discriminator into the generator,")
print("we need gradients to flow from D's output back through the selected word.")
print("But argmax is NOT differentiable. You cannot nudge 'fraud' toward 'fraud + 0.001'.")
print("There is no such thing as a word between two words.")
print()
print("Result: standard GAN training completely breaks for discrete text.")
print("(SeqGAN and similar work around this with reinforcement learning,")
print(" but that introduces its own instability problems.)")
```

**Expected output**: Numbers near 5.0 from the working numeric GAN; one word from the text GAN;
then the printed explanation of why gradients break.

**Notes**: Instructor should narrate: "Notice we only got ONE word - not a sentence. And we have
not even tried to train it yet - training would require gradients to flow through the argmax, which
is impossible." This is the pain moment.

---

### Cell 7: [type: markdown] - Beat 2 (GAN) + Peer Discussion Prompt

**Purpose**: Bridge from Beat 1 failure to the working model demo. Also the first peer discussion.

**Content**:
```
### Why GANs Fail for Text (The Gradient Problem)

The generator needs a gradient signal: "was this output good or bad, and in which direction
should I change?" For continuous data (pixel values, audio samples) that direction is a real
number. For discrete tokens the direction does not exist - words are not on a number line.

This is not a solvable engineering problem. It is a mathematical mismatch between the GAN
training recipe and the nature of language.

**So where do GANs shine?** Photorealistic face synthesis (StyleGAN), image-to-image
translation (Pix2Pix), and any task where the output is a continuous tensor.

**Peer Discussion (3 min)**

Think about a Barclays fraud detection system:
- Would you use a GAN to *classify* whether a complaint is about fraud?
- Would you use a GAN to *generate synthetic* complaint text for a test dataset?
- What are the risks of using synthetic complaint text in financial services?

Discuss with the person next to you. There is no single right answer.
```

**Notes**: Give 3 minutes. The second question is where the interesting tension lives:
yes, GANs can generate synthetic text (with tricks) but that is not what we want here.

---

### Cell 8: [type: markdown] - VAE and Diffusion Brief Explanation

**Purpose**: Complete the taxonomy tour quickly (VAE and Diffusion get less time - they are
less relevant to the narrative). No Beat 1 broken code needed for these two - the GAN demo
already landed the "wrong tool for text" lesson. One markdown cell per family, then move on.
Must be code before Cell 9 to avoid 3-markdown-chain rule.

**Content**:
```
### VAEs: Smooth Latent Space, Blurry Outputs

A Variational Autoencoder encodes input into a *probability distribution* in latent space
(not a single point), then samples from that distribution to decode a new output.

**The upside**: the latent space is smooth and continuous. Interpolating between two points
gives meaningful intermediate outputs. Great for anomaly detection: a complaint that lands
far from all training clusters is likely unusual.

**The downside for text**: same gradient problem as GANs when the output is discrete tokens.
VAEs are used for text *embeddings* (encoding), not text *generation*.

### Diffusion Models: SOTA Images, Wrong Tool for Text

Diffusion models learn to reverse a noise-adding process. Training: take an image, add
Gaussian noise step by step until it is pure noise. Learn to predict and remove that noise.
Inference: start from noise, denoise for 50-1000 steps.

**The upside**: state-of-the-art image quality (Stable Diffusion, DALL-E image backbone,
Midjourney). Also used for audio (MusicGen) and molecule design.

**The downside for text**: you cannot add Gaussian noise to a discrete token index.
Some research applies diffusion to continuous embeddings (MDLM, PLAID) but these are
not production-ready and are orders of magnitude slower than autoregressive LLMs.
```

**Notes**: Keep this fast - 2 minutes of narration. The goal is a complete map, not deep dives.

---

### Cell 9: [type: code] - Quick VAE Latent Space Demo (Beat 3 mini-demo)

**Purpose**: Code break after two markdown cells. Shows the VAE idea in 15 lines.
Demonstrates that the latent space is a distribution, not a point.

**Content**:
```python
# VAE in 15 lines: encode -> sample from latent distribution -> decode
# This is the CONCEPT, not a trained model. Just to make it concrete.

torch.manual_seed(0)

# Encoder: maps input to (mu, log_var) - a Gaussian in latent space
def encode(x, W_mu, W_logvar):
    mu = torch.matmul(x, W_mu)
    log_var = torch.matmul(x, W_logvar)
    return mu, log_var

# Reparameterisation trick: sample z = mu + eps * std (DIFFERENTIABLE)
def reparameterise(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)   # random noise
    return mu + eps * std         # shift and scale

# Simulate encoding a complaint embedding (dim=4) to latent dim=2
x = torch.tensor([[0.8, 0.1, 0.5, 0.2]])  # a "complaint" vector
W_mu    = torch.randn(4, 2)
W_logvar = torch.randn(4, 2)

mu, log_var = encode(x, W_mu, W_logvar)
z1 = reparameterise(mu, log_var)
z2 = reparameterise(mu, log_var)  # second sample from same input

print("Same complaint input -> two different latent samples:")
print("  z1:", z1.detach().numpy().round(3))
print("  z2:", z2.detach().numpy().round(3))
print()
print("This is the VAE trade-off: stochastic latent space enables generation")
print("but the decoder must reconstruct from a blurry, noisy latent - hence blurry outputs.")
```

**Notes**: Not a trained VAE - just a forward pass sketch. The point is the reparameterisation
trick and the fact that the same input maps to a distribution. Takes 30 seconds to run.

---

### Cell 10: [type: markdown] - Section 2 Transition to Autoregressive

**Purpose**: Pivot from "what does not work" to "what does". Sets up Beat 1 for autoregressive.

**Content**:
```
## Section 2: The Right Tool - Autoregressive Models

So far:
- GANs: gradient problem with discrete tokens
- VAEs: good for embeddings, not text generation
- Diffusion: SOTA for images, research-only for text

All three were designed around *continuous data*. Text is *discrete*.

### The Autoregressive Insight

What if we did not try to generate the whole complaint at once?
What if we generated it one word (actually: one token) at a time,
conditioning each new token on everything we have produced so far?

This is the autoregressive idea:

  P(token_1, token_2, ..., token_n) = P(token_1) * P(token_2 | token_1) * ...

No gradient problem: we train with cross-entropy on each token prediction.
Standard backpropagation. Scales to billions of parameters.

This is what GPT-4o, Claude, Llama, and every modern LLM is doing.
```

**Notes**: Write P(t1) * P(t2|t1) * ... on the whiteboard. Ask: "What does this remind you of
from your probability course?" (Chain rule of probability.) One minute, then the diagram.

---

### Cell 11: [type: markdown] - Diagram 2: Autoregressive Token Loop

**Purpose**: Second (and final) diagram. Visual anchor for the generation loop.

**Content**:
```
<!-- DIAGRAM: The autoregressive LLM token generation loop. Shows: (1) context window containing prompt tokens, (2) arrow to softmax distribution over vocabulary, (3) sampling a token (affected by temperature), (4) appending the new token back to context window, (5) repeat arrow. A second branch shows how temperature=0 picks the peak token (greedy) vs temperature=1.0 spreads the distribution (creative). -->
[View diagram](../../plans/topic_1/diagrams/autoregressive-loop.mmd)

Each step in this loop is a full forward pass through the transformer.
At GPT-4o scale that is hundreds of billions of floating-point operations per token.
The output is a probability distribution over ~100,000 vocabulary tokens.
Temperature controls how sharply peaked that distribution is.
```

**Notes**: This is the second consecutive markdown cell (Cell 10 was the first). Cell 12 must be code.

---

### Cell 12: [type: code] - Beat 3 (Autoregressive Loop Demo - Working)

**Purpose**: Beat 3 of the autoregressive concept. Build a tiny autoregressive model from scratch
(character-level, 2-layer, trivially small) and watch it generate a sequence. The point is to
make the loop tangible, not to build a production LLM.

**Content**:
```python
# ------------------------------------------------------------------
# BEAT 3: Autoregressive generation loop - made explicit
# ------------------------------------------------------------------
# We build the world's smallest autoregressive language model:
#   - Vocabulary: digits 0-9 plus a few characters
#   - Architecture: embedding -> 1-layer GRU -> linear -> softmax
#   - Training: 50 steps on a trivial sequence ("12345678901234...")
# The goal is to SEE the loop, not to impress anyone with quality.

torch.manual_seed(7)

# --- Vocabulary ---
chars = list("0123456789")
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# --- Tiny model ---
embed_dim = 8
hidden_dim = 16

embedding = nn.Embedding(vocab_size, embed_dim)
gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
linear = nn.Linear(hidden_dim, vocab_size)

params = list(embedding.parameters()) + list(gru.parameters()) + list(linear.parameters())
optimizer = optim.Adam(params, lr=0.05)
loss_fn = nn.CrossEntropyLoss()

# --- Training data: predict next digit in a repeating sequence ---
sequence = "01234567890123456789"
input_ids  = torch.tensor([[char_to_idx[c] for c in sequence[:-1]]])  # shape (1, 19)
target_ids = torch.tensor([[char_to_idx[c] for c in sequence[1:]]])   # shifted by 1

print("Training tiny autoregressive model...")
for step in range(150):
    optimizer.zero_grad()
    emb = embedding(input_ids)          # (1, 19, 8)
    out, _ = gru(emb)                   # (1, 19, 16)
    logits = linear(out)                # (1, 19, vocab_size)
    loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
print()

# --- The generation loop ---
# This is the EXACT same loop that GPT-4o uses at inference time.
# The only difference is scale (billions of parameters vs ~2K here).

def generate(seed_char, n_tokens=10, temperature=1.0):
    result = [seed_char]
    # Start with the seed token
    current_idx = torch.tensor([[char_to_idx[seed_char]]])
    hidden = None

    for _ in range(n_tokens):
        # Step 1: embed the current token
        emb = embedding(current_idx)            # (1, 1, 8)
        # Step 2: pass through model, update hidden state
        out, hidden = gru(emb, hidden)          # (1, 1, 16)
        logits = linear(out[:, -1, :])          # (1, vocab_size)

        # Step 3: apply temperature (lower = more deterministic)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # Step 4: sample the next token from the distribution
        next_idx = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Step 5: append to result, use as next input
        result.append(idx_to_char[next_idx.item()])
        current_idx = next_idx.unsqueeze(0)     # (1, 1, 1) -> shape for next step

    return "".join(result)

print("Autoregressive generation from seed '0':")
print("  temperature=0.1 (greedy)  :", generate("0", n_tokens=10, temperature=0.1))
print("  temperature=1.0 (balanced):", generate("0", n_tokens=10, temperature=1.0))
print("  temperature=2.0 (random)  :", generate("0", n_tokens=10, temperature=2.0))
print()
print("Notice: lower temperature = more predictable (repeats the learned pattern).")
print("        higher temperature = more random (deviates from the pattern).")
```

**Expected output**: With temperature=0.1 the model should reproduce "01234567890" closely.
Higher temperatures introduce noise. Students see the loop and the temperature effect in code.

**Notes**: Instructor: walk through the five steps inside the loop one by one, mapping each to
the diagram in Cell 11. "This is GPT-4o. Same loop. Just 175 billion more parameters."

---

### Cell 13: [type: markdown] - Beat 4 Lab 1: Autoregressive Temperature Exploration (Tier 1)

**Purpose**: Beat 4, Lab 1 (Tier 1, guided). Students modify the generate() function to
experiment with temperature and record observations.

**Content**:
```
### Lab 1: Temperature and Autoregressive Generation (Tier 1 - Guided)

**Situation**: You are evaluating whether an autoregressive model is suitable for generating
standardised complaint response templates at Barclays. You need to understand how temperature
controls the trade-off between consistency and variation.

**Task**: Run the generate() function with four different temperature values and record
what happens to the output.

**Action**: Fill in the temperatures in the starter code below, run the cell, and
answer the discussion questions in the markdown cell that follows.

**Result**: After completing the lab, you should be able to explain in one sentence
what temperature does and when you would use a low vs high value in production.

**Stretch (fast finishers)**: Modify generate() to use greedy decoding (argmax instead of
multinomial sampling) and compare it to temperature=0.1. Are they identical? Why or why not?

**Homework Extension**: See the cell after the safety-net cell.
```

---

### Cell 14: [type: code] - Lab 1 Starter Code

**Purpose**: Student lab code. One placeholder to fill in, verification block included.

**Content**:
```python
# Lab 1: Autoregressive Temperature Exploration
# Fill in the four temperature values you want to test.
# Then run this cell and observe the outputs.

# Step 1: choose four temperature values to explore (must be > 0)
temperatures_to_test = None  # YOUR CODE

# --- do not edit below this line ---
if temperatures_to_test is not None:
    print("Seed character: '5'")
    print("-" * 40)
    for t in temperatures_to_test:
        output = generate("5", n_tokens=12, temperature=t)
        print(f"  temperature={t:<4} -> {output}")
    print()
    print("Observation: at very low temperature, the model is _______________.")
    print("             at very high temperature, the model is _______________.")
```

**Notes**: The placeholder is intentionally vague - just a list of numbers.
Students must think about what range to explore. Verification is visual (they see the outputs).
This is a Tier 1 lab - the only thing to decide is which four temperatures to pick.

---

### Cell 15: [type: code] - Lab 1 Safety-Net

**Purpose**: Safety-net cell for Lab 1. Downstream cells print temperature outputs.
Must be here so students who did not finish Lab 1 can continue.

**Content**:
```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if temperatures_to_test is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    temperatures_to_test = [0.1, 0.5, 1.0, 2.0]
```

---

### Cell 16: [type: markdown] - Lab 1 Homework Extension

**Purpose**: Async deeper work for after class.

**Content**:
```
**Homework Extension - Lab 1**

The tiny model you used was trained on a repeating digit sequence.
Real LLMs are trained on trillions of tokens of text.

1. Read the OpenAI API documentation on the `temperature` parameter:
   https://platform.openai.com/docs/api-reference/chat/create

2. What is the default temperature for GPT-4o? What range is supported?

3. For each Barclays use case below, decide whether you would use a low, medium,
   or high temperature value, and justify your choice in 1-2 sentences:
   a. Generating a standardised rejection letter for a loan application
   b. Generating creative marketing copy for a new current account product
   c. Classifying an incoming complaint into one of 10 predefined categories

Write your answers in a markdown cell in a new notebook and bring them to the next session.
```

---

### Cell 17: [type: markdown] - Section 3 Header: LLM-as-a-Service

**Purpose**: Transition to the API section. Re-anchor on the Barclays narrative.
The conclusion of the taxonomy tour.

**Content**:
```
## Section 3: LLM-as-a-Service - The OpenAI API

We have established:
- GAN, VAE, Diffusion: wrong tools for text understanding and generation
- Autoregressive LLMs: the right tool, and you now know why at the code level

**The conclusion for our complaint system**:
We need an autoregressive LLM that understands text, and we need it *right now* -
not after training a model from scratch on Barclays data for six months.

This is where LLM-as-a-Service comes in.

### The Major Players (2025-2026)

| Provider | Model | Strengths |
|----------|-------|-----------|
| OpenAI | GPT-4o, GPT-4o-mini | Best general reasoning, huge ecosystem |
| Anthropic | Claude Sonnet/Opus | Strong reasoning, large context, safety focus |
| Google | Gemini Pro/Ultra | Multimodal, long context |
| Meta | Llama 3.x (open weights) | Self-hostable, fine-tunable |
| Mistral | Mistral Large | European, GDPR-friendly, open weights |

For this course we use **GPT-4o via the OpenAI API**.
All the concepts (tokens, temperature, system prompts, fine-tuning) apply equally to
every provider - only the API surface changes.

### Three Concepts You Must Understand Before Calling Any LLM API

1. **Tokens**: the atomic unit of text. Not words, not characters - subword pieces.
   "Barclays" is 1 token. "uncharacteristically" might be 4-6 tokens.
   You pay per token. You are rate-limited per token.

2. **Temperature**: you just built this from scratch. Range 0-2 in GPT-4o.
   Lower = deterministic. Higher = creative (and sometimes wrong).

3. **System prompt**: a privileged message that sets the model's role and rules.
   It appears before the user's message and is invisible to the end user.
   This is where you put: "You are a Barclays complaint triage agent. ..."
```

**Notes**: One minute to read this. Then immediately into code so we do not hit the 3-markdown rule.

---

### Cell 18: [type: code] - API Key Setup

**Purpose**: Collect the OpenAI API key securely. Hard rule: getpass.getpass() only.

**Content**:
```python
# Securely enter your OpenAI API key.
# This uses getpass() so the key is never visible in the notebook output
# and never stored in the notebook file (do not hardcode it).

openai_api_key = getpass.getpass("Paste your OpenAI API key and press Enter: ")

from openai import OpenAI
client = OpenAI(api_key=openai_api_key)

print("OpenAI client initialised. Key ends in:", openai_api_key[-4:])
```

**Notes**: Instructor: remind students to NEVER paste API keys directly into notebook cells.
If they accidentally commit an API key, they should rotate it immediately in the OpenAI dashboard.
The getpass() approach means the key is in memory only - not written to disk.

---

### Cell 19: [type: markdown] - Beat 1 (API) - No System Prompt = Bad Triage

**Purpose**: Beat 1 for the API section. Show what happens when you call GPT-4o with no
system prompt on a Barclays complaint. The output is unhelpfully generic.

**Content**:
```
### Beat 1: Calling the API Without a System Prompt

What happens if we just send the complaint text with no instructions?
```

---

### Cell 20: [type: code] - Beat 1 Code: API Call Without System Prompt

**Purpose**: Demonstrates the broken/naive baseline - API call with no system prompt.
Output will be a generic, unhelpful response with no routing decision.

**Content**:
```python
# Beat 1: Call GPT-4o with no system prompt - just the raw complaint text.
# Watch what happens.

complaint_text = (
    "I tried to make a payment of 500 pounds to my landlord last night and it just "
    "disappeared. The money left my account but my landlord says he never received it. "
    "The app just shows 'pending' with no further information. I need this resolved "
    "urgently as my rent is now overdue and I am being threatened with eviction."
)

# No system prompt - just a user message
response_no_system = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": complaint_text}
    ],
    temperature=0.7,
    max_tokens=200
)

print("Response WITHOUT a system prompt:")
print("-" * 50)
print(response_no_system.choices[0].message.content)
print("-" * 50)
print()
print("Tokens used:")
print("  prompt_tokens    :", response_no_system.usage.prompt_tokens)
print("  completion_tokens:", response_no_response.usage.completion_tokens)
print("  total_tokens     :", response_no_system.usage.total_tokens)
```

**Notes**: There is a deliberate bug in this cell: `response_no_response` (wrong variable name)
will cause a NameError. This is intentional Beat 1 - it fails. But the API call itself DOES
succeed and prints the generic response before the token count line crashes. Instructor should
let it run, point at the generic output, then show the NameError. "Two problems: the code has
a bug, AND the output is useless for routing." Beat 3 fixes both.

Actually - on reflection, the NameError is a code bug that stops execution of the token count
line ONLY. The response print will succeed. Students see the bad output THEN see the crash.
This is acceptable Beat 1 behavior (runs, fails). Make sure the instructor notes explain this.

---

### Cell 21: [type: markdown] - Beat 2 (API) - What a System Prompt Does

**Purpose**: Explain system prompts and their role before showing the working demo.
Includes the diagram placeholder for the API request/response flow.

**Content**:
```
### Beat 2: The System Prompt - Setting the Role and Rules

The model's response without a system prompt is empathetic but useless for operations:
it gives advice but does not classify, does not route, and does not follow any format.

<!-- DIAGRAM: openai-api-request-response-flow -->
[View diagram](../../plans/topic_1_overview_genai/diagrams/openai-api-request-response-flow.mmd)

The diagram shows the full request/response cycle: the system prompt and user message are
combined into a single HTTP request; the model returns a completion shaped by temperature
(low = peaked distribution, high = spread distribution) and the system prompt constraints.

A system prompt lets us give the model:
1. A **role** (who it is)
2. **Rules** (what it must and must not do)
3. An **output format** (JSON, structured text, etc.)
4. **Context** (what categories exist, what routing means)

The model will then apply these constraints to every user message it receives.
This is the difference between a general assistant and a production triage agent.

**Token note**: the system prompt counts toward your total token usage on every call.
A well-crafted, concise system prompt costs less than a verbose one - and often works better.
```

---

### Cell 22: [type: code] - Beat 3 (API) - Working Triage Call

**Purpose**: Beat 3. Full working demo with a system prompt, structured JSON output,
correct token printing, and temperature=0 for consistency. Heavily commented.

**Content**:
```python
# ------------------------------------------------------------------
# BEAT 3: Full working triage call with a system prompt
# ------------------------------------------------------------------

# A good system prompt has four components:
#   1. Role: who you are
#   2. Rules: what you must/must-not do
#   3. Categories: the routing options available
#   4. Output format: exactly what to return

system_prompt = """You are a complaint triage agent for Barclays Bank.
Your job is to read an incoming customer complaint and produce a structured triage decision.

Rules:
- You must respond in JSON only. No prose before or after the JSON.
- Do not include your reasoning in the output - only the structured decision.
- If the complaint is unclear, set category to 'unclassified'.

Categories (pick exactly one):
- 'payment_failure'  : money sent but not received, payment stuck or pending
- 'fraud_dispute'    : suspected unauthorised transaction or account takeover
- 'account_access'   : cannot log in, locked out, password or 2FA issues
- 'fee_complaint'    : unexpected charge, disputed fee
- 'unclassified'     : does not fit any category above

Output format (JSON only):
{
  "category": "<one of the five categories>",
  "urgency": "high | medium | low",
  "summary": "<one sentence, max 20 words>",
  "routing_team": "<Payments | Fraud | Digital | Fees | Triage>"
}"""

# The complaint (same one as Beat 1)
complaint_text = (
    "I tried to make a payment of 500 pounds to my landlord last night and it just "
    "disappeared. The money left my account but my landlord says he never received it. "
    "The app just shows 'pending' with no further information. I need this resolved "
    "urgently as my rent is now overdue and I am being threatened with eviction."
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},  # sets the role and rules
        {"role": "user",   "content": complaint_text}  # the complaint
    ],
    temperature=0,       # 0 = deterministic - we want consistent triage decisions
    max_tokens=200
)

# Extract the text content
raw_output = response.choices[0].message.content
print("Raw API response:")
print(raw_output)
print()

# Parse the JSON
try:
    triage = json.loads(raw_output)
    print("Parsed triage decision:")
    print(f"  Category     : {triage['category']}")
    print(f"  Urgency      : {triage['urgency']}")
    print(f"  Summary      : {triage['summary']}")
    print(f"  Routing team : {triage['routing_team']}")
except json.JSONDecodeError as e:
    print("JSON parse error:", e)
    print("Raw output was:", raw_output)

print()
print("Token usage:")
print(f"  prompt_tokens    : {response.usage.prompt_tokens}")
print(f"  completion_tokens: {response.usage.completion_tokens}")
print(f"  total_tokens     : {response.usage.total_tokens}")
print()
print("At $0.005 per 1K output tokens (gpt-4o as of 2025),")
print(f"this call cost approximately ${response.usage.total_tokens / 1000 * 0.005:.5f}")
```

**Expected output**: A JSON object with category='payment_failure', urgency='high',
a summary sentence, and routing_team='Payments'. Token counts shown. Cost shown.

**Notes**: Walk through each component of the system prompt out loud. Point at temperature=0.
"We set temperature to 0 because triage must be consistent - the same complaint must always
get the same category. Creative randomness is exactly what we do not want here."
Show the token count on the whiteboard - multiply by 1000 complaints per day. Costs matter.

---

### Cell 23: [type: markdown] - Lab 2: Build the Triage Prompt (Tier 2 - Hard)

**Purpose**: Beat 4 for the API section. Tier 2 lab (the ONE hard lab for Day 1).
Students build their own system prompt from scratch - less scaffolding than Tier 1.

**Content**:
```
### Lab 2: Build a Barclays Complaint Triage Prompt (Tier 2 - Hard)

**Situation**: The Barclays digital team has signed off on a complaint routing system
powered by GPT-4o. You are the engineer responsible for the system prompt.
The triage logic must handle five complaint categories and produce a JSON response
that the downstream routing system can parse without ambiguity.

**Task**: Write a system prompt that correctly classifies all five test complaints
below. Your prompt must:
1. Define the model's role clearly
2. List the five categories with unambiguous descriptions
3. Specify the exact JSON output format (category, urgency, summary, routing_team)
4. Handle the edge case: a complaint that fits two categories (pick the primary one)

**Action**: Fill in `my_system_prompt` below and run the test harness.
The test harness sends all five complaints and prints the triage decisions.
You have succeeded when all five complaints produce valid, sensible JSON.

**Result**: A working system prompt that classifies five real-world-style complaints.

**Stretch**: Add a sixth field `"confidence": "high | medium | low"` to your output format
and adjust the system prompt so the model sets confidence=low when the complaint is ambiguous.
```

---

### Cell 24: [type: code] - Lab 2 Starter Code

**Purpose**: The lab starter. `my_system_prompt` is the one placeholder to fill in.
The test harness and test complaints are provided.

**Content**:
```python
# Lab 2: Build a Barclays Complaint Triage Prompt
# Your job: write the system prompt. The test harness is provided.

# Step 1: Write your system prompt
my_system_prompt = None  # YOUR CODE

# --- Test complaints (do not edit) ---
test_complaints = [
    # Complaint 1: payment failure
    "I sent 1200 pounds to my sister three days ago and it still has not arrived. "
    "The transaction shows as completed on my end but she has nothing.",

    # Complaint 2: fraud dispute
    "I just got an alert for a 350 pound transaction at a shop in Manchester. "
    "I have not been to Manchester and I did not make this purchase. "
    "Please block my card immediately.",

    # Complaint 3: account access
    "I cannot log in to the app. It keeps asking for a code sent to my old phone number "
    "but I changed my number six months ago and never updated it.",

    # Complaint 4: fee complaint
    "You charged me a 25 pound overdraft fee last month but my account was only overdrawn "
    "for four hours overnight. This seems completely disproportionate.",

    # Complaint 5: ambiguous - could be payment_failure OR fraud
    "300 pounds left my account last night to someone called 'M. Johnson' but I have never "
    "heard of this person and I did not authorise any transfer to them.",
]

# --- Test harness (do not edit) ---
if my_system_prompt is not None:
    print("Running triage on 5 complaints...")
    print("=" * 60)
    for i, complaint in enumerate(test_complaints, 1):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": my_system_prompt},
                {"role": "user",   "content": complaint}
            ],
            temperature=0,
            max_tokens=150
        )
        raw = resp.choices[0].message.content
        print(f"Complaint {i}:")
        try:
            t = json.loads(raw)
            print(f"  category : {t.get('category', 'MISSING')}")
            print(f"  urgency  : {t.get('urgency', 'MISSING')}")
            print(f"  summary  : {t.get('summary', 'MISSING')}")
        except json.JSONDecodeError:
            print("  [JSON parse failed] raw output:", raw[:100])
        print()
else:
    print("my_system_prompt is None - fill it in and re-run this cell.")
```

**Notes**: Students will likely iterate 2-3 times before their prompt handles all five.
Complaint 5 (ambiguous) is the hard one - it should map to 'fraud_dispute' because the
money went to an unknown person and was unauthorised. Watch for students who map it to
'payment_failure'. Both are defensible - discuss as a class.

---

### Cell 25: [type: code] - Lab 2 Safety-Net

**Purpose**: Safety-net so downstream cells (token cost summary) run regardless.

**Content**:
```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.
if my_system_prompt is None:
    print("Using Lab 2 safety-net so the rest of the notebook can run.")
    my_system_prompt = """You are a complaint triage agent for Barclays Bank.
Respond in JSON only with fields: category, urgency, summary, routing_team.
Categories: payment_failure, fraud_dispute, account_access, fee_complaint, unclassified.
Urgency: high (financial loss or account access blocked), medium, low.
Routing teams: Payments, Fraud, Digital, Fees, Triage."""
```

---

### Cell 26: [type: markdown] - Lab 2 Homework Extension

**Purpose**: Async homework for students who want to go deeper.

**Content**:
```
**Homework Extension - Lab 2**

Your triage system is working in the notebook. Now extend it:

1. **Few-shot prompting**: Add two worked examples to your system prompt
   (a "user" message with a complaint and an "assistant" message with the correct JSON).
   Does this change the quality of the output on complaint 5 (the ambiguous one)?

2. **Edge cases**: Test your prompt on:
   - A complaint written entirely in capital letters (angry customer)
   - A complaint that is not about banking at all ("I want to complain about the weather")
   - A one-word complaint: "Fraud!"
   Does your prompt handle these gracefully? What changes would you make?

3. **Cost estimation**: Calculate the daily API cost if Barclays processes 10,000 complaints
   per day, using the token counts from your test harness calls.
   Assume the average complaint is the same length as complaint 1.
   Is this cost acceptable? What levers do you have to reduce it?

Write your answers and revised prompts in a new notebook section.
```

---

### Cell 27: [type: markdown] - Peer Discussion: Production Concerns

**Purpose**: Second peer discussion. Between the lab and the wrap-up. Covers real-world
implications of using an LLM API for financial complaint routing.

**Content**:
```
**Peer Discussion (4 min)**

You have just built a working complaint triage system using the OpenAI API.
Before Barclays could deploy this to production, what questions must the engineering
team answer? Consider:

1. **Reliability**: What happens when the OpenAI API is down? Does the complaint sit
   in a queue, or does a human take over? How do you detect the failure?

2. **Consistency**: We set temperature=0. But GPT-4o can still change between API
   versions (OpenAI updates the model). How do you test for regression?

3. **Compliance**: A Barclays complaint is personal financial data (PII). What are
   the data residency and regulatory implications of sending it to OpenAI's servers?
   (Think: GDPR, FCA, PCI-DSS.)

4. **Cost at scale**: If triage volume spikes 10x during an outage (when many customers
   complain simultaneously), what happens to your API costs and rate limits?

These are not rhetorical questions - the engineering team at Barclays is dealing with
exactly these trade-offs today. You will revisit them in the deployment topic (Day 3).
```

---

### Cell 28: [type: code] - Token Counting Demonstration

**Purpose**: Code cell to break the markdown chain after the peer discussion cell.
Also a genuinely useful demo: show how to count tokens BEFORE making an API call
so you can estimate cost without spending money.

**Content**:
```python
# Practical: count tokens before you make the API call
# This avoids surprises on the bill and helps you stay within rate limits.
#
# We use tiktoken, OpenAI's tokenizer library.
# Note: you do not need an API key to count tokens.

try:
    import tiktoken
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tiktoken"])
    import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

# Count tokens in our system prompt and a complaint
system_tokens = enc.encode(my_system_prompt)
complaint_tokens = enc.encode(test_complaints[0])

print("Token counts:")
print(f"  system prompt  : {len(system_tokens)} tokens")
print(f"  complaint 1    : {len(complaint_tokens)} tokens")
print(f"  total input    : {len(system_tokens) + len(complaint_tokens)} tokens")
print()

# Show tokenisation of a tricky word
word = "Barclays"
tokens = enc.encode(word)
print(f"How '{word}' is tokenised:")
print(f"  token ids: {tokens}")
print(f"  tokens   : {[enc.decode([t]) for t in tokens]}")
print()

word2 = "uncharacteristically"
tokens2 = enc.encode(word2)
print(f"How '{word2}' is tokenised:")
print(f"  token ids: {tokens2}")
print(f"  tokens   : {[enc.decode([t]) for t in tokens2]}")
print()
print("Rule of thumb: 1 token ~ 0.75 English words. 100 tokens ~ 75 words.")
```

**Notes**: The tokenisation of "Barclays" (likely 1 token) vs "uncharacteristically" (many tokens)
is a good talking point. Pricing is per token, not per word. Some technical jargon inflates token
count significantly.

---

### Cell 29: [type: markdown] - Few-Shot Prompting Preview

**Purpose**: Brief preview of few-shot prompting as a bridge to the next topic (Topic 2: Introducing LLMs).
This is NOT a full lab - just a concept intro to close the loop on the stretch goal.

**Content**:
```
### Preview: Few-Shot Prompting

Your Lab 2 system prompt is a **zero-shot** prompt: you describe what you want but give no examples.
GPT-4o can also learn from **examples in the prompt** - this is called few-shot prompting.

In the messages array, you can add pairs of ("user": complaint, "assistant": correct JSON)
before the actual complaint. The model treats these as demonstrations.

```python
messages = [
    {"role": "system",    "content": my_system_prompt},
    # Few-shot example 1
    {"role": "user",      "content": "I never authorised this 50 pound charge."},
    {"role": "assistant", "content": '{"category": "fraud_dispute", "urgency": "high", '
                                     '"summary": "Unauthorised 50 pound charge disputed.", '
                                     '"routing_team": "Fraud"}'},
    # Few-shot example 2
    {"role": "user",      "content": "Why was I charged a monthly fee?"},
    {"role": "assistant", "content": '{"category": "fee_complaint", "urgency": "low", '
                                     '"summary": "Customer queries monthly account fee.", '
                                     '"routing_team": "Fees"}'},
    # The actual complaint
    {"role": "user",      "content": complaint_text}
]
```

Each example pair costs tokens. More examples = higher cost = (usually) better quality.
Finding the right number is a prompt engineering decision you will practise in Topic 12.

The Homework Extension for Lab 2 asks you to try this - bring your results to the next session.
```

---

### Cell 30: [type: code] - Section Summary: The Right Tool Confirmed

**Purpose**: Code that ties the whole narrative together. Classify three complaints with
the working system and print a routing summary table. Confirms the complete pipeline works.
Also serves as a final "everything runs" verification.

**Content**:
```python
# Final demonstration: run triage on three complaints and display a routing table.
# This is the core of the Barclays complaint intelligence system you will build
# throughout this course.

demo_complaints = [
    ("Payment stuck for 3 days to landlord, eviction threatened.", "Expected: payment_failure / high"),
    ("Unauthorised 200 pound transaction appeared this morning.", "Expected: fraud_dispute / high"),
    ("Cannot log in, my old phone number is on the account.",     "Expected: account_access / medium"),
]

print("Barclays Complaint Triage System - Demo Run")
print("=" * 65)
print(f"{'#':<3} {'Category':<20} {'Urgency':<8} {'Team':<10} {'Note'}")
print("-" * 65)

for i, (complaint, note) in enumerate(demo_complaints, 1):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": my_system_prompt},
            {"role": "user",   "content": complaint}
        ],
        temperature=0,
        max_tokens=100
    )
    try:
        t = json.loads(resp.choices[0].message.content)
        cat  = t.get("category",      "???")[:18]
        urg  = t.get("urgency",       "???")[:6]
        team = t.get("routing_team",  "???")[:8]
        print(f"{i:<3} {cat:<20} {urg:<8} {team:<10} {note}")
    except json.JSONDecodeError:
        print(f"{i:<3} [JSON error]")

print("=" * 65)
print()
print("System works. This is the foundation we build on for the rest of the course.")
```

**Notes**: This cell is a satisfying end-of-section payoff. Students see their complaint triage
system working end-to-end. Instructor: "This is what Barclays actually wants to deploy. Over the
next three days you will learn how to fine-tune this, make it more reliable, and run it at scale
without relying on the OpenAI API for every single inference."

---

### Cell 31: [type: markdown] - Wrap-Up and Key Takeaways

**Purpose**: Close the notebook. Key takeaways, bridge to Topic 2.

**Content**:
```
## Wrap-Up: What You Built Today

In the last 50 minutes you:

1. **Mapped the generative AI landscape**: GAN, VAE, Diffusion, Autoregressive.
   You understand the key trade-offs and why three of the four families are not
   the right tool for text understanding.

2. **Saw the gradient problem**: You ran code that shows exactly why GANs cannot
   generate text - the argmax over a discrete vocabulary is not differentiable.

3. **Built the autoregressive loop from scratch**: embedding -> GRU -> softmax ->
   sample -> append -> repeat. You know this is the exact loop GPT-4o runs,
   just at a different scale.

4. **Called the OpenAI API**: system prompt, user message, temperature, token counts.
   You have a working complaint triage prototype for Barclays.

### Key Numbers to Remember

- Temperature range for GPT-4o: 0 to 2 (0 = deterministic, 2 = very random)
- Rule of thumb: 1 token is approximately 0.75 English words
- Zero-shot (no examples) vs few-shot (examples in prompt): more examples = more tokens = more cost
- argmax is not differentiable: this is why GANs fail for text

### What Comes Next (Topic 2: Introducing LLMs)

Topic 2 goes inside the black box. You will see:
- What a transformer architecture actually looks like in code
- What attention is and why it solves the long-range dependency problem
- How training works: cross-entropy loss, next-token prediction at scale

The complaint triage system continues - in Topic 2 you will understand *why* GPT-4o
is so much better at understanding "eviction threatened" (context) than older models.

### Questions to Bring to Topic 2

- Why does a longer context window cost more? (Hint: think about the attention mechanism.)
- If we remove the system prompt, does the model still produce JSON? Try it as homework.
- What would it take to fine-tune GPT-4o on Barclays complaint data? (You will do this in Topic 6.)
```

---

### Cell 32: [type: code] - Environment Teardown Check

**Purpose**: Final code cell. Prints a clean teardown confirmation. Good practice to end
every notebook with a "you are done" cell that also prints the key variables students created.

**Content**:
```python
# Notebook complete. Summary of what was created in this session.

print("Topic 1 Complete - Session Summary")
print("=" * 45)
print(f"  OpenAI client    : initialised (key ends ...{openai_api_key[-4:]})")
print(f"  System prompt    : {len(my_system_prompt)} characters, "
      f"{len(enc.encode(my_system_prompt))} tokens")
print(f"  Temperatures tested: {temperatures_to_test}")
print()
print("Variables available for Topic 2:")
print("  client           : OpenAI client (if key is still valid)")
print("  my_system_prompt : your triage prompt from Lab 2")
print("  test_complaints  : list of 5 test complaints")
print()
print("Next: Topic 2 - Introducing LLMs (what is inside the black box?)")
```

**Notes**: The OpenAI key will need to be re-entered in Topic 2 (new notebook, new kernel session).
This teardown cell is a good moment to remind students to rotate API keys if they accidentally
exposed them during the lab.

---

## Plan Metadata

| Field | Value |
|-------|-------|
| Topic number | 1 |
| Slug | overview_genai |
| Exercise path | `Exercises/topic_1_overview_genai/topic_1_overview_genai.ipynb` |
| Solution path | `Solutions/topic_1_overview_genai/topic_1_overview_genai.ipynb` |
| Total cells planned | 32 |
| Diagrams | 2 (genai-taxonomy, autoregressive-loop) |
| Remote training | None |
| Lab tiers | Lab 1: Tier 1 (temperature exploration), Lab 2: Tier 2 (build triage prompt) |
| Safety-net cells | Cell 15 (Lab 1), Cell 25 (Lab 2) |
| Peer discussions | Cell 7 (GAN failure implications), Cell 27 (production concerns) |
| Homework extensions | Cell 16 (Lab 1), Cell 26 (Lab 2) |
| numpy<2 pinned | Cell 2 |
| getpass for API key | Cell 18 |
| Model used | gpt-4o |
| evaluate library used | No |
| AI-tells (em dash, en dash, etc.) | None |
| Build order | Exercise first, solution second |

## Verification Checklist (for /verify-research)

- [ ] Four-beat arc present for GAN concept (Cells 6, 7, 9, 13-14) -- note: VAE/Diffusion share a compressed arc
- [ ] Four-beat arc present for autoregressive (Cells 10, 11, 12, 13-14)
- [ ] Four-beat arc present for API section (Cells 19-20, 21, 22, 23-24)
- [ ] Exactly 2 diagrams with correct paths and slugs
- [ ] No more than 3 consecutive markdown cells at any point
- [ ] Both labs have safety-net cells
- [ ] Both labs have homework extensions
- [ ] Two peer discussion prompts
- [ ] numpy<2 pinned in Cell 2
- [ ] getpass used in Cell 18
- [ ] gpt-4o used throughout (no other model)
- [ ] evaluate library NOT imported anywhere
- [ ] No em dashes, en dashes, Unicode multiplication signs, or emojis
- [ ] All # YOUR CODE placeholders are non-revealing
- [ ] Lab tiers: Lab 1 = Tier 1, Lab 2 = Tier 2 (correct for Day 1)
- [ ] Barclays narrative consistent across all cells
- [ ] Estimated time 50-60 min (plausible for 32 cells)
