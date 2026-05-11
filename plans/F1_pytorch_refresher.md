# F1: PyTorch Refresher - Cell-by-Cell Plan

## Context
- Topic: F1 - PyTorch Refresher
- Day: Framework Day (before Day 1)
- Deliverables:
  - exercises: Frameworks/pytorch_refresher.ipynb
  - solutions: Frameworks/pytorch_refresher_solution.ipynb
- Environment: SageMaker Studio kernel (no remote training)
- Source: condense PytorchPrimer 1-5 + add HuggingFace Trainer section
- Narrative: Barclays Customer Support Intelligence System - tensors=how text becomes numbers, autograd=how model learns, DataLoader=feeding complaint data, nn.Module=first classifier, nn.Sequential=clean rewrite, Trainer=production training
- Lab safety-net cells:
  - Lab 1 (X_complaints): feeds verification cell and Lab 2 discussion
  - Lab 2 (x_train, w, b, losses): feeds autograd bridge cell and Section 3
  - Lab 3 (X_500, train_loader): feeds Section 4 loader variable
  - Lab 4 (deep_model, deep_losses): feeds Section 5 discussion
  - Lab 5 (model_relu, relu_losses): feeds Section 6 narrative
  - Lab 6 (trainer6): feeds final sanity check
- Key decisions:
  - numpy<2 pinned in install cell
  - transformers>=4.35.0,<4.40.0 and tokenizers>=0.15.0,<0.20.0 (pre-built py312 wheels)
  - datasets>=2.18.0,<3.0.0
  - eval_strategy="epoch" (NOT evaluation_strategy - removed in 4.41+)
  - NO evaluate library - inline numpy for metrics
  - sagemaker>=2.200.0,<3.0.0 (v3 breaks get_execution_role)
  - matplotlib.use("Agg") for SageMaker JupyterLab
  - All synthetic data - no external dataset calls in main lab cells
  - HuggingFace Dataset (not PyTorch Dataset) for Trainer; ComplaintTrainer subclass for custom compute_loss
  - No GPU required for Sections 1-5; Section 6 auto-detects device

## Deliverables
exercises: Frameworks/pytorch_refresher.ipynb
solutions: Frameworks/pytorch_refresher_solution.ipynb

## Session Timing
Total: 90-120 min (sections only; labs add ~120 min)
- Section 1 (Tensors): 15 min
- Section 2 (Autograd): 20 min
- Section 3 (DataLoader): 15 min
- Section 4 (nn.Module): 20 min
- Section 5 (nn.Sequential): 10 min
- Section 6 (HF Trainer): 20 min

## Diagram Index
Diagram 1: slug=autograd-computation-graph, path=plans/F1_pytorch_refresher/diagrams/autograd-computation-graph.mmd
  Description: Computation graph showing how PyTorch builds a graph of operations from input tensors through loss, and how .backward() walks it in reverse to compute gradients for each parameter.

Diagram 2: slug=training-loop, path=plans/F1_pytorch_refresher/diagrams/training-loop.mmd
  Description: The four-step training loop: forward pass (predictions) -> loss computation -> backward pass (gradients) -> optimizer step (weight update). Cycles repeated for each batch and epoch.

---

# MAIN NOTEBOOK - Cell-by-Cell Content

## Cell 0 - markdown - Title and Learning Objectives
```markdown
# F1 - PyTorch Refresher
## Building a Barclays Customer Support Intelligence System

You are on the AI team at Barclays. The customer support department receives thousands
of complaint texts every day - account issues, fraud alerts, payment failures, loan
queries. Your job is to build a system that classifies each complaint automatically
so it reaches the right team.

In this notebook you build the foundation: the PyTorch skills every component of
that system depends on.

### What you will build
1. Tensors - how complaint text becomes numbers
2. Autograd - how the model learns from its mistakes
3. Dataset and DataLoader - how we feed complaint batches efficiently
4. Classifier with nn.Module - our first complaint classifier
5. Classifier with nn.Sequential - the same classifier, written cleanly
6. Training with HuggingFace Trainer - the production training loop

### Prerequisites
- Python 3.x, NumPy basics
- Basic understanding of what a neural network does (math derivations not required)

### Environment
This notebook runs entirely inside SageMaker Studio (JupyterLab kernel).
No remote training jobs. No GPU required for sections 1-5.
```

## Cell 1 - code - Install dependencies
```python
# Environment setup - runs in SageMaker Studio JupyterLab kernel
# No remote training in this notebook - all code runs locally in the kernel

import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy<2",
    "transformers>=4.35.0,<4.40.0",
    "tokenizers>=0.15.0,<0.20.0",
    "datasets>=2.18.0,<3.0.0",
    "sagemaker>=2.200.0,<3.0.0",
], check=True)

print("Install complete.")
```

## Cell 2 - code - Imports and SageMaker session
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
print("Environment OK - running in Studio kernel, no remote training needed for F1")
```

## Cell 3 - code - Core imports
```python
import torch as pt
import torch.nn as nn
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")           # SageMaker Studio: use non-interactive backend
import matplotlib.pyplot as plt

# Reproducibility seed - set once, used throughout
SEED = 42
pt.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"PyTorch version: {pt.__version__}")
print(f"CUDA available:  {pt.cuda.is_available()}")
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Device:          {device}")
```

## Cell 4 - markdown - Section 1 Header
```markdown
---
## Section 1 - Tensors: How Complaint Text Becomes Numbers

Before any model can read a customer complaint, we have to convert that text into
numbers. Those numbers live in **tensors** - the fundamental data structure in PyTorch.

A tensor is just a multi-dimensional array with one superpower: PyTorch knows how
to run it on a GPU and how to differentiate through it.

In this section you will learn to create, manipulate, and move tensors - the
operations that every downstream component depends on.
```

## Cell 5 - code - Beat 1 (Broken): Naive list approach for complaint encoding
```python
# --- Beat 1: The naive approach - why plain Python lists fail ---
#
# Suppose our customer complaints look like this (bag-of-words encoding):
# [word_count_account, word_count_fraud, word_count_payment, word_count_loan]

complaints_list = [
    [3, 0, 1, 0],   # complaint A: "account issue account payment"
    [0, 5, 0, 1],   # complaint B: "fraud fraud fraud fraud fraud loan"
    [1, 0, 4, 0],   # complaint C: "payment payment account payment payment"
]

labels_list = [0, 1, 0]   # 0=account_issue, 1=fraud

# Try to compute a dot product between complaint vectors and a weight vector
weights_list = [0.5, 2.0, 0.3, 1.0]

try:
    result = complaints_list * weights_list   # wrong: list * list = ERROR
    print(result)
except TypeError as e:
    print(f"ERROR: {e}")

# Even with a loop it is slow and has no gradients
dot_products = []
for complaint in complaints_list:
    dot = sum(c * w for c, w in zip(complaint, weights_list))
    dot_products.append(dot)

print(f"\nDot products (manual loop): {dot_products}")
print("Problem: slow for 100,000 complaints and PyTorch cannot")
print("compute gradients through plain Python loops.")
```

## Cell 6 - markdown - Beat 2: Diagram placeholder (autograd computation graph)
```markdown
<!-- DIAGRAM: autograd-computation-graph -->
[View diagram](../../plans/F1_pytorch_refresher/diagrams/autograd-computation-graph.mmd)

The diagram above shows how PyTorch builds a computation graph as you run tensor
operations. Every node records what operation created it. When you call `.backward()`,
PyTorch walks this graph in reverse to compute how much each parameter contributed
to the loss - this is what enables learning.
```

## Cell 7 - code - Beat 3 (Working): Tensor creation and operations
```python
# --- Beat 3: Tensors solve all three problems: fast, differentiable, GPU-ready ---

# 1. Create tensors from the same complaint data
complaints = pt.tensor([
    [3, 0, 1, 0],
    [0, 5, 0, 1],
    [1, 0, 4, 0],
], dtype=pt.float32)    # float32 is the standard dtype for neural nets

labels = pt.tensor([0, 1, 0], dtype=pt.long)  # long = int64, required by CrossEntropyLoss

weights = pt.tensor([0.5, 2.0, 0.3, 1.0], dtype=pt.float32)

print("complaints shape:", complaints.shape)   # [3, 4]: 3 complaints, 4 features
print("labels shape:    ", labels.shape)       # [3]
print("weights shape:   ", weights.shape)      # [4]

# 2. Matrix multiply: all 3 complaints x weights in ONE call
scores = complaints @ weights    # shape: [3]
print("\nScores (matmul):", scores)

# 3. Common creation patterns
zeros    = pt.zeros(3, 4)
ones     = pt.ones(3, 4)
rand     = pt.randn(3, 4)
arange   = pt.arange(0, 10, 2)
linspace = pt.linspace(0, 1, 5)
print("\narange:", arange)
print("linspace:", linspace)

# 4. Indexing and slicing (same as NumPy)
print("\nFirst complaint:", complaints[0])
print("Feature 1 for all complaints:", complaints[:, 1])
print("Top-left 2x2:\n", complaints[:2, :2])

# 5. Reductions
print("\nMax score:", scores.max().item())
print("Mean complaint vector:", complaints.mean(dim=0))

# 6. Reshape
flat = complaints.view(-1)    # flatten to 1D: [12]
back = flat.view(3, 4)        # back to original shape
print("\nFlattened shape:", flat.shape)
print("Reshaped shape:", back.shape)

# 7. GPU move (works even if no GPU - device is "cpu" then)
complaints_on_device = complaints.to(device)
print(f"\ncomplaints device: {complaints_on_device.device}")
```

## Cell 8 - markdown - Beat 4 Lab Instructions: Tensor Lab (Tier 1)
```markdown
### Lab 1 - Tensor Foundations (Tier 1, ~15 min)

**Situation**: You have received a batch of 200 Barclays customer complaints.
Each complaint has been encoded into 6 features (bag-of-words counts for:
account, fraud, payment, loan, dispute, refund).

**Task**: Create the complaint feature matrix and label tensor, then explore
the data structure the model will train on.

**Action**: Complete the steps below. Each `= None  # YOUR CODE` is one step.

**Result**: The verification cell at the end will confirm your shapes are correct.

Steps:
1. Create a float32 tensor named `X_complaints` with shape [200, 6] using `pt.randn`
   (random numbers simulate encoded complaint data)
2. Create a long tensor named `y_complaints` with shape [200] containing random class
   labels 0, 1, or 2 (use `pt.randint`)
3. Move both tensors to `device`
4. Compute `feature_means`: the mean value of each of the 6 features across all 200
   complaints (shape should be [6])
5. Compute `class_counts`: how many complaints fall into each class (use `pt.bincount`)

**Stretch**: Normalize `X_complaints` so each feature has mean 0 and std 1 across
all 200 samples (subtract mean, divide by std, using `dim=0`).
```

## Cell 9 - code - Lab 1 Starter Code
```python
# Lab 1 - Tensor Foundations
pt.manual_seed(SEED)

# Step 1: Create complaint feature matrix [200, 6], dtype float32
X_complaints = None  # YOUR CODE

# Step 2: Create class labels [200], dtype long, values in {0, 1, 2}
y_complaints = None  # YOUR CODE

# Step 3: Move both to device
X_complaints = None  # YOUR CODE
y_complaints = None  # YOUR CODE

# Step 4: Mean of each feature across all 200 complaints, shape [6]
feature_means = None  # YOUR CODE

# Step 5: Count of complaints in each class (0, 1, 2)
class_counts = None  # YOUR CODE

print("X_complaints shape:", X_complaints.shape if X_complaints is not None else "not set")
print("y_complaints shape:", y_complaints.shape if y_complaints is not None else "not set")
print("feature_means:", feature_means)
print("class_counts:", class_counts)
```

## Cell 10 - code - Lab 1 Safety-Net
```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
pt.manual_seed(SEED)
if X_complaints is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    X_complaints = pt.randn(200, 6, dtype=pt.float32).to(device)
    y_complaints = pt.randint(0, 3, (200,), dtype=pt.long).to(device)
    feature_means = X_complaints.mean(dim=0)
    class_counts = pt.bincount(y_complaints)
```

## Cell 11 - code - Lab 1 Verification
```python
# Verification - Lab 1
assert X_complaints is not None, "X_complaints not created"
assert X_complaints.shape == (200, 6), f"Expected (200,6), got {X_complaints.shape}"
assert X_complaints.dtype == pt.float32, "X_complaints must be float32"
assert y_complaints.shape == (200,), f"Expected (200,), got {y_complaints.shape}"
assert y_complaints.dtype == pt.long, "y_complaints must be long (int64)"
assert str(X_complaints.device).startswith(str(device).split(":")[0]), \
    f"X_complaints not on {device}"
assert feature_means.shape == (6,), f"feature_means shape wrong: {feature_means.shape}"
assert class_counts.shape == (3,), f"class_counts shape wrong: {class_counts.shape}"
print("Lab 1 passed. Shapes and dtypes are correct.")
print(f"  X_complaints: {X_complaints.shape} on {X_complaints.device}")
print(f"  feature_means: {feature_means}")
print(f"  class_counts per class: {class_counts.tolist()}")
```

## Cell 12 - markdown - Homework Extension 1
```markdown
**Homework Extension 1 - Tensor Operations**

1. Load the Yelp Polarity dataset from HuggingFace datasets and convert the first 1000
   reviews into a bag-of-words tensor using a vocabulary of the 50 most common words.
   Hint: `datasets.load_dataset("yelp_polarity")`.
2. Compute the cosine similarity between every pair of complaint vectors in
   `X_complaints`. What is the maximum similarity? What does a high cosine similarity
   between two complaint vectors mean for the classification task?
3. (Challenge) Implement a one-hot encoding function that takes a 1D integer tensor of
   class indices and returns a 2D float tensor of one-hot rows, using pure tensor
   operations only (no loops or list comprehensions).
```

## Cell 13 - markdown - Section 2 Header
```markdown
---
## Section 2 - Autograd: How the Model Learns from Mistakes

Right now our model has random weights. It will make wrong predictions on most
complaints. **Autograd** is how PyTorch figures out, for each weight, whether
increasing it would make the prediction better or worse - and by how much.

This section covers:
- `requires_grad=True`: telling PyTorch to track a tensor
- `backward()`: computing all gradients at once
- The manual training loop: forward -> loss -> backward -> update
- `no_grad()`: turning off tracking during inference
```

## Cell 14 - code - Beat 1 (Broken): Manual gradient computation
```python
# --- Beat 1: What life looks like without autograd ---
#
# Suppose we have a single weight w and want to minimize loss = (w*x - y)^2
# We need the gradient dL/dw = 2*(w*x - y)*x
# For one weight this is fine. For a 6-layer net with millions of weights,
# computing every partial derivative by hand is impossible.

# A "model" with just one weight
w = 0.5
x = 3.0
y_true = 1.0

# Forward pass
y_pred = w * x

# Loss (mean squared error)
loss = (y_pred - y_true) ** 2

# Gradient by hand: dL/dw = 2*(w*x - y)*x
gradient = 2 * (w * x - y_true) * x

print(f"y_pred: {y_pred:.4f}")
print(f"loss:   {loss:.4f}")
print(f"gradient by hand: {gradient:.4f}")

# Now imagine 6 features, 3 hidden layers, 64 units each...
# 6*64 + 64*64 + 64*64 + 64*3 = 8,576 parameters.
# Computing each partial derivative by hand would take days.
print("\nFor 8,576 parameters this approach is completely impractical.")
print("We need PyTorch autograd to compute ALL gradients in one backward() call.")
```

## Cell 15 - code - Beat 3 (Working): Autograd and manual training loop
```python
# --- Beat 3: PyTorch autograd computes all gradients for us ---

# Step 1: Mark tensors we want to differentiate through
w_auto = pt.tensor(0.5, requires_grad=True)   # our weight
x_auto = pt.tensor(3.0)                       # input (no grad needed for data)
y_true_auto = pt.tensor(1.0)

# Step 2: Forward pass - PyTorch records every operation
y_pred_auto = w_auto * x_auto              # records: mul(w, x)
loss_auto   = (y_pred_auto - y_true_auto) ** 2  # records: sub, pow

print(f"y_pred:   {y_pred_auto.item():.4f}")
print(f"loss:     {loss_auto.item():.4f}")
print(f"loss grad_fn: {loss_auto.grad_fn}")  # shows PyTorch built the graph

# Step 3: Backward pass - PyTorch walks the graph and fills .grad
loss_auto.backward()

print(f"\ndL/dw (autograd): {w_auto.grad.item():.4f}")
# Should match manual: 2*(0.5*3 - 1)*3 = 3.0
print(f"dL/dw (manual):   {2 * (w_auto.item() * x_auto.item() - y_true_auto.item()) * x_auto.item():.4f}")

# Step 4: A minimal training loop
print("\n--- Manual training loop (5 steps) ---")
w_loop = pt.tensor(0.5, requires_grad=True)
x_loop = pt.tensor(3.0)
y_loop = pt.tensor(1.0)
lr = 0.05

for step in range(5):
    # Forward
    y_pred_loop = w_loop * x_loop
    loss_loop   = (y_pred_loop - y_loop) ** 2

    # Backward
    loss_loop.backward()   # accumulates grad into w_loop.grad

    # Weight update (inside no_grad so this op is NOT recorded)
    with pt.no_grad():
        w_loop -= lr * w_loop.grad
        w_loop.grad.zero_()   # CRITICAL: zero out grad or it accumulates!

    print(f"  step {step}: loss={loss_loop.item():.4f}  w={w_loop.item():.4f}")

print(f"\nFinal w={w_loop.item():.4f}  (true ratio y/x = {y_loop.item()/x_loop.item():.4f})")

# Step 5: no_grad for inference (fast, no memory overhead)
print("\n--- Inference (no gradient tracking) ---")
with pt.no_grad():
    inference_pred = w_loop * x_loop
    print(f"Prediction: {inference_pred.item():.4f}")
    print(f"inference_pred.grad_fn: {inference_pred.grad_fn}")  # None - tracking is off
```

## Cell 16 - markdown - Beat 2: Training loop diagram
```markdown
<!-- DIAGRAM: training-loop -->
[View diagram](../../plans/F1_pytorch_refresher/diagrams/training-loop.mmd)

The diagram shows the four steps that repeat every batch:
forward (predictions) -> loss (how wrong we are) -> backward (how to fix it) ->
step (actually fix it). Understanding this loop is the foundation of everything
that follows in the course.
```

## Cell 17 - markdown - Beat 4 Lab Instructions: Autograd Lab (Tier 1)
```markdown
### Lab 2 - The Manual Training Loop (Tier 1, ~20 min)

**Situation**: You have a single complaint feature (average sentence length) and
you want to learn a linear relationship: `severity_score = w * sentence_length + b`.
This is the simplest possible model - but it uses the exact same training loop as
a 100-layer transformer.

**Task**: Implement the complete training loop: forward -> loss -> backward -> update.

**Action**: Complete the steps marked `# YOUR CODE`.

**Result**: After training, your learned `w` and `b` should produce predictions
close to the true scores. The verification cell plots training loss.

Steps:
1. Create `x_train` (100 sentence lengths, float32) and
   `y_train` (true severity scores = 0.7 * x_train + 0.3 + noise)
2. Initialize `w` and `b` as float32 tensors with `requires_grad=True`
3. Implement the forward function: `y_pred = w * x_train + b`
4. Implement the MSE loss: `loss = ((y_pred - y_train)**2).mean()`
5. Call `loss.backward()`
6. Update `w` and `b` with learning rate 0.01 (inside `pt.no_grad()`)
7. Zero the gradients
8. Run 100 epochs and record `losses` (a Python list of float values)

**Stretch**: Plot the learned line on top of the scatter plot of (x_train, y_train).
Add a title and axis labels that reference the Barclays complaint context.
```

## Cell 18 - code - Lab 2 Starter Code
```python
# Lab 2 - Manual Training Loop
pt.manual_seed(SEED)

# Step 1: Synthetic complaint dataset
x_train = None  # YOUR CODE: pt.rand, shape [100], dtype float32
y_train = None  # YOUR CODE: 0.7 * x_train + 0.3 + small noise

# Step 2: Initialize learnable parameters
w = None  # YOUR CODE: scalar tensor, requires_grad=True
b = None  # YOUR CODE: scalar tensor, requires_grad=True

lr = 0.01
losses = []

for epoch in range(100):
    # Step 3: Forward pass
    y_pred = None  # YOUR CODE

    # Step 4: Loss (MSE)
    loss = None  # YOUR CODE

    # Step 5: Backward pass
    # YOUR CODE (one line)

    # Step 6: Parameter update (inside no_grad)
    with pt.no_grad():
        pass  # YOUR CODE: update w and b

    # Step 7: Zero gradients
    # YOUR CODE (two lines, one per parameter)

    losses.append(loss.item() if loss is not None else 0.0)

if w is not None:
    print(f"Final w={w.item():.4f} (true: 0.7000)")
    print(f"Final b={b.item():.4f} (true: 0.3000)")
    print(f"Final loss: {losses[-1]:.6f}")
```

## Cell 19 - code - Lab 2 Safety-Net
```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.
pt.manual_seed(SEED)
if x_train is None or not isinstance(w, pt.Tensor) or not losses or losses[-1] == 0.0:
    print("Using Lab 2 safety-net so the rest of the notebook can run.")
    x_train = pt.rand(100, dtype=pt.float32)
    y_train = 0.7 * x_train + 0.3 + 0.05 * pt.randn(100)
    w = pt.tensor(0.0, requires_grad=True)
    b = pt.tensor(0.0, requires_grad=True)
    lr = 0.01
    losses = []
    for epoch in range(100):
        y_pred = w * x_train + b
        loss   = ((y_pred - y_train) ** 2).mean()
        loss.backward()
        with pt.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        losses.append(loss.item())
```

## Cell 20 - code - Lab 2 Verification
```python
# Verification - Lab 2
assert losses is not None and len(losses) == 100, "losses must have 100 values"
assert losses[-1] < losses[0], "Loss should decrease over training"
assert abs(w.item() - 0.7) < 0.15, f"w={w.item():.4f} too far from 0.7"
assert abs(b.item() - 0.3) < 0.15, f"b={b.item():.4f} too far from 0.3"

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Lab 2 - Training Loss (Barclays Complaint Severity Model)")
plt.tight_layout()
plt.savefig("/tmp/lab2_loss.png", dpi=80)
plt.show()
print(f"Lab 2 passed. Learned w={w.item():.4f}, b={b.item():.4f}")
```

## Cell 21 - markdown - Homework Extension 2
```markdown
**Homework Extension 2 - Autograd**

1. Modify the training loop to use **momentum**: instead of `w -= lr * w.grad`,
   implement `velocity = 0.9 * velocity + w.grad` and `w -= lr * velocity`.
   Does it converge faster?
2. Implement **gradient clipping**: before the update step, if `w.grad.abs() > 1.0`,
   scale it down to 1.0. This is used in transformers to prevent exploding gradients.
3. (Challenge) Implement the exact same training loop using `torch.optim.SGD` with
   momentum. Compare the final loss to your manual implementation. They should match
   if your momentum implementation was correct.
```

## Cell 22 - markdown - Discussion Prompt 1
```markdown
### Discussion (3 min) - Gradient Descent in Production

At Barclays, a fraud detection model is retrained every night on the previous day's
transactions. Consider:

1. The learning rate is a hyperparameter set by the team. Too high and training
   diverges; too low and it barely improves overnight. How would you find the right
   learning rate without running expensive experiments?
2. The training loop zeros gradients every step. What would happen if you forgot
   to zero them? Would the model still converge? Would it converge to the same place?
3. In production, should the fraud model do inference inside `pt.no_grad()`?
   What is the cost of forgetting this?

Share your answers with the person next to you. No wrong answers - the point is
to think about production tradeoffs.
```

## Cell 23 - code - Bridge: AdamW replaces the manual update step
```python
# Quick bridge: optimizers handle the update step for us
# Instead of:
#   w -= lr * w.grad
#   w.grad.zero_()
# We use:
#   optimizer.step()
#   optimizer.zero_grad()

from torch import optim

pt.manual_seed(SEED)
x_demo = pt.rand(100, dtype=pt.float32)
y_demo = 0.7 * x_demo + 0.3 + 0.05 * pt.randn(100)

w2 = pt.tensor(0.0, requires_grad=True)
b2 = pt.tensor(0.0, requires_grad=True)

optimizer = optim.AdamW([w2, b2], lr=0.05)

for epoch in range(100):
    y_pred2 = w2 * x_demo + b2
    loss2   = ((y_pred2 - y_demo) ** 2).mean()
    loss2.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"AdamW - Final w={w2.item():.4f} b={b2.item():.4f} loss={loss2.item():.6f}")
print("From here on we always use optimizer.step() and optimizer.zero_grad()")
```

## Cell 24 - markdown - Section 3 Header
```markdown
---
## Section 3 - Dataset and DataLoader: Feeding Complaint Batches Efficiently

In the previous section we trained on the full dataset every step. Real complaint
datasets at Barclays have millions of records - you cannot load them all into a tensor
and do matrix math on all of them at once (memory runs out).

The solution is **mini-batch training**: feed the model small batches of complaints,
update weights after each batch, and repeat until you have seen the full dataset
(one epoch).

`Dataset` defines what your data looks like. `DataLoader` handles shuffling, batching,
and (in production) parallel data loading from disk.
```

## Cell 25 - code - Beat 1 (Broken): Manual batching
```python
# --- Beat 1: Manual batching is error-prone ---
pt.manual_seed(SEED)

X_full = pt.randn(200, 6)
y_full = pt.randint(0, 3, (200,))

BATCH_SIZE = 32

print("Manual batch loop:")
for i in range(0, len(X_full), BATCH_SIZE):
    X_batch = X_full[i : i + BATCH_SIZE]
    y_batch = y_full[i : i + BATCH_SIZE]
    # Problem 1: last batch may be smaller - model might not handle it
    # Problem 2: data is NOT shuffled - model sees same order every epoch
    # Problem 3: no parallel loading - slow for data on disk
    if i == 0:
        print(f"  batch 0 shape: {X_batch.shape}")

last_start = (len(X_full) // BATCH_SIZE) * BATCH_SIZE
X_last = X_full[last_start:]
print(f"  last batch shape: {X_last.shape}")   # 200 % 32 = 8 samples
print("\nProblems: no shuffle, variable last-batch size, no parallel loading.")
print("PyTorch DataLoader solves all three.")
```

## Cell 26 - code - Beat 3 (Working): Dataset and DataLoader
```python
# --- Beat 3: Dataset and DataLoader ---
from torch.utils.data import Dataset, DataLoader, TensorDataset

pt.manual_seed(SEED)
X_full = pt.randn(200, 6, dtype=pt.float32).to(device)
y_full = pt.randint(0, 3, (200,), dtype=pt.long).to(device)

# 1. TensorDataset: the simplest Dataset - wraps existing tensors
dataset_demo = TensorDataset(X_full, y_full)
print(f"Dataset length: {len(dataset_demo)}")
print(f"First sample X shape: {dataset_demo[0][0].shape}")
print(f"First sample y:       {dataset_demo[0][1]}")

# 2. DataLoader: handles batching, shuffling, drop_last
loader_demo = DataLoader(
    dataset_demo,
    batch_size=32,
    shuffle=True,     # shuffle at the start of EACH epoch
    drop_last=False,  # keep the last partial batch
)
print(f"\nNumber of batches per epoch: {len(loader_demo)}")   # ceil(200/32) = 7

# 3. Iterate over one epoch
print("\nBatch shapes in one epoch:")
for batch_idx, (X_batch, y_batch) in enumerate(loader_demo):
    if batch_idx < 3 or batch_idx == len(loader_demo) - 1:
        print(f"  batch {batch_idx}: X={X_batch.shape} y={y_batch.shape}")
    elif batch_idx == 3:
        print("  ...")

# 4. Custom Dataset for more complex data (e.g. text that needs tokenization)
class ComplaintDataset(Dataset):
    """A Dataset for Barclays complaint records.

    In production this would tokenize raw text. Here we use pre-computed
    feature vectors to keep the example self-contained.
    """
    def __init__(self, features, labels):
        self.X = features.to(pt.float32)
        self.y = labels.to(pt.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

complaint_ds     = ComplaintDataset(X_full, y_full)
complaint_loader = DataLoader(complaint_ds, batch_size=32, shuffle=True)
X_b, y_b = next(iter(complaint_loader))
print(f"\nCustom Dataset - first batch: X={X_b.shape} y={y_b.shape}")
```

## Cell 27 - markdown - Beat 4 Lab Instructions: DataLoader Lab (Tier 1)
```markdown
### Lab 3 - DataLoader for Complaint Batches (Tier 1, ~15 min)

**Situation**: Your team has a synthetic dataset of 500 Barclays complaints with
8 features each, in 4 complaint categories (account, fraud, payment, general).

**Task**: Build a `ComplaintDataset`, wrap it in a `DataLoader`, and run one full
epoch of a dummy forward pass to confirm shapes are correct.

**Action**: Complete the steps marked `# YOUR CODE`.

**Result**: The verification cell counts the total samples seen across all batches
and confirms it equals 500.

Steps:
1. Create `X_500` (shape [500, 8], float32) and `y_500` (shape [500], long, values 0-3)
2. Implement the `ComplaintDataset.__getitem__` method (one line)
3. Create `train_loader` with `batch_size=64` and `shuffle=True`
4. Iterate over `train_loader` for one epoch, accumulating `total_seen`

**Stretch**: Add a `transform` parameter to `ComplaintDataset.__init__` that accepts
a callable. When `transform` is not None, apply it to `self.X[idx]` before returning.
Test it by passing a lambda that normalizes each row to unit norm.
```

## Cell 28 - code - Lab 3 Starter Code
```python
# Lab 3 - DataLoader for Complaint Batches
pt.manual_seed(SEED)

# Step 1: Create 500-complaint dataset
X_500 = None  # YOUR CODE: shape [500, 8], dtype float32
y_500 = None  # YOUR CODE: shape [500], dtype long, values 0-3

class ComplaintDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features.to(pt.float32)
        self.y = labels.to(pt.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return None  # YOUR CODE: return (X[idx], y[idx])

# Step 3: DataLoader
train_loader = None  # YOUR CODE: DataLoader with batch_size=64, shuffle=True

# Step 4: One epoch - count total samples seen
total_seen = 0
if train_loader is not None:
    for X_batch, y_batch in train_loader:
        total_seen += len(X_batch)

print(f"Total samples seen in one epoch: {total_seen}")
print(f"Expected: 500")
```

## Cell 29 - code - Lab 3 Safety-Net
```python
# Lab 3 safety-net: run this if you did not finish Lab 3.
# SKIP this cell if you DID finish Lab 3.
pt.manual_seed(SEED)
if X_500 is None or train_loader is None:
    print("Using Lab 3 safety-net so the rest of the notebook can run.")
    X_500 = pt.randn(500, 8, dtype=pt.float32)
    y_500 = pt.randint(0, 4, (500,), dtype=pt.long)

    class _ComplaintDataset(Dataset):
        def __init__(self, features, labels):
            self.X = features.to(pt.float32)
            self.y = labels.to(pt.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    _ds = _ComplaintDataset(X_500, y_500)
    train_loader = DataLoader(_ds, batch_size=64, shuffle=True)
    total_seen = sum(len(yb) for _, yb in train_loader)
```

## Cell 30 - code - Lab 3 Verification
```python
# Verification - Lab 3
assert total_seen == 500, f"Expected 500 samples, got {total_seen}"
X_b, y_b = next(iter(train_loader))
assert X_b.shape[1] == 8, f"Expected 8 features, got {X_b.shape[1]}"
assert y_b.dtype == pt.long, "y_b must be long dtype"
print(f"Lab 3 passed. {total_seen} samples seen across {len(train_loader)} batches.")
print(f"Batch shape: X={X_b.shape}, y={y_b.shape}")
```

## Cell 31 - markdown - Homework Extension 3
```markdown
**Homework Extension 3 - DataLoader**

1. Implement a `WeightedComplaintSampler` using `torch.utils.data.WeightedRandomSampler`.
   Fraud complaints (class 1) are rare - only 5% of the dataset. Make the sampler
   draw class 1 samples 10x more often to handle class imbalance.
2. Measure the throughput difference (samples/second) between `num_workers=0` and
   `num_workers=4` for a DataLoader with a dataset of 10,000 samples.
3. (Challenge) Build a `TextComplaintDataset` that reads raw complaint strings from
   a Python list, tokenizes them character-by-character, pads all sequences to the
   same length, and returns `(char_ids_tensor, label)`.
```

## Cell 32 - markdown - Section 4 Header
```markdown
---
## Section 4 - Complaint Classifier with nn.Module

We now have tensors, autograd, and data loading. It is time to build the first real
classifier: a two-layer neural network that takes complaint feature vectors and
predicts one of three categories (account issue, fraud, payment problem).

`nn.Module` is the base class for every neural network in PyTorch. You subclass it,
define your layers in `__init__`, and implement `forward()`. PyTorch handles
parameter registration, device placement, and gradient tracking automatically.
```

## Cell 33 - code - Beat 1 (Broken): Classifier without nn.Module
```python
# --- Beat 1: Building a classifier with raw tensors (do not do this) ---
pt.manual_seed(SEED)

W1 = pt.randn(6, 16, requires_grad=True)
b1 = pt.zeros(16, requires_grad=True)
W2 = pt.randn(16, 3, requires_grad=True)
b2 = pt.zeros(3, requires_grad=True)

def bad_forward(x):
    h = pt.relu(x @ W1 + b1)
    return h @ W2 + b2

X_sample = pt.randn(10, 6)
out = bad_forward(X_sample)
print(f"Output shape: {out.shape}")   # works so far

# Problem 1: saving requires manually listing ALL tensors
# Problem 2: moving to GPU requires updating every tensor by hand
# Problem 3: no model.eval(), model.train(), model.parameters()

try:
    bad_forward.to(device)
except AttributeError as e:
    print(f"\nERROR: {e}")

print("\nWe need nn.Module so PyTorch can manage parameters, device placement,")
print("and train/eval mode automatically.")
```

## Cell 34 - code - Beat 3 (Working): nn.Module Classifier with full training loop
```python
# --- Beat 3: Complaint classifier with nn.Module ---
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class ComplaintClassifier(nn.Module):
    """Two-layer feedforward classifier for Barclays complaint categories.

    Input:  6-dimensional complaint feature vector
    Output: 3-class logits (account_issue=0, fraud=1, payment=2)
    """
    def __init__(self, input_dim=6, hidden_dim=16, num_classes=3):
        super().__init__()
        # nn.Linear registers weights and bias as Parameters automatically
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.relu(self.layer1(x))   # [batch, hidden_dim]
        return self.layer2(h)           # [batch, num_classes]

pt.manual_seed(SEED)
model = ComplaintClassifier().to(device)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
# 6*16 + 16 + 16*3 + 3 = 163

# Synthetic complaint data (300 samples, 6 features, 3 classes)
pt.manual_seed(SEED)
X_data = pt.randn(300, 6, dtype=pt.float32).to(device)
y_data = pt.randint(0, 3, (300,), dtype=pt.long).to(device)
dataset = TensorDataset(X_data, y_data)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()    # combines LogSoftmax + NLL
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("\n--- Training ComplaintClassifier (5 epochs) ---")
for epoch in range(5):
    model.train()    # enables dropout, batch norm if present (good habit)
    total_loss = 0.0
    correct    = 0

    for X_batch, y_batch in loader:
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()

    avg_loss = total_loss / len(dataset)
    accuracy = correct / len(dataset)
    print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  acc={accuracy:.3f}")

# Inference mode
model.eval()
with pt.no_grad():
    sample = pt.randn(1, 6, dtype=pt.float32).to(device)
    logits = model(sample)
    probs  = pt.softmax(logits, dim=1)
    pred   = logits.argmax(dim=1)
    categories = ["account_issue", "fraud", "payment"]
    print(f"\nSample prediction: {categories[pred.item()]}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs[0].tolist()]}")
```

## Cell 35 - markdown - Beat 4 Lab Instructions: nn.Module Lab (Tier 1)
```markdown
### Lab 4 - Build a Deeper Complaint Classifier (Tier 1, ~20 min)

**Situation**: The two-layer classifier from Beat 3 gets about 35% accuracy on
random data (roughly chance for 3 classes). Your team wants to try a deeper network:
three hidden layers with dropout for regularization.

**Task**: Build a `DeeperComplaintClassifier` with three hidden layers (sizes 32, 16, 8)
and dropout (p=0.3) after the first two hidden layers. Train it for 10 epochs.

**Action**: Complete the steps marked `# YOUR CODE`.

**Result**: The verification cell confirms the model has more than 300 parameters
and that training loss decreases over 10 epochs.

Steps:
1. In `__init__`: create `self.layer1` (6->32), `self.drop1` (p=0.3),
   `self.layer2` (32->16), `self.drop2` (p=0.3), `self.layer3` (16->8),
   `self.output` (8->3), and `self.relu`
2. In `forward`: chain them (layer1 -> relu -> drop1 -> layer2 -> relu ->
   drop2 -> layer3 -> relu -> output)
3. Instantiate the model, move to `device`
4. Create `optimizer` (AdamW, lr=1e-3) and `criterion` (CrossEntropyLoss)
5. Train for 10 epochs using the existing `loader` from Beat 3
6. Record training losses in a list `deep_losses`

**Stretch**: Add a `batch_norm` layer (`nn.BatchNorm1d(32)`) between `layer1` and
`relu`. Does it train faster (fewer epochs to reach the same loss)?
```

## Cell 36 - code - Lab 4 Starter Code
```python
# Lab 4 - Deeper Complaint Classifier with nn.Module
pt.manual_seed(SEED)

class DeeperComplaintClassifier(nn.Module):
    """Three-layer classifier with dropout."""
    def __init__(self, input_dim=6, num_classes=3):
        super().__init__()
        # Step 1: define layers
        self.layer1 = None  # YOUR CODE: Linear 6->32
        self.drop1  = None  # YOUR CODE: Dropout p=0.3
        self.layer2 = None  # YOUR CODE: Linear 32->16
        self.drop2  = None  # YOUR CODE: Dropout p=0.3
        self.layer3 = None  # YOUR CODE: Linear 16->8
        self.output = None  # YOUR CODE: Linear 8->num_classes
        self.relu   = nn.ReLU()

    def forward(self, x):
        # Step 2: chain layers
        pass  # YOUR CODE: return final logits

# Step 3: Instantiate and move to device
deep_model = None  # YOUR CODE

# Step 4: Loss and optimizer
deep_criterion = None  # YOUR CODE
deep_optimizer = None  # YOUR CODE

# Steps 5-6: Training loop (10 epochs)
deep_losses = []

if deep_model is not None:
    for epoch in range(10):
        deep_model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            logits = None  # YOUR CODE: forward pass
            loss   = None  # YOUR CODE: compute loss
            # YOUR CODE: zero_grad, backward, step
            epoch_loss += loss.item() * len(y_batch) if loss is not None else 0
        deep_losses.append(epoch_loss / len(dataset))
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={deep_losses[-1]:.4f}")
```

## Cell 37 - code - Lab 4 Safety-Net
```python
# Lab 4 safety-net: run this if you did not finish Lab 4.
# SKIP this cell if you DID finish Lab 4.
pt.manual_seed(SEED)
if deep_model is None or not deep_losses:
    print("Using Lab 4 safety-net so the rest of the notebook can run.")

    class _DeeperComplaintClassifier(nn.Module):
        def __init__(self, input_dim=6, num_classes=3):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, 32)
            self.drop1  = nn.Dropout(0.3)
            self.layer2 = nn.Linear(32, 16)
            self.drop2  = nn.Dropout(0.3)
            self.layer3 = nn.Linear(16, 8)
            self.output = nn.Linear(8, num_classes)
            self.relu   = nn.ReLU()
        def forward(self, x):
            x = self.drop1(self.relu(self.layer1(x)))
            x = self.drop2(self.relu(self.layer2(x)))
            x = self.relu(self.layer3(x))
            return self.output(x)

    deep_model     = _DeeperComplaintClassifier().to(device)
    deep_criterion = nn.CrossEntropyLoss()
    deep_optimizer = optim.AdamW(deep_model.parameters(), lr=1e-3)
    deep_losses    = []
    for epoch in range(10):
        deep_model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            logits = deep_model(X_batch)
            loss   = deep_criterion(logits, y_batch)
            deep_optimizer.zero_grad()
            loss.backward()
            deep_optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
        deep_losses.append(epoch_loss / len(dataset))
```

## Cell 38 - code - Lab 4 Verification
```python
# Verification - Lab 4
assert deep_model is not None, "deep_model not created"
n_params = sum(p.numel() for p in deep_model.parameters())
assert n_params > 300, f"Expected >300 params, got {n_params}"
assert len(deep_losses) == 10, "Expected 10 loss values"
assert deep_losses[-1] < deep_losses[0], "Loss should decrease"
print(f"Lab 4 passed. Parameters: {n_params}")
print(f"Loss went from {deep_losses[0]:.4f} to {deep_losses[-1]:.4f}")
```

## Cell 39 - markdown - Homework Extension 4
```markdown
**Homework Extension 4 - nn.Module**

1. Add a `save_checkpoint` method to `DeeperComplaintClassifier` that calls
   `torch.save(self.state_dict(), path)` and a `load_checkpoint` class method
   that loads it back. Verify the loaded model produces identical predictions.
2. Register a forward hook on `self.layer2` using `module.register_forward_hook(fn)`.
   Print the shape of the intermediate activation every time `forward` is called.
   Hooks are how PyTorch visualizations (like Captum) inspect model internals.
3. (Challenge) Implement L2 regularization manually: after `loss.backward()` but
   before `optimizer.step()`, add `lambda_val * param.grad.add_(param.data)` for each
   parameter. Compare this to `AdamW(weight_decay=lambda_val)`. Are the results identical?
```

## Cell 40 - markdown - Discussion Prompt 2
```markdown
### Discussion (3 min) - Model Architecture Decisions

Your team is arguing about how to design the Barclays complaint classifier:

1. One engineer wants 10 hidden layers with 512 units each. Another wants 2 hidden
   layers with 32 units. You have 300 labeled complaints. Which is more dangerous
   and why?
2. Dropout randomly zeros out neurons during training. Why does this help the model
   generalize better? What does "generalize" mean in the fraud detection context?
3. We used `nn.CrossEntropyLoss`. Why not use MSE loss for a classification problem?
   What would go wrong?
```

## Cell 41 - markdown - Section 5 Header
```markdown
---
## Section 5 - Cleaner Classifiers with nn.Sequential

`nn.Module` with explicit `__init__` and `forward` is the right choice when you
need custom logic (skip connections, branching, shared weights). But for simple
feedforward networks like our complaint classifier, `nn.Sequential` is cleaner:
you describe the layers as a sequence and PyTorch wires `forward` automatically.

Think of it like a pipeline: complaint features go in one end, class logits come
out the other end, no branching.
```

## Cell 42 - code - Beat 1 (Broken): Sequential with wrong layer order
```python
# --- Beat 1: Wrong layer order in nn.Sequential ---
pt.manual_seed(SEED)

# A common mistake: layers in wrong order or wrong dimensions
bad_seq = nn.Sequential(
    nn.Linear(6, 16),
    nn.Linear(3, 16),   # WRONG: expects 3 inputs but previous layer outputs 16
    nn.ReLU(),
    nn.Linear(16, 3),
)

sample_bad = pt.randn(4, 6)
try:
    out_bad = bad_seq(sample_bad)
    print(f"Output shape: {out_bad.shape}")
except RuntimeError as e:
    print(f"ERROR: {e}")

print("\nnn.Sequential does not validate layer dimensions at construction time.")
print("The error only surfaces when you run the forward pass.")
print("Always test with a dummy input immediately after defining the model.")
```

## Cell 43 - code - Beat 3 (Working): nn.Sequential complaint classifier
```python
# --- Beat 3: Complaint classifier with nn.Sequential ---
pt.manual_seed(SEED)

# The same architecture as DeeperComplaintClassifier, expressed as nn.Sequential
seq_model = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
).to(device)

# Always test with a dummy input before training
dummy     = pt.randn(1, 6, dtype=pt.float32).to(device)
dummy_out = seq_model(dummy)
print(f"Dummy forward pass output shape: {dummy_out.shape}")   # [1, 3]

n_params = sum(p.numel() for p in seq_model.parameters())
print(f"Parameters: {n_params}")

# The same training loop works unchanged - nn.Sequential IS an nn.Module
seq_criterion = nn.CrossEntropyLoss()
seq_optimizer = optim.AdamW(seq_model.parameters(), lr=1e-3)
seq_losses    = []

print("\n--- Training Sequential Classifier (10 epochs) ---")
for epoch in range(10):
    seq_model.train()
    epoch_loss = 0.0
    correct    = 0
    for X_batch, y_batch in loader:
        logits = seq_model(X_batch)
        loss   = seq_criterion(logits, y_batch)
        seq_optimizer.zero_grad()
        loss.backward()
        seq_optimizer.step()
        epoch_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
    seq_losses.append(epoch_loss / len(dataset))
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}: loss={seq_losses[-1]:.4f}  "
              f"acc={correct/len(dataset):.3f}")

seq_model.eval()
with pt.no_grad():
    test_complaint = pt.randn(3, 6, dtype=pt.float32).to(device)
    logits_test    = seq_model(test_complaint)
    probs_test     = pt.softmax(logits_test, dim=1)
    preds_test     = logits_test.argmax(dim=1)
    cats           = ["account_issue", "fraud", "payment"]
    print("\nBatch predictions:")
    for i, (p, prob) in enumerate(zip(preds_test, probs_test)):
        print(f"  Complaint {i}: {cats[p.item()]}  (conf={prob[p].item():.3f})")
```

## Cell 44 - markdown - Beat 4 Lab Instructions: Sequential Lab (Tier 1)
```markdown
### Lab 5 - Sequential Multi-Class Classifier (Tier 1, ~15 min)

**Situation**: The Barclays data science team wants to experiment with different
activation functions. Your team lead asks you to build two variants of the
complaint classifier and compare their training loss after 5 epochs:
- Variant A: uses `nn.ReLU` (what we have been using)
- Variant B: uses `nn.Tanh` (an older activation, smoother gradient)

**Task**: Build both models as `nn.Sequential`, train both for 5 epochs on the
same data, and print which variant has lower final loss.

**Action**: Complete the steps marked `# YOUR CODE`.

**Result**: The verification cell confirms both models were trained and compares
their final losses.

Steps:
1. Define `model_relu` as `nn.Sequential` with architecture: 6->32->ReLU->16->ReLU->3
2. Define `model_tanh` as `nn.Sequential` with architecture: 6->32->Tanh->16->Tanh->3
3. Train each for 5 epochs on `loader`, record losses in `relu_losses` and `tanh_losses`
4. Print which model won (lower final loss)

**Stretch**: Add a third variant using `nn.LeakyReLU(negative_slope=0.01)`. Plot
all three loss curves on the same axes for visual comparison.
```

## Cell 45 - code - Lab 5 Starter Code
```python
# Lab 5 - Sequential: ReLU vs Tanh
pt.manual_seed(SEED)

# Step 1: Model A - ReLU activations
model_relu = None  # YOUR CODE: nn.Sequential with ReLU

# Step 2: Model B - Tanh activations
model_tanh = None  # YOUR CODE: nn.Sequential with Tanh

relu_losses = []
tanh_losses = []

def train_5_epochs(model, losses_list):
    """Train model for 5 epochs, appending avg loss per epoch to losses_list."""
    if model is None:
        return
    opt  = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            logits = None  # YOUR CODE: forward pass
            loss   = None  # YOUR CODE: compute loss
            # YOUR CODE: zero_grad, backward, step
            epoch_loss += loss.item() * len(y_batch) if loss is not None else 0
        losses_list.append(epoch_loss / len(dataset))

# Step 3: Train both
train_5_epochs(model_relu, relu_losses)
train_5_epochs(model_tanh, tanh_losses)

# Step 4: Compare
if relu_losses and tanh_losses:
    winner = "ReLU" if relu_losses[-1] < tanh_losses[-1] else "Tanh"
    print(f"ReLU final loss: {relu_losses[-1]:.4f}")
    print(f"Tanh final loss: {tanh_losses[-1]:.4f}")
    print(f"Winner on this random data: {winner}")
```

## Cell 46 - code - Lab 5 Safety-Net
```python
# Lab 5 safety-net: run this if you did not finish Lab 5.
# SKIP this cell if you DID finish Lab 5.
pt.manual_seed(SEED)
if model_relu is None or not relu_losses:
    print("Using Lab 5 safety-net so the rest of the notebook can run.")

    def _make_model(act_class):
        return nn.Sequential(
            nn.Linear(6, 32), act_class(), nn.Linear(32, 16), act_class(), nn.Linear(16, 3)
        ).to(device)

    model_relu  = _make_model(nn.ReLU)
    model_tanh  = _make_model(nn.Tanh)
    relu_losses = []
    tanh_losses = []

    for mdl, llist in [(model_relu, relu_losses), (model_tanh, tanh_losses)]:
        opt  = optim.AdamW(mdl.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for _ in range(5):
            mdl.train()
            el = 0.0
            for Xb, yb in loader:
                lg = mdl(Xb)
                ls = crit(lg, yb)
                opt.zero_grad(); ls.backward(); opt.step()
                el += ls.item() * len(yb)
            llist.append(el / len(dataset))
```

## Cell 47 - code - Lab 5 Verification
```python
# Verification - Lab 5
assert len(relu_losses) == 5, "relu_losses must have 5 values"
assert len(tanh_losses) == 5, "tanh_losses must have 5 values"
assert relu_losses[-1] < relu_losses[0] or tanh_losses[-1] < tanh_losses[0], \
    "At least one model should show decreasing loss"
print("Lab 5 passed.")
print(f"  ReLU: {relu_losses[0]:.4f} -> {relu_losses[-1]:.4f}")
print(f"  Tanh: {tanh_losses[0]:.4f} -> {tanh_losses[-1]:.4f}")
```

## Cell 48 - markdown - Homework Extension 5
```markdown
**Homework Extension 5 - nn.Sequential**

1. Use `torch.nn.utils.prune.l1_unstructured` to prune 30% of the weights in
   `model_relu[0]` (the first Linear layer). Compare inference speed before and after
   pruning. Does accuracy change?
2. Export `model_relu` to ONNX format using `torch.onnx.export`. Load it with
   `onnxruntime` and confirm it produces identical predictions to the PyTorch model.
   ONNX export is the standard way to deploy PyTorch models to production serving.
3. (Challenge) Implement an `nn.Sequential`-style model using only `nn.ModuleList`.
   Override `forward` to iterate over the module list. Confirm it produces the same
   output as `model_relu` for the same input.
```

## Cell 49 - markdown - Section 6 Header
```markdown
---
## Section 6 - HuggingFace Trainer: The Production Training Loop

Everything we have built so far is correct PyTorch. But in production, teams at
companies like Barclays use the **HuggingFace Trainer** instead of writing their
own training loop. Trainer adds:

- Automatic evaluation every N epochs
- Checkpointing (save and resume training)
- Learning rate scheduling
- Mixed precision (fp16/bf16) - roughly 2x faster on GPU
- Distributed training across multiple GPUs
- Progress bars and structured logging

The trade-off: Trainer is less transparent than a manual loop. You need to
understand the manual loop (Sections 2-5) before Trainer becomes a tool, not
a black box.

### The Dataset format Trainer expects
Trainer uses HuggingFace `datasets.Dataset` objects (not PyTorch `Dataset`).
They behave like pandas DataFrames with Arrow-backed columnar storage.
```

## Cell 50 - code - Beat 1 (Broken): Passing wrong dataset type to Trainer
```python
# --- Beat 1: Passing a PyTorch Dataset to Trainer (wrong type) ---
from transformers import Trainer, TrainingArguments

# Use the seq_model from Section 5 - Trainer wraps any nn.Module
# But what happens if we pass the wrong dataset type?

try:
    bad_args = TrainingArguments(
        output_dir="/tmp/bad_trainer_test",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        no_cuda=True,
    )
    # dataset here is the PyTorch TensorDataset from Section 4 - wrong type for Trainer
    bad_trainer = Trainer(
        model=seq_model,
        args=bad_args,
        train_dataset=dataset,   # PyTorch TensorDataset, not HF Dataset
    )
    bad_trainer.train()
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTrainer expects a HuggingFace datasets.Dataset, not a PyTorch Dataset.")
print("We need to convert our data using datasets.Dataset.from_dict()")
```

## Cell 51 - code - Beat 3 (Working): HuggingFace Trainer full demo
```python
# --- Beat 3: HuggingFace Trainer for Complaint Classification ---
import numpy as np
from datasets import Dataset as HFDataset
from transformers import Trainer, TrainingArguments

# 1. Rebuild model fresh (Trainer will manage the state)
pt.manual_seed(SEED)
hf_model = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(device)

# 2. Create HuggingFace datasets from numpy arrays
#    Trainer expects dicts with string keys; "labels" is the magic key for loss
pt.manual_seed(SEED)
X_np = pt.randn(300, 6).numpy().astype(np.float32)
y_np = np.random.randint(0, 3, size=300).astype(np.int64)

split    = int(0.8 * len(X_np))
hf_train = HFDataset.from_dict({"features": X_np[:split].tolist(),
                                  "labels":   y_np[:split].tolist()})
hf_eval  = HFDataset.from_dict({"features": X_np[split:].tolist(),
                                  "labels":   y_np[split:].tolist()})
hf_train = hf_train.with_format("torch")
hf_eval  = hf_eval.with_format("torch")

print(f"Train samples: {len(hf_train)}  Eval samples: {len(hf_eval)}")
print(f"Train columns: {hf_train.column_names}")

# 3. compute_metrics: inline numpy - NO evaluate library (incompatible with datasets 4.x)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()
    return {"accuracy": accuracy}

# 4. Subclass Trainer to override compute_loss
#    Needed because our model expects "features" not "input_ids"
class ComplaintTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels   = inputs.pop("labels")
        features = inputs["features"].to(pt.float32)
        logits   = model(features)
        loss     = nn.CrossEntropyLoss()(logits, labels)
        return (loss, logits) if return_outputs else loss

# 5. TrainingArguments - the production configuration
training_args = TrainingArguments(
    output_dir="/tmp/complaint_classifier",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",          # NOT evaluation_strategy (removed in transformers 4.41+)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=10,
    no_cuda=(device.type == "cpu"),
)

# 6. Trainer: wraps training loop, checkpointing, logging
trainer = ComplaintTrainer(
    model=hf_model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_eval,
    compute_metrics=compute_metrics,
)

print("\n--- Training with HuggingFace Trainer (5 epochs) ---")
train_result = trainer.train()
print(f"\nTraining complete.")
print(f"  Runtime: {train_result.metrics.get('train_runtime', 0):.1f}s")
print(f"  Loss:    {train_result.metrics.get('train_loss', 0):.4f}")

# 7. Evaluate
eval_result = trainer.evaluate()
print(f"\nEval accuracy: {eval_result.get('eval_accuracy', 'N/A'):.3f}")
```

## Cell 52 - markdown - Beat 4 Lab Instructions: HF Trainer Lab (Tier 2)
```markdown
### Lab 6 - HuggingFace Trainer with Custom Model (Tier 2, ~30 min)

**Situation**: The Barclays NLP team wants to experiment with model architecture
using the Trainer framework so they get checkpointing and evaluation for free.
They have asked you to adapt the Trainer setup to a NEW model architecture AND
a NEW dataset size (1000 complaints, 8 features, 4 classes).

**Task**: Build a `FourClassComplaintClassifier` (4 output classes: account, fraud,
payment, general), wrap it in a `ComplaintTrainer`, and run training for 8 epochs.
Report the best eval accuracy.

**Action**: This is a Tier 2 lab. You have the full pattern from Beat 3. Fewer hints
this time - steps are deliberately less prescriptive.

- Create `X_lab6` (1000 samples, 8 features) and `y_lab6` (1000 labels, 4 classes)
- Build `FourClassComplaintClassifier` as `nn.Sequential` with at least 3 layers
- Split data 80/20 into `hf_train6` and `hf_eval6` as HuggingFace datasets
- Define `compute_metrics_lab6` using inline numpy (no evaluate library)
- Define `TrainingArguments` with `num_train_epochs=8`, `eval_strategy="epoch"`
- Define `ComplaintTrainer6` (subclass of Trainer) with the correct `compute_loss`
- Train and report best eval accuracy from `trainer6.state.best_metric`

**Result**: The verification cell confirms training ran for 8 epochs and eval
accuracy is tracked.

**Stretch**: Add per-class accuracy to `compute_metrics_lab6`. For each of the
4 classes, compute what fraction of that class was correctly predicted (hint:
numpy masking on `predictions == labels`).
```

## Cell 53 - code - Lab 6 Starter Code (Tier 2 - Minimal scaffold)
```python
# Lab 6 - HuggingFace Trainer (Tier 2)
# You have the full pattern from Beat 3. Adapt it for:
#   - 1000 samples, 8 features, 4 classes
#   - Your own FourClassComplaintClassifier architecture
#   - 8 training epochs

pt.manual_seed(SEED)
np.random.seed(SEED)

# YOUR CODE: Create X_lab6 and y_lab6
X_lab6 = None
y_lab6 = None

# YOUR CODE: Build FourClassComplaintClassifier
class FourClassComplaintClassifier(nn.Module):
    pass  # YOUR CODE

# YOUR CODE: Create hf_train6 and hf_eval6 (80/20 split)
hf_train6 = None
hf_eval6  = None

# YOUR CODE: compute_metrics_lab6 (inline numpy, no evaluate library)
def compute_metrics_lab6(eval_pred):
    pass  # YOUR CODE

# YOUR CODE: ComplaintTrainer6 with correct compute_loss for 8-feature input

# YOUR CODE: TrainingArguments (8 epochs, eval_strategy="epoch")

# YOUR CODE: Create trainer6, call trainer6.train() and trainer6.evaluate()
trainer6 = None

if trainer6 is not None:
    print("Training complete.")
    print(f"Best eval accuracy: {trainer6.state.best_metric}")
```

## Cell 54 - code - Lab 6 Safety-Net
```python
# Lab 6 safety-net: run this if you did not finish Lab 6.
# SKIP this cell if you DID finish Lab 6.
pt.manual_seed(SEED)
np.random.seed(SEED)

if trainer6 is None:
    print("Using Lab 6 safety-net so the rest of the notebook can run.")

    X_lab6 = pt.randn(1000, 8).numpy().astype(np.float32)
    y_lab6 = np.random.randint(0, 4, size=1000).astype(np.int64)

    _four_model = nn.Sequential(
        nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 4),
    ).to(device)

    split    = 800
    _hf_train = HFDataset.from_dict({"features": X_lab6[:split].tolist(),
                                      "labels":   y_lab6[:split].tolist()})
    _hf_eval  = HFDataset.from_dict({"features": X_lab6[split:].tolist(),
                                      "labels":   y_lab6[split:].tolist()})
    _hf_train = _hf_train.with_format("torch")
    _hf_eval  = _hf_eval.with_format("torch")

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean().item()}

    class _FourTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            logits = model(inputs["features"].to(pt.float32))
            loss   = nn.CrossEntropyLoss()(logits, labels)
            return (loss, logits) if return_outputs else loss

    _args = TrainingArguments(
        output_dir="/tmp/lab6_safetynet",
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        no_cuda=(device.type == "cpu"),
    )
    trainer6 = _FourTrainer(
        model=_four_model,
        args=_args,
        train_dataset=_hf_train,
        eval_dataset=_hf_eval,
        compute_metrics=_compute_metrics,
    )
    trainer6.train()
    trainer6.evaluate()
```

## Cell 55 - code - Lab 6 Verification
```python
# Verification - Lab 6
assert trainer6 is not None, "trainer6 not created"
assert trainer6.state.epoch == 8.0, \
    f"Expected 8 epochs, got {trainer6.state.epoch}"
best = trainer6.state.best_metric
assert best is not None, "No best metric recorded - did you set compute_metrics?"
print("Lab 6 passed.")
print(f"  Epochs completed: {int(trainer6.state.epoch)}")
print(f"  Best eval accuracy: {best:.3f}")
print("  eval_strategy='epoch' used (NOT evaluation_strategy - removed in 4.41+)")
```

## Cell 56 - markdown - Homework Extension 6
```markdown
**Homework Extension 6 - HuggingFace Trainer**

1. Add a `transformers.EarlyStoppingCallback` to `trainer6`. Set `early_stopping_patience=3`.
   Rerun training with 20 epochs - does it stop before epoch 20?
2. Enable mixed precision by adding `fp16=True` to `TrainingArguments` (only if you
   have a GPU). Measure the speedup vs the fp32 run. Note: fp16 training is the
   primary reason Trainer is preferred over manual loops in production.
3. (Challenge) Replace the synthetic complaint features with TF-IDF encodings of real
   text. Write 20 synthetic complaint strings (5 per class), vectorize them with
   `sklearn.feature_extraction.text.TfidfVectorizer`, and use the TF-IDF matrix as
   `X_lab6`. Train with Trainer and compare accuracy to the random-feature baseline.
```

## Cell 57 - markdown - Wrap-Up and Key Takeaways
```markdown
---
## Wrap-Up - What You Built

In this notebook you built every layer of the PyTorch stack, from raw numbers to
a production training pipeline:

| Component | What it does | Barclays use |
|-----------|-------------|--------------|
| Tensor | Stores and operates on numbers | Complaint feature vectors |
| Autograd | Tracks gradients through computation | Model learns from prediction errors |
| Dataset + DataLoader | Batches data for efficient training | Feeds 10k complaints per hour |
| nn.Module | Defines model architecture | Custom complaint classifier |
| nn.Sequential | Clean alternative for feedforward nets | Rapid architecture experiments |
| HF Trainer | Production training loop with eval + ckpt | What the real models use |

### Key rules to remember

- Always `optimizer.zero_grad()` before `loss.backward()` (or use `set_to_none=True`)
- Always `model.train()` before the training loop, `model.eval()` before inference
- Always `pt.no_grad()` during inference (saves memory, speeds up inference 10-20%)
- `eval_strategy="epoch"` - NOT `evaluation_strategy` - in all TrainingArguments
- Never use the `evaluate` library - use inline numpy for metrics
- `numpy<2` is pinned everywhere - do not upgrade it

### What is next

The next notebook (Topic 3a - Seq2Seq and Bahdanau Attention) builds on everything
here. You will use DataLoader to feed sequence data, nn.Module to build an encoder-
decoder architecture, and the manual training loop (not Trainer) to understand
how sequence-to-sequence models learn.

The Barclays complaint classifier we built in 6 sections will become the decoder
that selects which customer support team should handle each complaint.
```

## Cell 58 - code - Final Sanity Check
```python
# Final sanity check - confirms all sections completed successfully
checks = {
    "X_complaints (Section 1)": lambda: X_complaints.shape == (200, 6),
    "w and b (Section 2)":      lambda: isinstance(w, pt.Tensor) and isinstance(b, pt.Tensor),
    "train_loader (Section 3)": lambda: train_loader is not None,
    "model (Section 4)":        lambda: isinstance(model, nn.Module),
    "seq_model (Section 5)":    lambda: isinstance(seq_model, nn.Sequential),
    "trainer6 (Section 6)":     lambda: trainer6 is not None,
}

all_pass = True
for name, check_fn in checks.items():
    try:
        result = check_fn()
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
    except Exception as e:
        status = f"ERROR: {e}"
        all_pass = False
    print(f"  {status}: {name}")

if all_pass:
    print("\nAll sections complete. You are ready for Topic 3a.")
else:
    print("\nSome sections incomplete. Re-run the safety-net cells for any FAIL/ERROR rows.")
```

---

# VERIFICATION CHECKLIST
- [x] numpy<2 in install cell (Cell 1)
- [x] sagemaker.Session() + get_execution_role() in setup (Cell 2)
- [x] Four-beat arc for every concept (Cells 5-7, 14-15, 25-26, 33-34, 42-43, 50-51)
- [x] 2 diagram placeholders with correct paths (Cells 6 and 16)
- [x] Safety-net after every lab whose output feeds downstream (Cells 10, 19, 29, 37, 46, 54)
- [x] No evaluate library - inline numpy for metrics (Cell 51 and Lab 6)
- [x] No em dashes, en dashes, unicode mult, emojis
- [x] # YOUR CODE does not hint at answer
- [x] No more than 3 consecutive markdown cells without a code cell
- [x] Tier 2 lab: HF Trainer section (Cell 52-55)
- [x] Every lab has stretch + homework extension
- [x] eval_strategy="epoch" in Beat 3 and Lab 6 safety-net
- [x] ComplaintTrainer subclass overrides compute_loss for custom feature key
- [x] matplotlib.use("Agg") for SageMaker JupyterLab (Cell 3)
- [x] transformers>=4.35.0,<4.40.0 and tokenizers>=0.15.0,<0.20.0 (Cell 1)
- [x] sagemaker>=2.200.0,<3.0.0 (Cell 1)
- [x] no_cuda=(device.type == "cpu") for transparent CPU/GPU handling (Cell 51, 54)

---

# RESEARCH VALIDATED
- Source 1: plans/CORE_TECHNOLOGIES_AND_DECISIONS.md - version matrix: numpy<2, transformers>=4.35.0,<4.40.0, tokenizers>=0.15.0,<0.20.0, sagemaker>=2.200.0,<3.0.0, eval_strategy="epoch"
- Source 2: plans/SAGEMAKER_LESSONS_LEARNED.md - L5: evaluation_strategy removed in 4.41+; L6: evaluate library incompatible with datasets 4.x - use inline numpy; L3: sagemaker v3 breaks get_execution_role
- Source 3: researches/F1_pytorch_refresher.md - full 58-cell plan with Barclays narrative, six-section structure, six labs, safety-nets, verification cells, homework extensions, discussion prompts
- Source 4: PytorchPrimer/ (5 exercise notebooks) - confirmed original content: Tensors (ex1), Autograd/GPU (ex2), Dataset/DataLoader (ex3), nn.Linear classifier (ex4), nn.Sequential (ex5) - all condensed and adapted for SageMaker Studio in this plan
