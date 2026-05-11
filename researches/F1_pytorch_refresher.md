# F1 - PyTorch Refresher: Cell-by-Cell Plan

## Narrative

Students are Barclays developers (2+ years Python, some ML exposure) building a
**Barclays Customer Support Intelligence System** that classifies customer complaint
texts. Each section of this notebook builds one component of that system:

- Tensors   = how complaint text becomes numbers the model can process
- Autograd  = how the model learns from its mistakes
- DataLoader = how we feed batches of complaint records efficiently
- nn.Module  = our complaint classifier
- HF Trainer = the production training loop used at real companies

The dataset throughout is **synthetic complaint data** (no external API calls), created
inline so the notebook is fully self-contained.

## Audience

Barclays developers. Comfortable with Python, NumPy basics, and ML concepts at a
high level. NOT beginners. Moving fast.

## Estimated Time

- Sections 1-4 (PyTorch fundamentals): 75 min
- Section 5 (HuggingFace Trainer): 30 min
- Labs: 5 x ~18 min = ~90 min
- Total in-class: ~3.5 hours (Day 1 morning + part of afternoon)

## Output path

`Frameworks/pytorch_refresher.ipynb` (exercise)
`Frameworks/pytorch_refresher_solution.ipynb` (solution)

---

## Diagram Index

| # | Slug | Path | Description |
|---|------|------|-------------|
| 1 | autograd-computation-graph | `plans/F1/diagrams/autograd-computation-graph.mmd` | Computation graph showing how PyTorch builds the graph of operations from input tensors through loss, and how .backward() walks it in reverse to compute gradients for each parameter |
| 2 | training-loop | `plans/F1/diagrams/training-loop.mmd` | The four-step training loop: forward pass (predictions) -> loss computation -> backward pass (gradients) -> optimizer step (weight update). Cycles repeated for each batch and epoch |

---

## Cell-by-Cell Plan

---

### Cell 1: [type: markdown] - Title and Learning Objectives

**Purpose**: Orient students, set the Barclays narrative, list what they will build.

**Content**:
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
4. Classifier with nn.Module - our complaint classifier and training loop
5. Training with HuggingFace Trainer - the production training loop

### Prerequisites
- Python 3.x, NumPy basics
- Basic understanding of what a neural network does (not required: math derivations)

### Environment
This notebook runs entirely inside SageMaker Studio (JupyterLab kernel).
No remote training jobs. No GPU required for sections 1-5.
```

**Notes**: No code cell here. Title cell only. Keep it short - developers want to
get to code fast.

---

### Cell 2: [type: code] - Environment Setup and Verification

**Purpose**: Verify SageMaker environment, install pinned dependencies. Students
see that F1 is Studio-kernel-only (no remote training).

**Content**:
```python
# Environment setup - runs in SageMaker Studio JupyterLab kernel
# No remote training in this notebook

# Pin numpy<2 to avoid compatibility issues with older torch ops
# transformers and datasets needed for Section 5 (HuggingFace Trainer)
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy<2",
    "transformers>=4.35.0,<4.40.0",
    "tokenizers>=0.15.0,<0.20.0",
    "datasets>=2.18.0,<3.0.0",
], check=True)

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
print("Environment OK - running in Studio kernel, no remote training needed for F1")
```

**Notes**: The `sagemaker.Session()` / `get_execution_role()` call is the canonical
pattern for every notebook in this course. Students see it here first. Install cell
uses `sys.executable` + `subprocess` to guarantee the correct kernel Python is used.
No `getpass` here - the execution role handles auth.

---

### Cell 3: [type: code] - Core Imports

**Purpose**: One cell with all imports for the whole notebook. Students can re-run
if kernel dies.

**Content**:
```python
import torch as pt
import torch.nn as nn
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")          # SageMaker Studio: use non-interactive backend
import matplotlib.pyplot as plt

# Reproducibility seed
SEED = 42
pt.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"PyTorch version: {pt.__version__}")
print(f"CUDA available:  {pt.cuda.is_available()}")
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Device:          {device}")
```

**Notes**: `matplotlib.use("Agg")` prevents display errors in SageMaker JupyterLab
when no GUI backend is available. All plots are saved with `plt.savefig` or rendered
inline with `%matplotlib inline` in subsequent cells.

---

### Cell 4: [type: markdown] - Section 1 Header: Tensors

**Purpose**: Introduce Section 1 with the Barclays narrative hook.

**Content**:
```markdown
## Section 1 - Tensors: How Complaint Text Becomes Numbers

Customer complaint text must become numbers before any model can process it.
Those numbers live in **tensors** -- PyTorch's core data structure, GPU-ready and differentiable.
```

**Notes**: One markdown cell is fine here (3-cell markdown chain limit not triggered yet).

---

### Cell 5: [type: code] - Beat 1 (Broken): Naive List Approach for Complaint Encoding

**Purpose**: Students feel the pain of using Python lists for ML data. Shows exactly
why tensors are needed.

**Content**:
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

labels_list = [0, 1, 0]   # 0=account_issue, 1=fraud, 0=account_issue

# Try to compute a dot product between complaint vectors and a weight vector
weights_list = [0.5, 2.0, 0.3, 1.0]

try:
    # This is what we WANT: matrix multiply complaints by weights
    result = complaints_list * weights_list   # <-- wrong: list * list = ERROR
    print(result)
except TypeError as e:
    print(f"ERROR: {e}")

# Even if we use a loop, it is SLOW and has no gradients
dot_products = []
for complaint in complaints_list:
    dot = sum(c * w for c, w in zip(complaint, weights_list))
    dot_products.append(dot)

print(f"\nDot products (manual loop): {dot_products}")
print("Problem: this is slow for 100,000 complaints and PyTorch cannot")
print("compute gradients through plain Python loops.")
```

**Notes**: The first `try/except` block actually crashes and shows the `TypeError`.
The manual loop succeeds but the instructor points out the performance and gradient
tracking problems. Students see the real output before the fix.

---

### Cell 6: [type: markdown] - Beat 2: Diagram Placeholder (Tensor ops)

**Purpose**: Visual anchor showing tensor shapes and operations used in this notebook.

**Content**:
```markdown
<!-- DIAGRAM: autograd-computation-graph -->
[View diagram](../../plans/F1/diagrams/autograd-computation-graph.mmd)

The diagram above shows how PyTorch builds a computation graph as you run tensor
operations. Every node records what operation created it. When you call `.backward()`,
PyTorch walks this graph in reverse to compute how much each parameter contributed
to the loss - this is what enables learning.
```

**Notes**: This is Diagram 1 of 2. Referenced in autograd section but placed here
so students have the mental model before they need it. Do not add another markdown
cell after this - go straight to code (Beat 3).

---

### Cell 7: [type: code] - Beat 3 (Working): Tensor Creation and Operations

**Purpose**: Full demo of tensor creation, shapes, operations, and device move.
Instructor live-codes this.

**Content**:
```python
# --- Beat 3: Tensors solve all three problems ---
# Fast, differentiable, GPU-ready

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
zeros = pt.zeros(3, 4)           # 3x4 block of zeros
ones  = pt.ones(3, 4)            # 3x4 block of ones
rand  = pt.randn(3, 4)           # 3x4 from standard normal
arange = pt.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = pt.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0]

print("\nzeros:\n", zeros)
print("ones:\n",  ones)
print("arange:",  arange)
print("linspace:", linspace)

# 4. Indexing and slicing (same as NumPy)
print("\nFirst complaint:", complaints[0])          # row 0
print("Feature 1 for all complaints:", complaints[:, 1])  # column 1
print("Top-left 2x2:", complaints[:2, :2])

# 5. Reductions
print("\nMax score:", scores.max().item())
print("Mean complaint vector:", complaints.mean(dim=0))  # mean per feature

# 6. Reshape
flat = complaints.view(-1)          # flatten to 1D: [12]
back = flat.view(3, 4)              # back to original shape
print("\nFlattened shape:", flat.shape)
print("Reshaped shape:", back.shape)

# 7. GPU move (works even if no GPU - device is "cpu" then)
complaints_on_device = complaints.to(device)
print(f"\ncomplaints device: {complaints_on_device.device}")
```

**Notes**: Instructor talks through each numbered block. Emphasize `dtype=pt.float32`
(not float64) and `dtype=pt.long` for class labels. The `.item()` call is introduced
early because students need it to extract scalars from tensors throughout the course.

---

### Cell 8: [type: markdown] - Beat 4 Lab Instructions: Tensor Lab

**Purpose**: STAR-method lab instructions for Tier 1 guided lab.

**Content**:
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
   (we use random numbers to simulate the encoded complaint data)
2. Create a long tensor named `y_complaints` with shape [200] containing random class
   labels 0, 1, or 2 (use `pt.randint`)
3. Move both tensors to `device`
4. Compute `feature_means`: the mean value of each of the 6 features across all 200
   complaints (shape should be [6])
5. Compute `class_counts`: how many complaints fall into each class (use `pt.bincount`)

**Stretch**: Normalize `X_complaints` so each feature has mean 0 and std 1 across
all 200 samples (hint: subtract mean, divide by std, using `dim=0`).
```

**Notes**: STAR method applied. The synthetic data uses `pt.randn` and `pt.randint`
so there are no external dependencies. Tier 1: all steps numbered, all variable names
given.

---

### Cell 9: [type: code] - Lab 1 Starter Code

**Purpose**: Student-facing scaffold. All variables set to None.

**Content**:
```python
# Lab 1 - Tensor Foundations
pt.manual_seed(SEED)   # keep results reproducible

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

**Notes**: The None-guards on the print statements ensure that incomplete labs do
not crash with `AttributeError`. Students see clear "not set" messages.

---

### Cell 10: [type: code] - Lab 1 Safety-Net

**Purpose**: Students who did not finish Lab 1 can run this and continue.

**Content**:
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

**Notes**: Checks only `X_complaints` (the primary output). If it is None the whole
block is set. Uses the same `SEED` so downstream cells that depend on these variables
produce consistent results.

---

### Cell 11: [type: code] - Lab 1 Verification

**Purpose**: Automated shape and type checks confirm student answers.

**Content**:
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

**Notes**: All asserts have informative messages. Runs without errors = lab is correct.

---

### Cell 12: [type: markdown] - Homework Extension 1

**Purpose**: Async deeper work for students who want more.

**Content**:
```markdown
**Homework Extension 1 - Tensor Operations**

1. Load the Yelp Polarity dataset from HuggingFace datasets (it is pre-encoded)
   and convert the first 1000 reviews into a bag-of-words tensor using a vocabulary
   of the 50 most common words. Hint: `datasets.load_dataset("yelp_polarity")`.
2. Compute the cosine similarity between every pair of complaint vectors in
   `X_complaints`. What is the maximum similarity? What does a high cosine similarity
   between two complaint vectors mean for the classification task?
3. (Challenge) Implement your own one-hot encoding function that takes a 1D integer
   tensor of class indices and returns a 2D float tensor of one-hot rows, without
   using any loop or list comprehension - pure tensor operations only.
```

**Notes**: These are for async/homework only. They require external dataset access
which students may or may not have. No verification code needed here.

---

### Cell 13: [type: markdown] - Section 2 Header: Autograd and the Training Loop

**Purpose**: Transition to autograd with Barclays narrative hook.

**Content**:
```markdown
## Section 2 - Autograd: How the Model Learns from Mistakes

Our model starts with random weights. **Autograd** computes, for every weight,
whether increasing it would reduce the loss -- and by exactly how much.
One call to `.backward()` gives us all gradients at once.
```

---

### Cell 14: [type: code] - Beat 1 (Broken): Manual Gradient Computation

**Purpose**: Show what gradient computation looks like without autograd - students
feel the combinatorial explosion.

**Content**:
```python
# --- Beat 1: What life looks like without autograd ---
#
# Suppose we have a single weight w and we want to minimize loss = (w*x - y)^2
# We need the gradient dL/dw = 2*(w*x - y)*x
# For one weight this is fine. For a 6-layer neural net with millions of weights,
# computing every partial derivative by hand is impossible.

import math

# A "model" with just one weight
w = 0.5      # our single parameter
x = 3.0      # one complaint feature value
y_true = 1.0 # ground truth label

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
# That is 6*64 + 64*64 + 64*64 + 64*3 = 8,576 parameters.
# Computing each partial derivative by hand would take days.
print("\nFor 8,576 parameters this approach is completely impractical.")
print("We need PyTorch autograd to compute ALL gradients in one backward() call.")
```

**Notes**: The code runs and produces numbers. The "pain" is not a crash but a
conceptual demonstration of intractability. Instructor pauses here to ask students
"what if there were 100 million parameters?" (GPT-4 has ~1.8 trillion).

---

### Cell 15: [type: code] - Beat 3 (Working): Autograd Demo

**Purpose**: Show `requires_grad`, computation graph, `.backward()`, `.grad`,
`no_grad()`, and the manual training loop - all with the complaint story.

**Content**:
```python
# --- Beat 3: PyTorch autograd computes all gradients for us ---
import torch as pt

# Step 1: Mark tensors we want to differentiate through
w = pt.tensor(0.5, requires_grad=True)   # our weight
x = pt.tensor(3.0)                       # input (no grad needed for data)
y_true = pt.tensor(1.0)

# Step 2: Forward pass - PyTorch records every operation
y_pred = w * x                           # records: mul(w, x)
loss = (y_pred - y_true) ** 2            # records: sub, pow

print(f"y_pred:   {y_pred.item():.4f}")
print(f"loss:     {loss.item():.4f}")
print(f"loss grad_fn: {loss.grad_fn}")   # shows PyTorch built the graph

# Step 3: Backward pass - PyTorch walks the graph and fills .grad
loss.backward()

print(f"\ndL/dw (autograd): {w.grad.item():.4f}")
# Should match our manual result: 2*(0.5*3 - 1)*3 = 2*(1.5-1)*3 = 3.0
print(f"dL/dw (manual):   {2 * (w.item() * x.item() - y_true.item()) * x.item():.4f}")

# Step 4: A minimal training loop
print("\n--- Manual training loop (5 steps) ---")
w = pt.tensor(0.5, requires_grad=True)
lr = 0.05    # learning rate

for step in range(5):
    # Forward
    y_pred = w * x
    loss = (y_pred - y_true) ** 2

    # Backward
    loss.backward()          # accumulates grad into w.grad

    # Weight update (inside no_grad so this op is NOT recorded)
    with pt.no_grad():
        w -= lr * w.grad     # gradient descent step
        w.grad.zero_()       # CRITICAL: zero out grad or it accumulates!

    print(f"  step {step}: loss={loss.item():.4f}  w={w.item():.4f}")

print(f"\nFinal w={w.item():.4f}  (true ratio y/x = {y_true.item()/x.item():.4f})")

# Step 5: no_grad for inference (fast, no memory overhead)
print("\n--- Inference (no gradient tracking) ---")
with pt.no_grad():
    inference_pred = w * x
    print(f"Prediction: {inference_pred.item():.4f}")
    print(f"inference_pred.grad_fn: {inference_pred.grad_fn}")  # None - tracking is off
```

**Notes**: Critical teaching moment: `w.grad.zero_()` must be called every step or
gradients accumulate across steps (a common bug). The `.item()` call is used throughout
to extract Python scalars from 0-dimensional tensors. Instructor should pause on
`grad_fn` printout and explain the computation graph.

---

### Cell 16: [type: markdown] - Beat 2: Training Loop Diagram

**Purpose**: Visual anchor for the four-step training loop students just saw.

**Content**:
```markdown
<!-- DIAGRAM: training-loop -->
[View diagram](../../plans/F1/diagrams/training-loop.mmd)

The diagram shows the four steps that repeat every batch:
forward (predictions) -> loss (how wrong we are) -> backward (how to fix it) ->
step (actually fix it). Understanding this loop is the foundation of everything
that follows in the course.
```

**Notes**: This is Diagram 2 of 2. Placed after Beat 3 (working demo) so students
have seen the code first and the diagram reinforces it. No more than 2 consecutive
markdown cells before we hit Beat 4 code.

---

### Cell 17: [type: markdown] - Beat 4 Lab Instructions: Autograd Lab

**Purpose**: STAR-method lab instructions for Tier 1 guided lab.

**Content**:
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
1. Create a synthetic dataset: `x_train` (100 sentence lengths, float32) and
   `y_train` (true severity scores = 0.7 * x_train + 0.3 + noise)
2. Initialize parameters `w` and `b` as float32 tensors with `requires_grad=True`
3. Implement the forward function: `y_pred = w * x_train + b`
4. Implement the MSE loss: `loss = ((y_pred - y_train)**2).mean()`
5. Call `loss.backward()`
6. Update `w` and `b` with learning rate 0.01 (inside `pt.no_grad()`)
7. Zero the gradients
8. Run 100 epochs and record `losses` (a Python list of float values)

**Stretch**: Plot the learned line on top of the scatter plot of (x_train, y_train).
Add a title and axis labels that reference the Barclays complaint context.
```

---

### Cell 18: [type: code] - Lab 2 Starter Code

**Purpose**: Student-facing scaffold.

**Content**:
```python
# Lab 2 - Manual Training Loop
pt.manual_seed(SEED)

# Step 1: Synthetic complaint dataset
# x_train: 100 complaint sentence lengths (normalized, roughly 0-1)
x_train = None  # YOUR CODE: pt.randn or pt.rand, shape [100], dtype float32
# y_train: severity score = 0.7 * x_train + 0.3 + small noise
y_train = None  # YOUR CODE

# Step 2: Initialize learnable parameters
w = None  # YOUR CODE: scalar tensor, requires_grad=True
b = None  # YOUR CODE: scalar tensor, requires_grad=True

lr = 0.01
losses = []   # record loss per epoch

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

    losses.append(loss.item())

print(f"Final w={w.item():.4f} (true: 0.7000)")
print(f"Final b={b.item():.4f} (true: 0.3000)")
print(f"Final loss: {losses[-1]:.6f}")
```

**Notes**: The `pass` placeholder in the `with no_grad()` block is intentional
so the loop runs without crashing (it just does nothing until students fill it in).

---

### Cell 19: [type: code] - Lab 2 Safety-Net

**Purpose**: Students who did not finish Lab 2 can continue.

**Content**:
```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2.
pt.manual_seed(SEED)
if x_train is None or not isinstance(w, pt.Tensor):
    print("Using Lab 2 safety-net so the rest of the notebook can run.")
    x_train = pt.rand(100, dtype=pt.float32)
    y_train = 0.7 * x_train + 0.3 + 0.05 * pt.randn(100)
    w = pt.tensor(0.0, requires_grad=True)
    b = pt.tensor(0.0, requires_grad=True)
    lr = 0.01
    losses = []
    for epoch in range(100):
        y_pred = w * x_train + b
        loss = ((y_pred - y_train)**2).mean()
        loss.backward()
        with pt.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        losses.append(loss.item())
```

---

### Cell 20: [type: code] - Lab 2 Verification

**Purpose**: Verify learned parameters and plot the loss curve.

**Content**:
```python
# Verification - Lab 2
assert losses is not None and len(losses) == 100, "losses must have 100 values"
assert losses[-1] < losses[0], "Loss should decrease over training"
assert abs(w.item() - 0.7) < 0.15, f"w={w.item():.4f} too far from 0.7"
assert abs(b.item() - 0.3) < 0.15, f"b={b.item():.4f} too far from 0.3"

# Plot training loss
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

**Notes**: Uses `/tmp/` for saving plots (always writable in SageMaker). `.show()`
also works with Agg backend in JupyterLab when `%matplotlib inline` is active.

---

### Cell 21: [type: markdown] - Homework Extension 2

**Purpose**: Async deeper work.

**Content**:
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

---

### Cell 22: [type: markdown] - Discussion Prompt 1

**Purpose**: Peer discussion between sections (3-5 min).

**Content**:
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

**Notes**: 3-5 min discussion. Instructor facilitates, not lectures. Do not add
another markdown cell after this - Cell 23 is a code cell.

---

### Cell 24: [type: markdown] - Section 3 Header: Dataset and DataLoader

**Purpose**: Introduce Section 3 with Barclays narrative.

**Content**:
```markdown
## Section 3 - Dataset and DataLoader: Feeding Complaint Batches Efficiently

Barclays complaint data has millions of records -- loading all of them into one tensor
exhausts memory. `Dataset` describes your data; `DataLoader` handles shuffling,
batching, and multi-worker loading automatically.
```

---

### Cell 25: [type: code] - Beat 1 (Broken): Manual Batching

**Purpose**: Show the pain of manually slicing tensors into batches.

**Content**:
```python
# --- Beat 1: Manual batching is error-prone ---
pt.manual_seed(SEED)

# 200 complaints, 6 features each
X_full = pt.randn(200, 6)
y_full = pt.randint(0, 3, (200,))

BATCH_SIZE = 32

# Attempt to iterate manually
print("Manual batch loop:")
for i in range(0, len(X_full), BATCH_SIZE):
    X_batch = X_full[i : i + BATCH_SIZE]
    y_batch = y_full[i : i + BATCH_SIZE]
    # Problem 1: last batch may be smaller - model might not handle it
    # Problem 2: data is NOT shuffled - model sees same order every epoch
    # Problem 3: no parallel loading - slow for data on disk
    if i == 0:
        print(f"  batch 0 shape: {X_batch.shape}")

# The last batch
last_start = (len(X_full) // BATCH_SIZE) * BATCH_SIZE
X_last = X_full[last_start:]
print(f"  last batch shape: {X_last.shape}")  # might be < BATCH_SIZE
print("\nProblems: no shuffle, variable last-batch size, no parallel loading.")
print("PyTorch DataLoader solves all three.")
```

**Notes**: The last batch has 200 % 32 = 8 samples. Students see the size mismatch.
This is a real problem when batch normalization layers expect a fixed batch size.

---

### Cell 26: [type: code] - Beat 3 (Working): Dataset and DataLoader

**Purpose**: Full working demo: custom Dataset class, DataLoader with shuffle,
iterating over batches, multi-epoch training.

**Content**:
```python
# --- Beat 3: Dataset and DataLoader ---
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. TensorDataset: the simplest Dataset - wraps existing tensors
pt.manual_seed(SEED)
X_full = pt.randn(200, 6, dtype=pt.float32).to(device)
y_full = pt.randint(0, 3, (200,), dtype=pt.long).to(device)

dataset = TensorDataset(X_full, y_full)
print(f"Dataset length: {len(dataset)}")
print(f"First sample X shape: {dataset[0][0].shape}")
print(f"First sample y:       {dataset[0][1]}")

# 2. DataLoader: handles batching, shuffling, drop_last
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,     # shuffle at the start of EACH epoch
    drop_last=False,  # keep the last partial batch
)
print(f"\nNumber of batches per epoch: {len(loader)}")   # ceil(200/32) = 7

# 3. Iterate over one epoch
print("\nBatch shapes in one epoch:")
for batch_idx, (X_batch, y_batch) in enumerate(loader):
    if batch_idx < 3 or batch_idx == len(loader) - 1:
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
        # Store data - no processing at init time
        self.X = features.to(pt.float32)
        self.y = labels.to(pt.long)

    def __len__(self):
        # DataLoader calls this to know how many samples exist
        return len(self.y)

    def __getitem__(self, idx):
        # DataLoader calls this for each index in a batch
        return self.X[idx], self.y[idx]

# Same data, but now in our custom Dataset
complaint_ds = ComplaintDataset(X_full, y_full)
complaint_loader = DataLoader(complaint_ds, batch_size=32, shuffle=True)
X_b, y_b = next(iter(complaint_loader))
print(f"\nCustom Dataset - first batch: X={X_b.shape} y={y_b.shape}")
```

**Notes**: Instructor explains `__len__` and `__getitem__` as the two required methods.
The `ComplaintDataset` docstring follows the Barclays story. `drop_last=False` is
the safe default for classification (never lose data). `drop_last=True` is used
when batch norm makes variable batch sizes dangerous - flag this for students.

---

### Cell 27: [type: markdown] - Beat 4 Lab Instructions: DataLoader Lab

**Purpose**: STAR-method lab instructions.

**Content**:
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
4. Iterate over `train_loader` for one epoch, accumulating `total_seen` (count of samples)

**Stretch**: Add a `transform` parameter to `ComplaintDataset.__init__` that accepts
a callable. When `transform` is not None, apply it to `self.X[idx]` before returning.
Test it by passing a lambda that normalizes each row to unit norm.
```

---

### Cell 28: [type: code] - Lab 3 Starter Code

**Purpose**: Student-facing scaffold.

**Content**:
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
        total_seen += len(X_batch)   # accumulate sample count

print(f"Total samples seen in one epoch: {total_seen}")
print(f"Expected: 500")
```

---

### Cell 29: [type: code] - Lab 3 Safety-Net

**Purpose**: Students who did not finish Lab 3 can continue.

**Content**:
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

---

### Cell 30: [type: code] - Lab 3 Verification

**Purpose**: Confirm total samples seen.

**Content**:
```python
# Verification - Lab 3
assert total_seen == 500, f"Expected 500 samples, got {total_seen}"
X_b, y_b = next(iter(train_loader))
assert X_b.shape[1] == 8, f"Expected 8 features, got {X_b.shape[1]}"
assert y_b.dtype == pt.long, "y_b must be long dtype"
print(f"Lab 3 passed. {total_seen} samples seen across {len(train_loader)} batches.")
print(f"Batch shape: X={X_b.shape}, y={y_b.shape}")
```

---

### Cell 31: [type: markdown] - Homework Extension 3

**Purpose**: Async deeper work.

**Content**:
```markdown
**Homework Extension 3 - DataLoader**

1. Implement a `WeightedComplaintSampler` using `torch.utils.data.WeightedRandomSampler`.
   Fraud complaints (class 1) are rare in real data - only 5% of the dataset.
   Make the sampler draw class 1 samples 10x more often to handle class imbalance.
2. Measure the throughput difference (samples/second) between `num_workers=0` and
   `num_workers=4` for a DataLoader with a dataset of 10,000 samples. Note: in
   SageMaker Studio this may not show a large difference since data is in-memory.
3. (Challenge) Build a `TextComplaintDataset` that reads raw complaint strings from
   a Python list, tokenizes them character-by-character (vocabulary = ASCII printable
   chars), pads all sequences to the same length, and returns `(char_ids_tensor, label)`.
```

---

### Cell 32: [type: markdown] - Section 4 Header: Classifier with nn.Module

**Purpose**: Introduce Section 4.

**Content**:
```markdown
## Section 4 - Complaint Classifier with nn.Module

Tensors, autograd, and data loading are in place. Now we wire them into a real
classifier. Subclass `nn.Module`, define layers in `__init__`, implement `forward()` --
PyTorch handles parameter registration, device placement, and gradient tracking.
```

---

### Cell 33: [type: code] - Beat 1 (Broken): Classifier Without nn.Module

**Purpose**: Show what happens when students try to build a classifier with raw tensors.
The broken code fails because parameter tracking does not work.

**Content**:
```python
# --- Beat 1: Building a classifier with raw tensors (do not do this) ---
pt.manual_seed(SEED)

# A two-layer "network" as plain tensors
W1 = pt.randn(6, 16, requires_grad=True)    # layer 1 weights
b1 = pt.zeros(16, requires_grad=True)        # layer 1 bias
W2 = pt.randn(16, 3, requires_grad=True)    # layer 2 weights
b2 = pt.zeros(3, requires_grad=True)         # layer 2 bias

# Forward pass
def bad_forward(x):
    h = pt.relu(x @ W1 + b1)
    return h @ W2 + b2

X_sample = pt.randn(10, 6)
out = bad_forward(X_sample)
print(f"Output shape: {out.shape}")   # works so far

# Problem 1: To save the model, we would need to manually list ALL tensors
# params_to_save = [W1, b1, W2, b2]   # what if we had 20 layers?

# Problem 2: Moving to GPU requires updating every tensor by hand
# W1 = W1.to("cuda"), b1 = b1.to("cuda"), W2 = W2.to("cuda"), ...

# Problem 3: No standardized interface - no model.eval(), model.train(), model.parameters()

try:
    # This is how you WANT to move a model to GPU - but it doesn't work here
    bad_forward.to(device)
except AttributeError as e:
    print(f"\nERROR: {e}")

print("\nWe need nn.Module so PyTorch can manage parameters, device placement,")
print("and train/eval mode automatically.")
```

**Notes**: The forward pass works. The failure is on `.to(device)`. Instructor explains
that `nn.Module` registers parameters via `self.layer = nn.Linear(...)` and then
`.to()`, `.parameters()`, `.state_dict()`, `model.eval()` all work for free.

---

### Cell 34: [type: code] - Beat 3 (Working): nn.Module Classifier

**Purpose**: Full working complaint classifier using nn.Module. Includes full training
loop with DataLoader.

**Content**:
```python
# --- Beat 3: Complaint classifier with nn.Module ---
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np   # numpy<2, already installed

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
        # No need to list them separately - model.parameters() finds all of them

    def forward(self, x):
        # Define the data flow: input -> linear -> relu -> linear -> logits
        h = self.relu(self.layer1(x))   # [batch, hidden_dim]
        return self.layer2(h)           # [batch, num_classes]

# Instantiate and move to device in ONE call
pt.manual_seed(SEED)
model = ComplaintClassifier().to(device)

# Inspect the model
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
# 6*16 + 16 + 16*3 + 3 = 96 + 16 + 48 + 3 = 163

# Make synthetic complaint data (reuse X_500, y_500 from Lab 3 but shrink to 3 classes)
pt.manual_seed(SEED)
X_data = pt.randn(300, 6, dtype=pt.float32).to(device)
y_data = pt.randint(0, 3, (300,), dtype=pt.long).to(device)
dataset = TensorDataset(X_data, y_data)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()    # combines LogSoftmax + NLL - standard for classification
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop: 5 epochs
print("\n--- Training ComplaintClassifier (5 epochs) ---")
for epoch in range(5):
    model.train()   # enables dropout, batch norm if present (good habit)
    total_loss = 0.0
    correct = 0

    for X_batch, y_batch in loader:
        # Forward
        logits = model(X_batch)         # [batch, 3]
        loss   = criterion(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
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

**Notes**: Critical details to emphasize: `super().__init__()` (often forgotten),
`model.train()` before training loop, `model.eval()` before inference, `optimizer.zero_grad()`
BEFORE `loss.backward()` (not after - AdamW needs clean grads). `CrossEntropyLoss` takes
raw logits (not softmax output) - common source of double-softmax bugs.

---

### Cell 35: [type: markdown] - Beat 4 Lab Instructions: nn.Module Lab

**Purpose**: STAR-method lab instructions.

**Content**:
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
2. In `forward`: chain them in order (layer1 -> relu -> drop1 -> layer2 -> relu ->
   drop2 -> layer3 -> relu -> output)
3. Instantiate the model, move to `device`
4. Create `optimizer` (AdamW, lr=1e-3) and `criterion` (CrossEntropyLoss)
5. Train for 10 epochs using the existing `loader` from Beat 3
6. Record training losses in a list `deep_losses`

**Stretch**: Add a `batch_norm` layer (`nn.BatchNorm1d(32)`) between `layer1` and
`relu`. Does it train faster (fewer epochs to reach the same loss)?
```

---

### Cell 36: [type: code] - Lab 4 Starter Code

**Purpose**: Student-facing scaffold.

**Content**:
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

# Step 5-6: Training loop (10 epochs)
deep_losses = []

if deep_model is not None:
    for epoch in range(10):
        deep_model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            logits = None  # YOUR CODE: forward pass
            loss   = None  # YOUR CODE: compute loss
            # YOUR CODE: backward and optimizer step
            epoch_loss += loss.item() * len(y_batch) if loss is not None else 0
        deep_losses.append(epoch_loss / len(dataset))
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}: loss={deep_losses[-1]:.4f}")
```

---

### Cell 37: [type: code] - Lab 4 Safety-Net

**Purpose**: Students who did not finish Lab 4 can continue.

**Content**:
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

    deep_model = _DeeperComplaintClassifier().to(device)
    deep_criterion = nn.CrossEntropyLoss()
    deep_optimizer = optim.AdamW(deep_model.parameters(), lr=1e-3)
    deep_losses = []
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

---

### Cell 38: [type: code] - Lab 4 Verification

**Purpose**: Confirm model depth and training progress.

**Content**:
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

---

### Cell 39: [type: markdown] - Homework Extension 4

**Purpose**: Async deeper work.

**Content**:
```markdown
**Homework Extension 4 - nn.Module**

1. Add a `save_checkpoint` method to `DeeperComplaintClassifier` that calls
   `torch.save(self.state_dict(), path)` and a `load_checkpoint` class method
   that loads it back. Verify that the loaded model produces identical predictions
   to the trained model.
2. Register a forward hook on `self.layer2` using `module.register_forward_hook(fn)`.
   Print the shape of the intermediate activation every time `forward` is called.
   Hooks are how PyTorch visualizations (like Captum) inspect model internals.
3. (Challenge) Implement `L2 regularization` manually: after `loss.backward()` but
   before `optimizer.step()`, add `lambda * param.grad.add_(param.data)` for each
   parameter. Compare this to `AdamW(weight_decay=lambda)`. Are the results identical?
```

---

### Cell 41: [type: markdown] - Section 5 Header: HuggingFace Trainer

**Purpose**: Transition to Section 5 with production motivation.

**Content**:
```markdown
## Section 5 - HuggingFace Trainer: The Production Training Loop

In production, teams use the **HuggingFace Trainer** instead of a manual loop.
It adds checkpointing, lr scheduling, mixed precision, and distributed training for free.
The one catch: Trainer expects `datasets.Dataset` objects (HuggingFace format),
not PyTorch `Dataset`. We convert using `HFDataset.from_dict()`.
```

---

### Cell 42: [type: code] - Beat 1 (Broken): Passing Wrong Dataset Type to Trainer

**Purpose**: Show the error students get when they pass a PyTorch Dataset to Trainer.

**Content**:
```python
# --- Beat 1: Passing a PyTorch Dataset to Trainer (wrong type) ---
from transformers import Trainer, TrainingArguments

# Trainer wraps any nn.Module - but it is strict about dataset type
# What happens when we pass a PyTorch TensorDataset?

try:
    bad_args = TrainingArguments(
        output_dir="/tmp/bad_trainer_test",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        no_cuda=True,   # force CPU for this demo
    )
    # dataset (PyTorch TensorDataset) - THIS IS THE WRONG TYPE for Trainer
    bad_trainer = Trainer(
        model=model,    # model from Section 4
        args=bad_args,
        train_dataset=dataset,   # <-- PyTorch TensorDataset, not HF Dataset
    )
    bad_trainer.train()
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTrainer expects a HuggingFace datasets.Dataset, not a PyTorch Dataset.")
print("We need to convert our data using datasets.Dataset.from_dict()")
```

**Notes**: The actual error message from Trainer when given a PyTorch TensorDataset
is a `TypeError` or `AttributeError` about missing `__getitem__` with string keys.
The error text varies by transformers version but students see that Trainer is strict
about its input types.

---

### Cell 43: [type: code] - Beat 3 (Working): HuggingFace Trainer Full Demo

**Purpose**: Full working Trainer setup. Includes HF Dataset creation, compute_metrics
(inline numpy, NO evaluate library), TrainingArguments, Trainer, and train + evaluate.

**Content**:
```python
# --- Beat 3: HuggingFace Trainer for Complaint Classification ---
import numpy as np
from datasets import Dataset as HFDataset
from transformers import Trainer, TrainingArguments

# 1. Rebuild our model fresh (Trainer will manage the state)
pt.manual_seed(SEED)
hf_model = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(device)

# 2. Create HuggingFace datasets from our numpy arrays
#    Trainer expects dicts with string keys; "labels" is the magic key for loss
pt.manual_seed(SEED)
X_np = pt.randn(300, 6).numpy().astype(np.float32)
y_np = np.random.randint(0, 3, size=300).astype(np.int64)

# 80/20 train/eval split
split = int(0.8 * len(X_np))
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
#    (needed because our model input is a tensor named "features", not "input_ids")
class ComplaintTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        features = inputs["features"].to(pt.float32)
        logits = model(features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, logits) if return_outputs else loss

# 5. TrainingArguments: the production configuration
training_args = TrainingArguments(
    output_dir="/tmp/complaint_classifier",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",        # NOT evaluation_strategy (removed in transformers 4.41+)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=10,
    no_cuda=(device.type == "cpu"),  # use GPU if available
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
print(f"  Runtime: {train_result.metrics.get('train_runtime', 'N/A'):.1f}s")
print(f"  Loss:    {train_result.metrics.get('train_loss', 'N/A'):.4f}")

# 7. Evaluate
eval_result = trainer.evaluate()
print(f"\nEval accuracy: {eval_result.get('eval_accuracy', 'N/A'):.3f}")
```

**Notes**: CRITICAL rules enforced here:
- `eval_strategy="epoch"` (NOT `evaluation_strategy`) - see L5 in SAGEMAKER_LESSONS_LEARNED
- NO `evaluate` library - inline numpy compute_metrics only - see L6
- `ComplaintTrainer` subclass is necessary because our model expects `features` not `input_ids`
- `no_cuda=(device.type == "cpu")` handles both GPU and CPU students transparently
Instructor should explicitly call out `eval_strategy` and explain why (transformers 4.41+).

---

### Cell 44: [type: markdown] - Beat 4 Lab Instructions: HF Trainer Lab

**Purpose**: STAR-method lab instructions (Tier 1).

**Content**:
```markdown
### Lab 5 - HuggingFace Trainer with Custom Model (Tier 1, ~20 min)

**Situation**: The Barclays NLP team wants checkpointing and automatic evaluation
on a 4-class complaint classifier (account, fraud, payment, general).

**Task**: Adapt the Trainer setup from Beat 3 to a new model (8 features, 4 classes)
and a larger dataset (1000 complaints). Train for 8 epochs. Report best eval accuracy.

**Action**: Complete the steps marked `# YOUR CODE`.

**Result**: The verification cell confirms training ran for 8 epochs and eval accuracy
is tracked in `trainer5.state.best_metric`.

Steps:
1. Create `X_lab5` (1000 samples, 8 features, float32 numpy array) and
   `y_lab5` (1000 labels, values 0-3, int64 numpy array)
2. Build `hf_train5` and `hf_eval5` using `HFDataset.from_dict()` with 80/20 split;
   call `.with_format("torch")` on both
3. Define `compute_metrics_lab5(eval_pred)` -- inline numpy, no evaluate library
4. Define `FourClassModel` as `nn.Sequential` (8->32->ReLU->16->ReLU->4) on `device`
5. Define `ComplaintTrainer5` subclassing `Trainer`, override `compute_loss` so it
   reads `inputs["features"]` and runs `CrossEntropyLoss`
6. Create `TrainingArguments` with `num_train_epochs=8`, `eval_strategy="epoch"`,
   `save_strategy="no"`, `no_cuda=(device.type=="cpu")`
7. Create `trainer5`, call `.train()` then `.evaluate()`

**Stretch**: Add per-class accuracy to `compute_metrics_lab5` using numpy masking.
```

**Notes**: Tier 1 guided lab. All variable names given, all steps numbered.

---

### Cell 45: [type: code] - Lab 5 Starter Code

**Purpose**: Student-facing scaffold.

**Content**:
```python
# Lab 5 - HuggingFace Trainer
pt.manual_seed(SEED)
np.random.seed(SEED)

# Step 1: Create numpy arrays (Trainer needs numpy, not tensors)
X_lab5 = None  # YOUR CODE: np.random.randn(1000, 8).astype(np.float32)
y_lab5 = None  # YOUR CODE: np.random.randint(0, 4, size=1000).astype(np.int64)

# Step 2: Build HuggingFace datasets (80/20 split)
split5 = 800
hf_train5 = None  # YOUR CODE: HFDataset.from_dict({...}).with_format("torch")
hf_eval5  = None  # YOUR CODE

# Step 3: compute_metrics (inline numpy - NO evaluate library)
def compute_metrics_lab5(eval_pred):
    pass  # YOUR CODE: return {"accuracy": ...}

# Step 4: Model
FourClassModel = None  # YOUR CODE: nn.Sequential 8->32->ReLU->16->ReLU->4, on device

# Step 5: Subclass Trainer
class ComplaintTrainer5(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pass  # YOUR CODE

# Step 6: TrainingArguments
training_args5 = None  # YOUR CODE: 8 epochs, eval_strategy="epoch", save_strategy="no"

# Step 7: Create trainer, train, evaluate
trainer5 = None  # YOUR CODE: ComplaintTrainer5(model=..., args=..., ...)

if trainer5 is not None:
    trainer5.train()
    trainer5.evaluate()
    print(f"Best eval accuracy: {trainer5.state.best_metric}")
```

**Notes**: All variable names given. Steps follow the Beat 3 pattern exactly.
The safety-net below covers students who do not finish.

---

### Cell 46: [type: code] - Lab 5 Safety-Net

**Purpose**: Students who did not finish Lab 5 can see a result.

**Content**:
```python
# Lab 5 safety-net: run this if you did not finish Lab 5.
# SKIP this cell if you DID finish Lab 5.
pt.manual_seed(SEED)
np.random.seed(SEED)

if trainer5 is None:
    print("Using Lab 5 safety-net so the rest of the notebook can run.")

    X_lab5 = np.random.randn(1000, 8).astype(np.float32)
    y_lab5 = np.random.randint(0, 4, size=1000).astype(np.int64)

    _four_model = nn.Sequential(
        nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 4),
    ).to(device)

    split5 = 800
    _hf_train = HFDataset.from_dict({"features": X_lab5[:split5].tolist(),
                                      "labels":   y_lab5[:split5].tolist()})
    _hf_eval  = HFDataset.from_dict({"features": X_lab5[split5:].tolist(),
                                      "labels":   y_lab5[split5:].tolist()})
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
            loss = nn.CrossEntropyLoss()(logits, labels)
            return (loss, logits) if return_outputs else loss

    _args = TrainingArguments(
        output_dir="/tmp/lab5_safetynet",
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        no_cuda=(device.type == "cpu"),
    )
    trainer5 = _FourTrainer(
        model=_four_model,
        args=_args,
        train_dataset=_hf_train,
        eval_dataset=_hf_eval,
        compute_metrics=_compute_metrics,
    )
    trainer5.train()
    trainer5.evaluate()
```

---

### Cell 47: [type: code] - Lab 5 Verification

**Purpose**: Confirm training ran.

**Content**:
```python
# Verification - Lab 5
assert trainer5 is not None, "trainer5 not created"
assert trainer5.state.epoch == 8.0, \
    f"Expected 8 epochs, got {trainer5.state.epoch}"
best = trainer5.state.best_metric
assert best is not None, "No best metric recorded - did you set compute_metrics?"
print(f"Lab 5 passed.")
print(f"  Epochs completed: {int(trainer5.state.epoch)}")
print(f"  Best eval accuracy: {best:.3f}")
print("  eval_strategy='epoch' used (NOT evaluation_strategy - that was removed in 4.41+)")
```

---

### Cell 48: [type: markdown] - Homework Extension 5

**Purpose**: Async deeper work.

**Content**:
```markdown
**Homework Extension 5 - HuggingFace Trainer**

1. Add a `transformers.EarlyStoppingCallback` to `trainer5`. Set `early_stopping_patience=3`.
   Rerun training with 20 epochs - does it stop before epoch 20?
2. Enable mixed precision by adding `fp16=True` to `TrainingArguments` (only if you
   have a GPU). Measure the speedup vs the fp32 run. Note: fp16 training is the
   primary reason Trainer is preferred over manual loops in production.
3. (Challenge) Replace the synthetic features with TF-IDF encodings of real text.
   Write 20 synthetic complaint strings, vectorize with `TfidfVectorizer`, and use
   the matrix as `X_lab5`. Compare Trainer accuracy to the random-feature baseline.
```

---

### Cell 49: [type: markdown] - Wrap-Up and Key Takeaways

**Purpose**: Synthesize the notebook, bridge to next topic.

**Content**:
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
```

**Notes**: The table reinforces the narrative that ran throughout. The "key rules"
section is the cheat sheet students take away. The bridge to Topic 3a is explicit.

---

### Cell 50: [type: code] - Final Sanity Check

**Purpose**: Run a top-to-bottom sanity check that all key variables are defined
and have expected types. Students can re-run this after any kernel restart.

**Content**:
```python
# Final sanity check - confirms all sections completed successfully
checks = {
    "X_complaints (Section 1)": lambda: X_complaints.shape == (200, 6),
    "w and b (Section 2)":      lambda: isinstance(w, pt.Tensor) and isinstance(b, pt.Tensor),
    "train_loader (Section 3)": lambda: train_loader is not None,
    "model (Section 4)":        lambda: isinstance(model, nn.Module),
    "trainer5 (Section 5)":     lambda: trainer5 is not None,
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

**Notes**: This is the last cell. Students run it at the end of class. The instructor
can see at a glance which students got through all 6 sections.

---

## Appendix: Variable Carryover Table

The following variables from this notebook are used in subsequent notebooks.
Verify these names match exactly when building Topic 3a and beyond.

| Variable | Type | Shape | Defined in |
|----------|------|-------|------------|
| `device` | `torch.device` | scalar | Cell 3 |
| `SEED` | `int` | scalar | Cell 3 |
| `X_complaints` | `Tensor float32` | [200, 6] | Lab 1 |
| `y_complaints` | `Tensor long` | [200] | Lab 1 |
| `model` | `ComplaintClassifier` | - | Section 4 |
| `loader` | `DataLoader` | - | Section 4 |
| `trainer5` | `ComplaintTrainer` subclass | - | Lab 5 |

---

## Appendix: Total Cell Count

| Section | Cells | Range |
|---------|-------|-------|
| Setup | 3 | 1-3 |
| Section 1 Tensors | 9 | 4-12 |
| Section 2 Autograd | 10 | 13-22 |
| Section 3 DataLoader | 8 | 23-30 |
| Section 4 nn.Module | 9 | 31-39 |
| Section 5 HF Trainer | 10 | 40-49 |
| Wrap-Up + Sanity | 2 | 49-50 |
| **Total** | **~50** | |

---

## Appendix: Diagram Mermaid Source (for /build-diagrams)

### autograd-computation-graph.mmd

```
graph LR
    x["x (input tensor)"] --> mul["mul (*)"]
    w["w (requires_grad=True)"] --> mul
    mul --> sub["sub (-)"]
    y_true["y_true (ground truth)"] --> sub
    sub --> pow2["pow2 (**)"]
    pow2 --> loss["loss (scalar)"]
    loss -->|backward| grad_loss["dloss/dloss = 1"]
    grad_loss -->|chain rule| grad_pow2["dloss/dsub"]
    grad_pow2 -->|chain rule| grad_sub["dloss/dmul"]
    grad_sub -->|chain rule| grad_w["dloss/dw stored in w.grad"]
    style w fill:#ff9,stroke:#333
    style loss fill:#f99,stroke:#333
    style grad_w fill:#9f9,stroke:#333
```

### training-loop.mmd

```
graph TD
    A["Batch from DataLoader\n(X_batch, y_batch)"] --> B["Forward Pass\nlogits = model(X_batch)"]
    B --> C["Loss Computation\nloss = criterion(logits, y_batch)"]
    C --> D["Backward Pass\nloss.backward()\ncomputes all gradients"]
    D --> E["Optimizer Step\noptimizer.step()\nupdates all weights"]
    E --> F["Zero Gradients\noptimizer.zero_grad()"]
    F -->|"next batch"| A
    style A fill:#dde,stroke:#333
    style C fill:#fdd,stroke:#333
    style D fill:#dfd,stroke:#333
    style E fill:#dff,stroke:#333
```
