# Topic 3b - Attention in PyTorch: Cell-by-Cell Plan

## Overview

**Audience**: Barclays developers, 2+ years Python, PyTorch fundamentals, deep learning basics.
**Estimated time**: 60-75 minutes in class.
**Environment**: AWS SageMaker Studio, JupyterLab kernel, ml.t3.medium (Studio default). All code runs in the notebook directly - no remote training jobs.
**Source notebook**: Adapts `Exercises/9_Attention_with_Torch.ipynb` - restructured for four-beat arc, Barclays narrative, SageMaker environment, and Tier 3 capstone.
**Output path**: `Exercises/topic_3b_attention_pytorch/topic_3b_attention_pytorch.ipynb`
**Solution path**: `Solutions/topic_3b_attention_pytorch/topic_3b_attention_pytorch.ipynb`

**Day 1 narrative slot**: This is the LAST topic of Day 1. Students have just implemented attention in NumPy (Topic 3a). Now they port the exact same math to PyTorch. This is intentional: they see that the framework does not add magic - it adds autograd, batching, and GPU support around the same equations. The capstone is Tier 3 (open-ended) because this is the last topic of Day 1.

**Why Tier 3 here**: Per the course rules, Tier 3 (function signature + docstring only, no YOUR CODE scaffold) is assigned to the last topic of the day (topics 3, 6, 9). Topic 3b is the last topic of Day 1. The capstone lab therefore gets only a function signature and docstring - no numbered steps, no placeholder assignments.

---

## Diagram Index

| Slug | Path | Description |
|------|------|-------------|
| `scaled-dot-product-formula` | `plans/topic_3b/diagrams/scaled-dot-product-formula.mmd` | Flowchart of Q, K, V -> MatMul(Q, K^T) -> divide by sqrt(d_k) -> Softmax -> MatMul with V -> Output context. All tensor shapes labelled at each step. Highlight the sqrt(d_k) scaling node in a different colour to emphasise the key difference from plain dot product. |
| `attention-heatmap-complaint` | `plans/topic_3b/diagrams/attention-heatmap-complaint.mmd` | Example 8x8 attention weight heatmap (Mermaid quadrant or matrix representation) showing a complaint-domain sequence where "unauthorised" and "charge" receive high attention weights when the decoder attends to the "fraud" concept. Annotate rows (query tokens) and columns (key tokens). Show that learned attention is NOT diagonal. |

---

## Key Changes from Source Notebook (`Exercises/9_Attention_with_Torch.ipynb`)

**Keeping**:
- The `DotProductAttention` nn.Module class structure (source cell-9) as the Beat 3 demo for dot product attention in PyTorch.
- The `MultiHeadAttentionModel` nn.Module structure (source cell-22) as a stretch/reference.
- The Q, K, V tensor construction pattern (source cell-13).
- The `torch.nn.MultiheadAttention` layer reference.

**Restructuring**:
- Source notebook goes straight to `DotProductAttention` without Beat 1 (no broken code). We add a proper Beat 1 showing gradient saturation in PyTorch when attention is not scaled.
- Source notebook has no Barclays context. All demos and labs use complaint-domain examples.
- Source notebook skips directly from dot product attention to MultiheadAttention without teaching scaled dot product attention from scratch. We add a full Beat 3 demo of scaled dot product, then make it the Tier 3 capstone lab.
- Source notebook has no safety-net cells, no discussion prompts, no homework extensions, no STAR method.
- Source `DotProductAttention` has stubs (`scores = None`, `attention_weights = None`, `context = None`). We promote the COMPLETED version to the Beat 3 demo and keep stubs for Lab 1 (Tier 1).
- Source notebook imports textblob, swifter, torchnlp - NOT available by default in SageMaker Studio and not needed for this notebook. Replaced with clean torch + numpy only.

**Replacing**:
- Remove textblob, swifter, torchnlp imports entirely.
- Replace `pytorch-nlp` with complaint-domain synthetic data generated inline.
- Replace source Beat 1 (there is none) with a concrete gradient saturation demo.
- MultiheadAttention is demoted to a stretch/reference section (still shown but not the main lab).
- The capstone `scaled_dot_product_attention` function is Tier 3: signature + docstring only.

---

## Bridging from Topic 3a (Critical)

The notebook MUST explicitly connect to Topic 3a vocabulary:
- "In Topic 3a you implemented `scaled_dot_product_attention(Q, K, V)` in NumPy."
- "Today we port the SAME function to PyTorch as an `nn.Module`."
- "Then we verify it matches `torch.nn.functional.scaled_dot_product_attention`."

Variable name continuity: Topic 3a used `alpha`, `energy`, `context_vector`, `encoder_states`, `decoder_state`. Topic 3b uses `attention_weights`, `scores`, `context`, `Q`, `K`, `V` (PyTorch conventions). The mapping should be made explicit.

---

## Cell-by-Cell Plan

### Cell 1: [markdown] - Title and Learning Objectives

**Purpose**: Set the scene. Explicitly bridge from Topic 3a. Establish the Day 1 capstone framing.

**Content**:
```
# Topic 3b - Attention in PyTorch

Barclays Customer Support Intelligence System | Day 1, Topic 3b (Capstone)

## What you will build
In Topic 3a you implemented scaled dot product attention from scratch in NumPy.
In this notebook you will:
1. Port dot product attention to a PyTorch nn.Module (with autograd)
2. Implement scaled dot product attention in PyTorch from scratch
3. Verify your implementation against torch.nn.functional.scaled_dot_product_attention
4. Apply your attention module to a complaint triage task and visualise attention weights

## Capstone lab (Tier 3 - open-ended)
Implement `ScaledDotProductAttention(nn.Module)` - signature and docstring only.
No step-by-step scaffold. You know the math from Topic 3a.

## Learning objectives
1. Translate the NumPy attention implementation to an nn.Module with gradient support
2. Understand how PyTorch handles batch dimensions and broadcasting automatically
3. Implement scaled dot product attention without scaffolding (Tier 3)
4. Verify a custom PyTorch implementation against a reference library function
5. Interpret attention weight heatmaps over complaint tokens
```

---

### Cell 2: [code] - Environment Setup and Installs

**Purpose**: SageMaker Studio session setup + installs. Canonical pattern. Clean imports - no textblob, no swifter, no torchnlp.

**Content**:
```python
# Environment setup for SageMaker Studio
# All attention demos run in this kernel - no remote training jobs.

!pip install -q "sagemaker>=2.200.0,<3.0.0" \
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
```

**Notes**: numpy<2 pinned. No textblob, no swifter, no torchnlp - those caused install issues in the source notebook and are not needed here.

---

### Cell 3: [code] - PyTorch Imports and Configuration

**Purpose**: Import torch, set device, set seeds. Establish the complaint-domain data we will use throughout.

**Content**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import warnings

warnings.filterwarnings("ignore")

# Reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(42)

# Device: SageMaker Studio default is CPU. This notebook runs fine on CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
print()

# Complaint-domain vocabulary for all demos in this notebook.
# These are the tokens we will visualise attention over throughout.
COMPLAINT_TOKENS = [
    "unauthorised", "charge", "account", "fraud",
    "refund", "dispute", "urgent", "branch"
]
print(f"Complaint vocabulary for demos: {COMPLAINT_TOKENS}")
```

**Notes**: `device` check is good practice even though we are on CPU. The complaint token list is defined ONCE here and reused in all demos - this is intentional for narrative continuity.

---

### Cell 4: [markdown] - Bridge from Topic 3a

**Purpose**: Explicit conceptual bridge. One markdown, then code immediately.

**Content**:
```
## Bridging from Topic 3a

In Topic 3a you implemented this function in NumPy:

    def scaled_dot_product_attention(Q, K, V):
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

In this notebook we port the SAME logic to PyTorch as an nn.Module.
The equations do not change. What changes:
- np.matmul -> torch.matmul (or the @ operator)
- numpy softmax -> F.softmax(..., dim=-1)
- Autograd handles gradients automatically (no backprop code needed)
- GPU support is free (just move tensors to device)

Let us start with dot product attention (the unscaled version) as a warm-up.
```

---

### Cell 5: [code] - Beat 1: Naive PyTorch Attention Without Scaling - Gradient Saturation

**Purpose**: Beat 1 - code that RUNS and visibly shows a failure mode. Demonstrate gradient saturation in PyTorch with large d_k. This is the PyTorch version of the Beat 1 saturation demo from Topic 3a Section 4.

**Content**:
```python
# Beat 1: PyTorch attention without scaling saturates at large d_k.
# The softmax gets stuck near 0 or 1 -> gradients vanish -> model does not learn.
#
# We demonstrate this with a concrete forward + backward pass.

def unscaled_attention_forward(Q, K, V):
    """
    Dot product attention WITHOUT the 1/sqrt(d_k) scaling.
    This is the naive version that fails at large d_k.
    """
    # Raw dot product scores: (batch, T_q, d_k) x (batch, d_k, T_k) -> (batch, T_q, T_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # Softmax over keys
    attention_weights = F.softmax(scores, dim=-1)
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

print("Gradient magnitude vs key dimension (unscaled attention)")
print(f"{'d_k':>8}  {'max score':>12}  {'max attn':>12}  {'Q grad norm':>14}")
print("-" * 55)

batch_size = 2
T_q = 5
T_k = 8

for d_k in [4, 16, 64, 256, 512]:
    set_seeds(42)
    Q = torch.randn(batch_size, T_q, d_k, requires_grad=True)
    K = torch.randn(batch_size, T_k, d_k)
    V = torch.randn(batch_size, T_k, d_k)
    
    output, attn = unscaled_attention_forward(Q, K, V)
    
    # Simulate a loss and compute gradients
    loss = output.sum()
    loss.backward()
    
    max_score  = float(torch.max(torch.abs(torch.matmul(Q.detach(), K.transpose(-2, -1)))).item())
    max_attn   = float(torch.max(attn.detach()).item())
    q_grad_norm = float(Q.grad.norm().item())
    
    print(f"{d_k:>8}  {max_score:>12.2f}  {max_attn:>12.6f}  {q_grad_norm:>14.6f}")
    Q.grad = None  # reset for next iteration

print()
print("At d_k=512: max attention weight approaches 1.0 (one token dominates).")
print("Q gradient norm collapses -> the query receives almost no learning signal.")
print("FIX: divide scores by sqrt(d_k) before softmax.")
```

**Notes**: The gradient norm collapse at large d_k is the Beat 1 failure. Students see concrete numbers: at d_k=512 the gradient norm should be very small compared to d_k=4. Instructor: "The gradient norm is how much the model learns per step. A near-zero gradient means the attention is frozen - it cannot learn which tokens to attend to." This motivates scaling.

---

### Cell 6: [markdown] - Section Header: Dot Product Attention as an nn.Module

**Purpose**: Short intro before Beat 3. One markdown, then code.

**Content**:
```
## Section 1 - Dot Product Attention in PyTorch

The warm-up: implement dot product attention as an `nn.Module`.
This is the unscaled version. We will add scaling in Section 2.

Notice how PyTorch's autograd handles the backward pass automatically -
no need to write gradient code by hand.
```

---

### Cell 7: [code] - Beat 3: DotProductAttention nn.Module - Full Working Demo

**Purpose**: Beat 3 for dot product attention in PyTorch. Complete working `nn.Module`. Instructor live-codes from this.

**Content**:
```python
# Beat 3: Complete DotProductAttention nn.Module.
# This is the WORKING version. The lab (Cell 9) will ask students to implement it themselves.
# Instructor: walk through each line. Point out the shapes at every step.

class DotProductAttention(nn.Module):
    """
    Dot product attention as an nn.Module.
    
    Formula: Attention(Q, K, V) = softmax(Q K^T) V
    
    Note: this is NOT scaled. For production use ScaledDotProductAttention instead.
    """
    
    def __init__(self):
        super().__init__()
        # No learnable parameters in basic dot product attention.
        # The Q, K, V projections (if needed) live outside this module.
    
    def forward(self, query, key, value):
        """
        Forward pass.
        
        Args:
            query: (batch, T_q, d_k)
            key:   (batch, T_k, d_k)  - key and query share d_k
            value: (batch, T_k, d_v)  - value can have different d_v
        
        Returns:
            context:          (batch, T_q, d_v)
            attention_weights: (batch, T_q, T_k)
        """
        # Step 1: Compute dot product scores.
        # query @ key.T: (batch, T_q, d_k) x (batch, d_k, T_k) -> (batch, T_q, T_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Softmax over key positions (dim=-1).
        # Each query attends over all key positions.
        attention_weights = F.softmax(scores, dim=-1)   # (batch, T_q, T_k)
        
        # Step 3: Weighted sum of values.
        # (batch, T_q, T_k) x (batch, T_k, d_v) -> (batch, T_q, d_v)
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights

# --- Demo with complaint-domain data ---
set_seeds(42)

batch_size = 2     # 2 complaint messages in the batch
T_q = 8            # query sequence length (complaint tokens)
T_k = 8            # key sequence length (same as query for self-attention)
d_k = 32           # key/query dimension
d_v = 32           # value dimension (same as d_k for simplicity)

query = torch.randn(batch_size, T_q, d_k)
key   = torch.randn(batch_size, T_k, d_k)
value = torch.randn(batch_size, T_k, d_v)

attn_module = DotProductAttention()
context_out, attn_weights = attn_module(query, key, value)

print("DotProductAttention demo")
print("=" * 40)
print(f"Query shape:            {query.shape}")
print(f"Key shape:              {key.shape}")
print(f"Value shape:            {value.shape}")
print(f"Context shape:          {context_out.shape}   -> (batch=2, T_q=8, d_v=32)")
print(f"Attention weights shape:{attn_weights.shape} -> (batch=2, T_q=8, T_k=8)")
print()
print(f"Row sums of attn_weights[0] (must all be 1.0):")
print(f"  {attn_weights[0].sum(dim=-1).detach().numpy().round(4)}")
print()

# Parameter count: dot product attention adds ZERO parameters.
total_params = sum(p.numel() for p in attn_module.parameters())
print(f"Total trainable parameters in DotProductAttention: {total_params}")
print("Attention is parameter-free when Q, K, V are provided externally.")
```

**Notes**: The parameter count printout (0) is a key teaching moment. "Attention itself has no parameters. The projections that create Q, K, V have parameters. This is why attention is cheap to add to an existing architecture." The shape printout at each step is mandatory for classroom readability.

---

### Cell 8: [markdown] - Beat 2: Diagram Reference - Scaled Dot Product Formula

**Purpose**: Beat 2 for the scaled dot product concept. Placed before the lab, between Beat 3 and Beat 4.

**Content**:
```python
# Beat 2: Visual anchor for the attention flow.
print("See diagram: scaled dot product attention formula")
```

Immediately followed by the diagram markdown (Cell 9).

---

### Cell 9: [markdown] - Diagram: Scaled Dot Product Formula

**Purpose**: Diagram embed.

**Content**:
```
<!-- DIAGRAM: Flowchart of scaled dot product attention: Q and K^T feed into a MatMul node; output divided by sqrt(d_k); Softmax over key dimension; result multiplied with V via MatMul to produce context output. Tensor shapes (batch, T_q, d_k), (batch, d_k, T_k), (batch, T_q, T_k), (batch, T_k, d_v), (batch, T_q, d_v) labelled at each arrow. The sqrt(d_k) scaling node highlighted in orange. -->
[View diagram](../../plans/topic_3b/diagrams/scaled-dot-product-formula.mmd)

The diagram shows the same computation you implemented in Topic 3a, now with
PyTorch tensor shapes. The sqrt(d_k) scaling (highlighted) is the only difference
from plain dot product attention.
```

---

### Cell 10: [markdown] - Lab 1 Header (Tier 1: Guided)

**Purpose**: Beat 4 begins. STAR method. Tier 1 guided lab.

**Content**:
```
## Lab 1 - Implement DotProductAttention nn.Module (Tier 1 - Guided)

**Time**: 15-20 minutes

### Situation
The Barclays complaints ML team uses a PyTorch model to route incoming complaints.
They need a reusable attention module that can be dropped into any encoder-decoder model.

### Task
Implement `DotProductAttention` as an `nn.Module` with the correct forward pass.
You have seen the full working version in the demo (Cell 7). Now implement it yourself.

### Action
Complete the three stubs in the forward method below.

### Result
The verification cell will check:
1. Output shape is (batch, T_q, d_v)
2. Attention weights sum to 1 along the key dimension
3. Your output matches the reference implementation numerically
```

---

### Cell 11: [code] - Lab 1 Starter Code

**Purpose**: Tier 1 lab. Three stubs to fill.

**Content**:
```python
# Lab 1: Implement DotProductAttention from scratch.
# Complete the three stubs in the forward method.

class MyDotProductAttention(nn.Module):
    """
    Dot product attention module.
    
    Formula: Attention(Q, K, V) = softmax(Q K^T) V
    
    Args (forward):
        query: (batch, T_q, d_k)
        key:   (batch, T_k, d_k)
        value: (batch, T_k, d_v)
    
    Returns:
        context:          (batch, T_q, d_v)
        attention_weights: (batch, T_q, T_k)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value):
        # Step 1: Compute raw dot product scores.
        # Shape: (batch, T_q, d_k) x (batch, d_k, T_k) -> (batch, T_q, T_k)
        scores = None  # YOUR CODE
        
        # Step 2: Softmax over key positions.
        # Use F.softmax with the correct dim argument.
        attention_weights = None  # YOUR CODE
        
        # Step 3: Weighted sum of values.
        # Shape: (batch, T_q, T_k) x (batch, T_k, d_v) -> (batch, T_q, d_v)
        context = None  # YOUR CODE
        
        return context, attention_weights

# Quick sanity check
set_seeds(42)
my_attn = MyDotProductAttention()
test_q = torch.randn(2, 5, 16)
test_k = torch.randn(2, 5, 16)
test_v = torch.randn(2, 5, 16)
try:
    test_ctx, test_w = my_attn(test_q, test_k, test_v)
    if test_ctx is not None:
        print(f"Output shape: {test_ctx.shape}  (expected: torch.Size([2, 5, 16]))")
    else:
        print("Return values are None - complete all three steps.")
except Exception as e:
    print(f"Error: {e}")
```

---

### Cell 12: [code] - Lab 1 Verification Cell

**Purpose**: Numerical verification against reference implementation.

**Content**:
```python
# Lab 1 Verification

set_seeds(42)
ref_attn = DotProductAttention()
my_attn_verify = MyDotProductAttention()

q_v = torch.randn(2, 8, 32)
k_v = torch.randn(2, 8, 32)
v_v = torch.randn(2, 8, 32)

ref_ctx, ref_w = ref_attn(q_v, k_v, v_v)

all_pass = True

try:
    my_ctx, my_w = my_attn_verify(q_v, k_v, v_v)
    
    if my_ctx is None or my_w is None:
        print("FAIL: Return values are None. Complete all three steps.")
        all_pass = False
    else:
        # Shape check
        if my_ctx.shape == ref_ctx.shape:
            print(f"PASS: Output shape {my_ctx.shape}")
        else:
            print(f"FAIL: Expected shape {ref_ctx.shape}, got {my_ctx.shape}")
            all_pass = False
        
        # Attention weights sum to 1
        row_sums = my_w.sum(dim=-1)
        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            print("PASS: Attention weights sum to 1 along key dimension")
        else:
            print(f"FAIL: Row sums: {row_sums[0].detach()}")
            all_pass = False
        
        # Numerical match
        if torch.allclose(my_ctx, ref_ctx, atol=1e-5):
            print("PASS: Output matches reference implementation")
        else:
            max_diff = float((my_ctx - ref_ctx).abs().max().item())
            print(f"FAIL: Max difference from reference: {max_diff:.6f}")
            all_pass = False
        
        # Gradient flow check
        q_grad_test = q_v.clone().requires_grad_(True)
        my_ctx2, _ = my_attn_verify(q_grad_test, k_v, v_v)
        my_ctx2.sum().backward()
        if q_grad_test.grad is not None:
            print("PASS: Gradients flow through the attention module")
        else:
            print("FAIL: No gradient on query - check your implementation")
            all_pass = False

except Exception as e:
    print(f"FAIL: Exception during forward pass: {e}")
    all_pass = False

if all_pass:
    print()
    print("All checks passed. DotProductAttention is correctly implemented.")
```

**Notes**: The gradient flow check is unique to this notebook (not in 3a). It verifies that autograd works through the implementation, which is the whole point of using PyTorch.

---

### Cell 13: [code] - Lab 1 Safety-Net

**Purpose**: Mandatory safety-net.

**Content**:
```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if 'MyDotProductAttention' not in dir():
    MyDotProductAttention = DotProductAttention
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
else:
    try:
        _test_q = torch.randn(1, 4, 8)
        _test_ctx, _ = MyDotProductAttention()(_test_q, _test_q, _test_q)
        if _test_ctx is None:
            MyDotProductAttention = DotProductAttention
            print("Using Lab 1 safety-net (implementation incomplete).")
    except Exception:
        MyDotProductAttention = DotProductAttention
        print("Using Lab 1 safety-net (implementation raised an error).")
```

**Notes**: Remove from solution notebook.

---

### Cell 14: [markdown] - Lab 1 Stretch and Homework

**Purpose**: Fast finisher and async extension.

**Content**:
```
### Stretch (fast finishers)

Add an optional `mask` parameter to your `MyDotProductAttention.forward` method.
A mask is a boolean tensor of shape `(batch, T_q, T_k)` where `True` means "ignore
this position". Apply the mask by setting masked positions to `-inf` before softmax.

This is used in decoders (causal masking) to prevent attending to future tokens.

```python
# Stretch: masked attention
# def forward(self, query, key, value, mask=None):
#     scores = ...
#     if mask is not None:
#         scores = scores.masked_fill(mask, float("-inf"))
#     attention_weights = F.softmax(scores, dim=-1)
#     context = ...
#     return context, attention_weights
```

### Homework Extension

Compare your `MyDotProductAttention` output to `F.scaled_dot_product_attention`
(PyTorch 2.0+ built-in). Note: you will need to adjust for the scaling factor.

```python
# Homework: Compare with PyTorch built-in
# output_builtin = F.scaled_dot_product_attention(q, k, v, scale=1.0)  # scale=1.0 disables scaling
# output_mine, _ = MyDotProductAttention()(q, k, v)
# print(torch.allclose(output_builtin, output_mine, atol=1e-4))
```
```

---

### Cell 15: [markdown] - Section Header: Scaled Dot Product Attention in PyTorch

**Purpose**: Transition to the main event. One markdown, then code immediately.

**Content**:
```
## Section 2 - Scaled Dot Product Attention in PyTorch

You implemented the unscaled version. Now we add the sqrt(d_k) scaling.

This is the EXACT operation at the core of every Transformer model.
After this section you will implement it yourself as the Tier 3 capstone.
```

---

### Cell 16: [code] - Beat 1: Gradient Saturation Fixed by Scaling (PyTorch Side-by-Side)

**Purpose**: Beat 1 for scaled dot product section - now show that scaling fixes the gradient problem from Cell 5. This is more of a "fix revealed" than a pure failure - students see the before and after.

**Content**:
```python
# Beat 1 resolved: scaling fixes the gradient saturation from Section 0.
# Side-by-side comparison of gradient norms: unscaled vs scaled at d_k=512.

def scaled_attention_forward(Q, K, V):
    """Scaled dot product attention with 1/sqrt(d_k) normalisation."""
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, V)
    return context, attention_weights

print("Unscaled vs Scaled attention - gradient norms at d_k=512")
print(f"{'version':>15}  {'max score':>12}  {'max attn':>12}  {'Q grad norm':>14}")
print("-" * 60)

d_k = 512
batch_size = 2
T_q = 5
T_k = 8

for label, forward_fn in [("Unscaled", unscaled_attention_forward),
                           ("Scaled", scaled_attention_forward)]:
    set_seeds(42)
    Q = torch.randn(batch_size, T_q, d_k, requires_grad=True)
    K = torch.randn(batch_size, T_k, d_k)
    V = torch.randn(batch_size, T_k, d_k)
    
    output, attn = forward_fn(Q, K, V)
    output.sum().backward()
    
    max_score   = float(torch.max(torch.abs(torch.matmul(Q.detach(), K.transpose(-2, -1)))).item())
    max_attn    = float(torch.max(attn.detach()).item())
    q_grad_norm = float(Q.grad.norm().item())
    
    print(f"{label:>15}  {max_score:>12.2f}  {max_attn:>12.6f}  {q_grad_norm:>14.6f}")

print()
print("Scaled version: smaller max attention weight, much larger gradient norm.")
print("The model can learn from both versions now - scaling prevents one token from dominating.")
```

---

### Cell 17: [code] - Beat 3: Scaled Dot Product Attention - Full Working Demo (Reference)

**Purpose**: Beat 3 for scaled dot product attention. This is the REFERENCE implementation that students will verify their capstone against. Heavily commented. Instructor live-codes from this.

**Content**:
```python
# Beat 3: Reference implementation of scaled dot product attention.
# This is what you will implement yourself in the Tier 3 capstone lab.
# Study this carefully before starting the capstone.
#
# Instructor: walk through EVERY line. Shapes at each step.

class ScaledDotProductAttentionReference(nn.Module):
    """
    Reference implementation of scaled dot product attention.
    
    Formula: Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
    
    This is the fundamental operation of the Transformer (Vaswani et al., 2017).
    """
    
    def __init__(self, dropout_p=0.0):
        """
        Args:
            dropout_p: dropout probability on attention weights (0 = no dropout).
                       Dropout on attention weights regularises the model during training
                       by randomly zeroing out some attention connections.
        """
        super().__init__()
        # Optional dropout on attention weights
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, query, key, value):
        """
        Args:
            query: (batch, T_q, d_k)
            key:   (batch, T_k, d_k)
            value: (batch, T_k, d_v)
        
        Returns:
            output:           (batch, T_q, d_v)
            attention_weights: (batch, T_q, T_k)
        """
        # d_k is the key/query dimension - we scale by its square root
        d_k = query.shape[-1]                                        # scalar
        
        # Step 1: Scaled dot product scores
        # query @ key^T: (batch, T_q, d_k) x (batch, d_k, T_k) -> (batch, T_q, T_k)
        scores = torch.matmul(query, key.transpose(-2, -1))          # (batch, T_q, T_k)
        scores = scores / (d_k ** 0.5)                               # divide by sqrt(d_k)
        
        # Step 2: Softmax -> attention weights
        # dim=-1 means we softmax over the key dimension (T_k)
        # Each query position gets weights over ALL key positions that sum to 1
        attention_weights = F.softmax(scores, dim=-1)                # (batch, T_q, T_k)
        attention_weights = self.dropout(attention_weights)          # optional regularisation
        
        # Step 3: Weighted sum of values
        # (batch, T_q, T_k) x (batch, T_k, d_v) -> (batch, T_q, d_v)
        output = torch.matmul(attention_weights, value)              # (batch, T_q, d_v)
        
        return output, attention_weights

# --- Demo: complaint triage self-attention ---
set_seeds(42)

batch_size = 2
T_seq = len(COMPLAINT_TOKENS)  # 8 complaint tokens
d_k = 64
d_v = 64

# Simulate complaint embeddings (Q = K = V for self-attention)
Q_complaint = torch.randn(batch_size, T_seq, d_k)
K_complaint = Q_complaint   # self-attention: query = key
V_complaint = Q_complaint   # self-attention: value = key

ref_module = ScaledDotProductAttentionReference(dropout_p=0.0)
ref_output, ref_attn_weights = ref_module(Q_complaint, K_complaint, V_complaint)

print("ScaledDotProductAttention (Reference) - Complaint Self-Attention Demo")
print("=" * 60)
print(f"Q shape: {Q_complaint.shape}  -> (batch=2, tokens=8, d_k=64)")
print(f"K shape: {K_complaint.shape}")
print(f"V shape: {V_complaint.shape}")
print(f"Output shape:          {ref_output.shape}          -> (batch=2, tokens=8, d_v=64)")
print(f"Attention weights shape: {ref_attn_weights.shape}  -> (batch=2, 8 queries, 8 keys)")
print()
row_sums = ref_attn_weights[0].sum(dim=-1).detach()
print(f"Row sums of attn_weights[0]: {row_sums.numpy().round(4)}")
print("All must be 1.0 - confirmed." if torch.allclose(row_sums, torch.ones(T_seq), atol=1e-5) else "WARNING: rows do not sum to 1!")
print()

# Compare with PyTorch built-in F.scaled_dot_product_attention (PyTorch 2.0+)
try:
    builtin_output = F.scaled_dot_product_attention(Q_complaint, K_complaint, V_complaint)
    match = torch.allclose(ref_output, builtin_output, atol=1e-4)
    print(f"Matches F.scaled_dot_product_attention: {match}")
    print("Our reference implementation agrees with PyTorch's built-in function.")
except AttributeError:
    print("F.scaled_dot_product_attention requires PyTorch 2.0+. Skipping comparison.")

# Visualise attention weights (batch item 0)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    ref_attn_weights[0].detach().numpy(),
    xticklabels=COMPLAINT_TOKENS,
    yticklabels=COMPLAINT_TOKENS,
    cmap="Blues",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Self-Attention Weights - Complaint Tokens (random, untrained)")
plt.xlabel("Key tokens (what is being attended to)")
plt.ylabel("Query tokens (what is asking the question)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
print("With random weights, attention is approximately uniform.")
print("After training on complaint-label pairs, high-severity tokens would")
print("attend strongly to each other: 'fraud' <-> 'unauthorised', 'charge' <-> 'refund'.")
```

**Notes**: The F.scaled_dot_product_attention comparison is a key DoD element - students must verify their capstone against this. The try/except handles SageMaker Studio kernels with PyTorch < 2.0. Instructor: "This is the function you are going to implement in the capstone. Before you start, study this reference implementation. The capstone gives you only the signature and docstring."

---

### Cell 18: [code] - Beat 2: Diagram Reference - Attention Heatmap

**Purpose**: Beat 2 for the attention heatmap concept.

**Content**:
```python
# Beat 2: Visual anchor for trained attention heatmap patterns.
print("See diagram: attention weight heatmap example with complaint tokens")
```

---

### Cell 19: [markdown] - Diagram: Attention Heatmap

**Purpose**: Diagram embed.

**Content**:
```
<!-- DIAGRAM: Example 8x8 attention weight heatmap over complaint tokens. Rows are query tokens (unauthorised, charge, account, fraud, refund, dispute, urgent, branch); columns are key tokens (same). Show a TRAINED attention pattern where fraud has high attention weight on unauthorised and charge; refund attends to dispute; the diagonal (self-attention) is moderate. Annotate the high-weight cells to show the semantic clustering. Contrast this with a random/diagonal-dominated heatmap to show what training achieves. -->
[View diagram](../../plans/topic_3b/diagrams/attention-heatmap-complaint.mmd)

A TRAINED attention heatmap is NOT diagonal. The model learns that "fraud" should
attend to "unauthorised" and "charge", and that "refund" is semantically related
to "dispute". This semantic clustering emerges from training on labelled complaints.
```

---

### Cell 20: [markdown] - Discussion Prompt

**Purpose**: Peer discussion before the capstone. 3-5 minutes.

**Content**:
```
### Discussion (3 minutes)

You have now seen DotProductAttention and ScaledDotProductAttention as nn.Modules.

1. The reference implementation has a `dropout_p` parameter on the attention weights.
   Why would you apply dropout specifically to attention weights rather than, say,
   to the output context vectors? What does "dropping out" an attention connection mean?

2. In Topic 3a we used word2vec embeddings as the Q, K, V inputs directly.
   In a real Transformer, Q, K, V are created by projecting the input embeddings
   through three separate learned weight matrices (W_Q, W_K, W_V).
   Why have separate projections? What would break if W_Q = W_K = W_V = Identity?

3. The capstone asks you to implement ScaledDotProductAttention from scratch.
   Without looking at Cell 17, how would you start? What is the first line of code
   you would write?
```

---

### Cell 21: [markdown] - Capstone Lab Header (Tier 3: Open-Ended)

**Purpose**: Tier 3 capstone. STAR framing. Explicit Tier 3 label. Signature + docstring only.

**Content**:
```
## Capstone Lab - Implement Scaled Dot Product Attention from Scratch (Tier 3 - Open-Ended)

**Time**: 25-35 minutes | **Tier**: 3 (open-ended - function signature + docstring only)

This is the Day 1 capstone. You know the math from Topic 3a. You have studied the
reference implementation in Cell 17. Now implement it yourself with NO scaffold.

### Situation
The Barclays ML platform team needs a production-quality attention module that:
- Works with batched complaint sequences
- Supports variable sequence lengths
- Returns both context vectors and attention weights (for interpretability)
- Passes a numerical verification against PyTorch's built-in function

### Task
Implement `ScaledDotProductAttention(nn.Module)`. The signature and docstring are
provided below. Do not look at Cell 17 while implementing.

### Action
Complete the implementation below. No numbered steps. No `# YOUR CODE` placeholders.
You decide the structure.

### Result
The verification cell will check:
1. Output shape matches expected
2. Attention weights sum to 1 along the key dimension
3. Numerical match with `ScaledDotProductAttentionReference` from Cell 17
4. Gradient flows through the module

**Stretch**: After passing verification, add a boolean `mask` parameter that allows
masking out padding tokens (set masked scores to -inf before softmax). Test it.

**Homework Extension**: Verify your implementation matches
`F.scaled_dot_product_attention(Q, K, V)` (PyTorch 2.0+). They should agree to
within floating-point tolerance. If they differ, find the bug.
```

---

### Cell 22: [code] - Capstone Lab Starter Code (Tier 3)

**Purpose**: Tier 3 lab - signature and docstring ONLY. No YOUR CODE stubs. No numbered steps. No hints about the implementation.

**Content**:
```python
# Capstone Lab: Implement scaled dot product attention.
# Signature and docstring provided. Implementation is yours.

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot product attention module.
    
    Implements: Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
    
    Parameters
    ----------
    dropout_p : float
        Dropout probability applied to attention weights during training.
        Default 0.0 (no dropout).
    
    Inputs (forward method)
    -----------------------
    query : torch.Tensor, shape (batch, T_q, d_k)
    key   : torch.Tensor, shape (batch, T_k, d_k)
    value : torch.Tensor, shape (batch, T_k, d_v)
    
    Returns
    -------
    output            : torch.Tensor, shape (batch, T_q, d_v)
    attention_weights : torch.Tensor, shape (batch, T_q, T_k)
                        Each row sums to 1.0 along the last dimension.
    """
    pass
```

**Notes**: This is intentionally terse. There are NO `= None  # YOUR CODE` lines, no step comments, no numbered hints. The student must produce the entire implementation. This is the Tier 3 contract. The solution notebook will have a complete implementation with explanation comments.

---

### Cell 23: [code] - Capstone Verification Cell

**Purpose**: Verify the capstone implementation. Four checks.

**Content**:
```python
# Capstone Verification - run after completing ScaledDotProductAttention above.

set_seeds(42)

# Test data
batch_size = 3
T_q = 6
T_k = 8
d_k = 64
d_v = 64

q_test = torch.randn(batch_size, T_q, d_k)
k_test = torch.randn(batch_size, T_k, d_k)
v_test = torch.randn(batch_size, T_k, d_v)

# Reference
ref = ScaledDotProductAttentionReference(dropout_p=0.0)
ref_out, ref_w = ref(q_test, k_test, v_test)

all_pass = True

try:
    student_module = ScaledDotProductAttention(dropout_p=0.0)
    student_out, student_w = student_module(q_test, k_test, v_test)
    
    if student_out is None or student_w is None:
        print("FAIL: Module returned None. Implement the forward method.")
        all_pass = False
    else:
        # Check 1: Output shape
        if student_out.shape == ref_out.shape:
            print(f"PASS: Output shape {student_out.shape}")
        else:
            print(f"FAIL: Expected output shape {ref_out.shape}, got {student_out.shape}")
            all_pass = False
        
        # Check 2: Attention weights shape
        if student_w.shape == ref_w.shape:
            print(f"PASS: Attention weights shape {student_w.shape}")
        else:
            print(f"FAIL: Expected weights shape {ref_w.shape}, got {student_w.shape}")
            all_pass = False
        
        # Check 3: Weights sum to 1
        row_sums = student_w.sum(dim=-1)
        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            print("PASS: Attention weights sum to 1 along key dimension")
        else:
            print(f"FAIL: Row sums min={row_sums.min():.4f} max={row_sums.max():.4f}")
            all_pass = False
        
        # Check 4: Numerical match with reference
        if torch.allclose(student_out, ref_out, atol=1e-5):
            print("PASS: Output matches ScaledDotProductAttentionReference")
        else:
            max_diff = float((student_out - ref_out).abs().max().item())
            print(f"FAIL: Max difference from reference: {max_diff:.6f}")
            print("Hint: check the scaling factor and the softmax dim.")
            all_pass = False
        
        # Check 5: Gradient flow
        q_grad = q_test.clone().requires_grad_(True)
        s_out, _ = student_module(q_grad, k_test, v_test)
        s_out.sum().backward()
        if q_grad.grad is not None:
            print("PASS: Gradients flow through ScaledDotProductAttention")
        else:
            print("FAIL: No gradient on query tensor")
            all_pass = False
        
        # Bonus: compare with PyTorch built-in
        try:
            builtin_out = F.scaled_dot_product_attention(q_test, k_test, v_test)
            builtin_match = torch.allclose(student_out, builtin_out, atol=1e-4)
            status = "PASS" if builtin_match else "FAIL"
            print(f"{status}: Matches F.scaled_dot_product_attention (PyTorch built-in)")
        except AttributeError:
            print("INFO: F.scaled_dot_product_attention not available (PyTorch < 2.0)")

except NotImplementedError:
    print("FAIL: Module raises NotImplementedError - remove the pass and implement forward()")
    all_pass = False
except Exception as e:
    print(f"FAIL: Exception during forward pass: {type(e).__name__}: {e}")
    all_pass = False

if all_pass:
    print()
    print("All capstone checks passed. Excellent work.")
    print("You have implemented the core operation of the Transformer from scratch in PyTorch.")
```

---

### Cell 24: [code] - Capstone Safety-Net

**Purpose**: Mandatory safety-net for the capstone. Students who did not finish can still continue.

**Content**:
```python
# Capstone safety-net: run this if you did not finish the capstone.
# SKIP this cell if you DID finish the capstone.
_need_safety_net = False
try:
    _m = ScaledDotProductAttention(dropout_p=0.0)
    _q = torch.randn(1, 4, 16)
    _out, _w = _m(_q, _q, _q)
    if _out is None:
        _need_safety_net = True
except Exception:
    _need_safety_net = True

if _need_safety_net:
    print("Using capstone safety-net so the rest of the notebook can run.")
    ScaledDotProductAttention = ScaledDotProductAttentionReference
```

**Notes**: Remove from solution notebook.

---

### Cell 25: [markdown] - Section Header: Applying Attention to Complaint Triage

**Purpose**: Bridge from isolated attention module to applied use case. One markdown, then code.

**Content**:
```
## Section 3 - Applying Your Attention Module to Complaint Triage

You have a working `ScaledDotProductAttention` module.
Let us use it in a minimal complaint triage model and visualise the learned attention pattern.

This is NOT a trained model - we use random weights. But the architecture shows
how attention would fit into a production complaints routing system.
```

---

### Cell 26: [code] - Applied Demo: Complaint Triage Attention Visualisation

**Purpose**: Apply the student's (or safety-net) ScaledDotProductAttention to complaint tokens and visualise. Narrative payoff for the whole day.

**Content**:
```python
# Applied demo: complaint triage self-attention with interpretability visualisation.
# We use the COMPLAINT_TOKENS from Cell 3 throughout.
#
# A real system would load actual complaint embeddings from the model's embedding layer.
# Here we simulate plausible embeddings where semantically related tokens are closer.

set_seeds(42)

n_tokens = len(COMPLAINT_TOKENS)  # 8
d_model = 64

# Simulate complaint embeddings with some structure:
# "unauthorised", "charge", "fraud" should be semantically close
# "refund", "dispute" should be close
# We add a small perturbation around two cluster centres.
cluster_fraud = torch.randn(d_model)
cluster_account = torch.randn(d_model)

def make_complaint_embeddings(cluster_fraud, cluster_account, d_model, seed=42):
    """Create plausible complaint token embeddings with semantic clustering."""
    torch.manual_seed(seed)
    embeddings = torch.zeros(n_tokens, d_model)
    # Fraud cluster: unauthorised, charge, fraud
    for i in [0, 1, 3]:   # indices in COMPLAINT_TOKENS
        embeddings[i] = cluster_fraud + 0.3 * torch.randn(d_model)
    # Account/resolution cluster: account, refund, dispute
    for i in [2, 4, 5]:
        embeddings[i] = cluster_account + 0.3 * torch.randn(d_model)
    # Standalone: urgent, branch
    embeddings[6] = torch.randn(d_model)
    embeddings[7] = torch.randn(d_model)
    return embeddings

complaint_emb = make_complaint_embeddings(cluster_fraud, cluster_account, d_model)

# Add batch dimension: (1, 8, 64) - one complaint message, 8 tokens
Q_appl = complaint_emb.unsqueeze(0)   # (1, 8, 64)
K_appl = Q_appl                        # self-attention
V_appl = Q_appl

# Use the student's attention module (or safety-net)
appl_module = ScaledDotProductAttention(dropout_p=0.0)
appl_output, appl_attn_weights = appl_module(Q_appl, K_appl, V_appl)

print(f"Input complaint embeddings: {Q_appl.shape}")
print(f"Output context vectors:     {appl_output.shape}")
print(f"Attention weights:          {appl_attn_weights.shape}")
print()

# Visualise
attn_np = appl_attn_weights[0].detach().numpy()

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    attn_np,
    xticklabels=COMPLAINT_TOKENS,
    yticklabels=COMPLAINT_TOKENS,
    cmap="Reds",
    annot=True,
    fmt=".3f",
    linewidths=0.5
)
plt.title("Self-Attention Weights - Complaint Triage\n(structured embeddings, no training)")
plt.xlabel("Key tokens (being attended to)")
plt.ylabel("Query tokens (asking the question)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

print("With structured embeddings (fraud cluster + account cluster):")
print("'unauthorised', 'charge', 'fraud' should attend to each other.")
print("'account', 'refund', 'dispute' should attend to each other.")
print()

# Verify clustering effect
fraud_tokens = [0, 1, 3]   # unauthorised, charge, fraud
acct_tokens  = [2, 4, 5]   # account, refund, dispute

avg_intra_fraud = float(attn_np[np.ix_(fraud_tokens, fraud_tokens)].mean())
avg_cross       = float(attn_np[np.ix_(fraud_tokens, acct_tokens)].mean())

print(f"Average attention within fraud cluster:   {avg_intra_fraud:.4f}")
print(f"Average attention fraud -> account cluster:{avg_cross:.4f}")
if avg_intra_fraud > avg_cross:
    print("Confirmed: fraud tokens attend more to each other than to account tokens.")
    print("This is what semantic clustering in embeddings produces.")
else:
    print("With random weights the clustering effect may not appear.")
    print("(Real trained attention would show clear intra-cluster patterns.)")
```

**Notes**: The `np.ix_` indexing may need numpy imported - it is already imported as np in Cell 3. The clustering verification (avg_intra_fraud vs avg_cross) gives a concrete numeric result. Instructor: "Notice the pattern emerges just from the embedding structure, before any training. Training would sharpen this."

---

### Cell 27: [markdown] - Section Header: Multi-Head Attention (Reference)

**Purpose**: Show the extension to multi-head attention. One markdown, then code.

**Content**:
```
## Section 4 - Multi-Head Attention: Reference Only

Multi-head attention runs several scaled dot product attentions in parallel,
each with its own Q, K, V projections, then concatenates the results.

You do not need to implement this today. The reference below shows how
`ScaledDotProductAttention` composes into `nn.MultiheadAttention`.
We will build the full multi-head attention module in Topic 4 (Transformers).
```

---

### Cell 28: [code] - Multi-Head Attention Reference Demo

**Purpose**: Show nn.MultiheadAttention using the same complaint tokens. Source cell-22 adapted. This is reference, not a lab.

**Content**:
```python
# Reference: nn.MultiheadAttention using PyTorch's built-in module.
# We will implement multi-head attention from scratch in Topic 4 (Transformers).
# For now, verify it works and understand the parameter count.

set_seeds(42)

embed_dim = 64    # must be divisible by num_heads
num_heads = 4     # 4 parallel attention heads, each with d_k = embed_dim // num_heads = 16

mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                             dropout=0.0, batch_first=True)

# batch_first=True means input is (batch, seq, embed_dim)
batch_size_mha = 2
T_seq_mha = len(COMPLAINT_TOKENS)   # 8

Q_mha = torch.randn(batch_size_mha, T_seq_mha, embed_dim)
K_mha = Q_mha
V_mha = Q_mha

attn_output_mha, attn_weights_mha = mha(Q_mha, K_mha, V_mha)

print("nn.MultiheadAttention reference demo")
print("=" * 40)
print(f"embed_dim={embed_dim}, num_heads={num_heads}, d_k_per_head={embed_dim//num_heads}")
print(f"Input shape:          {Q_mha.shape}")
print(f"Output shape:         {attn_output_mha.shape}")
print(f"Attention weights:    {attn_weights_mha.shape}  -> (batch, T_q, T_k) averaged across heads")
print()

# Parameter count: Q, K, V in_proj + out_proj
total_params_mha = sum(p.numel() for p in mha.parameters())
print(f"Total parameters in MultiheadAttention: {total_params_mha:,}")
print(f"Breakdown:")
print(f"  in_proj (Q+K+V projections):  3 x {embed_dim} x {embed_dim} = {3*embed_dim*embed_dim}")
print(f"  out_proj:                     {embed_dim} x {embed_dim} = {embed_dim*embed_dim}")
print(f"  biases:                       {4*embed_dim}")
print()
print(f"In Topic 4 you will build this from {num_heads} x ScaledDotProductAttention heads.")
```

**Notes**: The parameter count breakdown is important for the narrative "attention adds few parameters." The `batch_first=True` parameter requires PyTorch >= 1.9, which is satisfied in SageMaker Studio with PyTorch 2.8.0.

---

### Cell 29: [markdown] - Wrap-Up and Bridge to Topic 4

**Purpose**: Summarise Day 1. Bridge to Day 2 / Topic 4 (Transformers).

**Content**:
```
## Wrap-Up

### What you built today (Day 1 summary)

| Topic | What you implemented |
|-------|---------------------|
| 3a    | Bahdanau (additive) attention in NumPy - alignment scores, weights, context vector |
| 3a    | Dot product attention in NumPy |
| 3a    | Scaled dot product attention in NumPy |
| 3b    | DotProductAttention as nn.Module in PyTorch |
| 3b    | ScaledDotProductAttention as nn.Module (capstone) |
| 3b    | Applied self-attention to complaint tokens with heatmap visualisation |

### Key principles to carry forward

1. The seq2seq bottleneck is fixed by attention: instead of one context vector,
   we compute a different weighted average of encoder states at each decoder step.

2. Scaled dot product attention (Vaswani et al., 2017) is the core operation of
   every modern Transformer model. The scaling by 1/sqrt(d_k) prevents gradient
   saturation at large embedding dimensions.

3. Attention is largely parameter-free: the module itself has no weights.
   The learned weights live in the Q, K, V projection matrices outside the module.

4. Attention weights are interpretable: visualise them as a heatmap to understand
   what the model focuses on. This is critical for financial AI (explainability requirements).

### What is coming in Topic 4 - Transformers

In Topic 4 you will combine multiple ScaledDotProductAttention heads into
a full multi-head attention layer, add positional encoding, feed-forward blocks,
and build a complete Transformer encoder from scratch in PyTorch.
The capstone will be a GPU training job on a translation task.
```

---

### Cell 30: [markdown] - Homework Extensions

**Purpose**: Two async homework exercises.

**Content**:
```
## Homework Extensions

### Homework 1: Verify Against PyTorch Built-In

Your `ScaledDotProductAttention` should produce the same output as
`torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+).

```python
# Homework 1: verify numerical match with built-in
import torch.nn.functional as F

q = torch.randn(2, 6, 32)
k = torch.randn(2, 6, 32)
v = torch.randn(2, 6, 32)

my_output, _ = ScaledDotProductAttention()(q, k, v)
builtin_output = F.scaled_dot_product_attention(q, k, v)

match = torch.allclose(my_output, builtin_output, atol=1e-4)
print(f"Match with F.scaled_dot_product_attention: {match}")
# Should print True. If not, check your scaling factor and softmax dim.
```

### Homework 2: Add Causal Masking

Extend your `ScaledDotProductAttention` to support causal (autoregressive) masking.
A causal mask prevents position i from attending to any position j > i.

Signature extension:
```python
def forward(self, query, key, value, causal=False):
    ...
    if causal:
        # Build upper-triangular mask: positions in the future
        # Masked positions should have score = -inf so softmax gives 0
        pass
    ...
```

Verification: attention_weights[0] should be lower-triangular (zeros above diagonal).
Test with a sequence of 6 tokens.
```

---

### Cell 31: [markdown] - End of Notebook Marker

**Purpose**: Clear end marker.

**Content**:
```
---

*End of Topic 3b - Attention in PyTorch*

End of Day 1. Next session: Topic 4 - Transformers + Translator Capstone.
```

---

## Implementation Notes for /build-topic-notebook

1. Total cells: 31 markdown + code cells as planned. Several of the "single" cells above will split into pairs (markdown then code) during the 5-cell approval cadence, reaching the 40-55 target range. The build tool should count the Beat 2 diagram pairs (e.g., Cell 8 + Cell 9) as two cells each.

2. The critical Tier 3 rule: Cell 22 (capstone starter) must contain ONLY the class signature, docstring, and `pass`. No `= None  # YOUR CODE` lines, no numbered step comments, no hints about the implementation. Any hint inserted during build would violate the Tier 3 contract.

3. Variable continuity from Topic 3a: this notebook does NOT import word2vec or gensim. The `COMPLAINT_TOKENS` list defined in Cell 3 replaces the word2vec embeddings from Topic 3a. This is intentional - students see the same domain vocabulary but in a PyTorch tensor context.

4. Safety-nets: Cell 13 (Lab 1 safety-net) and Cell 24 (capstone safety-net) are both required. Both must be removed from the solution notebook.

5. The `ScaledDotProductAttentionReference` in Cell 17 is kept in BOTH notebooks (exercise and solution). In the solution notebook, Cell 22 is replaced with the full `ScaledDotProductAttention` implementation. The verification cell (Cell 23) remains identical in both notebooks.

6. No `evaluate` library, no OpenAI API, no getpass, no textblob, no swifter, no torchnlp. Clean imports only.

7. The `F.scaled_dot_product_attention` comparison in Cell 17 and Cell 23 uses a `try/except AttributeError` because some SageMaker Studio kernel images may have PyTorch < 2.0. The cell must not fail hard if the built-in is unavailable.

8. Both diagrams are in `plans/topic_3b/diagrams/`. The build-diagrams command will create these from the `<!-- DIAGRAM: -->` placeholders.

9. The source notebook (`9_Attention_with_Torch.ipynb`) imports `textblob`, `swifter`, `torchnlp` - DO NOT include these. They are not available by default in SageMaker Studio JupyterLab and are not needed for this notebook's content.

10. The multi-head attention section (Cells 27-28) is deliberately labelled "Reference Only" and explicitly says "You will build this in Topic 4." This prevents students from thinking they need to implement it today, and creates a clear forward bridge.
