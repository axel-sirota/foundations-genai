# Issues To Fix

Environment and notebook issues found during course delivery. Each entry is a fix to apply.

## 1. TensorFlow import conflicts in all notebooks

**Problem:** Notebooks hit TensorFlow-related errors (transformers tries to load the TF backend on SageMaker images).

**Fix:** Add this to the top of every notebook (Exercises and Solutions), before any `transformers` import:

```python
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

**Scope:** All notebooks, everywhere.

## 2. transformers / accelerate version pinning  [DONE]

**Problem:** Wrong/incompatible versions of `transformers` and `accelerate` caused errors.

**Original note said** `transformers>=4.40` + `accelerate==0.28.0` — but `0.28.0`
is stale (Mar 2024) and the real compatible matrix was researched against
transformers 4.53.0 `setup.py`:

| Package | Pin applied | Why |
|---|---|---|
| transformers | `>=4.53,<4.54` | target line, capped at minor |
| accelerate | `>=1.0.0` | transformers 4.53 floor is `>=0.26`; 1.0+ is safe |
| tokenizers | `>=0.21,<0.22` | transformers 4.53 REQUIRES this; old `<0.20` conflicts |
| huggingface_hub | `>=0.30.0,<1.0` | transformers 4.53 requires `>=0.30`; old `<0.25` conflicts |
| numpy | `<2` | unchanged (course constraint) |

CUDA note: transformers/accelerate are pure Python (no CUDA variant). Only
`torch` has a CUDA wheel — see issue 3.

**Applied:** 14 install cells across Exercises/Solutions topics 2,5,6a,6b,7a,7b,8.

**Scope:** Install cells in notebooks that use `transformers` / `accelerate`.

## 3. GPU image / PyTorch-CUDA mismatch in Spaces and training

**Problem:** GPU notebooks (Spaces, training) need a PyTorch build with CUDA support.
Spaces were run on `ml.t3.medium`, where the only available image was SageMaker
`4.1.0` — which is NOT GPU/CUDA-compatible. We had to reinstall PyTorch from scratch
to get a CUDA build.

**Fix (primary):** When a notebook runs GPU work even once, select the correct
GPU-capable image up front (SageMaker PyTorch+CUDA image). Search for and pin the
right image rather than relying on whatever the small instance defaults to.

**Fix (safety-net in notebook):** Add a CUDA detection + auto-repair cell. Logic:

1. If no CUDA device present -> no action (CPU run, nothing to fix).
2. If CUDA present, check whether the installed `torch` is a CUDA build
   (`torch.cuda.is_available()` and `torch.version.cuda`).
3. If CUDA present but `torch` is CPU-only -> reinstall the CUDA wheel as a fallback:

```python
import torch, subprocess, sys

# only act if a GPU is physically present but torch can't use it
if torch.version.cuda is None:
    # CPU-only torch on a CUDA box: reinstall the CUDA 12.6 wheel
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    ], check=True)
    print("Reinstalled torch with CUDA 12.6. Restart the kernel.")
else:
    print("torch CUDA build OK:", torch.version.cuda)
```

CUDA 12.6 wheel index: `https://download.pytorch.org/whl/cu126` (verified May 2026).

**Scope:** GPU notebooks (Spaces, training). Primary fix is image selection;
the safety-net cell is the fallback when the image is wrong.

## 4. Separate pip-install cells from Python code cells

**Problem:** Some notebooks mix `pip install` commands and Python code in the same
cell. It runs fine for students, but it is confusing.

**Fix:** Always keep `pip install` in its own cell, separate from Python code cells.

**Fix (kernel-restart reminder):** When a pip cell modifies anything environment-
critical (`numpy`, `pytorch`/`torch`, `transformers`, `accelerate`, etc.), add an
`echo` at the end of that cell reminding the student to restart the kernel:

```bash
pip install "transformers>=4.40" "accelerate==0.28.0"
echo "RESTART KERNEL before continuing (transformers/accelerate changed)."
```

**Scope:** All notebooks. Split mixed cells; add the restart `echo` to every pip
cell that touches `numpy` / `torch` / `transformers` / `accelerate`.

## 5. Attention / transformers topics too math-heavy; 6a sec 2-3 unclear

**Status:** OPEN - approach not yet decided. Discuss before acting.

**Problem (delivery feedback):**
- The attention and transformers topics lean too hard on the math / internals.
  Students did not care how it works under the hood; they care how they USE it.
  Delivery was rescued by skipping ahead into the fine-tuning / LLM part, but
  that is a poor experience and should not be the plan.
- Topic 6a, sections 2 and 3: not understandable as written.
- The attention notebook refactor also needs a legibility pass.

**Decision points (TBD with Axel):**
1. How far the attention + transformers rewrite should go:
   - cut the from-scratch math, reframe entirely around usage; OR
   - keep math but move it to an optional `topic_N_optional_<slug>.ipynb`
     deep-dive (CLAUDE.md optional-notebook pattern); OR
   - trim the math hard and wrap it in more intuition + usage framing.
2. Whether 6a sections 2-3 suffer the same root cause (too internals-heavy)
   or are just unclearly written and need a legibility pass only.
   -> needs a read of 6a sec 2-3 to diagnose.

**Scope:** Topics 3a/3b (attention), 4 (transformers), 6a sec 2-3. Likely the
largest item here - treat as a redesign, not a patch.

## 6. PyTorch refresher too long; split core vs optional

**Problem (delivery feedback):** The PyTorch refresher took too long in class, so
we never reached the content that actually matters (the PEFT / fine-tuning part).

**Fix:** Trim and split the PytorchPrimer track into:
- a lean CORE path - only what the course actually uses downstream; and
- an OPTIONAL "nice to know" notebook for the rest.
Goal: get to real content fast.

**Decision point (TBD):** which of the 5 Primer exercises (tensors, autograd/GPU,
Dataset/DataLoader, nn.Linear classifier, nn.Sequential classifier) are core vs
optional. Core = whatever the fine-tuning topics actually depend on.

**Scope:** PytorchPrimer track only (not the in-topic PyTorch sections).

## 7. Topic 6a: estimator.transformers_version AttributeError

**Problem:** Topic 6a builds a SageMaker `HuggingFace` estimator with
`transformers_version="4.56.2"` passed as a constructor arg, then a `print`
cell reads it back as `estimator.transformers_version`. That attribute is not
reliably exposed by the HuggingFace estimator after construction, so the cell
fails with `AttributeError` and the fine-tuning section breaks at that point.

Affected lines (both files have the bug):
- `Exercises/topic_6a_full_finetuning/topic_6a_full_finetuning.ipynb` ~L1110
- `Solutions/topic_6a_full_finetuning/topic_6a_full_finetuning.ipynb` ~L1187

```python
print(f"  transformers_version: {estimator.transformers_version}")  # AttributeError
```

**Fix:** Do not read framework versions back off the estimator. Print the
literals that were passed into the constructor instead (define them as variables
above the `HuggingFace(...)` call and reuse them), e.g.:

```python
TRANSFORMERS_VERSION = "4.56.2"
PYTORCH_VERSION = "2.1.0"
PY_VERSION = "py310"
estimator = HuggingFace(..., transformers_version=TRANSFORMERS_VERSION,
                        pytorch_version=PYTORCH_VERSION, py_version=PY_VERSION)
print(f"  transformers_version: {TRANSFORMERS_VERSION}")
```

Also check `estimator.pytorch_version` / `estimator.py_version` on the adjacent
print lines - same risk, same fix.

**Scope:** Topic 6a exercise + solution. Verify against the installed
`sagemaker` SDK version before finalizing the attribute names.
