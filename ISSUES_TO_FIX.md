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

## 3. GPU image / PyTorch-CUDA mismatch  [DONE]

**Problem:** A SageMaker kernel may land on a GPU instance or a CPU one depending
on quota. On a GPU box the image's `torch` is sometimes CPU-only, so
`torch.cuda.is_available()` is wrongly `False` and the notebook silently runs on
CPU (or fails). Originally hit on Spaces forced onto `ml.t3.medium`.

**AWS reality check** (profile `datacouch`, account 962804699607, us-west-2):
- Training-job GPU quota: ONLY `ml.g4dn.xlarge` is non-zero (100). All `g5.*`,
  `p3.*`, `p4.*` and bigger `g4dn` are 0.0 -> a job requesting them is rejected.
- Notebook-instance quota: `ml.g4dn.xlarge` = 200, `ml.t3.medium` = 100.
  So the kernel can be GPU or CPU -> safety-net must handle both.
- Every `train.py` / notebook already targets `ml.g4dn.xlarge` (T4 16GB) -
  consistent with the only available quota. No instance-type change needed.

**Fix applied:** Inserted a CUDA health-check cell (markdown header + code) right
after the first `import torch` cell in all 10 GPU notebooks (Exercises +
Solutions, topics 6a/6b/7a/7b/8). Logic:
1. No GPU hardware (`nvidia-smi` absent / non-zero exit) -> no action, CPU run.
2. GPU present + `torch.version.cuda` set -> print OK.
3. GPU present + `torch` CPU-only -> reinstall a matching CUDA wheel, tell the
   student to restart the kernel.

Wheel selection combines two signals: `nvidia-smi` driver max-CUDA, and the
SageMaker metadata file `/opt/ml/metadata/resource-metadata.json` (image/instance
identity, logged for diagnostics). `_pick_wheel` chooses the newest of
cu126/cu124/cu121 that is <= the driver CUDA; defaults to cu121 if unreadable.

**Scope:** 10 GPU notebooks. Cell source archived at `/tmp/cuda_safetynet_cell.py`.

## 4. Separate pip-install cells from Python code cells  [DONE]

**Problem:** Some notebooks mix `pip install` commands and Python code in the same
cell. It runs fine for students, but it is confusing.

**Fix applied:** Split every genuinely mixed cell into two: a pip cell + a code
cell. 18 mixed cells split across all topics (Exercises + Solutions).

Note on `%%bash`: NOT used. Install cells must stay Python cells with `!pip`
magics — `%%bash` runs in a separate subshell, breaks `!{sys.executable} -m pip`,
and can install into the wrong interpreter.

**Restart reminder:** every pip cell that installs an env-critical package
(`numpy` / `torch` / `transformers` / `accelerate`) ends with:

```python
print("RESTART KERNEL before continuing -- environment packages were installed/upgraded.")
```

6 pip-only cells (topic 4, 6b, sagemaker_fundamentals) got the reminder without
needing a split. Topic 2 cell 42 (sentence-transformers appendix) was correctly
skipped - its only "pip install" text is inside comments/strings.

**Scope:** All notebooks. Done via `/tmp/fix_issue4_split_cells.py`.

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
