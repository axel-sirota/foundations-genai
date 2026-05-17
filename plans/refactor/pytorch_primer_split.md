# Phase 7 - PyTorch Primer Split (ISSUES_TO_FIX issue 6)

## Problem
`Frameworks/pytorch_refresher.ipynb` (49 cells, 5 sections) takes too long in
class, so delivery never reaches the fine-tuning content. Split into a lean
CORE notebook + an OPTIONAL "nice to know" notebook.

## The 5 sections (current notebook)
1. Tensors - how complaint text becomes numbers
2. Autograd - manual `requires_grad` scalar training loop
3. Dataset and DataLoader - batching complaints
4. Classifier with `nn.Module` (uses `optim.AdamW`, `loss.backward()`, `optimizer.step()`)
5. HuggingFace Trainer - subclass `Trainer`, override `compute_loss`

## Core vs Optional decision

Rule from issue 6: "core = whatever the fine-tuning topics actually depend on."

The fine-tuning topics (topic_4, topic_5, topic_6a) depend on: torch tensors,
`Dataset`/`DataLoader`, and `nn.Module` model definitions. They use
`AutoModelForSequenceClassification` and SageMaker `HuggingFace` estimators -
they do NOT hand-write a `requires_grad` gradient loop, and they do NOT
subclass `Trainer.compute_loss` from scratch.

### CORE - `Frameworks/pytorch_refresher.ipynb` (trimmed)
- Section 1 Tensors
- Section 3 Dataset and DataLoader (renumbered Section 2)
- Section 4 nn.Module classifier (renumbered Section 3)

Section 4 already self-contains its training loop via the high-level `optim`
API (`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`). A short
"how training works" paragraph is added to the Section 4 markdown so
`backward()`/`step()` are not unexplained once the manual-autograd section is
gone. Section 4 Beat 1 (raw-tensor classifier) keeps its own `requires_grad`
usage - it is a self-contained "do not do this" demo.

### OPTIONAL - `Frameworks/pytorch_optional_deep_dive.ipynb` (new)
- Section 2 Autograd (the manual `requires_grad` scalar training loop and the
  Discussion cell) - conceptual deep dive, not used by the required path.
- Section 5 HuggingFace Trainer (subclassing `Trainer`) - the heaviest deps
  (transformers, datasets) and a full `trainer.train()`. The real fine-tuning
  topics use SageMaker estimators, so this is genuinely nice-to-know.

The OPTIONAL notebook is self-contained: it gets its own setup cells (TF
disable, env/imports) and re-creates any tensors it needs (it never imports
state from CORE).

## Cell mapping

### CORE notebook (target 33 cells)
Keep, in order, original indices:
`0,1,2,3` (title + setup),
`4,5,6,7,8,9,10,11,12` (Section 1 Tensors),
`23,24,25,26,27,28,29,30` (Section 3 -> renumber to Section 2),
`31,32,33,34,35,36,37,38` (Section 4 -> renumber to Section 3),
`47,48` (Wrap-Up + final sanity check).
- Drop Section 2 cells `13..22` and Section 5 cells `39..46`.
- Edit cell 0 (title list: drop Autograd + HF Trainer lines, add pointer to the
  optional notebook).
- Edit Section markdown headers `23` -> "Section 2", `31` -> "Section 3".
- Add an autograd primer paragraph into cell `31` (Section 4 -> 3 intro).
- Edit cell `47` Wrap-Up table (drop Autograd + HF Trainer rows, add pointer).
- Edit cell `48` final sanity check: drop the `w`/`b` (Section 2) and
  `trainer5` (Section 5) checks; renumber section labels.

### OPTIONAL notebook (target ~22 cells)
- New title markdown cell.
- New setup cell pair (copy of original `1` + `2`).
- New `import torch as pt` + seed cell (copy of original `3`).
- Autograd block: original `13,14,15,16,17,18,19,20,21,22` (renumber Section 1).
- HF Trainer block: original `39,40,41,42,43,44,45,46` (renumber Section 2).
  - The Trainer cells reference `model`/`dataset`/`loader` from CORE Section 4;
    the optional notebook re-creates these locally in a new bridge cell so it
    runs standalone.
- New Wrap-Up markdown cell.

## Runnability guarantee
- Each of the 4 resulting notebooks (core exercise, core solution, optional
  exercise, optional solution) is self-contained: defines every name it uses.
- Gates per notebook: `nbformat.validate`, per-cell `ast.parse`,
  concatenated-pyflakes 0 undefined names (magics/`!pip` stripped).
- Solutions built by copying the exercise split then swapping lab cells with
  the original solution lab bodies.

## Files
- Modified: `Frameworks/pytorch_refresher.ipynb` (CORE exercise, trimmed)
- Modified: `Frameworks/pytorch_refresher_solution.ipynb` (CORE solution, trimmed)
- New: `Frameworks/pytorch_optional_deep_dive.ipynb` (OPTIONAL exercise)
- New: `Frameworks/pytorch_optional_deep_dive_solution.ipynb` (OPTIONAL solution)
