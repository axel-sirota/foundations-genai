# Notebook Audit: topic_8_quantization
Generated: 2026-05-11

## Status: WARN (1 minor YOUR CODE hygiene issue; everything else PASS)

Exercise: `Exercises/topic_8_quantization/topic_8_quantization.ipynb` (54 cells)
Solution: `Solutions/topic_8_quantization/topic_8_quantization.ipynb` (54 cells)

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | PASS |
| Diagrams (2, markdown, Beat 2 position) | PASS |
| Safety-nets (exercise) | PASS |
| Safety-nets (solution kept) | PASS |
| AI-tells | PASS (clean on both notebooks) |
| Lab tiers | PASS (Labs 1/2/3 = Tier 1, QAT capstone = Tier 2, compress_model = Tier 3) |
| YOUR CODE hygiene | WARN (cell 42 has descriptive hint after `# YOUR CODE:`) |
| Notebook header (no Day N) | PASS |
| Narrative / Barclays STAR | PASS |
| SageMaker constraints | PASS |
| Discussion prompts | PASS (cells 13, 16, 52) |
| Cell parity (ex=54, sol=54) | PASS |
| No markdown chain > 3 | PASS |
| Solution clean | PASS |

---

## Detailed Findings

### AI-tells scan (programmatic)
Ran the canonical scanner on both notebooks. Result: **clean** on both. No em dashes, en dashes, unicode multiplication signs, or bare `---` separator lines anywhere.

### Cell parity
- Exercise: 54 cells
- Solution: 54 cells
- Cell-by-cell structure mirrors exactly (same section order, same beat order, same lab/safety-net/homework arrangement at identical indexes).

### Four-Beat Arc

**Section 1 (Quantization)** — PASS
- Beat 1 code: cell 5 (naive PTQ that runs and demonstrates collapse)
- Beat 2 markdown + diagram: cell 6 (`<!-- DIAGRAM: Quantization precision tradeoffs -->`)
- Beat 3 demo: cell 7 (proper static PTQ with calibration, heavily commented)
- Beat 4 lab: cell 8 (markdown) + cell 9 (Tier 1 starter)

**Section 2 (Weight Pruning)** — PASS (intentionally no diagram per topic spec)
- Beat 1 code: cell 15 (naive 80% pruning)
- Beat 2 substitute markdown: cell 16 (Discussion: How Much Pruning Is Safe?) — no diagram placeholder, per the topic decision
- Beat 3 demo: cell 17 (conservative 20% L1 unstructured)
- Beat 4 lab: cell 18 + cell 19

**Section 3 (Knowledge Distillation)** — PASS
- Beat 1 code: cell 25 (naive distillation with T=1)
- Beat 2 markdown + diagram: cell 26 (`<!-- DIAGRAM: Knowledge distillation architecture -->`)
- Beat 3 demo: cell 27 (proper distillation T=4, alpha=0.5)
- Beat 4 lab: cell 28 + cell 29

**Section 4 (QAT capstone)** — PASS (SageMaker remote job, Tier 2)

### Diagrams
Exactly 2 `<!-- DIAGRAM: -->` placeholders per notebook, both in markdown cells, both immediately before their Beat 3 demo:
1. Cell 6: `quantization-precision-tradeoffs.mmd` — file exists at `plans/topic_8_quantization/diagrams/quantization-precision-tradeoffs.mmd`
2. Cell 26: `knowledge-distillation-architecture.mmd` — file exists at `plans/topic_8_quantization/diagrams/knowledge-distillation-architecture.mmd`

Paths use full folder name (`topic_8_quantization`). PASS.

### Safety-nets
Exercise notebook safety-net cells (each immediately after the lab/job they protect):
- Cell 10: Lab 1 safety-net (after cell 9)
- Cell 20: Lab 2 safety-net (after cell 19)
- Cell 30: Lab 3 safety-net (after cell 29)
- Cell 38: `training_job_name` safety-net (after `.fit(wait=False)` in cell 37)

All use `# Safety-net: run this if...` + `# SKIP this cell if...` headers. Solution notebook retains all 4 safety-net cells (per spec, safety-nets are kept in solution).

### Lab Tiers
- **Tier 1 (guided)**: cells 8/9 (Lab 1), 18/19 (Lab 2), 28/29 (Lab 3) — each has numbered Steps, `= None  # YOUR CODE` stubs, hints on separate `# Hint:` lines, and a verification block.
- **Tier 2 (hard)**: cells 35/36/37 (QAT + LoRA SageMaker job) and cells 41/42 (qnnpack continuation). Labelled "Tier 2 Hard Lab (25 to 35 min)". Has stretch in cell 43.
- **Tier 3 (open-ended)**: cells 50/51 (`compress_model` pipeline). Labelled "Tier 3 Open-Ended". Exercise cell 51 contains function signature + docstring + `pass  # YOUR CODE` only (no numbered steps, no hints). Solution cell 51 has full implementation with branching on `target=size|latency`, size measurement via temp file, accuracy loop, and metrics dict.

Day 3 tier budget respected: ONE Tier 2 (Section 4) and ONE Tier 3 (final capstone), Tier 3 only in last topic.

### Homework Extensions
Present after every in-class lab:
- Cell 11/12: INT4 bitsandbytes (after Lab 1)
- Cell 21/22: Lottery Ticket pruning + fine-tune (after Lab 2)
- Cell 31/32: alpha/T grid search (after Lab 3)

### compress_model verification
- **Exercise cell 51**: signature + docstring + `pass  # YOUR CODE`. No implementation, no hints, no numbered steps. Correct Tier 3 form.
- **Solution cell 51**: full implementation supporting `target="size"` (dynamic int8) and `target="latency"` (20% L1 prune + dynamic int8), with size-on-disk measurement and accuracy loop over the dataset, returning `(compressed_model, metrics)` with `size_mb`, `accuracy`, `technique` keys.

### SageMaker constraints
- `numpy<2` present in install cells: PASS
- `import evaluate` / `from evaluate`: NOT present: PASS
- `evaluation_strategy=`: NOT present: PASS (no `eval_strategy=` either — TrainingArguments lives in `scripts_topic8/train.py`)
- `.fit(wait=False)`: cell 37 calls `estimator.fit(wait=False)`; cell 42 is configuration only and explicitly tells students not to call `.fit()`. PASS.
- `ml.g4dn.xlarge` present for HuggingFace estimator: PASS
- `ResourceNotFound` (not `ResourceNotFoundException`): PASS — cells 40 and 47 use `sm_client.exceptions.ResourceNotFound` and include the inline comment noting the boto3 naming gotcha.

### Discussion prompts
- Cell 13: Peer Discussion (3 min) between Section 1 and Section 2
- Cell 16: Discussion (3 min): How Much Pruning Is Safe? (between Section 2 beats)
- Cell 52: Discussion (5 min): Which Technique Would You Deploy at Barclays? (after Tier 3 capstone)

All are markdown cells. PASS.

### Header (no Day N)
Cell 0 begins with `# Topic 8 - Model Compression: Quantization, Pruning and Distillation` followed by `## Barclays Customer Support Intelligence System`. No "Day 1/2/3" anywhere in either notebook's header. PASS.

### Markdown chain
No streaks of more than 3 consecutive markdown cells anywhere in either notebook. PASS.

### Solution cleanliness
- No `= None  # YOUR CODE` remaining in any solution lab cell.
- Solution cell 51 (Tier 3) has full implementation (not `pass`).
- Solution safety-net cells retained as required.

---

## Issues to Fix

### WARN — YOUR CODE hygiene (Exercise cell 42, id `26709d0f`)

Current line:
```
# YOUR CODE: create a second HuggingFace estimator with qnnpack backend and lora_r=16.
```

The `# YOUR CODE:` form with a descriptive instruction after the colon violates the hygiene rule that `# YOUR CODE` should appear on the `variable = None  # YOUR CODE` placeholder line only, with hints kept on separate lines (e.g. `# Hint: ...`). The Tier 1 labs (cells 9, 19, 29) follow the correct pattern; cell 42 (Tier 2) is the only violator.

**Suggested fix**: rename the leading comment so it no longer collides with the hygiene rule. For example:
```
# Task: create a second HuggingFace estimator with qnnpack backend and lora_r=16.
# Refer to Cell 33 for the estimator pattern.
# Name it estimator_v2.

estimator_v2 = None  # YOUR CODE
```

This is a single-line rename in both Exercise and Solution cell 42 (same comment header text exists in both).

---

## MANDATORY Before Every Cell Edit (when fixing after audit)
1. Read the full notebook to get current cell list and IDs.
2. Identify the cell BEFORE and AFTER the insertion/edit point by cell_id.
3. Only then call NotebookEdit with the confirmed cell_id.
4. Never assume cell ordering from a previous read — re-read if in doubt.
