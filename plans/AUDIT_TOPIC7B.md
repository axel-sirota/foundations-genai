# Notebook Audit: topic_7b_peft_lora_distilbert
Generated: 2026-05-11

## Status: PASS (with minor WARNs)

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | WARN |
| Diagrams (2, markdown, Beat 2 position) | PASS |
| Safety-nets (exercise) | PASS |
| Safety-nets (solution kept) | PASS |
| AI-tells | PASS |
| Lab tiers | PASS |
| YOUR CODE hygiene | PASS |
| Notebook header (no Day N) | PASS |
| Narrative / Barclays STAR | WARN |
| SageMaker constraints | PASS |
| Discussion prompts | PASS |
| Cell parity (ex=44, sol=44) | PASS |
| No markdown chain > 3 | PASS |
| Solution clean | PASS |

### Per-section four-beat verification

| Section | B1 | B2 (diagram) | B3 | B4 (lab md + starter) | Result |
|---------|----|--------------|----|-----------------------|--------|
| 1 PEFT LoRA | cell 4 | cell 5 | cell 6 | cells 8 + 9 | PASS |
| 2 QLoRA | cell 16 | cell 17 | cell 18 | cells 19 + 20 | PASS |
| 3 Soft Prompts | cell 26 | (none — quota of 2 used) | cell 27 | cells 28 + 29 | WARN — no Beat 2 |
| 4 Capstone | (none — capstone) | (none) | cell 35 | cells 39 + 40 | WARN — no broken/naive Beat 1 |

### Diagram references verified
- `plans/topic_7b_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd` exists, full folder name used.
- `plans/topic_7b_peft_lora_distilbert/diagrams/qlora-architecture.mmd` exists, full folder name used.

### Safety-net inventory (exercise)
- Cell 10 — Lab 1 (`peft_model_r16`) — correct header, immediately after starter (cell 9). PASS
- Cell 21 — Lab 2 (`build_qlora_model`) — function-probe variant, correct header. PASS
- Cell 30 — Lab 1b (`prompt_model`) — correct header. PASS
- Cell 36 — Capstone `training_job_name` — immediately after `.fit(wait=False)` (cell 35). PASS

### Safety-net inventory (solution)
- Cells 10, 21, 30, 36 retained in solution. PASS

### AI-tells programmatic scan
- Em dash / en dash / unicode multiplication / bare `---` lines: 0 hits in both notebooks. PASS

### SageMaker constraints
- `numpy<2` pinned in install cell (cell 1). PASS
- No `import evaluate` anywhere. PASS
- `eval_strategy="epoch"` used (solution cell 40, train.py line 154). No `evaluation_strategy=`. PASS
- `estimator.fit(wait=False)` on cell 35. PASS
- HuggingFace estimator on `ml.g4dn.xlarge` (cell 35). PASS
- No PyTorch estimator needed in this topic (only one GPU job).
- `boto3` exception namespace: no usage (script uses `describe_training_job` polling, not exception). N/A — PASS
- No `mlflow` references. PASS
- `scripts_topic7b/requirements.txt`: `peft>=0.6.0,<0.8.0`, `bitsandbytes>=0.41.0`, `datasets==2.18.0`, `numpy<2`. PASS

### Cell parity
- Exercise: 44 cells.
- Solution: 44 cells. PASS

### Issues to Fix

**WARN — Section 3 (Soft Prompts) has no Beat 2 diagram.**
The notebook spec caps diagrams at exactly 2. Sections 1 and 2 use both quota slots. Section 3 jumps directly from B1 (cell 26 broken-naive) to B3 (cell 27 working demo) with no markdown beat in between. This is a structural tradeoff between the "exactly 2 diagrams per notebook" rule and the "B2 per section" rule. Acceptable if intentional; otherwise insert a markdown cell at position 27 (without a `<!-- DIAGRAM: -->` placeholder) explaining the soft-prompt mechanism visually to fulfil B2.

**WARN — Section 4 (Capstone) has no traditional Beat 1.**
Cell 33 is a section markdown intro; cell 34 is setup; cell 35 launches the working job (Beat 3). There is no broken/naive code that fails first. This is typical for a capstone section but technically violates the strict four-beat arc per the audit rubric. Acceptable if intentional.

**WARN — Lab 1b (Section 3, Soft Prompt) missing Stretch + Homework Extension cells.**
Cells around Lab 1b: 28 (lab md), 29 (starter), 30 (safety-net), 31 (comparison code). No `### Stretch (for fast finishers)` markdown and no Homework Extension starter cell exist for Lab 1b. Compare with Lab 1 (cells 12+13), Lab 2 (cells 22+23), and Capstone (cells 41+42), which all have both. Fix: add a Stretch markdown + Homework Extension code cell after cell 30 (and matching pair after cell 30 in the solution).

**WARN — Lab 1b STAR structure incomplete.**
Cell 28 has Situation and Task headers but no explicit Action and Result headers (compare Lab 1 cell 8, which includes all four). Minor narrative consistency issue.

**INFO — No Tier 3 lab in T7b.**
Spec explicitly states this is correct (Tier 3 is only in T8). PASS.

**INFO — Capstone has no auto-verification cell.**
Cell 39 markdown explicitly states "There is no auto-verification cell: you submit a SageMaker job and the instructor reviews the logs." Acceptable for Tier 2 capstone.

### Files audited
- `/Users/axelsirota/repos/genai_for_developers/Exercises/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb`
- `/Users/axelsirota/repos/genai_for_developers/Solutions/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb`
- `/Users/axelsirota/repos/genai_for_developers/Exercises/topic_7b_peft_lora_distilbert/scripts_topic7b/train.py`
- `/Users/axelsirota/repos/genai_for_developers/Exercises/topic_7b_peft_lora_distilbert/scripts_topic7b/requirements.txt`
- `/Users/axelsirota/repos/genai_for_developers/plans/topic_7b_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd`
- `/Users/axelsirota/repos/genai_for_developers/plans/topic_7b_peft_lora_distilbert/diagrams/qlora-architecture.mmd`
