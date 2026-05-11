# Notebook Audit: topic_7b_peft_lora_distilbert
Generated: 2026-05-11

## Status: PASS

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | PASS |
| Diagrams (2, markdown, Beat 2 position) | PASS |
| Safety-nets (exercise) | PASS |
| Safety-nets (solution kept) | PASS |
| AI-tells | PASS |
| Lab tiers | PASS |
| YOUR CODE hygiene | PASS |
| Notebook header (no Day N) | PASS |
| Narrative / Barclays STAR | PASS (15 Barclays mentions in both nbs) |
| SageMaker constraints | PASS |
| Discussion prompts | PASS |
| Cell parity (ex=44, sol=44) | PASS |
| No markdown chain > 3 | PASS (max chain = 2) |
| Solution clean | PASS |

### Detailed Findings

**AI-tells scan (programmatic):** Both notebooks scanned for em dash (U+2014), en dash (U+2013), unicode multiplication (U+00D7), and bare `---` separator lines. Result: `clean` on both.

**Cell parity:** Exercise = 44 cells, Solution = 44 cells, structure identical (verified by side-by-side first-line listing).

**Diagrams:**
- Cell 5 (markdown): `<!-- DIAGRAM: PEFT methods comparison ... -->` -> `[View diagram](../../plans/topic_7b_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd)` -- file exists (782B).
- Cell 17 (markdown): `<!-- DIAGRAM: QLoRA architecture ... -->` -> `[View diagram](../../plans/topic_7b_peft_lora_distilbert/diagrams/qlora-architecture.mmd)` -- file exists (583B).
- Exactly 2 placeholders, both in markdown cells, both positioned as Beat 2 (after Beat 1 broken code, before Beat 3 working demo). Full folder names used.

**Four-beat arc per section:**
- Section 1 (cells 3-13): B1=c4 (manual LoRA injection fails), B2=c5 (PEFT methods diagram), B3=c6 (PEFT library demo) + c7 (verification), B4=c8 (Lab 1 markdown) + c9 (lab starter).
- Section 2 (cells 15-23): B1=c16 (bitsandbytes CPU fail), B2=c17 (QLoRA diagram), B3=c18 (full QLoRA), B4=c19 (Lab 2 markdown) + c20 (lab starter).
- Section 3 (cells 25-31): B1=c26 (num_virtual_tokens > max_length mistake), B3=c27 (correct prompt config), B4=c28 (Lab 1b markdown) + c29 (lab starter). No Beat 2 -- acceptable since only 2 diagrams allowed per notebook.
- Section 4 / Capstone (cells 33-42): setup c34, launch c35, safety-net c36, polling c37, metrics c38, Capstone Lab c39 + c40. No Beat 2 -- already 2 diagrams used.

**Lab tiers (per expected layout for T7b):**
- Lab 1 cell 8: "Tier 1 Guided, ~15 min" -- correct.
- Lab 2 (QLoRA) cell 19: "Tier 2 Hard, 25-35 min" -- correct.
- Lab 1b (soft prompt) cell 28: "Tier 1 Guided, ~10 min" -- correct.
- Capstone cell 39: "Tier 2 Hard, 25-35 min" -- correct.
- No Tier 3 in this topic (matches spec).
- Each lab has a Stretch markdown cell + Homework Extension code cell.

**YOUR CODE hygiene:** All 19 `# YOUR CODE` lines across the exercise notebook are bare (no inline answer hints). Solution notebook has zero `YOUR CODE` markers remaining and zero bare `pass` lab stubs.

**Safety-nets (exercise):**
- Lab 1 safety-net cell 10: correct `# Safety-net: run this if...` / `# SKIP this cell if...` header, `if peft_model_r16 is None:` guard.
- Lab 2 safety-net cell 21: probe-based guard (calls `build_qlora_model` in try/except), correct header.
- Lab 1b safety-net cell 30: `if prompt_model is None:` guard, correct header.
- `training_job_name` safety-net cell 36 immediately follows the `.fit(wait=False)` cell 35.

**Safety-nets (solution):** All safety-net cells retained in solution notebook (cells 10, 21, 30, 36).

**Header (cell 0):** "Topic 7b - PEFT and LoRA with DistilBERT" -- no "Day 1/2/3" reference.

**SageMaker constraints:**
- `numpy<2` pinned in install cells (3 occurrences).
- No `import evaluate` / `from evaluate import` anywhere.
- `eval_strategy=` used; no `evaluation_strategy=`.
- 1 `.fit(` call, 2 `wait=False` occurrences (estimator + safety guidance).
- HuggingFace estimator on `ml.g4dn.xlarge` (GPU) -- correct.
- No `ResourceNotFoundException` (no `boto3` exception handling in this notebook -- N/A).
- No `mlflow` reference (correctly absent outside F2).

**Discussion prompts:** Markdown cells at 14 ("Discussion (3 minutes)"), 24 ("Peer Discussion (3 min)"), 32 ("Discussion (3 minutes)"). Three discussion checkpoints across four sections -- adequate.

**Markdown chain:** Max consecutive markdown run = 2 cells. Well under the 3-cell limit.

### Issues to Fix
None. Both notebooks pass every audit gate.

### Files Audited
- `/Users/axelsirota/repos/genai_for_developers/Exercises/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb`
- `/Users/axelsirota/repos/genai_for_developers/Solutions/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb`
- `/Users/axelsirota/repos/genai_for_developers/plans/topic_7b_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd`
- `/Users/axelsirota/repos/genai_for_developers/plans/topic_7b_peft_lora_distilbert/diagrams/qlora-architecture.mmd`
