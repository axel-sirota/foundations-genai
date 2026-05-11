# Topic 8 (Quantization) - Verification Report

Date: 2026-05-11
Notebooks audited:
- `/Users/axelsirota/repos/genai_for_developers/Exercises/topic_8_quantization/topic_8_quantization.ipynb`
- `/Users/axelsirota/repos/genai_for_developers/Solutions/topic_8_quantization/topic_8_quantization.ipynb`

## Pre-existing state (entering audit)

- Both notebooks already had 54 cells (parity OK).
- Diagram placeholders already wired correctly in Beat 2 markdown cells (Section 1 cell 6, Section 3 cell 26) with both `<!-- DIAGRAM: ... -->` and `[View diagram](...)` links pointing to existing `.mmd` files. No diagram changes needed.
- Both `.mmd` files present on disk:
  - `plans/topic_8_quantization/diagrams/quantization-precision-tradeoffs.mmd`
  - `plans/topic_8_quantization/diagrams/knowledge-distillation-architecture.mmd`
- Section 2 (Pruning) correctly has no diagram (intentional 2-diagram quota for Sections 1 and 3 only).
- AI-tells scan clean: no em dashes, en dashes, unicode multiplication signs, or bare `---` separator lines in either notebook.
- Lab tier distribution correct: Labs 1/2/3 = Tier 1 with numbered steps + `# YOUR CODE` stubs; QAT/qnnpack capstone = Tier 2; `compress_model` capstone = Tier 3 (exercise has only `pass # YOUR CODE`, solution has full implementation).
- Safety-nets present after every lab cell (Lab 1 cell 10, Lab 2 cell 20, Lab 3 cell 30, training_job_name cell 38) in both exercise and solution.
- `numpy<2` pinned in install cells; no `evaluation_strategy`; `wait=False` on `.fit()`; `ResourceNotFound` (not `ResourceNotFoundException`) in boto3 exception handlers (comments explicitly note the correct form); no `import evaluate`, no `mlflow`.
- HuggingFace estimator on `ml.g4dn.xlarge` GPU; endpoint on `ml.m5.xlarge`.
- Header cells (Cell 0/1) have no "Day N" reference.
- STAR (Situation/Task/Action/Result) wording present in all labs; Barclays narrative consistent.
- T8 self-contained setup note in place: cell 2 (`baseline_model` load) explicitly states model is reloaded fresh in T8, not carried from T7b kernel.

## Issues found and fixed

### 1. Stale cell-number cross-references (Section 4)

The Section 4 markdown cells referenced absolute cell numbers (`Cell 33`, `Cell 35`) that did not match the actual cells they meant. Cell 33 IS Section 4's own header, so "Use what you saw in Cell 33 as your reference" was self-referential. "Use Cell 35 to poll" pointed at the Tier 2 lab markdown rather than the boto3 polling cell (cell 40). "Run Cell 40 after training completes" in cell 46 pointed at the polling cell rather than the deploy cell.

Fixed by replacing absolute cell-number references with positional/descriptive references that survive future reordering:

- Exercise + Solution Cell 33 (Section 4 header): "Use what you saw in Cell 33 as your reference" -> "Use the HuggingFace estimator launch cell below as your reference".
- Exercise + Solution Cell 35 (Tier 2 lab markdown): "Then in Cell 33 launch..." -> "Then in the HuggingFace estimator launch cell below, launch..."; "Use Cell 35 to poll status" -> "Use the boto3 polling cell that follows to refresh status".
- Exercise + Solution Cell 41 (qnnpack lab markdown): "based on Cell 33" -> "based on the original launch cell above".
- Exercise Cell 42 (qnnpack starter code comment): "Refer to Cell 33 for the estimator pattern" -> "Refer to the original launch cell above for the estimator pattern". (Solution cell 42 had no such reference.)
- Exercise + Solution Cell 46 (endpoint test code): `print("Endpoint not yet deployed. Run Cell 40 after training completes.")` -> `print("Endpoint not yet deployed. Run the deploy cell above after training completes.")`

### 2. Stale internal reference in Lab 3 starter (exercise only)

- Exercise Cell 29 (Lab 3 starter code): "Reuse student_logits and teacher_logits from Cell 22." -> "Reuse student_logits and teacher_logits from the naive distillation demo above." Those variables are defined in Cell 25 (the naive distillation demo), not Cell 22, so the original line pointed students to the wrong cell.

## Post-fix verification (final scan)

```
== Exercise:  54 cells, AI-tells clean, 2 DIAGRAM placeholders, 0 stale Cell-N refs
== Solution:  54 cells, AI-tells clean, 2 DIAGRAM placeholders, 0 stale Cell-N refs
```

All audit checklist items pass:
- Four-beat arc preserved (B1 broken / B2 diagram / B3 demo / B4 lab) for Sections 1 and 3; Section 2 follows the same pattern minus diagram by design.
- Both diagrams wired in Beat 2 of Section 1 and Section 3 with comment placeholder + `[View diagram]` link to existing `.mmd` files.
- Safety-nets retained in both exercise and solution (rule explicitly forbids removing them from the solution).
- Cell parity 54/54.
- No markdown chain > 3.
- No AI-tells (em/en dashes, unicode mul, bare `---`).
- Lab tiers correct, Tier 3 `compress_model` open-ended (exercise = `pass # YOUR CODE` only with no hints; solution = full implementation with size/latency branches).
- SageMaker constraints respected (instance types, `wait=False`, `numpy<2`, no `evaluation_strategy`, no `import evaluate`, no `mlflow`, `ResourceNotFound` exception).
- STAR/Barclays narrative continuity preserved.
- Setup cell makes T8 self-contained (baseline_model reloaded fresh, not from T7b kernel).
