# Notebook Audit: topic_8_quantization
Generated: 2026-05-11

## Status: PASS (with minor WARN)

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
| Narrative / Barclays STAR | PASS |
| SageMaker constraints | PASS |
| Discussion prompts | WARN |
| Cell parity (ex=54, sol=54) | PASS |
| No markdown chain > 3 | PASS |
| Solution clean | PASS |

### Detail Notes

**Four-beat arc**
- Section 1 (Quantization): B1 cell 4-5 (naive PTQ broken demo), B2 cell 6 (markdown with diagram), B3 cell 7 (proper PTQ with calibration), B4 cells 8-9 (Lab 1 md + code). PASS.
- Section 2 (Pruning): no diagram per spec note. B1 cell 14-15 (80% naive collapse), B3 cell 17 (20% conservative), B4 cells 18-19 (Lab 2). Cell 16 is mid-section discussion. PASS (spec-compliant 3-beat).
- Section 3 (Distillation): B1 cells 23-25 (T=1 naive), B2 cell 26 (markdown + diagram), B3 cell 27 (proper T=4 KL loop), B4 cells 28-29 (Lab 3). PASS.
- Section 4 (Capstone QAT+LoRA): capstone pattern (Tier 2 hard lab), not strict four-beat — no broken/naive demo. This is the standard capstone structure; acceptable.
- Section 5 (Serving): deployment-only section, not a teaching arc. Acceptable.

**Diagrams**
- Exactly 2 `<!-- DIAGRAM: -->` placeholders found, both in markdown cells (cell 6 and cell 26).
- Both reference correct full folder path `topic_8_quantization` and the .mmd files exist.

**Safety-nets (exercise)**
- Lab 1 safety-net at cell 10 (`dynamic_quantized_model`).
- Lab 2 safety-net at cell 20 (`global_pruned_model`).
- Lab 3 safety-net at cell 30 (`kl_results`).
- `training_job_name` safety-net at cell 38 (immediately after `.fit(wait=False)` at cell 37).
- All use correct header comments (`# Safety-net: run this if...` + `# SKIP this cell if...`).

**Safety-nets (solution)**
- All four safety-net cells (10, 20, 30, 38) retained in the solution notebook. PASS.
- Solution Lab 1/2/3 cells have complete implementations.
- Solution `compress_model` (cell 51) has full implementation (not `pass`), handles both `target="size"` and `target="latency"`.

**AI-tells**
- Programmatic scan: clean. No em dashes, en dashes, unicode multiplication signs, or bare `---` separators in markdown cells of either notebook.

**Lab tiers**
- Lab 1, 2, 3: Tier 1 Guided, 15 min labels, numbered steps + `= None  # YOUR CODE`. PASS.
- Section 4: Tier 2 Hard Lab (25-35 min) labeled correctly, brief task description with no numbered substeps. PASS.
- Capstone (cell 50-51): Tier 3 Open-Ended labeled correctly. Exercise body is `pass  # YOUR CODE` only — no hints, no numbered steps. PASS.
- Stretch versions present (Lab 1 stretch in cell 8 md, Lab 2 stretch in 18, Lab 3 stretch in 28, Section 4 stretch in 35, dedicated cell 43 stretch markdown).
- Homework Extensions present after Labs 1/2/3 (cells 11-12, 21-22, 31-32).

**YOUR CODE hygiene**
- All `= None  # YOUR CODE` lines in cells 9, 19, 29, 42 inspected: no inline hints leak the answer after the placeholder. Hints are placed in preceding comment lines (e.g. `# Hint: torch.ao.quantization.quantize_dynamic(...)`).
- Tier 3 compress_model (cell 51 exercise) uses `pass  # YOUR CODE` correctly.

**Notebook header (no Day N)**
- Cell 0 title: "Topic 8 - Model Compression: Quantization, Pruning and Distillation". No "Day N" reference. PASS.

**Narrative / Barclays STAR**
- Cell 0 explicitly opens with "Barclays Customer Support Intelligence System".
- Labs 1, 2, 3 (cells 8, 18, 28) and Section 4 (cell 35) all use explicit **Situation / Task / Action / Result** structure.
- Cell 2 setup notes baseline_model is loaded fresh (not from T7b kernel state).

**SageMaker constraints**
- `numpy<2` pinned in install cell 1 and in scripts/requirements.txt.
- No `import evaluate` or `from evaluate` anywhere.
- `eval_strategy="epoch"` used in scripts_topic8/train.py:203 (NOT `evaluation_strategy`).
- `estimator.fit(wait=False)` used in cell 37.
- HuggingFace estimator on `ml.g4dn.xlarge` (cell 37 + cell 42 solution).
- `boto3` exception: `sm_client.exceptions.ResourceNotFound` used in cells 40 and 47 (NOT `ResourceNotFoundException`). Comments explicitly note this.
- No `mlflow` anywhere.
- `datasets==2.18.0` pinned in scripts_topic8/requirements.txt (both ex and sol).
- Endpoint instance `ml.m5.xlarge` (NOT ml.c5.large) in cell 45 with explicit OOM warning.

**Discussion prompts (WARN)**
- Discussion cells found: cell 13 (Peer Discussion 3 min, between Section 1 and 2), cell 16 (mid-Section 2 discussion "How Much Pruning Is Safe"), cell 52 (5-min discussion at end, between Section 5 and Wrap-Up).
- Missing dedicated peer discussion between Section 2 and Section 3, and between Section 3 and Section 4. The Section 2 discussion (cell 16) is placed mid-section rather than between sections.
- All present discussions are markdown cells with Barclays-framed questions.
- WARN: spec asks for "at least one peer discussion markdown cell between every two major sections" — with 5 sections that implies 4 inter-section discussions; only 2 are placed between sections (cell 13 and cell 52). Consider adding a brief discussion at the end of Section 3 (around cell 32-33 boundary) and at the end of Section 4 (around cell 43-44 boundary).

**Cell parity**
- Exercise: 54 cells. Solution: 54 cells. Match. PASS.

**No markdown chain > 3**
- Maximum consecutive markdown run is 2 cells. PASS.

**Solution clean**
- No `= None  # YOUR CODE` remaining in solution lab cells (9, 19, 29, 42, 51).
- Solution cell 51 (compress_model) has full implementation including pruning+quantization branches and accuracy measurement loop.

### Issues to Fix
- **Discussion prompts (WARN)**: Two inter-section discussion cells are missing. Recommended additions:
  - Between Section 3 (Distillation) and Section 4 (Capstone): a short peer discussion comparing PTQ/Pruning/Distillation choices before students commit to the QAT capstone.
  - Between Section 4 (Capstone) and Section 5 (Serving): a short reflection on production tradeoffs after seeing the QAT job launch.

No FAIL items identified. Notebook pair is production-ready for class delivery; the WARN above is a content-enrichment suggestion, not a blocker.
