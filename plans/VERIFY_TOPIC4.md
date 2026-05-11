# Verify Report: Topic 4 Transformers
Generated: 2026-05-11

## Status: FAIL — needs fix

### Issues to Fix

#### 1. Section 1 missing Beat 3 and Beat 4 (FAIL)
- Section 1 (Why Transformers?): Has Beat 1 (cell 6, RNN vanishing gradient) and Beat 2 (cell 7, diagram) but then jumps directly to Section 2 header (cell 8). No Beat 3 working demo and no Beat 4 lab.
- Fix: Insert a Beat 3 markdown+code cell after cell 7 showing a working self-attention demo (parallel processing, no hidden state bottleneck). Keep it brief — 1 code cell showing torch.nn.MultiheadAttention or a minimal attention pattern. Then insert a Beat 4 markdown cell as a brief guided observation (not a full lab — just 2-3 questions for students to observe from the Beat 3 output). No new diagram placeholder needed.
- Insert in BOTH exercise and solution notebooks.

#### 2. Section 3 missing Beat 1 and Beat 2 (FAIL)
- Section 3 (Building Transformer): jumps straight to two Beat 3 demo cells (cells 17, 18) with no Beat 1 broken code and no Beat 2 diagram.
- Fix: Insert a Beat 1 cell before cell 17 showing a naive/broken approach (e.g. trying to build a translator without positional encoding or without multi-head attention — showing degraded output). Then insert a Beat 2 markdown cell with a brief textual anchor (no new diagram placeholder — notebook already has exactly 2).
- Insert in BOTH exercise and solution notebooks.

#### 3. Notebook header says "Day 2" — remove day reference (FAIL)
- The title/header cell says "Day 2, Topic 4". Remove the "Day 2," part. Do not add "Day 1" either. Just say "Topic 4 - Transformers + Translator Capstone" or similar.
- Fix in BOTH exercise and solution notebooks.

#### 4. No Tier 3 lab (FAIL)
- Topic 4 is the last topic of Day 1. Must have exactly ONE Tier 3 lab (function signature + docstring + pass only, no numbered steps, no YOUR CODE hints beyond the signature).
- Fix: Find the most open-ended existing lab (likely Lab 2 or the capstone) and upgrade it to Tier 3: remove numbered sub-steps, keep only function signature + docstring + pass.
- In the solution notebook, the Tier 3 cell should have a complete implementation (not pass).
- Fix in BOTH exercise and solution notebooks.

#### 5. training_job_name safety-net missing (FAIL)
- After the .fit() call (cell 26), there is no safety-net for kernel restart recovery.
- Fix: Insert a safety-net cell immediately after cell 26:
  ```python
  # Safety-net: run this if your kernel restarted after launching the training job.
  # SKIP this cell if training_job_name is already defined.
  if 'training_job_name' not in dir() or training_job_name is None:
      training_job_name = "<PASTE YOUR JOB NAME HERE>"
      print(f"Using safety-net training_job_name: {training_job_name}")
  ```
- Insert in BOTH exercise and solution notebooks.

### Checks Summary
| Check | Result |
|-------|--------|
| AI-tells | PASS |
| Four-beat arc Sec 1 | FAIL — missing Beat 3 and Beat 4 |
| Four-beat arc Sec 2 | PASS |
| Four-beat arc Sec 3 | FAIL — missing Beat 1 and Beat 2 |
| Safety-nets (labs) | PASS |
| Safety-nets (solution) | KEEP — do not remove |
| training_job_name safety-net | FAIL — missing |
| Diagrams count (2) | PASS |
| Diagrams embedded | PASS |
| Lab tiers (Tier 3 required) | FAIL — no Tier 3 found |
| Notebook header day ref | FAIL — says "Day 2" |
| SageMaker constraints | PASS |
| numpy<2 | PASS |
| No evaluate | PASS |
| wait=False on .fit() | PASS |
| Cell parity | PASS (29/29) |
| Solution clean | PASS |

### MANDATORY Before Every Cell Insert or Edit
1. Read the full notebook to get current cell list and IDs
2. Identify the cell BEFORE and AFTER the insertion point by their cell_ids
3. Only then call NotebookEdit with the confirmed cell_id (insert after that ID)
4. Never assume cell ordering from a previous read — re-read if in doubt

### Notes
- Safety-net cells must NOT be removed from solution notebook — keep them.
- Keep exactly 2 diagram placeholders total — do NOT add new <!-- DIAGRAM: --> placeholders.
- Topic 4 IS Day 1 (last topic) — Tier 3 lab is required.
- Do not add "Day 1" to the header either — just remove the day reference entirely.
