# HANDOFF: day2-bigger-picture-fixes

## Goal
Fix Day 2 notebooks so students can see the bigger-picture system they're building and fix the Tier 3 misplacement.

## State
- Completed: 5-cycle research + code verification of all Day 2 notebooks (T4-T7b)
- In-flight: Implementing 5 ranked fixes in sequential subagents
- Remaining: All 5 fixes still to implement

## Confirmed Issues (code-verified)

### P0 — T6b Tier 3 misplaced + T7b missing Tier 3
- T6b Lab 6b is labeled "Tier 3 - Open-Ended" (Cell 29) but T6b is topic 4/6 of Day 2
- T7a and T7b come after T6b — rule says Tier 3 must be LAST topic of day only
- T7b has NO Tier 3 anywhere — only Tier 1 and Tier 2 labs
- Fix: relabel T6b Lab 6b → Tier 1 or Tier 2; add Tier 3 open-ended capstone to T7b
- Files: Exercises/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb (Cell 29)
          Solutions/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb (Cell 29)
          Exercises/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb (last cells)
          Solutions/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb (last cells)

### P1 — T4 wrap-up doesn't connect transformer architecture to T5-T7 models
- T4 wrap-up (last markdown cell) says "use this estimator pattern in Topics 6-9"
- Never says "distilbert-base-uncased IS the 6-encoder-block architecture you built today"
- Fix: one paragraph in T4's wrap-up cell connecting the built architecture to HF models
- Files: Exercises/topic_4_transformers/topic_4_transformers.ipynb (Cell 33, last markdown)
          Solutions/topic_4_transformers/topic_4_transformers.ipynb (Cell 33)

### P1 — No "you are here" progress table in any Day 2 header
- All notebooks pass system narrative test but none have a position marker
- Fix: add small markdown table to each Day 2 notebook header (cell after title)
- Files: T4, T5, T6a, T6b, T7a, T7b exercise AND solution notebooks (12 files total)

### P2 — T7b missing Day 2 "complete" closing moment
- T7b ends with wrap-up bullets; no closure that Day 2 is done
- Fix: add one markdown cell after T7b wrap-up showing assembled system + Day 3 preview
- Files: T7b exercise + solution

### P2 — T4, T6a, T7a missing explicit "Next session:" lines
- T5 and T6b have them; T4, T6a, T7a do not
- Fix: add one-line "Next session: ..." to each wrap-up
- Files: T4, T6a, T7a exercise + solution

## Implementation order
1. Fix T4 wrap-up (connect architecture to HF models) — T4 exercise + solution
2. Add "you are here" tables to all Day 2 headers — T4-T7b exercise + solution (12 files)
3. Fix T6b Tier 3 mislabeling — relabel Lab 6b to Tier 2, update exercise + solution
4. Add Tier 3 capstone to T7b + Day 2 closing cell — T7b exercise + solution
5. Add "Next session:" lines to T4, T6a, T7a — 6 files

## Key constraints
- ALLOW_DIRECT_MAIN=1 — commit directly to master, no feature branches
- 5-cell approval cadence NOT needed here (these are targeted surgical edits, not builds)
- No em dashes, en dashes, unicode chars, emojis in cell bodies
- Always read notebook with python json.load to get exact cell indices before editing
- Solutions must mirror every change made to exercises
- NotebookEdit requires cell_id (the cell to insert AFTER) for all insertions except cell 0

## Failed approaches
- None yet — research only so far

## Open questions
- None — all decisions made

## Next concrete step
Launch subagent 1: fix T4 wrap-up cell (Cell 33) in exercise + solution notebooks
