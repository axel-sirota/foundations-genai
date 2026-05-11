# Verify Report: Topic 3a Attention Python
Generated: 2026-05-11

## Status: FAIL — needs fix

### Issues to Fix

#### 1. Section 3 missing Beat 2 (FAIL)
- Section 3 (Dot Product Attention): Beat 1 broken code (cell 19) jumps directly to Beat 3 working demo (cell 20) with no Beat 2 between them.
- Fix: Insert a Beat 2 markdown cell between cells 19 and 20 with a brief visual/textual anchor explaining the dot product attention mechanism before the demo. Do NOT add a third <!-- DIAGRAM: --> placeholder (notebook already has exactly 2). Use prose explanation only.
- Insert in BOTH exercise and solution notebooks.

#### 2. Section 4 missing Beat 2 (FAIL)
- Section 4 (Scaled Dot Product Attention): Beat 1 broken code (cell 22) jumps directly to Beat 3 working demo (cell 23) with no Beat 2.
- Fix: Insert a Beat 2 markdown cell between cells 22 and 23 explaining the sqrt(d_k) scaling fix before the demo. No new diagram placeholder — prose only.
- Insert in BOTH exercise and solution notebooks.

#### 3. Safety-net placement (MINOR — fix if easy)
- Safety-net cell is at cell 16, but lab starter is at cell 14 with verification cell 15 between them.
- Strictly CLAUDE.md says safety-net immediately after lab starter. However functionally safe.
- Fix only if it does not require deleting/moving cells. If it requires deletion, skip this fix.

### Checks Summary
| Check | Result |
|-------|--------|
| AI-tells | PASS |
| Four-beat arc Sec 1 | PASS |
| Four-beat arc Sec 2 | PASS |
| Four-beat arc Sec 3 | FAIL — missing Beat 2 |
| Four-beat arc Sec 4 | FAIL — missing Beat 2 |
| Safety-nets | PASS (functionally) |
| Safety-nets (solution) | KEEP — do not remove |
| Diagrams count (2) | PASS |
| Diagrams embedded | PASS |
| Lab tiers (Tier 1 only) | PASS |
| numpy<2 | PASS |
| No evaluate | PASS |
| Narrative (Barclays) | PASS |
| Cell parity | PASS (31/31) |
| Solution clean | PASS |

### MANDATORY Before Every Cell Insert or Edit
1. Read the full notebook to get current cell list and IDs
2. Identify the cell BEFORE and AFTER the insertion point by their cell_ids
3. Only then call NotebookEdit with the confirmed cell_id (insert after that ID)
4. Never assume cell ordering from a previous read — re-read if in doubt

### Notes
- Safety-net cells must NOT be removed from solution notebook — keep them.
- No day number in notebook header — remove any "Day 1" or "Day 2" references in title cells.
- Keep exactly 2 diagram placeholders total after fix.
