# Verify Report: Topic 3b Attention PyTorch
Generated: 2026-05-11

## Status: FAIL — needs fix

### Issues to Fix

#### 1. Beat 2 appears AFTER Beat 3 in Section 1 (FAIL)
- Section 1 (Dot Product Attention): order is Beat 1 (cell 4) → Beat 3 (cell 6) → Beat 2/diagram (cell 7) → Beat 4/lab (cell 8).
- This is B1→B3→B2→B4 — diagram appears AFTER the working demo, violating mandatory B1→B2→B3→B4.
- Fix: Move the diagram cell (cell 7) to appear BEFORE the Beat 3 demo cell (cell 6). Do this by replacing cell 6's content with the diagram content, and cell 7's content with the demo content (effectively swapping them). Read cell IDs carefully before editing.
- Fix in BOTH exercise and solution notebooks.

#### 2. Beat 2 appears AFTER Beat 3 in Section 2 (FAIL)
- Section 2 (Scaled Dot Product): same violation — Beat 3 demo (cell 15) appears before Beat 2 diagram (cell 16).
- Fix: Swap cell 15 and cell 16 content so diagram comes before the demo.
- Fix in BOTH exercise and solution notebooks.

### Checks Summary
| Check | Result |
|-------|--------|
| AI-tells | PASS |
| Four-beat arc Sec 1 | FAIL — B1→B3→B2→B4 (diagram after demo) |
| Four-beat arc Sec 2 | FAIL — B1→B3→B2→B4 (diagram after demo) |
| Safety-nets | PASS |
| Safety-nets (solution) | KEEP — do not remove |
| Diagrams count (2) | PASS |
| Diagrams embedded | PASS |
| Lab tiers (Tier 1 + Tier 3) | PASS — Tier 3 capstone is correct for last topic |
| Discussion prompts (2) | PASS |
| numpy<2 | PASS |
| No evaluate | PASS |
| Cell parity | PASS (29/29) |
| Solution clean | PASS |

### MANDATORY Before Every Cell Insert or Edit
1. Read the full notebook to get current cell list and IDs
2. Identify the cell BEFORE and AFTER the swap/edit point by their cell_ids
3. Only then call NotebookEdit with the confirmed cell_id
4. Never assume cell ordering from a previous read — re-read if in doubt

### Notes
- Safety-net cells must NOT be removed from solution notebook — keep them.
- No day number in notebook header — remove any "Day 1" or "Day 2" references in title cells.
- Tier 3 capstone is intentional and correct — do NOT change it.
- The fix is a content swap, NOT a cell deletion/insertion.
