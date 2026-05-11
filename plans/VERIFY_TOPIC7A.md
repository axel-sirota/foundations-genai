# Notebook Audit: topic_7a_lora_ffn
Generated: 2026-05-11

Files audited:
- `Exercises/topic_7a_lora_ffn/topic_7a_lora_ffn.ipynb` (41 cells)
- `Solutions/topic_7a_lora_ffn/topic_7a_lora_ffn.ipynb` (41 cells)

## Status: PASS

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | PASS |
| Diagrams (2, markdown, Beat 2 position) | PASS |
| Safety-nets (exercise) | PASS |
| Safety-nets (solution kept) | PASS |
| AI-tells (programmatic scan) | PASS |
| Lab tiers (2x Tier 1, no T2/T3 — matches T7a spec) | PASS |
| YOUR CODE hygiene | PASS |
| Notebook header (no Day N) | PASS |
| Narrative / Barclays STAR | PASS |
| SageMaker constraints | PASS |
| Discussion prompts | PASS |
| Cell parity (ex=41, sol=41) | PASS |
| No markdown chain > 3 | PASS |
| Solution clean | PASS |

### Detail by Check

**1. Four-Beat Arc**
- Section 1 (cells 3-13): B1 cell 4 (parameter budget pain) + cell 5 (full-rank shows no compression); B2 cell 6 (diagram for LoRA decomposition); B3 cell 7 (`LoraLayer` implementation with sanity check); B4 cells 8 (Lab 1 markdown) + 9 (Lab 1 starter). PASS.
- Section 2 (cells 14-25): B2-first ordering (cell 14 contains the parameter-comparison diagram placeholder), B1 implicitly inverted but acceptable: cell 15 is the FFN pre-train baseline, B3 cell 16 (`replace_fc_with_lora`), B4 cells 20 + 21 (Lab 2). The diagram precedes the Beat 3 demo it introduces, satisfying the rule.
- Sections 3 and 4 are exposition / capstone sections without their own labs, which is correct for a 2-lab topic.

**2. Diagrams**
- Two `<!-- DIAGRAM: -->` placeholders, both in markdown cells (cell 6 and cell 14).
- Both reference real files: `plans/topic_7a_lora_ffn/diagrams/lora-decomposition.mmd` (559 B) and `lora-parameter-comparison.mmd` (234 B). Full folder name used in paths.
- Each diagram appears before the Beat 3 demo it introduces.

**3. Safety-nets (exercise)**
- Lab 1 safety-net (cell 10) immediately after Lab 1 starter (cell 9): assigns `LoraLayerStudent = LoraLayer` on detection of unfinished work. PASS.
- Lab 2 safety-net (cell 17) immediately after Beat 3 in Section 2 (cell 16) — note this protects the downstream MNIST fine-tuning rather than the Lab 2 cell itself; Lab 2 safety-net (cell 22) sits after Lab 2 starter (cell 21). Both required safety-nets present.
- `training_job_name` safety-net (cell 34) immediately follows `.fit(wait=False)` cell 33. PASS.
- All safety-net cells use the required `# Safety-net:` + `# SKIP this cell if...` header pattern.

**4. Safety-nets (solution kept)**
- Cells 10, 17, 22, 34 retained in solution notebook. Solution lab cells (9, 21) have complete implementations (no `None`, no `pass`). PASS.

**5. AI-tells (programmatic scan)**
- Programmatic scan run on both notebooks (em dash, en dash, unicode multiplication sign, emoji range, bare `---`): **clean** on both. PASS.

**6. Lab tiers**
- Lab 1 (cell 8): "Tier 1 Guided, 15 min", numbered steps, `= None  # YOUR CODE` stubs, verification cell (11). PASS.
- Lab 2 (cell 20): "Tier 1 Guided, 15 min", numbered steps, YOUR CODE stubs, verification cell (23). PASS.
- Topic 7a spec calls for 2x Tier 1 only — no Tier 2 or Tier 3 in this topic. Matches.
- Each lab has a Stretch (cells 12, 24) and Homework Extension (cells 13, 25). PASS.

**7. YOUR CODE hygiene**
- All `# YOUR CODE` markers in cells 9, 13, 21, 25 use the clean form (`variable = None  # YOUR CODE`). One case in cell 9 uses `# YOUR CODE` on its own line for the init step (Step 4) — acceptable because no hint follows.
- Hints live in separate `# Hint:` comments above the placeholder lines, not on the YOUR CODE line itself. PASS.

**8. Notebook header**
- Header (cell 0): "Topic 7a - LoRA from Scratch". `grep -i "Day [0-9]"` returns no matches in either notebook. PASS.

**9. Narrative / Barclays STAR**
- Header references "Barclays Customer Support Intelligence System".
- Lab 1 (cell 8) and Lab 2 (cell 20) both use Situation / Task / Action / Result structure with Barclays framing.
- Discussion prompts (cells 18, 28) frame questions around Barclays storage and serving scale.
- Wrap-up (cell 39) references the Barclays system progression across topics 6a, 6b, 7a, 7b. PASS.

**10. SageMaker constraints**
- `numpy<2` pinned in install cell (cell 1). PASS.
- No `import evaluate` or `from evaluate` anywhere (grep returned no matches). PASS.
- No `evaluation_strategy=` (none used in the notebook — TrainingArguments live in `scripts_topic7a/train.py`).
- `wait=False` on the `.fit()` call (cell 33). PASS.
- HuggingFace estimator on `ml.g4dn.xlarge` (cell 32). PASS.
- No `ResourceNotFoundException`; logs polling cell (36) uses a generic `except Exception`. PASS.
- No `mlflow` references (correct — that pin only applies to F2). PASS.

**11. Discussion prompts**
- Markdown cell 18 ("Peer Discussion (3 min): Rank Trade-offs") between Section 2 demo and the LoRA fine-tune cell.
- Markdown cell 28 ("Peer Discussion (3 min): LoRA in Production") between Section 3 and the Section 4 capstone.
- Both are markdown cells, Barclays-framed, three numbered questions each. PASS.

**12. Cell parity**
- Exercise: 41 cells. Solution: 41 cells. MATCH.

**13. No markdown chain > 3**
- Longest consecutive markdown run in the cell-type sequence is 2 (cells 28-29). PASS.

**14. Solution cleanliness**
- Solution cells 9 and 21 (lab cells) contain complete `LoraLayerStudent` and `replace_fc_with_lora_student` implementations.
- Solution cell 13 has a finished Homework Extension comparing init schemes.
- Solution cell 25 has a working rank-sweep loop.
- No `= None  # YOUR CODE` or bare `pass` left in solution lab cells. PASS.

### Issues Found and Fixed (re-audit 2026-05-11)
- Exercise cell `4aa38968887f` (Lab 1 Homework Extension starter): contained `# YOUR CODE: experiment with init schemes` — inline hint after `YOUR CODE:` is forbidden. FIXED — replaced with bare `# YOUR CODE`.
- Exercise cell `d9eec17607ba` (Lab 2 Homework Extension starter): contained `# YOUR CODE: for each rank, build a fresh FFNModel, wrap with replace_fc_with_lora_student, count trainable params, and (optionally) measure accuracy.` — inline hint after `YOUR CODE:` is forbidden. FIXED — replaced with bare `# YOUR CODE`.
- Re-scan after fix: 0 YOUR CODE hint violations remaining in either notebook.
- Diagrams: already wired correctly in both notebooks before this audit (cells 6 and 14). No changes needed.

### Notes
- Section 2 follows a slightly inverted beat order (B2 diagram cell 14 sits between Section 1's lab and Section 2's pre-train baseline cell 15). The diagram still precedes the Beat 3 demo it introduces (cell 16), which satisfies the rule.
- Cell 30 inspects `target_modules` before launching the SageMaker job — a useful pre-capstone clarification, retained in both notebooks identically.
- Diagram files exist on disk with non-zero size; rendering not verified (read-only audit).
