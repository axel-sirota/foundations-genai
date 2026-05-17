# /audit-notebook — Audit a Built Exercise + Solution Notebook Pair

Audits built `.ipynb` notebooks for correctness before delivery.
This is a READ-ONLY command — it reports issues, never edits.

## Usage

```
/audit-notebook topic_4_transformers
/audit-notebook topic_6a_full_finetuning
/audit-notebook F1
```

Resolves paths automatically:
- Exercise: `Exercises/topic_N_slug/topic_N_slug.ipynb`
- Solution: `Solutions/topic_N_slug/topic_N_slug.ipynb`
- Frameworks: `Frameworks/pytorch_refresher.ipynb` + `Frameworks/pytorch_refresher_solution.ipynb`

---

## GUARD: Read Core Decisions First

Before auditing, always read:
- `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`
- `plans/SAGEMAKER_LESSONS_LEARNED.md`

---

## Complete Checklist (check every item — read actual cell source, not just structure)

### 1. Four-Beat Arc (B1→B2→B3→B4, mandatory per section)
For every major section, read the actual cells and verify:
- **Beat 1**: code cell that runs and demonstrates the broken/naive approach
- **Beat 2**: standalone markdown cell with `<!-- DIAGRAM: -->` placeholder — AFTER B1, BEFORE B3
- **Beat 3**: full working demo code cell, heavily commented
- **Beat 4**: lab markdown cell + lab starter code cell

Report section name + cell IDs for any missing beat or wrong order.

### 2. Diagrams
- Exactly **2** `<!-- DIAGRAM: -->` placeholders per notebook — no more, no less
- Each must be in a **markdown cell** (grep for `<!-- DIAGRAM:` in code cells — that's a bug)
- Each must appear **before** the Beat 3 demo it introduces
- Format: `<!-- DIAGRAM: description -->` followed by `[View diagram](../../plans/topic_N_slug/diagrams/slug.mmd)`
- Verify the referenced `.mmd` file **exists on disk**
- Paths must use **full folder names** (e.g. `topic_7a_lora_ffn`, NOT `topic_7a`)

### 3. Safety-nets (exercise notebook)
- Every lab variable that feeds a downstream cell must have a safety-net **immediately after** the lab starter cell
- Format: `if variable is None:` fallback
- `training_job_name` safety-net **immediately after** every `.fit(wait=False)` cell
- Safety-net cells must use: `# Safety-net: run this if...` + `# SKIP this cell if...` comment header

### 4. Safety-nets (solution notebook)
- Safety-net cells must **remain** in the solution notebook — do NOT remove them
- Solution lab cells have complete implementations (not `pass`, not `None`)

### 5. AI-tells (zero tolerance — run programmatic scan)
Scan every cell source for:
- Em dash (— U+2014)
- En dash (– U+2013)
- Unicode multiplication sign (× U+00D7)
- Emojis
- Bare `---` separator lines in markdown cells

Use this scan:
```python
import json, sys
path = sys.argv[1]
with open(path) as f:
    nb = json.load(f)
hits = []
banned = [('—','em dash'), ('–','en dash'), ('×','unicode mul')]
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    for char, name in banned:
        if char in src:
            line = next((l for l in src.splitlines() if char in l), '')
            hits.append(f'Cell {i} ({cell["cell_type"]}): {name} -> {line.strip()[:80]}')
    if cell['cell_type'] == 'markdown':
        for j, line in enumerate(src.splitlines()):
            if line.strip() == '---':
                hits.append(f'Cell {i} (markdown): bare --- on line {j}')
if hits:
    for h in hits: print(h)
else:
    print('clean')
```

### 6. Lab Tiers
- Check every lab header cell for tier label
- Verify the tier is correct for this topic's position in the day:
  - **Tier 1**: has numbered steps + `= None  # YOUR CODE` stubs + verification cell
  - **Tier 2**: brief task description, no numbered sub-steps, YOUR CODE stubs, 25-35 min label
  - **Tier 3**: function signature + docstring + `pass` only — NO numbered steps, NO hints — last topic of day only
- Exactly ONE Tier 2 across the whole day, ONE Tier 3 in last topic only
- Every lab has a stretch version and a Homework Extension cell after it

### 7. YOUR CODE hygiene
- Check every `# YOUR CODE` line: no inline hints after it
- Correct: `variable = None  # YOUR CODE`
- Wrong: `variable = None  # YOUR CODE: use torch.softmax(...)` — flag these

### 8. Notebook header
- **No "Day N"** references in the title/header cell (Cell 0 or Cell 1)
- No "Day 1", "Day 2", "Day 3" anywhere in the header

### 9. Narrative continuity (Barclays STAR)
- All lab markdown cells reference **Barclays Customer Support Intelligence System**
- STAR structure: Situation → Task → Action → Result
- Check that variable names from the **prior topic** are referenced correctly (spot-check the setup cell)

### 10. SageMaker constraints (topics using SageMaker)
- `numpy<2` in every install cell
- **No `import evaluate`** or `from evaluate import` anywhere
- `eval_strategy=` (NOT `evaluation_strategy=`) in any TrainingArguments
- `wait=False` on every `.fit(` call
- HuggingFace estimator on `ml.g4dn.xlarge` (GPU)
- PyTorch estimator on `ml.m5.xlarge` (CPU)
- `boto3` exception: `ResourceNotFound` (NOT `ResourceNotFoundException`)
- `mlflow==2.13.2` only in F2/sagemaker_fundamentals

### 11. Discussion prompts
- At least one peer discussion markdown cell between every two major sections
- Must be a **markdown cell** — not a code cell with print()
- Contains Barclays-framed questions

### 12. Cell parity
- Count cells in exercise notebook
- Count cells in solution notebook
- They must match exactly

### 13. No markdown chain > 3
- No more than 3 consecutive markdown cells without a code cell between them

### 14. Solution cleanliness
- No `= None  # YOUR CODE` remaining in solution lab cells
- No `pass` in solution lab cells (except Tier 3 — which should have full implementation in solution)
- Solution cells have working code with explanation comments

---

## Output Format

```
# Notebook Audit: [topic slug]
Generated: [date]

## Status: PASS / FAIL

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | PASS/FAIL/WARN |
| Diagrams (2, markdown, Beat 2 position) | PASS/FAIL/WARN |
| Safety-nets (exercise) | PASS/FAIL/WARN |
| Safety-nets (solution kept) | PASS/FAIL/WARN |
| AI-tells | PASS/FAIL/WARN |
| Lab tiers | PASS/FAIL/WARN |
| YOUR CODE hygiene | PASS/FAIL/WARN |
| Notebook header (no Day N) | PASS/FAIL/WARN |
| Narrative / Barclays STAR | PASS/FAIL/WARN |
| SageMaker constraints | PASS/FAIL/WARN |
| Discussion prompts | PASS/FAIL/WARN |
| Cell parity (ex=N, sol=N) | PASS/FAIL |
| No markdown chain > 3 | PASS/FAIL/WARN |
| Solution clean | PASS/FAIL/WARN |

### Issues to Fix
[Every FAIL and WARN with cell ID and exact fix needed]

### MANDATORY Before Every Cell Edit (when fixing after audit)
1. Read the full notebook to get current cell list and IDs
2. Identify the cell BEFORE and AFTER the insertion/edit point by cell_id
3. Only then call NotebookEdit with the confirmed cell_id
4. Never assume cell ordering from a previous read — re-read if in doubt
```

Save the report to `plans/VERIFY_[TOPIC].md`.

---

## Non-Negotiables
- Read actual cell source — never guess from structure alone
- Run the AI-tells scan programmatically on both notebooks
- Report every failure with the specific cell ID
- Do NOT auto-fix anything — this command is read-only
- NEVER delete a cell — fixing means replace content or insert, never delete

---

## Notebook Edit Protocol (awareness)

If this skill ends up editing notebook cells (not just reading them), follow
the canonical procedure in `~/.claude/NOTEBOOK_EDIT_PROTOCOL.md`: normalize
cell ids, size-gate the mechanism, locate cells by id + content, read back and
assert after every edit, and run the structural + static code gates. Blind
bulk index-based rewrites are forbidden.
