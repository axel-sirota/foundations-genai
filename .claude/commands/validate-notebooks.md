# Validate Notebooks

Run comprehensive validation on exercise and/or solution notebooks.

## What This Command Checks

### Exercise notebooks
- All lab cells have `= None  # YOUR CODE` placeholders (Tier 1 and Tier 2 labs)
- Tier 3 (open-ended) labs have a function signature + docstring only - no `None` placeholders
- Every lab has a stretch version labeled clearly in the markdown cell above it
- Safety-net cells present for every lab whose output feeds a later cell
- Four-beat arc present for every concept: problem intro + naive code, diagram placeholder, full demo, lab
- `<!-- DIAGRAM: -->` placeholders present in markdown cells
- No solutions leaked in `# YOUR CODE` comments
- No branding (no Datatrainers, no Pluralsight)
- First-person instructor voice (spot-check markdown cells for "In this section..." - flag if found)

### Solution notebooks
- No `= None  # YOUR CODE` remaining in any lab cell
- Solution cells have complete implementations with explanation comments
- Safety-net cells are REMOVED (they exist in exercise, not solution)
- Solution is structurally a copy of the exercise with lab cells filled in
- Cell count matches exercise (minus removed safety-net cells)

### Both notebooks
- Cell order is correct (cell IDs chain correctly)
- No AI-tells (em dash, en dash, Unicode x, bare --- in markdown)
- numpy<2 pinned in install cell
- getpass + os.environ for API keys (no hardcoded keys)
- gpt-4o model used (no Anthropic/Claude model IDs)
- No torch import unless explicitly needed

## CRITICAL: Cell Order Verification

Before running the script, also verify cell ordering manually by reading the notebook JSON and
confirming each cell's `id` field matches the sequence expected. When using NotebookEdit to
edit an existing cell, ALWAYS fetch the current cell `id` from the notebook first - never assume
an id. Steps:

1. Read the notebook file
2. Print each cell's `id` and its first line of source to confirm order
3. Only then call NotebookEdit with the verified `cell_id`

If cell order looks wrong after an edit, re-read the notebook to confirm actual order before
making further edits.

## Instructions

Run the comprehensive notebook validation script to verify:
- Python syntax correctness in all code cells
- Exercise notebooks have proper `= None  # YOUR CODE` placeholders in lab cells
- Solution notebooks have no remaining `= None` in solution cells
- Paired notebooks have matching cell count and cell types

## Usage Patterns

### After each 5-cell batch (during /build-topic-notebook)
```bash
python validate_notebooks.py exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb --type exercise
```

### After completing exercise notebook
```bash
python validate_notebooks.py exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb --type exercise
```

### After completing solution notebook
```bash
python validate_notebooks.py solutions/topic_<N>_<slug>/topic_<N>_<slug>.ipynb --type solution
```

### Final pair check
```bash
python validate_notebooks.py --pair \
    exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb \
    solutions/topic_<N>_<slug>/topic_<N>_<slug>.ipynb
```

### Generate requirements.txt
```bash
python validate_notebooks.py exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb --requirements
```

## AI-Tells Scan (MANDATORY before marking any notebook done)

Run this grep on every completed notebook to catch forbidden typography in cell source:

```bash
# Em dash (—), en dash (–), Unicode multiplication (×), horizontal rule as separator
python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    nb = json.load(f)
hits = []
banned = [('—', 'em dash'), ('–', 'en dash'), ('×', 'unicode multiplication')]
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    for char, name in banned:
        if char in src:
            line = next((l for l in src.splitlines() if char in l), '')
            hits.append(f'Cell {i} ({cell[\"cell_type\"]}): {name} found -> {line.strip()[:80]}')
    # Check for --- as a markdown horizontal rule (only flag in markdown cells)
    if cell['cell_type'] == 'markdown':
        for j, line in enumerate(src.splitlines()):
            if line.strip() == '---':
                hits.append(f'Cell {i} (markdown): bare --- separator on line {j}')
if hits:
    print('AI-TELLS FOUND:')
    for h in hits: print(' ', h)
    sys.exit(1)
else:
    print('AI-tells scan: clean')
" <notebook_path>
```

Replace `<notebook_path>` with the actual path. Run on both exercise and solution notebooks.

**If hits are found**: use `/fixes` to log and correct them before marking the topic done.

Note: emojis are allowed and will NOT be flagged. Only the banned characters above are checked.

## Pass Criteria

- No syntax errors in code cells
- Exercise: all lab cells have `= None  # YOUR CODE` placeholders
- Solution: no lab cells have `= None` remaining
- Pair: cell count and cell types match
- AI-tells scan: clean (no em dashes, en dashes, Unicode multiplication, bare `---` separators)

## When to Run

- After every 5-cell batch during /build-topic-notebook (mandatory)
- After completing the full exercise notebook
- After completing the full solution notebook
- Before distributing notebooks to students

## See Also

- `validate_notebooks.py` - The validation script in repo root
