---
description: Validate a topic research plan file against all course standards
---

Verify research plan for topic: $ARGUMENTS

## GUARD: Read Course Manifest First

Before doing anything else, check for `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`:

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```

If the file does NOT exist, stop immediately and say:

> `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` not found. Run `/init-course` first to record the course decisions. Every command requires this file before proceeding.

If it exists, read the full file and keep its contents in mind throughout this command.

---

## What This Command Checks

Given a topic number N, locate `plans/topic_<N>_<slug>.md` and verify it is ready to be
consumed by `/build-topic-notebook`. Every check either passes or fails with a specific
line reference.

---

## Step 1: Locate the Plan File

```bash
ls plans/topic_<N>_*.md 2>/dev/null
```

If no file found, stop and say:

> No plan file found for Topic N. Run `/run-research-topic N` first.

Read the full file.

---

## Step 2: Run All Checks

### Check 1 - File path and slug

- Path must be exactly `plans/topic_<N>_<slug>.md` - never in a subfolder
- Slug must be snake_case and match the entry in `plans/TOPICS.md`
- Read `plans/TOPICS.md` and confirm the slug matches

### Check 2 - Required sections (all must be present)

- [ ] `# Topic N: <Title> - Cell-by-Cell Plan`
- [ ] `## Context`
- [ ] `## Deliverables`
- [ ] `## Session Timing`
- [ ] `# MAIN NOTEBOOK - Cell-by-Cell Content`
- [ ] `# VERIFICATION CHECKLIST`
- [ ] `# RESEARCH VALIDATED`

### Check 3 - Deliverables paths

Both must appear and use the correct slug:
- `exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb`
- `solutions/topic_<N>_<slug>/topic_<N>_<slug>.ipynb`

### Check 4 - Cell count and completeness

- Count cells: `## Cell N - ` headers. Target is 20-25 cells.
- Every cell must have a fenced code block (` ```markdown ` or ` ```python `) with actual
  content - not a placeholder like "add content here" or "TODO"
- Check that Cell 0 has the topic title with emoji and learning objectives
- Check that Cell 2 has a `!pip install` line with `numpy<2`
- Check that Cell 3 has `getpass` and `os.environ` for API keys

### Check 5 - Four-beat arc for every concept

For each concept section, verify all four beats are present:
- Beat 1: a Markdown cell with problem intro + a Code cell with naive/broken code
- Beat 2: a Markdown cell with `<!-- DIAGRAM: ...-->` placeholder
- Beat 3: a Code cell with full working demo
- Beat 4: a Markdown lab instructions cell + a Code lab starter cell

### Check 6 - Diagram index

In the Context section, every diagram must have all three fields:
- `Diagram slug: <slug>.mmd`
- `Diagram path: plans/topic_<N>_<slug>/diagrams/<slug>.mmd`
- `Description: <description>`

Every `<!-- DIAGRAM: -->` in the notebook cells must have a corresponding diagram index
entry. Count them and confirm they match.

### Check 7 - Lab tiers

- Confirm at least one Tier 1 guided lab (has `None  # YOUR CODE` with numbered step
  comments in the plan)
- Confirm lab tier assignment is noted in the plan (Tier 1 / Tier 2 / Tier 3)
- Confirm Day assignment is correct (Day 1: topics 1-3, Day 2: topics 4-6, Day 3: topics 7-9)
- Tier 3 only on last topic of each day (topic 3, 6, 9)
- Tier 2 (one per day) noted clearly

### Check 8 - Stretch labs

Every in-class lab must have a labeled stretch version in the plan. If any lab is missing
its stretch, list which one.

### Check 9 - Safety-net cells

For every lab whose output variable feeds a later cell, the plan must note a safety-net
cell. Check the `Lab safety-net cells` entry in the Context / Key decisions section.

### Check 10 - AI-tells scan

Run this on the raw plan file content:

```python
import re

with open('plans/topic_N_slug.md') as f:
    content = f.read()

hits = []
banned = [('--', 'em dash'), (chr(8211), 'en dash'), (chr(215), 'unicode multiplication')]
for char, name in banned:
    for i, line in enumerate(content.splitlines(), 1):
        if char in line:
            hits.append(f'Line {i} ({name}): {line.strip()[:80]}')

# bare --- separator (not in code fences)
in_fence = False
for i, line in enumerate(content.splitlines(), 1):
    if line.strip().startswith('```'):
        in_fence = not in_fence
    if not in_fence and line.strip() == '---':
        hits.append(f'Line {i}: bare --- separator')

print('AI-tells FOUND:' if hits else 'AI-tells: CLEAN')
for h in hits:
    print(' ', h)
```

Replace `N_slug` with the actual filename.

### Check 11 - OpenAI-only alignment

- No `anthropic` imports in any code cell plan content
- Model must be `gpt-4o` (not claude-*, not gpt-3.5)
- No Anthropic SDK patterns

### Check 12 - No branding

- No mention of "Datatrainers", "Pluralsight", or any external training company in cell
  content (mentions of "Barclays" are fine)

### Check 13 - RESEARCH VALIDATED block

- Must have at least 3 source URLs
- Each entry must have a URL and a specific fact extracted (not just a URL alone)

### Check 14 - TOPICS.md status

Read `plans/TOPICS.md` and confirm:
- Topic N status is `planned` or higher
- The research step checkbox is marked `[x]`

---

## Step 3: Report

Print a table:

```
Topic N Plan Verification Report
=================================
File: plans/topic_N_slug.md

Check                          | Result
-------------------------------|--------
File path and slug             | PASS / FAIL: <reason>
Required sections              | PASS / FAIL: missing <X>
Deliverables paths             | PASS / FAIL
Cell count (target 20-25)      | PASS (22 cells) / FAIL (N cells)
Cell content completeness      | PASS / FAIL: Cell X has placeholder
Four-beat arc                  | PASS / FAIL: Concept Y missing Beat Z
Diagram index                  | PASS / FAIL: N diagrams indexed, N found in cells
Lab tiers                      | PASS / FAIL
Stretch labs                   | PASS / FAIL: Lab X missing stretch
Safety-net cells               | PASS / FAIL
AI-tells                       | CLEAN / FAIL: <details>
OpenAI-only alignment          | PASS / FAIL
No branding                    | PASS / FAIL
RESEARCH VALIDATED block       | PASS / FAIL
TOPICS.md status               | PASS / FAIL
```

Then one of:

> All checks passed. Plan is ready for `/build-topic-notebook N`.

Or:

> N checks failed. Fix the issues above before running `/build-topic-notebook N`.
> Use `/fixes` to log and resolve any notebook issues once they are built.
