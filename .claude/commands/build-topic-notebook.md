---
description: Build exercise and solution notebooks for a specific topic, 5 cells at a time with approval checkpoints
---

Build notebooks for topic: $ARGUMENTS

## GUARD: Read Course Manifest First

Before doing anything else, check for `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`:

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```

If the file does NOT exist, stop immediately and say:

> `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` not found. Run `/init-course` first to record the course decisions. Every command requires this file before proceeding.

If it exists, read the full file. Use it to inform: the capstone scenario woven into labs, the running example dataset, delivery style (live coding vs independent), model names, API key source, and any compliance constraints.

---

**CRITICAL INSTRUCTIONS - READ CAREFULLY BEFORE STARTING**

## Command Arguments

This command takes ONE argument: the topic number (e.g., `4`)

Example usage: `/build-topic-notebook 4`

The environment is always **sagemaker** for this course. Do not ask.

---

## Pre-Work: MANDATORY Reading

### Step 1: Read All Context Files (DO THIS FIRST)

**YOU MUST READ THESE FILES BEFORE DOING ANYTHING:**

1. **CLAUDE.md** - Teaching philosophy, notebook structure, tone, environment, critical rules
2. **initial_docs/outline.pdf** (already in context from research phase, or re-read)
3. **plans/topic_<N>_<slug>.md** - The cell-by-cell plan produced by `/run-research-topic`
4. If topic N > 1: `exercises/topic_<N-1>_*/topic_<N-1>_*.ipynb` - confirm prior variable names

### Step 2: Show Plan Summary Before Building

**IN CHAT, SHOW ME:**
1. Topic title and learning objectives
2. List of concepts (2-4) with their Theory / Demo / Lab breakdown
3. Total estimated cell count
4. Which cells need safety-net cells and why
5. Any continuity bridges from the prior topic

**DO NOT PROCEED until I approve this summary.**

---

## Core Teaching Principles

### 1. Four-Beat Arc for Every Concept
Every concept in the notebook follows exactly this sequence:

**Beat 1 - Problem intro (Markdown + naive/broken code)**
Show what goes wrong WITHOUT this concept. Students feel the pain before the cure.
The naive code cell runs and produces bad/broken output - that's the point.

**Beat 2 - Diagram placeholder (Markdown)**
Insert this exact comment so `/build-diagrams` can find and fill it later:
```
<!-- DIAGRAM: describe what this diagram should show -->
```
Followed by 1-2 sentences explaining what the diagram will illustrate.

**Beat 3 - Full working demo (Code)**
Complete, runnable, heavily commented. Instructor live-codes from this.

**Beat 4 - Lab**
Students implement it. See lab tiers below.

### 2. Lab Tiers - Three Kinds, Distributed Across the Day
I include three lab tiers across each day (not per topic):

**Tier 1 - Guided (most labs)**
`variable = None  # YOUR CODE` with numbered step comments above each placeholder.
Verification code at the end. Medium difficulty, 15-20 min.
Example:
```python
# Step 1: create the OpenAI client
client = None  # YOUR CODE

# Step 2: call chat.completions.create with the system and user messages
response = None  # YOUR CODE

# Verification
if response:
    print(response.choices[0].message.content)
```

**Tier 2 - Hard (one per day)**
Still uses `None  # YOUR CODE` but the step comments are less prescriptive.
Students must combine multiple things learned so far. 25-35 min.
Mark in the plan which topic gets the hard lab for its day.

**Tier 3 - Open-ended (one per day, last topic of that day)**
NO `None` placeholders. NO step-by-step hints.
Just a clear problem statement and a function signature or docstring describing inputs/outputs.
Students figure out the approach. No verification - it works or it doesn't.
Example:
```python
def build_customer_assistant(documents: list[str], query: str) -> str:
    """
    Given a list of document strings and a customer query,
    return a grounded answer using RAG.
    Use the patterns we built in Topics 2-6.
    """
    # YOUR CODE
    pass
```

### 3. Tone - First Person and Friendly
- I write in first person throughout: "I'm going to show you...", "Let's see what happens if..."
- Encouraging and warm - students may be new to this
- Emojis welcome in markdown cell headers and section titles
- No jargon unless I explain it in the same cell

### 4. Heavy Comments
- Every non-obvious line has a comment explaining the WHY
- Demo cells are fully commented as live-coding reference
- Lab starter has enough structure to orient without revealing the answer

### 5. Public/Provided Data Only
- HuggingFace datasets, public URLs, or `barclays-prompt-eng-data` S3 bucket
- Never hardcode API keys - always `getpass.getpass()`

---

## Notebook Structure

### Header (Cells 0-4)
- **Cell 0 (Markdown)**: Topic title with emoji, learning objectives, which Customer Service Assistant component this builds
- **Cell 1 (Markdown)**: "Section 0: Environment Setup"
- **Cell 2 (Code)**: `!pip install -q ...` pinned versions + `numpy<2`
- **Cell 3 (Code)**: All imports + SageMaker session + API keys via getpass
- **Cell 4 (Markdown)**: "What are we building today?" - names the component, shows a before/after teaser

### Per Concept (repeat 2-4 times, each concept = 5-7 cells)
1. **Markdown**: Concept section header + problem intro paragraph (first person, what breaks without this)
2. **Code**: Naive/broken demo that students run to feel the problem
3. **Markdown**: Diagram reference cell - contains the `<!-- DIAGRAM: ... -->` comment from the plan PLUS a link to the diagram file:
   ```
   <!-- DIAGRAM: description from plan -->
   [View diagram: diagram title](../../plans/topic_<N>/diagrams/<slug>.mmd)
   > Diagram coming via /build-diagrams - the file above will contain the Mermaid source once built.
   ```
   Then 1-2 sentences explaining what the diagram shows conceptually.
4. **Code**: Full working demo (complete, heavily commented, runnable)
5. **Markdown**: Lab instructions
6. **Code**: Lab starter (Tier 1, Tier 2, or Tier 3 - one Tier 2 per day, one Tier 3 at end of last topic of the day only)
7. **Code** (conditional): Safety-net cell gated by `if variable is None:` when lab output feeds a later cell

**Diagram placeholder rule**: The link path `../../plans/topic_<N>/diagrams/<slug>.mmd` must match exactly the slug recorded in the plan's diagram index. The file does not exist yet - that is expected. `/build-diagrams` creates it. Do not skip the link or use a different path.

### Closing
- **Markdown**: Key takeaways, what I built today, what comes next, homework extensions

---

## CRITICAL: 5-Cell Approval Checkpoints - MANDATORY

**YOU MUST NEVER ADD MORE THAN 5 CELLS WITHOUT EXPLICIT USER APPROVAL**

### Process (DO NOT DEVIATE):

1. Add exactly 5 cells (or fewer for the final batch)
2. STOP IMMEDIATELY
3. Run validation: `python validate_notebooks.py exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb --type exercise`
4. **UNCONDITIONAL /save-state rule**: whenever the notebook cell count crosses a multiple of 10
   (reaches 10, 20, 30, ...), invoke the `/save-state` skill immediately. This fires regardless
   of whether the user said "go until the end", "continue", or anything else. It is NOT optional
   and is NOT skipped in continuous mode. Context compaction can happen at any time; save-state
   ensures a new session can resume from exactly where the build was interrupted.
5. Ask user: "I have added cells X-Y. How does it look? Should I continue?"
   If the user previously gave blanket "go until end" approval, skip this question and continue -
   but step 4 still fires unconditionally at every 10-cell boundary.
6. After approval (or under blanket approval): add ONLY the next 5 cells, then return to step 1.

### Forbidden:
- Adding 6+ cells in a single batch without approval (blanket "go" covers subsequent batches)
- Skipping validation between batches
- Skipping /save-state at 10-cell boundaries for any reason, including "go until end" mode

### Notebook Write Pattern (CRITICAL FOR FILE SIZE):

When writing large content (markdown with many lines, long code cells), NEVER write all cells in one tool call. Even within a single 5-cell batch:
- Add cells one at a time if any single cell exceeds ~30 lines
- This prevents JSON truncation in the notebook file

---

## Cell Order: ALWAYS Use cell_id

**After the first cell, ALWAYS pass `cell_id` to NotebookEdit.**

### MANDATORY: Verify Cell ID Before ANY NotebookEdit call

Before EVERY NotebookEdit (replace, insert, OR delete), run this python snippet to confirm
the cell_id is at the expected position. This prevents editing the wrong cell or deleting
unintended content:

```bash
.venv/bin/python3 -c "
import json
with open('PATH_TO_NOTEBOOK') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    print(i, c.get('id','?'), c['cell_type'], repr(''.join(c['source'])[:60]))
"
```

Check the output, confirm the target cell_id is where you expect it, THEN make the edit.
Never skip this step - it is the only way to be certain you are not deleting or overwriting
the wrong cell.

Workflow:
1. Create empty notebook with Write tool (minimal JSON skeleton)
2. Add Cell 0 with `edit_mode="insert"` - no `cell_id` needed
3. Before adding Cell 1: run python snippet to confirm Cell 0's id
4. Add Cell 1 with `cell_id="<id-of-cell-0>"` - insert after Cell 0
5. Before each subsequent cell: run python snippet to verify position
6. Continue chaining `cell_id` for every subsequent cell

**Never assume cells append to the end automatically. Always chain cell_id.**

### Empty Notebook JSON Skeleton:
```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "cells": []
}
```

---

## Markdown Rendering

Markdown cells use raw strings - no escaping needed:
```python
new_source = """# Topic 4: Prompt Engineering

## Learning Objectives

By the end of this section you will:
- Design few-shot prompts
- Control output format with JSON mode

```python
# Example inline code block
response = client.chat.completions.create(...)
```
"""
```

Never escape markdown characters. Never double-backslash headers.

---

## Exercise vs Solution Notebooks

### Exercise Notebook
Lab starter cells:
```python
# Your task: build a few-shot prompt for intent classification
# 1. Define at least 3 examples covering different intents
examples = None  # YOUR CODE

# 2. Format them into a messages list
messages = None  # YOUR CODE

# Verification - run this to check your answer
if examples is not None and messages is not None:
    print(f"Examples defined: {len(examples)}")
    print(f"Messages list length: {len(messages)}")
```

Safety-net cell (required when lab output feeds a later cell):
```python
# Safety-net: if you did not finish the lab above, run this cell so the
# rest of the notebook still works. If you did finish it, SKIP this cell.
if examples is None:
    print("Using safety-net implementation so the notebook can continue.")
    examples = [...]  # working implementation
    messages = [...]
```

### Solution Notebook
- Built AFTER the exercise is fully complete and approved
- Copy: `cp exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb solutions/topic_<N>_<slug>/topic_<N>_<slug>.ipynb`
- Walk through each lab cell, replace `= None  # YOUR CODE` with complete implementation
- **DO NOT delete safety-net cells.** Keep them in place - they maintain cell parity with
  the exercise. In the solution they will never fire (the lab above is complete) but
  removing them shifts all downstream cell positions and breaks the pair validator.
  Ask Axel before deleting any cell from a solution notebook.
- Add explanation comments: what the code does, why this approach, common mistakes
- Same 5-cell approval rhythm as the exercise

---

## Workflow Summary

1. Read all context files
2. Show plan summary - wait for approval
3. Create directories: `exercises/topic_<N>_<slug>/` and `solutions/topic_<N>_<slug>/`
4. Create empty notebook skeleton with Write tool
5. Build EXERCISE notebook:
   - Add 5 cells
   - Run validation
   - After every 10 cells: invoke /save-state before asking for approval
   - Show user and ask for approval
   - Repeat until complete
6. Build SOLUTION notebook AFTER exercise is fully approved:
   - Copy exercise to solutions/
   - Fill in placeholders, 5 cells at a time, with approval checkpoints
   - NEVER delete any cell - only replace None placeholders with implementations
7. Final validation:
   - `python validate_notebooks.py --pair exercises/.../notebook.ipynb solutions/.../notebook.ipynb`
   - Both notebooks run top-to-bottom without errors
   - File sizes reasonable (under 500 KB each)

---

## Checklist (complete before marking done)

- [ ] Topic title has emoji, learning objectives clear
- [ ] SageMaker setup: Session + execution role + getpass for API keys
- [ ] numpy<2 pinned in install cell
- [ ] Every concept has: problem intro code, diagram placeholder, full demo, lab
- [ ] Naive/broken demo cell precedes every full demo
- [ ] Diagram placeholder `<!-- DIAGRAM: ... -->` present for each concept
- [ ] All demo code is heavily commented
- [ ] First person tone throughout ("I'm going to show you...")
- [ ] Lab Tier 1 uses `None  # YOUR CODE` with numbered steps, no answer hints
- [ ] One Tier 2 hard lab present for this day (check against other topics on same day)
- [ ] One Tier 3 open-ended lab present at end of day (last topic only)
- [ ] Safety-net cells present for all labs whose output is used downstream
- [ ] Solution notebook has complete implementations with explanation comments
- [ ] Safety-net cells KEPT in solution (never deleted - they maintain pair parity)
- [ ] Wrap-up cell connects to next topic
- [ ] AI-tells scan passed (no em dashes, no en dashes, no `---` separators, no Unicode multiplication)
- [ ] Pair validation passes

---

## Update plans/TOPICS.md on Completion

When BOTH exercise and solution notebooks are fully built and validated:

1. Read `plans/TOPICS.md`
2. Find the entry for this topic
3. Update status: `- **Status**: in_progress` -> `- **Status**: done`
4. Mark the build and solution manifest checkboxes as done (`[ ]` -> `[x]`)

When starting the exercise build (before solution):
- Set status to `in_progress`

Use Edit (not Write) - do not rewrite the whole file, only the targeted lines.
