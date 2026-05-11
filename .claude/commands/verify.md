---
description: Verify everything for a topic - research plan, exercise notebook, and solution notebook in parallel
---

Verify topic: $ARGUMENTS

## GUARD: Read Course Manifest First

Before doing anything else, check for `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`:

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```

If the file does NOT exist, stop immediately and say:

> `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` not found. Run `/init-course` first to record the course decisions. Every command requires this file before proceeding.

If it exists, read the full file and keep its contents in mind throughout this command.

---

## What This Command Does

Given a topic number N, check what exists and run the appropriate verifications in parallel:

```bash
ls plans/topic_<N>_*.md 2>/dev/null
ls exercises/topic_<N>_*/*.ipynb 2>/dev/null
ls solutions/topic_<N>_*/*.ipynb 2>/dev/null
```

Then fan out in parallel based on what is found:

| What exists | What runs |
|-------------|-----------|
| Plan file | `/verify-research N` |
| Exercise notebook | `/validate-notebooks exercises/topic_<N>_<slug>/... --type exercise` |
| Solution notebook | `/validate-notebooks solutions/topic_<N>_<slug>/... --type solution` |
| Both notebooks | Also run `--pair` check |

---

## Process

### Step 1: Discover what exists

Run the three `ls` commands above. Note the exact file paths found.

### Step 2: Run verifications in parallel

Launch all applicable verifications simultaneously. Do not wait for one to finish before
starting another.

**If plan file exists**: invoke `/verify-research N` logic inline (do not spawn a separate
command - run the checks from verify-research.md directly in this session).

**If exercise notebook exists**: run the validate-notebooks checks for exercise type.

**If solution notebook exists**: run the validate-notebooks checks for solution type.

**If both notebooks exist**: also run the pair check.

### Step 3: Aggregate results

Print a combined report:

```
Topic N - Full Verification Report
====================================

RESEARCH PLAN (plans/topic_N_slug.md)
--------------------------------------
<verify-research output table>

EXERCISE NOTEBOOK (exercises/topic_N_slug/topic_N_slug.ipynb)
--------------------------------------------------------------
<validate-notebooks exercise output>

SOLUTION NOTEBOOK (solutions/topic_N_slug/topic_N_slug.ipynb)
-------------------------------------------------------------
<validate-notebooks solution output>

PAIR CHECK
----------
<pair validation output if both exist>

OVERALL
-------
Plan:     PASS / FAIL / NOT FOUND
Exercise: PASS / FAIL / NOT FOUND
Solution: PASS / FAIL / NOT FOUND
Pair:     PASS / FAIL / SKIPPED
```

### Step 4: Recommend next action

Based on results:

- If plan fails: fix the plan, then re-run `/verify N`
- If plan passes but no exercise exists: run `/build-topic-notebook N`
- If exercise fails: run `/fixes <description>` to log and fix
- If solution fails: run `/fixes <description>` to log and fix
- If everything passes: topic N is clean - ready for class or next topic

---

## Non-Negotiables

- Never skip a check because a file looks fine at a glance - always run all checks
- Report every failure with the specific cell number or line number
- Do not auto-fix anything - this command is read-only; it reports, never edits
- NEVER delete a cell. This command only reads and reports. If a fix is needed, use /fixes.
  /fixes must also never delete - only replace content or insert. Ask Axel before any deletion.
