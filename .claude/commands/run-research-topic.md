---
description: Research a topic and produce a cell-by-cell plan file at plans/topic_N_slug.md that /build-topic-notebook can consume directly
---

Research and plan topic: $ARGUMENTS

## GUARD: Read Course Manifest First

Before doing anything else, check for `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`:

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```

If the file does NOT exist, stop immediately and say:

> `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` not found. Run `/init-course` first to record the course decisions. Every command requires this file before proceeding.

If it exists, read the full file and keep its contents in mind throughout this command.

---

This command is the **natural precursor to `/build-topic-notebook`**. Its only job is to produce a plan file at `plans/topic_N_slug.md` so that `/build-topic-notebook N` can build notebooks cell by cell from it.

**This command does NOT create any notebooks.** It produces markdown only.

**Web research is MANDATORY. Use the `/research` skill for all web searches in this command.**

## FAILURE DEFINITION

This command has exactly two outcomes: COMPLETE SUCCESS or COMPLETE FAILURE. There is no partial credit.

**COMPLETE SUCCESS** = all of the following are true:
- All 5 research cycles completed and visible in chat
- Each cycle invoked the `/research` skill (Skill tool, skill="research")
- PRE-WRITE SELF-CHECK printed in chat with all YES answers
- Plan file written to `plans/topic_<N>_<slug>.md` with all required sections
- `plans/TOPICS.md` updated

**COMPLETE FAILURE** = any of the following happened:
- Fewer than 5 research cycles completed
- Any cycle skipped the `/research` skill invocation
- PRE-WRITE SELF-CHECK not printed, or any field was NO when Write was called
- Plan file not written (command ended without producing the file)
- Plan file written with stub/placeholder content instead of real cells

If this command ends in COMPLETE FAILURE, the correct response is: report what failed,
and restart from the beginning. Do not patch a partial result.

---

## Command Arguments

```
/run-research-topic <topic_number>
```

Example: `/run-research-topic 4`

After you invoke this command, I will ASK you three questions before doing any work:

1. **Main focuses / things to consider** - what to emphasize, go lighter on, or student confusion from prior topics to address.
2. **Confirmation of prior-topic continuity** - I will state which variables, patterns, and API objects carry over from prior topics and ask you to confirm.
3. **Any Barclays-specific context** - product names, scenarios, or compliance details to highlight.

**DO NOT proceed to research until I have all three answers.**

---

## Output Contract (NON-NEGOTIABLE)

1. **One file only**: `plans/topic_<N>_<topic_slug>.md` (always under `plans/`, never anywhere else).
2. **Structure** (every section required):
   - `# Topic N: <Title> - Cell-by-Cell Plan`
   - `## Context`
   - `## Deliverables`
   - `## Session Timing (~60-90 min)` table
   - `# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells)`
   - Each cell: `## Cell N - Markdown/Code: Description` with full content in fenced block
   - `# VERIFICATION CHECKLIST`
   - `# RESEARCH VALIDATED (Month Year)`
3. **No `.ipynb` files.** No Write calls except to `plans/topic_<N>_<topic_slug>.md`.
4. **Plan must be directly executable by `/build-topic-notebook`** - every cell has enough detail the builder does not need to invent content.

### Writing the Plan File in Batches

**If the plan file will exceed ~150 lines, write it in sections:**
1. First Write call: header through `## Session Timing` table
2. Subsequent Write calls: append cell-by-cell content in chunks of ~50 lines each (use Edit to append, not overwrite)
3. Final Write/Edit call: append VERIFICATION CHECKLIST and RESEARCH VALIDATED block

This prevents large single Write calls that can get truncated.

---

## Pre-Work: MANDATORY Reading

Before any research, read:

1. `CLAUDE.md` - teaching philosophy, notebook structure, tone, environment
2. `initial_docs/outline.pdf` - extract this topic's section (title, concepts, labs, objectives)
3. `.claude/commands/build-topic-notebook.md` - the consumer of your plan
4. If topic N > 1: open `exercises/topic_<N-1>_*/topic_<N-1>_*.ipynb` - list exact variable names, API patterns, and helpers students already have

---

## MANDATORY CYCLE TRACKER

At the start of research, post this tracker in chat and update it after completing each cycle. You MUST NOT write the plan file until all 5 boxes are checked:

```
CYCLE TRACKER
[_] Cycle 1 - Outline Alignment and Prior-Topic Continuity       /research invoked: NO
[_] Cycle 2 - Environment and Dependencies                        /research invoked: NO
[_] Cycle 3 - Pedagogical Structure                               /research invoked: NO
[_] Cycle 4 - Lab Difficulty and Stretch Labs                     /research invoked: NO
[_] Cycle 5 - Final Integration, Timing, Take-Homes              /research invoked: NO
```

After each cycle, update BOTH the `[_]` checkbox AND change `NO` to `YES` on the `/research invoked` field.
A cycle is NOT complete if `/research invoked` still says NO. There are no exceptions.

**HARD STOP before writing any file**: Check the tracker.
- If any cycle shows `[_]` (not checked): STOP. Complete that cycle fully before proceeding.
- If any cycle shows `/research invoked: NO`: STOP. That cycle must be redone with a real `/research` skill call.
- Only when all 5 show `[x]` AND all 5 show `/research invoked: YES` may you write the plan file.
- Writing the plan with any unchecked cycle or any missing `/research` invocation is a complete failure.
  The correct response to discovering this mid-run is to go back and complete the missing cycles,
  not to write a partial plan.

---

## Process: 5 TDD-Style Research Cycles

**All cycles visible in chat.** Do not skip, summarize, or collapse any cycle. After each cycle completes, update the CYCLE TRACKER above.

**MANDATORY: Every single cycle MUST invoke the `/research` skill.** This is not optional.
- Do NOT use WebSearch directly. Do NOT use WebFetch directly.
- Do NOT summarize what you "already know" instead of searching.
- The `/research` skill is the ONLY acceptable source of web-validated facts in this command.
- If you complete a cycle without invoking `/research`, that cycle does not count. Redo it.

---

### Cycle 1 - Outline Alignment and Prior-Topic Continuity

**Goal**: Lock what we teach and what students already have.

- Parse this topic's section from `initial_docs/outline.pdf`
- Open the prior topic's notebook; list every named variable, API object, credential pattern, and helper function students already have
- Propose the bridging approach: which objects carry over, which are new
- Default model IDs: `gpt-4o` for OpenAI, `claude-sonnet-4-6` for Anthropic (unless outline says otherwise)

**Invoke `/research`**: "What are current best practices for [this topic's core concept] with the OpenAI / Anthropic Python SDK as of 2025?"

**Refutation**: Does any proposed variable name collide with prior topics? Are we re-implementing something already built?

---

### Cycle 2 - Environment and Dependencies

**Goal**: Lock a working install cell, verify all external resources.

- Lock the full `!pip install -q` line with ALL packages pinned to current stable versions
- Always include `numpy<2`
- Verify every dataset URL or S3 path (`barclays-prompt-eng-data` bucket)
- Confirm library import paths against current docs (openai, anthropic, chromadb APIs change - verify exact import syntax)
- Credential pattern: `getpass.getpass()` for OpenAI/Anthropic keys; `sagemaker.Session()` + `get_execution_role()` for AWS

**Invoke `/research`**: "What is the current stable version of [library] and is import path [path] still correct? Does numpy<2 conflict with [other libraries]?"

**Refutation**: Does `numpy<2` conflict with any library? Are any import paths deprecated?

---

### Cycle 3 - Pedagogical Structure (Four-Beat Arc for EVERY concept)

**Goal**: Design the four-beat arc for each concept and decide lab tiers.

For EACH concept (2-4 per topic), plan these cells in order:

**Beat 1 - Problem intro**
- Markdown cell: section header + "here's what breaks without this" setup paragraph (first person)
- Code cell: naive/broken code that students run and see fail or produce bad output

**Beat 2 - Diagram placeholder**
- Markdown cell containing EXACTLY this line:
  `<!-- DIAGRAM: <description of what the diagram shows> -->`
- Followed by 1-2 sentences saying what the diagram will illustrate
- The file will live at `plans/topic_<N>/diagrams/<slug>.mmd` - reference it as:
  `[View diagram](../../plans/topic_<N>/diagrams/<slug>.mmd)` (link, not embed, since it is Mermaid source)
- Name the diagram file slug in the plan so `/build-diagrams` can find it

**Beat 3 - Full working demo**
- Code cell: complete runnable solution, heavily commented, first person inline comments

**Beat 4 - Lab**
- Markdown cell: lab instructions
- Code cell: lab starter

**Lab tier assignment (plan ONE per day across all topics that day):**
- Tier 1 (guided): most labs - `variable = None  # YOUR CODE` with numbered step comments, verification at end, 15-20 min
- Tier 2 (hard): ONE per day - multi-step, less prescriptive steps, 25-35 min, mark clearly in plan which topic gets it
- Tier 3 (open-ended): ONE per day, last topic of that day only - function signature + docstring only, no placeholders, no verification, students figure out the approach

Every concept has all four beats. No concept is beat-3-only or beat-4-only.
Build toward ONE main outcome that ties the whole topic together.

In the plan file, for each diagram placeholder also record:
```
- Diagram slug: `<slug>.md`
- Diagram path: `plans/topic_<N>/diagrams/<slug>.mmd`
- Description: <what it shows>
```
This is what `/build-diagrams` reads.

**Invoke `/research`**: "What are common gotchas when [doing this demo/lab concept] with [library]?"

**Refutation**: Does any `# YOUR CODE` comment leak the solution? Is Beat 1 actually broken/naive (not just a comment saying "naive approach")? Does each diagram description have enough detail for `/build-diagrams` to produce it without guessing?

---

### Cycle 4 - Lab Difficulty and Stretch Labs

**Goal**: Calibrate difficulty, add stretch content, identify safety-net requirements.

- For each in-class lab: standard version (15 min) + clearly labeled stretch version for fast finishers
- Test: cover the solution, read only the lab starter cell. Can a student solve it in under 30 seconds from the comments alone? If yes, hints are too strong - rewrite.
- Each lab gets a Homework Extension (async, deeper, production-oriented)
- Identify which labs need safety-net cells: any lab whose output variable is consumed by a later cell

**Invoke `/research`**: "What production concerns or edge cases in [lab topic] make a good homework extension?"

**Refutation**: Is the stretch lab genuinely harder, or just more volume?

---

### Cycle 5 - Final Integration, Timing, Take-Homes

**Goal**: Verify everything fits, write sources and checklist.

- Verify total session timing (Day 1: topics 1-3, Day 2: topics 4-6, Day 3: topics 7-9; each ~60-90 min)
- Wrap-up cell names the Customer Service Assistant component built this session and bridges to next topic
- Lock optional deep-dive notebook outline if applicable
- Produce RESEARCH VALIDATED block: every source URL + specific fact extracted
- Produce VERIFICATION CHECKLIST for this topic

**Invoke `/research`**: "What are the most important production best practices for [this topic] worth mentioning in take-homes?"

**Refutation**: Any unsourced claims? Any un-verified library versions or dataset URLs?

---

## HARD GATE: Pre-Write Self-Check (MANDATORY before any Write call)

Before calling Write to create the plan file, you MUST print this self-check in chat
and verify every line passes. If any line fails, do NOT write the file - go back and fix it.

```
PRE-WRITE SELF-CHECK
--------------------
Cycle tracker all 5 checked [x]:          YES / NO
All 5 /research invocations confirmed:    YES / NO
Plan file path is plans/topic_N_slug.md:  YES / NO
No .ipynb files created:                  YES / NO
RESEARCH VALIDATED block has URLs:        YES / NO
VERIFICATION CHECKLIST present:           YES / NO
All cells have full content (not stubs):  YES / NO
AI-tells scan passed (no em/en dashes):   YES / NO
```

If every answer is YES: proceed to Write.
If any answer is NO: this is a FAILURE. Fix the failing items, re-run the missing cycles,
and re-check before attempting Write again. Do not write a partial plan.

---

## Plan File Structure

```markdown
# Topic <N>: <Title> - Cell-by-Cell Plan

## Context

<What students arrive with - concrete variable names, API patterns>

<The key insight students leave with>

**Coordination with Topic <N-1>**: <one paragraph>
**Coordination with Topic <N+1>**: <one paragraph>

**Key decisions:**
- **Environment**: sagemaker (ml.t3.medium, no GPU)
- **Library choices**: <with pinned versions>
- **Model choices**: gpt-4o / claude-sonnet-4-6
- **Dataset / source choices**: <with URLs or S3 paths>
- **Continuity choices**: <which variables bridge from Topic N-1>
- **Lab safety-net cells**: <list which labs need them and why>

## Deliverables

- **Exercise**: `exercises/topic_<N>_<slug>/topic_<N>_<slug>.ipynb`
- **Solution**: `solutions/topic_<N>_<slug>/topic_<N>_<slug>.ipynb`

## Session Timing (~60-90 min)

| Section | Duration | Type |
|---------|----------|------|
| Section 0: Setup | 5 min | Code |
| Section 1: <Concept> | 20 min | Demo + Lab |
| **Total** | **~XX min** | |

---

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells)

## Cell 0 - Markdown: Title and Overview

```markdown
<full content>
```

## Cell 1 - Markdown: Section 0 - Environment Setup

```markdown
<content>
```

## Cell 2 - Code: Installs and Imports

```python
<actual pip install line + all imports + version checks>
```

<continue for every cell>

---

# VERIFICATION CHECKLIST

- [ ] Topic title and learning objectives clear
- [ ] SageMaker setup: Session + execution role + getpass for API keys
- [ ] numpy<2 pinned
- [ ] Every concept has Theory -> Demo -> Lab
- [ ] No lab starter comment leaks the solution
- [ ] Every in-class lab has a stretch version and Homework Extension
- [ ] All dataset URLs / S3 paths verified working
- [ ] All library versions pinned to web-verified current stable
- [ ] Prior-topic variables referenced by exact name
- [ ] Session timing fits Day slot
- [ ] Bridges to next topic in wrap-up

---

# RESEARCH VALIDATED (Month Year)

1. <Source URL> - <specific fact extracted>
2. <Source URL> - <specific fact extracted>
```

---

## Non-Negotiables (Violating Any = Complete Failure, Start Over)

1. Path is always `plans/topic_<N>_<topic_slug>.md` - never anywhere else
2. No `.ipynb` files created
3. `numpy<2` in every install cell
4. `# YOUR CODE` comments never reveal the answer
5. Every concept has a Demo cell AND a Lab cell
6. Every in-class lab has a stretch version and homework extension
7. All dataset URLs / S3 paths verified before finalizing
8. Prior-topic continuity confirmed by opening the actual prior notebook
9. All 5 cycles completed and visible in chat. CYCLE TRACKER must show [x] for all 5
   AND /research invoked: YES for all 5 before any Write call.
   Partial completion = failure. Missing /research invocation = failure.
   The correct recovery is to go back and complete missing cycles, not skip ahead.
10. EVERY CYCLE MUST INVOKE THE /research SKILL. No exceptions. No substitutions.
    - Raw WebSearch calls do not count.
    - Raw WebFetch calls do not count.
    - "I already know this" does not count.
    - Only a Skill tool call with skill="research" satisfies this requirement.
    A cycle completed without /research is not a completed cycle.
11. The PRE-WRITE SELF-CHECK must be printed in chat and must show all YES before Write.
    If any field is NO, do not write. Fix it first.
12. ZERO AI-TELLS IN THE PLAN FILE: no em dashes, no en dashes, no Unicode multiplication,
    no emoji - plain ASCII only. Run a final pass before Write.

---

## Update plans/TOPICS.md

After writing the plan file, update `plans/TOPICS.md`:

1. Read `plans/TOPICS.md`
2. Find the entry for this topic (search for `## Topic <NN>`)
3. Change `- **Status**: not_started` to `- **Status**: planned`
4. Mark the first manifest checkbox as done: change `- [ ] Run` to `- [x] Run` for the research step

Use Edit (not Write) to make this targeted change - do not rewrite the whole file.

---

## Handoff

Once the plan file and TOPICS.md are updated, end with:

> Plan written to `plans/topic_<N>_<topic_slug>.md`. TOPICS.md updated: status -> planned.
>
> Next step: run `/build-topic-notebook <N>` to generate notebooks from this plan, 5 cells at a time with approval between batches.

Do not offer to run `/build-topic-notebook` yourself.
