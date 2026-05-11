# Notebook Course SDD Workflow

This repo uses a specialised pipeline tuned for building pedagogically sound Jupyter notebook courses. It replaces the standard SDD software chain with a research-first, notebook-build flow. All work commits directly to `main` — no feature branches.

## The Pipeline

```
/request
    |
/specify          <- lock topic scope, outline, course context
    |
/run-research-topic   <- 5-cycle TDD research -> cell-by-cell plan at plans/topic_N_slug.md
    |
/verify-research      <- 14-point checklist: four-beat arc, diagram index, lab tiers, AI-tells
    |
/build-topic-notebook <- build exercise + solution notebooks, 5 cells at a time with approval
    |
/validate-notebooks   <- validate structure, cell order, pedagogy, placeholder completeness
    |
/build-diagrams       <- generate Mermaid diagrams for all <!-- DIAGRAM: --> placeholders
    |
  commit to main
```

## Why This Order

**`/run-research-topic` before building**: Every concept needs web-validated library versions, verified dataset URLs, and real-world gotchas baked into the plan. Building notebooks without research produces stale or wrong content.

**`/verify-research` as a gate**: The plan file is the single source of truth consumed by `/build-topic-notebook`. If the plan is broken (missing beats, bad diagram index, leaked lab solutions), the notebooks will be broken. Catching it in the plan is 10x cheaper than fixing it in the notebook.

**`/validate-notebooks` before diagrams**: Diagrams are expensive to generate. Only generate them once the notebook structure is confirmed correct.

## Teaching Methodology: Four-Beat Arc

Every concept in every notebook follows exactly this sequence:

| Beat | Type | Purpose |
|------|------|---------|
| 1 | Markdown + broken code | Students feel the pain before the cure |
| 2 | `<!-- DIAGRAM: ... -->` placeholder | Visual anchor for the concept |
| 3 | Full working demo (code) | Instructor live-codes from this |
| 4 | Lab (instructions + starter code) | Students implement it |

No concept is beat-3-only or beat-4-only. Every concept has all four beats.

## Lab Tiers (one per day across all topics that day)

- **Tier 1 (guided)**: `variable = None  # YOUR CODE` with numbered step comments + verification. 15-20 min. Most labs are this tier.
- **Tier 2 (hard)**: Multi-step, less prescriptive. 25-35 min. ONE per day.
- **Tier 3 (open-ended)**: Function signature + docstring only. No placeholders, no verification. ONE per day, last topic of that day only (topics 3, 6, 9).

Every lab also has a **stretch version** for fast finishers and a **Homework Extension** for async deeper work.

## STAR Method in Labs

Labs are framed using the STAR structure:
- **Situation**: What scenario are we in? (Barclays customer service assistant context)
- **Task**: What does the student need to build?
- **Action**: The `# YOUR CODE` scaffolding guides the steps
- **Result**: A verification cell confirms the expected output

## Diagram Convention

Diagrams are referenced but not embedded in notebooks (they are Mermaid source files):

```markdown
<!-- DIAGRAM: what this diagram shows -->
[View diagram](../../plans/topic_N/diagrams/slug.mmd)
```

The plan file indexes every diagram with slug + path + description so `/build-diagrams` can find and generate them without guessing.

## Commands Reference

| Command | Source | Role in pipeline |
|---------|--------|-----------------|
| `/request` | global | Entry point — capture topic idea |
| `/specify` | global | Lock scope and course context |
| `/run-research-topic` | project | 5-cycle research -> plan file |
| `/verify-research` | project | Validate plan before building |
| `/build-topic-notebook` | project | Build notebooks from plan |
| `/validate-notebooks` | project | QA notebooks |
| `/build-diagrams` | project | Generate Mermaid diagrams |
| `/research` | global | Used inside /run-research-topic cycles |
| `/read-context` | global | Load course context at session start |
| `/start-session` | global | Begin a session |
| `/save` / `/resume` | global | Survive context compaction |

## What Does NOT Apply Here

- No feature branches (`ALLOW_DIRECT_MAIN=1` in settings.json)
- No `/skeleton`, `/implement`, `/tests`, `/dod-test` — this is content, not software
- No `/finish-feature` — work commits straight to main after `/build-diagrams`
- No coverage gates, no PR flow
