---
description: Build Mermaid diagram files one by one from a topic's research plan into plans/topic_N/diagrams/
---

Build diagrams for topic: $ARGUMENTS

## GUARD: Read Course Manifest First

Before doing anything else, check for `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`:

```bash
ls plans/CORE_TECHNOLOGIES_AND_DECISIONS.md 2>/dev/null
```

If the file does NOT exist, stop immediately and say:

> `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md` not found. Run `/init-course` first to record the course decisions. Every command requires this file before proceeding.

If it exists, read the full file and keep its contents in mind throughout this command.

---

This command runs AFTER `/run-research-topic N` has produced the plan and AFTER `/build-topic-notebook N` has built the notebooks with diagram link placeholders.

It reads the diagram index from the plan file, builds each Mermaid diagram one at a time into `plans/topic_<N>/diagrams/<slug>.mmd`, and asks for approval between each one.

---

## Step 1: Read Context

1. Read `plans/topic_<N>_<slug>.md` - find the diagram index section (all entries with "Diagram slug:", "Diagram path:", "Description:")
2. Read the relevant notebook cells that reference each diagram to understand the exact context each diagram needs to serve

---

## Step 2: List All Diagrams

Print the full list of diagrams to build from the plan index:

```
Diagrams to build for Topic N:
1. <slug>.mmd - <description>
2. <slug>.mmd - <description>
...

I'll build them one at a time and ask for your approval before moving to the next.
```

**Ask**: "Ready to start with diagram 1? Or any specific order you want?"

---

## Step 3: Build Each Diagram (One at a Time, Approval Between Each)

For each diagram:

### 3a. Design the diagram in chat first

Before writing the file, show the Mermaid source in chat:

```
Diagram <N>: <slug>
Description: <from plan>
Context: <which cell this appears in, what concept it illustrates>

Proposed Mermaid source:

```mermaid
<source here>
```

Does this look right? I'll write the file once you approve.
```

**Wait for approval before writing the file.**

### 3b. Write the file

Once approved, create the directory if needed and write the file:

```bash
mkdir -p plans/topic_<N>_<slug>/diagrams
```

File path: `plans/topic_<N>_<slug>/diagrams/<slug>.mmd`

File contents - just the raw Mermaid source, no fences, no extra text:
```
graph TD
    A[...] --> B[...]
    ...
```

The file is plain Mermaid source so it can be rendered by any Mermaid-aware viewer (GitHub, JupyterLab extensions, VS Code).

### 3c. Confirm the notebook link resolves

Check that the notebook cell referencing this diagram has the correct relative path:
`../../plans/topic_<N>_<slug>/diagrams/<slug>.mmd`

If the path in the notebook cell is wrong, fix it with NotebookEdit (read the notebook first to get the cell_id).

### 3d. Ask before the next diagram

"Diagram `<slug>.mmd` written. Ready for diagram <N+1> (`<next-slug>`)?"

**Do not proceed until the user says yes.**

---

## Diagram Design Rules

- Keep diagrams focused: one concept per diagram, not a full system overview
- Use `graph TD` (top-down) for flow and pipeline diagrams
- Use `sequenceDiagram` for API call sequences (request/response flows)
- Use `flowchart LR` for decision trees (routing logic, guardrail checks)
- Node labels: short phrases, no em dashes, no special Unicode
- Max ~10 nodes per diagram - if it needs more, split into two diagrams
- Every node and edge label uses plain ASCII text only

### Good diagram types by concept:
- RAG pipeline -> `graph TD` showing query to embed to retrieve to augment to generate
- API call flow -> `sequenceDiagram` between User, App, and LLM API
- Chunking strategies -> `graph TD` showing document splitting options
- Guardrail routing -> `flowchart LR` with decision branches
- Memory management -> `graph TD` showing messages list growing and truncation

---

## Step 4: Summary

After all diagrams are built:

> All N diagrams written to `plans/topic_<N>_<slug>/diagrams/`.
>
> Files:
> - `<slug1>.mmd`
> - `<slug2>.mmd`
>
> Notebook links verified: yes / no (if no, list which cells still need fixing)
>
> Next step: open the notebooks in JupyterLab and confirm the diagram links resolve, or run `/validate-notebooks` to check overall notebook health.
