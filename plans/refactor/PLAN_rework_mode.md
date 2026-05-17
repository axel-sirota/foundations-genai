# Plan: add a "rework mode" to the notebook pipeline

Created 2026-05-17. Awaiting Axel approval before editing any skill file.

## Why

Phase 3 of the course restructure needs to APPLY the `plans/refactor/*.md` design
docs to EXISTING notebooks. The current pipeline (`/run-research-topic` ->
`/build-topic-notebook`) only builds notebooks FROM SCRATCH off a research plan.
We want to keep using the same skills and the 5-cell approval / validation /
save-state discipline, but drive edits from a design doc against a copy of an
existing notebook.

## Decisions (Axel, 2026-05-17)

1. Invocation: TWO explicit flags.
   `/build-topic-notebook <N> --base-from <notebook.ipynb> --design-doc plans/refactor/<doc>.md`
2. Gate: add a real `rework` state to `.claude/pipeline.yaml` with its own
   marker chain (not a silent skill-level waiver).
3. Scope: `build-topic-notebook` AND `build-diagrams` both get the flags.

## Behavior in rework mode (--base-from + --design-doc present)

build-topic-notebook:
- Skip the empty-skeleton creation. Instead `cp <base-from>` to the target
  Exercises path, and the Solutions twin to the Solutions path.
- Read `--design-doc` (a plans/refactor/*.md doc) INSTEAD of plans/topic_N_slug.md.
- Apply the doc's cell-by-cell actions (KEEP / EDIT / NEW / DELETE / MERGE) to
  the copy. EDIT = match the quoted old text and replace; NEW = insert at the
  stated position; DELETE/MERGE per the doc.
- KEEP every existing discipline: 5-edits-per-batch approval checkpoints,
  validate_notebooks.py between batches, /save-state at 10-cell boundaries,
  cell_id verification before every NotebookEdit, AI-tells scan, pair
  validation, four-beat / lab-tier rules still apply to any NEW cells.
- Solutions twin: apply the doc's "Solutions twin" instructions (lab cells
  filled, safety-net cells handled per doc).

build-diagrams:
- Accept the same flags. In rework mode it still just scans the resulting
  notebook for `<!-- DIAGRAM: -->` placeholders and builds the .mmd files;
  the flag only tells it which notebook/doc pair it belongs to and which
  marker to write. Diagram output dir stays plans/<topic>/diagrams/.

Default mode (no flags): UNCHANGED. Both skills behave exactly as today.

## pipeline.yaml change

Add a `rework` state, parallel to `build-topic-notebook`:

```yaml
  rework:
    next: validate-notebooks
    next_command: "/validate-notebooks {slug}"
    description: "Apply a plans/refactor/ design doc to an existing notebook"
    marker_written: "rework-{slug}.done"
    marker_required: null   # rework starts from a design doc, not research
    side_channel: false
```

Rationale for `marker_required: null`: a rework design doc has already been
through 3 Codex adversarial rounds - that IS its verification. There is no
`/run-research-topic` / `/verify-research` step in front of it. The branch
guard hook that blocked `/build-topic-notebook new` must recognise that when
`--base-from` is present the required marker is `rework`'s (none), not
`verify-research-{slug}.done`.

## Files to change

1. `.claude/pipeline.yaml` - add the `rework` state.
2. `.claude/commands/build-topic-notebook.md` - add a "Rework Mode" section:
   flag parsing, copy-not-create, design-doc-driven editing, marker handling.
3. `.claude/commands/build-diagrams.md` - accept the flags, write the right marker.
4. The hook that enforces markers (whatever blocked /build-topic-notebook new) -
   teach it the rework-mode marker exception. LOCATE it first; do not guess.

## Execution order (each step = stop point)

- Step 0: this plan approved.
- Step 1: locate the marker-enforcing hook, read it, confirm how to add the
  rework exception. Report back BEFORE editing it.
- Step 2: edit pipeline.yaml (add rework state).
- Step 3: edit build-topic-notebook.md (add Rework Mode section).
- Step 4: edit build-diagrams.md (accept flags).
- Step 5: edit the hook (rework marker exception).
- Step 6: dry-run check - invoke /build-topic-notebook in rework mode on ONE
  notebook (topic_3 or an optional) and confirm the hook lets it through and
  the skill reads the design doc. Stop for review before doing all notebooks.

## NOT in this plan

- Actually reworking all the notebooks (that is the build, after the skill
  changes land and are verified on one notebook).
- Any change to default-mode behavior.
