# Course Restructure Migration Plan

Created: 2026-05-16
Status: DRAFT — awaiting Axel approval before any execution.

## Why

Delivery feedback (ISSUES_TO_FIX.md item 5): attention + transformer topics are
too math/internals-heavy. Students are *users* of LLMs, not transformer authors.
The course buries the payoff (load / fine-tune / adapt / deploy) behind a
from-scratch build detour. Fix: demote the internals to an optional theory track,
pull the user-facing topics forward, end with a practical agent capstone.

## Target structure

OPTIONAL theory track (self-contained, advanced learners, taught from slides):
- `topic_optional_attention_python`   (was topic_3a)
- `topic_optional_attention_pytorch`  (was topic_3b)
- `topic_optional_transformers`       (was topic_4)
- `topic_optional_lora_ffn`           (was topic_7a)

REQUIRED user-facing path:

| New | Old  | Slug                        | Notes |
|-----|------|-----------------------------|-------|
| T1  | T1   | overview_genai              | unchanged |
| T2  | T2   | introducing_llms            | + theory-slide pointer for attention/transformers |
| T3  | T5   | huggingface                 | renumber |
| T4  | T6a  | full_finetuning             | renumber + FIX issue 7 (estimator bug) |
| T5  | T6b  | transfer_learning           | renumber |
| T6  | T7b  | peft_lora_distilbert        | renumber |
| T7  | T8   | quantization                | renumber |
| T8  | NEW  | agent_capstone              | new build via topic pipeline |

## Renumber map (mechanical)

```
topic_5  -> topic_3
topic_6a -> topic_4
topic_6b -> topic_5
topic_7b -> topic_6
topic_8  -> topic_7
topic_3a -> topic_optional_attention_python
topic_3b -> topic_optional_attention_pytorch
topic_4  -> topic_optional_transformers
topic_7a -> topic_optional_lora_ffn
```

Collision risk: old `topic_4` and new `topic_4` (from 6a) both exist mid-rename.
MUST rename to a temp namespace first, then to final names. Two-pass rename.

## Scope of changes (from cross-ref survey)

- 33 directories: Exercises/, Solutions/, plans/ x 11 topic dirs each.
- 13 researches/*.md files.
- 6 `scripts_topicN/` folders + `source_dir=` / `os.makedirs()` refs in notebooks.
- ~100+ in-notebook cross-references ("in Topic N", "T4", variable carryover).
- ~30+ diagram paths `../../plans/topic_N_slug/diagrams/`.
- Planning/config: plans/TOPICS.md, AUDIT_*.md, VERIFY_TOPIC*.md,
  docs/specs/setup/HANDOFF.md. (.claude/pipeline.yaml is safe — no hardcoding.)

## Variable-carryover chains to preserve

- T2 -> T3a(opt): COMPLAINT_TOKENS  — breaks if optional skipped; make 3a self-contained.
- T7a(opt) -> T7b: LoRA rank r=8, matrices A/B — T7b (new T6) must self-recap.
- T7b -> T8: trained checkpoints — both move together, chain intact.

## Execution phases

See "Revised phase list" below — each phase is one commit, stop for approval
between phases.

## Decisions (resolved with Axel 2026-05-17)

1. Optional notebooks -> fully separate `topic_optional_<slug>` namespace, no
   numbers. (As the renumber map above already states.)
2. T8 capstone local LLM -> a small HuggingFace model run IN-KERNEL via
   `transformers` (e.g. distilgpt2 / Qwen2-0.5B). No extra runtime install,
   ties back to T3-T7. Pure Python, no agent frameworks.
3. zips/ regeneration -> DEFERRED. Out of scope for this effort.
4. Theory slides (was Phase 8) -> DEFERRED. Out of scope for this effort.

## Revised phase list

- Phase 0: this plan approved.  [DONE]
- Phase 1: two-pass directory rename (temp namespace dodges topic_4 collision).  [DONE]
- Phase 2: fix in-notebook references.
  - 2a: diagram paths + scripts_topic refs.  [DONE]
  - 2b: MECHANICAL text fixes only — self-reference renumbering in the 5
    required notebooks, backward refs between required topics via the renumber
    map, lowercase topic_N tokens. NOT touched: `\bTN\b` shorthand (overloaded
    with the T4 GPU and T5 model family), anything referencing optional topics.
- Phase 3: NARRATIVE REWORK (bigger than originally scoped).
  - Rewrite the 4 optional notebooks to be genuinely standalone: drop the
    sequential "next topic" narrative, rework "YOU ARE HERE" tables, add
    self-contained recaps (COMPLAINT_TOKENS, LoRA setup). They are optional,
    so they cannot assume the linear path.
  - Tighten the 5 required notebooks' narrative around APPLICATION: reframe
    every reference to an optional topic ("you built a transformer in T4" ->
    "transformers, covered in the optional deep-dive, ...").
  - KNOWN STALE REFS left by Phase 2b (must fix here): references to the
    OLD topic_4 transformers now read "Topic 4" but new Topic 4 is
    full_finetuning. Affected: topic_3_huggingface c0 ("In Topic 4 you
    assembled a Transformer"), c6/c7 (COMPLAINT_TOKENS "carried over from
    Topic 4"). Also all "Topic 7a"/"7b"/bare "Topic 7"/`T7b` shorthand
    referring to LoRA topics, and "Topic 3a/3b" attention refs.
- Phase 4: FIX issue 7 — new-T4 (old 6a) estimator.transformers_version bug.
- Phase 5: update planning/config docs (TOPICS.md, CLAUDE.md table, etc.).
- Phase 6: build new T8 agent capstone (small HF model in-kernel as the tool)
  via /run-research-topic -> /build-topic-notebook.
- Phase 7 (issue 6): split PyTorch Primer core vs optional.
- DEFERRED: zip regeneration, theory slides.
```
