# Phase 3 Megaplan - Course Restructure Notebook Rework

THE master spec. Every build agent reads this FIRST, then its per-notebook
design doc. Do not start a rework without reading this whole file.

Created 2026-05-17. Status: ACTIVE.

---

## 0. Context in one paragraph

A GenAI course (Jupyter notebooks) was restructured: attention/transformers/
LoRA-from-scratch were demoted from the required path into a standalone OPTIONAL
track, the required topics were renumbered, and the course now builds ONE
running Barclays complaint-intelligence system whose state is passed topic-to-
topic via S3. Phases 0-2 (directory rename, reference fixes) are committed.
Phase 3 is the per-notebook content rework. `topic_3_huggingface` is already
reworked and Codex-approved - it is the proven template.

## 1. The notebook map

REQUIRED PATH (linear, S3-chained):

| New topic | Slug | Status |
|-----------|------|--------|
| 1 | topic_1_overview_genai | rework pending |
| 2 | topic_2_introducing_llms | rework pending (gets the transformer mini-lesson) |
| 3 | topic_3_huggingface | DONE - reworked, Codex-passed (template) |
| 4 | topic_4_full_finetuning | rework pending (+ issue 7 estimator fix) |
| 5 | topic_5_transfer_learning | rework pending |
| 6 | topic_6_peft_lora_distilbert | rework pending (gets the LoRA mini-lesson) |
| 7 | topic_7_quantization | rework pending |
| 8 | topic_8 agent capstone | OUT OF SCOPE - new build, separate plan |

OPTIONAL TRACK (standalone, NOT S3-chained, mutually independent):

| Slug | Status |
|------|--------|
| topic_optional_attention_python | rework pending |
| topic_optional_attention_pytorch | rework pending |
| topic_optional_transformers | rework pending |
| topic_optional_lora_ffn | rework pending (design doc not yet written) |

Every notebook has an Exercise copy and a Solution twin. One agent does BOTH
copies of its notebook as a pair.

## 2. Design docs - which agent reads which

Each agent reads MEGAPLAN.md (this file) + NOTEBOOK_EDIT_PROTOCOL.md +
its design doc:

| Notebook | Design doc |
|----------|-----------|
| required topics 1-7 | plans/refactor/required_path_continuity.md (one doc, per-topic sections) |
| topic_optional_attention_python | plans/refactor/optional_attention_python.md |
| topic_optional_attention_pytorch | plans/refactor/optional_attention_pytorch.md |
| topic_optional_transformers | plans/refactor/optional_transformers.md |
| topic_optional_lora_ffn | NOT YET WRITTEN - must be authored before rework |

## 3. Dispatch order (decided)

- OPTIONALS: mutually independent (no S3 chain, no shared state). The 3 with
  design docs may be reworked in PARALLEL - one subagent each.
- REQUIRED TOPICS: SERIAL, in numeric order 1 -> 2 -> 4 -> 5 -> 6 -> 7
  (topic_3 already done). They MUST be serial: the S3 handoff chain means
  topic N writes the artifact topic N+1 loads. Reworking N+1 before N is
  finalized risks drift against an artifact contract that is not yet locked.
- topic_8 capstone: not in this megaplan.

## 4. The S3 handoff chain (continuity contract)

This is the spiral. Every required notebook LOADS the previous topic's artifact
and WRITES its own. Key layout: `s3://<bucket>/barclays-course/topic_<N>/<file>`.

| Topic | LOADS from | WRITES |
|-------|-----------|--------|
| 1 | (nothing - first) | topic_1/triage_config.json (system prompt + test complaints) |
| 2 | topic_1/triage_config.json | topic_2/complaint_corpus.json |
| 3 | topic_2/complaint_corpus.json | topic_3/labelled_dataset.json + routing_labels.json |
| 4 | topic_3/labelled_dataset.json | topic_4/model_pointer.json (fine-tuned model URI) |
| 5 | topic_4/model_pointer.json | topic_5/model_pointer.json (transfer-learned) |
| 6 | topic_5/model_pointer.json | topic_6/model_pointer.json (PEFT checkpoint) |
| 7 | topic_6/model_pointer.json | topic_7/deployment.json (final endpoint) |

Hard rules for the handoff cells:
- LOAD cell goes right after the imports/setup cell. WRITE cell goes right
  before the end-of-topic footer.
- LOAD must have a FALLBACK: if the S3 artifact is absent (student starting
  mid-course), define a minimal local version and print a clear note. Same
  spirit as a lab safety-net cell. A mid-course student is never blocked.
- The OpenAI client is never serialized - it is recreated from the API key in
  each notebook. Only data artifacts (prompts, datasets, model URIs) persist.
- Optionals do NOT participate in the S3 chain - they are standalone.

## 5. What every subagent does (the standard procedure)

Each subagent, for its assigned notebook:

1. READ, in order: this MEGAPLAN.md, ~/.claude/NOTEBOOK_EDIT_PROTOCOL.md,
   CLAUDE.md, plans/CORE_TECHNOLOGIES_AND_DECISIONS.md,
   plans/SAGEMAKER_LESSONS_LEARNED.md, and its design doc (section 2).
2. Invoke rework mode:
   `/build-topic-notebook <N> --base-from <Exercise notebook> --design-doc <doc>`
   (For optionals, N is the slug; the pipeline routes via the `rework` state.)
3. Follow NOTEBOOK_EDIT_PROTOCOL.md for EVERY cell change:
   - normalize ids first
   - locate each cell by id AND assert its current ("OLD") content
   - apply the design doc's KEEP/EDIT/NEW/DELETE/MERGE action
   - control insert position explicitly
   - read back from disk and assert after every edit
4. Apply changes in batches of 5; between batches run the gates:
   nbformat.validate + per-cell ast.parse + concatenated-pyflakes (0 undefined
   names). Show the user the batch diff and wait for approval per the
   5-cell-checkpoint rule. /save-state at every 10-cell boundary.
5. Do the EXERCISE copy first, then apply the SAME structural changes to the
   SOLUTION twin. Lab cells in the solution stay filled; safety-net cells per
   the design doc. End with pair-parity check (same cell count, same cell-type
   sequence).
6. For required topics: verify the S3 LOAD/WRITE cells against section 4 - the
   LOAD reads the correct previous-topic artifact, the WRITE produces exactly
   what the next topic's LOAD expects.
7. Run `/build-diagrams` in rework mode: if the notebook's diagram cells were
   not changed and the diagrams already exist (inline Mermaid + .mmd files),
   this is a no-op - just verify the 2 diagrams resolve. If the rework ADDED a
   `<!-- DIAGRAM: -->` placeholder, build that diagram.
8. Commit the notebook pair with a message naming the topic and the design doc.
9. Hand off: report which artifact the topic now WRITES to S3, so the next
   agent in the chain knows the contract is locked.

## 6. Per-notebook agent assignments

### Optionals (parallel - 3 agents)

- AGENT-OPT-ATTN-PY: topic_optional_attention_python.
  Doc: optional_attention_python.md. Make it standalone: offline NLTK fallback,
  no Topic-2/3b chaining, supplementary banner. NOT S3-chained.
- AGENT-OPT-ATTN-PT: topic_optional_attention_pytorch.
  Doc: optional_attention_pytorch.md. Must DEFINE scaled_dot_product_attention
  and INTRODUCE nn.MultiheadAttention in-notebook (cold-run safe). NOT S3-chained.
- AGENT-OPT-TRANSFORMERS: topic_optional_transformers.
  Doc: optional_transformers.md. AWS-credentials guard cell before the GPU
  capstone; nn.MultiheadAttention introduced in-notebook. NOT S3-chained.

(topic_optional_lora_ffn: design doc must be written first - not yet an agent.)

### Required topics (serial - one agent at a time, in this order)

- AGENT-T1: topic_1_overview_genai. Doc: required_path_continuity.md Topic 1
  section. No LOAD cell (first topic). Adds the WRITE cell ->
  topic_1/triage_config.json. Replaces the old "Variables available for
  Topic 2" banner with a real S3 write.
- AGENT-T2: topic_2_introducing_llms. Doc: Topic 2 section + the R7 TRANSFORMER
  MINI-LESSON (6 new cells, Section 5.5). LOAD topic_1/triage_config.json,
  WRITE topic_2/complaint_corpus.json. The mini-lesson is the concept-level
  transformer teaching the required path depends on - it must land before any
  later topic references attention/CLS concepts.
- (topic_3 - DONE, skip.)
- AGENT-T4: topic_4_full_finetuning. Doc: Topic 4 section. LOAD
  topic_3/labelled_dataset.json, WRITE topic_4/model_pointer.json. ALSO fixes
  issue 7: the estimator.transformers_version AttributeError - print the
  literal version constants, do not read them back off the estimator.
- AGENT-T5: topic_5_transfer_learning. Doc: Topic 5 section. LOAD
  topic_4/model_pointer.json, WRITE topic_5/model_pointer.json.
- AGENT-T6: topic_6_peft_lora_distilbert. Doc: Topic 6 section + the R8 LoRA
  MINI-LESSON (3 new cells, after cell 5). LOAD topic_5/model_pointer.json,
  WRITE topic_6/model_pointer.json. The mini-lesson teaches rank/alpha/dropout
  so the topic does not depend on the optional lora_ffn notebook.
- AGENT-T7: topic_7_quantization. Doc: Topic 7 section. LOAD
  topic_6/model_pointer.json, WRITE topic_7/deployment.json. Fix the stale
  end-of-course tables (drop T3a/T3b/Flan-T5/T9 references), rebuild to
  Topics 1-7.

## 7. Definition of done (per notebook)

- All design-doc changes applied; nothing missing or half-applied.
- nbformat.validate passes; per-cell ast.parse passes; concatenated-pyflakes
  reports 0 undefined names.
- Exercise and Solution: equal cell count, matching cell-type sequence.
- For required topics: S3 LOAD reads the right artifact, WRITE produces what
  the next topic expects (section 4 contract).
- Diagrams resolve (2 per notebook unless the rework added one).
- Committed.
- A Codex (o3) review via /codex-consult passes with no corrective action
  (this is how topic_3 was signed off).

## 8. What is NOT in this megaplan

- topic_8 agent capstone (new from-scratch build - separate plan, needs
  /run-research-topic first).
- topic_optional_lora_ffn rework (its design doc is not written yet).
- Phase 5 doc updates (TOPICS.md, CLAUDE.md curriculum table).
- Phase 7 PyTorch Primer split.
- zip regeneration, theory slides (deferred).
