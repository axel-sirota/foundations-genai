---
slug: genai-course-restructure
saved_at: 2026-05-17
context_pct_at_save: 70%+
repo: /Users/axelsirota/repos/genai_for_developers
branch: master
---

## Goal

Rework and complete the GenAI-for-developers course: demote attention/transformers
to an optional track, renumber the required path, S3-chain the topics, build the
agent capstone. STATUS: structurally complete.

## State

- DONE: all 7 delivery issues (ISSUES_TO_FIX.md, all marked [DONE]).
- DONE: 7 required topics (1-7) reworked + renumbered + S3-chained.
- DONE: 4 optional notebooks reworked (attention_python, attention_pytorch,
  transformers, lora_ffn).
- DONE: topic_8 agent capstone BUILT (Exercise 13 cells, Solution 27 cells).
- DONE: 2 mini-lessons (transformer concepts in topic_2, LoRA in topic_6).
- DONE: Codex o3 continuity review + /verify sweep - all findings fixed.
- DONE: topic_1 fully converted 5-class -> 4-class taxonomy.
- Last commit: 3fa94fc (continuity fixes).

## In-flight reasoning

- The course is DONE. Only housekeeping + one optional item remain.
- ~6 commits are ahead of origin/master, UNPUSHED: includes 0faa2b1,
  1a7d5c9, 59d38fd, 3fa94fc. Next real action the user asked for: `git push`.
- The user then typed `/build-diagrams 8` and asked "what is this?" - the
  pipeline gate BLOCKED it (needs a validate-8.done marker first). See
  "Next concrete step".

## Failed approaches (do NOT retry)

- Do NOT use blind index-based bulk scripts to edit notebooks - forbidden by
  ~/.claude/NOTEBOOK_EDIT_PROTOCOL.md. Use located+asserted+read-back edits.
- Do NOT trust NotebookEdit on notebooks >25k tokens - its Read gate fails;
  use audited in-place JSON edits instead (per the protocol).

## Open questions / blockers

- [ ] User asked "what is /build-diagrams 8?" and the gate blocked it. Need to
  answer that, AND decide whether to actually build the topic_8 agent-loop.mmd
  diagram (placeholder exists in the notebook, .mmd file does not yet).

## Next concrete step

1. Answer the user: /build-diagrams generates Mermaid .mmd files from the
   <!-- DIAGRAM: --> placeholders. It was blocked because the pipeline gate
   wants a validate-8.done marker (chain: build -> validate-notebooks ->
   build-diagrams). topic_8 has ONE placeholder (agent-loop, cell 7).
2. Then: `git push origin master` (the ~6 unpushed commits).
3. Optional: /build-diagrams 8 (after /validate 8) to generate
   plans/topic_8_agent_capstone/diagrams/agent-loop.mmd.

## Don't re-litigate

- Capstone design: pure-Python ReAct agent, gpt-4o brain, 3 tools, RAG cut to
  Homework Extension. Settled via 4-cycle /research. Plan: plans/topic_8_agent_capstone.md.
- 4-class taxonomy (fraud and security / billing and charges / account access /
  general enquiry) is canonical course-wide. Settled.
- Safety-net cells STAY in solution notebooks (CLAUDE.md wins over the old
  validate-notebooks rule). Settled in commit 673a406.
- MEGAPLAN.md, NOTEBOOK_EDIT_PROTOCOL.md, restructure_course.md are the specs.
