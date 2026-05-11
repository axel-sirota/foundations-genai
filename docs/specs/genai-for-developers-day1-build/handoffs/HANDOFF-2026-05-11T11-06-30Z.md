---
slug: genai-for-developers-day1-build
saved_at: '2026-05-11T11:06:30Z'
type: side-save
---

# Side-save: genai-for-developers-day1-build

*Written by /side-save at 2026-05-11T11:05:50Z.*
*Use `/side-resume genai-for-developers-day1-build` to restore context.*

## Goal
Build all 13 notebooks for the Barclays "Generative AI for Developers" 3-day course running in AWS SageMaker Studio.

## State
- Completed:
  - Course mapping (INDEX.md, plans/TOPICS.md, plans/CORE_TECHNOLOGIES_AND_DECISIONS.md, plans/SAGEMAKER_LESSONS_LEARNED.md, plans/DEFINITION_OF_DONE.md)
  - SDD_WORKFLOW.md + pipeline.yaml + .claude/settings.json (ALLOW_DIRECT_MAIN=1)
  - Research plans for ALL Day 1 topics: researches/F1_pytorch_refresher.md, F2_sagemaker_fundamentals.md, topic_1_overview_genai.md, topic_2_introducing_llms.md, topic_3a_attention_python.md, topic_3b_attention_pytorch.md
  - Research plan + exercise + solution notebooks for Topic 4: researches/topic_4_transformers.md, Exercises/topic_4_transformers/topic_4_transformers.ipynb (27 cells), Solutions/topic_4_transformers/topic_4_transformers.ipynb (27 cells, parity OK)
  - scripts_topic4/train.py + requirements.txt (PyTorch estimator, ml.g4dn.xlarge)
- In-flight: User requested background research agents for ALL remaining topics (F1, F2, Topics 1, 2, 3a, 3b) invoking run-research-topic skill to write plans to researches/
- Remaining: Build notebooks for F1, F2, Topics 1-3b from research plans; Topics 5-9 research + build

## In-flight reasoning
- Day 1 research plans are done but notebooks not yet built from them
- Topic 4 notebooks are built and clean (AI-tells CLEAN, parity OK, safety-nets in both)
- The pattern that worked: research agent writes to researches/<slug>.md, then exercise builder polls for it, solution builder polls for exercise
- Exercise builder must NOT self-generate plan if polling times out -- it failed once and wrote its own plan. Fix: increase poll attempts or have it wait longer.
- The dedicated research agent hit 32k token limit once -- fixed by tightening prompt. Keep research prompts focused with "write to file, not stdout"

## Failed approaches (do NOT retry)
- Research agent with very long exhaustive prompt -> hit 32k output token limit, wrote nothing. Fix: tighter prompt, concise cell descriptions.
- Exercise builder using Monitor tool to wait for plan -> Monitor returned immediately, agent exited without building. Fix: explicit sleep/poll loop in bash.
- Solution builder ran before exercise was ready -> wrote 25-cell solution missing 2 safety-nets. Fix: always verify parity + restore missing cells manually.

## Open questions / blockers
- [ ] Topic 4 research plan has French narrative but notebooks use Spanish -- patched with NOTE comment in plan. Minor inconsistency.
- [ ] User asked to launch background agents for remaining topics -- not yet done (saving first due to context at 70%+)

## Next concrete step
Launch 6 background research agents (one per remaining Day 1 topic: F1, F2, topic_1, topic_2, topic_3a, topic_3b) each reading their researches/<slug>.md plan and invoking the run-research-topic workflow to produce a proper plans/topic_N_<slug>.md file in the plans/ folder (not researches/) -- OR if user wants notebooks built directly from the researches/ files, launch 6 build agents instead. Clarify with user which they want.

## Don't re-litigate
- ALLOW_DIRECT_MAIN=1: set in .claude/settings.json, no feature branches for this repo
- No emojis anywhere: CLAUDE.md wins over build-topic-notebook.md which says emojis ok in headers
- Safety-net cells stay in solution: CLAUDE.md rule, do NOT delete them
- PyTorch estimator not HuggingFace for custom models: L1 in SAGEMAKER_LESSONS_LEARNED.md
- py_version="py312" for framework_version="2.8.0": L2 in lessons learned
- SageMaker SDK pinned >=2.200.0,<3.0.0: L3 in lessons learned
- numpy<2 in every install cell: hard rule
- No evaluate library, inline numpy for metrics: L6 in lessons learned
- Diagram slugs in plan must match what notebooks actually reference: learned from Topic 4 conflict
