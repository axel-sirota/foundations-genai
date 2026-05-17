---
slug: setup
saved_at: '2026-05-11T03:33:01Z'
type: side-save
---

# Side-save: setup

*Written by /side-save at 2026-05-11T03:32:27Z.*
*Use `/side-resume setup` to restore context.*

## Goal
Complete course setup for "Generative AI for Developers" (Barclays, 3-day intensive) — mapping, tooling, and all course notebooks built and validated (8 required topics + 4 optional + framework prereqs).

## State
- Completed:
  - Pipeline hooks fixed (pipeline_gate.py + pipeline_suggest.py now respect project-local pipeline.yaml)
  - .claude/pipeline.yaml created with custom notebook course chain
  - .claude/settings.json created (ALLOW_DIRECT_MAIN=1, permissions)
  - .claude/hooks/ symlinked to global hooks
  - .claude/commands/ populated: run-research-topic, build-topic-notebook, validate-notebooks, verify-research, build-diagrams, verify
  - SDD_WORKFLOW.md created at repo root
  - CLAUDE.md updated with full teaching methodology (four-beat arc, lab tiers, STAR, safety-net cells, 5-cell cadence, # YOUR CODE hygiene, peer discussion prompts, homework extensions, optional deep-dive notebooks, no markdown chain, NotebookEdit cell_id rule, build order)
  - plans/CORE_TECHNOLOGIES_AND_DECISIONS.md created
  - plans/DEFINITION_OF_DONE.md created
  - plans/SAGEMAKER_LESSONS_LEARNED.md created (14 lessons L1-L14)
  - INDEX.md created at repo root with full notebook mapping
  - plans/TOPICS.md created with per-topic SDD checklists
  - run-research-topic.md and build-topic-notebook.md updated to mandate reading the 3 plans files

- In-flight: mapping complete, ready to start SDD workflow on first notebook
- Remaining: build all course notebooks via SDD pipeline (F1, F2, required Topics 1-8, optional track)

## Notebook Map (confirmed)

NOTE (2026-05-17, Phase 5): the course was restructured and renumbered after
this handoff was written. The table below has been updated to the post-
restructure numbering. See plans/restructure_course.md and plans/TOPICS.md.

Required path (linear, S3-chained):

| ID | Notebook | Source |
|----|----------|--------|
| F1 | Frameworks/pytorch_refresher.ipynb | condense PytorchPrimer 1-5 + add HF Trainer |
| F2 | Frameworks/sagemaker_fundamentals.ipynb | BUILD FROM SCRATCH |
| 1 | topic_1_overview_genai | BUILD FROM SCRATCH |
| 2 | topic_2_introducing_llms | BUILD FROM SCRATCH (slides ref) |
| 3 | topic_3_huggingface | BUILD FROM SCRATCH (was topic_5) |
| 4 | topic_4_full_finetuning | adapt Exercises/4-Finetuning.ipynb (was topic_6a) |
| 5 | topic_5_transfer_learning | adapt Exercises/13_Transfer_Learning.ipynb (was topic_6b) |
| 6 | topic_6_peft_lora_distilbert | adapt 12_PEFT_LoRA_DistillBert.ipynb (was topic_7b) |
| 7 | topic_7_quantization | adapt mastering-llm-deployments Lab4-6 scripts (was topic_8) |
| 8 | topic_8_agent_capstone | NEW BUILD (small HF model in-kernel) |

Optional theory track (standalone, NOT S3-chained):

| Slug | Source |
|------|--------|
| topic_optional_attention_python | adapt Exercises/8_Attention.ipynb (was topic_3a) |
| topic_optional_attention_pytorch | adapt Exercises/9_Attention_with_Torch.ipynb (was topic_3b) |
| topic_optional_transformers | adapt Exercises/11_Transformers_Translator.ipynb (was topic_4) |
| topic_optional_lora_ffn | adapt 11_Simplified_LoRA_FFN.ipynb (was topic_7a) |

(topic_9_rlhf remains PARKED / dropped.)

## Remote Training Split
- Local (Studio kernel): F1, Topics 1, 2, 3, 8; optional attention notebooks
- Remote CPU (ml.m5.xlarge, PyTorch estimator): F2 (demo), Topic 5
- Remote GPU (ml.g4dn.xlarge): Topics 4, 6, 7; optional transformers + lora_ffn

## Key Technical Decisions
- SageMaker SDK pinned >=2.200.0,<3.0.0 (v3 breaks get_execution_role)
- PyTorch estimator: framework_version="2.8.0", py_version="py312"
- HuggingFace estimator: GPU only (ml.g4dn.xlarge minimum)
- MLflow: MlflowVersion="2.13.2" only supported in us-west-2
- eval_strategy not evaluation_strategy (transformers 4.41+)
- No evaluate library — inline numpy for metrics
- requirements.txt exact name in source_dir (not requirements_cpu.txt etc.)
- boto3 exception: ResourceNotFound not ResourceNotFoundException
- numpy<2 in every install cell

## Open questions / blockers
- [ ] Need to decide order: start with F1+F2 first, or Topic 1?

## Next concrete step
Run `/run-research-topic` on F1 (PyTorch Refresher) or F2 (SageMaker Fundamentals) — confirm with user which to start with.

## Don't re-litigate
- Pipeline: project-local pipeline.yaml overrides global — fixed in both hooks
- No feature branches: ALLOW_DIRECT_MAIN=1 in settings.json
- HuggingFace estimator CPU: never, always GPU (L1 in lessons learned)
