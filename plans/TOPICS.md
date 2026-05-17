# Topic-to-Notebook Mapping
# Generative AI for Developers - Barclays
# Last updated: 2026-05-17 (Phase 5 - post-restructure renumbering)

The course was restructured (see plans/restructure_course.md): attention,
transformers, and LoRA-from-scratch were demoted from the required path into a
standalone OPTIONAL theory track, the required topics were renumbered, and the
course now builds ONE running Barclays complaint-intelligence system whose
state is passed topic-to-topic via S3.

## Framework Folders (Prerequisites - not in the 3-day outline)
<!-- F1 Status: done -->

These run in SageMaker Studio locally (no remote training). Adapted from PytorchPrimer.

### Frameworks/PyTorchReminder/ (5 notebooks)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| P1 | PyTorch Tensors | `PytorchPrimer/PyTorch_1_Exercise_PyTorch_Tensors.ipynb` | adapt to SageMaker env |
| P2 | Autograd and GPU | `PytorchPrimer/PyTorch_2_Exercise_Autograd_GPU.ipynb` | adapt to SageMaker env |
| P3 | Dataset and DataLoader | `PytorchPrimer/PyTorch_3_Exercise_PyTorch_Dataset_Dataloader.ipynb` | adapt to SageMaker env |
| P4 | Classifier with nn.Linear | `PytorchPrimer/PyTorch_4_Exercise_Classifier_nnLinear.ipynb` | adapt to SageMaker env |
| P5 | Classifier with nn.Sequential | `PytorchPrimer/PyTorch_5_Exercise_nnSequential_Classifier.ipynb` | adapt to SageMaker env |

**Adaptation needed**: replace Colab/local setup cell with SageMaker session setup; pin `numpy<2`.
No remote training - all runs in the notebook kernel.

### Frameworks/SageMakerReminder/ (1 notebook - BUILD FROM SCRATCH)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| S1 | SageMaker Fundamentals | none | BUILD FROM SCRATCH |

**Content**: SageMaker session setup, S3 read/write, launching a CPU training job, MLflow tracking,
model registry, endpoint deployment. Uses exact patterns from CORE_TECHNOLOGIES_AND_DECISIONS.md.
This is the only notebook that teaches "how to use SageMaker" before students see it in capstones.

---

## Required Path (linear, S3-chained)

Every required notebook LOADS the previous topic's S3 artifact and WRITES its
own. Key layout: `s3://<bucket>/barclays-course/topic_<N>/<file>`.

| New topic | Slug | Old slug | LOADS | WRITES |
|-----------|------|----------|-------|--------|
| 1 | `topic_1_overview_genai` | topic_1 | (nothing - first) | topic_1/triage_config.json |
| 2 | `topic_2_introducing_llms` | topic_2 | topic_1/triage_config.json | topic_2/complaint_corpus.json |
| 3 | `topic_3_huggingface` | topic_5 | topic_2/complaint_corpus.json | topic_3/labelled_dataset.json + routing_labels.json |
| 4 | `topic_4_full_finetuning` | topic_6a | topic_3/labelled_dataset.json | topic_4/model_pointer.json |
| 5 | `topic_5_transfer_learning` | topic_6b | topic_4/model_pointer.json | topic_5/model_pointer.json |
| 6 | `topic_6_peft_lora_distilbert` | topic_7b | topic_5/model_pointer.json | topic_6/model_pointer.json |
| 7 | `topic_7_quantization` | topic_8 | topic_6/model_pointer.json | topic_7/deployment.json |
| 8 | `topic_8_agent_capstone` | NEW | topic_7/deployment.json | (capstone - final) |

### Topic 1: Overview of Generative AI

**Content**: What is GenAI, generative model families (GANs, VAEs, Diffusion, Autoregressive),
LLM-as-a-Service landscape (OpenAI APIs, tokens, temperature). No heavy code - conceptual +
interactive demos via API. Writes the triage config (system prompt + test complaints) to S3.
**Runs**: in Studio notebook (API calls only, no training).
**Status**: built; Phase 3 narrative rework pending.

### Topic 2: Introducing LLMs

**Content**: What lies behind ChatGPT, LLMs as Transformers, types of transformers and tasks,
famous models (BERT, GPT, T5, LLaMA overview), GenAI project lifecycle. Includes the
concept-level transformer mini-lesson (so the required path does not depend on the optional
attention/transformer notebooks).
**Runs**: in Studio notebook (conceptual + light inference demos, no training).
**Status**: built; Phase 3 narrative rework pending (gets the transformer mini-lesson).

### Topic 3: Hugging Face

**Content**: HuggingFace Hub intro, datasets library, loading and using a model, pipeline API,
tokenizers, model cards. Produces the 4-class labelled routing dataset for downstream topics.
**Runs**: in Studio notebook (inference only, no training jobs).
**Status**: DONE - reworked and Codex-approved (the proven Phase 3 template).

### Topic 4: Full Fine-Tuning

**Content**: when to train vs not, compute costs, full fine-tuning, catastrophic forgetting,
single vs multitask fine-tuning. Capstone: full fine-tune remotely, measure forgetting.
Includes the issue-7 fix (estimator.transformers_version AttributeError).
**Runs**: REMOTE GPU (HuggingFace estimator).
**Status**: built; Phase 3 narrative rework pending.

### Topic 5: Transfer Learning

**Content**: transfer learning to avoid full fine-tuning, DistilBERT on the routing task.
Capstone: transfer learning of DistilBERT.
**Runs**: REMOTE CPU (PyTorch estimator) - intentional CPU demo.
**Status**: built; Phase 3 narrative rework pending.

### Topic 6: PEFT / LoRA on DistilBERT

**Content**: introducing PEFT, LoRA explained (low-rank decomposition), rank/alpha/dropout.
Includes the LoRA mini-lesson so the topic does not depend on the optional lora_ffn notebook.
Capstone: PEFT/LoRA fine-tune of DistilBERT.
**Runs**: REMOTE GPU (HuggingFace estimator with PEFT library).
**Status**: built; Phase 3 narrative rework pending.

### Topic 7: Quantization and Serving

**Content**: introducing quantization, post-training quantization vs quantization-aware
training, weight pruning, serving quantized models. Writes the final deployment artifact.
**Runs**: REMOTE GPU.
**Status**: built; Phase 3 narrative rework pending (drop stale end-of-course tables).

### Topic 8: Agent Capstone

**Content**: a pure-Python ReAct agent (no frameworks), gpt-4o brain, with 3 tools
that surface the student's prior work - a code tool, a fine-tuned classifier, and a
PEFT classifier - each loading from the S3 handoff with a public-model fallback.
Ties back to Topics 1-7.
**Runs**: in Studio notebook (OpenAI API + in-kernel HuggingFace pipelines).
**Plan**: `plans/topic_8_agent_capstone.md`.
**Status**: BUILT - exercise + solution notebooks built and verified.

---

## Optional Theory Track (standalone, NOT S3-chained, mutually independent)

For advanced learners, taught from slides. Each notebook is self-contained: no
sequential "next topic" narrative, no S3 handoff, self-contained recaps.

| Slug | Old slug | Content |
|------|----------|---------|
| `topic_optional_attention_python` | topic_3a | Seq2Seq limitations, Bahdanau attention, dot-product and scaled dot-product attention in plain Python/numpy. |
| `topic_optional_attention_pytorch` | topic_3b | Attention in PyTorch: scaled_dot_product_attention, nn.MultiheadAttention, custom attention classes. |
| `topic_optional_transformers` | topic_4 | Full transformer architecture (positional encoding, layer norm, feed-forward, encoder/decoder); translator capstone on GPU. |
| `topic_optional_lora_ffn` | topic_7a | LoRA from scratch on a feed-forward network: low-rank matrices A/B, rank r. |

**Status**: attention_python / attention_pytorch / transformers have Phase 3
design docs and rework pending; `topic_optional_lora_ffn` design doc not yet written.

---

## Summary Table

| Topic | Track | Remote Training | Build status |
|-------|-------|-----------------|--------------|
| P: PyTorch Reminder | prereq | none | done (adapt only) |
| S: SageMaker Reminder | prereq | demo job | built |
| 1: Overview of GenAI | required | none | built, rework pending |
| 2: Introducing LLMs | required | none | built, rework pending |
| 3: Hugging Face | required | none | DONE (reworked) |
| 4: Full Fine-Tuning | required | GPU | built, rework pending |
| 5: Transfer Learning | required | CPU | built, rework pending |
| 6: PEFT / LoRA | required | GPU | built, rework pending |
| 7: Quantization | required | GPU | built, rework pending |
| 8: Agent Capstone | required | none (in-kernel) | not built |
| optional: attention_python | optional | none | rework pending |
| optional: attention_pytorch | optional | none | rework pending |
| optional: transformers | optional | GPU | rework pending |
| optional: lora_ffn | optional | GPU | design doc pending |
