# Topic-to-Notebook Mapping
# Generative AI for Developers - Barclays
# Last updated: 2026-05-11

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

## Day 1 Topics (runs in Studio notebook, no remote training)

### Topic 1: Overview of Generative AI (1 notebook)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 1 | Overview of Generative AI | none | BUILD FROM SCRATCH |

**Content**: What is GenAI, generative model families (GANs, VAEs, Diffusion, Autoregressive),
LLM-as-a-Service landscape (OpenAI/Anthropic APIs, tokens, temperature) - 30 min context-setting.
No heavy code - conceptual + interactive demos via API.
**Runs**: in Studio notebook (API calls only, no training).

- **Status**: not_started
- [ ] Run `/run-research-topic 1`
- [ ] Run `/build-topic-notebook 1`
- [ ] Run `/validate-notebooks 1`
- [ ] Run `/build-diagrams 1`

---

### Topic 2: Introducing LLMs (1 notebook)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 2 | Introducing LLMs | none | BUILD FROM SCRATCH |

**Content**: What lies behind ChatGPT, LLMs as Transformers, types of transformers and tasks,
famous models (BERT, GPT, T5, LLaMA overview), GenAI project lifecycle vs ML lifecycle.
**Runs**: in Studio notebook (conceptual + light inference demos, no training).

- **Status**: done
- [x] Run `/run-research-topic 2`
- [x] Run `/build-topic-notebook 2`
- [ ] Run `/validate-notebooks 2`
- [ ] Run `/build-diagrams 2`

---

### Topic 3: Introducing Attention (2 notebooks)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 3a | Seq2Seq and Bahdanau Attention (pure Python) | `Exercises/8_Attention.ipynb` | heavy adapt |
| 3b | Attention in PyTorch | `Exercises/9_Attention_with_Torch.ipynb` | heavy adapt |

**3a content**: word2vec recap, seq2seq limitations, Bahdanau attention, dot-product and scaled
dot-product attention in plain Python/numpy. Capstone: implement scaled dot-product attention from scratch.
**3b content**: PyTorch MultiheadAttention, custom DotProductAttention and MultiHeadAttention classes.
**Runs**: both in Studio notebook (no training jobs).

**Adaptation needed**: SageMaker env setup, four-beat arc restructure, STAR method labs, diagram placeholders.

- **Status**: done
- [x] Run `/run-research-topic 3`
- [x] Run `/build-topic-notebook 3`
- [x] Run `/validate-notebooks 3`
- [ ] Run `/build-diagrams 3`

---

### Topic 4: Transformers (1 notebook + first remote GPU job)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 4 | Transformer Architecture + Translator Capstone | `Exercises/11_Transformers_Translator.ipynb` | heavy adapt + remote GPU |

**Content**: why seq2seq+attention had drawbacks, multi-head attention deep dive, full transformer
architecture (positional encoding, layer norm, feed-forward, encoder/decoder), encoder vs decoder vs
encoder-decoder. Capstone: build a Transformer translator from scratch.
**Runs**: theory and architecture cells run in notebook; CAPSTONE TRAINS REMOTELY on `ml.g4dn.xlarge`
(first GPU remote training job in the course - introduces the pattern).

**Note**: `Exercises/10_NMT_with_Attention.ipynb` (NMT with cross-attention) can be used as
a Beat 1 "before transformers" reference in the theory section but is not a separate notebook.

**Adaptation needed**: major restructure, remote training integration, SageMaker estimator for capstone.

- **Status**: not_started
- [ ] Run `/run-research-topic 4`
- [ ] Run `/build-topic-notebook 4`
- [ ] Run `/validate-notebooks 4`
- [ ] Run `/build-diagrams 4`

---

## Day 2 Topics (mix of local and remote training)

### Topic 5: Hugging Face (1 notebook)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 5 | Hugging Face Hub and Ecosystem | none | BUILD FROM SCRATCH |

**Content**: HuggingFace Hub intro, datasets library, loading and using a model, uploading a checkpoint.
Light practical demos - pipeline API, tokenizers, model cards. No fine-tuning yet.
**Runs**: in Studio notebook (inference only, no training jobs).

- **Status**: not_started
- [ ] Run `/run-research-topic 5`
- [ ] Run `/build-topic-notebook 5`
- [ ] Run `/validate-notebooks 5`
- [ ] Run `/build-diagrams 5`

---

### Topic 6: Training LLMs - The Easy Part (2 notebooks)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 6a | Full Fine-Tuning + Catastrophic Forgetting | `Exercises/4-Finetuning.ipynb` (partial) | heavy adapt + remote GPU |
| 6b | Transfer Learning with DistilBERT | `Exercises/13_Transfer_Learning.ipynb` | heavy adapt + remote GPU |

**6a content**: when to train vs not, compute costs of LLM training, full fine-tuning of Flan-T5,
verifying catastrophic forgetting, single vs multitask fine-tuning.
Capstone: full fine-tune Flan-T5 remotely, measure forgetting.
**Runs**: REMOTE GPU (`ml.g4dn.xlarge`, HuggingFace estimator).

**6b content**: transfer learning to avoid full fine-tuning, DistilBERT on sentiment analysis.
Capstone: transfer learning of DistilBERT on SST-2.
**Runs**: REMOTE CPU (`ml.m5.xlarge`, PyTorch estimator) - intentional CPU demo.

**Note**: `Exercises/4-Finetuning.ipynb` covers GloVe fine-tuning (too basic); use only as
structural reference, not content. `Exercises/13_Transfer_Learning.ipynb` covers BERT on IMDB -
reuse structure, adapt to DistilBERT + SageMaker remote training.

**Adaptation needed**: major - remote training integration, SageMaker estimators, Flan-T5 target.

- **Status**: not_started
- [ ] Run `/run-research-topic 6`
- [ ] Run `/build-topic-notebook 6`
- [ ] Run `/validate-notebooks 6`
- [ ] Run `/build-diagrams 6`

---

### Topic 7: Training LLMs - PEFT (1 notebook)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 7 | PEFT: LoRA, QLoRA, Soft Prompts | none | BUILD FROM SCRATCH |

**Content**: introducing PEFT, LoRA explained (low-rank decomposition), QLoRA (quantization + LoRA),
soft prompts / prompt tuning. Capstone: fine-tune Flan-T5 with LoRA for NER and summarization.
**Runs**: REMOTE GPU (`ml.g4dn.xlarge`, HuggingFace estimator with PEFT library).

- **Status**: not_started
- [ ] Run `/run-research-topic 7`
- [ ] Run `/build-topic-notebook 7`
- [ ] Run `/validate-notebooks 7`
- [ ] Run `/build-diagrams 7`

---

## Day 3 Topics (remote training throughout)

### Topic 8: Deploying LLMs - Quantization (1 notebook)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 8 | Quantization and Serving | none | BUILD FROM SCRATCH |

**Content**: introducing quantization, post-training quantization vs quantization-aware training,
weight pruning in PyTorch. Capstone: quantization-aware training for LLMs with LoRA. Serving quantized models.
**Runs**: REMOTE GPU (`ml.g4dn.xlarge`).

- **Status**: not_started
- [ ] Run `/run-research-topic 8`
- [ ] Run `/build-topic-notebook 8`
- [ ] Run `/validate-notebooks 8`
- [ ] Run `/build-diagrams 8`

---

### Topic 9: RLHF - Time Permitting (1 notebook)

| # | Notebook | Existing source | Status |
|---|----------|----------------|--------|
| 9 | RLHF | none | BUILD FROM SCRATCH |

**Content**: RLHF intro, PPO algorithm, reward model, reward hacking and scaling issues.
**Runs**: REMOTE GPU if capstone included; otherwise Studio notebook for conceptual parts.
**Note**: time-permitting topic - design so it can be skipped without breaking flow.

- **Status**: not_started
- [ ] Run `/run-research-topic 9`
- [ ] Run `/build-topic-notebook 9`
- [ ] Run `/validate-notebooks 9`
- [ ] Run `/build-diagrams 9`

---

## Notebooks Not Used (from foundations-genai, retired for this course)

| Notebook | Reason not used |
|----------|----------------|
| `2-Text_Processing_Logistic_Regression_and_Boosting.ipynb` | Classical ML (logistic regression, boosting) - below course level, audience already knows this |
| `3-CBOW.ipynb` | CBOW word embeddings - too basic for this audience; word2vec is covered as a brief recap in Topic 3 |
| `5-NewsClassification_MLP.ipynb` | MLP text classification - below course level |
| `6_RentalGenerator_LSTM.ipynb` | LSTM text generation - covered by PyTorch Primer; not in outline |
| `7_NER_BiLSTM.ipynb` | BiLSTM NER - below course level for this audience |
| `10_NMT_with_Attention.ipynb` | NMT with cross-attention - used as Beat 1 reference in Topic 4, not a standalone notebook |
| `12_Prompt_Engineering.ipynb` | Prompt engineering with Flan-T5 - covered as part of Topic 6/7, not a standalone notebook |

---

## Summary Table

| Topic | # Notebooks | Remote Training | Existing coverage | Build effort |
|-------|-------------|-----------------|-------------------|--------------|
| P: PyTorch Reminder | 5 | none | full (adapt only) | low |
| S: SageMaker Reminder | 1 | demo job | none | high |
| 1: Overview of GenAI | 1 | none | none | medium |
| 2: Introducing LLMs | 1 | none | none | medium |
| 3: Introducing Attention | 2 | none | partial (8, 9) | medium |
| 4: Transformers | 1 | GPU (first job) | partial (11) | high |
| 5: Hugging Face | 1 | none | none | medium |
| 6: Training LLMs Easy | 2 | GPU + CPU | partial (4, 13) | high |
| 7: PEFT | 1 | GPU | none | high |
| 8: Quantization | 1 | GPU | none | high |
| 9: RLHF | 1 | GPU (optional) | none | medium |
| **Total** | **17** | **6 jobs** | | |
