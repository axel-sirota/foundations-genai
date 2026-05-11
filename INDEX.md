# Generative AI for Developers - Course Index
# Barclays | 3-day intensive | Last updated: 2026-05-11

---

## Frameworks (Prerequisites)

| # | Notebook | Path | Remote Training | Reuse | Status |
|---|----------|------|-----------------|-------|--------|
| F1 | PyTorch Refresher | `Frameworks/pytorch_refresher.ipynb` | none (Studio kernel) | condense PytorchPrimer 1-5 + add HF Trainer section | not started |
| F2 | SageMaker Fundamentals | `Frameworks/sagemaker_fundamentals.ipynb` | CPU (`ml.m5.xlarge`) | BUILD FROM SCRATCH | not started |

**F1 - PyTorch Refresher**: tensors, autograd, Dataset/DataLoader, training loop with nn.Module, training with HuggingFace Trainer. Condenses PytorchPrimer 5 exercises into one focused notebook.

**F2 - SageMaker Fundamentals**: SageMaker session setup, S3 read/write, remote CPU training job with PyTorch estimator, MLflow tracking, model registry, endpoint deployment. The one notebook students must internalize before Day 2 capstones.

---

## Day 1 - Foundations (all run in Studio notebook, no remote training)

| # | Notebook | Path | Remote Training | Reuse | Status |
|---|----------|------|-----------------|-------|--------|
| 1 | Overview of Generative AI | `Exercises/topic_1_overview_genai/topic_1_overview_genai.ipynb` | none | BUILD FROM SCRATCH | not started |
| 2 | Introducing LLMs | `Exercises/topic_2_introducing_llms/topic_2_introducing_llms.ipynb` | none | BUILD FROM SCRATCH (slides exist as reference) | not started |
| 3a | Seq2Seq and Bahdanau Attention | `Exercises/topic_3a_attention_python/topic_3a_attention_python.ipynb` | none | adapt `Exercises/8_Attention.ipynb` | not started |
| 3b | Attention in PyTorch | `Exercises/topic_3b_attention_pytorch/topic_3b_attention_pytorch.ipynb` | none | adapt `Exercises/9_Attention_with_Torch.ipynb` | not started |
| 4 | Transformers + Translator Capstone | `Exercises/topic_4_transformers/topic_4_transformers.ipynb` | GPU `ml.g4dn.xlarge` | adapt `Exercises/11_Transformers_Translator.ipynb` + add remote GPU capstone | not started |

---

## Day 2 - Fine-Tuning (remote training begins)

| # | Notebook | Path | Remote Training | Reuse | Status |
|---|----------|------|-----------------|-------|--------|
| 5 | Hugging Face Ecosystem | `Exercises/topic_5_huggingface/topic_5_huggingface.ipynb` | none | BUILD FROM SCRATCH (inference only - pipeline, AutoModel, Hub) | not started |
| 6a | Full Fine-Tuning + Forgetting | `Exercises/topic_6a_full_finetuning/topic_6a_full_finetuning.ipynb` | GPU `ml.g4dn.xlarge` | adapt `Exercises/4-Finetuning.ipynb` - replace GloVe with small encoder LLM, show catastrophic forgetting | not started |
| 6b | Transfer Learning with DistilBERT | `Exercises/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb` | CPU `ml.m5.xlarge` | adapt `Exercises/13_Transfer_Learning.ipynb` - swap BERT/IMDB for DistilBERT/SST-2, add remote CPU training | not started |
| 7a | LoRA on Feed-Forward Networks | `Exercises/topic_7a_lora_ffn/topic_7a_lora_ffn.ipynb` | GPU `ml.g4dn.xlarge` | adapt `~/repos/finetuning-llms-hf/3-LLMs/11_Simplified_LoRA_FFN.ipynb` | not started |
| 7b | PEFT LoRA on DistilBERT | `Exercises/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb` | GPU `ml.g4dn.xlarge` | adapt `~/repos/finetuning-llms-hf/3-LLMs/12_PEFT_LoRA_DistillBert.ipynb` | not started |

---

## Day 3 - Deployment and Alignment

| # | Notebook | Path | Remote Training | Reuse | Status |
|---|----------|------|-----------------|-------|--------|
| 8 | Quantization, Pruning and Distillation | `Exercises/topic_8_quantization/topic_8_quantization.ipynb` | GPU `ml.g4dn.xlarge` | adapt scripts from `~/repos/mastering-llm-deployments/LLM_Deployments_Course/Lab6_Quantization/`, `Lab5_Pruning/`, `Lab4_Distillation_TrainingLoop/` | not started |
| 9 | RLHF | `Exercises/topic_9_rlhf/topic_9_rlhf.ipynb` | GPU `ml.g4dn.xlarge` | PARKED | parked |

---

## Solutions

Each exercise notebook has a matching solution:

| Exercise path | Solution path |
|---------------|---------------|
| `Exercises/topic_N_<slug>/topic_N_<slug>.ipynb` | `Solutions/topic_N_<slug>/topic_N_<slug>.ipynb` |
| `Frameworks/pytorch_refresher.ipynb` | `Frameworks/pytorch_refresher_solution.ipynb` |
| `Frameworks/sagemaker_fundamentals.ipynb` | `Frameworks/sagemaker_fundamentals_solution.ipynb` |

---

## Existing Notebooks (foundations-genai origin, retired for this course)

| Notebook | Reason |
|----------|--------|
| `Exercises/2-Text_Processing_*.ipynb` | Classical ML - below course level |
| `Exercises/3-CBOW.ipynb` | Word2vec - brief recap only in Topic 3a |
| `Exercises/5-NewsClassification_MLP.ipynb` | MLP classification - below course level |
| `Exercises/6_RentalGenerator_LSTM.ipynb` | LSTM generation - not in outline |
| `Exercises/7_NER_BiLSTM.ipynb` | BiLSTM NER - below course level |
| `Exercises/10_NMT_with_Attention.ipynb` | Used as Beat 1 reference in Topic 4 only |
| `Exercises/12_Prompt_Engineering.ipynb` | Folded into Topics 6/7 |
| `PytorchPrimer/` | Condensed into F1 (PyTorch Refresher) |
