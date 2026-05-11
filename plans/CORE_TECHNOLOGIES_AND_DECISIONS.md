# Core Technologies and Decisions
# Generative AI for Developers - Barclays
# Last updated: 2026-05-11

## Course Identity

- **Audience**: Developers with 2+ years Python, PyTorch fundamentals, deep learning basics. NOT beginners.
- **Duration**: 3 days
- **Environment**: AWS SageMaker Studio (JupyterLab), domain `barclays-training-v2` (d-iwf4df7ijy95), us-west-2
- **Studio URL**: https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/studio/open/d-iwf4df7ijy95
- **Account**: 962804699607 (datacouch profile)
- **Execution role**: `SageMakerStudentExecutionRole`
- **Users**: instructor-axel + student-01 through student-25 (26 total)

---

## Training Execution Model (Critical)

### Day 1 and early Day 2 (up to and including Attention)
- All code runs **in the SageMaker Studio notebook directly** (JupyterLab kernel)
- No remote training jobs
- Instance: ml.t3.medium (Studio default) — sufficient for attention demos
- Same pattern as Colab, just different environment setup cell

### From "Transformer Translator" capstone onwards (Day 2+)
- **Remote training jobs** for all capstones and heavy fine-tuning
- CPU jobs: `sagemaker.pytorch.PyTorch` estimator, `ml.m5.xlarge`
- GPU jobs: `sagemaker.huggingface.HuggingFace` estimator, `ml.g4dn.xlarge`
- One CPU demo job included early to show the pattern before GPU jobs
- All training code lives in `source_dir` folders with `train.py` + `requirements.txt`

### Hard rule: HuggingFace estimator = GPU only
The HuggingFace training DLC has no CPU variant. Any job using `HuggingFace` estimator
MUST use a GPU instance (`ml.g4dn.xlarge` minimum). For CPU training use `PyTorch` estimator
and install transformers via `requirements.txt`.

---

## Pinned Version Matrix (Verified 2026-05-11)

| Component | Version | Notes |
|-----------|---------|-------|
| SageMaker SDK (notebook) | `>=2.200.0,<3.0.0` | v3 breaks `from sagemaker import get_execution_role` |
| transformers (notebook install) | `>=4.35.0,<4.40.0` | py312 wheels exist; do NOT pin 4.26 |
| tokenizers (notebook install) | `>=0.15.0,<0.20.0` | py312 wheels exist |
| PyTorch estimator | `framework_version="2.8.0"`, `py_version="py312"` | CPU + GPU; py311 not supported for 2.8.0 |
| HuggingFace estimator | `transformers_version="4.56.2"`, `pytorch_version="2.8.0"`, `py_version="py312"` | GPU only |
| GPU instance | `ml.g4dn.xlarge` | NVIDIA T4, 16GB VRAM, ~$0.74/hr |
| CPU training instance | `ml.m5.xlarge` | 4 vCPU, 16GB RAM - minimum for DistilBERT |
| Endpoint instance | `ml.m5.xlarge` | ml.c5.large (4GB RAM) OOMs on DistilBERT |
| MLflow tracking server | `MlflowVersion="2.13.2"` | ONLY supported version in us-west-2 as of 2026-05-11 |
| mlflow client | `mlflow==2.16.2` + `sagemaker-mlflow==0.1.0` | client version does NOT need to match server |
| numpy (everywhere) | `numpy<2` | pin in every install cell |
| TrainingArguments eval | `eval_strategy="epoch"` | `evaluation_strategy` removed in transformers 4.41+ |
| evaluate library | **DO NOT USE** | incompatible with datasets 4.x; use inline numpy instead |

### scripts_gpu/requirements.txt (canonical)
```
sagemaker-mlflow==0.1.0
mlflow==2.16.2
numpy<2
```

### scripts_cpu/requirements.txt (canonical)
```
transformers==4.40.0
datasets==2.18.0
numpy<2
```

### Notebook install cell (canonical)
```python
!pip install -q "sagemaker>=2.200.0,<3.0.0" \
    "transformers>=4.35.0,<4.40.0" \
    "tokenizers>=0.15.0,<0.20.0" \
    "datasets>=2.18.0,<3.0.0" \
    "numpy<2"
```

---

## IAM Role: SageMakerStudentExecutionRole

### Managed policies attached
- `AmazonSageMakerFullAccess`
- `AmazonSageMakerCanvasSMDataScienceAssistantAccess`

### Inline policies
- **BarclaysS3Access**: read/write to `barclays-genai-devs-data`, `barclays-prompt-eng-data`, `barclays-training-scratch`
- **SageMakerTrainingSupport**: `sagemaker:ListTrainingJobs` + ECR auth + CloudWatch Logs
- **SageMakerMLflowAccess**: `sagemaker-mlflow:*` on `arn:aws:sagemaker:us-west-2:962804699607:mlflow-tracking-server/*`

### Critical: sagemaker-mlflow:* is a SEPARATE IAM namespace
`AmazonSageMakerFullAccess` covers `sagemaker:*` but NOT `sagemaker-mlflow:*`. Both inline
policies are required for MLflow to work. Do not assume full access includes MLflow.

---

## S3 Buckets

| Bucket | Region | Purpose |
|--------|--------|---------|
| `sagemaker-us-west-2-962804699607` | us-west-2 | Default SM bucket, training artifacts |
| `barclays-genai-devs-data` | ap-south-1 | Course datasets (cross-region access works) |
| `barclays-prompt-eng-data` | us-west-2 | Prompt engineering datasets |
| `barclays-training-scratch` | us-west-2 | Scratch space for students |

---

## SageMaker Session Setup (canonical pattern for every notebook)

```python
import sagemaker
from sagemaker import get_execution_role
import boto3

sess = sagemaker.Session()
role = get_execution_role()
bucket = sess.default_bucket()
region = sess.boto_region_name

print(f"Role: {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
```

NO `getpass` for AWS credentials — the execution role handles authentication automatically
inside SageMaker Studio. Use `getpass.getpass()` only for external API keys (OpenAI, HuggingFace Hub token).

---

## Models Used in This Course

| Purpose | Model | Source |
|---------|-------|--------|
| Word embeddings demo | word2vec (gensim) | gensim |
| Seq2seq baseline | custom LSTM | built from scratch |
| Attention demo | custom attention | built from scratch |
| Transformer capstone | custom transformer | built from scratch |
| Transfer learning | DistilBERT | HuggingFace Hub |
| Full fine-tuning | Flan-T5 | HuggingFace Hub |
| PEFT/LoRA | Flan-T5 | HuggingFace Hub |
| Quantization | Flan-T5 or DistilBERT | HuggingFace Hub |

---

## Hard AWS Limits (Cannot Be Changed)

- Presigned Studio URLs max TTL = 12 hours (cannot make 7-day links)
- Domain VPC and auth mode are immutable after creation
- HuggingFace training DLCs are GPU-only — no CPU variant exists for any version
- Cannot change subnet IDs on a domain if the original VPC was deleted
- SageMaker Managed MLflow: only `2.13.2` supported in us-west-2 as of 2026-05-11
