# SageMaker Lessons Learned
# Hard-won from the Barclays permissions validation session (2026-05-10/11)
# READ THIS before writing any SageMaker notebook cell or training script.

---

## L1: HuggingFace estimator is GPU-ONLY. No exceptions.

**What broke**: `ValueError: Unsupported processor: cpu` when using `HuggingFace` estimator on `ml.m5.xlarge`.

**Rule**: Every HuggingFace training DLC is GPU-only — this is true for ALL versions including 4.26, 4.36, 4.56.
- Want CPU training? Use `sagemaker.pytorch.PyTorch` estimator + install transformers via `requirements.txt`.
- Want HuggingFace estimator? Use `ml.g4dn.xlarge` (cheapest GPU, NVIDIA T4, ~$0.74/hr).

**Never write**: `HuggingFace(instance_type="ml.m5.xlarge", ...)`

---

## L2: PyTorch 2.8.0 requires py_version="py312", NOT py311.

**What broke**: `ValueError: Unsupported Python version: py311. Supported: py312.`

**Rule**: Always verify the exact `py_version` for the chosen `framework_version` before writing estimator code.
For PyTorch 2.8.0: `py_version="py312"` is the only valid value.

---

## L3: SageMaker SDK v3 breaks everything. Pin to v2.

**What breaks**: `from sagemaker import get_execution_role` raises `ImportError` on sagemaker>=3.
SageMaker v3.0 (released Nov 2025) changed `__init__.py` to a namespace shim with zero exports.

**Rule**: Always pin `"sagemaker>=2.200.0,<3.0.0"` in the notebook install cell. Never let pip resolve to v3.

---

## L4: requirements.txt MUST be named exactly "requirements.txt" in source_dir.

**What broke**: `ModuleNotFoundError: No module named 'transformers'` — script had a subprocess pip hack
because the file was named `requirements_cpu.txt` instead of `requirements.txt`.

**Rule**: The sagemaker-training-toolkit auto-installs `requirements.txt` at the root of `source_dir` before
running the entry point. The filename MUST be exactly `requirements.txt`. Any other name is silently ignored.
Remove any subprocess pip install hacks from train.py — they are unnecessary and fragile.

**Structure required**:
```
source_dir/
  train.py          <- entry point (mandatory name configurable via entry_point=)
  requirements.txt  <- auto-installed (MUST be this exact name)
```

---

## L5: evaluation_strategy was renamed to eval_strategy in transformers 4.41+.

**What broke**: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`

**Rule**: Use `eval_strategy="epoch"` everywhere. The PyTorch 2.8.0 container ships transformers 4.41+.
`evaluation_strategy` was silently removed. Check both train.py files (scripts_cpu/ and scripts_gpu/).

---

## L6: Do NOT use the evaluate library. Use inline numpy instead.

**What broke**: `AttributeError: 'DownloadConfig' object has no attribute 'use_auth_token'`
`evaluate==0.4.1` tries to download the accuracy metric from HuggingFace Hub at eval time.
The `datasets` 4.x package removed `use_auth_token` from `DownloadConfig`. They are incompatible.

**Rule**: Never add `evaluate` to requirements.txt. Replace `evaluate.load("accuracy")` with:
```python
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()
    return {"accuracy": accuracy}
```
No Hub call, no version conflicts, identical result.

---

## L7: boto3 SageMaker exception is ResourceNotFound, NOT ResourceNotFoundException.

**What broke**: `AttributeError: <botocore.errorfactory.SageMakerExceptions object> has no attribute ResourceNotFoundException`

**Rule**: The boto3 SageMaker client uses `ResourceNotFound` (no "Exception" suffix).
```python
# WRONG
except sm_client.exceptions.ResourceNotFoundException:

# CORRECT
except sm_client.exceptions.ResourceNotFound:
```

---

## L8: SageMaker Managed MLflow only supports MlflowVersion="2.13.2" in us-west-2.

**What broke**: `ValidationException: The provided MLflow version is not supported`

**Rule**: As of 2026-05-11, only `MlflowVersion="2.13.2"` is accepted by the SageMaker Managed MLflow
create API in us-west-2. Versions 2.14.x through 2.18.x all return "not supported".

The mlflow CLIENT pip package version does NOT need to match the server version.
`mlflow==2.16.2` on the client talking to a `2.13.2` server works fine.

**Always use**:
```python
sm_client.create_mlflow_tracking_server(
    TrackingServerName="barclays-mlflow",
    ArtifactStoreUri=f"s3://{bucket}/mlflow-artifacts",
    MlflowVersion="2.13.2",   # <-- only supported version
    RoleArn=role,
)
```

---

## L9: sagemaker-mlflow:* is a SEPARATE IAM namespace from sagemaker:*.

**What broke**: `403 User is not authorized to perform: sagemaker-mlflow:GetExperimentByName`

**Rule**: `AmazonSageMakerFullAccess` covers `sagemaker:*` actions but does NOT cover `sagemaker-mlflow:*`.
These are completely separate IAM action namespaces. Both are required for MLflow to work.

**Required inline policy on the execution role**:
```json
{
  "Effect": "Allow",
  "Action": "sagemaker-mlflow:*",
  "Resource": "arn:aws:sagemaker:us-west-2:962804699607:mlflow-tracking-server/*"
}
```

---

## L10: HuggingFace Hub token needed for some model downloads — use getpass.

**Rule**: Some models on HuggingFace Hub are gated (e.g., LLaMA variants). Always use:
```python
import getpass
hf_token = getpass.getpass("HuggingFace Hub token: ")
```
Never hardcode tokens. For public models (DistilBERT, Flan-T5) no token is needed.

---

## L11: transformers notebook install — do NOT pin 4.26.

**What broke**: `error: can't find Rust compiler` — transformers 4.26 has no pre-built py312 wheels,
so pip tried to compile tokenizers from source. Rust is not installed in SageMaker Studio images.

**Rule**: Pin `transformers>=4.35.0,<4.40.0` and `tokenizers>=0.15.0,<0.20.0` for the notebook kernel.
These versions have pre-built py312 wheels. The training container versions are set inside the estimator
and are independent of the notebook install.

---

## L12: Training job status counters omit zero-count keys — use .get(), not direct access.

**What broke**: `KeyError: 'Failed'` when polling `TrainingJobStatusCounters` on a running HPT job.

**Rule**: AWS omits keys with a count of 0 from `TrainingJobStatusCounters`. Always use `.get()`:
```python
# WRONG
counts['Failed']

# CORRECT
counts.get('Failed', 0)
counts.get('Completed', 0)
counts.get('InProgress', 0)
```

---

## L13: Domain VPC is immutable — if the VPC is deleted, create a new domain.

**What broke**: Attempting to use domain `barclays-training` (d-zzyx1yfvygpc) after its VPC was deleted.
Error: `SageMaker is unable to find one or more of the subnets`.

**Rule**: You cannot update the VPC or subnets on an existing SageMaker domain. If the VPC is gone,
the domain is dead. Create a new domain pointing at a working VPC.

---

## L14: Presigned Studio URLs expire after 12 hours maximum. This cannot be extended.

**Rule**: Do NOT give students presigned URLs (`create-presigned-domain-url`). They expire in max 12 hours.
Give students the permanent console deep-link instead:
```
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/studio/open/d-iwf4df7ijy95
```
This requires AWS login but never expires.
