# F2 - SageMaker Fundamentals: Cell-by-Cell Plan

## Narrative and Audience

**Notebook**: `Frameworks/sagemaker_fundamentals.ipynb`
**Solution**: `Frameworks/sagemaker_fundamentals_solution.ipynb`
**Audience**: Developers who have completed F1 (PyTorch Refresher). They can write a training loop but have never touched SageMaker. They know AWS exists but think of it as "servers in the cloud."
**Estimated time**: 3.5 to 4 hours (90 min instructor-led demo + 90 min labs + 30 min MLflow/registry/endpoint walkthrough)
**Remote training**: ONE CPU job on `ml.m5.xlarge` using `sagemaker.pytorch.PyTorch` estimator. The model is intentionally tiny (2-layer MLP on synthetic complaint text features). Goal is the PATTERN, not the model.
**Day narrative**: "Building a Barclays Customer Support Intelligence System." You wrote your classifier locally. Now let's run it on the infrastructure Barclays actually uses. Everything from Topic 4 onwards lives in this pattern.
**Key constraint**: This notebook MUST be self-contained. A student who did not attend F1 can still follow the SageMaker pattern. The PyTorch model inside train.py is kept trivially simple on purpose.

---

## Diagram Index

Exactly 2 diagrams. Both must appear as `<!-- DIAGRAM: -->` placeholders in the notebook.

### Diagram 1
- **Slug**: `sagemaker_training_lifecycle`
- **Path**: `../../plans/F2/diagrams/sagemaker_training_lifecycle.mmd`
- **Description**: SageMaker remote training job lifecycle. Shows the flow: Studio notebook (estimator.fit()) -> S3 input data upload -> SageMaker control plane spins up training instance -> training instance downloads data from S3, installs requirements.txt, runs train.py -> artifacts saved to S3 -> job status queryable from Studio. Nodes: Studio Notebook, S3 Input, Training Instance (ml.m5.xlarge), S3 Artifacts, CloudWatch Logs. Directed graph, left to right.

### Diagram 2
- **Slug**: `sagemaker_mlflow_architecture`
- **Path**: `../../plans/F2/diagrams/sagemaker_mlflow_architecture.mmd`
- **Description**: SageMaker Managed MLflow architecture. Shows: train.py (running on training instance) calls mlflow.log_metric() -> MLflow Tracking Server (managed by AWS, MlflowVersion 2.13.2) -> experiment runs stored in tracking server DB -> artifacts (model files) stored in S3 artifact store bucket. Also shows: Studio notebook calls mlflow.search_runs() <- Tracking Server. IAM callout: sagemaker-mlflow:* permission is separate from sagemaker:*. Two-sided arrows where appropriate.

---

## Source Dir Structure

The source directory is created in the notebook as `source_dir = "scripts_cpu"`. The notebook writes both files using Python's `pathlib` / open() calls. Students see the files being written inline so they understand exactly what runs on the remote instance.

### `scripts_cpu/requirements.txt`

MUST be named exactly `requirements.txt` - the sagemaker-training-toolkit auto-installs this before running train.py. Any other name is silently ignored (Lesson L4).

```
transformers==4.40.0
datasets==2.18.0
numpy<2
```

No `evaluate` library (Lesson L6). No `sagemaker` or `sagemaker-mlflow` here - those are for the GPU variant. For the MLflow section (Beat 3 and Homework Extension), we add them via a second requirements file shown separately but explain that the canonical CPU file above does NOT include them unless you are doing MLflow tracking in the training script.

For the MLflow-enabled variant (shown in Section 6):
```
transformers==4.40.0
datasets==2.18.0
numpy<2
sagemaker-mlflow==0.1.0
mlflow==2.16.2
```

### `scripts_cpu/train.py` (FULL CONTENT - canonical)

This is the exact script that gets uploaded to S3 and runs on `ml.m5.xlarge`. It must:
- Accept SageMaker environment variables (SM_MODEL_DIR, SM_CHANNEL_TRAIN)
- Use argparse for hyperparameters
- Train a 2-layer MLP on synthetic 20-dimensional text features (no real data download needed)
- Log metrics to stdout (CloudWatch picks these up automatically)
- Save model artifacts to SM_MODEL_DIR
- Use inline numpy for accuracy (no evaluate library)
- Include MLflow tracking calls that are OPTIONAL (guarded by an arg flag)

```python
"""
train.py - Barclays Customer Support Complaint Classifier
Remote training script for SageMaker ml.m5.xlarge (CPU).

This script trains a tiny 2-layer MLP on synthetic complaint text features.
The goal is NOT model quality. The goal is to learn the SageMaker pattern:
- How environment variables carry paths
- How argparse carries hyperparameters from the estimator
- How artifacts land in SM_MODEL_DIR
- How stdout becomes CloudWatch logs

SageMaker injects these environment variables automatically:
  SM_MODEL_DIR   -> where to save your model (e.g. /opt/ml/model)
  SM_CHANNEL_TRAIN -> path to your training data (e.g. /opt/ml/input/data/train)
  SM_NUM_GPUS    -> number of GPUs (0 on ml.m5.xlarge - we are CPU-only here)
"""

import argparse
import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------
# Logging: everything printed to stdout appears in CloudWatch Logs.
# SageMaker streams /opt/ml/output/data/algo-1.log to CloudWatch.
# Use the standard logging module so log level can be controlled.
# -----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Model definition
# A tiny 2-layer MLP for complaint category classification.
# Input: 20-dim synthetic feature vector (simulates TF-IDF or embeddings)
# Hidden: configurable (default 64)
# Output: 5 classes (billing, fraud, account_access, product_complaint, other)
# -----------------------------------------------------------------------
class ComplaintClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------
# Synthetic data generator
# In a real job you would load from SM_CHANNEL_TRAIN.
# Here we generate synthetic data so the job has no external dependencies
# and completes quickly on CPU (approx 2-3 minutes on ml.m5.xlarge).
# -----------------------------------------------------------------------
def make_synthetic_data(n_samples=1000, input_dim=20, num_classes=5, seed=42):
    """
    Generate synthetic complaint features and labels.
    Each class has a slightly different mean so the model can actually learn.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    per_class = n_samples // num_classes
    for cls in range(num_classes):
        # Class-specific mean shifts features so the problem is learnable
        mean = np.zeros(input_dim)
        mean[cls * (input_dim // num_classes) : (cls + 1) * (input_dim // num_classes)] = 2.0
        X_cls = rng.normal(loc=mean, scale=1.0, size=(per_class, input_dim))
        y_cls = np.full(per_class, cls, dtype=np.int64)
        X_list.append(X_cls)
        y_list.append(y_cls)
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# -----------------------------------------------------------------------
# Accuracy metric - inline numpy, no evaluate library (Lesson L6)
# -----------------------------------------------------------------------
def compute_accuracy(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return float((preds == labels).mean())


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
def train(args):
    logger.info("SageMaker environment:")
    logger.info(f"  SM_MODEL_DIR     = {args.model_dir}")
    logger.info(f"  SM_CHANNEL_TRAIN = {args.train_dir}")
    logger.info(f"  epochs           = {args.epochs}")
    logger.info(f"  batch_size       = {args.batch_size}")
    logger.info(f"  lr               = {args.lr}")
    logger.info(f"  hidden_dim       = {args.hidden_dim}")

    # -- MLflow (optional, enabled by --mlflow-tracking-uri) ------------
    mlflow_enabled = bool(args.mlflow_tracking_uri)
    if mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.start_run(run_name="complaint-classifier-cpu")
        logger.info(f"MLflow tracking to: {args.mlflow_tracking_uri}")
        logger.info(f"Experiment: {args.mlflow_experiment_name}")
        # Log hyperparameters
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
        })

    # -- Data -----------------------------------------------------------
    logger.info("Generating synthetic complaint data...")
    X, y = make_synthetic_data(n_samples=2000, input_dim=20, num_classes=5)
    # 80/20 train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # -- Model, optimizer, loss -----------------------------------------
    model = ComplaintClassifier(
        input_dim=20,
        hidden_dim=args.hidden_dim,
        num_classes=5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # -- Training loop --------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        avg_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                all_logits.append(logits.numpy())
                all_labels.append(y_batch.numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        val_acc = compute_accuracy(all_logits, all_labels)

        logger.info(
            f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if mlflow_enabled:
            mlflow.log_metrics(
                {"train_loss": avg_loss, "val_accuracy": val_acc},
                step=epoch,
            )

    # -- Save model artifact --------------------------------------------
    # SageMaker expects the model to be saved under SM_MODEL_DIR.
    # After training completes, SageMaker tars everything in SM_MODEL_DIR
    # and uploads it to S3 as model.tar.gz.
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Also save the model config so the inference script knows the architecture
    config = {
        "input_dim": 20,
        "hidden_dim": args.hidden_dim,
        "num_classes": 5,
    }
    config_path = os.path.join(args.model_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    logger.info(f"Config saved to {config_path}")

    # -- Finalize MLflow run --------------------------------------------
    if mlflow_enabled:
        # Log the model artifact to MLflow as well
        mlflow.pytorch.log_model(model, artifact_path="complaint-classifier")
        mlflow.end_run()
        logger.info("MLflow run closed.")

    logger.info("Training complete.")


# -----------------------------------------------------------------------
# Argument parsing
# SageMaker passes hyperparameters as CLI args to the entry point.
# The training toolkit also sets SM_* env vars which map to some of these.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complaint classifier training")

    # Hyperparameters - passed via estimator hyperparameters={}
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)

    # SageMaker-injected paths - do NOT change the names or defaults
    # SageMaker sets these env vars and the training toolkit maps them to args
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )

    # MLflow (optional - only set if you want MLflow tracking)
    parser.add_argument("--mlflow-tracking-uri", type=str, default="")
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="barclays-complaint-classifier",
    )

    args = parser.parse_args()
    train(args)
```

---

## Cell-by-Cell Plan

Total cells: 43 (28 code + 15 markdown). Markdown cells never chain more than 3 in a row without a code cell.

---

### Cell 1: [type: markdown] - Title and Learning Objectives

**Purpose**: Orient students. Tell them what they will build, why it matters, and what they will know at the end.
**Content**:
```
# F2 - SageMaker Fundamentals

## Building a Barclays Customer Support Intelligence System

You have a PyTorch model that classifies customer complaints into five categories:
billing, fraud, account access, product complaint, and other.

You wrote it locally. It works on your laptop with 200 samples. But the real
Barclays complaint queue has 40,000 tickets a day. Your laptop will not cut it.

This notebook teaches you the SageMaker pattern you will use for every
capstone from Topic 4 onwards:

1. Set up a SageMaker session (your entry point to AWS)
2. Read and write data on S3 (the data layer)
3. Package your training code into a source directory
4. Launch a remote training job on ml.m5.xlarge (CPU)
5. Monitor the job: poll status, read CloudWatch logs
6. Track experiments with SageMaker Managed MLflow
7. Register the trained model in the SageMaker Model Registry
8. Deploy the model to a real-time endpoint and invoke it

The model we train is intentionally tiny. A 2-layer MLP on synthetic features.
The goal is NOT model accuracy. The goal is that you internalize the pattern.

**Estimated time**: 3.5 to 4 hours
**Instance required**: Studio notebook kernel (ml.t3.medium or similar)
**Remote job**: ml.m5.xlarge (CPU, ~$0.19/hr, approx 5-7 min total)
```
**Notes**: Keep objectives concrete and numbered. Students need to know this is infrastructure training, not ML training.

---

### Cell 2: [type: code] - Install dependencies

**Purpose**: Pin all versions. First code cell students run. Must succeed cleanly.
**Content**:
```python
# Install the SageMaker SDK and supporting libraries.
# Pin sagemaker to v2 -- v3 (released Nov 2025) removed get_execution_role from __init__.py.
# Pin numpy<2 everywhere to avoid the numpy 2.0 breaking API changes.
!pip install -q \
    "sagemaker>=2.200.0,<3.0.0" \
    "boto3>=1.34.0" \
    "numpy<2" \
    "mlflow==2.16.2" \
    "sagemaker-mlflow==0.1.0"
```
**Notes**: Instructor calls out the `<3.0.0` pin explicitly. "SageMaker v3 dropped the imports we rely on. This will bite you if you forget the pin." No transformers here - transformers runs INSIDE the training container, not the notebook kernel.

---

### Cell 3: [type: markdown] - Section 1: SageMaker Session

**Purpose**: Section header, one line of context, straight into broken code.
**Content**:
```
## Section 1: SageMaker Session Setup

Three objects anchor every SageMaker notebook: a Session, an IAM Role, and an S3 Bucket.
Let us see what happens when developers try to skip the session and call AWS directly.
```
**Notes**: Two lines only. Immediately followed by broken code.

---

### Cell 4: [type: code] - Beat 1: Broken session setup (naive direct boto3)

**Purpose**: Students feel the pain of trying to manually construct what the session gives you for free. Actually runs. Produces an error or clearly wrong output.
**Content**:
```python
# Beat 1: What developers try before they learn about sagemaker.Session()
# This will fail or produce incorrect output -- read the error carefully.

import boto3

# Naive approach: just grab the role from STS and build the bucket name manually
sts = boto3.client("sts")
identity = sts.get_caller_identity()
account_id = identity["Account"]
region = boto3.Session().region_name

# This is NOT how you get the SageMaker execution role.
# get_caller_identity returns YOUR user/role, not the SageMaker execution role.
naive_role = identity["Arn"]
naive_bucket = f"sagemaker-{region}-{account_id}"

print(f"Account:      {account_id}")
print(f"Region:       {region}")
print(f"Naive role:   {naive_role}")
print(f"Naive bucket: {naive_bucket}")

# Now try to use this naive_role to launch a training job later...
# It will fail with: "The provided role does not have the required permissions."
# The SageMaker service needs to ASSUME the role -- it cannot assume your user ARN.
print()
print("Problem: the role above is your Studio user, not a SageMaker-assumable role.")
print("SageMaker needs a role with 'sagemaker.amazonaws.com' as a trusted service.")
print("get_execution_role() reads the correct role from the Studio execution context.")
```
**Notes**: This actually runs and produces output. The "problem" explanation at the bottom is printed text, not a comment. Instructor pauses here for 2 min to discuss IAM trust relationships.

---

### Cell 5: [type: markdown] - Beat 2: Diagram placeholder for session setup

**Purpose**: Visual anchor. Show what the session wires together before the working code.
**Content**:
```
### How SageMaker Session wires your environment

<!-- DIAGRAM: SageMaker remote training job lifecycle (Studio -> S3 -> Training instance -> S3 artifacts -> CloudWatch) -->
[View diagram](../../plans/F2/diagrams/sagemaker_training_lifecycle.mmd)

The session is the thread connecting all four components above.
When you call `estimator.fit()`, the session translates that Python call
into SageMaker API calls that spin up the infrastructure on the right side.
```
**Notes**: Exactly 2 sentences after the diagram link. Do not exceed.

---

### Cell 6: [type: code] - Beat 3: Working session setup

**Purpose**: The canonical session pattern every notebook will use. Heavily commented.
**Content**:
```python
# Beat 3: The correct way to set up a SageMaker session.
# This is the canonical pattern -- copy it into every SageMaker notebook you write.

import sagemaker
from sagemaker import get_execution_role  # requires sagemaker<3.0.0
import boto3

# sagemaker.Session() connects to the SageMaker service in the current region.
# Inside SageMaker Studio this automatically uses your domain's AWS context.
# Outside Studio you need AWS credentials configured (e.g. via aws configure).
sess = sagemaker.Session()

# get_execution_role() reads the IAM role attached to this Studio user profile.
# This role has SageMaker permissions AND is trusted by sagemaker.amazonaws.com,
# meaning SageMaker can assume it when running training instances on your behalf.
role = get_execution_role()

# The default bucket is created by SageMaker the first time you use it.
# Name pattern: sagemaker-{region}-{account_id}
# Already exists in this account so this call is instant.
bucket = sess.default_bucket()

# Region is inferred from the boto3 session used by sagemaker.Session().
region = sess.boto_region_name

print(f"Role:   {role}")
print(f"Bucket: {bucket}")
print(f"Region: {region}")
```
**Notes**: Instructor live-codes this. Each variable gets a sentence of explanation. The print outputs are what students will reference throughout the rest of the notebook.

---

### Cell 7: [type: markdown] - Beat 4 Lab 1 instructions (Tier 1 - guided)

**Purpose**: Lab instruction header in STAR format.
**Content**:
```
### Lab 1 (Tier 1, guided) -- SageMaker Session Exploration (15 min)

**Situation**: You are a Barclays ML engineer. Every notebook you write
starts with this session block. You need to understand what each object
exposes before you can use it.

**Task**: Use the session and boto3 client to answer four questions about
your environment.

**Action**: Fill in each `= None  # YOUR CODE` line.

**Result**: The verification cell below prints a summary. All four
values should be non-empty strings.
```
**Notes**: STAR method applied. "Tier 1" and time estimate stated.

---

### Cell 8: [type: code] - Beat 4 Lab 1 starter code

**Purpose**: Guided lab with numbered steps. Tier 1 means every step is scaffolded.
**Content**:
```python
# Lab 1: SageMaker Session Exploration
# Complete the four lines below. Use sess, role, bucket, and region from Cell 6.
# All four answers are already computed above -- this is about learning the API.

import json

# Step 1: What is the AWS account ID that owns this SageMaker domain?
# Hint: sess.boto_session gives you a boto3 Session.
# From that session, create an STS client and call get_caller_identity().
account_id = None  # YOUR CODE

# Step 2: What is the S3 URI prefix where SageMaker will store training artifacts?
# SageMaker uses: s3://{bucket}/sagemaker/{job-name}/output/
# Build the base prefix (without job name): s3://{bucket}/sagemaker/
artifact_base = None  # YOUR CODE

# Step 3: List the first 3 SageMaker training jobs in this account (most recent first).
# Use boto3 to call sagemaker.list_training_jobs(SortBy='CreationTime', SortOrder='Descending', MaxResults=3)
# Store the list of job name strings (not the full response).
sm_client = sess.boto_session.client("sagemaker")
recent_jobs = None  # YOUR CODE

# Step 4: What IAM role name (not the full ARN) is being used for training?
# The ARN looks like: arn:aws:iam::962804699607:role/SageMakerStudentExecutionRole
# Extract just the role name after the last '/'.
role_name = None  # YOUR CODE

# -- Do not edit below this line --
print("=== Lab 1 Results ===")
print(f"Account ID:      {account_id}")
print(f"Artifact base:   {artifact_base}")
print(f"Role name:       {role_name}")
print(f"Recent jobs (up to 3):")
if recent_jobs:
    for j in recent_jobs:
        print(f"  {j}")
else:
    print("  (none yet - this account is fresh)")
```
**Notes**: Step 3 may return an empty list if no jobs have run yet. That is valid and expected. The verification cell below handles both cases.

---

### Cell 9: [type: code] - Lab 1 safety-net

**Purpose**: Safety-net so downstream cells work even if Lab 1 is incomplete.
**Content**:
```python
# Lab 1 safety-net: run this if you did not finish Lab 1.
# SKIP this cell if you DID finish Lab 1.
if account_id is None:
    print("Using Lab 1 safety-net so the rest of the notebook can run.")
    sts = sess.boto_session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    artifact_base = f"s3://{bucket}/sagemaker/"
    sm_client = sess.boto_session.client("sagemaker")
    resp = sm_client.list_training_jobs(SortBy="CreationTime", SortOrder="Descending", MaxResults=3)
    recent_jobs = [j["TrainingJobName"] for j in resp.get("TrainingJobSummaries", [])]
    role_name = role.split("/")[-1]
```
**Notes**: Safety-net is removed from the solution notebook.

---

### Cell 10: [type: code] - Lab 1 verification

**Purpose**: Confirm lab answers are correct before moving on.
**Content**:
```python
# Lab 1 verification -- run this after completing (or safety-netting) Lab 1.
assert account_id is not None and len(account_id) == 12, \
    "account_id should be a 12-digit string"
assert artifact_base.startswith("s3://"), \
    "artifact_base should be an S3 URI starting with s3://"
assert role_name and "/" not in role_name, \
    "role_name should be just the name, no slashes"

print("Lab 1 PASSED.")
print(f"  account_id:    {account_id}")
print(f"  artifact_base: {artifact_base}")
print(f"  role_name:     {role_name}")
print(f"  recent_jobs:   {recent_jobs}")
```
**Notes**: Assertions are descriptive. Students who see a failure message know exactly what to fix.

---

### Cell 11: [type: markdown] - Section 2: S3 Read/Write header

**Purpose**: Section header. One markdown cell before the broken code.
**Content**:
```
## Section 2: S3 Read and Write

S3 is the data layer for all SageMaker training. Input goes in. Artifacts come out.
```
**Notes**: Two lines only. Immediately followed by broken code in Cell 12.

---

### Cell 12: [type: code] - Beat 1: Broken S3 upload (wrong API)

**Purpose**: Show the wrong way to upload. Runs and fails (or produces wrong result).
**Content**:
```python
# Beat 1: What happens when you try to upload with raw boto3 put_object
# and forget that SageMaker expects a specific prefix structure.

import boto3
import io

s3 = boto3.client("s3", region_name=region)

# Naive: upload a small CSV with put_object, no prefix structure
sample_csv = "complaint_id,category\n1,billing\n2,fraud\n3,account_access\n"

try:
    s3.put_object(
        Bucket="this-bucket-does-not-exist-12345",  # wrong bucket name
        Key="data/complaints.csv",
        Body=sample_csv.encode(),
    )
    print("Uploaded? Something is wrong if this printed.")
except s3.exceptions.NoSuchBucket as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error (type={type(e).__name__}): {e}")

# Even if the bucket existed, SageMaker training jobs expect data at a specific
# S3 path that you pass via input channels. Uploading to an arbitrary key and
# then hardcoding the path in train.py breaks the estimator input abstraction.
print()
print("The SageMaker session gives you upload_data() which handles the prefix and")
print("returns the S3 URI you pass directly to estimator.fit(input_data).")
```
**Notes**: The wrong bucket name ensures a clean error. The print at the bottom bridges to the working demo.

---

### Cell 13: [type: code] - Beat 3: Working S3 upload and download

**Purpose**: The canonical S3 pattern. Heavily commented.
**Content**:
```python
# Beat 3: Correct S3 read/write using the SageMaker session helpers.

import os
import tempfile

# -- Upload: sess.upload_data() -------------------------------------------
# Creates a small synthetic complaints CSV in a temp directory
# and uploads it to S3 using the session helper.
# Returns the S3 URI, which we will pass to the estimator later.

os.makedirs("local_data", exist_ok=True)
with open("local_data/complaints_sample.csv", "w") as f:
    f.write("complaint_id,text_snippet,category\n")
    f.write("1,My statement shows charges I did not make,fraud\n")
    f.write("2,I cannot log in to online banking,account_access\n")
    f.write("3,My monthly fee increased without notice,billing\n")
    f.write("4,The mobile app crashes on deposit,product_complaint\n")
    f.write("5,My card was declined at checkout,other\n")

# upload_data() uploads the directory (or file) to S3 and returns the S3 URI.
# The key_prefix argument sets the S3 folder prefix.
s3_data_uri = sess.upload_data(
    path="local_data",                         # local file or directory
    bucket=bucket,                             # target bucket
    key_prefix="barclays-f2/data",             # S3 prefix (folder path)
)
print(f"Data uploaded to: {s3_data_uri}")

# -- Download: boto3 S3 get_object() or sagemaker.s3.S3Downloader ---------
# After a training job completes, model.tar.gz lands in S3.
# We use S3Downloader to pull it back to the notebook instance.
# (We will do the real download after the training job in Section 5.)
# For now, verify we can read the file we just uploaded.

from sagemaker.s3 import S3Downloader

# List objects under our prefix to confirm upload worked
objects = S3Downloader.list(s3_data_uri)
print(f"\nObjects under {s3_data_uri}:")
for obj in objects:
    print(f"  {obj}")

# Read one file directly from S3 into memory
content = S3Downloader.read_file(objects[0])
print(f"\nFirst 200 chars of {objects[0].split('/')[-1]}:")
print(content[:200])
```
**Notes**: Uses the session's upload_data helper. Introduces S3Downloader for the artifact download pattern that comes later in Section 5.

---

### Cell 14: [type: markdown] - Section 3: Source Directory header

**Purpose**: Transition to source dir creation. One markdown cell.
**Content**:
```
## Section 3: Packaging Your Training Code

SageMaker uploads your source directory to S3 and runs it inside a container.
The directory MUST contain `train.py` and `requirements.txt` (exactly these names).
```
**Notes**: Two lines. Followed immediately by code.

---

### Cell 15: [type: code] - Create source_dir, requirements.txt, and train.py

**Purpose**: Write both files to disk. Students see the full content of what runs on the remote instance. This is a critical pedagogical moment -- instructors should slow down here.
**Content**:
```python
# Create the source directory and write both required files.
# After this cell runs, inspect scripts_cpu/ in the file browser to see what will be uploaded.

import os

os.makedirs("scripts_cpu", exist_ok=True)

# requirements.txt -- MUST be this exact name.
# The sagemaker-training-toolkit reads this file and installs it before
# running train.py. Naming it anything else (requirements_cpu.txt, etc.)
# causes a silent failure where your script fails with ModuleNotFoundError.
requirements = """\
transformers==4.40.0
datasets==2.18.0
numpy<2
"""

with open("scripts_cpu/requirements.txt", "w") as f:
    f.write(requirements)

print("Created scripts_cpu/requirements.txt:")
print(requirements)
```
**Notes**: Only requirements.txt in this cell. train.py is long and gets its own cells (17 and 18) split at a natural boundary so the full script is visible without scrolling past 50 lines per cell.

---

### Cell 16: [type: code] - Write train.py (part 1: imports, model, data)

**Purpose**: First half of train.py. Students read the model definition and synthetic data generator.
**Content**:
```python
# train.py -- Part 1: imports, model definition, and synthetic data.
# This code runs REMOTELY on ml.m5.xlarge, not in your notebook kernel.

train_py_part1 = '''"""
train.py - Barclays Customer Support Complaint Classifier
Runs on SageMaker ml.m5.xlarge (CPU).

SageMaker injects these environment variables before calling this script:
  SM_MODEL_DIR      -> /opt/ml/model  (save your model here)
  SM_CHANNEL_TRAIN  -> /opt/ml/input/data/train  (your input data)
  SM_NUM_GPUS       -> 0 on CPU instances
"""

import argparse
import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ComplaintClassifier(nn.Module):
    """2-layer MLP. Input: 20-dim synthetic features. Output: 5 complaint categories."""
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_synthetic_data(n_samples=2000, input_dim=20, num_classes=5, seed=42):
    """Generate class-separable synthetic features. No external data needed."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    per_class = n_samples // num_classes
    for cls in range(num_classes):
        mean = np.zeros(input_dim)
        mean[cls * (input_dim // num_classes):(cls + 1) * (input_dim // num_classes)] = 2.0
        X_cls = rng.normal(loc=mean, scale=1.0, size=(per_class, input_dim))
        X_list.append(X_cls.astype(np.float32))
        y_list.append(np.full(per_class, cls, dtype=np.int64))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def compute_accuracy(logits, labels):
    """Inline numpy accuracy -- no evaluate library required."""
    preds = np.argmax(logits, axis=-1)
    return float((preds == labels).mean())
'''

with open("scripts_cpu/train.py", "w") as f:
    f.write(train_py_part1)

print("Wrote Part 1 of train.py (model + data)")
print(f"Lines so far: {len(train_py_part1.splitlines())}")
```
**Notes**: The triple-quote string approach lets students see the file content inline. Instructor walks through ComplaintClassifier and make_synthetic_data explaining why synthetic data is used.

---

### Cell 17: [type: code] - Write train.py (part 2: training loop and argparse)

**Purpose**: Second half of train.py. Students see the SageMaker-specific patterns: SM_MODEL_DIR, argparse, saving artifacts.
**Content**:
```python
# train.py -- Part 2: training loop, artifact saving, and argparse.
# Appended to scripts_cpu/train.py written in the previous cell.

train_py_part2 = '''

def train(args):
    logger.info(f"SM_MODEL_DIR={args.model_dir}  SM_CHANNEL_TRAIN={args.train_dir}")
    logger.info(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # Optional MLflow (enabled by --mlflow-tracking-uri hyperparameter)
    mlflow_enabled = bool(args.mlflow_tracking_uri)
    if mlflow_enabled:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.start_run(run_name="complaint-classifier-cpu")
        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size,
                           "lr": args.lr, "hidden_dim": args.hidden_dim})

    X, y = make_synthetic_data()
    split = int(0.8 * len(X))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split])),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:])),
        batch_size=args.batch_size,
    )

    model = ComplaintClassifier(hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_b)
        avg_loss = total_loss / (len(X) * 0.8)

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                all_logits.append(model(X_b).numpy())
                all_labels.append(y_b.numpy())
        val_acc = compute_accuracy(
            np.concatenate(all_logits), np.concatenate(all_labels)
        )
        # These log lines appear in CloudWatch Logs automatically.
        logger.info(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        if mlflow_enabled:
            mlflow.log_metrics({"train_loss": avg_loss, "val_accuracy": val_acc}, step=epoch)

    # Save to SM_MODEL_DIR -- SageMaker tars this directory and uploads to S3.
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))
    with open(os.path.join(args.model_dir, "model_config.json"), "w") as fp:
        json.dump({"input_dim": 20, "hidden_dim": args.hidden_dim, "num_classes": 5}, fp)
    logger.info("Model saved. Training complete.")

    if mlflow_enabled:
        mlflow.pytorch.log_model(model, artifact_path="complaint-classifier")
        mlflow.end_run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    # SageMaker-injected paths -- do NOT rename these
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    # MLflow optional
    p.add_argument("--mlflow-tracking-uri", type=str, default="")
    p.add_argument("--mlflow-experiment-name", type=str, default="barclays-complaint-classifier")
    train(p.parse_args())
'''

with open("scripts_cpu/train.py", "a") as f:
    f.write(train_py_part2)

print("Appended Part 2 of train.py (training loop + argparse)")

# Count total lines in the final file
with open("scripts_cpu/train.py") as f:
    lines = f.readlines()
print(f"Total lines in train.py: {len(lines)}")

# Verify both required files exist
import os
assert os.path.exists("scripts_cpu/train.py"), "train.py missing"
assert os.path.exists("scripts_cpu/requirements.txt"), "requirements.txt missing"
print("scripts_cpu/ is ready for upload.")
```
**Notes**: Instructor highlights `SM_MODEL_DIR` and argparse patterns. "This is the contract between SageMaker and your script. Break the contract and your model never saves."

---

### Cell 18: [type: markdown] - Section 4 header: Launching the job

**Purpose**: Bridge into the main lab. One markdown.
**Content**:
```
## Section 4: Launching a Remote CPU Training Job

Calling `.fit()` on a PyTorch estimator uploads your source directory to S3 and
submits the job. The notebook kernel stays free -- the ml.m5.xlarge does the work.
```
**Notes**: Two lines. Straight into Beat 1.

---

### Cell 19: [type: code] - Beat 1: Wrong estimator config (HuggingFace on CPU)

**Purpose**: Show the specific error that L1 from SAGEMAKER_LESSONS_LEARNED.md documents. Runs and produces the actual exception.
**Content**:
```python
# Beat 1: What happens if you use the HuggingFace estimator on a CPU instance.
# This is the most common mistake. The error is clear but easy to miss in docs.

from sagemaker.huggingface import HuggingFace

try:
    wrong_estimator = HuggingFace(
        entry_point="train.py",
        source_dir="scripts_cpu",
        instance_type="ml.m5.xlarge",         # CPU instance
        instance_count=1,
        transformers_version="4.56.2",
        pytorch_version="2.8.0",
        py_version="py312",
        role=role,
        hyperparameters={"epochs": 2},
    )
    print("No error? That should not happen.")
except ValueError as e:
    print(f"ValueError: {e}")
    print()
    print("Root cause: HuggingFace training DLCs are GPU-only.")
    print("There is no CPU variant for ANY HuggingFace estimator version.")
    print("Fix: use sagemaker.pytorch.PyTorch estimator + install transformers via requirements.txt")

# The correct pattern for CPU training:
# from sagemaker.pytorch import PyTorch
# estimator = PyTorch(
#     entry_point="train.py",
#     source_dir="scripts_cpu",
#     framework_version="2.8.0",
#     py_version="py312",        # py311 is NOT valid for 2.8.0
#     instance_type="ml.m5.xlarge",
#     ...
# )
```
**Notes**: This ACTUALLY raises a ValueError. Instructor reads the error with the class before moving on. The commented correct pattern is a preview of Cell 21.

---

### Cell 20: [type: code] - Beat 3: Working estimator setup (do not call .fit() yet)

**Purpose**: Build the correct estimator. Do NOT call .fit() in this cell - that is for the lab.
**Content**:
```python
# Beat 3: The correct PyTorch estimator for CPU training.
# Note: we build the estimator here but do NOT call .fit() yet.
# That is Lab 2 (the main lab of this notebook).

from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    # Entry point: the script SageMaker runs inside the container
    entry_point="train.py",

    # Source directory: uploaded to S3, extracted on the training instance
    # Must contain requirements.txt (exactly that name)
    source_dir="scripts_cpu",

    # Container version -- py311 is NOT valid for framework_version 2.8.0
    framework_version="2.8.0",
    py_version="py312",

    # Instance: CPU (HuggingFace estimator would fail here with ValueError)
    instance_type="ml.m5.xlarge",
    instance_count=1,

    # IAM role that the training instance assumes
    role=role,

    # Hyperparameters are passed as CLI args to train.py
    # e.g. --epochs 5 --batch-size 32 --lr 0.001 --hidden-dim 64
    hyperparameters={
        "epochs": 5,
        "batch-size": 32,
        "lr": 1e-3,
        "hidden-dim": 64,
    },

    # Use the current SageMaker session
    sagemaker_session=sess,

    # Keep output compact -- only the model artifact, no full debug output
    disable_profiler=True,
)

print("Estimator created successfully.")
print(f"  framework_version: {estimator.framework_version}")
print(f"  py_version:        {estimator.py_version}")
print(f"  instance_type:     {estimator.instance_type}")
print(f"  entry_point:       {estimator.entry_point}")
```
**Notes**: Each constructor argument gets an inline comment. Instructor pauses on `py_version="py312"` -- "this breaks you silently in 2025 if you forget." Do not call .fit() here.

---

### Cell 21: [type: markdown] - Beat 4 Lab 2 instructions (Tier 1 - guided)

**Purpose**: Main lab. STAR format. Tier 1 = numbered steps, fully scaffolded.
**Content**:
```
### Lab 2 (Tier 1, guided) -- Launch the Training Job (20 min)

**Situation**: The estimator from the cell above is configured.
Your source directory `scripts_cpu/` contains `train.py` and `requirements.txt`.

**Task**: Submit the job to SageMaker and capture the job name.

**Action**: Fill in the two `= None  # YOUR CODE` lines.

Step 1: Call `estimator.fit(inputs=None, wait=False)` to submit without blocking.
Step 2: Capture `estimator.latest_training_job.name` and store it in `job_name`.

**Result**: The verification cell prints the job name and confirms the job is InProgress.
The job will appear in SageMaker Studio -> Training jobs within 60 seconds.
```
**Notes**: Tier 1, fully guided. F2 is a framework prereq so all labs are Tier 1.

---

### Cell 22: [type: code] - Beat 4 Lab 2 starter code

**Purpose**: Tier 1 guided lab with numbered step comments.
**Content**:
```python
# Lab 2: Launch the training job.
# estimator is already built in the cell above.

# Step 1: Submit the job without blocking the notebook kernel.
# Hint: call estimator.fit() with inputs=None and wait=False.
estimator.fit(None)  # YOUR CODE: replace None with the correct call

# Step 2: Capture the training job name.
# SageMaker auto-generates a name like "pytorch-training-2026-..."
# Hint: use estimator.latest_training_job.name
job_name = None  # YOUR CODE

print(f"Job submitted: {job_name}")
print(f"CloudWatch logs: /aws/sagemaker/TrainingJobs / {job_name}/algo-1")
```
**Notes**: Tier 1 -- each step is spelled out. Two placeholders to fill.

---

### Cell 23: [type: code] - Lab 2 safety-net

**Purpose**: Safety-net for the main lab so downstream monitoring cells work.
**Content**:
```python
# Lab 2 safety-net: run this if you did not finish Lab 2.
# SKIP this cell if you DID finish Lab 2 and job_name is set.
if job_name is None:
    print("Using Lab 2 safety-net -- launching training job now.")
    estimator.fit(inputs=None, wait=False)
    job_name = estimator.latest_training_job.name
    log_group = f"/aws/sagemaker/TrainingJobs"
    print(f"Job name:  {job_name}")
    print(f"Log group: {log_group}")
```
**Notes**: Critical safety-net. All monitoring cells in Section 5 depend on job_name.

---

### Cell 24: [type: code] - Lab 2 verification

**Purpose**: Confirm job was submitted before moving to monitoring.
**Content**:
```python
# Lab 2 verification
assert job_name is not None, "job_name is None -- run the safety-net cell above"
assert job_name.startswith("pytorch-training-"), \
    f"Unexpected job name format: {job_name}. Did .fit() run?"

sm_client = sess.boto_session.client("sagemaker")
resp = sm_client.describe_training_job(TrainingJobName=job_name)
status = resp["TrainingJobStatus"]
print(f"Lab 2 PASSED.")
print(f"  Job name: {job_name}")
print(f"  Status:   {status}  (expected: InProgress or Completed)")
print(f"  Log group: /aws/sagemaker/TrainingJobs")
print(f"  Log stream: {job_name}/algo-1")
```
**Notes**: Status check uses describe_training_job. No assertion on status value since timing varies.

---

### Cell 25: [type: markdown] - Section 5 header: Monitoring

**Purpose**: Section header before monitoring cells.
**Content**:
```
## Section 5: Monitoring the Job

Poll `describe_training_job` for status. Read logs from CloudWatch at
`/aws/sagemaker/TrainingJobs / {job_name}/algo-1`.
```
**Notes**: Two lines. Followed by code.

---

### Cell 26: [type: code] - Beat 3: Polling job status with timeout

**Purpose**: The canonical polling pattern. Used in all subsequent capstone notebooks.
**Content**:
```python
# Beat 3: Poll training job status until completion or timeout.
# This is the canonical monitoring pattern for all remote training jobs.

import time

def wait_for_training_job(sm_client, job_name, poll_interval_sec=30, timeout_sec=600):
    """
    Poll describe_training_job until status is Completed or Failed.

    Args:
        sm_client: boto3 SageMaker client
        job_name:  training job name
        poll_interval_sec: seconds between polls (default 30)
        timeout_sec: maximum wait time in seconds (default 600 = 10 min)

    Returns:
        final_status: "Completed" or "Failed"
    """
    start = time.time()
    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:4d}s] Status: {status}")

        if status in ("Completed", "Failed", "Stopped"):
            if status == "Failed":
                reason = resp.get("FailureReason", "unknown")
                print(f"  FAILED: {reason}")
            return status

        if elapsed > timeout_sec:
            print(f"  Timeout after {timeout_sec}s -- job may still be running")
            return "Timeout"

        time.sleep(poll_interval_sec)


# Run the poller (this cell blocks for 3-7 minutes on a fresh ml.m5.xlarge job)
print(f"Polling job: {job_name}")
print("(ml.m5.xlarge startup takes ~2-3 min, training takes ~1-2 min)")
sm_client_poll = sess.boto_session.client("sagemaker")
final_status = wait_for_training_job(sm_client_poll, job_name)
print(f"\nFinal status: {final_status}")
```
**Notes**: This cell BLOCKS for a few minutes. Instructor tells students to use this time to open the SageMaker console and watch the job appear. Timeout is set to 600s (10 min) which is generous for a 5-job.

---

### Cell 27: [type: code] - CloudWatch log tailing

**Purpose**: Show how to read the training logs from CloudWatch. Students see their own stdout output from train.py.
**Content**:
```python
# Read the last N lines of CloudWatch logs for this training job.
# Every print() and logger.info() in train.py ends up in this log stream.

import boto3

logs_client = sess.boto_session.client("logs", region_name=region)

log_group = "/aws/sagemaker/TrainingJobs"
log_stream = f"{job_name}/algo-1"

def tail_cloudwatch_logs(logs_client, log_group, log_stream, last_n=30):
    """Fetch the last N log events from a CloudWatch log stream."""
    try:
        resp = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=log_stream,
            limit=last_n,
            startFromHead=False,
        )
        events = resp.get("events", [])
        if not events:
            print("No log events yet. The job may still be starting up.")
            return
        print(f"Last {len(events)} log events from {log_stream}:")
        print("-" * 60)
        for event in events:
            print(event["message"])
    except logs_client.exceptions.ResourceNotFoundException:
        print(f"Log stream not found: {log_stream}")
        print("The job may not have started yet, or the stream name is incorrect.")


tail_cloudwatch_logs(logs_client, log_group, log_stream)
```
**Notes**: Uses `ResourceNotFoundException` correctly per boto3 logs client (this is a CloudWatch Logs client exception, not SageMaker -- the Lesson L7 applies to the SageMaker client specifically). Instructor points out students can see their own `logger.info()` calls from train.py in this output.

---

### Cell 28: [type: code] - Download model artifacts from S3

**Purpose**: Show how to get the trained model back from S3 and load it locally.
**Content**:
```python
# After the job completes, SageMaker uploads everything in SM_MODEL_DIR as model.tar.gz.
# Locate the artifact URI from the describe response and download it.

import tarfile
import os

resp = sm_client_poll.describe_training_job(TrainingJobName=job_name)
model_artifact_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
print(f"Model artifact: {model_artifact_uri}")

# Download using S3Downloader
from sagemaker.s3 import S3Downloader

os.makedirs("downloaded_model", exist_ok=True)
S3Downloader.download(model_artifact_uri, "downloaded_model/")

# Extract the tarball
tarball_path = "downloaded_model/model.tar.gz"
with tarfile.open(tarball_path, "r:gz") as tar:
    tar.extractall("downloaded_model/")

# List extracted files
extracted = os.listdir("downloaded_model/")
print(f"Extracted files: {extracted}")

# Load and inspect the model config
import json
with open("downloaded_model/model_config.json") as f:
    config = json.load(f)
print(f"Model config: {config}")

# Verify the state dict loads
import torch
import torch.nn as nn

class ComplaintClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)

model = ComplaintClassifier(**config)
model.load_state_dict(torch.load("downloaded_model/model.pt", map_location="cpu"))
model.eval()
print("Model loaded successfully from S3 artifacts.")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```
**Notes**: Redefines ComplaintClassifier locally because this is a notebook -- the class definition from train.py is not in the notebook namespace. Students will hit this in capstones and need to see the pattern.

---

### Cell 29: [type: markdown] - Section 6 header: MLflow

**Purpose**: Section header for MLflow. One markdown before Beat 1.
**Content**:
```
## Section 6: MLflow Experiment Tracking

Which of the 20 training runs produced the best model? SageMaker Managed MLflow
records hyperparameters, per-epoch metrics, and model artifacts automatically.

IMPORTANT: `AmazonSageMakerFullAccess` covers `sagemaker:*` but NOT `sagemaker-mlflow:*`.
If you see a 403 on GetExperimentByName, add the `sagemaker-mlflow:*` inline policy.
```
**Notes**: IAM callout kept, trimmed to 2 lines. Followed immediately by Beat 1 code.

---

### Cell 30: [type: code] - Beat 1: Wrong MLflow version

**Purpose**: Show the "not supported" error when using the wrong MlflowVersion.
**Content**:
```python
# Beat 1: What happens if you specify the wrong MLflow server version.
# SageMaker Managed MLflow only supports MlflowVersion="2.13.2" in us-west-2.

sm_client_mlflow = sess.boto_session.client("sagemaker")

try:
    resp = sm_client_mlflow.create_mlflow_tracking_server(
        TrackingServerName="barclays-mlflow-wrong-version",
        ArtifactStoreUri=f"s3://{bucket}/mlflow-artifacts-test",
        MlflowVersion="2.18.0",   # <-- NOT supported in us-west-2
        RoleArn=role,
    )
    print("Created? Something is wrong if this printed.")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    print()
    print("Only MlflowVersion='2.13.2' is supported in us-west-2 as of 2026-05-11.")
    print("Versions 2.14.x through 2.18.x all return ValidationException.")
```
**Notes**: Will raise ValidationException. Students see the exact error. Instructor: "This is not documented clearly in the AWS docs. We hit it the hard way."

---

### Cell 31: [type: markdown] - Beat 2: Diagram placeholder MLflow

**Purpose**: Second and final diagram placeholder.
**Content**:
```
### How SageMaker Managed MLflow fits into the training pattern

<!-- DIAGRAM: SageMaker Managed MLflow architecture (tracking server, artifact store, experiment runs, IAM separation) -->
[View diagram](../../plans/F2/diagrams/sagemaker_mlflow_architecture.mmd)

The training instance talks to the tracking server over HTTPS.
The tracking server writes run metadata to its own managed database
and writes model artifacts to the S3 artifact store you specify.
Your notebook queries the tracking server via the mlflow Python client.
```
**Notes**: Two sentences after the diagram link. The MLflow architecture diagram is the second and final diagram in this notebook.

---

### Cell 32: [type: code] - Beat 3: Get or create MLflow tracking server

**Purpose**: Check if the server already exists, create it if not, and get the tracking URI.
**Content**:
```python
# Beat 3: Get or create the SageMaker Managed MLflow tracking server.
# The server for this course (barclays-mlflow) already exists in this account.
# This cell is idempotent -- safe to re-run.

TRACKING_SERVER_NAME = "barclays-mlflow"
sm_client_mlflow = sess.boto_session.client("sagemaker")

def get_or_create_mlflow_server(sm_client, server_name, bucket, role):
    """Return the tracking URI for an existing or newly created server."""
    try:
        resp = sm_client.describe_mlflow_tracking_server(
            TrackingServerName=server_name
        )
        tracking_uri = resp["TrackingServerUrl"]
        status = resp["TrackingServerStatus"]
        print(f"Server '{server_name}' exists. Status: {status}")
        print(f"Tracking URI: {tracking_uri}")
        return tracking_uri
    except sm_client.exceptions.ResourceNotFound:
        # ResourceNotFound (no "Exception" suffix) -- Lesson L7 applies here too
        print(f"Server '{server_name}' not found. Creating...")
        sm_client.create_mlflow_tracking_server(
            TrackingServerName=server_name,
            ArtifactStoreUri=f"s3://{bucket}/mlflow-artifacts",
            MlflowVersion="2.13.2",   # ONLY supported version in us-west-2
            RoleArn=role,
        )
        # Wait for Active status
        import time
        for _ in range(30):
            resp = sm_client.describe_mlflow_tracking_server(TrackingServerName=server_name)
            if resp["TrackingServerStatus"] == "Created":
                tracking_uri = resp["TrackingServerUrl"]
                print(f"Server ready. Tracking URI: {tracking_uri}")
                return tracking_uri
            time.sleep(10)
        raise TimeoutError("MLflow server did not become ready in 5 min")


mlflow_tracking_uri = get_or_create_mlflow_server(
    sm_client_mlflow, TRACKING_SERVER_NAME, bucket, role
)
```
**Notes**: `ResourceNotFound` (correct) vs `ResourceNotFoundException` (wrong). Instructor repeats Lesson L7 here because it's the second time they see it (Lesson L7 was mentioned in section 5 for the logs client but that uses the logs client not sagemaker client).

---

### Cell 33: [type: code] - Beat 3 continued: Query MLflow runs from the previous job

**Purpose**: Show the mlflow client querying runs from the notebook (not the training script).
**Content**:
```python
# Connect the mlflow client to the SageMaker Managed MLflow server
# and query experiment runs.
# Note: mlflow client version (2.16.2) does NOT need to match server version (2.13.2).

import mlflow

mlflow.set_tracking_uri(mlflow_tracking_uri)

# List experiments
experiments = mlflow.search_experiments()
print(f"Experiments on this tracking server:")
for exp in experiments:
    print(f"  {exp.experiment_id}: {exp.name}")

# Search runs in our experiment (if any exist)
experiment_name = "barclays-complaint-classifier"
try:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        max_results=5,
    )
    if runs.empty:
        print(f"\nNo runs found in '{experiment_name}'.")
        print("The previous training job did not use MLflow tracking.")
        print("Section 6.2 (Homework) shows how to add MLflow to the training script.")
    else:
        print(f"\nLatest runs in '{experiment_name}':")
        cols = [c for c in runs.columns if c.startswith("metrics.") or c in ["run_id", "status"]]
        print(runs[cols].head())
except Exception as e:
    print(f"Could not query experiment '{experiment_name}': {e}")
```
**Notes**: The first training job (Lab 2) did NOT use MLflow -- the hyperparameter `mlflow-tracking-uri` was not set. This cell may show "No runs found" which is expected. The Homework Extension in Cell 45 shows how to re-run with MLflow tracking enabled.

---

### Cell 34: [type: markdown] - Section 7 header: Model Registry

**Purpose**: Section header. One markdown.
**Content**:
```
## Section 7: Registering the Trained Model

Register the artifact before you deploy it. Every version starts in
"PendingManualApproval" -- an approver promotes it to "Approved" before deployment.
```
**Notes**: Two lines. Followed immediately by code.

---

### Cell 35: [type: code] - Beat 3: Register model in Model Registry

**Purpose**: Create model package group and register the model artifact.
**Content**:
```python
# Register the trained model in SageMaker Model Registry.
# Steps: (1) create a model package group (once), (2) register a version.

import time

sm_client_registry = sess.boto_session.client("sagemaker")

MODEL_PACKAGE_GROUP = "barclays-complaint-classifier"

# Step 1: create the model package group (idempotent -- skip if already exists)
try:
    sm_client_registry.create_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageGroupDescription="Barclays customer complaint classifier",
    )
    print(f"Created model package group: {MODEL_PACKAGE_GROUP}")
except sm_client_registry.exceptions.ResourceInUse:
    print(f"Model package group already exists: {MODEL_PACKAGE_GROUP}")

# Step 2: register a model version
# ModelDataUrl points to the S3 location of model.tar.gz
# The inference image must match the framework used for training
inference_image = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="2.8.0",
    py_version="py312",
    instance_type="ml.m5.xlarge",
    image_scope="inference",
)
print(f"Inference image URI: {inference_image}")

resp = sm_client_registry.create_model_package(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelPackageDescription=f"Trained by job: {job_name}",
    InferenceSpecification={
        "Containers": [
            {
                "Image": inference_image,
                "ModelDataUrl": model_artifact_uri,
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",
)

model_package_arn = resp["ModelPackageArn"]
print(f"Registered model version: {model_package_arn}")
print("Status: PendingManualApproval")
print("(An approver must change this to 'Approved' before deployment)")
```
**Notes**: Shows the approval workflow. Instructor discusses: in Barclays, model approval is a governance step involving risk and model validation teams.

---

### Cell 36: [type: code] - Approve and promote model version

**Purpose**: Approve the model so it can be deployed.
**Content**:
```python
# Approve the model version (simulate the governance step).
# In production this would be done by a separate approver after validation.

sm_client_registry.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",
)

resp = sm_client_registry.describe_model_package(ModelPackageArn=model_package_arn)
print(f"Model package status: {resp['ModelApprovalStatus']}")
print(f"Model package ARN:    {model_package_arn}")
print("Ready for deployment.")
```
**Notes**: Short cell. One action, one confirmation print. Keeps the registry section tight.

---

### Cell 37: [type: markdown] - Section 8 header: Endpoint deployment

**Purpose**: Section header.
**Content**:
```
## Section 8: Deploying to a Real-Time Endpoint

`.deploy()` creates a managed HTTPS endpoint. Send JSON, get predictions back.
Deploying takes 3-5 minutes. Endpoints cost money while running -- delete when done.
```
**Notes**: Two lines. Cost reminder kept. Followed immediately by code.

---

### Cell 38: [type: code] - Deploy endpoint from registered model

**Purpose**: Deploy the registered model version to a real-time endpoint.
**Content**:
```python
# Deploy the approved model version to a real-time SageMaker endpoint.
# This takes 3-5 minutes (container download + instance startup).

from sagemaker.pytorch import PyTorchModel

ENDPOINT_NAME = "barclays-complaint-classifier-ep"

# Build a SageMaker Model object from the registered model artifact
pytorch_model = PyTorchModel(
    model_data=model_artifact_uri,
    role=role,
    framework_version="2.8.0",
    py_version="py312",
    # inference.py would go here for custom preprocessing -- we use default for now
    # entry_point="inference.py",
    sagemaker_session=sess,
)

# Deploy creates:
#   1. A SageMaker Model (wrapper around the container + artifact)
#   2. An Endpoint Configuration (which model, which instance)
#   3. An Endpoint (the live HTTPS service)
print(f"Deploying to endpoint: {ENDPOINT_NAME}")
print("(This takes 3-5 minutes -- watch for 'Endpoint in service' in the console)")

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=ENDPOINT_NAME,
    wait=True,   # block until endpoint is InService
)
print(f"\nEndpoint deployed: {ENDPOINT_NAME}")
print(f"Endpoint status: InService")
```
**Notes**: `wait=True` blocks until InService. This is intentional -- students see the progression in the cell output. Takes 3-5 min. Use this time for questions.

---

### Cell 39: [type: code] - Invoke the endpoint

**Purpose**: Show how to send a prediction request to the live endpoint.
**Content**:
```python
# Invoke the endpoint with a synthetic complaint feature vector.
# In production, these features would come from a preprocessing pipeline.

import json
import numpy as np
import boto3

runtime_client = sess.boto_session.client("sagemaker-runtime")

# Synthetic 20-dimensional feature vector representing a "billing" complaint
# (class 0: features 0-3 are elevated)
billing_complaint_features = np.zeros(20, dtype=np.float32)
billing_complaint_features[:4] = 2.5   # billing features elevated

payload = json.dumps({"inputs": billing_complaint_features.tolist()})

resp = runtime_client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=payload,
)

raw_output = resp["Body"].read().decode("utf-8")
print(f"Raw endpoint output: {raw_output[:200]}")

# The default PyTorch container returns a JSON list of logits
# Parse and apply argmax to get the predicted class
try:
    logits = json.loads(raw_output)
    predicted_class = int(np.argmax(logits))
    class_names = ["billing", "fraud", "account_access", "product_complaint", "other"]
    print(f"\nPredicted class index: {predicted_class}")
    print(f"Predicted class name:  {class_names[predicted_class]}")
    print(f"(Expected: billing -- features for class 0 were elevated)")
except Exception as e:
    print(f"Could not parse output: {e}")
    print(f"Raw output was: {raw_output}")
```
**Notes**: Shows the full round trip. Instructor notes: in production you need an inference.py to handle preprocessing. The default container just wraps the forward() call.

---

### Cell 40: [type: code] - Delete endpoint (cleanup)

**Purpose**: Critical cleanup step. Endpoints cost money while running.
**Content**:
```python
# IMPORTANT: Delete the endpoint when done to stop incurring costs.
# An ml.m5.xlarge endpoint costs $0.19/hr.
# Forgetting to delete is the most common source of unexpected AWS bills.

resp = runtime_client = None  # release client before deletion

sm_client_endpoint = sess.boto_session.client("sagemaker")
sm_client_endpoint.delete_endpoint(EndpointName=ENDPOINT_NAME)
print(f"Endpoint deletion initiated: {ENDPOINT_NAME}")
print("The endpoint will disappear from the console within 60 seconds.")
print()
print("If you want to redeploy, re-run Cell 40.")
print("If you are done for the day, also delete the endpoint configuration:")
print(f"  aws sagemaker delete-endpoint-config --endpoint-config-name {ENDPOINT_NAME}")
```
**Notes**: Explicit cost reminder. The delete call is immediate but deletion takes ~60s in the console.

---

### Cell 41: [type: markdown] - Homework Extension header

**Purpose**: Homework instruction markdown.
**Content**:
```
## Homework Extension: Add MLflow Tracking to the Training Script

The training job you ran in Lab 2 did not log to MLflow.
This homework adds MLflow tracking.

**Step 1**: Update `scripts_cpu/requirements.txt` to add:
```
sagemaker-mlflow==0.1.0
mlflow==2.16.2
```

**Step 2**: Confirm `train.py` already contains the MLflow tracking block
(look for `if mlflow_enabled:` in the train() function).
It is already there -- it just needs the `--mlflow-tracking-uri` hyperparameter to activate it.

**Step 3**: Create a new estimator with `mlflow-tracking-uri` in hyperparameters:

```python
estimator_mlflow = PyTorch(
    entry_point="train.py",
    source_dir="scripts_cpu",
    framework_version="2.8.0",
    py_version="py312",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=role,
    hyperparameters={
        "epochs": 10,
        "batch-size": 32,
        "lr": 1e-3,
        "hidden-dim": 64,
        "mlflow-tracking-uri": mlflow_tracking_uri,
        "mlflow-experiment-name": "barclays-complaint-classifier",
    },
    sagemaker_session=sess,
)
estimator_mlflow.fit(inputs=None, wait=True)
```

**Step 4**: After the job completes, run Cell 35 again to see the run in MLflow.

**Expected result**: The experiment `barclays-complaint-classifier` shows a new run
with 10 rows of metrics (one per epoch), hyperparameters logged, and a model artifact.
```
**Notes**: Homework is detailed enough to be self-directed. No starter code cell needed -- students write the estimator themselves.

---

### Cell 42: [type: code] - Homework Extension starter cell

**Purpose**: Blank cell for homework work. Keeps the notebook structure clean.
**Content**:
```python
# Homework: MLflow-enabled training job.
# Write your estimator and .fit() call here.
# Refer to the markdown cell above for the exact hyperparameters to add.

estimator_mlflow = None  # YOUR CODE
```
**Notes**: Simple scaffold. No safety-net needed -- this is homework, not in-class.

---

### Cell 43: [type: markdown] - Wrap-up and bridge to Topic 4

**Purpose**: Close the notebook. Key takeaways. Bridge to next topic.
**Content**:
```
## Wrap-Up: What You Learned

1. **Session setup**: `sagemaker.Session()` + `get_execution_role()` + `default_bucket()`
   is the entry point to every SageMaker operation.

2. **S3 read/write**: `sess.upload_data()` for input, `S3Downloader.download()` for artifacts.
   The SageMaker session manages the S3 URI structure automatically.

3. **Source directory contract**: `train.py` + `requirements.txt` (exactly these names).
   SM_MODEL_DIR and SM_CHANNEL_TRAIN are your I/O contract with the container.

4. **Estimator choice**: `PyTorch` estimator for CPU and GPU training.
   `HuggingFace` estimator is GPU-ONLY -- using it on ml.m5.xlarge raises ValueError.

5. **Monitoring**: poll `describe_training_job` for status, read CloudWatch at
   `/aws/sagemaker/TrainingJobs/{job_name}/algo-1`.

6. **MLflow**: `MlflowVersion="2.13.2"` is the only supported version in us-west-2.
   `sagemaker-mlflow:*` is a separate IAM namespace from `sagemaker:*` -- both required.

7. **Model Registry**: register -> approve -> deploy. Never deploy without registering.

8. **Endpoint cleanup**: always delete endpoints when done. They cost $0.19-$0.74/hr
   even when idle.

### Bridge to Topic 4: Transformer Translator Capstone

Topic 4 uses a GPU training job (`ml.g4dn.xlarge`, HuggingFace estimator) for the
Transformer Translator capstone. The pattern is identical to what you did here:
- source_dir with train.py + requirements.txt
- estimator.fit() with wait=False
- poll for completion
- download artifacts

The only differences: `HuggingFace` estimator instead of `PyTorch`, GPU instance,
and a much longer training time (~15-20 min instead of 5 min).

**You now know the pattern. From here, everything is just a different model.**
```
**Notes**: Eight numbered takeaways. Bridge to Topic 4 is explicit and concrete. "You now know the pattern" is the emotional close.

---

## Version Constraint Checklist (all must appear in plan)

Every constraint from SAGEMAKER_LESSONS_LEARNED.md and CORE_TECHNOLOGIES_AND_DECISIONS.md is addressed:

| Constraint | Cell(s) where enforced |
|------------|------------------------|
| `sagemaker>=2.200.0,<3.0.0` | Cell 2 (install), Cell 43 (takeaway 4) |
| `framework_version="2.8.0"`, `py_version="py312"` | Cell 20 (estimator), Cell 19 (beat 1 comment) |
| HuggingFace estimator = GPU only | Cell 19 (Beat 1 broken code), Cell 37 (section header), Cell 43 (takeaway 4) |
| `requirements.txt` exact name | Cell 14 (section header), Cell 15 (comments), Cell 43 (takeaway 3) |
| `eval_strategy` not `evaluation_strategy` | Cell 16/17 train.py -- not applicable (no TrainingArguments in MLP) |
| No `evaluate` library | Cell 16/17 train.py (compute_accuracy inline), Cell 15 requirements.txt |
| `ResourceNotFound` not `ResourceNotFoundException` | Cell 32 (MLflow get_or_create) |
| `MlflowVersion="2.13.2"` | Cell 30 (Beat 1 broken), Cell 32 (Beat 3 correct) |
| `numpy<2` | Cell 2 (install), Cell 15 (requirements.txt), Cell 16/17 (train.py) |
| Plain ASCII only | All cells -- no em dashes, en dashes, Unicode multiplication signs, emojis |
| No hardcoded secrets | Cell 2 (no API keys needed), Cell 29 (section header IAM note) |
| `getpass` for external APIs | Not applicable (no external API keys in this notebook) |

---

## Safety-Net Cell Summary

| Lab | Variable(s) | Safety-net cell | Downstream cells |
|-----|-------------|-----------------|------------------|
| Lab 1 | `account_id`, `artifact_base`, `recent_jobs`, `role_name` | Cell 9 | Cell 10 (verification) |
| Lab 2 | `job_name` | Cell 23 | Cells 24, 26, 27, 28, 35, 36, 38, 39 |

Both safety-net cells must be removed from the solution notebook.

---

## Diagram Checklist

- [ ] Cell 5: `<!-- DIAGRAM: SageMaker remote training job lifecycle ... -->` with link to `../../plans/F2/diagrams/sagemaker_training_lifecycle.mmd`
- [ ] Cell 31: `<!-- DIAGRAM: SageMaker Managed MLflow architecture ... -->` with link to `../../plans/F2/diagrams/sagemaker_mlflow_architecture.mmd`
- [ ] Both diagram `.mmd` files generated by `/build-diagrams` before final commit

---

## Cell Count Summary

| Type | Count |
|------|-------|
| Markdown cells | 15 |
| Code cells | 28 |
| **Total** | **43** |

No markdown chain of more than 3 consecutive cells exists anywhere in this plan.
Every lab has both a safety-net cell and a verification cell.
Every major section uses the four-beat arc (Beat 1 broken -> Beat 2 diagram -> Beat 3 demo -> Beat 4 lab).
Beat 2 appears once per section as required -- only 2 diagrams total for F2.
All labs are Tier 1 (guided, numbered steps, fully scaffolded).
