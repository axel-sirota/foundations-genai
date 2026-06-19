# Barclays Training Environment — Desired State

Source of truth for the **new** AWS environment we want Datacouch to provision for the
Barclays courses. Distilled from the technical setup spec (Axel Sirota, April 2026).

This describes the **target**. It is what we will check live AWS against. Nothing here
is assumed to exist yet — the previous environment was destroyed.

---

## 0. TL;DR — what "ready" means

The environment is ready when, for a given course, **every** one of these is true:

1. 25 Ubuntu VMs (one per student) reachable by SSH, baked from the spec AMI.
2. 1 SageMaker Studio domain with 25 user profiles (`student-01`..`student-25`).
3. Each student can launch a JupyterLab Space pinned to the course's instance type.
4. Inside the Space: `pip install` works, GPU is visible (Gen AI for Devs only),
   ChromaDB runs local, S3 read on both dataset buckets, S3 read/write on scratch,
   HF Hub reachable, OpenAI + Anthropic reachable.
5. 25 IAM users in the `barclays-students` group, each with console password + access key.
6. Service quotas raised (see Section 6) — approved, not just requested.
7. 3 S3 buckets exist and are populated (see Section 7).
8. The lab-0 sanity notebook passes end to end for a student identity.

If any of these fails, the environment is NOT ready. Section 12 is the literal checklist.

---

## 1. Courses covered

| Course | Length | Cohort size |
|---|---|---|
| Generative AI: Prompt Engineering for Software Developers | 3 days | 20 students |
| Generative AI for Developers | 3 days | 20 students |

Note: spec body provisions for **25** per course (and quotas doubled to 50) for headroom,
even though nominal cohort is 20. We build for 25.

---

## 2. Region

**us-west-2 (Oregon)** for ALL resources.

> RESOLVED 2026-06-19 from live AWS: the spec originally said us-east-2, but that region is
> EMPTY. The live `barclays-training-v2` domain is in **us-west-2**, and the shared
> `dc-ec2-policy` only allows EC2 in `us-west-2, us-west-1, ap-south-1, us-east-1` (NOT
> us-east-2). So **us-west-2 is canonical.** The account (962804699607, datacouch) is
> shared across many instructors — anything not in us-west-2 is treated as nonexistent.
> Every resource (domain, VMs, all 3 buckets, IAM region-scoped ARNs, quotas) lives in
> us-west-2.

---

## 3. Architecture (what each student gets)

Two things per student:

1. **Personal Ubuntu VM (EC2)** — thin client. Terminal + browser, reaches AWS console,
   runs local Python if needed. Mirrors "my laptop" in a Colab workflow.
2. **Personal SageMaker Studio user profile** in a shared domain. From Studio UI the
   student launches a **JupyterLab Space** pinned to one instance type. All real work
   (notebooks, training, RAG, chatbot) happens here. Inside the Space everything is
   **localhost**: pip install works, ChromaDB local, API keys in env vars, localhost URLs.

Critically, in the NEW model: **no SageMaker training jobs, no endpoints, no distributed
training**. Everything runs *inside* the Space on its single pinned instance. (The old
"just in case" training/endpoint IAM permissions are kept as headroom, Section 5, but the
class flow does not require them.)

```
Barclays student laptop (browser + SSH)
   |
   +--> Ubuntu VM (per student, t3.medium)
   |       python3, git, awscli, vscode, pytorch, transformers
   |
   +--> AWS Console / SageMaker Studio
           Studio Domain (1)
            +-- User Profile: participant-01..participant-25
            |     JupyterLab Space  (pinned instance type per course)
            |       localhost: notebook + ChromaDB + training-in-notebook
            |
            +-- S3 buckets (us-west-2)
                  datacouch-barclays-prompt-eng-usw2   (read-only)
                  datacouch-barclays-genai-devs-usw2   (read-only)
                  datacouch-barclays-scratch-usw2      (read/write)
```

---

## 4. Ubuntu VM (per student)

### Spec
| Item | Value |
|---|---|
| AMI | Ubuntu Server 24.04 LTS (latest) |
| Instance type | t3.medium (2 vCPU, 4 GB RAM) |
| Root volume | 30 GB gp3 |
| Count | 25 per course (50 total if both run concurrently) |
| Access | SSH (key pair) + browser VS Code Remote / code-server |
| Public IP | Yes, or behind Datacouch bastion (Datacouch choice) |
| Security group | Inbound 22 from instructor + student IPs, egress all |

### Pre-installed software (bake into AMI)
System: `build-essential curl wget unzip jq tree htop`
Python: `python3.11 python3.11-venv python3-pip` (pip upgraded)
Git: `git` (>= 2.30)
AWS CLI v2 (from awscli-exe-linux-x86_64.zip)
VS Code: `snap install code --classic`
Python libs (CPU-only): `torch (cpu index)`, `transformers`, `datasets`, `accelerate`,
`sentence-transformers`, `chromadb`, `openai`, `anthropic`, `pymupdf`, `beautifulsoup4`,
`jupyter`, `jupyterlab`, `pandas numpy scikit-learn matplotlib`.

**No Docker.** Per instructor request.

### AWS credentials on the VM
Each VM has AWS CLI configured with that student's IAM access key. Datacouch drops the key
into `~/.aws/credentials` per VM before class, tied to that student's IAM user.

---

## 5. SageMaker Studio Domain + IAM

### Domain
| Item | Value |
|---|---|
| Domain name | barclays-training-v2 (live; `d-bquhieanzgod`) |
| Region | us-west-2 |
| Authentication | IAM (one IAM user per participant) |
| Network | VPC-only or public (Datacouch preference; public is simpler) |
| Default execution role | SageMakerStudentExecutionRole |
| Storage | Default EFS (Studio-managed) |

### User profiles
25 profiles per course: `participant-01` .. `participant-25`. Same domain, same execution
role, each gets its own EFS home automatically. (The live domain currently has 87 mixed
profiles — student-01..60, partial participant-*, and test profiles — which are cruft to
ignore/clean. Canonical scheme going forward is `participant-NN`.)

### JupyterLab Space instance type (pinned per course)
| Course | Instance type | Specs | ~$/hr |
|---|---|---|---|
| Prompt Engineering | ml.c5.xlarge | 4 vCPU, 8 GB, no GPU | ~$0.20 |
| Gen AI for Devs | ml.g4dn.xlarge | 4 vCPU, 16 GB, 1x T4 16GB | ~$0.74 |

Rationale: Prompt Eng is API calls + PyMuPDF/BS4/ChromaDB/web — no GPU ever. **The spec
originally pinned `ml.t3.medium` here, but in practice it was found too small; we use
`ml.c5.xlarge` instead** (4 vCPU, 8 GB, compute-optimized, no GPU). Gen AI for Devs runs
full-finetune Flan-T5, transfer-learn DistilBERT, LoRA, QLoRA, quant-aware training. T4 is
the Colab-equivalent and smallest AWS option that handles all of these at educational
scale. No A10G+ needed (instructor confirmed).

### Space storage
Default 100 GB EBS per Space for Gen AI for Devs (checkpoints, HF cache). 30 GB for Prompt Eng.

### Lifecycle / auto-shutdown
No idle shutdown. Datacouch tears Spaces down after class.

### Studio image
Default SageMaker Distribution image (latest; PyTorch, Transformers, CUDA). Students
`pip install` anything missing. No custom images.

### IAM model
**Per-student IAM user**, each with: console access (open Studio in browser), access key +
secret (for `aws configure` on the VM), membership in group `barclays-students`.

#### `barclays-students` group policy (Sids we care about)
- `SageMakerStudioAccess` — CreatePresignedDomainUrl, Describe Domain/UserProfile,
  List/Create/Delete/Describe **App**, Create/Update/Delete/Describe/List **Space**. (`*`)
- `SageMakerTrainingAndInferenceJustInCase` — Create/Describe/Stop TrainingJob,
  ProcessingJob, CreateModel, Create/Delete Endpoint + EndpointConfig, InvokeEndpoint,
  Describe/List Endpoints. (`*`) — **headroom, not required by class flow.**
- `PassRoleToSageMaker` — `iam:PassRole` on
  `arn:aws:iam::<ACCOUNT_ID>:role/SageMakerStudentExecutionRole`, condition
  `iam:PassedToService = sagemaker.amazonaws.com`.
- `S3DatasetsRead` — GetObject, ListBucket on `datacouch-barclays-prompt-eng-usw2` and
  `datacouch-barclays-genai-devs-usw2` (+ `/*`).
- `S3ScratchBucketReadWrite` — Get/Put/Delete/ListBucket on `datacouch-barclays-scratch-usw2` (+ `/*`).
- `ECRReadForDeepLearningContainers` — ecr + ecr-public auth/pull (`*`).
- `CloudWatchLogsForTrainingJobs` — log group/stream/put/describe/get on
  `arn:aws:logs:*:*:log-group:/aws/sagemaker/*`.
- `BedrockInvokeOptional` — InvokeModel, InvokeModelWithResponseStream,
  ListFoundationModels (`*`).
- `IAMSelfInspection` — GetUser, ListAttachedUserPolicies, ListGroupsForUser on
  `arn:aws:iam::*:user/${aws:username}`.
- `ServiceQuotasRead` — ListServiceQuotas, GetServiceQuota (`*`).

(Full JSON lives in `new_infra/docs/iam/` once we transcribe it — see Section 13.)

#### `SageMakerStudentExecutionRole` (the role the Space assumes)
- Attach **AmazonSageMakerFullAccess** (AWS managed) — pragmatic classroom choice,
  removes a class of in-class surprises. Tight least-privilege acceptable to Datacouch
  if preferred.
- Custom inline policy: read on the two dataset buckets, read/write on scratch bucket.
- Trust policy: `sagemaker.amazonaws.com`.

#### KMS note
If the buckets use customer-managed KMS (SSE-KMS), both the student users AND the exec
role additionally need `kms:Decrypt` (+ `kms:GenerateDataKey` for scratch writes) on the
key. SSE-S3 (default) needs no extra KMS permission.

---

## 6. Service quotas (raise BEFORE class — 24-72h approval)

These are the quota lines we actually track. The Prompt Eng JupyterLab moved from
`ml.t3.medium` to **`ml.c5.xlarge`** (t3.medium too small), so the relevant SageMaker
quota families are **`ml.c5.xlarge`** and **`ml.g4dn.xlarge`**.

### JupyterLab Apps (Studio Space) quotas
| Quota | Min needed | Request | Why |
|---|---|---|---|
| Studio JupyterLab Apps on ml.c5.xlarge | 25 | 50 | One per student, Prompt Eng, doubled |
| Studio JupyterLab Apps on ml.g4dn.xlarge | 25 | 50 | One per student, Gen AI for Devs, doubled |

### Training / processing / endpoint quotas (headroom — class flow runs in-notebook, but kept)
| Quota | Why |
|---|---|
| ml.g4dn.xlarge for training job usage | GPU remote training, just in case |
| ml.g4dn.xlarge for processing job usage | GPU remote processing, just in case |
| ml.g4dn.xlarge for endpoint usage | GPU remote inference, just in case |
| ml.c5.xlarge for training job usage | CPU remote training, just in case |
| ml.c5.xlarge for processing job usage | CPU remote processing, just in case |
| ml.c5.xlarge for endpoint usage | CPU remote inference, just in case |

### EC2 + account-level quotas
| Quota | Min needed | Request | Why |
|---|---|---|---|
| Running On-Demand G and VT instances (vCPU) | 100 | 200 | g4dn.xlarge = 4 vCPU x 25 |
| Running On-Demand C instances (vCPU) | 100 | 200 | c5.xlarge Studio Spaces (Prompt Eng) |
| Running On-Demand T instances (vCPU) | 100 | 200 | t3.medium Ubuntu VMs |
| Studio domains | 1 | 3 | Room for a staging domain |
| User profiles per domain | 50 | 100 | 25 x 2 courses, doubled |

NOTE on regions/families: the JupyterLab-App quotas and the training/processing/endpoint
quotas live under **SageMaker**; the G / C / T vCPU quotas live under **EC2**. All must be
raised in the SAME target region. Submit early (24-72h approval).

> CONFIRM: exact current values per line (what we have today vs what to request). The list
> above is the set of quota lines we care about; fill in actual numbers from the Service
> Quotas console for the target region.

---

## 7. S3 buckets (us-west-2)

> Names changed from the spec. S3 names are GLOBALLY unique and the spec's three
> `barclays-*` names are already taken (they exist in ap-south-1, not ours). New
> us-west-2 names below, decided 2026-06-19.

| Bucket | Access | Contents |
|---|---|---|
| `datacouch-barclays-prompt-eng-usw2` | read-only | `barclays-products/` (brochures, T&Cs, FAQ PDFs for NLP+RAG), `sample-conversations/` |
| `datacouch-barclays-genai-devs-usw2` | read-only | `datasets/` (IMDB subset, small summarization, small NER, tiny Multi30k), `models/` (optional local Flan-T5-base, DistilBERT-base mirrors) |
| `datacouch-barclays-scratch-usw2` | read/write | participant prefixes `s3://.../participant-NN/` for checkpoints/artifacts. No IAM prefix isolation — convention only. |

### Bucket config (all three)
- Versioning **on**
- Public access **fully blocked**
- Default encryption **SSE-S3** (unless Barclays requires SSE-KMS — then KMS note applies)
- Lifecycle on scratch: delete objects older than **30 days**

---

## 8. API keys (OpenAI, Anthropic)

Instructor hands out verbally in class. Students paste into a notebook cell as env vars
(Colab-style). Datacouch does NOT provision these and does NOT inject via lifecycle configs.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

---

## 9. What Datacouch must deliver (per course, before day 1)

- 25 Ubuntu VMs (Section 4 AMI), SSH-reachable, one per participant.
- 1 SageMaker Studio domain in us-west-2 with 25 user profiles (`participant-01..25`).
- 25 IAM users in the `barclays-students` group, each with console password + access key.
- Service quotas raised per Section 6 (approved).
- 3 S3 buckets populated per Section 7.
- Per-student handout: SSH command + key, AWS Console URL, IAM username + password,
  Studio profile name, dataset + scratch bucket names.
- Lab-0 sanity notebook (Section 11) that verifies the whole stack.

---

## 10. Cost estimate (rough; 3 days x 8h, 25 students)

| Course | Resource | Hours | Students | ~Total |
|---|---|---|---|---|
| Prompt Eng | ml.c5.xlarge Space | 24 | 25 | $120 |
| Prompt Eng | t3.medium VM | 24 | 25 | $25 |
| Gen AI for Devs | ml.g4dn.xlarge Space | 24 | 25 | $445 |
| Gen AI for Devs | t3.medium VM | 24 | 25 | $25 |
| All | S3, CloudWatch, misc | | | $15 |

Per-course rough: Prompt Eng ~$145 (up from ~$55 after the t3.medium -> c5.xlarge Space
change), Gen AI for Devs ~$470. Assumes Datacouch tears down Spaces + VMs at end of each
day (or at least end of course). The EC2 thin-client VM stays t3.medium for both courses;
only the Prompt Eng **Studio Space** changed.

---

## 11. Lab-0 sanity check (must pass for a student identity)

The notebook students run first. It verifies:
- Python version
- `torch.cuda.is_available()` -> True (Gen AI for Devs only), False (Prompt Eng) — intended
- S3 read on both dataset buckets
- S3 write on scratch
- HF Hub reachability
- OpenAI + Anthropic connectivity with a throwaway key

---

## 12. Readiness checklist (the literal gate)

Per course:

- [x] Region = us-west-2, confirmed against live AWS (Section 2) — RESOLVED
- [x] Quotas healthy (JupyterLab 100 c5 / 200 g4dn; EC2 768 G / 2500 Standard) — no action
- [x] SageMaker domain exists in us-west-2 (`barclays-training-v2` / `d-bquhieanzgod`), IAM auth, exec role attached
- [ ] 3 S3 buckets exist **in us-west-2** (`datacouch-barclays-prompt-eng-usw2`,
      `datacouch-barclays-genai-devs-usw2`, `datacouch-barclays-scratch-usw2`), versioned,
      public-blocked, SSE-S3 — **MISSING today, must create**
- [ ] Dataset buckets populated; scratch lifecycle (30d) set
- [ ] 25 user profiles `participant-01..25` InService (clean set; ignore student-*/test cruft)
- [ ] JupyterLab Space launches on the pinned instance type (c5.xlarge / g4dn.xlarge)
- [ ] Exec role grants what the class NEEDS (dataset RO + scratch RW on the NEW bucket names).
      Over-scope is fine; only UNDER-scope blocks. Verify the new bucket ARNs are reachable.
- [ ] Participant IAM users exist with the permissions they NEED for the JupyterLab flow
      (only under-scope matters). Normalize how they get it (group `barclays-students` OR
      reuse a `Barclays-batch-*` group) — but do NOT disturb other instructors' resources.
- [ ] 25 Ubuntu VMs up, SSH-reachable, AMI software present, `~/.aws/credentials` set
- [ ] Lab-0 passes end to end as a `participant-NN` identity
- [ ] (If SSE-KMS) kms:Decrypt / GenerateDataKey granted to users + exec role

---

## 13. Consistency goal (Colab parity)

Student code must work exactly as in Colab:
- `pip install <x>` works
- `torch.cuda.is_available()` -> True (Gen AI for Devs) / False (Prompt Eng), intended
- `chromadb.PersistentClient(path="./chroma_db")` stores in Space EFS home
- `OpenAI()` / `Anthropic()` read env vars set at notebook top
- `AutoModel.from_pretrained("google/flan-t5-base")` downloads to local HF cache
- Any Colab notebook runs in the Space with zero changes

The Ubuntu VM is the consistent shell for non-notebook work (git clone, curl, aws s3 cp).

---

## 14. Resolved vs still-open (post live-AWS research, 2026-06-19)

### RESOLVED against live AWS (see `aws_state_findings.md`)
1. **Region = us-west-2.** Spec's us-east-2 is wrong (empty + EC2 policy region-locked).
2. **Account = 962804699607 (datacouch)**, shared across many instructors. us-west-2 only.
3. **Domain exists**: `barclays-training-v2` / `d-bquhieanzgod`, InService, IAM, public.
4. **Quotas healthy** — no action needed.
5. **Identity scheme = `participant-01..25`** (25). student-*/test profiles are cruft.
6. **Bucket names** = `datacouch-barclays-{prompt-eng,genai-devs,scratch}-usw2` (spec names
   collide globally with ap-south-1 buckets that aren't ours).
7. **Scope policy: only UNDER-scoped permissions matter.** Over-scoped roles/policies are
   explicitly fine and NOT to be trimmed.

### STILL OPEN
1. **Create the 3 us-west-2 buckets** + populate datasets + set scratch 30d lifecycle.
2. **Bucket encryption** — SSE-S3 (default, assumed) vs SSE-KMS. If KMS, add kms perms.
3. **Participant permission path** — the `Barclays-batch-*` groups have 0 members, so how
   do participant users actually get permissions today? Confirm and normalize WITHOUT
   touching other instructors' shared resources.
4. **Exec role new-bucket access** — current role has S3FullAccess (so it'll work), but if
   anyone later tightens it, ensure the NEW bucket ARNs are covered. Under-scope is the
   only failure mode we care about.
5. **VM fleet** — 25 Ubuntu t3.medium VMs not yet verified/provisioned in us-west-2.

---

_Authored from the Barclays Technical Setup Specification (Axel Sirota, April 2026),_
_reconciled against live AWS on 2026-06-19. See `aws_state_findings.md` for the evidence._
