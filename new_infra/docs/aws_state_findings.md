# AWS Live-State Research Findings — Barclays Environment

**Date:** 2026-06-19
**Account:** 962804699607 (datacouch) — **shared across many instructors**
**Identity used:** `instructor-01` (read-scoped, but IAM reads worked this time)
**Region scope:** **us-west-2 ONLY.** Per Axel: the account is shared; anything not in
us-west-2 is treated as nonexistent for our purposes.
**Method:** 4-cycle research (hypothesis -> live AWS reads -> antithesis -> refinement).
No resources modified. Read-only.

> Tooling note: the `aws` CLI is wrapped by rtk, which compresses JSON output into a
> schema view. All reads below were run via `command /opt/homebrew/bin/aws ... --output json`
> to bypass the wrapper and get real values.

---

## Executive summary

The environment **exists and is live in us-west-2**, but it is **not the clean state the
spec describes** — it is accumulated, multi-instructor, multi-run cruft. The spec's region
(us-east-2) is **wrong**; us-west-2 is the real region. The biggest readiness gap is **the
dataset/scratch S3 buckets do not exist in us-west-2** (the `barclays-*` buckets live in
ap-south-1 and are not ours). IAM is over-provisioned and inconsistently named, with an
explicit `iam:*` Deny in play. Quotas are healthy — not a blocker.

**Verdict: NOT ready.** Provisionable, but needs cleanup + the missing buckets + IAM
normalization before it matches the spec.

---

## Cycle 1 — Landscape (what exists)

**Hypothesis:** Old env destroyed; expect little. Region unknown.

**Evidence:**
- `sagemaker list-domains`:
  - **us-east-2: ZERO domains.** (Spec region is empty.)
  - **us-west-2: ONE domain** — `barclays-training-v2`, **DomainId `d-bquhieanzgod`**,
    InService, created **2026-05-20**. (Different ID than the old snapshot's
    `d-iwf4df7ijy95` -> the domain was rebuilt.)
- `list-user-profiles` on the live domain: **87 user profiles** — a messy superset:
  - `student-01..student-60` (60, old bread-financial-style global numbering) + strays
    `student-019`, `student-4`.
  - `participant-01..25` but **incomplete** (~14 present), plus a typo `Partcipant-10`
    alongside `Participant-10`.
  - ad-hoc human/test profiles: `MikeR17`, `Nicoles-lab`, `Srislab`, `VladsLab`,
    `vueslab`, `aaron-04`, `garylobermier-lab`, `TestingUser1`, `instructor-axel`.

**Antithesis / refinement:** The env is real and in us-west-2, NOT us-east-2. Profiles are
NOT the clean `participant-01..25` the spec wants — they're cruft from many runs and many
instructors sharing the account.

---

## Cycle 2 — Domain + execution role (where permissions live)

**Hypothesis:** Permission issues come from the exec role / student policy not matching spec.

**Evidence — `describe-domain d-bquhieanzgod`:**
| Field | Value | vs spec |
|---|---|---|
| AuthMode | IAM | ✓ matches |
| AppNetworkAccessType | PublicInternetOnly | ✓ matches ("public is simpler") |
| ExecutionRole | `SageMakerStudentExecutionRole` | ✓ name matches |
| VpcId | vpc-0b82cd4a70dd491a3 | (us-west-2) |
| SubnetIds | [subnet-0167807db1a69d541] | single subnet |

**Evidence — `SageMakerStudentExecutionRole` policies:**
- Attached managed: `AmazonSageMakerFullAccess` (✓ spec), **PLUS** `AmazonEC2FullAccess`,
  `CloudWatchLogsFullAccess`, `AmazonS3FullAccess`,
  `AmazonSageMakerCanvasSMDataScienceAssistantAccess`, and custom
  **`BreadFinancialCourseExtras`** (wrong course — leftover).
- Inline: `BedrockAgentCoreMWAAFromStudio`, `KBVectorStoreAccess`,
  `SageMakerMLflowAccess`, `SageMakerTrainingSupport` — **all bread-financial-specific**
  (MWAA/Airflow, KB vector store, MLflow). None are in the Barclays spec.

**Antithesis / refinement:** The exec role is a **reused bread-financial role**, heavily
over-provisioned. Because it has `AmazonS3FullAccess`, **S3 access from the Space is NOT
permission-limited** — so the permission pain is unlikely to be the exec role's S3. The
role is "too much," not "too little." Real least-privilege per spec would be a rebuild.

---

## Cycle 3 — Student IAM groups/policies (the explicit-Deny landmine)

**Hypothesis:** Students get permissions via a `barclays-students` group (per spec).

**Evidence:**
- **`barclays-students` group does NOT exist** (NoSuchEntity).
- Groups that DO exist: `Barclays-batch-1`, `Barclays_Batch-2` (inconsistent
  hyphen/underscore naming), `StudentSageMakerGroup`, plus bread-financial groups
  (`bread-academy-*`, `BreadFinancialStudents`), `AWS_Cloud_Foundations`, `nutanix`,
  `training`.
- **All three candidate student groups have ZERO members.** So participant IAM users (if
  any) are NOT getting permissions through these groups — permissions are attached some
  other way (directly to users, or users don't exist yet).
- `Barclays-batch-1` vs `Barclays_Batch-2` carry **divergent** policy sets (batch-2 has
  extra `AmazonS3FullAccess` + different inline `barclays_policy` vs `barclays_policy_v2`).
- Both Barclays groups attach **`dc-ec2-policy`**.

**Evidence — `dc-ec2-policy` (default version v164 — heavily iterated, 20 statements):**
- **Explicit `Deny` Sid `DenyAttachIAMPolicy` on `iam:*`** — overrides every Allow. This
  is what AccessDenied'd `instructor-01` on group-policy reads in the old snapshot, and it
  kills the spec's `IAMSelfInspection` Sid for students too.
- **Explicit `Deny` Sid `EC2TpyeInstanceDeny`** — only blocks LEGACY instance types
  (m2/c3/c4/g2/etc.). Does **NOT** block t3.medium, c5.xlarge, or g4dn.xlarge. So it is
  **not** the cause of any JupyterLab instance-type problem.
- **`EC2` Allow is region-restricted** to `us-west-2, us-west-1, ap-south-1, us-east-1`.
  **us-east-2 is NOT in the list** -> further proof the spec's us-east-2 is wrong and
  us-west-2 is correct.

**Antithesis / refinement:** The student-permission surface is fragmented and partly
self-blocking (`iam:*` Deny). Empty group membership means the live permission path for
participants is unverified from here — it's whatever is attached directly to their users.

---

## Cycle 4 — S3 buckets + quotas (the explicit asks)

**Hypothesis:** Buckets + quotas exist per spec.

**Evidence — S3:**
- Buckets named `barclays-prompt-eng-data`, `barclays-genai-devs-data`,
  `barclays-training-scratch` **DO** exist by name — **but all three are in `ap-south-1`
  (Mumbai).**
- **In us-west-2 (our only region), these buckets DO NOT EXIST.** Per the
  us-west-2-only rule, treat them as **absent**. They are someone else's / another
  region's resources.

**Evidence — Quotas (us-west-2):**

SageMaker JupyterLab App quotas (the ones that gate Spaces):
| Quota | Value | Need | OK? |
|---|---|---|---|
| Studio JupyterLab Apps on ml.c5.xlarge | **100** | 25 | ✓ |
| Studio JupyterLab Apps on ml.g4dn.xlarge | **200** | 25 | ✓ |
| Studio JupyterLab Apps on ml.t3.medium | 100 | (n/a now) | ✓ |

SageMaker training/processing/endpoint (headroom):
| Family | training job | processing job | endpoint |
|---|---|---|---|
| ml.c5.xlarge | 100 | 100 | 100 |
| ml.g4dn.xlarge | 100 | 100 | 100 |

EC2 On-Demand vCPU (the real launch gate):
| Quota | Value | Need | OK? |
|---|---|---|---|
| Running On-Demand G and VT instances | **768** | 100 (g4dn x25) | ✓ |
| Running On-Demand Standard (A,C,D,H,I,M,R,T,Z) | **2500** | c5 Spaces + t3 VMs | ✓ |
| Running On-Demand P instances | 768 | n/a | ✓ |

**Antithesis / refinement:** **Quotas are NOT a blocker** — all healthy with large
headroom (consistent with a shared multi-instructor account that's been raised before).
The S3 region mismatch IS a blocker for the spec as written.

---

## Key findings (what we actually have, us-west-2)

| Thing | Spec wants | Live us-west-2 reality | Gap |
|---|---|---|---|
| Region | us-east-2 | **us-west-2** (domain + EC2 policy region-locked) | **Spec is wrong; use us-west-2** |
| SageMaker domain | `barclays-training` | `barclays-training-v2` `d-bquhieanzgod`, InService | name differs; exists |
| User profiles | `participant-01..25` (clean) | 87 mixed (student-01..60, partial participant-*, test profiles) | **messy; needs cleanup** |
| Exec role | SageMakerStudentExecutionRole + minimal inline | exists, but over-provisioned w/ bread-financial leftovers | over-scoped, not under |
| Student group | `barclays-students` | does NOT exist; `Barclays-batch-1`/`Barclays_Batch-2` (0 members, divergent) | **missing/forked** |
| IAM self-inspect | allowed | **explicitly denied** (`dc-ec2-policy` `iam:*` Deny) | blocked |
| Dataset buckets | 2 read-only in region | exist only in **ap-south-1** -> absent in us-west-2 | **MISSING in us-west-2** |
| Scratch bucket | 1 RW in region | exists only in **ap-south-1** -> absent in us-west-2 | **MISSING in us-west-2** |
| JupyterLab quotas | 25 each (req 50) | 100 (c5) / 200 (g4dn) | ✓ healthy |
| EC2 vCPU quotas | 200 G, 200 C/T | 768 G, 2500 Standard | ✓ healthy |
| Active Spaces | per-student on demand | **0 right now** | none running |

### Assumptions that were WRONG going in
1. "Probably nothing exists" — wrong, a full domain + 87 profiles is live.
2. Spec region us-east-2 — wrong, real region is us-west-2 (3x confirmed: domain location,
   EC2-policy region allow-list, empty us-east-2).
3. Permission problems = too little S3 access — wrong, exec role has S3FullAccess; the
   real IAM issue is fragmentation + an `iam:*` Deny, and the real data issue is
   cross-region buckets.

---

## Trade-offs / what this means

- **Buckets are the #1 readiness blocker.** The spec's three buckets do not exist in
  us-west-2. Either create them in us-west-2 (clean, matches spec intent) or repoint the
  course at whatever us-west-2 data location actually holds the datasets. Cross-region to
  ap-south-1 is not acceptable for a class.
- **IAM is over-scoped, not under-scoped.** Lower risk for "students can't do X," higher
  risk for drift/security and the `iam:*` self-inspection Deny. A clean `barclays-students`
  group per spec would remove ambiguity, but in a SHARED account that touches other
  instructors' setups — proceed carefully, do not delete shared policies.
- **Profiles are cruft.** 87 vs the 25 wanted. Safe to leave extras, but a clean
  `participant-01..25` set (consistent casing, no typos) avoids handout confusion.
- **Quotas need no action.**

---

## Decisions locked (2026-06-19)

- **Region = us-west-2.** (Spec's us-east-2 is wrong/empty.)
- **Identity scheme = `participant-01..25`** (25). student-*/test profiles are cruft.
- **Bucket names** (spec names collide globally with ap-south-1, can't reuse):
  - `datacouch-barclays-prompt-eng-usw2` (read-only datasets)
  - `datacouch-barclays-genai-devs-usw2` (read-only datasets)
  - `datacouch-barclays-scratch-usw2` (read/write)
- **Scope rule: only UNDER-scoped permissions are a problem.** Over-scoped roles/policies
  (the exec role's bread-financial leftovers, FullAccess attachments) are **fine — do NOT
  trim them.** The only failure mode we care about is "a participant can't do something the
  class needs."

## Proposed plan (NOT executed — findings only)

1. **Create the 3 us-west-2 buckets** (`datacouch-barclays-prompt-eng-usw2`,
   `datacouch-barclays-genai-devs-usw2`, `datacouch-barclays-scratch-usw2`): versioning on,
   public blocked, SSE-S3, 30-day lifecycle on scratch. Populate datasets.
2. **Verify the participant permission path** — the `Barclays-batch-*` groups have 0
   members, so confirm how participant IAM users actually receive permissions today, and
   that the JupyterLab train/register/deploy-in-notebook flow is NOT under-scoped. Do this
   WITHOUT disturbing other instructors' shared resources.
3. **Ensure the new bucket ARNs are reachable** by the exec role + participant users
   (today the exec role has S3FullAccess so this is satisfied; just confirm).
4. **Clean participant profile set** `participant-01..25` (optional — extras are harmless).
5. **Run lab-0 as a `participant-NN` identity** to prove the full stack end to end.
6. Quotas: no action (healthy).

> Note: the `iam:*` Deny in `dc-ec2-policy` blocks the spec's `IAMSelfInspection` Sid. That
> is an UNDER-scope (a thing students can't do), so it matters ONLY if lab-0 / the course
> actually needs self-inspection. If not, drop that Sid from the spec. If yes, it must be
> reconciled — but carefully, since `dc-ec2-policy` is shared.

---

## Items 2 + 3 verified (participant permission path + bucket reachability) — 2026-06-19

Followed up on the two open items with live reads + IAM policy simulation.

### How participants actually get permissions (the real path)
- **60 `participant-NN` IAM users exist** (participant-01..60, clean sequence).
- A sampled participant (`participant-01`) has **no direct/inline policies**. Its single
  permission source is membership in group **`bread-academy-students`** — a BREAD-FINANCIAL
  group, NOT a Barclays one. (The `Barclays-batch-*` groups are empty/unused.)
- `bread-academy-students` (120 members) carries: managed `BreadAcademyStudentPolicy` (v6)
  + inline `SageMakerAllInline` (`sagemaker:*`) + inline `S3VectorsAllInline`.

### What that grants vs the Barclays JupyterLab flow needs
| Need | Source | Verdict |
|---|---|---|
| Open Studio + launch JupyterLab Space (CreatePresignedDomainUrl / CreateApp / CreateSpace) | inline `sagemaker:*` | ✅ allowed (simulated) |
| PassRole -> SageMakerStudentExecutionRole | `BreadAcademyStudentPolicy` SageMakerPassRole | ✅ |
| CreateModel / Endpoint / TrainingJob / ProcessingJob / register ModelPackage | `SageMakerWorkflow` Sid | ✅ |
| ECR pull (DLC images) | `ECRRead` Sid | ✅ |
| Bedrock invoke | `BedrockInference` Sid | ✅ |
| CloudWatch logs | `CloudWatchAndLogs` Sid | ✅ |
| S3 on the NEW `datacouch-barclays-*` buckets (as the IAM user) | `S3CourseBuckets` lists only `bread-academy-*` | **❌ implicitDeny (GAP)** |

### IAM simulation results (hard evidence)
- `participant-01` -> `s3:GetObject/PutObject/ListBucket` on the 3 new buckets: **implicitDeny**.
- `SageMakerStudentExecutionRole` -> `s3:GetObject/PutObject` on the new buckets: **allowed**
  (has `AmazonS3FullAccess`).
- `participant-01` -> `sagemaker:CreatePresignedDomainUrl / CreateApp / CreateSpace`: **allowed**.

### Conclusion (item 2 + 3)
- **Item 3 (bucket reachability): RESOLVED.** Once the 3 `datacouch-barclays-*` buckets are
  created, the **Studio Space (exec role) reads/writes them with no policy change.** Inside
  the notebook — where all the class work happens — S3 is NOT under-scoped.
- **Item 2 (participant path): RESOLVED with ONE gap.** Participants get everything the
  in-Space JupyterLab flow needs (open Studio, launch Space, train/register/deploy
  in-notebook) via the `bread-academy-students` group. The ONLY under-scope is the
  **participant IAM user** (not the Space) accessing the new buckets directly — e.g.
  `aws s3 cp` from the Ubuntu VM or a boto3 call running under the user's own creds. That
  returns implicitDeny because `BreadAcademyStudentPolicy`'s `S3CourseBuckets` Sid only
  lists `bread-academy-*` ARNs.

### The one fix needed
Add the 3 new bucket ARNs to participant S3 access. Cleanest: extend the `S3CourseBuckets`
Sid in `BreadAcademyStudentPolicy` (or add a small dedicated policy) to include:
`arn:aws:s3:::datacouch-barclays-prompt-eng-usw2(/*)`,
`arn:aws:s3:::datacouch-barclays-genai-devs-usw2(/*)`,
`arn:aws:s3:::datacouch-barclays-scratch-usw2(/*)`.
CAUTION: `BreadAcademyStudentPolicy` is SHARED (120 members across bread + barclays). Adding
ARNs is purely additive (over-scope, which is fine) and safe; do NOT remove anything.

> Whether this fix is even required depends on the course: if students only ever touch the
> buckets from INSIDE the Space (exec role), the gap never bites. It only bites for direct
> IAM-user S3 access (VM `aws s3 cp`, or boto3 under user creds). The lab-0 sanity notebook
> runs in the Space -> exec role -> already allowed.

## Snapshot finding — participant Studio profiles are INCOMPLETE (2026-06-19)

The scoped snapshot (`new_infra/snapshot/`) surfaced a new gap by cross-checking IAM users
vs SageMaker Studio user profiles:

- **60 `participant-NN` IAM users exist** (participant-01..60).
- But only **14 of participant-01..25 have a SageMaker Studio user PROFILE**:
  present = 01,02,06,07,08,09,11,12,14,15,19,20,21,25.
- **11 are MISSING a profile**: participant-03, 04, 05, 10, 13, 16, 17, 18, 22, 23, 24.

Implication: those 11 participants have an IAM user (can log into the console) but **no
Studio profile -> cannot open a JupyterLab Space.** Terraform must create the missing
`participant-NN` user profiles (all 25) in domain `d-bquhieanzgod`, exec role
`SageMakerStudentExecutionRole`. (0 spaces are live right now — students create on demand.)

This is a concrete "NOT ready" item, additive to the bucket gap.

## Snapshot tooling notes (for re-running)

- Snapshot script: `new_infra/snapshot/snapshot.sh` (Steampipe, us-west-2 + Barclays scope).
- Steampipe AWS plugin v1.30.2 has **no** `aws_sagemaker_user_profile` or
  `aws_sagemaker_space` table -> those two sections use an **AWS CLI fallback** in the
  script. The script sets `export AWS_PROFILE=datacouch` so the CLI fallbacks hit the right
  account (without it they silently return the default account = empty).
- Output is **gitignored** (`new_infra/snapshot/`): it contains full IAM policy JSON,
  account/resource IDs, participant ARNs. Re-run with `bash new_infra/snapshot/snapshot.sh`.

## Sources (live AWS reads, us-west-2, account 962804699607)

- `sts get-caller-identity`
- `sagemaker list-domains` (us-east-2 empty, us-west-2 = d-bquhieanzgod)
- `sagemaker describe-domain --domain-id d-bquhieanzgod`
- `sagemaker list-user-profiles --domain-id-equals d-bquhieanzgod` (87)
- `sagemaker list-spaces --domain-id-equals d-bquhieanzgod` (0)
- `iam list-attached-role-policies / list-role-policies --role-name SageMakerStudentExecutionRole`
- `iam list-groups`, `get-group`, `list-attached-group-policies`, `list-group-policies`
- `iam get-policy / get-policy-version --policy-arn .../dc-ec2-policy` (v164)
- `s3api list-buckets`, `get-bucket-location` (all 3 -> ap-south-1)
- `service-quotas list-service-quotas --service-code sagemaker --region us-west-2`
- `service-quotas list-service-quotas --service-code ec2 --region us-west-2`
