# RAW_FINDINGS — barclays-tf-converge

Intake for the Terraform convergence epic. Each finding is a concrete delta between
**desired state** (`new_infra/docs/desired_state.md`) and **live us-west-2 reality**
(`new_infra/docs/aws_state_findings.md` + `new_infra/snapshot/`).

- Scope: **us-west-2 only**, account **962804699607 (datacouch)**, domain
  **barclays-training-v2 / d-bquhieanzgod**.
- Existing SHARED resources (domain, exec role `SageMakerStudentExecutionRole`, the
  Barclays VPC, the `bread-academy-students` group + `BreadAcademyStudentPolicy`) are
  referenced as TF **data sources** and NEVER recreated.
- Rule: over-scoped IAM is fine; **only under-scope is a defect.**
- User runs `terraform apply` himself (never automated here).

Adapted skill semantics: "object" = an AWS/TF resource; "DDL/schema phase" = a resource
that CREATES infra other tasks depend on (e.g. a bucket whose ARN a policy references);
"consume phase" = a resource/policy that READS/REFERENCES an already-created object;
"DoD test" = `terraform validate` + a clean `terraform plan` (and a targeted read-back).

---

## Findings

### B1 — TF project scaffold does not exist
**Verbatim (state):** "new_infra/terraform/" does not exist. There is no provider config,
no backend, no data-source wiring for the existing domain/role/VPC.
**Why it matters:** nothing can be planned/applied until the project, provider (pinned
us-west-2 + profile datacouch), and the data sources for shared resources exist.
**Tags:** scaffold, provider, data-sources, foundational
**Guessed section:** 00-foundation

### B2 — The 3 us-west-2 dataset/scratch buckets do NOT exist
**Verbatim (findings):** "In us-west-2 (our only region), these buckets DO NOT EXIST ...
they are someone else's / another region's resources." Target names (collision-safe):
`datacouch-barclays-prompt-eng-usw2`, `datacouch-barclays-genai-devs-usw2`,
`datacouch-barclays-scratch-usw2`.
**Why it matters:** #1 readiness blocker. No datasets, no scratch -> labs can't read/write.
**Tags:** s3, create, blocker, schema-phase
**Guessed section:** 01-s3-buckets

### B3 — Bucket configuration (versioning, public-block, encryption, lifecycle)
**Verbatim (desired):** "Versioning on. Public access fully blocked. Default encryption
SSE-S3. Lifecycle on scratch: delete objects older than 30 days."
**Why it matters:** required config on each of the 3 buckets; scratch needs the 30d rule.
**Tags:** s3, config, versioning, encryption, lifecycle, public-block
**Guessed section:** 01-s3-buckets

### B4 — Dataset buckets must be POPULATED (data, not just created)
**Verbatim (desired):** prompt-eng = `barclays-products/` brochures+T&Cs+FAQ PDFs +
`sample-conversations/`; genai-devs = `datasets/` (IMDB/summarization/NER/Multi30k subsets)
+ optional `models/` (Flan-T5-base, DistilBERT-base mirrors).
**Why it matters:** an empty bucket passes `terraform plan` but fails the class. Population
is a content step, likely OUT of pure-TF scope (or `aws s3 cp`/objects).
**Tags:** s3, data, content, out-of-tf-maybe
**Guessed section:** 01-s3-buckets

### B5 — 11 of 25 participant Studio user PROFILES are missing
**Verbatim (findings):** "only 14 of participant-01..25 have a SageMaker Studio user
PROFILE ... 11 are MISSING a profile: participant-03, 04, 05, 10, 13, 16, 17, 18, 22, 23,
24." Present: 01,02,06,07,08,09,11,12,14,15,19,20,21,25.
**Why it matters:** those 11 have an IAM user but cannot open a JupyterLab Space.
**Tags:** sagemaker, user-profile, create, blocker, consume-phase (reads existing domain+role)
**Guessed section:** 02-sagemaker-profiles

### B6 — Participant IAM users lack S3 access to the NEW buckets (under-scope)
**Verbatim (findings):** "participant-01 -> s3:GetObject/PutObject/ListBucket on the 3 new
buckets: implicitDeny ... the policy's S3 grant only lists bread-academy-* ARNs."
**Why it matters:** under-scope (the only defect class we care about). Bites only for
DIRECT IAM-user S3 access (VM `aws s3 cp` / boto3 under user creds); in-Space (exec role)
is fine. Fix = additively add the 3 new bucket ARNs. CAUTION: `BreadAcademyStudentPolicy`
is SHARED by 120 members -> add-only, never remove.
**Tags:** iam, s3, under-scope, additive, shared-policy, consume-phase (refs bucket ARNs)
**Guessed section:** 03-iam-s3-access

### B7 — Exec role already reaches the new buckets (NON-action, confirm only)
**Verbatim (findings):** "SageMakerStudentExecutionRole -> s3:GetObject/PutObject on the
new buckets: allowed (has AmazonS3FullAccess)." No change needed.
**Why it matters:** confirms in-Space S3 works the moment buckets exist; documents that we
do NOT touch the exec role. Over-scoped = fine.
**Tags:** iam, exec-role, no-action, confirm
**Guessed section:** 03-iam-s3-access

### B8 — Quotas are healthy (NON-action, confirm only)
**Verbatim (findings):** JupyterLab Apps c5.xlarge=100 / g4dn.xlarge=200; EC2 G/VT=768,
Standard=2500. "Quotas are NOT a blocker."
**Why it matters:** no TF/quota action; record so the plan doesn't invent quota tasks.
**Tags:** quotas, no-action, confirm
**Guessed section:** 00-foundation

### B9 — Region is us-west-2, not the spec's us-east-2 (constraint, not a task)
**Verbatim (findings):** "us-west-2 is canonical ... spec's us-east-2 is wrong/empty."
**Why it matters:** every resource + provider must be us-west-2. A constraint baked into B1.
**Tags:** region, constraint, provider
**Guessed section:** 00-foundation

### B10 — TF must NOT recreate or disturb shared resources
**Verbatim (context):** existing domain, exec role, VPC, `bread-academy-students` group are
shared across many instructors. Reference as data sources; never `terraform destroy`/replace.
**Why it matters:** a careless `resource` (vs `data`) on a shared object, or a policy
overwrite, breaks other instructors. Must be a hard invariant in every card.
**Tags:** safety, shared, data-source, invariant
**Guessed section:** 00-foundation

### B11 — State backend / workspace isolation undecided
**Verbatim (state):** no backend chosen. Account is shared; other instructors operate here.
**Why it matters:** local vs remote state, and how this TF avoids colliding with any other
TF in the account, is an open decision affecting B1.
**Tags:** state, backend, open-question, foundational
**Guessed section:** 00-foundation

---

## Gate 1 triage decisions (RATIFIED 2026-06-19)
1. **B4 (population):** OUT of TF scope. TF creates + configures buckets only; populating
   datasets is a separate content step (`aws s3 cp`/manual). B4 becomes a noted follow-up,
   NOT a card.
2. **B6 (policy):** NEW ADDITIVE policy `datacouch-barclays-s3-usw2` attached to the 25
   participants, granting RO on the 2 dataset buckets + RW on scratch. NEVER edit the shared
   `BreadAcademyStudentPolicy`. (Keeps the fix isolated to our 25 users.)
3. **B11 (backend):** LOCAL state in `new_infra/terraform/` (gitignored).
4. **B5 scope:** TF owns ALL 25 profiles. Import the 14 existing so TF manages the full set;
   create the 11 missing.

### Card-eligible findings (post-triage)
- **B1** -> foundation scaffold (provider, data sources, locals) — Wave 1.
- **B2 + B3** -> the 3 buckets + their config — Wave 1 (the "schema"/producer layer; their
  ARNs are consumed by B6).
- **B5** -> 25 participant profiles (import 14 + create 11) — Wave 1 (consumes existing
  domain+role data sources; independent of buckets).
- **B6** -> additive participant S3 policy — Wave 2 (CONSUMES the bucket ARNs from B2).
- **B7, B8, B9, B10** -> NON-action constraints/confirmations folded into B1's foundation
  card + the INDEX invariants, not standalone cards.
- **B4** -> follow-up note, not a card.
