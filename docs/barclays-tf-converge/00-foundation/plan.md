# 00-foundation — plan

Covers B1 (scaffold) + folds in B7/B8/B9/B10 (non-action constraints/invariants).

## Goal
Stand up a Terraform project under `new_infra/terraform/` that can `init/validate/plan`
against us-west-2 / datacouch, references the SHARED resources as data sources (never
recreates them), and exposes the IDs/ARNs other waves consume.

## Tasks

### F1 — TF project skeleton + provider + locals  (Wave 1, producer)
- `new_infra/terraform/versions.tf` — required_version, aws provider (~> 5.x) pin.
- `new_infra/terraform/providers.tf` — provider "aws" { region = "us-west-2", profile =
  "datacouch" }. (B9 region constraint; B10 — never another region.)
- `new_infra/terraform/locals.tf` — account_id 962804699607, domain_id d-bquhieanzgod,
  exec_role name, the 25 participant numbers, the 3 bucket names, the 11-missing list.
- `new_infra/terraform/backend.tf` — local backend (B11). State path gitignored.
- `new_infra/terraform/.gitignore` — `*.tfstate*`, `.terraform/`, `terraform_plans/`.
- DoD: `terraform init` + `terraform validate` exit 0 (no resources yet).

### F2 — Data sources for SHARED resources (Wave 1, producer)
- `new_infra/terraform/data.tf`:
  - `data.aws_caller_identity` / `data.aws_region` (assert account + region).
  - the existing SageMaker domain (referenced by id d-bquhieanzgod) — for exec-role +
    network defaults the new profiles inherit.
  - `data.aws_iam_role.exec` = SageMakerStudentExecutionRole (ARN consumed by profiles).
  - the Barclays VPC + subnet (read-only, documented; domain already owns them).
  - the 25 existing/target participant IAM users (`data.aws_iam_user` per N) — needed by
    the Wave-2 policy attachment.
- B10 INVARIANT: every shared object is a `data` block, NEVER a `resource`. No `terraform
  destroy` of anything here.
- DoD: `terraform plan` shows these as data reads, 0 resource changes.

## Non-action constraints folded in (no task)
- **B7** exec role already has S3FullAccess -> in-Space S3 works once buckets exist. We do
  NOT modify the exec role.
- **B8** quotas healthy -> no quota resources.
- **B9** us-west-2 only -> provider + every resource.
- **B10** shared resources are data-only, never recreated/destroyed.
