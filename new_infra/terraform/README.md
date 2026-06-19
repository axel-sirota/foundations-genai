# Barclays env Terraform (new_infra/terraform)

Converges the live us-west-2 Barclays training env to `../docs/desired_state.md`.
Account **962804699607 (datacouch)**, region **us-west-2** ONLY, profile **datacouch**.
Built from the waved plan in `../../docs/barclays-tf-converge/`.

## What it manages (plan: 14 import, 27 add, 14 change, 0 destroy)
- **3 S3 buckets** (`datacouch-barclays-{prompt-eng,genai-devs,scratch}-usw2`) +
  versioning + public-access-block + SSE-S3 + 30-day lifecycle on scratch.
- **All 25** participant Studio user profiles on `SageMakerStudentExecutionRole`:
  the 14 existing are IMPORTED (by full ARN) and converged to the student role (the 14
  in-place changes); the 11 missing are created.
- **1 new additive IAM policy** `datacouch-barclays-s3-usw2` attached to BOTH Barclays
  groups (`Barclays-batch-1`, `Barclays_Batch-2`): RO on the 2 dataset buckets, RW on
  scratch. Closes the participant under-scope; attaching to the GROUP survives user churn.

## What it does NOT touch (shared / out of scope)
- SageMaker domain `d-bquhieanzgod`, the exec role, the VPC: data sources, never changed.
- The Barclays groups' OTHER policies + their membership: we add ONE policy, nothing else.
- Shared `bread-academy-students` group / `BreadAcademyStudentPolicy`: not referenced.
- **Participant IAM users**: NOT managed by TF. They are provisioned/deleted out-of-band by
  whoever owns the cohort (they churn in this shared account — 60 existed earlier in the
  build session, 0 hours later). The S3 policy attaches to GROUPS, so it works no matter
  when the users are (re)created and added to a group.

## Profile import note (not a provider bug)
`aws_sagemaker_user_profile` imports by FULL ARN
(`arn:aws:sagemaker:us-west-2:<acct>:user-profile/<domain>/<name>`), NOT `domain/name`. The
short form makes the provider's arn.Parse fail with "arn: invalid prefix". `sagemaker_profiles.tf`
uses the ARN form (see its `import {}` block).

## Apply sequence (AXEL runs apply — never automated)
```bash
export AWS_PROFILE=datacouch
terraform -chdir=new_infra/terraform init

# Pre-apply check: bucket names are globally free (S3 namespace is global).
aws s3api head-bucket --bucket datacouch-barclays-prompt-eng-usw2   # expect 404 = free
aws s3api head-bucket --bucket datacouch-barclays-genai-devs-usw2   # expect 404 = free
aws s3api head-bucket --bucket datacouch-barclays-scratch-usw2      # expect 404 = free

# Plan (saved to terraform_plans/, gitignored):
terraform -chdir=new_infra/terraform plan -out=terraform_plans/$(date +%Y%m%d-%H%M%S).tfplan
# review: expect "14 to import, 27 to add, 14 to change, 0 to destroy"
terraform -chdir=new_infra/terraform apply terraform_plans/<that-file>.tfplan   # <-- AXEL

# Post-apply verification:
aws sagemaker list-user-profiles --region us-west-2 --domain-id-equals d-bquhieanzgod
  # all 25 InService, all on SageMakerStudentExecutionRole
aws iam list-attached-group-policies --group-name Barclays-batch-1
  # datacouch-barclays-s3-usw2 present
```

## Out of scope (separate steps)
- **B4 dataset population** — buckets created EMPTY; upload brochures/datasets via `aws s3 cp`.
- **Participant IAM users** — provisioned by whoever owns the cohort; add them to a Barclays
  group to inherit the S3 policy.
- State is **local + gitignored** (`.gitignore` here). Single-operator.

> NOTE on the participant under-scope verification: you can't `simulate-principal-policy` a
> participant user until those users exist again. Once a participant exists and is in a
> Barclays group, simulate `s3:PutObject` on `datacouch-barclays-scratch-usw2/...` -> allowed.
