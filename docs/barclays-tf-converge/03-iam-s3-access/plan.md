# 03-iam-s3-access — plan

Covers B6: a NEW additive policy giving the 25 participants S3 access to the new buckets.
B7 (exec role already OK) is a confirm-only non-action.

## Goal
Close the ONLY under-scope: participant IAM users get `implicitDeny` on the new buckets.
Add a NEW small policy (never edit the shared `BreadAcademyStudentPolicy`).

## Tasks

### I1 — datacouch-barclays-s3-usw2 policy + attach to 25 participants  (Wave 2, CONSUMER)
- `new_infra/terraform/iam_participant_s3.tf`:
  - `aws_iam_policy.barclays_s3_usw2` with:
    - RO (`s3:GetObject`, `s3:ListBucket`) on `datacouch-barclays-prompt-eng-usw2` +
      `datacouch-barclays-genai-devs-usw2` (+ `/*`).
    - RW (`s3:GetObject/PutObject/DeleteObject/ListBucket`) on
      `datacouch-barclays-scratch-usw2` (+ `/*`).
    - Resource ARNs come from the Wave-1 bucket resources (`aws_s3_bucket.this[...].arn`) —
      this is the CONSUME edge: I1 references B2's created objects.
  - `aws_iam_user_policy_attachment` with `for_each` over the 25 participant users
    (`data.aws_iam_user` from 00-foundation/F2).
- B6 CAUTION honored: this is a brand-new standalone policy. The shared
  `BreadAcademyStudentPolicy` / `bread-academy-students` group is NOT touched.
- B10: additive only; no detach/remove of any existing attachment.
- DoD: `terraform plan` adds 1 policy + 25 attachments, 0 destroys. Post-apply:
  `aws iam simulate-principal-policy` for participant-01 on the 3 new buckets returns
  `allowed` (was implicitDeny).

## Non-action (no task)
- **B7** — exec role (the Space identity) already has AmazonS3FullAccess and simulates
  `allowed` on the new buckets. We do NOT modify the exec role. Confirm-only.

## Wave placement rationale (the consume edge)
I1 references the bucket ARNs created in Wave 1 (S1). In Terraform a single `apply` resolves
this via the dependency graph, BUT the adapted wave rule treats "policy that references a
not-yet-created bucket" as a CONSUME-of-producer-output -> I1 is Wave 2 so the buckets are
real/applied first. (If we ever split applies per wave, Wave 1 must apply before Wave 2.)
