# 02-sagemaker-profiles — plan

Covers B5: TF owns all 25 participant Studio user profiles (import 14 existing, create 11).

## Goal
Every `participant-01..25` has a SageMaker Studio user profile in domain d-bquhieanzgod
using the existing exec role, so all 25 can launch a JupyterLab Space.

## State today (from snapshot)
- PRESENT (14): 01,02,06,07,08,09,11,12,14,15,19,20,21,25
- MISSING (11): 03,04,05,10,13,16,17,18,22,23,24

## Tasks

### P1 — 25 participant user profiles (for_each), import the 14  (Wave 1, consumer)
- `new_infra/terraform/sagemaker_profiles.tf`:
  - `aws_sagemaker_user_profile` with `for_each = toset(local.participants)` (01..25).
  - `domain_id = local.domain_id` (d-bquhieanzgod — from the data source / locals).
  - `user_settings { execution_role = data.aws_iam_role.exec.arn }` (the existing shared
    role; B10 — referenced, not created).
- IMPORT the 14 existing profiles so TF adopts them instead of trying to recreate (a
  recreate would error / churn). Import ids: `<domain-id>/<profile-name>`.
  - Provide an `import {}` block per existing profile (TF 1.5+) OR a documented
    `terraform import` list for the 14.
- DoD: `terraform plan` shows 11 profiles to ADD and 0 to destroy/replace (the 14 imported
  ones show no-change). Post-apply: all 25 `list-user-profiles` InService.

## Notes / risks
- This CONSUMES the existing domain + exec role (data sources from 00-foundation/F2). It
  does NOT depend on the buckets — fully parallel with section 01.
- Importing must use the EXACT current settings of the 14 so the plan is clean (no diff).
  If the 14 were created with different user_settings than our resource declares, the plan
  will show in-place updates — must reconcile the resource to match reality (over-scope is
  fine, but avoid changing the shared exec role wiring).
