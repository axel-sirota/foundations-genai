# 01-s3-buckets — plan

Covers B2 (create the 3 buckets) + B3 (their config). B4 (population) is OUT of TF scope.

## Goal
Create the 3 us-west-2 buckets with the required config. These are the PRODUCER layer:
their ARNs are consumed by the Wave-2 participant S3 policy (B6).

## Buckets
1. `datacouch-barclays-prompt-eng-usw2` — read-only datasets
2. `datacouch-barclays-genai-devs-usw2` — read-only datasets
3. `datacouch-barclays-scratch-usw2`    — read/write scratch

## Tasks

### S1 — Create the 3 buckets + config  (Wave 1, producer/"schema")
- `new_infra/terraform/s3.tf`:
  - `aws_s3_bucket` x3 (use `for_each` over a locals map).
  - `aws_s3_bucket_versioning` = Enabled (all 3). [B3]
  - `aws_s3_bucket_public_access_block` = all four flags true (all 3). [B3]
  - `aws_s3_bucket_server_side_encryption_configuration` = SSE-S3 (AES256) (all 3). [B3]
  - `aws_s3_bucket_lifecycle_configuration` on scratch ONLY = expire objects after 30 days.
    [B3]
- B10: bucket NAMES must be the new collision-safe ones (the bare `barclays-*` names exist
  globally in ap-south-1 and are NOT ours — a create on those names will 409).
- DoD: `terraform plan` shows exactly 3 buckets + their config sub-resources to ADD, no
  destroys; post-apply read-back `aws s3api get-bucket-versioning/-encryption` matches.

## Out of scope (B4 follow-up, NOT a card)
Populating the dataset buckets (brochures, IMDB/NER/Multi30k subsets, model mirrors) is a
separate content step done with `aws s3 cp` / manual upload after the buckets exist.
