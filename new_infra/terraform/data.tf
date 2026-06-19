# w1-f2-datasources — SHARED resources, READ-ONLY.
# HARD INVARIANT (B10): every block here is a `data` source. NONE of these is ever a
# `resource`, NONE is created, modified, or destroyed by this Terraform. The domain, the
# exec roles, the VPC, and the participant IAM users are shared across many instructors.

# Assert we are in the right account + region before anything plans.
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# The named student exec role -> assigned to the 11 NEW profiles (w1-p1).
data "aws_iam_role" "student_exec" {
  name = local.student_exec_role
}

# Participant IAM users are NOT managed or referenced here. They are provisioned + deleted
# out-of-band by whoever owns the cohort (they churn in this shared account — they existed
# earlier in the session and were gone hours later). TF must not depend on them. The new S3
# policy attaches to the BARCLAYS GROUPS instead (w2-i1), which survive user churn.
# (aws_iam_group_policy_attachment takes the group NAME directly, so no data source needed.)

# ---- Guards: fail the plan if we are pointed at the wrong account/region. ----
check "account_and_region" {
  assert {
    condition     = data.aws_caller_identity.current.account_id == local.account_id
    error_message = "Wrong AWS account: expected ${local.account_id} (datacouch), got ${data.aws_caller_identity.current.account_id}."
  }
  assert {
    condition     = data.aws_region.current.name == local.region
    error_message = "Wrong region: expected ${local.region}, got ${data.aws_region.current.name}."
  }
}
