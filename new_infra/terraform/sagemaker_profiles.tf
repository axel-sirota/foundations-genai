# w1-p1-profiles — all 25 participant SageMaker Studio user profiles (B5).
#
# UNIFORM fleet: all 25 on the named student exec role SageMakerStudentExecutionRole
# (per Axel: "I want all 25 with the role I gave"). The 14 that already exist are IMPORTED;
# importing converges their execution_role to the student role (in-place, 1 change each).
# The 11 missing are created.
#
# IMPORT-ID FORMAT: aws_sagemaker_user_profile imports by FULL ARN, not "domain-id/name".
#   arn:aws:sagemaker:<region>:<account>:user-profile/<domain-id>/<profile-name>
# (Using the short "domain-id/name" form makes the provider's arn.Parse fail with
#  "arn: invalid prefix" — that was the earlier error, a wrong-id mistake, not a provider bug.)
#
# HARD INVARIANT (B10): the domain + the exec role are EXISTING shared resources, referenced
# only via data sources. This file creates/updates user profiles; never the domain/role.

resource "aws_sagemaker_user_profile" "participant" {
  for_each          = toset(local.all_participants) # participant-01 .. participant-25
  domain_id         = local.domain_id
  user_profile_name = each.value

  user_settings {
    execution_role = data.aws_iam_role.student_exec.arn
  }
}

# Import the 14 already-existing profiles by full ARN so TF adopts + converges them.
import {
  for_each = toset(local.existing_profiles)
  to       = aws_sagemaker_user_profile.participant[each.value]
  id       = "arn:aws:sagemaker:${local.region}:${local.account_id}:user-profile/${local.domain_id}/${each.value}"
}
