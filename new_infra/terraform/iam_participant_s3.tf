# w2-i1-s3policy — NEW additive policy giving the 25 participants S3 access to the new
# us-west-2 buckets (B6). Closes the only under-scope: participant IAM users get implicitDeny
# on these buckets today.
#
# HARD INVARIANT (B6 + B10): this is a BRAND-NEW standalone policy. The shared
# BreadAcademyStudentPolicy / bread-academy-students group is NEVER touched. Additive only.
#
# CONSUME edge: the Resource ARNs come from aws_s3_bucket.this[*].arn (created in w1-s1), so
# the dependency is real and Terraform orders the buckets before this policy.

data "aws_iam_policy_document" "participant_s3" {
  # Read-only on the two dataset buckets.
  statement {
    sid     = "DatasetsRead"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = flatten([
      for k in ["prompt_eng", "genai_devs"] : [
        aws_s3_bucket.this[k].arn,
        "${aws_s3_bucket.this[k].arn}/*",
      ]
    ])
  }

  # Read/write on the scratch bucket.
  statement {
    sid    = "ScratchReadWrite"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.this["scratch"].arn,
      "${aws_s3_bucket.this["scratch"].arn}/*",
    ]
  }
}

resource "aws_iam_policy" "barclays_s3_usw2" {
  name        = "datacouch-barclays-s3-usw2"
  description = "Barclays participants: RO on usw2 dataset buckets, RW on usw2 scratch. Additive; does not touch BreadAcademyStudentPolicy."
  policy      = data.aws_iam_policy_document.participant_s3.json
}

# Attach to the two Barclays cohort GROUPS (not individual users) so the grant survives the
# participant IAM users being recreated. Additive: aws_iam_group_policy_attachment manages
# ONLY this one (group, policy) pair -- it never touches the group's other policies or its
# membership. Participants inherit the S3 access by being members of these groups. (B6 + B10)
resource "aws_iam_group_policy_attachment" "barclays_s3" {
  for_each   = toset(local.barclays_groups)
  group      = each.value
  policy_arn = aws_iam_policy.barclays_s3_usw2.arn
}
