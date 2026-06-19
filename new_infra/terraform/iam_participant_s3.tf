# w2-i1 — participant policy, 100% aligned with the spec doc's "barclays-students" policy
# (Section 5), plus the one addition /research proved the notebooks need (the SageMaker
# default bucket, which the spec omits but every notebook chains through).
#
# Structural divergences from the spec are RATIFIED and intentional (live reality the April
# spec predates): region us-west-2, bucket names datacouch-*-usw2, participant-NN identities,
# and attaching to the existing Barclays cohort GROUPS instead of a "barclays-students" group
# that does not exist. The POLICY CONTENT below mirrors the spec Sid-for-Sid.
#
# Over-scope is fine; this restores every spec Sid (Bedrock, IAM self-inspection, service
# quotas, ecr-public, logs write) even where the current required notebooks don't exercise
# them -- the spec includes them as headroom + for the lab-0 sanity checks.
#
# HARD INVARIANT (B6 + B10): brand-new standalone policy attached to the Barclays GROUPS.
# The shared BreadAcademyStudentPolicy / bread-academy-students group is NEVER touched.

data "aws_iam_policy_document" "participant" {

  # Spec Sid: SageMakerStudioAccess -- open Studio + manage apps/spaces.
  statement {
    sid    = "SageMakerStudioAccess"
    effect = "Allow"
    actions = [
      "sagemaker:CreatePresignedDomainUrl",
      "sagemaker:DescribeDomain",
      "sagemaker:DescribeUserProfile",
      "sagemaker:ListApps",
      "sagemaker:CreateApp",
      "sagemaker:DeleteApp",
      "sagemaker:DescribeApp",
      "sagemaker:CreateSpace",
      "sagemaker:UpdateSpace",
      "sagemaker:DeleteSpace",
      "sagemaker:DescribeSpace",
      "sagemaker:ListSpaces",
    ]
    resources = ["*"]
  }

  # Spec Sid: SageMakerTrainingAndInferenceJustInCase -- fit()/deploy() + endpoints.
  statement {
    sid    = "SageMakerTrainingAndInferenceJustInCase"
    effect = "Allow"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:ListTrainingJobs",
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:CreateModel",
      "sagemaker:CreateEndpoint",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:InvokeEndpoint",
      "sagemaker:DeleteEndpoint",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:DeleteModel",
      "sagemaker:DescribeEndpoint",
      "sagemaker:ListEndpoints",
    ]
    resources = ["*"]
  }

  # Spec Sid: PassRoleToSageMaker.
  statement {
    sid       = "PassRoleToSageMaker"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [data.aws_iam_role.student_exec.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # Spec Sid: S3DatasetsRead (the 2 dataset buckets, RO).
  statement {
    sid     = "S3DatasetsRead"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = flatten([
      for k in ["prompt_eng", "genai_devs"] : [
        aws_s3_bucket.this[k].arn,
        "${aws_s3_bucket.this[k].arn}/*",
      ]
    ])
  }

  # Spec Sid: S3ScratchBucketReadWrite (scratch, RW).
  statement {
    sid     = "S3ScratchBucketReadWrite"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.this["scratch"].arn,
      "${aws_s3_bucket.this["scratch"].arn}/*",
    ]
  }

  # ADDITION (not in spec, found by /research): the SageMaker DEFAULT bucket
  # sagemaker-<region>-<account>. Every notebook chains artifacts through
  # sagemaker.Session().default_bucket(); estimator.fit()/.deploy() read/write it.
  statement {
    sid    = "SageMakerDefaultBucketReadWrite"
    effect = "Allow"
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
      "s3:ListBucket", "s3:GetBucketLocation",
    ]
    resources = [
      "arn:aws:s3:::${local.sagemaker_default_bucket}",
      "arn:aws:s3:::${local.sagemaker_default_bucket}/*",
    ]
  }

  # Spec Sid: ECRReadForDeepLearningContainers (incl. ecr-public).
  statement {
    sid    = "ECRReadForDeepLearningContainers"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr-public:GetAuthorizationToken",
      "ecr-public:BatchCheckLayerAvailability",
      "ecr-public:GetDownloadUrlForLayer",
      "ecr-public:BatchGetImage",
    ]
    resources = ["*"]
  }

  # Spec Sid: CloudWatchLogsForTrainingJobs (write + read).
  statement {
    sid    = "CloudWatchLogsForTrainingJobs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
      "logs:GetLogEvents",
    ]
    resources = ["arn:aws:logs:*:*:log-group:/aws/sagemaker/*"]
  }

  # Spec Sid: BedrockInvokeOptional.
  statement {
    sid    = "BedrockInvokeOptional"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:ListFoundationModels",
    ]
    resources = ["*"]
  }

  # Spec Sid: IAMSelfInspection (scoped to the calling user).
  statement {
    sid    = "IAMSelfInspection"
    effect = "Allow"
    actions = [
      "iam:GetUser",
      "iam:ListAttachedUserPolicies",
      "iam:ListGroupsForUser",
    ]
    resources = ["arn:aws:iam::*:user/$${aws:username}"]
  }

  # Spec Sid: ServiceQuotasRead.
  statement {
    sid    = "ServiceQuotasRead"
    effect = "Allow"
    actions = [
      "servicequotas:ListServiceQuotas",
      "servicequotas:GetServiceQuota",
    ]
    resources = ["*"]
  }

  # sts:GetCallerIdentity (lab-0 sanity check).
  statement {
    sid       = "StsIdentity"
    effect    = "Allow"
    actions   = ["sts:GetCallerIdentity"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "participant" {
  name        = "datacouch-barclays-students-usw2"
  description = "Barclays participant policy (spec Section 5 barclays-students, verbatim) + the SageMaker default bucket the notebooks chain through. Additive; does not touch shared policies."
  policy      = data.aws_iam_policy_document.participant.json
}

# Attach to both Barclays cohort GROUPS (survives participant-user churn). Additive only.
resource "aws_iam_group_policy_attachment" "participant" {
  for_each   = toset(local.barclays_groups)
  group      = each.value
  policy_arn = aws_iam_policy.participant.arn
}
