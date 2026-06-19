# w2-i1 (revised per /research) — the participant gap policy.
#
# The Barclays groups already carry AmazonSageMakerFullAccess, which covers sagemaker:*
# (CreateTrainingJob/CreateModel/CreateEndpoint/InvokeEndpoint/CreatePresignedDomainUrl/
# CreateApp/CreateSpace/Describe*/List* — everything fit()/deploy()/Studio-launch need on the
# sagemaker namespace). It does NOT cover four things the labs require. This ADDITIVE policy
# fills exactly those gaps + the S3 the course actually uses. Over-scope is fine; this targets
# only the real under-scope.
#
# Evidence (/research over every executed code cell in the Solution notebooks):
#   - topics 4,5,6,7 + optional run estimator.fit() = real CreateTrainingJob (needs PassRole,
#     ECR pull, the exec role does the S3/logs/metrics).
#   - topics 5,7 run .deploy() = CreateModel + CreateEndpoint (needs PassRole).
#   - EVERY topic chains artifacts through sagemaker.Session().default_bucket() =
#     sagemaker-us-west-2-<acct> (NOT the barclays-* buckets).
#   - notebooks read CloudWatch /aws/sagemaker/TrainingJobs logs + call sts:GetCallerIdentity.
#
# HARD INVARIANT (B6 + B10): brand-new standalone policy attached to the Barclays GROUPS.
# The shared BreadAcademyStudentPolicy / bread-academy-students group is NEVER touched.

data "aws_iam_policy_document" "participant_gaps" {

  # --- Gap 1: PassRole to the SageMaker exec role (fit/deploy require it; FullAccess omits it).
  statement {
    sid       = "PassExecRoleToSageMaker"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [data.aws_iam_role.student_exec.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # --- Gap 2: S3 on the SageMaker DEFAULT bucket (the real artifact chain).
  statement {
    sid    = "SageMakerDefaultBucketRW"
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

  # --- Gap 2b: S3 on the barclays-* dataset buckets (RO) + scratch (RW). Datasets the
  #     instructor populates live here; kept per decision. Notebooks may read them too.
  statement {
    sid     = "BarclaysDatasetsRead"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
    resources = flatten([
      for k in ["prompt_eng", "genai_devs"] : [
        aws_s3_bucket.this[k].arn,
        "${aws_s3_bucket.this[k].arn}/*",
      ]
    ])
  }
  statement {
    sid     = "BarclaysScratchReadWrite"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.this["scratch"].arn,
      "${aws_s3_bucket.this["scratch"].arn}/*",
    ]
  }

  # --- Gap 3: ECR pull (the HuggingFace/PyTorch DLC images training jobs use).
  statement {
    sid    = "ECRPullDeepLearningContainers"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"] # GetAuthorizationToken requires * ; layer reads are cross-account DLC repos
  }

  # --- Gap 4: CloudWatch Logs read for training-job logs (the notebooks fetch these).
  # Needs the log-GROUP ARN (DescribeLogStreams/FilterLogEvents) AND the log-STREAM ARN
  # (GetLogEvents reads individual streams). DescribeLogGroups is account-level (resource *).
  statement {
    sid    = "TrainingJobLogGroups"
    effect = "Allow"
    actions = [
      "logs:DescribeLogStreams",
      "logs:FilterLogEvents",
      "logs:GetLogEvents",
    ]
    resources = [
      "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*",
      "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*:log-stream:*",
    ]
  }
  statement {
    sid       = "TrainingJobDescribeLogGroups"
    effect    = "Allow"
    actions   = ["logs:DescribeLogGroups"]
    resources = ["*"] # DescribeLogGroups does not support resource-level scoping
  }

  # --- sts:GetCallerIdentity (lab-0 sanity check). Allowed to all by default, but explicit.
  statement {
    sid       = "StsIdentity"
    effect    = "Allow"
    actions   = ["sts:GetCallerIdentity"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "participant_gaps" {
  name        = "datacouch-barclays-participant-gaps-usw2"
  description = "Barclays participants: the gaps AmazonSageMakerFullAccess misses (PassRole, S3 on SageMaker default + barclays buckets, ECR pull, training-job log read). Additive; does not touch shared policies."
  policy      = data.aws_iam_policy_document.participant_gaps.json
}

# Attach to both Barclays cohort GROUPS (survives participant-user churn). Additive only.
resource "aws_iam_group_policy_attachment" "participant_gaps" {
  for_each   = toset(local.barclays_groups)
  group      = each.value
  policy_arn = aws_iam_policy.participant_gaps.arn
}
