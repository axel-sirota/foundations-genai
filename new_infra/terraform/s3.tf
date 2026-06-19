# w1-s1-buckets — the 3 us-west-2 buckets + required config (B2 + B3).
# Producer layer: bucket ARNs are consumed by the participant S3 policy (w2-i1).

resource "aws_s3_bucket" "this" {
  for_each = local.all_buckets
  bucket   = each.value
}

# Versioning ON (all 3). [B3]
resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Public access fully blocked (all 3). [B3]
resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Default encryption SSE-S3 / AES256 (all 3). [B3]
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle: scratch ONLY -> expire objects after 30 days. [B3]
resource "aws_s3_bucket_lifecycle_configuration" "scratch" {
  bucket = aws_s3_bucket.this["scratch"].id

  rule {
    id     = "expire-scratch-after-30d"
    status = "Enabled"

    filter {} # whole bucket

    expiration {
      days = 30
    }
  }
}

# NOTE (B4, OUT of TF scope): the dataset buckets are created EMPTY. Populating them
# (brochures, IMDB/NER/Multi30k subsets, model mirrors) is a separate `aws s3 cp` step.
