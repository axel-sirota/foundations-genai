# w1-f1-scaffold — AWS provider pinned to us-west-2 / datacouch profile (B9).
# HARD INVARIANT: us-west-2 ONLY. The account (962804699607) is shared across many
# instructors; nothing here may touch another region.
provider "aws" {
  region  = local.region
  profile = "datacouch"

  # Belt-and-suspenders: refuse to run against the wrong account.
  allowed_account_ids = [local.account_id]

  default_tags {
    tags = {
      Project   = "barclays-training"
      ManagedBy = "terraform"
      Epic      = "barclays-tf-converge"
    }
  }
}
