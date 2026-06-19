# w1-f1-scaffold — Terraform + provider version pins.
terraform {
  required_version = ">= 1.5.0" # import {} blocks need >= 1.5

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
