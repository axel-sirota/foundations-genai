# w1-f1-scaffold — local state backend (B11). State file is gitignored.
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}
