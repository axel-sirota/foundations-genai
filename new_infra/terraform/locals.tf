# w1-f1-scaffold — central config for the whole epic.
locals {
  account_id = "962804699607" # datacouch
  region     = "us-west-2"
  domain_id  = "d-bquhieanzgod" # barclays-training-v2 (existing, shared — data source only)

  # Exec role (EXISTING shared role — referenced, never created). All 25 profiles converge
  # onto this named student role (per Axel: "I want all 25 with the role I gave").
  student_exec_role = "SageMakerStudentExecutionRole"

  # Participant identity: 01..25 is the canonical set.
  participant_nums = [for i in range(1, 26) : format("%02d", i)] # "01".."25"

  # 14 profiles that ALREADY exist (import, no-op) vs 11 missing (create).
  existing_profile_nums = ["01", "02", "06", "07", "08", "09", "11", "12", "14", "15", "19", "20", "21", "25"]
  missing_profile_nums  = ["03", "04", "05", "10", "13", "16", "17", "18", "22", "23", "24"]

  existing_profiles = [for n in local.existing_profile_nums : "participant-${n}"]
  missing_profiles  = [for n in local.missing_profile_nums : "participant-${n}"]
  all_participants  = [for n in local.participant_nums : "participant-${n}"]

  # The Barclays cohort IAM groups the additive participant policy attaches to. Attaching to
  # the GROUP (not users) survives user churn. Participants must be MEMBERS of one of these to
  # inherit the policy. (A live impersonation test of participant-01 caught that the 3 real
  # participants were parked in the dead bread-academy-students group instead of a Barclays
  # group -> they get added to Barclays-batch-1 out-of-band; bread is dead so that group is
  # not used for anything active.)
  barclays_groups = ["Barclays-batch-1", "Barclays_Batch-2"]

  # The 3 us-west-2 buckets (names are collision-safe; the bare barclays-* names are taken
  # globally in ap-south-1 and are NOT ours).
  bucket_prompt_eng = "datacouch-barclays-prompt-eng-usw2"
  bucket_genai_devs = "datacouch-barclays-genai-devs-usw2"
  bucket_scratch    = "datacouch-barclays-scratch-usw2"

  dataset_buckets = {
    prompt_eng = local.bucket_prompt_eng
    genai_devs = local.bucket_genai_devs
  }
  all_buckets = {
    prompt_eng = local.bucket_prompt_eng
    genai_devs = local.bucket_genai_devs
    scratch    = local.bucket_scratch
  }

  # The SageMaker DEFAULT bucket sagemaker-<region>-<account>. This is the bucket the course
  # notebooks ACTUALLY chain artifacts through (sagemaker.Session().default_bucket()), and the
  # bucket estimator.fit()/.deploy() read/write. It already exists. (Found by /research over
  # the notebooks: NONE reference the barclays-* buckets; ALL use this one.)
  sagemaker_default_bucket = "sagemaker-${local.region}-${local.account_id}"
}
