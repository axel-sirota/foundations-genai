# CODEX o3 whole-plan review — barclays-tf-converge

Verdict: READY TO EXECUTE after 2 pre-apply checks (both folded into card DoD).

Run: codex exec -m o3 reasoning_effort=high, 2026-06-19.

---

READY TO EXECUTE (after fixing 2 blockers)

Fix & re-check  
1. Prepare/verify the 14 `terraform import` commands (exact `for_each` keys + `user_settings`). Any drift will force a spurious profile replacement.  
2. Run an out-of-band `aws s3api head-bucket` for the three planned names; abort if any already exist (S3 namespace is global).

Everything else passes:

- No card declares or destroys the shared domain, execution role, VPC or bread-academy group (all are `data` blocks).
- Wave split is correct: buckets (W1-s1) are produced before the policy (W2-i1); all other edges are compile-time data reads—no intra-wave runtime traps.
- Import-vs-create logic for profiles is sound once the imports above are done; duplicates cannot occur.
- Buckets use unique `datacouch-barclays-*-usw2` names, versioning, SSE-S3, public-block, and 30-day scratch lifecycle—collision risk handled by check #2.
- Plan fully covers desired additions (3 buckets, 25 profiles, 1 policy+attachments); no finding is dropped.
- IAM policy is additive and at least minimally scoped (RO on datasets, RW on scratch). Over-scope permitted; no under-scope detected.

Proceed after the two fixes.
