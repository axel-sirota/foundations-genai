# INDEX — barclays-tf-converge (waved Terraform plan)

Converge the live us-west-2 Barclays env to `new_infra/docs/desired_state.md`. Produced by
/workflow-plan (Terraform-adapted). Inputs: desired_state.md + aws_state_findings.md +
new_infra/snapshot/. Account **962804699607 (datacouch)**, region **us-west-2**, domain
**barclays-training-v2 / d-bquhieanzgod**.

**Hard invariants (every card):**
- us-west-2 ONLY; shared account, many instructors.
- Shared resources (domain, exec role, VPC, bread-academy-students group) are TF **data
  sources**, NEVER `resource`, NEVER destroyed. (B10)
- Over-scope is fine; only UNDER-scope is a defect. (no role-trimming tasks)
- User runs `terraform apply` himself.

---

## Waves

### Wave 1 — Foundation + Producers (the things others reference)
Build the project, read the shared resources, create the buckets and the profiles. The only
cross-edges are producer outputs that Terraform interpolation + `depends_on` order correctly.

| Card | Title | phase | depends_on | section |
|---|---|---|---|---|
| **w1-f1-scaffold** | TF root: provider us-west-2/datacouch, locals, local backend | schema | - | 00-foundation |
| **w1-f2-datasources** | Data sources: domain, exec role, VPC, 25 participant users | schema | f1 | 00-foundation |
| **w1-s1-buckets** | 3 buckets + versioning/public-block/SSE-S3/scratch-30d | schema | f1 | 01-s3-buckets |
| **w1-p1-profiles** | 25 Studio profiles (import 14, create 11) | consume | f2 | 02-sagemaker-profiles |

Within Wave 1 the producer->consumer edges (f1->all, f2->p1) are **config/data dependencies**,
not runtime - safe under schema->barrier->consume phasing. p1 reads f2's data outputs (role
ARN, domain id) which are STATIC reads available at plan time, not unmerged runtime output.
No intra-wave runtime trap.

### Wave 2 — Consumer (references Wave-1 created objects)
| Card | Title | phase | depends_on | section |
|---|---|---|---|---|
| **w2-i1-s3policy** | New additive policy datacouch-barclays-s3-usw2 + attach 25 | consume | s1, f2 | 03-iam-s3-access |

w2-i1 is Wave 2 because it **consumes the bucket ARNs CREATED in w1-s1** (the Rule-5 class).
Putting it in Wave 1 would be the runtime-dependency trap. Wave 2 guarantees buckets exist first.

---

## RUNTIME-DEPENDENCY SCAN (the Wave-2 trap check)

| Edge | Kind | Verdict |
|---|---|---|
| f1 -> f2, s1 | config (project root) | same-wave OK (root must exist; static) |
| f2 -> p1 | data-read (role ARN, domain id) | same-wave OK (static reads, not runtime output) |
| s1 -> i1 | **created-object reference** (bucket ARN) | **SPLIT across waves** (i1 in W2) OK |
| f2 -> i1 | data-read (user ARNs) | satisfied (f2 in W1, i1 in W2) OK |

No intra-wave consumer references another card's newly-CREATED runtime object. The one such
edge (s1->i1) is split W1->W2. **No trap.**

---

## Conflict report

Full: `ddl_conflict_report.md`. Summary: **No HARD CONFLICTS, no folds.** Each AWS resource
has exactly one creating card. Cross-card edges are clean producer->consumer ARN references
(s1->i1, f2->{p1,i1}), handled by Terraform interpolation + the wave boundary.

### Risks to ratify at Gate 2
1. **w1-p1 import:** the 14 existing profiles must import with attributes matching reality
   (else spurious in-place updates). The build step must capture their real `user_settings`.
2. **w1-s1 names:** must use `datacouch-*-usw2` (bare names collide globally in ap-south-1).
3. **Shared-resource safety:** every shared object is `data`, never `resource`.
4. **B4 (population) is OUT of scope** - buckets created empty; datasets uploaded separately.

---

## Apply order (for the operator)
```
# Wave 1
terraform -chdir=new_infra/terraform init
terraform -chdir=new_infra/terraform plan -out=terraform_plans/<ts>-wave1.tfplan
# (review, then) terraform apply <plan>   <-- AXEL runs this

# Wave 2 (after W1 applied -> buckets exist)
terraform -chdir=new_infra/terraform plan -out=terraform_plans/<ts>-wave2.tfplan
# (review, then) terraform apply <plan>   <-- AXEL runs this
```
A single full `apply` also works (graph orders s1 before i1); the wave split is the safe,
reviewable path and the one the runner uses.

<!-- BEGIN ddl_conflict_report (auto-embedded by ddl_plan.py) -->
<!-- superseded by the TF-adapted ddl_conflict_report.md; see that file -->
<!-- END ddl_conflict_report -->

---

## GATE 2 — RATIFIED (2026-06-19)
Axel approved: wave grouping, the 5 cards, import-14/create-11 profiles, additive S3 policy.
Verification: conflict report clean, loader exit 0 (both waves), codex o3 READY (2 pre-apply
checks folded into w1-s1 + w1-p1 DoD). Hand-off target: /workflow-execute (full runner).
