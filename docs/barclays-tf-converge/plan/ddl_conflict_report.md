# Conflict report — Terraform-adapted (workflow-plan Step 6)

> NOTE: `ddl_plan.py` keys on SQL `ddl_intents`; this epic uses `tf_intents` (AWS resources),
> so the automated pass found 0 schema cards (vacuous "no conflict"). This report applies the
> SAME 5-rule framework, adapted to Terraform/AWS semantics, BY HAND. "object" = an AWS
> resource; "schema phase" = a resource that CREATES an object others reference; "consume" =
> a resource/policy that REFERENCES an already-created object.

- cards analysed: 5 (w1-f1, w1-f2, w1-s1, w1-p1, w2-i1)
- producer ("schema") cards: w1-f1 (project root), w1-f2 (data outputs), w1-s1 (buckets)
- consumer cards: w1-p1 (profiles consume domain+role), w2-i1 (policy consumes bucket+user ARNs)

## Per-object operation timeline (the resources / references)

| Object | Created by | Referenced by | Edge |
|---|---|---|---|
| provider.aws / locals / backend | w1-f1 | everything | root |
| data.aws_iam_role.exec (ARN) | w1-f2 (read) | w1-p1 (profile exec role) | producer->consumer |
| data.aws_iam_user.participant[*] (ARN) | w1-f2 (read) | w2-i1 (policy attach) | producer->consumer |
| data.shared_domain (d-bquhieanzgod) | w1-f2 (read) | w1-p1 (domain_id) | producer->consumer |
| aws_s3_bucket.*[3] (ARN) | w1-s1 (create) | w2-i1 (policy resource ARNs) | producer->consumer |
| aws_sagemaker_user_profile[25] | w1-p1 (create/import) | - | leaf |
| aws_iam_policy.barclays_s3_usw2 | w2-i1 (create) | - | leaf |

## 5-rule analysis (adapted)

**Rule 1 - Multiple cards touch ONE object (fold to one owner).**
No object is CREATED by more than one card. Each AWS resource has exactly one creating card.
-> No fold needed. One owner per object already. OK

**Rule 2 - Irreconcilable shape disagreement on the same object (HARD-CONFLICT).**
No two cards define the same resource with conflicting attributes. The 3 buckets, 25
profiles, 1 policy are each declared once. -> No HARD-CONFLICT. OK

**Rule 3 - rename X->Y.**
No renames. (The bucket NAME change vs spec is a naming DECISION already settled, not a
live-resource rename - the bare names aren't ours.) OK

**Rule 4 - create_enum then column-uses-enum (ordering edge).**
TF analogue: a resource whose attribute is consumed by another. Present and HANDLED:
- w1-s1 creates buckets -> w2-i1's policy ARNs reference them. Ordering edge s1->i1.
- w1-f2 exposes role/user ARNs -> w1-p1 / w2-i1 consume them. Edges f2->p1, f2->i1.
- w1-f1 root -> all. Edges f1->{f2,s1,p1,i1}.
-> All ordering edges captured by `depends_on`; see wave placement. OK

**Rule 5 - a reference to another created object (the FK-to-this-wave class).**
The dangerous analogue: w2-i1's policy `Resource` ARNs point at buckets created THIS epic.
If i1 applied before s1, the ARNs would reference non-existent buckets. In a SINGLE
`terraform apply`, the graph orders s1 before i1 automatically (i1 interpolates
`aws_s3_bucket.*.arn`). In a PER-WAVE apply, the wave split (s1=W1, i1=W2) enforces it.
-> Mitigated by both the dependency interpolation AND the wave boundary. OK
  (We deliberately use `aws_s3_bucket.x.arn` interpolation, NOT a hardcoded ARN string, so
  the graph edge is real - a hardcoded ARN would HIDE the dependency. The builder MUST do this.)

## Verdict
**No HARD-CONFLICTS. No folds required.** The only cross-card edges are clean
producer->consumer ARN references, all expressible as `depends_on` + Terraform interpolation.
The single risk class (Rule 5: policy referencing this-epic buckets) is double-mitigated by
interpolation + the Wave-1->Wave-2 boundary.

## TF-specific risks NOT in the SQL framework (flagged for Gate 2)
1. **Import correctness (w1-p1).** The 14 existing profiles must be imported with resource
   attributes matching their REAL settings, or the plan shows spurious in-place updates. This
   is the TF analogue of "schema card's column already exists" - verify before apply.
2. **Bucket-name global collision (w1-s1).** Apply 409s if the bare names are ever used; the
   datacouch-*-usw2 names are mandatory.
3. **Shared-resource safety (w1-f2, all).** Every shared object MUST be `data`, never
   `resource`. A single mis-typed `resource "aws_sagemaker_domain"` would try to recreate a
   shared domain. Hard invariant.
