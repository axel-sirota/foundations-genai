---
name: workflow-plan
description: PLAN a wave-structured epic from raw human feedback. Takes app feedback (an alpha-test walkthrough, a bug list, a feature ask) and produces the `plan/` directory that /workflow-execute consumes — findings → Jira-like task cards (.card.yaml with explicit files/inputs/outputs/DoD/DDL-intents) → a SEMANTIC DDL conflict analysis (catches the migration-collision class AT PLAN TIME) → human-curated dependency-ordered waves → INDEX.md. This is the PLANNER half of the pair; /workflow-execute is the RUNNER. Two human gates: triage findings, then approve the waves + DDL conflict report. Usage: /workflow-plan <raw-feedback-file | findings-dir> [--epic <slug>] [--sections N]
argument-hint: "<raw-feedback-file | findings-dir> [--epic <slug>] [--sections N]"
disable-model-invocation: false
user-invocable: true
---

# workflow-plan

The PLANNER. Turns raw human feedback into a wave-structured plan that `/workflow-execute` runs.
Designed via a 3-cycle Opus×codex-o3 GAN — `docs/researches/workflow-skill-pair-design.md` is the
blueprint (read it for the full rationale + the GAN transcripts).

`$ARGUMENTS` = `<raw-feedback-file | findings-dir> [--epic <slug>] [--sections N]`. The epic slug
defaults to the input filename stem; the plan lands in `docs/<epic>/`.

## Why this exists (the split)
`workflow-edit` fused planning + execution. The planning half — findings → tasks → waves — was done
ad-hoc by hand (that's how `docs/mvp/first_test/` got built). This skill formalizes it, and adds the
ONE thing that would have prevented the Wave-1 migration disaster: **plan-time DDL conflict
detection** over explicit per-task `ddl_intents`. The seam to the runner is DATA ONLY: the `plan/`
directory. The runner re-uses nothing from here except by reading the cards.

## The output contract — `docs/<epic>/`
```
docs/<epic>/
  RAW_FINDINGS.md           # captured input, labeled (id, verbatim, tags, section)
  <NN>-<section>/           # one dir per app-area section (the parallel axis)
    plan.md                 # human-readable per-section plan (tasks in prose)
    index_of_code_and_test.md  # finding → file:line → covering tests / MISSING
    research.md             # best-practice research for this area
  plan/
    INDEX.md                # human-readable wave rationale + the DDL conflict report
    tasks/<task-id>.card.yaml   # the MACHINE CONTRACT /workflow-execute reads
    ddl_conflict_report.md  # the semantic-DDL analysis output
```

## The pipeline

### Step 1 — Intake (raw feedback → findings)
Read the raw input VERBATIM. Extract discrete **findings** → `RAW_FINDINGS.md`, each with: a stable
id (B1, F3, ...), the verbatim user text, tags, and a guessed section. Do NOT yet design solutions.
**[HUMAN GATE 1]** Show the findings list; the user triages which become tasks (drop noise, split
compound findings, confirm priorities). Do not proceed until ratified.

### Step 2 — Section (group by app area — the PARALLEL axis)
Cluster findings into ~6-10 **sections** by area of the app (auth, home, upload, review, ...). Each
section becomes a `docs/<epic>/<NN>-<section>/` dir. Sections are what fan out in parallel later.

### Step 3 — Per-section panel (the proven first_test pattern)
For each section, run a small panel (use the Workflow engine or foreground agents):
- **code-map agent:** each finding → implementing code (file:line) → covering tests (or **MISSING**).
  This populates the card's `files_touched` from REAL grep, not prose guesses.
- **research agent:** best-practice for this area via `/wsk.search` (token-efficient). → `research.md`.
- **synthesis agent:** merge → per-section `plan.md` with tasks.

### Step 4 — Emit Jira-like CARDS (the machine contract)
Every task becomes a `plan/tasks/<id>.card.yaml`. Schema (v1):
```yaml
id: w1-01t3                 # stable, wave-prefixed
title: "approval_status column + migration"
type: feature              # bugfix | feature | test
priority: P0               # P0 | P1 | P2
section: 01-auth-onboarding-approval
files_touched: [apps/api/db/models.py, migrations/versions/]   # from the code-map grep
shared_resources: [test_db]            # test_db | s3 | redis | node_modules | none
depends_on: []             # other task ids
wave: 1
phase: schema              # schema | consume   (schema = authors DDL; consume = reads it)
dod_tests: ["tests/unit/test_approval_status_default.py::..."]   # the proving tests
human_owner: axel          # who ratifies the domain decision (see gates)
# --- schema-phase ONLY: the SEMANTIC DDL declaration (the crown-jewel field) ---
ddl_intents:
  - object: users.approval_status
    operation: add         # add | alter | rename | drop | create_table | create_enum
    details: {type: text, nullable: true, default: "'approved'"}
# fields kept but IGNORED in v1: data_migrations(bool), external_contracts[]
```
Cards are a REVIEW ARTIFACT, not a trust-blindly contract — the panel proposes `files_touched` +
`ddl_intents`; the human ratifies at Gate 2.

### Step 5 — Codex o3 section audit
Per section, run an adversarial codex o3 pass over the proposed tasks (like the first_test audits) →
fold corrections back into the cards.

### Step 6 — SEMANTIC DDL PLANNING (the crown jewel — catches the Wave-1 class at PLAN time)
Run the planner over ALL cards' `ddl_intents`. Run it `--apply` so it doesn't just REPORT the folds
but actually REWRITES the cards (assigns one schema-owner per object, moves the other cards' DDL onto
it preserving create→add→alter order, flips the losers to `consume`):
```bash
python3 .claude/skills/workflow-plan/scripts/ddl_plan.py docs/<epic>/plan/tasks/ --apply
```
It builds a **per-object operation timeline** and applies 5 rules (o3-converged):
1. **Multiple cards touch one object** (add→alter, create→add-column) → FOLD: one schema-owner gets
   ALL of that object's ops in order (create→add→alter), the other cards become `consume`. `--apply`
   rewrites the cards (the ops are MOVED to the owner, never dropped — a NOT-NULL alter on a loser
   becomes a 2nd op on the owner so the constraint still ships).
2. **Irreconcilable shape disagreement** on the same object → HARD CONFLICT → halt. A `type` clash
   (INT4 vs BIGINT) is HARD across any two defining ops; a `nullable`/`default`/`length` clash is HARD
   only among ops of the SAME kind (≥2 `add`, or ≥2 `alter`) — a same-stage contradiction. An
   add-nullable→alter-NOT-NULL SEQUENCE is NOT a clash — that's Rule 1's fold.
3. **rename X→Y** → ordering edge X→Y + flag every other card still referencing X (object, fk_target,
   OR default SQL) so its reference gets rewritten.
4. **create_enum then column-uses-enum** → add an ordering edge in the schema phase.
5. **`fk_target` OR a `default`** (fk, `nextval('seq')`, `::enum` cast) **references another created
   object** → ordering edge target→op (the FK-to-a-table-created-this-wave class — Alembic dies
   "relation does not exist" if run parallel).
Output: `plan/ddl_conflict_report.md` (also EMBEDDED into `plan/INDEX.md` for Gate 2) + the
schema-owner assignment. `--apply` will NOT fold while a HARD CONFLICT is present (a fold would bake
in a wrong shape) — the human reconciles the type/nullability first, then re-run `--apply`. This is
ONE planner reasoning over DECLARED intent (not parallel-blind agents) → reliable, catching the
operation/type/order/FK conflicts that simple object-name matching misses.

### Step 7 — Human-curated waves (NOT a pure topo-sort)
Group tasks into dependency-ordered **waves**. This is HUMAN-CURATED, not a mechanical DAG sort: real
cycles (e.g. seat-invariant ↔ approve-with-role) get broken by CHANGING SEMANTICS — a judgment call.
Section + priority + dependency are co-equal axes. The DEFAULT shape (from the design):
- **Wave 1 = Schema + Primitives** — every card with `phase: schema` + the primitives consumers need.
- **Wave 2+ = Consumers** — pure `consume` cards, layered by `depends_on`.
Within any wave that has schema cards, the runner enforces `schema-phase → barrier → consume-phase`.
Emit `plan/INDEX.md` (the wave list + rationale + the embedded DDL conflict report).

#### THE RUNTIME-DEPENDENCY RULE (Wave-2 first_test, 2026-06-12 — the hardest wave-grouping lesson)
**A `depends_on` between two CONSUME cards in the SAME wave is a TRAP — split it across waves, OR pull
the pair into one card.** Why: tasks in a wave build in PARALLEL isolated worktrees off the wave's
START commit. So if B `depends_on` A and B consumes A's RUNTIME output (a queue, an endpoint, a
service method — not just a schema column), B CANNOT build against it: A isn't merged yet during the
parallel build. In Wave 2, `f2-approve-with-role depends_on f1-approval-queue`, both in the same wave —
f2 built off a base WITHOUT f1's queue, delivered no working consumer, the audit DO-NOT-SHIP'd the wave.
Two failure modes the planner MUST prevent at plan time:
1. **Build-base mismatch.** B can't consume A's unmerged runtime output → B builds wrong or empty.
2. **Inert-half ship.** A landed without its consumer B → the FEATURE is non-functional (f1's approval
   queue with no f2 gate = pending users still log in; the queue moderates nothing). A producer and
   its REQUIRED enforcement point belong in the same shippable unit.
Distinguish the dependency KIND:
- **Schema dependency** (B reads a column/table/enum A creates): SAFE within a wave via the
  schema→barrier→consume phasing — the schema is frozen before consumers build. This is the normal case.
- **Runtime dependency** (B calls A's new endpoint / queue / service fn): NOT safe in the same wave.
  → put B in a LATER wave (so A is merged + on B's base), OR merge A and B into ONE card if they're a
  single feature (a producer + its enforcement gate — e.g. approval-queue + approve/login-gate).
At Gate 2, scan every intra-wave `depends_on`: is it schema (ok) or runtime (re-wave or merge)? The
loader's `merge_order` honors deps for MERGING, but it does NOT fix the parallel BUILD base — only
re-waving or card-merging does.

**[HUMAN GATE 2]** The user approves: the wave grouping (incl. the runtime-dependency scan above), the
schema-owner assignment, and the DDL conflict report (esp. any HARD CONFLICT from rule 2). Do not
finalize until ratified.

### Step 7.5 — FINAL whole-plan audit (MANDATORY — runs before hand-off, every time)
Step 5 audits each section in isolation; it CANNOT catch cross-plan + against-reality problems. This
final gate does — it is NOT optional, and a plan does NOT hand off to the runner until it passes.

**A. PATH VERIFICATION (the execution-readiness blocker).** Every card's `files_touched` is, by
construction, LLM-PROPOSED — it WILL drift from the real tree (missing Next.js route groups like
`app/(app)/...`, invented `*_service.py` names, non-existent `auth/`/`validators/`/`workers/` dirs).
For EACH card, confirm every `files_touched` path EXISTS (`ls`/`git ls-files`); for any that doesn't,
GREP the finding's keywords to find the real owner and PATCH the card. A wrong path = that worktree
agent thrashes hunting a missing file. This is the #1 reason a real run wastes itself — do it before,
not during, execution. Also confirm each schema card's `ddl_intents` object does NOT already exist in
`models.py` (an `add` of an existing column is a no-op/conflict).

**B. CODEX o3 WHOLE-PLAN REVIEW (mandatory second pair of eyes).** Run a codex review over the WHOLE
plan + the real tree:
```bash
codex exec -s workspace-write --json -m o3 -c 'reasoning_effort="high"' \
  "Review docs/<epic>/plan/tasks/*.card.yaml + ddl_conflict_report.md against the input
   docs/<epic>/FIRST_MVP_TEST.md and the REAL app tree. Per card: is every files_touched path real
   (grep it)? Do the schema cards' columns already exist in models.py? Did any finding get DROPPED
   (no card) or any card not trace to a finding? Will load_wave.py exit 0 (phase/wave set, no consume
   card carrying ddl_intents)?
   INTRA-WAVE RUNTIME DEPENDENCY CHECK (the Wave-2 trap): for every card B with a depends_on A where A
   is in the SAME wave, classify the dep — SCHEMA (B reads a column/table A creates: safe, the
   barrier handles it) vs RUNTIME (B calls A's new endpoint/queue/service fn: NOT safe — B builds in a
   parallel worktree off the wave start, before A is merged, so it can't consume A). Flag every runtime
   intra-wave dep: B must move to a LATER wave, OR A+B merge into one card if they're a single feature
   (a producer + its required enforcement gate). ALSO flag any 'inert half': a producer card landing
   without its required consumer/enforcement in the SAME shippable wave (e.g. an approval-queue with no
   approve/login-gate → moderates nothing).
   End: READY TO EXECUTE (after fixing X) or NOT READY (list everything)."
```
Fold its findings back into the cards. **The plan is execution-ready ONLY when codex says READY and
every path resolves.** Write the verdict to `plan/CODEX_PLAN_REVIEW.md`.

**C. LOADER SELF-CHECK.** Run `load_wave.py docs/<epic>/plan --wave N` for every wave — each must
exit 0 (the integrity gate). A non-zero exit means the plan is unsafe; fix it here, not in the runner.

#### Lessons from the first real run (first_test, 2026-06-12) — the path-fix is PREDICTABLE
The first whole-plan run hit codex's NOT-READY on **16 path mismaps out of 24 cards** while the
STRUCTURE was sound (codex: "DDL wave is sound… no deeper logic conflicts"). Expect this exact shape —
the plan's waves/DDL/deps are usually right; only the LEAF paths drift. The mismaps clustered into a
few mechanical patterns — check these FIRST and most fixes fall out:
- **Next.js route groups.** Every `app/<route>/page.tsx` is really `app/(app)/<route>/page.tsx` (or
  `(auth)/`). The LLM omits the route-group segment every time. → prepend `(app)/`.
- **Component subfolders.** `components/Foo.tsx` is really `components/{shell,home,review}/Foo.tsx`
  (e.g. `shell/Sidebar.tsx`, `shell/CompanySwitcher.tsx`, `shell/SeatBadge.tsx`,
  `home/CompanyStatusTable.tsx`, `review/ReviewTable.tsx`). → grep `find components -iname`.
- **Phantom backend services.** The LLM invents `*_service.py` per concept; the repo doesn't split
  that way. Real owners: seats→`firm_service.py`+`member_service.py`; home counts→
  `firm_overview_service.py`; review/edit + reconcile + period-validation→`routes/jobs.py` (NOT a
  service); retry→`job_service.py`. There is NO `apps/api/auth/` dir → auth/permissions live in
  `apps/api/auth_perms.py`; NO `routes/admin.py` → use `routes/users.py`; NO `validators/` dir.
- **Phantom seed.** No `db/seed.py` — drop it; the seat-recompute owner is `firm_service.py`.
- **Fixing paths SURFACES HIDDEN OVERLAPS.** Once corrected, `load_wave.py`'s overlap map revealed
  `routes/jobs.py` is touched by 3 tasks (period-validate + lost-job + edit-persist) — masked while
  the paths were wrong. Re-read the overlap map AFTER the fix; the merger needs the real one.
- **Schema cards were all valid** (codex confirmed none of the 5 columns already existed in
  `models.py`) — so the §A "already exists?" check usually passes; the path check is where the work is.

## Hand-off to the runner
When `plan/` is ratified AND Step 7.5 passed (paths verified + codex READY + loader exit 0):
`/workflow-execute docs/<epic>/plan --wave 1`. The runner reads the cards
(`phase`, `ddl_intents`, `shared_resources`, `files_touched`, `depends_on`) — it does NOT re-derive
them from prose. The frozen-wave invariant holds: once execution starts, wave numbers never change;
re-planning only appends NEW tasks to FUTURE waves (`/workflow-plan --incremental`, post-v1).

## Hard rules (the GAN-converged invariants)
1. **Two human gates** — triage (G1) + plan-approval-with-DDL-report (G2). The user is the domain
   oracle (only they know "approve-with-role means X", "holding-firm vs nullable firm_id"). Don't
   skip them; the panel PROPOSES, the human RATIFIES.
2. **Contention is prevented at PLAN time** by the semantic DDL planner + the schema-owner assignment
   — NOT discovered at runtime. The runner's reconciler is a LAST-RESORT backstop, not the primary fix.
3. **One schema-owner per object.** If two cards declare the same object, the planner assigns the DDL
   to ONE schema card; the others become `consume`. (This is what Wave 1 lacked — no plan-time intent.)
4. **Deterministic, never placeholder, revision ids** — the schema-owner derives ids from slug + the
   real current head. Encoded in the schema card's prompt by the runner.
5. **v1 scope (o3-trimmed):** `external_contracts` + `data_migrations` ordering are FIELDS only,
   ignored in v1 logic. Build the 5-rule DDL planner first — it's the irreducible fix.

## Reference
- Blueprint + GAN transcripts: `docs/researches/workflow-skill-pair-design.md`,
  `docs/researches/gan-workflow-skill-pair/`.
- The runner: `.claude/skills/workflow-execute/skill.md` (consumes this skill's `plan/` output).
- Migration redesign rationale: `docs/researches/wave-migration-redesign.md`.
- A real (hand-built) example of this skill's output: `docs/mvp/first_test/` (INDEX + sections).
