---
name: workflow-execute
description: EXECUTE a wave-structured plan (the `plan/` directory produced by /workflow-plan, or a wave-ordered INDEX.md) as a worktree-per-task build→verify→merge panel. Each task gets its OWN git worktree, so tasks in a wave run in parallel EVEN IF they touch the same files — overlap is resolved by YOU (the workflow master / sole merger) when you merge each finished branch back into the feature branch. Per task: create worktree → write a transition/context file → agent builds the task → per-task verifier+audit → you merge into feat + resolve conflicts. After ALL of a wave's tasks are merged: a full audit + codex-consult (o3) review of the whole merged wave, then the next wave. This is the RUNNER half of the pair; /workflow-plan is the PLANNER that produces what this consumes. Usage: /workflow-execute <plan-dir|INDEX.md> [--wave N] [--tier simple|ui] [--max-stacks N] [--verify "<cmd>"] [--dry-run]
argument-hint: "<plan-dir|INDEX.md> [--wave N] [--tier simple|ui] [--max-stacks 8] [--verify \"pytest -m hp\"] [--dry-run]"
disable-model-invocation: false
user-invocable: true
---

# workflow-execute — the RUNNER

The execution half of the workflow pair. **`/workflow-plan` is the PLANNER** — it produces the
`plan/` directory (Jira-like `.card.yaml` cards + INDEX.md + the DDL conflict report) that THIS skill
consumes. Generalizes this project's wave-execution pattern (backend —
`docs/specs/_panel-2026-06-08/PARALLELIZATION.md`; UI e2e — `docs/ui-audit/HANDOFF.md`).

`$ARGUMENTS` = `<plan-dir | INDEX.md> [flags]`. First positional = a `plan/` directory from
`/workflow-plan` (preferred) OR a wave-ordered `INDEX.md` (the legacy hand-built form, e.g.
`docs/mvp/first_test/INDEX.md`).

## Read the CARDS via `load_wave.py` — don't re-derive from prose (the contract with /workflow-plan)
When given a `plan/` directory, the FIRST thing you do is run the card loader — it parses
`plan/tasks/*.card.yaml`, splits the wave into schema/consume, computes lock groups + overlaps +
merge order, reads the planner's ordering edges, and GATES on an unsafe plan. You then drive from its
structured output; you do NOT re-grep prose to re-derive what the planner already decided:
```bash
python3 .claude/skills/workflow-execute/scripts/load_wave.py docs/<epic>/plan --wave <N> --json
```
- **Exit 2 = UNSAFE — STOP.** The loader found a HARD CONFLICT still in `ddl_conflict_report.md`, or
  a `phase: consume` card that still carries `ddl_intents` (the planner didn't fold it). Do NOT start
  the wave — go back to `/workflow-plan` (re-run `ddl_plan.py --apply`, re-ratify Gate 2). The loader
  is the gate that stops a broken plan from ever reaching worktrees.
- **Exit 0** → use the JSON directly:
  - `schema_tasks` / `consume_tasks` → the two phases (you do NOT re-classify — Step 0.5 just HONORS
    this split + the `ordering_edges`).
  - `ordering_edges` → schema-phase build order (predecessor → dependent), from the planner's report.
  - `lock_groups` → tasks sharing a lock (`test_db|s3|redis`) must NOT run concurrently — serialize
    them within the wave (do not launch both worktrees' verifiers against the same DB at once).
  - `overlaps` → the files ≥2 tasks touch = where you (the merger) will resolve conflicts.
  - `merge_order` → the `depends_on` topo order you merge in.
  - `expect_migrations` → pass `--expect-migrations` to `migration_reconcile.py` when true (so a
    schema phase that produced ZERO migrations fails closed instead of silently lifting the barrier).
  - `dod_tests[<id>]` → that task's per-task verify command.
The TRANSITION.md you write per task is DERIVED FROM THE CARD (the card IS the spec; read the matching
`plan/tasks/<id>.card.yaml` for `title`, `files_touched`, `ddl_intents`). For a legacy `INDEX.md` with
no cards, `load_wave.py` does not apply — fall back to the prose-parsing Step 0 below.

**FROZEN-WAVE INVARIANT:** once execution of a wave starts, wave NUMBERS never change. Re-planning
(`/workflow-plan --incremental`, post-v1) may only APPEND new tasks to FUTURE waves — it never
renumbers or moves an in-flight wave. This keeps resume + traceability deterministic.

## The mental model — worktree-per-TASK, master is the merger

**One git worktree per task. The worktree IS the isolation.** Tasks in a wave are parallelizable
because each runs in its own worktree on its own branch — so **two tasks may touch the SAME file and
that is fine**. There is NO "disjoint-file grouping" requirement (that was wrong). Overlap is
resolved at **merge time** by YOU, the workflow master and **sole merger**, who merges each finished
branch back into the **feature branch** and resolves any conflicts.

The per-task pipeline (all tasks of a wave run this concurrently):
```
create worktree  →  write transition/context file  →  agent builds the task  →
   per-task VERIFIER + AUDIT (in/against the worktree)  →  MASTER merges into feat + resolves conflicts
```
After **all** of a wave's tasks are merged into feat:
```
→  WAVE-LEVEL full audit  +  codex-consult (o3) reviewing the WHOLE merged wave  →  next wave
```

So there are **two audit points**: (1) per-task, gating each branch before it merges; (2) wave-level,
reviewing the merged whole. The master merges **as each task finishes** (not a barrier) — early
branches merge clean, later branches conflict-resolve onto a growing feat.

## Flags
- `--wave N` — run a single wave (default: the wave named by the arg, e.g. `wave0`).
- `--tier simple|ui` (default `simple`) — `ui` adds a per-worktree Docker stack + Playwright (below).
- `--max-stacks N` (Tier B cap, default 8) · `--verify "<cmd>"` (override the verify/audit cmd) ·
  `--dry-run` (decompose the wave into tasks+worktrees+files and STOP — always do this first).

## Tiers (Docker is Tier B only — backend used NO Docker)
- **Tier A — `simple`** (backend default): worktree + `pytest` verify. No Docker.
- **Tier B — `ui`** (real-stack e2e): worktree + an isolated Docker stack per task
  (`scripts/ui_e2e_stack.sh up <N>`, ports offset `N*10`, capped by **MAX_STACKS**), verify = real
  Chromium Playwright. Tier-B worktrees also need symlinked `.venv`, `.env`, AND
  `apps/web-next/node_modules` (pnpm) — use `scripts/worktree.sh up <slug> <plan-slug> --ui`.

---

## Step 0 — Parse the wave (always `--dry-run` first)
Read the plan. For the target wave, list each **task** (ID + one-line) and its **files to touch**
(from the per-section `plan.md` "Files:" lines). One task = one worktree = one agent. Files MAY
overlap across tasks — that is expected; note the overlaps so you know where conflicts will surface
at merge. Emit and STOP if `--dry-run`:
```
Wave 0  (gate: prior waves' deps met)
  T-§01-T1   worktree wf/<slug>/01-t1   files: routes/session.py, services/company_service.py, ...
  T-§03-T0   worktree wf/<slug>/03-t0   files: services/firm_overview_service.py, db/models.py, ...
  ...
  overlaps (expect merge-conflict resolution by master): db/models.py {§02-T1, §03-T0}, ...
```

## Step 0.5 — Migration discipline (THE fix for the Wave-1 disaster — read before Step 1)
**IF YOU HAVE A `plan/` FROM /workflow-plan: the planner ALREADY did this, and `load_wave.py` handed
you the result.** Use the loader's JSON, not your own re-derivation:
- `schema_tasks` / `consume_tasks` ARE the two phases — run `schema_tasks` first (barrier), then the
  reconciler, then `consume_tasks`. Do NOT re-classify.
- `ordering_edges` is the within-schema-phase build order — when two schema tasks have an edge
  `A → B`, B's worktree must base off A's merged result (build A first, merge, then B). The
  planner's `ddl_plan.py --apply` already folded multi-owner objects to ONE schema-owner, so two
  schema tasks will NOT both author the same column — that's guaranteed at plan time, not hoped for.
- `expect_migrations` from the loader → pass `--expect-migrations` to the reconciler iff true.
- If `load_wave.py` exited 2 you never got here — the plan was unsafe and you stopped.

The detailed steps below are the FALLBACK for a legacy `INDEX.md` with no cards (where you must
classify schema/consume yourself).

Wave 1 died because 8 tasks independently authored alembic migrations: 6 hard-coded the SAME
placeholder revision id and 3 pairs RE-CREATED the same schema (each agent built in an isolated
worktree, blind to its siblings). The redesign (codex o3 + `docs/researches/wave-migration-redesign.md`):

**A migration WAVE is split into two phases with a BARRIER between them.** Producers of DDL run
first; consumers of that DDL run after the schema is frozen. This is expand/contract applied to waves.

1. **Classify every task `--dry-run` as `schema` or `consume`.** A `schema` task's deliverable
   INCLUDES a `migrations/versions/*.py` file or a model/column/table change. Everything else is
   `consume`. Emit both buckets in the dry-run.
2. **`Needs-DDL:` tags.** Every TRANSITION.md gets a header line:
   `Needs-DDL: <none | add column users.approval_status TEXT default 'approved' | create table audit_log(...) | ...>`.
   The orchestrator collects all non-none tags into `<repo>/wave<N>/ddl_intents.yaml`.
3. **ONE-OWNER RULE — a `consume` task must NEVER commit `migrations/versions/` files.** Its
   TRANSITION.md states: "Do NOT author any alembic migration. The column/table you need is created
   by the schema phase; build against it. State your need in `Needs-DDL:`." If `--dry-run` finds ≥2
   tasks that would author real migrations and they're not all in the schema phase → **ABORT** and
   re-bucket.
4. **Deterministic revision ids — NEVER a placeholder.** Each schema task's migration id is derived
   from its slug (e.g. `w1s_<slug>` truncated/uuid5-of-slug), and `down_revision` chains off the
   ACTUAL current head (`alembic heads`, not assumed). No two tasks can collide.
5. **Schema phase has its own barrier + a reconciler.** Topology (the Workflow `parallel()` IS the
   barrier — it awaits all):
   ```js
   phase('schema')
   const schema = await parallel(schemaTasks.map(t => agent(schemaPrompt(t, ddlIntents), {schema:V})))
   // BARRIER reached. MASTER merges the schema branches, THEN runs the reconciler BEFORE consumers:
   //   python3 .claude/skills/workflow-execute/scripts/migration_reconcile.py wave<N> \
   //       --since <pre-wave-head> --expect-migrations --apply
   //   --apply auto-regenerates colliding ids + linearizes the chain + writes the sentinel; a
   //   DUPLICATE-DDL collision stays fail-closed (exit 2, no sentinel — resolve which body keeps the
   //   object by hand). --expect-migrations (from load_wave.py's expect_migrations) fails closed if
   //   the schema phase produced ZERO migrations. THEN run the throwaway-PG drift gate
   //   (test_UP_infra_foundations_alembic_drift → empty autogenerate diff) before consumers/apply.
   phase('consume')
   const consumers = await parallel(consumeTasks.map(t => agent(consumePrompt(t), {schema:V})))
   ```
6. **Verify hook (master, after schema merges):** reject the wave if the same `revision` value
   appears in two files OR an `op.create_table`/`op.add_column` duplicates an existing object. The
   repo's `tests/integration/test_UP_infra_foundations_alembic_drift.py` (V29) is the model-sync
   gate — the reconciler must produce a state that passes it (`alembic check` → empty diff).
7. **Migrations are AUTHORED by Claude, APPLIED BY AXEL.** The reconciler verifies against a THROWAWAY
   local Postgres; it NEVER runs `upgrade` on a shared/prod DB. Multiple migrations → one linear chain
   (NOT a multi-head merge unless genuinely unavoidable).
8. **THE MULTI-HEAD THE FIRST RUN HIT (2026-06-12 first_test, now FIXED in the reconciler).** Every
   schema agent correctly chains off the SAME real head (Step 0.5.4) — so after merge you have N
   migrations ALL with `down_revision=<that head>` = an N-way fork = N alembic heads. This is EXPECTED
   and the reconciler `--apply` linearizes it. BUT it only works if the reconciler can SEE the fork:
   - Run the reconciler with the env that lets `alembic heads` run (`source .env`), OR trust its
     FILE-LEVEL detection (it now flags any `down_revision` shared by ≥2 wave migrations even when
     alembic can't run — the original miss was `alembic heads`→None silently skipping the check).
   - After `--apply`, CONFIRM a single head: `source .env && alembic heads` must show exactly ONE.
   - The first run's reconciler reported "concordant (no-op)" on a real 5-way fork because alembic
     couldn't run from the repo root → the master had to linearize by hand. The code now catches it;
     still verify the single head before consumers.
9. **WORKFLOW-DIED-MID-RUN RECOVERY (also first_test).** The schema-phase Workflow can die after agents
   BUILD but before they COMMIT (4/5 agents had the migration + model change uncommitted in their
   worktree, only 1 committed). This is RECOVERABLE, not a re-run: (a) inspect each worktree's
   uncommitted work, (b) AUDIT each migration (real non-placeholder revision id, correct
   `down_revision`, DDL scoped to only that task), (c) commit the audited work on its branch, (d) merge
   + reconcile as normal. Do NOT just re-run the workflow — the work exists; re-running re-does it.

If the wave has NO schema tasks, skip the schema phase — it's a pure `consume` wave (one `parallel()`).

## Step 1 — Create worktrees + transition files (one per task)
For each task, create its worktree off the **feature branch HEAD** and write a transition file that
gives the agent everything it needs:
```bash
# worktree per task, branched off feat (NOT main — we merge back into feat)
.claude/skills/workflow-execute/scripts/worktree.sh up <task-slug> <plan-slug>      # add --ui for Tier B
```
Then write `../<repo>-<task-slug>/TRANSITION.md` containing: the task ID + full text from the
section `plan.md`; the files to touch; the relevant context files to read (the section's
`index_of_code_and_test.md`, `research.md`, and any code the plan cites); the Test Constitution
minimums; and the explicit rules below. This transition file is the agent's brief.

**Each worktree agent runs the FULL SDD RITUAL on its own task — autonomously, autopilot ON — NOT a
bare implement.** This is the corrected model: every task is a real SDD feature in its OWN worktree
(its own branch = its own micro-slug = its own markers), and the agent walks the entire kryla pipeline
end-to-end without ever pausing for a human. "Atomic" means ONE functionality per worktree — it does
NOT mean "skip the ceremony." Skipping straight to implement is what produced under-tested, unspecced
task branches; the ritual is what makes each branch independently shippable BEFORE it merges.

**The agent's micro-slug = its worktree branch slug** (`<task-slug>`, e.g. `g4-edit-persist`). All its
kryla artifacts (spec, manifest, markers) are namespaced to THAT slug, so parallel worktrees never
collide on markers. The TRANSITION.md MUST instruct the agent to, IN ITS OWN WORKTREE:

1. **Turn autopilot ON first:** `touch .claude/markers/autopilot.on` (skips every human-in-the-loop
   pause across the kryla skills). NOTE: autopilot does NOT bypass the `/kryla.skeleton` review — so
   the agent SELF-APPROVES its own skeleton (it is the sole owner of this atomic task; type the `YES`
   the skeleton gate asks for itself). It is non-interactive; only the MASTER ever talks to the human.
2. **Run the full ritual on its micro-slug, in order, autonomously — but SKIP the branch-creating
   bookends.** The worktree is ALREADY on its branch `wf/<plan>/<task-slug>`, so the agent does NOT run
   `/kryla.request` (it would try to create `feat/<task-slug>` from main and fail — exit 2/4) and does
   NOT run `/kryla.finish-feature` (the master merges). Instead the agent:
   - lays its own request marker so the gates open: `touch .claude/markers/dod-read-<task-slug>.done`
     (the override path — `/kryla.request`'s only effect the agent needs here; its branch already exists);
   - then runs the CONTENT steps of the ritual, in order, on `<task-slug>`:
     **`/research` FIRST for any `type: bugfix` card** (or any card whose ROOT CAUSE is unknown) —
       investigate WHY the bug happens in THIS codebase before designing a fix. The card states a
       SYMPTOM ("completed extraction → 'no se encontró la carga'"), not the cause; `/research` finds
       the real cause (e.g. "job-detail is gated on active-company, which the workspace never sets") so
       the fix targets the cause, not the symptom. (`type: feature` cards may skip `/research` — their
       cause isn't in question; `/kryla.specify`'s own research covers best-practice.) →
     `/kryla.specify <task-slug>` (research → spec → manifest; the TRANSITION + the card's `files_touched`
     + the `/research` findings are the input, so spec research is fast — pass `--no-websearch` if offline) →
     `/kryla.clarify <task-slug>` (DoD + verification matrix) →
     `/kryla.skeleton <task-slug>` (xfail stubs — SELF-APPROVE the review) →
     `/kryla.implement <task-slug>` (TDD task-by-task: red → green) →
     `/kryla.tests <task-slug>` (coverage ≥90% + full suite) →
     `/kryla.dod-test <task-slug>` (resolve every skeleton stub, all V-rows green).
   The branch stays `wf/<plan>/<task-slug>` throughout — the ritual writes spec/markers FOR `<task-slug>`
   on that branch; it never re-branches. The MASTER merges `wf/<plan>/<task-slug>` into feat (agents
   never push/merge/finish-feature).
3. **Report back** green/red + branch + which ritual markers it wrote (tests-<slug>.done,
   dod-test-<slug>.done, etc.) so the master can carry them into the wave's finish.
- **Do NOT ask the human anything, do NOT pause** — everything needed is in TRANSITION.md (the task came
  through research → code-map → codex-audit at plan time). Self-approve the skeleton; execute end-to-end.
- **A `Needs-DDL:` header line** (see Step 0.5): the exact column/table the task needs.
- **For a `consume` task:** "**Do NOT author or commit any `migrations/versions/` file and do NOT
  change `apps/api/db/models.py` schema.** The column/table you need is ALREADY created by the schema
  phase (on your base branch — `alembic heads`/read models.py to confirm). Build against it; if missing,
  STOP and report — never create it yourself." (Its `/kryla.skeleton`+`/kryla.implement` stay within this.)
- **For a `schema` task:** "Author your migration with a revision id derived from THIS slug,
  `down_revision` = the ACTUAL latest head (`alembic heads`, never assume/placeholder). Create ONLY the
  DDL in your `Needs-DDL:` + `ddl_intents.yaml` — never schema another task owns. Author the file; never
  run `alembic upgrade` on a shared DB (Axel applies)."

**Markers are PER-WORKTREE and per-micro-slug.** Each agent writes ITS OWN `tests-<task-slug>.done`,
`dod-test-<task-slug>.done`, etc. in its worktree — that's correct and required (the ritual produces
them). The landmine "never `git add markers/` blindly" still holds: an agent commits only ITS slug's
markers, never the whole `markers/` dir (that would leak/wipe sibling slugs — see Landmine #9). The
master still lays the wave-level branch-guard override so the hook permits source writes.

## Step 2 — Run the wave via the NAMED workflow (do NOT author a per-wave script)
**DON'T hand-write a workflow script per wave — 95% is identical.** There is ONE canonical, runnable
5-step workflow at **`.claude/workflows/wave-execute.js`** (build → verify → review → MERGE → audit,
all in-script). It resolves **by name**, driven ENTIRELY by `args` — you pass `load_wave.py`'s JSON
straight in. Merge + audit are real workflow phases, so they show in `/workflows`:
```js
// 1. get the plan (the workflow's contract — NO translation):
//    python3 .claude/skills/workflow-execute/scripts/load_wave.py docs/<epic>/plan --wave N --json
// 2. run the named workflow with that .plan:
Workflow({
  name: 'wave-execute',
  args: {
    plan,                         // <- the .plan object from load_wave.py (tasks/merge_order/overlaps/…)
    feat: 'feat/<branch>',        // merge target (never main)
    base: '<pre-wave SHA>',       // for the Step-5 audit diff (git diff base..feat)
    epicDir: 'docs/<epic>',       // where cards + the WAVE<N>_CODEX_AUDIT.md live
    tier: 'simple',               // or 'ui'
    fullRitual: true,             // each worktree runs the full SDD ritual (Step 1)
  },
})
```
That single call runs all 5 steps engine-side: avoids the 600s watchdog, is **resumable**
(`resumeFromRunId` — cached agents + the idempotent is-ancestor merge mean a resume re-does nothing),
and satisfies the sequential-tools rule. You do NOT edit the workflow per wave — ONLY the args change.
`.claude/skills/workflow-execute/templates/wave.workflow.js` is just a POINTER to the canonical file.
(Author a bespoke script ONLY for a genuinely non-standard wave — e.g. a schema-phase barrier that
needs the reconciler between sub-phases; see Step 0.5. The named workflow covers the normal wave.)

Before launching: create the worktrees + write each `TRANSITION.md` (Step 1), and lay the wave-level
override marker. The template's build prompt tells each agent to `cd` its worktree, read TRANSITION.md,
and run its ritual.
- **Foreground Agent calls** for a tiny wave (≤2 tasks). Never background standalone agents for
  builds — the 600s watchdog kills them.

Each agent's job: read its `TRANSITION.md`, then run the **FULL SDD RITUAL on its micro-slug**
(autopilot ON → request → specify → clarify → skeleton[self-approve] → implement → tests → dod-test,
per Step 1) for ONLY its one task in its worktree, following the Test Constitution. It **commits** on
its branch (source + its OWN slug's markers) and reports green/red + branch + files-changed + the
ritual markers it wrote (use a structured `schema`). It does NOT run finish-feature — the master
merges. Tier B: bring up its stack (`ui_e2e_stack.sh up <N>`), Playwright at `:$((3000+N*10))`,
`down <N>`; commit with `ALLOW_DIRECT_MAIN=1` (branch_guard false-fires in worktrees).

## Step 3 — Per-task VERIFIER + AUDIT (before merge)
As each build agent reports done, run its verifier+audit **against that task's worktree branch**
before merging:
- **Verifier:** run the task's tests / `--verify` cmd on the worktree (Tier B: its stack's Playwright).
- **Audit:** a quick adversarial check — did it actually satisfy the task, break an invariant
  (tenancy/firm_id, B8 seat `used_seats==verified_count`, optimistic-lock version, SSE reconnect
  counter, exactly-once per file), or fake green? If RED → bounce back to the build agent (same
  worktree) to fix, then re-verify. Only a green-and-audited branch proceeds to merge.

## Step 4 — MASTER merges into feat + resolves conflicts (INCREMENTALLY, as each verifier ends)
You (the master) are the **sole merger**, and you merge **one-by-one the moment each task's
verify+audit passes** — NOT batched at the end. Earlier branches merge clean; conflicts surface and
resolve incrementally instead of 14-at-once.

**Two ways to merge — both honor the single-merger rule:**
- **(Preferred) In-workflow Step 4** (see "Canonical workflow shape" below + `templates/wave.workflow.js`):
  the SCRIPT, after the fanout returns, runs a SERIAL deterministic merge loop (one merge-agent), with an
  is-ancestor post-condition after each. The FANOUT never merges; the SCRIPT does. This is the reliable
  path — a hand-rolled merge outside the workflow once silently dropped 10/14 branches.
- **(Legacy) Master-merges-on-sentinel:** the verify stage writes `echo "<branch>" >
  <repo>/wave<N>/<slug>.mergeready` on `mergeReady=true`; the master polls + merges each as it appears.
  Use only when you're driving merges by hand outside a workflow. EITHER way: only the master/Step-4
  agent merges, into FEAT, never main; the parallel fanout never merges.

As each task passes Step 3, merge its branch into the feature branch and resolve conflicts (overlap
is expected):
```bash
git checkout <feat-branch>
git merge --no-ff wf/<plan-slug>/<task-slug>        # resolve conflicts here — you own this
```
Merge **incrementally as tasks finish**, not in one barrier. Never let agents merge; never push
(the user pushes to origin). If a conflict is non-trivial, you may spawn a dedicated
conflict-resolution agent against the merge, but the decision to merge stays with the master.

**STALE-FORK RE-VERIFY (o3 review #1 — the biggest correctness gap).** Each task's tests ran on its
OWN fork, NOT on the ever-moving feat branch. Two individually-green branches can interact badly once
the earlier one is merged. So AFTER every merge (especially a conflict-resolved one), **re-run the
just-merged task's tests AND every already-merged task's tests on the new feat HEAD.** If green→red,
the merge introduced an integration break — fix it before merging the next branch. A conflict
resolution that compiles + passes one side's tests can SILENTLY DROP the other side's behavior
(adversarial #2): when two branches edit the same function (e.g. one adds a tenancy check, the other
a version bump), keep BOTH changes — never take one side wholesale. Diff the merged function against
BOTH parents to confirm nothing was lost.

**FAKE-GREEN GUARD (adversarial #1).** An agent can over-mock its own seam, narrow assertions, or test
only its happy path. Before trusting a task's green: the per-task AUDIT (Step 3) must check the diff
for test-weakening (new mocks of the OWNED boundary, deleted/loosened assertions, no negative test)
and demand the agent's failing-before/green-after evidence. Don't accept a green that only the agent
asserts.

**SHARED-FIXTURE GUARD (adversarial #6 — the Wave-0 CSRF-flag class).** A task editing
`tests/conftest.py`, a settings singleton, or a global fixture changes behavior for ALL tasks. After
merging any branch that touches a shared fixture/conftest/config, re-run the FULL suite (not just the
task's tests) before the next merge — a shared-fixture change scoped wrong (e.g. CSRF off globally
instead of non-real-deps-only) reddens unrelated tasks.

## Step 5 — WAVE-LEVEL full audit + codex-consult (o3) — after ALL of the wave is merged
Once every task in the wave is merged into feat:
1. **Full audit:** run the whole suite on merged feat — `--verify` (default
   `.venv/bin/python3 -m pytest tests/ -m "not e2e and not skeleton"`), Tier B adds a fresh stack +
   the wave's Playwright + backend regression. **Flag only NEW failures** (known baseline:
   `test_UP_alembic_check_detects_model_drift`, `test_afip_client::test_default_initialization`).
2. **codex-consult o3 review of the whole merged wave:**
   ```bash
   ~/venvs/global/bin/python3 .claude/skills/codex-consult/scripts/run.py --model o3 \
     "Adversarially review the merged Wave-<N> diff (git diff <feat-base>..HEAD) of <plan-slug>:
      every hole, broken invariant, missed edge case, or task that didn't deliver. Cite file:line."
   ```
   (o3 reads the workspace under the consult's read-only context; if it can't read files, fall back to
   the codex-adversarial `run.py <slug>` which feeds the diff inline.)
   **TRUNCATION GUARD (adversarial #8):** o3's diff window is bounded; for a big wave (>~25kLOC or
   >~30 files changed) it WILL truncate and false-pass on the visible subset. If the diff is large,
   split the audit BY SUBSYSTEM (one o3 run per top-level dir touched) + one final integration run,
   and require each run to enumerate the files it actually read. Never accept a SHIP on a diff that
   exceeded the window unreviewed.
3. **Fix-panel** any RED / o3 finding: spawn fix-agents (own worktrees), re-verify, re-merge,
   re-audit. Use **loop-until-dry** for open-ended fixes (repeat until K dry rounds). Then advance to
   the next wave.
   **BUDGET / DoS GUARD (adversarial #13):** cap the loops — max fix-panel rounds (e.g. 3), max o3
   re-runs, max e2e retries, a per-task timeout. If a budget is hit (loop-until-dry never dries, a fat
   task re-hangs, o3 never reaches SHIP), STOP and escalate to the human rather than burn unbounded
   tokens/$/external-API calls.

## Step 5.5 — e2e + CI gate (LEARNED IN WAVE 0 — don't skip)
Steps 3–5 run `-m "not e2e and not skeleton"`. Before declaring a wave done you MUST also clear e2e,
skeleton, and CI — and TRIAGE failures (most are pre-existing, NOT yours):
1. **skeleton sentinel:** `pytest tests/skeletons/ -m skeleton`. Flaky asyncpg "another operation in
   progress" = cross-loop pooled connection → `DB_USE_NULLPOOL=1` in the test lane.
2. **e2e (real-deps), bisect per MODULE** (one file, serial — parallel hammers the shared DB and
   manufactures flakes): `for f in tests/e2e/test_e2e_*.py; do pytest "$f" -m "e2e and not paid"
   --timeout=120 ...; done`. Real-deps env = docker Redis (real, NOT fakeredis) + **real AWS S3**
   (do NOT source `.env`'s MinIO endpoints; conftest forces real AWS in the real-deps branch) + RDS
   on :5434 + real OpenAI/Mistral/AFIP. `E2E_REAL_DEPS=1`. **Timeout ≥120s** — SSE tests run a real
   30s keepalive and the in-process worker; `--timeout=30` FALSE-kills them.
3. **PROVE pre-existing vs regression** for every failure: run the failing module on a FRESH
   `origin/main` worktree + fresh migrated DB. If it fails identically on main → PRE-EXISTING →
   record in `BUGS.md`, do NOT fix on this wave. Only fix what THIS wave introduced.
4. **CI:** push, watch `gh pr checks`. `_derive_test_db_name()` reads the branch name and is
   non-deterministic under CI's detached HEAD → pin `TEST_DB_NAME` in any skeleton/test workflow.
5. **Merge to main is USER-ONLY:** `gh pr merge` is AI-blocked by a guard needing
   `merge-approved-<slug>.done`. For a NON-SDD wave branch finish_feature can't write it → the USER
   runs `gh pr merge <PR> --squash --admin`. Never bypass the guard yourself.

## Step 6 — Resume / re-run
Interrupted? If you used the Workflow engine: `Workflow({scriptPath, resumeFromRunId:"<runId>"})` —
cached agents return instantly, only the failed/new task re-runs (stop a stuck run with `TaskStop`
first). Otherwise re-run `--wave N` — already-merged tasks are detected by their merge commit.

## Step 7 — CLEANUP (per wave, after the wave-level audit is green)
```bash
.claude/skills/workflow-execute/scripts/worktree.sh down <task-slug>     # removes worktree + branch
# Tier B: ensure every stack slot was torn down (ui_e2e_stack.sh down <N>)
```

## Landmines (hard rules — from real incidents)
1. **Worktree-per-task is the isolation — same-file overlap is OK.** Do NOT force disjoint grouping.
   The master resolves overlap at merge. (This is the corrected core model.)
2. **Single merger.** Exactly ONE merger, into the FEATURE branch (never main), resolving conflicts —
   either the master by hand OR the in-workflow Step-4 merge-agent (one serial agent, see the 5-step
   model). The FANOUT (parallel build agents) never merges; build agents never push/merge. "Single
   merger" = single SERIAL writer, NOT "humans only" — a deterministic Step-4 loop is a valid merger.
3. **Two audit points.** Per-task verifier+audit gates each branch BEFORE merge; the wave-level full
   audit + codex-consult o3 reviews the merged whole AFTER all merges.
4. **Branch off feat, merge back to feat** — never main. The user pushes feat to origin.
5. **Tier→worktree mechanism.** Tier A worktrees via `worktree.sh` (or engine `isolation:'worktree'`);
   Tier B via `worktree.sh --ui` + Docker stack. Backend used NO Docker. **MAX_STACKS** caps Tier B.
6. **600s watchdog** is BACKGROUND-only — the Workflow engine path avoids it; hand-spawned build
   agents must run foreground.
7. **Known-baseline failures.** Audits flag only NEW reds.
8. **Harness merge-conflict auto-commit** — a `wip: ctx NN% snapshot` MERGE can land with `<<<<<<<`
   markers if a merge conflicts mid-context-save → `git reset --hard <last-clean-merge>` and re-resolve.
9. **marker_invalidate** wipes other slugs' markers — never `git add markers/` blindly; run the full
   skeleton sentinel before push.
10. **Migrations applied by Axel, not Claude.** Multiple migrations in one wave → alembic multi-head
    → author a merge migration; Axel applies.
11. **Fat tasks hang the engine.** A 3-in-1 task stalled mid-build and blocked the whole Workflow
    (pipeline awaits every item). Split fat tasks; if one straggles, run a per-task FINISHER workflow
    for just it (it can resume the partial worktree work) rather than restarting the wave.
12. **Wave-level codex o3 catches what per-task audits CANNOT** — cross-task seams only exist after
    merge: a §01 admin-gate locked out §02's new superadmin (`role!='admin'` → fix superadmin⊐admin
    via `is_superadmin`); a CSRF test seeded a member so a 403 came from the admin gate, not CSRF.
    Always run Step 5's o3 and re-run it until SHIP after fixes.
13. **The pydantic `settings` singleton is the recurring villain.** It's built at first import; setting
    `os.environ` AFTER does NOT update it. Bit CSRF_ENABLED, S3 endpoints, GEMINI keys. Fix pattern:
    mutate the singleton FIELD directly in conftest — and SCOPE it to the right lane (e.g. CSRF off in
    NON-real-deps only; disabling it globally broke the real-deps e2e CSRF gate, 200 not 403).
14. **Most e2e "failures" are PRE-EXISTING, not yours.** Prove it: run the failing module on a fresh
    `origin/main` worktree+DB. If it fails identically → BUGS.md, not a wave fix. (13 of 14 in Wave 0.)
15. **NEVER let parallel tasks author migrations independently (the Wave-1 disaster).** 8 tasks each
    wrote a migration: 6 hard-coded the SAME placeholder revision id, 3 pairs RE-CREATED the same
    table/column (worktrees are isolated — agents can't see each other). The user's "shared
    migrations.md read-write-read" CANNOT work (cross-worktree writes are invisible until merge + a
    merge-HEAD race). FIX (Step 0.5): split a migration wave into `phase('schema')` (DDL owners,
    deterministic slug-derived revision ids, barrier) → reconciler → `phase('consume')` (no migrations
    allowed). `Needs-DDL:` tags + the one-owner rule + `scripts/migration_reconcile.py`. A `consume`
    task committing a `migrations/versions/` file is a HARD error. See
    `docs/researches/wave-migration-redesign.md` + `wave-reorg-codex-o3.md`.
16. **Collapse to 3 waves: 0 (done), 1 = Schema + Primitives, 2 = Consumers.** The wave boundary now
    means "schema is frozen" — DDL producers in Wave 1, pure consumers in Wave 2 (expand/contract).
17. **INTRA-WAVE RUNTIME DEPENDENCY = a build-base trap (Wave-2 first_test, 2026-06-12).** When B
    `depends_on` A and BOTH are in the same wave, B's worktree is branched off the wave's START commit —
    so B CANNOT consume A's unmerged runtime output (a queue/endpoint/service fn). In Wave 2,
    `f2-approve-with-role depends_on f1-approval-queue`: f2 built off a base without f1's queue, produced
    no working consumer, and the audit DO-NOT-SHIP'd the wave (f1's queue was left INERT — pending users
    still logged in, because f2's login gate never landed). The PLANNER should prevent this (re-wave or
    merge the cards — see workflow-plan Step 7 "runtime-dependency rule"). If it reaches the runner
    anyway: **branch the dependent's worktree off the dependency's MERGED result, not the wave start** —
    i.e. merge A first, THEN create B's worktree off the updated feat (the schema-phase barrier already
    does this for schema deps; do it manually for a runtime dep). On re-run, tear down B's stale worktree
    and recreate it off current feat. SCHEMA deps are safe in-wave (the barrier handles them); RUNTIME
    deps are not.
18. **THE AUDIT (Step 5) EARNS ITS KEEP on the inert-half class.** The Wave-2 o3 audit caught that f1
    shipped without its consumer f2 → the approval feature moderated nothing. No per-task review can see
    this — f1 alone passes every per-task gate. Only the whole-wave audit, asking "does the FEATURE work
    end to end", catches a producer landed without its required enforcement point. Always run it; a
    green per-task panel is NOT a working feature.

## Canonical workflow shape — the 5-STEP model (MERGE is Step 4, INSIDE the script)
**The old rule "the workflow must NEVER merge" was only ever true of the FANOUT, never the SCRIPT.**
"Workflow" means two different things — separate them:
- **The FANOUT** = `pipeline()`/`parallel()` of `agent()` calls. Parallel LLM agents. NEVER merge here
  (git is a single-writer — concurrent merges collide on `.git/index.lock`).
- **The SCRIPT** = the plain deterministic JavaScript that CONTAINS the fanout. After `await pipeline(...)`
  RETURNS, you are back in ordinary sequential JS — and **the next lines CAN merge. That is Step 4.**

So ONE Workflow run does all five steps (see `templates/wave.workflow.js` — the worked reference):
```
Step 1 build         ┐
Step 2 verify+audit  ├─ inside pipeline()  (the FANOUT — never merge here)
Step 3 codex review  ┘  (per-task: run codex-consult run.py on the branch diff → mergeReady)
── pipeline() RETURNS → plain sequential JS ──
Step 4 MERGE         deterministic JS loop → ONE merge-agent w/ Bash. SERIAL, IDEMPOTENT
                     (skip already-`is-ancestor` branches), POST-CONDITION-VERIFIED (after each
                     merge, ASSERT the branch is now an ancestor of feat — else STOP). HARD
                     conflict → the merge-agent ABORTS and returns it for the human (never force).
Step 5 big codex     o3 reviews the WHOLE merged wave diff (git diff base..feat) — the cross-task
                     seams per-task review CANNOT see.
```
**Why Step 4 belongs in the workflow (the first_test proof):** a hand-rolled merge OUTSIDE the workflow
silently dropped **10 of 14 branches** (a bash `for…; grep -q CONFLICT` loop skipped them with no error,
caught only by accident). The Step-4 `is-ancestor` post-condition assert catches that immediately. And the
Step-5 big audit caught a HIGH + MED cross-task bug per-task review missed — it must run EVERY wave, which
only happens reliably if it's a workflow step, not a thing the master remembers to do.

```js
// CORRECT — fanout (1-3) returns, THEN Step 4 merge, THEN Step 5 audit — all one run.
const reviews = (await pipeline(TASKS,
  (t) => agent(buildPrompt(t),  {phase:'W0:build',  schema:VERDICT}),
  (b) => agent(verifyPrompt(b), {phase:'W0:verify', schema:VERDICT}),
  (v) => agent(reviewPrompt(v), {phase:'W0:review', schema:REVIEW}),   // Step 3: per-task codex
)).filter(Boolean)
phase('W0:merge')                                                       // ── back in sequential JS ──
for (const br of mergeOrder(reviews)) {                                 // Step 4: SERIAL merge
  const m = await agent(mergePrompt(br), {phase:'W0:merge', schema:MERGE_RESULT})
  if (m.hardConflict || !m.isAncestor) break                           // STOP — human or failed post-cond
}
phase('W0:audit')
const audit = await agent(bigAuditPrompt(), {phase:'W0:audit', schema:AUDIT})  // Step 5: big o3
// The workflow merges into FEAT only (never main); it RETURNS the audit verdict for the human's push/PR call.
```
The single-merger rule is HONORED — exactly one serial merge-agent, never the fanout, never main.
`resumeFromRunId` makes Step 4 safe to resume (cached merges = no-op; the is-ancestor check is idempotent).

### Tiers — the template handles BOTH via `args.tier` (you don't write a per-tier script)
`templates/wave.workflow.js` implements both patterns; pick the tier in `args.tier`:
- **Tier A — `simple`** (backend/logic): per-task ritual-build → verify → review → merge → audit, NO
  Docker, `pytest` verify. The default. Historical reference patterns:
  `docs/specs/_panel-2026-06-08/PARALLELIZATION.md`, `plans/first_test_pipeline.workflow.js`.
- **Tier B — `ui`** (real-stack browser-e2e): same 5 steps + a per-worktree `ui_e2e_stack.sh` Docker
  stack (MAX_STACKS cap) + Playwright verify. Historical reference: `docs/ui-audit/HANDOFF.md`.
These are REFERENCES for what the template does — you do not re-implement them per wave.

## Reference files
- Toolset rationale: `docs/researches/workflow-edit-toolset.md`. Recovered pattern:
  `docs/researches/workflow-edit-pattern.md`.
- Working workflow examples: `plans/wave0_build.workflow.js` (build→verify+audit pipeline),
  `plans/first_test_pipeline.workflow.js`, `plans/first_test_docfix_synth.workflow.js`.
  Template: `templates/wave.workflow.js`.
- Originals: `docs/ui-audit/HANDOFF.md` (UI tier), `docs/specs/_panel-2026-06-08/PARALLELIZATION.md`
  (backend tier), `scripts/ui_e2e_stack.sh`, `.claude/skills/codex-consult/scripts/run.py`,
  `.claude/skills/codex-adversarial/scripts/run.py`, `.claude/skills/workflow-execute/scripts/worktree.sh`.
