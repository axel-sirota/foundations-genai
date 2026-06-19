#!/usr/bin/env python3
"""Generate TRANSITION.md into each Wave-0 worktree.

Each transition file is the atomic agent's complete, autonomous brief: the task text
extracted from the section plan.md + the files to touch + context pointers + the hard
rules (no SDD, no markers, no human approval, TDD-only, commit-don't-merge).
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path("/Users/axelsirota/repos/pocs/ticket-ocr")
BASE = ROOT / "docs/mvp/first_test"
WT_PARENT = ROOT.parent

# task-slug -> (section-dir, [task-id-prefixes to extract from plan.md])
TASKS = {
    "w0-01t1": ("01-auth-onboarding-approval", ["T1 "]),
    "w0-01t2": ("01-auth-onboarding-approval", ["T2 "]),
    "w0-02t1": ("02-superadmin-console", ["T1 "]),
    "w0-02t4": ("02-superadmin-console", ["T4 "]),
    "w0-03t0": ("03-home-dashboard", ["T0 "]),
    "w0-03t6": ("03-home-dashboard", ["T6 "]),
    "w0-06t1": ("06-upload-flow", ["T1 "]),
    "w0-06t2": ("06-upload-flow", ["T2 "]),
    "w0-05t1": ("05-sidebar-seats", ["T1 "]),
    "w0-04t1": ("04-company-selection-workspace", ["T1 "]),
    "w0-04t2": ("04-company-selection-workspace", ["T2 "]),
    "w0-04t3": ("04-company-selection-workspace", ["T3 "]),
    "w0-04t11": ("04-company-selection-workspace", ["T11 "]),
    "w0-08t1": ("08-review-workstation", ["T1 ", "T2 ", "T11 "]),
}

RULES = """\
## Hard rules (read first — you are an ATOMIC, AUTONOMOUS task agent)
- You own exactly ONE functionality (this task). You are **NOT in the SDD / kryla pipeline.**
- Do **NOT** run any `/kryla.*` step. Do **NOT** create, touch, or `git add` anything under
  `.claude/markers/`. The override marker is already laid for you.
- **Do NOT ask for human approval. Do NOT pause for confirmation or clarification.** Everything you
  need is in THIS file — the task already went through the full research → code-map → codex-audit
  pipeline. There is nothing left to clarify. Execute end-to-end and finish.
- Develop **TDD-style**: write the failing test(s) for THIS task first (red), then implement until
  they pass (green). Honor the repo Test Constitution layers relevant to your change (unit / HP /
  UP / smoke / ui / e2e as the task implies).
- Run tests with the venv: `.venv/bin/python3 -m pytest ...` (symlinked in this worktree).
- **Commit** your work on THIS branch with `ALLOW_DIRECT_MAIN=1 git commit` (branch_guard
  false-fires inside worktrees). Do **NOT** push. Do **NOT** merge — the master merges your branch.
- Stay within your task's files where possible; if you must touch a shared file, that's fine — the
  master resolves overlap at merge. Do not refactor unrelated code.
- When done, report: green/red, the branch name, the list of files you changed, and any note the
  master needs for merging (especially shared-file edits).
"""


def extract_task(plan_text: str, prefixes: list[str]) -> str:
    """Pull the block(s) for the given task IDs out of plan.md."""
    out = []
    lines = plan_text.splitlines()
    # Tasks are headed by lines like '**T1 — ...** | ...' or '## T0 — ...'
    i = 0
    while i < len(lines):
        line = lines[i]
        # a task head matches the prefix as: **T1 — / ### T1 — / ## T1 — / - **T1 —
        def _is_task_head(text: str, p: str) -> bool:
            t = text.strip()
            pp = re.escape(p.strip())
            return bool(
                re.match(rf"^\*\*{pp}\b", t)
                or re.match(rf"^#+\s*{pp}\b", t)
                or re.match(rf"^-\s*\*\*{pp}\b", t)
            )

        # ANY task head (to detect the boundary to the NEXT task)
        def _is_any_task_head(text: str) -> bool:
            t = text.strip()
            return bool(
                re.match(r"^\*\*T\d+\b", t)
                or re.match(r"^#+\s*T\d+\b", t)
                or re.match(r"^-\s*\*\*T\d+\b", t)
            )

        is_head = any(_is_task_head(line, p) for p in prefixes)
        if is_head:
            block = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                # stop at the next task head (that is NOT one of our wanted prefixes) or a top section
                if (_is_any_task_head(nxt) and not any(_is_task_head(nxt, p) for p in prefixes)) \
                   or nxt.strip().startswith("## Codex audit deltas") \
                   or re.match(r"^##\s+(Test|Cross|Open|Summary)", nxt.strip()):
                    break
                block.append(nxt)
                i += 1
            out.append("\n".join(block).rstrip())
        else:
            i += 1
    return "\n\n".join(out) if out else "(task block not auto-extracted — read the section plan.md directly)"


def main() -> int:
    repo = ROOT.name
    written = 0
    for slug, (sec, prefixes) in TASKS.items():
        wt = WT_PARENT / f"{repo}-{slug}"
        if not wt.exists():
            print(f"!! worktree missing for {slug}: {wt}", file=sys.stderr)
            continue
        plan = (BASE / sec / "plan.md").read_text(encoding="utf-8")
        task_block = extract_task(plan, prefixes)
        content = f"""# TRANSITION — {slug}  (Wave 0, section {sec})

You are the build agent for **{slug}**. Work ONLY inside this worktree:
`{wt}` (branch `wf/first-mvp-test-doc/{slug}`).

{RULES}

## Context to read (in this worktree — paths are repo-relative)
- `docs/mvp/first_test/{sec}/plan.md` — your section's full plan (your task block is below).
- `docs/mvp/first_test/{sec}/index_of_code_and_test.md` — behaviour → code → tests map (file:line).
- `docs/mvp/first_test/{sec}/codex_audit.md` — the adversarial audit; honor its corrections.
- `docs/mvp/first_test/FIRST_MVP_TEST.md` — the original finding.
- Repo roots: backend `apps/api/` + `packages/core/`; frontend `apps/web-next/app/`; tests
  `tests/{{unit,integration,smoke,ui,e2e,skeletons}}/` (TS under `apps/web-next/tests/`).

## YOUR TASK (atomic — verbatim from the codex-audited plan)
{task_block}

## Definition of done
The failing test(s) you write for this task pass (green), the task's behaviour is implemented as
specified above, the relevant Test Constitution layers are covered, and you have committed on this
branch. Then report back to the master. Do not push, do not merge, do not ask for approval.
"""
        (wt / "TRANSITION.md").write_text(content, encoding="utf-8")
        written += 1
        print(f"✓ {slug}: TRANSITION.md ({len(task_block)} chars of task text)")
    print(f"\n{written}/{len(TASKS)} transition files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
