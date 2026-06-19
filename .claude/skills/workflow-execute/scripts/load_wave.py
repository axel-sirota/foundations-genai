#!/usr/bin/env python3
"""load_wave.py — the RUNNER's card reader (the contract with /workflow-plan).

o3 review #6: the runner skill DESCRIBED reading `plan/tasks/*.card.yaml` but no script did it — the
helpers still expected a prose INDEX.md, so `phase`, `shared_resources`, `depends_on`, and the
planner's ordering edges were never actually consumed. This script makes the data-only seam REAL: it
loads the cards for one wave and emits the structured execution plan the master drives from.

It deliberately REUSES the planner's card loader (`ddl_plan.py:_load_cards`) so the runner and planner
parse the identical schema — there is exactly one parser, not two that can drift.

Emits (text by default, --json for machine):
  - schema_tasks / consume_tasks      : split by card `phase`
  - lock_groups                       : {test_db:[ids], s3:[...], redis:[...]} from shared_resources
                                        (tasks sharing a lock must NOT run concurrently → serialize)
  - ordering_edges                    : schema-phase build order (predecessor → dependent), read from
                                        plan/ddl_conflict_report.md (the planner's output)
  - overlaps                          : files_touched ∩ across tasks (where merge conflicts surface)
  - merge_order                       : a topological order of depends_on (master merges in this order)
  - expect_migrations                 : bool — any schema task? → feeds migration_reconcile.py
                                        --expect-migrations (fail-closed if the schema phase produced 0)

Usage:  load_wave.py <plan-dir> --wave N [--json]
Exit:   0 ok · 2 the plan is unsafe to execute (HARD CONFLICT unresolved, or a schema/consume
        integrity violation — a consume card still carries ddl_intents) · 3 bad input.

A non-zero exit means /workflow-execute must NOT start the wave — the plan is broken, fix it in
/workflow-plan first. This is the gate that stops a bad plan from reaching worktrees.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the planner's ONE card loader so planner+runner never drift on the schema.
_PLAN_DDL = Path(__file__).resolve().parents[2] / "workflow-plan" / "scripts" / "ddl_plan.py"


def _load_ddl_module():
    spec = importlib.util.spec_from_file_location("ddl_plan", _PLAN_DDL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load planner card loader at {_PLAN_DDL}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wave_of(card: dict) -> int | None:
    w = card.get("wave")
    if w is None:
        return None
    try:
        return int(w)
    except (TypeError, ValueError):
        m = re.search(r"\d+", str(w))
        return int(m.group()) if m else None


def _toposort(ids: list[str], deps: dict[str, list[str]]) -> list[str]:
    """Stable topo order of ids by deps (depends_on). Cycles → the remaining nodes appended in id
    order (the planner's human-curated waves already broke real cycles; this is a safety net)."""
    order: list[str] = []
    remaining = set(ids)
    while remaining:
        ready = sorted(i for i in remaining if all(d not in remaining for d in deps.get(i, [])))
        if not ready:  # cycle — append the rest deterministically
            order.extend(sorted(remaining))
            break
        order.extend(ready)
        remaining -= set(ready)
    return order


def _ordering_edges(plan_dir: Path, wave_objects: set[str] | None = None) -> list[tuple[str, str]]:
    """Parse the planner's ddl_conflict_report.md 'Ordering edges' section (predecessor → dependent).
    The report is GLOBAL (all waves); pass `wave_objects` (the DDL objects this wave's cards touch) to
    keep only edges where the DEPENDENT belongs to this wave — otherwise a later wave's edge leaks into
    an earlier wave's plan."""
    report = plan_dir / "ddl_conflict_report.md"
    edges: list[tuple[str, str]] = []
    if not report.exists():
        return edges
    in_section = False
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Ordering edges"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            m = re.match(r"\s*-\s*(.+?)\s*→\s*(.+?)\s*$", line)
            if m:
                pred, dep = m.group(1).strip(), m.group(2).strip()
                if wave_objects is None or dep in wave_objects or pred in wave_objects:
                    edges.append((pred, dep))
    return edges


def build_plan(plan_dir: Path, wave: int) -> tuple[dict, list[str]]:
    """Return (execution_plan, problems). problems non-empty ⇒ the wave is unsafe to run."""
    ddl = _load_ddl_module()
    tasks_dir = plan_dir / "tasks"
    if not tasks_dir.is_dir():
        return {}, [f"no tasks dir at {tasks_dir}"]
    try:
        all_cards = ddl._load_cards(tasks_dir)
    except ddl.CardParseError as e:
        return {}, [f"card parse failed (fail-closed): {e}"]

    cards = [c for c in all_cards if _wave_of(c) == wave]
    problems: list[str] = []
    if not cards:
        return {}, [
            f"no cards for wave {wave} (found waves: "
            f"{sorted({_wave_of(c) for c in all_cards if _wave_of(c) is not None})})"
        ]

    def cid(c):
        return ddl._cid(c)

    schema_tasks, consume_tasks = [], []
    for c in cards:
        phase = str(c.get("phase", "consume")).strip()
        (schema_tasks if phase == "schema" else consume_tasks).append(cid(c))
        # INTEGRITY: a consume card must NOT carry ddl_intents (the planner should have folded them).
        if phase != "schema" and ddl._intents(c):
            problems.append(
                f"INTEGRITY: card {cid(c)} is phase=consume but still declares ddl_intents "
                f"{[i.get('object') for i in ddl._intents(c)]} — re-run `ddl_plan.py --apply` to "
                f"fold it onto a schema-owner before executing."
            )

    # If the planner left a HARD CONFLICT in the report, the wave is not safe to run.
    report = plan_dir / "ddl_conflict_report.md"
    if report.exists() and "HARD CONFLICT" in report.read_text(encoding="utf-8"):
        problems.append(
            "HARD CONFLICT present in ddl_conflict_report.md — the human must reconcile it in "
            "/workflow-plan (Gate 2) before /workflow-execute runs this wave."
        )

    lock_groups: dict[str, list[str]] = defaultdict(list)
    deps: dict[str, list[str]] = {}
    files_by_task: dict[str, list[str]] = {}
    for c in cards:
        i = cid(c)
        for lock in c.get("shared_resources") or []:
            lock = str(lock).strip()
            if lock and lock != "none":
                lock_groups[lock].append(i)
        deps[i] = [str(d).strip() for d in (c.get("depends_on") or [])]
        files_by_task[i] = [str(f).strip() for f in (c.get("files_touched") or [])]

    # overlaps: which files ≥2 tasks touch (merge-conflict sites the master must resolve)
    file_owners: dict[str, list[str]] = defaultdict(list)
    for t, fs in files_by_task.items():
        for f in fs:
            file_owners[f].append(t)
    overlaps = {f: sorted(ts) for f, ts in file_owners.items() if len(ts) > 1}

    # the DDL objects this wave's cards touch — to filter the GLOBAL ordering-edge report to this wave
    wave_objects = {
        str(it.get("object", "")).strip()
        for c in cards
        for it in ddl._intents(c)
        if str(it.get("object", "")).strip()
    }

    plan = {
        "wave": wave,
        "schema_tasks": sorted(schema_tasks),
        "consume_tasks": sorted(consume_tasks),
        "expect_migrations": bool(schema_tasks),
        "lock_groups": {k: sorted(v) for k, v in lock_groups.items() if len(v) > 1},
        "ordering_edges": _ordering_edges(plan_dir, wave_objects),
        "overlaps": overlaps,
        "merge_order": _toposort([cid(c) for c in cards], deps),
        "dod_tests": {cid(c): (c.get("dod_tests") or []) for c in cards},
    }
    return plan, problems


def _print_text(plan: dict, problems: list[str]) -> None:
    print(f"# Wave {plan.get('wave')} execution plan (from the cards)\n")
    if problems:
        print("## ❌ UNSAFE TO RUN — fix in /workflow-plan first")
        for p in problems:
            print(f"- {p}")
        print()
    print(f"schema-phase tasks ({len(plan.get('schema_tasks', []))}): {plan.get('schema_tasks')}")
    print(
        f"consume-phase tasks ({len(plan.get('consume_tasks', []))}): {plan.get('consume_tasks')}"
    )
    print(f"expect_migrations (→ reconciler --expect-migrations): {plan.get('expect_migrations')}")
    if plan.get("lock_groups"):
        print("\nshared-resource lock groups (SERIALIZE within these — do NOT run concurrently):")
        for lock, ids in plan["lock_groups"].items():
            print(f"  {lock}: {ids}")
    if plan.get("ordering_edges"):
        print("\nschema-phase ordering edges (build predecessor BEFORE dependent):")
        for a, b in plan["ordering_edges"]:
            print(f"  {a}  →  {b}")
    if plan.get("overlaps"):
        print("\nfile overlaps (master resolves these at merge):")
        for f, ts in plan["overlaps"].items():
            print(f"  {f}: {ts}")
    print(f"\nmerge order (depends_on topo): {plan.get('merge_order')}")


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    as_json = "--json" in argv
    wave = None
    if "--wave" in argv:
        i = argv.index("--wave")
        if i + 1 < len(argv):
            try:
                wave = int(argv[i + 1])
            except ValueError:
                wave = None
    if not args or wave is None:
        print("usage: load_wave.py <plan-dir> --wave N [--json]", file=sys.stderr)
        return 3
    plan_dir = Path(args[0])
    # accept either the plan/ dir or its parent (docs/<epic>/)
    if not (plan_dir / "tasks").is_dir() and (plan_dir / "plan" / "tasks").is_dir():
        plan_dir = plan_dir / "plan"
    if not plan_dir.is_dir():
        print(f"ERROR: {plan_dir} is not a plan directory", file=sys.stderr)
        return 3

    plan, problems = build_plan(plan_dir, wave)
    if not plan:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return 3
    if as_json:
        print(json.dumps({"plan": plan, "problems": problems}, indent=2))
    else:
        _print_text(plan, problems)
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
