// POINTER — not the live script. The canonical, runnable wave workflow is:
//     .claude/workflows/wave-execute.js
// which is resolvable by NAME:
//     Workflow({ name: 'wave-execute', args: { plan, feat, base, epicDir, tier, fullRitual } })
//
// where `plan` is the .plan object from:
//     python3 .claude/skills/workflow-execute/scripts/load_wave.py docs/<epic>/plan --wave N --json
//
// You do NOT author or edit a per-wave script. ONE workflow, args change per wave. See the skill's
// "Step 2 — Run the wave via the TEMPLATE" for the exact call. This file exists only as a signpost so
// older references to templates/wave.workflow.js still lead you to the real one.
//
// (Kept as a comment-only file to avoid two diverging copies of the 140-line script.)
export const meta = {
  name: 'wave.workflow (pointer)',
  description: 'POINTER → .claude/workflows/wave-execute.js (run by name). Not a live script.',
  phases: [],
}
throw new Error('templates/wave.workflow.js is a POINTER — run the canonical workflow by name instead: Workflow({ name: "wave-execute", args: {...} }). See .claude/workflows/wave-execute.js')
