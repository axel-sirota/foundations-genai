# /audit — Audit a Day 3 Research Plan File

Audits a research plan markdown file (in `researches/`) before building notebooks.
This is a READ-ONLY command — it reports issues, never edits.

## Usage

```
/audit researches/topic_7a_lora_ffn.md
/audit researches/topic_7b_peft_lora_distilbert.md
/audit researches/topic_8_quantization.md
```

---

## GUARD: Read Core Decisions First

Before auditing, always read:
- `plans/CORE_TECHNOLOGIES_AND_DECISIONS.md`
- `plans/SAGEMAKER_LESSONS_LEARNED.md`

---

## Complete Checklist (check every item)

### 1. Four-Beat Arc (B1→B2→B3→B4, mandatory per section)
For every major section:
- **Beat 1**: broken/naive code that runs and fails (students feel the pain)
- **Beat 2**: standalone markdown cell with `<!-- DIAGRAM: -->` placeholder OR prose anchor — must come AFTER B1, BEFORE B3
- **Beat 3**: full working demo, heavily commented
- **Beat 4**: lab markdown + starter code cell

Flag any section where a beat is missing OR the order is wrong (e.g. B2 before B1, B3 before B2).

### 2. Diagrams
- Exactly **2** diagram placeholders per notebook — no more, no less
- Each must be planned as a **markdown cell** (never inside print() or a code cell)
- Each must appear as **Beat 2** — before the Beat 3 working demo it introduces
- Format: `<!-- DIAGRAM: description -->` + `[View diagram](../../plans/topic_N_slug/diagrams/slug.mmd)`
- All referenced `.mmd` files must **exist on disk** — check `plans/topic_N_slug/diagrams/`
- Diagram paths must use **full folder names** (e.g. `topic_7a_lora_ffn`, NOT `topic_7a`)

### 3. Safety-nets
- Every lab variable that feeds a downstream cell must have a safety-net cell **immediately after** the lab starter
- Format: `if variable is None: variable = <working implementation>`
- `training_job_name` safety-net must be planned **after every `.fit(wait=False)` call**
- Safety-nets planned for **both** exercise and solution notebooks
- Safety-nets must **NOT be removed** from solution notebooks

### 4. AI-tells (zero tolerance)
- No em dashes (— U+2014)
- No en dashes (– U+2013)
- No Unicode multiplication sign (× U+00D7)
- No emojis
- No bare `---` separator lines in any planned markdown cell body

### 5. Lab Tiers (Day rules — apply the correct day's rules)
**Per day across all topics that day:**
- Exactly **ONE Tier 2** lab (multi-step, less prescriptive, 25-35 min)
- Exactly **ONE Tier 3** lab — in the **LAST topic of that day only**
- All other labs: **Tier 1** (numbered steps + `= None  # YOUR CODE` + verification cell)

**Tier definitions:**
- Tier 1: numbered steps + YOUR CODE stubs + verification cell. ~15-20 min.
- Tier 2: brief task description, no numbered sub-steps, YOUR CODE stubs. ~25-35 min.
- Tier 3: function signature + docstring + `pass` only. No hints. No steps. Last topic of day only.

**Every lab** (all tiers) must have:
- A stretch version for fast finishers
- A Homework Extension markdown cell + starter code cell after it

### 6. YOUR CODE hygiene
- Stub line: `variable = None  # YOUR CODE` — **no inline hints**
- Hints go in numbered-step **comment lines above** the stub
- Never: `= None  # YOUR CODE: F.softmax(...)` or `= None  # YOUR CODE: call prune.global_unstructured(...)`

### 7. Narrative continuity (Barclays STAR)
- All labs framed as **Barclays Customer Support Intelligence System**
- STAR method: Situation (Barclays context) → Task (what to build) → Action (YOUR CODE) → Result (verification)
- Variable names must carry over **exactly** from the prior topic notebook
- A "Variable Continuity from Topic N" section must exist documenting what carries in
- **No "Day N" references** in planned notebook header/title cells (no "Day 1", "Day 2", "Day 3")

### 8. SageMaker constraints (for any topic using SageMaker)
- `numpy<2` pinned in every install cell
- **No `evaluate` library** — use inline metric functions only
- `eval_strategy="epoch"` (NOT the deprecated `evaluation_strategy=`)
- `wait=False` on **all** `.fit()` calls
- `requirements.txt` named **exactly** "requirements.txt" in source_dir
- Scripts folder planned **inside** the topic exercise folder: `Exercises/topic_N_slug/scripts_topic_N/`
- Scripts folder also mirrored in `Solutions/topic_N_slug/scripts_topic_N/`
- **HuggingFace estimator** = GPU only (`ml.g4dn.xlarge`)
- **PyTorch estimator** = CPU only (`ml.m5.xlarge`)
- `mlflow==2.13.2` only if mlflow is used (F2/sagemaker_fundamentals only — not in topic notebooks)
- `boto3` exception: `ResourceNotFound` NOT `ResourceNotFoundException`
- Version pins: `sagemaker>=2.200.0,<3.0.0`, `transformers>=4.35.0,<4.40.0`, `tokenizers>=0.15.0,<0.20.0`
- `datasets==2.18.0` pinned in scripts `requirements.txt` when datasets is used
- `transformers_version="4.56.2"`, `pytorch_version="2.8.0"`, `py_version="py312"` in estimator

### 9. Discussion prompts
- At least **one peer discussion markdown cell** between every two major sections
- Must be a **markdown cell** — never print() inside a code cell
- **Barclays-framed** questions: tradeoffs, production consequences, "what if"
- 3-5 questions, 3-5 min

### 10. Homework Extensions
- Every lab must have a **Homework Extension** markdown cell + starter code cell **after** it

### 11. Cell count and markdown chains
- Estimate total planned cell count (target 35-55 cells per notebook)
- No more than **3 consecutive markdown cells** without a code cell between them
- Exercise and solution notebooks must have the **same planned cell count**

### 12. Cross-topic tier distribution
- Confirm tier counts across ALL topics in the same day
- Exactly 1 Tier 2 total, exactly 1 Tier 3 total (last topic only)

---

## Output Format

```
# Audit Report: [topic slug]
Generated: [date]

## Status: PASS / FAIL — [one line summary]

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | PASS/FAIL/WARN |
| Diagrams (count=2, markdown, Beat 2) | PASS/FAIL/WARN |
| Safety-nets | PASS/FAIL/WARN |
| AI-tells | PASS/FAIL/WARN |
| Lab tiers | PASS/FAIL/WARN |
| YOUR CODE hygiene | PASS/FAIL/WARN |
| Narrative / Barclays STAR | PASS/FAIL/WARN |
| SageMaker constraints | PASS/FAIL/WARN |
| Discussion prompts | PASS/FAIL/WARN |
| Homework extensions | PASS/FAIL/WARN |
| Cell count / markdown chains | PASS/FAIL/WARN |
| Cross-topic tier distribution | PASS/FAIL/WARN |

### Issues to Fix
[List every FAIL and WARN with specific section/cell reference and exact fix needed]

### MANDATORY Before Every Fix
1. Read the full file to understand context
2. Identify exact location (line number or section header)
3. Never assume structure — re-read if in doubt
```

Save the report to `plans/AUDIT_[TOPIC].md`.

---

## Non-Negotiables
- Never skip a check because a section looks fine at a glance — check everything
- Report every failure with the specific section or line reference
- Do NOT auto-fix anything — this command is read-only
- If a fix is needed, the fix must go through a dedicated fix agent or `/fixes`

---

## Notebook Edit Protocol (awareness)

If this skill ends up editing notebook cells (not just reading them), follow
the canonical procedure in `~/.claude/NOTEBOOK_EDIT_PROTOCOL.md`: normalize
cell ids, size-gate the mechanism, locate cells by id + content, read back and
assert after every edit, and run the structural + static code gates. Blind
bulk index-based rewrites are forbidden.
