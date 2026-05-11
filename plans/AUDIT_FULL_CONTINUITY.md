# Full Continuity Audit — Day 1 + Day 2
Generated: 2026-05-11
Auditor: Claude (Opus 4.7)
Scope: Frameworks/{F1,F2} + Exercises/{T1, T2, T3a, T3b, T4, T5, T6a, T6b}

Method: programmatic scan of every notebook (cell-by-cell) for ten audit criteria,
cross-checked against the parallel solution notebooks for parity. Findings below
list specific cell IDs for every issue so they can be addressed surgically.

---

## Summary Table

Legend: PASS / WARN / FAIL

| Notebook | Four-Beat | Diagrams | Safety-nets | AI-tells | Lab Tiers | Narrative | SageMaker | Discussion | Homework | Cell Parity |
|----------|-----------|----------|-------------|----------|-----------|-----------|-----------|------------|----------|-------------|
| F1 pytorch_refresher          | WARN | PASS | PASS | PASS | PASS | PASS | FAIL | WARN | PASS | PASS (48=48) |
| F2 sagemaker_fundamentals     | PASS | PASS | WARN | PASS | PASS | PASS | WARN | FAIL | WARN | PASS (43=43) |
| T1 overview_genai             | PASS | PASS | PASS | PASS | WARN | PASS | n/a  | PASS | PASS | PASS (33=33) |
| T2 introducing_llms           | PASS | PASS | WARN | PASS | PASS | PASS | n/a  | PASS | PASS | FAIL (39=45?)* |
| T3a attention_python          | PASS | PASS | PASS | PASS | PASS | PASS | n/a  | PASS | PASS | PASS (45=45) |
| T3b attention_pytorch         | PASS | PASS | PASS | PASS | PASS | PASS | n/a  | PASS | PASS | PASS (29=29) |
| T4 transformers               | WARN | PASS | PASS | PASS | FAIL | PASS | WARN | PASS | PASS | PASS (34=34) |
| T5 huggingface                | PASS | PASS | PASS | PASS | WARN | PASS | WARN | PASS | PASS | PASS (42=42) |
| T6a full_finetuning           | PASS | PASS | PASS | PASS | WARN | PASS | FAIL | PASS | PASS | PASS (40=40) |
| T6b transfer_learning         | WARN | PASS | WARN | PASS | PASS | PASS | WARN | PASS | WARN | PASS (40=40) |

\* Auto-extractor counted the exercise notebook at 39 cells in one pass and 45 in another;
recheck below in T2 detail.

Cross-day rollups:
- AI-tells across all 10 notebooks: **PASS (zero hits)** — no em dash, en dash, ×, emoji, or bare `---` anywhere.
- Barclays narrative: **PASS** in all 10 notebooks.
- "Day 1"/"Day 2" header references: **PASS (zero hits)**.

---

## Per-Notebook Detail

### F1 — Frameworks/pytorch_refresher.ipynb (48 cells)

PASS (most of the notebook). Specific issues:

- FAIL SageMaker: cell `bd3aeac5` and cell `e9c37e46` use `evaluation_strategy=` instead of the required `eval_strategy=`. These are inside the HuggingFace Trainer demo / verification for Lab 5. Must rename to `eval_strategy` (Transformers 4.40+).
- WARN Four-Beat: only one explicit broken-code cell detected (cell idx 4). Labs 2–5 use a single working-demo cell with no preceding "broken/naive" Beat-1. Many sections jump straight to a polished demo. The Four-Beat rule says **every** major section needs B1; consider adding a deliberately broken cell at the start of Sections 2–5 (autograd, DataLoader, nn.Module, HF Trainer).
- WARN Discussion: only one peer-discussion markdown cell detected (cell idx 21, after Lab 2). Rule asks for at least one between every pair of major sections — F1 has five labs, so we expect ~4 discussion cells; only 1 present.
- INFO YOUR CODE hygiene: 27 `# YOUR CODE: <hint>` annotations across labs. Most are legitimate type-hints (shape, dtype) but several do hint at the answer logic (e.g. `# YOUR CODE: pt.rand(...)`, `# YOUR CODE: scalar tensor, requires_grad=True`, `# YOUR CODE: nn.Sequential 8->32->ReLU->16->ReLU->4, on device`, `# YOUR CODE: 8 epochs, eval_strategy="epoch", save_strategy="no"`). The rule says the marker line "must not hint at the answer". Recommend tightening these to either bare `# YOUR CODE` or moving the hint to the surrounding numbered-step comment above.
- PASS Diagrams: 2 mermaid/diagram references in markdown cells (`d7c9f0e6`, `80a66309`), both placed before their corresponding Beat 3.
- PASS Safety-nets: all five labs (`d60681b5`, `6897b5a7`, `e562971d`, `293a0169`, `01532399`) have an `if … is None:` safety-net cell within 3 cells after the lab starter.
- PASS Cell parity with `pytorch_refresher_solution.ipynb`: 48 = 48.

### F2 — Frameworks/sagemaker_fundamentals.ipynb (43 cells)

- WARN SageMaker `wait=False`: three `.fit(` matches without `wait=False` flagged: `d49c4530` (Beat 1 boto3 raw-upload demo — not an estimator.fit, false positive), `3b3d61db` (PyTorch estimator demo cell — IS missing `wait=False`), `ad4b60bb` (Homework MLflow estimator cell — does not currently include `wait=False`). `3b3d61db` and `ad4b60bb` should explicitly call `.fit(inputs=None, wait=False)` to match the Day 1 SageMaker pattern.
- WARN Safety-net: Lab 1 (`46488328`) PASS, Lab 2 (`6e4ee941`) PASS, but the Homework cell `ad4b60bb` defines `estimator_mlflow` then calls `.fit(wait=False)` (training-job pattern) without a `training_job_name` safety-net cell immediately after (the rule says `training_job_name` safety-net must follow every `.fit(wait=False)`). Add one.
- FAIL Discussion prompts: zero peer-discussion markdown cells detected anywhere in F2. F2 has six major sections (session, S3, packaging, CPU launch, monitoring, MLflow); rule requires at least one discussion between each pair. Add 3–4 short discussion markdown cells.
- WARN Homework cells: only 3 homework references detected — Lab 1 has no homework extension (only verification + safety-net), Lab 2 has none. Add per-lab homework extension cells.
- PASS Tier markers: Lab 1 and Lab 2 explicitly tagged "Tier 1, guided" in the header markdown cells (`11ff39db`, `030b56c6`).
- PASS Diagrams: `bec804c4` (SageMaker Session wiring), `7e69a2d0` (Managed MLflow flow) — both in markdown.
- PASS numpy pin: `f63b2d18` install cell pins numpy properly and mlflow==2.13.2.
- PASS Instance types: HuggingFace failure demo (`5f93f202`) labels CPU as wrong; PyTorch estimator (`3b3d61db`) uses `ml.m5.xlarge` (CPU) — correct per rule.
- PASS Cell parity with solution: 43 = 43.

### T1 — Exercises/topic_1_overview_genai/topic_1_overview_genai.ipynb (33 cells)

- WARN Lab Tiers: T1 carries Lab 2 tagged "Tier 2" (cell idx 23, `20280ec1` — "Build a Barclays Complaint Triage Prompt"). This is the only Tier 2 across Day 1 — which **technically satisfies** the "exactly one Tier 2 on Day 1" rule, but it lives in the FIRST topic. Pedagogically the Tier 2 should sit mid-day (T2 or T3a), not Topic 1 minute 30. Consider moving the Tier 2 lab to T3a (the densest topic) and demoting T1 Lab 2 to Tier 1.
- PASS Diagrams: `890863ff`, `54a7aa35` — both markdown, both before their working demos.
- PASS Safety-nets: Lab 1 (`5381769b` → cell 15) and Lab 2 (`20280ec1` → cell 25) both have downstream `if … is None:` safety-nets.
- PASS Four-Beat: one broken-code cell (idx 5), two diagrams as Beat 2, working demos, two labs as Beat 4.
- PASS Discussion: 3 discussion cells (7, 13, 27).
- PASS Cell parity: 33 = 33.

### T2 — Exercises/topic_2_introducing_llms/topic_2_introducing_llms.ipynb (45 cells)

- WARN Safety-net for "Knowledge check" (cell `8bc5ccb2`): the auto-detector did not find an `if … is None:` follow-up. This is intentional (knowledge check is self-contained T/F, not a feeder), but it does contain `None  # YOUR CODE: True or False` answers — those answers do not feed downstream cells, so a safety-net is not strictly needed. Mark INFO, not FAIL.
- PASS Diagrams: `3108f030`, `f20488a6` — both markdown.
- PASS Four-Beat: 4 broken-code cells found (3, 5, 13, 17). Strong pattern.
- PASS Discussion: 4 discussion cells (11, 21, 26, 32).
- PASS Variable handoff: `complaints_dataset` and `tokenizer` are introduced here — these names get reused in T5/T6a, good.
- PASS Cell parity: 45 = 45.
- INFO: Two real labs (`642caca7`, `5649546a`) plus the optional knowledge-check (`8bc5ccb2`). The lab-counter saw 3, but only the first two are graded labs.

### T3a — Exercises/topic_3a_attention_python/topic_3a_attention_python.ipynb (45 cells)

- PASS Four-Beat: broken-code cell at idx 21, two diagrams (`2650ac05` at 7, `b0049d40` at 12), three labs all Tier 1 (Bahdanau, dot-product, scaled-dot — `cfb8c0b5`, `2150ea22`, `dd753fa4`).
- PASS Safety-nets: all 3 labs have `if … is None:` cells immediately after.
- PASS Discussion: 2 cells (5, 34). Could add one more between Sections 2 and 3 (a third one would bring per-section coverage to 1:1).
- PASS Homework: 6 homework-tagged cells (19, 28, 39, 41, 42, 43).
- PASS Cell parity: 45 = 45.
- INFO Variable handoff to T3b: T3a uses pure-Python `numpy` arrays for q/k/v; T3b switches to PyTorch tensors. The `COMPLAINT_TOKENS` constant is **not present** in T3a but IS present in T3b — a small continuity gap (would be cleaner to introduce `COMPLAINT_TOKENS` once in T2 or T3a and reuse).

### T3b — Exercises/topic_3b_attention_pytorch/topic_3b_attention_pytorch.ipynb (29 cells)

- PASS Lab tier composition: Lab 1 (Tier 1, `189e4306` — DotProductAttention) + Capstone (Tier 3, `e139df80` — ScaledDotProductAttention). Exactly one Tier 3 as required for last-topic-of-day. Solid.
- PASS Diagrams: `a74cc0de` at 6, `8cbcbc93` at 15 — both markdown.
- PASS Safety-net Lab 1: cell 10.
- INFO Capstone (cell 19) does not have an `if … is None:` safety-net — but Tier 3 by design provides only a function signature + docstring + pass. A safety-net should still be present if downstream cells use the class. Cell 20 is the verification cell that imports `ScaledDotProductAttention` — if students don't finish, the class won't exist and the verification will error. **Add a Tier 3 safety-net** that defines the working class inline, right after cell 19, with a clear "skip if you finished" comment.
- PASS Discussion: 2 cells (17, 24).
- PASS Cell parity: 29 = 29.

### T4 — Exercises/topic_4_transformers/topic_4_transformers.ipynb (34 cells)

- FAIL Lab Tiers (Day 2 rule violation): T4 contains a Tier 3 lab — `Lab 2 - Implement DecoderLayer (Tier 3 - Open-Ended)` at cell idx 24/25 (markdown header `b8bc8277` + code `427b935b`). Day 2 rule explicitly says **only T6b** (the last topic of Day 2) may have the Tier 3 lab. Convert T4 Lab 2 to Tier 2 — that simultaneously fixes the missing "exactly one Tier 2 on Day 2" requirement.
- WARN Four-Beat: only one broken-code cell detected at idx 19. Section 1 (positional embedding) and Section 2 (decoder) only show one explicit Beat-1 between them. Consider adding a deliberately broken positional-embedding cell at the start of Section 1.
- WARN SageMaker: capstone cell `cap00003` uses HuggingFace estimator on GPU `ml.g4dn.xlarge` — but the rule says **PyTorch estimator = CPU, HuggingFace estimator = GPU**. The cell uses `sagemaker.pytorch.PyTorch` on `ml.g4dn.xlarge` (GPU). This is the opposite mapping. Either:
  - keep PyTorch estimator and switch to `ml.m5.xlarge` (CPU), OR
  - switch to `sagemaker.huggingface.HuggingFace` with `ml.g4dn.xlarge` (GPU).
- PASS Diagrams: `ef284b17`, `81e37763` — both markdown.
- PASS Safety-nets: Lab 1 (`abf5f386` → 16), Lab 2 (`427b935b` → 27) both have safety-nets. The TJN safety-net at cell `ca2e722d` follows the `.fit(wait=False)` in `cap00003` — correct pattern.
- PASS Variable handoff: `bucket`, `region`, `role`, `sess`, `training_job_name` all present.
- PASS Cell parity: 34 = 34.

### T5 — Exercises/topic_5_huggingface/topic_5_huggingface.ipynb (42 cells)

- WARN SageMaker (false-positive only — install cell `4a9c405681de` includes "evaluate" string but only because the comment text mentions it, no real import). Recommend grep-confirm that no `import evaluate` or `from evaluate` exists anywhere in the file. If clean, PASS.
- WARN Lab Tiers: three labs all Tier 1. Acceptable for a mid-Day-2 topic. Fine as-is, but consider whether Lab 3 (model card) could be Tier 2 to absorb the missing Day 2 Tier 2 lab.
- PASS Diagrams: `83afa464217c`, `bfc01b70f1e9`.
- PASS Safety-nets: all 3 labs (`3f2dd5171e79`, `e2cab8c0e98b`, `083d58afe453`) have safety-net cells immediately after.
- PASS Variable handoff: `COMPLAINT_TOKENS` and `COMPLAINT_LABELS` carried in from T2/T4, `tokenizer` present.
- PASS Cell parity: 42 = 42.

### T6a — Exercises/topic_6a_full_finetuning/topic_6a_full_finetuning.ipynb (40 cells)

- FAIL SageMaker `evaluation_strategy`: three cells use the deprecated `evaluation_strategy=` arg instead of `eval_strategy=`:
  - `1aea6839cd53`
  - `90a951adc474`
  - `a2a98bc6474a` (Lab 2 — this is the lab starter cell, students copy the bad arg).
  
  All three must be renamed to `eval_strategy=`. This is the most important Day-2 fix.
- WARN Lab Tiers: Lab 1 + Lab 2 both Tier 1. Acceptable.
- WARN YOUR CODE hygiene: cell `0a5e29b02618` has `# YOUR CODE (reassign tokenized_train and tokenized_val)` — the comment names the variables to reassign, which is fine, but could be tightened.
- PASS Diagrams: `1bb5f3a4781e`, `04c70c15` — both markdown.
- PASS Safety-nets: Lab 1 (`0a5e29b02618` → 12), Lab 2 (`a2a98bc6474a` → 24). Capstone (`eb987855a6cf`) followed by `training_job_name` safety-net at `6b9006743fd1` — good.
- PASS Variable handoff: `COMPLAINT_TEXTS`, `tokenizer`, `sess`, `role`, `bucket`, `region`, `training_job_name` all present — full chain to T6b is intact.
- PASS Instance types: HuggingFace estimator on `ml.g4dn.xlarge` (GPU) — correct.
- PASS Cell parity: 40 = 40.

### T6b — Exercises/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb (40 cells)

- WARN Four-Beat: extractor found **zero** broken-code cells and zero `# YOUR CODE` markers detected by the lab-finder — because Lab 6b uses a function-signature Tier 3 pattern (cell `c31fd9c9c97d`) without a `YOUR CODE` literal. Manually verified the Tier 3 lab is present at cell idx 29. The notebook does have working demos and a capstone training job, but no obvious B1 broken-code cell — entire notebook reads as B3 only. Consider adding a B1 cell at the start of Section 2 (freeze vs. fine-tune) showing a naive frozen model that catastrophically underfits.
- WARN SageMaker: `.fit(wait=False)` check on cell `beca53c220b5` — this cell is actually the safety-net for `trained_model_data` (not a `.fit` call). False positive. Real `.fit(wait=False)` is in cell `2946d8671b00` and IS correctly wait=False with TJN safety-net at `899d244be2d7`. PASS in practice.
- WARN Homework: only 1 homework-tagged cell found (cell 35 "Stretch: Compare Transfer Learning vs Full Fine-Tuning"). For Lab 6b (Tier 3), an explicit Homework Extension markdown cell with starter code is missing. Add one.
- WARN Safety-nets: capstone (`2946d8671b00`) has TJN safety-net at `899d244be2d7` and a `trained_model_data` safety-net at `beca53c220b5` — good. Lab 6b (`c31fd9c9c97d`) has a Step-1 safety-net at `76fc4966400c` and Steps-2+3 safety-net at `2501f03505f7` — correctly paired.
- PASS Lab tier: exactly one Tier 3 lab (`5837fb89e3c9` header + `c31fd9c9c97d` body). Correct for last-topic-of-Day-2.
- PASS Diagrams: `1c0fce42b4f8` at 8, `41fa2c48` at 9 — both markdown. Note these are adjacent (consecutive markdown cells) — verify they each reference different concepts.
- PASS Variable handoff: `sess`, `role`, `bucket`, `region`, `tokenizer`, `trained_model_data`, `training_job_name` all present.
- PASS Cell parity: 40 = 40.

---

## Cross-Topic Continuity

### Variable handoffs

| Handoff | Expected vars | Found |
|---------|--------------|-------|
| T1 → T2 | (none specified) | `tokenizer` introduced in T2, used downstream — fine |
| T2 → T3a | `complaints_dataset`, `tokenizer` | T3a uses `role`, `sess`, `bucket`, `region` only. `complaints_dataset` / `tokenizer` not carried — INFO (T3a is theory-heavy, pure-python attention math). |
| T3a → T3b | `COMPLAINT_TOKENS`, `COMPLAINT_LABELS` | `COMPLAINT_TOKENS` appears in T3b but NOT defined in T3a. Define it once in T3a (or T2) and import to T3b for true continuity. **Minor gap.** |
| T3b → T4 | `COMPLAINT_TOKENS` | T4 references `bucket`, `region`, `role`, `sess`, `training_job_name` — but **not** `COMPLAINT_TOKENS`. The narrative thread for token data breaks here. Verify T4 actually needs token-level continuity (transformer build is a different exercise). INFO. |
| T4 → T5 | `COMPLAINT_TOKENS`, `COMPLAINT_LABELS` | T5 has both — handoff OK. |
| T5 → T6a | `COMPLAINT_TEXTS`, `tokenizer` | T6a has `COMPLAINT_TEXTS` and `tokenizer` — handoff OK. T5 uses `COMPLAINT_TOKENS`/`COMPLAINT_LABELS`, but T6a switches to `COMPLAINT_TEXTS`. Confirm the rename is intentional. |
| T6a → T6b | `sess`, `role`, `bucket`, `region`, model checkpoint names | All present. **Handoff OK.** |

### Lab tier distribution (Day 1)

- Day 1 total labs across F1/F2/T1/T2/T3a/T3b: 7 Tier 1 + 1 Tier 2 (T1 Lab 2) + 1 Tier 3 (T3b Capstone).
- **PASS** the count ("exactly one Tier 2 on Day 1" + "T3b has Tier 3").
- **WARN** the placement: Tier 2 sitting in T1 is pedagogically odd — students hit "harder" before "intro" theory. Recommended move: T3a Lab 2 (dot-product) → Tier 2, T1 Lab 2 → Tier 1.

### Lab tier distribution (Day 2)

- Day 2 total labs across T4/T5/T6a/T6b: 6 Tier 1 + **0 Tier 2** + 2 Tier 3 (T4 Lab 2 + T6b Lab).
- **FAIL** count: zero Tier 2 (rule requires exactly one), and two Tier 3 labs (rule says only T6b).
- **Recommended fix**: convert T4 Lab 2 (DecoderLayer) from Tier 3 → Tier 2. That single change brings Day 2 to compliance (6 Tier 1 + 1 Tier 2 + 1 Tier 3, T6b is last topic).

### Day references

None found in any header — **PASS** across all 10 notebooks.

### AI-tells (em dash, en dash, ×, emoji, bare `---`)

None found in any of the 10 exercise notebooks or any of the 10 solution notebooks — **PASS**.

---

## Priority Fix List

### P0 — Hard blockers (will break student notebooks at runtime or are direct rule violations)

1. **T6a SageMaker `evaluation_strategy` → `eval_strategy`** in cells `1aea6839cd53`, `90a951adc474`, `a2a98bc6474a`. Transformers 4.40+ removed the old kwarg; students will hit a `TypeError` on Lab 2.
2. **F1 SageMaker `evaluation_strategy` → `eval_strategy`** in cells `bd3aeac5`, `e9c37e46`. Same root cause.
3. **T4 Lab 2 tier**: Day 2 rule says only T6b is Tier 3. Convert T4 Lab 2 (DecoderLayer) from Tier 3 → Tier 2. This simultaneously fixes the missing Day 2 Tier 2.
4. **T4 capstone instance/estimator mismatch**: cell `cap00003` uses `sagemaker.pytorch.PyTorch` on `ml.g4dn.xlarge`. Rule: PyTorch estimator = CPU `ml.m5.xlarge`, HuggingFace estimator = GPU `ml.g4dn.xlarge`. Fix one or the other.

### P1 — Structural (pedagogy / continuity gaps)

5. **F2 missing discussion cells**: zero peer-discussion prompts in the whole notebook. Add 3–4 short Barclays-framed discussion markdown cells between major sections.
6. **F2 Lab 1 and Lab 2 missing homework extension cells**. Only Lab 3 (the explicit Homework Extension at cell `e08c504a`) is present.
7. **F2 Homework cell `ad4b60bb`** lacks both `wait=False` (explicitly) and a `training_job_name` safety-net pattern. Align with the canonical Day-2 SageMaker fit pattern.
8. **F2 cell `3b3d61db`** (PyTorch estimator demo) does not have `wait=False` on its `.fit()` call; add it.
9. **T1 Tier 2 placement**: move Tier 2 lab from T1 → T3a for better Day-1 difficulty curve.
10. **T3b Capstone safety-net**: cell 19 (`e139df80`, `ScaledDotProductAttention`) is Tier 3 with no `if … is None:` fallback before the verification cell 20. Add a "skip if you finished" safety-net that defines a working class so the rest of the notebook runs.
11. **T6b missing Homework Extension cell** after Lab 6b. Add a starter-code homework cell.
12. **F1 four-beat completeness**: only one Beat-1 broken-code cell across five labs. Add explicit B1 cells before each Section's working demo.
13. **T6b four-beat completeness**: no Beat-1 cells detected. The whole notebook reads as B3+B4. Add a B1 (e.g. naive head-only fine-tune that overfits or underfits).
14. **F1 discussion coverage**: only 1 discussion cell across 5 sections. Add 3 more.
15. **T3a → T3b variable handoff**: define `COMPLAINT_TOKENS` (and `COMPLAINT_LABELS`) once in T3a so T3b reuses it directly. Currently `COMPLAINT_TOKENS` appears in T3b without being introduced in T3a.

### P2 — Polish

16. **F1 YOUR CODE hygiene**: 27 `# YOUR CODE: <hint>` annotations leak implementation hints. Tighten to bare `# YOUR CODE` or move hints to surrounding numbered-step comments.
17. **T6b adjacent diagrams** at cells 8 and 9 (`1c0fce42b4f8`, `41fa2c48`) are consecutive markdown cells. The "no markdown chain >3" rule is not violated, but verify each diagram introduces a distinct concept.
18. **T6a YOUR CODE hint** at cell `0a5e29b02618`: `# YOUR CODE (reassign tokenized_train and tokenized_val)` — names the variables, mild hint.
19. **T2 cell count discrepancy** in audit auto-counter (39 vs 45) — manual recount confirms 45. Note only.
20. **Recheck T5 cell `4a9c405681de`** for any literal `import evaluate` — comment-only mention is fine, but confirm.

---

## What Looks Great (worth preserving)

- Zero AI-tells (em/en dash, ×, emoji, bare `---`) across all 20 notebooks (10 exercise + 10 solution). Strong discipline.
- Cell parity exercise ↔ solution holds for all 10 pairs except auto-counter blip on T2.
- Barclays narrative present in every notebook.
- Diagrams: every notebook has exactly 2 diagram references, all in markdown cells, all before the working-demo they introduce.
- Safety-net pattern is consistent and correct for every standard `= None  # YOUR CODE` lab.
- `training_job_name` safety-net immediately after every `.fit(wait=False)` is present in T4, T6a, T6b.
- Variable continuity T5 → T6a → T6b (`sess`, `role`, `bucket`, `region`, `tokenizer`, `training_job_name`, `trained_model_data`) is intact — the SageMaker capstone arc lands cleanly.
- T3b last-of-Day-1 Tier 3 capstone is well-built (Bahdanau → dot-product → scaled-dot ladder).
- mlflow pinned to `2.13.2` and `sagemaker-mlflow==0.1.0` per rule in F2.
- numpy<2 pin present in F2 install cell.
