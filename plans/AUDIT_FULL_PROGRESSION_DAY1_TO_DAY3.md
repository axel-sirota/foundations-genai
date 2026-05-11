# Full Course Progression Audit: F1 through T8
Generated: 2026-05-11
Audit scope: 13 notebook pairs (26 notebooks total)
Mode: READ-ONLY. No files were modified.

## Notes on naming

The audit prompt referenced topic slugs (e.g. `topic_1_pytorch_nlp`, `topic_5_rlhf`, `topic_6b_deployment`, `topic_3b_transformers`) that do **not** exist on disk. The actual on-disk slugs were used instead:

| Prompt slug | Actual slug |
|-------------|-------------|
| topic_1_pytorch_nlp | topic_1_overview_genai |
| topic_2_embeddings | topic_2_introducing_llms |
| topic_3a_attention | topic_3a_attention_python |
| topic_3b_transformers | topic_3b_attention_pytorch |
| topic_5_rlhf | topic_5_huggingface |
| topic_6b_deployment | topic_6b_transfer_learning |

This is itself worth flagging: either the audit prompt is stale, or the on-disk slugs were renamed without updating the curriculum index. **P1**.

---

## Overall Status: **FAIL** (P0 issues present)

Two P0 issues block delivery; several P1 issues degrade student experience.

---

## Per-Notebook Summary

| Notebook | Cells (ex/sol) | AI-tells | Parity | Day-in-header | Diagrams | YOUR-CODE hints | Tier markers | SM constraints | Status |
|----------|----------------|----------|--------|----------------|----------|-----------------|--------------|----------------|--------|
| F1  pytorch_refresher          | 48/48 | PASS | PASS | PASS | 2/2 + mmd OK | PASS (md only) | T1x5 | n/a | PASS |
| F2  sagemaker_fundamentals     | 45/45 | PASS | PASS | PASS | 2/2 + mmd OK | **FAIL** (Cell 22 inline hint) | T1x2 | numpy<2 OK, RNE-FullWord, wait=True in HW | **WARN** |
| T1  overview_genai             | 33/33 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1x2 | n/a | PASS |
| T2  introducing_llms           | 45/45 | PASS | PASS | PASS | 2/2 + mmd OK | **FAIL** (Cell 7 x3, Cell 36 x1) | T1x2 (ex), T1x2 (sol)* | n/a | **WARN** |
| T3a attention_python           | 46/46 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1+T2+T1 | n/a | PASS |
| T3b attention_pytorch          | 30/30 | PASS | PASS | PASS | 2/2 + mmd OK | PASS (md ref) | T3 markers (see note) | n/a | PASS |
| T4  transformers               | 34/34 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1+T2 (ex), T1+T2+T2 (sol)* | numpy<2 OK, instance OK, RNE-FullWord | **WARN** |
| T5  huggingface                | 42/42 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1+T1+T1 | numpy<2 OK, no fit/instance | PASS |
| T6a full_finetuning            | 40/40 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1+T1 | numpy<2 OK, g4dn OK, RNE-FullWord, eval_strategy refs only | **WARN** |
| T6b transfer_learning          | 43/43 | PASS | PASS | PASS | 2/2 BUT mmd path WRONG | PASS | T3+T3 | numpy<2 OK (no install in cell shown), m5 OK, fit(wait=True) in HW stretch | **WARN** |
| T7a lora_ffn                   | 44/44 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1+T1 | numpy<2 OK, g4dn OK | PASS |
| T7b peft_lora_distilbert       | 47/47 | PASS | PASS | PASS | 2/2 + mmd OK | PASS | T1+T2+T2+T1+T2+T2 | numpy<2 OK, g4dn OK, **Solution Cell 33 leaves `= None # YOUR CODE`** | **FAIL** |
| T8  quantization               | 56/56 | PASS | PASS | PASS | 2/2 + mmd OK | PASS (md refs only) | T1x3+T2x4+T3 | numpy<2 OK, g4dn+m5 OK, RNE-FullWord | PASS |

\* Sol/ex tier-marker parity diffs are false positives — markers occur in counted text strings only.

**AI-tells scan**: ALL 26 notebooks pass — zero em-dashes, en-dashes, unicode-mul characters, or bare `---` markdown lines.

**Cell-count parity**: ALL 13 pairs pass.

**Day N in header**: ALL 26 notebooks pass — no "Day 1/2/3" leaks in cells 0–1.

**Diagram count**: All 13 pairs have exactly 2 `<!-- DIAGRAM: -->` per notebook (ex and sol).

**Diagram folders**: All `plans/topic_*/diagrams/` directories exist with .mmd files (F1, F2 do not have a `plans/<slug>/diagrams/` folder — they live in `Frameworks/` only; this matches their non-topic status).

---

## Cross-Day Variable Continuity

Reviewed the first non-install code cell ("setup") of each receiving topic.

| Transition | Expected variables | Found in receiver setup | Status |
|------------|--------------------|-------------------------|--------|
| F1 → F2 | sess, role, bucket, region | F2 install cell creates them fresh; no kernel carry-over expected (F1 is PyTorch-only) | PASS |
| F2 → T1 | sess, role, bucket, region | T1 does **NOT** use SageMaker session vars; runs in Studio kernel only. Acceptable for content but the chain claim in the audit prompt is incorrect — T1 is intro/GAN/OpenAI API. | OK (chain claim invalid) |
| T1 → T2 | device, set_seeds | T2 does **not** define `device` or `set_seeds` in setup. T2 setup begins at "Beat 1 -- Naive tokenization". No carry-over. | **WARN** (broken chain) |
| T2 → T3a | model/tokenizer/embedding vars | T3a setup imports gensim+nltk word2vec_sample fresh; no T2 variables carried. T3a is also where `sess/role/bucket/region` is FIRST introduced. | **WARN** (chain restarted) |
| T3a → T3b | attention mechanism vars | T3b setup re-imports torch and re-defines `set_seeds`+`device` fresh; no T3a softmax/attention vars carried. | **WARN** (chain restarted) |
| T3b → T4 | sess, role, bucket, region | T4 setup re-imports sagemaker + get_execution_role and re-defines `set_seeds`. Comment says "carried over from Topics 3a and 3b" but the code re-creates them. | PASS (code path works; comment is misleading) |
| T4 → T5 | training_job_name / model artifacts | T5 setup re-defines `set_seeds`, imports torch, defines COMPLAINT_TOKENS as "carried from Topic 4". No training_job_name. T5 is the HuggingFace intro topic, not a continuation of T4's translator capstone. | OK (intentional) |
| T5 → T6a | sess/role/bucket/region/model config | T6a setup re-defines `set_seeds`, references "Complaint vocabulary carried over from Topic 5" via COMPLAINT_TEXTS (different var name). T6a install cell creates fresh sess/role/bucket/region. | PASS |
| T6a → T6b | trained model / training_job_name | T6b install cell is **empty** in the exercise (cell-0 install was probably consolidated). T6b setup not captured by the install-cell scan. Needs manual verification — see P1 below. | **WARN** |
| T6b → T7a | sess/role/bucket/region/device/set_seeds | T7a install cell creates fresh sess/role/bucket/region; setup imports torch + set_seeds + device. No kernel carry-over. | PASS (fresh) |
| T7a → T7b | lora_r, device, set_seeds, sess/role/bucket/region | T7b setup says explicitly `lora_r = 8   # rank, carried forward from T7a`. Sess/role/bucket/region created fresh in install. | PASS |
| T7b → T8 | Fresh load expected | T8 setup says "Load a fresh DistilBERT classifier. Note: this is NOT carried over from Topic 7b kernel state." Correct per audit spec. | PASS |

**Conclusion**: The variable-continuity model the audit prompt describes is largely aspirational — most topics restart fresh and announce the carry-over in *comments* without actually depending on kernel state. This is the correct robustness pattern (every notebook is self-contained) but the audit prompt's chain assertions do not match implementation reality. P2: align prompt wording with the as-built pattern.

---

## Pedagogical Progression

1. **Complexity arc F1 → T8** — sound. F1 (PyTorch refresher) → F2 (SM session/estimator) → T1 (intro/GAN) → T2 (tokenisation) → T3a (attention by hand) → T3b (attention in torch) → T4 (transformer + first remote CPU job) → T5 (HuggingFace pipelines/AutoModel) → T6a (Flan-T5 full FT, first GPU) → T6b (DistilBERT transfer learning) → T7a (LoRA from scratch) → T7b (PEFT/LoRA + QLoRA) → T8 (quantization + distillation + pruning + Tier 3 capstone).
2. **Lab tier distribution**:
   - Day 1 (T1–T3b): only Tier 1 + one Tier 2 (T3a Cell 25) + Tier 3 cluster in T3b. Conforms to the "1 Tier 2 / day, Tier 3 last topic of day" guideline.
   - Day 2 (T4–T6b): T4 has one Tier 2 (Cell 24), T5 all Tier 1, T6a all Tier 1, T6b has Tier 3 (Cells 29–30). Conforms.
   - Day 3 (T7a–T8): T7a all Tier 1, T7b has 2x Tier 2 (Cells 19, 42), T8 has 3x Tier 1 + 4x Tier 2 markers + 1x Tier 3 (Cell 52 compress_model). Matches the prompt's expected Day 3 distribution.
3. **Barclays STAR narrative** — consistent. Complaint classification + Barclaycard scenarios show up coherently from T2 onwards. T1 introduces the business context with GAN-on-text as the "why text needs LLMs" hook.
4. **Discussion sophistication** — Cells around peer-discussion prompts get more domain-aware over time (e.g. cost/latency tradeoffs in T8 vs basic tokenisation discussion in T2). No regressions found.
5. **Four-beat arc** — visible in every topic: Beat 1 marker present in setup cells (T2, T3b, T4, T6a, T7b, T8 explicitly comment Beat 1). No topic appears to break the arc.

---

## Critical Issues (P0 — must fix before delivery)

### P0-1: T7b Solution Cell 33 has unfilled `= None # YOUR CODE` lines
File: `Solutions/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb` Cell 33
```
transferred_accuracy = None  # YOUR CODE
fresh_accuracy = None  # YOUR CODE
```
This is a code cell in the SOLUTION notebook. The solution must contain working code, not placeholders. This breaks the solution notebook downstream wherever those variables are referenced.

### P0-2: T6b broken diagram path
File: `Exercises/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb` Cell 9
File: `Solutions/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb` Cell 9
Diagram link: `[View diagram](../../plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd)`
Resolved path: `plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd` — **does not exist**.
The actual file is at `plans/topic_6b_transfer_learning/diagrams/tl-vs-finetuning-comparison.mmd`. The link is missing the full slug. Students will see a 404 when clicking "View diagram".

---

## Important Issues (P1)

### P1-1: F2 Lab Cell 22 contains an inline hint that gives the answer
`None  # YOUR CODE: replace this line with estimator.fit(inputs=None, wait=False)`
This violates the YOUR-CODE hygiene rule ("placeholder line must not hint at the answer"). The line above already says "Hint: call estimator.fit() with inputs=None and wait=False." That is sufficient.

### P1-2: T2 Lab inline hints
`Exercises/topic_2_introducing_llms/topic_2_introducing_llms.ipynb`
- Cell 7: three lines like `result_short = None  # YOUR CODE: call analyze_complaint_tokens on short_complaint`
- Cell 36: `... : None,  # YOUR CODE: True or False`
The `# YOUR CODE:` annotation contains the operative verb. Replace with `# YOUR CODE` (no colon-prefixed instruction on the placeholder line).

### P1-3: Audit prompt directory names do not match repo
The audit prompt references slugs that do not exist on disk (topic_1_pytorch_nlp, topic_5_rlhf, topic_6b_deployment, etc.). Either the audit prompt or the on-disk slugs need to be reconciled. This is likely a stale prompt copy.

### P1-4: T6b Cell 36 (stretch) references `estimator_full.fit(wait=True)` in markdown
This is a markdown stretch instructions cell, not executable code, but `wait=True` will block the notebook for the duration of training. If a student copy-pastes the snippet they will hang their kernel. Recommend `wait=False` consistently and add a poll-loop snippet.

### P1-5: F2 Cell 41 homework instructions show `wait=True` example
`estimator_mlflow.fit(inputs=None, wait=True)` in a code snippet inside a markdown cell. Same issue as P1-4 — pedagogical inconsistency with the project rule "wait=False on all .fit calls".

### P1-6: T6b empty install cell in exercise
The install cell scan returned empty text for `Exercises/topic_6b_transfer_learning/topic_6b_transfer_learning.ipynb`. If the install cell was consolidated into the setup cell, that is fine; if it was dropped entirely students may hit `ModuleNotFoundError`. Worth a manual spot-check.

---

## Minor Issues (P2)

### P2-1: `ResourceNotFoundException` appears in F2, T4, T6a, T8
The boto3 SageMaker client uses `ResourceNotFound` (no `Exception` suffix). Per L7 in SAGEMAKER_LESSONS_LEARNED. Cells flagged:
- `Frameworks/sagemaker_fundamentals.ipynb` Cell 27 (and solution)
- `Exercises/topic_4_transformers/topic_4_transformers.ipynb` Cell 32 (and solution)
- `Exercises/topic_6a_full_finetuning/topic_6a_full_finetuning.ipynb` Cell 34 (and solution)
- `Exercises/topic_8_quantization/topic_8_quantization.ipynb` Cells 41 and 49 (and solution)

These need inspection — the literal string may appear in error messages or docstrings (acceptable), but any `except sm_client.exceptions.ResourceNotFoundException:` is wrong and will raise `AttributeError` at runtime.

### P2-2: F1/F2/T8 markdown text references `= None  # YOUR CODE` as a pattern
Solution-notebook markdown cells like "Each `= None  # YOUR CODE` is one step" trip the lab-remnant scan. Not a real problem — pedagogical text — but worth noting if any tooling treats these as blockers.

### P2-3: Variable-continuity claims in audit prompt are not implemented
The audit prompt claims kernel-level chains like "T2 → T3a: model, tokenizer, or embedding vars carry forward". In practice each topic re-initialises (which is correct). Update the curriculum doc or the audit checklist to reflect "each topic is self-contained; carry-over is via comments/business context, not kernel state".

### P2-4: T4 setup comment claims `set_seeds` "carried over from Topics 3a and 3b" but redefines it
Same function body. Either remove the comment ("carry-over") or rely on import. Minor accuracy issue in narration.

---

## Spot-check: Solution Completeness

| Notebook | Lab cells inspected | Issues |
|----------|---------------------|--------|
| T7b sol | Cell 33 | **`= None # YOUR CODE` remains** (P0-1) |
| T8 sol  | Cells 8, 18, 28, 43, 52 | All markdown except 43 (qnnpack estimator) and 52/53 (Tier 3 docstring-only — expected per audit spec) — PASS |
| T6a sol | Cells 22, 31 | Full estimator code present, `eval_strategy=` correct — PASS |
| T4 sol  | Cell 30 | Full PyTorch estimator + `.fit(wait=False, job_name=...)` — PASS |
| F2 sol  | Cell 22 | `estimator.fit(inputs=None, wait=False)` present — PASS |
| F1 sol  | Cell 7 markdown intro | (Code cells not in flagged set) — PASS |

No bare `pass`-only lab cells found in any solution notebook (T8's Tier 3 has docstring-only stub as expected per the audit spec).

---

## Numeric Recap

- Total notebooks audited: 26 (13 pairs)
- Total banned-char hits: 0
- Total cell-parity failures: 0
- Total "Day N in header" leaks: 0
- Total diagram-link 404s: 2 (both T6b, same .mmd)
- Total solution-notebook code cells with unfilled placeholder: 1 (T7b Cell 33, 2 lines)
- Total inline YOUR-CODE hint violations: 4 lines across F2 + T2
- Total `evaluation_strategy=` keyword-arg usages: 0 (all 8 string occurrences are explanatory comments warning against the old name)
- Total HuggingFace estimators on non-GPU: 0
- Total PyTorch estimators on non-m5.xlarge: 0
- Total .fit(wait=True) in executable code: 0 (only in markdown stretch/homework hint snippets — P1-4, P1-5)
- Total `mlflow` imports outside F2: 0

## Sign-off

**Status**: FAIL — 2 P0 issues block delivery.
Fix P0-1 (T7b sol Cell 33) and P0-2 (T6b mmd link) before running the course.
P1 cosmetic/hygiene items should be batched into a follow-up cleanup pass.
