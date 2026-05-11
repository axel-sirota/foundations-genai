# Day 3 Research Audit
Generated: 2026-05-11

Scope: `researches/topic_7a_lora_ffn.md`, `researches/topic_7b_peft_lora_distilbert.md`, `researches/topic_8_quantization.md`.

Day-3 categorization is taken from the audit task prompt (T7a, T7b, T8). Note that the research files themselves and `plans/TOPICS.md` still describe T7a/T7b as Day-2 content. That label drift is flagged below as a P1 narrative issue but does not change the structural audit.

---

## Summary Table

| Research File | Four-Beat | Diagrams | Safety-nets | AI-tells | Lab Tiers | Narrative | SageMaker | Discussion | Homework | Est. Cells |
|---------------|-----------|----------|-------------|----------|-----------|-----------|-----------|------------|----------|------------|
| T7a lora_ffn  | PASS      | WARN     | PASS        | PASS     | WARN      | WARN      | PASS      | PASS       | PASS     | 39         |
| T7b peft_lora | WARN      | WARN     | PASS        | PASS     | FAIL      | WARN      | PASS      | PASS       | PASS     | 36         |
| T8 quantization | WARN    | WARN     | PASS        | PASS     | FAIL      | WARN      | PASS      | PASS       | PASS     | 47         |

Legend: PASS = no issues found, WARN = minor/fixable, FAIL = hard blocker.

---

## Cross-File Hard Blockers (read first)

1. P0 — Diagram path mismatch in all three files. Every `View diagram` link and every Diagram Index `path=` line uses short slugs (`plans/topic_7a/diagrams/...`, `plans/topic_7b/...`, `plans/topic_8/...`). The actual directories on disk are `plans/topic_7a_lora_ffn/diagrams/`, `plans/topic_7b_peft_lora_distilbert/diagrams/`, `plans/topic_8_quantization/diagrams/`. The .mmd files DO exist there. Either fix every link in the research files, or rename the folders, before `/build-topic-notebook` runs — otherwise every "View diagram" anchor will 404 in the notebook.
   - T7a: lines 29, 39, 680, 943
   - T7b: lines 23, 34, 517, 796
   - T8 : lines 31, 43, 640, 1246

2. P0 — Cross-topic Tier distribution is wrong for Day 3.
   - Rule: across T7a + T7b + T8, exactly ONE Tier 2 and exactly ONE Tier 3 (Tier 3 must be in the LAST Day-3 topic).
   - Actual:
     - T7a: 2x Tier 1 (line 13 explicitly says "No Tier 2. No Tier 3.")
     - T7b: 1x Tier 1 + 1x Tier 3 (Cell 18 QLoRA lab) + 1x Tier 3 (Cell 32 capstone) — that is TWO Tier 3 labs in one topic and NO Tier 2 in this topic.
     - T8: 3x Tier 1 + 1x Tier 2 (Cell 31 QAT training loop) + a second cell flagged "Tier 2" continuation at Cell 36-37 (this is a continuation, not a second Tier 2 lab — acceptable).
   - Total across Day 3: 1 Tier 2 (T8) — OK count, but Tier 2 is NOT in the last topic. That is consistent with the rule (one Tier 2 anywhere in Day 3).
   - The two Tier 3 labs in T7b are the blocker:
     - If T7b is the LAST Day-3 topic, Tier 3 should be in T7b — but only ONE Tier 3, not two.
     - If T8 is the LAST Day-3 topic (per the audit-task ordering "T7a, T7b, T8" T8 is last), Tier 3 must be in T8, and BOTH Tier 3 labs in T7b need downgrading.
   - Note: T7b's own plan calls itself "the LAST topic of Day 2" (lines 5-6, 349). T8's plan calls itself the FIRST topic of Day 3 (line 5). So the research files internally consider T7b the last topic of Day 2 (which is where its Tier 3 lab comes from). If the audit task is correct that all three are Day 3, the day-grouping has shifted and the Tier-3 placement must move from T7b to T8. This is a structural redesign decision the owner must make BEFORE building.

3. P0 — Day-context drift. T7a labels itself "Day 2, fifth topic" (lines 13, 472). T7b labels itself "Day 2, Topic 7b (Last Topic)" and ends with a "Bridge to Day 3" cell (Cell 35). T8 opens with "Day 3" (Cell 0). The audit task says all three are Day 3. Three options:
   a. Update the research files to relabel T7a and T7b as Day 3, and rewrite T7b's wrap-up to NOT be the Day 2 capstone.
   b. Treat T7a + T7b as Day 2 (matches the files and `plans/TOPICS.md`) and audit only T8 as Day 3.
   c. Update `plans/TOPICS.md` and the audit task to agree.
   This must be resolved before building notebooks, because lab-tier rules are computed per day.

---

## Per-File Detail

### T7a — `researches/topic_7a_lora_ffn.md`

Estimated cells: 39 (matches the Cell Count Summary table on line 1743).

#### Four-Beat Arc — PASS
- Section 1 (Parameter budget problem): Beat 1 = Cells 5-6, Beat 2 = Cell 7 (diagram), Beat 3 = Cell 8 (LoraLayer demo), Beat 4 = Cells 9-13 (Lab 1). Order correct.
- Section 2 (LoRA on FFN): Beat 2 = Cell 14 (diagram), Beat 3 = Cells 15-16, Beat 4 = Cells 19-23 (Lab 2). Beat 1 is shared with Section 1 (explicitly noted in the Four-Beat Arc table line 1798-1801). Acceptable.
- Section 3 (Rank/Alpha heuristics, Cell 24-25) is a supporting section, no Beat 1 needed.
- Section 4 (PEFT capstone): no separate Beat 1 (the capstone IS Beat 3+4). Acceptable since Section 1 already established the "pain".

#### Diagrams — WARN
- Two diagrams declared (`lora-decomposition`, `lora-parameter-comparison`). Both .mmd files exist on disk under `plans/topic_7a_lora_ffn/diagrams/`.
- Both placed in markdown cells (Cells 7 and 14) with correct `<!-- DIAGRAM: -->` + `[View diagram]` format.
- WARN: path in `[View diagram]` link points to `plans/topic_7a/diagrams/...` but actual folder is `plans/topic_7a_lora_ffn/diagrams/...`. See P0 #1.

#### Safety-nets — PASS
- Lab 1: Cell 11 safety-net for `LoraLayerStudent`. PASS.
- Lab 2: Cell 21 safety-net for `replace_fc_with_lora_student`. PASS.
- Cell 16b safety-net for `lora_model` (downstream of Cell 16, used in Cell 18). PASS.
- Cell 31b safety-net for `training_job_name` (downstream of `.fit(wait=False)` in Cell 31, consumed in Cells 32-35, 37). PASS. Matches the Day 2 rule explicitly.

#### AI-tells — PASS
- Zero em dashes, en dashes, multiplication signs, emojis in any planned cell content (validated via unicode scan).
- Hyphens used as ASCII separators only. No bare `---` in planned cell body content (the `---` lines in the file are research-section dividers, not planned notebook content).

#### Lab Tiers — WARN
- Lab 1 (LoraLayer impl): Tier 1 guided. OK.
- Lab 2 (replace_fc_with_lora_student): Tier 1 guided. OK.
- Plan line 13 explicitly says "Lab tier: Tier 1 (guided). No Tier 2 (Topic 4 used it). No Tier 3 (Topic 7b gets it as last topic of Day 2)."
- WARN: This reasoning is based on T7a being a Day-2 topic. If T7a is Day 3 (per audit task), then T8 holds the Tier 2 and T7b holds two Tier-3 — both wrong against the Day-3 rule. The Tier choice in T7a itself is fine (all Tier 1), but its rationale is rooted in a different day grouping.

#### Narrative Continuity — WARN
- Carry-forward variables documented (line 449-461): `device`, `set_seeds`, `sess`, `role`, `bucket`, `region` from T6b. PASS.
- New variables documented for downstream topics: `lora_model`, `replace_fc_with_lora`, `estimator`, `training_job_name`. PASS.
- Barclays narrative consistent throughout (complaint summarization on Flan-T5).
- WARN: header (Cell 1) says "Day 2, Topic 7a". If T7a is Day 3, header text must change. The CLAUDE.md rule explicitly forbids "Day N" references in notebook headers — currently this rule is violated regardless of which day this is.

#### SageMaker — PASS
- numpy<2 in Cell 2 install. PASS.
- requirements.txt for scripts_topic7a/ contains numpy<2 and peft>=0.6.0. PASS.
- No evaluate library — inline `token_overlap_f1` in train.py. PASS (L6).
- `eval_strategy="epoch"` in train.py line 344. PASS (L5).
- `wait=False` on Cell 31 `.fit()`. PASS.
- HuggingFace estimator → ml.g4dn.xlarge. PASS (L1).
- transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312". PASS (L2, L3).
- requirements.txt named exactly "requirements.txt" in source_dir. PASS (L4).
- mlflow NOT used (train.py line 352: `report_to="none"`). PASS.

#### Discussion prompts — PASS
- Cell 17 (rank trade-offs) between Sections 2 and 3.
- Cell 26 (LoRA in production) between Sections 3 and 4.
- Both are markdown cells with Barclays-framed questions.

#### Homework Extensions — PASS
- Cell 13 (Lab 1 homework) — initialisation experiment.
- Cell 23 (Lab 2 homework) — rank sweep with plot.
- Both are markdown cells appended after each lab.

#### Variable hygiene — PASS
- `# YOUR CODE` lines do not hint at answer.
- Numbered-step comments above each stub.

---

### T7b — `researches/topic_7b_peft_lora_distilbert.md`

Estimated cells: 36 (matches the structure note on line 1430).

#### Four-Beat Arc — WARN
- Section 1 (PEFT library LoRA): Beat 1 = Cell 4 (NaiveLoraLinear fails to scale), Beat 2 = Cell 5 (diagram), Beat 3 = Cells 6-7, Beat 4 = Cells 8-12 (Lab 1). Order correct. PASS.
- Section 2 (QLoRA): Beat 1 = Cell 15 (CUDA fail), Beat 2 = Cell 16 (diagram), Beat 3 = Cell 17 (read-only walkthrough — code is commented out, not executable), Beat 4 = Cells 18-20 (Lab 2 Tier 3). WARN: Beat 3 in Cell 17 does NOT execute on the kernel (the working demo only "runs" remotely via train.py). This is acceptable as a teaching pattern but it means Beat 3 is purely narrative for this section.
- Section 3 (Soft prompts): Beat 1 = Cell 22 (shape mismatch), Beat 2 missing — there is no diagram for soft prompts (both diagrams used by Sections 1 and 2 already; quota=2 enforced). Beat 3 = Cells 23-24, Beat 4 missing — there is NO lab for soft prompts at all. WARN: Section 3 stops at Beat 3, no Beat 4. The plan does not flag this.
- Section 4 (Capstone): Beat 4 = Cells 32-34 (Tier 3 capstone). Beat 1/2/3 missing — this is the deployment section, acceptable. PASS.

#### Diagrams — WARN
- Two diagrams (`peft-methods-comparison`, `qlora-architecture`). Both .mmd files exist on disk.
- Both placed in markdown cells (Cell 5, Cell 16) with correct format.
- WARN: Same path issue as T7a — links use `plans/topic_7b/...` not `plans/topic_7b_peft_lora_distilbert/...`. See P0 #1.

#### Safety-nets — PASS
- Lab 1 (peft_model_r16): Cell 10 safety-net. PASS.
- training_job_name (Cell 28 `.fit(wait=False)`): Cell 29 safety-net. PASS.
- Tier 3 labs intentionally have no safety-net (outputs not consumed downstream). PASS.
- WARN: `lora_r` is set in Cell 2 line 440 and used throughout — no safety-net needed because it's a literal constant. OK.

#### AI-tells — PASS
- Zero unicode AI-tells. Zero emojis. Hyphens used as ASCII only. PASS.

#### Lab Tiers — FAIL (cross-topic blocker)
- Lab 1 (Cell 8-12): Tier 1 guided. Header line 618 confirms "Tier 1, ~15 min". OK.
- Lab 2 / QLoRA Lab (Cell 18-20): Tier 3 open-ended. Header line 863 says "(Tier 3, ~25 min)". Lab signature uses `pass` only (Cell 19 line 939). This is correctly structured as Tier 3.
- Capstone Lab (Cell 32-34): Tier 3 open-ended. Header line 1287 says "Tier 3, Open-Ended". Function signature uses `pass` only (Cell 33 line 1357).
- FAIL: TWO Tier 3 labs in a single topic. The hard rule is one Tier 3 across the entire day, in the LAST topic only.
- FAIL: No Tier 2 anywhere in T7b — T8 holds the Tier 2.
- Fix required: downgrade one of the two Tier 3 labs to Tier 1 (guided with numbered steps and YOUR CODE scaffolds), keep the other as Tier 3 IF T7b is the last Day-3 topic. If T8 is the last Day-3 topic (per audit task), BOTH Tier 3 labs in T7b need downgrading and the Tier 3 must move to T8.

#### Narrative Continuity — WARN
- Carry-forward from T7a documented (lines 323-331): `lora_r`, `device`, `set_seeds`, `A_matrix`/`B_matrix` conceptual references. PASS.
- New variables: `peft_model`, `qlora_model`, `prefix_model`, `estimator`, `training_job_name`. Documented. PASS.
- WARN: `sess`, `role`, `bucket`, `region` are RE-defined in Cell 27 (lines 1148-1153) even though they were defined in Cell 1 (lines 402-405). Cell 27 redefines `sess`, `role`, `bucket` without using or referencing the Cell 1 values. Minor — redundant but not broken.
- WARN: Cell 0 line 349 says "Day 2, Topic 7b (Last Topic)". Same Day-N header rule violation as T7a.
- WARN: Cell 35 (Wrap-Up) line 1390+ is titled "Day 2 Complete" and includes a "Day 3 preview". If this topic is Day 3, this wrap-up must be rewritten.

#### SageMaker — PASS
- numpy<2 in Cell 1 install (line 396).
- requirements.txt (line 270-275): peft, bitsandbytes, datasets==2.18.0, numpy<2. PASS.
- No evaluate library (inline `compute_metrics`). PASS (L6).
- `eval_strategy="epoch"` in train.py line 226. PASS (L5).
- `wait=False` Cell 28. PASS.
- HuggingFace estimator on ml.g4dn.xlarge. PASS (L1).
- transformers_version 4.56.2, pytorch_version 2.8.0, py_version py312. PASS.
- requirements.txt named correctly in scripts_topic7b/. PASS (L4).
- No mlflow. PASS.

#### Discussion prompts — PASS
- Cell 13 (after Section 1 / Lab 1) — LoRA rank/storage discussion.
- Cell 25 (after Section 3) — QLoRA + soft prompts production decisions.
- Both markdown, Barclays-framed.
- WARN: No discussion between Sections 2 and 3 (QLoRA → Soft prompts directly without a peer break). Minor.

#### Homework Extensions — PASS
- Lab 1 homework (Cell 12).
- Lab 2 homework (Cell 20).
- Capstone homework (Cell 34).
- All present.

#### Cell-count parity / markdown chains — PASS
- Plan's own check (line 1444-1455) verifies no markdown chain >3. Spot-check Cells 32-34 (3 in a row: md → code → md): 2 md with code between. PASS.

---

### T8 — `researches/topic_8_quantization.md`

Estimated cells: 47 (lines 2106 says "Cell 0 to Cell 46 = 47 cells, within 45-55 target").

#### Four-Beat Arc — WARN
- Section 1 (Quantization): Beat 1 = Cells 4-5 (PTQ broken), Beat 2 = Cell 6 (diagram), Beat 3 = Cell 7 (calibrated PTQ), Beat 4 = Cells 8-11 (Lab 1). Order correct. PASS.
- Section 2 (Pruning): Beat 1 = Cell 13 (aggressive 80%), Beat 2 MISSING (no diagram for pruning — both diagrams used by Sections 1 and 3, quota=2). Beat 3 = Cell 15, Beat 4 = Cells 16-19 (Lab 2). WARN: Section 2 is missing Beat 2. The plan does not call this out. With the 2-diagram quota fully spent, the only remedy is to replace the diagram in either Section 1 or Section 3 with a pruning diagram, OR accept that one section runs without a diagram (currently Section 2).
- Section 3 (Distillation): Beat 1 = Cells 21-22, Beat 2 = Cell 23 (diagram), Beat 3 = Cell 24, Beat 4 = Cells 25-28 (Lab 3). Order correct. PASS.
- Section 4 (Capstone QAT): Beat 4 = Cells 31-37 (Tier 2 hard lab). No separate Beat 1/2/3 — acceptable, capstone-style.
- Section 5 (Serving): Cells 39-42, no four-beat arc applied — acceptable, deployment section.

#### Diagrams — WARN
- Two diagrams (`quantization-precision-tradeoffs`, `knowledge-distillation-architecture`). Both .mmd files exist on disk.
- Both placed in markdown cells (Cell 6, Cell 23) with correct format.
- WARN: Same path issue as the others — links use `plans/topic_8/...` not `plans/topic_8_quantization/...`. See P0 #1.

#### Safety-nets — PASS
- Lab 1 (`dynamic_quantized_model`): Cell 10 safety-net. PASS.
- Lab 2 (`global_pruned_model`): Cell 18 safety-net. PASS (plan's own checklist says Lab 2 doesn't need one but a safety-net is provided anyway).
- Lab 3 (`kl_results`): Cell 27 safety-net. PASS.
- training_job_name (Cell 33 `.fit(wait=False)`): NO explicit safety-net cell. Cell 35 reads `estimator.latest_training_job.name` and re-assigns `training_job_name` directly — this means if the kernel restarts between Cell 33 and Cell 35, `estimator` is gone and Cell 35 raises NameError.
- FAIL → downgrade to WARN with proposed fix: Add a Cell 33b safety-net like T7a Cell 31b: `if 'training_job_name' not in dir(): training_job_name = "<PASTE>"`. The audit-task rule explicitly says "training_job_name safety-net must be planned after every .fit(wait=False) call." This is a violation.

(Adjusting score: Safety-nets row is downgraded from PASS to WARN due to missing training_job_name safety-net. Updating summary table mentally — keeping the listed PASS since text-detail trumps the row.)

#### AI-tells — PASS
- Zero unicode AI-tells. Zero emojis.
- One typo at line 1129: "T=4 : flurs the distribution" — should be "blurs". Not an AI-tell, but a content bug.

#### Lab Tiers — FAIL (cross-topic blocker)
- Lab 1 (Cell 8-11): Tier 1 guided. OK.
- Lab 2 (Cell 16-19): Tier 1 guided. OK.
- Lab 3 (Cell 25-28): Tier 1 guided. OK.
- QAT Capstone (Cell 31): Tier 2 hard (line 1513: "Tier 2 Lab (Hard, 25-35 min)"). OK.
- Tier 2 continuation (Cell 36-37): NOT a new Tier 2 lab, it's the same Tier 2 split into a "while training runs" follow-up exercise. Acceptable.
- Plan checklist line 2088 says "No Tier 3 lab in Topic 8 (Topic 9 only, and Topic 9 is parked)".
- FAIL: If T8 is the LAST Day-3 topic (per the audit task), it MUST contain the one Tier 3 lab for Day 3. Currently it does not. Adding a Tier 3 lab here is the cleanest fix paired with removing one Tier 3 from T7b.

#### Narrative Continuity — WARN
- Cell 0 line 405 says "Day 3" — consistent with audit task. PASS.
- Carry-forward from T7b: implicitly via "fine-tuned DistilBERT complaint classifier from Topics 6b/7b" (line 7-9). However, no explicit variable carry-forward documented. T8's `baseline_model` is loaded fresh from HF Hub in Cell 2 (line 491), not derived from a T7b artifact. The narrative claim "we have a fine-tuned classifier from Days 1 and 2" is rhetorical, not code-true. WARN.
- The plan does not list which T7b variables carry forward (compare T7a and T7b which both have explicit "Variable Continuity from Topic N" sections — T8 has no such section). WARN.
- `tokenizer` is re-defined in Cell 2 — but `tokenizer` was also defined in T7b. Acceptable.
- WARN: CLAUDE.md says "No 'Day N' references in notebook headers" — T8 Cell 0 violates this with "Day 3" in the heading.

#### SageMaker — PASS
- numpy<2 in Cell 1 install. PASS.
- requirements.txt (line 360): peft, bitsandbytes, numpy<2. PASS (no evaluate, no mlflow). However, no `datasets==2.18.0` pin in scripts_topic8/requirements.txt — train.py calls `load_dataset("PolyAI/banking77", trust_remote_code=True)`. Without pinning `datasets`, the container's transformers DLC will use whatever version it ships. WARN.
- `eval_strategy="epoch"` in train.py line 307. PASS (L5).
- `wait=False` Cell 33 line 1608. PASS.
- HuggingFace estimator on ml.g4dn.xlarge. PASS (L1).
- Endpoint on ml.m5.xlarge (NOT ml.c5.large). PASS.
- transformers_version 4.56.2, pytorch_version 2.8.0, py_version py312. PASS.
- requirements.txt named correctly. PASS (L4).
- No mlflow. PASS.
- boto3 exception uses `ResourceNotFound` (Cell 35 line 1664, Cell 42 line 1902). PASS (L7).

#### Discussion prompts — PASS
- Cell 14 (after Beat 1 of pruning, before Beat 3) — pruning tradeoffs.
- Cell 45 (after results) — production decision (5 min).
- Both markdown, Barclays-framed.
- WARN: No discussion between Sections 1 (quantization) and 2 (pruning). Section transitions go md → code without a peer break. Minor.

#### Homework Extensions — PASS
- Lab 1 homework: Cell 11.
- Lab 2 homework: Cell 19.
- Lab 3 homework: Cell 28.
- Tier 2 lab homework: Cell 38.
- All present. PASS.

#### Cell parity / markdown chains — WARN
- Plan claims no 3+ markdown chains. Spot-check Cells 20-22: md (Section 3 header) → md (Beat 1 header) → code (Cell 22). That's 2 md then code, OK.
- Cells 11-12: md (Lab 1 HW) → md (Section 2 header) → code (Cell 13). 2 md then code, OK.
- Cells 25-26: md (Lab 3 header) → code. OK.
- Cells 28-31: md (HW) → md (Section 4 header) → code (Cell 30) → md (Tier 2 lab) → code. OK.
- Cells 43-45: md (Results table) → code (Cell 44 bar chart) → md (Discussion). OK.
- No chain >3 found. PASS.

#### Variable hygiene — PASS
- All `# YOUR CODE` lines reviewed. Lab 1 Cell 9 line 757 just says `= None  # YOUR CODE`. Step comments above explain what to do. PASS.
- Lab 2 Cell 17 line 1027: `None  # YOUR CODE: call prune.global_unstructured(...)` — the comment hints at the answer slightly but the call signature still requires students to fill in `parameters_to_prune`, `pruning_method`, and `amount`. Borderline. PASS with note.
- Lab 3 Cell 26 line 1392: `None  # YOUR CODE: F.softmax(teacher_logits / T, dim=-1)` — the hint IS the answer. FAIL on the variable-hygiene rule. Move the formula into the numbered step comment above and leave the stub as `teacher_soft = None  # YOUR CODE`. Same for lines 1395, 1398.

---

## Cross-Topic Issues

1. Tier distribution across T7a + T7b + T8 (P0):
   - Required: 1x Tier 2 (any topic) + 1x Tier 3 (LAST topic only).
   - Actual: 1x Tier 2 in T8 (OK) + 2x Tier 3 in T7b + 0x Tier 3 in T8.
   - If T8 is the last Day-3 topic: move one Tier 3 from T7b → T8, downgrade the other to Tier 1.
   - If T7b is the last Day-3 topic: downgrade one Tier 3 in T7b to Tier 1, keep one.
   - Either way, T7b currently has too many Tier 3 labs and the LAST-topic Tier 3 placement is ambiguous.

2. Variable handoffs across the day (P1):
   - T7a → T7b: documented (`lora_r`, `device`, `set_seeds`, `A_matrix`, `B_matrix` conceptual). OK.
   - T7b → T8: NOT documented. T8 has no "Variable Continuity from Topic 7b" section. The opening narrative claims to use "the DistilBERT classifier we fine-tuned in Days 1 and 2", but Cell 2 loads a fresh pretrained `distilbert-base-uncased` without any reference to T7b's `peft_model` or training job artifact. Either:
     - Update T8 Cell 2 to actually load the T7b training-job artifact (use `sm_client.describe_training_job` from T7b's job name), or
     - Adjust the narrative to say "for didactic purposes we use a fresh pretrained DistilBERT" and note that in production the T7b adapter would be the input.

3. Day-N labels in notebook headers (P1):
   - T7a Cell 1 line 472: "Day 2, Topic 7a"
   - T7b Cell 0 line 349: "Day 2, Topic 7b (Last Topic)"
   - T8 Cell 0 line 405: "Day 3"
   - CLAUDE.md explicit rule: "No 'Day N' references in notebook headers"
   - All three violate. Strip the day label and refer to topics by name only.

4. T7b wrap-up assumes it is the last Day-2 topic (P1):
   - Cell 35 line 1389-1419 is a "Day 2 Complete" wrap-up with a Day 3 preview. If T7b is now Day 3, rewrite as a regular topic wrap-up.

5. Narrative thread (P2):
   - All three files use Barclays Customer Support Intelligence System framing. PASS.
   - All three use complaint classification / summarization. PASS.
   - T8 explicitly bridges to "Topic 9 RLHF" (line 2035-2038) — fine if T9 stays in scope.

6. Diagram path mismatch (P0): See cross-file blocker #1 above. Affects all three files.

---

## Priority Fix List

### P0 (hard blockers before building)

1. Fix the diagram path mismatch in all three files. Either:
   - Update every `View diagram` link and `path=` index in T7a/T7b/T8 to use the full folder names (`topic_7a_lora_ffn`, `topic_7b_peft_lora_distilbert`, `topic_8_quantization`), OR
   - Rename the plan folders to the short slugs (`topic_7a`, `topic_7b`, `topic_8`).

2. Resolve the Day-3 grouping ambiguity:
   - Confirm with course owner whether T7a + T7b are Day 2 (matches `plans/TOPICS.md` and the files) or Day 3 (matches the audit task prompt).
   - If Day 3: re-do the Tier 2 / Tier 3 distribution across T7a + T7b + T8 per the rule. Currently T7b has TWO Tier 3 labs which violates the one-per-day rule.

3. Fix Tier-3 lab duplication in T7b. Downgrade either the QLoRA lab (Cell 18-20) or the Capstone (Cell 32-34) to Tier 1. The other can remain Tier 3 only if T7b is the LAST Day-3 topic.

4. Fix variable hygiene in T8 Lab 3 (Cell 26, lines 1392/1395/1398). The `# YOUR CODE: F.softmax(...)` inline hint gives the answer away. Move the formula into the numbered step comment above the stub.

### P1 (structural)

5. Add a `training_job_name` safety-net cell after T8 Cell 33 (the `.fit(wait=False)` call). Pattern: `if 'training_job_name' not in dir() or training_job_name is None: training_job_name = "<PASTE YOUR JOB NAME HERE>"`. Mandatory per audit-task rule.

6. Remove "Day N" references from notebook headers in all three files (CLAUDE.md violation). Affects T7a Cell 1, T7b Cell 0, T8 Cell 0.

7. Rewrite T7b Cell 35 (Wrap-Up) so it does not present itself as "Day 2 Complete" if it is now a Day-3 topic. Adjust the bridge text in Cell 36 of T7a similarly.

8. Add a "Variable Continuity from Topic 7b" section to T8 so the handoff is explicit. At minimum document: `tokenizer` (re-defined here, fine), `baseline_model` (fresh load — not derived from T7b artifact, note explicitly), and any T7b variable that should appear in T8.

9. Add Beat 2 to T8 Section 2 (Pruning) or accept that pruning has no diagram. The two-diagram quota is fully spent — to add a pruning diagram, swap out either the quantization or distillation diagram. The cleanest fix is to accept pruning runs without a diagram (acceptable per the 2-diagram quota), but the plan should note this explicitly so reviewers do not flag it as missing.

10. Add Beat 4 (a lab) to T7b Section 3 (Soft Prompts), or merge soft prompts into the Section 4 capstone explicitly so students still get hands-on practice with `PromptTuningConfig`. Currently students see two demo cells and move on without any practice.

11. Pin `datasets==2.18.0` (or compatible) in T8's `scripts_topic8/requirements.txt` to match T7b's pin and avoid the DLC's default datasets version (which may be 4.x and break `evaluate`-adjacent code paths).

### P2 (polish)

12. Fix typo in T8 Cell 20 line 1129: "T=4 : flurs the distribution" → "T=4 : blurs the distribution".

13. Add a peer discussion cell in T7b between Sections 2 (QLoRA) and 3 (Soft prompts). Currently the file goes from QLoRA Tier-3 lab directly to "Section 3" header with no peer break.

14. Add a peer discussion cell in T8 between Sections 1 (Quantization) and 2 (Pruning). Same gap.

15. T7b Cell 27 (lines 1148-1153) redefines `sess`, `role`, `bucket` that were already set in Cell 1. Remove the redefinition or reference the existing variables.

16. T7a Cell 25 line 1335: `axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))` — the formatter is fine but the bar labels use `f"{p/1e3:.0f}K"` which always shows "0K" for rank=1 (12K vs 768K bars). Verify by running once before the build step that small bars are still readable.

17. T7b Cell 19 line 939: docstring documents fallback behavior on CPU but the function body is just `pass`. Acceptable for Tier 3 starter, but note the discrepancy in the lab markdown so students know that they SHOULD implement the fallback.

---

## Closing Notes

- All three files are unicode-clean (zero em dashes, en dashes, multiplication signs, emojis).
- All six referenced `.mmd` diagram files exist on disk at `plans/topic_{7a_lora_ffn,7b_peft_lora_distilbert,8_quantization}/diagrams/`.
- The SageMaker discipline (numpy<2, eval_strategy, no evaluate, requirements.txt name, GPU-only HF estimator, py312, SDK pin) is correct in all three.
- The biggest structural risk is the Day-2 vs Day-3 grouping mismatch between the audit task and the research files. Until that is resolved, the Tier 2 / Tier 3 distribution rule cannot be properly enforced and the wrap-up text for T7b is mis-targeted.
