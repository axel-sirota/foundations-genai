# Day 2 Research Audit Report
Generated: 2026-05-11

---

## Summary Table

| File | Four-Beat | Safety-nets | Diagrams | AI-tells | Lab Tiers | Narrative | Discussion | Homework |
|------|-----------|-------------|----------|----------|-----------|-----------|------------|----------|
| topic_5_huggingface.md | PASS | PASS | PASS | WARN | PASS | PASS | PASS | PASS |
| topic_6a_full_finetuning.md | WARN | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| topic_6b_transfer_learning.md | WARN | WARN | FAIL | WARN | PASS | PASS | PASS | PASS |
| topic_7a_lora_ffn.md | WARN | WARN | PASS | PASS | PASS | PASS | PASS | PASS |
| topic_7b_peft_lora_distilbert.md | WARN | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

Legend: PASS = no issues found, WARN = minor/fixable issues found, FAIL = hard blocker

---

## Per-File Issues

---

### topic_5_huggingface.md

#### Four-Beat Arc

- [PASS] Section 1 (Hub): Beat 1 = Cell 6 (wrong task on sentiment pipeline), Beat 2 = Cell 7 (DIAGRAM hub ecosystem), Beat 3 = Cell 8 (model_info demo), Beat 4 = Lab 1 (Cell 10). All four beats present.
- [PASS] Section 2 (pipeline): Beat 1 = Cell 15 (wrong task name), Beat 2 not explicitly listed as a separate cell here (Diagram 1 was placed earlier; the section transitions directly to Beat 3). This is acceptable because Diagram 1 in Cell 7 still introduces the ecosystem concept. No second diagram is placed within Section 2 but the plan counts exactly 2 diagrams total which is correct.
- [PASS] Section 4 (AutoModel): Beat 1 = Cell 25 (AutoModel vs AutoModelForSeqClass), Beat 2 = Cell 24 (DIAGRAM automodel hierarchy), Beat 3 = Cell 26 (full working demo), Beat 4 = Lab 2 (Cell 28). All four beats present. Note: Beat 2 is placed BEFORE Beat 1 in the cell order (Cell 24 precedes Cell 25). The plan text acknowledges this and explains it as a deliberate choice (Beat 2 placed after concept intro but before broken code). This ordering deviates from strict Beat 1 -> Beat 2 -> Beat 3 -> Beat 4 sequence. Flag for review.
- [ISSUE] Section 4 Four-Beat ordering: Cell 24 (Beat 2 diagram) appears BEFORE Cell 25 (Beat 1 broken code). Per the mandatory arc definition, Beat 1 (broken code that fails) must come BEFORE Beat 2 (diagram). Students should feel the pain before seeing the explanation. Recommend swapping: Cell 24 and Cell 25 so broken code runs first, then the diagram.
- [PASS] Section 5 (Hub upload): Beat 1 = Cell 33 (push without auth), Beat 3 = Cell 34 (full working pattern). Beat 2 (diagram) is missing for this section -- but there are only 2 diagrams total (correct quota). Section 5 is a short capstone section; the plan treats Beat 4 as Lab 3. Acceptable since the 2-diagram quota is already met by Sections 1 and 4.
- [PASS] Section 3 (datasets): Beat 1 = Cell 20 (wrong split / DatasetDict mistake), Beat 3 = Cell 21 (correct loading). Beat 2 diagram missing for this concept but Section 3 is a supporting section without its own diagram (both diagrams are allocated to Sections 1 and 4). Acceptable.

#### Safety-nets

- [PASS] Lab 1 safety-net: Cell 12 sets zs_info, zs_pipeline_tag, zs_tags. Present and correct.
- [PASS] Lab 2 safety-net: Cell 30 sets ner_tokenizer, ner_model, ner_inputs, ner_probs, ner_predictions. Present and correct.
- [PASS] Lab 3: Plan explicitly notes no safety-net needed because Lab 3 outputs do not feed downstream cells. Correct.
- [PASS] classifier (Cell 16) feeds Cell 22 (financial_phrasebank evaluation). No safety-net cell between 16 and 22, but classifier is set in a Beat 3 demonstration cell (not a lab), so no safety-net is required. This is acceptable.
- [PASS] tokenizer and model (Cell 26) feed Cell 34 (push_to_hub demo). These are instructor demo cells, not lab output variables, so no safety-net required. Acceptable.

#### Diagrams

- [PASS] Exactly 2 diagrams found.
- [PASS] Diagram 1: slug=hf-hub-ecosystem, path=plans/topic_5/diagrams/hf-hub-ecosystem.mmd, placed at Cell 7 with <!-- DIAGRAM: ... --> placeholder and [View diagram] link. Correct format.
- [PASS] Diagram 2: slug=automodel-class-hierarchy, path=plans/topic_5/diagrams/automodel-class-hierarchy.mmd, placed at Cell 24 with <!-- DIAGRAM: ... --> placeholder and [View diagram] link. Correct format.

#### AI-tells

- [ISSUE] Cell 40 (End of Notebook Marker) contains a bare `---` separator line at the top of the content block. Per the rule, `---` bare separator lines are forbidden in cell bodies. The markdown cell starts with `---` as a horizontal rule. This needs to be removed or replaced with a plain heading or blank line.
- [PASS] No em dashes (--) used in cell bodies. Double hyphens are used in comments, which is acceptable. No Unicode em dash character U+2014 found.
- [PASS] No en dashes found.
- [PASS] No Unicode multiplication sign found.
- [PASS] No emojis found.
- [WARN] Cell 40 `---` separator: the CLAUDE.md rule says "no bare `---` separator lines." This cell contains exactly that at its opening. Needs fix.

#### Lab Tiers

- [PASS] Lab 1 (Cell 10): Tier 1 guided, 3 stubs, numbered steps in markdown header, verification cell. Correct.
- [PASS] Lab 2 (Cell 28): Tier 1 guided, 5 stubs, numbered steps in markdown header, verification cell. Correct.
- [PASS] Lab 3 (Cell 36): Tier 1 guided, 3 stubs, numbered steps in markdown header, verification cell. Correct.
- [PASS] No Tier 2 or Tier 3 labs. Correct for Topic 5.

#### Narrative

- [PASS] All sections reference Barclays Customer Support Intelligence System.
- [PASS] Hub exploration uses Barclays complaint routing and sentiment examples throughout.
- [PASS] COMPLAINT_TOKENS and COMPLAINT_LABELS carry over from Topic 4 by the same names.
- [PASS] SST-2 classifier evaluation in Cell 22 uses financial_phrasebank to motivate domain gap and transition to Topic 6a.

#### Discussion prompts

- [PASS] Cell 18: Discussion (3 min) between Sections 2 and 3. Three production-relevant questions covering latency, domain transfer, and cold-start. Correctly placed between major sections.
- [ISSUE] No discussion prompt between Sections 3 (datasets) and Section 4 (AutoModel). These are two major concept sections. A peer discussion after the domain gap evaluation (Cell 22) would reinforce the motivation for fine-tuning before moving to AutoModel internals. Minor gap -- not a hard blocker.

#### Homework Extensions

- [PASS] Lab 1 Homework Extension: Cell 13 contains `top_zero_shot_models` function stub with docstring.
- [PASS] Lab 2 Homework Extension: Cell 31 contains `entity_f1` function stub with docstring.
- [PASS] Lab 3 Homework Extension: Cell 38 contains `validate_model_card` function stub with docstring.

---

### topic_6a_full_finetuning.md

#### Four-Beat Arc

The plan provides an explicit Four-Beat Arc checklist table at the bottom:

| Section | Concept | Beat 1 | Beat 2 | Beat 3 | Beat 4 |
|---------|---------|--------|--------|--------|--------|
| 1 | Memory cost | Cell 5 | Cell 6 | Cells 8-9 | Lab 1 |
| 3 | Catastrophic forgetting | Cells 17-18 | Cell 19 | Cells 20-21 | Lab 2 |
| 4 | SageMaker GPU job | Cell 29 | -- | Cells 30-31 | Cells 32-34 |

- [PASS] Section 1: Beat 1 = Cell 5 (OOM calculation, CPU cost demo), Beat 2 = Cell 6 (diagram), Beat 3 = Cells 8-9 (Trainer demo), Beat 4 = Lab 1 (tokenize). All four beats present.
- [ISSUE] Section 2 (HuggingFace Trainer): Cells 7, 8, 9 cover the Trainer concept with Beat 3 working demos. However, there is NO Beat 1 (broken/failing code) for Section 2's core concept of "how to use Trainer correctly." The Section 2 markdown (Cell 7) introduces the Trainer. Cell 8 immediately shows correct code. The concept of Trainer misuse (e.g., wrong eval_strategy, wrong column names, forgetting no_cuda) is not demonstrated as a failure first. The plan's Beat 1 for Section 1 (Cell 5) is about memory cost, not Trainer API errors. This means the Trainer API concept skips Beat 1.
- [PASS] Section 3 (Catastrophic forgetting): Beat 1 = Cells 17-18 (baseline measurement + forgetting measurement), Beat 2 = Cell 19 (diagram), Beat 3 = Cells 20-21 (multitask mitigation), Beat 4 = Lab 2. All four beats present.
- [WARN] Section 4 (SageMaker GPU job): Beat 2 is listed as "--" (missing). The plan acknowledges this explicitly. A diagram for the SageMaker job flow (estimator -> job -> artifacts -> S3) would make the architecture explicit for students. Not a hard blocker since the two-diagram quota is already used, but worth noting.

#### Safety-nets

- [PASS] Lab 1 safety-net: Cell 12 sets tokenized_train and tokenized_val. Present and correct.
- [PASS] Lab 2 safety-net: Cell 24 sets lab2_model, lab2_args, lab2_trainer, lab2_metrics. Present and correct.
- [PASS] pre_train_sst2_acc (Cell 17) and post_train_sst2_acc (Cell 18): these are local variables used only within the forgetting demo cells (Cells 17-18), not consumed in student lab cells or downstream notebook sections. No safety-net required.
- [ISSUE] training_job_name (Cell 31) is consumed in Cells 32, 33, 34, and Cell 38 (variable inventory). If the kernel restarts or Cell 31 fails, all polling and log cells fail. No safety-net is described for training_job_name. Should add: `if 'training_job_name' not in dir(): training_job_name = "<paste your job name here>"` fallback cell after Cell 31.

#### Diagrams

- [PASS] Exactly 2 diagrams found.
- [PASS] Diagram 1: slug=full-finetuning-parameter-update, path=plans/topic_6a/diagrams/full-finetuning-parameter-update.mmd, Cell 6 with <!-- DIAGRAM: ... --> placeholder and [View diagram] link.
- [PASS] Diagram 2: slug=catastrophic-forgetting, path=plans/topic_6a/diagrams/catastrophic-forgetting.mmd, Cell 19 with <!-- DIAGRAM: ... --> placeholder and [View diagram] link.

#### AI-tells

- [PASS] No em dashes found in any cell body text.
- [PASS] No en dashes found.
- [PASS] No Unicode multiplication sign found.
- [PASS] No emojis found.
- [PASS] The compliance checklist at the bottom confirms: "No em dashes used (checked: 'to' used instead of '--')."
- [WARN] Cell 38 (Final Recap Code Cell) description mentions "prevents 3-markdown-chain" -- this is an internal note that should not appear in the actual cell. If this header comment appears in the built notebook's cell source, it would be an implementation note rather than student-facing content. Minor: ensure this text is only in the plan and not in the built cell.

#### Lab Tiers

- [PASS] Lab 1 (Cell 11): Tier 1 guided, 4 steps, YOUR CODE placeholders, verification cell (Cell 13). Correct.
- [PASS] Lab 2 (Cell 23): Tier 1 guided, 5 steps, YOUR CODE placeholders, verification cell (Cell 25). Correct.
- [PASS] No Tier 2 or Tier 3. Correct for Topic 6a (Tier 2 was used in Topic 4, Tier 3 reserved for Topic 7b).

#### Narrative

- [PASS] Uses Barclays Customer Support Intelligence System throughout.
- [PASS] COMPLAINT_TEXTS carried over from Topic 5 by the same name (Cell 3).
- [PASS] tokenizer and model reuse the same names from Topic 5.
- [PASS] Synthetic complaint dataset uses Barclays-domain language consistently.

#### Discussion prompts

- [PASS] Cell 15: Discussion (3-5 min) between Lab 1 and Section 3. Questions on data cost, labelling, and active learning. Correctly placed.
- [PASS] Cell 27: Discussion (3-5 min) between Lab 2 and Section 4. Questions on forgetting, degradation detection, and multi-task architecture. Correctly placed.

#### Homework Extensions

- [PASS] Lab 1 Homework Extension: Cell 14 contains multilingual tokenizer comparison task.
- [PASS] Lab 2 Homework Extension: Cell 26 contains WeightedRandomSampler investigation task.

---

### topic_6b_transfer_learning.md

#### Four-Beat Arc

The plan's checklist at the bottom confirms: Beat 1 (two broken code cells), Beat 2 (two diagram placeholders), Beat 3 (four working demo cells), Beat 4 (capstone + lab).

- [PASS] Beat 1: Cell 6 (wrong LR for frozen head -- lr=5e-6) and Cell 7 (full fine-tune CPU time projection). Both run and demonstrate failure. Correct.
- [PASS] Beat 2: Cell 8 contains the first diagram placeholder (transfer-learning-arch). Correctly placed after the two Beat 1 failure cells.
- [PASS] Beat 3: Cells 10, 11, 12, 14 show model inspection, freezing, gradient verification, and train.py preview. Correct.
- [ISSUE] Beat 2 for Diagram 2 (tl-vs-finetuning-comparison) is placed at Cell 15 as a PRINT STATEMENT, not as a proper markdown cell with a <!-- DIAGRAM: --> placeholder. Cell 15 is a CODE cell that prints the diagram reference string:
  ```python
  print("<!-- DIAGRAM: Accuracy vs epochs comparison ... -->")
  print("[View diagram](../../plans/topic_6b/diagrams/tl-vs-finetuning-comparison.mmd)")
  ```
  A diagram placeholder must be in a MARKDOWN cell, not in a print statement inside a code cell. The plan then places the actual markdown diagram cell at Cell 34 (wrap-up section). Cell 34 has the correct markdown format. The issue is that the Beat 2 diagram for the accuracy comparison appears in the wrap-up section (Cell 34), not near the Beat 3 demo that introduces the concept (Section 2 / Cells 9-15). This is a structural gap: the diagram explaining WHY transfer learning converges faster is not shown before the working demo of the capstone.
- [ISSUE] Section 3 (Capstone): The capstone occupies Cells 16-23 (launch estimator, retrieve output, discussion, deploy endpoint, write inference.py, test endpoint, cleanup). Lab 6b comes after (Cells 24-31). There is no Beat 1 broken code for the SageMaker deployment concept itself -- the capstone jumps directly to working code. Acceptable because the Beat 1 failures were placed in Cells 6-7 (before the entire notebook's concept arc), and the capstone is a Beat 4 activity (lab-equivalent). However, the plan's checklist only lists Beat 1 for Section 1 (naive failures), not for the SageMaker/deployment concept in Section 3. Minor structural note.

#### Safety-nets

- [PASS] Lab 6b Step 1 safety-net: Cell 26 sets test_raw and test_encoded. Present and correct.
- [PASS] Lab 6b Steps 2+3 safety-net: Cell 29 sets model_lab, trainable, all_logits, all_labels, accuracy. Present and correct.
- [ISSUE] The estimator variable (Cell 17) is consumed in Cell 18 (job output) and Cell 33 (comparison table). If Cell 17 fails or the job has not been submitted, Cells 18 and 33 will fail. No fallback is described. Should note: if estimator.fit() was not run, provide a way to set `estimator.model_data` manually using an S3 URI.
- [ISSUE] training_job_name (Cell 18) is consumed in Cell 18 itself (boto3 describe) -- not propagated beyond. But the comparison table in Cell 33 uses `estimator.model_data` which requires estimator to be set. This is a dependency without a safety-net for the demo cells (not lab cells), so it is lower priority but should be documented.
- [ISSUE] The inference.py file is written by Cell 21, which must run BEFORE Cell 20 (which deploys using `entry_point="inference.py"`). But the plan cell order is: Cell 20 (deploy) then Cell 21 (write inference.py). This is a sequencing error: inference.py does not exist when the deployment cell runs. The deploy cell (Cell 20) references `entry_point="inference.py"` but the file is written in Cell 21. Fix: swap Cells 20 and 21 so the inference.py write happens before the deploy call.

#### Diagrams

- [FAIL] Diagram 2 (tl-vs-finetuning-comparison) is not properly placed as a Beat 2 markdown cell near the concept introduction. Its markdown placeholder appears at Cell 34 (wrap-up) and as a print statement in Cell 15 (code). Neither constitutes a valid Beat 2 placement.
- [PASS] Diagram 1 (transfer-learning-arch): Cell 8 contains a proper markdown <!-- DIAGRAM: ... --> placeholder with [View diagram] link. Correct.
- [PASS] Total diagram count is 2 (one proper in Cell 8, one in wrap-up Cell 34). The quota of exactly 2 is met, but Diagram 2's placement is not ideal (it should be near the Beat 3 concept demo, not in the wrap-up).
- [ISSUE] Diagram 2 should be moved from the wrap-up Cell 34 to somewhere around Cell 9 (after the concept intro and before the working demo cells), to serve as a proper Beat 2 for the accuracy comparison concept.

#### AI-tells

- [PASS] No em dashes found.
- [PASS] No en dashes found.
- [PASS] No Unicode multiplication sign found.
- [PASS] No emojis found.
- [WARN] Cell 15 contains a bare `---` inside a print statement (the DIAGRAM placeholder is printed as a string, not in markdown). This is not the same as a markdown cell `---` separator, but the string `<!-- DIAGRAM: ... -->` inside a print statement is conceptually wrong regardless of AI-tells. See diagram issue above.

#### Lab Tiers

- [PASS] Lab 6b (Cells 24-31): Tier 1 guided. Three steps with numbered action items, YOUR CODE placeholders, safety-nets, verification (Cell 30), stretch, and Homework Extension (Cell 31).
- [PASS] No Tier 2 or Tier 3. Correct for Topic 6b.

#### Narrative

- [PASS] Uses Barclays Customer Support Intelligence System throughout.
- [PASS] Carries forward sess, role, bucket, region from Topic 6a by the same names.
- [PASS] Lab endpoint test in Cell 22 uses Barclays-style complaint text samples.
- [PASS] Discussion cell (Cell 19) is framed around Barclays weekly retrain scenarios.

#### Discussion prompts

- [PASS] Cell 19: Discussion (3 min) framed as Barclays weekly retrain decision. Three questions on transfer learning vs full fine-tuning vs LoRA. Correctly placed between capstone demo and lab section.
- [PASS] Cell 35: Peer discussion printed from a code cell (using print statements). This is an unusual pattern -- discussions should be in markdown cells, not printed from code cells. However, the content is correct and the questions are relevant. Minor: recommend converting Cell 35 to a markdown cell.

#### Homework Extensions

- [PASS] Lab 6b Homework Extension: Cell 31 contains partial fine-tuning task (freeze_layers argument, plot accuracy vs trainable parameter count). Present.

---

### topic_7a_lora_ffn.md

#### Four-Beat Arc

The plan includes an explicit Four-Beat Arc Verification table at the bottom.

- [PASS] LoRA motivation concept: Beat 1 = Cells 5-6 (delta_W explosion, rank=d failure), Beat 2 = Cell 7 (lora-decomposition diagram), Beat 3 = Cell 8 (LoraLayer from scratch), Beat 4 = Cells 9-13 (Lab 1). All four beats present.
- [PASS] LoRA on FFN concept: Beat 1 shared with LoRA motivation (the plan notes this explicitly), Beat 2 = Cell 14 (parameter comparison diagram), Beat 3 = Cells 15-16 (pre-train FFN, replace layers), Beat 4 = Cells 19-23 (Lab 2). All four beats present.
- [ISSUE] The plan states "Beat 1 shared with above" for the LoRA on FFN concept. The Beat 1 cells (5-6) are separated from the FFN application section (Section 2, starting at Cell 14) by 8 cells. Students may not connect the Beat 1 failure to the Section 2 concept. A brief reminder at the start of Section 2 (Cell 14 markdown) referencing Cells 5-6 would strengthen the arc. The current Cell 14 does not reference Cells 5-6 directly.
- [WARN] Cell 14 is a markdown cell that contains the Section 2 header AND the second diagram placeholder (lora-parameter-comparison). This means Beat 2 for the FFN concept (the diagram) is embedded inside the section header markdown. When built, the diagram placeholder will appear in the same cell as the explanatory text. This is structurally acceptable but not ideal -- Beat 2 is typically a standalone diagram cell. Minor.

#### Safety-nets

- [PASS] Lab 1 safety-net: Cell 11 falls back to LoraLayerStudent = LoraLayer. Present. However, the safety-net check is fragile: it instantiates LoraLayerStudent(nn.Linear(4, 8)) and checks if .lora_A is None. If a student does a partial implementation where __init__ runs but lora_A is set incorrectly, the safety-net may not trigger. This is a minor robustness issue in the safety-net check, not a missing safety-net.
- [PASS] Lab 2 safety-net: Cell 21 replaces replace_fc_with_lora_student with a lambda fallback. Present.
- [ISSUE] lora_model (Cell 16) is consumed in Cell 18 (MNIST fine-tuning), Cell 22 (Lab 2 verification uses trainable_lora which depends on lora_model), and Cell 37 (summary). If Cell 16 fails, all downstream cells fail. No safety-net for lora_model is described. The pretrained_model (from Cell 15) and lora_model are part of the instructor demo flow, not a lab, so a safety-net is less critical. But given that Cell 15 takes ~3 minutes to train on CPU, a failure would be disruptive. Recommend adding a note: "If Cell 15 times out, re-run with PRETRAIN_EPOCHS=2."
- [ISSUE] trainable_lora (Cell 16) is consumed in Cell 22 (Lab 2 verification ratio check) and Cell 37 (summary). If Cell 16 is never run, Cell 22's assertion fails. No safety-net for trainable_lora.
- [ISSUE] training_job_name (Cell 31) is consumed in Cell 32 (polling) and Cell 37 (summary). No safety-net described. Same issue as topic_6a and topic_6b.

#### Diagrams

- [PASS] Exactly 2 diagrams found.
- [PASS] Diagram 1: slug=lora-decomposition, path=plans/topic_7a/diagrams/lora-decomposition.mmd, Cell 7 markdown with <!-- DIAGRAM: ... --> and [View diagram] link. Correct format.
- [PASS] Diagram 2: slug=lora-parameter-comparison, path=plans/topic_7a/diagrams/lora-parameter-comparison.mmd, Cell 14 markdown with <!-- DIAGRAM: ... --> and [View diagram] link. Correct format.

#### AI-tells

- [PASS] No em dashes found.
- [PASS] No en dashes found.
- [PASS] No Unicode multiplication sign found.
- [PASS] No emojis found.
- [PASS] The compliance checklist confirms AI-tells check passed.

#### Lab Tiers

- [PASS] Lab 1 (Cells 9-13): Tier 1 guided, 7 steps (Steps 1-7 in comments), YOUR CODE placeholders, verification cell. Correct.
- [PASS] Lab 2 (Cells 19-23): Tier 1 guided, 6 steps (fc1-fc6 replacement), YOUR CODE placeholders, verification cell. Correct.
- [PASS] No Tier 2 or Tier 3. Correct for Topic 7a.

#### Narrative

- [PASS] Uses Barclays Customer Support Intelligence System throughout.
- [PASS] Framing: Barclays hosts dozens of NLP models; LoRA adapters let you deploy one frozen base model with task-specific adapters. Correct Barclays motivation.
- [PASS] Flan-T5 capstone uses synthetic Barclays complaint-summary pairs as training data.
- [PASS] Discussion prompts 1 and 2 frame questions in Barclays production context.
- [WARN] The FFN scratch demo uses FashionMNIST/MNIST datasets, not Barclays data. The plan explicitly acknowledges this as a metaphor ("we keep that pair for the scratch demo ... but frame it as a metaphor"). The Barclays connection is maintained through framing text, not the actual data. This is acceptable given the pedagogical goal (verify math before applying to real LLM), but the lab instructions (Cells 9 and 19) should explicitly remind students that the FFN demo is a proxy for the real Barclays Flan-T5 capstone.

#### Discussion prompts

- [PASS] Cell 17: Discussion (3 min) on rank trade-offs (rank=1 vs rank=128, multi-task adapter storage, why lower rank generalises). Correctly placed between the FFN LoRA application and the MNIST fine-tuning.
- [PASS] Cell 26: Discussion (3 min) on LoRA in production (storage savings, deployment pipeline differences, alpha risks). Correctly placed between Section 3 (rank heuristics) and Section 4 (PEFT capstone).

#### Homework Extensions

- [PASS] Lab 1 Homework Extension: Cell 13 contains init comparison task (zeros/Normal permutations).
- [PASS] Lab 2 Homework Extension: Cell 23 contains rank sweep task (r=1,4,8,16, plot accuracy vs rank).

---

### topic_7b_peft_lora_distilbert.md

#### Four-Beat Arc

- [PASS] Section 1 (PEFT library LoRA): Beat 1 = Cell 4 (naive manual LoRA fails to scale -- partial injection, head still trainable), Beat 2 = Cell 5 (DIAGRAM peft-methods-comparison), Beat 3 = Cells 6-7 (PEFT library demo + forward pass verification), Beat 4 = Lab 1 (Cells 8-11). All four beats present.
- [PASS] Section 2 (QLoRA): Beat 1 = Cell 15 (bitsandbytes fails on CPU -- RuntimeError), Beat 2 = Cell 16 (DIAGRAM qlora-architecture), Beat 3 = Cell 17 (QLoRA code walkthrough), Beat 4 = covered by the capstone (Section 4 --peft_method=qlora path). All four beats nominally present.
- [ISSUE] Section 2 Beat 4 is missing an explicit lab for QLoRA. The plan says the capstone in Section 4 covers QLoRA as a capstone option. However, the Tier 3 capstone (Cell 28-29) is open-ended with no required QLoRA step. Students who choose LoRA in the capstone will never have a Beat 4 practice moment for QLoRA. This is a structural gap in the QLoRA four-beat arc. Recommend: add a guided question or short guided cell after Cell 17 that asks students to compare the memory footprint numbers (e.g., "calculate what QLoRA saves for a 7B model vs DistilBERT").
- [PASS] Section 3 (Soft Prompts): Beat 1 = Cell 19 (shape mismatch with num_virtual_tokens=200), Beat 2 implied by the diagram that is placed in Cell 5 (PEFT methods comparison shows soft prompts panel). No dedicated Beat 2 diagram cell for soft prompts alone -- Diagram 1 covers all three methods. Beat 3 = Cells 20-21 (working demo + comparison table). Beat 4 = addressed in the capstone (students can choose prefix tuning). Acceptable given the shared diagram.
- [WARN] Section 3 has no dedicated Beat 4 lab (only the open-ended Tier 3 capstone). This means soft prompts have Beat 1, Beat 2 (via shared diagram), Beat 3, but no dedicated Tier 1 guided lab practice. Students who are slow may not attempt soft prompts in the capstone. Minor gap.

#### Safety-nets

- [PASS] Lab 1 safety-net: Cell 10 sets peft_model_r16 using the fallback construction. Present and correct.
- [PASS] Tier 3 capstone: no safety-net required (open-ended, no downstream dependency). Correct.
- [PASS] peft_model (Cell 6) is consumed in Cell 11 (Lab 1 verification) and Cell 21 (parameter comparison table). Cell 10 (safety-net for Lab 1) only creates peft_model_r16, not peft_model itself. If Cell 6 fails, Cell 11 and 21 fail. However, Cell 6 is a Beat 3 instructor demo cell (not a student lab), so its failure would require re-running the demo. No safety-net is needed for instructor demo cells.
- [PASS] prefix_model (Cell 20) is consumed in Cell 21 (parameter comparison). No safety-net, but Cell 20 is an instructor demo cell. Acceptable.
- [ISSUE] training_job_name (Cell 25) is consumed in Cells 26, 27. If Cell 25 fails or is re-run with a new job name, Cells 26-27 lose context. No safety-net described (same pattern as all other topics with estimator.fit). Recommend the same fallback note pattern used elsewhere.

#### Diagrams

- [PASS] Exactly 2 diagrams found.
- [PASS] Diagram 1: slug=peft-methods-comparison, path=plans/topic_7b/diagrams/peft-methods-comparison.mmd, Cell 5 markdown with <!-- DIAGRAM: ... --> and [View diagram] link. Correct format.
- [PASS] Diagram 2: slug=qlora-architecture, path=plans/topic_7b/diagrams/qlora-architecture.mmd, Cell 16 markdown with <!-- DIAGRAM: ... --> and [View diagram] link. Correct format.

#### AI-tells

- [PASS] No em dashes found.
- [PASS] No en dashes found.
- [PASS] No Unicode multiplication sign found.
- [PASS] No emojis found.
- [PASS] The plan's AI-tells check section confirms PASS.

#### Lab Tiers

- [PASS] Lab 1 (Cells 8-12): Tier 1 guided. 4 steps (load model, define config, wrap, count params), YOUR CODE placeholders, verification cell (Cell 11), safety-net (Cell 10). Correct.
- [PASS] Tier 3 capstone (Cells 28-29): Open-ended, function signature + docstring + pass only. No verification cell (by design). Correctly placed on Topic 7b, the last topic of Day 2. Correct.
- [PASS] Lab tier note in the checklist at bottom: "Topic 7b: Tier 1 Lab 1 + Tier 3 Capstone -- CORRECT. Day 2 gets exactly ONE Tier 3 lab, which must be the last topic. Topic 7b is last. PASS."

#### Narrative

- [PASS] Uses Barclays Customer Support Intelligence System throughout.
- [PASS] Carries forward lora_r from Topic 7a explicitly (Cell 2: `lora_r = 8   # rank -- carried forward from 7a`).
- [PASS] All complaint examples use Barclays-domain language (cards, ATM, online banking).
- [PASS] Discussion prompts (Cells 13 and 22) frame questions around Barclays production scenarios.
- [PASS] Uses financial_phrasebank as the complaint classification proxy dataset -- consistent with Topic 5 which also evaluated this dataset.

#### Discussion prompts

- [PASS] Cell 13: Discussion (3 min) on LoRA adapter storage, versioning, rollback, and base model updates. Correctly placed between Lab 1 and Section 2.
- [PASS] Cell 22: Discussion (3 min) on QLoRA team access implications, soft prompt management for 20 complaint categories, and when to choose soft prompts over LoRA. Correctly placed between Section 3 and the capstone.

#### Homework Extensions

- [PASS] Lab 1 Homework Extension: Cell 12 contains rank vs parameter count plot task (r=2,4,8,16,32).
- [PASS] Tier 3 capstone stretch (Cell 30): two-job comparison (LoRA vs QLoRA) and metric logging as CloudWatch custom metric. Homework Extension: merge_and_unload() research task.

---

## Cross-Topic Issues

### Tier 2 Lab Quota (Day 2 total: Topics 5, 6a, 6b, 7a, 7b)

The rule requires EXACTLY ONE Tier 2 lab across all Day 2 topics.

- Topic 5: 0 Tier 2 labs. PASS.
- Topic 6a: 0 Tier 2 labs. PASS.
- Topic 6b: 0 Tier 2 labs. PASS.
- Topic 7a: 0 Tier 2 labs. PASS.
- Topic 7b: 0 Tier 2 labs. PASS.

[ISSUE] The Day 2 research plans describe ZERO Tier 2 labs across all five topics. The rules require EXACTLY ONE Tier 2 lab per day. All plans note "Tier 2 was used in Topic 4" but Topic 4 is a Day 2 topic in this curriculum (the plan overview says Day 2 = Topics 5, 6a, 6b, 7a, 7b). If Topic 4 is actually on Day 2 (as a preceding topic before Topic 5), and its Tier 2 counts toward the Day 2 quota, then the quota is satisfied. However, the five files audited here (Topics 5-7b) contain no Tier 2 lab. If Topic 4 is considered Day 1, then Day 2 is missing its required Tier 2 lab entirely. This needs clarification. The plans consistently note "Tier 2 was used by Topic 4" -- if Topic 4 is part of Day 2, the quota is met. If not, a Tier 2 lab must be added to one of Topics 5-7a.

### Tier 3 Lab Placement

- [PASS] Tier 3 appears only in Topic 7b (last topic of Day 2). PASS.
- [PASS] No Tier 3 lab appears in Topics 5, 6a, 6b, or 7a. PASS.

### training_job_name Safety-Net Pattern

- [ISSUE] ALL four topics that launch SageMaker training jobs (6a, 6b, 7a, 7b) use the same pattern: `training_job_name = estimator.latest_training_job.name` immediately after `estimator.fit(wait=False)`. None of the four plans include a safety-net or fallback for the case where:
  (a) The kernel is restarted after launching the job.
  (b) The estimator.fit() call is re-run with a new job, overwriting training_job_name.
  (c) A student opens the notebook after the instructor launched the job.
  All polling, log, and artifact cells depend on training_job_name being set. A one-line fallback should be added after the fit() cell in all four notebooks:
  ```python
  # If you restarted the kernel after the job was launched, paste the job name here:
  # training_job_name = "topic6a-..."  # uncomment and fill in
  ```

### Variable Name Continuity Across Topics

- [PASS] device, set_seeds(), sess, role, bucket, region are consistently named across all five topics.
- [PASS] tokenizer name is reused across Topics 5, 6a, 6b. In Topics 7a and 7b the tokenizer is local to the training script.
- [WARN] Topic 6a introduces compute_metrics as a local function and notes it is "same signature used in Topic 7a." Topic 7a does NOT import or reuse compute_metrics from 6a -- it defines its own in train.py. The name is the same but the function is not shared across kernel sessions. This is correct behavior (each notebook is self-contained) but the variable continuity documentation could be clearer.

### Peer Discussion Prompts Between Topics

- All five topics have at least one discussion prompt. All are between major sections. No cross-topic discussion gaps found.

### Homework Extension Coverage

- All five topics have Homework Extensions after every lab. No gaps found.

---

## Priority Fixes

Ranked by severity (blocker first, then structural, then polish):

### P0 - Hard Blockers (must fix before building)

1. **topic_6b: Cell sequencing error -- inference.py written AFTER deploy call**
   Cell 20 (deploy using entry_point="inference.py") comes before Cell 21 (write inference.py). The file does not exist when Cell 20 runs. Swap Cells 20 and 21.

2. **topic_6b: Diagram 2 is a print statement, not a markdown cell**
   The second diagram placeholder (tl-vs-finetuning-comparison) appears inside a print() call in Cell 15 (code), not in a markdown cell. It is then placed again in Cell 34 (wrap-up markdown). Move the proper Beat 2 diagram markdown cell to after Cell 13 (transition to remote training) or at the start of Section 2, and remove the print() version from Cell 15.

3. **Cross-topic: Tier 2 lab quota unclear**
   Day 2 has ZERO Tier 2 labs in Topics 5-7b. If Topic 4 is Day 2, the quota is satisfied. If Topic 4 is Day 1, one of Topics 5-7a must have a Tier 2 lab added. Clarify which day Topic 4 belongs to before building.

### P1 - Structural Issues (fix before building, but not hard blockers)

4. **topic_5: Beat 1/Beat 2 ordering in Section 4**
   Cell 24 (Beat 2 diagram) comes before Cell 25 (Beat 1 broken code). The mandatory arc requires Beat 1 before Beat 2. Swap the cells.

5. **topic_6a: No Beat 1 for the Trainer API concept**
   Section 2 introduces HuggingFace Trainer (Cells 7-9) with working code immediately. Add a Beat 1 cell showing a common Trainer mistake (e.g., wrong eval_strategy name, forgetting to rename label -> labels, or missing no_cuda=True on CPU) that fails, then fix it in Beat 3.

6. **topic_6a: training_job_name has no safety-net**
   Add a one-line fallback cell after Cell 31 so students who restart the kernel can paste in their job name.

7. **topic_6b: training_job_name / estimator has no safety-net**
   Add one-line fallback cell after Cell 17 (estimator.fit()).

8. **topic_7a: training_job_name has no safety-net**
   Add one-line fallback cell after Cell 31 (estimator.fit()).

9. **topic_7b: training_job_name has no safety-net**
   Add one-line fallback cell after Cell 25 (estimator.fit()).

10. **topic_7a: lora_model and trainable_lora have no safety-net**
    Cells 16 and 22 depend on lora_model and trainable_lora from Cell 16. Add a fallback: if lora_model is None, rebuild it from pretrained_model (or a fresh FFNModel) with replace_fc_with_lora.

### P2 - Polish and Minor Issues (fix before or during build)

11. **topic_5: Bare `---` separator in Cell 40**
    Remove the `---` horizontal rule at the top of the End of Notebook Marker markdown cell.

12. **topic_5: Missing discussion prompt between Sections 3 and 4**
    Add a 3-minute discussion cell after Cell 22 (financial_phrasebank evaluation) to close the gap before AutoModel section.

13. **topic_6b: Cell 35 discussion is a code print cell, not markdown**
    Convert Cell 35 (peer discussion questions) from a code cell with print() calls to a markdown cell.

14. **topic_7b: QLoRA four-beat arc missing Beat 4 practice**
    After Cell 17 (QLoRA walkthrough), add a short guided question cell or calculation exercise so students practice thinking about QLoRA memory savings, bridging the QLoRA concept to the capstone.

15. **topic_7a: Lab 2 instructions do not reference Cells 5-6 shared Beat 1**
    Cell 14 (Section 2 header) should explicitly point back to Cells 5-6 as the Beat 1 failure that motivates the FFN LoRA application. Currently the connection is implicit.

16. **topic_6b: Diagram 2 placement is in the wrap-up (Cell 34)**
    Even after fixing the print() issue (P0 item 2), ensure the accuracy comparison diagram appears near the Beat 3 demo (Cell 15 area), not only in the wrap-up. The wrap-up placement of a concept diagram is too late to prime students during learning.
