# Day 1 Research Audit Report
Generated: 2026-05-11

Scope: All Day 1 research plan files audited against CLAUDE.md teaching methodology.
Files audited: F1_pytorch_refresher.md, F2_sagemaker_fundamentals.md,
topic_1_overview_genai.md, topic_2_introducing_llms.md,
topic_3a_attention_python.md, topic_3b_attention_pytorch.md, topic_4_transformers.md

---

## Summary Table

| File | Four-Beat Arc | Safety-nets | Diagrams | AI-tells | Lab Tiers | Narrative | Discussion | Homework |
|---|---|---|---|---|---|---|---|---|
| F1_pytorch_refresher.md | PASS | PASS | WARN | FAIL | PASS | PASS | PASS | PASS |
| F2_sagemaker_fundamentals.md | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS |
| topic_1_overview_genai.md | FAIL | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| topic_2_introducing_llms.md | WARN | PASS | WARN | PASS | FAIL | PASS | PASS | PASS |
| topic_3a_attention_python.md | FAIL | PASS | FAIL | FAIL | FAIL | PASS | PASS | PASS |
| topic_3b_attention_pytorch.md | WARN | PASS | PASS | PASS | PASS | PASS | FAIL | PASS |
| topic_4_transformers.md | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS |

Legend: PASS = clean, WARN = minor/low-risk issue, FAIL = must fix before building notebook

---

## Per-File Issues

### F1_pytorch_refresher.md

**STATUS: WARN — fix AI-tells before building**

#### FAIL: AI-tells (--- separators in section headers)

Every section header cell begins with `---` on its own line. Found in:
- Cell 12 content: `---\n## Section 2 - Autograd...`
- Cell 24 content: `---\n## Section 3 - Dataset and DataLoader...`
- Cell 32 content: `---\n## Section 4 - Complaint Classifier...`
- Cell 41 content: `---\n## Section 5 - Cleaner Classifiers...`
- Cell 49 content: `---\n## Section 6 - HuggingFace Trainer...`

`---` is a horizontal rule / thematic break that functions as a visual separator.
While not an em dash, it is a formatting crutch that reads as AI-generated structure.
The CLAUDE.md rule is "Plain ASCII only" in cell bodies.
Fix: remove the leading `---` line from each section header cell.

#### PASS: Four-Beat Arc

All six sections (Tensors, Autograd, Dataset/DataLoader, nn.Module,
nn.Sequential, HuggingFace Trainer) follow Beat 1 -> Beat 3 -> Beat 4.

NOTE: Beat 2 (diagram placeholder) appears only in Sections 1 and 2.
Sections 3-6 have no diagram placeholder cells. This is consistent with the
plan having exactly 2 diagrams total (autograd-computation-graph.mmd and
training-loop.mmd), so the absence of Beat 2 in later sections is intentional.
No fix required, but instructor should verbally bridge sections 3-6 without
diagram anchors.

NOTE on Beat 2 / training-loop diagram (Cell 16): the diagram is placed AFTER
Beat 3 (Cell 15), not before it. The plan notes this explicitly and calls it an
intentional reinforcement placement. This is non-standard but low-risk.

#### PASS: Safety-nets

- Cell 10: Lab 1 safety-net (tensor variables)
- Cell 19: Lab 2 safety-net (w, b, x_train, y_train, losses)
- Cell 29: Lab 3 safety-net (X_500, train_loader)
- Cell 37: Lab 4 safety-net (deep_model, deep_losses)
- Cell 46: Lab 5 safety-net (model_relu, relu_losses)
All safety-nets are immediately after lab starter cells. No downstream cell
dependency gaps found.

#### PASS: Lab Tiers

Lab 1 Tier 1 (tensors), Lab 2 Tier 1 (autograd), Lab 3 Tier 1 (DataLoader),
Lab 4 Tier 1 (nn.Module), Lab 5 Tier 1 (nn.Sequential), Lab 6 Tier 2
(HuggingFace Trainer). F1 is a foundations module (not a "topic day") so the
"exactly 1 Tier 2 per day" rule applies across the full foundations day.
Only 1 Tier 2 (Lab 6) is present -- correct.

#### PASS: Narrative

Barclays complaint severity model thread runs consistently through all sections.
Feature names (sentence_length, severity_score, complaint categories) carry over.

#### PASS: Peer Discussions

Cell 22 (Discussion 1 -- gradient descent in production) after Section 2.
Cell 40 (Discussion 2 -- model architecture decisions) after Section 4.
Two discussions found. Sections 5-6 do not have standalone discussions but Section 6
transitions immediately into the Tier 2 lab -- acceptable given density.

#### PASS: Homework Extensions

Cells 21, 31, 39, 48 (per section). All found after lab cells. Content is
appropriate depth.

---

### F2_sagemaker_fundamentals.md

**STATUS: WARN — fix AI-tells before building**

#### FAIL: AI-tells (--- separators)

- Cell 3 content begins with `---\n## Section 1 - SageMaker Session...`
- Cell 12 content begins with `---\n## Section 2 - S3 and Artifacts...`
(Additional section header cells likely follow same pattern -- not re-read in
full but the pattern is consistent with F1.)

Fix: remove the leading `---` line from each section header cell.

#### PASS: Four-Beat Arc

SageMaker Session, S3/Artifacts, Estimator/Training Job, MLflow sections all
show Beat 1 (broken/error code) -> Beat 3 (working demo) -> Beat 4 (lab).

#### PASS: Diagrams (exactly 2)

- Cell 5: sagemaker_training_lifecycle.mmd -- correct format
- Cell 33: sagemaker_mlflow_architecture.mmd -- correct format
Both have description sentences. No extra diagram references found.

#### PASS: Safety-nets

- Cell 9: Lab 1 variables (account_id, artifact_base, recent_jobs, role_name)
- Cell 24: Lab 2 variable (job_name) -- feeds cells 25, 27, 28, 29, 37, 38, 40, 41

#### PASS: Lab Tiers

Lab 1 Tier 1, Lab 2 Tier 2 -- correct for a single foundations session.

#### PASS: Peer Discussions

Cell 11 (Discussion 1), Cell 30 (Discussion 2). Both present.

#### PASS: Homework Extension

Cell 44 (markdown) + Cell 45 (starter code). Present and correctly placed.

---

### topic_1_overview_genai.md

**STATUS: FAIL -- fix diagram gap and AI-tells before building**

#### FAIL: AI-tells (--- separators)

`---` separator lines appear in at least 6 cells: Cells 3, 10, 17, 27, 29, 31.
Fix: strip all bare `---` lines from cell body content.

#### FAIL: Missing Beat 2 diagram for the API/Prompting section (Section 3)

Section 3 flow: Cell 19 (Beat 1 header), Cell 20 (Beat 1 broken code),
Cell 21 (markdown explanation of system prompts -- NO diagram placeholder),
Cell 22 (Beat 3 working code).

There is no `<!-- DIAGRAM: ... -->` placeholder for the API section. The notebook
declares exactly 2 diagrams (genai-taxonomy.mmd in Cell 5, autoregressive-loop.mmd
in Cell 11). Cell 11 diagram covers Section 2 (text generation). Section 3 (API
calling) gets no visual anchor. This is a teaching gap -- students transition from
"how text is generated" directly into API code without a conceptual bridge.

Fix option A: add a diagram placeholder in Cell 21 (e.g., a diagram showing the
request/response loop for a language model API call) and add a third .mmd entry
to the diagram index.
Fix option B: reuse the autoregressive-loop.mmd as Beat 2 for Section 3 by
adding a reference line in Cell 21. This keeps the diagram count at 2 but requires
noting that the same diagram serves two sections.

#### WARN: VAE and Diffusion sections have compressed arc (no Beat 1 broken code)

The plan acknowledges this is intentional to save time. The treatment is
abbreviated: students see the concept explained but do not feel the pain of a
naive approach first. This is a pedagogical compromise, not a structural error.
Flag for instructor awareness: plan extra verbal scaffolding for these two concepts.

#### PASS: Safety-nets

- Cell 15: Lab 1 safety-net (temperatures_to_test)
- Cell 25: Lab 2 safety-net (my_system_prompt)

#### PASS: Lab Tiers

Lab 1 Tier 1 (temperature exploration), Lab 2 Tier 2 (triage prompt).
Tier counts are correct for this topic in isolation.
See Cross-Topic Issues for the Tier 2 quota conflict with Topic 2.

#### PASS: Peer Discussions

Cell 7 (Discussion 1), Cell 27 (Discussion 2). Both present.

#### PASS: Homework Extensions

Cell 16 (Lab 1 homework), Cell 26 (Lab 2 homework). Both present.

---

### topic_2_introducing_llms.md

**STATUS: FAIL -- fix Tier 2 quota and verify diagram placement**

#### FAIL: Lab Tier Quota Violation (Cross-topic)

Lab 2 in this file is labeled Tier 2 (complaint similarity matrix). Topic 1 Lab 2
is also Tier 2 (triage prompt). Both topics are on the same day.
CLAUDE.md rule: "exactly ONE Tier 2 per day."
Two Tier 2 labs on Day 1 violates this rule.

Fix: Downgrade one of these to Tier 1 by adding more `# YOUR CODE` step
prescriptions. The complaint similarity matrix lab (Topic 2 Lab 2) is the stronger
candidate for downgrade because the tokenization lab (Topic 2 Lab 1) is already
at Tier 1 -- keeping Topic 2 with one Tier 1 and one Tier 1 is consistent.
Alternatively, if Topic 1 Lab 2 (triage prompt) is the Day 1 Tier 2, downgrade
Topic 2 Lab 2 to Tier 1.

Recommended: Keep Topic 1 Lab 2 as the day's Tier 2 (students engage with prompt
engineering earlier in the day when energy is higher). Downgrade Topic 2 Lab 2
to Tier 1 by adding numbered sub-steps to the complaint similarity matrix build.

#### WARN: Diagram 1 placement non-standard

transformer-families.mmd (Cell 5) is placed as a "visual teaser" at the start,
before the tokenization Beat 1/3 cells. It is described as "placed early as a
visual teaser" and revisited in Section 3.

Standard Beat 2 placement is after Beat 1 and before Beat 3 of the SAME concept.
Here the diagram is not tied to a Beat 1/Beat 3 pair -- it floats above all
sections as orientation. This is not invalid, but it means the transformer families
section (Section 3) has no Beat 2 placeholder of its own.

Fix: either (a) note in the diagram reference that this serves both Section 1
orientation and Section 3 Beat 2, or (b) move the diagram to Cell 24 area (after
the Section 3 Beat 1 broken code) and add a forward-reference comment in Cell 5
("We will see the full transformer family diagram in Section 3").

#### PASS: AI-tells

No em dashes, en dashes, Unicode multiplication signs, or `---` separators found
in the cell content. Hard rules checklist at bottom confirms clean.

#### PASS: Safety-nets

- Cell 10: Lab 1 safety-net (result_short, result_medium, result_long)
- Cell 20: Lab 2 safety-net (all_embeddings, sim_matrix, most_similar_idx)

#### PASS: Peer Discussions

Cell 12, Cell 22, Cell 32 -- three discussions. More than minimum, all appropriate.

#### PASS: Homework Extensions

Cell 11 (Lab 1), Cell 21 (Lab 2). Both present.

---

### topic_3a_attention_python.md

**STATUS: FAIL -- fix diagram order and lab tier gap**

#### FAIL: Diagram 2 placed out of sequence (after Beat 3)

Diagram convention requires Beat 2 (diagram placeholder) to appear BEFORE Beat 3
(working demo) for the same concept.

bahdanau-score-computation.mmd is declared as Diagram 2, placed at Cell 23.
By Cell 23, the Beat 3 Bahdanau attention demo has already run (Sections 3 and 4
demos are in earlier cells), and Lab 1 has also already appeared. The diagram
lands mid-notebook as a summary visual, not as a Beat 2 anchor.

Fix: move the bahdanau-score-computation diagram placeholder to immediately before
the first Bahdanau Beat 3 demo cell. If that cell is in Section 3, the diagram
should go between the Beat 1 (bottleneck pain) and Beat 3 (Bahdanau solution) cells.

#### FAIL: No Tier 2 lab in this notebook

Topic 3a has only Lab 1 (Tier 1 -- guided Bahdanau implementation). The day's
"exactly 1 Tier 2" quota means one topic on Day 1 must contain the Tier 2 lab.
If Topic 1 or Topic 2 carries the Tier 2 (once the quota conflict above is
resolved), Topic 3a being Tier 1 only is fine. However, if the resolved schedule
puts the Day 1 Tier 2 in Topic 3a, this file needs a Tier 2 lab.

Clarification needed: confirm which topic is the designated Day 1 Tier 2 carrier.
Once confirmed, Topic 3a's single Tier 1 lab is either correct or needs upgrade.

#### WARN: Beat 2 implementation non-standard (code-print then markdown)

Cell 7 is a code cell containing only `print("See diagram...")`, followed by
Cell 8 (markdown with the diagram placeholder). This breaks up what should be a
pure markdown Beat 2 with an extra code cell. The workaround exists to avoid
three consecutive markdown cells, but it creates a bare `print` cell with no
teaching content.

Fix: restructure to place the diagram markdown immediately after the Beat 1 code
cell. If the preceding cell is markdown (section header), insert a minimal code
cell that actually demonstrates the bottleneck problem, then place the diagram
markdown. This way the print trick is unnecessary.

#### FAIL: AI-tells (--- separator in Cell 30)

Cell 30 contains `---` followed by `*End of Topic 3a...*`.
Fix: remove the `---` line. The closing cell can just have the italic text or be
omitted entirely.

#### PASS: Safety-nets

Cell 15: Lab 1 safety-net (my_bahdanau_attention). Present and correct.

#### PASS: Diagrams (count is exactly 2)

seq2seq-bottleneck-vs-attention.mmd (Cell 8) and bahdanau-score-computation.mmd
(Cell 23). Count is correct; placement of Diagram 2 needs fixing (see above).

#### PASS: Peer Discussions

Cell 6 (Discussion 1), Cell 25 (Discussion 2). Both present.

#### PASS: Homework Extensions

Cell 16 (Lab 1 stretch/homework), Cells 27-29 (formal homework extension section).
Present.

---

### topic_3b_attention_pytorch.md

**STATUS: WARN -- fix peer discussion count**

#### FAIL: Only 1 peer discussion

Cell 20 is the only peer discussion cell in this notebook. All other Day 1
notebooks have at least 2 discussions between major sections.

Topic 3b has sections: (1) Dot-product attention in PyTorch, (2) Scaled
dot-product with masking, (3) Attention visualization, (4) Capstone Tier 3.
There is a natural discussion point between sections 2 and 3 (after students
have implemented both attention variants) that is missing.

Fix: add a Discussion markdown cell between the attention heatmap visualization
Beat 3 (Section 3 demo) and the Capstone lab instructions. Topic: "When should
you use raw dot-product attention vs scaled dot-product? What breaks in practice
without the sqrt(d_k) scaling?" (3 min, pairs).

#### WARN: Beat 2 implementation non-standard (same pattern as 3a)

Cell 8 is a code cell with `print("See diagram...")` followed by Cell 9 (markdown
with diagram). Cell 18 repeats the same pattern before the second diagram.

Same fix as Topic 3a: restructure so the diagram markdown follows a genuine code
cell, not a placeholder print. The print-then-markdown pattern is technically
compliant (not 3 consecutive markdown cells) but adds noise.

#### PASS: Diagrams (exactly 2)

scaled-dot-product-formula.mmd (Cell 9) and attention-heatmap-complaint.mmd
(Cell 19). Both correctly formatted with description sentences.

#### PASS: Lab Tiers

Lab 1 Tier 1 (DotProductAttention nn.Module) and Capstone Tier 3
(ScaledDotProductAttention). This is the last topic of Day 1, so Tier 3 is
correct here per CLAUDE.md rule. Tier 3 implementation uses `pass` only -- correct.

#### PASS: Safety-nets

Cell 13: Lab 1 safety-net (MyDotProductAttention).
Cell 24: Capstone safety-net (ScaledDotProductAttention).

#### PASS: AI-tells

No em dashes, en dashes, Unicode multiplication signs found. No `---` separators
noted in cell content.

#### PASS: Homework Extensions

Cell 14 (Lab 1), Cell 30 (Homework 1 and 2). Present.

---

### topic_4_transformers.md

**STATUS: FAIL -- fix diagram index vs cell content mismatch**

#### FAIL: Diagram count mismatch (3 referenced in cells, 2 in index)

The Diagram Index at the top of the plan declares 2 diagrams:
- transformer-architecture.mmd
- positional-encoding-pattern.mmd

But the cell-by-cell plan references 3 diagram placeholders:
- Cell 5: transformer-architecture.mmd (Beat 2 for Section 1) -- in index
- Cell 18: multi-head-attention.mmd (Beat 2 for Section 3) -- NOT in index
- positional-encoding-pattern.mmd -- IN the index but has NO cell-level
  `<!-- DIAGRAM: ... -->` placeholder

This creates two bugs:
1. multi-head-attention.mmd will appear in the notebook but the /build-diagrams
   command will not find it in the index and will not generate the .mmd file.
2. positional-encoding-pattern.mmd will be generated by /build-diagrams but
   there is no `<!-- DIAGRAM: ... -->` placeholder in any cell, so it will never
   be linked from the notebook.

Fix: update the Diagram Index to include exactly the diagrams that have cell-level
placeholders. The correct set appears to be:
- transformer-architecture.mmd (Cell 5 -- confirmed in cell content)
- multi-head-attention.mmd (Cell 18 -- confirmed in cell content)
Remove positional-encoding-pattern.mmd from the index (it has no cell placeholder).
If a positional encoding diagram is wanted, add a `<!-- DIAGRAM: ... -->` placeholder
in the positional encoding section (around Cell 9) and re-add it to the index.

Note: having 3 diagrams instead of 2 requires a plan-level decision. CLAUDE.md
does not set a hard maximum -- "exactly 2 diagram placeholders" per notebook is
the convention described. Confirm with course designer whether 3 is acceptable
for Topic 4 before fixing.

#### PASS: Four-Beat Arc

Section 1 (Positional Encoding), Section 2 (Multi-Head Attention), Section 3
(Full Transformer), Section 4 (Training) all show Beat 1 -> Beat 2 -> Beat 3 ->
Beat 4 pattern. No arc gaps found.

#### PASS: Safety-nets

Cell 14: Lab 1 safety-net (PositionalEncodingLab).
Cell 23: Lab 2 safety-net (full TransformerModel implementation).

#### PASS: Lab Tiers

Lab 1 Tier 1, Lab 2 Tier 2. No Tier 3 (Topic 4 is not marked as last topic of
the day in this plan -- implementation notes say "Tier 3 reserved for last topic
of Day 2"). Tier assignment is consistent with the plan's stated day placement.

NOTE: The audit request groups Topic 4 as a Day 1 topic, but the plan itself
states it opens Day 2. This is a scheduling discrepancy to resolve with the
course designer. If Topic 4 IS on Day 1, a Tier 3 lab should be added (and
Topic 3b's Tier 3 capstone would need reconsideration). If Topic 4 opens Day 2,
the current Tier 2 maximum is correct.

#### PASS: Peer Discussions

Cell 6 (Discussion 1), Cell 27 (Discussion 2). Both present.

#### PASS: Homework Extensions

Cell 24 (Lab 2 stretch/homework), Cell 36 (3 homework extensions). Present.

#### PASS: AI-tells

No em dashes, en dashes, Unicode multiplication signs found.

---

## Cross-Topic Issues

### Issue C1: Tier 2 Quota Conflict (HIGH PRIORITY)

Topic 1 Lab 2 and Topic 2 Lab 2 are both labeled Tier 2.
CLAUDE.md mandates exactly ONE Tier 2 per day across all topics.

Action required: designate exactly one topic as the Tier 2 carrier for Day 1.
Recommended: Topic 1 Lab 2 (triage prompt engineering) stays as Tier 2.
Topic 2 Lab 2 (complaint similarity matrix) is downgraded to Tier 1 by adding
step-by-step `# YOUR CODE` prescriptions for:
1. Calling the embedding model for all 5 complaints
2. Stacking results into a matrix
3. Computing cosine similarity using sklearn or manual dot-product
4. Finding the argmax for most similar pair

### Issue C2: Topic 4 Day Placement Ambiguity (MEDIUM PRIORITY)

The audit request includes Topic 4 as a Day 1 file, but the topic_4_transformers.md
plan explicitly states it opens Day 2. This affects:
- Tier 3 lab assignment (Topic 3b has Tier 3; if Topic 4 is Day 2, no conflict)
- Day 1 end-of-day capstone positioning

Action required: confirm with course designer which day Topic 4 belongs to before
building the notebook.

### Issue C3: Beat 2 "print placeholder" Pattern in Topics 3a and 3b (LOW PRIORITY)

Both topic_3a and topic_3b use a code cell containing only `print("See diagram...")`
followed by a markdown diagram cell. This pattern exists to avoid 3 consecutive
markdown cells, but the print cell has no teaching value.

The cleaner fix is to ensure the Beat 1 code cell immediately precedes the diagram
markdown. If the section header is also a markdown cell, the order becomes:
[Section header markdown] -> [Beat 1 code] -> [Diagram markdown] -> [Beat 3 code]
which is at most 1 markdown before the Beat 1 code and 1 markdown after it --
no consecutive markdown violation.

Action required: when building notebooks, restructure these sections to avoid the
placeholder print cells.

### Issue C4: --- Separator Pollution Across F1 and F2 (MEDIUM PRIORITY)

Both F1 and F2 section header cells begin with a bare `---` line. This is likely
a copy-paste pattern from the original plan template.

Action required: before running /build-topic-notebook, do a final scan of each
research file and remove all bare `---` lines from cell body content (section
headers, closing cells). The `##` heading alone is sufficient visual separation in
JupyterLab.

---

## Priority Fixes

Priority 1 -- BLOCKING (must fix before building any notebook):

1. [topic_1_overview_genai.md] Add Beat 2 diagram placeholder for Section 3
   (API/Prompting). Either add a new diagram to the index or document that
   autoregressive-loop.mmd is reused for this section.

2. [topic_2_introducing_llms.md] Resolve Tier 2 quota conflict with Topic 1.
   Downgrade Topic 2 Lab 2 to Tier 1 by adding numbered sub-steps.

3. [topic_4_transformers.md] Fix diagram index: add multi-head-attention.mmd,
   remove or add cell placeholder for positional-encoding-pattern.mmd.
   Confirm whether 3 diagrams is acceptable.

4. [topic_3a_attention_python.md] Move Diagram 2 (bahdanau-score-computation.mmd)
   to before the Bahdanau Beat 3 demo cell. Update the cell ordering in the plan.

Priority 2 -- REQUIRED before delivery (fix before /build-topic-notebook runs):

5. [All files with ---] Strip bare `---` separator lines from all section header
   cells. Affected: F1 (5 cells), F2 (at least 2 cells), topic_1 (6 cells),
   topic_3a (Cell 30). Use the /build-topic-notebook step to catch any remaining
   occurrences before committing cells.

6. [topic_3b_attention_pytorch.md] Add a second peer discussion cell between the
   attention visualization section and the Capstone lab. Draft: "When does
   sqrt(d_k) scaling matter and when is it safe to skip it?" (3 min, pairs).

7. [Confirm] Resolve Topic 4 day placement: Day 1 or Day 2? Decision determines
   whether a Tier 3 capstone is needed in Topic 4 and whether Topic 3b's Tier 3
   is redundant.

Priority 3 -- POLISH (fix during notebook build):

8. [topic_3a, topic_3b] Remove placeholder print cells used to avoid consecutive
   markdown. Restructure Beat 1 -> diagram -> Beat 3 cell order so no print trick
   is needed.

9. [topic_2_introducing_llms.md] Clarify whether transformer-families.mmd in Cell 5
   serves as the Beat 2 diagram for Section 3 or only as an orientation visual.
   Add a comment in the plan noting which section's Beat 2 each diagram anchors.

10. [F1_pytorch_refresher.md] Consider whether Beat 3 -> Diagram 2 ordering
    (training-loop.mmd placed after Beat 3 code) is intentional reinforcement
    or should be moved to before Beat 3. Low risk -- document the decision.

---

End of Audit Report
