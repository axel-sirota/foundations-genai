# Required-Path Continuity Refactor - Design Doc (Round 2, Codex-corrected)

## Status

This is the SECOND, corrected, expanded version. The first draft passed an adversarial
review by Codex (o3) which found blocking defects. The defect list is in
`plans/refactor/CODEX_FINDINGS_R1.md`. Findings R6, R7, R8, R9, R10, R11, R12 are
resolved here. Resolution principle is OPTION B: the required path teaches concepts
properly with genuine MINI-LESSONS; the optional notebooks are only the from-scratch
BUILDS.

## Purpose

The course was renumbered. The NEW required path is:

| New # | Slug |
|-------|------|
| topic_1 | overview_genai |
| topic_2 | introducing_llms |
| topic_3 | huggingface |
| topic_4 | full_finetuning |
| topic_5 | transfer_learning |
| topic_6 | peft_lora_distilbert |
| topic_7 | quantization |
| topic_8 | agent capstone (future, not yet built) |

Four notebooks were moved OUT of the required path into a standalone OPTIONAL track:

- topic_optional_attention_python
- topic_optional_attention_pytorch
- topic_optional_transformers
- topic_optional_lora_ffn

A prior phase mechanically renumbered cross-references BETWEEN required topics. What
remains, and what this doc specifies, is:

1. The narrative continuity rework: passages that assume the old linear path went
   attention -> transformers -> LoRA-from-scratch as required topics must be reframed.
2. TWO new MINI-LESSONS (Option B): a transformer-concepts mini-lesson in topic_2
   (R7) and a LoRA-mechanics mini-lesson in topic_6 (R8). The required path must TEACH
   these concepts, not point at optional notebooks for them.
3. Self-containment: required notebooks must not load artifacts produced by optional
   notebooks (R6).
4. Consistent wording: the CONCEPT is the required mini-lesson, the from-scratch BUILD
   is optional (R12).

## Conventions for the editing agents

- **Every change in this doc applies to BOTH copies of the notebook**: the
  `Exercises/<topic>/<topic>.ipynb` file AND the matching
  `Solutions/<topic>/<topic>.ipynb` file. The cells listed here are markdown or
  narrative code-comment cells that are identical in both copies. Apply the same OLD
  -> NEW replacement, and the same new-cell insertions, in both files.
- For the two new mini-lessons: insert the SAME cells in both copies. The mini-lesson
  lab cell (the one small runnable code cell) is identical in Exercises and Solutions
  because it is a fully-worked demo, not a `# YOUR CODE` lab. No safety-net cell is
  needed (no downstream variable depends on it).
- Cell indices below are 0-based and were read from the `Exercises/` copy. The
  `Solutions/` copy has the same cell ordering for these narrative cells; if an index
  is off by one in a Solutions file, match on the quoted OLD text instead.
- Plain ASCII only. No em-dashes, en-dashes, Unicode multiplication signs, emojis.
- Preserve the Barclays customer-support through-line and the four-beat arc.
- Replace ONLY the quoted OLD text. Do not reflow or re-wrap surrounding lines unless
  the new text changes line length enough to need it; keep the surrounding markdown
  structure intact.
- No more than 3 consecutive markdown cells without a code cell (CLAUDE.md). The R7
  and R8 mini-lessons each include one runnable code cell to honour this.

## Canonical reframing language (reuse across notebooks)

R12 resolution: the contradiction "transformers are the architecture behind every
model" vs "you can skip the optional notebook" is resolved by separating CONCEPT from
BUILD. The required path teaches the concept (the R7 and R8 mini-lessons). The optional
notebooks are only the from-scratch implementation. Every passage must use this split.

CONCEPT is required. Use, when introducing the idea:

> Transformers are the architecture behind every model you use in this course. This
> topic teaches you the concepts you need to read and reason about them.

BUILD is optional. Use, when pointing at the optional notebook:

> If you want to see one built from scratch - attention, positional encoding, the
> decoder loop - there is an optional deep-dive notebook (topic_optional_transformers).
> It is not required for what follows here.

Analogous one-liners:

- Attention-from-scratch BUILD: "If you want to implement attention by hand, in pure
  Python and then in PyTorch, there are two optional deep-dive notebooks
  (topic_optional_attention_python and topic_optional_attention_pytorch). The concepts
  you need are taught here; the by-hand build is optional."
- LoRA-from-scratch BUILD: "If you want to see LoRA implemented by hand on a
  feed-forward network, there is an optional deep-dive notebook
  (topic_optional_lora_ffn). The required path teaches the mechanics here and uses the
  PEFT library directly."

Never say "you did X in Topic N" when X is the from-scratch attention build, the
transformer-from-scratch build, the translator build, or LoRA-from-scratch. Never imply
the optional notebook is a prerequisite for understanding.

---

## Topic 1 - overview_genai

File(s): `Exercises/topic_1_overview_genai/topic_1_overview_genai.ipynb` and the
Solutions twin.

### Edit 1.1 - cell 33, wrap-up "Questions to Bring to Topic 2"

OLD:
```
- What would it take to fine-tune GPT-4o on Barclays complaint data? (You will do this in Topic 6.)
```

NEW:
```
- What would it take to fine-tune a model on Barclays complaint data? (You will do this with an open model in Topic 4 and Topic 6.)
```

Rationale: The new Topic 6 is PEFT/LoRA on DistilBERT, and the course never fine-tunes
GPT-4o itself. Point the student at where fine-tuning actually happens (full
fine-tuning in Topic 4, parameter-efficient fine-tuning in Topic 6).

### Edit 1.2 - cell 33, "What Comes Next (Topic 2: Introducing LLMs)" paragraph

OLD:
```
Topic 2 goes inside the black box. You will see:
- What a transformer architecture actually looks like in code
- What attention is and why it solves the long-range dependency problem
- How training works: cross-entropy loss, next-token prediction at scale
```

NEW:
```
Topic 2 goes inside the black box. You will see:
- The three transformer architecture families and how to pick the right one
- A concept-level mini-lesson on how self-attention works: queries, keys, values, and multi-head attention
- How tokenization and embeddings turn text into something a model can use
```

Rationale: Under Option B, Topic 2 now contains a genuine transformer-concepts
mini-lesson (see the dedicated R7 section). The old bullet list promised a code-level
architecture walk-through and training internals that remain optional; the new bullets
match what Topic 2 actually delivers, including the new mini-lesson.

No other cells in Topic 1 need continuity changes. The autoregressive-loop and GAN
demos are self-contained within Topic 1 and do not depend on optional notebooks.

---

## Topic 2 - introducing_llms

File(s): `Exercises/topic_2_introducing_llms/topic_2_introducing_llms.ipynb` and the
Solutions twin. This notebook gets the R7 transformer-concepts mini-lesson; see the
dedicated section "R7 Mini-Lesson" below for the new cells. The edits in this section
are the narrative-continuity replacements.

### Edit 2.1 - cell 34, Peer Discussion question 2

OLD:
```
2. The lifecycle says "you almost never train from scratch."
   But this course teaches you to build a transformer FROM SCRATCH in Topics 3 and 4.
   Why is that useful if you will never do it in production?
```

NEW:
```
2. The lifecycle says "you almost never train from scratch."
   The optional deep-dive notebooks still show you how to build a transformer from
   scratch. Why is it useful to understand that internal machinery even if you will
   never write it yourself in production?
```

Rationale: Transformer-from-scratch is now the optional transformers notebook, not
required Topics 3 and 4. Keep the discussion point (it is pedagogically good) but
reframe the build as optional.

### Edit 2.2 - cell 37, "The Self-Attention Mechanism" block

OLD:
```
### The Self-Attention Mechanism
We said "N transformer blocks" and skipped straight to the output.
The inside of those blocks -- queries, keys, values, attention weights -- is the
mathematical heart of the transformer. That is **Topic 3 (Attention)**.
You will implement it from scratch in pure Python, then in PyTorch.
```

NEW:
```
### The Self-Attention Mechanism, in Depth
We covered the core idea in the mini-lesson above: queries, keys, values, and
multi-head attention. That is enough to read attention-head visualizations and reason
about model behaviour for the rest of this course.
If you want to implement attention by hand and watch every matrix multiply, there are
two optional deep-dive notebooks (topic_optional_attention_python and
topic_optional_attention_pytorch). They are not required for the path you are on.
```

Rationale: Under Option B, the attention concept is now taught in Topic 2 itself (the
R7 mini-lesson). This cell, which sits in the "What We Did NOT Cover" section, must no
longer claim attention is uncovered; it now points only at the optional from-scratch
BUILD. The required Topic 3 is HuggingFace, not attention.

### Edit 2.3 - cell 37, "Training and Fine-Tuning" block

OLD:
```
### Training and Fine-Tuning
Everything today was INFERENCE -- we used pretrained weights and did not update them.
In **Topics 6 and 7** you will fine-tune DistilBERT and Flan-T5
on a Barclays complaint dataset on a remote GPU instance.
```

NEW:
```
### Training and Fine-Tuning
Everything today was INFERENCE -- we used pretrained weights and did not update them.
In **Topic 4 (Full Fine-Tuning)** and **Topic 6 (PEFT and LoRA)** you will fine-tune
DistilBERT on a Barclays complaint dataset on a remote GPU instance.
```

Rationale: Renumbered. Full fine-tuning is now Topic 4, parameter-efficient
fine-tuning is Topic 6. Flan-T5 fine-tuning is no longer a required-path deliverable;
drop it to keep the promise accurate.

### Edit 2.4 - cell 37, "Positional Encoding (the Details)" block

OLD:
```
### Positional Encoding (the Details)
We mentioned it briefly in the pipeline. The math behind sinusoidal and learned
positional encodings is in **Topic 4 (Transformers)**.
```

NEW:
```
### Positional Encoding (the Full Math)
The mini-lesson above explained why positional encoding exists and what it does. The
full sinusoidal formula and the proof that it encodes relative position are in the
optional transformers deep-dive notebook (topic_optional_transformers).
```

Rationale: Under Option B the positional-encoding CONCEPT is taught in the R7
mini-lesson. Only the full derivation remains optional. The transformer-internals topic
is now optional, not required Topic 4.

### Edit 2.5 - cell 39, wrap-up "Where We Are in the Course" code block

OLD:
```
Topic 1 (done): What is GenAI?
Topic 2 (done): What is an LLM? How does it tokenize and embed text?
Topic 3 (next): How does self-attention work? Build it from scratch in Python.
Topic 4:        Full transformer. Build the encoder-decoder from scratch in PyTorch.
Later:          Fine-tune DistilBERT and Flan-T5 on Barclays complaint data.
Later:          Deploy, quantize, align.
```

NEW:
```
Topic 1 (done): What is GenAI?
Topic 2 (done): What is an LLM? Tokenization, embeddings, and how self-attention works.
Topic 3 (next): The HuggingFace ecosystem. Load and run pretrained models.
Topic 4:        Full fine-tuning on Barclays complaint data, and catastrophic forgetting.
Topic 5-6:      Transfer learning, then parameter-efficient fine-tuning with PEFT and LoRA.
Topic 7:        Compress and deploy the model: quantization, pruning, distillation.
Optional:       Build attention and the transformer from scratch (deep-dive notebooks).
```

Rationale: The roadmap listed the old linear path. Replace it with the new required
sequence. Topic 2's line now mentions self-attention because the R7 mini-lesson covers
it. The optional deep-dives get their own line.

### Edit 2.6 - cell 39, paragraph after the code block

OLD:
```
In Topic 3, we open the black box we used today.
You will implement the attention mechanism -- the core innovation that makes transformers
better than RNNs and LSTMs -- in pure Python, with no libraries.
```

NEW:
```
In Topic 3, we move from theory to tooling: the HuggingFace ecosystem, where you load
pretrained models and run them in a few lines of code. You already have the attention
concepts you need from the mini-lesson in this topic. If you want to implement the
attention mechanism yourself in pure Python, the optional attention deep-dive notebooks
are there for you, but they are not required for what comes next.
```

Rationale: Topic 3 is now HuggingFace. The "next we implement attention" handoff is
stale. The concept is already taught here (R7 mini-lesson); only the build is optional.

### Edit 2.7 - cell 40, cleanup code comment and print

OLD (comment line):
```
# Optional cleanup -- free memory before moving to Topic 3.
```
NEW:
```
# Optional cleanup -- free memory before moving to Topic 3 (HuggingFace).
```

OLD (print line):
```
print("Memory cleanup complete. Ready for Topic 3.")
```
NEW:
```
print("Memory cleanup complete. Ready for Topic 3 (HuggingFace).")
```

Rationale: Low-stakes clarity fix so the transition names the new Topic 3 content.

Note: cell 0 ("In Topic 1 we decided...") and cell 38/41 (knowledge check, model-size
appendix) need NO change. "Train from scratch" elsewhere refers to classical ML
training, not to building a transformer, and is correct as written.

---

## Topic 3 - huggingface

File(s): `Exercises/topic_3_huggingface/topic_3_huggingface.ipynb` and the Solutions
twin. This is the most affected notebook for narrative continuity.

### Edit 3.1 - cell 0, "What you will build today" opening paragraph

OLD:
```
In Topic 4 you assembled a Transformer encoder-decoder from scratch and trained it on Spanish-to-English
complaint tickets. You had full control, and you wrote every line: positional encoding, multi-head
attention, the decoder loop, the SageMaker estimator.

Today you stop reinventing the wheel. HuggingFace gives you pre-trained models trained on
billions of sentences, curated datasets, and a four-line inference API. By the end of this
topic you will classify Barclays complaint sentiment, route complaints to the correct team using
zero-shot classification, extract named entities, and understand how to share a checkpoint to the
Hub.
```

NEW:
```
In Topic 2 you learned what a transformer is: the three architecture families, and a
concept-level mini-lesson on self-attention - queries, keys, values, and multi-head
attention. That is the foundation for everything here.

This topic is about USING those models, not building them. HuggingFace gives you
pre-trained models trained on billions of sentences, curated datasets, and a four-line
inference API. By the end of this topic you will classify Barclays complaint sentiment,
route complaints to the correct team using zero-shot classification, extract named
entities, and understand how to share a checkpoint to the Hub.

If you want to see a transformer built from scratch - positional encoding, multi-head
attention, the decoder loop - there is an optional deep-dive notebook
(topic_optional_transformers). It is not required for what follows here.
```

Rationale: The opening falsely claimed the student had just done the old transformers
topic as required Topic 4. The new Topic 4 is full_finetuning. The accurate prior topic
is Topic 2, which now teaches the transformer concepts (R7 mini-lesson). Reframe the
build as optional and pivot to application. Resolves the R10-class self-reference for
Topic 3.

### Edit 3.2 - cell 2, "Day 2 System Overview" table

OLD:
```
| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | T4 Transformers | Build the architecture from scratch |
| 2 | T5 HuggingFace | Load pre-trained models from the Hub (YOU ARE HERE) |
| 3 | T6a Full Fine-Tuning | Adapt a model to Barclays complaints |
| 4 | T6b Transfer Learning | Freeze the encoder, train only the head |
| 5 | T7a LoRA from Scratch | Implement parameter-efficient adaptation |
| 6 | T7b PEFT + LoRA | Apply PEFT library to a full classifier |
```

NEW:
```
| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | Topic 3 HuggingFace | Load pre-trained models from the Hub (YOU ARE HERE) |
| 2 | Topic 4 Full Fine-Tuning | Adapt a model to Barclays complaints |
| 3 | Topic 5 Transfer Learning | Freeze the encoder, train only the head |
| 4 | Topic 6 PEFT and LoRA | Apply the PEFT library to a full classifier |
| 5 | Topic 7 Quantization | Compress and deploy the model |
```

Also change the heading text and the sentence under the table:

OLD heading: `## Day 2 System Overview`
NEW heading: `## Where This Topic Fits`

OLD sentence:
```
By end of Day 2 you will have a fine-tuned, PEFT-adapted DistilBERT complaint classifier
running as a SageMaker endpoint.
```
NEW sentence:
```
By the end of the required path you will have a fine-tuned, PEFT-adapted DistilBERT
complaint classifier compressed and running as a SageMaker endpoint.
```

Rationale: The table used the old T4/T5/T6a/T6b/T7a/T7b numbering and listed
transformers-from-scratch and LoRA-from-scratch as required steps. Replace with the
new required sequence (Topics 3-7). The "Day 2" framing no longer matches; generalize.

### Edit 3.3 - cell 6, COMPLAINT_TOKENS comment (self-containment, R6)

OLD:
```
# Complaint tokens carried over from Topic 4 for tokenizer comparison demos.
COMPLAINT_TOKENS = [
    "unauthorised", "charge", "account", "fraud",
    "refund", "dispute", "urgent", "branch"
]
```

NEW:
```
# Complaint vocabulary defined locally for the tokenizer comparison demos in this
# notebook. Topic 3 is self-contained and does not depend on any earlier notebook.
COMPLAINT_TOKENS = [
    "unauthorised", "charge", "account", "fraud",
    "refund", "dispute", "urgent", "branch"
]
```

Rationale: R6. The "carried over from Topic 4" claim was false. The list is defined
inline, so the only change needed is the comment: state it is defined locally.
COMPLAINT_TOKENS is only printed in this cell and not consumed downstream.

### Edit 3.4 - cell 7, "What are we building today?" opening

OLD:
```
In Topic 4 we trained a translator from scratch on 50,000 sentence pairs.
It took 15-25 minutes on an NVIDIA T4 GPU and reached a modest BLEU score.

That same model exists on HuggingFace Hub, trained on 50 million sentence pairs,
fine-tuned by professional teams, with a 95 BLEU score, and it downloads in seconds.
```

NEW:
```
Training a translator from scratch - the kind of build shown in the optional
transformers deep-dive - takes a custom architecture, 50,000 sentence pairs, and
15-25 minutes on a GPU just to reach a modest BLEU score.

That same kind of model already exists on the HuggingFace Hub, trained on 50 million
sentence pairs, fine-tuned by professional teams, with a 95 BLEU score, and it
downloads in seconds.
```

Rationale: The passage claimed "in Topic 4 we trained a translator", which was the old
transformers topic and is now optional. Keep the from-scratch-vs-Hub contrast but
attribute the build to the optional deep-dive.

### Edit 3.5 - cell 25, code comment

OLD:
```
# This motivates Section 4 (AutoModel) and the next topic (full fine-tuning).
```
NEW:
```
# This motivates Section 4 (AutoModel) and Topic 4 (full fine-tuning).
```

Rationale: Make the forward reference explicit and correct. The next topic IS full
fine-tuning (Topic 4). Low-stakes.

### Edit 3.6 - cell 43, wrap-up "What is coming next"

OLD:
```
### What is coming next

In the next topic on Full Fine-Tuning:
You will fine-tune a pre-trained DistilBERT on a Barclays-domain dataset using a remote
GPU job. You will see catastrophic forgetting in action, what happens when you fine-tune
too aggressively on a small dataset.

In Topics 6-7 on Transfer Learning with frozen encoder:
You will freeze the DistilBERT encoder and train only the classification head.
The Hub models you used today are your starting points.
```

NEW:
```
### What is coming next

In Topic 4 on Full Fine-Tuning:
You will fine-tune a pre-trained DistilBERT on a Barclays-domain dataset using a remote
GPU job. You will see catastrophic forgetting in action, what happens when you fine-tune
too aggressively on a small dataset.

In Topic 5 on Transfer Learning:
You will freeze the DistilBERT encoder and train only the classification head.
The Hub models you used today are your starting points.
```

Rationale: "Topics 6-7 on Transfer Learning" was the old numbering. Transfer learning
is now Topic 5, full fine-tuning is Topic 4.

### Edit 3.7 - cell 44, end-of-topic footer

OLD:
```
*End of Topic 3 - HuggingFace Ecosystem*

Next session: Full Fine-Tuning and Catastrophic Forgetting.
Fine-tune DistilBERT on Barclays complaint data. See what happens when you train too hard.
```

NEW:
```
*End of Topic 3 - HuggingFace Ecosystem*

Next: Topic 4 - Full Fine-Tuning and Catastrophic Forgetting.
Fine-tune DistilBERT on Barclays complaint data. See what happens when you train too hard.
```

Rationale: Consistency of forward reference; name Topic 4 explicitly. Low-stakes.

---

## Topic 4 - full_finetuning

File(s): `Exercises/topic_4_full_finetuning/topic_4_full_finetuning.ipynb` and the
Solutions twin.

### R10 verification note

CODEX R10 reports that topic_4 keeps an opening recap reading "assembled a Transformer
encoder-decoder from scratch in Topic 4" and "added attention ON TOP of an RNN in
Topics 3a/3b". This text was inspected cell by cell in the current notebook (commit
13187c0). It is NOT present. Topic 4 cell 0 currently reads "In Topic 3 you used
pre-trained models off the shelf, zero training, pure inference" and "In Topic 3 you
used AutoModel for inference" - both correct under the new numbering. A prior commit
("fix(Day2+Day3): add Beat 1-4 labels...") already removed the stale recap.

R10 is therefore resolved by verification: the topic_4 opening recap is clean, no edit
is required for it. The editing agent must still confirm the same in the Solutions
twin by matching on the OLD strings "assembled a Transformer encoder-decoder" and "ON
TOP of an RNN"; if either string is found in the Solutions copy, replace the
containing sentence with the Exercises-copy wording. The remaining topic_4 edits
(4.1-4.6 below) are the renumbering edits the first draft listed and are still valid.

### Edit 4.1 - cell 4, "Day 2 System Overview" table

OLD:
```
## Day 2 System Overview

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | T4 Transformers | Build the architecture from scratch |
| 2 | T5 HuggingFace | Load pre-trained models from the Hub |
| 3 | T6a Full Fine-Tuning | Adapt a model to Barclays complaints (YOU ARE HERE) |
| 4 | T6b Transfer Learning | Freeze the encoder, train only the head |
| 5 | T7a LoRA from Scratch | Implement parameter-efficient adaptation |
| 6 | T7b PEFT + LoRA | Apply PEFT library to a full classifier |

By end of Day 2 you will have a fine-tuned, PEFT-adapted DistilBERT complaint classifier
running as a SageMaker endpoint.
```

NEW:
```
## Where This Topic Fits

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | Topic 3 HuggingFace | Load pre-trained models from the Hub |
| 2 | Topic 4 Full Fine-Tuning | Adapt a model to Barclays complaints (YOU ARE HERE) |
| 3 | Topic 5 Transfer Learning | Freeze the encoder, train only the head |
| 4 | Topic 6 PEFT and LoRA | Apply the PEFT library to a full classifier |
| 5 | Topic 7 Quantization | Compress and deploy the model |

By the end of the required path you will have a fine-tuned, PEFT-adapted DistilBERT
complaint classifier compressed and running as a SageMaker endpoint.
```

Rationale: Same stale-numbering table as Topic 3. Replace with the new required
sequence; drop transformers-from-scratch and LoRA-from-scratch rows.

### Edit 4.2 - cell 5, COMPLAINT_TEXTS comment (self-containment, R6)

OLD:
```
# Complaint vocabulary carried over from Topic 3.
COMPLAINT_TEXTS = [
```

NEW:
```
# Sample complaint texts defined locally for this notebook's demos.
COMPLAINT_TEXTS = [
```

Rationale: R6. The list is fully defined inline; it does not depend on Topic 3 having
run. Make the self-containment explicit.

### Edit 4.3 - cell 25, mitigation strategies "Option B"

OLD:
```
Option B, LoRA and PEFT: freeze the pre-trained weights and only train small adapter layers.
The original knowledge is locked in the frozen weights; only the adapters change.
(This is Topic 7a and 7b.)

Option C, Elastic Weight Consolidation (EWC): add a regularisation term that penalises
changes to parameters that were important for the original task.

For this course we focus on A (multitask, shown below) and B (LoRA, next topic).
```

NEW:
```
Option B, LoRA and PEFT: freeze the pre-trained weights and only train small adapter layers.
The original knowledge is locked in the frozen weights; only the adapters change.
(This is Topic 6.)

Option C, Elastic Weight Consolidation (EWC): add a regularisation term that penalises
changes to parameters that were important for the original task.

For this course we focus on A (multitask, shown below) and B (LoRA, in Topic 6).
```

Rationale: LoRA/PEFT is now the single required Topic 6, not the old "7a and 7b" pair.
Topic 6 is two topics ahead of Topic 4, so "next topic" is also wrong; name it.

### Edit 4.4 - cell 41, decision-framework table

OLD:
```
| Memory or compute constrained     | PEFT and LoRA (Topic 7)       |
| Task changes frequently           | PEFT and LoRA (Topic 7)       |
```

NEW:
```
| Memory or compute constrained     | PEFT and LoRA (Topic 6)       |
| Task changes frequently           | PEFT and LoRA (Topic 6)       |
```

Rationale: PEFT/LoRA is the new Topic 6, not Topic 7. Topic 7 is now Quantization.

### Edit 4.5 - cell 42, cost-comparison code (comment, dict entry, print)

OLD (comment):
```
# Cost comparison: full fine-tuning vs inference only vs PEFT (preview for Topic 7)
```
NEW (comment):
```
# Cost comparison: full fine-tuning vs inference only vs PEFT (preview for Topic 6)
```

OLD (dict name field):
```
        "name": "PEFT LoRA (preview: Topic 7)",
```
NEW:
```
        "name": "PEFT LoRA (preview: Topic 6)",
```

OLD (print line):
```
print("Key takeaway: PEFT (Topic 7) gets 89% accuracy at 40% the cost of full fine-tuning")
```
NEW:
```
print("Key takeaway: PEFT (Topic 6) gets 89% accuracy at 40% the cost of full fine-tuning")
```

Rationale: PEFT is Topic 6 in the new numbering.

### Edit 4.6 - cell 43, wrap-up "What comes next"

OLD:
```
Topic 5 (Transfer Learning with DistilBERT) extends this to a CPU training job using the
PyTorch estimator, you will see how to use transformers without the HuggingFace DLC.

Topic 7a and 7b (LoRA plus PEFT) show how to get 89% of full fine-tuning accuracy at 40%
the GPU cost with zero catastrophic forgetting. Those are the techniques Barclays
production teams actually use today.
```

NEW:
```
Topic 5 (Transfer Learning with DistilBERT) extends this to a CPU training job using the
PyTorch estimator, you will see how to use transformers without the HuggingFace DLC.

Topic 6 (PEFT and LoRA) shows how to get 89% of full fine-tuning accuracy at 40%
the GPU cost with zero catastrophic forgetting. Those are the techniques Barclays
production teams actually use today.
```

Rationale: "Topic 7a and 7b" was the old LoRA-from-scratch plus PEFT pair. The
required path now has a single PEFT/LoRA topic: Topic 6.

### Edit 4.7 - cell 43, end-of-cell "Next session" footer

OLD:
```
Next session: Topic 5 -- Transfer Learning.
```
NEW: no change. Topic 5 is still Transfer Learning. Leave this line as is.

### Edit 4.8 - cell 44, variable-inventory code (comment and print)

OLD (comment):
```
# Quick recap: variable inventory for Topic 5 and Topic 7a.
```
NEW:
```
# Quick recap: variable inventory for Topic 5.
```

OLD (print line):
```
print("All variables above are available for Topic 5 and 7a.")
```
NEW:
```
print("All variables above are available for Topic 5.")
```

Rationale: "Topic 7a" (LoRA from scratch) is now optional and does not consume these
variables. The variables are carried forward to Topic 5 (transfer learning) only.

---

## Topic 5 - transfer_learning

File(s): `Exercises/topic_5_transfer_learning/topic_5_transfer_learning.ipynb` and the
Solutions twin.

### Edit 5.1 - cell 3, "Day 2 System Overview" table

OLD:
```
## Day 2 System Overview

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | T4 Transformers | Build the architecture from scratch |
| 2 | T5 HuggingFace | Load pre-trained models from the Hub |
| 3 | T6a Full Fine-Tuning | Adapt a model to Barclays complaints |
| 4 | T6b Transfer Learning | Freeze the encoder, train only the head (YOU ARE HERE) |
| 5 | T7a LoRA from Scratch | Implement parameter-efficient adaptation |
| 6 | T7b PEFT + LoRA | Apply PEFT library to a full classifier |

By end of Day 2 you will have a fine-tuned, PEFT-adapted DistilBERT complaint classifier
running as a SageMaker endpoint.
```

NEW:
```
## Where This Topic Fits

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | Topic 3 HuggingFace | Load pre-trained models from the Hub |
| 2 | Topic 4 Full Fine-Tuning | Adapt a model to Barclays complaints |
| 3 | Topic 5 Transfer Learning | Freeze the encoder, train only the head (YOU ARE HERE) |
| 4 | Topic 6 PEFT and LoRA | Apply the PEFT library to a full classifier |
| 5 | Topic 7 Quantization | Compress and deploy the model |

By the end of the required path you will have a fine-tuned, PEFT-adapted DistilBERT
complaint classifier compressed and running as a SageMaker endpoint.
```

Rationale: Same stale-numbering table. Replace with the new required sequence.

### Edit 5.2 - cell 0, intro paragraph "compare against 6a"

OLD:
```
- Compare accuracy and training cost against 6a full fine-tuning
```

NEW:
```
- Compare accuracy and training cost against Topic 4 full fine-tuning
```

Rationale: Full fine-tuning is now Topic 4, not "6a".

### Edit 5.3 - cell 28, Discussion bullet on LoRA

OLD:
```
- LoRA (Topic 7): trainable adapters inserted into frozen layers, best of both worlds

We will revisit this comparison after Topic 7.
```

NEW:
```
- LoRA (Topic 6): trainable adapters inserted into frozen layers, best of both worlds

We will revisit this comparison in Topic 6.
```

Rationale: LoRA/PEFT is Topic 6 in the new numbering.

### Edit 5.4 - cell 45, Discussion bullet on LoRA

OLD:
```
- Topic 7 introduces LoRA: trainable low-rank adapters injected INTO the frozen encoder layers.
  How is that different from what we built today? What limitation of today's approach does it
  address?
```

NEW:
```
- Topic 6 introduces LoRA: trainable low-rank adapters injected INTO the frozen encoder layers.
  How is that different from what we built today? What limitation of today's approach does it
  address?
```

Rationale: LoRA is Topic 6.

### Edit 5.5 - cell 46, "Up next" block

OLD:
```
**Up next - Topic 7a: LoRA on Feed-Forward Networks**
What if we could get even better accuracy than full fine-tuning while only touching
0.1 percent of the parameters? LoRA inserts low-rank adapter matrices into the frozen
transformer layers. We will build one from scratch before using PEFT in 7b.
```

NEW:
```
**Up next - Topic 6: PEFT and LoRA with DistilBERT**
What if we could get even better accuracy than full fine-tuning while only touching
0.1 percent of the parameters? LoRA inserts low-rank adapter matrices into the frozen
transformer layers. Topic 6 teaches the LoRA mechanics and uses the production-grade
HuggingFace PEFT library to do exactly that. If you want to see LoRA implemented by
hand on a feed-forward network first, there is an optional deep-dive notebook
(topic_optional_lora_ffn).
```

Rationale: The next required topic is Topic 6 (PEFT library on DistilBERT, with the R8
mini-lesson teaching the mechanics), not the old "7a LoRA from scratch".
LoRA-from-scratch is now the optional lora_ffn notebook.

Note: cell 8 ("Topic 4 showed full fine-tuning") is CORRECT under the new numbering -
full fine-tuning is Topic 4. No change. cells 12 and 33 ("learn from scratch", "training
from scratch is too slow") refer to training a model from random initialization as a
general concept, not to building a transformer; no change.

---

## Topic 6 - peft_lora_distilbert

File(s): `Exercises/topic_6_peft_lora_distilbert/topic_6_peft_lora_distilbert.ipynb`
and the Solutions twin. This notebook gets the R8 LoRA-mechanics mini-lesson; see the
dedicated section "R8 Mini-Lesson" below for the new cells. The edits in this section
are the narrative-continuity replacements. The notebook currently leans on "you built
LoRA from scratch in 7a" as a prerequisite; that prerequisite is now an OPTIONAL
notebook, and the R8 mini-lesson replaces the missing knowledge inside this topic.

### Edit 6.1 - cell 0, "What you will build" opening

OLD:
```
In Topic 7a you built LoRA from scratch on feed-forward networks.
Now you use the production-grade HuggingFace PEFT library to apply LoRA to a full
DistilBERT complaint classifier, the same pattern used in real ML pipelines at scale.
```

NEW:
```
This topic teaches you how LoRA (Low-Rank Adaptation) works and how to apply it with
the production-grade HuggingFace PEFT library. A short mini-lesson below explains the
mechanics - low-rank decomposition, and what rank, alpha, and dropout actually do - so
you can reason about the knobs you tune.

You then use the PEFT library to apply LoRA to a full DistilBERT complaint classifier,
the same pattern used in real ML pipelines at scale. If you want to see LoRA built by
hand on a feed-forward network, there is an optional deep-dive notebook
(topic_optional_lora_ffn); it is not required for this topic.
```

Rationale: The notebook opened by asserting the student had done the old required "7a".
Under Option B, the LoRA mechanics are now taught here (R8 mini-lesson). Reframe:
announce the mini-lesson, then go to the application, and mention the optional build.

### Edit 6.2 - cell 4, "Day 2 System Overview" table

OLD:
```
## Day 2 System Overview

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | T4 Transformers | Build the architecture from scratch |
| 2 | T5 HuggingFace | Load pre-trained models from the Hub |
| 3 | T6a Full Fine-Tuning | Adapt a model to Barclays complaints |
| 4 | T6b Transfer Learning | Freeze the encoder, train only the head |
| 5 | T7a LoRA from Scratch | Implement parameter-efficient adaptation |
| 6 | T7b PEFT + LoRA | Apply PEFT library to a full classifier (YOU ARE HERE) |

By end of Day 2 you will have a fine-tuned, PEFT-adapted DistilBERT complaint classifier
running as a SageMaker endpoint.
```

NEW:
```
## Where This Topic Fits

We are building the Barclays Customer Support Intelligence System end to end.
Each topic adds one layer. Today you are here:

| Step | Topic | What it adds to the system |
|------|-------|---------------------------|
| 1 | Topic 3 HuggingFace | Load pre-trained models from the Hub |
| 2 | Topic 4 Full Fine-Tuning | Adapt a model to Barclays complaints |
| 3 | Topic 5 Transfer Learning | Freeze the encoder, train only the head |
| 4 | Topic 6 PEFT and LoRA | Apply the PEFT library to a full classifier (YOU ARE HERE) |
| 5 | Topic 7 Quantization | Compress and deploy the model |

By the end of the required path you will have a fine-tuned, PEFT-adapted DistilBERT
complaint classifier compressed and running as a SageMaker endpoint.
```

Rationale: Same stale-numbering table.

### Edit 6.3 - cell 5, code comment and print referencing "7a"

OLD:
```
# Recall from Topic 7a: we built LoRA matrices A and B by hand.
# Reminder of what those dimensions meant.
lora_r = 8   # rank, carried forward from 7a
print(f"LoRA rank from 7a: {lora_r}")
```

NEW:
```
# LoRA decomposes a weight update into two small matrices A and B of rank r.
# The mini-lesson below explains rank, alpha, and dropout in detail.
lora_r = 8   # rank: a small r means very few trainable parameters
print(f"LoRA rank: {lora_r}")
```

Rationale: The cell assumed the student carried `lora_r` and the A/B intuition from a
required prior topic. Define the value locally and point at the R8 mini-lesson (which
the editing agent inserts immediately after this cell - see the R8 section).

### Edit 6.4 - cell 8, "From Scratch to Library" section intro

OLD:
```
## Beat 1 -- Section 1: From Scratch to Library, PEFT LoRA on DistilBERT


In Topic 7a you implemented LoRA by hand:

  output = W_frozen @ x + (B @ A) @ x

The PEFT library automates that injection for any HuggingFace model.
Three function calls replace two custom classes and a manual layer-replacement loop.

### Beat 1: What happens if we try to apply LoRA without PEFT?
```

NEW:
```
## Beat 1 -- Section 1: From Scratch to Library, PEFT LoRA on DistilBERT


From the mini-lesson above, LoRA looks like this:

  output = W_frozen @ x + (B @ A) @ x

W_frozen stays fixed; only the small matrices A and B are trained. Doing that injection
by hand for every attention projection is tedious and error-prone. The PEFT library
automates it for any HuggingFace model. Three function calls replace two custom classes
and a manual layer-replacement loop. (The optional notebook topic_optional_lora_ffn
shows the by-hand build in full.)

### Beat 1: What happens if we try to apply LoRA without PEFT?
```

Rationale: Same reframe - present the LoRA formula as a callback to the R8 mini-lesson
in this notebook, not as "recall from Topic 7a", and point at the optional build.

### Edit 6.5 - cell 9, code docstring and comment

OLD:
```
class NaiveLoraLinear(nn.Module):
    """Manual LoRA wrapper from Topic 7a, applied naively to DistilBERT."""
```

NEW:
```
class NaiveLoraLinear(nn.Module):
    """Hand-rolled LoRA wrapper (the optional-deep-dive style), applied naively to DistilBERT."""
```

Also update the comment two lines below:

OLD:
```
# Manual attempt: replace q_lin in each attention block with a custom linear.
# This is what we did in 7a. Let's see why it is incomplete here.
```
NEW:
```
# Manual attempt: replace q_lin in each attention block with a custom linear.
# This is the hand-rolled style. Let's see why it is incomplete here.
```

Rationale: Remove the "we did this in 7a" assumption; the hand-rolled style is shown in
the optional notebook, not a required prior topic.

### Edit 6.6 - cell 11, code comments referencing "Topic 7a"

OLD:
```
# This replaces the entire manual injection from Topic 7a with 5 lines.
```
NEW:
```
# This replaces the entire hand-rolled manual injection with 5 lines.
```

OLD:
```
    r=lora_r,                      # rank from Topic 7a (r=8)
```
NEW:
```
    r=lora_r,                      # rank r=8 (see the LoRA mini-lesson above)
```

Rationale: Drop the "Topic 7a" attribution; keep the technical content and point at the
R8 mini-lesson.

### Edit 6.7 - cell 51, wrap-up "Bridge to Topic 7"

OLD:
```
### Bridge to Topic 7

Next: Topic 7 - Quantization. We take a pretrained classifier and shrink its
footprint with post-training quantization, pruning, and distillation so it can
run on smaller, cheaper instances at inference time.
```

NEW: no change. Topic 7 is still Quantization in the new numbering, and this bridge is
correct as written. Leave cell 51 as is.

### Edit 6.8 - cell 52, capstone "Situation" paragraph

OLD:
```
### Situation
You have completed the full Day 2 arc: architecture (T4), pre-trained models (T5),
full fine-tuning (T6a), transfer learning (T6b), LoRA from scratch (T7a), and PEFT
library (T7b). The Barclays ML Platform team now asks you to design and document the
production PEFT strategy for a new complaint routing system.
```

NEW:
```
### Situation
You have completed the core training arc of the course: pre-trained models from the
Hub (Topic 3), full fine-tuning (Topic 4), transfer learning (Topic 5), and the PEFT
library (Topic 6, this notebook). The Barclays ML Platform team now asks you to design
and document the production PEFT strategy for a new complaint routing system.
```

Rationale: The capstone summary listed the old T4-T7b path including
architecture-from-scratch and LoRA-from-scratch as required deliverables. Replace with
the new required arc (Topics 3-6).

### Edit 6.9 - cell 54, "Day 2 Complete" recap table

OLD:
```
## Day 2 Complete

You have built the Barclays Customer Support Intelligence System training pipeline:

| Topic | What you built | Key insight |
|-------|---------------|-------------|
| T4 | Transformer encoder-decoder from scratch | Parallelism beats RNNs |
| T5 | HuggingFace Hub inference pipeline | Pre-trained beats from-scratch |
| T6a | Full fine-tuning with catastrophic forgetting | Adapting costs memory |
| T6b | Transfer learning (frozen encoder + trained head) | Freeze expensive, train cheap |
| T7a | LoRA from scratch on a feed-forward network | Low-rank beats full delta |
| T7b | PEFT library LoRA + QLoRA on DistilBERT | Library beats hand-rolled |

**Day 3 preview**: Quantization, pruning, and distillation -- making the trained model
smaller and faster for production serving without sacrificing accuracy.
```

NEW:
```
## The Training Arc, Complete

You have built the Barclays Customer Support Intelligence System training pipeline:

| Topic | What you built | Key insight |
|-------|---------------|-------------|
| Topic 3 | HuggingFace Hub inference pipeline | Pre-trained beats from-scratch |
| Topic 4 | Full fine-tuning with catastrophic forgetting | Adapting costs memory |
| Topic 5 | Transfer learning (frozen encoder + trained head) | Freeze expensive, train cheap |
| Topic 6 | PEFT library LoRA + QLoRA on DistilBERT | Library beats hand-rolled |

**Next preview**: Topic 7 - Quantization, pruning, and distillation -- making the
trained model smaller and faster for production serving without sacrificing accuracy.
```

Rationale: The recap table listed transformer-from-scratch (old T4) and
LoRA-from-scratch (old T7a) as required deliverables and used old numbering. Replace
with the actual required Topics 3-6. "Day 2 Complete" no longer maps; generalize.

### Edit 6.10 - cell 54, "What Comes Next: Day 3" block

OLD:
```
### What Comes Next: Day 3

Your PEFT-adapted model is accurate -- but it is still float32 and larger than a production endpoint wants.
Topic 7 (Day 3) teaches you to compress it without losing accuracy:
Post-Training Quantization, Quantization-Aware Training, structured pruning, and knowledge distillation.
Then you will serve it from a real SageMaker endpoint -- replacing the GPT-4o API call from Topic 1.
```

NEW:
```
### What Comes Next

Your PEFT-adapted model is accurate -- but it is still float32 and larger than a production endpoint wants.
Topic 7 teaches you to compress it without losing accuracy:
Post-Training Quantization, Quantization-Aware Training, structured pruning, and knowledge distillation.
Then you will serve it from a real SageMaker endpoint -- replacing the GPT-4o API call from Topic 1.
```

Rationale: Topic 7 is correct; only the "Day 3" framing is stale. Drop the day labels.
The reference to "the GPT-4o API call from Topic 1" is correct and stays.

---

## Topic 7 - quantization

File(s): `Exercises/topic_7_quantization/topic_7_quantization.ipynb` and the Solutions
twin. Topic 7 mostly references "Topic 6 / T7b" as its upstream; T7b is the old name
for what is now Topic 6.

### R6 self-containment note for Topic 7

Topic 7 cells 7, ~17, and ~21 save and reload a `model.pt` file inside a `tempfile`
directory created WITHIN the same notebook (`os.path.join(tmpdir, "model.pt")`). These
are self-contained round-trips, not loads of an optional-notebook artifact. No change
is needed for R6 in topic_7. Cell 7 already re-loads `distilbert-base-uncased` fresh
and states the notebook is self-contained. Confirmed: no required notebook loads
`attention_weights.npy`, `translator_checkpoint.pt`, or any optional-notebook artifact.
R6 is satisfied across all seven required notebooks.

### Edit 7.1 - cell 0, opening and prerequisites

OLD:
```
In Topic 6 you assembled the Barclays Complaint Intelligence System using PEFT/LoRA on DistilBERT.
That model is accurate. Now you will make it production-ready: smaller, faster, and cheap to serve.

This notebook picks up from where Topic 6 left off. The PEFT checkpoint you saved in T7b is the
starting point for the compression pipeline you build here.

**Note**: Each section loads the checkpoint fresh from S3 -- you do not need to re-run T7b.
The kernel state is independent, but the logical progression is continuous.
```

NEW:
```
In Topic 6 you assembled the Barclays Complaint Intelligence System using PEFT/LoRA on DistilBERT.
That model is accurate. Now you will make it production-ready: smaller, faster, and cheap to serve.

This notebook picks up from where Topic 6 left off. The PEFT checkpoint you saved in Topic 6 is the
starting point for the compression pipeline you build here.

**Note**: Each section loads the checkpoint fresh from S3 -- you do not need to re-run Topic 6.
The kernel state is independent, but the logical progression is continuous.
```

Rationale: "T7b" is the old label for what is now Topic 6. Use the new name.

### Edit 7.2 - cell 0, prerequisites bullet

OLD:
```
- Completed Topic 6 or have a DistilBERT PEFT checkpoint in S3
```
NEW: no change. This is already correct.

### Edit 7.3 - cell 6, "Day 3 System Overview" table and surrounding text

OLD:
```
> **Note**: Each section loads checkpoints from S3. Kernel state is fresh, but logically this continues from T7b.

## Day 3 System Overview -- Where You Are in the Journey

This is the FINAL topic of the course. By the time you finish Topic 7, the Barclays
complaint-intelligence assistant is production-ready and cost-efficient.

| Day | Topics | What You Built |
|-----|--------|---------------|
| Day 1 | T1-T4 | GPT-4o prototype --> attention --> full Transformer from scratch |
| Day 2 | T5-T7b | HuggingFace ecosystem --> fine-tuning --> PEFT/LoRA assembly |
| **Day 3** | **T8 (this notebook)** | **Quantization + pruning + distillation --> deployable endpoint** |

### YOU ARE HERE: Day 3 -- Topic 7 (1 of 1)

The fine-tuned, LoRA-adapted DistilBERT you shipped in T7b is accurate but heavy.
Topic 7 teaches you to make it fast and cheap without sacrificing quality:
```

NEW:
```
> **Note**: Each section loads checkpoints from S3. Kernel state is fresh, but logically this continues from Topic 6.

## Where You Are in the Journey

This is the final required topic before the capstone. By the time you finish Topic 7, the
Barclays complaint-intelligence assistant is production-ready and cost-efficient.

| Stage | Topics | What You Built |
|-------|--------|---------------|
| Foundations | Topic 1-2 | GPT-4o prototype, then the LLM taxonomy, transformer concepts, and tokenization |
| Training | Topic 3-6 | HuggingFace ecosystem, fine-tuning, transfer learning, PEFT/LoRA |
| **Compression** | **Topic 7 (this notebook)** | **Quantization + pruning + distillation --> deployable endpoint** |

### YOU ARE HERE: Topic 7

The fine-tuned, LoRA-adapted DistilBERT you shipped in Topic 6 is accurate but heavy.
Topic 7 teaches you to make it fast and cheap without sacrificing quality:
```

Rationale: R9. The Day 1/2/3 table used the old T1-T7b numbering and listed
"attention --> full Transformer from scratch" as required Day 1 content - both are now
optional. Replace with the new required arc (Topics 1-7); the Foundations row mentions
transformer concepts because Topic 2 now teaches them (R7 mini-lesson). "FINAL topic of
the course" is no longer strictly true since topic_8 (agent capstone) is planned;
soften to "final required topic before the capstone". "T7b" becomes "Topic 6".

### Edit 7.4 - cell 7, code comment

OLD:
```
# Load a fresh DistilBERT classifier. Note: this is NOT carried over from Topic 6
# kernel state. We re-load distilbert-base-uncased here so Topic 7 is self-contained
# and students who lost their T7b kernel can still run all compression demos.
```

NEW:
```
# Load a fresh DistilBERT classifier. Note: this is NOT carried over from Topic 6
# kernel state. We re-load distilbert-base-uncased here so Topic 7 is self-contained
# and students who lost their Topic 6 kernel can still run all compression demos.
```

Rationale: Replace the stale "T7b" label with "Topic 6". The self-containment is
already correct and good.

### Edit 7.5 - cell 0 and cell 6, distillation reference to "T7b DistilBERT"

In cell 0, "What You Will Build" bullet 4:

OLD:
```
4. Run knowledge distillation -- transfer T7b DistilBERT knowledge to a smaller student
```
NEW:
```
4. Run knowledge distillation -- transfer Topic 6 DistilBERT knowledge to a smaller student
```

In cell 6, the matching distillation bullet:

OLD:
```
- **Knowledge Distillation**: transfer T7b DistilBERT knowledge into a smaller student model
```
NEW:
```
- **Knowledge Distillation**: transfer Topic 6 DistilBERT knowledge into a smaller student model
```

Rationale: "T7b" is the old name for Topic 6.

### Edit 7.6 - cell 63, "Course Complete" pipeline table

OLD:
```
| Stage | Topic | What You Built | Status |
|-------|-------|---------------|--------|
| Prototype | T1 | GPT-4o complaint router (raw API call) | Done in Day 1 |
| Understanding | T2 | LLM taxonomy: BERT vs GPT vs T5 families | Done in Day 1 |
| Attention | T3a/T3b | Bahdanau + scaled dot-product attention from scratch | Done in Day 1 |
| Architecture | T4 | Full Transformer encoder/decoder from scratch | Done in Day 1 |
| Ecosystem | T5 | HuggingFace Hub, tokenizers, pipeline API | Done in Day 2 |
| Fine-Tuning | T6a | Full fine-tune Flan-T5; measure catastrophic forgetting | Done in Day 2 |
| Transfer | T6b | Transfer learning with DistilBERT on SST-2 | Done in Day 2 |
| PEFT Theory | T7a | LoRA math: low-rank decomposition, FFN injection | Done in Day 2 |
| PEFT Practice | T7b | PEFT library end-to-end; Day 2 capstone assembly | Done in Day 2 |
| **Compression** | **T8** | **PTQ + QAT + pruning + distillation + endpoint** | **Done -- YOU ARE HERE** |
```

NEW:
```
| Stage | Topic | What You Built | Status |
|-------|-------|---------------|--------|
| Prototype | Topic 1 | GPT-4o complaint router (raw API call) | Done |
| Understanding | Topic 2 | LLM taxonomy and a transformer-concepts mini-lesson | Done |
| Ecosystem | Topic 3 | HuggingFace Hub, tokenizers, pipeline API | Done |
| Fine-Tuning | Topic 4 | Full fine-tune DistilBERT; measure catastrophic forgetting | Done |
| Transfer | Topic 5 | Transfer learning with DistilBERT on SST-2 | Done |
| PEFT | Topic 6 | LoRA mechanics mini-lesson + PEFT library: LoRA and QLoRA on DistilBERT | Done |
| **Compression** | **Topic 7** | **PTQ + QAT + pruning + distillation + endpoint** | **Done -- YOU ARE HERE** |
```

Rationale: R9. The recap table listed attention-from-scratch (T3a/T3b) and
Transformer-from-scratch (T4) as required Day-1 deliverables - both are now optional and
removed. It used old T6a/T6b/T7a/T7b numbering. It claimed full fine-tuning was on
Flan-T5; Topic 4 fine-tunes DistilBERT, so "Flan-T5" is corrected to "DistilBERT". The
old T7a "PEFT Theory" row is folded into the Topic 6 row, which now also names the R8
mini-lesson. The "T8" row becomes "Topic 7". The "Done in Day N" detail is reduced to
"Done" since day boundaries shifted. Topic 2 and Topic 6 rows mention the new
mini-lessons so the recap is accurate about what was taught.

### Edit 7.7 - cell 63, "What Comes Next" RLHF bullet

OLD:
```
- Explore RLHF (Topic 9, time-permitting) to align your model with human feedback
```

NEW:
```
- Explore RLHF to align your model with human feedback (an optional advanced direction)
```

Rationale: R9. There is no Topic 9 in the new structure (the planned next topic is
topic_8, an agent capstone). Drop the dangling topic number; keep RLHF as a suggested
direction.

### Edit 7.8 - cell 63, "Course Complete" congratulations / "T8" wording

OLD (the cell also contains lines that congratulate finishing "T8" and frame the start
as "T1 (Day 1 start)" / end as "T8 (Day 3 end)"):
```
**T1 (Day 1 start)**:
```
NEW:
```
**Topic 1 (course start)**:
```

OLD:
```
**T8 (Day 3 end)**: Your own model, on your own endpoint, under your control:
```
NEW:
```
**Topic 7 (required path complete)**: Your own model, on your own endpoint, under your control:
```

Rationale: R9. The notebook congratulates the student on finishing "T8" - there is no
T8 quantization topic; this notebook IS Topic 7. The planned topic_8 is a separate,
not-yet-built agent capstone. Rename "T1" to "Topic 1" and "T8" to "Topic 7" so the
end-of-notebook framing matches the new numbering and does not imply the student
finished a topic that does not exist. The editing agent must match these by the quoted
OLD strings; if the surrounding heading also says "Course Complete", leave that heading
as is (the required path IS complete at Topic 7; topic_8 is optional/future).

Note: cell 26 ("trained from scratch to the same accuracy") is the Lottery Ticket
Hypothesis definition - a general ML statement, correct as written, no change. cell 39
("LoRA: fine-tune efficiently (from Topic 6)") is correct under the new numbering, no
change. cell 54 ("Replacing the Topic 1 API Call", "Topic 1 prototype") is correct, no
change.

---

## R7 Mini-Lesson: Transformer Concepts (new content for topic_2)

### Placement decision and justification

The R7 mini-lesson lives in **topic_2 (introducing_llms)**, NOT topic_3. Justification:

- topic_2 already establishes tokenization, embeddings, the three transformer families,
  and "N transformer blocks" as a black box (cell 35-37). The mini-lesson fills exactly
  the gap topic_2 itself names in cell 37 ("we skipped straight to the output").
- topic_3 (huggingface) is about USING models with a 4-line API; inserting architecture
  theory there would break its application-focused arc.
- Downstream notebooks that need the concepts - topic_4 attention-head and CLS-token
  discussion, topic_5 CLS-pooling diagram, topic_7 attention-head pruning - all come
  AFTER topic_2, so teaching it in topic_2 makes the concepts available everywhere they
  are used.
- It is concept-level only: Q/K/V intuition, positional encoding, multi-head attention,
  the encoder block. NO from-scratch derivation, NO training loop. The from-scratch
  build stays in the optional notebooks (R12 split: concept = required, build =
  optional).

The mini-lesson is inserted as a new Section 5.5, immediately AFTER cell 36 (the
pipeline-trace code cell) and BEFORE cell 37 ("What We Did NOT Cover Today"). This
placement means cell 37's edited "in depth" framing (Edit 2.2) correctly refers back to
a mini-lesson the student has just seen.

Insert FIVE new cells after cell 36 (using NotebookEdit with `cell_id` of cell 36, then
chaining). All five cells are identical in the Exercises and Solutions copies. Cadence:
this is one 5-cell batch; run /validate-notebooks after inserting.

### New cell A (markdown) - "Section 5.5: How Self-Attention Works"

```
## Section 5.5 -- How Self-Attention Works (Concept Mini-Lesson)

We have called the middle of a transformer "N transformer blocks" and treated it as a
black box. This mini-lesson opens that box at the CONCEPT level. You will not derive
any math or write a training loop here. The goal is narrower and practical: after this
section you can read an attention-head heatmap, explain what the [CLS] token is, and
reason about why transformers handle long-range dependencies better than RNNs.

If you later want to BUILD attention and a full transformer from scratch, that is what
the optional deep-dive notebooks are for (topic_optional_attention_python,
topic_optional_attention_pytorch, topic_optional_transformers). They are not required.

### The problem attention solves

A complaint reads: "The branch told me the refund was processed, but it never arrived."
To classify this correctly the model must connect "refund" with "never arrived" - words
that are eight tokens apart. An RNN reads left to right and compresses everything seen
so far into one fixed-size hidden state, so distant words get diluted. Self-attention
removes that bottleneck: every token can look directly at every other token in one step.

### Queries, Keys, and Values

Self-attention gives each token three learned vectors:

- Query (Q): "what am I looking for?"
- Key (K): "what do I offer to others?"
- Value (V): "what information do I carry?"

For one token, the model compares its Query against every token's Key (a dot product).
High dot product means "these two are relevant to each other". Those scores are scaled
by the square root of the key dimension (this keeps gradients stable), then passed
through a softmax so they become weights that sum to 1. The token's new representation
is the weighted sum of every token's Value. In short:

  attention output = softmax( Q dot K_transposed / sqrt(d_k) ) times V

That is the whole operation. The word "fraud" ends up with a representation that has
pulled in information from "unauthorised" and "charge" because those tokens scored high
against its Query.
```

### New cell B (code) - small runnable demo of one attention step

This is the mandatory runnable code cell (CLAUDE.md no-3-markdown-chain rule). It is a
fully-worked demo, identical in Exercises and Solutions (not a `# YOUR CODE` lab).

```
# Concept demo: one self-attention step over a tiny complaint sequence.
# This is a CONCEPT illustration, not a from-scratch build. We use small random
# Q, K, V so you can watch the weights form. The optional deep-dive notebooks
# implement the real thing properly.
import numpy as np

np.random.seed(0)

tokens = ["the", "refund", "never", "arrived"]
d_k = 8  # key/query dimension

# Each token gets a random Query, Key, Value vector (a real model LEARNS these).
Q = np.random.randn(len(tokens), d_k)
K = np.random.randn(len(tokens), d_k)
V = np.random.randn(len(tokens), d_k)

# Step 1: score every Query against every Key, then scale by sqrt(d_k).
scores = (Q @ K.T) / np.sqrt(d_k)

# Step 2: softmax each row so the weights sum to 1.
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights = weights / weights.sum(axis=1, keepdims=True)

# Step 3: each token's new vector is the weighted sum of all Values.
attended = weights @ V

print("Attention weight matrix (rows = query token, columns = key token):")
print("        " + "  ".join(f"{t:>8}" for t in tokens))
for tok, row in zip(tokens, weights):
    print(f"{tok:>8} " + "  ".join(f"{w:8.3f}" for w in row))
print()
print("Each row sums to 1.0:", np.allclose(weights.sum(axis=1), 1.0))
print("Output shape (one new vector per token):", attended.shape)
print()
print("Reading this matrix: row 'refund' shows how much 'refund' attends to each")
print("other token. After training on real data, semantically related tokens get")
print("the high weights. This same matrix is what an attention-head heatmap shows.")
```

### New cell C (markdown) - "Multi-Head Attention and Positional Encoding"

```
### Multi-Head Attention

One set of Q/K/V vectors learns one kind of relationship. Real transformers run several
in parallel - these are the attention HEADS. DistilBERT has 12 heads per layer. One head
may learn "this pronoun refers to that noun", another "this adjective modifies that
noun". Each head produces its own weighted-sum output; the outputs are concatenated and
projected back down. When you see an attention-head visualization (you will, in Topic 4
and Topic 7), you are looking at ONE head's weight matrix - exactly the matrix the demo
above printed.

### Positional Encoding

Self-attention has no built-in sense of order: "refund never arrived" and "never refund
arrived" would look identical to it, because the weighted sum does not depend on
position. To fix this, transformers ADD a position signal to each token embedding
before the first block. This is positional encoding. The concept is all you need here:
order information is injected once, up front, so attention can use it. The exact
sinusoidal formula and the proof that it encodes relative distance are in the optional
transformers deep-dive.

### The Encoder Block

Stack these pieces and you have one encoder block:

1. Multi-head self-attention (every token looks at every token)
2. Add and LayerNorm (a residual connection plus normalization for stable training)
3. A small feed-forward network applied to each token
4. Add and LayerNorm again

DistilBERT stacks 6 of these blocks. "N transformer blocks" from Section 6 is exactly
this, repeated N times.
```

### New cell D (markdown) - "The CLS Token"

```
### The [CLS] Token

In Section 2 you saw BERT-style tokenizers prepend a special [CLS] token to every
input. Now it makes sense. [CLS] is a token with no word meaning of its own. As it
passes through the attention blocks, it attends to every real token and accumulates a
summary of the whole sequence. After the final block, the vector sitting at the [CLS]
position is used as the sentence representation: a classification head reads that single
vector to predict a label.

This is why, in Topic 4 and Topic 5, the complaint classifier takes the [CLS] vector and
feeds it to a small trainable head. The encoder does the understanding; the [CLS] vector
is where that understanding is collected; the head turns it into a Barclays complaint
category. You now have everything you need to follow that discussion.
```

### New cell E (markdown) - "Mini-Lesson Recap" plus discussion prompt

```
### Mini-Lesson Recap

- Self-attention lets every token look directly at every other token, removing the RNN
  bottleneck for long-range dependencies.
- Q, K, V: a token's Query is matched against all Keys to produce weights; the output
  is the weighted sum of Values.
- Multi-head attention runs several of these in parallel; each head is one heatmap.
- Positional encoding adds order information, because attention itself is order-blind.
- An encoder block = multi-head attention + add-and-norm + feed-forward + add-and-norm;
  DistilBERT stacks 6 of them.
- The [CLS] token accumulates a whole-sequence summary that a classification head reads.

**Peer discussion (3-5 min)**: A teammate says "we should just use an RNN, it is
simpler". Using what you just learned, give two concrete reasons a transformer encoder
will classify long Barclays complaints more reliably. Then name one cost of that choice
(hint: think about how attention scales with sequence length).
```

Note: cells A, C, D, E are four markdown cells, but cell B (code) sits between A and C,
so the longest markdown run is three (C, D, E) - within the CLAUDE.md limit. The
preceding cell (36) is code and the following cell (37) is markdown, so no violation is
introduced.

---

## R8 Mini-Lesson: LoRA Mechanics (new content for topic_6)

### Placement decision and justification

The R8 mini-lesson lives in **topic_6 (peft_lora_distilbert)**, inserted immediately
AFTER cell 5 (the setup cell that defines `lora_r`, edited by Edit 6.3) and BEFORE
cell 6 (the CUDA health check). Justification:

- Edit 6.3 makes cell 5 say "the mini-lesson below explains rank, alpha, and dropout";
  the mini-lesson must therefore be the next thing the student sees.
- Edit 6.1 (cell 0) announces "a short mini-lesson below explains the mechanics".
- It must come before cell 8 ("From Scratch to Library"), which Edit 6.4 rewrites to say
  "From the mini-lesson above, LoRA looks like this".
- Concept-level only: what low-rank decomposition is, what rank / alpha / dropout do,
  and why it works. The from-scratch build stays in topic_optional_lora_ffn (R12 split).

Insert THREE new cells after cell 5. All three are identical in Exercises and Solutions
copies. The code cell is a fully-worked demo, not a `# YOUR CODE` lab, so no safety-net
cell is needed. Cadence: one batch of 3 cells; run /validate-notebooks after inserting.

### New cell A (markdown) - "LoRA Mechanics: What the Knobs Do"

```
## LoRA Mechanics -- A Concept Mini-Lesson

Before you call the PEFT library, you need to understand what it is doing, so you can
reason about the knobs you tune. This mini-lesson is concept-level. If you want to see
LoRA implemented by hand on a feed-forward network, the optional deep-dive notebook
topic_optional_lora_ffn does exactly that; it is not required here.

### The problem: full fine-tuning updates everything

Fine-tuning a transformer the full way (Topic 4) updates every weight matrix W. For
DistilBERT that is about 66 million parameters, and the Adam optimizer needs extra
memory for each one. That is expensive, and you get a whole new copy of the model per
task.

### The idea: the UPDATE is low-rank

Fine-tuning changes a weight matrix W into W + deltaW. The key empirical finding behind
LoRA (Hu et al., 2021) is that deltaW - the CHANGE - does not need to be full rank. It
can be well approximated by the product of two much smaller matrices:

  deltaW (shape d by k)  is approximated by  B (shape d by r) times A (shape r by k)

Here r, the RANK, is small - typically 4, 8, or 16. Instead of training all d times k
numbers in deltaW, you train only r times (d + k) numbers in A and B. For a 768 by 768
attention projection with r = 8, that is about 12,000 trainable numbers instead of
about 590,000 - roughly 50 times fewer.

During training W is FROZEN. Only A and B are learned. At inference the layer computes:

  output = W x  +  (B A) x

A is initialized random, B is initialized to zero, so at the start B A = 0 and the
model behaves exactly like the pretrained one - training then nudges it from there.
```

### New cell B (code) - small runnable demo of low-rank decomposition

This is the mandatory runnable code cell. Fully-worked demo, identical in both copies.

```
# Concept demo: how many parameters does LoRA actually train?
# This counts parameters for one attention projection. It is an illustration of the
# mechanics, not the real injection (the PEFT library does that later in this notebook).

d, k = 768, 768   # a DistilBERT attention projection is 768 x 768

full_finetune = d * k                      # every number in delta-W

for r in [4, 8, 16, 32]:
    # LoRA trains A (r x k) and B (d x r) instead of the full delta-W.
    lora_params = r * k + d * r
    ratio = full_finetune / lora_params
    print(f"rank r = {r:>2}:  LoRA trains {lora_params:>8,} params  "
          f"vs {full_finetune:,} full  ->  {ratio:5.1f}x fewer")

print()
print("Bigger r = more capacity to fit the new task, but more parameters and more")
print("risk of overfitting a small dataset. Smaller r = cheaper and more regularized.")
print("r = 8 is the common default for DistilBERT-sized models.")
```

### New cell C (markdown) - "Rank, Alpha, Dropout" plus discussion prompt

```
### The Three Knobs You Tune

**rank (r)**: the inner dimension of A and B. It sets how much the adapter can change
the model. Small r (4-8) is cheap and resists overfitting on small datasets; larger r
(16-32) gives more capacity for harder or more distant tasks. The demo above shows the
parameter cost of each choice.

**lora_alpha**: a scaling factor. The adapter's contribution is scaled by alpha / r
before it is added to the frozen path. Think of it as a learning-rate-like gain on the
adapter: raising alpha makes the adapter's effect stronger, lowering it makes the
adapter more conservative. A common convention is to set alpha = 2 times r, so the
effective scale stays steady as you change r.

**lora_dropout**: ordinary dropout applied to the adapter input during training. It
randomly zeroes some activations so the adapter cannot rely on any single feature,
which regularizes a small adapter trained on a small dataset. Typical values are 0.05
to 0.1. It is active during training only and does nothing at inference.

### Why it works

The pretrained model already knows language. Adapting it to Barclays complaints is a
small, low-dimensional shift, not a from-scratch relearning - so a low-rank deltaW is
enough. Because W stays frozen, the original knowledge cannot be overwritten: this is
why LoRA does not suffer the catastrophic forgetting you saw in Topic 4. And because
A and B are tiny, you can store one small adapter per task instead of a full model copy.

**Peer discussion (3-5 min)**: You fine-tune a DistilBERT complaint classifier with
LoRA at r = 4 and accuracy plateaus below full fine-tuning. Would you raise r, raise
lora_alpha, or both - and what is the risk of each? When would you instead leave r low
on purpose?
```

Note: cells A, C are markdown with code cell B between them - no markdown-chain
violation. Cell 5 before is code, cell 6 after is markdown; the longest resulting
markdown run is two. Within the CLAUDE.md limit.

---

## R11: Diagram Files - Audit Result

CODEX R11 states that Mermaid diagram captions still embed stale "YOU ARE HERE -> Topic
N" progressions. Every `.mmd` file under `plans/<topic>/diagrams/` for the seven
required topics, and every optional-topic `.mmd`, was read in full and grep-checked for
the patterns `YOU ARE HERE`, `Topic <number>`, `T<number>[ab]`, and `Day <number>`.

**Audit result: NO `.mmd` file contains a stale "YOU ARE HERE -> Topic N" caption or
any topic-number / day-number reference.** The diagram source files are purely
structural (nodes, edges, formulas). The grep hits that look like matches are false
positives inside technical content: "T5-small" / "Flan-T5" as model names in
`transformer-families.mmd` and `lora-parameter-comparison.mmd`; "T=1 / T=4" as the
distillation temperature in `knowledge-distillation-architecture.mmd`; "INT8 / INT4" as
quantization precisions in `quantization-precision-tradeoffs.mmd`. None of these are
course-position captions.

Required-topic diagram files audited (all clean, NO edit needed):

| File | Verdict |
|------|---------|
| plans/topic_1_overview_genai/diagrams/autoregressive-loop.mmd | clean |
| plans/topic_1_overview_genai/diagrams/openai-api-request-response-flow.mmd | clean |
| plans/topic_2_introducing_llms/diagrams/genai-lifecycle.mmd | clean |
| plans/topic_2_introducing_llms/diagrams/transformer-families.mmd | clean (T5/Flan-T5 are model names) |
| plans/topic_3_huggingface/diagrams/automodel-class-hierarchy.mmd | clean |
| plans/topic_3_huggingface/diagrams/hf-hub-ecosystem.mmd | clean |
| plans/topic_4_full_finetuning/diagrams/catastrophic-forgetting.mmd | clean |
| plans/topic_4_full_finetuning/diagrams/full-finetuning-parameter-update.mmd | clean |
| plans/topic_5_transfer_learning/diagrams/tl-vs-finetuning-comparison.mmd | clean |
| plans/topic_5_transfer_learning/diagrams/transfer-learning-arch.mmd | clean |
| plans/topic_6_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd | clean |
| plans/topic_6_peft_lora_distilbert/diagrams/qlora-architecture.mmd | clean |
| plans/topic_7_quantization/diagrams/knowledge-distillation-architecture.mmd | clean (T=1/T=4 is temperature) |
| plans/topic_7_quantization/diagrams/quantization-precision-tradeoffs.mmd | clean (INT8/INT4 are precisions) |

Optional-topic diagram files (all clean, NO edit needed): bahdanau-score-computation,
seq2seq-bottleneck-vs-attention, attention-heatmap-complaint, scaled-dot-product-formula,
lora-decomposition, lora-parameter-comparison, positional-encoding-pattern,
transformer-architecture.

### Where the stale captions ACTUALLY live: in-notebook diagram text

The stale course-position text R11 is concerned about is NOT in the `.mmd` files. It is
in the NOTEBOOK markdown cells - the `<!-- DIAGRAM: -->` placeholder comments and the
prose captions that surround the embedded diagram blocks. Those are markdown cells and
are covered by the per-notebook edits above. The specific in-notebook diagram-related
text already handled:

- topic_2 cell 37 / cell 39 (Edits 2.2, 2.5): the "Topic 3 (Attention)" and roadmap
  text that introduces the transformer-families context.
- topic_5 cell ~50 (the transfer-learning-arch diagram caption block): the surrounding
  prose is part of the cells covered by Edits 5.3-5.5; the `<!-- DIAGRAM: -->` comment
  itself is structural and has no topic number.
- topic_7 cell 6 (Edit 7.3): the "Day 3 System Overview" text that sits next to the
  journey table is rewritten.

No `<!-- DIAGRAM: ... -->` placeholder comment in any of the seven required notebooks
contains a topic number or "YOU ARE HERE" string (verified by grep across all seven
`.ipynb` files). Therefore R11 requires NO diagram-file edits and NO additional
notebook edits beyond those already specified. R11 is resolved by this audit: the
premise that `.mmd` files carry stale captions does not hold for this repo; the
course-position text lives in the system-overview tables, which Edits 3.2, 4.1, 5.1,
6.2, and 7.3 already rebuild.

---

## Summary of edits

| Notebook | Continuity edits | New mini-lesson cells | Notes |
|----------|------------------|-----------------------|-------|
| topic_1_overview_genai | 2 | 0 | wrap-up forward references |
| topic_2_introducing_llms | 7 | 5 (R7 transformer mini-lesson) | roadmap reframes + new Section 5.5 |
| topic_3_huggingface | 7 | 0 | false "Topic 4 transformers" prereq, COMPLAINT_TOKENS |
| topic_4_full_finetuning | 8 (4.7 is a no-op) | 0 | system table + LoRA/PEFT renumbering; R10 verified clean |
| topic_5_transfer_learning | 5 | 0 | system table + LoRA handoff reframe |
| topic_6_peft_lora_distilbert | 10 (6.7 is a no-op) | 3 (R8 LoRA mini-lesson) | "you built LoRA in 7a" reframing |
| topic_7_quantization | 8 (7.2 is a no-op) | 0 | T7b->Topic 6, Flan-T5 fix, T8/T9 cleanup, recap table |

Notebooks needing changes: 7 of 7.
Continuity edits specified: 47 entries; 3 (4.7, 6.7, 7.2) are explicit "no change"
confirmations, leaving 44 actual replacement edits.
New cells inserted: 8 total (5 in topic_2 for the R7 mini-lesson, 3 in topic_6 for the
R8 mini-lesson).
Diagram files changed: 0 (R11 audit found no stale `.mmd` captions).

Every edit and every new-cell insertion applies to BOTH the `Exercises/` copy and the
`Solutions/` copy of the named notebook. The mini-lesson cells are fully-worked demos
and are identical in both copies (no `# YOUR CODE` blanks, no safety-net cell needed).

---

## Codex R1 findings resolved

| Finding | Resolution location in this doc |
|---------|---------------------------------|
| R6 (required notebooks load optional artifacts) | Edits 3.3, 4.2 (COMPLAINT_TOKENS / COMPLAINT_TEXTS comments corrected to "defined locally"); "R6 self-containment note for Topic 7" (model.pt is a same-notebook round-trip). Audit confirms no required notebook loads attention_weights.npy, translator_checkpoint.pt, or any optional artifact. |
| R7 (transformer concept never taught) | Dedicated section "R7 Mini-Lesson: Transformer Concepts" - 5 new cells inserted in topic_2 after cell 36 (Section 5.5: Q/K/V, multi-head attention, positional encoding, encoder block, CLS token, plus one runnable demo). Placement justified there. Edits 1.2, 2.2, 2.4, 2.6 reframe topic_2/topic_3 prose around the new mini-lesson. |
| R8 (LoRA mechanics hand-waved) | Dedicated section "R8 Mini-Lesson: LoRA Mechanics" - 3 new cells inserted in topic_6 after cell 5 (low-rank decomposition, rank/alpha/dropout, why it works, plus one runnable parameter-count demo). Edits 6.1, 6.3, 6.4, 6.6 reframe topic_6 prose around the new mini-lesson. |
| R9 (topic_7 stale end-of-course table, Flan-T5, T8, T9) | Edits 7.3, 7.6, 7.7, 7.8 - rebuild the journey table and the Course Complete table to Topics 1-7, correct "Flan-T5" to "DistilBERT", remove the "T8" congratulations and "Topic 9 RLHF" dangling reference. |
| R10 (topic_4 self-reference recap "from scratch in Topic 4", "RNN in 3a/3b") | "R10 verification note" under Topic 4 - the stale recap text was already removed in commit 13187c0; topic_4 cell 0 is clean. Editing agent must confirm the same strings are absent in the Solutions twin. Topic_4 edits 4.1-4.6 are the still-valid renumbering edits. |
| R11 (Mermaid diagram captions embed stale "YOU ARE HERE -> Topic N") | Dedicated section "R11: Diagram Files - Audit Result" - every `.mmd` file read and grep-checked; none contains a stale caption or topic number. The course-position text lives in notebook system-overview tables, rebuilt by Edits 3.2, 4.1, 5.1, 6.2, 7.3. 0 diagram-file edits required. |
| R12 (concept-vs-build contradiction) | "Canonical reframing language" section codifies the split: CONCEPT is the required mini-lesson (R7, R8), the from-scratch BUILD is optional. Every edit that mentions transformers, attention, or LoRA uses this split consistently (Edits 2.2, 2.4, 2.6, 3.1, 3.4, 5.5, 6.1, 6.4 and both mini-lesson sections). |

Note on R1-R5 and R13: these are OWNED by the optional-notebooks design doc, not this
required-path doc. They are listed in CODEX_FINDINGS_R1.md for completeness but are out
of scope here; this doc covers only the seven required notebooks.
