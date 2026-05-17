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

Insert SIX new cells after cell 36 (using NotebookEdit with `cell_id` of cell 36, then
chaining): A(md), B(code), C(md), D(md), E(code), F(md). All six cells are identical in
the Exercises and Solutions copies. Cadence: insert as a 5-cell batch then a 1-cell
batch (or two batches of 3); run /validate-notebooks after each batch.

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

### New cell E (code) - tiny runnable check, breaks the markdown run

Codex R2 finding N3: cells C and D are markdown and the original cell 37 (which
follows the mini-lesson) is also markdown. Without a code cell here the run would
be C, D, 37 plus the recap markdown = a 4-cell markdown chain, violating the
CLAUDE.md "no more than 3 consecutive markdown cells" rule. This short runnable
cell sits between the multi-head/encoder-block markdown and the recap, breaking
the chain. It is a fully-worked demo, identical in Exercises and Solutions.

```
# Concept check: confirm the multi-head numbers for DistilBERT.
# This uses only the config, no model download, no training.
from transformers import DistilBertConfig

cfg = DistilBertConfig()  # the standard distilbert-base-uncased configuration
print(f"Transformer (encoder) blocks : {cfg.n_layers}")
print(f"Attention heads per block    : {cfg.n_heads}")
print(f"Hidden size                  : {cfg.dim}")
print(f"Total attention heads        : {cfg.n_layers * cfg.n_heads}")
print()
print("Each of those heads produces one attention weight matrix - one heatmap -")
print("exactly like the matrix the demo above printed. When Topic 4 visualizes an")
print("attention head, it is showing one of these.")
```

### New cell F (markdown) - "Mini-Lesson Recap" plus discussion prompt

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

Note (markdown-run check, Codex R2 finding N3): the mini-lesson is six cells -
A(md), B(code), C(md), D(md), E(code), F(md). Cell B breaks the A-C run; cell E
(the DistilBertConfig check) breaks the C-D run AND separates D from cell F.
After cell F (markdown) comes the original cell 37 (also markdown), so the
longest markdown run anywhere is two (F, 37) - within the CLAUDE.md limit of 3.
The preceding cell 36 is code. No violation.

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

## S3 Handoff Chain

### Why this section exists

Codex (o3) Round 3 (findings C1, C2, C3 in `CODEX_FINDINGS_R1.md`) found that the
required path is not actually a spiral. Each notebook recreates its demo state from
scratch instead of extending one running Barclays Customer Support Intelligence System.
Topic 1 even ends with a "Variables available for Topic 2" banner that Topic 2 never
references. The "variables carry over exactly" rule (CLAUDE.md) is violated at the very
first handoff.

The fix, decided with Axel: cross-notebook handoff is done via a course S3 bucket, not
kernel variables. The class runs on SageMaker in AWS, where every notebook can read and
write a course bucket. Each required notebook ENDS by writing the artifacts the next
topic needs to a known S3 prefix, and STARTS (after setup) by reading the previous
topic's artifacts back, with a clear fallback if absent. This makes the spiral literal
and survives kernel restarts between sessions.

This section is implementable on its own: a notebook-build agent can apply it without
re-deriving anything. Every cell body below is complete, plain-ASCII Python.

### Scope and conventions for this section

- Every change in this section applies to BOTH copies of each notebook: the
  `Exercises/<topic>/<topic>.ipynb` file AND the matching `Solutions/<topic>/<topic>.ipynb`
  file. The handoff cells are fully-worked infrastructure code (no `# YOUR CODE` blanks),
  so they are byte-identical in the Exercises and Solutions copies.
- The handoff cells are NOT labs. They need no safety-net cell of the CLAUDE.md lab
  pattern; instead, the LOAD cell has its OWN built-in fallback (same spirit as a
  safety-net: if the S3 artifact is absent, define a minimal local version and print a
  clear note) so a student starting mid-course is never blocked.
- Plain ASCII only. No em-dashes, en-dashes, Unicode multiplication signs, emojis.
- Cell indices below are 0-based, read from the `Exercises/` copy at commit 13187c0.
  The `Solutions/` copy has the same ordering for these cells; if an index is off by
  one in a Solutions file, match on the quoted neighbouring cell content instead.
- Insertion cadence: the handoff cells are added in the same 5-cell-batch approval
  cadence as everything else (CLAUDE.md). Per notebook there are at most two new cells
  (one LOAD, one WRITE), so one batch per notebook is enough. Run `/validate-notebooks`
  after each notebook.
- These handoff cells are ADDITIVE. They do not replace any continuity edit above. Where
  a continuity edit already rewrites a wrap-up cell (for example topic_1 Edit on cell
  34, topic_4 Edit 4.8), the WRITE cell is inserted as a NEW cell AFTER that wrap-up
  cell, and the wrap-up prose is additionally adjusted as noted per topic below.

### Course-wide S3 key layout

All handoff artifacts live under a single course prefix in the SageMaker default bucket:

```
s3://<bucket>/barclays-course/topic_<N>/<artifact>
```

- `<bucket>` is the SageMaker default bucket for the account and region
  (`sagemaker.Session().default_bucket()`), the same bucket the remote-training topics
  already use. It resolves to `sagemaker-<region>-<account-id>`.
- `barclays-course/` is the fixed course namespace, so handoff artifacts never collide
  with the per-topic training-job output prefixes (`topic7b-peft/`, etc.) the notebooks
  already write.
- `topic_<N>/` is the PRODUCING topic's number (1 through 7). A topic always writes
  under its own number and reads from `topic_<N-1>/`.
- `<artifact>` is a stable file name. JSON for small structured data
  (`triage_config.json`, `label_map.json`, `dataset.json`), plain text for a single
  string, and a pointer JSON (`model_pointer.json`) for anything that is itself an S3
  object (a `model.tar.gz`), because an S3 model tarball is referenced by URI, not
  copied.

Full artifact map (the exact keys each topic writes):

| Topic | Writes to `barclays-course/topic_<N>/` |
|-------|-----------------------------------------|
| topic_1 | `triage_config.json` (system prompt + the 5 test complaints + routing categories) |
| topic_2 | `complaint_corpus.json` (the worked complaint texts + tokenizer name) |
| topic_3 | `labelled_dataset.json` (labelled complaint examples + `label_map`) and `routing_labels.json` |
| topic_4 | `model_pointer.json` (fine-tuned `model.tar.gz` S3 URI + `training_job_name` + label map) |
| topic_5 | `model_pointer.json` (transfer-learned `model.tar.gz` S3 URI + endpoint name if deployed) |
| topic_6 | `model_pointer.json` (PEFT/LoRA checkpoint `model.tar.gz` S3 URI + `training_job_name` + LoRA config) |
| topic_7 | `deployment.json` (final quantized endpoint name + compression summary). Last topic: nothing reads it yet; topic_8 capstone will. |

### Shared helper: the handoff module

Every LOAD and WRITE cell uses the same four tiny helpers. To avoid repeating them,
they are defined inline in each cell that needs them (the cells are self-contained;
notebooks do not share a kernel). The canonical helper block, reused verbatim:

```
# --- S3 handoff helpers (course-wide, identical in every notebook) ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def _handoff_key(topic_n, artifact):
    return f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"

def handoff_write(bucket, topic_n, artifact, obj):
    """Write a JSON-serialisable object to this topic's handoff prefix."""
    key = _handoff_key(topic_n, artifact)
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

def handoff_read(bucket, topic_n, artifact):
    """Read a JSON object from a topic's handoff prefix. Returns None if absent."""
    key = _handoff_key(topic_n, artifact)
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None
```

Topics 4, 5, 6, 7 already define `bucket` in their SageMaker setup cell, so their
handoff cells reuse it. Topics 1, 2, 3 have no SageMaker session today; their LOAD or
WRITE cell first resolves `bucket` with this snippet (added at the top of the relevant
handoff cell):

```
# Topics 1-3 have no sagemaker.Session(); resolve the course bucket directly.
import sagemaker
bucket = sagemaker.Session().default_bucket()  # sagemaker-<region>-<account-id>
```

`sagemaker` is already a dependency of topics 4-7. For topics 1-3 the install cell
gains `"sagemaker>=2.200.0,<3.0.0"` (a small, additive change to the existing `pip
install` cell; documented per topic below). If a fully offline / no-AWS student runs
topics 1-3, the LOAD fallback and a try/except around `default_bucket()` keep them
unblocked (see the per-topic fallback notes).

### Honest note on non-serialisable objects (C1)

The OpenAI `client` object created in topic_1 cell 20 CANNOT be serialised to S3 or
JSON. The handoff therefore does NOT attempt to carry the client. What persists to S3
is the DURABLE data: the triage system prompt and the test complaints. The `client` is
simply recreated from the API key with `getpass.getpass()` in any later notebook that
needs to call the OpenAI API (topic_7 already does exactly this in its endpoint-vs-API
comparison). Every LOAD cell that mentions the client says so explicitly. This is the
honest resolution of C1: prompt and complaints are state and persist; the client is a
live connection and is rebuilt, never stored.

---

### topic_1 - overview_genai (PRODUCES only, no predecessor)

LOADS: nothing. Topic 1 is first.

PRODUCES: `barclays-course/topic_1/triage_config.json` containing the triage system
prompt, the five test complaints, and the routing categories. These are the durable
artifacts behind the old "Variables available for Topic 2" banner.

Install-cell change: the topic_1 install cell (cell 2) gains `sagemaker` so the WRITE
cell can resolve the bucket.

OLD (cell 2):
```
!pip install -q "numpy<2" "openai>=1.0.0"
```
NEW (cell 2):
```
!pip install -q "numpy<2" "openai>=1.0.0" "sagemaker>=2.200.0,<3.0.0"
```

WRITE cell: insert ONE new code cell immediately AFTER cell 34 (the existing
"Notebook complete" session-summary cell). The continuity edit above already rewrites
cell 34's banner; this new cell turns the banner into a real S3 write.

Additionally, in cell 34 itself, the "Variables available for Topic 2:" block is
replaced so it no longer claims a kernel handoff:

OLD (cell 34, the banner block):
```
print("Variables available for Topic 2:")
print("  client           : OpenAI client (if key is still valid)")
print("  my_system_prompt : your triage prompt from Lab 2")
print("  test_complaints  : list of 5 test complaints")
print()
print("Next: Topic 2 - Introducing LLMs (what is inside the black box?)")
```
NEW (cell 34, the banner block):
```
print("Durable artifacts to save for Topic 2 (written to S3 in the next cell):")
print("  my_system_prompt : your triage prompt from Lab 2")
print("  test_complaints  : list of 5 test complaints")
print("Note: the OpenAI client is a live connection, not data - it cannot be saved.")
print("Later notebooks recreate it from your API key with getpass.")
print()
print("Next: Topic 2 - Introducing LLMs (what is inside the black box?)")
```

New WRITE cell (insert after cell 34), full content:
```
# Handoff to Topic 2: save the durable Barclays triage artifacts to the course S3 bucket.
# In the next topic you load these back and extend the system - this is the spiral.
# The OpenAI client is NOT saved: it is a live connection, recreated from your key later.

# Topics 1-3 have no sagemaker.Session(); resolve the course bucket directly.
import sagemaker
bucket = sagemaker.Session().default_bucket()  # sagemaker-<region>-<account-id>

# --- S3 handoff helpers (course-wide, identical in every notebook) ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def _handoff_key(topic_n, artifact):
    return f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"

def handoff_write(bucket, topic_n, artifact, obj):
    """Write a JSON-serialisable object to this topic's handoff prefix."""
    key = _handoff_key(topic_n, artifact)
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

# Routing categories the triage prompt uses (kept with the prompt for downstream topics).
routing_categories = [
    "payment_failure", "fraud_dispute", "account_access",
    "fee_complaint", "unclassified",
]

triage_config = {
    "system_prompt": my_system_prompt,
    "test_complaints": test_complaints,
    "routing_categories": routing_categories,
}

handoff_write(bucket, 1, "triage_config.json", triage_config)
print()
print("Topic 2 will load this and tokenize the same test complaints.")
```

Narrative line topic_1 adds (this is the print output above plus the rewritten cell 34
banner): "Topic 2 will load this and tokenize the same test complaints." No load cell
in topic_1 - it has no predecessor.

---

### topic_2 - introducing_llms

LOADS: `barclays-course/topic_1/triage_config.json` - the triage system prompt and the
five test complaints from Topic 1.

USES what it loads: the loaded `test_complaints` are fed into the Section 1 tokenization
demos and the Lab 1 `analyze_complaint_tokens` function as real example inputs, so the
handoff is not decorative. The narrative explicitly connects "the complaints you triaged
in Topic 1" to "the same complaints, now tokenized".

PRODUCES: `barclays-course/topic_2/complaint_corpus.json` containing the worked
complaint texts used across the notebook and the tokenizer name (`distilbert-base-uncased`)
so downstream topics know which tokenizer the corpus was prepared with.

Install-cell change: topic_2 cell 2 already pins `transformers`, `datasets`, etc. Add
`sagemaker` so the handoff cells can resolve the bucket.

OLD (cell 2, last dependency line):
```
    "scikit-learn>=1.3.0,<2.0.0"
```
NEW (cell 2, last dependency lines):
```
    "scikit-learn>=1.3.0,<2.0.0" \
    "sagemaker>=2.200.0,<3.0.0"
```

LOAD cell: insert ONE new code cell immediately AFTER cell 3 (the imports/setup cell)
and BEFORE cell 4 ("Section 1 - What Is a Token?"). Full content:
```
# Handoff from Topic 1: load the Barclays triage artifacts you produced there.
# In Topic 1 you wrote a triage system prompt and 5 test complaints, and saved them
# to S3. We load them now and extend the system: this topic tokenizes those same
# complaints and turns them into embeddings.
# The OpenAI client is NOT loaded - it is a live connection. Topic 2 does no API
# calls, so no client is needed here.

# Topics 1-3 have no sagemaker.Session(); resolve the course bucket directly.
import sagemaker
bucket = sagemaker.Session().default_bucket()

# --- S3 handoff helpers (course-wide, identical in every notebook) ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def _handoff_key(topic_n, artifact):
    return f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"

def handoff_read(bucket, topic_n, artifact):
    """Read a JSON object from a topic's handoff prefix. Returns None if absent."""
    key = _handoff_key(topic_n, artifact)
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None

_t1 = handoff_read(bucket, 1, "triage_config.json")

if _t1 is not None:
    test_complaints = _t1["test_complaints"]
    triage_system_prompt = _t1["system_prompt"]
    print(f"Loaded {len(test_complaints)} test complaints from Topic 1.")
else:
    # Fallback: student is starting at Topic 2. Define a minimal local version
    # so the rest of the notebook runs. Same spirit as a lab safety-net cell.
    print("Using a local fallback complaint set so Topic 2 runs standalone.")
    test_complaints = [
        "I sent 1200 pounds to my sister three days ago and it still has not arrived.",
        "I just got an alert for a 350 pound transaction in Manchester I did not make.",
        "I cannot log in to the app; it texts a code to my old phone number.",
        "You charged me a 25 pound overdraft fee for being four hours overdrawn.",
        "300 pounds left my account to someone I have never heard of.",
    ]
    triage_system_prompt = "You are a complaint triage agent for Barclays Bank."

print("These are the complaints you triaged in Topic 1. Now we look inside the model:")
print("we tokenize them and turn them into embeddings.")
```

Downstream wiring: cell 7 (the DistilBERT tokenization demo) and cell 9 (Lab 1) should
draw their example text from `test_complaints[0]` instead of a fresh hardcoded string,
so the handoff is genuinely used. Minimal adjustment - in cell 7, the line:
```
complaint = (
    "I've been charged twice for the same transaction on my Barclaycard. "
    "This is unacceptable and I need a refund immediately."
)
```
becomes:
```
# Use the first complaint carried over from Topic 1 (loaded in the handoff cell above).
complaint = test_complaints[0]
```
This is the only consuming change needed; the rest of cell 7 is unchanged. Cell 9's
hardcoded `short_complaint` / `medium_complaint` may stay (they exercise length
thresholds) but a one-line comment notes that `test_complaints` from Topic 1 are also
valid inputs to `analyze_complaint_tokens`.

WRITE cell: insert ONE new code cell immediately AFTER cell 39 (the Topic 2 wrap-up)
and AFTER the existing cell 40 cleanup cell would be acceptable too; place it right
after cell 39 so it runs before the optional cleanup. Full content:
```
# Handoff to Topic 3: save the Barclays complaint corpus for the HuggingFace topic.
# Topic 3 loads this corpus to run pipeline() classification on real complaint text
# instead of toy strings.

# --- S3 handoff helpers ---
import json, boto3

COURSE_PREFIX = "barclays-course"

def handoff_write(bucket, topic_n, artifact, obj):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

complaint_corpus = {
    "complaints": test_complaints,
    "tokenizer_name": "distilbert-base-uncased",
}

handoff_write(bucket, 2, "complaint_corpus.json", complaint_corpus)
print()
print("Topic 3 will load this corpus and classify it with the HuggingFace pipeline API.")
```

Narrative line topic_2 adds: "In Topic 1 you produced a triage prompt and test
complaints and saved them to S3; we load them now and look inside the model that would
process them."

---

### topic_3 - huggingface

LOADS: `barclays-course/topic_2/complaint_corpus.json` - the Barclays complaint texts
from Topic 2.

USES what it loads: the loaded complaints become the inputs to the Section 2
`pipeline()` sentiment and zero-shot classification demos and to Lab 2's NER, instead
of fresh hardcoded strings.

PRODUCES: two artifacts under `barclays-course/topic_3/`:
- `routing_labels.json` - the `COMPLAINT_LABELS` routing categories used for zero-shot
  classification.
- `labelled_dataset.json` - a small labelled complaint dataset (complaint text plus an
  integer label) and the `label_map` (id-to-name). This is the dataset Topic 4 needs to
  fine-tune on. It is built from the complaints classified during the notebook (the
  zero-shot routing output is the cheap labelling step) plus a fixed `label_map`.

Install-cell change: topic_3 cell 4 gains `sagemaker`.

OLD (cell 4, last dependency line):
```
    "numpy<2"
```
NEW (cell 4, last dependency lines):
```
    "numpy<2" \
    "sagemaker>=2.200.0,<3.0.0"
```

LOAD cell: insert ONE new code cell immediately AFTER cell 6 (the imports + seeds +
COMPLAINT_TOKENS/COMPLAINT_LABELS cell) and BEFORE cell 7 ("What are we building
today?"). Note this places the load AFTER the COMPLAINT_TOKENS definition that Edit 3.3
already corrects; that is fine - COMPLAINT_TOKENS stays local, the load adds the
complaint corpus. Full content:
```
# Handoff from Topic 2: load the Barclays complaint corpus you produced there.
# In Topic 2 you tokenized and embedded a set of complaints and saved them to S3.
# We load them now and extend the system: this topic classifies and routes them
# with pretrained HuggingFace models.

# Topics 1-3 have no sagemaker.Session(); resolve the course bucket directly.
import sagemaker
bucket = sagemaker.Session().default_bucket()

# --- S3 handoff helpers ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def handoff_read(bucket, topic_n, artifact):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None

_t2 = handoff_read(bucket, 2, "complaint_corpus.json")

if _t2 is not None:
    course_complaints = _t2["complaints"]
    print(f"Loaded {len(course_complaints)} complaints carried over from Topic 2.")
else:
    # Fallback: student is starting at Topic 3. Define a local complaint set.
    print("Using a local fallback complaint set so Topic 3 runs standalone.")
    course_complaints = [
        "I sent 1200 pounds to my sister three days ago and it still has not arrived.",
        "I just got an alert for a 350 pound transaction in Manchester I did not make.",
        "I cannot log in to the app; it texts a code to my old phone number.",
        "You charged me a 25 pound overdraft fee for being four hours overdrawn.",
        "300 pounds left my account to someone I have never heard of.",
    ]

print("These are the same Barclays complaints from Topic 2. Now we route them with")
print("pretrained HuggingFace models instead of looking at their embeddings.")
```

Downstream wiring: in cell 20 (the zero-shot classification demo) the complaint inputs
should be `course_complaints` rather than fresh hardcoded strings; capture the predicted
label per complaint into a list `routed_examples` of `{"text": ..., "label": <int>}` so
the WRITE cell has a labelled dataset to persist. This is a small additive change to
cell 20: after the existing zero-shot loop, add a few lines building `routed_examples`
from the loop's predictions mapped through `COMPLAINT_LABELS`.

WRITE cell: insert ONE new code cell immediately AFTER cell 43 (the "What is coming
next" wrap-up) and BEFORE cell 44 (the end-of-topic footer). Full content:
```
# Handoff to Topic 4: save the labelled Barclays complaint dataset and routing labels.
# Topic 4 loads this dataset to run full fine-tuning on a real, domain dataset
# instead of a toy one.

# --- S3 handoff helpers ---
import json, boto3

COURSE_PREFIX = "barclays-course"

def handoff_write(bucket, topic_n, artifact, obj):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

# Routing labels (zero-shot categories) used across the rest of the course.
handoff_write(bucket, 3, "routing_labels.json", {"labels": COMPLAINT_LABELS})

# Labelled dataset for Topic 4 fine-tuning. label_map is id -> category name.
label_map = {i: name for i, name in enumerate(COMPLAINT_LABELS)}

# routed_examples was built in the zero-shot section above. If that cell was not run,
# fall back to a minimal labelled set so the handoff still completes.
try:
    examples = routed_examples
except NameError:
    examples = [{"text": t, "label": 0} for t in course_complaints]

labelled_dataset = {
    "examples": examples,
    "label_map": label_map,
}
handoff_write(bucket, 3, "labelled_dataset.json", labelled_dataset)
print()
print("Topic 4 will load this labelled dataset and fine-tune DistilBERT on it.")
```

Narrative line topic_3 adds: "In Topic 2 you produced a complaint corpus and saved it
to S3; we load it now and route those complaints with pretrained models, then save a
labelled dataset for Topic 4 to fine-tune on."

---

### topic_4 - full_finetuning

LOADS: `barclays-course/topic_3/labelled_dataset.json` and
`barclays-course/topic_3/routing_labels.json` - the labelled complaint dataset and
routing labels Topic 3 produced.

USES what it loads: the loaded `label_map` sets `num_labels` for the model and the
loaded examples seed the Lab 1 / Section 2 fine-tuning data, so the fine-tuning runs on
the dataset the course has been building rather than an unrelated toy set.

PRODUCES: `barclays-course/topic_4/model_pointer.json` - a pointer JSON holding the
fine-tuned model's `model.tar.gz` S3 URI (`estimator.model_data`), the
`training_job_name`, and the `label_map`. A model tarball is referenced by URI, not
copied, hence a pointer.

`bucket` already exists (cell 3, SageMaker setup). No install-cell change needed
(`sagemaker` is already pinned).

LOAD cell: insert ONE new code cell immediately AFTER cell 5 (the `import numpy`/setup
cell, after the SageMaker session in cell 3) and BEFORE cell 6 ("CUDA Health Check").
Full content:
```
# Handoff from Topic 3: load the labelled Barclays complaint dataset.
# In Topic 3 you routed complaints with pretrained models and saved a labelled
# dataset to S3. We load it now and extend the system: this topic fine-tunes
# DistilBERT on that exact dataset.

# --- S3 handoff helpers ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def handoff_read(bucket, topic_n, artifact):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None

_t3 = handoff_read(bucket, 3, "labelled_dataset.json")

if _t3 is not None:
    course_examples = _t3["examples"]
    label_map = {int(k): v for k, v in _t3["label_map"].items()}
    print(f"Loaded {len(course_examples)} labelled examples and "
          f"{len(label_map)} labels from Topic 3.")
else:
    # Fallback: student is starting at Topic 4. Define a minimal labelled set.
    print("Using a local fallback labelled dataset so Topic 4 runs standalone.")
    label_map = {0: "fraud and security", 1: "billing and charges",
                 2: "account access", 3: "general enquiry"}
    course_examples = [
        {"text": "Unauthorised charge appeared on my account.", "label": 0},
        {"text": "You charged me an overdraft fee I did not expect.", "label": 1},
        {"text": "I cannot log in to the mobile app.", "label": 2},
        {"text": "What are your branch opening hours?", "label": 3},
    ]

num_labels = len(label_map)
print(f"Fine-tuning target: {num_labels}-class Barclays complaint classifier.")
```

Downstream wiring: cell 16's `LAB_TRAIN_TEXTS` dataset stays as the in-notebook lab
data, but a one-line comment notes that `course_examples` loaded above is the
course-spiral dataset; where the notebook builds a `Dataset` for fine-tuning it may
extend it with `course_examples`. `num_labels` from the handoff feeds
`AutoModelForSequenceClassification.from_pretrained(..., num_labels=num_labels)`
wherever the notebook currently hardcodes a label count.

WRITE cell: insert ONE new code cell immediately AFTER cell 44 (the existing
"variable inventory" recap cell that Edit 4.8 rewrites). The WRITE cell is the real
mechanism behind that inventory. Full content:
```
# Handoff to Topic 5: save a pointer to the fine-tuned model artifacts.
# A model.tar.gz lives in S3 already; we save its URI (a pointer), not a copy.
# Topic 5 loads this pointer to compare transfer learning against full fine-tuning.

# --- S3 handoff helpers ---
import json, boto3

COURSE_PREFIX = "barclays-course"

def handoff_write(bucket, topic_n, artifact, obj):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

# estimator.model_data and training_job_name come from the Section 4 GPU job.
# If the remote job was not run in this kernel, fall back to None so the handoff
# still records the label map for Topic 5.
try:
    finetuned_model_uri = estimator.model_data
except (NameError, Exception):
    finetuned_model_uri = None

try:
    _job = training_job_name
except NameError:
    _job = None

model_pointer = {
    "model_tar_uri": finetuned_model_uri,
    "training_job_name": _job,
    "label_map": label_map,
    "kind": "full_finetune",
}
handoff_write(bucket, 4, "model_pointer.json", model_pointer)
print()
if finetuned_model_uri is None:
    print("Note: no model URI recorded (GPU job not run in this kernel).")
    print("Topic 5 will fall back to fine-tuning fresh if the URI is missing.")
print("Topic 5 will load this pointer and compare it against transfer learning.")
```

Narrative line topic_4 adds: "In Topic 3 you produced a labelled complaint dataset and
saved it to S3; we load it now and fine-tune DistilBERT on it, then save the trained
model's S3 location for Topic 5."

---

### topic_5 - transfer_learning

LOADS: `barclays-course/topic_4/model_pointer.json` - the fine-tuned model URI and
label map from Topic 4.

USES what it loads: the loaded `label_map` configures the transfer-learning head; the
loaded full-fine-tune model URI is the baseline that Section 5's comparison
(`compare_checkpoints`) measures the transfer-learned model against - so the Topic 4 vs
Topic 5 comparison uses the real Topic 4 artifact.

PRODUCES: `barclays-course/topic_5/model_pointer.json` - a pointer JSON with the
transfer-learned model's `model.tar.gz` URI (`trained_model_data`), the
`training_job_name`, the endpoint name if one was deployed, and the `label_map`.

`bucket` already exists (cell 4). No install-cell change needed.

LOAD cell: insert ONE new code cell immediately AFTER cell 4 (the SageMaker session
setup) and BEFORE cell 5 ("CUDA Health Check"). Full content:
```
# Handoff from Topic 4: load the fine-tuned model pointer you produced there.
# In Topic 4 you fully fine-tuned DistilBERT and saved the model's S3 URI.
# We load it now and extend the system: this topic trains a transfer-learning
# variant and compares it against that full-fine-tune baseline.

# --- S3 handoff helpers ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def handoff_read(bucket, topic_n, artifact):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None

_t4 = handoff_read(bucket, 4, "model_pointer.json")

if _t4 is not None:
    finetune_model_uri = _t4.get("model_tar_uri")
    label_map = {int(k): v for k, v in _t4["label_map"].items()}
    print(f"Loaded Topic 4 full-fine-tune pointer; {len(label_map)} labels.")
    if finetune_model_uri is None:
        print("Topic 4 recorded no model URI; the Section 5 comparison will note this.")
else:
    # Fallback: student is starting at Topic 5.
    print("Using a local fallback label map so Topic 5 runs standalone.")
    finetune_model_uri = None
    label_map = {0: "negative", 1: "positive"}

num_labels = len(label_map)
print(f"Transfer-learning target: {num_labels}-class classifier.")
```

Downstream wiring: cell 44 (`compare 6a full fine-tune vs 6b transfer learning`) uses
`finetune_model_uri` as the full-fine-tune checkpoint argument to `compare_checkpoints`
instead of a hardcoded or placeholder URI; if `finetune_model_uri` is None, the cell
prints a clear note that the comparison runs transfer-learning-only.

WRITE cell: insert ONE new code cell immediately AFTER cell 46 (the "Key Takeaways"
final cell). Full content:
```
# Handoff to Topic 6: save a pointer to the transfer-learned model.
# Topic 6 loads this so its PEFT/LoRA work continues the same Barclays classifier.

# --- S3 handoff helpers ---
import json, boto3

COURSE_PREFIX = "barclays-course"

def handoff_write(bucket, topic_n, artifact, obj):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

try:
    transfer_model_uri = trained_model_data
except NameError:
    transfer_model_uri = None

try:
    _job = training_job_name
except NameError:
    _job = None

try:
    _endpoint = predictor.endpoint_name
except (NameError, AttributeError):
    _endpoint = None

model_pointer = {
    "model_tar_uri": transfer_model_uri,
    "training_job_name": _job,
    "endpoint_name": _endpoint,
    "label_map": label_map,
    "kind": "transfer_learning",
}
handoff_write(bucket, 5, "model_pointer.json", model_pointer)
print()
print("Topic 6 will load this pointer and adapt the classifier with PEFT and LoRA.")
```

Narrative line topic_5 adds: "In Topic 4 you produced a fully fine-tuned model and
saved its S3 location; we load it now as the baseline, train a transfer-learning
variant, and save that for Topic 6."

---

### topic_6 - peft_lora_distilbert

LOADS: `barclays-course/topic_5/model_pointer.json` - the transfer-learned model
pointer and label map from Topic 5.

USES what it loads: the loaded `label_map` sets `num_labels` for the PEFT model; the
loaded transfer-learned model URI is referenced as the starting checkpoint the LoRA
adapters extend, so the PEFT topic continues the same classifier rather than starting
from raw `distilbert-base-uncased` with no lineage.

PRODUCES: `barclays-course/topic_6/model_pointer.json` - a pointer JSON with the
PEFT/LoRA fine-tuned `model.tar.gz` URI (from the Section 4 GPU job, currently written
under `s3://{bucket}/topic7b-peft/output/`), the `training_job_name`, the LoRA config
(`lora_r`, `lora_alpha`), and the `label_map`.

`bucket` already exists (cell 3). No install-cell change needed.

Placement note vs the R8 mini-lesson: the R8 LoRA mini-lesson is inserted after cell 5
(see the "R8 Mini-Lesson" section). The LOAD cell goes EARLIER - immediately AFTER
cell 3 (the SageMaker session setup) and BEFORE cell 4 ("Day 2 System Overview" /
"Where This Topic Fits"). This keeps the load right after the session is available and
well before the mini-lesson, so there is no ordering conflict.

LOAD cell: insert ONE new code cell immediately AFTER cell 3 and BEFORE cell 4. Full
content:
```
# Handoff from Topic 5: load the transfer-learned model pointer you produced there.
# In Topic 5 you trained a transfer-learning classifier and saved its S3 URI.
# We load it now and extend the system: this topic adapts the same classifier with
# parameter-efficient fine-tuning (PEFT and LoRA).

# --- S3 handoff helpers ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def handoff_read(bucket, topic_n, artifact):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None

_t5 = handoff_read(bucket, 5, "model_pointer.json")

if _t5 is not None:
    prior_model_uri = _t5.get("model_tar_uri")
    label_map = {int(k): v for k, v in _t5["label_map"].items()}
    print(f"Loaded Topic 5 transfer-learning pointer; {len(label_map)} labels.")
else:
    # Fallback: student is starting at Topic 6.
    print("Using a local fallback label map so Topic 6 runs standalone.")
    prior_model_uri = None
    label_map = {0: "negative", 1: "positive"}

num_labels = len(label_map)
print(f"PEFT/LoRA target: {num_labels}-class Barclays complaint classifier.")
```

Downstream wiring: wherever the notebook builds the base
`AutoModelForSequenceClassification` it uses `num_labels=num_labels` from the handoff;
the Section 4 launch cell comment notes that `prior_model_uri` is the lineage of the
classifier being adapted.

WRITE cell: insert ONE new code cell immediately AFTER cell 46 (the cell that reads the
metrics summary and surfaces the training-job status), so the pointer is written once
the GPU job name is known. Full content:
```
# Handoff to Topic 7: save a pointer to the PEFT/LoRA fine-tuned model.
# Topic 7 loads this checkpoint and compresses it (quantization, pruning, distillation).

# --- S3 handoff helpers ---
import json, boto3

COURSE_PREFIX = "barclays-course"

def handoff_write(bucket, topic_n, artifact, obj):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

# The PEFT GPU job writes model.tar.gz under s3://<bucket>/topic7b-peft/output/.
try:
    peft_model_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
except (NameError, KeyError, TypeError):
    peft_model_uri = None

try:
    _job = training_job_name
except NameError:
    _job = None

model_pointer = {
    "model_tar_uri": peft_model_uri,
    "training_job_name": _job,
    "lora_config": {"lora_r": 8, "lora_alpha": 16},
    "label_map": label_map,
    "kind": "peft_lora",
}
handoff_write(bucket, 6, "model_pointer.json", model_pointer)
print()
if peft_model_uri is None:
    print("Note: no PEFT model URI recorded (GPU job not run in this kernel).")
    print("Topic 7 will fall back to a fresh DistilBERT if the URI is missing.")
print("Topic 7 will load this pointer and compress the model for production serving.")
```

Narrative line topic_6 adds: "In Topic 5 you produced a transfer-learned model and
saved its S3 location; we load it now and adapt it with PEFT and LoRA, then save the
PEFT checkpoint for Topic 7."

---

### topic_7 - quantization

LOADS: `barclays-course/topic_6/model_pointer.json` - the PEFT/LoRA checkpoint pointer
and label map from Topic 6.

USES what it loads: the loaded `label_map` sets `num_labels` for the model loaded in
cell 7; the loaded PEFT checkpoint URI is the model the compression pipeline
(quantization, pruning, distillation) operates on. Topic 7 cell 0 and cell 7 already
say "Each section loads the checkpoint fresh from S3"; this handoff makes that literal
by giving cell 7 the actual Topic 6 URI. The existing self-containment fallback in
cell 7 (re-load `distilbert-base-uncased`) becomes the LOAD cell's fallback branch.

PRODUCES: `barclays-course/topic_7/deployment.json` - the final quantized endpoint name
and a compression summary (size before/after, latency before/after). Topic 7 is the
last required topic, so nothing reads this yet; the planned topic_8 agent capstone will
consume it. Writing it completes the chain and keeps the spiral honest.

`bucket` already exists (cell 3). No install-cell change needed.

LOAD cell: insert ONE new code cell immediately AFTER cell 5 (the CUDA health-check
cell) and BEFORE cell 6 (the "Each section loads checkpoints from S3" note). Full
content:
```
# Handoff from Topic 6: load the PEFT/LoRA checkpoint pointer you produced there.
# In Topic 6 you adapted the Barclays classifier with PEFT and LoRA and saved the
# checkpoint's S3 URI. We load it now and extend the system: this topic compresses
# that model for cheap production serving.

# --- S3 handoff helpers ---
import json, boto3, botocore

COURSE_PREFIX = "barclays-course"

def handoff_read(bucket, topic_n, artifact):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Handoff loaded: s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"No handoff found at s3://{bucket}/{key} (starting mid-course is fine).")
        return None

_t6 = handoff_read(bucket, 6, "model_pointer.json")

if _t6 is not None:
    peft_checkpoint_uri = _t6.get("model_tar_uri")
    label_map = {int(k): v for k, v in _t6["label_map"].items()}
    print(f"Loaded Topic 6 PEFT checkpoint pointer; {len(label_map)} labels.")
    if peft_checkpoint_uri is None:
        print("Topic 6 recorded no checkpoint URI; compressing a fresh model instead.")
else:
    # Fallback: student is starting at Topic 7. Compress a fresh classifier.
    print("Using a fresh DistilBERT classifier so Topic 7 runs standalone.")
    peft_checkpoint_uri = None
    label_map = {0: "fraud and security", 1: "billing and charges",
                 2: "account access", 3: "general enquiry", 4: "unclassified"}

num_labels = len(label_map)
print(f"Compression target: a {num_labels}-class Barclays complaint classifier.")
```

Downstream wiring: cell 7 currently hardcodes `num_labels=5`; it uses `num_labels` from
the handoff instead. Cell 7's comment is updated to note that when `peft_checkpoint_uri`
is set the production pipeline would load that tarball; the fresh-load path remains as
the documented fallback (consistent with Edit 7.4).

WRITE cell: insert ONE new code cell immediately AFTER cell 63 (the "Course Complete"
cell). Full content:
```
# Handoff to the future agent capstone (topic_8): save the final deployment record.
# Topic 7 is the last required topic. Writing this completes the course S3 chain;
# the planned agent capstone will load it as the deployed Barclays classifier.

# --- S3 handoff helpers ---
import json, boto3

COURSE_PREFIX = "barclays-course"

def handoff_write(bucket, topic_n, artifact, obj):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(obj, indent=2).encode("utf-8"),
    )
    print(f"Handoff written: s3://{bucket}/{key}")
    return key

# endpoint_name is set in Section 5 when the quantized model is deployed.
try:
    _endpoint = endpoint_name
except NameError:
    _endpoint = None

deployment = {
    "endpoint_name": _endpoint,
    "label_map": label_map,
    "compression": "PTQ + pruning + distillation (see Section 5 results)",
    "kind": "quantized_endpoint",
}
handoff_write(bucket, 7, "deployment.json", deployment)
print()
print("Course S3 handoff chain complete: topic_1 -> topic_7.")
print("The Barclays Complaint Intelligence System is one artifact lineage in S3.")
```

Narrative line topic_7 adds: "In Topic 6 you produced a PEFT-adapted model and saved
its S3 location; we load it now and compress it for production, then record the final
deployment so the whole course is one artifact lineage in S3."

---

### Handoff chain summary table

| Topic | LOAD (from prev) | PRODUCE (own prefix) | LOAD cell after | WRITE cell after |
|-------|------------------|----------------------|-----------------|------------------|
| topic_1 | nothing (first) | `topic_1/triage_config.json` | n/a | cell 34 |
| topic_2 | `topic_1/triage_config.json` | `topic_2/complaint_corpus.json` | cell 3 | cell 39 |
| topic_3 | `topic_2/complaint_corpus.json` | `topic_3/labelled_dataset.json`, `topic_3/routing_labels.json` | cell 6 | cell 43 |
| topic_4 | `topic_3/labelled_dataset.json` + `routing_labels.json` | `topic_4/model_pointer.json` | cell 5 | cell 44 |
| topic_5 | `topic_4/model_pointer.json` | `topic_5/model_pointer.json` | cell 4 | cell 46 |
| topic_6 | `topic_5/model_pointer.json` | `topic_6/model_pointer.json` | cell 3 | cell 46 |
| topic_7 | `topic_6/model_pointer.json` | `topic_7/deployment.json` | cell 5 | cell 63 |

New cells added for the handoff chain: 13 total (topic_1 has WRITE only = 1; topics 2-7
each have LOAD + WRITE = 12). Every cell is fully-worked infrastructure code, identical
in the Exercises and Solutions copies. The LOAD cells carry their own fallback (the
safety-net spirit). No `# YOUR CODE` blanks, no separate safety-net cell needed.

---

## Summary of edits

| Notebook | Continuity edits | New mini-lesson cells | S3 handoff cells | Notes |
|----------|------------------|-----------------------|------------------|-------|
| topic_1_overview_genai | 2 | 0 | 1 (WRITE only) | wrap-up forward references; banner -> S3 write |
| topic_2_introducing_llms | 7 | 5 (R7 transformer mini-lesson) | 2 (LOAD + WRITE) | roadmap reframes + new Section 5.5 |
| topic_3_huggingface | 7 | 0 | 2 (LOAD + WRITE) | false "Topic 4 transformers" prereq, COMPLAINT_TOKENS |
| topic_4_full_finetuning | 8 (4.7 is a no-op) | 0 | 2 (LOAD + WRITE) | system table + LoRA/PEFT renumbering; R10 verified clean |
| topic_5_transfer_learning | 5 | 0 | 2 (LOAD + WRITE) | system table + LoRA handoff reframe |
| topic_6_peft_lora_distilbert | 10 (6.7 is a no-op) | 3 (R8 LoRA mini-lesson) | 2 (LOAD + WRITE) | "you built LoRA in 7a" reframing |
| topic_7_quantization | 8 (7.2 is a no-op) | 0 | 2 (LOAD + WRITE) | T7b->Topic 6, Flan-T5 fix, T8/T9 cleanup, recap table |

Notebooks needing changes: 7 of 7.
Continuity edits specified: 47 entries; 3 (4.7, 6.7, 7.2) are explicit "no change"
confirmations, leaving 44 actual replacement edits.
New mini-lesson cells inserted: 8 total (5 in topic_2 for the R7 mini-lesson, 3 in
topic_6 for the R8 mini-lesson).
New S3 handoff cells inserted: 13 total (topic_1 WRITE only; topics 2-7 each LOAD +
WRITE). See the "S3 Handoff Chain" section for full per-cell content.
Install-cell adjustments: 3 (topics 1, 2, 3 add `"sagemaker>=2.200.0,<3.0.0"` so the
handoff cells can resolve the course bucket; topics 4-7 already pin it).
Diagram files changed: 0 (R11 audit found no stale `.mmd` captions).

Every edit, every new mini-lesson cell, and every new S3 handoff cell applies to BOTH
the `Exercises/` copy and the `Solutions/` copy of the named notebook. The mini-lesson
cells and the handoff cells are fully-worked code, identical in both copies (no
`# YOUR CODE` blanks; the LOAD cells carry their own fallback, so no separate
safety-net cell is needed).

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

---

## Codex R3 findings resolved

Round 3 checked teaching-narrative continuity. The three findings and the S3 handoff
DECISION are resolved as follows.

| Finding | Resolution location in this doc |
|---------|---------------------------------|
| C1 (Topic 1 -> Topic 2 breaks the spiral: the "Variables available for Topic 2" banner lists `client` / `my_system_prompt` / `test_complaints`, but Topic 2 references none of them) | "S3 Handoff Chain" section, subsections "topic_1" and "topic_2". The topic_1 cell-34 banner is rewritten to drop the false kernel-handoff claim, and a new WRITE cell persists the DURABLE artifacts (`my_system_prompt`, `test_complaints`, routing categories) to `barclays-course/topic_1/triage_config.json`. topic_2 gains a LOAD cell that reads them back AND a downstream-wiring change (cell 7 uses `test_complaints[0]` as its tokenization input) so the artifacts are genuinely used. The OpenAI `client` is handled honestly: it is a live connection, cannot be serialised, and is not carried; it is recreated from the API key with `getpass` in any notebook that needs it (see "Honest note on non-serialisable objects (C1)"). |
| C2 (the whole required path restarts state per notebook instead of extending one running Barclays system) | The entire "S3 Handoff Chain" section. Every required notebook ends with a WRITE cell and (topics 2-7) starts with a LOAD cell, chained topic_1 -> topic_7 through `s3://<bucket>/barclays-course/topic_<N>/`. Each LOAD cell genuinely feeds downstream cells (corpus -> tokenization, labelled dataset -> fine-tuning, model pointers -> comparison and adaptation), so the course is one artifact lineage in S3, robust to kernel restarts. The "Handoff chain summary table" lists every link. |
| C3 (mini-lesson placement risk: orphaned references to optional topics must be removed and the mini-lessons must land BEFORE the first such reference) | Verified below. All orphaned references are scheduled for removal by existing edits, and both mini-lessons land before the first reference they replace. |

### C3 verification - orphaned references and mini-lesson ordering

Codex C3 named two orphaned-reference sites. Both are covered by existing edits in this
doc, and the cell indices were re-checked against the notebooks at commit 13187c0:

1. **topic_3 cell ~14, "In Topic 4 you assembled a Transformer".** The actual location
   is topic_3 **cell 0**, the "What you will build today" opening paragraph, whose
   first sentence reads "In Topic 4 you assembled a Transformer encoder-decoder from
   scratch...". This is removed by **Edit 3.1**, which rewrites cell 0 to "In Topic 2
   you learned what a transformer is..." and points the from-scratch build at the
   optional notebook. Codex's "cell ~14" was approximate; the grep target string lives
   in cell 0. Edit 3.1 fires on the exact OLD string, so it is correctly scheduled.
   Ordering: the R7 transformer mini-lesson is inserted in **topic_2 after cell 36**,
   which is an EARLIER notebook than topic_3. Any reference in topic_3 therefore comes
   after the mini-lesson the student has already seen. No mis-ordering.

2. **topic_6 cells 13 / 148 / 263 / 296 / 346, "recall LoRA from Topic 7a".** The
   topic_6 notebook at commit 13187c0 has 55 cells (0-54), so the literal indices
   148/263/296/346 do not exist; Codex was working from a stale or pre-split cell
   numbering. The ACTUAL "Topic 7a" / "7a" orphaned references in topic_6 are in cells
   **0, 5, 8, 9, 11** and the capstone recap cells **52, 54**. Every one of them is
   already scheduled for removal:
   - cell 0 -> **Edit 6.1** (rewrites the "In Topic 7a you built LoRA from scratch"
     opening).
   - cell 5 -> **Edit 6.3** (replaces "Recall from Topic 7a... carried forward from 7a").
   - cell 8 -> **Edit 6.4** (replaces "In Topic 7a you implemented LoRA by hand").
   - cell 9 -> **Edit 6.5** (replaces the `NaiveLoraLinear` docstring "from Topic 7a"
     and the "This is what we did in 7a" comment).
   - cell 11 -> **Edit 6.6** (replaces "the entire manual injection from Topic 7a" and
     "rank from Topic 7a").
   - cell 52 -> **Edit 6.8** (rewrites the capstone "Situation" paragraph listing
     "LoRA from scratch (T7a)").
   - cell 54 -> **Edit 6.9** (rebuilds the "Day 2 Complete" recap table, removing the
     "T7a LoRA from scratch" row).
   No orphaned "7a" reference in topic_6 is left unscheduled. Codex's specific cell
   indices were stale, but the editing agent must match on the quoted OLD strings (as
   every Edit 6.x instructs), so the wrong indices cause no miss.
   Ordering: the R8 LoRA mini-lesson is inserted in **topic_6 after cell 5**. The first
   orphaned reference that the mini-lesson is meant to replace conceptually is in cell 8
   ("From the mini-lesson above, LoRA looks like this", per Edit 6.4) - which is AFTER
   cell 5, so the mini-lesson lands before it. The cell-0 opening (Edit 6.1) and the
   cell-5 setup (Edit 6.3) come BEFORE the mini-lesson, but both are rewritten to
   ANNOUNCE the mini-lesson ("a short mini-lesson below explains the mechanics" /
   "the mini-lesson below explains rank, alpha, and dropout") rather than to teach or
   assume the content - so they are forward references to the mini-lesson, not orphaned
   backward references. This is correct: a student reads cell 0, then cell 5, then the
   mini-lesson (cells A-C after cell 5), then cell 8 which builds on it. No teaching
   content is needed before the mini-lesson; every cell before it only points forward.

C3 ordering rule satisfied: the R7 mini-lesson is in topic_2 (before all of topic_3-7),
and the R8 mini-lesson is in topic_6 after cell 5, before the first cell (cell 8) that
relies on its content. No edit is missing and no edit is mis-ordered. No additional
edits are required for C3 beyond those already specified as Edits 3.1 and 6.1-6.9.

### S3 handoff DECISION resolved

The Round 3 DECISION ("cross-notebook handoff is done via S3, not kernel variables")
is implemented in full by the "S3 Handoff Chain" section: the course-wide key layout
(`s3://<bucket>/barclays-course/topic_<N>/<artifact>`), the per-topic LOADS and
PRODUCES, the exact LOAD and WRITE cell content with built-in fallbacks, and the
per-notebook narrative line. topic_1 has a WRITE cell only (no predecessor). The
"S3 Handoff Chain" section is self-contained and implementable by a notebook-build
agent without re-deriving anything.
