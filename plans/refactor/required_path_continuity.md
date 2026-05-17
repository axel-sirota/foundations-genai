# Required-Path Continuity Refactor - Design Doc

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
remains, and what this doc specifies, is the **narrative continuity** rework: the
required notebooks still contain passages that assume the old linear path went
attention -> transformers -> LoRA-from-scratch as required topics. Those passages must
be reframed so that:

1. Attention internals, transformer-from-scratch, and LoRA-from-scratch are described
   as OPTIONAL deep-dives, never as "you did this in Topic N".
2. Any data/variable carryover that supposedly came from a now-optional notebook is
   made self-contained (defined locally).
3. The narrative tightens toward APPLICATION: the required path is for USERS of LLMs.

## Conventions for the editing agents

- **Every change in this doc applies to BOTH copies of the notebook**: the
  `Exercises/<topic>/<topic>.ipynb` file AND the matching
  `Solutions/<topic>/<topic>.ipynb` file. The cells listed here are markdown or
  narrative code-comment cells that are identical in both copies. Apply the same OLD
  -> NEW replacement in both files.
- Cell indices below are 0-based and were read from the `Exercises/` copy. The
  `Solutions/` copy has the same cell ordering for these narrative cells; if an index
  is off by one in a Solutions file, match on the quoted OLD text instead.
- Plain ASCII only. No em-dashes, en-dashes, Unicode multiplication signs, emojis.
- Preserve the Barclays customer-support through-line and the four-beat arc.
- Replace ONLY the quoted OLD text. Do not reflow or re-wrap surrounding lines unless
  the new text changes line length enough to need it; keep the surrounding markdown
  structure intact.

## Canonical reframing language (reuse across notebooks)

When a passage says "you built/assembled/trained X from scratch in Topic N", and X is
attention, the transformer architecture, a translator, or LoRA-from-scratch, use this
pattern:

> Transformers are the architecture behind every model you use in this course. If you
> want to see one built from scratch - attention, positional encoding, the decoder
> loop - there is an optional deep-dive notebook (topic_optional_transformers). It is
> not required for what follows here.

Analogous one-liners:

- Attention internals: "How self-attention works internally is covered in the optional
  attention deep-dive notebooks (topic_optional_attention_python and
  topic_optional_attention_pytorch). It is not required for the path you are on."
- LoRA from scratch: "If you want to see LoRA implemented by hand on a feed-forward
  network, there is an optional deep-dive notebook (topic_optional_lora_ffn). The
  required path uses the PEFT library directly."

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
- How tokenization and embeddings turn text into something a model can use
- How an LLM runs inference, and how the GenAI project lifecycle differs from classical ML
```

Rationale: Topic 2 (introducing_llms) does not teach attention math or training-loop
internals; that is the optional attention deep-dive. The old bullet list promised
content that now lives off the required path. Reframe to what Topic 2 actually
delivers.

No other cells in Topic 1 need continuity changes. The autoregressive-loop and GAN
demos are self-contained within Topic 1 and do not depend on optional notebooks.

---

## Topic 2 - introducing_llms

File(s): `Exercises/topic_2_introducing_llms/topic_2_introducing_llms.ipynb` and the
Solutions twin.

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
### The Self-Attention Mechanism
We said "N transformer blocks" and skipped straight to the output.
The inside of those blocks -- queries, keys, values, attention weights -- is the
mathematical heart of the transformer. If you want to see it implemented from scratch,
in pure Python and then in PyTorch, there are two optional deep-dive notebooks
(topic_optional_attention_python and topic_optional_attention_pytorch). They are not
required for the path you are on.
```

Rationale: Attention internals moved to the optional track. The required Topic 3 is now
HuggingFace, not attention.

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
### Positional Encoding (the Details)
We mentioned it briefly in the pipeline. The math behind sinusoidal and learned
positional encodings is covered in the optional transformers deep-dive notebook
(topic_optional_transformers).
```

Rationale: The transformer-internals topic is now optional, not required Topic 4.

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
Topic 2 (done): What is an LLM? How does it tokenize and embed text?
Topic 3 (next): The HuggingFace ecosystem. Load and run pretrained models.
Topic 4:        Full fine-tuning on Barclays complaint data, and catastrophic forgetting.
Topic 5-6:      Transfer learning, then parameter-efficient fine-tuning with PEFT and LoRA.
Topic 7:        Compress and deploy the model: quantization, pruning, distillation.
Optional:       Build attention and the transformer from scratch (deep-dive notebooks).
```

Rationale: The roadmap listed the old linear path. Replace it with the new required
sequence and call out the optional deep-dives as a separate line.

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
pretrained models and run them in a few lines of code. If you want to open the black
box and implement the attention mechanism yourself, the optional attention deep-dive
notebooks are there for you, but they are not required for what comes next.
```

Rationale: Topic 3 is now HuggingFace. The "next we implement attention" handoff is
stale; reframe attention as optional.

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
Optional but recommended for consistency; safe to apply.

Note: cell 0 ("In Topic 1 we decided...") and cell 32 / cell 44 (GenAI lifecycle,
"train from scratch" as a general ML concept) need NO change. "Train from scratch"
there refers to classical ML training, not to building a transformer, and is correct
as written.

---

## Topic 3 - huggingface

File(s): `Exercises/topic_3_huggingface/topic_3_huggingface.ipynb` and the Solutions
twin. This is the most affected notebook.

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
Transformers are the architecture behind every model you will use in this course. If you
want to see one built from scratch - positional encoding, multi-head attention, the
decoder loop - there is an optional deep-dive notebook (topic_optional_transformers).
It is not required for what follows here.

This topic is about USING those models, not building them. HuggingFace gives you
pre-trained models trained on billions of sentences, curated datasets, and a four-line
inference API. By the end of this topic you will classify Barclays complaint sentiment,
route complaints to the correct team using zero-shot classification, extract named
entities, and understand how to share a checkpoint to the Hub.
```

Rationale: The opening assumed the student had just done the old transformers topic as
required Topic 4. The new Topic 4 is full_finetuning. Reframe the transformer build as
an optional deep-dive and pivot the framing to application.

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
new required sequence (Topics 3-7). The "Day 2" framing no longer matches; generalize
it.

### Edit 3.3 - cell 6, COMPLAINT_TOKENS comment (self-containment)

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

Rationale: The "carried over from Topic 4" claim was false even before the renumber
(it referred to the old transformers topic). The list is already defined inline in
this cell, so the only change needed is the comment: state that it is defined locally.
COMPLAINT_TOKENS is only printed in this cell and not consumed downstream, so no
further work is needed.

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
transformers topic and is now optional. Keep the from-scratch-vs-Hub contrast (it
motivates the whole notebook) but attribute the build to the optional deep-dive
instead of a required prior topic.

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
fine-tuning (Topic 4), so this is consistent; naming it removes ambiguity. Low-stakes.

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

### Edit 4.2 - cell 5, COMPLAINT_TEXTS comment (self-containment)

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

Rationale: The list is fully defined inline in this cell; it does not actually depend
on Topic 3 having run. Make the self-containment explicit so the notebook is robust to
being run on its own.

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
transformer layers. Topic 6 uses the production-grade HuggingFace PEFT library to do
exactly that. If you want to see LoRA implemented by hand on a feed-forward network
first, there is an optional deep-dive notebook (topic_optional_lora_ffn).
```

Rationale: The next required topic is Topic 6 (PEFT library on DistilBERT), not the
old "7a LoRA from scratch". LoRA-from-scratch is now the optional lora_ffn notebook.
Reframe the handoff to Topic 6 and mention the optional build as a side path.

Note: cell 8 ("Topic 4 showed full fine-tuning") is CORRECT under the new numbering -
full fine-tuning is Topic 4. No change. cells 12 and 33 ("learn from scratch", "training
from scratch is too slow") refer to training a model from random initialization as a
general concept, not to building a transformer; no change.

---

## Topic 6 - peft_lora_distilbert

File(s): `Exercises/topic_6_peft_lora_distilbert/topic_6_peft_lora_distilbert.ipynb`
and the Solutions twin. This notebook leans heavily on "you built LoRA from scratch in
7a" as a prerequisite; that prerequisite is now an OPTIONAL notebook, so the framing
must change from "recall what you did" to "here is the idea, optionally see it built".

### Edit 6.1 - cell 0, "What you will build" opening

OLD:
```
In Topic 7a you built LoRA from scratch on feed-forward networks.
Now you use the production-grade HuggingFace PEFT library to apply LoRA to a full
DistilBERT complaint classifier, the same pattern used in real ML pipelines at scale.
```

NEW:
```
LoRA (Low-Rank Adaptation) freezes a model's pretrained weights and trains a small pair
of low-rank matrices instead. If you want to see that idea built by hand on a
feed-forward network, there is an optional deep-dive notebook (topic_optional_lora_ffn);
it is not required for this topic.

Here you use the production-grade HuggingFace PEFT library to apply LoRA to a full
DistilBERT complaint classifier, the same pattern used in real ML pipelines at scale.
```

Rationale: The notebook opened by asserting the student had done the old required "7a".
Reframe: introduce LoRA self-containedly, point at the optional build, then go straight
to the application (the PEFT library).

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
# The optional deep-dive notebook topic_optional_lora_ffn builds these by hand.
lora_r = 8   # rank: a small r means very few trainable parameters
print(f"LoRA rank: {lora_r}")
```

Rationale: The cell assumed the student carried `lora_r` and the A/B intuition from a
required prior topic. Define the concept and the value locally; point at the optional
notebook for the hand-built version.

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


Implemented by hand, LoRA looks like this:

  output = W_frozen @ x + (B @ A) @ x

W_frozen stays fixed; only the small matrices A and B are trained. The optional
deep-dive notebook topic_optional_lora_ffn builds exactly this by hand on a
feed-forward network. Here we go straight to the production approach: the PEFT
library automates that injection for any HuggingFace model. Three function calls
replace two custom classes and a manual layer-replacement loop.

### Beat 1: What happens if we try to apply LoRA without PEFT?
```

Rationale: Same reframe - present the LoRA formula self-containedly rather than as
"recall from Topic 7a", and point at the optional notebook.

### Edit 6.5 - cell 9, code docstring

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
    r=lora_r,                      # rank r=8
```

Rationale: Drop the "Topic 7a" attribution; keep the technical content.

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
LoRA-from-scratch (old T7a) as required deliverables and used the old numbering.
Replace with the actual required Topics 3-6. "Day 2 Complete" no longer maps cleanly;
generalize the heading.

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

Rationale: Topic 7 is correct; only the "Day 3" framing is stale (Day boundaries
shifted with the renumber). Drop the day labels. The reference to "the GPT-4o API call
from Topic 1" is correct and stays.

---

## Topic 7 - quantization

File(s): `Exercises/topic_7_quantization/topic_7_quantization.ipynb` and the Solutions
twin. Topic 7 mostly references "Topic 6 / T7b" as its upstream; T7b is the old name
for what is now Topic 6.

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
| Foundations | Topic 1-2 | GPT-4o prototype, then the LLM taxonomy and tokenization |
| Training | Topic 3-6 | HuggingFace ecosystem, fine-tuning, transfer learning, PEFT/LoRA |
| **Compression** | **Topic 7 (this notebook)** | **Quantization + pruning + distillation --> deployable endpoint** |

### YOU ARE HERE: Topic 7

The fine-tuned, LoRA-adapted DistilBERT you shipped in Topic 6 is accurate but heavy.
Topic 7 teaches you to make it fast and cheap without sacrificing quality:
```

Rationale: The Day 1/2/3 table used the old T1-T7b numbering and listed
"attention --> full Transformer from scratch" as required Day 1 content. Replace with
the new required arc (Topics 1-7). "FINAL topic of the course" is no longer strictly
true since topic_8 (agent capstone) is planned; soften to "final required topic before
the capstone". "T7b" becomes "Topic 6".

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
| Understanding | Topic 2 | LLM taxonomy: BERT vs GPT vs T5 families | Done |
| Ecosystem | Topic 3 | HuggingFace Hub, tokenizers, pipeline API | Done |
| Fine-Tuning | Topic 4 | Full fine-tune DistilBERT; measure catastrophic forgetting | Done |
| Transfer | Topic 5 | Transfer learning with DistilBERT on SST-2 | Done |
| PEFT | Topic 6 | PEFT library end-to-end: LoRA and QLoRA on DistilBERT | Done |
| **Compression** | **Topic 7** | **PTQ + QAT + pruning + distillation + endpoint** | **Done -- YOU ARE HERE** |
```

Rationale: The recap table listed attention-from-scratch (T3a/T3b) and
Transformer-from-scratch (T4) as required Day-1 deliverables - both are now optional.
It also used old T6a/T6b/T7a/T7b numbering and claimed full fine-tuning was on Flan-T5
(Topic 4 fine-tunes DistilBERT). Replace with the actual required Topics 1-7. Drop the
"Done in Day N" column detail to "Done" since day boundaries shifted.

### Edit 7.7 - cell 63, "What Comes Next" RLHF bullet

OLD:
```
- Explore RLHF (Topic 9, time-permitting) to align your model with human feedback
```

NEW:
```
- Explore RLHF to align your model with human feedback (an optional advanced direction)
```

Rationale: There is no Topic 9 in the new structure (the planned next topic is topic_8,
an agent capstone). Drop the dangling topic number; keep RLHF as a suggested direction.

Note: cell 26 ("trained from scratch to the same accuracy") is the Lottery Ticket
Hypothesis definition - a general ML statement, correct as written, no change. cell 39
("LoRA: fine-tune efficiently (from Topic 6)") is correct under the new numbering, no
change. cell 54 ("Replacing the Topic 1 API Call", "Topic 1 prototype") is correct, no
change.

---

## Summary of edits

| Notebook | Distinct edits | Notes |
|----------|----------------|-------|
| topic_1_overview_genai | 2 | wrap-up forward references |
| topic_2_introducing_llms | 7 | roadmap + attention/transformer reframes |
| topic_3_huggingface | 7 | most affected: false "Topic 4 transformers" prereq, COMPLAINT_TOKENS |
| topic_4_full_finetuning | 7 (4.7 is a no-op) | system table + LoRA/PEFT renumbering |
| topic_5_transfer_learning | 5 | system table + LoRA handoff reframe |
| topic_6_peft_lora_distilbert | 9 (6.7 is a no-op) | heavy "you built LoRA in 7a" reframing |
| topic_7_quantization | 7 (7.2 is a no-op) | T7b->Topic 6, day-label and recap-table cleanup |

Notebooks needing changes: 7 of 7.
Total distinct edits specified: 44 entries; 3 of those (4.7, 6.7, 7.2) are explicit
"no change, leave as is" confirmations, leaving 41 actual edits to apply.

Each edit applies to BOTH the `Exercises/` copy and the `Solutions/` copy of the named
notebook.
