# Codex (o3) Adversarial Findings - Round 1

These are blocking defects found in the first draft of the Phase 3 design docs.
The redo MUST resolve every one. Resolution principle chosen by Axel: OPTION B -
required notebooks get genuine MINI-LESSONS (not paragraph hand-waves); optional
notebooks must be genuinely runnable cold from a fresh kernel.

## Hard runtime breaks (a cold run errors out)

R1. optional_attention_pytorch calls `scaled_dot_product_attention()` but that
    function is only DEFINED in the python attention notebook. Opening the
    PyTorch notebook first -> NameError. FIX: the PyTorch notebook must DEFINE
    the function itself (a real code cell), not just restate the math in prose.

R2. optional_transformers tells students to compare against `nn.MultiheadAttention`
    "you saw in Topic 5" - Topic 5 is now transfer_learning and never shows it.
    FIX: the notebook must itself introduce `nn.MultiheadAttention` with a real
    runnable example before asking for the comparison.

R3. Both attention optionals re-run `nltk.download()` for `word2vec_sample`.
    Fails on offline classroom images. FIX: wrap the download in a try/except
    with a clear message AND a self-contained fallback (define the few embedding
    vectors inline) so the first demo runs offline.

R4. optional_transformers capstone assumes AWS creds / default bucket configured
    once per course. A standalone learner hits NoCredentialsError on cell 1 of
    the capstone. FIX: add an explicit "this capstone needs AWS / SageMaker
    credentials" prerequisite note + a guard cell that checks creds and explains
    what to do if absent.

## Variable / artifact leakage

R5. `found_tokens` (produced by a helper) is used in later demo cells of
    optional_attention_python before definition if cells run out of order; the
    old linear flow masked this via shared kernel state from Topic 2. FIX:
    ensure every variable a cell uses is defined earlier IN THIS NOTEBOOK.

R6. Optional notebooks save artifacts (attention_weights.npy,
    translator_checkpoint.pt) to cwd; required notebooks later `load()` them
    assuming they exist. FIX: required notebooks must NOT load artifacts produced
    by optional notebooks - regenerate or define what they need locally.

## Pedagogical incoherence (Option B: real mini-lessons)

R7. Required path fine-tunes DistilBERT (a 6-layer transformer) without ever
    explaining what a transformer is - no Q/K/V, no positional encoding, no
    multi-head attention. Required notebooks still discuss CLS tokens and
    attention-head visualizations. FIX (Option B): add a genuine, no-derivation
    MINI-LESSON on transformer concepts to the required path (in topic_2 and/or
    topic_3) - enough that a user can interpret attention-head viz and CLS
    tokens. Concept-level, not from-scratch math.

R8. topic_6 PEFT/LoRA hand-waves LoRA mechanics ("see optional deep-dive").
    FIX (Option B): topic_6 must contain a real mini-lesson on LoRA mechanics -
    what low-rank decomposition is, what rank/alpha/dropout do - so a student
    can reason about the knobs they tune. The optional lora_ffn notebook remains
    the from-scratch build; topic_6 no longer depends on it.

## Continuity-doc misses (the doc scheduled 0 edits for these)

R9. topic_7_quantization still has an end-of-course table listing T3a/T3b/T4/T6a
    as "Done in Day 1/2" and claims Flan-T5 was fine-tuned (never happens in the
    required path). Also congratulates finishing "T8" and suggests "Topic 9
    RLHF". FIX: the continuity doc MUST include topic_7 edits - rebuild that
    table to the new required sequence, drop Flan-T5/T9 claims.

R10. topic_4_full_finetuning keeps its opening recap "assembled a Transformer
     encoder-decoder from scratch in Topic 4" (self-reference bug) and "added
     attention ON TOP of an RNN in Topics 3a/3b". The continuity doc did not
     list these for rewrite. FIX: add these to the continuity doc.

R11. Mermaid diagram captions still embed "YOU ARE HERE -> Topic 4" etc. The
     refactor only touched markdown text. FIX: the continuity doc must enumerate
     every diagram file with a stale caption and specify the new caption (the
     diagram files live under plans/<topic>/diagrams/*.mmd).

## Narrative contradictions to resolve

R12. Continuity doc commits to BOTH "Transformers are the architecture behind
     every model you use" AND "you can complete the course without this
     notebook". Under Option B this is resolved by R7 (the required mini-lesson
     teaches the concept; the OPTIONAL notebook is only the from-scratch BUILD).
     Wording across all docs must reflect: concept = required mini-lesson,
     from-scratch build = optional.

R13. optional_transformers calls its GPU job "the first remote training job in
     the course" - false, required topic_4 already launches one. FIX: drop
     "first in the course" framing; describe it neutrally.
