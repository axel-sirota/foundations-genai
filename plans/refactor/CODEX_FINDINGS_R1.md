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

---

# Codex (o3) Round 2 - verification + new findings

Round 2 verified all 13 R1 findings RESOLVED. It raised 5 new issues (N1-N5).
Resolution (Axel: fix N1-N4 directly in the docs, ignore N5):

- N1 [FIXED] optional_attention_python.md: `seaborn` import now guarded with
  try/except + `_HAS_SEABORN`; the heatmap helper falls back to matplotlib
  `imshow` when seaborn is absent. Offline guarantee restored.
- N2 [FIXED] optional_transformers.md: the AWS guard cell now wraps
  `import boto3` / botocore itself in try/except (`_HAS_BOTO3`); credential
  and bucket probes are skipped when the SDK is absent. No ModuleNotFoundError.
- N3 [FIXED] required_path_continuity.md: the R7 transformer mini-lesson gains
  a sixth cell - a runnable DistilBertConfig check (new cell E) between the
  multi-head/encoder markdown and the recap - so the longest markdown run is 2,
  not 4. Cell count and the (previously wrong) self-note corrected.
- N4 [FIXED] optional_attention_pytorch.md: the merged safety-net was already
  conditional, not unconditional as Codex phrased it; strengthened so that when
  it DOES swap the student's class it announces it loudly and prints the probe
  error + the likely cause (constructor arg not named `dropout_p`). No silent
  replacement.
- N5 [IGNORED] fixed-seed fallback embeddings - reproducibility is a feature for
  an optional lesson; no change.

---

# Codex (o3) Round 3 - pedagogical continuity (surgical)

Round 3 checked TEACHING-NARRATIVE continuity, not code. Findings:

C1. Topic 1 -> Topic 2 breaks the spiral. Topic 1 ends with a "Variables
    available for Topic 2" banner (client, my_system_prompt, test_complaints)
    but Topic 2 never references any of them (zero grep hits). The "variables
    carry over exactly" rule is violated at the very first handoff. Pre-existing,
    not caused by the renumber - but a real philosophy violation.

C2. The whole required path restarts state per notebook instead of extending
    one running Barclays system. Demo variables are recreated fresh each topic.

C3. Mini-lesson placement risk: required_path_continuity.md adds the transformer
    mini-lesson to topic_2 and the LoRA mini-lesson to topic_6, but topic_3
    cell ~14 already says "In Topic 4 you assembled a Transformer" and topic_6
    cells 13/148/263/296/346 say "recall LoRA from Topic 7a". Edits removing
    those orphaned references must be VERIFIED to fire, and any in-place teaching
    must land BEFORE the first such reference.

## Resolution (decided with Axel)

DECISION: cross-notebook handoff is done via S3, NOT kernel variables. The
class runs on SageMaker in AWS; every notebook can read/write the course S3
bucket. This makes the spiral literal and robust to kernel restarts between
sessions.

Apply COURSE-WIDE:
- Each required notebook ENDS with a "handoff" cell that writes the artifacts
  the next topic needs to a known S3 prefix (e.g.
  s3://<bucket>/barclays-course/topic_<N>/...). Examples: topic_1 writes the
  triage system prompt + the test complaints; later topics write datasets,
  checkpoints, label maps, etc.
- Each required notebook STARTS (after setup) with a "load handoff" cell that
  reads the previous topic's artifacts from S3, with a clear fallback if absent
  (define a minimal local version + print a note) so a student starting mid
  course is not blocked.
- The notebook narrative explicitly says "in the previous topic you produced X,
  saved to S3; we load it now and extend it."
- topic_1 has no predecessor: it only writes a handoff, no load.
- This is the literal mechanism behind the "one running system, built layer by
  layer" spiral.

The continuity design doc (required_path_continuity.md) must gain a new section
"S3 Handoff Chain" specifying, per topic: what it LOADS, what it WRITES, the S3
key layout, and the fallback behavior. Plus C1/C3 fixes above.
