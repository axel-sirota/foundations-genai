# Topic 7b Verification Report

**Date:** 2026-05-11
**Notebooks audited:**
- `Exercises/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb`
- `Solutions/topic_7b_peft_lora_distilbert/topic_7b_peft_lora_distilbert.ipynb`

## Summary

Both notebooks passed full audit with **zero violations found**. All required
diagrams were already wired in correctly. No edits were needed.

## Audit results

| Check | Exercise | Solution |
|-------|----------|----------|
| Cell count | 44 | 44 (parity OK) |
| DIAGRAM placeholders | 2 (cell 5, cell 17) | 2 (cell 5, cell 17) |
| Diagram link format | OK (`[View diagram](../../plans/...)`) | OK |
| Both .mmd files exist on disk | yes | yes |
| AI-tells (em/en dash, unicode mul) | clean | clean |
| Bare `---` in markdown | none | none |
| `YOUR CODE:` hints (forbidden) | none | none |
| `mlflow` references | none | none |
| `evaluation_strategy=` misuse | none (only mentioned in constraint text "not evaluation_strategy") | same |
| `import evaluate` | none | none |
| Markdown chain > 3 | none | none |
| `wait=False` on `.fit(` | yes (cell 35) | yes (cell 35) |
| `training_job_name` safety-net after fit | yes (cell 36) | yes (cell 36) |
| `numpy<2` in install cell | yes (cell 1) | yes (cell 1) |
| No "Day N" in header | confirmed | confirmed |
| Setup vars from T7a (`lora_r`, `device`, `set_seeds`, `sess`, `role`, `bucket`, `region`) | all present (cells 1-2) | all present |

## Four-Beat Arc verification

### Section 1 (PEFT LoRA on DistilBERT)
- Beat 1: cell 4 (naive manual injection failure)
- Beat 2: cell 5 (DIAGRAM placeholder + link to `peft-methods-comparison.mmd`)
- Beat 3: cells 6-7 (PEFT library working demo + inference check)
- Beat 4: cells 8 (Lab 1 markdown, Tier 1) + 9 (starter), 10 (safety-net), 11 (verify)

### Section 2 (QLoRA)
- Beat 1: cell 16 (bitsandbytes on CPU failure)
- Beat 2: cell 17 (DIAGRAM placeholder + link to `qlora-architecture.mmd`)
- Beat 3: cell 18 (QLoRA walkthrough)
- Beat 4: cells 19 (Lab 2 markdown, Tier 2) + 20 (starter), 21 (safety-net)

### Section 3 (Soft prompts)
- Beat 1: cell 26 (prefix length mismatch)
- Beat 3: cell 27 (correct soft prompt config) — T7b has exactly 2 diagrams (already used)
- Beat 4: cells 28 (Lab 1b, Tier 1) + 29 (starter), 30 (safety-net), 31 (3-way comparison)

### Section 4 (Capstone)
- Cells 33-38: setup + SageMaker GPU job launch + polling
- Capstone Lab: cell 39 (Tier 2) + cell 40 (starter)

## Lab tier distribution (matches T7b spec)
- Lab 1 = Tier 1 (guided, r=16 PEFT) — OK
- Lab 1b = Tier 1 (guided, soft prompt config) — OK
- Lab 2 = Tier 2 (hard, build_qlora_model) — OK
- Capstone = Tier 2 (hard, train_peft_complaint_classifier) — OK
- No Tier 3 (correct — Tier 3 is reserved for T8) — OK

## Safety-nets verified

Exercise notebook safety-net cells (all preserved in solution per rule):
- Cell 10: Lab 1 (`peft_model_r16`)
- Cell 21: Lab 2 (`build_qlora_model` via probe)
- Cell 30: Lab 1b (`prompt_model`)
- Cell 36: SageMaker `training_job_name`

## Solution lab cells (complete implementations)
- Cell 9: full r=16 LoRA build
- Cell 20: full `build_qlora_model` with CUDA branch + CPU fallback
- Cell 29: full `prompt_config` / `prompt_model` / `trainable_soft`
- Cell 40: full `train_peft_complaint_classifier` with HF Trainer

## Discussion prompts (peer)
- Cell 14: 3-min discussion after Section 1
- Cell 24: 3-min peer discussion after Section 2
- Cell 32: 3-min discussion after Section 3

## Diagrams wired

Exercise cell 5 and Solution cell 5 (identical):
```
<!-- DIAGRAM: PEFT methods comparison (LoRA, QLoRA, Soft Prompts) showing parameter efficiency of each -->
[View diagram](../../plans/topic_7b_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd)
```

Exercise cell 17 and Solution cell 17 (identical):
```
<!-- DIAGRAM: QLoRA architecture showing 4-bit NF4 base model with float16 LoRA adapters -->
[View diagram](../../plans/topic_7b_peft_lora_distilbert/diagrams/qlora-architecture.mmd)
```

Both target .mmd files exist on disk:
- `plans/topic_7b_peft_lora_distilbert/diagrams/peft-methods-comparison.mmd` (782 B)
- `plans/topic_7b_peft_lora_distilbert/diagrams/qlora-architecture.mmd` (583 B)

## Fixes applied

None — both notebooks were already fully compliant.

## Conclusion

Topic 7b exercise and solution notebooks are production-ready.
