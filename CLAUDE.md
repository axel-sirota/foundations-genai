# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A Generative AI course for developers (originally `foundations-genai`). Content is organized into:

- **`Exercises/`** — student-facing notebooks with blanks to fill in (numbered 2–13)
- **`Solutions/`** — complete reference implementations matching each exercise
- **`PytorchPrimer/`** — PyTorch-specific exercises and demos (tensors, autograd, Dataset/DataLoader, classifiers)
- **`student-environments/`** — Terraform + scripts to provision per-student AWS infrastructure

## Environment Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Key packages: `torch`, `transformers`, `gensim`, `spacy`, `evaluate`, `scikit-learn`, `pandas`, `numpy`.

No virtualenv is pre-configured — create one before installing if needed.

## Notebook Curriculum (in order)

| # | Topic |
|---|-------|
| 2 | Text processing, logistic regression, boosting |
| 3 | CBOW word embeddings (gensim) |
| 4 | Fine-tuning HuggingFace models |
| 5 | News classification with MLP |
| 6 | Text generation with LSTM |
| 7 | Named entity recognition with BiLSTM |
| 8–9 | Attention mechanisms (pure Python, then PyTorch) |
| 10 | Neural machine translation with attention |
| 11 | Transformer-based translator |
| 12 | Prompt engineering |
| 13 | Transfer learning |

**PyTorch Primer** (separate track, 5 exercises): tensors → autograd/GPU → Dataset/DataLoader → `nn.Linear` classifier → `nn.Sequential` classifier.

Every exercise in `Exercises/` has a matching solution in `Solutions/`. When editing an exercise, keep the solution in sync.

## Notebook Course Pipeline

All notebook work follows the pipeline defined in `SDD_WORKFLOW.md` and `.claude/pipeline.yaml`. **No feature branches — all commits go directly to main** (`ALLOW_DIRECT_MAIN=1` is set in `.claude/settings.json`).

**Chain:**
```
/request -> /specify -> /run-research-topic -> /verify-research
  -> /build-topic-notebook -> /validate-notebooks -> /build-diagrams -> commit
```

There is no `/finish-feature`, no PR flow, no skeleton/implement/tests gates. This is a content repo.

### Teaching Methodology: Four-Beat Arc

Every concept in every notebook follows exactly this sequence — no exceptions:

1. **Beat 1** — Markdown header + broken/naive code that runs and fails (students feel the pain)
2. **Beat 2** — `<!-- DIAGRAM: description -->` placeholder + 1-2 sentences of what it shows
3. **Beat 3** — Full working demo, heavily commented, instructor live-codes from this
4. **Beat 4** — Lab instructions (Markdown) + lab starter code

### Lab Tiers (plan ONE per day across all topics that day)

- **Tier 1 (guided)**: `variable = None  # YOUR CODE` with numbered steps + verification. 15-20 min. Most labs.
- **Tier 2 (hard)**: Multi-step, less prescriptive. 25-35 min. ONE per day.
- **Tier 3 (open-ended)**: Function signature + docstring only. ONE per day, last topic of day only (topics 3, 6, 9).

Every lab has a **stretch version** for fast finishers and a **Homework Extension**.

### STAR Method

Labs are framed as: **Situation** (Barclays customer service context) → **Task** (what to build) → **Action** (`# YOUR CODE` scaffolding) → **Result** (verification cell).

### Diagram Convention

```markdown
<!-- DIAGRAM: what this diagram shows -->
[View diagram](../../plans/topic_N/diagrams/slug.mmd)
```

Diagrams are Mermaid source files, not embedded images. `/build-diagrams` generates them from `<!-- DIAGRAM: -->` placeholders.

### Key Rules

- `numpy<2` pinned in every install cell
- `getpass.getpass()` for all API keys (never hardcoded)
- Model: `gpt-4o` for OpenAI — no Anthropic SDK in student notebooks
- Prior-topic variable names must carry over exactly (check the previous notebook before building)
- No AI-tells: no em dashes, en dashes, Unicode multiplication signs, or emojis anywhere in cell bodies, print statements, or markdown headers. Plain ASCII only.

### Notebook Authoring Rules (Hard-Won)

**5-cells approval cadence**: Add max 5 cells with NotebookEdit, STOP, ask "I've added cells X-Y. How does it look? Should I continue?", wait for explicit approval before the next batch. Never interpret "continue" as permission to do all remaining cells. Run `/validate-notebooks` after each 5-cell batch.

**Cell insertion order**: After the first cell, always pass `cell_id` (the cell to insert AFTER) to NotebookEdit. Inserting without `cell_id` puts the cell at the top of the notebook.

**Build order**: Exercise first, solution second. Solution = `cp exercises/.../nb.ipynb solutions/.../nb.ipynb`, then replace each `= None  # YOUR CODE` lab cell with a complete implementation. Never build exercise and solution in parallel.

**Safety-net cells**: If a lab variable feeds a downstream cell, add a safety-net cell immediately after the lab starter:

```python
# Lab N safety-net: run this if you did not finish Lab N.
# SKIP this cell if you DID finish Lab N.
if my_variable is None:
    print("Using Lab N safety-net so the rest of the notebook can run.")
    my_variable = <working implementation>
```

Remove safety-net cells from the solution notebook (the lab cell IS the solution there).

**`# YOUR CODE` hygiene**: The placeholder line must not hint at the answer.
- Correct: `result = None  # YOUR CODE`
- Wrong: `result = None  # YOUR CODE: filter df where amount > 1000`

**Peer discussion prompts**: Between major sections add a Discussion markdown cell (3-5 min). Focus on tradeoffs, consequences, and real-world implications — not just "how" but "why" and "what if". Frame from the student's professional perspective.

**Homework Extensions**: Keep in-class labs ~15 min each. Add a "Homework Extension" markdown + starter code cell after every lab for async exercises that build on in-class work.

**Optional deep-dive notebooks**: When a topic has both a practical and theoretical side, the main notebook covers the practical (required for all students). Create a separate `topic_N_optional_<slug>.ipynb` for theory/internals (advanced learners, self-contained, clearly marked supplementary).

**No markdown chain**: Never chain more than 3 consecutive markdown cells without a code cell in between — at minimum one line showing the concept in action.

## Student Infrastructure (Terraform)

Located in `student-environments/`. Provisions per-student: IAM user + access key + login profile, personal S3 bucket (`devint-<name>-<random>`), SageMaker notebook instance (CPU: `ml.t3.2xlarge` in us-west-1; GPU: `ml.g5.xlarge` in us-east-1).

**Workflow to add/change students:**

```bash
# 1. Edit names.txt with desired usernames
# 2. Regenerate terraform.tfvars
python changer.py > tmp && mv tmp terraform.tfvars

# 3. Apply (script exports PGP key automatically)
cd student-environments
./terraform.sh plan
./terraform.sh apply
```

The `terraform.sh` wrapper exports your GPG key (`gpg --export "Axel Sirota"`) as `TF_VAR_pgp_key` and runs `terraform init` before every command. AWS profile used: `di`.

**Never run `terraform destroy` manually** — use `./terraform.sh destroy` so the PGP key is set correctly.

Terraform state lives in `student-environments/terraform.tfstate` (committed). Per-user state is in `tf-user<N>/` subdirectories.
