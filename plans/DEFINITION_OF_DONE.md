# Definition of Done
# Generative AI for Developers - Barclays
# Last updated: 2026-05-11

## Course-Level Definition of Done

The course is done when ALL of the following are true:

1. Every topic in the outline has at least one notebook (exercise + solution pair)
2. Every notebook runs top-to-bottom without errors inside SageMaker Studio
3. Notebooks that use remote training have been verified with an actual training job
4. All diagram placeholders (`<!-- DIAGRAM: -->`) have been filled by `/build-diagrams`
5. `plans/TOPICS.md` shows `status: done` for every topic
6. The PyTorch reminder and SageMaker reminder framework folders exist and are complete
7. The outline-to-notebook mapping in `plans/TOPICS.md` is complete and accurate

---

## Topic-Level Definition of Done

A topic is done when:

- [ ] `/run-research-topic` completed: plan file at `plans/topic_N_<slug>.md` with all required sections
- [ ] `/verify-research` passed: 14-point checklist all green
- [ ] `/build-topic-notebook` completed: exercise notebook built 5 cells at a time with approval
- [ ] Solution notebook built: copied from exercise, all `= None  # YOUR CODE` replaced
- [ ] `/validate-notebooks` passed: structure, pedagogy, placeholders all valid
- [ ] `/build-diagrams` completed: all `<!-- DIAGRAM: -->` placeholders resolved
- [ ] `plans/TOPICS.md` updated: status -> done, all checkboxes marked

---

## Notebook-Level Definition of Done

A single notebook (exercise + solution pair) is done when:

### Structure
- [ ] Title cell with learning objectives and which Customer Service / Barclays component it builds
- [ ] Environment setup: `sagemaker.Session()` + `get_execution_role()` + pinned installs
- [ ] Every concept follows the four-beat arc: Beat 1 (broken) -> Beat 2 (diagram) -> Beat 3 (demo) -> Beat 4 (lab)
- [ ] No more than 3 consecutive markdown cells without a code cell in between
- [ ] Wrap-up cell with key takeaways, homework extensions, bridge to next topic

### Teaching quality
- [ ] Beat 1 code actually runs and actually fails or produces bad output (not just a comment)
- [ ] Beat 3 demo is heavily commented, runnable, first-person tone
- [ ] Lab tier assigned correctly: Tier 1 (guided, most labs), Tier 2 (hard, one per day), Tier 3 (open-ended, last topic of the day only)
- [ ] Every in-class lab has a stretch version for fast finishers
- [ ] Every in-class lab has a Homework Extension
- [ ] Peer discussion prompts between major sections (3-5 min, tradeoffs and production concerns)
- [ ] STAR method applied to labs: Situation (Barclays context) -> Task -> Action (YOUR CODE) -> Result (verification)

### Code correctness
- [ ] `numpy<2` pinned in install cell
- [ ] `getpass.getpass()` for all external API keys; no hardcoded secrets
- [ ] `eval_strategy="epoch"` (NOT `evaluation_strategy`) in all TrainingArguments
- [ ] `evaluate` library NOT used; inline numpy for metrics instead
- [ ] `requirements.txt` (not requirements_cpu.txt or any other name) in each `source_dir`
- [ ] HuggingFace estimator only on GPU instances (`ml.g4dn.xlarge` minimum)
- [ ] PyTorch estimator uses `framework_version="2.8.0"`, `py_version="py312"`
- [ ] SageMaker SDK pinned `>=2.200.0,<3.0.0`
- [ ] boto3 SageMaker exception: `ResourceNotFound` (NOT `ResourceNotFoundException`)
- [ ] MLflow tracking server: `MlflowVersion="2.13.2"` only

### Safety and placeholders
- [ ] Safety-net cells present for every lab whose output feeds a downstream cell
- [ ] Safety-net cells REMOVED from solution notebook
- [ ] `# YOUR CODE` placeholder comments do NOT reveal the answer
- [ ] Solution notebook has complete implementations with explanation comments

### No AI-tells
- [ ] No em dashes, en dashes, Unicode multiplication signs, or emojis in cell bodies or print statements
- [ ] Plain ASCII only in all text

### Validation
- [ ] `python validate_notebooks.py --pair exercises/.../nb.ipynb solutions/.../nb.ipynb` passes
- [ ] Both notebooks run top-to-bottom without errors
- [ ] File size under 500KB each

---

## Remote Training Job Definition of Done (applies to all capstones from Transformer Translator onwards)

- [ ] `source_dir` contains exactly `train.py` and `requirements.txt` (mandatory names)
- [ ] `requirements.txt` does NOT include `evaluate`; uses inline numpy for metrics
- [ ] `eval_strategy` (not `evaluation_strategy`) in `TrainingArguments`
- [ ] Job runs to completion without error (verified by polling `DescribeTrainingJob`)
- [ ] CloudWatch logs accessible and show training progress
- [ ] Model artifacts land in S3 default bucket
- [ ] If using MLflow: tracking server exists at `MlflowVersion="2.13.2"`, inline policy `sagemaker-mlflow:*` is in role

---

## Framework Folders Definition of Done

### Frameworks/PyTorchReminder/
- [ ] Covers: tensors, autograd/GPU, Dataset/DataLoader, nn.Linear classifier, nn.Sequential classifier
- [ ] All 5 exercises from PytorchPrimer adapted to run in SageMaker Studio
- [ ] Solutions updated to match

### Frameworks/SageMakerReminder/
- [ ] Covers: SageMaker session setup, S3 read/write, launching a training job, MLflow tracking, model registry, endpoint deployment
- [ ] Uses the canonical version matrix from CORE_TECHNOLOGIES_AND_DECISIONS.md
- [ ] Walks through the exact patterns students will use in all subsequent capstones
