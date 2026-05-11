# Notebook Audit: topic_7a_lora_ffn
Generated: 2026-05-11

## Status: PASS (with minor WARNs)

### Checks Summary
| Check | Result |
|-------|--------|
| Four-beat arc | WARN |
| Diagrams (2, markdown, Beat 2 position) | PASS |
| Safety-nets (exercise) | PASS |
| Safety-nets (solution kept) | PASS |
| AI-tells | PASS |
| Lab tiers | PASS |
| YOUR CODE hygiene | PASS |
| Notebook header (no Day N) | PASS |
| Narrative / Barclays STAR | PASS |
| SageMaker constraints | PASS |
| Discussion prompts | WARN |
| Cell parity (ex=41, sol=41) | PASS |
| No markdown chain > 3 | PASS |
| Solution clean | PASS |

### Issues to Fix

#### WARN — Four-beat arc (Section 2)
- Section 2 header markdown (cell id `d35d4b64efd1`, index 14) places the `<!-- DIAGRAM: -->` Beat-2 placeholder at the very top of the section, BEFORE any Beat 1 broken/naive code cell. Cell 15 (`b09f64809213`) is FFN pre-training (legitimate setup, not a broken/naive demo), and cell 16 (`ed5950a44d61`) jumps straight to the working Beat-3 LoRA replacement. The expected order is B1 -> B2 -> B3 -> B4; here it is effectively B2 -> setup -> B3 -> B4. Suggested fix: either reorder so the diagram appears just before cell 16, or add a short Beat-1 cell that demonstrates a naive alternative (e.g. unfreezing the whole FFN and showing the cost), then place the diagram before cell 16.

#### WARN — Four-beat arc (Section 3)
- Section 3 (cells 26-28: header md, viz code, discussion md) has no Beat 1 broken/naive code, no diagram, no Beat 4 lab. It is a pure heuristics/visualization section. This is consistent with the "2 Tier 1 labs only" requirement (labs sit in Sec 1 and Sec 2) but the four-beat arc rule says "every major section". Acceptable as a knowledge interlude — flagging for awareness, not requiring a fix.

#### WARN — Four-beat arc (Section 4 capstone)
- Section 4 (cells 29-38) is a capstone-style walkthrough. No B1 (broken/naive), no `<!-- DIAGRAM: -->` placeholder, no Lab/B4. By design — capstone replaces structured beats with a real training job — but again does not fit the strict four-beat rule. Acceptable as capstone; flagging for awareness.

#### WARN — Discussion prompts cadence
- Only 2 peer discussion markdown cells in the notebook: cell 18 (`e33980047d98`, mid-Section 2) and cell 28 (`ac866245836a`, end of Section 3). There is no discussion between Section 1 (LoraLayer scratch implementation) and Section 2 (applying to FFN). The rule says "at least one peer discussion markdown cell between every two major sections" — between Sec1 and Sec2 there is none. Suggested fix: add a 3-min peer-discussion markdown cell after the Section 1 wrap-up (after cell 13) framed with Barclays context (e.g. "Why does freezing the backbone matter for our complaint-classifier deployment story?").

### Detail: PASS items worth noting
- AI-tells programmatic scan: both notebooks clean (no em dash, en dash, unicode multiplication, emoji, bare `---` separators).
- Cell parity: exercise = 41, solution = 41, identical cell ids.
- Diagrams: exactly 2 placeholders, both in markdown cells. Files exist at `/Users/axelsirota/repos/genai_for_developers/plans/topic_7a_lora_ffn/diagrams/lora-decomposition.mmd` and `lora-parameter-comparison.mmd`. Paths use full folder name `topic_7a_lora_ffn`.
- Safety-nets present after every variable-producing lab/training cell: cell 10 (Lab 1 -> LoraLayerStudent), cell 17 (LoRA replacement -> lora_model), cell 22 (Lab 2 -> replace_fc_with_lora_student), cell 34 (`.fit(wait=False)` -> training_job_name). All use `# Safety-net: ... # SKIP this cell if ...` header.
- Solution lab cells (9, 21) and homework starter (13, 25) all have working implementations; no `None  # YOUR CODE` or `pass` leakage.
- Safety-net cells (10, 17, 22, 34) retained in solution per requirement.
- Lab tiers: Lab 1 (cells 8-13) Tier 1 guided, Lab 2 (cells 20-25) Tier 1 guided. Both have Stretch + Homework Extension. No Tier 2, no Tier 3 — matches T7a allocation.
- YOUR CODE hygiene: every `# YOUR CODE` placeholder is bare or follows `variable = None`. No inline answer hints after the marker.
- Header (cell 0): "Topic 7a - LoRA from Scratch" — no Day-N reference.
- Barclays STAR: both labs have explicit Situation/Task/Action/Result subsections framed as Barclays Customer Support Intelligence System.
- Setup cell (cell 1): defines `sess, role, bucket, region` via `get_execution_role()`. Cell 2 defines `device, set_seeds`. Carry-over from T6b naming preserved.
- SageMaker constraints: `numpy<2` pinned in install cell; `sagemaker>=2.200.0,<3.0.0`; `transformers>=4.35.0,<4.40.0`; HuggingFace estimator pinned to `transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312"` on `ml.g4dn.xlarge`; `.fit(wait=False)`; no `evaluate` import, no `evaluation_strategy`, no `ResourceNotFoundException`, no MLflow.
- Markdown chains: longest run is 2 consecutive markdown cells (28 + 29), well under the 3-cell ceiling.
