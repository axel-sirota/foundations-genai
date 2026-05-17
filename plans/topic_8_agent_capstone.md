# Topic 8: The Barclays Complaint Agent - Cell-by-Cell Plan

Research basis: 4-cycle /research on capstone agent design (see session log).
Pipeline: this is a FROM-SCRATCH build (/build-topic-notebook 8), NOT rework mode.

## Capstone Design Note (READ BEFORE BUILDING)

Topic 8 is the CAPSTONE. It deliberately BREAKS the standard four-beat /
lab-tier structure. There are NO Tier 1/2 guided labs, NO `= None # YOUR CODE`
scaffolds, NO numbered steps. The notebook is:

1. SETUP cells - fully given, runnable as-is: installs, env, imports, the S3
   handoff load, and the agent-loop scaffold (the loop is plumbing, not the
   lesson - the research found students should not burn the day on
   tool_call_id wiring).
2. GUIDANCE cells - markdown only. They state the goal, the spec, the
   acceptance criteria, and the constraints. They do NOT show code.
3. The student writes everything else from a blank code cell: the 3 tools,
   the tool schemas, the system prompt, and the wiring into the loop.

The whole capstone is effectively one big Tier 3 task. The student has the
setup handed to them and the goal explained; the integration is theirs.

Solution notebook: the same notebook with the blank student code cells filled
in with a complete reference implementation. Setup and markdown cells are
identical between Exercise and Solution.

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~22 cells)

## Cell 0 - Markdown: Title and capstone framing

```
# Topic 8: The Barclays Complaint Agent (Capstone)

Barclays Customer Support Intelligence System | Topic 8 | CAPSTONE

This is the capstone. Across Topics 1 to 7 you built every piece of a
complaint-intelligence system: you loaded pretrained models, fine-tuned one,
transfer-learned another, adapted one with LoRA, and compressed a model for
serving. Each topic saved its work to S3.

Today you connect them. You will build ONE agent - a pure-Python program, no
agent frameworks - that uses an LLM as its brain and calls your earlier models
as tools to triage a Barclays complaint end to end.

## What this capstone is

This notebook hands you the setup: installs, environment, the S3 load of your
prior work, and the bare agent loop. Everything else - the tools, the tool
schemas, the system prompt, the orchestration - you write yourself, from
scratch. There are no numbered steps and no fill-in-the-blank scaffolds. You
have spent seven topics earning this.

## What you will build

A "Barclays Complaint Agent" that, given a raw customer complaint, decides on
its own which of its tools to call, calls them, and returns a triage decision
(category, urgency, a recommended action) with its reasoning.

## Estimated time

90 minutes. This is the hardest and most open task of the course.
```

## Cell 1 - Markdown: Section 0 - Environment Setup

```
## Section 0 - Environment Setup

The next four cells are GIVEN. Run them in order, top to bottom, and do not
edit them. They install dependencies, disable the TensorFlow backend, import
what you need, and load the artifacts your earlier topics saved to S3.
```

## Cell 2 - Code: TF-disable env vars (GIVEN, do not edit)

```
# Disable the TensorFlow backend in transformers (SageMaker image compatibility).
# Must run before any transformers import.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```

## Cell 3 - Code: install cell (GIVEN, do not edit)

```
# Capstone install cell. Pinned to the course-wide matrix.
!pip install -q \
    "openai>=1.0.0" \
    "transformers>=4.53,<4.54" \
    "accelerate>=1.0.0" \
    "tokenizers>=0.21,<0.22" \
    "sagemaker>=2.200.0,<3.0.0" \
    "numpy<2"

print("RESTART KERNEL before continuing -- environment packages were installed/upgraded.")
```

## Cell 4 - Code: imports + OpenAI client + SageMaker bucket (GIVEN, do not edit)

```
# Imports, the OpenAI client, and the course S3 bucket. GIVEN - do not edit.
import json
import getpass

from openai import OpenAI
import sagemaker

# gpt-4o is the course brain model. The key is entered at runtime, never hardcoded.
_api_key = getpass.getpass("OpenAI API key: ")
client = OpenAI(api_key=_api_key)
BRAIN_MODEL = "gpt-4o"

# The course S3 bucket - every topic saved its handoff artifacts here.
bucket = sagemaker.Session().default_bucket()
COURSE_PREFIX = "barclays-course"

print("OpenAI client ready. Brain model:", BRAIN_MODEL)
print("Course bucket:", bucket)
```

## Cell 5 - Markdown: Section 1 - What your earlier topics left you

```
## Section 1 - What Your Earlier Topics Left You

Every required topic wrote a handoff artifact to
`s3://<bucket>/barclays-course/topic_<N>/`. The cell below loads them. This is
the raw material your agent's tools will stand on:

- Topic 1 - the triage system prompt and the test complaints.
- Topic 3 - the routing label set and a labelled complaint dataset.
- Topic 4 - a fully fine-tuned classifier (model pointer).
- Topic 5 - a transfer-learned classifier (model pointer).
- Topic 6 - a PEFT / LoRA-adapted classifier (model pointer).
- Topic 7 - the compressed, deployment-ready model and endpoint.

The load cell has a fallback: if you skipped a topic or a training job did not
finish, the artifact may be missing. That is expected and handled. Your tools
must cope with a missing artifact - the design guidance below tells you how.
```

## Cell 6 - Code: S3 handoff load (GIVEN, do not edit)

```
# Load every prior topic's handoff artifact. GIVEN - do not edit.
# Anything missing comes back as None; your tools must handle that gracefully.
import boto3, botocore

def handoff_read(topic_n, artifact):
    key = f"{COURSE_PREFIX}/topic_{topic_n}/{artifact}"
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Loaded s3://{bucket}/{key}")
        return json.loads(body)
    except botocore.exceptions.ClientError:
        print(f"MISSING s3://{bucket}/{key} (a tool fallback will cover this)")
        return None

prior_work = {
    "triage":     handoff_read(1, "triage_config.json"),
    "labels":     handoff_read(3, "routing_labels.json"),
    "dataset":    handoff_read(3, "labelled_dataset.json"),
    "finetuned":  handoff_read(4, "model_pointer.json"),
    "transfer":   handoff_read(5, "model_pointer.json"),
    "peft":       handoff_read(6, "model_pointer.json"),
    "deployment": handoff_read(7, "deployment.json"),
}

# A sensible default routing label set if Topic 3 was skipped.
ROUTING_LABELS = (prior_work["labels"] or {}).get("labels") or [
    "fraud and security", "billing and charges",
    "account access", "general enquiry",
]
print()
print("Routing labels:", ROUTING_LABELS)
print("Prior-work artifacts present:",
      [k for k, v in prior_work.items() if v is not None])
```

## Cell 7 - Markdown: Section 2 - the agent loop (scaffold explanation)

```
## Section 2 - The Agent Loop

An agent is not magic. It is a loop:

1. Send the conversation and the list of available tools to the LLM.
2. The LLM replies. Either it answers, or it asks to call one or more tools.
3. If it asked for tools: run them yourself, append each result to the
   conversation, and go back to step 1.
4. If it answered: stop and return the answer.

The LLM never runs the tools. It only decides which to call and with what
arguments. Your code does the running.

The next cell GIVES you this loop. It is plumbing, not the point of the
capstone - so you do not have to write it. Read it carefully: it expects two
things YOU will define later - a list called `TOOL_SCHEMAS` (the JSON
descriptions the LLM sees) and a dict called `TOOL_FUNCTIONS` (mapping each
tool name to the Python function that runs it).
```

## Cell 8 - Code: the agent loop scaffold (GIVEN, do not edit)

```
# The agent loop. GIVEN - do not edit. It drives the conversation and the
# tool calls. It depends on TOOL_SCHEMAS and TOOL_FUNCTIONS, which YOU define.

def run_agent(system_prompt, user_message, max_turns=8):
    """Run the agent loop. Returns (final_answer, full_message_history).

    full_message_history is the agent's memory: every message, tool call, and
    tool result, in order. The loop re-sends it every turn.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        choice = response.choices[0].message
        messages.append(choice)

        if not choice.tool_calls:
            return choice.content, messages

        for call in choice.tool_calls:
            fn = TOOL_FUNCTIONS.get(call.function.name)
            args = json.loads(call.function.arguments)
            if fn is None:
                result = f"ERROR: unknown tool {call.function.name}"
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f"ERROR running {call.function.name}: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })
    return "Agent stopped: reached the turn limit.", messages
```

## Cell 9 - Markdown: Section 3 - YOUR capstone (the brief)

```
## Section 3 - Your Capstone

Everything above was setup. From here on, you write the code.

### The goal

Build the Barclays Complaint Agent. Given a raw customer complaint string, it
must decide for itself which tools to use, call them, and return a triage
result: a routing category, an urgency judgement, and a recommended action,
with the reasoning that led there.

### What you must produce

Three TOOLS, their SCHEMAS, a SYSTEM PROMPT, and the wiring - then a run.

1. Tool A - a code / computation tool. The agent can hand it a small piece of
   Python (or a fixed analysis) to compute something about complaint data -
   for example, how many complaints in the loaded Topic 3 dataset fall in each
   category. You decide its exact contract.

2. Tool B - a classifier built from a model you trained earlier. Wrap the
   Topic 4 fine-tuned model OR the Topic 5 transfer-learned model. Load the
   real artifact from its S3 model pointer if it is present in `prior_work`.
   If it is missing, fall back to a small public HuggingFace classifier doing
   the same job (a `transformers` pipeline, `framework="pt"`). Either way the
   tool returns a routing category for a complaint.

3. Tool C - a second classifier, from your PEFT / LoRA work (Topic 6). Same
   load-or-fallback pattern. The agent now has two classifiers and must decide
   which to trust, or call both and compare - that decision is part of the
   capstone.

4. TOOL_SCHEMAS - a list of JSON tool descriptions, one per tool, in the
   OpenAI function-calling format. This is what the LLM sees.

5. TOOL_FUNCTIONS - a dict mapping each tool name to its Python function.

6. A SYSTEM PROMPT that makes gpt-4o behave as the Barclays triage agent:
   tells it its job, its tools, and what a finished answer looks like.

7. Call `run_agent(system_prompt, complaint)` on at least two of the Topic 1
   test complaints and show the agent's reasoning and final triage.

### Acceptance criteria

- Run the agent on a fraud-type complaint and a billing-type complaint.
- For each, the agent must call at least one classifier tool and return a
  routing category from `ROUTING_LABELS`, an urgency, and a recommended action.
- The full message history (the agent's memory) must show the tool calls and
  their results - the decision must be traceable, not a black box.
- A missing S3 artifact must not crash the agent: the fallback path must work.

### Constraints

- Pure Python and the `openai` / `transformers` SDKs only. No agent frameworks
  (no LangChain, no CrewAI, no LlamaIndex).
- The brain is `gpt-4o` via the `client` already created. Do not hardcode keys.
- Use the GIVEN `run_agent` loop as-is. Your job is the tools and the brains
  around it, not the loop.
- HuggingFace pipelines must pass `framework="pt"` (SageMaker TF-backend rule).

### How you are assessed

There is no verification cell that grades you. The capstone works when you can
run it end to end and read a sensible, traceable triage decision out of the
agent. That is the bar. You have no scaffolding and no hints - this is the
point of a capstone.
```

## Cell 10 - Markdown: Discussion / planning prompt

```
### Before you write code - plan (5 minutes)

With the person next to you, agree on:

- Each tool's exact input and output. A tool that returns a clean string the
  LLM can read is far easier to orchestrate than one that returns a raw object.
- What goes in the system prompt so the agent actually USES the tools instead
  of guessing. What happens if you do not mention a tool in the prompt?
- The order you expect a good agent to call tools in - and whether you should
  force that order or let the LLM decide. Which is more in the spirit of an
  agent?
- How you will SEE the agent's reasoning. The returned message history is your
  window into it - plan how you will print it readably.
```

## Cell 11 - Code: blank capstone cell (Exercise: empty; Solution: full implementation)

```
# Your capstone starts here. Build the three tools, the schemas, the system
# prompt, and the wiring. Add as many cells as you need.
#
# (Exercise notebook: this cell is intentionally near-empty - the comment above
#  plus a single blank line. The student writes everything from here.)
```

SOLUTION NOTEBOOK - Cell 11 onward: a complete reference implementation,
broken across several code cells with explanatory markdown between them:

- SOL Cell 11a (code): Tool A - `analyze_complaint_dataset(category=None)` -
  counts complaints per routing category in `prior_work["dataset"]`, with a
  fallback message if the dataset is absent.
- SOL Cell 11b (code): a shared `_load_classifier(pointer, fallback_model_id)`
  helper - if `pointer` has a `model_tar_uri`, note it and (for the capstone)
  load a `transformers` pipeline; if absent, load `fallback_model_id`. Always
  `pipeline("text-classification", model=..., framework="pt")`.
- SOL Cell 11c (code): Tool B - `classify_finetuned(complaint)` - uses the
  Topic 4/5 pointer via `_load_classifier`, maps the pipeline label to one of
  `ROUTING_LABELS`, returns the category string.
- SOL Cell 11d (code): Tool C - `classify_peft(complaint)` - same pattern with
  the Topic 6 pointer.
- SOL Cell 11e (code): `TOOL_SCHEMAS` - the three OpenAI function-calling JSON
  descriptions; `TOOL_FUNCTIONS` - the name-to-function dict.
- SOL Cell 11f (code): `SYSTEM_PROMPT` - a full triage-agent system prompt
  naming the three tools and defining a finished answer (category + urgency +
  action + reasoning).
- SOL Cell 11g (code): runs `run_agent` on two Topic 1 test complaints and
  pretty-prints the final triage and the tool-call trace from the message
  history.

## Cell 12 - Markdown: Wrap-Up

```
## Wrap-Up - You Built the Whole System

If your agent runs and returns a traceable triage decision, you have just
connected every topic of this course into one working program:

- The LLM brain is the prompt-engineering and LLM-API work from Topics 1-2.
- The classifier tools are the models you fine-tuned, transfer-learned, and
  LoRA-adapted in Topics 4-6.
- The graceful fallbacks are the HuggingFace ecosystem skills from Topic 3.
- The whole thing is deployable because of the compression work in Topic 7.

You did this with pure Python and one loop. No framework. You now know what an
agent actually is - a loop, some tools, and a model that decides - because you
built one with nothing hidden from you.

### Homework Extension - give the agent a fourth tool

Add a naive RAG tool: take a handful of internal Barclays policy snippets
(plain strings), embed them and the query, score with cosine similarity,
return the top-2 snippets as context the agent can cite. No vector database -
a list and a dot product. Wire it in as a fourth tool and re-run the agent on
a complaint whose answer depends on a policy.
```

# DIAGRAM INDEX

## Diagram 1
- Diagram slug: `agent-loop`
- Diagram path: `../../plans/topic_8_agent_capstone/diagrams/agent-loop.mmd`
- Placeholder cell: a `<!-- DIAGRAM: the agent loop - LLM decides, code runs
  tools, results return, repeat until the LLM answers -->` line should be
  added to Cell 7 (the agent-loop explanation markdown). Description: a cycle -
  LLM call -> tool calls? -> run tools -> append results -> back to LLM; exit
  branch when the LLM returns a final answer instead of tool calls.

(One diagram only. The capstone is light on diagrams by design - it is a build,
not a concept lesson.)

# BUILD NOTES FOR /build-topic-notebook

- This is a FROM-SCRATCH build. Target dirs: Exercises/topic_8_agent_capstone/
  and Solutions/topic_8_agent_capstone/.
- The capstone deliberately has NO four-beat arc and NO Tier 1/2 labs. Do not
  add naive/broken "Beat 1" cells or `= None # YOUR CODE` scaffolds. The
  validate-notebooks check for those does not apply to topic 8 - note this in
  the build.
- Exercise Cell 11 is a near-empty blank cell (comment + blank line). The
  Solution replaces it with the multi-cell reference implementation (11a-11g).
  This is the ONE place Exercise and Solution diverge; all other cells are
  identical.
- No S3 WRITE cell - topic 8 is the last topic, it produces nothing downstream.
- Plain ASCII only, no AI-tells.
- Add the `<!-- DIAGRAM: -->` line to Cell 7 per the diagram index, so
  /build-diagrams can later generate agent-loop.mmd.
