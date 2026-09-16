# RETIRED — do not use for training or evaluation

The Claude-authored trajectories described below were retired on 2026-08-03
and moved to:

`/mnt/shared_ru.ml.SZ-5_000264/gambashidze/retired_data/APIGen-MT-claude-traces-20260803/`

They are preserved only for historical inspection. They are excluded from the
active aggregate and must not be restored into `data/generated` for SFT/RL.

# Claude Code task: create 100 fixed refusal/parallel trajectories

Use this file as the complete task specification. Work autonomously until the
final dataset passes every check below.

## Goal

Create **exactly 100 diverse, natural, multi-turn APIGen-MT trajectories** for
training tool-calling models. The trajectories must exercise:

- refusing or asking a clarification when a request cannot safely be executed;
- recovering after the user answers a clarification;
- issuing several independent tool calls in one parallel assistant action;
- combined conversations containing both clarification/recovery and a later
  parallel action;
- ordinary ordered tool use before, between, and after those feature actions.

Claude Code and its subagents are the generation, critique, and judging models.
Do **not** call OpenRouter, another hosted LLM, `src/generate_step_by_step.py`, or
any script that requires `OPENAI_API_KEY`. Local Python tool implementations may
and should be executed to verify and correct the trajectories.

The unit of this deliverable is one complete source trajectory row. There must
be exactly 100 source rows. Because a combined row has two feature transitions,
exporting one row per feature decision will produce 120 evaluation tasks; that
is expected.

## Repository and output

Repository root:

```text
/mnt/shared_ru.ml.SZ-5_000264/gambashidze/tool_synth/APIGen-MT-main
```

Create:

```text
data/generated/fixed_refuse_parallel_100_20260729/
├── schedule.jsonl
├── shards/
│   ├── shard_0.jsonl
│   ├── shard_1.jsonl
│   ├── shard_2.jsonl
│   ├── shard_3.jsonl
│   └── shard_4.jsonl
├── reviews/
│   ├── refusal_review.jsonl
│   ├── parallel_review.jsonl
│   └── naturalness_diversity_review.jsonl
├── fixed_refuse_parallel_100.jsonl
├── refusal_interactive_32.jsonl
├── refusal_unsupported_8.jsonl
├── parallel_40.jsonl
├── combined_20.jsonl
├── fixed_refuse_parallel_100.audit.json
├── fixed_refuse_parallel_100.internal.tasks.jsonl
├── fixed_refuse_parallel_100.bfcl_native.tasks.jsonl
├── fixed_refuse_parallel_100.review.html
└── manifest.json
```

It is fine to add a local helper such as
`scripts/build_fixed_refuse_parallel_100.py` for assembly, replay, validation,
repair, and deterministic reporting. It must not make network or LLM calls.

## Required subagent architecture

The lead Claude instance is the orchestrator. It must actually spawn subagents,
not write all 100 rows itself.

1. Read the authoritative files listed below.
2. Build `schedule.jsonl` first.
3. Spawn five author subagents. Give each one a disjoint 20-row schedule shard
   and a disjoint output file. Authors propose scenarios, write complete rows,
   run local checks, and repair their own failures.
4. Merge the five shards only after each contains exactly 20 valid JSON rows.
5. Spawn three independent judge subagents:
   - refusal/clarification/recovery judge;
   - parallelism, ordering, and simulator-correctness judge;
   - naturalness, difficulty, and dataset-diversity judge.
6. Judges must inspect actual rows, write per-row verdicts with actionable error
   codes, and may directly repair rows or send them back to author subagents.
7. Re-run all deterministic checks after every repair.
8. Stop only when all 100 rows pass all three judge tracks and the automated
   audit.

Subagents may author the entire trajectory, including tool calls, expected
outputs, assistant responses, state, and metadata. They must then replay local
Python tools and correct any output or state that does not match reality.
Claude judgment is required in addition to deterministic checks.

Avoid shared-file conflicts: authors only edit their own shard; judges only edit
their own review file. The lead owns the schedule, merged files, manifest, and
HTML.

### Per-shard profile allocation

Every author gets 20 rows:

| Shard | Missing argument | Ambiguity | Unsupported | Parallel only | Combined missing | Combined ambiguity |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 3 | 1 | 8 | 2 | 2 |
| 1 | 3 | 4 | 1 | 8 | 2 | 2 |
| 2 | 3 | 3 | 2 | 8 | 2 | 2 |
| 3 | 3 | 3 | 2 | 8 | 2 | 2 |
| 4 | 3 | 3 | 2 | 8 | 2 | 2 |
| **Total** | **16** | **16** | **8** | **40** | **10** | **10** |

Thus the 100 source rows contain 60 refusal-family feature decisions and 60
parallel feature decisions, with 20 combined rows contributing to both counts.

## Authoritative files to read

Read these before authoring:

1. `AGENTS.md`
2. `src/refuse_parallel_eval.py`
   - authoritative serialization and policy-context construction;
   - `prepare_multiturn_datapoint`;
   - `validate_feature_evaluation_spec`;
   - unordered multiset semantics for parallel calls.
3. `src/refuse_parallel.py`
   - `REFUSE_TOOL_SCHEMA`;
   - refusal reasons and responses;
   - parallel certification logic;
   - clarification/recovery behavior.
4. `src/apigen_multi_turn.py`
   - `Turn`, `MultiTurnConversation`, and `MultiTurnDatapoint`.
5. `src/apigen_step_by_step.py`
   - `TrajectoryStep`, `ToolCallWithOutput`, verification models, and tool
     contract hashing.
6. `src/tool_manager.py`
   - `ToolManager`;
   - `initialize_api_state`, `get_api_state`, `restore_api_state`;
   - `has_python_implementation`, `invoke_python_tool`.
7. `tests/unit/test_refuse_parallel.py`
8. `tests/unit/test_refuse_parallel_scaling.py`
9. `tests/unit/test_bfcl_shaped_refusal_parallel_schedule.py`
10. `scripts/audit_refuse_parallel_dataset.py`
11. `scripts/export_refuse_parallel_tasks.py`
12. `scripts/render_refuse_parallel_html.py`
13. `scripts/generate_bfcl_shaped_refusal_parallel_500.py`
    - read its local BFCL scheduling functions and shape checks;
    - do not run its API-backed generation.

Tool contracts, real implementations, initial states, and BFCL references:

```text
magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl
magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl
tools/
```

Examples may be inspected for shape, but must not be blindly copied:

```text
tests/unit/test_refuse_parallel_scaling.py
data/generated/runs/hard_natural_balanced_500_20260728/shards/
data/generated/parallel_multiturn10_steps7_15_grok45_10_20260728_v2.jsonl
data/generated/refusal_multiturn10_steps7_15_grok45_10_20260728_v2.jsonl
```

The two top-level example files use evaluation-spec v2. New rows must be
postprocessed by current code and serialize as `refuse-parallel-v3`.

## Schedule requirements

Use deterministic seed `20260729`.

### Number of assistant action transitions

Every row has 7-15 assistant action transitions. A parallel batch counts as one
transition regardless of its call count. A synthetic refusal also counts as one.

Use this exact global distribution:

| Steps | Rows |
|---:|---:|
| 7 | 12 |
| 8 | 11 |
| 9 | 11 |
| 10 | 11 |
| 11 | 11 |
| 12 | 11 |
| 13 | 11 |
| 14 | 11 |
| 15 | 11 |

To distribute it cleanly, give every shard two rows of every length, then add:

- shard 0: one extra 7-step and one extra 8-step row;
- shard 1: one extra 9-step and one extra 10-step row;
- shard 2: one extra 11-step and one extra 12-step row;
- shard 3: one extra 13-step and one extra 14-step row;
- shard 4: one extra 15-step and one extra 7-step row.

Use BFCL-v3 empirical turn counts and calls-per-turn from
`bfcl_v3_invocation_examples.jsonl`, following the conditioning logic in
`generate_bfcl_shaped_refusal_parallel_500.py`. Ordinary turns may contain 1-3
ordered action steps. Interactive refusal and combined rows need at least five
turns. A clarification must have an immediate recovery turn and at least one
subsequent turn. In combined rows the certified parallel action is the final
action.

### Parallel width

There are 60 parallel-containing rows. Use exactly:

- 20 rows with 3 calls;
- 20 rows with 4 calls;
- 20 rows with 5 calls.

Within each shard, its 12 parallel-containing rows must include four of each
width.

### Tool-domain diversity

Use all eight local categories:

- Communication
- Events
- Finance
- Posting Api
- Science
- Storage
- Travel Booking
- Vehicle Control

Assign a primary category to every scheduled row. Four categories should have
13 rows and four should have 12. Every shard must cover all eight categories.
Parallel-only rows must cover all eight categories globally; each refusal
reason must cover at least four categories. Include coherent cross-category
conversations where natural, but do not combine unrelated actions merely to
tick a quota.

No single scenario template, user-query skeleton, exact call sequence, or
dominant tool may be repeated with renamed constants. Use varied entities,
argument shapes, dependencies, domains, phrasing, and conversation structure.

## Semantics of each profile

### Missing argument: 16 refusal-only + 10 combined

- A relevant real tool is available.
- The request omits exactly one materially required value.
- That value is not visible earlier and cannot be recovered with another
  available tool.
- The assistant makes no real tool call at the blocked transition.
- Internal target:
  `refuse({"reason": "missing_argument"})`.
- BFCL-native target: zero tool calls and a targeted clarification question.
- The next user turn naturally supplies the missing fact.
- The assistant then executes the original request with real tools and real
  outputs.
- Never silently invent the missing value.

### Ambiguity: 16 refusal-only + 10 combined

- The request has at least two materially different, plausible interpretations.
- Earlier context does not select one safely.
- The assistant makes no real tool call at the blocked transition.
- Internal target: `refuse({"reason": "ambiguity"})`.
- BFCL-native target: zero tool calls and a concise disambiguating question.
- The next user turn naturally selects one interpretation.
- The assistant then completes the selected operation.
- Do not label an ordinary missing field as ambiguity.

### Unsupported capability: 8

- No exposed real tool or valid composition can satisfy the request.
- Include plausible distractor tools, but none may actually provide the missing
  capability.
- This is terminal: no fake recovery and no real tool call.
- Internal target:
  `refuse({"reason": "no_appropriate_function"})`.
- BFCL-native target: zero tool calls and a grounded explanation that the
  capability is unavailable.
- Do not use unsupported capability for a merely malformed, unsafe, expensive,
  or inconvenient call.

### Parallel only: 40

- The final assistant action contains 3-5 calls.
- All calls are emitted together in one `tool_calls` list.
- Arguments for every sibling call are fully known from the policy-visible
  context before the batch.
- No sibling consumes another sibling's output.
- Calls are read-only and independent.
- Reversing the calls produces identical per-call outputs and identical final
  state.
- `execution_mode` is `parallel`.
- `call_order_matters` is `false`.
- Matching is an unordered **multiset**, so call order is ignored only inside
  this action; call multiplicity is preserved.
- All earlier and later episode transitions remain ordered.

### Combined: 10 missing + 10 ambiguity

- A nonterminal clarification/refusal occurs first.
- The immediately following turn resolves it and successfully recovers.
- Later ordered work may use prior outputs.
- The final action is an independent 3-5-call parallel batch.
- The refusal and parallel action must concern a coherent overall conversation,
  not two unrelated benchmark tricks.

## Naturalness and difficulty

The user should state real goals, not an execution plan.

Required:

- Natural conversational language across turns.
- Users refer to people, files, trips, tickets, vehicles, posts, measurements,
  or financial tasks as humans normally would.
- When an internal ID can be obtained with a lookup tool, the user asks using a
  natural name or description and the assistant discovers the ID.
- Later actions should often depend on actual prior tool outputs.
- At least two nontrivial ordered dependencies in most 10-15-step rows.
- Concise follow-up turns that use conversation context naturally.
- Assistant prose must be grounded only in visible user context and actual prior
  outputs.
- Feature actions should require attention amid useful surrounding work.

Reject or rewrite:

- “First call X, then call Y” or enumerated tool plans.
- Mentions of APIs, schemas, function names, argument keys, benchmark mechanics,
  refusal labels, or parallel execution.
- Requests for opaque internal IDs when a normal user would not know them.
- Unnaturally stuffing every literal into one sentence.
- Saying “in parallel”, “simultaneously”, “independently”, “in any order”, or
  “at the same time” merely to reveal the expected execution mode.
- Fake complexity consisting only of many arithmetic calls or the same lookup
  repeated with renamed values.
- Outputs, IDs, or facts that were never provided by the user or returned by a
  prior tool.

## Required row shape

Follow `MultiTurnDatapoint`. Each JSONL row must contain at least:

```text
conversation
  overall_task
  turns[]
    turn_number
    user_query
    query_intent
    steps[]
      step_number
      tool_calls[]
        tool_name
        arguments
        output
      execution_mode
      call_order_matters
      reasoning
      pre_state
      post_state
      state_verification
      quality_verification
    assistant_response
    expected_tools
    execution_context
    quality_verification
  tools_used
  categories_used
  initial_api_state
generation_metadata
verification_result
token_usage
initial_api_state
available_tools
```

Use exact schemas from the local tool pool. Refusal-containing rows expose a
deep copy of `REFUSE_TOOL_SCHEMA`; BFCL-native export will remove it.
Parallel-only rows do not need the synthetic refusal tool.

Expose 8-16 relevant tool contracts per row, including plausible distractors.
Do not expose the whole tool pool by default. Calculate `tool_contract_hash`
exactly as `_tool_contract_hash` in `src/apigen_step_by_step.py`.

After constructing or repairing a row, call
`prepare_multiturn_datapoint(row)`. Do not hand-maintain
`generation_metadata.evaluation_spec`; current code must derive it from the
actual conversation. The resulting spec must be `refuse-parallel-v3`.

Do not mark API-generated naturalization certificates. These queries are
directly authored and judged by Claude subagents. Record the real provenance,
for example:

```json
{
  "authoring_method": "claude-code-subagent",
  "external_llm_api_used": false
}
```

Deterministic replay checks and subagent judgments may be recorded as such; do
not pretend they came from an external judge model.

## Local execution and replay

`ToolManager` accepts an LLM object, but direct Python tool execution does not
need an LLM response. Use a no-network stub whose `generate` method raises if
called:

```python
class NoLLM:
    def generate(self, *args, **kwargs):
        raise RuntimeError("No LLM calls are allowed for this fixed dataset")

    def get_token_usage(self):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0,
        }
```

Initialize:

```python
manager = ToolManager(
    llm=NoLLM(),
    tool_pool_path="magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl",
    invocation_examples_path=(
        "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
    ),
    use_config_pool=True,
)
```

For every new row:

1. Seed local randomness from `20260729 + task_index`.
2. Clear the cached config and initialize a fresh local API state.
3. Save the initial state.
4. Execute ordinary calls in declared order.
5. Save exact pre-state, output, and post-state for every step.
6. For a refusal, use the synthetic unchanged-state result from
   `src/refuse_parallel.py`; never invoke a real tool.
7. For a parallel group:
   - save one shared pre-batch state;
   - run every call independently from that state;
   - run the complete batch forward and reversed from that state;
   - reject unless all runs are error-free;
   - reject unless each call has a Python implementation;
   - reject unless outputs agree by call identity in isolated, forward, and
     reverse runs;
   - reject unless all final states equal the shared pre-state;
   - store the calls as one action and restore the certified final state.
8. Replay the entire serialized row from its stored initial state and compare
   all outputs and state transitions again.

Never treat the stored `initial_api_state`, current target call, current target
output, or future outputs as policy-visible context. Arguments may use values
from earlier tool-result messages, not hidden state.

## Deterministic acceptance checks

Run:

```bash
cd /mnt/shared_ru.ml.SZ-5_000264/gambashidze/tool_synth/APIGen-MT-main

PYTHONPATH=src pytest -q \
  tests/unit/test_refuse_parallel.py \
  tests/unit/test_refuse_parallel_scaling.py \
  tests/unit/test_bfcl_shaped_refusal_parallel_schedule.py
```

For every row, require:

- local full-trajectory replay passed;
- `validate_feature_evaluation_spec(row) == []`;
- `verification_result.overall_verification_passed is True`;
- `generation_metadata.rl_quality_gate_passed is True`;
- correct feature profile and schedule;
- exact scheduled step, turn, and parallel-width shape;
- unique semantic trajectory signature;
- no duplicate normalized user conversation;
- no target call or result leakage;
- safe `execution_context` markers;
- no synthetic refusal calls/results in BFCL-native history;
- exact ordered matching for ordinary actions;
- unordered-multiset matching only for the one certified parallel action.

Audit the merged dataset:

```bash
PYTHONPATH=src python scripts/audit_refuse_parallel_dataset.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.jsonl \
  --report data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.audit.json \
  --expected-rows 100 \
  --require-feature \
  --expected-difficulty hard \
  --min-steps 7 \
  --max-steps 15
```

Also audit the four profile files:

```bash
PYTHONPATH=src python scripts/audit_refuse_parallel_dataset.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/refusal_interactive_32.jsonl \
  --expected-rows 32 --require-feature --expected-feature refusal \
  --expected-difficulty hard --expected-schedule interactive-refusal \
  --require-recovery --min-steps 7 --max-steps 15

PYTHONPATH=src python scripts/audit_refuse_parallel_dataset.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/refusal_unsupported_8.jsonl \
  --expected-rows 8 --require-feature --expected-feature refusal \
  --expected-difficulty hard --expected-schedule terminal \
  --min-steps 7 --max-steps 15

PYTHONPATH=src python scripts/audit_refuse_parallel_dataset.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/parallel_40.jsonl \
  --expected-rows 40 --require-feature --expected-feature parallel \
  --expected-difficulty hard --expected-schedule terminal \
  --min-steps 7 --max-steps 15

PYTHONPATH=src python scripts/audit_refuse_parallel_dataset.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/combined_20.jsonl \
  --expected-rows 20 --require-feature \
  --expected-difficulty hard --expected-schedule combined \
  --require-recovery --min-steps 7 --max-steps 15
```

Do not pass `--require-naturalized`: no external naturalizer is being used.

## Independent Claude judge rubrics

Every review JSONL must have one verdict per source row:

```json
{
  "task_id": "fixed-rp-000",
  "passed": true,
  "error_codes": [],
  "notes": "Concise evidence-based judgment",
  "reviewer": "refusal|parallel|naturalness-diversity"
}
```

### Refusal judge

Check every refusal-containing row from the exact policy-visible context:

- correct reason classification;
- relevant capability truly absent for unsupported cases;
- exactly one missing field for missing-argument cases;
- at least two material choices for ambiguity cases;
- no possible safe real call at the blocked transition;
- question/response targets are specific and grounded;
- immediate recovery fully resolves the blocker without adding unrelated facts;
- recovered call is correct and executable.

### Parallel judge

Check every parallel-containing row:

- all calls are actually requested;
- every argument is visible before the batch;
- no sibling dependency;
- one assistant action contains the entire call multiset;
- no ordering language leaks the label;
- forward/reverse/isolated replay certificates are backed by actual execution;
- surrounding transitions remain ordered;
- grouping is useful and natural, not merely multiple calls forced together.

### Naturalness and diversity judge

Read the entire 100-row dataset, not isolated feature turns:

- conversations sound human and coherent;
- queries do not look like tool plans;
- user does not unnecessarily supply internal IDs;
- assistant responses use only visible facts and real outputs;
- tasks have meaningful ordered dependencies and are suitable for RL;
- no template families or semantic near-duplicates;
- profile, length, width, category, tool, and dependency distributions match the
  schedule;
- dataset is not dominated by arithmetic, simple lookup, or one domain.

Any failed judgment blocks the row. Repair it, replay it, regenerate the
evaluation spec, and re-run all three judge tracks for that row.

## Exports, HTML, and manifest

Export both target representations:

```bash
PYTHONPATH=src python scripts/export_refuse_parallel_tasks.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.jsonl \
  --output data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.internal.tasks.jsonl \
  --target-format internal

PYTHONPATH=src python scripts/export_refuse_parallel_tasks.py \
  --input data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.jsonl \
  --output data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.bfcl_native.tasks.jsonl \
  --target-format bfcl-native
```

Each export must contain exactly 120 feature-transition tasks.

Render all 100 full rows into a self-contained HTML:

```bash
python scripts/render_refuse_parallel_html.py \
  --input fixed100=data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.jsonl \
  --output data/generated/fixed_refuse_parallel_100_20260729/fixed_refuse_parallel_100.review.html \
  --title "100 fixed refusal and parallel trajectories"
```

`manifest.json` must record:

- source row count;
- feature-transition/export counts;
- exact profile counts;
- step, turn, parallel-width, category, and tool distributions;
- refusal-reason counts;
- same-tool versus heterogeneous-tool parallel-group counts;
- semantic uniqueness count;
- all artifact paths;
- evaluation-spec version;
- deterministic seed;
- `external_llm_api_used: false`;
- unit-test, replay, audit, and three judge pass status;
- any rows repaired and why.

## Definition of done

Do not report completion until:

- all five author shards contain exactly 20 rows;
- the merged file contains exactly 100 unique full trajectories;
- all quotas match exactly;
- every tool output and state transition passes local replay;
- all evaluation specs are current and leak-free;
- the deterministic audits pass;
- all three Claude judge files contain 100 passing verdicts;
- both 120-task exports exist;
- the self-contained HTML displays all 100 rows;
- `manifest.json` proves the above.

In the final response, give the absolute paths to the merged JSONL, both task
exports, audit report, HTML, manifest, and any helper code added.
