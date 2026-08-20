#!/usr/bin/env python3
"""Gold-leak-free interactive pass@k evaluator for APIGen trajectories.

Version 3 differs from the legacy exact-next-step evaluator in four important
ways:

* certified parallel actions are exact unordered multisets;
* consecutive independent read-only actions are scheduled as a dependency
  aware ready set, so harmless batching and reordering are accepted;
* state-changing actions remain ordered barriers;
* the synthetic dataset-internal ``refuse`` tool is hidden and refusals or
  clarifications are judged as natural assistant responses.

The policy never sees the initial simulator state, future turns, future tool
outputs, scheduler mode, or gold values.  Only exact ready gold calls are
executed, by replaying their recorded outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import check_apigen_trajectories_passk as legacy


PROTOCOL_VERSION = "apigen-semantic-reasoning-passk-v8"
DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent
    / "prompts/reasoning_next_action_system_v4.txt"
)
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

# The training projection removes these simulator-only input properties from
# both its schemas and targets.  Keep the evaluator on that exact surface.
REMOVED_INPUT_PROPERTIES: dict[str, frozenset[str]] = {
    "book_flight": frozenset({"travel_cost"}),
    "delete_message": frozenset({"message_id"}),
}

GORILLA_TO_OPENAI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

REFUSAL_REASONS = {
    "missing_argument",
    "missing_function",
    "ambiguity",
    "no_appropriate_function",
}


@dataclass(frozen=True)
class ScheduleSegment:
    """One ordered scheduler segment inside one user turn."""

    segment_index: int
    turn_index: int
    mode: str
    step_indices: tuple[int, ...]
    dependencies: tuple[tuple[int, tuple[int, ...]], ...] = ()

    def dependency_map(self) -> dict[int, set[int]]:
        return {
            step_index: set(dependencies)
            for step_index, dependencies in self.dependencies
        }


@dataclass(frozen=True)
class SchedulerTarget:
    segment: ScheduleSegment
    ready_step_indices: tuple[int, ...]


@dataclass(frozen=True)
class RefusalDecision:
    passed: bool
    reason: str
    raw: str = ""
    error: str = ""


def _tool_name(schema: Mapping[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def _real_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool for tool in tools if _tool_name(tool) != "refuse"]


def _canonical_schema(value: Any, *, nested_item: bool = False) -> Any:
    """Mirror the exact BFCL/OpenAI input-schema projection used for SFT."""

    if isinstance(value, list):
        return [
            _canonical_schema(item, nested_item=nested_item) for item in value
        ]
    if not isinstance(value, Mapping):
        return value
    result = {
        key: _canonical_schema(child, nested_item=(key == "items"))
        for key, child in value.items()
    }
    raw_type = result.get("type")
    if isinstance(raw_type, str):
        result["type"] = GORILLA_TO_OPENAI.get(raw_type, "string")
        if raw_type == "float" and not nested_item:
            result["format"] = "float"
            description = result.get("description", "")
            if not isinstance(description, str):
                description = str(description)
            suffix = " This is a float type value."
            if not description.endswith(suffix):
                result["description"] = description + suffix
    return result


def _canonical_policy_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    raw = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
    name = str(raw.get("name") or "")
    description = raw.get("description")
    parameters = raw.get("parameters")
    if not name or not isinstance(description, str) or not isinstance(parameters, Mapping):
        raise ValueError(f"Invalid policy tool schema for {name!r}")
    projected = _canonical_schema(parameters)
    projected["type"] = "object"
    properties = projected.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"Tool {name!r} parameters have no properties object")
    removed = REMOVED_INPUT_PROPERTIES.get(name, frozenset())
    for key in removed:
        properties.pop(key, None)
    required = projected.get("required")
    if isinstance(required, list) and removed:
        projected["required"] = [key for key in required if key not in removed]
    return {
        "type": "function",
        "function": {
            "name": re.sub(r"\.", "_", name),
            "description": description,
            "parameters": projected,
        },
    }


def _canonical_policy_tools(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _canonical_policy_tool(tool)
        for tool in tools
        if _tool_name(tool) != "refuse"
    ]


def _normalise_target_arguments(name: str, arguments: Any) -> dict[str, Any]:
    result = dict(arguments) if isinstance(arguments, Mapping) else {}
    for key in REMOVED_INPUT_PROPERTIES.get(name, frozenset()):
        result.pop(key, None)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_projection_manifest(jsonl_path: str | Path) -> dict[str, Any]:
    """Fail closed before trusting compact next-action annotations."""

    path = Path(jsonl_path).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Next-action manifest is required: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = _sha256(path)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError(
            "Dataset SHA does not match its next-action manifest: "
            f"{actual_sha}"
        )
    contract = manifest.get("training_contract") or {}
    required_contract = {
        "supervision": "next_action_group_and_terminal_stop",
        "golden_history": True,
        "parallel_group_is_one_target": True,
        "terminal_stop_is_separate_target": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Dataset manifest does not declare the required evaluation "
            f"contract: {mismatches}"
        )
    return {
        "path": str(manifest_path),
        "dataset_sha256": actual_sha,
        "rows": int(manifest.get("rows") or 0),
        "statistics": manifest.get("statistics") or {},
        "argument_visibility": manifest.get("argument_visibility") or {},
        "training_contract": contract,
    }


def _raw_step(task: legacy.Task, gold_step: Mapping[str, Any]) -> dict[str, Any]:
    if gold_step.get("projected_transition"):
        turn = task.user_turns[int(gold_step["turn_index"])]
        return {
            "quality_verification": {
                "native_response": str(turn.get("assistant_response") or ""),
            }
        }
    turn = task.user_turns[int(gold_step["turn_index"])]
    steps = turn.get("steps") or []
    step_index = int(gold_step["step_index"])
    return steps[step_index] if 0 <= step_index < len(steps) else {}


def _projected_parallel_has_sibling_dependency(
    task: legacy.Task,
    gold_step: Mapping[str, Any],
) -> bool:
    """Reject a projected parallel group if one sibling consumes another.

    The next-action projection deliberately keeps only the compact
    deterministic certificate.  Reconstruct the important safety property
    from the recorded calls: a value produced by one sibling cannot be used by
    another unless it was already literal in the current user request.
    """

    calls = list(gold_step.get("calls") or [])
    turn = task.user_turns[int(gold_step["turn_index"])]
    visible_text = str(turn.get("query") or "")
    produced: list[set[tuple[str, str]]] = []
    for call in calls:
        values = _value_fingerprints(call.get("output"))
        values.difference_update(
            _value_fingerprints(call.get("arguments") or {})
        )
        produced.append(values)
    for consumer_index, call in enumerate(calls):
        arguments = _value_fingerprints(call.get("arguments") or {})
        hidden_arguments = {
            value
            for value in arguments
            if not _scalar_visible_in_text(value, visible_text)
        }
        for producer_index, values in enumerate(produced):
            if producer_index != consumer_index and hidden_arguments & values:
                return True
    return False


def _projected_parallel_is_certified(
    task: legacy.Task,
    gold_step: Mapping[str, Any],
) -> bool:
    """Validate the compact parallel proof retained by the SFT projection."""

    raw_step = _raw_step(task, gold_step)
    calls = list(gold_step.get("calls") or [])
    quality = raw_step.get("quality_verification") or {}
    return bool(
        len(calls) > 1
        and raw_step.get("execution_mode") == "parallel"
        and raw_step.get("call_order_matters") is False
        and quality.get("passed") is True
        and raw_step.get("pre_state") is not None
        and raw_step.get("pre_state") == raw_step.get("post_state")
        and all(call.get("output") is not None for call in calls)
        and not _projected_parallel_has_sibling_dependency(task, gold_step)
    )


def prepare_next_action_tasks(
    tasks: Sequence[legacy.Task],
    *,
    trust_projected_parallel: bool = False,
) -> dict[str, int]:
    """Compile the exact supervised next-action contract for evaluation.

    Leading ``sft_supervision:false`` turns are golden input context, not
    decisions to score.  A supervised no-tool turn contributes one empty
    assistant target.  Every supervised actionable turn contributes its
    recorded action groups followed by one empty terminal target.
    """

    counts: Counter[str] = Counter()
    for task in tasks:
        by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for step in task.gold_steps:
            by_turn[int(step["turn_index"])].append(step)

        expanded: list[dict[str, Any]] = []
        evaluated_calls: list[dict[str, Any]] = []
        evaluation_start_turn: int | None = None
        supervised_seen = False
        for turn_index, turn in enumerate(task.user_turns):
            turn_steps = sorted(
                by_turn.get(turn_index, []),
                key=lambda step: int(step.get("step_index", -1)),
            )
            supervised = turn.get("sft_supervision") is not False
            if not supervised:
                if supervised_seen:
                    raise ValueError(
                        f"Row {task.position} has an unsupervised turn after "
                        "evaluation already started"
                    )
                counts["golden_prefix_turns"] += 1
                counts["golden_prefix_actions"] += len(turn_steps)
                counts["golden_prefix_calls"] += sum(
                    len(step.get("calls") or []) for step in turn_steps
                )
                continue

            supervised_seen = True
            if evaluation_start_turn is None:
                evaluation_start_turn = turn_index
            counts["supervised_turns"] += 1
            no_tool_target = turn.get("no_tool_target") is True
            if no_tool_target:
                if turn_steps:
                    raise ValueError(
                        f"Row {task.position} turn {turn_index} has both "
                        "no_tool_target and executable steps"
                    )
                reason = str(turn.get("no_tool_reason") or "")
                certificate = turn.get("no_tool_certificate") or {}
                if reason not in REFUSAL_REASONS:
                    raise ValueError(
                        f"Row {task.position} turn {turn_index} has unknown "
                        f"no-tool reason {reason!r}"
                    )
                if certificate.get("passed") is not True:
                    raise ValueError(
                        f"Row {task.position} turn {turn_index} has no valid "
                        "no-tool certificate"
                    )
                expanded.append(
                    {
                        "turn_index": turn_index,
                        "step_index": -1,
                        "calls": [],
                        "call_indices": [],
                        "execution_mode": "no_tool_stop",
                        "call_order_matters": True,
                        "parallel_certified": False,
                        "refusal_certified": False,
                        "refusal_reason": reason,
                        "projected_transition": "no_tool_target",
                    }
                )
                counts["projected_no_tool_targets"] += 1
                counts[f"projected_no_tool_reason:{reason}"] += 1
                continue

            for step in turn_steps:
                raw_step = _raw_step(task, step)
                calls = [
                    {
                        **call,
                        "arguments": _normalise_target_arguments(
                            str(call.get("name") or ""),
                            call.get("arguments"),
                        ),
                    }
                    for call in step.get("calls") or []
                ]
                if not calls:
                    continue
                if len(calls) > 1:
                    projected_parallel = bool(
                        raw_step.get("execution_mode") == "parallel"
                        and raw_step.get("call_order_matters") is False
                    )
                    if not trust_projected_parallel or not projected_parallel:
                        raise ValueError(
                            f"Row {task.position} turn {turn_index} step "
                            f"{step.get('step_index')} has an uncertified "
                            "multi-call target"
                        )
                    execution_mode = "parallel"
                    call_order_matters = False
                    parallel_certified = True
                    counts["recertified_parallel_steps"] += 1
                else:
                    execution_mode = "sequential"
                    call_order_matters = True
                    parallel_certified = False
                call_indices = list(
                    range(
                        len(evaluated_calls),
                        len(evaluated_calls) + len(calls),
                    )
                )
                evaluated_calls.extend(calls)
                expanded.append(
                    {
                        **step,
                        "calls": calls,
                        "call_indices": call_indices,
                        "execution_mode": execution_mode,
                        "call_order_matters": call_order_matters,
                        "parallel_certified": parallel_certified,
                        "refusal_certified": False,
                        "refusal_reason": None,
                    }
                )
                counts["evaluated_action_targets"] += 1
                counts["evaluated_calls"] += len(calls)

            # A complete interactive episode must prove that the policy stops
            # after finishing the current request.  This is the same terminal
            # decision supervised by the next-action SFT view.
            expanded.append(
                {
                    "turn_index": turn_index,
                    "step_index": len(turn.get("steps") or []),
                    "calls": [],
                    "call_indices": [],
                    "execution_mode": "terminal_stop",
                    "call_order_matters": True,
                    "parallel_certified": False,
                    "refusal_certified": False,
                    "refusal_reason": None,
                    "projected_transition": "terminal_stop",
                }
            )
            counts["projected_terminal_stops"] += 1

        if evaluation_start_turn is None:
            raise ValueError(f"Row {task.position} has no supervised turn")
        task.gold_steps = expanded
        task.gold_calls = evaluated_calls
        task.query = str(task.user_turns[evaluation_start_turn].get("query") or "")
        task.evaluation_start_turn = evaluation_start_turn
        task.data_issues = [
            issue
            for issue in task.data_issues
            if issue
            not in {
                "no_gold_calls",
                "uncertified_multi_call_step",
                "order_invariant_non_parallel_step",
            }
        ]
        counts["tasks"] += 1
        counts["evaluated_decisions"] += len(expanded)
    return dict(counts)


def tools_for_turn(
    task: legacy.Task,
    turn_index: int,
) -> list[dict[str, Any]]:
    """Return the exact catalog visible on the current user turn."""

    turn = task.user_turns[turn_index]
    override = turn.get("available_tools")
    if override is None:
        return _canonical_policy_tools(task.tools)
    if not isinstance(override, list):
        raise ValueError(
            f"Row {task.position} turn {turn_index} available_tools is not a list"
        )
    schemas = legacy._declared_tool_schemas({"available_tools": override})
    if len(schemas) != len(override):
        by_name = {_tool_name(tool): tool for tool in task.tools}
        schemas = [
            by_name[name]
            for name in override
            if isinstance(name, str) and name in by_name
        ]
    if len(schemas) != len(override):
        raise ValueError(
            f"Row {task.position} turn {turn_index} has unresolved tool snapshot"
        )
    tools = _canonical_policy_tools(schemas)
    names = [_tool_name(tool) for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError(
            f"Row {task.position} turn {turn_index} has duplicate tool names"
        )
    return tools


def _is_state_preserving_single_read(
    task: legacy.Task, gold_step: Mapping[str, Any]
) -> bool:
    if gold_step.get("execution_mode") in {
        "parallel",
        "refusal",
        "uncertified_parallel",
        "uncertified_refusal",
    }:
        return False
    if len(gold_step.get("calls") or []) != 1:
        return False
    raw_step = _raw_step(task, gold_step)
    pre_state = raw_step.get("pre_state")
    post_state = raw_step.get("post_state")
    return (
        pre_state is not None
        and post_state is not None
        and pre_state == post_state
    )


def _json_fingerprint(value: Any) -> tuple[str, str] | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        value = round(value, 8)
    if isinstance(value, str):
        value = " ".join(value.strip().casefold().split())
        if not value:
            return None
    return (
        type(value).__name__,
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
    )


def _value_fingerprints(value: Any) -> set[tuple[str, str]]:
    """Return leaf fingerprints usable for conservative provenance edges."""

    result: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            result.update(_value_fingerprints(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(_value_fingerprints(child))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"[", "{"}:
            try:
                result.update(_value_fingerprints(json.loads(stripped)))
            except json.JSONDecodeError:
                pass
        fingerprint = _json_fingerprint(value)
        if fingerprint is not None:
            result.add(fingerprint)
    else:
        fingerprint = _json_fingerprint(value)
        if fingerprint is not None:
            result.add(fingerprint)
    return result


def _scalar_visible_in_text(fingerprint: tuple[str, str], text: str) -> bool:
    kind, encoded = fingerprint
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        return False
    haystack = text.casefold()
    if kind == "str":
        needle = str(value).casefold()
        return len(needle) >= 2 and needle in haystack
    if kind in {"int", "float"}:
        return re.search(
            rf"(?<!\w){re.escape(str(value))}(?!\w)",
            haystack,
        ) is not None
    return False


def _infer_read_dependencies(
    task: legacy.Task, step_indices: Sequence[int]
) -> dict[int, set[int]]:
    """Infer conservative output dependencies inside one read-only segment.

    An edge is added when a later gold argument contains a value produced by an
    earlier call and that value was not already literal in the current user
    request. False-positive edges only serialize an otherwise safe pair; they
    never permit an unsafe reordering.
    """

    dependencies = {step_index: set() for step_index in step_indices}
    if not step_indices:
        return dependencies
    turn_index = int(task.gold_steps[step_indices[0]]["turn_index"])
    visible_text = str(task.user_turns[turn_index].get("query") or "")
    output_fingerprints: dict[int, set[tuple[str, str]]] = {}
    for step_index in step_indices:
        output_fingerprints[step_index] = set()
        for call in task.gold_steps[step_index]["calls"]:
            output_fingerprints[step_index].update(
                _value_fingerprints(call.get("output"))
            )
            # Many local tools echo their inputs in the result. An echoed
            # value was already known before the producer ran and therefore
            # cannot create a real sibling dependency.
            output_fingerprints[step_index].difference_update(
                _value_fingerprints(call.get("arguments") or {})
            )

    for offset, step_index in enumerate(step_indices):
        argument_fingerprints: set[tuple[str, str]] = set()
        for call in task.gold_steps[step_index]["calls"]:
            argument_fingerprints.update(
                _value_fingerprints(call.get("arguments") or {})
            )
        hidden_argument_values = {
            fingerprint
            for fingerprint in argument_fingerprints
            if not _scalar_visible_in_text(fingerprint, visible_text)
        }
        for earlier_index in step_indices[:offset]:
            if hidden_argument_values & output_fingerprints[earlier_index]:
                dependencies[step_index].add(earlier_index)
    return dependencies


def build_schedule(task: legacy.Task) -> tuple[ScheduleSegment, ...]:
    """Compile the recorded episode into ordered barrier/ready-set segments."""

    by_turn: dict[int, list[int]] = defaultdict(list)
    for step_index, step in enumerate(task.gold_steps):
        by_turn[int(step["turn_index"])].append(step_index)

    segments: list[ScheduleSegment] = []
    for turn_index in sorted(by_turn):
        indices = by_turn[turn_index]
        cursor = 0
        while cursor < len(indices):
            step_index = indices[cursor]
            step = task.gold_steps[step_index]
            if step.get("execution_mode") == "no_tool_stop":
                mode = "no_tool_stop"
                group = [step_index]
            elif step.get("execution_mode") == "terminal_stop":
                mode = "terminal_stop"
                group = [step_index]
            elif step.get("execution_mode") == "parallel":
                mode = "certified_parallel"
                group = [step_index]
            elif _is_state_preserving_single_read(task, step):
                # Consecutive state-preserving reads may be executed in any
                # dependency-respecting subset/order.  This accepts semantic
                # schedules that differ only from the generator's arbitrary
                # step grouping while retaining exact tool/argument matching.
                mode = "ready_read_set"
                group = []
                while cursor < len(indices):
                    candidate = indices[cursor]
                    if not _is_state_preserving_single_read(
                        task, task.gold_steps[candidate]
                    ):
                        break
                    group.append(candidate)
                    cursor += 1
                dependencies = _infer_read_dependencies(task, group)
                segments.append(
                    ScheduleSegment(
                        segment_index=len(segments),
                        turn_index=turn_index,
                        mode=mode,
                        step_indices=tuple(group),
                        dependencies=tuple(
                            (index, tuple(sorted(dependencies[index])))
                            for index in group
                        ),
                    )
                )
                continue
            else:
                mode = "ordered_barrier"
                group = [step_index]

            segments.append(
                ScheduleSegment(
                    segment_index=len(segments),
                    turn_index=turn_index,
                    mode=mode,
                    step_indices=tuple(group),
                )
            )
            cursor += 1
    return tuple(segments)


def scheduler_target(
    task: legacy.Task,
    matched_step_indices: set[int],
    schedule: Sequence[ScheduleSegment] | None = None,
) -> SchedulerTarget | None:
    schedule = schedule or build_schedule(task)
    for segment in schedule:
        unmatched = [
            index
            for index in segment.step_indices
            if index not in matched_step_indices
        ]
        if not unmatched:
            continue
        if segment.mode != "ready_read_set":
            return SchedulerTarget(segment, (unmatched[0],))
        dependencies = segment.dependency_map()
        ready = tuple(
            index
            for index in unmatched
            if dependencies.get(index, set()) <= matched_step_indices
        )
        if not ready:
            raise ValueError(
                f"No ready node in segment {segment.segment_index}; "
                "dependency cycle or invalid replay state"
            )
        return SchedulerTarget(segment, ready)
    return None


SEMANTIC_TEXT_FIELDS = {
    ("comment", "comment_content"),
    ("create_ticket", "description"),
    ("create_ticket", "title"),
    ("post_tweet", "content"),
    ("resolve_ticket", "resolution"),
    ("send_message", "message"),
}


def _schema_for_tool(
    task: legacy.Task | None, name: str
) -> Mapping[str, Any]:
    if task is None:
        return {}
    for tool in task.tools:
        function = tool.get("function") or {}
        if function.get("name") == name:
            return function.get("parameters") or {}
    return {}


def _arguments_with_defaults(
    task: legacy.Task | None,
    name: str,
    arguments: Any,
) -> dict[str, Any]:
    result = dict(arguments) if isinstance(arguments, Mapping) else {}
    properties = (_schema_for_tool(task, name).get("properties") or {})
    for key, schema in properties.items():
        if key not in result and isinstance(schema, Mapping) and "default" in schema:
            result[key] = schema["default"]
    return result


def _argument_leaf_fingerprints(value: Any) -> set[tuple[str, str]]:
    """Return factual leaves whose provenance an argument requires.

    Object keys come from the tool schema, so only values need provenance.
    JSON-encoded strings are treated as their decoded structure when possible.
    Empty containers, booleans, and null contain no hidden factual literal.
    """

    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, Mapping):
        result: set[tuple[str, str]] = set()
        for child in value.values():
            result.update(_argument_leaf_fingerprints(child))
        return result
    if isinstance(value, (list, tuple)):
        result = set()
        for child in value:
            result.update(_argument_leaf_fingerprints(child))
        return result
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"[", "{"}:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(decoded, (Mapping, list, tuple)):
                    return _argument_leaf_fingerprints(decoded)
        # Tool outputs often serialise numeric IDs as strings while a later
        # schema expects an integer. Treat lossless numeric spellings as the
        # same visible value, but preserve leading-zero identifiers as text.
        if re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", stripped):
            numeric: int | float
            numeric = float(stripped) if "." in stripped else int(stripped)
            if isinstance(numeric, float) and numeric.is_integer():
                numeric = int(numeric)
            fingerprint = _json_fingerprint(numeric)
            return {fingerprint} if fingerprint is not None else set()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    fingerprint = _json_fingerprint(value)
    return {fingerprint} if fingerprint is not None else set()


def _schema_declares_argument_value(
    argument_schema: Mapping[str, Any], value: Any
) -> bool:
    """Whether a value is available directly from the public tool schema."""

    def same(left: Any, right: Any) -> bool:
        return json.dumps(
            left, ensure_ascii=False, sort_keys=True, default=str
        ) == json.dumps(
            right, ensure_ascii=False, sort_keys=True, default=str
        )

    if "const" in argument_schema and same(value, argument_schema["const"]):
        return True
    if "default" in argument_schema and same(value, argument_schema["default"]):
        return True
    enum = argument_schema.get("enum")
    if isinstance(enum, list) and any(same(value, item) for item in enum):
        return True

    # BFCL schemas frequently encode enums and symbolic modes only in the
    # human-readable description (for example "[Enum]: START, STOP" or
    # "'w' for words"). Those literals are policy-visible too.
    # Search only the human-facing description.  Looking through the entire
    # schema would incorrectly treat metadata such as ``type: string`` or a
    # numeric bound as an explicitly supplied argument value.
    description = argument_schema.get("description")
    if not isinstance(description, str):
        return False
    schema_text = description.casefold()
    leaves = _argument_leaf_fingerprints(value)
    for kind, encoded in leaves:
        try:
            leaf = json.loads(encoded)
        except json.JSONDecodeError:
            return False
        if kind == "str":
            if re.search(
                rf"(?<!\w){re.escape(str(leaf).casefold())}(?!\w)",
                schema_text,
            ) is None:
                return False
        elif kind in {"int", "float"}:
            if re.search(
                rf"(?<![\w.]){re.escape(str(leaf))}(?![\w.])",
                schema_text,
            ) is None:
                return False
        else:
            return False
    return bool(leaves)


def _relative_date_fingerprints(
    text: str, current_date: str
) -> set[tuple[str, str]]:
    """Resolve unambiguous relative dates exposed by the system contract."""

    try:
        anchor = date.fromisoformat(current_date)
    except ValueError:
        return set()
    lowered = text.casefold()
    values: list[str] = []
    if re.search(r"\btoday\b", lowered):
        values.append(anchor.isoformat())
    if re.search(r"\btomorrow\b", lowered):
        values.append((anchor + timedelta(days=1)).isoformat())
    if re.search(r"\byesterday\b", lowered):
        values.append((anchor - timedelta(days=1)).isoformat())
    return {
        fingerprint
        for value in values
        if (fingerprint := _json_fingerprint(value)) is not None
    }


def _argument_visibility_validation(
    tasks: Sequence[legacy.Task],
    *,
    current_date: str,
    include_initial_state: bool = False,
    max_examples: int = 100,
) -> dict[str, Any]:
    """Estimate how many stored gold arguments require a hidden value.

    The audit follows the same reveal order as the policy.  A top-level real
    tool argument is visible when all of its factual leaves come from visible
    user/history text, the system date, a schema declaration, an accepted prior
    call, or a prior returned tool output.  Initial simulator state and sibling
    parallel outputs are unavailable unless explicitly configured otherwise.

    This is deliberately conservative and lexical.  It is an upper-bound
    estimate: a semantically inferable paraphrase can still be marked hidden.
    """

    total_arguments = 0
    hidden_arguments = 0
    required_arguments = 0
    hidden_required_arguments = 0
    hidden_leaf_count = 0
    excluded_synthetic_arguments = 0
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_hidden_tasks: dict[str, set[int]] = defaultdict(set)
    visibility_sources: Counter[str] = Counter()
    hidden_by_tool: Counter[str] = Counter()
    hidden_examples: list[dict[str, Any]] = []
    tasks_with_hidden: set[int] = set()

    system_fingerprints = _argument_leaf_fingerprints(current_date)

    for task in tasks:
        source = str(
            (task.raw.get("aggregation_metadata") or {}).get("source_dataset")
            or "unknown"
        )
        visible_user_text: list[str] = []
        visible_assistant_text: list[str] = []
        visible_structured: set[tuple[str, str]] = set()
        derived_system_values: set[tuple[str, str]] = set(system_fingerprints)
        if include_initial_state:
            visible_structured.update(
                _argument_leaf_fingerprints(task.initial_state)
            )

        steps_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for step in task.gold_steps:
            steps_by_turn[int(step["turn_index"])].append(step)

        for turn_index, turn in enumerate(task.user_turns):
            query = str(turn.get("query") or "")
            visible_user_text.append(query)
            derived_system_values.update(
                _relative_date_fingerprints(query, current_date)
            )

            turn_has_refusal = False
            for step in steps_by_turn.get(turn_index, []):
                if step.get("execution_mode") == "refusal":
                    turn_has_refusal = True
                pending_visible: set[tuple[str, str]] = set()
                for call in step.get("calls") or []:
                    tool_name = str(call.get("name") or "")
                    arguments = call.get("arguments") or {}
                    if not isinstance(arguments, Mapping):
                        continue
                    if tool_name == "refuse":
                        excluded_synthetic_arguments += len(arguments)
                        continue

                    parameters = _schema_for_tool(task, tool_name)
                    properties = parameters.get("properties") or {}
                    required = set(parameters.get("required") or [])
                    combined_text = "\n".join(
                        visible_user_text + visible_assistant_text
                    )
                    for argument_name, value in arguments.items():
                        total_arguments += 1
                        source_counts[source]["total_arguments"] += 1
                        is_required = argument_name in required
                        if is_required:
                            required_arguments += 1
                            source_counts[source]["required_arguments"] += 1

                        argument_schema = properties.get(argument_name) or {}
                        leaves = _argument_leaf_fingerprints(value)
                        if _schema_declares_argument_value(
                            argument_schema, value
                        ):
                            visibility = "schema_declared"
                        elif not leaves:
                            visibility = "no_factual_literal"
                        elif all(
                            _scalar_visible_in_text(leaf, combined_text)
                            for leaf in leaves
                        ):
                            visibility = "visible_text"
                        elif all(leaf in visible_structured for leaf in leaves):
                            visibility = "prior_call_or_tool_output"
                        elif all(
                            leaf in derived_system_values for leaf in leaves
                        ):
                            visibility = "system_date"
                        elif all(
                            _scalar_visible_in_text(leaf, combined_text)
                            or leaf in visible_structured
                            or leaf in derived_system_values
                            for leaf in leaves
                        ):
                            visibility = "mixed_policy_visible_sources"
                        else:
                            visibility = "hidden"

                        visibility_sources[visibility] += 1
                        source_counts[source][visibility] += 1
                        if visibility == "hidden":
                            hidden_arguments += 1
                            tasks_with_hidden.add(task.position)
                            source_hidden_tasks[source].add(task.position)
                            hidden_by_tool[tool_name] += 1
                            missing_leaves = [
                                leaf
                                for leaf in leaves
                                if not (
                                    _scalar_visible_in_text(
                                        leaf, combined_text
                                    )
                                    or leaf in visible_structured
                                    or leaf in derived_system_values
                                )
                            ]
                            hidden_leaf_count += len(missing_leaves)
                            if is_required:
                                hidden_required_arguments += 1
                                source_counts[source][
                                    "hidden_required_arguments"
                                ] += 1
                            if len(hidden_examples) < max_examples:
                                hidden_examples.append(
                                    {
                                        "row_position": task.position,
                                        "source_dataset": source,
                                        "turn_index": turn_index,
                                        "step_index": int(
                                            step.get("step_index", -1)
                                        ),
                                        "tool_name": tool_name,
                                        "argument_name": str(argument_name),
                                        "required": is_required,
                                        "missing_leaf_count": len(
                                            missing_leaves
                                        ),
                                    }
                                )

                    # Calls in one stored step are simultaneous. Their arguments
                    # and outputs become visible only after every sibling has
                    # been audited, preventing parallel sibling-output leakage.
                    pending_visible.update(
                        _argument_leaf_fingerprints(arguments)
                    )
                    pending_visible.update(
                        _argument_leaf_fingerprints(call.get("output"))
                    )
                visible_structured.update(pending_visible)

            # This mirrors the current evaluator. Native refusals retain the
            # policy's own text; ordinary turns receive the recorded grounded
            # assistant response before the next user turn is revealed.
            if not turn_has_refusal:
                assistant_response = str(
                    turn.get("assistant_response") or ""
                )
                if assistant_response:
                    visible_assistant_text.append(assistant_response)

    by_source: dict[str, Any] = {}
    for source, counts in sorted(source_counts.items()):
        source_total = counts["total_arguments"]
        source_hidden = counts["hidden"]
        by_source[source] = {
            **dict(counts),
            "hidden": source_hidden,
            "hidden_argument_ratio": (
                source_hidden / source_total if source_total else 0.0
            ),
            "tasks_with_hidden_arguments": len(
                source_hidden_tasks[source]
            ),
        }

    return {
        "method": "conservative_policy_visible_top_level_arguments_v1",
        "definition": (
            "Top-level real-tool gold arguments not grounded in revealed "
            "text, system date, schema declarations, accepted prior calls, "
            "or prior tool outputs; initial state and parallel sibling outputs "
            "are hidden."
        ),
        "initial_state_policy_visible": include_initial_state,
        "total_arguments": total_arguments,
        "hidden_argument_count": hidden_arguments,
        "hidden_argument_ratio": (
            hidden_arguments / total_arguments if total_arguments else 0.0
        ),
        "required_argument_count": required_arguments,
        "hidden_required_argument_count": hidden_required_arguments,
        "hidden_required_argument_ratio": (
            hidden_required_arguments / required_arguments
            if required_arguments
            else 0.0
        ),
        "hidden_leaf_count": hidden_leaf_count,
        "tasks_with_hidden_arguments": len(tasks_with_hidden),
        "excluded_synthetic_refuse_arguments": (
            excluded_synthetic_arguments
        ),
        "visibility_source_counts": dict(visibility_sources),
        "hidden_by_tool": dict(hidden_by_tool.most_common()),
        "by_source": by_source,
        "hidden_examples": hidden_examples,
        "hidden_examples_truncated": hidden_arguments > len(hidden_examples),
    }


def _normalise_semantic_text(name: str, key: str, value: Any) -> Any:
    if (name, key) not in SEMANTIC_TEXT_FIELDS or not isinstance(value, str):
        return value
    return " ".join(
        re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).split()
    )


def _call_key(
    call: Mapping[str, Any],
    task: legacy.Task | None = None,
) -> tuple[str, str]:
    def canonical_numbers(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: canonical_numbers(
                    _normalise_semantic_text(
                        str(call.get("name") or ""), key, child
                    )
                )
                for key, child in sorted(value.items())
            }
        if isinstance(value, list):
            return [canonical_numbers(child) for child in value]
        if isinstance(value, float):
            rounded = round(value, 8)
            return int(rounded) if rounded.is_integer() else rounded
        return value

    name = str(call.get("name") or "")
    arguments = legacy.normalise_arguments(
        _arguments_with_defaults(task, name, call.get("arguments"))
    )
    if name in {"add", "multiply"} and set(arguments) == {"a", "b"}:
        operands = sorted(
            (
                canonical_numbers(arguments["a"]),
                canonical_numbers(arguments["b"]),
            ),
            key=lambda value: json.dumps(
                value, ensure_ascii=False, sort_keys=True, default=str
            ),
        )
        arguments = {"operands": operands}
    return (
        name,
        json.dumps(
            canonical_numbers(arguments),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _pair_calls(
    task: legacy.Task,
    predicted: Sequence[dict[str, Any]],
    candidates: Sequence[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]] | None:
    remaining = list(candidates)
    paired: list[tuple[int, dict[str, Any]]] = []
    for predicted_call in predicted:
        key = _call_key(predicted_call, task)
        match_position = next(
            (
                position
                for position, (_, gold_call) in enumerate(remaining)
                if _call_key(gold_call, task) == key
            ),
            None,
        )
        if match_position is None:
            return None
        paired.append(remaining.pop(match_position))
    return paired


def match_scheduler_calls(
    task: legacy.Task,
    target: SchedulerTarget,
    predicted: Sequence[dict[str, Any]],
) -> tuple[list[int], list[int], list[dict[str, Any]]] | None:
    """Match one assistant action and preserve predicted-call output pairing."""

    mode = target.segment.mode
    if mode == "native_refusal":
        return None

    if mode == "certified_parallel":
        step_index = target.ready_step_indices[0]
        step = task.gold_steps[step_index]
        if len(predicted) != len(step["calls"]):
            return None
        paired = _pair_calls(
            task,
            predicted,
            list(zip(step["call_indices"], step["calls"])),
        )
        if paired is None:
            return None
        return (
            [step_index],
            [call_index for call_index, _ in paired],
            [call for _, call in paired],
        )

    if mode == "ordered_barrier":
        step_index = target.ready_step_indices[0]
        step = task.gold_steps[step_index]
        gold_calls = step["calls"]
        if len(predicted) != len(gold_calls) or not all(
            _call_key(predicted_call, task) == _call_key(gold_call, task)
            for predicted_call, gold_call in zip(predicted, gold_calls)
        ):
            return None
        return (
            [step_index],
            list(step["call_indices"]),
            list(gold_calls),
        )

    if not predicted:
        return None
    candidates = [
        (
            step_index,
            task.gold_steps[step_index]["call_indices"][0],
            task.gold_steps[step_index]["calls"][0],
        )
        for step_index in target.ready_step_indices
    ]
    remaining = list(candidates)
    matched: list[tuple[int, int, dict[str, Any]]] = []
    for predicted_call in predicted:
        key = _call_key(predicted_call, task)
        match_position = next(
            (
                position
                for position, (_, _, gold_call) in enumerate(remaining)
                if _call_key(gold_call, task) == key
            ),
            None,
        )
        if match_position is None:
            return None
        matched.append(remaining.pop(match_position))
    return (
        [step_index for step_index, _, _ in matched],
        [call_index for _, call_index, _ in matched],
        [call for _, _, call in matched],
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


class SemanticRefusalJudge:
    """Independent semantic judge for native clarification/refusal responses."""

    def __init__(
        self,
        client: legacy.VLLMClient,
        *,
        model: str,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str, str, str], RefusalDecision] = {}

    def judge(
        self,
        *,
        query: str,
        reason: str,
        response: str,
        reference: str,
        available_tools: Sequence[Mapping[str, Any]] = (),
    ) -> RefusalDecision:
        tool_context = [
            {
                "name": _tool_name(tool),
                "description": str(
                    (tool.get("function") or {}).get("description") or ""
                ),
            }
            for tool in available_tools
        ]
        tool_context_json = json.dumps(
            tool_context, ensure_ascii=False, sort_keys=True
        )
        key = (query, reason, response, reference, tool_context_json)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        if reason not in REFUSAL_REASONS:
            decision = RefusalDecision(False, "unknown_refusal_reason")
        elif not response.strip():
            decision = RefusalDecision(False, "empty_response")
        else:
            rubric = {
                "missing_argument": (
                    "It asks a targeted question for the specific material "
                    "missing value, does not invent it, and does not claim the "
                    "operation happened. It may reasonably suggest a format or "
                    "ask for slightly more specific detail than the reference."
                ),
                "missing_function": (
                    "No currently available tool implements the exact "
                    "requested operation. It does not call a tempting sibling "
                    "tool as a substitute, invent execution, or claim success. "
                    "It naturally states that the capability is unavailable "
                    "for this turn or that it can continue after tools update."
                ),
                "ambiguity": (
                    "It asks a targeted question that resolves the material "
                    "ambiguity, and neither invents a choice nor claims action."
                ),
                "no_appropriate_function": (
                    "It correctly explains that the requested capability is "
                    "unavailable with the provided tools, and does not claim "
                    "the operation happened."
                ),
            }[reason]
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict binary evaluator of an assistant "
                            "clarification/refusal. Judge semantic correctness, "
                            "not wording similarity. The reference is one valid "
                            "example, not a maximum-detail template: do not fail "
                            "a targeted clarification merely because it requests "
                            "a useful format or extra specificity. Output one JSON object only: "
                            '{"pass":true|false,"reason":"brief reason"}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_request": query,
                                "expected_behavior": reason,
                                "rubric": rubric,
                                "available_tools": tool_context,
                                "reference_response": reference,
                                "candidate_response": response,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": self.max_tokens,
            }
            try:
                raw_response = self.client._request("/chat/completions", payload)
                choices = raw_response.get("choices") or []
                message = (choices[0] or {}).get("message") if choices else {}
                content = str((message or {}).get("content") or "")
                parsed = _extract_json_object(content)
                if parsed is None or not isinstance(parsed.get("pass"), bool):
                    decision = RefusalDecision(
                        False,
                        "judge_unparseable",
                        raw=content,
                        error="Judge did not return a boolean pass field",
                    )
                else:
                    decision = RefusalDecision(
                        bool(parsed["pass"]),
                        str(parsed.get("reason") or ""),
                        raw=content,
                    )
            except Exception as error:
                decision = RefusalDecision(
                    False,
                    "judge_error",
                    error=str(error),
                )

        with self._lock:
            self._cache[key] = decision
        return decision


class LexicalRefusalJudge:
    """Deterministic smoke-test judge; not used for authoritative full runs."""

    def judge(
        self,
        *,
        query: str,
        reason: str,
        response: str,
        reference: str,
        available_tools: Sequence[Mapping[str, Any]] = (),
    ) -> RefusalDecision:
        passed = False
        try:
            from src.refuse_parallel_eval import conservative_native_refusal_match

            passed = conservative_native_refusal_match(response, reason)
        except (ImportError, ValueError):
            text = response.casefold()
            if reason in {"missing_argument", "ambiguity"}:
                passed = "?" in response
            else:
                passed = any(
                    phrase in text
                    for phrase in ("can't", "cannot", "don't have", "not available")
                )
        return RefusalDecision(passed, "lexical_smoke_test")


class InteractivePassKV3Checker:
    """Dependency-aware exact-action replay with native refusal semantics."""

    def __init__(
        self,
        client: legacy.VLLMClient,
        refusal_judge: Any,
        *,
        pass_k: int = 16,
        sampling: legacy.SamplingConfig = legacy.SamplingConfig(max_tokens=8192),
        workers: int = 32,
        include_initial_state: bool = False,
        current_date: str | None = None,
    ) -> None:
        self.client = client
        self.refusal_judge = refusal_judge
        self.pass_k = pass_k
        self.sampling = sampling
        self.workers = max(1, workers)
        self.include_initial_state = include_initial_state
        self.current_date = current_date or date.today().isoformat()
        self._schedules: dict[int, tuple[ScheduleSegment, ...]] = {}

    def _schedule(self, task: legacy.Task) -> tuple[ScheduleSegment, ...]:
        key = id(task)
        if key not in self._schedules:
            self._schedules[key] = build_schedule(task)
        return self._schedules[key]

    def _seed(self, state: legacy.RolloutState) -> int:
        return (
            self.sampling.seed
            + state.task.position * self.pass_k
            + state.sample_index
            + state.next_turn * 1_000_003
        )

    def _initial_messages(
        self, task: legacy.Task, include_initial_state: bool
    ) -> list[dict[str, Any]]:
        start_turn = int(getattr(task, "evaluation_start_turn", 0))
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # Unsupervised leading turns are exactly the golden history rendered by
        # the trainer.  They condition the first scored decision but are not
        # themselves sampled or counted.
        for turn_index, turn in enumerate(task.user_turns[:start_turn]):
            messages.append(
                {"role": "user", "content": str(turn.get("query") or "")}
            )
            turn_number = turn.get("turn_number", 0)
            for step in turn.get("steps") or []:
                raw_calls = step.get("tool_calls") or []
                tool_calls: list[dict[str, Any]] = []
                tool_messages: list[dict[str, Any]] = []
                for call_position, call in enumerate(raw_calls):
                    name = str(
                        call.get("tool_name") or call.get("name") or ""
                    )
                    if not name:
                        continue
                    call_id = (
                        f"call_t{turn_number}_{step.get('step_number', 0)}_"
                        f"{call_position}"
                    )
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    _normalise_target_arguments(
                                        name, call.get("arguments")
                                    ),
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                    output = call.get("output", "")
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": (
                                output
                                if isinstance(output, str)
                                else json.dumps(output, ensure_ascii=False)
                            ),
                        }
                    )
                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": tool_calls,
                        }
                    )
                    messages.extend(tool_messages)
            messages.append({"role": "assistant", "content": ""})

        content = str(task.user_turns[start_turn].get("query") or "")
        if include_initial_state:
            content = (
                "Initial environment context:\n"
                + json.dumps(task.initial_state, ensure_ascii=False, indent=2)
                + "\n\nUser request:\n"
                + content
            )
        messages.append({"role": "user", "content": content})
        return messages

    def build_states(
        self, tasks: Sequence[legacy.Task]
    ) -> list[legacy.RolloutState]:
        states = [
            legacy.RolloutState(
                task=task,
                sample_index=sample_index,
                messages=self._initial_messages(task, self.include_initial_state),
            )
            for task in tasks
            for sample_index in range(self.pass_k)
        ]
        for state in states:
            state.task.tools = _canonical_policy_tools(state.task.tools)
            if not state.task.gold_steps:
                state.status = "failed"
                state.failure = "no_gold_steps"
        return states

    def _mismatch_reason(
        self,
        state: legacy.RolloutState,
        target: SchedulerTarget,
        calls: Sequence[dict[str, Any]],
    ) -> str:
        if any(call.get("name") == "refuse" for call in calls):
            return "synthetic_refuse_call"
        current_turn = target.segment.turn_index
        unmatched_current = [
            step
            for index, step in enumerate(state.task.gold_steps)
            if index not in state.matched_gold_step_indices
            and int(step["turn_index"]) == current_turn
        ]
        future_keys = Counter(
            _call_key(call, state.task)
            for step in unmatched_current
            for call in step["calls"]
        )
        predicted_keys = Counter(
            _call_key(call, state.task) for call in calls
        )
        if predicted_keys and predicted_keys <= future_keys:
            return "unsafe_or_nonready_batch"
        expected_calls = [
            call
            for step_index in target.ready_step_indices
            for call in state.task.gold_steps[step_index]["calls"]
        ]
        expected_names = Counter(call["name"] for call in expected_calls)
        predicted_names = Counter(str(call.get("name") or "") for call in calls)
        if not calls:
            return "no_tool_call"
        if not (predicted_names & expected_names):
            return "wrong_tool"
        if predicted_names != expected_names:
            return "wrong_call_set"
        return "wrong_arguments"

    def _generate_event(self, state: legacy.RolloutState) -> dict[str, Any]:
        target = scheduler_target(
            state.task,
            state.matched_gold_step_indices,
            self._schedule(state.task),
        )
        if target is None:
            raise ValueError("Active rollout has no scheduler target")
        current_tools = tools_for_turn(
            state.task, target.segment.turn_index
        )
        try:
            response = self.client.chat(
                state.messages,
                current_tools,
                self.sampling,
                self._seed(state),
                parallel_tool_calls=True,
            )
            calls, parse_errors, content = legacy.parse_response_calls(
                response, current_tools
            )
            choices = response.get("choices") or []
            choice = choices[0] or {} if choices else {}
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")
            # vLLM 0.23 exposes parsed Qwen thoughts as ``reasoning`` while
            # several other OpenAI-compatible servers use
            # ``reasoning_content``.  Preserve either spelling for RL audits.
            reasoning_content = str(
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            )
            usage = response.get("usage") or {}
        except Exception as error:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "row_position": state.task.position,
                "sample_index": state.sample_index,
                "turn": state.next_turn,
                "gold_user_turn_index": target.segment.turn_index,
                "scheduler_mode": target.segment.mode,
                "ready_step_indices": list(target.ready_step_indices),
                "matched": False,
                "failure": "api_error",
                "error": str(error),
                "parse_errors": 0,
                "predicted_calls": [],
                "raw_completion": "",
                "reasoning_content": "",
                "tool_only_compliant": False,
                "finish_reason": None,
                "usage": {},
                "matched_gold_step_indices": [],
                "matched_gold_indices": [],
                "matched_gold_calls": [],
                "tool_outputs": [],
            }

        refusal_decision: RefusalDecision | None = None
        matched: tuple[list[int], list[int], list[dict[str, Any]]] | None = None
        if parse_errors:
            failure = "parse_error"
        elif target.segment.mode in {"terminal_stop", "no_tool_stop"}:
            step_index = target.ready_step_indices[0]
            if calls:
                failure = (
                    "tool_call_for_no_tool_target"
                    if target.segment.mode == "no_tool_stop"
                    else "tool_call_after_completion"
                )
            elif finish_reason == "length":
                failure = f"truncated_{target.segment.mode}"
            else:
                failure = ""
                matched = ([step_index], [], [])
        else:
            if content.strip():
                failure = "visible_prose_with_tool_call"
            else:
                matched = match_scheduler_calls(
                    state.task,
                    target,
                    calls,
                )
                failure = "" if matched is not None else self._mismatch_reason(
                    state, target, calls
                )

        step_indices, call_indices, gold_calls = (
            matched if matched is not None else ([], [], [])
        )
        original_unmatched = [
            index
            for index, step in enumerate(state.task.gold_steps)
            if index not in state.matched_gold_step_indices
            and int(step["turn_index"]) == target.segment.turn_index
        ]
        strict_compatible = (
            bool(step_indices)
            and len(step_indices) == 1
            and step_indices[0] == original_unmatched[0]
            and len(calls) == len(
                state.task.gold_steps[original_unmatched[0]]["calls"]
            )
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "row_position": state.task.position,
            "sample_index": state.sample_index,
            "turn": state.next_turn,
            "gold_user_turn_index": target.segment.turn_index,
            "segment_index": target.segment.segment_index,
            "scheduler_mode": target.segment.mode,
            "ready_step_indices": list(target.ready_step_indices),
            "matched": not failure,
            "failure": failure,
            "parse_errors": parse_errors,
            "predicted_calls": calls,
            "raw_completion": content,
            "reasoning_content": reasoning_content,
            # BFCL-style function-call correctness treats any response with
            # no decoded call as a stop.  Keep the stricter RL output contract
            # as a separate metric instead of collapsing both notions of
            # success into one pass@k number.
            "tool_only_compliant": not bool(content.strip()),
            "finish_reason": finish_reason,
            "usage": usage,
            "matched_gold_step_indices": step_indices,
            "matched_gold_indices": call_indices,
            "matched_gold_calls": [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in gold_calls
            ],
            "tool_outputs": [call.get("output") for call in gold_calls],
            "available_tool_names": [
                _tool_name(tool) for tool in current_tools
            ],
            "refusal_reason": (
                state.task.gold_steps[target.ready_step_indices[0]].get(
                    "refusal_reason"
                )
                if target.segment.mode == "no_tool_stop"
                else None
            ),
            "refusal_judge": (
                asdict(refusal_decision) if refusal_decision is not None else None
            ),
            "strict_schedule_compatible": strict_compatible,
        }

    @staticmethod
    def apply_event(
        state: legacy.RolloutState, event: Mapping[str, Any]
    ) -> None:
        if state.status != "active":
            raise ValueError("Cannot apply an event to a completed rollout")
        if int(event["turn"]) != state.next_turn:
            raise ValueError("Non-consecutive rollout event")
        state.turns.append(dict(event))
        if not event.get("matched"):
            state.status = "failed"
            state.failure = str(event.get("failure") or "call_mismatch")
            return

        step_indices = [
            int(index) for index in event.get("matched_gold_step_indices") or []
        ]
        call_indices = [
            int(index) for index in event.get("matched_gold_indices") or []
        ]
        if not step_indices:
            raise ValueError("Matched event has no gold step")
        if any(index in state.matched_gold_step_indices for index in step_indices):
            raise ValueError("Gold step matched more than once")
        if any(index in state.matched_gold_indices for index in call_indices):
            raise ValueError("Gold call matched more than once")

        state.matched_gold_step_indices.update(step_indices)
        state.matched_gold_indices.update(call_indices)
        predicted_calls = list(event.get("predicted_calls") or [])
        scheduler_mode = str(event.get("scheduler_mode") or "")
        reasoning_content = str(event.get("reasoning_content") or "")

        if scheduler_mode in {"terminal_stop", "no_tool_stop"}:
            assistant_message = {
                "role": "assistant",
                "content": str(event.get("raw_completion") or ""),
            }
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            state.messages.append(assistant_message)
        else:
            outputs = list(event.get("tool_outputs") or [])
            if len(predicted_calls) != len(outputs):
                raise ValueError("Predicted-call/tool-output count mismatch")
            call_ids = [
                predicted.get("tool_call_id")
                or (
                    f"call_{state.task.position}_{state.sample_index}_"
                    f"{state.next_turn}_{position}"
                )
                for position, predicted in enumerate(predicted_calls, 1)
            ]
            assistant_message = {
                "role": "assistant",
                "content": str(event.get("raw_completion") or ""),
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": predicted["name"],
                            "arguments": json.dumps(
                                predicted["arguments"], ensure_ascii=False
                            ),
                        },
                    }
                    for call_id, predicted in zip(call_ids, predicted_calls)
                ],
            }
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            state.messages.append(assistant_message)
            for call_id, predicted, output in zip(
                call_ids, predicted_calls, outputs
            ):
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": predicted["name"],
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )

        if len(state.matched_gold_step_indices) == len(state.task.gold_steps):
            state.status = "success"
            return

        completed_turn = int(event["gold_user_turn_index"])
        remaining_same_turn = any(
            index not in state.matched_gold_step_indices
            and int(step["turn_index"]) == completed_turn
            for index, step in enumerate(state.task.gold_steps)
        )
        if remaining_same_turn:
            return
        next_turn_index = min(
            int(step["turn_index"])
            for index, step in enumerate(state.task.gold_steps)
            if index not in state.matched_gold_step_indices
        )
        state.messages.append(
            {
                "role": "user",
                "content": state.task.user_turns[next_turn_index]["query"],
            }
        )

    def run(
        self,
        tasks: Sequence[legacy.Task],
        *,
        states: list[legacy.RolloutState] | None = None,
        on_event: Callable[[dict[str, Any], legacy.RolloutState], None] | None = None,
        on_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> list[legacy.RolloutState]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        states = states or self.build_states(tasks)
        while True:
            active = [state for state in states if state.status == "active"]
            if not active:
                break
            interaction_turn = min(state.next_turn for state in active)
            turn_states = [
                state for state in active if state.next_turn == interaction_turn
            ]
            with ThreadPoolExecutor(
                max_workers=min(self.workers, len(turn_states))
            ) as executor:
                futures = {
                    executor.submit(self._generate_event, state): state
                    for state in turn_states
                }
                for processed, future in enumerate(as_completed(futures), 1):
                    state = futures[future]
                    event = future.result()
                    self.apply_event(state, event)
                    if on_event:
                        on_event(event, state)
                    if on_progress:
                        counts = Counter(item.status for item in states)
                        on_progress(
                            {
                                "turn": interaction_turn,
                                "processed": processed,
                                "turn_total": len(turn_states),
                                "active": counts["active"],
                                "success": counts["success"],
                                "failed": counts["failed"],
                                "total": len(states),
                            }
                        )
        return states


def _scheduler_validation(tasks: Sequence[legacy.Task]) -> dict[str, Any]:
    modes: Counter[str] = Counter()
    dependency_edges = 0
    read_segment_sizes: Counter[int] = Counter()
    issues: Counter[str] = Counter()
    for task in tasks:
        for turn_index, turn in enumerate(task.user_turns):
            no_tool_steps = [
                step
                for step in task.gold_steps
                if int(step["turn_index"]) == turn_index
                and step["execution_mode"] == "no_tool_stop"
            ]
            total_steps = sum(
                int(step["turn_index"]) == turn_index
                for step in task.gold_steps
            )
            if no_tool_steps and total_steps != 1:
                issues["no_tool_not_only_transition_in_turn"] += 1
        for segment in build_schedule(task):
            modes[segment.mode] += 1
            dependency_edges += sum(
                len(dependencies) for _, dependencies in segment.dependencies
            )
            if segment.mode == "ready_read_set":
                read_segment_sizes[len(segment.step_indices)] += 1
    return {
        "segments_by_mode": dict(modes),
        "read_segment_size_histogram": {
            str(size): count for size, count in sorted(read_segment_sizes.items())
        },
        "inferred_read_dependency_edges": dependency_edges,
        "scheduler_issue_counts": dict(issues),
    }


def _bucket_metrics(
    tasks: Sequence[legacy.Task],
    states: Sequence[legacy.RolloutState],
    pass_k: int,
    key: Callable[[legacy.Task], str],
) -> dict[str, Any]:
    states_by_position: dict[int, list[legacy.RolloutState]] = defaultdict(list)
    for state in states:
        states_by_position[state.task.position].append(state)
    buckets: dict[str, list[float]] = defaultdict(list)
    for task in tasks:
        samples = states_by_position[task.position]
        successes = sum(state.status == "success" for state in samples)
        buckets[key(task)].append(
            legacy.estimate_pass_at_k(
                len(samples), successes, min(pass_k, len(samples))
            )
        )
    return {
        label: {
            "tasks": len(values),
            f"pass_at_{pass_k}": sum(values) / max(len(values), 1),
        }
        for label, values in sorted(buckets.items())
    }


def summarize_v3(
    tasks: Sequence[legacy.Task],
    states: Sequence[legacy.RolloutState],
    pass_k: int,
) -> dict[str, Any]:
    summary = legacy.summarize(tasks, states, pass_k)
    events = [event for state in states for event in state.turns]
    mode_attempts = Counter(str(event.get("scheduler_mode")) for event in events)
    mode_matches = Counter(
        str(event.get("scheduler_mode"))
        for event in events
        if event.get("matched")
    )
    no_tool_attempts = Counter(
        str(event.get("refusal_reason"))
        for event in events
        if event.get("scheduler_mode") == "no_tool_stop"
    )
    no_tool_matches = Counter(
        str(event.get("refusal_reason"))
        for event in events
        if event.get("scheduler_mode") == "no_tool_stop"
        and event.get("matched")
    )
    reached_turns = Counter(
        int(event.get("gold_user_turn_index", -1)) + 1 for event in events
    )
    matched_turn_events = Counter(
        int(event.get("gold_user_turn_index", -1)) + 1
        for event in events
        if event.get("matched")
    )
    summary.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "scheduler_validation": _scheduler_validation(tasks),
            "transition_metrics": {
                mode: {
                    "attempts": attempts,
                    "matched": mode_matches[mode],
                    "match_rate": mode_matches[mode] / max(attempts, 1),
                }
                for mode, attempts in sorted(mode_attempts.items())
            },
            "no_tool_metrics": {
                reason: {
                    "attempts": attempts,
                    "matched": no_tool_matches[reason],
                    "match_rate": no_tool_matches[reason] / max(attempts, 1),
                }
                for reason, attempts in sorted(no_tool_attempts.items())
            },
            "user_turn_reach": {
                str(turn): {
                    "events": count,
                    "matched_events": matched_turn_events[turn],
                }
                for turn, count in sorted(reached_turns.items())
            },
            "strict_schedule_compatibility_rate": (
                sum(bool(event.get("strict_schedule_compatible")) for event in events)
                / max(len(events), 1)
            ),
            "finish_reason_counts": dict(
                Counter(str(event.get("finish_reason")) for event in events)
            ),
            "truncated_event_count": sum(
                event.get("finish_reason") == "length" for event in events
            ),
            "judge_error_count": sum(
                (event.get("refusal_judge") or {}).get("error") not in {"", None}
                for event in events
            ),
            "step_length_metrics": _bucket_metrics(
                tasks,
                states,
                pass_k,
                lambda task: str(len(task.gold_steps)),
            ),
            "user_turn_length_metrics": _bucket_metrics(
                tasks,
                states,
                pass_k,
                lambda task: str(len(task.user_turns)),
            ),
        }
    )
    states_by_position: dict[int, list[legacy.RolloutState]] = defaultdict(list)
    for state in states:
        states_by_position[state.task.position].append(state)
    strict_curve: list[dict[str, Any]] = []
    for k in range(1, pass_k + 1):
        values: list[float] = []
        for task in tasks:
            samples = states_by_position[task.position]
            strict_successes = sum(
                state.status == "success"
                and all(
                    bool(event.get("strict_schedule_compatible"))
                    for event in state.turns
                )
                for state in samples
            )
            values.append(
                legacy.estimate_pass_at_k(
                    len(samples), strict_successes, min(k, len(samples))
                )
            )
        strict_curve.append(
            {
                "k": k,
                "strict_reference_pass_at_k": (
                    sum(values) / max(len(values), 1)
                ),
            }
        )
    summary["strict_reference_metrics"] = {
        "definition": (
            "Whole-rollout success with every decision also matching the "
            "recorded next source step/group; semantic ready-read schedule "
            "equivalences are excluded"
        ),
        "num_successful_rollouts": sum(
            state.status == "success"
            and all(
                bool(event.get("strict_schedule_compatible"))
                for event in state.turns
            )
            for state in states
        ),
        "pass_curve": strict_curve,
        "pass_at_1": (
            strict_curve[0]["strict_reference_pass_at_k"]
            if strict_curve
            else 0.0
        ),
        f"pass_at_{pass_k}": (
            strict_curve[-1]["strict_reference_pass_at_k"]
            if strict_curve
            else 0.0
        ),
    }
    tool_only_curve: list[dict[str, Any]] = []
    for k in range(1, pass_k + 1):
        values: list[float] = []
        for task in tasks:
            samples = states_by_position[task.position]
            compliant_successes = sum(
                state.status == "success"
                and all(
                    bool(event.get("tool_only_compliant"))
                    for event in state.turns
                )
                for state in samples
            )
            values.append(
                legacy.estimate_pass_at_k(
                    len(samples),
                    compliant_successes,
                    min(k, len(samples)),
                )
            )
        tool_only_curve.append(
            {
                "k": k,
                "strict_tool_only_pass_at_k": (
                    sum(values) / max(len(values), 1)
                ),
            }
        )
    stop_events = [
        event
        for event in events
        if event.get("scheduler_mode") in {"terminal_stop", "no_tool_stop"}
    ]
    compliant_stop_events = sum(
        bool(event.get("tool_only_compliant")) for event in stop_events
    )
    summary["success_definition"] = (
        "BFCL-like function-call correctness: every required call is exact "
        "and the final decision emits no further tool call; visible terminal "
        "or no-tool prose is allowed"
    )
    summary["strict_tool_only_metrics"] = {
        "definition": (
            "BFCL-like whole-rollout success plus zero visible assistant "
            "characters on every decision"
        ),
        "num_successful_rollouts": sum(
            state.status == "success"
            and all(
                bool(event.get("tool_only_compliant"))
                for event in state.turns
            )
            for state in states
        ),
        "stop_events": len(stop_events),
        "compliant_stop_events": compliant_stop_events,
        "stop_event_compliance_rate": (
            compliant_stop_events / max(len(stop_events), 1)
        ),
        "pass_curve": tool_only_curve,
        "pass_at_1": (
            tool_only_curve[0]["strict_tool_only_pass_at_k"]
            if tool_only_curve
            else 0.0
        ),
        f"pass_at_{pass_k}": (
            tool_only_curve[-1]["strict_tool_only_pass_at_k"]
            if tool_only_curve
            else 0.0
        ),
    }
    return summary


def rollout_record(state: legacy.RolloutState) -> dict[str, Any]:
    record = legacy.rollout_record(state)
    record["protocol_version"] = PROTOCOL_VERSION
    record["events"] = state.turns
    return record


def load_events(
    path: Path, states: Sequence[legacy.RolloutState]
) -> int:
    if not path.exists():
        return 0
    by_key = {
        (state.task.position, state.sample_index): state for state in states
    }
    count = 0
    for line_number, event in enumerate(legacy.read_jsonl(path), 1):
        if event.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"Event line {line_number} has wrong protocol version"
            )
        key = int(event["row_position"]), int(event["sample_index"])
        if key not in by_key:
            raise ValueError(f"Unknown event key at line {line_number}: {key}")
        InteractivePassKV3Checker.apply_event(by_key[key], event)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--tool-pool", default=str(legacy.DEFAULT_TOOL_POOL))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tool-scope",
        choices=["category", "gold", "all", "declared"],
        default="declared",
    )
    parser.add_argument("--pass-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--current-date",
        default=date.today().isoformat(),
        help="Policy-visible date used to resolve relative/omitted dates",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--include-initial-state", action="store_true")
    parser.add_argument(
        "--chat-template-path",
        help=(
            "Chat template used by the serving backend; its SHA is verified "
            "against the dataset manifest and recorded for reproducibility"
        ),
    )
    thinking = parser.add_mutually_exclusive_group(required=True)
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen's hidden reasoning channel for every policy call",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen's hidden reasoning channel",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("LLM_PROXY_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.6-35B-A3B-FP8",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_PROXY_MASTER_KEY")
        or os.getenv("VLLM_API_KEY"),
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument(
        "--judge-url",
        default=os.getenv("LLM_PROXY_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--judge-model",
        default="nvidia/Llama-3.3-70B-Instruct-FP8",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("LLM_PROXY_MASTER_KEY")
        or os.getenv("VLLM_API_KEY"),
    )
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument(
        "--lexical-refusal-judge",
        action="store_true",
        help="Smoke tests only; authoritative runs use the semantic judge",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pass_k < 1 or args.workers < 1:
        raise ValueError("--pass-k and --workers must be positive")
    projection_manifest = validate_projection_manifest(args.jsonl)
    chat_template_validation: dict[str, Any] | None = None
    if args.chat_template_path:
        chat_template_path = Path(args.chat_template_path).resolve()
        actual_template_sha = _sha256(chat_template_path)
        expected_template_sha = projection_manifest[
            "training_contract"
        ].get("chat_template_sha256")
        if actual_template_sha != expected_template_sha:
            raise ValueError(
                "Serving chat template differs from the signed corpus "
                f"template: {actual_template_sha} != {expected_template_sha}"
            )
        chat_template_validation = {
            "path": str(chat_template_path),
            "sha256": actual_template_sha,
            "matches_manifest": True,
        }
    tasks, _ = legacy.load_tasks(
        args.jsonl,
        args.tool_pool,
        tool_scope=args.tool_scope,
        max_samples=args.max_samples,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    for task in tasks:
        task.tools = _canonical_policy_tools(task.tools)
        # Recorded assistant prose was deliberately excluded from the SFT
        # policy history and must never become an argument-provenance source.
        for turn in task.user_turns:
            turn["assistant_response"] = ""
    projection_preparation = prepare_next_action_tasks(
        tasks,
        trust_projected_parallel=True,
    )
    full_unsharded = (
        args.max_samples is None
        and args.shard_count == 1
        and args.shard_index == 0
    )
    manifest_stats = projection_manifest.get("statistics") or {}
    if full_unsharded:
        expected = {
            "tasks": projection_manifest.get("rows"),
            "evaluated_action_targets": manifest_stats.get("action_targets"),
            "evaluated_calls": manifest_stats.get("tool_calls"),
            "recertified_parallel_steps": manifest_stats.get("parallel_targets"),
            "projected_no_tool_targets": manifest_stats.get("no_call_targets"),
            "evaluated_decisions": manifest_stats.get("decision_targets"),
        }
        mismatches = {
            key: {"expected": value, "actual": projection_preparation.get(key, 0)}
            for key, value in expected.items()
            if value is not None and projection_preparation.get(key, 0) != value
        }
        if mismatches:
            raise ValueError(
                f"Prepared evaluation view disagrees with manifest: {mismatches}"
            )

    missing_gold_tools: list[dict[str, Any]] = []
    for task in tasks:
        for step in task.gold_steps:
            if not step.get("calls"):
                continue
            turn_index = int(step["turn_index"])
            available = {
                _tool_name(tool) for tool in tools_for_turn(task, turn_index)
            }
            for call in step["calls"]:
                if call["name"] not in available:
                    missing_gold_tools.append(
                        {
                            "row_position": task.position,
                            "turn_index": turn_index,
                            "tool_name": call["name"],
                        }
                    )
    if missing_gold_tools:
        raise ValueError(
            "Evaluated gold calls are absent from their turn tool menus: "
            f"{missing_gold_tools[:10]}"
        )
    scheduler_validation = _scheduler_validation(tasks)
    argument_visibility = (
        projection_manifest.get("argument_visibility", {}).get("output") or {}
    )
    scored_total_arguments = sum(
        len(call.get("arguments") or {})
        for task in tasks
        for call in task.gold_calls
    )
    validation = {
        "protocol_version": PROTOCOL_VERSION,
        "projection_manifest": projection_manifest,
        "projection_preparation": projection_preparation,
        "chat_template": chat_template_validation,
        "tasks": len(tasks),
        "steps": sum(len(task.gold_steps) for task in tasks),
        "calls": sum(len(task.gold_calls) for task in tasks),
        "multi_turn_tasks": sum(len(task.user_turns) > 1 for task in tasks),
        "no_tool_steps": sum(
            step["execution_mode"] == "no_tool_stop"
            for task in tasks
            for step in task.gold_steps
        ),
        "certified_parallel_steps": sum(
            step["execution_mode"] == "parallel"
            for task in tasks
            for step in task.gold_steps
        ),
        "synthetic_refuse_tools_visible": sum(
            _tool_name(tool) == "refuse"
            for task in tasks
            for tool in task.tools
        ),
        "initial_state_policy_visible": args.include_initial_state,
        "enable_thinking": args.enable_thinking,
        "hidden_argument_count": argument_visibility[
            "hidden_argument_count"
        ],
        "total_arguments": argument_visibility["total_arguments"],
        "hidden_argument_ratio": argument_visibility[
            "hidden_argument_ratio"
        ],
        "argument_visibility": argument_visibility,
        "scored_argument_visibility": {
            "total_arguments": scored_total_arguments,
            "hidden_argument_count": 0,
            "hidden_argument_ratio": 0.0,
            "evidence": (
                "Subset of the SHA-locked source visibility certificate; "
                "golden-prefix arguments are excluded from this denominator"
            ),
        },
        "scheduler": scheduler_validation,
    }
    print(json.dumps({"data_validation": validation}, indent=2), flush=True)
    if args.validate_only:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    if args.overwrite and events_path.exists():
        events_path.unlink()
    if events_path.exists() and not args.resume:
        raise FileExistsError(
            f"{events_path} exists; pass --resume or --overwrite"
        )

    sampling = legacy.SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "jsonl": str(Path(args.jsonl).resolve()),
        "tool_pool": str(Path(args.tool_pool).resolve()),
        "tool_scope": args.tool_scope,
        "pass_k": args.pass_k,
        "sampling": asdict(sampling),
        "workers": args.workers,
        "include_initial_state": args.include_initial_state,
        "current_date": args.current_date,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_path": str(DEFAULT_SYSTEM_PROMPT_PATH),
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "chat_template": chat_template_validation,
        "enable_thinking": args.enable_thinking,
        "parallel_tool_calls": "always_enabled_no_gold_hint",
        "synthetic_refuse_tool_visible": False,
        "vllm_url": legacy._normalise_vllm_url(args.vllm_url),
        "model": args.model,
        "no_tool_scoring": "strict_empty_visible_assistant_target",
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "data_validation": validation,
    }
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume configuration does not match config.json")
    legacy.atomic_write_json(config_path, config)

    policy_client = legacy.VLLMClient(
        args.vllm_url,
        args.model,
        args.api_key,
        timeout=args.request_timeout,
        retries=args.request_retries,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
    )
    # No semantic judge is involved: the SFT no-tool/terminal target is the
    # literal empty visible assistant response.  The object is retained only
    # for the checker's backward-compatible constructor.
    refusal_judge: Any = LexicalRefusalJudge()
    checker = InteractivePassKV3Checker(
        policy_client,
        refusal_judge,
        pass_k=args.pass_k,
        sampling=sampling,
        workers=args.workers,
        include_initial_state=args.include_initial_state,
        current_date=args.current_date,
    )
    states = checker.build_states(tasks)
    resumed = load_events(events_path, states) if args.resume else 0
    if resumed:
        print(f"Resumed {resumed} events", flush=True)

    event_mode = "a" if args.resume else "w"
    with contextlib.ExitStack() as stack:
        event_file = stack.enter_context(
            events_path.open(event_mode, encoding="utf-8")
        )

        def on_event(
            event: dict[str, Any], _state: legacy.RolloutState
        ) -> None:
            event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            event_file.flush()

        def on_progress(progress: dict[str, int]) -> None:
            legacy.atomic_write_json(out_dir / "progress.json", progress)
            if progress["processed"] == progress["turn_total"]:
                print(json.dumps(progress), flush=True)

        checker.run(
            tasks,
            states=states,
            on_event=on_event,
            on_progress=on_progress,
        )

    with (out_dir / "rollouts.jsonl").open("w", encoding="utf-8") as output:
        for state in states:
            output.write(
                json.dumps(rollout_record(state), ensure_ascii=False) + "\n"
            )
    summary = summarize_v3(tasks, states, args.pass_k)
    summary["config"] = config
    legacy.atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
