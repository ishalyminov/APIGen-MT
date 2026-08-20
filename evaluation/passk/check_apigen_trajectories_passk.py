#!/usr/bin/env python3
"""Interactive pass@k checker for APIGen trajectory JSONL files.

This is a standalone, OpenAI-compatible-server adaptation of
``toolcallrl/verl/toolcall_rl/eval_vllm_native_interactive_jsonl.py``.

The checker does not execute arbitrary generated calls. Instead, each rollout:

1. gives the policy the current user query and an APIGen tool catalog;
2. requires one gold step per assistant turn (one call for a sequential step,
   or the complete call set for a certified parallel step);
3. exact-matches that step against an unmatched gold step;
4. replays the recorded gold outputs only after an exact match;
5. advances to the next recorded user turn only after the current turn's gold
   steps are complete; and
6. succeeds only after every gold step has been matched once.

By default, ``initial_api_state`` is deliberately hidden. This makes the check a
test of policy-visible solvability: arguments must come from the user query,
tool schemas, or outputs returned by earlier matched calls. Use
``--include-initial-state`` only when reproducing the older evaluator's setup.

The module can be imported later (``VLLMClient``, ``load_tool_pool``,
``task_from_record``, and ``InteractivePassKChecker`` are the integration
surface), or run as a CLI. The CLI can use an already-running vLLM URL or own a
temporary vLLM API-server process for the duration of the check.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_TOOL_POOL = (
    HERE.parents[1]
    / "magnet_tool_extraction"
    / "bfcl_v3_tools_with_outputs.jsonl"
)
DEFAULT_QWEN36_27B = Path(
    "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.6-27B-FP8/snapshots/"
    "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
)

SYSTEM_PROMPT = """You are a precise interactive tool-calling assistant.

Complete the user's request with the available tools. After each correct
assistant tool-call step, the environment will return the tool output or
outputs. Use returned values when constructing dependent later calls.

Rules:
- Call only tools from the provided catalog.
- For a sequential or dependent step, emit exactly one tool call.
- When the user explicitly requests multiple independent read-only operations
  that can run from the same context, emit those calls together in one
  assistant response.
- Do not emit a dependent future call before receiving its input-producing
  tool output.
- Emit tool calls rather than a final natural-language answer.
- If the request is genuinely missing a required value, materially ambiguous,
  or unsupported and the `refuse` tool is available, emit exactly one `refuse`
  call with the matching reason. Do not call a real tool at that transition.
- Use only values available in the user request, tool schemas, or prior tool
  outputs. The simulator's private API state is not policy-visible.
- Do not invent tool names or argument values.
- You may think before the tool call, but do not place calls inside thinking.
"""

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    category: str
    schema: dict[str, Any]


@dataclass
class Task:
    position: int
    raw: dict[str, Any]
    query: str
    initial_state: dict[str, Any]
    tools: list[dict[str, Any]]
    gold_calls: list[dict[str, Any]]
    gold_steps: list[dict[str, Any]]
    user_turns: list[dict[str, Any]]
    step_order_matters: bool
    data_issues: list[str]
    focus_category: str = ""


@dataclass
class RolloutState:
    task: Task
    sample_index: int
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]] = field(default_factory=list)
    matched_gold_indices: set[int] = field(default_factory=set)
    matched_gold_step_indices: set[int] = field(default_factory=set)
    status: str = "active"
    failure: str = ""

    @property
    def next_turn(self) -> int:
        return len(self.turns) + 1

    def unmatched_gold_calls(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for index, call in enumerate(self.task.gold_calls):
            if index not in self.matched_gold_indices:
                yield index, call

    def unmatched_gold_steps(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for index, step in enumerate(self.task.gold_steps):
            if index not in self.matched_gold_step_indices:
                yield index, step


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096
    seed: int = 42


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalise_schema(value: Any) -> Any:
    """Convert BFCL-ish schema types into ordinary JSON Schema."""
    if isinstance(value, list):
        return [_normalise_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _normalise_schema(child)
        for key, child in value.items()
        if key != "optional"
    }
    schema_type = result.get("type")
    type_map = {
        "dict": "object",
        "list": "array",
        "float": "number",
        "double": "number",
        "int": "integer",
        "bool": "boolean",
    }
    if isinstance(schema_type, str):
        result["type"] = type_map.get(schema_type.lower(), schema_type.lower())
    return result


def load_tool_pool(path: str | Path) -> dict[str, ToolDefinition]:
    catalog: dict[str, ToolDefinition] = {}
    for line_number, item in enumerate(read_jsonl(path), 1):
        name = str(item.get("name") or item.get("api_name") or "").strip()
        if not name:
            raise ValueError(f"Tool-pool line {line_number} has no name/api_name")
        if name in catalog:
            raise ValueError(f"Duplicate tool name in pool: {name}")
        parameters = _normalise_schema(item.get("parameters") or {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool {name!r} has non-object parameters")
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        description = str(
            item.get("description")
            or item.get("api_description")
            or item.get("tool_description")
            or name
        )
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        catalog[name] = ToolDefinition(
            name=name,
            category=str(item.get("category") or "Unknown"),
            schema=schema,
        )
    return catalog


def _user_turns_from_record(raw: dict[str, Any]) -> list[dict[str, Any]]:
    conversation_turns = (raw.get("conversation") or {}).get("turns") or []
    if conversation_turns:
        return [
            {
                "query": str(turn.get("user_query") or ""),
                "turn_number": turn.get("turn_number", 0),
                "assistant_response": str(turn.get("assistant_response") or ""),
                "steps": list(turn.get("steps") or []),
                # Keep projected next-action metadata available to newer
                # interactive evaluators.  Legacy callers ignore these keys.
                "no_tool_target": turn.get("no_tool_target") is True,
                "no_tool_reason": turn.get("no_tool_reason"),
                "no_tool_certificate": turn.get("no_tool_certificate"),
                "available_tools": turn.get("available_tools"),
                "sft_supervision": turn.get("sft_supervision") is not False,
            }
            for turn in conversation_turns
        ]

    trajectory = raw.get("trajectory") or {}
    return [
        {
            "query": str(trajectory.get("query") or ""),
            "turn_number": trajectory.get("turn_number", 0),
            "assistant_response": str(trajectory.get("final_response") or ""),
            "steps": list(trajectory.get("steps") or []),
            "no_tool_target": trajectory.get("no_tool_target") is True,
            "no_tool_reason": trajectory.get("no_tool_reason"),
            "no_tool_certificate": trajectory.get("no_tool_certificate"),
            "available_tools": trajectory.get("available_tools"),
            "sft_supervision": trajectory.get("sft_supervision") is not False,
        }
    ]


def _gold_from_record(
    raw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    user_turns = _user_turns_from_record(raw)
    for turn_index, turn in enumerate(user_turns):
        for step_index, step in enumerate(turn["steps"]):
            step_calls: list[dict[str, Any]] = []
            call_indices: list[int] = []
            for call in step.get("tool_calls", []):
                gold_call = {
                    "name": str(call.get("tool_name") or call.get("name") or ""),
                    "arguments": call.get("arguments") or {},
                    "output": call.get("output"),
                }
                call_indices.append(len(calls))
                calls.append(gold_call)
                step_calls.append(gold_call)
            if step_calls:
                quality = step.get("quality_verification") or {}
                explicit_mode = str(step.get("execution_mode") or "")
                declared_refusal = (
                    explicit_mode == "refusal"
                    or (
                        len(step_calls) == 1
                        and step_calls[0]["name"] == "refuse"
                    )
                )
                refusal_reason = (
                    step_calls[0].get("arguments", {}).get("reason")
                    if declared_refusal and len(step_calls) == 1
                    else None
                )
                refusal_certified = (
                    declared_refusal
                    and len(step_calls) == 1
                    and step_calls[0]["name"] == "refuse"
                    and refusal_reason
                    in {
                        "missing_argument",
                        "ambiguity",
                        "no_appropriate_function",
                    }
                    and quality.get("passed") is True
                    and quality.get("mode") == "refusal"
                    and bool(
                        quality.get("native_response")
                        or turn.get("assistant_response")
                    )
                )
                declared_parallel = (
                    not declared_refusal
                    and (
                        explicit_mode == "parallel"
                        or quality.get("mode") == "parallel"
                    )
                )
                per_call_checks = quality.get("per_call_checks") or []
                certified_parallel = (
                    len(step_calls) > 1
                    and declared_parallel
                    and quality.get("passed") is True
                    and quality.get("read_only") is True
                    and quality.get("same_pre_batch_context") is True
                    and quality.get("forward_reverse_outputs_equal") is True
                    and quality.get("forward_reverse_state_equal") is True
                    and quality.get("isolated_outputs_equal") is True
                    and (
                        quality.get("argument_visibility_certificate") or {}
                    ).get("passed")
                    is True
                    and len(per_call_checks) == len(step_calls)
                    and all(
                        check.get("passed") is True
                        and check.get("read_only") is True
                        for check in per_call_checks
                    )
                )
                if declared_refusal:
                    execution_mode = (
                        "refusal"
                        if refusal_certified
                        else "uncertified_refusal"
                    )
                else:
                    execution_mode = (
                        "parallel"
                        if certified_parallel
                        else (
                            "uncertified_parallel"
                            if declared_parallel
                            else explicit_mode or "sequential"
                        )
                    )
                call_order_matters = step.get("call_order_matters")
                if declared_refusal:
                    call_order_matters = True
                elif not isinstance(call_order_matters, bool):
                    call_order_matters = not certified_parallel
                elif not certified_parallel:
                    call_order_matters = True
                steps.append(
                    {
                        "turn_index": turn_index,
                        "step_index": step_index,
                        "calls": step_calls,
                        "call_indices": call_indices,
                        "execution_mode": execution_mode,
                        "call_order_matters": call_order_matters,
                        "parallel_certified": certified_parallel,
                        "refusal_certified": refusal_certified,
                        "refusal_reason": refusal_reason,
                    }
                )
    return calls, steps, user_turns


def _flatten_gold_calls(raw: dict[str, Any]) -> list[dict[str, Any]]:
    calls, _, _ = _gold_from_record(raw)
    return calls


def _declared_tool_schemas(raw: dict[str, Any]) -> list[dict[str, Any]]:
    declared = raw.get("available_tools") or []
    schemas: list[dict[str, Any]] = []
    for item in declared:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            function = dict(item["function"])
        else:
            name = item.get("name") or item.get("api_name")
            if not name:
                continue
            function = {
                "name": str(name),
                "description": str(
                    item.get("description")
                    or item.get("api_description")
                    or name
                ),
                "parameters": item.get("parameters") or {},
            }
        function["parameters"] = _normalise_schema(
            function.get("parameters") or {}
        )
        function["parameters"].setdefault("type", "object")
        function["parameters"].setdefault("properties", {})
        schemas.append({"type": "function", "function": function})
    return schemas


def _schema_name(schema: dict[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def _focus_category(raw: dict[str, Any]) -> str:
    metadata = raw.get("generation_metadata") or {}
    category = metadata.get("focus_category")
    if category:
        return str(category)
    categories = raw.get("trajectory", {}).get("categories_used") or []
    return str(categories[0]) if len(categories) == 1 else ""


def _tool_names_for_task(
    raw: dict[str, Any],
    gold_calls: Sequence[dict[str, Any]],
    catalog: dict[str, ToolDefinition],
    scope: str,
) -> list[str]:
    gold_names = list(dict.fromkeys(call["name"] for call in gold_calls))
    if scope == "gold":
        return gold_names
    if scope == "all":
        return list(catalog)
    if scope == "declared":
        declared = raw.get("available_tools") or []
        names = [
            _schema_name(schema)
            for schema in _declared_tool_schemas(raw)
            if _schema_name(schema)
        ]
        if not names:
            names = [str(name) for name in declared if isinstance(name, str)]
        return list(dict.fromkeys(names)) if names else gold_names

    category = _focus_category(raw)
    categories = (
        {category}
        if category
        else {catalog[name].category for name in gold_names if name in catalog}
    )
    return [
        name
        for name, definition in catalog.items()
        if definition.category in categories
    ]


def _nested_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _nested_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_values(child)
    elif value is not None:
        yield value


def _data_issues(
    raw: dict[str, Any], gold_calls: Sequence[dict[str, Any]], available: set[str]
) -> list[str]:
    issues: list[str] = []
    if not gold_calls:
        issues.append("no_gold_calls")
    if any(not call["name"] or call["name"] not in available for call in gold_calls):
        issues.append("gold_tool_not_available")
    if any(call.get("output") is None for call in gold_calls):
        issues.append("missing_output")
    if any(not isinstance(call.get("arguments"), dict) for call in gold_calls):
        issues.append("gold_arguments_not_object")

    for turn in _user_turns_from_record(raw):
        steps = turn["steps"]
        if any(
            index > 0
            and step.get("pre_state") is not None
            and steps[index - 1].get("post_state") is not None
            and step.get("pre_state") != steps[index - 1].get("post_state")
            for index, step in enumerate(steps)
        ):
            issues.append("broken_state_chain")
        if any(
            step.get("state_verification", {}).get("is_valid") is False
            for step in steps
        ):
            issues.append("invalid_gold_step")
    _, gold_steps, _ = _gold_from_record(raw)
    if any(
        len(step["calls"]) > 1 and not step["parallel_certified"]
        for step in gold_steps
    ):
        issues.append("uncertified_multi_call_step")
    if any(
        not step["call_order_matters"] and step["execution_mode"] != "parallel"
        for step in gold_steps
    ):
        issues.append("order_invariant_non_parallel_step")
    if any(
        step["execution_mode"] == "uncertified_refusal"
        for step in gold_steps
    ):
        issues.append("uncertified_refusal_step")
    if any(
        any(call["name"] == "refuse" for call in step["calls"])
        and step["execution_mode"] != "refusal"
        for step in gold_steps
    ):
        issues.append("malformed_refusal_step")
    if any(
        isinstance(value, str) and value.strip().lower().startswith("error:")
        for call in gold_calls
        for value in _nested_values(call.get("output"))
    ):
        issues.append("gold_execution_error")
    return issues


def load_tasks(
    jsonl_path: str | Path,
    tool_pool_path: str | Path = DEFAULT_TOOL_POOL,
    *,
    tool_scope: str = "category",
    max_samples: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> tuple[list[Task], dict[str, ToolDefinition]]:
    """Load APIGen records and turn their BFCL schemas into OpenAI tools."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard_count/shard_index")
    catalog = load_tool_pool(tool_pool_path)
    raw_rows = read_jsonl(jsonl_path)
    if max_samples is not None:
        raw_rows = raw_rows[:max_samples]

    tasks = [
        task_from_record(position, raw, catalog, tool_scope=tool_scope)
        for position, raw in enumerate(raw_rows)
        if position % shard_count == shard_index
    ]
    return tasks, catalog


def task_from_record(
    position: int,
    raw: dict[str, Any],
    catalog: dict[str, ToolDefinition],
    *,
    tool_scope: str = "category",
) -> Task:
    """Adapt one single- or multi-turn APIGen record for checking."""
    gold_calls, gold_steps, user_turns = _gold_from_record(raw)
    query = user_turns[0]["query"] if user_turns else ""
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Row {position} has no initial user query")
    names = _tool_names_for_task(raw, gold_calls, catalog, tool_scope)
    declared_schemas = (
        _declared_tool_schemas(raw) if tool_scope == "declared" else []
    )
    declared_by_name = {
        _schema_name(schema): schema for schema in declared_schemas
        if _schema_name(schema)
    }
    unknown = sorted(set(names) - set(catalog) - set(declared_by_name))
    if unknown:
        raise ValueError(
            f"Row {position} references tools missing from pool: {unknown}"
        )
    available = set(names)
    tools = [
        declared_by_name[name] if name in declared_by_name else catalog[name].schema
        for name in names
    ]
    trajectory = raw.get("trajectory") or {}
    initial_state = (
        raw.get("initial_api_state") or trajectory.get("initial_api_state") or {}
    )
    return Task(
        position=position,
        raw=raw,
        query=query,
        initial_state=initial_state,
        tools=tools,
        gold_calls=gold_calls,
        gold_steps=gold_steps,
        user_turns=user_turns,
        step_order_matters=bool((raw.get("conversation") or {}).get("turns")),
        data_issues=_data_issues(raw, gold_calls, available),
        focus_category=_focus_category(raw),
    )


def _normalise_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _normalise_value(child_key, child)
            for child_key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_normalise_value(key, child) for child in value]
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        result = value.strip()
        if key in {"symbol", "stock"}:
            return result.upper()
        if key in {"name", "sector", "order_type"}:
            return result.lower()
        return result
    return value


def normalise_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    return {
        key: _normalise_value(key, value) for key, value in sorted(arguments.items())
    }


def exact_call_match(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    return predicted.get("name") == gold.get("name") and normalise_arguments(
        predicted.get("arguments")
    ) == normalise_arguments(gold.get("arguments"))


def initial_messages(
    task: Task, include_initial_state: bool = False
) -> list[dict[str, Any]]:
    content = task.query
    if include_initial_state:
        content = (
            "Initial environment context:\n"
            + json.dumps(task.initial_state, ensure_ascii=False, indent=2)
            + "\n\nUser request:\n"
            + task.query
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_states(
    tasks: Sequence[Task], pass_k: int, include_initial_state: bool
) -> list[RolloutState]:
    if pass_k < 1:
        raise ValueError("pass_k must be positive")
    states = [
        RolloutState(
            task=task,
            sample_index=sample_index,
            messages=initial_messages(task, include_initial_state),
        )
        for task in tasks
        for sample_index in range(pass_k)
    ]
    for state in states:
        if not state.task.gold_steps:
            state.status = "failed"
            state.failure = "no_gold_steps"
    return states


def _schema_properties(tools: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name") == name:
            return (function.get("parameters") or {}).get("properties") or {}
    return {}


def _coerce_xml_value(value: str, schema: dict[str, Any]) -> Any:
    value = value.strip("\n")
    schema_type = str(schema.get("type") or "string").lower()
    if value.lower() == "null":
        return None
    if schema_type == "string":
        return value
    if schema_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if schema_type == "number":
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    if schema_type == "boolean":
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value
    if schema_type in {"array", "object"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return value
    return value


def parse_qwen_xml_calls(
    content: str, tools: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    if "<tool_call>" not in content:
        return [], 0
    calls: list[dict[str, Any]] = []
    errors = 0
    blocks = TOOL_CALL_RE.findall(content)
    if not blocks:
        return [], 1
    for block in blocks:
        function_matches = FUNCTION_RE.findall(block)
        if len(function_matches) != 1:
            errors += 1
            continue
        name, body = function_matches[0]
        name = name.strip()
        properties = _schema_properties(tools, name)
        arguments = {
            param_name.strip(): _coerce_xml_value(
                raw_value, properties.get(param_name.strip(), {})
            )
            for param_name, raw_value in PARAMETER_RE.findall(body)
        }
        calls.append({"name": name, "arguments": arguments})
    return calls, errors


def parse_response_calls(
    response: dict[str, Any], tools: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, str]:
    choices = response.get("choices") or []
    if not choices:
        return [], 1, ""
    message = choices[0].get("message") or {}
    content = str(message.get("content") or "")
    raw_calls = message.get("tool_calls") or []
    if not raw_calls:
        calls, errors = parse_qwen_xml_calls(content, tools)
        return calls, errors, content

    calls: list[dict[str, Any]] = []
    errors = 0
    for raw_call in raw_calls:
        try:
            function = raw_call.get("function") or {}
            name = str(function["name"])
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments are not an object")
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "tool_call_id": raw_call.get("id"),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors += 1
    return calls, errors, content


def _normalise_vllm_url(value: str) -> str:
    result = value.rstrip("/")
    suffix = "/chat/completions"
    if result.endswith(suffix):
        result = result[: -len(suffix)]
    if not result.endswith("/v1"):
        result += "/v1"
    return result


class VLLMClient:
    """Small OpenAI-compatible client suitable for importing into APIGen."""

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 300.0,
        retries: int = 2,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = _normalise_vllm_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.chat_template_kwargs = (
            dict(chat_template_kwargs)
            if chat_template_kwargs is not None
            else None
        )

    def _request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = self.base_url + path
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.request(
                    "GET" if payload is None else "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    detail = response.text[:2000]
                    last_error = RuntimeError(
                        f"HTTP {response.status_code} from {url}: {detail}"
                    )
                    if response.status_code < 500:
                        break
                else:
                    return response.json()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if isinstance(error, json.JSONDecodeError):
                    break
            if attempt < self.retries:
                time.sleep(min(2**attempt, 5))
        raise RuntimeError(f"vLLM request failed: {last_error}") from last_error

    def list_models(self) -> list[str]:
        response = self._request("/models")
        return [str(item["id"]) for item in response.get("data", []) if item.get("id")]

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        models = self.list_models()
        if not models:
            raise RuntimeError(f"No model advertised by {self.base_url}")
        self.model = models[0]
        return self.model

    def is_ready(self) -> bool:
        try:
            return bool(self.list_models())
        except Exception:
            return False

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        sampling: SamplingConfig,
        seed: int,
        *,
        parallel_tool_calls: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "model": self.resolve_model(),
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": parallel_tool_calls,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "presence_penalty": sampling.presence_penalty,
            "seed": seed,
            # These are accepted as top-level extensions by vLLM.
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
        }
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        return self._request("/chat/completions", payload)


class ManagedVLLMServer:
    """Optionally own a temporary vLLM OpenAI API server process."""

    def __init__(
        self,
        *,
        launch_model: str | None,
        client: VLLMClient,
        out_dir: Path,
        python: str,
        served_model_name: str,
        host: str,
        port: int,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        tool_call_parser: str,
        reasoning_parser: str | None,
        startup_timeout: float,
        cuda_visible_devices: str | None,
        extra_args: Sequence[str],
        reuse_running_server: bool,
    ) -> None:
        self.launch_model = launch_model
        self.client = client
        self.out_dir = out_dir
        self.python = python
        self.served_model_name = served_model_name
        self.host = host
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.startup_timeout = startup_timeout
        self.cuda_visible_devices = cuda_visible_devices
        self.extra_args = list(extra_args)
        self.reuse_running_server = reuse_running_server
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None

    def __enter__(self) -> VLLMClient:
        if not self.launch_model:
            return self.client
        if self.client.is_ready():
            if self.reuse_running_server:
                return self.client
            raise RuntimeError(
                f"A server already responds at {self.client.base_url}; use "
                "--reuse-running-server or omit --launch-vllm"
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.out_dir / "vllm_server.log"
        self.log_handle = log_path.open("ab")
        command = [
            self.python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.launch_model,
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            self.tool_call_parser,
        ]
        if self.reasoning_parser:
            command.extend(["--reasoning-parser", self.reasoning_parser])
        command.extend(self.extra_args)
        environment = os.environ.copy()
        if self.cuda_visible_devices:
            environment["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log_handle.flush()
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[
                        -4000:
                    ]
                    raise RuntimeError(f"vLLM exited during startup:\n{tail}")
                if self.client.is_ready():
                    self.client.model = self.served_model_name
                    return self.client
                time.sleep(2)
            raise TimeoutError(
                f"vLLM did not become ready within {self.startup_timeout}s; see {log_path}"
            )
        except BaseException:
            self._stop()
            raise

    def _stop(self) -> None:
        """Stop only the server process owned by this context."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop()


class InteractivePassKChecker:
    """Reusable interactive exact-replay checker."""

    def __init__(
        self,
        client: VLLMClient,
        *,
        pass_k: int = 16,
        sampling: SamplingConfig = SamplingConfig(),
        workers: int = 16,
        include_initial_state: bool = False,
        ordered: bool = False,
    ) -> None:
        self.client = client
        self.pass_k = pass_k
        self.sampling = sampling
        self.workers = max(1, workers)
        self.include_initial_state = include_initial_state
        self.ordered = ordered

    def _seed(self, state: RolloutState) -> int:
        return (
            self.sampling.seed
            + state.task.position * self.pass_k
            + state.sample_index
            + state.next_turn * 1_000_003
        )

    @staticmethod
    def _call_key(call: dict[str, Any]) -> tuple[str, str]:
        return (
            str(call.get("name") or ""),
            json.dumps(
                normalise_arguments(call.get("arguments")),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    @classmethod
    def _exact_step_match(
        cls,
        predicted: Sequence[dict[str, Any]],
        gold_step: dict[str, Any],
    ) -> bool:
        gold = gold_step["calls"]
        if gold_step["call_order_matters"]:
            return len(predicted) == len(gold) and all(
                exact_call_match(predicted_call, gold_call)
                for predicted_call, gold_call in zip(predicted, gold)
            )
        # A Counter is deliberately used rather than a set: certified parallel
        # calls are order-invariant, but duplicate calls still have multiplicity.
        return Counter(cls._call_key(call) for call in predicted) == Counter(
            cls._call_key(call) for call in gold
        )

    def _generate_event(self, state: RolloutState) -> dict[str, Any]:
        try:
            unmatched = list(state.unmatched_gold_steps())
            current_user_turn = (
                min(step["turn_index"] for _, step in unmatched)
                if unmatched
                else 0
            )
            current_turn_steps = [
                (index, step)
                for index, step in unmatched
                if step["turn_index"] == current_user_turn
            ]
            candidates = (
                current_turn_steps[:1]
                if self.ordered or state.task.step_order_matters
                else current_turn_steps
            )
            response = self.client.chat(
                state.messages,
                state.task.tools,
                self.sampling,
                self._seed(state),
                # Keep the policy capability constant. Toggling this from the
                # current gold step would leak whether the expected answer is
                # sequential or parallel.
                parallel_tool_calls=True,
            )
            calls, parse_errors, content = parse_response_calls(
                response, state.task.tools
            )
            choices = response.get("choices") or []
            finish_reason = (
                (choices[0] or {}).get("finish_reason") if choices else None
            )
            usage = response.get("usage") or {}
        except Exception as error:
            return {
                "row_position": state.task.position,
                "sample_index": state.sample_index,
                "turn": state.next_turn,
                "matched": False,
                "failure": "api_error",
                "error": str(error),
                "parse_errors": 0,
                "predicted_call": None,
                "predicted_calls": [],
                "raw_completion": "",
                "finish_reason": None,
                "usage": {},
                "matched_gold_index": None,
                "matched_gold_call": None,
                "tool_output": None,
            }

        matched_item = next(
            (
                (index, gold_step)
                for index, gold_step in candidates
                if not parse_errors
                and self._exact_step_match(calls, gold_step)
            ),
            None,
        )
        if parse_errors:
            failure = "parse_error"
        elif not calls:
            failure = "no_tool_call"
        elif matched_item is None:
            expected_sizes = {len(step["calls"]) for _, step in candidates}
            expected_name_sets = [
                Counter(call["name"] for call in step["calls"])
                for _, step in candidates
            ]
            predicted_names = Counter(call.get("name") for call in calls)
            if len(calls) not in expected_sizes:
                failure = "wrong_call_count"
            elif predicted_names not in expected_name_sets:
                failure = "wrong_tool"
            else:
                failure = "wrong_arguments"
        else:
            failure = ""

        matched_step_index, matched_step = (
            matched_item if matched_item else (None, None)
        )
        matched_calls: list[dict[str, Any]] = []
        matched_call_indices: list[int] = []
        if matched_step is not None:
            if matched_step["call_order_matters"]:
                matched_call_indices.extend(matched_step["call_indices"])
                matched_calls.extend(matched_step["calls"])
            else:
                remaining = list(
                    zip(matched_step["call_indices"], matched_step["calls"])
                )
                for predicted in calls:
                    predicted_key = self._call_key(predicted)
                    match_position = next(
                        index
                        for index, (_, gold_call) in enumerate(remaining)
                        if self._call_key(gold_call) == predicted_key
                    )
                    call_index, gold_call = remaining.pop(match_position)
                    matched_call_indices.append(call_index)
                    matched_calls.append(gold_call)
        return {
            "row_position": state.task.position,
            "sample_index": state.sample_index,
            "turn": state.next_turn,
            "matched": not failure,
            "failure": failure,
            "parse_errors": parse_errors,
            "predicted_call": calls[0] if len(calls) == 1 else None,
            "predicted_calls": calls,
            "raw_completion": content,
            "finish_reason": finish_reason,
            "usage": usage,
            "matched_gold_step_index": matched_step_index,
            "matched_execution_mode": (
                matched_step["execution_mode"] if matched_step is not None else None
            ),
            "call_order_matters": (
                matched_step["call_order_matters"]
                if matched_step is not None
                else None
            ),
            "matched_gold_indices": matched_call_indices,
            "matched_gold_index": (
                matched_call_indices[0] if len(matched_call_indices) == 1 else None
            ),
            "matched_gold_calls": [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in matched_calls
            ],
            "matched_gold_call": (
                {
                    "name": matched_calls[0]["name"],
                    "arguments": matched_calls[0]["arguments"],
                }
                if len(matched_calls) == 1
                else None
            ),
            "tool_outputs": [call.get("output") for call in matched_calls],
            "tool_output": matched_calls[0].get("output") if len(matched_calls) == 1 else None,
        }

    @staticmethod
    def apply_event(state: RolloutState, event: dict[str, Any]) -> None:
        if state.status != "active":
            raise ValueError("Cannot apply an event to a completed rollout")
        if int(event["turn"]) != state.next_turn:
            raise ValueError("Non-consecutive rollout event")
        state.turns.append(event)
        if not event.get("matched"):
            state.status = "failed"
            state.failure = str(event.get("failure") or "call_mismatch")
            return

        if event.get("matched_gold_step_index") is not None:
            matched_step_index = int(event["matched_gold_step_index"])
            matched_indices = [int(index) for index in event["matched_gold_indices"]]
            predicted_calls = list(event["predicted_calls"])
            tool_outputs = list(event["tool_outputs"])
        else:
            # Backward-compatible event replay for existing single-call runs.
            matched_step_index = int(event["matched_gold_index"])
            matched_indices = [int(event["matched_gold_index"])]
            predicted_calls = [event["predicted_call"]]
            tool_outputs = [event["tool_output"]]
        if matched_step_index in state.matched_gold_step_indices:
            raise ValueError("Gold step matched more than once")
        if any(index in state.matched_gold_indices for index in matched_indices):
            raise ValueError("Gold call matched more than once")
        if len(predicted_calls) != len(tool_outputs):
            raise ValueError("Predicted-call/tool-output count mismatch")
        state.matched_gold_step_indices.add(matched_step_index)
        state.matched_gold_indices.update(matched_indices)

        call_ids = [
            predicted.get("tool_call_id")
            or (
                f"call_{state.task.position}_{state.sample_index}_"
                f"{state.next_turn}_{call_index}"
            )
            for call_index, predicted in enumerate(predicted_calls, 1)
        ]
        state.messages.append(
            {
                "role": "assistant",
                "content": event.get("raw_completion") or "",
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
        )
        for call_id, predicted, tool_output in zip(
            call_ids, predicted_calls, tool_outputs
        ):
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": predicted["name"],
                    "content": json.dumps(tool_output, ensure_ascii=False),
                }
            )

        if len(state.matched_gold_step_indices) == len(state.task.gold_steps):
            state.status = "success"
            return

        matched_turn_index = state.task.gold_steps[matched_step_index]["turn_index"]
        next_step = next(state.unmatched_gold_steps())[1]
        next_turn_index = next_step["turn_index"]
        if next_turn_index > matched_turn_index:
            assistant_response = state.task.user_turns[matched_turn_index].get(
                "assistant_response"
            )
            if assistant_response:
                state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
            state.messages.append(
                {
                    "role": "user",
                    "content": state.task.user_turns[next_turn_index]["query"],
                }
            )

    def run(
        self,
        tasks: Sequence[Task],
        *,
        states: list[RolloutState] | None = None,
        on_event: Callable[[dict[str, Any], RolloutState], None] | None = None,
        on_progress: Callable[[dict[str, int]], None] | None = None,
    ) -> list[RolloutState]:
        states = states or build_states(tasks, self.pass_k, self.include_initial_state)
        while True:
            active = [state for state in states if state.status == "active"]
            if not active:
                break
            current_turn = min(state.next_turn for state in active)
            turn_states = [state for state in active if state.next_turn == current_turn]
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
                                "turn": current_turn,
                                "processed": processed,
                                "turn_total": len(turn_states),
                                "active": counts["active"],
                                "success": counts["success"],
                                "failed": counts["failed"],
                                "total": len(states),
                            }
                        )
        return states


def rollout_record(state: RolloutState) -> dict[str, Any]:
    return {
        "row_position": state.task.position,
        "sample_index": state.sample_index,
        "status": state.status,
        "success": state.status == "success",
        "failure": state.failure,
        "matched_steps": len(state.matched_gold_step_indices),
        "matched_calls": len(state.matched_gold_indices),
        "matched_gold_step_indices": sorted(state.matched_gold_step_indices),
        "matched_gold_indices": sorted(state.matched_gold_indices),
        "num_gold_steps": len(state.task.gold_steps),
        "num_gold_calls": len(state.task.gold_calls),
        "data_issues": state.task.data_issues,
        "calls": [turn.get("predicted_calls") or [] for turn in state.turns],
    }


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate from n samples with c successes."""
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n], got k={k}, n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _task_behavior(task: Task) -> str:
    has_refusal = any(
        step["execution_mode"] == "refusal" for step in task.gold_steps
    )
    has_parallel = any(
        step["execution_mode"] == "parallel" for step in task.gold_steps
    )
    if has_refusal and has_parallel:
        return "combined_refusal_parallel"
    if has_refusal:
        return "refusal_or_clarification"
    if has_parallel:
        return "parallel"
    if len(task.user_turns) > 1:
        return "ordinary_multi_turn"
    return "ordinary_single_turn"


def _task_source(task: Task) -> str:
    aggregation = task.raw.get("aggregation_metadata") or {}
    return str(aggregation.get("source_dataset") or "unaggregated")


def _task_training_eligible(task: Task) -> bool:
    aggregation = task.raw.get("aggregation_metadata") or {}
    value = aggregation.get("eligible_for_sft_rl")
    return bool(value) if isinstance(value, bool) else True


def summarize(
    tasks: Sequence[Task], states: Sequence[RolloutState], pass_k: int
) -> dict[str, Any]:
    grouped: dict[int, list[RolloutState]] = defaultdict(list)
    for state in states:
        grouped[state.task.position].append(state)
    for samples in grouped.values():
        samples.sort(key=lambda item: item.sample_index)

    task_by_position = {task.position: task for task in tasks}
    clean_positions = {task.position for task in tasks if not task.data_issues}
    task_results = []
    for position, samples in sorted(grouped.items()):
        n = len(samples)
        c = sum(state.status == "success" for state in samples)
        task = task_by_position[position]
        task_results.append(
            {
                "row_position": position,
                "source_dataset": _task_source(task),
                "eligible_for_sft_rl": _task_training_eligible(task),
                "behavior": _task_behavior(task),
                "focus_category": task.focus_category or "Unknown",
                "num_samples": n,
                "successful_rollouts": c,
                "rollout_success_rate": c / max(n, 1),
                "pass_at_1": estimate_pass_at_k(n, c, 1),
                f"pass_at_{pass_k}": estimate_pass_at_k(
                    n, c, min(pass_k, n)
                ),
                f"empirical_prefix_pass_at_{pass_k}": any(
                    state.status == "success" for state in samples[:pass_k]
                ),
            }
        )

    curve = []
    for k in range(1, pass_k + 1):
        all_values = [
            estimate_pass_at_k(
                len(samples),
                sum(state.status == "success" for state in samples),
                k,
            )
            for samples in grouped.values()
        ]
        clean_values = [
            estimate_pass_at_k(
                len(grouped[position]),
                sum(
                    state.status == "success"
                    for state in grouped[position]
                ),
                k,
            )
            for position in sorted(clean_positions)
        ]
        prefix_values = [
            any(state.status == "success" for state in samples[:k])
            for samples in grouped.values()
        ]
        clean_prefix_values = [
            any(
                state.status == "success"
                for state in grouped[position][:k]
            )
            for position in sorted(clean_positions)
        ]
        curve.append(
            {
                "k": k,
                "pass_at_k_all": sum(all_values) / max(len(all_values), 1),
                "pass_at_k_clean": sum(clean_values) / max(len(clean_values), 1),
                "empirical_prefix_pass_at_k_all": (
                    sum(prefix_values) / max(len(prefix_values), 1)
                ),
                "empirical_prefix_pass_at_k_clean": (
                    sum(clean_prefix_values)
                    / max(len(clean_prefix_values), 1)
                ),
            }
        )

    def grouped_metrics(key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in task_results:
            buckets[str(result[key])].append(result)
        return {
            label: {
                "tasks": len(values),
                "successful_rollouts": sum(
                    int(value["successful_rollouts"]) for value in values
                ),
                "rollout_success_rate": (
                    sum(float(value["rollout_success_rate"]) for value in values)
                    / len(values)
                ),
                "pass_at_1": (
                    sum(float(value["pass_at_1"]) for value in values)
                    / len(values)
                ),
                f"pass_at_{pass_k}": (
                    sum(float(value[f"pass_at_{pass_k}"]) for value in values)
                    / len(values)
                ),
            }
            for label, values in sorted(buckets.items())
        }

    failure_counts = Counter(
        state.failure for state in states if state.status == "failed"
    )
    issue_counts = Counter(issue for task in tasks for issue in task.data_issues)
    category_tasks = Counter(task.focus_category or "Unknown" for task in tasks)
    solved_positions = {
        position
        for position, samples in grouped.items()
        if any(state.status == "success" for state in samples)
    }
    category_solved = Counter(
        task.focus_category or "Unknown"
        for task in tasks
        if task.position in solved_positions
    )
    return {
        "num_tasks": len(tasks),
        "num_clean_tasks": len(clean_positions),
        "num_rollouts": len(states),
        "num_successful_rollouts": sum(state.status == "success" for state in states),
        "pass_at_1_all": curve[0]["pass_at_k_all"] if curve else 0.0,
        f"pass_at_{pass_k}_all": curve[-1]["pass_at_k_all"] if curve else 0.0,
        "pass_at_1_clean": curve[0]["pass_at_k_clean"] if curve else 0.0,
        f"pass_at_{pass_k}_clean": curve[-1]["pass_at_k_clean"] if curve else 0.0,
        "failure_counts": dict(failure_counts),
        "data_issue_task_counts": dict(issue_counts),
        "category_tasks": dict(category_tasks),
        f"category_solved_at_{pass_k}": dict(category_solved),
        "source_metrics": grouped_metrics("source_dataset"),
        "behavior_metrics": grouped_metrics("behavior"),
        "eligibility_metrics": grouped_metrics("eligible_for_sft_rl"),
        "pass_curve": curve,
        "task_results": task_results,
    }


def load_events(path: Path, states: Sequence[RolloutState]) -> int:
    if not path.exists():
        return 0
    by_key = {(state.task.position, state.sample_index): state for state in states}
    count = 0
    for line_number, event in enumerate(read_jsonl(path), 1):
        key = int(event["row_position"]), int(event["sample_index"])
        if key not in by_key:
            raise ValueError(f"Unknown event key at line {line_number}: {key}")
        InteractivePassKChecker.apply_event(by_key[key], event)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jsonl", required=True, help="APIGen trajectory JSONL")
    parser.add_argument("--tool-pool", default=str(DEFAULT_TOOL_POOL))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tool-scope",
        choices=["category", "gold", "all", "declared"],
        default="category",
        help="Which tools the policy sees for each task",
    )
    parser.add_argument("--pass-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--include-initial-state",
        action="store_true",
        help="Expose generator state to the policy (off for policy-visible checks)",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Require the next gold call specifically; default mirrors the old unordered matcher",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--model", help="Served model ID; omitted means discover via /v1/models"
    )
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=2)

    launch = parser.add_argument_group("optional managed vLLM server")
    launch.add_argument(
        "--launch-vllm",
        metavar="MODEL_PATH",
        help=f"Launch vLLM for this run (Qwen default path: {DEFAULT_QWEN36_27B})",
    )
    launch.add_argument("--vllm-python", default=sys.executable)
    launch.add_argument("--served-model-name", default="qwen3.6-27b")
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=8000)
    launch.add_argument("--tensor-parallel-size", type=int, default=4)
    launch.add_argument("--max-model-len", type=int, default=49152)
    launch.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    launch.add_argument("--tool-call-parser", default="qwen3_xml")
    launch.add_argument("--reasoning-parser", default="qwen3")
    launch.add_argument("--startup-timeout", type=float, default=900.0)
    launch.add_argument("--cuda-visible-devices")
    launch.add_argument(
        "--vllm-extra-arg",
        action="append",
        default=[],
        help="One extra api_server argument; repeat for multiple tokens",
    )
    launch.add_argument("--reuse-running-server", action="store_true")

    output = parser.add_argument_group("run control")
    output.add_argument("--resume", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pass_k < 1:
        parser.error("--pass-k must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.launch_vllm and args.vllm_url == "http://127.0.0.1:8000/v1":
        args.vllm_url = f"http://{args.host}:{args.port}/v1"

    tasks, _ = load_tasks(
        args.jsonl,
        args.tool_pool,
        tool_scope=args.tool_scope,
        max_samples=args.max_samples,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    validation = {
        "tasks": len(tasks),
        "steps": sum(len(task.gold_steps) for task in tasks),
        "calls": sum(len(task.gold_calls) for task in tasks),
        "step_ordered_tasks": sum(task.step_order_matters for task in tasks),
        "order_invariant_parallel_steps": sum(
            step["execution_mode"] == "parallel"
            and not step["call_order_matters"]
            for task in tasks
            for step in task.gold_steps
        ),
        "certified_refusal_steps": sum(
            step["execution_mode"] == "refusal"
            for task in tasks
            for step in task.gold_steps
        ),
        "multi_turn_tasks": sum(len(task.user_turns) > 1 for task in tasks),
        "source_tasks": dict(
            Counter(_task_source(task) for task in tasks)
        ),
        "training_eligible_tasks": sum(
            _task_training_eligible(task) for task in tasks
        ),
        "clean_tasks": sum(not task.data_issues for task in tasks),
        "issue_counts": dict(
            Counter(issue for task in tasks for issue in task.data_issues)
        ),
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
        raise FileExistsError(f"{events_path} exists; pass --resume or --overwrite")

    sampling = SamplingConfig(
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
        "jsonl": str(Path(args.jsonl).resolve()),
        "tool_pool": str(Path(args.tool_pool).resolve()),
        "tool_scope": args.tool_scope,
        "pass_k": args.pass_k,
        "sampling": asdict(sampling),
        "workers": args.workers,
        "include_initial_state": args.include_initial_state,
        "ordered": args.ordered,
        "parallel_tool_calls": "always_enabled_no_gold_hint",
        "vllm_url": _normalise_vllm_url(args.vllm_url),
        "model": args.model,
        "launch_vllm": args.launch_vllm,
        "served_model_name": args.served_model_name,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "evaluation": (
            "interactive_exact_ordered_step_replay"
            if args.ordered
            else "interactive_exact_unordered_step_replay"
        ),
        "data_validation": validation,
    }
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume configuration does not match config.json")
    atomic_write_json(config_path, config)

    states = build_states(tasks, args.pass_k, args.include_initial_state)
    resumed = load_events(events_path, states) if args.resume else 0
    if resumed:
        print(f"Resumed {resumed} events", flush=True)

    client = VLLMClient(
        args.vllm_url,
        args.model,
        args.api_key,
        timeout=args.request_timeout,
        retries=args.request_retries,
    )
    server = ManagedVLLMServer(
        launch_model=args.launch_vllm,
        client=client,
        out_dir=out_dir,
        python=args.vllm_python,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser or None,
        startup_timeout=args.startup_timeout,
        cuda_visible_devices=args.cuda_visible_devices,
        extra_args=args.vllm_extra_arg,
        reuse_running_server=args.reuse_running_server,
    )

    checker = InteractivePassKChecker(
        client,
        pass_k=args.pass_k,
        sampling=sampling,
        workers=args.workers,
        include_initial_state=args.include_initial_state,
        ordered=args.ordered,
    )
    event_mode = "a" if args.resume else "w"
    with contextlib.ExitStack() as stack:
        active_client = stack.enter_context(server)
        checker.client = active_client
        event_file = stack.enter_context(events_path.open(event_mode, encoding="utf-8"))

        def on_event(event: dict[str, Any], _state: RolloutState) -> None:
            event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            event_file.flush()

        def on_progress(progress: dict[str, int]) -> None:
            atomic_write_json(out_dir / "progress.json", progress)
            if progress["processed"] == progress["turn_total"]:
                print(json.dumps(progress), flush=True)

        checker.run(tasks, states=states, on_event=on_event, on_progress=on_progress)

    with (out_dir / "rollouts.jsonl").open("w", encoding="utf-8") as output:
        for state in states:
            output.write(json.dumps(rollout_record(state), ensure_ascii=False) + "\n")
    summary = summarize(tasks, states, args.pass_k)
    summary["config"] = config
    atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
