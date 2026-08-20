"""Serialization and evaluation helpers for refusal/parallel trajectories.

The generator keeps a synthetic ``refuse`` tool internally because it gives RL a
stable discrete target.  This module additionally emits a benchmark-native view
where refusal/clarification means *no real tool call* plus a grounded response.

Parallel groups are scoped per transition.  Only calls inside the same group are
order-invariant; the surrounding episode remains ordered.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence


EVALUATION_SPEC_VERSION = "refuse-parallel-v3"
SUPPORTED_EVALUATION_SPEC_VERSIONS = {
    "refuse-parallel-v2",
    EVALUATION_SPEC_VERSION,
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def normalise_tool_call(call: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise internal and OpenAI-native function-call representations.

    Accepted examples include ``{"name": ..., "arguments": {...}}`` and
    ``{"function": {"name": ..., "arguments": "{...}"}}``.  A
    malformed JSON argument string remains a raw string and therefore cannot
    accidentally match a structured gold argument object.
    """

    function = call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name", call.get("name", call.get("tool_name", "")))
        arguments = function.get("arguments", call.get("arguments", {}))
    else:
        name = call.get("name", call.get("tool_name", ""))
        arguments = call.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {"name": str(name), "arguments": arguments}


def canonical_tool_call(call: Mapping[str, Any]) -> str:
    return canonical_json(normalise_tool_call(call))


def tool_call_multiset(calls: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(canonical_tool_call(call) for call in calls)


def match_parallel_calls(
    predicted_actions: Sequence[Mapping[str, Any]],
    gold_calls: Sequence[Mapping[str, Any]],
) -> bool:
    """Match one parallel assistant action as an unordered multiset.

    ``predicted_actions`` is a sequence of assistant actions.  Each action must
    contain a ``tool_calls`` list.  A parallel target succeeds only when exactly
    one assistant action contains the entire call multiset.  Emitting one sibling,
    observing its output, and emitting the next sibling is therefore rejected.
    """

    if len(predicted_actions) != 1:
        return False
    calls = predicted_actions[0].get("tool_calls", [])
    if not isinstance(calls, list):
        return False
    return tool_call_multiset(calls) == tool_call_multiset(gold_calls)


def match_internal_refusal(
    predicted_actions: Sequence[Mapping[str, Any]],
    reason: str,
) -> bool:
    if len(predicted_actions) != 1:
        return False
    calls = predicted_actions[0].get("tool_calls", [])
    if not isinstance(calls, list) or len(calls) != 1:
        return False
    call = normalise_tool_call(calls[0])
    return (
        call.get("name") == "refuse"
        and call.get("arguments") == {"reason": reason}
    )


def _normalise_text(text: str) -> str:
    return " ".join((text or "").casefold().split())


def conservative_native_refusal_match(response: str, reason: str) -> bool:
    """Conservative lexical prefilter for benchmark-native refusal.

    For authoritative benchmark scoring, an external semantic judge may set a
    rollout-level ``semantic_refusal_match`` boolean.  This helper is intentionally
    conservative and is primarily useful for smoke tests.
    """

    text = _normalise_text(response)
    if not text:
        return False
    if reason == "no_appropriate_function":
        inability = any(
            phrase in text
            for phrase in (
                "can't",
                "cannot",
                "unable",
                "not available",
                "no available",
                "don't have",
                "do not have",
            )
        )
        capability = any(
            word in text
            for word in ("tool", "capability", "function", "option", "support")
        )
        return inability and capability
    if reason == "missing_argument":
        asks = "?" in response or any(
            phrase in text
            for phrase in ("please provide", "what is", "which", "could you provide")
        )
        missing = any(
            word in text
            for word in ("need", "missing", "provide", "specify", "which", "what")
        )
        return asks and missing
    if reason == "ambiguity":
        asks = "?" in response or any(
            phrase in text for phrase in ("please clarify", "which do you mean", "which one")
        )
        ambiguity = any(
            word in text
            for word in ("clarify", "which", "ambiguous", "mean", "choose", "option")
        )
        return asks and ambiguity
    return False


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator used by code-generation benchmarks."""

    if n < 0 or c < 0 or c > n:
        raise ValueError(f"Invalid counts: n={n}, c={c}")
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n; got k={k}, n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _object_value(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _set_object_value(obj: Any, field: str, value: Any) -> None:
    if isinstance(obj, MutableMapping):
        obj[field] = value
    else:
        setattr(obj, field, value)


def _call_target(call: Any, *, call_id: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "name": str(_object_value(call, "tool_name", "")),
        "arguments": copy.deepcopy(_object_value(call, "arguments", {}) or {}),
    }


def _assistant_tool_message(calls: Sequence[Any], *, prefix: str) -> Dict[str, Any]:
    tool_calls = []
    for index, call in enumerate(calls, 1):
        call_id = f"{prefix}c{index}"
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(_object_value(call, "tool_name", "")),
                    "arguments": canonical_json(
                        _object_value(call, "arguments", {}) or {}
                    ),
                },
            }
        )
    return {"role": "assistant", "tool_calls": tool_calls}


def _tool_result_messages(calls: Sequence[Any], *, prefix: str) -> List[Dict[str, Any]]:
    messages = []
    for index, call in enumerate(calls, 1):
        call_id = f"{prefix}c{index}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": str(_object_value(call, "tool_name", "")),
                "content": canonical_json(_object_value(call, "output")),
            }
        )
    return messages


def _response_rubric(reason: str) -> Dict[str, Any]:
    if reason == "missing_argument":
        return {
            "reason": reason,
            "must_make_real_tool_calls": False,
            "must_ask_targeted_question": True,
            "must_not_invent_missing_value": True,
        }
    if reason == "ambiguity":
        return {
            "reason": reason,
            "must_make_real_tool_calls": False,
            "must_ask_targeted_question": True,
            "must_request_material_disambiguation": True,
        }
    return {
        "reason": reason,
        "must_make_real_tool_calls": False,
        "must_explain_capability_unavailable": True,
    }


def _transition(
    *,
    transition_id: str,
    turn_number: int,
    step_number: int,
    messages: Sequence[Mapping[str, Any]],
    calls: Sequence[Any],
    assistant_response: str,
    execution_mode: str | None = None,
    call_order_matters: bool | None = None,
) -> Dict[str, Any]:
    targets = [
        _call_target(call, call_id=f"t{turn_number}s{step_number}c{index}")
        for index, call in enumerate(calls, 1)
    ]
    is_refusal = (
        len(targets) == 1
        and targets[0]["name"] == "refuse"
    )
    if is_refusal:
        reason = str(targets[0]["arguments"].get("reason", "no_appropriate_function"))
        mode = "clarification" if reason in {"missing_argument", "ambiguity"} else "refusal"
        matching = {
            "internal": {
                "tool_calls": "exact_single_refuse_reason",
                "must_be_same_assistant_action": True,
            },
            "bfcl_native": {
                "tool_calls": "none",
                "assistant_response": "semantic",
                "rubric": _response_rubric(reason),
            },
        }
        native_target = {
            "tool_calls": [],
            "assistant_response": assistant_response,
            "reason": reason,
        }
    explicit_parallel = (
        execution_mode == "parallel"
        or (call_order_matters is False and len(targets) > 1)
    )
    if execution_mode is None:
        execution_mode = "parallel" if explicit_parallel else "sequential"
    if call_order_matters is None:
        call_order_matters = not explicit_parallel

    if is_refusal:
        execution_mode = "refusal"
        call_order_matters = True
    elif explicit_parallel:
        mode = "parallel"
        matching = {
            "tool_calls": "unordered_multiset",
            "must_be_same_assistant_action": True,
            "preserve_duplicate_multiplicity": True,
            "surrounding_episode_ordered": True,
        }
        native_target = {"tool_calls": copy.deepcopy(targets)}
    else:
        mode = "sequential"
        matching = {
            "tool_calls": "ordered_exact",
            "must_be_same_assistant_action": True,
            "surrounding_episode_ordered": True,
        }
        native_target = {"tool_calls": copy.deepcopy(targets)}

    return {
        "transition_id": transition_id,
        "turn_number": turn_number,
        "step_number": step_number,
        "mode": mode,
        "execution_mode": execution_mode,
        "call_order_matters": call_order_matters,
        "policy_messages": copy.deepcopy(list(messages)),
        "internal_target": {
            "tool_calls": copy.deepcopy(targets),
            "assistant_response": assistant_response if is_refusal else None,
        },
        "bfcl_native_target": native_target,
        "matching": matching,
    }


def build_multiturn_evaluation_spec(
    conversation: Any,
    *,
    available_tools: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    internal_transcript: List[Dict[str, Any]] = []
    native_transcript: List[Dict[str, Any]] = []

    for turn in _object_value(conversation, "turns", []) or []:
        turn_number = int(_object_value(turn, "turn_number", len(transitions) + 1))
        user_query = str(_object_value(turn, "user_query", ""))
        user_message = {"role": "user", "content": user_query}
        internal_transcript.append(copy.deepcopy(user_message))
        native_transcript.append(copy.deepcopy(user_message))
        steps = _object_value(turn, "steps", []) or []
        assistant_response = str(_object_value(turn, "assistant_response", ""))

        for step in steps:
            step_number = int(_object_value(step, "step_number", 1))
            calls = _object_value(step, "tool_calls", []) or []
            transition_id = f"t{turn_number}_s{step_number}"
            transition = _transition(
                    transition_id=transition_id,
                    turn_number=turn_number,
                    step_number=step_number,
                    messages=internal_transcript,
                    calls=calls,
                    assistant_response=assistant_response,
                    execution_mode=_object_value(step, "execution_mode"),
                    call_order_matters=_object_value(
                        step, "call_order_matters"
                    ),
                )
            transition["bfcl_native_policy_messages"] = copy.deepcopy(
                native_transcript
            )
            transitions.append(transition)
            prefix = f"t{turn_number}s{step_number}"
            assistant_tool_message = _assistant_tool_message(
                calls,
                prefix=prefix,
            )
            tool_result_messages = _tool_result_messages(
                calls,
                prefix=prefix,
            )
            internal_transcript.append(assistant_tool_message)
            internal_transcript.extend(tool_result_messages)
            if transition["mode"] not in {"refusal", "clarification"}:
                native_transcript.append(copy.deepcopy(assistant_tool_message))
                native_transcript.extend(copy.deepcopy(tool_result_messages))

        if assistant_response:
            response_message = {
                "role": "assistant",
                "content": assistant_response,
            }
            internal_transcript.append(copy.deepcopy(response_message))
            native_transcript.append(copy.deepcopy(response_message))

    return {
        "version": EVALUATION_SPEC_VERSION,
        "initial_api_state_is_policy_visible": False,
        "policy_context_source": "conversation.turns",
        "tool_contract_source": "datapoint.available_tools",
        "synthetic_refuse_tool_present": any(
            tool.get("name") == "refuse" for tool in available_tools
        ),
        "transitions": transitions,
    }


def build_step_by_step_evaluation_spec(
    trajectory: Any,
    *,
    available_tools: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    query = str(_object_value(trajectory, "query", ""))
    steps = _object_value(trajectory, "steps", []) or []
    final_response = str(_object_value(trajectory, "final_response", ""))
    transcript: List[Dict[str, Any]] = [{"role": "user", "content": query}]
    transitions: List[Dict[str, Any]] = []

    for step in steps:
        step_number = int(_object_value(step, "step_number", 1))
        calls = _object_value(step, "tool_calls", []) or []
        transitions.append(
            _transition(
                transition_id=f"t1_s{step_number}",
                turn_number=1,
                step_number=step_number,
                messages=transcript,
                calls=calls,
                assistant_response=final_response,
                execution_mode=_object_value(step, "execution_mode"),
                call_order_matters=_object_value(step, "call_order_matters"),
            )
        )
        prefix = f"t1s{step_number}"
        transcript.append(_assistant_tool_message(calls, prefix=prefix))
        transcript.extend(_tool_result_messages(calls, prefix=prefix))

    return {
        "version": EVALUATION_SPEC_VERSION,
        "initial_api_state_is_policy_visible": False,
        "policy_context_source": "trajectory.query_and_prior_steps",
        "tool_contract_source": "datapoint.available_tools",
        "synthetic_refuse_tool_present": any(
            tool.get("name") == "refuse" for tool in available_tools
        ),
        "transitions": transitions,
    }


def _actual_overall_task(queries: Sequence[str]) -> str:
    return "Actual conversation requests: " + " | ".join(
        f"Turn {index}: {query}" for index, query in enumerate(queries, 1)
    )


def prepare_multiturn_datapoint(datapoint: Any) -> Any:
    """Make a feature-enabled multi-turn datapoint safe for RL/evaluation."""

    if datapoint is None:
        return None
    conversation = _object_value(datapoint, "conversation")
    metadata = _object_value(datapoint, "generation_metadata", {})
    verification = _object_value(datapoint, "verification_result", {}) or {}
    available_tools = _object_value(datapoint, "available_tools", []) or []
    turns = _object_value(conversation, "turns", []) or []

    actual_queries = [str(_object_value(turn, "user_query", "")) for turn in turns]
    actual_expected = [
        list(_object_value(turn, "expected_tools", []) or []) for turn in turns
    ]
    source_queries = copy.deepcopy(
        metadata.get(
            "source_blueprint_queries",
            metadata.get("blueprint_queries", []),
        )
    )
    source_overall = metadata.get(
        "source_overall_task",
        metadata.get(
            "overall_task", _object_value(conversation, "overall_task", "")
        ),
    )
    rewritten = [
        index + 1
        for index, query in enumerate(actual_queries)
        if index >= len(source_queries) or source_queries[index] != query
    ]
    overall = _actual_overall_task(actual_queries)

    metadata["source_blueprint_queries"] = source_queries
    metadata["source_overall_task"] = source_overall
    metadata["blueprint_queries"] = copy.deepcopy(actual_queries)
    metadata["turn_expected_tools"] = copy.deepcopy(actual_expected)
    metadata["feature_rewritten_turns"] = rewritten
    metadata["overall_task"] = overall
    _set_object_value(conversation, "overall_task", overall)

    spec = build_multiturn_evaluation_spec(
        conversation,
        available_tools=available_tools,
    )
    metadata["evaluation_spec"] = spec
    metadata["policy_context_version"] = EVALUATION_SPEC_VERSION
    metadata["parallel_order_invariant"] = False
    metadata["parallel_order_invariance_scope"] = "per_transition_only"
    metadata["parallel_groups"] = [
        {
            "transition_id": item["transition_id"],
            "turn_number": item["turn_number"],
            "step_number": item["step_number"],
            "call_count": len(item["internal_target"]["tool_calls"]),
            "order_invariant": True,
            "must_be_same_assistant_action": True,
        }
        for item in spec["transitions"]
        if item["mode"] == "parallel"
    ]

    refusal_transitions = [
        item
        for item in spec["transitions"]
        if item["mode"] in {"refusal", "clarification"}
    ]
    metadata["refusal_transitions"] = [
        {
            "transition_id": item["transition_id"],
            "turn_number": item["turn_number"],
            "mode": item["mode"],
            "reason": item["bfcl_native_target"].get("reason"),
        }
        for item in refusal_transitions
    ]

    # ``Turn.execution_context`` used to be a shallow post-action snapshot and
    # could retroactively acquire current/future outputs.  It is debug metadata,
    # not a policy input.  Replace it with an explicit safe marker; exact policy
    # inputs live in ``evaluation_spec.transitions[*].policy_messages``.
    for turn in turns:
        turn_number = int(_object_value(turn, "turn_number", 0))
        transition_ids = [
            item["transition_id"]
            for item in spec["transitions"]
            if item["turn_number"] == turn_number
        ]
        _set_object_value(
            turn,
            "execution_context",
            {
                "policy_visible_only": True,
                "contains_current_turn_outputs": False,
                "contains_future_turn_outputs": False,
                "evaluation_transition_ids": transition_ids,
            },
        )

    if isinstance(verification, MutableMapping):
        gate = verification.setdefault("rl_quality_gate", {})
        if isinstance(gate, MutableMapping):
            gate["evaluation_spec_present"] = True
            gate["policy_context_leakage_checked"] = True
    return datapoint


def prepare_step_by_step_datapoint(datapoint: Any) -> Any:
    if datapoint is None:
        return None
    trajectory = _object_value(datapoint, "trajectory")
    metadata = _object_value(datapoint, "generation_metadata", {})
    verification = _object_value(datapoint, "verification_result", {}) or {}
    available_tools = _object_value(datapoint, "available_tools", []) or []
    spec = build_step_by_step_evaluation_spec(
        trajectory,
        available_tools=available_tools,
    )
    metadata["evaluation_spec"] = spec
    metadata["policy_context_version"] = EVALUATION_SPEC_VERSION
    metadata["parallel_order_invariant"] = False
    metadata["parallel_order_invariance_scope"] = "per_transition_only"
    metadata["parallel_groups"] = [
        {
            "transition_id": item["transition_id"],
            "turn_number": item["turn_number"],
            "step_number": item["step_number"],
            "call_count": len(item["internal_target"]["tool_calls"]),
            "order_invariant": True,
            "must_be_same_assistant_action": True,
        }
        for item in spec["transitions"]
        if item["mode"] == "parallel"
    ]
    if isinstance(verification, MutableMapping):
        gate = verification.setdefault("rl_quality_gate", {})
        if isinstance(gate, MutableMapping):
            gate["evaluation_spec_present"] = True
            gate["policy_context_leakage_checked"] = True
    return datapoint


def validate_feature_evaluation_spec(row: Mapping[str, Any]) -> List[str]:
    """Return deterministic dataset-audit errors for one serialized row."""

    errors: List[str] = []
    metadata = row.get("generation_metadata", {})
    spec = metadata.get("evaluation_spec")
    if not isinstance(spec, Mapping):
        return ["MISSING_EVALUATION_SPEC"]
    if spec.get("version") not in SUPPORTED_EVALUATION_SPEC_VERSIONS:
        errors.append("UNEXPECTED_EVALUATION_SPEC_VERSION")
    if spec.get("initial_api_state_is_policy_visible") is not False:
        errors.append("INITIAL_STATE_VISIBILITY_NOT_FALSE")
    transitions = spec.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        errors.append("MISSING_EVALUATION_TRANSITIONS")
        return errors

    for item in transitions:
        transition_id = str(item.get("transition_id", "unknown"))
        messages = item.get("policy_messages", [])
        if not isinstance(messages, list):
            errors.append(f"{transition_id}:INVALID_POLICY_MESSAGES")
            continue
        target_ids = {
            call.get("id")
            for call in item.get("internal_target", {}).get("tool_calls", [])
        }
        for message in messages:
            if message.get("role") == "tool" and message.get("tool_call_id") in target_ids:
                errors.append(f"{transition_id}:TARGET_RESULT_LEAK")
            for call in message.get("tool_calls", []) or []:
                if call.get("id") in target_ids:
                    errors.append(f"{transition_id}:TARGET_CALL_LEAK")
        native_messages = item.get(
            "bfcl_native_policy_messages",
            messages,
        )
        if not isinstance(native_messages, list):
            errors.append(f"{transition_id}:INVALID_NATIVE_POLICY_MESSAGES")
        else:
            for message in native_messages:
                for call in message.get("tool_calls", []) or []:
                    function = call.get("function", {})
                    if function.get("name") == "refuse":
                        errors.append(
                            f"{transition_id}:NATIVE_CONTEXT_HAS_REFUSE_CALL"
                        )
                if (
                    message.get("role") == "tool"
                    and message.get("name") == "refuse"
                ):
                    errors.append(
                        f"{transition_id}:NATIVE_CONTEXT_HAS_REFUSE_RESULT"
                    )

        mode = item.get("mode")
        matching = item.get("matching", {})
        if mode == "parallel":
            if item.get("execution_mode") != "parallel":
                errors.append(f"{transition_id}:PARALLEL_MODE_NOT_EXPLICIT")
            if item.get("call_order_matters") is not False:
                errors.append(f"{transition_id}:PARALLEL_ORDER_NOT_FALSE")
            if matching.get("tool_calls") != "unordered_multiset":
                errors.append(f"{transition_id}:PARALLEL_MATCHER_NOT_MULTISET")
            if matching.get("must_be_same_assistant_action") is not True:
                errors.append(f"{transition_id}:PARALLEL_NOT_SAME_ACTION")
        if mode in {"refusal", "clarification"}:
            if item.get("execution_mode") != "refusal":
                errors.append(f"{transition_id}:REFUSAL_MODE_NOT_EXPLICIT")
            native = item.get("bfcl_native_target", {})
            if native.get("tool_calls") != []:
                errors.append(f"{transition_id}:NATIVE_REFUSAL_HAS_TOOL_CALL")
            if not native.get("assistant_response"):
                errors.append(f"{transition_id}:NATIVE_REFUSAL_MISSING_RESPONSE")

    if metadata.get("parallel_order_invariant") is True:
        errors.append("EPISODE_WIDE_PARALLEL_INVARIANCE")

    conversation = row.get("conversation")
    if isinstance(conversation, Mapping):
        turns = conversation.get("turns", [])
        actual_queries = [turn.get("user_query", "") for turn in turns]
        if metadata.get("blueprint_queries") != actual_queries:
            errors.append("STALE_BLUEPRINT_QUERIES")
        for turn in turns:
            context = turn.get("execution_context", {})
            if context.get("contains_current_turn_outputs") is not False:
                errors.append("UNSAFE_TURN_EXECUTION_CONTEXT")
            if context.get("contains_future_turn_outputs") is not False:
                errors.append("UNSAFE_TURN_EXECUTION_CONTEXT")
    return errors
