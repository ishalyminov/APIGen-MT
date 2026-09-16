import copy

import pytest

from refuse_parallel_eval import (
    estimate_pass_at_k,
    match_internal_refusal,
    match_parallel_calls,
    prepare_multiturn_datapoint,
    validate_feature_evaluation_spec,
)


def _call(name, arguments, output):
    return {
        "tool_name": name,
        "arguments": arguments,
        "output": output,
    }


def _step(number, calls):
    is_refusal = len(calls) == 1 and calls[0]["tool_name"] == "refuse"
    is_parallel = len(calls) > 1
    return {
        "step_number": number,
        "tool_calls": calls,
        "execution_mode": (
            "refusal" if is_refusal else "parallel" if is_parallel else "sequential"
        ),
        "call_order_matters": not is_parallel,
        "quality_verification": {"passed": True},
    }


def _feature_datapoint():
    return {
        "conversation": {
            "overall_task": "Stale blueprint task",
            "turns": [
                {
                    "turn_number": 1,
                    "user_query": "Look up alpha.",
                    "steps": [
                        _step(1, [_call("lookup", {"key": "alpha"}, {"value": "A"})])
                    ],
                    "assistant_response": "Alpha is A.",
                    "expected_tools": ["lookup"],
                    # Deliberately leaked future output from a shallow copy.
                    "execution_context": {
                        "turn_outputs": [
                            {"lookup": {"value": "A"}},
                            {"lookup": [{"value": "B"}, {"value": "C"}]},
                        ]
                    },
                    "quality_verification": {"passed": True},
                },
                {
                    "turn_number": 2,
                    "user_query": "Look up beta and gamma at the same time.",
                    "steps": [
                        _step(
                            1,
                            [
                                _call("lookup", {"key": "beta"}, {"value": "B"}),
                                _call("lookup", {"key": "gamma"}, {"value": "C"}),
                            ],
                        )
                    ],
                    "assistant_response": "Beta is B and gamma is C.",
                    "expected_tools": ["lookup", "lookup"],
                    "execution_context": {"parallel_batches": [{"calls": []}]},
                    "quality_verification": {"passed": True},
                },
                {
                    "turn_number": 3,
                    "user_query": "Archive both files as a ZIP.",
                    "steps": [
                        _step(
                            1,
                            [
                                _call(
                                    "refuse",
                                    {"reason": "no_appropriate_function"},
                                    {
                                        "status": "refused",
                                        "reason": "no_appropriate_function",
                                    },
                                )
                            ],
                        )
                    ],
                    "assistant_response": (
                        "I cannot create a ZIP archive because that capability "
                        "is not available."
                    ),
                    "expected_tools": ["refuse"],
                    "execution_context": {"refusal": {"status": "refused"}},
                    "quality_verification": {"passed": True},
                },
            ],
            "tools_used": ["lookup", "refuse"],
            "categories_used": ["Demo", "Control"],
            "initial_api_state": {"hidden": {"value": 1}},
        },
        "generation_metadata": {
            "overall_task": "Stale blueprint task",
            "blueprint_queries": [
                "Look up alpha.",
                "Old second turn.",
                "Old third turn.",
            ],
            "turn_expected_tools": [["lookup"], ["old"], ["old"]],
            "parallel_order_invariant": True,
            "rl_quality_gate_passed": True,
        },
        "verification_result": {
            "overall_verification_passed": True,
            "rl_quality_gate": {"passed": True},
        },
        "available_tools": [
            {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
            {
                "name": "refuse",
                "synthetic_terminal_tool": True,
                "parameters": {"type": "object"},
            },
        ],
    }


def test_parallel_match_is_one_action_unordered_multiset_with_multiplicity():
    gold = [
        {"name": "lookup", "arguments": {"key": "a"}},
        {"name": "lookup", "arguments": {"key": "b"}},
    ]
    reversed_one_action = [
        {
            "tool_calls": [
                {"name": "lookup", "arguments": {"key": "b"}},
                {"name": "lookup", "arguments": {"key": "a"}},
            ]
        }
    ]
    assert match_parallel_calls(reversed_one_action, gold)
    assert not match_parallel_calls(
        [
            {"tool_calls": [{"name": "lookup", "arguments": {"key": "a"}}]},
            {"tool_calls": [{"name": "lookup", "arguments": {"key": "b"}}]},
        ],
        gold,
    )
    assert not match_parallel_calls(
        [
            {
                "tool_calls": [
                    {"name": "lookup", "arguments": {"key": "a"}},
                    {"name": "lookup", "arguments": {"key": "a"}},
                ]
            }
        ],
        gold,
    )


def test_internal_refusal_match_is_stable_reason_only():
    assert match_internal_refusal(
        [
            {
                "tool_calls": [
                    {
                        "name": "refuse",
                        "arguments": {"reason": "missing_argument"},
                    }
                ]
            }
        ],
        "missing_argument",
    )
    assert not match_internal_refusal([], "missing_argument")


def test_standard_pass_at_k_estimator():
    assert estimate_pass_at_k(10, 0, 5) == 0.0
    assert estimate_pass_at_k(10, 10, 5) == 1.0
    assert estimate_pass_at_k(10, 1, 10) == 1.0
    assert estimate_pass_at_k(10, 1, 1) == pytest.approx(0.1)


def test_multiturn_postprocessing_removes_leaks_and_scopes_parallelism():
    datapoint = _feature_datapoint()
    prepared = prepare_multiturn_datapoint(copy.deepcopy(datapoint))
    metadata = prepared["generation_metadata"]
    turns = prepared["conversation"]["turns"]

    assert metadata["source_blueprint_queries"][1] == "Old second turn."
    assert metadata["blueprint_queries"] == [
        turn["user_query"] for turn in turns
    ]
    assert metadata["feature_rewritten_turns"] == [2, 3]
    assert metadata["parallel_order_invariant"] is False
    assert metadata["parallel_order_invariance_scope"] == "per_transition_only"
    assert metadata["parallel_groups"] == [
        {
            "transition_id": "t2_s1",
            "turn_number": 2,
            "step_number": 1,
            "call_count": 2,
            "order_invariant": True,
            "must_be_same_assistant_action": True,
        }
    ]

    for turn in turns:
        assert turn["execution_context"]["contains_current_turn_outputs"] is False
        assert turn["execution_context"]["contains_future_turn_outputs"] is False
        assert "turn_outputs" not in turn["execution_context"]
        assert "parallel_batches" not in turn["execution_context"]
        assert "refusal" not in turn["execution_context"]

    spec = metadata["evaluation_spec"]
    parallel = next(item for item in spec["transitions"] if item["mode"] == "parallel")
    assert parallel["matching"]["tool_calls"] == "unordered_multiset"
    assert parallel["matching"]["must_be_same_assistant_action"] is True
    assert len(parallel["internal_target"]["tool_calls"]) == 2

    # The current target and results are absent from its policy context.
    target_ids = {call["id"] for call in parallel["internal_target"]["tool_calls"]}
    assert all(
        message.get("tool_call_id") not in target_ids
        for message in parallel["policy_messages"]
    )
    assert all(
        call.get("id") not in target_ids
        for message in parallel["policy_messages"]
        for call in message.get("tool_calls", [])
    )

    refusal = next(
        item for item in spec["transitions"] if item["mode"] == "refusal"
    )
    assert refusal["bfcl_native_target"]["tool_calls"] == []
    assert refusal["bfcl_native_target"]["reason"] == "no_appropriate_function"
    assert "ZIP" in refusal["bfcl_native_target"]["assistant_response"]
    assert validate_feature_evaluation_spec(prepared) == []


def test_native_history_after_clarification_omits_synthetic_refuse_messages():
    datapoint = _feature_datapoint()
    turns = datapoint["conversation"]["turns"]
    turns.append(
        {
            "turn_number": 4,
            "user_query": "Use customer key alpha and go ahead with the lookup.",
            "steps": [
                _step(
                    1,
                    [_call("lookup", {"key": "alpha"}, {"value": "A"})],
                )
            ],
            "assistant_response": "The value is A.",
            "expected_tools": ["lookup"],
            "execution_context": {},
            "quality_verification": {"passed": True},
        }
    )

    prepared = prepare_multiturn_datapoint(copy.deepcopy(datapoint))
    recovery = next(
        item
        for item in prepared["generation_metadata"]["evaluation_spec"][
            "transitions"
        ]
        if item["transition_id"] == "t4_s1"
    )
    native_messages = recovery["bfcl_native_policy_messages"]
    assert any(
        message.get("role") == "assistant"
        and "ZIP" in message.get("content", "")
        for message in native_messages
    )
    assert not any(
        call.get("function", {}).get("name") == "refuse"
        for message in native_messages
        for call in message.get("tool_calls", [])
    )
    assert not any(
        message.get("role") == "tool"
        and message.get("name") == "refuse"
        for message in native_messages
    )
    assert validate_feature_evaluation_spec(prepared) == []


def test_openai_native_calls_normalise_before_parallel_matching():
    gold = [
        {"name": "lookup", "arguments": {"key": "a"}},
        {"name": "lookup", "arguments": {"key": "b"}},
    ]
    predicted = [
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key":"b"}',
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key":"a"}',
                    },
                },
            ]
        }
    ]
    assert match_parallel_calls(predicted, gold)


def test_openai_native_refuse_call_matches_internal_reason():
    predicted = [
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "refuse",
                        "arguments": '{"reason":"ambiguity"}',
                    },
                }
            ]
        }
    ]
    assert match_internal_refusal(predicted, "ambiguity")
