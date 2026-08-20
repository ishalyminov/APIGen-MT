import json
import os
from pathlib import Path

import pytest

import check_apigen_trajectories_passk as checker


def external_fixture(name):
    default_root = (
        Path(__file__).resolve().parents[4] / "tool_synth" / "APIGen-MT-main"
    )
    root = Path(os.environ.get("APIGEN_PASSK_FIXTURE_ROOT", default_root))
    path = root / "data" / "generated" / name
    if not path.is_file():
        pytest.skip(
            "large pass@k integration fixture is external; set "
            "APIGEN_PASSK_FIXTURE_ROOT"
        )
    return path


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def step(calls, *, order_matters):
    return {
        "calls": calls,
        "call_order_matters": order_matters,
        "execution_mode": "sequential" if order_matters else "parallel",
    }


def test_parallel_step_matches_as_an_order_invariant_multiset():
    gold = step(
        [call("lookup", key="alpha"), call("lookup", key="beta")],
        order_matters=False,
    )
    predicted = [call("lookup", key="beta"), call("lookup", key="alpha")]
    assert checker.InteractivePassKChecker._exact_step_match(predicted, gold)


def test_sequential_call_order_is_not_interchangeable():
    gold = step(
        [call("lookup", key="alpha"), call("lookup", key="beta")],
        order_matters=True,
    )
    predicted = [call("lookup", key="beta"), call("lookup", key="alpha")]
    assert not checker.InteractivePassKChecker._exact_step_match(predicted, gold)


def test_parallel_multiset_preserves_duplicate_multiplicity():
    duplicate = call("lookup", key="alpha")
    gold = step([duplicate, duplicate], order_matters=False)
    assert not checker.InteractivePassKChecker._exact_step_match([duplicate], gold)
    assert checker.InteractivePassKChecker._exact_step_match(
        [duplicate, duplicate], gold
    )


def test_uncertified_parallel_label_does_not_disable_ordering():
    raw = {
        "conversation": {
            "turns": [
                {
                    "user_query": "Look up alpha and beta.",
                    "steps": [
                        {
                            "execution_mode": "parallel",
                            "call_order_matters": False,
                            "quality_verification": {
                                "mode": "parallel",
                                "passed": False,
                            },
                            "tool_calls": [
                                {
                                    "tool_name": "lookup",
                                    "arguments": {"key": "alpha"},
                                    "output": {"value": 1},
                                },
                                {
                                    "tool_name": "lookup",
                                    "arguments": {"key": "beta"},
                                    "output": {"value": 2},
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }
    _, gold_steps, _ = checker._gold_from_record(raw)
    assert gold_steps[0]["parallel_certified"] is False
    assert gold_steps[0]["execution_mode"] == "uncertified_parallel"
    assert gold_steps[0]["call_order_matters"] is True


def test_generated_parallel_steps_are_explicitly_order_invariant():
    path = external_fixture(
        "parallel_multistep_grok45_10_20260728.jsonl"
    )
    tasks, _ = checker.load_tasks(path, tool_scope="declared")
    assert tasks
    assert all(task.step_order_matters for task in tasks)
    for task in tasks:
        parallel_steps = [
            gold_step
            for gold_step in task.gold_steps
            if gold_step["execution_mode"] == "parallel"
        ]
        assert len(parallel_steps) == 1
        assert parallel_steps[0]["parallel_certified"] is True
        assert parallel_steps[0]["call_order_matters"] is False


class ReverseParallelClient:
    def __init__(self, task):
        self.task = task
        self.step_index = 0
        self.parallel_flags = []

    def chat(
        self,
        messages,
        tools,
        sampling,
        seed,
        *,
        parallel_tool_calls=False,
    ):
        gold_step = self.task.gold_steps[self.step_index]
        self.step_index += 1
        self.parallel_flags.append(parallel_tool_calls)
        calls = list(gold_step["calls"])
        if gold_step["execution_mode"] == "parallel":
            calls.reverse()
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call-{self.step_index}-{index}",
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item["arguments"]),
                                },
                            }
                            for index, item in enumerate(calls)
                        ],
                    }
                }
            ]
        }


def test_reverse_parallel_replay_pairs_each_output_with_its_call():
    path = external_fixture(
        "parallel_multistep_grok45_10_20260728.jsonl"
    )
    tasks, _ = checker.load_tasks(path, tool_scope="declared", max_samples=1)
    task = tasks[0]
    client = ReverseParallelClient(task)
    state = checker.InteractivePassKChecker(
        client,
        pass_k=1,
        workers=1,
        ordered=True,
    ).run(tasks)[0]

    assert state.status == "success"
    assert client.parallel_flags == [True] * len(task.gold_steps)
    final_gold = task.gold_steps[-1]
    expected_output_by_call = {
        checker.InteractivePassKChecker._call_key(gold_call): gold_call["output"]
        for gold_call in final_gold["calls"]
    }
    final_assistant_index = max(
        index
        for index, message in enumerate(state.messages)
        if message["role"] == "assistant" and message.get("tool_calls")
    )
    final_assistant = state.messages[final_assistant_index]
    final_tool_messages = state.messages[
        final_assistant_index + 1 : final_assistant_index + 1 + len(final_gold["calls"])
    ]
    for predicted, tool_message in zip(
        final_assistant["tool_calls"], final_tool_messages
    ):
        function = predicted["function"]
        key = checker.InteractivePassKChecker._call_key(
            {
                "name": function["name"],
                "arguments": json.loads(function["arguments"]),
            }
        )
        assert json.loads(tool_message["content"]) == expected_output_by_call[key]


def test_standard_pass_at_k_estimator():
    assert checker.estimate_pass_at_k(16, 0, 1) == 0.0
    assert checker.estimate_pass_at_k(16, 16, 1) == 1.0
    assert checker.estimate_pass_at_k(16, 1, 16) == 1.0
    assert checker.estimate_pass_at_k(16, 1, 1) == 1 / 16


class GoldStepClient:
    def __init__(self, task):
        self.task = task
        self.step_index = 0
        self.message_snapshots = []

    def chat(
        self,
        messages,
        tools,
        sampling,
        seed,
        *,
        parallel_tool_calls=False,
    ):
        self.message_snapshots.append(list(messages))
        gold_step = self.task.gold_steps[self.step_index]
        self.step_index += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"gold-{self.step_index}-{index}",
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item["arguments"]),
                                },
                            }
                            for index, item in enumerate(gold_step["calls"])
                        ],
                    }
                }
            ]
        }


def test_interactive_refusal_replays_clarification_before_recovery_turn():
    raw = {
        "available_tools": [
            {
                "name": "refuse",
                "description": "Refuse or clarify",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "enum": [
                                "missing_argument",
                                "ambiguity",
                                "no_appropriate_function",
                            ],
                        }
                    },
                    "required": ["reason"],
                },
            },
            {
                "name": "lookup",
                "description": "Look up a key",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        ],
        "conversation": {
            "turns": [
                {
                    "user_query": "Please look it up.",
                    "assistant_response": "Which key should I look up?",
                    "steps": [
                        {
                            "execution_mode": "refusal",
                            "tool_calls": [
                                {
                                    "tool_name": "refuse",
                                    "arguments": {
                                        "reason": "missing_argument"
                                    },
                                    "output": {
                                        "status": "refused",
                                        "reason": "missing_argument",
                                    },
                                }
                            ],
                            "quality_verification": {
                                "passed": True,
                                "mode": "refusal",
                                "reason": "missing_argument",
                                "native_response": (
                                    "Which key should I look up?"
                                ),
                            },
                        }
                    ],
                },
                {
                    "user_query": "Use alpha.",
                    "assistant_response": "The result is 1.",
                    "steps": [
                        {
                            "tool_calls": [
                                {
                                    "tool_name": "lookup",
                                    "arguments": {"key": "alpha"},
                                    "output": {"value": 1},
                                }
                            ]
                        }
                    ],
                },
            ]
        },
    }
    catalog = {
        name: checker.ToolDefinition(
            name=name,
            category="test",
            schema=schema,
        )
        for name, schema in (
            (
                item["name"],
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": item["parameters"],
                    },
                },
            )
            for item in raw["available_tools"]
        )
    }
    task = checker.task_from_record(0, raw, catalog, tool_scope="declared")
    assert not task.data_issues
    assert task.gold_steps[0]["execution_mode"] == "refusal"
    assert task.gold_steps[0]["refusal_certified"] is True

    client = GoldStepClient(task)
    state = checker.InteractivePassKChecker(
        client,
        pass_k=1,
        workers=1,
        ordered=True,
    ).run([task])[0]

    assert state.status == "success"
    second_request = client.message_snapshots[1]
    assert any(
        message.get("role") == "tool"
        and json.loads(message["content"])["reason"] == "missing_argument"
        for message in second_request
    )
    clarification_index = next(
        index
        for index, message in enumerate(second_request)
        if message == {
            "role": "assistant",
            "content": "Which key should I look up?",
        }
    )
    recovery_index = next(
        index
        for index, message in enumerate(second_request)
        if message == {"role": "user", "content": "Use alpha."}
    )
    assert clarification_index < recovery_index
