#!/usr/bin/env python3
"""Fast structural tests for next-action SFT snapshots."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_toolcalling_toolonly import (
    build_training_examples,
    build_training_examples_with_tools,
)


PROMPT = "tool-only"


def call(name: str, value: int) -> dict:
    return {
        "tool_name": name,
        "arguments": {"value": value},
        "output": {"result": value},
    }


def supervised(messages, mask):
    return [message for message, target in zip(messages, mask) if target]


def test_trajectory_snapshots() -> None:
    row = {
        "trajectory": {
            "query": "Do the work.",
            "steps": [
                {"step_number": 1, "tool_calls": [call("first", 1)]},
                {"step_number": 2, "tool_calls": [call("second", 2)]},
            ],
        }
    }
    examples = build_training_examples(row, PROMPT)
    assert len(examples) == 3
    targets = [supervised(*example) for example in examples]
    assert [len(target) for target in targets] == [1, 1, 1]
    assert targets[0][0]["tool_calls"][0]["function"]["name"] == "first"
    assert targets[1][0]["tool_calls"][0]["function"]["name"] == "second"
    assert not targets[2][0].get("tool_calls")
    assert all(not message.get("tool_calls") for message in examples[0][0][:-1])
    assert any(message.get("role") == "tool" for message in examples[1][0][:-1])
    assert sum(examples[1][1]) == 1


def test_parallel_is_one_target() -> None:
    row = {
        "trajectory": {
            "query": "Read both.",
            "steps": [
                {
                    "step_number": 1,
                    "execution_mode": "parallel",
                    "call_order_matters": False,
                    "tool_calls": [call("left", 1), call("right", 2)],
                }
            ],
        }
    }
    examples = build_training_examples(row, PROMPT)
    assert len(examples) == 2
    target = supervised(*examples[0])
    assert len(target) == 1
    assert len(target[0]["tool_calls"]) == 2
    assert not supervised(*examples[1])[0].get("tool_calls")


def test_no_call_is_not_duplicated_as_stop() -> None:
    row = {
        "conversation": {
            "turns": [
                {
                    "turn_number": 1,
                    "user_query": "Unsupported request.",
                    "steps": [],
                    "no_tool_target": True,
                    "no_tool_reason": "no_appropriate_function",
                }
            ]
        }
    }
    examples = build_training_examples(row, PROMPT)
    assert len(examples) == 1
    target = supervised(*examples[0])
    assert len(target) == 1 and not target[0].get("tool_calls")


def test_unsupervised_turn_stays_in_history() -> None:
    row = {
        "conversation": {
            "turns": [
                {
                    "turn_number": 1,
                    "user_query": "First.",
                    "steps": [
                        {"step_number": 1, "tool_calls": [call("first", 1)]}
                    ],
                    "sft_supervision": False,
                },
                {
                    "turn_number": 2,
                    "user_query": "Next.",
                    "steps": [
                        {"step_number": 1, "tool_calls": [call("second", 2)]}
                    ],
                },
            ]
        }
    }
    examples = build_training_examples(row, PROMPT)
    assert len(examples) == 2
    assert any(
        message.get("role") == "tool" and message.get("name") == "first"
        for message in examples[0][0]
    )
    assert sum(examples[0][1]) == 1


def test_record_with_no_supervised_turn_emits_nothing() -> None:
    row = {
        "available_tools": [
            {
                "name": "first",
                "description": "First tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            }
        ],
        "conversation": {
            "turns": [
                {
                    "turn_number": 1,
                    "user_query": "Context only.",
                    "steps": [
                        {"step_number": 1, "tool_calls": [call("first", 1)]}
                    ],
                    "sft_supervision": False,
                }
            ]
        },
    }
    assert build_training_examples(row, PROMPT) == []
    assert build_training_examples_with_tools(row, PROMPT, {}) == []


def main() -> None:
    test_trajectory_snapshots()
    test_parallel_is_one_target()
    test_no_call_is_not_duplicated_as_stop()
    test_unsupervised_turn_stays_in_history()
    test_record_with_no_supervised_turn_emits_nothing()
    print("PASS next-action supervision snapshots")


if __name__ == "__main__":
    main()
