"""Tests for the economical 1,000-row GLM-5.2 schedule."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/generate_diverse_glm52_1000.py"
)
SPEC = importlib.util.spec_from_file_location("diverse_glm52_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_exact_feature_category_and_step_balance():
    specs, summary = launcher.build_schedule(launcher.DEFAULT_SEED)

    assert len(specs) == 1000
    assert Counter(spec.profile for spec in specs) == Counter(
        launcher.PROFILE_COUNTS
    )
    assert Counter(spec.category for spec in specs) == Counter(
        {category: 125 for category in launcher.CATEGORIES}
    )
    assert Counter(spec.steps for spec in specs) == Counter(
        launcher.STEP_COUNTS
    )
    assert sum(spec.contains_refusal for spec in specs) == 100
    assert sum(spec.contains_parallel for spec in specs) == 100
    assert summary["mean_steps"] == 20
    assert specs[0].profile == "normal"
    assert specs[0].category == "Storage"
    assert specs[0].turns == 5
    assert specs[0].actual_steps_per_turn == (4, 4, 4, 4, 4)


def test_turn_and_action_vectors_are_variable_and_exact():
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)

    assert set(spec.turns for spec in specs) == set(range(3, 9))
    assert set(
        value
        for spec in specs
        for value in spec.actual_steps_per_turn
    ) == set(range(1, 7))
    for spec in specs:
        assert 3 <= spec.turns <= 8
        assert 12 <= spec.steps <= 28
        assert len(spec.actual_steps_per_turn) == spec.turns
        assert len(spec.blueprint_actions_per_turn) == spec.turns
        assert sum(spec.actual_steps_per_turn) == spec.steps
        assert all(1 <= value <= 6 for value in spec.actual_steps_per_turn)
        if spec.interactive_refusal_turn is not None:
            index = spec.interactive_refusal_turn - 1
            assert spec.actual_steps_per_turn[index] == 1
            assert (
                spec.blueprint_actions_per_turn[index]
                == spec.actual_steps_per_turn[index + 1]
            )
            assert spec.blueprint_actions_per_turn[index + 1] == 1
        if spec.schedule in {"terminal", "combined"}:
            assert spec.actual_steps_per_turn[-1] == 1


def test_parallel_widths_are_balanced_and_reasoning_is_disabled():
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)
    widths = Counter(
        spec.parallel_width for spec in specs if spec.contains_parallel
    )
    assert widths == Counter({2: 25, 3: 25, 4: 25, 5: 25})
    assert launcher.reasoning_payload("disabled") == {
        "enabled": False,
        "exclude": True,
    }


def _valid_row_for(spec):
    turns = []
    for turn_index, action_count in enumerate(spec.actual_steps_per_turn):
        steps = [
            {
                "execution_mode": "sequential",
                "call_order_matters": True,
                "tool_calls": [
                    {"tool_name": "ordinary_tool", "arguments": {}}
                ],
            }
            for _ in range(action_count)
        ]
        expected_tools = ["ordinary_tool"] * action_count

        if (
            spec.schedule in {"interactive-refusal", "combined"}
            and spec.interactive_refusal_turn == turn_index + 1
        ):
            steps = [{"tool_calls": []}]
            expected_tools = []
        if turn_index == spec.turns - 1 and spec.contains_parallel:
            steps = [
                {
                    "execution_mode": "parallel",
                    "call_order_matters": False,
                    "tool_calls": [
                        {
                            "tool_name": f"parallel_tool_{call_index}",
                            "arguments": {},
                        }
                        for call_index in range(spec.parallel_width)
                    ],
                }
            ]
            expected_tools = [
                f"parallel_tool_{call_index}"
                for call_index in range(spec.parallel_width)
            ]
        elif (
            turn_index == spec.turns - 1
            and spec.feature == "refusal"
            and spec.schedule == "terminal"
        ):
            steps = [{"tool_calls": []}]
            expected_tools = []

        turns.append(
            {
                "turn_number": turn_index + 1,
                "steps": steps,
                "expected_tools": expected_tools,
                "quality_verification": {"passed": True},
            }
        )

    metadata = {
        "focus_category": spec.category,
        "rl_quality_gate_passed": True,
        "contains_refusal": spec.contains_refusal,
        "contains_parallel": spec.contains_parallel,
        "feature_schedule": spec.schedule,
        "feature_difficulty": "hard",
        "query_naturalization": {
            "enabled": True,
            "protected_tokens_preserved": True,
            "certificate": {
                "semantic_plan_preserved": True,
                "natural_conversation": True,
                "no_tool_syntax": True,
                "avoids_unnecessary_internal_ids": True,
            },
        },
    }
    if spec.schedule in {"interactive-refusal", "combined"}:
        metadata["clarification_recovered"] = True
        metadata["refusal_turns"] = [spec.interactive_refusal_turn]
    return {
        "conversation": {"turns": turns},
        "generation_metadata": metadata,
        "verification_result": {"overall_verification_passed": True},
    }


def test_feature_replaced_turns_do_not_use_blueprint_expected_tool_count():
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)
    parallel = next(
        spec
        for spec in specs
        if spec.feature == "parallel" and spec.parallel_width > 1
    )
    terminal_refusal = next(
        spec
        for spec in specs
        if spec.feature == "refusal" and spec.schedule == "terminal"
    )

    assert launcher.validate_row(parallel, _valid_row_for(parallel)) == []
    assert (
        launcher.validate_row(
            terminal_refusal,
            _valid_row_for(terminal_refusal),
        )
        == []
    )


def test_subprocess_usage_includes_discarded_attempts(tmp_path):
    usage = tmp_path / "usage"
    usage.mkdir()
    (usage / "accepted.r1.json").write_text(
        json.dumps(
            {
                "accepted_rows": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost_usd": 0.01,
                "total_llm_calls": 5,
            }
        )
    )
    (usage / "discarded.r1.json").write_text(
        json.dumps(
            {
                "accepted_rows": 0,
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
                "cost_usd": 0.005,
                "total_calls": 3,
            }
        )
    )

    result = launcher.subprocess_usage_summary(tmp_path)

    assert result["reports"] == 2
    assert result["accepted_reports"] == 1
    assert result["discarded_reports"] == 1
    assert result["total_llm_calls"] == 8
    assert result["total_tokens"] == 180
    assert result["cost_usd"] == 0.015
