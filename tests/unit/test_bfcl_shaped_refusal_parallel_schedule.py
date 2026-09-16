"""Tests for the exact BFCL-shaped 500-row generation schedule."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apigen_multi_turn import MultiTurnGenerator
from generate_step_by_step import parse_actions_per_turn
from refuse_parallel import RefusalParallelMultiTurnGenerator


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/generate_bfcl_shaped_refusal_parallel_500.py"
)
SPEC = importlib.util.spec_from_file_location("bfcl_shaped_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_schedule_has_exact_balance_and_lengths():
    specs, summary = launcher.build_schedule(launcher.DEFAULT_SEED)

    assert len(specs) == 500
    assert Counter(spec.profile for spec in specs) == Counter(
        {
            "refusal_missing": 80,
            "refusal_ambiguity": 80,
            "refusal_unsupported": 40,
            "parallel": 200,
            "combined_missing": 50,
            "combined_ambiguity": 50,
        }
    )
    assert Counter(spec.steps for spec in specs) == Counter(
        {7: 55, 8: 55, 9: 55, 10: 56, 11: 56, 12: 56,
         13: 56, 14: 56, 15: 55}
    )
    assert summary["rows"] == 500


def test_schedule_vectors_match_feature_execution_semantics():
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)

    for spec in specs:
        actual = spec.actual_steps_per_turn
        blueprint = spec.blueprint_actions_per_turn
        assert len(actual) == len(blueprint) == spec.turns
        assert sum(actual) == spec.steps
        assert all(1 <= count <= 3 for count in actual)
        assert all(1 <= count <= 3 for count in blueprint)
        if spec.interactive_refusal_turn is not None:
            index = spec.interactive_refusal_turn - 1
            assert actual[index] == 1
            assert blueprint[index] == actual[index + 1]
            assert blueprint[index + 1] == 1
        if spec.schedule in {"terminal", "combined"}:
            assert actual[-1] == 1


def test_parallel_widths_are_exactly_balanced():
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)
    widths = Counter(
        spec.parallel_width
        for spec in specs
        if spec.feature in {"parallel", "mixed"}
    )
    assert widths == Counter({3: 100, 4: 100, 5: 100})


def test_exact_action_schedule_cli_and_constructor_validation():
    assert parse_actions_per_turn("1, 3,2") == [1, 3, 2]
    llm = MagicMock()
    manager = MagicMock()
    generator = MultiTurnGenerator(
        llm,
        manager,
        num_turns=3,
        blueprint_max_actions_per_turn=3,
        blueprint_actions_per_turn=[1, 3, 2],
    )
    assert generator.blueprint_actions_per_turn == [1, 3, 2]

    with pytest.raises(ValueError, match="exactly 3 entries"):
        MultiTurnGenerator(
            llm,
            manager,
            num_turns=3,
            blueprint_max_actions_per_turn=3,
            blueprint_actions_per_turn=[1, 2],
        )


def test_forced_interactive_turn_respects_recovery_boundaries():
    llm = MagicMock()
    manager = MagicMock()
    generator = RefusalParallelMultiTurnGenerator(
        llm_client=llm,
        tool_manager=manager,
        num_turns=5,
        actions_per_turn=3,
        blueprint_max_actions_per_turn=3,
        blueprint_actions_per_turn=[1, 2, 2, 1, 1],
        allow_refusal=True,
        multi_turn_feature_schedule="interactive-refusal",
        interactive_refusal_turn=3,
    )
    assert generator.feature_config.interactive_refusal_turn == 3

    with pytest.raises(ValueError, match="leave two prior turns"):
        RefusalParallelMultiTurnGenerator(
            llm_client=llm,
            tool_manager=manager,
            num_turns=5,
            actions_per_turn=3,
            allow_refusal=True,
            multi_turn_feature_schedule="interactive-refusal",
            interactive_refusal_turn=2,
        )
