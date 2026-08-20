"""Tests for the deterministic two-teacher 200-row launcher."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/generate_dual_teacher_200.py"
)
SPEC = importlib.util.spec_from_file_location("dual_teacher_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_exact_teacher_feature_step_and_category_quotas():
    specs, summary = launcher.build_schedule(launcher.DEFAULT_SEED)

    assert len(specs) == 200
    assert Counter(spec.teacher for spec in specs) == Counter(
        {"deepseek_flash": 100, "glm52": 100}
    )
    assert Counter(spec.category for spec in specs) == Counter(
        {category: 25 for category in launcher.CATEGORIES}
    )
    assert summary["overall_contains_refusal"] == 20
    assert summary["overall_contains_parallel"] == 20

    for teacher, _ in launcher.TEACHERS:
        selected = [spec for spec in specs if spec.teacher == teacher]
        assert Counter(spec.profile for spec in selected) == Counter(
            launcher.PROFILE_COUNTS
        )
        assert Counter(spec.steps for spec in selected) == Counter(
            launcher.STEP_COUNTS
        )
        assert sum(spec.contains_refusal for spec in selected) == 10
        assert sum(spec.contains_parallel for spec in selected) == 10
        assert sum(spec.steps for spec in selected) / len(selected) == 20

    # First two rows are representative one-row-per-teacher smoke targets.
    assert [spec.teacher for spec in specs[:2]] == [
        "deepseek_flash",
        "glm52",
    ]
    assert all(spec.profile == "normal" for spec in specs[:2])
    assert all(spec.category == "Science" for spec in specs[:2])
    assert all(spec.actual_steps_per_turn == (4, 4, 4, 4, 4) for spec in specs[:2])


def test_every_teacher_hard_targets_all_129_tools():
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)
    tools_by_category = launcher.load_tools()
    all_tools = {tool for tools in tools_by_category.values() for tool in tools}
    assert len(all_tools) == 129

    category_by_tool = {
        tool: category
        for category, tools in tools_by_category.items()
        for tool in tools
    }
    for teacher, _ in launcher.TEACHERS:
        selected = [spec for spec in specs if spec.teacher == teacher]
        targets = Counter(tool for spec in selected for tool in spec.required_tools)
        assert set(targets) == all_tools
        assert sum(targets.values()) == 135
        assert all(1 <= len(spec.required_tools) <= 2 for spec in selected)
        assert all(
            category_by_tool[tool] == spec.category
            for spec in selected
            for tool in spec.required_tools
        )


def test_qwen_is_commanded_only_as_writer_and_teacher_remains_grounder(tmp_path):
    specs, _ = launcher.build_schedule(launcher.DEFAULT_SEED)
    spec = specs[0]
    args = SimpleNamespace(
        python="python",
        max_calls_per_row=30,
        max_tokens_per_row=160_000,
        max_candidate_starts_per_row=2,
    )
    command = launcher.command_for(
        spec,
        args=args,
        output=tmp_path / "row.jsonl",
        registry=tmp_path / "registry",
        usage_report=tmp_path / "usage.json",
        checkpoint=tmp_path / "checkpoint.json",
        candidate_archive=tmp_path / "archive",
    )

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--model") == spec.teacher_model
    assert value("--judge-model") == spec.teacher_model
    assert value("--final-response-model") == launcher.QWEN_WRITER_MODEL
    assert "--use-qwen-final-stages" not in command
    assert "--grounding-model" not in command
    assert [
        command[index + 1]
        for index, token in enumerate(command)
        if token == "--required-tool"
    ] == list(spec.required_tools)


def test_nested_subprocess_usage_counts_discarded_attempts(tmp_path):
    usage = tmp_path / "usage" / "deepseek_flash"
    usage.mkdir(parents=True)
    (usage / "accepted.json").write_text(
        '{"accepted_rows":1,"total_tokens":120,"cost_usd":0.01,'
        '"total_llm_calls":5}'
    )
    other = tmp_path / "usage" / "glm52"
    other.mkdir()
    (other / "discarded.json").write_text(
        '{"accepted_rows":0,"total_tokens":60,"cost_usd":0.005,'
        '"total_calls":3}'
    )

    result = launcher.subprocess_usage_summary(tmp_path)
    assert result["reports"] == 2
    assert result["accepted_reports"] == 1
    assert result["discarded_reports"] == 1
    assert result["total_tokens"] == 180
    assert result["total_llm_calls"] == 8
    assert result["cost_usd"] == 0.015
