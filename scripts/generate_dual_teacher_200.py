#!/usr/bin/env python3
"""Generate a balanced 200-row RL corpus with two teacher models.

The schedule is deterministic and resumable.  Each teacher contributes 100
accepted rows with exactly ten refusal-containing and ten parallel-containing
episodes (two rows contain both features).  Qwen is used only to write all
ordinary per-turn assistant responses in one batched call.  Blueprinting,
semantic judging, turn compilation, and final grounding stay on the teacher.

Every one of the 129 BFCL-v3 tools is assigned as a hard executed target at
least once *per teacher*.  This is stronger than the curriculum's advisory
coverage targets: a candidate that omits an assigned tool is rejected.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_diverse_glm52_1000 as base


ROOT = SCRIPT_DIR.parent
TOOL_POOL = base.TOOL_POOL
BFCL_EXAMPLES = base.BFCL_EXAMPLES
CATEGORIES = base.CATEGORIES
DEFAULT_SEED = 20260805
QWEN_WRITER_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
TEACHERS = (
    ("deepseek_flash", "deepseek/deepseek-v4-flash-0731"),
    ("glm52", "z-ai/glm-5.2"),
)

# Per teacher: 100 rows, exactly 10 refusal-containing and 10
# parallel-containing rows.  The two combined rows count in both quotas.
PROFILE_COUNTS = {
    "normal": 82,
    "refusal_missing": 3,
    "refusal_ambiguity": 3,
    "refusal_unsupported": 2,
    "parallel": 8,
    "combined_missing": 1,
    "combined_ambiguity": 1,
}

# Per teacher: symmetric around 20 total action transitions.
STEP_COUNTS = {
    12: 2,
    13: 2,
    14: 3,
    15: 4,
    16: 6,
    17: 8,
    18: 10,
    19: 10,
    20: 10,
    21: 10,
    22: 10,
    23: 8,
    24: 6,
    25: 4,
    26: 3,
    27: 2,
    28: 2,
}


@dataclass(frozen=True)
class DualTeacherSpec:
    index: int
    teacher_row: int
    teacher: str
    teacher_model: str
    reasoning_effort: str
    profile: str
    category: str
    feature: str
    schedule: str
    refusal_reason: str
    steps: int
    turns: int
    actual_steps_per_turn: tuple[int, ...]
    blueprint_actions_per_turn: tuple[int, ...]
    parallel_width: int
    interactive_refusal_turn: int | None
    required_tools: tuple[str, ...] = ()

    @property
    def stem(self) -> str:
        category = self.category.lower().replace(" ", "_")
        return (
            f"{self.index:03d}.{self.teacher}.{self.profile}.{category}."
            f"s{self.steps}.t{self.turns}.w{self.parallel_width}"
        )

    @property
    def contains_refusal(self) -> bool:
        return self.feature in {"refusal", "mixed"}

    @property
    def contains_parallel(self) -> bool:
        return self.feature in {"parallel", "mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "data/generated/runs/dual_teacher_200_20260805"
        ),
    )
    parser.add_argument("--python", default=os.getenv("APIGEN_PYTHON", sys.executable))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-task-restarts", type=int, default=1)
    parser.add_argument("--task-timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-calls-per-row", type=int, default=30)
    parser.add_argument("--max-tokens-per-row", type=int, default=160_000)
    parser.add_argument("--max-candidate-starts-per-row", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Run only the first N interleaved rows (two means one/teacher).",
    )
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument("--skip-provider-probe", action="store_true")
    parser.add_argument("--quiet-schedule", action="store_true")
    parser.add_argument(
        "--dedupe-against",
        action="append",
        default=[],
        type=Path,
        metavar="JSONL",
    )
    return parser.parse_args()


def load_tools() -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    all_names: list[str] = []
    for row in base.read_jsonl(TOOL_POOL):
        name = str(row["api_name"])
        category = str(row["category"])
        by_category[category].append(name)
        all_names.append(name)
    if len(all_names) != 129 or len(set(all_names)) != 129:
        raise RuntimeError(
            f"Expected 129 uniquely named BFCL tools, got "
            f"{len(all_names)}/{len(set(all_names))}"
        )
    if set(by_category) != set(CATEGORIES):
        raise RuntimeError("Tool-pool categories differ from the schedule")
    return {category: sorted(names) for category, names in by_category.items()}


def _category_counts(teacher_index: int) -> dict[str, int]:
    # Complementary 12/13 splits make every category exactly 25/200 overall.
    return {
        category: 13 if index % 2 == teacher_index else 12
        for index, category in enumerate(CATEGORIES)
    }


def _move_one(values: list[Any], value: Any, destination: int = 0) -> None:
    source = values.index(value)
    values[source], values[destination] = values[destination], values[source]


def _teacher_schedule(
    *,
    teacher_index: int,
    teacher: str,
    teacher_model: str,
    seed: int,
) -> list[DualTeacherSpec]:
    rng = random.Random(seed + 10_007 * (teacher_index + 1))
    profiles = [
        profile
        for profile, count in PROFILE_COUNTS.items()
        for _ in range(count)
    ]
    steps = [
        value for value, count in STEP_COUNTS.items() for _ in range(count)
    ]
    categories = [
        category
        for category, count in _category_counts(teacher_index).items()
        for _ in range(count)
    ]
    rng.shuffle(profiles)
    rng.shuffle(steps)
    rng.shuffle(categories)

    # Row zero for each teacher is a representative 5-turn x 4-action smoke
    # row.  Swaps preserve every exact distribution.
    _move_one(profiles, "normal")
    _move_one(steps, 20)
    _move_one(categories, "Science")

    # Ten parallel-containing rows per teacher.  Across both teachers every
    # certified width occurs exactly five times.
    width_counts = (
        {2: 3, 3: 2, 4: 2, 5: 3}
        if teacher_index == 0
        else {2: 2, 3: 3, 4: 3, 5: 2}
    )
    widths = [
        width for width, count in width_counts.items() for _ in range(count)
    ]
    rng.shuffle(widths)
    width_index = 0

    specs: list[DualTeacherSpec] = []
    for row_index, (profile, step_count, category) in enumerate(
        zip(profiles, steps, categories)
    ):
        feature, schedule, refusal_reason = base.profile_properties(profile)
        if row_index == 0:
            turns = 5
            actual = (4, 4, 4, 4, 4)
            blueprint = actual
            refusal_index = None
        else:
            turns = base.choose_turn_count(
                rng, schedule=schedule, steps=step_count
            )
            refusal_index: int | None = None
            fixed: dict[int, int] = {}
            if schedule == "terminal":
                fixed[turns - 1] = 1
            elif schedule in {"interactive-refusal", "combined"}:
                refusal_index = rng.choice(list(range(2, turns - 2)))
                fixed[refusal_index] = 1
                if schedule == "combined":
                    fixed[turns - 1] = 1
            actual = base.choose_action_vector(
                rng, steps=step_count, turns=turns, fixed=fixed
            )
            blueprint_list = list(actual)
            if refusal_index is not None:
                blueprint_list[refusal_index] = actual[refusal_index + 1]
                blueprint_list[refusal_index + 1] = 1
            blueprint = tuple(blueprint_list)

        if feature in {"parallel", "mixed"}:
            parallel_width = widths[width_index]
            width_index += 1
        else:
            parallel_width = 2
        specs.append(
            DualTeacherSpec(
                index=-1,
                teacher_row=row_index,
                teacher=teacher,
                teacher_model=teacher_model,
                reasoning_effort="low",
                profile=profile,
                category=category,
                feature=feature,
                schedule=schedule,
                refusal_reason=refusal_reason,
                steps=step_count,
                turns=turns,
                actual_steps_per_turn=actual,
                blueprint_actions_per_turn=blueprint,
                parallel_width=parallel_width,
                interactive_refusal_turn=(
                    refusal_index + 1 if refusal_index is not None else None
                ),
            )
        )
    return specs


def _assign_required_tools(
    specs: list[DualTeacherSpec],
    tools_by_category: dict[str, list[str]],
    *,
    seed: int,
) -> list[DualTeacherSpec]:
    """Give every row 1-2 hard targets and cover all 129 tools."""

    rng = random.Random(seed)
    assigned: dict[int, list[str]] = {index: [] for index in range(len(specs))}
    for category in CATEGORIES:
        row_indices = [
            index for index, spec in enumerate(specs) if spec.category == category
        ]
        rng.shuffle(row_indices)
        # Keep the representative row first and make its sole target simple.
        representative = next(
            (index for index in row_indices if specs[index].teacher_row == 0),
            None,
        )
        if representative is not None:
            row_indices.remove(representative)
            row_indices.insert(0, representative)

        tools = list(tools_by_category[category])
        rng.shuffle(tools)
        if representative is not None and category == "Science":
            tools.remove("add")
            tools.insert(0, "add")

        # One target for every row.  Only categories with fewer tools than
        # rows repeat a target here (Communication and Events).
        for position, row_index in enumerate(row_indices):
            assigned[row_index].append(tools[position % len(tools)])

        # Assign tools not reached by the primary pass as second targets.
        remaining_tools = tools[len(row_indices) :]
        secondary_rows = row_indices[1:] + row_indices[:1]
        for tool, row_index in zip(remaining_tools, secondary_rows):
            assigned[row_index].append(tool)

    result = [
        replace(spec, required_tools=tuple(assigned[index]))
        for index, spec in enumerate(specs)
    ]
    if any(not 1 <= len(spec.required_tools) <= 2 for spec in result):
        raise RuntimeError("Every row must have one or two required tools")
    return result


def build_schedule(seed: int) -> tuple[list[DualTeacherSpec], dict[str, Any]]:
    tools_by_category = load_tools()
    per_teacher: list[list[DualTeacherSpec]] = []
    for teacher_index, (teacher, teacher_model) in enumerate(TEACHERS):
        specs = _teacher_schedule(
            teacher_index=teacher_index,
            teacher=teacher,
            teacher_model=teacher_model,
            seed=seed,
        )
        specs = _assign_required_tools(
            specs,
            tools_by_category,
            seed=seed + 50_021 * (teacher_index + 1),
        )
        per_teacher.append(specs)

    # Interleaving makes --max-tasks 2 a useful one-row-per-teacher smoke test.
    ordered: list[DualTeacherSpec] = []
    for teacher_row in range(100):
        for teacher_specs in per_teacher:
            ordered.append(teacher_specs[teacher_row])
    specs = [replace(spec, index=index) for index, spec in enumerate(ordered)]

    all_tool_names = {
        name for names in tools_by_category.values() for name in names
    }
    teacher_summaries: dict[str, Any] = {}
    for teacher, teacher_model in TEACHERS:
        selected = [spec for spec in specs if spec.teacher == teacher]
        target_counts = Counter(
            tool for spec in selected for tool in spec.required_tools
        )
        missing = sorted(all_tool_names - set(target_counts))
        if missing:
            raise RuntimeError(f"{teacher} schedule misses tools: {missing}")
        refusal_count = sum(spec.contains_refusal for spec in selected)
        parallel_count = sum(spec.contains_parallel for spec in selected)
        if refusal_count != 10 or parallel_count != 10:
            raise RuntimeError(
                f"{teacher} feature quotas are {refusal_count}/{parallel_count}"
            )
        teacher_summaries[teacher] = {
            "model": teacher_model,
            "rows": len(selected),
            "profiles": dict(sorted(Counter(s.profile for s in selected).items())),
            "categories": dict(sorted(Counter(s.category for s in selected).items())),
            "contains_refusal": refusal_count,
            "contains_parallel": parallel_count,
            "step_distribution": dict(sorted(Counter(s.steps for s in selected).items())),
            "mean_steps": sum(s.steps for s in selected) / len(selected),
            "turn_distribution": dict(sorted(Counter(s.turns for s in selected).items())),
            "parallel_width_distribution": dict(
                sorted(
                    Counter(
                        s.parallel_width for s in selected if s.contains_parallel
                    ).items()
                )
            ),
            "hard_target_assignments": sum(len(s.required_tools) for s in selected),
            "unique_hard_target_tools": len(target_counts),
            "repeated_hard_target_assignments": sum(target_counts.values())
            - len(target_counts),
        }

    summary = {
        "seed": seed,
        "rows": len(specs),
        "teachers": teacher_summaries,
        "overall_categories": dict(
            sorted(Counter(spec.category for spec in specs).items())
        ),
        "overall_contains_refusal": sum(spec.contains_refusal for spec in specs),
        "overall_contains_parallel": sum(spec.contains_parallel for spec in specs),
        "qwen_role": "batched_final_response_writer_only",
        "qwen_model": QWEN_WRITER_MODEL,
        "grounding_policy": "same_large_teacher_as_each_row",
        "hard_coverage_policy": (
            "all 129 BFCL-v3 tools assigned and execution-enforced per teacher"
        ),
    }
    if len(specs) != 200:
        raise RuntimeError(f"Expected 200 rows, got {len(specs)}")
    if summary["overall_categories"] != {category: 25 for category in CATEGORIES}:
        raise RuntimeError("Overall category schedule is not exactly balanced")
    return specs, summary


def spec_path(spec: DualTeacherSpec, work_dir: Path) -> Path:
    return work_dir / "rows" / spec.teacher / f"{spec.stem}.jsonl"


def trace_path(spec: DualTeacherSpec, work_dir: Path) -> Path:
    return work_dir / "traces" / spec.teacher / f"{spec.stem}.jsonl"


def _actual_tools(row: dict[str, Any]) -> list[str]:
    return [
        str(call.get("tool_name", ""))
        for turn in row.get("conversation", {}).get("turns", [])
        for step in turn.get("steps", [])
        for call in step.get("tool_calls", [])
    ]


def _provenance_sources(row: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for turn in row.get("conversation", {}).get("turns", []):
        for step in turn.get("steps", []):
            roots = [
                step.get("quality_verification", {}).get(
                    "argument_provenance", {}
                ),
                step.get("quality_verification", {}).get(
                    "argument_visibility_certificate", {}
                ),
            ]
            stack: list[Any] = roots
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    source = value.get("source")
                    if isinstance(source, str):
                        result[source.casefold()] += 1
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
    return result


def _trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(base.read_jsonl(path))


def validate_row(
    spec: DualTeacherSpec,
    row: dict[str, Any],
    *,
    llm_trace: Path | None = None,
) -> list[str]:
    errors = list(base.validate_row(spec, row))
    metadata = row.get("generation_metadata", {})
    routing = metadata.get("model_routing", {})
    expected_routing = {
        "generator": spec.teacher_model,
        "semantic_judge": spec.teacher_model,
        "final_response_writer": QWEN_WRITER_MODEL,
        "grounding_judge": spec.teacher_model,
    }
    if routing != expected_routing:
        errors.append(f"MODEL_ROUTING:{routing}!={expected_routing}")
    if metadata.get("generation_pipeline") != "turn_compiler_v1_batched_turn_responses":
        errors.append("WRONG_GENERATION_PIPELINE")
    if metadata.get("turn_response_policy") != "batched_grounded_per_turn":
        errors.append("WRONG_TURN_RESPONSE_POLICY")

    actual_tools = set(_actual_tools(row))
    missing_tools = sorted(set(spec.required_tools) - actual_tools)
    if missing_tools:
        errors.append("REQUIRED_TOOLS_MISSING:" + ",".join(missing_tools))
    directive = metadata.get("generation_directive", {})
    if set(directive.get("hard_required_tools", [])) != set(spec.required_tools):
        errors.append("HARD_REQUIRED_TOOL_METADATA_MISMATCH")

    turns = row.get("conversation", {}).get("turns", [])
    if any(not str(turn.get("assistant_response", "")).strip() for turn in turns):
        errors.append("EMPTY_ASSISTANT_RESPONSE")
    serialized = json.dumps(row, ensure_ascii=False)
    if "The requested actions completed successfully." in serialized:
        errors.append("LEGACY_RESPONSE_PLACEHOLDER")

    allowed_sources = {
        "user",
        "history",
        "tool_output",
        "schema_default",
        "visible_context",
        "literal",
        "prevalidated_override",
    }
    sources = _provenance_sources(row)
    invalid_sources = sorted(set(sources) - allowed_sources)
    if invalid_sources:
        errors.append("HIDDEN_OR_UNKNOWN_ARGUMENT_SOURCES:" + ",".join(invalid_sources))

    if llm_trace is not None:
        events = _trace_events(llm_trace)
        if not events:
            errors.append("MISSING_LLM_TRACE")
        successful = [event for event in events if event.get("status") == "success"]
        qwen_events = [
            event for event in successful if event.get("model") == QWEN_WRITER_MODEL
        ]
        if not qwen_events:
            errors.append("NO_QWEN_WRITER_CALL")
        if any(event.get("purpose") != "final_response_generate" for event in qwen_events):
            errors.append("QWEN_USED_OUTSIDE_FINAL_RESPONSE_WRITING")
        grounding = [
            event
            for event in successful
            if event.get("purpose") == "final_response_grounding_judge"
        ]
        if not grounding:
            errors.append("NO_FINAL_GROUNDING_CALL")
        if any(event.get("model") != spec.teacher_model for event in grounding):
            errors.append("GROUNDING_NOT_DONE_BY_TEACHER")
    return errors


def feature_args(spec: DualTeacherSpec) -> list[str]:
    return base.feature_args(spec)


def command_for(
    spec: DualTeacherSpec,
    *,
    args: argparse.Namespace,
    output: Path,
    registry: Path,
    usage_report: Path,
    checkpoint: Path,
    candidate_archive: Path,
) -> list[str]:
    num_actions = spec.parallel_width if spec.feature != "normal" else 6
    command = [
        args.python,
        "src/generate_step_by_step.py",
        "--mode",
        "multi-turn",
        "--num-turns",
        str(spec.turns),
        "--num-datapoints",
        "1",
        "--num-actions",
        str(num_actions),
        "--blueprint-max-actions-per-turn",
        "6",
        "--blueprint-actions-per-turn",
        ",".join(map(str, spec.blueprint_actions_per_turn)),
        "--max-parallel-width",
        str(spec.parallel_width),
        "--min-total-steps",
        str(spec.steps),
        "--max-total-steps",
        str(spec.steps),
        "--category",
        spec.category,
        "--model",
        spec.teacher_model,
        "--judge-model",
        spec.teacher_model,
        # Deliberately do not use --use-qwen-final-stages: only the writer is
        # local Qwen, while the default grounding client remains the teacher.
        "--final-response-model",
        QWEN_WRITER_MODEL,
        "--optimized-pipeline",
        "--max-calls-per-candidate",
        str(args.max_calls_per_row),
        "--max-calls-per-accepted-row",
        str(args.max_calls_per_row),
        "--max-tokens-per-accepted-row",
        str(args.max_tokens_per_row),
        "--max-candidate-starts-per-row",
        str(args.max_candidate_starts_per_row),
        "--max-turn-attempts",
        "2",
        "--usage-report",
        str(usage_report),
        "--candidate-archive-dir",
        str(candidate_archive),
        "--checkpoint",
        str(checkpoint),
        "--tool-pool",
        str(TOOL_POOL),
        "--invocation-examples",
        str(BFCL_EXAMPLES),
        "--dedupe-registry",
        str(registry),
    ]
    for tool in spec.required_tools:
        command.extend(["--required-tool", tool])
    command.extend(feature_args(spec))
    if spec.interactive_refusal_turn is not None:
        command.extend(
            ["--interactive-refusal-turn", str(spec.interactive_refusal_turn)]
        )
    command.extend(["--output", str(output)])
    return command


def _archive_invalid_row(
    output: Path,
    *,
    work_dir: Path,
    spec: DualTeacherSpec,
) -> Path:
    destination_dir = work_dir / "scheduler_rejected" / spec.teacher
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{spec.stem}.{time.time_ns()}.jsonl"
    output.replace(destination)
    return destination


def run_spec(
    spec: DualTeacherSpec,
    *,
    args: argparse.Namespace,
    work_dir: Path,
    registry: Path,
) -> tuple[int, str]:
    output = spec_path(spec, work_dir)
    trace = trace_path(spec, work_dir)
    log = work_dir / "logs" / spec.teacher / f"{spec.stem}.log"
    usage_dir = work_dir / "usage" / spec.teacher
    checkpoint = work_dir / "checkpoints" / spec.teacher / f"{spec.stem}.json"
    candidate_archive = work_dir / "candidate_archive" / spec.teacher / spec.stem
    for directory in (
        output.parent,
        trace.parent,
        log.parent,
        usage_dir,
        checkpoint.parent,
        candidate_archive,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    rows = base.count_rows(output)
    if rows == 1:
        errors = validate_row(spec, next(base.read_jsonl(output)), llm_trace=trace)
        if not errors:
            return spec.index, "already-complete"
        archived = _archive_invalid_row(output, work_dir=work_dir, spec=spec)
        raise RuntimeError(
            f"Existing row failed validation and was archived at {archived}: {errors}"
        )
    if rows > 1:
        raise RuntimeError(f"{output} contains more than one row")

    last_status = ""
    for restart in range(1, args.max_task_restarts + 1):
        usage_report = usage_dir / f"{spec.stem}.r{restart}.json"
        command = command_for(
            spec,
            args=args,
            output=output,
            registry=registry,
            usage_report=usage_report,
            checkpoint=checkpoint,
            candidate_archive=candidate_archive,
        )
        environment = {
            **os.environ,
            "APIGEN_LLM_TIMEOUT": os.getenv("APIGEN_LLM_TIMEOUT", "900"),
            "APIGEN_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
            "APIGEN_REASONING_EFFORT": spec.reasoning_effort,
            "APIGEN_HTTP_ATTEMPTS": "1",
            "APIGEN_APPLICATION_LLM_ATTEMPTS": "1",
            "APIGEN_LLM_TRACE_PATH": str(trace),
        }
        with log.open("a", encoding="utf-8") as destination:
            destination.write(
                json.dumps(
                    {
                        "event": "launch",
                        "restart": restart,
                        "spec": asdict(spec),
                        "command": command,
                        "reasoning_effort": spec.reasoning_effort,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=destination,
                    stderr=subprocess.STDOUT,
                    timeout=args.task_timeout_seconds,
                    check=False,
                )
                last_status = f"exit={completed.returncode}"
            except subprocess.TimeoutExpired:
                last_status = f"timeout={args.task_timeout_seconds}s"
                destination.write(
                    json.dumps({"event": "timeout", "restart": restart}) + "\n"
                )

        if base.count_rows(output) == 1:
            row = next(base.read_jsonl(output))
            errors = validate_row(spec, row, llm_trace=trace)
            if not errors:
                return spec.index, f"generated-restart-{restart}"
            archived = _archive_invalid_row(output, work_dir=work_dir, spec=spec)
            raise RuntimeError(
                f"Generated row failed validation and was archived at {archived}: {errors}"
            )

        if log.exists():
            with log.open("rb") as source:
                source.seek(max(0, log.stat().st_size - 32_768))
                tail = source.read().decode("utf-8", errors="replace").casefold()
            fatal_markers = (
                "key limit exceeded",
                "requires more credits",
                "invalid api key",
                "authentication failed",
                "insufficient_quota",
                "user not found",
            )
            if any(marker in tail for marker in fatal_markers):
                raise base.FatalProviderError(
                    f"{spec.stem} hit a provider/account limit; see {log}"
                )
        if restart < args.max_task_restarts:
            time.sleep(min(2**restart, 60))
    raise RuntimeError(
        f"{spec.stem} exhausted {args.max_task_restarts} restart(s) "
        f"({last_status}); see {log}"
    )


def complete_rows(
    specs: Iterable[DualTeacherSpec], work_dir: Path
) -> list[tuple[DualTeacherSpec, dict[str, Any]]]:
    complete: list[tuple[DualTeacherSpec, dict[str, Any]]] = []
    for spec in specs:
        path = spec_path(spec, work_dir)
        if base.count_rows(path) != 1:
            continue
        row = next(base.read_jsonl(path))
        if not validate_row(spec, row, llm_trace=trace_path(spec, work_dir)):
            complete.append((spec, row))
    return complete


def subprocess_usage_summary(work_dir: Path) -> dict[str, Any]:
    """Account for accepted and discarded nested per-teacher reports."""

    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_prompt_tokens",
        "cost_usd",
        "total_llm_calls",
    )
    result: dict[str, float | int] = {key: 0 for key in keys}
    reports = sorted((work_dir / "usage").glob("**/*.json"))
    accepted_reports = 0
    for report in reports:
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(payload.get("accepted_rows", 0) or 0) > 0:
            accepted_reports += 1
        for key in keys:
            if key == "total_llm_calls":
                value = payload.get(key, payload.get("total_calls", 0))
            else:
                value = payload.get(key, 0)
            result[key] += value or 0
    result["reports"] = len(reports)
    result["accepted_reports"] = accepted_reports
    result["discarded_reports"] = len(reports) - accepted_reports
    return result


def _coverage_summary(
    complete: Iterable[tuple[DualTeacherSpec, dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for teacher, _ in TEACHERS:
        selected = [(spec, row) for spec, row in complete if spec.teacher == teacher]
        tools = Counter(tool for _, row in selected for tool in set(_actual_tools(row)))
        result[teacher] = {
            "accepted_rows": len(selected),
            "unique_tools_executed": len(set(tools) - {"refuse"}),
            "tools_executed": dict(sorted(tools.items())),
        }
    return result


def write_progress(
    *,
    specs: list[DualTeacherSpec],
    selected: list[DualTeacherSpec],
    work_dir: Path,
    output_dir: Path,
    failures: list[str],
) -> dict[str, Any]:
    complete = complete_rows(selected, work_dir)
    partial = output_dir / "accepted.partial.jsonl"
    base.atomic_jsonl(partial, (row for _, row in complete))
    accepted_usage = base.usage_summary(row for _, row in complete)
    all_usage = subprocess_usage_summary(work_dir)
    usage_keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_prompt_tokens",
        "cost_usd",
        "total_llm_calls",
    )
    summary = {
        "scheduled_total": len(specs),
        "selected_this_run": len(selected),
        "accepted": len(complete),
        "accepted_by_teacher": dict(
            sorted(Counter(spec.teacher for spec, _ in complete).items())
        ),
        "remaining_selected": len(selected) - len(complete),
        "failures": len(failures),
        "partial_dataset": str(partial),
        "executed_tool_coverage": _coverage_summary(complete),
        "usage": accepted_usage,
        "all_subprocess_usage": all_usage,
        "discarded_subprocess_usage": {
            key: max(0, all_usage.get(key, 0) - accepted_usage.get(key, 0))
            for key in usage_keys
        },
        "updated_at_epoch": time.time(),
    }
    base.atomic_json(output_dir / "status.json", summary)
    return summary


def finalize(
    *,
    specs: list[DualTeacherSpec],
    schedule_summary: dict[str, Any],
    work_dir: Path,
    output_dir: Path,
    registry: Path,
    registry_seed: dict[str, int],
) -> dict[str, Any]:
    complete = complete_rows(specs, work_dir)
    if len(complete) != 200:
        raise RuntimeError(f"Cannot finalize {len(complete)}/200 rows")
    rows = [row for _, row in complete]
    signatures = [base.semantic_signature(row) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("Generated dataset contains semantic duplicates")

    merged = output_dir / "dual_teacher_200.jsonl"
    base.atomic_jsonl(merged, rows)
    teacher_files: dict[str, str] = {}
    for teacher, _ in TEACHERS:
        destination = output_dir / f"{teacher}_100.jsonl"
        selected_rows = [row for spec, row in complete if spec.teacher == teacher]
        if len(selected_rows) != 100:
            raise RuntimeError(f"{teacher} has {len(selected_rows)}/100 rows")
        base.atomic_jsonl(destination, selected_rows)
        teacher_files[teacher] = str(destination)

    observed: dict[str, Any] = {}
    all_tools = {name for names in load_tools().values() for name in names}
    for teacher, teacher_model in TEACHERS:
        selected = [(spec, row) for spec, row in complete if spec.teacher == teacher]
        executed = {
            tool for _, row in selected for tool in _actual_tools(row)
        }
        missing = sorted(all_tools - executed)
        if missing:
            raise RuntimeError(f"{teacher} accepted rows miss tools: {missing}")
        observed[teacher] = {
            "model": teacher_model,
            "rows": len(selected),
            "profiles": dict(sorted(Counter(s.profile for s, _ in selected).items())),
            "categories": dict(sorted(Counter(s.category for s, _ in selected).items())),
            "contains_refusal": sum(s.contains_refusal for s, _ in selected),
            "contains_parallel": sum(s.contains_parallel for s, _ in selected),
            "unique_tools_executed": len(executed - {"refuse"}),
            "usage": base.usage_summary(row for _, row in selected),
        }

    compressed_logs = 0
    for log in (work_dir / "logs").glob("**/*.log"):
        target = log.with_suffix(".log.gz")
        if target.exists():
            continue
        with log.open("rb") as source, gzip.open(
            target, "wb", compresslevel=6
        ) as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
        log.unlink()
        compressed_logs += 1

    manifest = {
        "dataset": "dual-teacher-balanced-rl-200",
        "total_rows": len(rows),
        "merged": str(merged),
        "teacher_files": teacher_files,
        "model_policy": {
            "teachers": dict(TEACHERS),
            "qwen_writer": QWEN_WRITER_MODEL,
            "qwen_roles": ["final_response_generate"],
            "grounding": "row teacher",
        },
        "observed": observed,
        "unique_semantic_signatures": len(set(signatures)),
        "usage": base.usage_summary(rows),
        "all_subprocess_usage": subprocess_usage_summary(work_dir),
        "schedule": schedule_summary,
        "semantic_dedupe_registry": str(registry),
        "registry_seed": registry_seed,
        "compressed_log_files": compressed_logs,
    }
    base.atomic_json(output_dir / "dataset_manifest.json", manifest)
    base.atomic_json(output_dir / "cost_report.json", manifest["usage"])
    return manifest


def _default_dedupe_sources() -> list[Path]:
    canonical = (
        ROOT
        / "data/generated/canonical_sft_rl_corpus_565_no_claude_20260803.jsonl"
    )
    result = base.default_dedupe_sources()
    if canonical.exists():
        result.append(canonical)
    return list(dict.fromkeys(path.resolve() for path in result))


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.max_task_restarts < 1:
        raise ValueError("--max-task-restarts must be positive")

    specs, schedule_summary = build_schedule(args.seed)
    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_jsonl(
        output_dir / "generation_schedule.jsonl",
        (asdict(spec) for spec in specs),
    )
    base.atomic_json(output_dir / "schedule_summary.json", schedule_summary)
    if not args.quiet_schedule:
        print(json.dumps(schedule_summary, ensure_ascii=False, indent=2))
    if args.schedule_only:
        return 0

    if not args.skip_provider_probe:
        for _, model in TEACHERS:
            base.provider_probe(
                model=model,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
            )

    registry = output_dir / "semantic_signatures.registry"
    dedupe_sources = [
        *_default_dedupe_sources(),
        *(path.resolve() for path in args.dedupe_against),
    ]
    dedupe_sources = list(dict.fromkeys(dedupe_sources))
    registry_seed = base.seed_registry(registry, dedupe_sources)
    base.atomic_json(
        output_dir / "dedupe_sources.json",
        {
            "sources": [str(path) for path in dedupe_sources],
            **registry_seed,
        },
    )

    selected = specs[: args.max_tasks] if args.max_tasks else specs
    failures: list[str] = []
    progress_lock = threading.Lock()
    progress = write_progress(
        specs=specs,
        selected=selected,
        work_dir=work_dir,
        output_dir=output_dir,
        failures=failures,
    )
    print(
        f"Starting/resuming {len(selected)} tasks; "
        f"{progress['accepted']} already accepted",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                run_spec,
                spec,
                args=args,
                work_dir=work_dir,
                registry=registry,
            ): spec
            for spec in selected
        }
        done = 0
        fatal = False
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                _, status = future.result()
            except base.FatalProviderError as exc:
                failures.append(f"{spec.stem}: {exc}")
                fatal = True
                status = f"FATAL: {exc}"
            except Exception as exc:
                failures.append(f"{spec.stem}: {type(exc).__name__}: {exc}")
                status = f"FAILED: {exc}"
            done += 1
            with progress_lock:
                progress = write_progress(
                    specs=specs,
                    selected=selected,
                    work_dir=work_dir,
                    output_dir=output_dir,
                    failures=failures,
                )
            print(
                f"[{done}/{len(selected)}] {spec.stem}: {status}; "
                f"accepted={progress['accepted']}, "
                f"cost=${progress['all_subprocess_usage']['cost_usd']:.4f}",
                flush=True,
            )
            if fatal:
                for pending in futures:
                    pending.cancel()
                break

    if failures:
        base.atomic_json(
            output_dir / "generation_errors.json",
            {"failures": failures, "count": len(failures)},
        )
        print(
            f"Run ended with {len(failures)} failed work item(s); rerun the "
            "same command to resume.",
            file=sys.stderr,
        )
        return 1

    if args.max_tasks:
        print(
            f"Partial run complete: {progress['accepted']}/{len(selected)} "
            f"selected rows are in {progress['partial_dataset']}"
        )
        return 0

    manifest = finalize(
        specs=specs,
        schedule_summary=schedule_summary,
        work_dir=work_dir,
        output_dir=output_dir,
        registry=registry,
        registry_seed=registry_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
