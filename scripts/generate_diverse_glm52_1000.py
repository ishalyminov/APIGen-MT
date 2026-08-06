#!/usr/bin/env python3
"""Generate 1,000 diverse, economical multi-turn RL trajectories.

The deterministic schedule has:

* 820 ordinary multi-turn rows;
* 80 refusal-only rows;
* 80 parallel-only rows;
* 20 combined clarification/recovery + parallel rows.

Thus exactly 10% of rows contain a refusal and exactly 10% contain a
certified unordered parallel transition.  Every row has 3-8 turns and 12-28
assistant action transitions.  The exact transition distribution is symmetric
around 20.  Work is split into independently resumable one-row shards.

GLM-5.2 on OpenRouter does not advertise a native low reasoning tier, but the
gateway accepts ``low`` and maps it to the model's nearest supported tier.  The
launcher sends ``low`` explicitly: disabling reasoning was cheaper per call but
produced enough invalid plans to be a false economy in a 20-action trace.
OpenRouter's default routing already price-weights stable providers; explicit
``:floor`` routing was rejected after a live probe because its latency made a
1,000-row run impractical.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
TOOL_POOL = ROOT / "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
BFCL_EXAMPLES = (
    ROOT / "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
)
DEFAULT_SEED = 20260729
DEFAULT_MODEL = "z-ai/glm-5.2"
CATEGORIES = (
    "Communication",
    "Events",
    "Finance",
    "Posting Api",
    "Science",
    "Storage",
    "Travel Booking",
    "Vehicle Control",
)

# Exactly 1,000 rows and a mean of exactly 20 action transitions.
STEP_COUNTS = {
    12: 20,
    13: 20,
    14: 30,
    15: 40,
    16: 60,
    17: 80,
    18: 100,
    19: 100,
    20: 100,
    21: 100,
    22: 100,
    23: 80,
    24: 60,
    25: 40,
    26: 30,
    27: 20,
    28: 20,
}
TURN_WEIGHTS = {3: 6, 4: 18, 5: 30, 6: 24, 7: 14, 8: 8}
PROFILE_COUNTS = {
    "normal": 820,
    "refusal_missing": 32,
    "refusal_ambiguity": 32,
    "refusal_unsupported": 16,
    "parallel": 80,
    "combined_missing": 10,
    "combined_ambiguity": 10,
}


class FatalProviderError(RuntimeError):
    """A provider/account error that should stop scheduling new work."""


@dataclass(frozen=True)
class Spec:
    index: int
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

    @property
    def stem(self) -> str:
        category = self.category.lower().replace(" ", "_")
        return (
            f"{self.index:04d}.{self.profile}.{category}."
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
            / "data/generated/runs/"
            "diverse_mtms_glm52_1000_20260729"
        ),
    )
    parser.add_argument("--python", default=os.getenv("APIGEN_PYTHON", sys.executable))
    parser.add_argument("--model", default=os.getenv("APIGEN_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--judge-model",
        default=os.getenv("APIGEN_JUDGE_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--max-task-restarts",
        type=int,
        default=1,
        help=(
            "Subprocess restarts per row (default: 1). The generator already "
            "has bounded candidate/turn repair; higher values reset that "
            "budget and should be used only deliberately."
        ),
    )
    parser.add_argument("--task-timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-calls-per-row", type=int, default=30)
    parser.add_argument("--max-tokens-per-row", type=int, default=100_000)
    parser.add_argument("--max-candidate-starts-per-row", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("APIGEN_REASONING_EFFORT", "low"),
        help=(
            "APIGen/OpenRouter reasoning control. Defaults to low; OpenRouter "
            "maps it to GLM-5.2's nearest supported effort."
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Run only the first N scheduled rows (smoke/partial run).",
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def profile_properties(profile: str) -> tuple[str, str, str]:
    if profile == "normal":
        return "normal", "ordinary", "random"
    if profile == "parallel":
        return "parallel", "terminal", "random"
    if profile == "refusal_unsupported":
        return "refusal", "terminal", "no_appropriate_function"
    if profile == "refusal_missing":
        return "refusal", "interactive-refusal", "missing_argument"
    if profile == "refusal_ambiguity":
        return "refusal", "interactive-refusal", "ambiguity"
    if profile == "combined_missing":
        return "mixed", "combined", "missing_argument"
    if profile == "combined_ambiguity":
        return "mixed", "combined", "ambiguity"
    raise ValueError(f"Unknown profile: {profile}")


def valid_turn_count(*, schedule: str, steps: int, turns: int) -> bool:
    if not 3 <= turns <= 8:
        return False
    if schedule in {"interactive-refusal", "combined"} and turns < 5:
        return False
    fixed_ones = 0
    if schedule in {"terminal", "interactive-refusal"}:
        fixed_ones = 1
    elif schedule == "combined":
        fixed_ones = 2
    maximum = 6 * turns - 5 * fixed_ones
    return turns <= steps <= maximum


def choose_turn_count(
    rng: random.Random,
    *,
    schedule: str,
    steps: int,
) -> int:
    candidates = [
        turns
        for turns in TURN_WEIGHTS
        if valid_turn_count(schedule=schedule, steps=steps, turns=turns)
    ]
    if not candidates:
        raise RuntimeError(
            f"No turn count supports schedule={schedule}, steps={steps}"
        )
    return rng.choices(
        candidates,
        weights=[TURN_WEIGHTS[turns] for turns in candidates],
        k=1,
    )[0]


def choose_action_vector(
    rng: random.Random,
    *,
    steps: int,
    turns: int,
    fixed: dict[int, int],
) -> tuple[int, ...]:
    """Create a variable, reasonably balanced 1-6 action composition."""

    vector = [fixed.get(index, 1) for index in range(turns)]
    remaining = steps - sum(vector)
    if remaining < 0:
        raise RuntimeError("Fixed action vector exceeds requested steps")

    while remaining:
        candidates = [
            index
            for index in range(turns)
            if index not in fixed and vector[index] < 6
        ]
        if not candidates:
            raise RuntimeError(
                f"Cannot distribute {steps} actions over {turns} turns"
            )
        minimum = min(vector[index] for index in candidates)
        balanced = [
            index for index in candidates if vector[index] <= minimum + 1
        ]
        pool = balanced if rng.random() < 0.82 else candidates
        vector[rng.choice(pool)] += 1
        remaining -= 1

    # A few transfers create natural unevenness without changing totals/fixed
    # feature positions or allowing pathological 0/7-action turns.
    for _ in range(turns * 2):
        donors = [
            index
            for index in range(turns)
            if index not in fixed and vector[index] > 1
        ]
        receivers = [
            index
            for index in range(turns)
            if index not in fixed and vector[index] < 6
        ]
        if not donors or not receivers or rng.random() > 0.35:
            continue
        donor = rng.choice(donors)
        receiver = rng.choice(receivers)
        if donor != receiver:
            vector[donor] -= 1
            vector[receiver] += 1

    if sum(vector) != steps or any(not 1 <= value <= 6 for value in vector):
        raise RuntimeError(f"Invalid action vector: {vector}")
    if any(vector[index] != value for index, value in fixed.items()):
        raise RuntimeError(f"Fixed positions changed: {vector}, fixed={fixed}")
    return tuple(vector)


def build_schedule(seed: int) -> tuple[list[Spec], dict[str, Any]]:
    rng = random.Random(seed)

    profiles = [
        profile
        for profile, count in PROFILE_COUNTS.items()
        for _ in range(count)
    ]
    steps = [
        value
        for value, count in STEP_COUNTS.items()
        for _ in range(count)
    ]
    categories = [
        category for category in CATEGORIES for _ in range(125)
    ]
    widths = [width for width in (2, 3, 4, 5) for _ in range(25)]
    rng.shuffle(profiles)
    rng.shuffle(steps)
    rng.shuffle(categories)
    rng.shuffle(widths)

    if not (len(profiles) == len(steps) == len(categories) == 1000):
        raise RuntimeError("Internal 1,000-row schedule length mismatch")

    width_index = 0
    pending: list[Spec] = []
    for index, (profile, step_count, category) in enumerate(
        zip(profiles, steps, categories)
    ):
        feature, schedule, refusal_reason = profile_properties(profile)
        turns = choose_turn_count(
            rng,
            schedule=schedule,
            steps=step_count,
        )
        refusal_index: int | None = None
        fixed: dict[int, int] = {}
        if schedule == "terminal":
            fixed[turns - 1] = 1
        elif schedule in {"interactive-refusal", "combined"}:
            # Two completed turns before clarification, immediate recovery,
            # and at least one later turn.
            refusal_index = rng.choice(list(range(2, turns - 2)))
            fixed[refusal_index] = 1
            if schedule == "combined":
                fixed[turns - 1] = 1

        actual = choose_action_vector(
            rng,
            steps=step_count,
            turns=turns,
            fixed=fixed,
        )
        blueprint = list(actual)
        if refusal_index is not None:
            # The feature generator replaces this planned source turn with the
            # clarification transition and replays its plan on the next turn.
            blueprint[refusal_index] = actual[refusal_index + 1]
            blueprint[refusal_index + 1] = 1

        if feature in {"parallel", "mixed"}:
            parallel_width = widths[width_index]
            width_index += 1
        else:
            parallel_width = 2

        pending.append(
            Spec(
                index=index,
                profile=profile,
                category=category,
                feature=feature,
                schedule=schedule,
                refusal_reason=refusal_reason,
                steps=step_count,
                turns=turns,
                actual_steps_per_turn=actual,
                blueprint_actions_per_turn=tuple(blueprint),
                parallel_width=parallel_width,
                interactive_refusal_turn=(
                    refusal_index + 1 if refusal_index is not None else None
                ),
            )
        )

    refusal_rows = sum(spec.contains_refusal for spec in pending)
    parallel_rows = sum(spec.contains_parallel for spec in pending)
    summary = {
        "seed": seed,
        "rows": len(pending),
        "profiles": dict(sorted(Counter(s.profile for s in pending).items())),
        "categories": dict(sorted(Counter(s.category for s in pending).items())),
        "features": {
            "contains_refusal": refusal_rows,
            "contains_parallel": parallel_rows,
            "ordinary": sum(s.feature == "normal" for s in pending),
            "refusal_only": sum(s.feature == "refusal" for s in pending),
            "parallel_only": sum(s.feature == "parallel" for s in pending),
            "combined": sum(s.feature == "mixed" for s in pending),
        },
        "step_distribution": dict(
            sorted(Counter(s.steps for s in pending).items())
        ),
        "mean_steps": sum(s.steps for s in pending) / len(pending),
        "turn_distribution": dict(
            sorted(Counter(s.turns for s in pending).items())
        ),
        "actions_per_turn_distribution": dict(
            sorted(
                Counter(
                    count
                    for spec in pending
                    for count in spec.actual_steps_per_turn
                ).items()
            )
        ),
        "parallel_width_distribution": dict(
            sorted(
                Counter(
                    s.parallel_width for s in pending if s.contains_parallel
                ).items()
            )
        ),
        "reasoning_policy": (
            "request low explicitly; OpenRouter maps unsupported effort values "
            "to the model's nearest supported tier (GLM-5.2 currently exposes "
            "high/xhigh)"
        ),
        "provider_routing": (
            "OpenRouter default: stability-filtered, price-weighted provider "
            "load balancing; :floor was too slow in an 8-worker live probe"
        ),
    }
    if refusal_rows != 100 or parallel_rows != 100:
        raise RuntimeError(
            f"Feature cap mismatch: refusal={refusal_rows}, parallel={parallel_rows}"
        )
    if summary["mean_steps"] != 20:
        raise RuntimeError(f"Mean steps is not 20: {summary['mean_steps']}")

    # Put one clean 5-turn x 4-action Storage row first.  A --max-tasks 1
    # smoke test then exercises exactly the user's representative shape and is
    # not accidentally dominated by a 26-28-action tail example.
    representative = next(
        spec
        for spec in pending
        if (
            spec.profile == "normal"
            and spec.category == "Storage"
            and spec.turns == 5
            and spec.actual_steps_per_turn == (4, 4, 4, 4, 4)
        )
    )
    ordered = [representative, *(spec for spec in pending if spec is not representative)]
    pending = [replace(spec, index=index) for index, spec in enumerate(ordered)]
    return pending, summary


def spec_path(spec: Spec, work_dir: Path) -> Path:
    return work_dir / "rows" / f"{spec.stem}.jsonl"


def row_shape(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    turns = row.get("conversation", {}).get("turns", [])
    step_vector = [len(turn.get("steps", [])) for turn in turns]
    call_vector = [
        sum(
            len(step.get("tool_calls", []))
            for step in turn.get("steps", [])
        )
        for turn in turns
    ]
    return step_vector, call_vector


def validate_row(spec: Spec, row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    metadata = row.get("generation_metadata", {})
    verification = row.get("verification_result", {})
    turns = row.get("conversation", {}).get("turns", [])
    step_vector, call_vector = row_shape(row)

    if tuple(step_vector) != spec.actual_steps_per_turn:
        errors.append(
            f"STEP_VECTOR:{step_vector}!={list(spec.actual_steps_per_turn)}"
        )
    if sum(step_vector) != spec.steps:
        errors.append(f"STEP_COUNT:{sum(step_vector)}!={spec.steps}")
    if len(turns) != spec.turns:
        errors.append(f"TURN_COUNT:{len(turns)}!={spec.turns}")
    if metadata.get("focus_category") != spec.category:
        errors.append(
            f"CATEGORY:{metadata.get('focus_category')}!={spec.category}"
        )
    if metadata.get("rl_quality_gate_passed") is not True:
        errors.append("RL_QUALITY_GATE_NOT_PASSED")
    if verification.get("overall_verification_passed") is not True:
        errors.append("OVERALL_VERIFICATION_NOT_PASSED")
    if any(
        turn.get("quality_verification", {}).get("passed") is not True
        for turn in turns
    ):
        errors.append("TURN_QUALITY_NOT_PASSED")
    def feature_replaces_blueprint_count(turn_index: int) -> bool:
        # Feature generators intentionally replace one ordinary blueprint
        # transition.  A clarification/refusal turn can contain no tool call,
        # while the terminal parallel transition contains one step but several
        # expected tools.  Compare expected-tool counts only on untouched
        # ordinary turns.
        if (
            spec.schedule in {"interactive-refusal", "combined"}
            and spec.interactive_refusal_turn == turn_index + 1
        ):
            return True
        if turn_index == len(turns) - 1:
            if spec.contains_parallel:
                return True
            if spec.feature == "refusal" and spec.schedule == "terminal":
                return True
        return False

    if any(
        len(turn.get("expected_tools", [])) != expected
        for turn_index, (turn, expected) in enumerate(
            zip(turns, spec.actual_steps_per_turn)
        )
        if not feature_replaces_blueprint_count(turn_index)
    ):
        errors.append("EXPECTED_TOOL_COUNT_MISMATCH")

    if spec.feature == "normal":
        if metadata.get("contains_refusal") is True:
            errors.append("UNEXPECTED_REFUSAL")
        if metadata.get("contains_parallel") is True:
            errors.append("UNEXPECTED_PARALLEL")
        return errors

    if metadata.get("feature_schedule") != spec.schedule:
        errors.append(
            f"SCHEDULE:{metadata.get('feature_schedule')}!={spec.schedule}"
        )
    if metadata.get("feature_difficulty") != "hard":
        errors.append("DIFFICULTY_NOT_HARD")
    naturalization = metadata.get("query_naturalization", {})
    certificate = naturalization.get("certificate", {})
    if not (
        naturalization.get("enabled") is True
        and naturalization.get("protected_tokens_preserved") is True
        and certificate.get("semantic_plan_preserved") is True
        and certificate.get("natural_conversation") is True
        and certificate.get("no_tool_syntax") is True
        and certificate.get("avoids_unnecessary_internal_ids") is True
    ):
        errors.append("FEATURE_QUERIES_NOT_CERTIFIED_NATURAL")

    contains_refusal = metadata.get("contains_refusal") is True
    contains_parallel = metadata.get("contains_parallel") is True
    if contains_refusal != spec.contains_refusal:
        errors.append("REFUSAL_FEATURE_MISMATCH")
    if contains_parallel != spec.contains_parallel:
        errors.append("PARALLEL_FEATURE_MISMATCH")

    if spec.schedule in {"interactive-refusal", "combined"}:
        if metadata.get("clarification_recovered") is not True:
            errors.append("CLARIFICATION_NOT_RECOVERED")
        expected = [spec.interactive_refusal_turn]
        if metadata.get("refusal_turns") != expected:
            errors.append(
                f"REFUSAL_TURNS:{metadata.get('refusal_turns')}!={expected}"
            )
    if spec.contains_parallel:
        if not call_vector or call_vector[-1] != spec.parallel_width:
            errors.append(
                f"PARALLEL_WIDTH:{call_vector[-1] if call_vector else None}!="
                f"{spec.parallel_width}"
            )
        final_steps = turns[-1].get("steps", []) if turns else []
        if not (
            len(final_steps) == 1
            and final_steps[0].get("execution_mode") == "parallel"
            and final_steps[0].get("call_order_matters") is False
        ):
            errors.append("FINAL_PARALLEL_GROUP_NOT_UNORDERED")
    return errors


def semantic_signature(row: dict[str, Any]) -> str:
    payload = []
    if isinstance(row.get("conversation"), dict):
        for turn in row["conversation"].get("turns", []):
            groups = []
            for step in turn.get("steps", []):
                calls = [
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in step.get("tool_calls", [])
                ]
                if step.get("call_order_matters", True) is False:
                    calls.sort(
                        key=lambda value: json.dumps(
                            value,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        )
                    )
                groups.append(
                    {
                        "execution_mode": step.get(
                            "execution_mode", "sequential"
                        ),
                        "calls": calls,
                    }
                )
            payload.append(
                {"query": turn.get("user_query", ""), "groups": groups}
            )
    else:
        for step in row.get("trajectory", {}).get("steps", []):
            payload.append(
                [
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in step.get("tool_calls", [])
                ]
            )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_dedupe_sources() -> list[Path]:
    generated = ROOT / "data/generated"
    sources = [
        generated / "long7_15_grok45_500_20260727_naturalized.jsonl",
        generated / "runs/hard_natural_balanced_500_20260728/shards",
    ]
    result: list[Path] = []
    for source in sources:
        if source.is_file():
            result.append(source)
        elif source.is_dir():
            result.extend(sorted(source.glob("*.jsonl")))
    return result


def seed_registry(registry: Path, sources: Iterable[Path]) -> dict[str, int]:
    existing = (
        {
            line.strip()
            for line in registry.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if registry.exists()
        else set()
    )
    source_rows = 0
    seeded = 0
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as destination:
        fcntl.flock(destination.fileno(), fcntl.LOCK_EX)
        for source in sources:
            for row in read_jsonl(source):
                if not (
                    isinstance(row.get("conversation"), dict)
                    or isinstance(row.get("trajectory"), dict)
                ):
                    continue
                source_rows += 1
                signature = semantic_signature(row)
                if signature in existing:
                    continue
                destination.write(signature + "\n")
                existing.add(signature)
                seeded += 1
        destination.flush()
        os.fsync(destination.fileno())
    return {
        "source_rows": source_rows,
        "new_seed_signatures": seeded,
        "registry_signatures": len(existing),
    }


def feature_args(spec: Spec) -> list[str]:
    if spec.feature == "normal":
        return ["--no-allow-refusal", "--no-allow-parallel"]
    common = [
        "--require-feature",
        "--feature-difficulty",
        "hard",
        "--naturalize-queries",
        "--multi-turn-feature-schedule",
        spec.schedule,
        "--refusal-reason",
        spec.refusal_reason,
    ]
    if spec.feature == "refusal":
        return [
            *common,
            "--allow-refusal",
            "--refusal-rate",
            "1.0",
            "--no-allow-parallel",
        ]
    if spec.feature == "parallel":
        return [
            *common,
            "--allow-parallel",
            "--parallel-rate",
            "1.0",
            "--no-allow-refusal",
        ]
    return [
        *common,
        "--allow-refusal",
        "--refusal-rate",
        "0.5",
        "--allow-parallel",
        "--parallel-rate",
        "0.5",
    ]


def command_for(
    spec: Spec,
    *,
    args: argparse.Namespace,
    output: Path,
    registry: Path,
    usage_report: Path,
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
        args.model,
        "--judge-model",
        args.judge_model,
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
        "--tool-pool",
        str(TOOL_POOL),
        "--invocation-examples",
        str(BFCL_EXAMPLES),
        "--dedupe-registry",
        str(registry),
        *feature_args(spec),
        "--output",
        str(output),
    ]
    if spec.interactive_refusal_turn is not None:
        command.extend(
            [
                "--interactive-refusal-turn",
                str(spec.interactive_refusal_turn),
            ]
        )
    return command


def run_spec(
    spec: Spec,
    *,
    args: argparse.Namespace,
    work_dir: Path,
    registry: Path,
) -> tuple[int, str]:
    output = spec_path(spec, work_dir)
    log = work_dir / "logs" / f"{spec.stem}.log"
    usage_report = work_dir / "usage" / f"{spec.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    usage_report.parent.mkdir(parents=True, exist_ok=True)

    rows = count_rows(output)
    if rows == 1:
        errors = validate_row(spec, next(read_jsonl(output)))
        if not errors:
            return spec.index, "already-complete"
        raise RuntimeError(f"Existing row {output} failed validation: {errors}")
    if rows > 1:
        raise RuntimeError(f"{output} contains more than one row")

    command = command_for(
        spec,
        args=args,
        output=output,
        registry=registry,
        usage_report=usage_report,
    )
    environment = {
        **os.environ,
        "APIGEN_LLM_TIMEOUT": os.getenv("APIGEN_LLM_TIMEOUT", "240"),
        "APIGEN_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
        "APIGEN_REASONING_EFFORT": args.reasoning_effort,
    }
    last_status = ""
    for restart in range(1, args.max_task_restarts + 1):
        restart_usage_report = usage_report.with_name(
            f"{usage_report.stem}.r{restart}{usage_report.suffix}"
        )
        restart_command = list(command)
        usage_index = restart_command.index("--usage-report") + 1
        restart_command[usage_index] = str(restart_usage_report)
        with log.open("a", encoding="utf-8") as destination:
            destination.write(
                json.dumps(
                    {
                        "event": "launch",
                        "restart": restart,
                        "spec": asdict(spec),
                        "command": restart_command,
                        "reasoning_effort": args.reasoning_effort,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
            try:
                completed = subprocess.run(
                    restart_command,
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
                    json.dumps(
                        {"event": "timeout", "restart": restart}
                    )
                    + "\n"
                )

        if count_rows(output) == 1:
            row = next(read_jsonl(output))
            errors = validate_row(spec, row)
            if not errors:
                return spec.index, f"generated-restart-{restart}"
            raise RuntimeError(
                f"Generated row {output} failed validation: {errors}"
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
                raise FatalProviderError(
                    f"{spec.stem} hit a provider/account limit; see {log}"
                )
        if restart < args.max_task_restarts:
            time.sleep(min(2**restart, 60))
    raise RuntimeError(
        f"{spec.stem} exhausted {args.max_task_restarts} restarts "
        f"({last_status}); see {log}"
    )


def reasoning_payload(reasoning_effort: str) -> dict[str, Any]:
    normalized = reasoning_effort.strip().lower()
    if normalized in {"off", "none", "disabled", "false", "0"}:
        return {"enabled": False, "exclude": True}
    return {"effort": normalized, "exclude": True}


def provider_probe(
    *,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> None:
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError("OPENAI_API_BASE and OPENAI_API_KEY must be set")
    response: requests.Response | None = None
    for attempt in range(1, 6):
        response = requests.post(
            api_base.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Reply with exactly OK."}
                ],
                "temperature": 0,
                "max_tokens": min(max_output_tokens, 8),
                "reasoning": reasoning_payload(reasoning_effort),
            },
            timeout=60,
        )
        if response.status_code < 400:
            return
        if response.status_code not in {408, 409, 429} and not (
            500 <= response.status_code < 600
        ):
            break
        if attempt < 5:
            time.sleep(min(2**attempt, 16))
    assert response is not None
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = response.text
    error_type = (
        FatalProviderError
        if response.status_code in {401, 402, 403}
        else RuntimeError
    )
    raise error_type(
        f"Provider probe failed for {model}: HTTP "
        f"{response.status_code}: {message[:300]}"
    )


def complete_rows(
    specs: Iterable[Spec],
    work_dir: Path,
) -> list[tuple[Spec, dict[str, Any]]]:
    complete = []
    for spec in specs:
        path = spec_path(spec, work_dir)
        if count_rows(path) != 1:
            continue
        row = next(read_jsonl(path))
        if not validate_row(spec, row):
            complete.append((spec, row))
    return complete


def usage_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
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
    count = 0
    for row in rows:
        count += 1
        usage = row.get("token_usage", {})
        for key in keys:
            result[key] += usage.get(key, 0) or 0
    result["rows"] = count
    result["average_cost_usd"] = (
        float(result["cost_usd"]) / count if count else 0.0
    )
    result["average_total_tokens"] = (
        float(result["total_tokens"]) / count if count else 0.0
    )
    return result


def subprocess_usage_summary(work_dir: Path) -> dict[str, Any]:
    """Account for accepted and discarded child processes alike."""
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
    reports = sorted((work_dir / "usage").glob("*.json"))
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


def write_progress(
    *,
    specs: list[Spec],
    selected: list[Spec],
    work_dir: Path,
    output_dir: Path,
    failures: list[str],
) -> dict[str, Any]:
    complete = complete_rows(selected, work_dir)
    partial = output_dir / "accepted.partial.jsonl"
    atomic_jsonl(partial, (row for _, row in complete))
    accepted_usage = usage_summary(row for _, row in complete)
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
    discarded_usage = {
        key: max(0, all_usage.get(key, 0) - accepted_usage.get(key, 0))
        for key in usage_keys
    }
    summary = {
        "scheduled_total": len(specs),
        "selected_this_run": len(selected),
        "accepted": len(complete),
        "remaining_selected": len(selected) - len(complete),
        "failures": len(failures),
        "partial_dataset": str(partial),
        "usage": accepted_usage,
        "all_subprocess_usage": all_usage,
        "discarded_subprocess_usage": discarded_usage,
        "updated_at_epoch": time.time(),
    }
    atomic_json(output_dir / "status.json", summary)
    return summary


def finalize(
    *,
    args: argparse.Namespace,
    specs: list[Spec],
    schedule_summary: dict[str, Any],
    work_dir: Path,
    output_dir: Path,
    registry: Path,
    registry_seed: dict[str, int],
) -> dict[str, Any]:
    complete = complete_rows(specs, work_dir)
    if len(complete) != 1000:
        raise RuntimeError(f"Cannot finalize {len(complete)}/1000 rows")
    rows = [row for _, row in complete]
    signatures = [semantic_signature(row) for row in rows]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("Generated dataset contains semantic duplicates")

    merged = output_dir / "diverse_mtms_glm52_1000.jsonl"
    atomic_jsonl(merged, rows)
    observed_steps = Counter()
    observed_turns = Counter()
    observed_categories = Counter()
    observed_profiles = Counter()
    for spec, row in complete:
        step_vector, _ = row_shape(row)
        observed_steps[sum(step_vector)] += 1
        observed_turns[len(step_vector)] += 1
        observed_categories[spec.category] += 1
        observed_profiles[spec.profile] += 1

    if dict(sorted(observed_steps.items())) != schedule_summary[
        "step_distribution"
    ]:
        raise RuntimeError("Observed step distribution differs from schedule")

    compressed_logs = 0
    for log in (work_dir / "logs").glob("*.log"):
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
        "dataset": "diverse-multiturn-multistep-glm52-1000",
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "total_rows": len(rows),
        "merged": str(merged),
        "observed_profile_counts": dict(sorted(observed_profiles.items())),
        "observed_category_counts": dict(sorted(observed_categories.items())),
        "observed_step_distribution": dict(sorted(observed_steps.items())),
        "observed_turn_distribution": dict(sorted(observed_turns.items())),
        "unique_semantic_signatures": len(set(signatures)),
        "usage": usage_summary(rows),
        "all_subprocess_usage": subprocess_usage_summary(work_dir),
        "schedule": schedule_summary,
        "semantic_dedupe_registry": str(registry),
        "registry_seed": registry_seed,
        "compressed_log_files": compressed_logs,
    }
    atomic_json(output_dir / "dataset_manifest.json", manifest)
    atomic_json(output_dir / "cost_report.json", manifest["usage"])
    return manifest


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
    atomic_jsonl(
        output_dir / "generation_schedule.jsonl",
        (asdict(spec) for spec in specs),
    )
    atomic_json(output_dir / "schedule_summary.json", schedule_summary)
    if not args.quiet_schedule:
        print(json.dumps(schedule_summary, ensure_ascii=False, indent=2))
    if args.schedule_only:
        return 0

    if not args.skip_provider_probe:
        provider_probe(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )

    registry = output_dir / "semantic_signatures.registry"
    dedupe_sources = [
        *default_dedupe_sources(),
        *(path.resolve() for path in args.dedupe_against),
    ]
    dedupe_sources = list(dict.fromkeys(dedupe_sources))
    registry_seed = seed_registry(registry, dedupe_sources)
    atomic_json(
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
            except FatalProviderError as exc:
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
                f"cost=${progress['usage']['cost_usd']:.4f}",
                flush=True,
            )
            if fatal:
                for pending in futures:
                    pending.cancel()
                break

    if failures:
        atomic_json(
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
        args=args,
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
