#!/usr/bin/env python3
"""Generate 500 BFCL-shaped, natural refusal/parallel trajectories.

The schedule is deterministic and intentionally separates three quantities:

* conversation turns, sampled from the local BFCL-v3 empirical distribution;
* assistant action transitions, exactly balanced across 7-15;
* calls inside a certified parallel transition, balanced across widths 3-5.

Feature balance:

* 200 refusal-only rows (160 clarification/recovery, 40 unsupported);
* 200 parallel-only rows;
* 100 combined clarification/recovery + final parallel rows.

Generation is one row per resumable work item. Completed rows are validated
before being accepted, so rerunning this launcher only fills missing work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = tuple(range(7, 16))
DEFAULT_SEED = 20260728
BFCL_EXAMPLES = (
    ROOT / "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
)
TOOL_POOL = ROOT / "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"


class FatalProviderError(RuntimeError):
    """Non-retryable provider/account failure shared by all work items."""


@dataclass(frozen=True)
class Profile:
    name: str
    rows: int
    feature: str
    schedule: str
    refusal_reason: str
    length_offset: int


@dataclass(frozen=True)
class Spec:
    index: int
    profile: str
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
        return (
            f"{self.index:04d}.{self.profile}.s{self.steps}.t{self.turns}."
            f"w{self.parallel_width}"
        )


PROFILES = (
    Profile(
        "refusal_missing",
        80,
        "refusal",
        "interactive-refusal",
        "missing_argument",
        0,
    ),
    Profile(
        "refusal_ambiguity",
        80,
        "refusal",
        "interactive-refusal",
        "ambiguity",
        3,
    ),
    Profile(
        "refusal_unsupported",
        40,
        "refusal",
        "terminal",
        "no_appropriate_function",
        6,
    ),
    Profile("parallel", 200, "parallel", "terminal", "random", 2),
    Profile(
        "combined_missing",
        50,
        "mixed",
        "combined",
        "missing_argument",
        4,
    ),
    Profile(
        "combined_ambiguity",
        50,
        "mixed",
        "combined",
        "ambiguity",
        10,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "data/generated/runs/"
            "refusal_parallel_bfcl_shaped_steps7_15_500_20260728"
        ),
    )
    parser.add_argument("--python", default=os.getenv("APIGEN_PYTHON", sys.executable))
    parser.add_argument("--model", default=os.getenv("APIGEN_MODEL", "x-ai/grok-4.5"))
    parser.add_argument(
        "--judge-model",
        default=os.getenv("APIGEN_JUDGE_MODEL", "x-ai/grok-4.5"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--max-task-restarts", type=int, default=12)
    parser.add_argument("--task-timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-tasks",
        type=int,
        help=(
            "Run only the first N scheduled tasks and stop without final merge; "
            "intended for a smoke test."
        ),
    )
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="Write and validate the deterministic schedule without API calls.",
    )
    parser.add_argument(
        "--quiet-schedule",
        action="store_true",
        help="Do not print the full schedule summary on every resumable launch.",
    )
    parser.add_argument(
        "--skip-provider-probe",
        action="store_true",
        help="Skip the small OpenAI-compatible endpoint availability probe.",
    )
    parser.add_argument(
        "--dedupe-against",
        action="append",
        default=[],
        type=Path,
        metavar="JSONL",
        help="Seed the shared semantic-signature registry from this JSONL.",
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


def bfcl_distributions() -> dict[str, Counter[int]]:
    """Reconstruct BFCL testcase turn lengths and calls per turn locally."""

    grouped: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in read_jsonl(BFCL_EXAMPLES):
        grouped[str(row["test_case_id"])][int(row["turn_index"])] += 1

    turn_counts: Counter[int] = Counter()
    calls_per_turn: Counter[int] = Counter()
    capped_calls_per_turn: Counter[int] = Counter()
    total_calls: Counter[int] = Counter()
    for turns in grouped.values():
        turn_counts[len(turns)] += 1
        total_calls[sum(turns.values())] += 1
        calls_per_turn.update(turns.values())
        capped_calls_per_turn.update(min(value, 3) for value in turns.values())
    return {
        "turn_counts": turn_counts,
        "calls_per_turn": calls_per_turn,
        "capped_calls_per_turn": capped_calls_per_turn,
        "total_calls": total_calls,
    }


def balanced_length_counts(profile: Profile) -> dict[int, int]:
    base, remainder = divmod(profile.rows, len(LENGTHS))
    return {
        steps: base + ((index - profile.length_offset) % len(LENGTHS) < remainder)
        for index, steps in enumerate(LENGTHS)
    }


def valid_turn_count(
    *,
    schedule: str,
    steps: int,
    turns: int,
) -> bool:
    if schedule in {"interactive-refusal", "combined"} and turns < 5:
        return False
    # Refusal and parallel transitions each occupy one action transition even
    # when the parallel transition contains several independent calls.
    maximum = 3 * turns - (4 if schedule == "combined" else 2)
    return turns <= steps <= maximum


def choose_turn_count(
    *,
    rng: random.Random,
    turn_weights: Counter[int],
    schedule: str,
    steps: int,
) -> int:
    candidates = [
        (turns, weight)
        for turns, weight in sorted(turn_weights.items())
        if turns >= 2
        and valid_turn_count(schedule=schedule, steps=steps, turns=turns)
    ]
    if not candidates:
        raise RuntimeError(
            f"No BFCL-shaped turn count supports schedule={schedule}, steps={steps}"
        )
    return rng.choices(
        [turns for turns, _ in candidates],
        weights=[weight for _, weight in candidates],
        k=1,
    )[0]


def choose_action_vector(
    *,
    rng: random.Random,
    calls_per_turn_weights: Counter[int],
    steps: int,
    turns: int,
    fixed: dict[int, int],
) -> tuple[int, ...]:
    candidates: list[tuple[int, ...]] = []
    weights: list[int] = []
    for vector in itertools.product((1, 2, 3), repeat=turns):
        if sum(vector) != steps:
            continue
        if any(vector[index] != value for index, value in fixed.items()):
            continue
        weight = math.prod(calls_per_turn_weights[value] for value in vector)
        candidates.append(vector)
        weights.append(weight)
    if not candidates:
        raise RuntimeError(
            f"No action vector supports steps={steps}, turns={turns}, fixed={fixed}"
        )
    return rng.choices(candidates, weights=weights, k=1)[0]


def build_schedule(seed: int) -> tuple[list[Spec], dict[str, Any]]:
    empirical = bfcl_distributions()
    rng = random.Random(seed)
    pending: list[dict[str, Any]] = []
    width_cursor = 0

    for profile in PROFILES:
        for steps, rows in balanced_length_counts(profile).items():
            for _ in range(rows):
                turns = choose_turn_count(
                    rng=rng,
                    turn_weights=empirical["turn_counts"],
                    schedule=profile.schedule,
                    steps=steps,
                )
                refusal_index: int | None = None
                fixed: dict[int, int]
                if profile.schedule == "terminal":
                    fixed = {turns - 1: 1}
                else:
                    # The feature implementation requires two prior turns,
                    # immediate recovery, and at least one subsequent turn.
                    refusal_index = rng.choice(list(range(2, turns - 2)))
                    fixed = {refusal_index: 1}
                    if profile.schedule == "combined":
                        fixed[turns - 1] = 1

                actual = choose_action_vector(
                    rng=rng,
                    calls_per_turn_weights=empirical[
                        "capped_calls_per_turn"
                    ],
                    steps=steps,
                    turns=turns,
                    fixed=fixed,
                )
                blueprint = list(actual)
                if refusal_index is not None:
                    # The blocked source turn is replayed on the immediately
                    # following recovery turn. Its original following plan is
                    # intentionally replaced by the recovery.
                    blueprint[refusal_index] = actual[refusal_index + 1]
                    blueprint[refusal_index + 1] = 1

                if profile.feature in {"parallel", "mixed"}:
                    parallel_width = (3, 4, 5)[width_cursor % 3]
                    width_cursor += 1
                else:
                    parallel_width = 2
                pending.append(
                    {
                        "profile": profile,
                        "steps": steps,
                        "turns": turns,
                        "actual": actual,
                        "blueprint": tuple(blueprint),
                        "parallel_width": parallel_width,
                        "interactive_refusal_turn": (
                            refusal_index + 1
                            if refusal_index is not None
                            else None
                        ),
                    }
                )

    if len(pending) != 500:
        raise RuntimeError(f"Internal schedule error: {len(pending)} != 500")
    # Interleave feature families so a smoke run and early partial artifact are
    # representative rather than containing only the first profile.
    rng.shuffle(pending)
    specs = [
        Spec(
            index=index,
            profile=item["profile"].name,
            feature=item["profile"].feature,
            schedule=item["profile"].schedule,
            refusal_reason=item["profile"].refusal_reason,
            steps=item["steps"],
            turns=item["turns"],
            actual_steps_per_turn=item["actual"],
            blueprint_actions_per_turn=item["blueprint"],
            parallel_width=item["parallel_width"],
            interactive_refusal_turn=item["interactive_refusal_turn"],
        )
        for index, item in enumerate(pending)
    ]

    summary = {
        "seed": seed,
        "rows": len(specs),
        "profiles": dict(Counter(spec.profile for spec in specs)),
        "features": dict(Counter(spec.feature for spec in specs)),
        "schedules": dict(Counter(spec.schedule for spec in specs)),
        "step_distribution": dict(
            sorted(Counter(spec.steps for spec in specs).items())
        ),
        "turn_distribution": dict(
            sorted(Counter(spec.turns for spec in specs).items())
        ),
        "scheduled_actions_per_turn": dict(
            sorted(
                Counter(
                    count
                    for spec in specs
                    for count in spec.actual_steps_per_turn
                ).items()
            )
        ),
        "parallel_width_distribution": dict(
            sorted(
                Counter(
                    spec.parallel_width
                    for spec in specs
                    if spec.feature in {"parallel", "mixed"}
                ).items()
            )
        ),
        "bfcl_reference": {
            "source": str(BFCL_EXAMPLES),
            "testcases": sum(empirical["turn_counts"].values()),
            "turn_distribution": dict(
                sorted(empirical["turn_counts"].items())
            ),
            "calls_per_turn_distribution": dict(
                sorted(empirical["calls_per_turn"].items())
            ),
            "capped_calls_per_turn_distribution": dict(
                sorted(empirical["capped_calls_per_turn"].items())
            ),
        },
        "conditioning": (
            "BFCL empirical turn-count and calls-per-turn weights, conditioned "
            "on exact 7-15 steps and feature-schedule feasibility; call counts "
            "above 3 are capped to the generator's certified ordinary-turn max."
        ),
    }
    return specs, summary


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
    step_vector, call_vector = row_shape(row)

    if tuple(step_vector) != spec.actual_steps_per_turn:
        errors.append(
            f"STEP_VECTOR:{step_vector}!={list(spec.actual_steps_per_turn)}"
        )
    if sum(step_vector) != spec.steps:
        errors.append(f"STEP_COUNT:{sum(step_vector)}!={spec.steps}")
    if len(step_vector) != spec.turns:
        errors.append(f"TURN_COUNT:{len(step_vector)}!={spec.turns}")
    if metadata.get("feature_schedule") != spec.schedule:
        errors.append(
            f"SCHEDULE:{metadata.get('feature_schedule')}!={spec.schedule}"
        )
    if metadata.get("feature_difficulty") != "hard":
        errors.append("DIFFICULTY_NOT_HARD")
    if metadata.get("rl_quality_gate_passed") is not True:
        errors.append("RL_QUALITY_GATE_NOT_PASSED")
    if verification.get("overall_verification_passed") is not True:
        errors.append("OVERALL_VERIFICATION_NOT_PASSED")
    naturalization = metadata.get("query_naturalization", {})
    certificate = naturalization.get("certificate", {})
    if not (
        naturalization.get("enabled") is True
        and naturalization.get("rewritten") is True
        and naturalization.get("protected_tokens_preserved") is True
        and certificate.get("semantic_plan_preserved") is True
        and certificate.get("natural_conversation") is True
        and certificate.get("no_tool_syntax") is True
        and certificate.get("avoids_unnecessary_internal_ids") is True
    ):
        errors.append("BLUEPRINT_NOT_CERTIFIED_NATURAL")

    contains_refusal = metadata.get("contains_refusal") is True
    contains_parallel = metadata.get("contains_parallel") is True
    if spec.feature == "refusal" and (not contains_refusal or contains_parallel):
        errors.append("REFUSAL_ONLY_FEATURE_MISMATCH")
    if spec.feature == "parallel" and (contains_refusal or not contains_parallel):
        errors.append("PARALLEL_ONLY_FEATURE_MISMATCH")
    if spec.feature == "mixed" and not (
        contains_refusal and contains_parallel
    ):
        errors.append("COMBINED_FEATURE_MISMATCH")

    if spec.schedule in {"interactive-refusal", "combined"}:
        if metadata.get("clarification_recovered") is not True:
            errors.append("CLARIFICATION_NOT_RECOVERED")
        expected_turn = spec.interactive_refusal_turn
        if metadata.get("refusal_turns") != [expected_turn]:
            errors.append(
                f"REFUSAL_TURN:{metadata.get('refusal_turns')}!="
                f"{[expected_turn]}"
            )
    if spec.feature in {"parallel", "mixed"}:
        if not call_vector or call_vector[-1] != spec.parallel_width:
            errors.append(
                f"PARALLEL_WIDTH:{call_vector[-1] if call_vector else None}!="
                f"{spec.parallel_width}"
            )
        final_steps = row.get("conversation", {}).get("turns", [])[-1].get(
            "steps", []
        )
        if not (
            len(final_steps) == 1
            and final_steps[0].get("execution_mode") == "parallel"
            and final_steps[0].get("call_order_matters") is False
        ):
            errors.append("FINAL_PARALLEL_GROUP_NOT_UNORDERED")
    return errors


def feature_args(spec: Spec) -> list[str]:
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
) -> list[str]:
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
        str(spec.parallel_width),
        "--blueprint-max-actions-per-turn",
        "3",
        "--blueprint-actions-per-turn",
        ",".join(map(str, spec.blueprint_actions_per_turn)),
        "--max-parallel-width",
        str(spec.parallel_width),
        "--min-total-steps",
        str(spec.steps),
        "--max-total-steps",
        str(spec.steps),
        "--model",
        args.model,
        "--judge-model",
        args.judge_model,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    if count_rows(output) == 1:
        row = next(read_jsonl(output))
        errors = validate_row(spec, row)
        if not errors:
            return spec.index, "already-complete"
        raise RuntimeError(
            f"Existing row {output} failed validation: {errors}"
        )
    if count_rows(output) > 1:
        raise RuntimeError(f"{output} contains more than one row")

    command = command_for(
        spec,
        args=args,
        output=output,
        registry=registry,
    )
    environment = {
        **os.environ,
        "APIGEN_LLM_TIMEOUT": os.getenv("APIGEN_LLM_TIMEOUT", "240"),
        "APIGEN_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
    }
    last_status = ""
    for restart in range(1, args.max_task_restarts + 1):
        with log.open("a", encoding="utf-8") as destination:
            destination.write(
                json.dumps(
                    {
                        "event": "launch",
                        "restart": restart,
                        "spec": asdict(spec),
                        "command": command,
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
            )
            if any(marker in tail for marker in fatal_markers):
                raise FatalProviderError(
                    f"{spec.stem} hit a non-retryable provider/account limit; "
                    f"see {log}"
                )
        if restart < args.max_task_restarts:
            time.sleep(min(2**restart, 60))
    raise RuntimeError(
        f"{spec.stem} exhausted {args.max_task_restarts} restarts "
        f"({last_status}); see {log}"
    )


def provider_probe(*, model: str, max_output_tokens: int) -> None:
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
    if response.status_code >= 400:
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


def semantic_signature(row: dict[str, Any]) -> str:
    """Match generate_step_by_step's multi-turn dedupe signature."""

    if isinstance(row.get("conversation"), dict):
        payload_turns = []
        for turn in row["conversation"].get("turns", []):
            groups = []
            for step in turn.get("steps", []):
                group = [
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in step.get("tool_calls", [])
                ]
                if step.get("call_order_matters", True) is False:
                    group = sorted(
                        group,
                        key=lambda call: json.dumps(
                            call,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    )
                groups.append(
                    {
                        "execution_mode": step.get(
                            "execution_mode", "sequential"
                        ),
                        "calls": group,
                    }
                )
            payload_turns.append(
                {"query": turn.get("user_query", ""), "groups": groups}
            )
        payload: Any = payload_turns
    else:
        steps = row.get("trajectory", {}).get("steps", [])
        has_parallel_step = any(
            len(step.get("tool_calls", [])) > 1 for step in steps
        )
        if has_parallel_step:
            payload = []
            for step in steps:
                group = [
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in step.get("tool_calls", [])
                ]
                if len(group) > 1:
                    group = sorted(
                        group,
                        key=lambda call: json.dumps(
                            call,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    )
                payload.append(group)
        else:
            payload = [
                {
                    "tool_name": call.get("tool_name"),
                    "arguments": call.get("arguments", {}),
                }
                for step in steps
                for call in step.get("tool_calls", [])
            ]
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
        generated
        / "refusal_multiturn10_steps7_15_grok45_10_20260728_v2.jsonl",
        generated
        / "parallel_multiturn10_steps7_15_grok45_10_20260728_v2.jsonl",
        generated / "refusal_multistep_grok45_10_20260728.jsonl",
        generated / "parallel_multistep_grok45_10_20260728.jsonl",
    ]
    sources.extend(
        sorted(
            (
                generated
                / "runs/hard_natural_balanced_500_20260728/shards"
            ).glob("*.jsonl")
        )
    )
    return [path for path in sources if path.exists()]


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
        for source in sources:
            for row in read_jsonl(source):
                source_rows += 1
                signature = semantic_signature(row)
                if signature in existing:
                    continue
                destination.write(signature + "\n")
                existing.add(signature)
                seeded += 1
    return {
        "source_rows": source_rows,
        "new_seed_signatures": seeded,
        "registry_signatures": len(existing),
    }


def atomic_merge(paths: Iterable[Path], output: Path) -> int:
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for path in paths:
            for row in read_jsonl(path):
                destination.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                rows += 1
    temporary.replace(output)
    return rows


def run_audit(
    *,
    args: argparse.Namespace,
    input_path: Path,
    report: Path,
    rows: int,
    expected_feature: str | None = None,
    expected_schedule: str | None = None,
    require_recovery: bool = False,
) -> None:
    command = [
        args.python,
        "scripts/audit_refuse_parallel_dataset.py",
        "--input",
        str(input_path),
        "--report",
        str(report),
        "--expected-rows",
        str(rows),
        "--require-feature",
        "--expected-difficulty",
        "hard",
        "--require-naturalized",
        "--min-steps",
        "7",
        "--max-steps",
        "15",
    ]
    if expected_feature:
        command.extend(["--expected-feature", expected_feature])
    if expected_schedule:
        command.extend(["--expected-schedule", expected_schedule])
    if require_recovery:
        command.append("--require-recovery")
    subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )


def finalize(
    *,
    args: argparse.Namespace,
    specs: list[Spec],
    schedule_summary: dict[str, Any],
    output_dir: Path,
    work_dir: Path,
    registry: Path,
    registry_seed: dict[str, int],
) -> dict[str, Any]:
    by_profile: dict[str, list[Spec]] = defaultdict(list)
    for spec in specs:
        by_profile[spec.profile].append(spec)

    groups = {
        "refusal_bfcl_shaped_hard_natural_200.jsonl": {
            "profiles": {
                "refusal_missing",
                "refusal_ambiguity",
                "refusal_unsupported",
            },
            "rows": 200,
            "expected_feature": "refusal",
            "expected_schedule": None,
            "recovery": False,
        },
        "parallel_bfcl_shaped_hard_natural_200.jsonl": {
            "profiles": {"parallel"},
            "rows": 200,
            "expected_feature": "parallel",
            "expected_schedule": "terminal",
            "recovery": False,
        },
        "combined_bfcl_shaped_hard_natural_100.jsonl": {
            "profiles": {"combined_missing", "combined_ambiguity"},
            "rows": 100,
            "expected_feature": None,
            "expected_schedule": "combined",
            "recovery": True,
        },
    }
    canonical: list[Path] = []
    for filename, group in groups.items():
        selected = sorted(
            (
                spec
                for spec in specs
                if spec.profile in group["profiles"]
            ),
            key=lambda spec: spec.index,
        )
        output = output_dir / filename
        rows = atomic_merge(
            [spec_path(spec, work_dir) for spec in selected],
            output,
        )
        if rows != group["rows"]:
            raise RuntimeError(f"{output} has {rows}, need {group['rows']}")
        run_audit(
            args=args,
            input_path=output,
            report=output.with_suffix(".audit.json"),
            rows=group["rows"],
            expected_feature=group["expected_feature"],
            expected_schedule=group["expected_schedule"],
            require_recovery=group["recovery"],
        )
        canonical.append(output)

    merged = output_dir / "refusal_parallel_bfcl_shaped_500.jsonl"
    rows = atomic_merge(
        [spec_path(spec, work_dir) for spec in sorted(specs, key=lambda x: x.index)],
        merged,
    )
    if rows != 500:
        raise RuntimeError(f"{merged} has {rows}, need 500")
    run_audit(
        args=args,
        input_path=merged,
        report=merged.with_suffix(".audit.json"),
        rows=500,
    )

    observed_steps: Counter[int] = Counter()
    observed_turns: Counter[int] = Counter()
    observed_profiles: Counter[str] = Counter()
    signatures: set[str] = set()
    for spec in specs:
        row = next(read_jsonl(spec_path(spec, work_dir)))
        errors = validate_row(spec, row)
        if errors:
            raise RuntimeError(f"{spec.stem}: {errors}")
        step_vector, _ = row_shape(row)
        observed_steps[sum(step_vector)] += 1
        observed_turns[len(step_vector)] += 1
        observed_profiles[spec.profile] += 1
        signature = semantic_signature(row)
        if signature in signatures:
            raise RuntimeError(f"Duplicate generated signature at {spec.stem}")
        signatures.add(signature)
    if dict(sorted(observed_steps.items())) != schedule_summary[
        "step_distribution"
    ]:
        raise RuntimeError("Observed step distribution differs from schedule")

    task_exports: dict[str, dict[str, Any]] = {}
    for target_format in ("internal", "bfcl-native"):
        destination = output_dir / (
            f"refusal_parallel_bfcl_shaped_500.{target_format}.tasks.jsonl"
        )
        subprocess.run(
            [
                args.python,
                "scripts/export_refuse_parallel_tasks.py",
                "--input",
                str(merged),
                "--output",
                str(destination),
                "--target-format",
                target_format,
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src"},
            check=True,
        )
        task_exports[target_format] = {
            "path": str(destination),
            "rows": count_rows(destination),
        }

    # Logs are useful for diagnosis but compress extremely well and otherwise
    # dominate inode/space use for this verbose generator.
    compressed_logs = 0
    for log in (work_dir / "logs").glob("*.log"):
        gz_path = log.with_suffix(log.suffix + ".gz")
        if gz_path.exists():
            continue
        with log.open("rb") as source, gzip.open(gz_path, "wb", compresslevel=6) as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        log.unlink()
        compressed_logs += 1

    manifest = {
        "dataset": "refusal-parallel-bfcl-shaped-steps7-15-500",
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "total_rows": 500,
        "merged": str(merged),
        "canonical_parts": [
            {"path": str(path), "rows": count_rows(path)}
            for path in canonical
        ],
        "observed_profile_counts": dict(observed_profiles),
        "observed_step_distribution": dict(sorted(observed_steps.items())),
        "observed_turn_distribution": dict(sorted(observed_turns.items())),
        "unique_generated_semantic_signatures": len(signatures),
        "schedule": schedule_summary,
        "semantic_dedupe_registry": str(registry),
        "registry_seed": registry_seed,
        "task_exports": task_exports,
        "compressed_log_files": compressed_logs,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
    schedule_path = output_dir / "generation_schedule.jsonl"
    schedule_path.write_text(
        "".join(
            json.dumps(asdict(spec), ensure_ascii=False) + "\n"
            for spec in specs
        ),
        encoding="utf-8",
    )
    (output_dir / "schedule_summary.json").write_text(
        json.dumps(schedule_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.quiet_schedule:
        print(json.dumps(schedule_summary, ensure_ascii=False, indent=2))
    if args.schedule_only:
        return 0

    if not args.skip_provider_probe:
        provider_probe(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )

    registry = output_dir / "semantic_signatures.registry"
    dedupe_sources = [
        *default_dedupe_sources(),
        *(path.resolve() for path in args.dedupe_against),
    ]
    # Preserve input order while removing path duplicates.
    dedupe_sources = list(dict.fromkeys(dedupe_sources))
    registry_seed = seed_registry(registry, dedupe_sources)
    (output_dir / "dedupe_sources.json").write_text(
        json.dumps(
            {
                "sources": [str(path) for path in dedupe_sources],
                **registry_seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    selected = specs[: args.max_tasks] if args.max_tasks else specs
    failures: list[str] = []
    completed = sum(
        1
        for spec in selected
        if count_rows(spec_path(spec, work_dir)) == 1
    )
    print(
        f"Starting {len(selected)} scheduled tasks; "
        f"{completed} already have one row",
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
        fatal_provider_failure = False
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                _, status = future.result()
                done += 1
                print(
                    f"[{done:3d}/{len(selected)}] {spec.stem}: {status}",
                    flush=True,
                )
            except Exception as exc:
                failure = f"{spec.stem}: {exc}"
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
                if isinstance(exc, FatalProviderError):
                    fatal_provider_failure = True
                    for pending in futures:
                        pending.cancel()
                    break
        if fatal_provider_failure:
            print(
                "Cancelled queued tasks after a non-retryable provider error",
                flush=True,
            )

    progress = {
        "scheduled": len(selected),
        "accepted": sum(
            count_rows(spec_path(spec, work_dir)) == 1
            for spec in selected
        ),
        "failures": failures,
    }
    (output_dir / "generation_progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} task(s) failed; rerun to resume"
        )
    if args.max_tasks:
        print(json.dumps(progress, ensure_ascii=False, indent=2))
        return 0

    manifest = finalize(
        args=args,
        specs=specs,
        schedule_summary=schedule_summary,
        output_dir=output_dir,
        work_dir=work_dir,
        registry=registry,
        registry_seed=registry_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
