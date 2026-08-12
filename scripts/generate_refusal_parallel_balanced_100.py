#!/usr/bin/env python3
"""Generate a balanced, resumable 100-row refusal/parallel add-on corpus.

The four equally sized profiles are deliberately different training regimes:

* 25 terminal-multistep refusals: useful tool work, then a certified refusal;
* 25 interactive refusals: clarification, user recovery, and later tool work;
* 25 single-turn parallel batches of 3-5 independent calls;
* 25 multi-turn episodes ending in a hard, history-grounded parallel batch.

Each subprocess attempts exactly one candidate. The call ceiling is derived from
the number of real conversational turns and feature stages (10-15 calls), with
one whole-turn repair but no nested candidate retry. All usage, traces, logs,
and emitted candidate archives are kept. The scheduler is resumable and admits
no work that could cross its cost cap.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import generate_bfcl_shaped_refusal_parallel_500 as base


ROOT = Path(__file__).resolve().parents[1]
TOOL_POOL = ROOT / "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
BFCL_EXAMPLES = ROOT / "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
DEFAULT_OUTPUT = (
    ROOT
    / "data/generated/refusal_parallel_balanced_100_deepseek_coreweave_20260811"
)
MODEL = "~deepseek/deepseek-v4-flash-latest"
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
PROFILE_ORDER = (
    "refusal_terminal_multistep",
    "refusal_interactive_multiturn",
    "parallel_single_turn",
    "parallel_multi_turn",
)
# DeepSeek V4 Flash has been below this reserve for a 100k-token candidate.
# The reserve also covers a missing usage report from an interrupted process.
MAX_RESERVED_COST_PER_CANDIDATE = 0.04


@dataclass(frozen=True)
class Spec:
    index: int
    profile: str
    feature: str
    mode: str
    category: str
    feature_schedule: str
    refusal_reason: str
    steps: int
    turns: int
    actual_steps_per_turn: tuple[int, ...]
    blueprint_actions_per_turn: tuple[int, ...]
    parallel_width: int
    interactive_refusal_turn: int | None = None

    @property
    def stem(self) -> str:
        category = self.category.lower().replace(" ", "_")
        return (
            f"{self.index:03d}.{self.profile}.{category}."
            f"t{self.turns}.s{self.steps}.w{self.parallel_width}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--provider", default="CoreWeave")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts-per-slot", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--total-budget-usd", type=float, default=18.0)
    parser.add_argument("--budget-safety-reserve-usd", type=float, default=0.25)
    parser.add_argument(
        "--max-new-candidates",
        type=int,
        help="Stop after this many new subprocess attempts (smoke/debug use).",
    )
    parser.add_argument("--schedule-only", action="store_true")
    parser.add_argument(
        "--dedupe-against", action="append", type=Path, default=[]
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _profile_categories(profile_index: int) -> list[str]:
    # Three of each category plus one rotating extra = 25/profile. Across all
    # profiles no category differs by more than one row.
    result = list(CATEGORIES) * 3
    result.append(CATEGORIES[profile_index % len(CATEGORIES)])
    return result


def _cycle(values: tuple[Any, ...], count: int, offset: int = 0) -> list[Any]:
    return [values[(index + offset) % len(values)] for index in range(count)]


def build_schedule(seed: int) -> tuple[list[Spec], dict[str, Any]]:
    rng = random.Random(seed)
    empirical = base.bfcl_distributions()
    pending: list[Spec] = []

    for profile_index, profile in enumerate(PROFILE_ORDER):
        categories = _profile_categories(profile_index)
        rng.shuffle(categories)
        if profile == "refusal_terminal_multistep":
            lengths = _cycle(tuple(range(7, 13)), 25, profile_index)
            # Until ambiguity/unsupported refusals have an equally strong
            # deterministic witness, production uses the mechanically
            # checkable required-field omission path only.
            reasons = ["missing_argument"] * 25
            for slot, (steps, reason, category) in enumerate(
                zip(lengths, reasons, categories)
            ):
                turns = base.choose_turn_count(
                    rng=rng,
                    turn_weights=empirical["turn_counts"],
                    schedule="terminal",
                    steps=steps,
                )
                actual = base.choose_action_vector(
                    rng=rng,
                    calls_per_turn_weights=empirical["capped_calls_per_turn"],
                    steps=steps,
                    turns=turns,
                    fixed={turns - 1: 1},
                )
                pending.append(
                    Spec(
                        slot,
                        profile,
                        "refusal",
                        "multi-turn",
                        category,
                        "terminal",
                        reason,
                        steps,
                        turns,
                        actual,
                        actual,
                        3,
                    )
                )
        elif profile == "refusal_interactive_multiturn":
            lengths = _cycle(tuple(range(8, 16)), 25, profile_index)
            reasons = ["missing_argument"] * 25
            for slot, (steps, reason, category) in enumerate(
                zip(lengths, reasons, categories)
            ):
                turns = base.choose_turn_count(
                    rng=rng,
                    turn_weights=empirical["turn_counts"],
                    schedule="interactive-refusal",
                    steps=steps,
                )
                refusal_index = rng.choice(list(range(2, turns - 2)))
                actual = base.choose_action_vector(
                    rng=rng,
                    calls_per_turn_weights=empirical["capped_calls_per_turn"],
                    steps=steps,
                    turns=turns,
                    fixed={refusal_index: 1},
                )
                blueprint = list(actual)
                blueprint[refusal_index] = actual[refusal_index + 1]
                blueprint[refusal_index + 1] = 1
                pending.append(
                    Spec(
                        slot,
                        profile,
                        "refusal",
                        "multi-turn",
                        category,
                        "interactive-refusal",
                        reason,
                        steps,
                        turns,
                        actual,
                        tuple(blueprint),
                        3,
                        refusal_index + 1,
                    )
                )
        elif profile == "parallel_single_turn":
            widths = _cycle((3, 4, 5), 25, profile_index)
            for slot, (width, category) in enumerate(zip(widths, categories)):
                pending.append(
                    Spec(
                        slot,
                        profile,
                        "parallel",
                        "step-by-step",
                        category,
                        "terminal",
                        "random",
                        1,
                        1,
                        (1,),
                        (1,),
                        width,
                    )
                )
        else:
            lengths = _cycle(tuple(range(7, 16)), 25, profile_index)
            widths = _cycle((3, 4, 5), 25, profile_index)
            for slot, (steps, width, category) in enumerate(
                zip(lengths, widths, categories)
            ):
                turns = base.choose_turn_count(
                    rng=rng,
                    turn_weights=empirical["turn_counts"],
                    schedule="terminal",
                    steps=steps,
                )
                actual = base.choose_action_vector(
                    rng=rng,
                    calls_per_turn_weights=empirical["capped_calls_per_turn"],
                    steps=steps,
                    turns=turns,
                    fixed={turns - 1: 1},
                )
                pending.append(
                    Spec(
                        slot,
                        profile,
                        "parallel",
                        "multi-turn",
                        category,
                        "terminal",
                        "random",
                        steps,
                        turns,
                        actual,
                        actual,
                        width,
                    )
                )

    if len(pending) != 100:
        raise RuntimeError(f"Schedule error: expected 100, got {len(pending)}")
    rng.shuffle(pending)
    specs = [
        Spec(index=index, **{k: v for k, v in asdict(spec).items() if k != "index"})
        for index, spec in enumerate(pending)
    ]
    summary = {
        "rows": len(specs),
        "seed": seed,
        "profile_distribution": dict(Counter(x.profile for x in specs)),
        "feature_distribution": dict(Counter(x.feature for x in specs)),
        "mode_distribution": dict(Counter(x.mode for x in specs)),
        "category_distribution": dict(sorted(Counter(x.category for x in specs).items())),
        "multiturn_transition_distribution": dict(
            sorted(Counter(x.steps for x in specs if x.mode == "multi-turn").items())
        ),
        "turn_distribution": dict(
            sorted(Counter(x.turns for x in specs if x.mode == "multi-turn").items())
        ),
        "parallel_width_distribution": dict(
            sorted(Counter(x.parallel_width for x in specs if x.feature == "parallel").items())
        ),
        "refusal_reason_distribution": dict(
            sorted(Counter(x.refusal_reason for x in specs if x.feature == "refusal").items())
        ),
        "interpretation": {
            "refusal_terminal_multistep": (
                "ordinary successful tool transitions followed by one terminal refusal"
            ),
            "refusal_interactive_multiturn": (
                "mid-dialogue clarification/refusal, explicit user recovery, later work"
            ),
            "parallel_single_turn": "one user request and one unordered 3-5-call batch",
            "parallel_multi_turn": (
                "ordinary history followed by one hard history-grounded unordered batch"
            ),
        },
    }
    return specs, summary


def row_path(root: Path, spec: Spec) -> Path:
    return root / "rows" / f"{spec.stem}.jsonl"


def trace_glob(root: Path, spec: Spec) -> list[Path]:
    return sorted((root / "traces").glob(f"{spec.stem}.a*.jsonl"))


def call_limit(spec: Spec) -> int:
    if spec.profile == "parallel_single_turn":
        return 10
    if spec.profile == "refusal_interactive_multiturn":
        return min(15, spec.turns + 9)
    if spec.profile == "parallel_multi_turn":
        return min(15, spec.turns + 8)
    return min(14, spec.turns + 7)


def token_limit(spec: Spec) -> int:
    return 175_000 if spec.mode == "multi-turn" else 125_000


def validate_row(spec: Spec, row: dict[str, Any], *, model: str) -> list[str]:
    errors: list[str] = []
    metadata = row.get("generation_metadata", {})
    verification = row.get("verification_result", {})
    if verification.get("overall_verification_passed") is not True:
        errors.append("OVERALL_VERIFICATION_FAILED")
    if metadata.get("rl_quality_gate_passed") is not True:
        errors.append("RL_QUALITY_GATE_FAILED")
    routing = metadata.get("model_routing", {})
    expected_routing = {
        "generator": model,
        "semantic_judge": model,
        "final_response_writer": model,
        "grounding_judge": model,
    }
    if routing != expected_routing:
        errors.append(f"MODEL_ROUTING:{routing}!={expected_routing}")
    symbolic_metrics = metadata.get("symbolic_plan_metrics", {})
    if symbolic_metrics and int(symbolic_metrics.get("hidden_argument_count", -1)) != 0:
        errors.append(
            f"HIDDEN_ARGUMENTS:{symbolic_metrics.get('hidden_argument_count')}"
        )
    calls = int(
        row.get("token_usage", {}).get(
            "total_llm_calls", row.get("token_usage", {}).get("total_calls", 0)
        )
        or 0
    )
    if calls > call_limit(spec):
        errors.append(f"CALL_BUDGET:{calls}>{call_limit(spec)}")
    serialized = json.dumps(row, ensure_ascii=False)
    if "The requested actions completed successfully." in serialized:
        errors.append("LEGACY_PLACEHOLDER")

    if spec.mode == "multi-turn":
        turns = row.get("conversation", {}).get("turns", [])
        vector = [len(turn.get("steps", [])) for turn in turns]
        if tuple(vector) != spec.actual_steps_per_turn:
            errors.append(f"STEP_VECTOR:{vector}!={list(spec.actual_steps_per_turn)}")
        if metadata.get("feature_schedule") != spec.feature_schedule:
            errors.append("FEATURE_SCHEDULE_MISMATCH")
        if metadata.get("feature_difficulty") != "hard":
            errors.append("FEATURE_NOT_HARD")
        if any(not str(turn.get("assistant_response", "")).strip() for turn in turns):
            errors.append("EMPTY_ASSISTANT_RESPONSE")
        # These rows are generated naturally in one pass. Running the former
        # style-only rewrite+judge pair here added 2-3 paid calls while the
        # blueprint semantic judge and feature certifier already fail closed on
        # plan-like, discontinuous, or tool-syntax wording.
        if any(not str(turn.get("user_query", "")).strip() for turn in turns):
            errors.append("EMPTY_USER_QUERY")

        if spec.feature == "refusal":
            if metadata.get("contains_refusal") is not True or metadata.get(
                "contains_parallel"
            ) is True:
                errors.append("REFUSAL_FEATURE_MISMATCH")
            expected_turn = (
                spec.interactive_refusal_turn
                if spec.feature_schedule == "interactive-refusal"
                else spec.turns
            )
            if metadata.get("refusal_turns") != [expected_turn]:
                errors.append(
                    f"REFUSAL_TURN:{metadata.get('refusal_turns')}!=[{expected_turn}]"
                )
            feature_steps = turns[expected_turn - 1].get("steps", []) if turns else []
            feature_calls = [
                call for step in feature_steps for call in step.get("tool_calls", [])
            ]
            if len(feature_calls) != 1 or feature_calls[0].get("tool_name") != "refuse":
                errors.append("REFUSAL_ACTION_INVALID")
            elif feature_calls[0].get("arguments", {}).get("reason") != spec.refusal_reason:
                errors.append("REFUSAL_REASON_MISMATCH")
            feature_quality = (
                turns[expected_turn - 1].get("quality_verification", {})
                if turns
                else {}
            )
            refusal_certificate = feature_quality.get(
                "query_preflight", {}
            ).get("refusal_certificate", {})
            witness = refusal_certificate.get(
                "deterministic_missing_argument_witness", {}
            )
            if spec.refusal_reason == "missing_argument" and not (
                witness.get("passed") is True
                and witness.get("required_without_default") is True
                and witness.get("removed_from_source_query") is True
                and witness.get("absent_from_prior_history") is True
            ):
                errors.append("MISSING_ARGUMENT_WITNESS_INVALID")
            if spec.feature_schedule == "interactive-refusal":
                if metadata.get("clarification_recovered") is not True:
                    errors.append("CLARIFICATION_NOT_RECOVERED")
                recovery_index = int(expected_turn or 0)
                if recovery_index >= len(turns) or not turns[recovery_index].get("steps"):
                    errors.append("RECOVERY_TURN_EMPTY")
        else:
            if metadata.get("contains_parallel") is not True or metadata.get(
                "contains_refusal"
            ) is True:
                errors.append("PARALLEL_FEATURE_MISMATCH")
            final_steps = turns[-1].get("steps", []) if turns else []
            if not (
                len(final_steps) == 1
                and final_steps[0].get("execution_mode") == "parallel"
                and final_steps[0].get("call_order_matters") is False
                and len(final_steps[0].get("tool_calls", [])) == spec.parallel_width
            ):
                errors.append("FINAL_PARALLEL_GROUP_INVALID")
    else:
        trajectory = row.get("trajectory", {})
        steps = trajectory.get("steps", [])
        if not (
            len(steps) == 1
            and steps[0].get("execution_mode") == "parallel"
            and steps[0].get("call_order_matters") is False
            and len(steps[0].get("tool_calls", [])) == spec.parallel_width
        ):
            errors.append("SINGLE_TURN_PARALLEL_GROUP_INVALID")
        if metadata.get("contains_parallel") is not True:
            errors.append("MISSING_PARALLEL_METADATA")
        if metadata.get("feature_difficulty") != "standard":
            errors.append("SINGLE_TURN_DIFFICULTY_MISMATCH")
        certificate = metadata.get("parallel_certificate", {})
        if certificate.get("passed") is not True:
            errors.append("PARALLEL_CERTIFICATE_FAILED")
        if not str(trajectory.get("query", "")).strip():
            errors.append("EMPTY_USER_QUERY")
        if not str(trajectory.get("final_response", "")).strip():
            errors.append("EMPTY_FINAL_RESPONSE")
    return errors


def feature_args(spec: Spec) -> list[str]:
    difficulty = "standard" if spec.mode == "step-by-step" else "hard"
    result = [
        "--require-feature",
        "--feature-difficulty",
        difficulty,
    ]
    if spec.feature == "refusal":
        result += [
            "--allow-refusal",
            "--refusal-rate",
            "1",
            "--no-allow-parallel",
            "--multi-turn-feature-schedule",
            spec.feature_schedule,
            "--refusal-reason",
            spec.refusal_reason,
        ]
        if spec.interactive_refusal_turn is not None:
            result += [
                "--interactive-refusal-turn",
                str(spec.interactive_refusal_turn),
            ]
    else:
        result += [
            "--allow-parallel",
            "--parallel-rate",
            "1",
            "--no-allow-refusal",
            "--max-parallel-width",
            str(spec.parallel_width),
        ]
        if spec.mode == "multi-turn":
            result += ["--multi-turn-feature-schedule", "terminal"]
    return result


def command_for(
    spec: Spec,
    *,
    args: argparse.Namespace,
    output: Path,
    usage: Path,
    archive: Path,
    registry: Path,
) -> list[str]:
    command = [
        sys.executable,
        "src/generate_step_by_step.py",
        "--mode",
        spec.mode,
        "--num-datapoints",
        "1",
        "--num-actions",
        str(spec.parallel_width),
        "--category",
        spec.category,
        "--model",
        args.model,
        "--judge-model",
        args.model,
        "--final-response-model",
        args.model,
        "--grounding-model",
        args.model,
        "--optimized-pipeline",
        "--tool-pool",
        str(TOOL_POOL),
        "--invocation-examples",
        str(BFCL_EXAMPLES),
        "--dedupe-registry",
        str(registry),
        "--max-calls-per-candidate",
        str(call_limit(spec)),
        "--max-calls-per-accepted-row",
        str(call_limit(spec)),
        "--max-tokens-per-accepted-row",
        str(token_limit(spec)),
        "--max-candidate-starts-per-row",
        "1",
        "--max-turn-attempts",
        "2",
        "--no-resume",
        "--usage-report",
        str(usage),
        "--candidate-archive-dir",
        str(archive),
        *feature_args(spec),
        "--output",
        str(output),
    ]
    if spec.mode == "multi-turn":
        insertion = [
            "--num-turns",
            str(spec.turns),
            "--blueprint-max-actions-per-turn",
            "3",
            "--blueprint-actions-per-turn",
            ",".join(map(str, spec.blueprint_actions_per_turn)),
            "--min-total-steps",
            str(spec.steps),
            "--max-total-steps",
            str(spec.steps),
        ]
        command[4:4] = insertion
    return command


def next_attempt(root: Path, spec: Spec) -> int:
    attempts: list[int] = []
    for directory, suffix in (
        (root / "usage", ".json"),
        (root / "logs", ".log"),
        (root / "traces", ".jsonl"),
    ):
        for path in directory.glob(f"{spec.stem}.a*{suffix}"):
            try:
                attempts.append(int(path.name.split(".a")[-1].split(".")[0]))
            except ValueError:
                pass
    return max(attempts, default=0) + 1


def generate_once(
    spec: Spec,
    *,
    args: argparse.Namespace,
    root: Path,
    registry: Path,
) -> tuple[int, str]:
    output = row_path(root, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    if base.count_rows(output) == 1:
        errors = validate_row(spec, next(base.read_jsonl(output)), model=args.model)
        if not errors:
            return spec.index, "already-complete"
        quarantine = root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        output.replace(quarantine / f"{spec.stem}.{time.time_ns()}.jsonl")

    attempt = next_attempt(root, spec)
    usage = root / "usage" / f"{spec.stem}.a{attempt}.json"
    trace = root / "traces" / f"{spec.stem}.a{attempt}.jsonl"
    log = root / "logs" / f"{spec.stem}.a{attempt}.log"
    archive = root / "candidate_archive" / spec.stem / f"a{attempt}"
    for directory in (usage.parent, trace.parent, log.parent, archive):
        directory.mkdir(parents=True, exist_ok=True)
    command = command_for(
        spec,
        args=args,
        output=output,
        usage=usage,
        archive=archive,
        registry=registry,
    )
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv(
        "OPENROUTER_API_KEY_TOOL_CALLING"
    )
    environment = {
        **os.environ,
        "OPENAI_API_BASE": os.getenv(
            "OPENAI_API_BASE", "https://openrouter.ai/api/v1"
        ),
        "OPENAI_API_KEY": str(api_key or ""),
        "APIGEN_OPENROUTER_PROVIDER": args.provider,
        "APIGEN_REASONING_EFFORT": "off",
        "APIGEN_BLUEPRINT_GENERATE_REASONING_EFFORT": "low",
        "APIGEN_BLUEPRINT_TURN_COMPILE_REASONING_EFFORT": "low",
        "APIGEN_BLUEPRINT_QUERY_ALIGN_REASONING_EFFORT": "low",
        "APIGEN_BLUEPRINT_SEMANTIC_JUDGE_REASONING_EFFORT": "off",
        "APIGEN_REFUSAL_QUERY_GENERATE_REASONING_MAX_TOKENS": "1024",
        "APIGEN_REFUSAL_SEMANTIC_JUDGE_REASONING_EFFORT": "off",
        "APIGEN_CLARIFICATION_RECOVERY_GENERATE_REASONING_EFFORT": "off",
        "APIGEN_FINAL_RESPONSE_GENERATE_REASONING_EFFORT": "off",
        "APIGEN_FINAL_RESPONSE_GROUNDING_JUDGE_REASONING_EFFORT": "off",
        "APIGEN_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
        "APIGEN_HTTP_ATTEMPTS": "1",
        "APIGEN_APPLICATION_LLM_ATTEMPTS": "1",
        "APIGEN_LLM_TIMEOUT": "300",
        "APIGEN_LLM_TRACE_PATH": str(trace),
    }
    with log.open("w", encoding="utf-8") as destination:
        destination.write(
            json.dumps(
                {"spec": asdict(spec), "attempt": attempt, "command": command},
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
                timeout=args.timeout_seconds,
                check=False,
            )
            process_status = f"exit={completed.returncode}"
        except subprocess.TimeoutExpired:
            process_status = "timeout"

    if base.count_rows(output) == 1:
        row = next(base.read_jsonl(output))
        errors = validate_row(spec, row, model=args.model)
        if not errors:
            return spec.index, f"accepted-a{attempt}"
        quarantine = root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        output.replace(quarantine / f"{spec.stem}.a{attempt}.{time.time_ns()}.jsonl")
        return spec.index, "validation=" + ",".join(errors)

    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-32768:].casefold()
        fatal = (
            "insufficient credits",
            "requires more credits",
            "invalid api key",
            "authentication failed",
            "user not found",
        )
        if any(marker in tail for marker in fatal):
            raise base.FatalProviderError(f"provider/account failure; see {log}")
    return spec.index, f"failed-a{attempt}:{process_status}"


def valid_rows(
    specs: Iterable[Spec], root: Path, *, model: str
) -> list[tuple[Spec, dict[str, Any]]]:
    result: list[tuple[Spec, dict[str, Any]]] = []
    for spec in specs:
        path = row_path(root, spec)
        if base.count_rows(path) != 1:
            continue
        row = next(base.read_jsonl(path))
        if not validate_row(spec, row, model=model):
            result.append((spec, row))
    return result


def usage_summary(root: Path) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    reports = list((root / "usage").glob("*.json"))
    for path in reports:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_prompt_tokens",
            "total_llm_calls",
        ):
            totals[key] += int(payload.get(key, 0) or 0)
        totals["cost_usd"] += float(payload.get("cost_usd", 0) or 0)
        totals["accepted_reports"] += bool(payload.get("accepted_rows", 0))
    return {**dict(totals), "reports": len(reports)}


def budget_snapshot(
    root: Path, *, hard_cap: float, safety_reserve: float
) -> dict[str, Any]:
    usage = usage_summary(root)
    missing_reports = []
    for log in (root / "logs").glob("*.log"):
        usage_path = root / "usage" / (log.stem + ".json")
        if not usage_path.exists():
            missing_reports.append(str(log))
    missing_reserve = len(missing_reports) * MAX_RESERVED_COST_PER_CANDIDATE
    effective = float(usage.get("cost_usd", 0)) + missing_reserve + safety_reserve
    return {
        "hard_cap_usd": hard_cap,
        "reported_cost_usd": float(usage.get("cost_usd", 0)),
        "effective_spend_usd": effective,
        "missing_report_reserve_usd": missing_reserve,
        "missing_reports": missing_reports,
        "safety_reserve_usd": safety_reserve,
        "remaining_after_reserves_usd": max(0.0, hard_cap - effective),
        "usage": usage,
    }


def write_status(
    specs: list[Spec], root: Path, *, model: str, target: int = 100
) -> dict[str, Any]:
    accepted = valid_rows(specs, root, model=model)
    rows = [row for _, row in accepted]
    atomic_jsonl(root / "accepted.partial.jsonl", rows)
    profile_counts = Counter(spec.profile for spec, _ in accepted)
    status = {
        "target": target,
        "accepted": len(accepted),
        "remaining": target - len(accepted),
        "profile_counts": dict(profile_counts),
        "feature_counts": dict(Counter(spec.feature for spec, _ in accepted)),
        "mode_counts": dict(Counter(spec.mode for spec, _ in accepted)),
        "usage": usage_summary(root),
        "updated_at": time.time(),
    }
    atomic_json(root / "status.json", status)
    return status


def finalize(specs: list[Spec], root: Path, *, model: str) -> None:
    accepted = valid_rows(specs, root, model=model)
    if len(accepted) != 100:
        return
    ordered = sorted(accepted, key=lambda item: item[0].index)
    atomic_jsonl(root / "refusal_parallel_balanced_100.jsonl", (row for _, row in ordered))
    atomic_jsonl(
        root / "refusal_50.jsonl",
        (row for spec, row in ordered if spec.feature == "refusal"),
    )
    atomic_jsonl(
        root / "parallel_50.jsonl",
        (row for spec, row in ordered if spec.feature == "parallel"),
    )
    atomic_json(
        root / "dataset_manifest.json",
        {
            "rows": 100,
            "model": model,
            "parts": {"refusal": 50, "parallel": 50},
            "profiles": dict(Counter(spec.profile for spec, _ in ordered)),
            "usage": usage_summary(root),
            "merged": str(root / "refusal_parallel_balanced_100.jsonl"),
        },
    )


def main() -> int:
    args = parse_args()
    if args.max_workers < 1 or args.max_attempts_per_slot < 1:
        raise ValueError("worker and attempt limits must be positive")
    if not 0 < args.total_budget_usd <= 50:
        raise ValueError("--total-budget-usd must be in (0, 50]")
    if args.max_new_candidates is not None and args.max_new_candidates < 1:
        raise ValueError("--max-new-candidates must be positive")
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY_TOOL_CALLING")):
        raise RuntimeError("OPENROUTER_API_KEY_TOOL_CALLING (or OPENAI_API_KEY) is required")

    specs, summary = build_schedule(args.seed)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(root / "schedule.jsonl", (asdict(spec) for spec in specs))
    atomic_json(root / "schedule_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.schedule_only:
        return 0

    registry = root / "dedupe.registry"
    external_canonical = Path(
        "/mnt/shared_ru.ml.SZ-5_000264/gambashidze/tool_synth/APIGen-MT-main/"
        "data/generated/canonical_sft_rl_corpus_565_no_claude_20260803.jsonl"
    )
    stopped_general = (
        ROOT
        / "data/generated/deepseek_v4_flash_uniform8_20_500_20260810/"
        "production_50x2/deepseek.partial.jsonl"
    )
    sources = [
        *base.default_dedupe_sources(),
        external_canonical,
        stopped_general,
        *(path.resolve() for path in args.dedupe_against),
    ]
    sources = list(dict.fromkeys(path for path in sources if path.exists()))
    seed_result = base.seed_registry(registry, sources)
    atomic_json(
        root / "dedupe_sources.json",
        {"sources": [str(path) for path in sources], **seed_result},
    )

    status = write_status(specs, root, model=args.model)
    pending = [
        spec
        for spec in specs
        if not any(x.index == spec.index for x, _ in valid_rows([spec], root, model=args.model))
    ]
    attempts = Counter(
        {
            spec.index: next_attempt(root, spec) - 1
            for spec in specs
        }
    )
    queue = list(pending)
    in_flight: dict[concurrent.futures.Future, Spec] = {}
    new_candidates = 0
    lock = threading.Lock()
    budget_blocked = False
    fatal = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while queue or in_flight:
            while (
                queue
                and len(in_flight) < args.max_workers
                and status["accepted"] + len(in_flight) < 100
                and (
                    args.max_new_candidates is None
                    or new_candidates + len(in_flight) < args.max_new_candidates
                )
            ):
                budget = budget_snapshot(
                    root,
                    hard_cap=args.total_budget_usd,
                    safety_reserve=args.budget_safety_reserve_usd,
                )
                prospective = (
                    budget["effective_spend_usd"]
                    + (len(in_flight) + 1) * MAX_RESERVED_COST_PER_CANDIDATE
                )
                if prospective > args.total_budget_usd:
                    budget_blocked = True
                    break
                spec = queue.pop(0)
                attempts[spec.index] += 1
                future = executor.submit(
                    generate_once,
                    spec,
                    args=args,
                    root=root,
                    registry=registry,
                )
                in_flight[future] = spec
            if not in_flight:
                break
            done, _ = concurrent.futures.wait(
                in_flight,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                spec = in_flight.pop(future)
                new_candidates += 1
                try:
                    _, result = future.result()
                except base.FatalProviderError as exc:
                    result = f"fatal:{exc}"
                    fatal = True
                    queue.clear()
                except Exception as exc:
                    result = f"exception:{type(exc).__name__}:{exc}"
                accepted = result.startswith(("accepted-", "already-complete"))
                if (
                    not accepted
                    and not fatal
                    and attempts[spec.index] < args.max_attempts_per_slot
                ):
                    queue.append(spec)
                with lock:
                    status = write_status(specs, root, model=args.model)
                    budget = budget_snapshot(
                        root,
                        hard_cap=args.total_budget_usd,
                        safety_reserve=args.budget_safety_reserve_usd,
                    )
                    atomic_json(root / "budget_status.json", budget)
                print(
                    f"[{new_candidates}] {spec.stem}: {result}; "
                    f"accepted={status['accepted']}/100; "
                    f"profiles={status['profile_counts']}; "
                    f"cost=${budget['reported_cost_usd']:.4f}",
                    flush=True,
                )
            if fatal:
                for future in in_flight:
                    future.cancel()
                queue.clear()

    finalize(specs, root, model=args.model)
    if budget_blocked:
        print("Hard cost cap stopped new candidate admission.", file=sys.stderr)
    if status["accepted"] < 100:
        print(
            f"Incomplete but resumable: {status['accepted']}/100 accepted.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
