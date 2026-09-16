#!/usr/bin/env python3
"""Fail-closed audit for refusal/parallel generator output."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from refuse_parallel_eval import validate_feature_evaluation_spec


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-feature", action="store_true")
    parser.add_argument(
        "--expected-feature",
        choices=("parallel", "refusal"),
        help=(
            "Require this feature in every row. Clarification transitions count "
            "as refusal-family targets."
        ),
    )
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--expected-turns", type=int)
    parser.add_argument(
        "--expected-difficulty",
        choices=("standard", "hard"),
    )
    parser.add_argument(
        "--expected-schedule",
        choices=("terminal", "interactive-refusal", "combined"),
    )
    parser.add_argument("--require-naturalized", action="store_true")
    parser.add_argument("--require-recovery", action="store_true")
    return parser.parse_args()


def trajectory_shape(row: dict[str, Any]) -> tuple[int, int]:
    trajectory = row.get("trajectory")
    if isinstance(trajectory, dict):
        return len(trajectory.get("steps", [])), 1
    conversation = row.get("conversation", {})
    turns = conversation.get("turns", []) if isinstance(conversation, dict) else []
    return sum(len(turn.get("steps", [])) for turn in turns), len(turns)


def semantic_signature(row: dict[str, Any]) -> str:
    trajectory = row.get("trajectory")
    if isinstance(trajectory, dict):
        turns = [
            {
                "query": trajectory.get("query", ""),
                "steps": trajectory.get("steps", []),
            }
        ]
    else:
        turns = row.get("conversation", {}).get("turns", [])
    payload = []
    for turn in turns:
        steps = []
        for step in turn.get("steps", []):
            calls = [
                {
                    "name": call.get("tool_name", call.get("name", "")),
                    "arguments": call.get("arguments", {}),
                }
                for call in step.get("tool_calls", [])
            ]
            if step.get("call_order_matters", True) is False:
                calls.sort(
                    key=lambda call: json.dumps(
                        call,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
            steps.append(
                {
                    "execution_mode": step.get("execution_mode", "sequential"),
                    "calls": calls,
                }
            )
        payload.append({"query": turn.get("user_query", turn.get("query", "")), "steps": steps})
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    failures: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    step_counts: list[int] = []
    turn_counts: list[int] = []
    signatures: dict[str, int] = {}
    rows = list(read_jsonl(args.input))

    if args.expected_rows is not None and len(rows) != args.expected_rows:
        failures.append(
            {
                "row": None,
                "errors": [
                    f"ROW_COUNT_MISMATCH:{len(rows)}!={args.expected_rows}"
                ],
            }
        )

    for index, row in enumerate(rows):
        metadata = row.get("generation_metadata", {})
        verification = row.get("verification_result", {})
        errors = validate_feature_evaluation_spec(row)
        if metadata.get("rl_quality_gate_passed") is not True:
            errors.append("RL_QUALITY_GATE_NOT_PASSED")
        if verification.get("overall_verification_passed") is not True:
            errors.append("OVERALL_VERIFICATION_NOT_PASSED")
        if (
            args.expected_difficulty is not None
            and metadata.get("feature_difficulty")
            != args.expected_difficulty
        ):
            errors.append(
                "FEATURE_DIFFICULTY_MISMATCH:"
                f"{metadata.get('feature_difficulty')}"
                f"!={args.expected_difficulty}"
            )
        if (
            args.expected_schedule is not None
            and metadata.get("feature_schedule") != args.expected_schedule
        ):
            errors.append(
                "FEATURE_SCHEDULE_MISMATCH:"
                f"{metadata.get('feature_schedule')}"
                f"!={args.expected_schedule}"
            )

        spec = metadata.get("evaluation_spec", {})
        feature_modes = {
            transition.get("mode")
            for transition in spec.get("transitions", [])
            if transition.get("mode") in {"parallel", "refusal", "clarification"}
        }
        for mode in feature_modes:
            counts[mode] += 1
        if args.require_feature and not feature_modes:
            errors.append("NO_REFUSAL_OR_PARALLEL_TRANSITION")
        if args.expected_feature == "parallel" and "parallel" not in feature_modes:
            errors.append("EXPECTED_PARALLEL_TRANSITION_MISSING")
        if args.expected_feature == "refusal" and not (
            feature_modes & {"refusal", "clarification"}
        ):
            errors.append("EXPECTED_REFUSAL_TRANSITION_MISSING")
        if args.expected_schedule == "combined" and not (
            "parallel" in feature_modes
            and bool(feature_modes & {"refusal", "clarification"})
        ):
            errors.append("COMBINED_FEATURES_MISSING")

        naturalization = metadata.get("query_naturalization", {})
        feature_naturalizations = metadata.get(
            "feature_query_naturalizations", []
        )
        if args.require_naturalized:
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
            if not feature_naturalizations or any(
                item.get("enabled") is not True
                or item.get("rewritten") is not True
                or item.get("protected_tokens_preserved") is not True
                for item in feature_naturalizations
            ):
                errors.append("FEATURE_QUERY_NOT_NATURALIZED")

        clarification_events = metadata.get("clarification_events", [])
        if args.require_recovery:
            if not clarification_events:
                errors.append("CLARIFICATION_RECOVERY_MISSING")
            for event in clarification_events:
                recovery_certificate = event.get(
                    "recovery_certificate", {}
                )
                if not (
                    event.get("recovered") is True
                    and event.get("recovery_turn")
                    == event.get("refusal_turn", 0) + 1
                    and recovery_certificate.get("passed") is True
                    and recovery_certificate.get(
                        "protected_tokens_preserved"
                    )
                    is True
                ):
                    errors.append("CLARIFICATION_RECOVERY_NOT_CERTIFIED")

        steps, turns = trajectory_shape(row)
        step_counts.append(steps)
        turn_counts.append(turns)
        if args.min_steps is not None and steps < args.min_steps:
            errors.append(f"TOO_FEW_STEPS:{steps}<{args.min_steps}")
        if args.max_steps is not None and steps > args.max_steps:
            errors.append(f"TOO_MANY_STEPS:{steps}>{args.max_steps}")
        if args.expected_turns is not None and turns != args.expected_turns:
            errors.append(f"TURN_COUNT_MISMATCH:{turns}!={args.expected_turns}")

        signature = semantic_signature(row)
        if signature in signatures:
            errors.append(f"DUPLICATE_OF_ROW:{signatures[signature]}")
        else:
            signatures[signature] = index
        if errors:
            failures.append({"row": index, "errors": sorted(set(errors))})

    report = {
        "input": str(args.input),
        "rows": len(rows),
        "feature_transition_counts": dict(counts),
        "step_count_range": (
            [min(step_counts), max(step_counts)] if step_counts else []
        ),
        "turn_count_range": (
            [min(turn_counts), max(turn_counts)] if turn_counts else []
        ),
        "unique_semantic_trajectories": len(signatures),
        "difficulty_counts": dict(
            collections.Counter(
                row.get("generation_metadata", {}).get(
                    "feature_difficulty", "unknown"
                )
                for row in rows
            )
        ),
        "schedule_counts": dict(
            collections.Counter(
                row.get("generation_metadata", {}).get(
                    "feature_schedule", "unknown"
                )
                for row in rows
            )
        ),
        "naturalized_rows": sum(
            row.get("generation_metadata", {})
            .get("query_naturalization", {})
            .get("rewritten")
            is True
            for row in rows
        ),
        "recovered_clarification_rows": sum(
            row.get("generation_metadata", {}).get(
                "clarification_recovered"
            )
            is True
            for row in rows
        ),
        "passed": not failures,
        "failures": failures,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
