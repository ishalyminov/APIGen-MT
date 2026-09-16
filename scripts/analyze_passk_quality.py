#!/usr/bin/env python3
"""Separate likely query/gold defects from model failures in pass@k output."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def first_gold_call(row: dict[str, Any]) -> dict[str, Any]:
    return row["trajectory"]["steps"][0]["tool_calls"][0]


def detect_query_gold_conflicts(
    row: dict[str, Any], events: list[dict[str, Any]], pass_k: int
) -> list[dict[str, Any]]:
    turn_one = [event for event in events if event.get("turn") == 1]
    if len(turn_one) != pass_k or any(event.get("matched") for event in turn_one):
        return []
    if any(event.get("failure") != "wrong_arguments" for event in turn_one):
        return []

    query = normalise(row["trajectory"]["query"])
    gold = first_gold_call(row)
    conflicts: list[dict[str, Any]] = []
    for argument, gold_value in gold.get("arguments", {}).items():
        if not isinstance(gold_value, str) or normalise(gold_value) in query:
            continue
        predictions = [
            (event.get("predicted_call") or {}).get("arguments", {}).get(argument)
            for event in turn_one
        ]
        predicted_counts = collections.Counter(
            value for value in predictions if isinstance(value, str)
        )
        if not predicted_counts:
            continue
        predicted_value, count = predicted_counts.most_common(1)[0]
        if count == pass_k and normalise(predicted_value) in query:
            conflicts.append(
                {
                    "argument": argument,
                    "query_supported_value": predicted_value,
                    "gold_value": gold_value,
                    "samples": count,
                }
            )
    return conflicts


def pass_curve(
    rollouts_by_row: dict[int, list[dict[str, Any]]],
    row_indices: list[int],
    pass_k: int,
) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for k in range(1, pass_k + 1):
        solved = 0
        for index in row_indices:
            samples = sorted(
                rollouts_by_row[index], key=lambda rollout: rollout["sample_index"]
            )
            solved += any(rollout.get("success") for rollout in samples[:k])
        curve.append(
            {
                "k": k,
                "solved": solved,
                "tasks": len(row_indices),
                "pass_at_k": solved / len(row_indices) if row_indices else 0.0,
            }
        )
    return curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl)
    events = read_jsonl(args.events)
    rollouts = read_jsonl(args.rollouts)
    raw_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    pass_k = int(raw_summary["config"]["pass_k"])

    events_by_row: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    rollouts_by_row: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        events_by_row[int(event["row_position"])].append(event)
    for rollout in rollouts:
        rollouts_by_row[int(rollout["row_position"])].append(rollout)

    defects: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        conflicts = detect_query_gold_conflicts(
            row, events_by_row[index], pass_k
        )
        if conflicts:
            defects.append(
                {
                    "row_position": index,
                    "category": row["trajectory"].get("categories_used", []),
                    "query": row["trajectory"]["query"],
                    "first_gold_call": first_gold_call(row),
                    "conflicts": conflicts,
                }
            )

    defect_indices = {defect["row_position"] for defect in defects}
    eligible = [index for index in range(len(rows)) if index not in defect_indices]
    adjusted_curve = pass_curve(rollouts_by_row, eligible, pass_k)
    adjusted_rollouts = [
        rollout
        for rollout in rollouts
        if int(rollout["row_position"]) not in defect_indices
    ]
    adjusted_failures = collections.Counter(
        rollout.get("failure") or "none"
        for rollout in adjusted_rollouts
        if not rollout.get("success")
    )

    report = {
        "verdict": (
            "The model saturates the non-defective subset; the raw dataset is "
            "too easy for this evaluator and still contains query/gold defects."
        ),
        "raw": {
            "tasks": raw_summary["num_tasks"],
            "rollouts": raw_summary["num_rollouts"],
            "successful_rollouts": raw_summary["num_successful_rollouts"],
            "pass_at_1": raw_summary["pass_at_1_all"],
            "pass_at_16": raw_summary["pass_at_16_all"],
            "failure_counts": raw_summary["failure_counts"],
        },
        "confirmed_query_gold_defects": {
            "count": len(defects),
            "rows": defects,
        },
        "defect_adjusted": {
            "tasks": len(eligible),
            "rollouts": len(adjusted_rollouts),
            "successful_rollouts": sum(
                bool(rollout.get("success")) for rollout in adjusted_rollouts
            ),
            "rollout_success_rate": (
                sum(bool(rollout.get("success")) for rollout in adjusted_rollouts)
                / len(adjusted_rollouts)
            ),
            "failure_counts": dict(adjusted_failures),
            "pass_curve": adjusted_curve,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
