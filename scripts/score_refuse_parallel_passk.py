#!/usr/bin/env python3
"""Score structured refusal/parallel rollouts and compute standard pass@k.

Rollout rows should contain:

- ``task_id`` and ``sample_index``;
- preferably ``predicted_actions``: a list of assistant actions, each with a
  ``tool_calls`` list;
- alternatively ``predicted_tool_calls`` (treated as one assistant action);
- optional ``assistant_response``;
- optional ``semantic_refusal_match`` for authoritative BFCL-native refusal
  scoring by an external judge.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

from refuse_parallel_eval import (
    canonical_tool_call,
    conservative_native_refusal_match,
    estimate_pass_at_k,
    match_internal_refusal,
    match_parallel_calls,
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def normalise_actions(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    actions = rollout.get("predicted_actions")
    if isinstance(actions, list):
        return actions
    calls = rollout.get("predicted_tool_calls", [])
    if isinstance(calls, list):
        return [{"tool_calls": calls}] if calls else []
    return []


def score_rollout(
    task: dict[str, Any],
    rollout: dict[str, Any],
    *,
    allow_lexical_refusal_fallback: bool = False,
) -> tuple[bool, str]:
    actions = normalise_actions(rollout)
    mode = task["mode"]
    target_format = task.get("target_format", "internal")
    target = task.get("target", {})

    if mode == "parallel":
        success = match_parallel_calls(actions, target.get("tool_calls", []))
        return success, "" if success else "parallel_mismatch"

    if mode in {"refusal", "clarification"}:
        if target_format == "internal":
            target_calls = target.get("tool_calls", [])
            reason = (
                target_calls[0].get("arguments", {}).get("reason")
                if len(target_calls) == 1
                else None
            )
            success = match_internal_refusal(actions, str(reason))
            return success, "" if success else "refusal_tool_mismatch"

        reason = target.get("reason")
        # BFCL-native refusal: no real tool call plus a semantically appropriate
        # response.  Prefer an authoritative external-judge label when supplied.
        if any(action.get("tool_calls") for action in actions):
            return False, "unexpected_tool_call"
        semantic = rollout.get("semantic_refusal_match")
        if isinstance(semantic, bool):
            return semantic, "" if semantic else "semantic_refusal_mismatch"
        if not allow_lexical_refusal_fallback:
            return False, "missing_semantic_refusal_label"
        response = str(rollout.get("assistant_response", ""))
        success = conservative_native_refusal_match(response, str(reason))
        return success, "" if success else "native_refusal_prefilter_failed"

    gold_calls = target.get("tool_calls", [])
    if len(actions) != 1:
        return False, "sequential_action_count_mismatch"
    predicted = actions[0].get("tool_calls", [])
    success = [canonical_tool_call(call) for call in predicted] == [
        canonical_tool_call(call) for call in gold_calls
    ]
    return success, "" if success else "tool_call_mismatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, action="append", default=[])
    parser.add_argument(
        "--allow-lexical-refusal-fallback",
        action="store_true",
        help=(
            "Permit the conservative lexical refusal matcher when a rollout "
            "does not contain semantic_refusal_match. Disabled by default so "
            "BFCL-native refusal scoring fails closed."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    rollouts_by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    detailed: list[dict[str, Any]] = []

    for rollout in read_jsonl(args.rollouts):
        task_id = str(rollout["task_id"])
        if task_id not in tasks:
            raise ValueError(f"Unknown task_id in rollout: {task_id}")
        success, failure = score_rollout(
            tasks[task_id],
            rollout,
            allow_lexical_refusal_fallback=args.allow_lexical_refusal_fallback,
        )
        scored = dict(rollout)
        scored["success"] = success
        scored["failure"] = failure or None
        detailed.append(scored)
        rollouts_by_task[task_id].append(scored)

    per_task = []
    max_n = max((len(values) for values in rollouts_by_task.values()), default=0)
    ks = sorted(set(args.k or ([1, max_n] if max_n else [])))
    overall: dict[str, float] = {}

    for task_id, task in tasks.items():
        samples = sorted(
            rollouts_by_task.get(task_id, []),
            key=lambda item: int(item.get("sample_index", 0)),
        )
        n = len(samples)
        c = sum(bool(sample["success"]) for sample in samples)
        metrics = {
            f"pass_at_{k}": estimate_pass_at_k(n, c, k)
            for k in ks
            if 1 <= k <= n
        }
        per_task.append(
            {
                "task_id": task_id,
                "mode": task["mode"],
                "n": n,
                "c": c,
                **metrics,
            }
        )

    for k in ks:
        values = [
            row[f"pass_at_{k}"]
            for row in per_task
            if f"pass_at_{k}" in row
        ]
        if values:
            overall[f"pass_at_{k}"] = sum(values) / len(values)

    report = {
        "tasks": len(tasks),
        "rollouts": len(detailed),
        "k_values": ks,
        "overall": overall,
        "per_task": per_task,
        "rollouts_scored": detailed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rollouts_scored"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
