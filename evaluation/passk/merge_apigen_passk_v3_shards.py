#!/usr/bin/env python3
"""Replay, validate, and merge APIGen interactive pass@k v3 shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import check_apigen_trajectories_passk as legacy
import check_apigen_trajectories_passk_v3 as v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--tool-pool", default=str(legacy.DEFAULT_TOOL_POOL))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--pass-k", type=int, default=16)
    parser.add_argument(
        "--tool-scope",
        choices=["category", "gold", "all", "declared"],
        default="declared",
    )
    parser.add_argument("--current-date", default="2026-07-30")
    return parser


def solution_rate_distribution(
    task_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(
        min(int(float(result["rollout_success_rate"]) * 10), 9)
        for result in task_results
    )
    total = len(task_results)
    return [
        {
            "range_percent": (
                f"{lower}-{lower + 10}" if lower < 90 else "90-100"
            ),
            "tasks": counts[lower // 10],
            "fraction": counts[lower // 10] / max(total, 1),
        }
        for lower in range(0, 100, 10)
    ]


def main() -> int:
    args = build_parser().parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")

    jsonl = Path(args.jsonl).resolve()
    tool_pool = Path(args.tool_pool).resolve()
    run_root = Path(args.run_root).resolve()
    tasks, _ = legacy.load_tasks(
        jsonl,
        tool_pool,
        tool_scope=args.tool_scope,
    )
    checker = v3.InteractivePassKV3Checker(
        client=None,  # type: ignore[arg-type]
        refusal_judge=None,
        pass_k=args.pass_k,
        current_date=args.current_date,
    )
    states = checker.build_states(tasks)

    shard_configs: list[dict[str, Any]] = []
    event_count = 0
    for shard_index in range(args.shard_count):
        shard_dir = run_root / f"shard{shard_index}"
        config_path = shard_dir / "config.json"
        events_path = shard_dir / "events.jsonl"
        summary_path = shard_dir / "summary.json"
        if not all(path.exists() for path in (config_path, events_path, summary_path)):
            raise FileNotFoundError(f"Incomplete shard directory: {shard_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checks = {
            "protocol_version": v3.PROTOCOL_VERSION,
            "jsonl": str(jsonl),
            "tool_pool": str(tool_pool),
            "tool_scope": args.tool_scope,
            "pass_k": args.pass_k,
            "shard_count": args.shard_count,
            "shard_index": shard_index,
            "include_initial_state": False,
            "current_date": args.current_date,
            "synthetic_refuse_tool_visible": False,
            "parallel_tool_calls": "always_enabled_no_gold_hint",
        }
        mismatches = {
            key: {"expected": expected, "actual": config.get(key)}
            for key, expected in checks.items()
            if config.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Shard {shard_index} config mismatch: "
                f"{json.dumps(mismatches, ensure_ascii=False)}"
            )
        shard_configs.append(config)
        event_count += v3.load_events(events_path, states)

    active = [state for state in states if state.status == "active"]
    if active:
        examples = [
            (state.task.position, state.sample_index, state.next_turn)
            for state in active[:10]
        ]
        raise RuntimeError(
            f"{len(active)} rollouts are still active; examples={examples}"
        )
    if len(states) != len(tasks) * args.pass_k:
        raise RuntimeError("Unexpected rollout cardinality")
    terminal_keys = {
        (state.task.position, state.sample_index) for state in states
    }
    if len(terminal_keys) != len(states):
        raise RuntimeError("Duplicate terminal rollout keys")

    rollouts_path = run_root / "rollouts.jsonl"
    with rollouts_path.open("w", encoding="utf-8") as output:
        for state in states:
            output.write(
                json.dumps(v3.rollout_record(state), ensure_ascii=False) + "\n"
            )

    summary = v3.summarize_v3(tasks, states, args.pass_k)
    summary["solution_rate_distribution_10pct"] = solution_rate_distribution(
        summary["task_results"]
    )
    summary["merge"] = {
        "jsonl": str(jsonl),
        "tool_pool": str(tool_pool),
        "tool_scope": args.tool_scope,
        "pass_k": args.pass_k,
        "shard_count": args.shard_count,
        "events_replayed": event_count,
        "terminal_rollouts": len(states),
        "unique_terminal_keys": len(terminal_keys),
        "all_rollouts_terminal": True,
        "shard_configs": shard_configs,
    }
    legacy.atomic_write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
