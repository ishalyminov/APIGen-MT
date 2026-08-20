#!/usr/bin/env python3
"""Summarize per-task rollout solve rates in equal-width percentage bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", default="long500")
    parser.add_argument("--rollouts-per-task", type=int, default=16)
    parser.add_argument("--bin-width", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if 100 % args.bin_width:
        raise ValueError("--bin-width must divide 100 evenly")
    results = json.loads(args.results.read_text(encoding="utf-8"))
    models = results["datasets"][args.dataset]
    bin_count = 100 // args.bin_width
    report = {
        "source": str(args.results),
        "dataset": args.dataset,
        "rollouts_per_task": args.rollouts_per_task,
        "bin_width_percentage_points": args.bin_width,
        "models": {},
    }

    for model, summary in models.items():
        task_count = int(summary["num_tasks"])
        sparse = {
            str(task_id): int(successes)
            for task_id, successes in summary.get(
                "successes_per_task", {}
            ).items()
        }
        successes = list(sparse.values()) + [0] * (task_count - len(sparse))
        bins = [0] * bin_count
        for count in successes:
            rate = 100.0 * count / args.rollouts_per_task
            index = min(int(rate // args.bin_width), bin_count - 1)
            bins[index] += 1

        rows = []
        for index, count in enumerate(bins):
            lower = index * args.bin_width
            upper = lower + args.bin_width
            rows.append(
                {
                    "lower_percent_inclusive": lower,
                    "upper_percent_exclusive": (
                        None if upper == 100 else upper
                    ),
                    "label": (
                        f"{lower}-{upper}%"
                        if upper < 100
                        else f"{lower}-100%"
                    ),
                    "tasks": count,
                    "task_percent": 100.0 * count / task_count,
                }
            )
        report["models"][model] = {
            "tasks": task_count,
            "bins": rows,
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
