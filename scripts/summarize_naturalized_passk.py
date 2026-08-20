#!/usr/bin/env python3
"""Create pass@k tables and solve-rate distributions for naturalized long500."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "data/generated/long7_15_grok45_500_20260727_naturalized.jsonl"
)
DEFAULT_RESULTS = (
    ROOT / "data/generated/pass16_qwen35_2b_4b_naturalized500_20260728"
)
OLD_RESULTS = (
    ROOT / "data/generated/pass16_qwen35_2b_4b_4gpu_20260728"
)
MODELS = ("qwen35_2b", "qwen35_4b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--old-results-dir", type=Path, default=OLD_RESULTS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def solve_rate_bin(successes: int, samples: int) -> int:
    percent = 100.0 * successes / samples
    return min(int(percent // 10), 9)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 dataset rows, found {len(rows)}")
    step_by_row = {
        index: len(row["trajectory"]["steps"])
        for index, row in enumerate(rows)
    }
    category_by_row = {
        index: (
            row["generation_metadata"].get("focus_category")
            or (
                row["trajectory"].get("categories_used") or ["Unknown"]
            )[0]
        )
        for index, row in enumerate(rows)
    }

    analysis: dict[str, Any] = {
        "dataset": str(args.dataset.resolve()),
        "tasks": len(rows),
        "models": {},
    }
    pass_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for model in MODELS:
        model_dir = args.results_dir / model / "naturalized500"
        summary = json.loads(
            (model_dir / "summary.json").read_text(encoding="utf-8")
        )
        rollouts = read_jsonl(model_dir / "rollouts.jsonl")
        samples = int(summary["num_rollouts"]) // len(rows)
        successes = Counter(
            int(rollout["row_position"])
            for rollout in rollouts
            if rollout.get("success")
        )
        totals = Counter(
            int(rollout["row_position"]) for rollout in rollouts
        )
        if set(totals) != set(range(len(rows))) or any(
            totals[index] != samples for index in range(len(rows))
        ):
            raise RuntimeError(f"{model}: incomplete per-task rollouts")

        per_task = {
            str(index): {
                "successes": successes[index],
                "samples": samples,
                "solve_rate": successes[index] / samples,
                "solve_percent": 100.0 * successes[index] / samples,
                "steps": step_by_row[index],
                "category": category_by_row[index],
            }
            for index in range(len(rows))
        }
        bins = Counter(
            solve_rate_bin(successes[index], samples)
            for index in range(len(rows))
        )
        bin_records = []
        for bin_index in range(10):
            count = bins[bin_index]
            record = {
                "lower_percent_inclusive": bin_index * 10,
                "upper_percent_exclusive": (
                    (bin_index + 1) * 10 if bin_index < 9 else None
                ),
                "label": (
                    f"{bin_index * 10}-{(bin_index + 1) * 10}%"
                    if bin_index < 9
                    else "90-100%"
                ),
                "tasks": count,
                "task_percent": 100.0 * count / len(rows),
            }
            bin_records.append(record)
            distribution_rows.append({"model": model, **record})

        by_steps = {}
        for steps in sorted(set(step_by_row.values())):
            indices = [
                index
                for index, value in step_by_row.items()
                if value == steps
            ]
            success_count = sum(successes[index] for index in indices)
            solved = sum(successes[index] > 0 for index in indices)
            rates = [successes[index] / samples for index in indices]
            record = {
                "tasks": len(indices),
                "successful_rollouts": success_count,
                "total_rollouts": len(indices) * samples,
                "mean_solve_rate": statistics.mean(rates),
                "median_solve_rate": statistics.median(rates),
                "tasks_solved_at_least_once": solved,
                "pass_at_16_empirical": solved / len(indices),
                "decile_task_counts": [
                    sum(
                        solve_rate_bin(successes[index], samples)
                        == bin_index
                        for index in indices
                    )
                    for bin_index in range(10)
                ],
            }
            by_steps[str(steps)] = record
            step_rows.append({"model": model, "steps": steps, **record})

        indices_by_category: dict[str, list[int]] = defaultdict(list)
        for index, category in category_by_row.items():
            indices_by_category[category].append(index)
        by_category = {}
        for category, indices in sorted(indices_by_category.items()):
            solved = sum(successes[index] > 0 for index in indices)
            success_count = sum(successes[index] for index in indices)
            record = {
                "tasks": len(indices),
                "successful_rollouts": success_count,
                "total_rollouts": len(indices) * samples,
                "mean_solve_rate": success_count / (len(indices) * samples),
                "tasks_solved_at_least_once": solved,
                "pass_at_16_empirical": solved / len(indices),
            }
            by_category[category] = record
            category_rows.append(
                {"model": model, "category": category, **record}
            )

        for item in summary["pass_curve"]:
            pass_rows.append(
                {
                    "model": model,
                    "k": item["k"],
                    "pass_at_k_all": item["pass_at_k_all"],
                    "pass_at_k_clean": item["pass_at_k_clean"],
                }
            )

        old_summary_path = (
            args.old_results_dir / model / "long500/summary.json"
        )
        comparison = None
        if old_summary_path.exists():
            old = json.loads(old_summary_path.read_text(encoding="utf-8"))
            comparison = {
                "old_pass_at_1": old["pass_at_1_all"],
                "naturalized_pass_at_1": summary["pass_at_1_all"],
                "delta_pass_at_1": (
                    summary["pass_at_1_all"] - old["pass_at_1_all"]
                ),
                "old_pass_at_16": old["pass_at_16_all"],
                "naturalized_pass_at_16": summary["pass_at_16_all"],
                "delta_pass_at_16": (
                    summary["pass_at_16_all"] - old["pass_at_16_all"]
                ),
            }
            comparison_rows.append({"model": model, **comparison})

        rates = [successes[index] / samples for index in range(len(rows))]
        analysis["models"][model] = {
            "rollouts_per_task": samples,
            "num_rollouts": len(rollouts),
            "successful_rollouts": sum(successes.values()),
            "mean_task_solve_rate": statistics.mean(rates),
            "median_task_solve_rate": statistics.median(rates),
            "tasks_solved_at_least_once": sum(rate > 0 for rate in rates),
            "tasks_never_solved": sum(rate == 0 for rate in rates),
            "pass_curve": summary["pass_curve"],
            "solution_percentage_distribution": bin_records,
            "by_steps": by_steps,
            "by_category": by_category,
            "comparison_to_plan_like_source": comparison,
            "per_task": per_task,
            "failure_counts": summary.get("failure_counts", {}),
            "data_issue_task_counts": summary.get(
                "data_issue_task_counts", {}
            ),
        }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_dir / "analysis.json", analysis)
    write_csv(
        args.results_dir / "pass_at_k_table.csv",
        ["model", "k", "pass_at_k_all", "pass_at_k_clean"],
        pass_rows,
    )
    write_csv(
        args.results_dir / "solution_percentage_distribution.csv",
        [
            "model",
            "lower_percent_inclusive",
            "upper_percent_exclusive",
            "label",
            "tasks",
            "task_percent",
        ],
        distribution_rows,
    )
    write_csv(
        args.results_dir / "solve_rate_by_steps.csv",
        [
            "model",
            "steps",
            "tasks",
            "successful_rollouts",
            "total_rollouts",
            "mean_solve_rate",
            "median_solve_rate",
            "tasks_solved_at_least_once",
            "pass_at_16_empirical",
            "decile_task_counts",
        ],
        step_rows,
    )
    write_csv(
        args.results_dir / "solve_rate_by_category.csv",
        [
            "model",
            "category",
            "tasks",
            "successful_rollouts",
            "total_rollouts",
            "mean_solve_rate",
            "tasks_solved_at_least_once",
            "pass_at_16_empirical",
        ],
        category_rows,
    )
    if comparison_rows:
        write_csv(
            args.results_dir / "comparison_to_plan_like_source.csv",
            [
                "model",
                "old_pass_at_1",
                "naturalized_pass_at_1",
                "delta_pass_at_1",
                "old_pass_at_16",
                "naturalized_pass_at_16",
                "delta_pass_at_16",
            ],
            comparison_rows,
        )

    markdown = [
        "# Naturalized long500 interactive pass@16",
        "",
        "| Model | pass@1 | pass@4 | pass@8 | pass@16 | Never solved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = analysis["models"][model]
        curve = {
            int(item["k"]): item["pass_at_k_all"]
            for item in data["pass_curve"]
        }
        markdown.append(
            f"| {model} | {curve[1]:.4f} | {curve[4]:.4f} | "
            f"{curve[8]:.4f} | {curve[16]:.4f} | "
            f"{data['tasks_never_solved']} |"
        )
    markdown.extend(
        [
            "",
            "Detailed tables:",
            "",
            "- `pass_at_k_table.csv`",
            "- `solution_percentage_distribution.csv`",
            "- `solve_rate_by_steps.csv`",
            "- `solve_rate_by_category.csv`",
            "- `comparison_to_plan_like_source.csv`",
            "- `analysis.json`",
            "",
        ]
    )
    (args.results_dir / "README.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(json.dumps({
        model: {
            "pass@1": analysis["models"][model]["pass_curve"][0][
                "pass_at_k_all"
            ],
            "pass@16": analysis["models"][model]["pass_curve"][15][
                "pass_at_k_all"
            ],
            "never_solved": analysis["models"][model][
                "tasks_never_solved"
            ],
        }
        for model in MODELS
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
