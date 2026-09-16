#!/usr/bin/env python3
"""Generate the balanced hard/natural 500-row multi-turn dataset.

The dataset is intentionally split into three canonical parts:

* 200 refusal-family rows
  * 80 missing-argument clarification + recovery
  * 80 ambiguity clarification + recovery
  * 40 terminal unsupported-capability refusals
* 200 certified-parallel rows
* 100 combined rows with clarification + recovery + final parallel

Every row has exactly 10-20 assistant action transitions. Lengths are allocated
as evenly as integer counts permit. Ordinary turns contain one action, while
parallel turns contain a 3-5-call unordered batch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LENGTHS = tuple(range(10, 21))


@dataclass(frozen=True)
class Profile:
    name: str
    rows: int
    feature: str
    schedule: str
    refusal_reason: str
    length_offset: int


@dataclass(frozen=True)
class Task:
    profile: Profile
    steps: int
    width: int
    shard: int
    rows: int

    @property
    def stem(self) -> str:
        return (
            f"{self.profile.name}.s{self.steps}.w{self.width}."
            f"part{self.shard:03d}"
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
        default=ROOT
        / "data/generated/runs/hard_natural_balanced_500_20260728",
    )
    parser.add_argument("--python", default=os.getenv("APIGEN_PYTHON", sys.executable))
    parser.add_argument("--model", default=os.getenv("APIGEN_MODEL", "x-ai/grok-4.5"))
    parser.add_argument(
        "--judge-model",
        default=os.getenv("APIGEN_JUDGE_MODEL", "x-ai/grok-4.5"),
    )
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--rows-per-shard", type=int, default=3)
    return parser.parse_args()


def balanced_counts(total: int, offset: int = 0) -> dict[int, int]:
    base, remainder = divmod(total, len(LENGTHS))
    return {
        steps: base
        + (
            (index - offset) % len(LENGTHS) < remainder
        )
        for index, steps in enumerate(LENGTHS)
    }


def tasks(rows_per_shard: int) -> list[Task]:
    result: list[Task] = []
    width_cursor = 0
    for profile in PROFILES:
        for steps, count in balanced_counts(
            profile.rows,
            profile.length_offset,
        ).items():
            shard = 0
            while count:
                rows = min(rows_per_shard, count)
                width = (3, 4, 5)[width_cursor % 3]
                if profile.feature in {"parallel", "mixed"}:
                    width_cursor += 1
                else:
                    width = 2
                result.append(
                    Task(
                        profile=profile,
                        steps=steps,
                        width=width,
                        shard=shard,
                        rows=rows,
                    )
                )
                shard += 1
                count -= rows
    return result


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def feature_args(task: Task) -> list[str]:
    common = [
        "--require-feature",
        "--feature-difficulty",
        "hard",
        "--naturalize-queries",
        "--multi-turn-feature-schedule",
        task.profile.schedule,
        "--refusal-reason",
        task.profile.refusal_reason,
    ]
    if task.profile.feature == "refusal":
        return [
            *common,
            "--allow-refusal",
            "--refusal-rate",
            "1.0",
            "--no-allow-parallel",
        ]
    if task.profile.feature == "parallel":
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


def run_task(
    task: Task,
    *,
    args: argparse.Namespace,
    shard_dir: Path,
    log_dir: Path,
    registry: Path,
) -> tuple[str, int]:
    output = shard_dir / f"{task.stem}.jsonl"
    log = log_dir / f"{task.stem}.log"
    existing = count_rows(output)
    if existing > task.rows:
        raise RuntimeError(
            f"{output} has {existing} rows; expected at most {task.rows}"
        )
    remaining = task.rows - existing
    if remaining == 0:
        return task.stem, task.rows

    command = [
        args.python,
        "src/generate_step_by_step.py",
        "--mode",
        "multi-turn",
        "--num-turns",
        str(task.steps),
        "--num-datapoints",
        str(remaining),
        "--num-actions",
        str(task.width),
        # One ordinary action per turn makes the requested length exact. The
        # parallel transition remains a single action containing 3-5 calls.
        "--blueprint-max-actions-per-turn",
        "1",
        "--max-parallel-width",
        str(task.width),
        "--min-total-steps",
        str(task.steps),
        "--max-total-steps",
        str(task.steps),
        "--model",
        args.model,
        "--judge-model",
        args.judge_model,
        "--tool-pool",
        "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl",
        "--invocation-examples",
        "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl",
        "--dedupe-registry",
        str(registry),
        *feature_args(task),
        "--output",
        str(output),
    ]
    with log.open("a", encoding="utf-8") as destination:
        destination.write(
            json.dumps(
                {
                    "event": "launch",
                    "existing": existing,
                    "remaining": remaining,
                    "command": command,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        destination.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=destination,
            stderr=subprocess.STDOUT,
            check=False,
        )
    final_rows = count_rows(output)
    if completed.returncode != 0 or final_rows != task.rows:
        raise RuntimeError(
            f"{task.stem} failed: exit={completed.returncode}, "
            f"rows={final_rows}/{task.rows}; see {log}"
        )
    return task.stem, final_rows


def atomic_merge(paths: Iterable[Path], output: Path) -> int:
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for path in paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        json.loads(line)
                        destination.write(line)
                        rows += 1
    temporary.replace(output)
    return rows


def audit_part(
    *,
    args: argparse.Namespace,
    input_path: Path,
    report_path: Path,
    expected_rows: int,
    schedule: str | None,
    require_recovery: bool,
) -> None:
    command = [
        args.python,
        "scripts/audit_refuse_parallel_dataset.py",
        "--input",
        str(input_path),
        "--report",
        str(report_path),
        "--expected-rows",
        str(expected_rows),
        "--require-feature",
        "--expected-difficulty",
        "hard",
        "--require-naturalized",
        "--min-steps",
        "10",
        "--max-steps",
        "20",
    ]
    if schedule is not None:
        command.extend(["--expected-schedule", schedule])
    if require_recovery:
        command.append("--require-recovery")
    subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_BASE"):
        raise RuntimeError("OPENAI_API_KEY and OPENAI_API_BASE must be set")
    if args.max_workers < 1 or args.rows_per_shard < 1:
        raise ValueError("worker and shard sizes must be positive")
    # Avoid allowing a stalled TLS/proxy connection to pin one generation
    # worker for the legacy 15-minute client default.
    os.environ.setdefault("APIGEN_LLM_TIMEOUT", "180")

    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    log_dir = output_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    registry = output_dir / "semantic_signatures.registry"
    work = tasks(args.rows_per_shard)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                run_task,
                task,
                args=args,
                shard_dir=shard_dir,
                log_dir=log_dir,
                registry=registry,
            ): task
            for task in work
        }
        completed_rows = 0
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                stem, rows = future.result()
                completed_rows += rows
                print(
                    f"[{completed_rows:3d}/500] {stem}: {rows} rows",
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"{task.stem}: {exc}")
                print(f"FAILED {task.stem}: {exc}", flush=True)
    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} shard(s) failed; rerun to resume")

    groups = {
        "refusal_hard_natural_200.jsonl": {
            "profiles": {
                "refusal_missing",
                "refusal_ambiguity",
                "refusal_unsupported",
            },
            "rows": 200,
            "schedule": None,
            "recovery": False,
        },
        "parallel_hard_natural_200.jsonl": {
            "profiles": {"parallel"},
            "rows": 200,
            "schedule": "terminal",
            "recovery": False,
        },
        "combined_hard_natural_100.jsonl": {
            "profiles": {"combined_missing", "combined_ambiguity"},
            "rows": 100,
            "schedule": "combined",
            "recovery": True,
        },
    }
    canonical: list[Path] = []
    for filename, group in groups.items():
        selected = [
            shard_dir / f"{task.stem}.jsonl"
            for task in work
            if task.profile.name in group["profiles"]
        ]
        output = output_dir / filename
        rows = atomic_merge(selected, output)
        if rows != group["rows"]:
            raise RuntimeError(
                f"{output} has {rows} merged rows; expected {group['rows']}"
            )
        audit_part(
            args=args,
            input_path=output,
            report_path=output.with_suffix(".audit.json"),
            expected_rows=group["rows"],
            schedule=group["schedule"],
            require_recovery=group["recovery"],
        )
        canonical.append(output)

    # The refusal-only file intentionally mixes terminal and interactive
    # schedules, so audit those requirements globally here.
    refusal_rows = [
        json.loads(line)
        for line in canonical[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schedule_counts = Counter(
        row["generation_metadata"].get("feature_schedule")
        for row in refusal_rows
    )
    if schedule_counts != Counter(
        {"interactive-refusal": 160, "terminal": 40}
    ):
        raise RuntimeError(f"Unexpected refusal schedule counts: {schedule_counts}")
    if sum(
        row["generation_metadata"].get("clarification_recovered") is True
        for row in refusal_rows
    ) != 160:
        raise RuntimeError("Not all 160 interactive refusal rows recovered")

    # Export one compact task stream across all three canonical source parts.
    task_outputs: dict[str, list[Path]] = {"internal": [], "bfcl-native": []}
    for source in canonical:
        for target_format in task_outputs:
            task_output = output_dir / (
                source.stem + f".{target_format}.tasks.jsonl"
            )
            subprocess.run(
                [
                    args.python,
                    "scripts/export_refuse_parallel_tasks.py",
                    "--input",
                    str(source),
                    "--output",
                    str(task_output),
                    "--target-format",
                    target_format,
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": "src"},
                check=True,
            )
            task_outputs[target_format].append(task_output)
    merged_tasks = {}
    for target_format, paths in task_outputs.items():
        destination = output_dir / f"balanced_500.{target_format}.tasks.jsonl"
        merged_tasks[target_format] = {
            "path": str(destination),
            "rows": atomic_merge(paths, destination),
        }

    length_counts = Counter()
    profile_counts = Counter()
    for task in work:
        length_counts[task.steps] += task.rows
        profile_counts[task.profile.name] += task.rows
    manifest = {
        "dataset": "hard-natural-balanced-500",
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "total_rows": 500,
        "canonical_parts": [
            {"path": str(path), "rows": count_rows(path)}
            for path in canonical
        ],
        "profile_counts": dict(profile_counts),
        "step_distribution": dict(sorted(length_counts.items())),
        "parallel_widths": [3, 4, 5],
        "ordinary_actions_per_turn": 1,
        "task_exports": merged_tasks,
        "semantic_dedupe_registry": str(registry),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
