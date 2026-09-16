#!/usr/bin/env python3
"""Run interactive pass@16 for Qwen3.5 2B and 4B on naturalized long500."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/gambashidze")
TOOL_SYNTH = ROOT / "tool_synth"
APIGEN = TOOL_SYNTH / "APIGen-MT-main"
sys.path.insert(0, str(TOOL_SYNTH))

import run_qwen35_small_4gpu_pass16 as base  # noqa: E402


DATASET = (
    APIGEN
    / "data/generated/long7_15_grok45_500_20260727_naturalized.jsonl"
)
OUTPUT_ROOT = (
    APIGEN
    / "data/generated/pass16_qwen35_2b_4b_naturalized500_20260728"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unordered",
        action="store_true",
        help=(
            "Allow exact unmatched gold steps from the current user turn in "
            "any order. This is an upper-bound diagnostic for trajectories "
            "without an explicit dependency DAG."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    output_root = (args.output_root or OUTPUT_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = output_root
    base.DATASETS = {
        "naturalized500": {
            "path": DATASET,
            "tool_scope": "category",
        }
    }
    if args.unordered:
        ordered_checker_command = base.checker_command

        def unordered_checker_command(**kwargs: object) -> list[str]:
            command = ordered_checker_command(**kwargs)
            command.remove("--ordered")
            return command

        base.checker_command = unordered_checker_command

    for model in base.MODELS.values():
        if not Path(model["path"]).is_dir():
            raise FileNotFoundError(model["path"])

    servers: list[dict[str, object]] = []
    checker_environment = base.runtime_environment()
    try:
        for model_name, model in base.MODELS.items():
            for replica_index, (gpu, port) in enumerate(model["replicas"]):
                log_path = output_root / f"vllm_{model_name}_gpu{gpu}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                command = base.server_command(model, port)
                log_handle.write("COMMAND: " + " ".join(command) + "\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=TOOL_SYNTH,
                    env=base.runtime_environment(gpu),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                servers.append(
                    {
                        "label": (
                            f"{model_name}/replica{replica_index}/gpu{gpu}"
                        ),
                        "model_name": model_name,
                        "replica_index": replica_index,
                        "gpu": gpu,
                        "port": port,
                        "served_name": model["served_name"],
                        "process": process,
                        "log_handle": log_handle,
                        "log_path": log_path,
                    }
                )

        base.wait_for_servers(servers)
        results = base.run_dataset_wave(
            "naturalized500",
            servers,
            checker_environment,
        )
        payload = {
            "pass_k": base.PASS_K,
            "dataset": str(DATASET),
            "evaluation": (
                "interactive_exact_unordered_step_replay"
                if args.unordered
                else "interactive_exact_ordered_step_replay"
            ),
            "models": {
                name: {
                    "path": str(model["path"]),
                    "served_name": model["served_name"],
                    "replicas": model["replicas"],
                }
                for name, model in base.MODELS.items()
            },
            "results": results,
        }
        base.atomic_json(output_root / "results.json", payload)
        print(
            json.dumps(
                {
                    model: {
                        "pass@1": summary["pass_at_1_all"],
                        "pass@16": summary["pass_at_16_all"],
                    }
                    for model, summary in results.items()
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        for server in servers:
            process = server["process"]
            assert isinstance(process, subprocess.Popen)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            log_handle = server["log_handle"]
            if not log_handle.closed:
                log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
