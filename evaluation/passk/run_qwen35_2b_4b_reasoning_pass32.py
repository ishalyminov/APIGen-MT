#!/usr/bin/env python3
"""Run Qwen3.5 2B/4B reasoning pass@32 at temperatures 0.7 and 1.0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


OWNER = Path("/mnt/shared_ru.ml.SZ-5_000264/gambashidze")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CHECKER = ROOT / "check_apigen_trajectories_passk_v3.py"
DATA = OWNER / (
    "qwen35_toolonly_sft_sweep_artifacts/data/"
    "apigen_toolonly_sft_next_action_targeted200_v1.jsonl"
)
ENV_ROOT = Path("/home/jovyan/.mlspace/envs/verl_q35_cu128_clean")
PYTHON = ENV_ROOT / "bin/python"
DEFAULT_OUTPUT = ROOT / (
    "results/pass32_qwen35_2b_4b_reasoning_apigen1391_promptv4_20260817"
)
CHAT_TEMPLATE = REPO_ROOT / "templates/qwen35_toolonly_base.jinja"
TOOL_POOL = (
    REPO_ROOT
    / "magnet_tool_extraction"
    / "bfcl_v3_tools_with_outputs.jsonl"
)

CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "label": "qwen35_2b/t0p7",
        "gpu": 0,
        "port": 8210,
        "temperature": 0.7,
        "served_name": "qwen3.5-2b",
        "model_path": Path(
            "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-2B/snapshots/"
            "15852e8c16360a2fea060d615a32b45270f8a8fc"
        ),
    },
    {
        "label": "qwen35_2b/t1p0",
        "gpu": 1,
        "port": 8211,
        "temperature": 1.0,
        "served_name": "qwen3.5-2b",
        "model_path": Path(
            "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-2B/snapshots/"
            "15852e8c16360a2fea060d615a32b45270f8a8fc"
        ),
    },
    {
        "label": "qwen35_4b/t0p7",
        "gpu": 2,
        "port": 8212,
        "temperature": 0.7,
        "served_name": "qwen3.5-4b",
        "model_path": Path(
            "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/"
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ),
    },
    {
        "label": "qwen35_4b/t1p0",
        "gpu": 3,
        "port": 8213,
        "temperature": 1.0,
        "served_name": "qwen3.5-4b",
        "model_path": Path(
            "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/"
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ),
    },
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def runtime_environment(gpu: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    library_paths = [
        str(ENV_ROOT / "lib64"),
        str(ENV_ROOT / "lib"),
        str(ENV_ROOT / "targets/x86_64-linux/lib"),
        environment.get("LD_LIBRARY_PATH", ""),
    ]
    environment.update(
        {
            "CONDA_PREFIX": str(ENV_ROOT),
            "CUDA_HOME": str(ENV_ROOT),
            "CUDA_PATH": str(ENV_ROOT),
            "PATH": f"{ENV_ROOT / 'bin'}:{environment.get('PATH', '')}",
            "LD_LIBRARY_PATH": ":".join(library_paths),
            "PYTHONPATH": (
                f"{OWNER / 'toolcallrl/verl'}:"
                f"{environment.get('PYTHONPATH', '')}"
            ),
            "HF_HOME": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
            "TOOLCALL_VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "TOOLCALL_GDN_PREFILL_BACKEND": "triton",
            "TOOLCALL_MM_ENCODER_ATTN_BACKEND": "FLASHINFER",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(key, None)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def server_command(condition: dict[str, Any]) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(condition["model_path"]),
        "--served-model-name",
        str(condition["served_name"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(condition["port"]),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.65",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
        "--chat-template",
        str(CHAT_TEMPLATE),
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "16384",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--generation-config",
        "vllm",
        "--disable-uvicorn-access-log",
        "--enforce-eager",
        "--gdn-prefill-backend",
        "triton",
    ]


def server_ready(condition: dict[str, Any]) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{condition['port']}/v1/models", timeout=3
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return condition["served_name"] in {
            str(item.get("id")) for item in payload.get("data", [])
        }
    except (OSError, ValueError, urllib.error.URLError):
        return False


def checker_command(
    condition: dict[str, Any],
    output_dir: Path,
    *,
    max_samples: int | None,
    overwrite: bool,
) -> list[str]:
    command = [
        str(PYTHON),
        str(CHECKER),
        "--jsonl",
        str(DATA),
        "--out-dir",
        str(output_dir),
        "--tool-pool",
        str(TOOL_POOL),
        "--tool-scope",
        "declared",
        "--pass-k",
        "32",
        "--temperature",
        str(condition["temperature"]),
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--min-p",
        "0",
        "--presence-penalty",
        "0",
        "--repetition-penalty",
        "1",
        "--max-tokens",
        "8192",
        "--seed",
        "42",
        "--current-date",
        "2026-08-17",
        "--workers",
        "64",
        "--enable-thinking",
        "--chat-template-path",
        str(CHAT_TEMPLATE),
        "--vllm-url",
        f"http://127.0.0.1:{condition['port']}/v1",
        "--model",
        str(condition["served_name"]),
        "--request-timeout",
        "600",
        "--request-retries",
        "3",
        "--overwrite" if overwrite else "--resume",
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    return command


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_pilot(output_dir: Path) -> dict[str, Any]:
    events = read_events(output_dir / "events.jsonl")
    if not events:
        raise RuntimeError(f"Pilot emitted no events: {output_dir}")
    api_errors = sum(event.get("failure") == "api_error" for event in events)
    truncated = sum(event.get("finish_reason") == "length" for event in events)
    reasoning = [
        len(str(event.get("reasoning_content") or "")) for event in events
    ]
    nonempty_reasoning = sum(length > 0 for length in reasoning)
    audit = {
        "events": len(events),
        "api_errors": api_errors,
        "truncated": truncated,
        "truncated_rate": truncated / len(events),
        "reasoning_nonempty": nonempty_reasoning,
        "reasoning_nonempty_rate": nonempty_reasoning / len(events),
        "reasoning_chars_mean": sum(reasoning) / len(reasoning),
        "reasoning_chars_max": max(reasoning),
    }
    if api_errors or truncated / len(events) > 0.01:
        raise RuntimeError(f"Pilot transport/truncation failure: {audit}")
    if nonempty_reasoning == 0:
        raise RuntimeError(f"Thinking was not enabled: {audit}")
    return audit


def run_checkers(
    conditions: tuple[dict[str, Any], ...],
    output_root: Path,
    *,
    phase: str,
    max_samples: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for condition in conditions:
        output_dir = output_root / phase / str(condition["label"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and (output_dir / "summary.json").exists():
            print(f"SKIP completed {phase} {condition['label']}", flush=True)
            continue
        log_path = output_dir / "runner.log"
        command = checker_command(
            condition,
            output_dir,
            max_samples=max_samples,
            overwrite=overwrite,
        )
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write("COMMAND: " + " ".join(command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=runtime_environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        jobs.append(
            {
                "condition": condition,
                "output_dir": output_dir,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
            }
        )

    last_status = 0.0
    while any(job["process"].poll() is None for job in jobs):
        now = time.monotonic()
        if now - last_status >= 30:
            status: dict[str, Any] = {"phase": phase, "conditions": {}}
            for job in jobs:
                progress_path = job["output_dir"] / "progress.json"
                progress = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.exists()
                    else {}
                )
                status["conditions"][job["condition"]["label"]] = {
                    "pid": job["process"].pid,
                    "returncode": job["process"].poll(),
                    "progress": progress,
                }
            atomic_json(output_root / "status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            last_status = now
        time.sleep(2)

    failures: list[str] = []
    for job in jobs:
        job["log_handle"].close()
        if job["process"].returncode:
            tail = "\n".join(
                job["log_path"].read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-80:]
            )
            failures.append(
                f"{phase} {job['condition']['label']} exited "
                f"{job['process'].returncode}:\n{tail}"
            )
    if failures:
        raise RuntimeError("\n\n".join(failures))

    results: dict[str, Any] = {}
    for condition in conditions:
        output_dir = output_root / phase / str(condition["label"])
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing summary: {summary_path}")
        results[str(condition["label"])] = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-samples", type=int, default=32)
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if (
        not DATA.is_file()
        or not CHECKER.is_file()
        or not PYTHON.is_file()
        or not CHAT_TEMPLATE.is_file()
        or not TOOL_POOL.is_file()
    ):
        raise FileNotFoundError(
            "Dataset, checker, runtime Python, or chat template is missing"
        )
    for condition in CONDITIONS:
        if not condition["model_path"].is_dir():
            raise FileNotFoundError(condition["model_path"])

    servers: list[dict[str, Any]] = []
    try:
        for condition in CONDITIONS:
            log_path = output_root / (
                f"vllm_{str(condition['label']).replace('/', '_')}.log"
            )
            log_handle = log_path.open("a", encoding="utf-8")
            command = server_command(condition)
            log_handle.write("COMMAND: " + " ".join(command) + "\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=runtime_environment(int(condition["gpu"])),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            servers.append(
                {
                    "condition": condition,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                }
            )

        deadline = time.monotonic() + 1200
        pending = set(range(len(servers)))
        while pending and time.monotonic() < deadline:
            for index in list(pending):
                server = servers[index]
                process = server["process"]
                if process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited for {server['condition']['label']}; "
                        f"see {server['log_path']}"
                    )
                if server_ready(server["condition"]):
                    print(f"READY {server['condition']['label']}", flush=True)
                    pending.remove(index)
            if pending:
                time.sleep(3)
        if pending:
            raise TimeoutError(f"vLLM startup timed out: {sorted(pending)}")

        aggregate: dict[str, Any] = {
            "dataset": str(DATA),
            "pass_k": 32,
            "thinking": True,
            "conditions": [condition["label"] for condition in CONDITIONS],
        }
        results_path = output_root / "results.json"
        if args.skip_pilot and results_path.exists():
            previous = json.loads(results_path.read_text(encoding="utf-8"))
            if all(previous.get(key) == value for key, value in aggregate.items()):
                aggregate = previous
        if not args.skip_pilot:
            pilot = run_checkers(
                CONDITIONS,
                output_root,
                phase="pilot",
                max_samples=args.pilot_samples,
                overwrite=True,
            )
            aggregate["pilot_audits"] = {
                condition["label"]: audit_pilot(
                    output_root / "pilot" / str(condition["label"])
                )
                for condition in CONDITIONS
            }
            aggregate["pilot_summaries"] = pilot
            atomic_json(results_path, aggregate)
        if args.pilot_only:
            return 0

        aggregate["full_summaries"] = run_checkers(
            CONDITIONS,
            output_root,
            phase="full",
            max_samples=None,
            overwrite=False,
        )
        atomic_json(results_path, aggregate)
        print(f"ALL DONE: {results_path}", flush=True)
        return 0
    finally:
        for server in servers:
            process = server["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            if not server["log_handle"].closed:
                server["log_handle"].close()


if __name__ == "__main__":
    raise SystemExit(main())
