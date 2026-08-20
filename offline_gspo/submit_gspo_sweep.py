#!/usr/bin/env python3
"""Submit the conservative one-epoch offline-GSPO learning-rate sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep.submit_sft_sweep import (  # noqa: E402
    normalize_job_desc,
    parse_learning_rates,
    stage_repo,
)


OWNER = Path("/mnt/shared_ru.ml.SZ-5_000264/gambashidze")
ARTIFACTS = OWNER / "qwen35_toolonly_sft_sweep_artifacts"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
INSTANCE_TYPE = "a100plus.1gpu.80vG.12C.244G"
DEFAULT_BASE = OWNER / (
    "models/models--Qwen--Qwen3.5-2B/snapshots/"
    "15852e8c16360a2fea060d615a32b45270f8a8fc"
)
DEFAULT_RUN_ID = "q35-2b-offline-gspo-pass32-v1"
DEFAULT_REPLAY_ROOT = ARTIFACTS / "offline_gspo/q35_2b_pass32_reasoning_v4_gspo_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--learning-rates", default="5e-8,1e-7,2.5e-7")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--priority", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--dry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = parse_learning_rates(args.learning_rates)
    base_model = args.base_model.resolve()
    replay_root = args.replay_root.resolve()
    prepared_dataset = replay_root / "prepared"
    prepared_manifest = replay_root / "prepared/manifest.json"
    for required in (
        base_model / "config.json",
        prepared_dataset / "state.json",
        prepared_manifest,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    dirty = [line for line in git("status", "--porcelain").splitlines() if line and not line.startswith("??")]
    if dirty and not args.dry:
        raise RuntimeError("tracked working tree is dirty; commit first:\n" + "\n".join(dirty))
    commit = git("rev-parse", "HEAD")
    staged = ROOT if args.dry else stage_repo(commit)
    output_root = ARTIFACTS / "checkpoints" / args.run_id
    log_root = ARTIFACTS / "logs" / args.run_id

    if args.dry:
        client = None
        region = f"<from-profile:{args.profile}>"
        in_progress: dict[str, dict[str, object]] = {}
    else:
        from mls.manager.job.utils import (
            get_in_progress_jobs,
            run_job_with_retry,
            training_job_api_from_profile,
        )

        client, extra = training_job_api_from_profile(args.profile)
        region = extra["region"]
        in_progress = {
            normalize_job_desc(job.get("job_desc", "")): job
            for job in get_in_progress_jobs()
        }

    jobs: list[dict[str, str]] = []
    for tag, learning_rate in variants:
        marker = output_root / f"{tag}_vlserving/.vlserving_done"
        if marker.is_file():
            print(f"skip complete: {tag}")
            jobs.append(
                {
                    "tag": tag,
                    "learning_rate": learning_rate,
                    "job_name": "",
                    "state": "complete",
                }
            )
            continue
        description = f"q35-2b-offline-gspo-{args.run_id}-{tag}-ep1"
        job_desc = f"{description} #gambashidze @alexander"
        normalized_desc = normalize_job_desc(job_desc)
        if normalized_desc in in_progress:
            print(f"skip in queue: {tag}")
            queued = in_progress[normalized_desc]
            jobs.append(
                {
                    "tag": tag,
                    "learning_rate": learning_rate,
                    "job_name": str(queued.get("job_name") or ""),
                    "state": "in_progress",
                }
            )
            continue
        payload = {
            "script": f"cd {shlex.quote(str(staged))} && bash offline_gspo/run_one_gspo.sh",
            "job_desc": job_desc,
            "env_variables": {
                "WORKDIR": str(staged),
                "BASE_MODEL": str(base_model),
                "PREPARED_DATASET": str(prepared_dataset),
                "PREPARED_MANIFEST": str(prepared_manifest),
                "OUTPUT_ROOT": str(output_root),
                "LOG_ROOT": str(log_root),
                "LR_TAG": tag,
                "LEARNING_RATE": learning_rate,
                "EPSILON_LOW": "0.0003",
                "EPSILON_HIGH": "0.0004",
                "EPISODES_PER_BATCH": "1",
                "HF_HOME": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface",
                "HF_HUB_CACHE": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "instance_type": INSTANCE_TYPE,
            "region": region,
            "type": "binary_exp",
            "shm_size_class": "large",
            "base_image": BASE_IMAGE,
            "n_workers": 1,
            "processes_per_worker": 1,
            "priority_class": args.priority,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.dry:
            continue
        assert client is not None
        result = run_job_with_retry(client, payload, profile=args.profile)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        jobs.append(
            {
                "tag": tag,
                "learning_rate": learning_rate,
                "job_name": str(job_name or ""),
                "state": "submitted",
            }
        )
        print("result", result)

    if not args.dry:
        manifest_path = ARTIFACTS / f"offline_gspo_sweep_jobs.{args.run_id}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "commit": commit,
                    "run_id": args.run_id,
                    "base_model": str(base_model),
                    "replay_root": str(replay_root),
                    "prepared_manifest_sha256": file_sha256(prepared_manifest),
                    "reference_policy": "online_frozen_base_same_process",
                    "jobs": jobs,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
