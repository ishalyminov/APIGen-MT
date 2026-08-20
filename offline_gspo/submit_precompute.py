#!/usr/bin/env python3
"""Submit the single resumable frozen-policy logprob cache job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep.submit_sft_sweep import normalize_job_desc, stage_repo  # noqa: E402


OWNER = Path("/mnt/shared_ru.ml.SZ-5_000264/gambashidze")
ARTIFACTS = OWNER / "qwen35_toolonly_sft_sweep_artifacts"
REPLAY_ROOT = ARTIFACTS / "offline_gspo/q35_2b_pass32_reasoning_v4_gspo_v1"
BASE_MODEL = OWNER / (
    "models/models--Qwen--Qwen3.5-2B/snapshots/"
    "15852e8c16360a2fea060d615a32b45270f8a8fc"
)
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
INSTANCE_TYPE = "a100plus.1gpu.80vG.12C.244G"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--priority", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    prepared = REPLAY_ROOT / "prepared"
    prepared_manifest = prepared / "manifest.json"
    old_dataset = REPLAY_ROOT / "old_logps"
    done = REPLAY_ROOT / "old_logps.done"
    for required in (BASE_MODEL / "config.json", prepared / "state.json", prepared_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)
    if done.is_file():
        print(f"already complete: {done}")
        return 0

    dirty = [line for line in git("status", "--porcelain").splitlines() if line and not line.startswith("??")]
    if dirty and not args.dry:
        raise RuntimeError("tracked working tree is dirty; commit first:\n" + "\n".join(dirty))
    commit = git("rev-parse", "HEAD")
    staged = ROOT if args.dry else stage_repo(commit)
    description = "q35-2b-offline-gspo-precompute-pass32-v1"
    job_desc = f"{description} #gambashidze @alexander"
    payload = {
        "script": f"cd {shlex.quote(str(staged))} && bash offline_gspo/run_precompute.sh",
        "job_desc": job_desc,
        "env_variables": {
            "WORKDIR": str(staged),
            "BASE_MODEL": str(BASE_MODEL),
            "PREPARED_DATASET": str(prepared),
            "PREPARED_MANIFEST": str(prepared_manifest),
            "OLD_LOGPS_DATASET": str(old_dataset),
            "LOG_ROOT": str(REPLAY_ROOT / "logs"),
            "BATCH_SIZE": "4",
            "HF_HOME": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface",
            "HF_HUB_CACHE": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "instance_type": INSTANCE_TYPE,
        "region": "SR008",
        "type": "binary_exp",
        "shm_size_class": "large",
        "base_image": BASE_IMAGE,
        "n_workers": 1,
        "processes_per_worker": 1,
        "priority_class": args.priority,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry:
        return 0

    from mls.manager.job.utils import (
        get_in_progress_jobs,
        run_job_with_retry,
        training_job_api_from_profile,
    )

    in_progress = {
        normalize_job_desc(item.get("job_desc", ""))
        for item in get_in_progress_jobs()
    }
    if normalize_job_desc(job_desc) in in_progress:
        print("already queued")
        return 0
    client, extra = training_job_api_from_profile(args.profile)
    payload["region"] = extra["region"]
    result = run_job_with_retry(client, payload, profile=args.profile)
    job_name = result.get("job_name") if isinstance(result, dict) else None
    job_manifest = ARTIFACTS / "offline_gspo_precompute_job.q35-2b-pass32-v1.json"
    job_manifest.write_text(
        json.dumps(
            {
                "commit": commit,
                "job_name": job_name,
                "prepared_manifest_sha256": file_sha256(prepared_manifest),
                "output": str(old_dataset),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {job_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
