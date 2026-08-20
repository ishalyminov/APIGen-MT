#!/usr/bin/env python3
"""Validate the next-action corpus and submit a one-epoch Qwen3.5 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(
    "/mnt/shared_ru.ml.SZ-5_000264/gambashidze/"
    "qwen35_toolonly_sft_sweep_artifacts"
)
DEFAULT_DATA = ARTIFACTS / (
    "data/apigen_toolonly_sft_next_action_v2.jsonl"
)
PROMPT = ROOT / "prompts/tool_only_system.txt"
TEMPLATE = ROOT / "templates/qwen35_toolonly_base.jinja"
STAGING_ROOT = ARTIFACTS / "code"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
INSTANCE_TYPE = "a100plus.1gpu.80vG.12C.244G"
DEFAULT_LEARNING_RATES = "1e-7,2.5e-7,5e-7"
EXPECTED_SUPERVISION = "next_action_group_and_terminal_stop"
EXPECTED_PREFIX_UNIT = "one_per_next_action_or_parallel_group_with_golden_history"
EXPECTED_SCHEMA_PROJECTION = "openai_name_description_parameters_v3"
REPEAT_BEHAVIORS = ("no_tool", "single_call", "parallel", "other")


def parse_learning_rates(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse a comma-separated LR sweep into stable ``(tag, value)`` pairs."""
    variants: list[tuple[str, str]] = []
    seen: set[Decimal] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            decimal = Decimal(item)
            value = float(decimal)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid learning rate: {item!r}") from exc
        if not decimal.is_finite() or decimal <= 0 or not math.isfinite(value):
            raise ValueError(f"learning rate must be finite and positive: {item!r}")
        decimal = decimal.normalize()
        if decimal in seen:
            raise ValueError(f"duplicate learning rate: {item!r}")
        seen.add(decimal)
        # Decimal's scientific format is stable across equivalent spellings.
        canonical = format(decimal, "E").lower().replace("e+", "e")
        canonical = re.sub(r"e(-?)0+(\d+)$", r"e\1\2", canonical)
        variants.append(("lr" + canonical.replace(".", "p"), canonical))
    if not variants:
        raise ValueError("--learning-rates must contain at least one value")
    return tuple(variants)


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


def normalize_job_desc(value: str) -> str:
    return " ".join(
        token for token in value.split() if not token.startswith(("#", "@"))
    )


def count_jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            rows += 1
    return rows


def resolve_manifest_source(raw_path: str, manifest_path: Path) -> Path:
    source = Path(raw_path)
    if not source.is_absolute():
        source = manifest_path.parent / source
    return source.resolve()


def verify_source_hashes(manifest: dict, manifest_path: Path) -> list[dict[str, str]]:
    """Verify every immutable source reference represented by the manifest.

    The v2 builder used ``input``/``input_sha256``.  The multi-source v3
    builder may use ``inputs``/``source_files`` objects with ``path`` and
    ``sha256`` or a ``source_hashes`` path-to-hash map, so accept those
    explicit forms too.
    """

    references: list[tuple[str, str]] = []
    legacy_path = manifest.get("input")
    legacy_sha = manifest.get("input_sha256")
    if legacy_path is not None or legacy_sha is not None:
        if not isinstance(legacy_path, str) or not isinstance(legacy_sha, str):
            raise ValueError("manifest input/input_sha256 contract is incomplete")
        references.append((legacy_path, legacy_sha))

    for collection_name in ("inputs", "source_files"):
        inputs = manifest.get(collection_name)
        if inputs is None:
            continue
        if isinstance(inputs, dict):
            for raw_path, expected in inputs.items():
                if not isinstance(raw_path, str) or not isinstance(expected, str):
                    raise ValueError(
                        f"manifest {collection_name} must map paths to hashes"
                    )
                references.append((raw_path, expected))
            continue
        if not isinstance(inputs, list):
            raise ValueError(f"manifest {collection_name} must be a list or object")
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise ValueError(
                    f"manifest {collection_name}[{index}] must be an object"
                )
            raw_path = item.get("path", item.get("input", item.get("source")))
            expected = item.get("sha256", item.get("input_sha256"))
            if not isinstance(raw_path, str) or not isinstance(expected, str):
                raise ValueError(
                    f"manifest {collection_name}[{index}] needs path and sha256 strings"
                )
            references.append((raw_path, expected))

    source_hashes = manifest.get("source_hashes")
    if source_hashes is not None:
        if not isinstance(source_hashes, dict):
            raise ValueError("manifest source_hashes must be an object")
        for raw_path, expected in source_hashes.items():
            if not isinstance(raw_path, str) or not isinstance(expected, str):
                raise ValueError("manifest source_hashes must map paths to hashes")
            references.append((raw_path, expected))

    verified: list[dict[str, str]] = []
    seen: dict[Path, str] = {}
    for raw_path, expected in references:
        source = resolve_manifest_source(raw_path, manifest_path)
        if source in seen:
            if seen[source] != expected:
                raise ValueError(
                    f"manifest gives conflicting hashes for {source}: "
                    f"{seen[source]} vs {expected}"
                )
            continue
        seen[source] = expected
        if not source.is_file():
            raise FileNotFoundError(f"manifest source is missing: {source}")
        actual = file_sha256(source)
        if actual != expected:
            raise ValueError(
                f"immutable source changed: {source}; expected={expected} actual={actual}"
            )
        verified.append({"path": str(source), "sha256": actual})
    return verified


def load_and_validate_contract(data: Path) -> dict:
    data = data.resolve()
    manifest_path = data.with_suffix(".manifest.json")
    for required in (data, manifest_path, PROMPT, TEMPLATE):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    expected_rows = manifest.get("expected_rows")
    if not isinstance(rows, int) or rows <= 0 or rows != expected_rows:
        raise ValueError(
            f"manifest row contract failed: rows={rows} expected={expected_rows}"
        )
    actual_rows = count_jsonl_rows(data)
    if actual_rows != rows:
        raise ValueError(
            f"dataset row count differs from manifest: manifest={rows} actual={actual_rows}"
        )

    output = manifest.get("output")
    if output is not None:
        if not isinstance(output, str):
            raise ValueError("manifest output must be a path string")
        if resolve_manifest_source(output, manifest_path) != data:
            raise ValueError(f"manifest output does not identify --data: {output}")

    actual_data_sha = file_sha256(data)
    if manifest.get("output_sha256") != actual_data_sha:
        raise ValueError(
            "dataset hash differs from its manifest: "
            f"manifest={manifest.get('output_sha256')} actual={actual_data_sha}"
        )
    if (
        manifest.get("unique_semantic_trajectories") is not None
        and manifest.get("unique_semantic_trajectories") != rows
    ):
        raise ValueError("corpus is not semantically unique")

    verified_sources = verify_source_hashes(manifest, manifest_path)
    contract = manifest.get("training_contract")
    if not isinstance(contract, dict):
        raise ValueError("manifest has no training_contract object")
    if contract.get("enable_thinking") is not False:
        raise ValueError("corpus contract does not disable thinking")
    if contract.get("supervision") != EXPECTED_SUPERVISION:
        raise ValueError("unexpected SFT supervision contract")
    if contract.get("prefix_unit") != EXPECTED_PREFIX_UNIT:
        raise ValueError("unexpected SFT prefix-unit contract")
    if contract.get("golden_history") is not True:
        raise ValueError("next-action corpus must retain golden history")
    if contract.get("parallel_group_is_one_target") is not True:
        raise ValueError("parallel groups must remain one supervised action")
    if contract.get("terminal_stop_is_separate_target") is not True:
        raise ValueError("terminal stop must be a separate supervised action")
    if contract.get("tool_schema_projection") != EXPECTED_SCHEMA_PROJECTION:
        raise ValueError("unexpected tool schema projection contract")
    visibility = manifest.get("argument_visibility")
    if not isinstance(visibility, dict):
        raise ValueError("manifest has no argument_visibility object")
    output_visibility = visibility.get("output")
    if not isinstance(output_visibility, dict):
        raise ValueError("manifest has no output visibility audit")
    if output_visibility.get("hidden_argument_count") != 0:
        raise ValueError(
            "training corpus contains policy-hidden gold arguments: "
            f"{output_visibility.get('hidden_argument_count')}"
        )

    prompt_hash = hashlib.sha256(
        PROMPT.read_text(encoding="utf-8").strip().encode("utf-8")
    ).hexdigest()
    expected_prompt_hash = contract.get(
        "system_prompt_content_sha256", contract.get("system_prompt_sha256")
    )
    if expected_prompt_hash != prompt_hash:
        raise ValueError(
            "system prompt differs from the corpus contract: "
            f"manifest={expected_prompt_hash} actual={prompt_hash}"
        )
    template_hash = file_sha256(TEMPLATE)
    if contract.get("chat_template_sha256") != template_hash:
        raise ValueError(
            "chat template differs from the corpus contract: "
            f"manifest={contract.get('chat_template_sha256')} actual={template_hash}"
        )

    repeat_factors = contract.get("train_repeat_factors")
    if not isinstance(repeat_factors, dict) or set(repeat_factors) != set(
        REPEAT_BEHAVIORS
    ):
        raise ValueError(
            f"train_repeat_factors must contain exactly {REPEAT_BEHAVIORS}: "
            f"{repeat_factors}"
        )
    for behavior in REPEAT_BEHAVIORS:
        value = repeat_factors[behavior]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"repeat factor {behavior!r} must be an integer >= 1: {value!r}"
            )
    # The trainer exposes explicit knobs for the first three behaviors.  Its
    # ordinary/other view is deliberately fixed at one, so reject an
    # unrepresentable manifest rather than silently training a different mix.
    if repeat_factors["other"] != 1:
        raise ValueError("trainer requires train_repeat_factors.other == 1")

    return {
        "data": data,
        "manifest": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "rows": rows,
        "data_sha256": actual_data_sha,
        "prompt_sha256": prompt_hash,
        "template_sha256": template_hash,
        "repeat_factors": repeat_factors,
        "prefix_unit": contract["prefix_unit"],
        "verified_sources": verified_sources,
    }


def stage_repo(commit: str) -> Path:
    target = STAGING_ROOT / commit
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-hardlinks", str(ROOT), str(target)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", commit],
        check=True,
    )
    return target


def verify_completed_checkpoint(
    serving_marker: Path,
    validated: dict,
    learning_rate: str,
    *,
    model_tag: str,
    base_model: Path,
    base_model_config_sha256: str,
) -> None:
    """Do not let a same-data but incompatible completed run be reused."""

    contract_path = serving_marker.parent / "toolonly_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"completed checkpoint has no training contract: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        "model_tag": model_tag,
        "base_model": str(base_model),
        "base_model_config_sha256": base_model_config_sha256,
        "dataset_sha256": validated["data_sha256"],
        "system_prompt_sha256": validated["prompt_sha256"],
        "chat_template_sha256": validated["template_sha256"],
        "enable_thinking": False,
        "supervision": EXPECTED_SUPERVISION,
        "prefix_unit": EXPECTED_PREFIX_UNIT,
        "tool_schema_projection": EXPECTED_SCHEMA_PROJECTION,
        "train_repeat_factors": validated["repeat_factors"],
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if contract.get("epochs") != 1 and contract.get("epochs") != 1.0:
        mismatches["epochs"] = {
            "expected": 1,
            "actual": contract.get("epochs"),
        }
    try:
        actual_lr = float(contract.get("learning_rate"))
    except (TypeError, ValueError):
        actual_lr = None
    if actual_lr != float(learning_rate):
        mismatches["learning_rate"] = {
            "expected": learning_rate,
            "actual": contract.get("learning_rate"),
        }
    if mismatches:
        raise ValueError(
            f"completed checkpoint contract mismatch at {contract_path}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Absolute local Qwen3.5 base-model directory.",
    )
    parser.add_argument(
        "--model-tag",
        required=True,
        help="Filesystem-safe model identity included in every artifact path.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="validate all local contracts and print payloads without MLSpace access",
    )
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--learning-rates",
        default=DEFAULT_LEARNING_RATES,
        help=(
            "Comma-separated positive learning rates. Default is the conservative "
            f"one-epoch sweep: {DEFAULT_LEARNING_RATES}"
        ),
    )
    parser.add_argument(
        "--priority", choices=("low", "medium", "high"), default="high"
    )
    args = parser.parse_args()
    variants = parse_learning_rates(args.learning_rates)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.model_tag):
        raise ValueError(
            "--model-tag must contain only lowercase letters, digits, '.', '_' or '-'"
        )
    base_model = args.base_model.resolve()
    base_config = base_model / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    base_model_config_sha256 = file_sha256(base_config)

    validated = load_and_validate_contract(args.data)
    data: Path = validated["data"]
    manifest_path: Path = validated["manifest"]
    rows: int = validated["rows"]
    actual_data_sha: str = validated["data_sha256"]
    repeat_factors: dict[str, int] = validated["repeat_factors"]
    contract_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "supervision": EXPECTED_SUPERVISION,
                "prefix_unit": validated["prefix_unit"],
                "repeat_factors": repeat_factors,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:8]
    corpus_run_id = (
        f"{args.model_tag}-nextaction{rows}_{actual_data_sha[:8]}_"
        f"{contract_fingerprint}"
    )
    output_root = ARTIFACTS / "checkpoints" / corpus_run_id
    log_root = ARTIFACTS / "logs" / corpus_run_id

    print(
        json.dumps(
            {
                "validation": "ok",
                "data": str(data),
                "manifest": str(manifest_path),
                "rows": rows,
                "data_sha256": actual_data_sha,
                "run_id": corpus_run_id,
                "model_tag": args.model_tag,
                "base_model": str(base_model),
                "base_model_config_sha256": base_model_config_sha256,
                "train_repeat_factors": repeat_factors,
                "prefix_unit": validated["prefix_unit"],
                "verified_sources": validated["verified_sources"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    dirty = [
        line
        for line in git("status", "--porcelain").splitlines()
        if line and not line.startswith("??")
    ]
    if dirty and not args.dry:
        raise RuntimeError("tracked working tree is dirty; commit first:\n" + "\n".join(dirty))
    commit = git("rev-parse", "HEAD")
    staged = ROOT if args.dry else stage_repo(commit)
    if args.dry:
        client = None
        region = f"<from-profile:{args.profile}>"
        in_progress: set[str] = set()
    else:
        # Keep the CLI import out of --dry so corpus/contract validation also
        # works in the lightweight local Python environment.
        from mls.manager.job.utils import (
            get_in_progress_jobs,
            run_job_with_retry,
            training_job_api_from_profile,
        )

        client, extra = training_job_api_from_profile(args.profile)
        region = extra["region"]
        in_progress = {
            normalize_job_desc(job.get("job_desc", ""))
            for job in get_in_progress_jobs()
        }

    launched: list[dict[str, str]] = []
    for tag, learning_rate in variants:
        serving = output_root / f"{tag}_vlserving/.vlserving_done"
        if serving.is_file():
            verify_completed_checkpoint(
                serving,
                validated,
                learning_rate,
                model_tag=args.model_tag,
                base_model=base_model,
                base_model_config_sha256=base_model_config_sha256,
            )
            print(f"skip complete: {tag}")
            continue
        description = f"q35-toolonly-sft-{corpus_run_id}-{tag}-ep1"
        job_desc = f"{description} #gambashidze @alexander"
        if normalize_job_desc(job_desc) in in_progress:
            print(f"skip in queue: {tag}")
            continue
        script = f"cd {shlex.quote(str(staged))} && bash sweep/run_one_sft.sh"
        payload = {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {
                "WORKDIR": str(staged),
                "BASE_MODEL": str(base_model),
                "BASE_MODEL_CONFIG_SHA256": base_model_config_sha256,
                "MODEL_TAG": args.model_tag,
                "HF_HOME": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface",
                "HF_HUB_CACHE": "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DATA_PATH": str(data),
                "DATA_SHA256": actual_data_sha,
                "MANIFEST_PATH": str(manifest_path),
                "MANIFEST_SHA256": validated["manifest_sha256"],
                "PROMPT_SHA256": validated["prompt_sha256"],
                "TEMPLATE_SHA256": validated["template_sha256"],
                "CORPUS_RUN_ID": corpus_run_id,
                "OUTPUT_ROOT": str(output_root),
                "LOG_ROOT": str(log_root),
                "LR_TAG": tag,
                "LEARNING_RATE": learning_rate,
                "EPOCHS": "1",
                "MAX_LENGTH": "12288",
                "NO_TOOL_REPEAT_FACTOR": str(repeat_factors["no_tool"]),
                "SINGLE_CALL_REPEAT_FACTOR": str(repeat_factors["single_call"]),
                "PARALLEL_REPEAT_FACTOR": str(repeat_factors["parallel"]),
                "OTHER_REPEAT_FACTOR": str(repeat_factors["other"]),
                "SUPERVISION_CONTRACT": EXPECTED_SUPERVISION,
                "PREFIX_UNIT": EXPECTED_PREFIX_UNIT,
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
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.dry:
            continue
        assert client is not None
        result = run_job_with_retry(client, payload, profile=args.profile)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        launched.append({"tag": tag, "job_name": str(job_name or "")})
        print("result", result)

    if args.dry:
        print("dry validation complete: no files written and no jobs submitted")
        return 0

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    jobs_path = ARTIFACTS / f"sft_sweep_jobs.{corpus_run_id}.json"
    jobs_path.write_text(
        json.dumps(
            {
                "commit": commit,
                "run_id": corpus_run_id,
                "model_tag": args.model_tag,
                "base_model": str(base_model),
                "base_model_config_sha256": base_model_config_sha256,
                "data": str(data),
                "data_sha256": actual_data_sha,
                "manifest": str(manifest_path),
                "manifest_sha256": validated["manifest_sha256"],
                "train_repeat_factors": repeat_factors,
                "jobs": launched,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {jobs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
