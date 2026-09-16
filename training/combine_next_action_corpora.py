#!/usr/bin/env python3
"""Combine immutable next-action corpora without changing source rows.

This builder is intentionally narrow: both inputs must already have passed the
same hidden-argument, prompt, template, schema-projection, and next-action
supervision contract.  It performs training-semantic deduplication and writes a
multi-source manifest that the sweep launcher verifies before submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


OWNER = Path("/mnt/shared_ru.ml.SZ-5_000264/gambashidze")
ARTIFACTS = OWNER / "qwen35_toolonly_sft_sweep_artifacts"
DEFAULT_BASE = ARTIFACTS / "data/apigen_toolonly_sft_next_action_v2.jsonl"
DEFAULT_ADDON = OWNER / (
    "APIGen-MT-github/data/generated/"
    "bfcl_transfer_targeted_200_20260816_v13/audits/"
    "next_action_visibility.jsonl"
)
DEFAULT_OUTPUT = ARTIFACTS / (
    "data/apigen_toolonly_sft_next_action_targeted200_v1.jsonl"
)
RELEASE_TIMESTAMP = "2026-08-17T00:00:00+00:00"

CONTRACT_KEYS = (
    "tool_schema_projection",
    "system_prompt_content_sha256",
    "chat_template_sha256",
    "enable_thinking",
    "supervision",
    "prefix_unit",
    "golden_history",
    "parallel_group_is_one_target",
    "terminal_stop_is_separate_target",
    "split_unit",
    "hidden_argument_policy",
    "train_repeat_factors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            result.append(value)
    return result


def source_manifest(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def validate_source(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = source_manifest(path)
    for required in (path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(path)
    expected_rows = manifest.get("rows")
    if (
        not isinstance(expected_rows, int)
        or expected_rows <= 0
        or expected_rows != manifest.get("expected_rows")
        or len(rows) != expected_rows
    ):
        raise ValueError(f"row contract failed for {path}")
    actual_sha = sha256(path)
    if manifest.get("output_sha256") != actual_sha:
        raise ValueError(f"output hash mismatch for {path}")
    if manifest.get("unique_semantic_trajectories") != expected_rows:
        raise ValueError(f"source is not semantically unique: {path}")
    contract = manifest.get("training_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"missing training contract: {manifest_path}")
    if contract.get("enable_thinking") is not False:
        raise ValueError(f"thinking is not disabled: {manifest_path}")
    visibility = manifest.get("argument_visibility", {}).get("output", {})
    if visibility.get("hidden_argument_count") != 0:
        raise ValueError(f"hidden gold arguments remain in {path}")
    return rows, manifest


def semantic_fingerprint(row: dict[str, Any]) -> str:
    """Hash only fields that can affect the trainer's rendered examples."""
    payload = {
        "conversation": row.get("conversation"),
        "trajectory": row.get("trajectory"),
        "available_tools": row.get("available_tools"),
    }
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def projected_contract(contract: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(f"training contract is missing keys: {missing}")
    return {key: contract[key] for key in CONTRACT_KEYS}


def iter_supervised_turns(row: dict[str, Any]):
    conversation = row.get("conversation")
    if isinstance(conversation, dict):
        for turn in conversation.get("turns", []):
            if turn.get("sft_supervision") is not False:
                yield turn
        return
    trajectory = row.get("trajectory")
    if isinstance(trajectory, dict):
        yield trajectory


def dataset_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    tools: set[str] = set()
    for row in rows:
        stats["input_rows"] += 1
        for turn in iter_supervised_turns(row):
            stats["supervised_turns"] += 1
            if turn.get("no_tool_target") is True:
                stats["no_call_targets"] += 1
            # Every supervised turn produces one terminal empty action.
            stats["terminal_stop_targets"] += 1
            for step in turn.get("steps", []):
                calls = step.get("tool_calls", [])
                if not calls:
                    continue
                stats["action_targets"] += 1
                stats[f"action_width:{len(calls)}"] += 1
                if len(calls) > 1:
                    stats["parallel_targets"] += 1
                for call in calls:
                    stats["tool_calls"] += 1
                    name = call.get("tool_name")
                    if isinstance(name, str):
                        tools.add(name)
    stats["decision_targets"] = (
        stats["action_targets"] + stats["terminal_stop_targets"]
    )
    stats["output_rows"] = len(rows)
    result = dict(sorted(stats.items()))
    result["unique_tools"] = len(tools)
    return result


def sum_visibility(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = Counter()
    source_counts = Counter()
    for manifest in manifests:
        output = manifest["argument_visibility"]["output"]
        for key in (
            "total_arguments",
            "hidden_argument_count",
            "required_argument_count",
            "hidden_required_argument_count",
            "tasks_with_hidden_arguments",
        ):
            numeric[key] += int(output.get(key, 0))
        source_counts.update(output.get("visibility_source_counts", {}))
    total = numeric["total_arguments"]
    required = numeric["required_argument_count"]
    return {
        "method": "conservative_policy_visible_top_level_arguments_v1",
        "definition": (
            "Sum of the already-audited immutable source views; initial state "
            "and parallel sibling outputs are hidden."
        ),
        "initial_state_policy_visible": False,
        **dict(numeric),
        "hidden_argument_ratio": (
            numeric["hidden_argument_count"] / total if total else 0.0
        ),
        "hidden_required_argument_ratio": (
            numeric["hidden_required_argument_count"] / required
            if required else 0.0
        ),
        "visibility_source_counts": dict(sorted(source_counts.items())),
        "hidden_by_tool": {},
    }


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--addon", type=Path, default=DEFAULT_ADDON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-addon-rows", type=int, default=200)
    args = parser.parse_args()
    base = args.base.resolve()
    addon = args.addon.resolve()
    output = args.output.resolve()
    if OWNER not in output.parents:
        raise ValueError(f"output must remain under {OWNER}")
    if output in {base, addon}:
        raise ValueError("output may not overwrite an immutable source")

    base_rows, base_manifest = validate_source(base)
    addon_rows, addon_manifest = validate_source(addon)
    if len(addon_rows) != args.expected_addon_rows:
        raise ValueError(
            f"expected {args.expected_addon_rows} addon rows, got {len(addon_rows)}"
        )
    base_contract = projected_contract(base_manifest["training_contract"])
    addon_contract = projected_contract(addon_manifest["training_contract"])
    if base_contract != addon_contract:
        raise ValueError("base and addon training contracts differ")

    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = Counter()
    for role, rows in (("base", base_rows), ("addon", addon_rows)):
        for row in rows:
            fingerprint = semantic_fingerprint(row)
            if fingerprint in seen:
                duplicates[role] += 1
                continue
            seen.add(fingerprint)
            combined.append(row)
    if duplicates:
        raise ValueError(
            "training-semantic duplicates found; sources must be fixed rather "
            f"than silently downsampled: {dict(duplicates)}"
        )
    expected = len(base_rows) + len(addon_rows)
    if len(combined) != expected:
        raise AssertionError("combined row count changed unexpectedly")

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output, combined)
    visibility = sum_visibility([base_manifest, addon_manifest])
    if visibility["hidden_argument_count"] != 0:
        raise AssertionError("combined visibility audit is not clean")
    manifest = {
        "version": "next_action_combined_targeted200_v1",
        "created_at": RELEASE_TIMESTAMP,
        "inputs": [
            {"path": str(base), "sha256": sha256(base), "role": "base"},
            {"path": str(addon), "sha256": sha256(addon), "role": "targeted_addon"},
        ],
        "source_files": [
            {
                "path": str(source_manifest(base)),
                "sha256": sha256(source_manifest(base)),
                "role": "base_manifest",
            },
            {
                "path": str(source_manifest(addon)),
                "sha256": sha256(source_manifest(addon)),
                "role": "addon_manifest",
            },
        ],
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": len(combined),
        "expected_rows": expected,
        "unique_semantic_trajectories": len(seen),
        "source_file_modified": False,
        "duplicates_skipped": 0,
        "statistics": dataset_statistics(combined),
        "argument_visibility": {
            "policy_history": base_manifest["argument_visibility"]["policy_history"],
            "source": visibility,
            "output": visibility,
        },
        "training_contract": base_contract,
    }
    manifest_path = output.with_suffix(".manifest.json")
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
