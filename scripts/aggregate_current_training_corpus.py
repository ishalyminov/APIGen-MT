#!/usr/bin/env python3
"""Build the canonical APIGen-MT SFT/RL corpus from active accepted outputs.

The source files intentionally mix the legacy single-turn ``trajectory`` shape
and the multi-turn ``conversation`` shape.  This aggregator preserves every
source row verbatim apart from adding top-level provenance/eligibility metadata.
It rejects semantic duplicates and shuffles deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_VERSION = "apigen_mt_canonical_no_claude_20260803_v2"
RETIRED_SOURCE_LABELS = frozenset(
    {
        "claude_diverse_refuse_parallel_400",
        "claude_fixed_refuse_parallel_100",
    }
)


@dataclass(frozen=True)
class Source:
    label: str
    path: str
    quality_tier: str
    eligible_by_default: bool = True
    use_current_reviews: bool = False


SOURCES = (
    Source(
        "grok45_naturalized_long500",
        "data/generated/long7_15_grok45_500_20260727_naturalized.jsonl",
        "audited_naturalized",
    ),
    Source(
        "glm52_partial_accepted_63",
        "data/generated/runs/diverse_mtms_glm52_1000_20260729/"
        "accepted.partial.jsonl",
        "legacy_pipeline_accepted",
    ),
    Source(
        "optimized_qwen36_smoke_1turn",
        "data/generated/smoke/"
        "optimized_turn_compiler_qwen36_debug_v3_1t2s_20260730.jsonl",
        "optimized_pipeline_smoke",
        eligible_by_default=False,
    ),
    Source(
        "optimized_qwen36_smoke_2turn",
        "data/generated/smoke/"
        "optimized_turn_compiler_qwen36_v7_2t4s_20260730.jsonl",
        "optimized_pipeline_smoke",
        eligible_by_default=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/generated/canonical_sft_rl_corpus_565_no_claude_20260803.jsonl"
        ),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def semantic_signature(datapoint: dict[str, Any]) -> str:
    """Match the generator's order-aware / parallel-order-invariant dedupe."""
    if isinstance(datapoint.get("conversation"), dict):
        payload_turns = []
        for turn in datapoint["conversation"].get("turns", []):
            groups = []
            for step in turn.get("steps", []):
                group = [
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in step.get("tool_calls", [])
                ]
                if step.get("call_order_matters", True) is False:
                    group = sorted(group, key=canonical_json)
                groups.append(
                    {
                        "execution_mode": step.get(
                            "execution_mode", "sequential"
                        ),
                        "calls": group,
                    }
                )
            payload_turns.append(
                {
                    "query": turn.get("user_query", ""),
                    "groups": groups,
                }
            )
        payload: Any = payload_turns
    else:
        steps = datapoint.get("trajectory", {}).get("steps", [])
        has_parallel = any(
            len(step.get("tool_calls", [])) > 1 for step in steps
        )
        if has_parallel:
            payload = []
            for step in steps:
                group = [
                    {
                        "tool_name": call.get("tool_name"),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in step.get("tool_calls", [])
                ]
                if len(group) > 1:
                    group = sorted(group, key=canonical_json)
                payload.append(group)
        else:
            payload = [
                {
                    "tool_name": call.get("tool_name"),
                    "arguments": call.get("arguments", {}),
                }
                for step in steps
                for call in step.get("tool_calls", [])
            ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(row: dict[str, Any]) -> str | None:
    return (
        row.get("task_id")
        or (row.get("generation_metadata") or {}).get("task_id")
    )


def current_review_flags(source_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Use only reviews produced after the current aggregate was built.

    Older review files describe a pre-repair version and must not quarantine
    rows that were subsequently rebuilt.
    """
    review_dir = source_path.parent / "reviews"
    if not review_dir.is_dir():
        return {}
    source_mtime = source_path.stat().st_mtime_ns
    flags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review_path in sorted(review_dir.glob("*.jsonl")):
        if review_path.stat().st_mtime_ns < source_mtime:
            continue
        with review_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                verdict = json.loads(line)
                if verdict.get("passed") is not False:
                    continue
                row_task_id = verdict.get("task_id")
                if not row_task_id:
                    raise ValueError(
                        f"missing task_id in {review_path}:{line_number}"
                    )
                codes = verdict.get("error_codes") or []
                if isinstance(codes, str):
                    codes = [codes]
                flags[str(row_task_id)].append(
                    {
                        "review": review_path.stem,
                        "error_codes": sorted(str(code) for code in codes),
                    }
                )
    return dict(flags)


def main() -> None:
    args = parse_args()
    configured_retired = RETIRED_SOURCE_LABELS.intersection(
        source.label for source in SOURCES
    )
    if configured_retired:
        raise ValueError(
            "retired sources must never enter the active corpus: "
            + ", ".join(sorted(configured_retired))
        )
    output = (REPO_ROOT / args.output).resolve()
    manifest = (
        (REPO_ROOT / args.manifest).resolve()
        if args.manifest
        else output.with_suffix(".manifest.json")
    )
    if output == manifest:
        raise ValueError("output and manifest paths must differ")

    aggregated: list[dict[str, Any]] = []
    seen_signatures: dict[str, tuple[str, int]] = {}
    source_counts: Counter[str] = Counter()
    source_eligible: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    held_out_reasons: Counter[str] = Counter()
    input_manifest: list[dict[str, Any]] = []
    current_review_flag_ids: dict[str, list[str]] = {}

    for source in SOURCES:
        source_path = (REPO_ROOT / source.path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        review_flags = (
            current_review_flags(source_path)
            if source.use_current_reviews
            else {}
        )
        current_review_flag_ids[source.label] = sorted(review_flags)
        rows_read = 0
        with source_path.open(encoding="utf-8") as handle:
            for source_row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                rows_read += 1
                row = json.loads(line)
                signature = semantic_signature(row)
                if signature in seen_signatures:
                    previous = seen_signatures[signature]
                    raise ValueError(
                        "semantic duplicate: "
                        f"{source.label}:{source_row_index} duplicates "
                        f"{previous[0]}:{previous[1]}"
                    )
                seen_signatures[signature] = (
                    source.label,
                    source_row_index,
                )

                row_task_id = task_id(row)
                flags = review_flags.get(str(row_task_id), [])
                verification_passed = (
                    (row.get("verification_result") or {}).get(
                        "overall_verification_passed"
                    )
                    is True
                )
                eligible = (
                    source.eligible_by_default
                    and verification_passed
                    and not flags
                )
                if not source.eligible_by_default:
                    held_out_reasons["diagnostic_smoke"] += 1
                if not verification_passed:
                    held_out_reasons["verification_not_passed"] += 1
                if flags:
                    held_out_reasons["current_review_failure"] += 1

                schema = (
                    "conversation"
                    if isinstance(row.get("conversation"), dict)
                    else "trajectory"
                )
                row["aggregation_metadata"] = {
                    "corpus_version": CORPUS_VERSION,
                    "source_dataset": source.label,
                    "source_path": source.path,
                    "source_row_index": source_row_index,
                    "source_task_id": row_task_id,
                    "semantic_signature": signature,
                    "schema": schema,
                    "quality_tier": source.quality_tier,
                    "eligible_for_sft_rl": eligible,
                    "review_flags": flags,
                }
                aggregated.append(row)
                source_counts[source.label] += 1
                source_eligible[source.label] += int(eligible)
                schema_counts[schema] += 1

        input_manifest.append(
            {
                "label": source.label,
                "path": source.path,
                "sha256": file_sha256(source_path),
                "rows": rows_read,
                "quality_tier": source.quality_tier,
                "eligible_by_default": source.eligible_by_default,
                "current_review_flagged_rows": len(review_flags),
            }
        )

    random.Random(args.seed).shuffle(aggregated)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for row in aggregated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_output.replace(output)

    eligible_count = sum(
        bool(row["aggregation_metadata"]["eligible_for_sft_rl"])
        for row in aggregated
    )
    manifest_payload = {
        "corpus_version": CORPUS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "shuffle_seed": args.seed,
        "rows": len(aggregated),
        "unique_semantic_trajectories": len(seen_signatures),
        "eligible_for_sft_rl": eligible_count,
        "held_out": len(aggregated) - eligible_count,
        "held_out_reasons": dict(sorted(held_out_reasons.items())),
        "schema_counts": dict(sorted(schema_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_eligible_counts": dict(sorted(source_eligible.items())),
        "current_review_flag_ids": current_review_flag_ids,
        "inputs": input_manifest,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)

    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest),
                "rows": len(aggregated),
                "unique": len(seen_signatures),
                "eligible_for_sft_rl": eligible_count,
                "held_out": len(aggregated) - eligible_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
