#!/usr/bin/env python3
"""Build a non-destructive, fail-closed next-action SFT corpus view.

The rich source trajectories remain immutable.  This builder copies eligible
rows into a new versioned JSONL, truncating a conversation before its first
turn whose stored gold arguments fail the conservative policy-visible audit.
A single-turn trajectory with such a failure is quarantined as a whole.

Actual next-action expansion happens in ``train_toolcalling_toolonly.py`` so
all prefixes from one source trajectory stay in the same train/validation
split.  The manifest fixes that inference-shaped supervision contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


OWNER = Path("/mnt/shared_ru.ml.SZ-5_000264/gambashidze")
ROOT = Path(__file__).resolve().parents[1]
BUNDLED_PASSK = ROOT / "evaluation" / "passk"
TOOL_SYNTH = (
    BUNDLED_PASSK
    if (BUNDLED_PASSK / "check_apigen_trajectories_passk_v3.py").is_file()
    else OWNER / "tool_synth"
)
DEFAULT_INPUT = OWNER / (
    "qwen35_toolonly_sft_sweep_artifacts/data/"
    "apigen_toolonly_sft_local_transfer_v4.jsonl"
)
DEFAULT_OUTPUT = OWNER / (
    "qwen35_toolonly_sft_sweep_artifacts/data/"
    "apigen_toolonly_sft_next_action_v2.jsonl"
)
PROMPT = ROOT / "prompts/tool_only_system.txt"
TEMPLATE = ROOT / "templates/qwen35_toolonly_base.jinja"
PROJECTION_VERSION = "next_action_visible_v2"
SUPERVISION = "next_action_group_and_terminal_stop"
PREFIX_UNIT = "one_per_next_action_or_parallel_group_with_golden_history"
SCHEMA_PROJECTION = "openai_name_description_parameters_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def supervised_view(row: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that can change emitted SFT prefixes or targets.

    Aggregation/source metadata and final assistant prose are intentionally
    absent: the tool-only trainer never renders them.  This makes duplicate
    detection match the actual supervised view rather than row identity.
    """

    def step_view(step: dict[str, Any]) -> dict[str, Any]:
        return {
            "calls": [
                {
                    "name": call.get("tool_name", call.get("name")),
                    "arguments": call.get("arguments"),
                    "output": call.get("output"),
                }
                for call in (step.get("tool_calls") or [])
            ],
        }

    def turn_view(turn: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": turn.get("user_query", turn.get("query")),
            "steps": [step_view(step) for step in (turn.get("steps") or [])],
            "sft_supervision": turn.get("sft_supervision") is not False,
            "no_tool_target": turn.get("no_tool_target") is True,
            "no_tool_reason": turn.get("no_tool_reason"),
            "available_tools": turn.get("available_tools"),
        }

    conversation = row.get("conversation")
    if isinstance(conversation, dict):
        turns = conversation.get("turns") or []
    else:
        turns = [row.get("trajectory") or {}]
    return {
        "available_tools": row.get("available_tools"),
        "turns": [turn_view(turn) for turn in turns],
    }


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def row_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = row.get("conversation")
    if isinstance(conversation, dict):
        turns = conversation.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("conversation has no turns")
        return turns
    trajectory = row.get("trajectory")
    if isinstance(trajectory, dict):
        return [trajectory]
    raise ValueError("row has neither conversation nor trajectory")


def turn_is_supervised(turn: dict[str, Any]) -> bool:
    return turn.get("sft_supervision") is not False


def action_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    names: set[str] = set()
    for row in records:
        for turn in row_turns(row):
            if not turn_is_supervised(turn):
                continue
            steps = turn.get("steps") or []
            if not isinstance(steps, list):
                raise ValueError("turn steps must be a list")
            executable = 0
            for step in steps:
                calls = step.get("tool_calls") or []
                if not isinstance(calls, list) or not calls:
                    continue
                executable += 1
                counts["action_targets"] += 1
                counts[f"action_width:{len(calls)}"] += 1
                if len(calls) > 1:
                    counts["parallel_targets"] += 1
                for call in calls:
                    name = call.get("tool_name") or call.get("name")
                    if isinstance(name, str) and name:
                        names.add(name)
                    counts["tool_calls"] += 1
            # Exactly one terminal decision for every supervised real user
            # turn. A no-call turn therefore contributes one target, not two.
            counts["terminal_or_no_call_targets"] += 1
            if executable == 0:
                counts["no_call_targets"] += 1
            else:
                counts["terminal_stop_targets"] += 1
    counts["decision_targets"] = (
        counts["action_targets"] + counts["terminal_or_no_call_targets"]
    )
    counts["unique_tools"] = len(names)
    return counts


def audit_hidden_turns(
    input_path: Path,
    *,
    current_date: str,
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    if str(TOOL_SYNTH) not in sys.path:
        sys.path.insert(0, str(TOOL_SYNTH))
    import check_apigen_trajectories_passk as legacy
    import check_apigen_trajectories_passk_v3 as v3

    tasks, _ = legacy.load_tasks(
        str(input_path),
        tool_scope="declared",
    )
    for task in tasks:
        task.tools = v3._real_tools(task.tools)
        # Tool-only SFT deliberately removes recorded assistant prose.  Do not
        # let that omitted text make a later gold argument appear observable.
        for turn in task.user_turns:
            turn["assistant_response"] = ""
    report = v3._argument_visibility_validation(
        tasks,
        current_date=current_date,
        include_initial_state=False,
        max_examples=1_000_000,
    )
    examples = report.get("hidden_examples") or []
    if len(examples) != int(report.get("hidden_argument_count", -1)):
        raise ValueError("visibility audit examples are unexpectedly truncated")
    bad_turns = {
        (int(item["row_position"]), int(item["turn_index"]))
        for item in examples
    }
    return report, bad_turns


def validate_output(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row_index, row in enumerate(records):
        signature = canonical_hash(supervised_view(row))
        if signature in seen:
            raise ValueError(f"duplicate projected row at position {row_index}")
        seen.add(signature)
        turns = row_turns(row)
        if not turns:
            raise ValueError(f"row {row_index} has no turns")
        for turn_index, turn in enumerate(turns):
            query = turn.get("user_query", turn.get("query"))
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"row {row_index} turn {turn_index} has no query"
                )
            steps = turn.get("steps") or []
            if not isinstance(steps, list):
                raise ValueError(
                    f"row {row_index} turn {turn_index} steps is not a list"
                )
            if turn.get("no_tool_target") is True and steps:
                raise ValueError("no_tool_target turn has executable steps")
        if not any(turn_is_supervised(turn) for turn in turns):
            raise ValueError(
                f"row {row_index} has no supervised turns and emits no targets"
            )


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def visibility_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "method",
            "definition",
            "initial_state_policy_visible",
            "total_arguments",
            "hidden_argument_count",
            "hidden_argument_ratio",
            "required_argument_count",
            "hidden_required_argument_count",
            "hidden_required_argument_ratio",
            "tasks_with_hidden_arguments",
            "visibility_source_counts",
            "hidden_by_tool",
            "by_source",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--current-date", default="2026-08-12")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output = args.output.resolve()
    quarantine = (
        args.quarantine.resolve()
        if args.quarantine
        else output.with_suffix(".quarantine.jsonl")
    )
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else output.with_suffix(".manifest.json")
    )
    for path in (input_path, PROMPT, TEMPLATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output, quarantine, manifest):
        if OWNER not in path.parents:
            raise ValueError(f"output must stay under {OWNER}: {path}")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    source_records = list(rows(input_path))
    source_visibility, bad_turns = audit_hidden_turns(
        input_path,
        current_date=args.current_date,
    )
    first_bad: dict[int, int] = {}
    for row_index, turn_index in bad_turns:
        first_bad[row_index] = min(first_bad.get(row_index, turn_index), turn_index)

    projected: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    projected_view_indices: dict[str, int] = {}
    stats: Counter[str] = Counter()
    for row_index, source in enumerate(source_records):
        bad_index = first_bad.get(row_index)
        row = copy.deepcopy(source)
        if bad_index is None:
            stats["rows_unchanged"] += 1
        elif isinstance(row.get("conversation"), dict) and bad_index > 0:
            turns = row["conversation"].get("turns") or []
            removed = turns[bad_index:]
            row["conversation"]["turns"] = turns[:bad_index]
            stats["rows_truncated_before_hidden_turn"] += 1
            stats["turns_quarantined"] += len(removed)
            quarantined.append(
                {
                    "source_row_index": row_index,
                    "first_hidden_turn_index": bad_index,
                    "reason": "hidden_argument_visibility_audit",
                    "removed_turns": removed,
                }
            )
        else:
            stats["rows_quarantined"] += 1
            stats["turns_quarantined"] += len(row_turns(row))
            quarantined.append(
                {
                    "source_row_index": row_index,
                    "first_hidden_turn_index": bad_index,
                    "reason": "hidden_argument_visibility_audit",
                    "source_row": row,
                }
            )
            continue

        if not any(turn_is_supervised(turn) for turn in row_turns(row)):
            stats["rows_quarantined_no_supervised_target"] += 1
            stats["turns_quarantined"] += len(row_turns(row))
            quarantined.append(
                {
                    "source_row_index": row_index,
                    "reason": "no_supervised_target",
                    "source_row": row,
                }
            )
            continue

        metadata = row.setdefault("aggregation_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("aggregation_metadata must be an object")
        metadata["parent_corpus_version"] = metadata.get("corpus_version")
        metadata["corpus_version"] = PROJECTION_VERSION
        metadata["next_action_projection"] = {
            "source_row_index": row_index,
            "source_row_sha256": canonical_hash(source),
            "first_hidden_turn_index": bad_index,
            "hidden_turn_policy": "truncate_before_first_affected_turn",
        }
        metadata["eligible_for_sft_rl"] = True
        view_hash = canonical_hash(supervised_view(row))
        duplicate_of = projected_view_indices.get(view_hash)
        if duplicate_of is not None:
            stats["rows_quarantined_semantic_duplicate"] += 1
            quarantined.append(
                {
                    "source_row_index": row_index,
                    "reason": "duplicate_supervised_view",
                    "duplicate_of_projected_row_index": duplicate_of,
                    "source_row": row,
                }
            )
            continue
        projected_view_indices[view_hash] = len(projected)
        projected.append(row)

    validate_output(projected)
    # Re-run the exact policy-history visibility audit on the materialized
    # projection before admitting it.  A source-position or truncation bug must
    # fail closed rather than leak into training.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix="next_action_visibility_",
        dir=output.parent,
        delete=False,
    ) as handle:
        audit_path = Path(handle.name)
        for row in projected:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    try:
        output_visibility, remaining_bad_turns = audit_hidden_turns(
            audit_path,
            current_date=args.current_date,
        )
    finally:
        audit_path.unlink(missing_ok=True)
    if output_visibility.get("hidden_argument_count") != 0 or remaining_bad_turns:
        raise ValueError(
            "projected corpus still contains hidden arguments: "
            f"count={output_visibility.get('hidden_argument_count')} "
            f"turns={len(remaining_bad_turns)}"
        )
    counts = action_counts(projected)
    stats.update(counts)
    stats["input_rows"] = len(source_records)
    stats["output_rows"] = len(projected)
    stats["visibility_bad_turns"] = len(bad_turns)
    stats["visibility_bad_rows"] = len(first_bad)

    atomic_write_jsonl(output, projected)
    atomic_write_jsonl(quarantine, quarantined)
    output_hash = sha256(output)
    manifest_payload = {
        "version": PROJECTION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "output": str(output),
        "output_sha256": output_hash,
        "rows": len(projected),
        "expected_rows": len(projected),
        "unique_semantic_trajectories": len(projected),
        "quarantine": str(quarantine),
        "quarantine_sha256": sha256(quarantine),
        "quarantined_records": len(quarantined),
        "source_file_modified": False,
        "statistics": dict(sorted(stats.items())),
        "argument_visibility": {
            "policy_history": (
                "user queries, accepted prior calls and tool outputs, schema "
                "declarations, and system date; recorded assistant prose omitted"
            ),
            "source": visibility_summary(source_visibility),
            "output": visibility_summary(output_visibility),
        },
        "training_contract": {
            "tool_schema_projection": SCHEMA_PROJECTION,
            "system_prompt_content_sha256": hashlib.sha256(
                PROMPT.read_text(encoding="utf-8").strip().encode("utf-8")
            ).hexdigest(),
            "chat_template_sha256": sha256(TEMPLATE),
            "enable_thinking": False,
            "supervision": SUPERVISION,
            "prefix_unit": PREFIX_UNIT,
            "golden_history": True,
            "parallel_group_is_one_target": True,
            "terminal_stop_is_separate_target": True,
            "split_unit": "source_trajectory_before_prefix_expansion",
            "hidden_argument_policy": "truncate_before_first_affected_turn",
            "train_repeat_factors": {
                "no_tool": 1,
                "single_call": 1,
                "parallel": 1,
                "other": 1,
            },
        },
    }
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
