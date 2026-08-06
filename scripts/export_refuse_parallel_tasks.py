#!/usr/bin/env python3
"""Export feature transitions as exact pass@k tasks.

The input is generator JSONL containing ``generation_metadata.evaluation_spec``.
Each output row contains one action target with the exact policy-visible messages,
tool contracts, matching semantics, and either an internal or BFCL-native target.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable


FEATURE_MODES = {"parallel", "refusal", "clarification"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-format",
        choices=("internal", "bfcl-native"),
        default="internal",
    )
    parser.add_argument(
        "--include-sequential",
        action="store_true",
        help="Also export ordinary sequential transitions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    exported = 0

    with temporary.open("w", encoding="utf-8") as destination:
        for row_index, row in enumerate(read_jsonl(args.input)):
            metadata = row.get("generation_metadata", {})
            spec = metadata.get("evaluation_spec")
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Row {row_index}: missing generation_metadata.evaluation_spec"
                )
            tools = copy.deepcopy(row.get("available_tools", []))
            if args.target_format == "bfcl-native":
                tools = [tool for tool in tools if tool.get("name") != "refuse"]

            for transition in spec.get("transitions", []):
                mode = transition.get("mode")
                if mode not in FEATURE_MODES and not args.include_sequential:
                    continue
                target_key = (
                    "internal_target"
                    if args.target_format == "internal"
                    else "bfcl_native_target"
                )
                output = {
                    "task_id": f"row{row_index}:{transition['transition_id']}",
                    "source_row": row_index,
                    "transition_id": transition["transition_id"],
                    "mode": mode,
                    "messages": (
                        transition.get(
                            "bfcl_native_policy_messages",
                            transition["policy_messages"],
                        )
                        if args.target_format == "bfcl-native"
                        else transition["policy_messages"]
                    ),
                    "available_tools": tools,
                    "target": transition[target_key],
                    "matching": transition["matching"],
                    "target_format": args.target_format,
                    "tool_contract_hash": metadata.get("tool_contract_hash"),
                    "evaluation_spec_version": spec.get("version"),
                }
                destination.write(json.dumps(output, ensure_ascii=False) + "\n")
                exported += 1

    temporary.replace(args.output)
    print(f"Exported {exported} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
