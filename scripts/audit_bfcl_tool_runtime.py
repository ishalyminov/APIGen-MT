#!/usr/bin/env python3
"""Execute and replay-audit every BFCL V3 multi-turn tool implementation.

This is deliberately independent of generation prompts.  Each of the 129
stateful BFCL V3 environment APIs is invoked against a deterministic fixture,
its top-level output is checked against the executable schema, and the same
call is replayed on a second fresh simulator instance.  Output and resulting
state must match exactly after normalising private temporary-directory paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_bfcl_output_schemas import ARGS, CLASS_NAMES, CONFIGS, json_type


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonicalize(value: Any, *, temp_root: str = "") -> Any:
    """Remove process-local details while preserving observable semantics."""
    if isinstance(value, dict):
        result = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if key == "_temp_dir":
                result[key] = "<VFS_ROOT>"
                continue
            result[str(key)] = canonicalize(child, temp_root=temp_root)
        return result
    if isinstance(value, list):
        return [canonicalize(item, temp_root=temp_root) for item in value]
    if isinstance(value, tuple):
        return [canonicalize(item, temp_root=temp_root) for item in value]
    if isinstance(value, str) and temp_root:
        return value.replace(temp_root, "<VFS_ROOT>")
    return value


def snapshot(instance: Any) -> dict[str, Any]:
    if callable(getattr(instance, "get_state", None)):
        state = instance.get_state()
    else:
        state = copy.deepcopy(vars(instance))
    temp_root = str(getattr(instance, "_temp_dir", ""))
    return canonicalize(state, temp_root=temp_root)


def schema_issues(output: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    issues: list[dict[str, Any]] = []
    for key, value in output.items():
        field = properties.get(key)
        if not isinstance(field, dict):
            issues.append({"field": key, "issue": "missing_output_field_schema"})
            continue
        expected = field.get("type")
        actual = json_type(value)
        if expected != actual:
            issues.append({
                "field": key,
                "issue": "output_type_mismatch",
                "expected": expected,
                "actual": actual,
            })
    return issues


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        default="magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl",
    )
    parser.add_argument(
        "--output",
        default="bfcl_v3_tool_runtime_audit.json",
    )
    args = parser.parse_args()

    rows = load_jsonl(ROOT / args.pool)
    results: list[dict[str, Any]] = []

    for row in rows:
        module_name = row["tool_name"]
        api_name = row["api_name"]
        module = importlib.import_module(f"tools.{module_name}")
        cls = getattr(module, CLASS_NAMES[module_name])
        call_args = copy.deepcopy(ARGS.get(api_name, {}))

        first = cls(copy.deepcopy(CONFIGS[module_name]))
        second = cls(copy.deepcopy(CONFIGS[module_name]))
        first_output = getattr(first, api_name)(**copy.deepcopy(call_args))
        second_output = getattr(second, api_name)(**copy.deepcopy(call_args))
        first_temp_root = str(getattr(first, "_temp_dir", ""))
        second_temp_root = str(getattr(second, "_temp_dir", ""))
        canonical_first_output = canonicalize(
            first_output,
            temp_root=first_temp_root,
        )
        canonical_second_output = canonicalize(
            second_output,
            temp_root=second_temp_root,
        )
        first_state = snapshot(first)
        second_state = snapshot(second)

        issues: list[dict[str, Any]] = []
        if not isinstance(first_output, dict):
            issues.append({
                "issue": "non_object_output",
                "actual": type(first_output).__name__,
            })
        else:
            issues.extend(schema_issues(first_output, row.get("output_schema", {})))
        if canonical_first_output != canonical_second_output:
            issues.append({"issue": "nondeterministic_output"})
        if first_state != second_state:
            issues.append({"issue": "nondeterministic_post_state"})

        results.append({
            "category": row["category"],
            "tool_class": module_name,
            "api_name": api_name,
            "arguments": call_args,
            "output": canonical_first_output,
            "output_hash": digest(canonical_first_output),
            "post_state_hash": digest(first_state),
            "passed": not issues,
            "issues": issues,
        })

    report = {
        "scope": "BFCL V3 stateful multi-turn environment APIs",
        "tool_count": len(rows),
        "passed_count": sum(item["passed"] for item in results),
        "failed_count": sum(not item["passed"] for item in results),
        "passed": len(rows) == 129 and all(item["passed"] for item in results),
        "tools": results,
    }
    output_path = ROOT / args.output
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": report["passed"],
        "tool_count": report["tool_count"],
        "passed_count": report["passed_count"],
        "failed_count": report["failed_count"],
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
