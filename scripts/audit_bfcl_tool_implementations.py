#!/usr/bin/env python3
"""Audit BFCL V3 multi-turn schemas against local Python implementations."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_NAMES = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "PostingAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelBooking",
    "vehicle_control": "VehicleControl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definitions", default="magnet_tool_extraction/bfcl_v3_tool_definitions.jsonl")
    parser.add_argument("--pool", default="magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl")
    parser.add_argument("--json-report", default="bfcl_v3_tool_implementation_audit.json")
    parser.add_argument("--markdown-report", default="bfcl_v3_tool_implementation_audit.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    definitions = load_jsonl(root / args.definitions)
    pool = load_jsonl(root / args.pool)
    pool_by_name = {row["api_name"]: row for row in pool}

    rows = []
    for definition in definitions:
        module_name = definition["tool_name"]
        api_name = definition["api_name"]
        module = importlib.import_module(f"tools.{module_name}")
        cls = getattr(module, CLASS_NAMES[module_name])
        method = getattr(cls, api_name, None)
        callable_method = callable(method)
        signature_match = False
        signature = ""
        mismatch = []
        if callable_method:
            sig = inspect.signature(method)
            signature = str(sig)
            impl = [p for p in sig.parameters.values() if p.name != "self"]
            impl_names = {p.name for p in impl}
            impl_required = {p.name for p in impl if p.default is inspect.Parameter.empty}
            schema_names = set(definition["parameters"].get("properties", {}))
            schema_required = set(definition["parameters"].get("required", []))
            signature_match = impl_names == schema_names and impl_required == schema_required
            if impl_names != schema_names:
                mismatch.append({"kind": "parameter_names", "schema": sorted(schema_names), "implementation": sorted(impl_names)})
            if impl_required != schema_required:
                mismatch.append({"kind": "required_parameters", "schema": sorted(schema_required), "implementation": sorted(impl_required)})

        pool_row = pool_by_name.get(api_name)
        rows.append({
            "category": definition["category"],
            "tool_class": module_name,
            "api_name": api_name,
            "callable": callable_method,
            "signature": signature,
            "signature_match": signature_match,
            "signature_issues": mismatch,
            "present_in_executable_pool": pool_row is not None,
            "has_output_schema": bool(pool_row and pool_row.get("output_schema")),
        })

    report = {
        "definition_count": len(definitions),
        "definition_unique_tools": len({row["api_name"] for row in definitions}),
        "pool_count": len(pool),
        "pool_unique_tools": len(pool_by_name),
        "category_counts": dict(Counter(row["category"] for row in definitions)),
        "callable_count": sum(row["callable"] for row in rows),
        "signature_match_count": sum(row["signature_match"] for row in rows),
        "output_schema_count": sum(row["has_output_schema"] for row in rows),
        "missing_implementations": [row for row in rows if not row["callable"]],
        "signature_mismatches": [row for row in rows if row["callable"] and not row["signature_match"]],
        "missing_from_pool": [row for row in rows if not row["present_in_executable_pool"]],
        "missing_output_schemas": [row for row in rows if not row["has_output_schema"]],
        "tools": rows,
    }
    passed = all(
        report[key] == 129
        for key in ("definition_count", "definition_unique_tools", "pool_count", "pool_unique_tools", "callable_count", "signature_match_count", "output_schema_count")
    )
    report["passed"] = passed

    json_path = root / args.json_report
    md_path = root / args.markdown_report
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# BFCL V3 multi-turn tool implementation audit",
            "",
            f"- Definitions: **{report['definition_unique_tools']}**",
            f"- Executable-pool tools: **{report['pool_unique_tools']}**",
            f"- Callable implementations: **{report['callable_count']}**",
            f"- Exact input-signature matches: **{report['signature_match_count']}**",
            f"- Output schemas: **{report['output_schema_count']}**",
            f"- Overall: **{'PASS' if passed else 'FAIL'}**",
            "",
            "## Categories",
            "",
            *[f"- {category}: {count}" for category, count in sorted(report["category_counts"].items())],
            "",
            "## Missing implementations",
            "",
            *( ["None."] if not report["missing_implementations"] else [f"- {row['api_name']}" for row in report["missing_implementations"]] ),
            "",
            "## Signature mismatches",
            "",
            *( ["None."] if not report["signature_mismatches"] else [f"- {row['api_name']}: {row['signature_issues']}" for row in report["signature_mismatches"]] ),
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("passed", "definition_unique_tools", "pool_unique_tools", "callable_count", "signature_match_count", "output_schema_count")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
