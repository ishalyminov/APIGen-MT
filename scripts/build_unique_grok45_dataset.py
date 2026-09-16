#!/usr/bin/env python3
"""Build and audit a semantically unique APIGen JSONL dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


AUTHENTICATION_TOOLS = {
    "authenticate_twitter",
    "authenticate_travel",
    "message_login",
    "ticket_login",
    "trading_login",
}
CREATE_TOOLS = {
    "add_contact",
    "book_flight",
    "create_ticket",
    "place_order",
    "post_tweet",
    "register_credit_card",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def semantic_signature(row: dict[str, Any]) -> str:
    calls = [
        {
            "tool_name": call["tool_name"],
            "arguments": call["arguments"],
        }
        for step in row["trajectory"]["steps"]
        for call in step["tool_calls"]
    ]
    return json.dumps(
        calls,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def normalized_query(row: dict[str, Any]) -> str:
    return " ".join(row["trajectory"]["query"].casefold().split())


def nested_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from nested_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_scalars(child)
    elif value is not None and not isinstance(value, bool):
        yield value


def uses_prior_output(row: dict[str, Any]) -> bool:
    steps = row["trajectory"]["steps"]
    first_output = steps[0]["tool_calls"][0].get("output")
    second_arguments = steps[1]["tool_calls"][0].get("arguments", {})
    outputs = {
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        for value in nested_scalars(first_output)
    }
    arguments = {
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        for value in nested_scalars(second_arguments)
    }
    return bool(outputs & arguments)


def validate_row(row: dict[str, Any], index: int) -> None:
    metadata = row["generation_metadata"]
    verification = row["verification_result"]
    gate = verification["rl_quality_gate"]
    steps = row["trajectory"]["steps"]
    if metadata["rl_quality_gate_passed"] is not True:
        raise ValueError(f"Row {index}: generation RL gate did not pass")
    if verification["overall_verification_passed"] is not True:
        raise ValueError(f"Row {index}: overall verification did not pass")
    if gate["passed"] is not True:
        raise ValueError(f"Row {index}: final RL gate did not pass")
    if gate["query_preflight"]["passed"] is not True:
        raise ValueError(f"Row {index}: query preflight did not pass")
    if gate["final_response_grounding"]["passed"] is not True:
        raise ValueError(f"Row {index}: final response is not grounded")
    if len(steps) != 2:
        raise ValueError(f"Row {index}: expected two steps, got {len(steps)}")
    if not all(step["quality_verification"]["passed"] for step in steps):
        raise ValueError(f"Row {index}: a transition quality check failed")
    if not all(check["passed"] for check in gate["transition_checks"]):
        raise ValueError(f"Row {index}: an aggregate transition check failed")

    tools = row["available_tools"]
    payload = json.dumps(
        tools,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != metadata["tool_contract_hash"]:
        raise ValueError(f"Row {index}: tool contract hash mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--supplement",
        type=Path,
        action="append",
        default=[],
        help="Supplemental JSONL or shard; repeat in desired selection order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected: list[dict[str, Any]] = []
    semantic_seen: set[str] = set()
    query_seen: set[str] = set()
    source_counts: collections.Counter[str] = collections.Counter()

    for path in [args.raw, *args.supplement]:
        for row in read_jsonl(path):
            semantic = semantic_signature(row)
            query = normalized_query(row)
            if semantic in semantic_seen or query in query_seen:
                continue
            semantic_seen.add(semantic)
            query_seen.add(query)
            selected.append(row)
            source_counts[str(path)] += 1

    if len(selected) != args.target:
        raise ValueError(
            f"Expected exactly {args.target} unique rows, found {len(selected)}"
        )

    categories: collections.Counter[str] = collections.Counter()
    tools: collections.Counter[str] = collections.Counter()
    tool_pairs: collections.Counter[tuple[str, ...]] = collections.Counter()
    total_calls = 0
    total_tokens = 0
    direct_dependency_rows = 0
    authentication_dependency_rows = 0
    state_change_rows = 0
    create_rows = 0
    explicitly_ordered_queries = 0
    query_word_total = 0
    for index, row in enumerate(selected):
        validate_row(row, index)
        categories.update([row["generation_metadata"]["focus_category"]])
        used = tuple(row["trajectory"]["tools_used"])
        tools.update(used)
        tool_pairs.update([used])
        total_calls += row["token_usage"]["total_llm_calls"]
        total_tokens += row["token_usage"]["total_tokens"]
        first_tool, second_tool = used
        direct_dependency_rows += int(uses_prior_output(row))
        authentication_dependency_rows += int(
            first_tool in AUTHENTICATION_TOOLS
        )
        state_change_rows += int(
            any(
                step.get("pre_state") != step.get("post_state")
                for step in row["trajectory"]["steps"]
            )
        )
        create_rows += int(
            any(tool in CREATE_TOOLS for tool in used)
        )
        query = normalized_query(row)
        explicitly_ordered_queries += int(
            any(marker in query for marker in ("first ", " then ", "after "))
        )
        query_word_total += len(query.split())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in selected:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "output": str(args.output.resolve()),
        "sha256": output_sha256,
        "rows": len(selected),
        "unique_semantic_trajectories": len(semantic_seen),
        "unique_normalized_queries": len(query_seen),
        "all_quality_checks_passed": True,
        "all_tool_contract_hashes_match": True,
        "categories": dict(sorted(categories.items())),
        "distinct_tools": len(tools),
        "distinct_tool_pairs": len(tool_pairs),
        "top_tool_pairs": [
            {"tools": list(pair), "count": count}
            for pair, count in tool_pairs.most_common(20)
        ],
        "total_llm_calls": total_calls,
        "total_tokens": total_tokens,
        "average_llm_calls_per_row": total_calls / len(selected),
        "average_tokens_per_row": total_tokens / len(selected),
        "average_query_words": query_word_total / len(selected),
        "direct_output_dependency_rows": direct_dependency_rows,
        "authentication_dependency_rows": authentication_dependency_rows,
        "dependent_rows_union_lower_bound": sum(
            uses_prior_output(row)
            or row["trajectory"]["tools_used"][0] in AUTHENTICATION_TOOLS
            for row in selected
        ),
        "state_change_rows": state_change_rows,
        "create_rows": create_rows,
        "explicitly_ordered_query_rows": explicitly_ordered_queries,
        "selected_source_counts": dict(source_counts),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_temporary.replace(args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
