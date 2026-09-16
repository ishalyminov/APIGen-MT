#!/usr/bin/env python3
"""Flag APIGen query/gold alignment risks without calling an LLM.

This is a conservative review-candidate finder, not an authoritative semantic
judge.  It detects:

* long free-text tool arguments with little lexical grounding in the visible
  conversation;
* concrete gold date years absent from the visible conversation; and
* gold ``post_tweet`` + ``mention`` plans for which the post schema already
  admits mentions, making the exact gold path non-unique.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FREE_TEXT_FIELDS = {
    "comment_content",
    "content",
    "description",
    "message",
    "resolution",
    "title",
}
STOPWORDS = set(
    "a an the to for and or of in on at with that this it is are be as by "
    "from please me my i you your we our".split()
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w']+", str(value).casefold())
        if len(token) > 1 and token not in STOPWORDS
    }


def leaves(value: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from leaves(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child, key)
    else:
        yield key, value


def turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = (row.get("conversation") or {}).get("turns") or []
    if conversation:
        return conversation
    trajectory = row.get("trajectory") or {}
    return [
        {
            "user_query": trajectory.get("query") or "",
            "assistant_response": trajectory.get("final_response") or "",
            "steps": trajectory.get("steps") or [],
        }
    ]


def source(row: dict[str, Any]) -> str:
    return str(
        (row.get("aggregation_metadata") or {}).get("source_dataset")
        or "unaggregated"
    )


def audit(rows: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    source_rows: dict[str, set[int]] = defaultdict(set)
    source_kinds: dict[str, Counter[str]] = defaultdict(Counter)

    def add(
        *,
        row_position: int,
        row_source: str,
        turn_index: int,
        kind: str,
        severity: str,
        evidence: dict[str, Any],
    ) -> None:
        findings.append(
            {
                "row_position": row_position,
                "source_dataset": row_source,
                "turn_index": turn_index,
                "kind": kind,
                "severity": severity,
                "evidence": evidence,
            }
        )
        source_rows[row_source].add(row_position)
        source_kinds[row_source][kind] += 1

    for row_position, row in enumerate(rows):
        row_source = source(row)
        visible_prefix = ""
        for turn_index, turn in enumerate(turns(row)):
            query = str(turn.get("user_query") or "")
            visible = f"{visible_prefix} {query}"
            visible_tokens = tokens(visible)
            steps = turn.get("steps") or []
            for step_offset, step in enumerate(steps):
                calls = step.get("tool_calls") or []
                for call in calls:
                    tool_name = str(
                        call.get("tool_name") or call.get("name") or ""
                    )
                    arguments = call.get("arguments") or {}
                    for key, value in leaves(arguments):
                        if (
                            isinstance(value, str)
                            and re.fullmatch(r"20\d\d-\d\d-\d\d", value)
                            and re.search(r"\b20\d\d\b", visible) is None
                        ):
                            add(
                                row_position=row_position,
                                row_source=row_source,
                                turn_index=turn_index,
                                kind="gold_date_year_not_literal_visible",
                                severity="review",
                                evidence={
                                    "query": query,
                                    "tool_name": tool_name,
                                    "argument": key,
                                    "gold_value": value,
                                },
                            )
                        if (
                            key in FREE_TEXT_FIELDS
                            and isinstance(value, str)
                            and len(value.split()) >= 4
                        ):
                            value_tokens = tokens(value)
                            overlap = len(value_tokens & visible_tokens) / max(
                                len(value_tokens), 1
                            )
                            if overlap < 0.45:
                                add(
                                    row_position=row_position,
                                    row_source=row_source,
                                    turn_index=turn_index,
                                    kind="low_visible_gold_text_overlap",
                                    severity=(
                                        "high_risk"
                                        if overlap < 0.20
                                        else "review"
                                    ),
                                    evidence={
                                        "query": query,
                                        "tool_name": tool_name,
                                        "argument": key,
                                        "gold_value": value,
                                        "token_recall": round(overlap, 4),
                                    },
                                )

                post_calls = [
                    call
                    for call in calls
                    if (
                        call.get("tool_name") or call.get("name")
                    )
                    == "post_tweet"
                ]
                if post_calls and not (
                    post_calls[0].get("arguments") or {}
                ).get("mentions"):
                    later_calls = [
                        call
                        for later_step in steps[step_offset + 1 :]
                        for call in later_step.get("tool_calls") or []
                    ]
                    if any(
                        (
                            call.get("tool_name") or call.get("name")
                        )
                        == "mention"
                        for call in later_calls
                    ):
                        add(
                            row_position=row_position,
                            row_source=row_source,
                            turn_index=turn_index,
                            kind="non_unique_post_then_mention_plan",
                            severity="diagnostic",
                            evidence={
                                "query": query,
                                "detail": (
                                    "post_tweet accepts mentions directly, but "
                                    "gold uses a later mention call"
                                ),
                            },
                        )
            visible_prefix += (
                f" {query} {str(turn.get('assistant_response') or '')}"
            )

    severity_counts = Counter(item["severity"] for item in findings)
    kind_counts = Counter(item["kind"] for item in findings)
    return {
        "input": str(input_path.resolve()),
        "method": "deterministic_review_candidate_heuristics_v1",
        "authoritative_semantic_audit": False,
        "rows": len(rows),
        "findings": len(findings),
        "unique_flagged_rows": len(
            {item["row_position"] for item in findings}
        ),
        "severity_counts": dict(severity_counts),
        "kind_counts": dict(kind_counts),
        "source_summary": {
            row_source: {
                "unique_flagged_rows": len(source_rows[row_source]),
                "finding_counts": dict(source_kinds[row_source]),
            }
            for row_source in sorted(source_rows)
        },
        "finding_records": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report = audit(read_jsonl(input_path), input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "finding_records"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
