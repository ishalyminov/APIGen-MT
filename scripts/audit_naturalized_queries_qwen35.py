#!/usr/bin/env python3
"""Independently audit naturalized APIGen queries with the cluster Qwen judge."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re
import statistics
import threading
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/generated/long7_15_grok45_500_20260727.jsonl"
NATURALIZED = (
    ROOT / "data/generated/long7_15_grok45_500_20260727_naturalized.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/generated/long7_15_grok45_500_20260727_naturalized."
    "independent_qwen35_audit.json"
)
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response was not a JSON object")
    return value


def judge_one(
    client: OpenAI,
    row_id: int,
    source: dict[str, Any],
    naturalized: dict[str, Any],
) -> dict[str, Any]:
    source_query = source["trajectory"]["query"]
    rewritten_query = naturalized["trajectory"]["query"]
    steps = [
        [
            {
                "tool": call["tool_name"],
                "arguments": call["arguments"],
            }
            for call in step["tool_calls"]
        ]
        for step in source["trajectory"]["steps"]
    ]
    prompt = f"""You are independently auditing a synthetic tool-calling task.
Return the requested JSON immediately, without showing analysis.

Compare the original request and its rewrite against the fixed gold tool plan.
Judge only what a normal user and assistant could infer from the request and
prior tool results. Be strict. A rewrite can preserve the requested outcomes
while no longer requiring one unique total ordering.

Return exactly one JSON object with these fields:
- semantic_equivalent: boolean
- all_user_values_preserved: boolean
- no_new_user_facts: boolean
- no_hidden_tool_output_leakage: boolean
- rewrite_is_natural_human_request: boolean
- naturalness_improved: boolean
- source_naturalness_score: integer 1..5
- rewrite_naturalness_score: integer 1..5
- source_total_order_constraint_preserved: boolean
- required_dependency_and_state_order_preserved: boolean
- stateful_timing_became_ambiguous: boolean
- meaningful_order_constraint_lost: boolean
- exact_gold_order_uniquely_required: boolean
- at_least_one_alternative_valid_order: boolean
- issue_codes: array of short strings
- concise_reason: string, at most 45 words

`exact_gold_order_uniquely_required` is true only when dependencies or explicit
language make every adjacent order in the listed plan mandatory. If two
independent calls could be swapped while still satisfying the rewritten user,
set it false and `at_least_one_alternative_valid_order` true.

`source_total_order_constraint_preserved` asks whether the rewrite retains the
original request's complete first/then/next ordering, even where that order was
artificial. Separately, `required_dependency_and_state_order_preserved` asks
whether all meaningful partial-order constraints remain clear: create before
using its ID, inspect before updating when inspection is requested, read before
write when the user wants the pre-write value, and transformations before
consumers. Set `stateful_timing_became_ambiguous` when before-versus-after can
change the observed result. Set `meaningful_order_constraint_lost` only for a
dependency/state/user-outcome constraint, not merely because independent calls
can swap. Treat loss of a meaningful constraint as a semantic-equivalence
failure.

ROW: {row_id}
ORIGINAL REQUEST:
{source_query}

REWRITTEN REQUEST:
{rewritten_query}

FIXED GOLD TOOL PLAN:
{json.dumps(steps, ensure_ascii=False)}
"""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content or ""
            result = extract_json(content)
            required_booleans = (
                "semantic_equivalent",
                "all_user_values_preserved",
                "no_new_user_facts",
                "no_hidden_tool_output_leakage",
                "rewrite_is_natural_human_request",
                "naturalness_improved",
                "source_total_order_constraint_preserved",
                "required_dependency_and_state_order_preserved",
                "stateful_timing_became_ambiguous",
                "meaningful_order_constraint_lost",
                "exact_gold_order_uniquely_required",
                "at_least_one_alternative_valid_order",
            )
            if any(not isinstance(result.get(key), bool) for key in required_booleans):
                raise ValueError("Judge omitted a required boolean")
            for key in ("source_naturalness_score", "rewrite_naturalness_score"):
                if not isinstance(result.get(key), int) or not 1 <= result[key] <= 5:
                    raise ValueError(f"Invalid {key}")
            result["row_id"] = row_id
            result["model"] = MODEL
            result["attempt"] = attempt + 1
            return result
        except Exception as error:  # Retry transient proxy/format failures.
            last_error = error
    raise RuntimeError(f"row {row_id}: {last_error}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    source = read_jsonl(SOURCE)
    naturalized = read_jsonl(NATURALIZED)
    if len(source) != len(naturalized):
        raise RuntimeError("Source/naturalized row counts differ")
    if not 1 <= args.sample_size <= len(source):
        raise ValueError("--sample-size is out of range")

    rng = random.Random(20260728)
    by_steps: dict[int, list[int]] = {}
    for index, row in enumerate(source):
        by_steps.setdefault(len(row["trajectory"]["steps"]), []).append(index)
    selected: list[int] = []
    for indices in by_steps.values():
        rng.shuffle(indices)
    while len(selected) < args.sample_size:
        made_progress = False
        for steps in sorted(by_steps):
            indices = by_steps[steps]
            if indices and len(selected) < args.sample_size:
                selected.append(indices.pop())
                made_progress = True
        if not made_progress:
            break
    selected.sort()

    client = OpenAI()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_one,
                client,
                index,
                source[index],
                naturalized[index],
            ): index
            for index in selected
        }
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            if completed % 10 == 0 or completed == len(futures):
                print(f"[{completed}/{len(futures)}] independently judged", flush=True)

    records.sort(key=lambda record: int(record["row_id"]))
    boolean_fields = (
        "semantic_equivalent",
        "all_user_values_preserved",
        "no_new_user_facts",
        "no_hidden_tool_output_leakage",
        "rewrite_is_natural_human_request",
        "naturalness_improved",
        "source_total_order_constraint_preserved",
        "required_dependency_and_state_order_preserved",
        "stateful_timing_became_ambiguous",
        "meaningful_order_constraint_lost",
        "exact_gold_order_uniquely_required",
        "at_least_one_alternative_valid_order",
    )
    summary = {
        "source": str(SOURCE),
        "naturalized": str(NATURALIZED),
        "model": MODEL,
        "sample_size": len(records),
        "sample_seed": 20260728,
        "rates": {
            key: sum(bool(record[key]) for record in records) / len(records)
            for key in boolean_fields
        },
        "mean_source_naturalness": statistics.mean(
            record["source_naturalness_score"] for record in records
        ),
        "mean_rewrite_naturalness": statistics.mean(
            record["rewrite_naturalness_score"] for record in records
        ),
        "records": records,
    }
    atomic_json(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
