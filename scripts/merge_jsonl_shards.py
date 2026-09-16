#!/usr/bin/env python3
"""Atomically merge validated JSONL shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    for path in args.input:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(
            f"Merged row count {len(rows)} != expected {args.expected_rows}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Merged {len(rows)} rows into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
