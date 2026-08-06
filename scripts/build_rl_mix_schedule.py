#!/usr/bin/env python3
"""Schedule for an RL-shaped mix, not a feature benchmark.

The earlier runs made every row carry a refusal or a parallel batch, because the
original task asked for exactly that.  For RL training the mix is wrong: a policy
trained on 100% feature rows learns that refusing and fanning out are the normal
moves.  This schedule puts ordinary multi-step tool use in the majority and keeps
the two features as a minority signal.

    80%  ordinary multi-step   (half single-turn, half multi-turn)
    10%  contains one certified parallel batch
    10%  contains one refusal / clarification

Ordinary rows still carry 7-15 action transitions and real ordered dependencies;
"ordinary" means no synthetic feature, not easy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
LENGTHS = tuple(range(7, 16))
CATEGORIES = ("Communication", "Events", "Finance", "Posting Api",
              "Science", "Storage", "Travel Booking", "Vehicle Control")

# (profile, feature, schedule, refusal_reason, share)
MIX = [
    ("ordinary_single", "none", "terminal", None, 0.40),
    ("ordinary_multi", "none", "terminal", None, 0.40),
    ("parallel_only", "parallel", "terminal", None, 0.10),
    ("missing_argument", "refusal", "interactive-refusal", "missing_argument", 0.04),
    ("ambiguity", "refusal", "interactive-refusal", "ambiguity", 0.04),
    ("unsupported", "refusal", "terminal", "no_appropriate_function", 0.02),
]


def bfcl_weights() -> tuple[Counter, Counter]:
    grouped: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with BFCL.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[str(row["test_case_id"])][int(row["turn_index"])] += 1
    turns: Counter = Counter()
    calls: Counter = Counter()
    for testcase in grouped.values():
        turns[len(testcase)] += 1
        calls.update(min(value, 3) for value in testcase.values())
    return turns, calls


def action_vector(rng: random.Random, weights: Counter, steps: int, turns: int,
                  fixed: dict[int, int]) -> tuple[int, ...]:
    options, scores = [], []
    for vector in itertools.product((1, 2, 3), repeat=turns):
        if sum(vector) != steps:
            continue
        if any(vector[i] != v for i, v in fixed.items()):
            continue
        options.append(vector)
        scores.append(math.prod(weights[v] for v in vector))
    if not options:
        raise RuntimeError(f"no vector for steps={steps} turns={turns} fixed={fixed}")
    return rng.choices(options, weights=scores, k=1)[0]


def build(total: int, seed: int, shards: int) -> list[dict]:
    turn_w, call_w = bfcl_weights()
    rng = random.Random(seed)
    rows: list[dict] = []

    for profile, feature, schedule, reason, share in MIX:
        count = round(total * share)
        for index in range(count):
            steps = LENGTHS[index % len(LENGTHS)]
            if profile == "ordinary_single":
                turns, vector, fixed = 1, None, {}
                # one turn carrying every action: 7-15 steps in a single request
                vector = (steps,)
            else:
                if profile == "ordinary_multi":
                    candidates = [t for t in sorted(turn_w) if 2 <= t <= steps
                                  and steps <= 3 * t]
                elif schedule == "terminal":
                    candidates = [t for t in sorted(turn_w) if 2 <= t <= steps
                                  and steps <= 3 * t - 2]
                else:
                    candidates = [t for t in sorted(turn_w) if 5 <= t <= steps
                                  and steps <= 3 * t - 2]
                turns = rng.choices(candidates,
                                    weights=[turn_w[t] for t in candidates], k=1)[0]
                fixed = {}
                refusal_index = None
                if feature == "parallel":
                    fixed = {turns - 1: 1}
                elif schedule == "interactive-refusal":
                    refusal_index = rng.choice(list(range(2, turns - 2)))
                    fixed = {refusal_index: 1}
                elif feature == "refusal":
                    fixed = {turns - 1: 1}
                vector = action_vector(rng, call_w, steps, turns, fixed)
            rows.append({
                "profile": profile, "feature": feature, "schedule": schedule,
                "refusal_reason": reason, "steps": steps, "turns": turns,
                "steps_per_turn": list(vector),
                "parallel_width": (3, 4, 5)[index % 3] if feature == "parallel" else None,
                "refusal_turn": (
                    (fixed and min(fixed) + 1) if schedule == "interactive-refusal"
                    else (turns if feature == "refusal" else None)),
                "recovery_turn": (
                    min(fixed) + 2 if schedule == "interactive-refusal" else None),
            })

    rng.shuffle(rows)
    per = len(rows) // shards
    for index, row in enumerate(rows):
        row["task_id"] = f"rl-mix-{index:04d}"
        row["index"] = index
        row["shard"] = min(index // per, shards - 1)
        row["primary_category"] = CATEGORIES[index % len(CATEGORIES)]
        row["state_seed"] = seed + index
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--out", default="data/generated/rl_mix_1000_20260730")
    args = parser.parse_args()

    rows = build(args.total, args.seed, args.shards)
    out = ROOT / args.out
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    with (out / "schedule.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total": len(rows), "seed": args.seed, "shards": args.shards,
        "profiles": dict(sorted(Counter(r["profile"] for r in rows).items())),
        "feature_share": {
            "ordinary": round(sum(r["feature"] == "none" for r in rows) / len(rows), 3),
            "parallel": round(sum(r["feature"] == "parallel" for r in rows) / len(rows), 3),
            "refusal": round(sum(r["feature"] == "refusal" for r in rows) / len(rows), 3),
        },
        "single_turn_rows": sum(r["turns"] == 1 for r in rows),
        "steps": dict(sorted(Counter(r["steps"] for r in rows).items())),
        "turns": dict(sorted(Counter(r["turns"] for r in rows).items())),
        "widths": dict(sorted(Counter(r["parallel_width"] for r in rows
                                      if r["parallel_width"]).items())),
        "categories": dict(sorted(Counter(r["primary_category"] for r in rows).items())),
        "rows_per_shard": dict(sorted(Counter(str(r["shard"]) for r in rows).items())),
    }
    (out / "schedule.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
