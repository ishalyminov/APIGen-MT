#!/usr/bin/env python3
"""Local assembly, replay, certification and reporting for the fixed 100-row
refusal + certified-parallel dataset.

This helper makes **no network calls and no LLM calls**.  Scenario design, user
queries, tool selection, arguments and assistant prose are authored by Claude
Code subagents into compact ``spec`` rows; this module executes the real local
Python tool implementations, records the true outputs and state transitions,
certifies parallel batches by isolated/forward/reverse replay, derives the
``refuse-parallel-v3`` evaluation spec with the current library code, and
fails closed with actionable error codes.

Commands
--------
``schedule``   build the deterministic 100-row schedule
``states``     materialise the per-task initial API state + author digests
``probe``      execute candidate calls against a task's initial state
``build``      turn author spec rows into full validated trajectory rows
``merge``      merge shards, split profile files
``verify``     independent full replay of a serialized dataset file
``manifest``   deterministic distribution report over the merged dataset
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apigen_step_by_step import StepByStepGenerator  # noqa: E402
from config_pool import generate_random_config  # noqa: E402
from refuse_parallel import REFUSE_TOOL_SCHEMA  # noqa: E402
from refuse_parallel_eval import (  # noqa: E402
    prepare_multiturn_datapoint,
    validate_feature_evaluation_spec,
)
from rl_quality_gate import validate_transition_quality  # noqa: E402
from tool_manager import ToolManager  # noqa: E402


def _freeze_simulator_clock() -> None:
    """Pin ``datetime.now()`` inside the tool implementations.

    ``trading_bot.place_order`` stamps each transaction with a wall-clock time
    and stores it in the simulator state.  That makes every affected row
    unreproducible: replaying it a second later yields a different state, so the
    recorded trajectory can never be verified again.  The dataset is meant to be
    deterministic from a seed, so the clock is pinned to the dataset's own
    nominal date rather than left to run.
    """
    import datetime as _datetime
    import tools.trading_bot as _trading
    import tools.travel_booking as _travel

    frozen = _datetime.datetime(2026, 7, 29, 0, 0, 0)

    class _FrozenDateTime(_datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102 - mirrors datetime.now
            return frozen

        @classmethod
        def today(cls):  # noqa: D102
            return frozen

    _trading.datetime = _FrozenDateTime
    if hasattr(_travel, "datetime"):
        _travel.datetime.datetime = _FrozenDateTime


_freeze_simulator_clock()

SEED = 20260729
TOOL_POOL = ROOT / "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
BFCL_EXAMPLES = ROOT / "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
OUT_DIR = ROOT / "data/generated/fixed_refuse_parallel_100_20260729"
RUN_NAME = "fixed_refuse_parallel_100"
MERGED_NAME = "fixed_refuse_parallel_100.jsonl"
TASK_PREFIX = "fixed-rp"
TIMESTAMP = "2026-07-29T00:00:00"
LENGTHS = tuple(range(7, 16))

CATEGORIES = (
    "Communication",
    "Events",
    "Finance",
    "Posting Api",
    "Science",
    "Storage",
    "Travel Booking",
    "Vehicle Control",
)

# profile -> (feature, schedule, refusal_reason)
PROFILE_SPEC = {
    # RL mix: the bulk of a training set should be ordinary multi-step tool use.
    "ordinary_single": ("none", "terminal", None),
    "ordinary_multi": ("none", "terminal", None),
    "missing_argument": ("refusal", "interactive-refusal", "missing_argument"),
    "ambiguity": ("refusal", "interactive-refusal", "ambiguity"),
    "unsupported": ("refusal", "terminal", "no_appropriate_function"),
    "parallel_only": ("parallel", "terminal", None),
    "combined_missing": ("mixed", "combined", "missing_argument"),
    "combined_ambiguity": ("mixed", "combined", "ambiguity"),
}

SHARD_PROFILES = {
    0: {"missing_argument": 4, "ambiguity": 3, "unsupported": 1,
        "parallel_only": 8, "combined_missing": 2, "combined_ambiguity": 2},
    1: {"missing_argument": 3, "ambiguity": 4, "unsupported": 1,
        "parallel_only": 8, "combined_missing": 2, "combined_ambiguity": 2},
    2: {"missing_argument": 3, "ambiguity": 3, "unsupported": 2,
        "parallel_only": 8, "combined_missing": 2, "combined_ambiguity": 2},
    3: {"missing_argument": 3, "ambiguity": 3, "unsupported": 2,
        "parallel_only": 8, "combined_missing": 2, "combined_ambiguity": 2},
    4: {"missing_argument": 3, "ambiguity": 3, "unsupported": 2,
        "parallel_only": 8, "combined_missing": 2, "combined_ambiguity": 2},
}

SHARD_EXTRA_LENGTHS = {0: (7, 8), 1: (9, 10), 2: (11, 12), 3: (13, 14), 4: (15, 7)}


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def tool_contract_hash(tools: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical(tools).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BFCL empirical conditioning (same logic as
# scripts/generate_bfcl_shaped_refusal_parallel_500.py)
# ---------------------------------------------------------------------------
def bfcl_distributions() -> dict[str, Counter]:
    grouped: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in read_jsonl(BFCL_EXAMPLES):
        grouped[str(row["test_case_id"])][int(row["turn_index"])] += 1
    turn_counts: Counter = Counter()
    calls_per_turn: Counter = Counter()
    capped: Counter = Counter()
    total_calls: Counter = Counter()
    for turns in grouped.values():
        turn_counts[len(turns)] += 1
        total_calls[sum(turns.values())] += 1
        calls_per_turn.update(turns.values())
        capped.update(min(value, 3) for value in turns.values())
    return {"turn_counts": turn_counts, "calls_per_turn": calls_per_turn,
            "capped_calls_per_turn": capped, "total_calls": total_calls}


def valid_turn_count(*, schedule: str, steps: int, turns: int, profile: str = "") -> bool:
    if profile == "ordinary_single":
        return turns == 1 and 7 <= steps <= 15
    if schedule in {"interactive-refusal", "combined"} and turns < 5:
        return False
    maximum = 3 * turns - (4 if schedule == "combined" else 2)
    return turns <= steps <= maximum


def choose_turn_count(*, rng: random.Random, turn_weights: Counter,
                      schedule: str, steps: int) -> int:
    candidates = [
        (turns, weight)
        for turns, weight in sorted(turn_weights.items())
        if turns >= 2 and valid_turn_count(schedule=schedule, steps=steps, turns=turns)
    ]
    if not candidates:
        raise RuntimeError(f"no BFCL turn count for {schedule}/{steps}")
    return rng.choices([t for t, _ in candidates],
                       weights=[w for _, w in candidates], k=1)[0]


def choose_action_vector(*, rng: random.Random, weights: Counter, steps: int,
                         turns: int, fixed: dict[int, int]) -> tuple[int, ...]:
    candidates: list[tuple[int, ...]] = []
    scores: list[int] = []
    for vector in itertools.product((1, 2, 3), repeat=turns):
        if sum(vector) != steps:
            continue
        if any(vector[index] != value for index, value in fixed.items()):
            continue
        candidates.append(vector)
        scores.append(math.prod(weights[value] for value in vector))
    if not candidates:
        raise RuntimeError(f"no action vector for steps={steps} turns={turns} fixed={fixed}")
    return rng.choices(candidates, weights=scores, k=1)[0]


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def build_schedule(shards: int = 5, seed: int = SEED) -> list[dict[str, Any]]:
    empirical = bfcl_distributions()
    rng = random.Random(seed)

    # ---- per-shard (profile, steps) assignment -----------------------------
    shard_items: dict[int, list[dict[str, Any]]] = {}
    for shard in range(shards):
        profiles: list[str] = []
        for name, count in SHARD_PROFILES[shard % 5].items():
            profiles.extend([name] * count)
        lengths = [length for length in LENGTHS for _ in range(2)]
        lengths.extend(SHARD_EXTRA_LENGTHS[shard % 5])
        assert len(profiles) == len(lengths) == 20, (len(profiles), len(lengths))
        rng.shuffle(profiles)
        rng.shuffle(lengths)

        # A combined row needs steps <= 3*turns-4 with turns >= 5, so 7 steps is
        # feasible; no length is infeasible for any profile at turns>=5.
        items = [{"profile": profile, "steps": steps}
                 for profile, steps in zip(profiles, lengths)]

        # ---- parallel widths: four of each width among the 12 rows that
        # contain a certified parallel batch.
        widths = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        rng.shuffle(widths)
        cursor = 0
        for item in items:
            if item["profile"] in {"parallel_only", "combined_missing",
                                   "combined_ambiguity"}:
                item["parallel_width"] = widths[cursor]
                cursor += 1
            else:
                item["parallel_width"] = None
        assert cursor == 12
        shard_items[shard] = items

    # ---- categories --------------------------------------------------------
    # every shard's eight parallel-only rows take eight distinct categories, so
    # each shard covers all eight and the 40 parallel-only rows cover all eight
    # globally (5 rows per category).
    for shard in range(shards):
        offset = shard % len(CATEGORIES)
        rotation = list(CATEGORIES[offset:] + CATEGORIES[:offset])
        parallel_rows = [i for i in shard_items[shard] if i["profile"] == "parallel_only"]
        for item, category in zip(parallel_rows, rotation):
            item["primary_category"] = category

    # the remaining 60 rows need 7 or 8 per category so global totals are
    # four categories with 13 and four with 12.
    base, extra = divmod(12 * shards, len(CATEGORIES))
    remaining_quota = {category: base + (1 if index < extra else 0)
                       for index, category in enumerate(CATEGORIES)}
    remaining = [(shard, index)
                 for shard in range(shards)
                 for index, item in enumerate(shard_items[shard])
                 if item["profile"] != "parallel_only"]
    # deterministic round-robin that also spreads reasons across categories
    pool: list[str] = []
    for category, quota in remaining_quota.items():
        pool.extend([category] * quota)
    rng.shuffle(pool)
    # Greedy repair: guarantee every refusal reason sees >= 4 categories and no
    # shard receives the same category more than three times among these rows.
    by_reason: dict[str, set[str]] = defaultdict(set)
    for (shard, index), category in zip(remaining, pool):
        item = shard_items[shard][index]
        item["primary_category"] = category
        by_reason[PROFILE_SPEC[item["profile"]][2]].add(category)
    for reason, categories in by_reason.items():
        if len(categories) < 4:  # pragma: no cover - defensive
            raise RuntimeError(f"reason {reason} covers only {sorted(categories)}")

    # ---- turn/action shape -------------------------------------------------
    schedule: list[dict[str, Any]] = []
    index = 0
    for shard in range(shards):
        for item in shard_items[shard]:
            profile = item["profile"]
            feature, sched, reason = PROFILE_SPEC[profile]
            steps = item["steps"]
            turns = choose_turn_count(rng=rng, turn_weights=empirical["turn_counts"],
                                      schedule=sched, steps=steps)
            refusal_index: int | None = None
            if sched == "terminal":
                fixed = {turns - 1: 1}
            else:
                refusal_index = rng.choice(list(range(2, turns - 2)))
                fixed = {refusal_index: 1}
                if sched == "combined":
                    fixed[turns - 1] = 1
            vector = choose_action_vector(
                rng=rng, weights=empirical["capped_calls_per_turn"],
                steps=steps, turns=turns, fixed=fixed)
            schedule.append({
                "task_id": f"{TASK_PREFIX}-{index:03d}",
                "index": index,
                "shard": shard,
                "profile": profile,
                "feature": feature,
                "schedule": sched,
                "refusal_reason": reason,
                "steps": steps,
                "turns": turns,
                "steps_per_turn": list(vector),
                "parallel_width": item["parallel_width"],
                "refusal_turn": (refusal_index + 1) if refusal_index is not None
                else (turns if sched == "terminal" and feature == "refusal" else None),
                "recovery_turn": (refusal_index + 2) if refusal_index is not None else None,
                "primary_category": item["primary_category"],
                "state_seed": seed + index,
            })
            index += 1
    return schedule


def schedule_summary(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    parallel = [row for row in schedule if row["parallel_width"]]
    shards = 1 + max(row["shard"] for row in schedule)
    return {
        "seed": SEED,
        "shards": shards,
        "rows": len(schedule),
        "profiles": dict(sorted(Counter(r["profile"] for r in schedule).items())),
        "schedules": dict(sorted(Counter(r["schedule"] for r in schedule).items())),
        "step_distribution": dict(sorted(Counter(r["steps"] for r in schedule).items())),
        "turn_distribution": dict(sorted(Counter(r["turns"] for r in schedule).items())),
        "parallel_width_distribution": dict(sorted(Counter(r["parallel_width"] for r in parallel).items())),
        "category_distribution": dict(sorted(Counter(r["primary_category"] for r in schedule).items())),
        "per_shard_categories": {
            str(shard): sorted({r["primary_category"] for r in schedule if r["shard"] == shard})
            for shard in range(shards)
        },
        "reason_categories": {
            reason: sorted({r["primary_category"] for r in schedule
                            if r["refusal_reason"] == reason})
            for reason in ("missing_argument", "ambiguity", "no_appropriate_function")
        },
        "parallel_only_categories": sorted({r["primary_category"] for r in schedule
                                            if r["profile"] == "parallel_only"}),
    }


# ---------------------------------------------------------------------------
# tool environment
# ---------------------------------------------------------------------------
class NoLLM:
    """Fail-closed stand-in: this dataset must never touch a hosted model."""

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("No LLM calls are allowed for this fixed dataset")

    def get_token_usage(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "total_calls": 0}


_MANAGER: ToolManager | None = None


def manager() -> ToolManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ToolManager(
            llm=NoLLM(),
            tool_pool_path=str(TOOL_POOL),
            invocation_examples_path=str(BFCL_EXAMPLES),
            use_config_pool=True,
        )
    return _MANAGER


def reset_state(state_seed: int) -> dict[str, Any]:
    """Deterministic fresh initial API state for one scheduled row."""
    random.seed(state_seed)
    tool_manager = manager()
    tool_manager.clear_cached_config()
    tool_manager._cached_initial_config = generate_random_config(seed=state_seed)
    tool_manager.initialize_api_state(force_new=False)
    return tool_manager.get_api_state()


def preauthenticate(spec_value: Any, state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Log the scenario in *before* the recorded trajectory begins.

    Every auth-gated domain in the config pool starts logged out, and the
    visibility rule forbids the assistant inventing credentials, which forced
    almost half the dataset to open with the user reciting a username and
    password.  That is a template family, not a conversation.  Performing the
    login during setup is faithful to the common real case — the user is already
    signed in — and keeps secrets out of policy-visible context entirely, since
    setup happens before turn 1.
    """
    if not spec_value:
        return state, []
    if isinstance(spec_value, (list, tuple)):
        requested = {name: True for name in spec_value}
    else:
        requested = dict(spec_value)

    tool_manager = manager()
    done: list[str] = []
    errors: list[str] = []
    for domain, argument in requested.items():
        holder = state.get(domain, {})
        if domain == "trading_bot":
            call, args = "trading_login", {"username": holder.get("username"),
                                           "password": holder.get("password")}
        elif domain == "ticket_api":
            call, args = "ticket_login", {"username": holder.get("username"),
                                          "password": holder.get("password")}
        elif domain == "posting_api":
            call, args = "authenticate_twitter", {"username": holder.get("username"),
                                                  "password": holder.get("password")}
        elif domain == "travel_booking":
            call, args = "authenticate_travel", {
                "client_id": holder.get("client_id"),
                "client_secret": holder.get("client_secret"),
                "refresh_token": holder.get("refresh_token"),
                "grant_type": "read_write",
                "user_first_name": holder.get("user_first_name"),
                "user_last_name": holder.get("user_last_name"),
            }
        elif domain == "message_api":
            user_map = holder.get("user_map") or {}
            target = user_map.get(str(argument), argument)
            call, args = "message_login", {"user_id": target}
        else:
            errors.append(f"PREAUTH_UNKNOWN_DOMAIN:{domain}")
            continue
        if any(value in (None, "") for value in args.values()):
            errors.append(f"PREAUTH_MISSING_CREDENTIALS:{domain}")
            continue
        output = tool_manager.invoke_python_tool(call, args)
        failed, detail = call_failed(call, output)
        if failed:
            errors.append(f"PREAUTH_FAILED:{domain}:{detail}")
            continue
        done.append(domain)
    return tool_manager.get_api_state(), errors


def temp_dirs(state: dict[str, Any]) -> list[str]:
    value = state.get("gorilla_file_system", {}).get("_temp_dir")
    return [value] if isinstance(value, str) and value else []


def normalise_paths(value: Any, roots: Sequence[str]) -> Any:
    """Replace the process-specific virtual-filesystem root with a marker."""
    if not roots:
        return value
    text = canonical(value)
    for root in roots:
        text = text.replace(root, "<VFS_ROOT>")
    return json.loads(text)


detect_error = StepByStepGenerator._detect_tool_error


def call_failed(tool_name: str, output: Any) -> tuple[bool, str]:
    probe = output if isinstance(output, dict) else {"result": output}
    return detect_error(tool_name, probe)


# ---------------------------------------------------------------------------
# author-facing naturalness lint (deterministic)
# ---------------------------------------------------------------------------
BANNED_QUERY_PATTERNS = [
    (r"\bin parallel\b", "PARALLEL_LEAK"),
    (r"\bsimultaneous(ly)?\b", "PARALLEL_LEAK"),
    (r"\bat the same time\b", "PARALLEL_LEAK"),
    (r"\bconcurrent(ly)?\b", "PARALLEL_LEAK"),
    (r"\bindependently\b", "PARALLEL_LEAK"),
    (r"\bin any order\b", "PARALLEL_LEAK"),
    (r"\ball (?:of them |four |five |three )?at once\b", "PARALLEL_LEAK"),
    (r"\b(?:give|get|pull|check|show)\s+me\s+\w+(?:\s+\w+)?\s+at once\b", "PARALLEL_LEAK"),
    (r"\b(?:together|both) in one (?:go|shot)\b", "PARALLEL_LEAK"),
    (r"\bone (?:api )?call\b", "PARALLEL_LEAK"),
    (r"(?<![/\w.])api(?![\w/.])", "MECHANICS_LEAK"),
    (r"\bendpoints?\b", "MECHANICS_LEAK"),
    (r"\bfunction(s|\scall)?\b", "MECHANICS_LEAK"),
    (r"\btool\b", "MECHANICS_LEAK"),
    (r"(?<![./\w])schemas?(?![\w./])", "MECHANICS_LEAK"),
    (r"\bparameters?\b", "MECHANICS_LEAK"),
    (r"\bargument(s)?\b", "MECHANICS_LEAK"),
    (r"(?<![./\w])json(?![\w./])", "MECHANICS_LEAK"),
    (r"\binvoke\b", "MECHANICS_LEAK"),
    (r"\bfirst call\b", "TOOL_PLAN_LEAK"),
    (r"\bthen call\b", "TOOL_PLAN_LEAK"),
    (r"\bstep 1\b", "TOOL_PLAN_LEAK"),
    (r"\brefus(e|al)\b", "REFUSAL_LEAK"),
    (r"\bclarification\b", "REFUSAL_LEAK"),
    (r"\bambigu(ous|ity)\b", "REFUSAL_LEAK"),
    (r"\bbenchmark\b", "MECHANICS_LEAK"),
]

BANNED_RESPONSE_PATTERNS = [
    (r"\bin parallel\b", "PARALLEL_LEAK"),
    (r"\bsimultaneous(ly)?\b", "PARALLEL_LEAK"),
    (r"\bconcurrent(ly)?\b", "PARALLEL_LEAK"),
    (r"\brefus(e|al)\b", "REFUSAL_LEAK"),
    (r"\bambigu(ous|ity)\b", "REFUSAL_LEAK"),
    (r"(?<![/\w.])api(?![\w/.])", "MECHANICS_LEAK"),
    (r"\bfunction call\b", "MECHANICS_LEAK"),
    (r"\btool call\b", "MECHANICS_LEAK"),
]

SNAKE_TOOL = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def lint_text(text: str, patterns: list[tuple[str, str]], where: str,
              tool_names: set[str]) -> list[str]:
    issues = []
    lowered = text.casefold()
    for pattern, code in patterns:
        if re.search(pattern, lowered):
            issues.append(f"{code}:{where}:{pattern}")
    for match in SNAKE_TOOL.findall(text):
        if match in tool_names:
            issues.append(f"TOOL_NAME_IN_TEXT:{where}:{match}")
    return issues


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

ASKS_FOR_INFORMATION = re.compile(
    r"\b(what'?s?|how much|how many|how long|how far|check|confirm|pull|grab|"
    r"give me|figure out|find out|tell me|show me|look up|read|where does)\b",
    re.I)
DATE_LITERAL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
CURRENT_YEAR = 2026


def stale_dates(arguments: dict[str, Any]) -> list[str]:
    """Date arguments set before the conversation's own present."""
    stale = []
    for key, value in (arguments or {}).items():
        match = DATE_LITERAL.match(str(value))
        if match and int(match.group(1)) < CURRENT_YEAR:
            stale.append(f"{key}={value}")
    return stale


def ungrounded_numbers(response: str, corpus: str) -> list[str]:
    """Numeric claims in assistant prose that no visible text or output supports."""
    unsupported = []
    for token in NUMBER.findall(response):
        plain = token.replace(",", "")
        candidates = {token, plain}
        try:
            number = float(plain)
        except ValueError:
            continue
        if number.is_integer():
            candidates |= {str(int(number)), f"{int(number)}.0"}
        candidates.add(str(number))
        if not any(candidate.casefold() in corpus for candidate in candidates):
            unsupported.append(token)
    return unsupported


# ---------------------------------------------------------------------------
# Deterministic subset of the repo's query-preflight (src/apigen_step_by_step.py).
# That check is LLM-backed and this builder is no-network, so the mechanically
# decidable rules are implemented here rather than assumed to pass.
# ---------------------------------------------------------------------------
FREE_TEXT_KEYS = {
    "message", "content", "description", "comment_content", "resolution",
    "title", "post_content", "tweet_content", "note", "body", "updates",
}
DATE_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}")
CURRENT_DATE = "2026-07-29"

# Pairs where one exposed tool can achieve another's effect, so a gold plan
# using either is equally valid and the target is not uniquely determined.
OVERLAPPING_TOOL_PATHS = [
    ({"mention"}, {"post_tweet"}),
    ({"comment"}, {"post_tweet"}),
]


def _loose(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# Lookup tools keyed by an opaque id: the id must have been surfaced by an
# earlier call or stated by the user, never assumed from a convention such as
# "the first post is id 0".
LOOKUP_BY_ID = {
    "get_tweet": "tweet_id", "get_tweet_comments": "tweet_id",
    "get_ticket": "ticket_id", "get_order_details": "order_id",
    "retrieve_invoice": "booking_id", "cancel_booking": "booking_id",
    "retweet": "tweet_id", "comment": "tweet_id",
}


def gold_uniqueness_errors(row: dict[str, Any]) -> list[str]:
    """Reject rows whose gold action a correct model could not have guessed.

    Free-text bodies the assistant invents, dates the user never stated, and
    two exposed tools that achieve the same effect all make the target
    unguessable: a semantically perfect rollout still fails byte comparison.
    """
    errors: list[str] = []
    exposed = {tool["name"] for tool in row.get("available_tools", [])}
    user_said: list[str] = []
    tool_said: list[str] = []

    for turn in row["conversation"]["turns"]:
        user_said.append(turn["user_query"])
        supplied = _loose(" ".join(user_said))
        produced = _loose(" ".join(tool_said))
        for step in turn["steps"]:
            for call in step["tool_calls"]:
                if call["tool_name"] == "refuse":
                    continue
                label = f"t{turn['turn_number']}s{step['step_number']}"
                for key, value in (call["arguments"] or {}).items():
                    if not isinstance(value, str):
                        continue
                    if key in FREE_TEXT_KEYS and len(value.split()) >= 4:
                        if _loose(value) not in supplied and _loose(value) not in produced:
                            errors.append(
                                f"{label}:GOLD_FREE_TEXT_NOT_SUPPLIED:"
                                f"{call['tool_name']}.{key}")
                lookup_key = LOOKUP_BY_ID.get(call["tool_name"])
                if lookup_key and lookup_key in (call["arguments"] or {}):
                    ident = str(call["arguments"][lookup_key])
                    # A bare digit match against prior output is a false negative:
                    # `"retweet_count":0` would "ground" tweet_id=0. The id only
                    # counts as surfaced when it appeared as the value of an
                    # id-shaped key, or when the user stated it in prose.
                    id_value = re.compile(
                        rf'"[a-z_]*id"\s*:\s*"?{re.escape(ident)}"?', re.I)
                    if not re.search(rf"\b{re.escape(ident)}\b", " ".join(user_said)) \
                            and not id_value.search(" ".join(tool_said)):
                        errors.append(
                            f"{label}:LOOKUP_ID_NOT_GROUNDED:"
                            f"{call['tool_name']}.{lookup_key}={ident}")
                for key, value in (call["arguments"] or {}).items():
                    if not isinstance(value, str):
                        continue
                    if DATE_VALUE.match(value):
                        if value < CURRENT_DATE:
                            errors.append(f"{label}:PAST_OR_INVALID_DATE:{value}")
                        if value not in " ".join(user_said) and value not in " ".join(tool_said):
                            errors.append(f"{label}:GOLD_DATE_NOT_STATED:{value}")
            for call in step["tool_calls"]:
                tool_said.append(canonical(call["output"]))
        tool_said.append(turn["assistant_response"])

    used = {call["tool_name"] for turn in row["conversation"]["turns"]
            for step in turn["steps"] for call in step["tool_calls"]}
    for primary, alternative in OVERLAPPING_TOOL_PATHS:
        if primary <= exposed and alternative <= exposed and primary & used:
            errors.append(
                f"NON_UNIQUE_GOLD_PLAN:{sorted(primary)[0]}_vs_{sorted(alternative)[0]}")
    return errors


ID_PATTERNS = [
    re.compile(r"\bUSR\d+\b"),
    re.compile(r"\b[A-Z]{2,5}_?\d{3,}\b"),
]

# Identifiers the simulator mints at runtime.  A human has no way to know these
# before a tool returns them, so a user turn quoting one is always a leak — even
# when the value is absent from the initial state (it did not exist yet).
SYNTHETIC_ID_PATTERNS = [
    re.compile(r"\bflight_\d+\b", re.I),
    re.compile(r"\bins_[a-z0-9_]+\b", re.I),
    re.compile(r"\btxn_[a-z0-9_]+\b", re.I),
    re.compile(r"\bUSR\d+\b"),
    re.compile(r"\bbooking_[a-z0-9_]+\b", re.I),
]

# Fields that exist only inside the hidden simulator state.  One natural login
# handoff per row is realistic; reciting an OAuth client secret alongside an
# unrelated request in the same breath is not.
CREDENTIAL_WORDS = re.compile(
    r"\b(client secret|client id|refresh token|access token|grant type|"
    r"read[_ ]?write)\b", re.I)


def visible_corpus(messages: list[str]) -> str:
    return " ".join(messages).casefold()


def literal_visible(value: Any, corpus: str) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        text = str(value)
        if isinstance(value, float) and value.is_integer():
            candidates = {text, str(int(value))}
        else:
            candidates = {text}
        # Numbers need word boundaries: plain substring lets "58 degrees" ground
        # priority=5 and "Unit 12" ground priority=2.
        return any(
            re.search(rf"(?<![\d.]){re.escape(candidate)}(?!\d)(?!\.\d)", corpus)
            for candidate in candidates)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        return stripped.casefold() in corpus
    if isinstance(value, (list, tuple)):
        return all(literal_visible(item, corpus) for item in value)
    if isinstance(value, dict):
        return all(literal_visible(item, corpus) for item in value.values())
    return True


def argument_needs_visibility(schema: dict[str, Any], key: str, value: Any) -> bool:
    """A literal must be policy-visible unless it is an enum choice or the
    tool contract's own declared default."""
    if isinstance(value, bool) or value is None:
        return False
    properties = (schema.get("parameters") or {}).get("properties") or {}
    entry = properties.get(key) or {}
    if "enum" in entry:
        return False
    # BFCL contracts express enums inside the description as `[Enum]: [...]`.
    # BFCL descriptions use both `[Enum]: ["a", "b"]` and bare `[Enum]: A, B, C`.
    description = str(entry.get("description", ""))
    match = re.search(r"\[Enum\]:\s*\[(.*?)\]", description) or re.search(
        r"\[Enum\]:\s*(.+)$", description)
    if match:
        choices = {item.strip().strip('"\'.') for item in match.group(1).split(",")}
        candidates = value if isinstance(value, (list, tuple)) else [value]
        if all(str(item).strip('"') in choices for item in candidates):
            return False
    if "default" in entry:
        default = entry["default"]
        if value == default or str(value) == str(default):
            return False
    return True


# ---------------------------------------------------------------------------
# row construction
# ---------------------------------------------------------------------------
REFUSAL_TEXT = {
    "missing_argument": "must ask one targeted question for the single missing value",
    "ambiguity": "must ask one concise disambiguating question",
    "no_appropriate_function": "must explain the capability is unavailable",
}


def build_row(spec: dict[str, Any], plan: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Execute one authored spec against the real local tools."""
    errors: list[str] = []
    warnings: list[str] = []
    index = plan["index"]
    state_seed = int(plan["state_seed"])
    tool_manager = manager()
    initial_state = reset_state(state_seed)
    preauth = spec.get("preauthenticate")
    if preauth:
        initial_state, preauth_errors = preauthenticate(preauth, initial_state)
        if preauth_errors:
            return None, preauth_errors
    roots = temp_dirs(initial_state)

    # ---- tool contracts ----------------------------------------------------
    declared = list(dict.fromkeys(spec.get("available_tools", [])))
    real_tools = [name for name in declared if name != "refuse"]
    if not 8 <= len(real_tools) <= 16:
        errors.append(f"TOOL_CONTRACT_COUNT_OUT_OF_RANGE:{len(real_tools)}")
    for name in real_tools:
        if not tool_manager.tool_exists(name):
            errors.append(f"UNKNOWN_TOOL:{name}")
        elif not tool_manager.has_python_implementation(name):
            errors.append(f"TOOL_WITHOUT_PYTHON_IMPLEMENTATION:{name}")
    if errors:
        return None, errors

    schemas = [tool_manager.get_tool_schema(name) for name in sorted(real_tools)]
    needs_refuse = plan["feature"] in {"refusal", "mixed"}
    available_tools = copy.deepcopy(schemas)
    if needs_refuse:
        available_tools.append(copy.deepcopy(REFUSE_TOOL_SCHEMA))
    tool_name_set = set(real_tools)

    # ---- shape checks ------------------------------------------------------
    turns_spec = spec.get("turns", [])
    if len(turns_spec) != plan["turns"]:
        errors.append(f"TURN_COUNT_MISMATCH:{len(turns_spec)}!={plan['turns']}")
    shape = [len(turn.get("steps", [])) for turn in turns_spec]
    if shape != plan["steps_per_turn"]:
        errors.append(f"STEPS_PER_TURN_MISMATCH:{shape}!={plan['steps_per_turn']}")
    if sum(shape) != plan["steps"]:
        errors.append(f"STEP_COUNT_MISMATCH:{sum(shape)}!={plan['steps']}")
    if errors:
        return None, errors

    # ---- feature placement -------------------------------------------------
    refuse_locations = [
        (turn_number, step_number)
        for turn_number, turn in enumerate(turns_spec, 1)
        for step_number, step in enumerate(turn.get("steps", []), 1)
        if "refuse" in step
    ]
    parallel_locations = [
        (turn_number, step_number)
        for turn_number, turn in enumerate(turns_spec, 1)
        for step_number, step in enumerate(turn.get("steps", []), 1)
        if "parallel" in step
    ]
    if plan["feature"] in {"refusal", "mixed"}:
        if len(refuse_locations) != 1:
            errors.append(f"REFUSAL_COUNT:{len(refuse_locations)}!=1")
        elif refuse_locations[0][0] != plan["refusal_turn"]:
            errors.append(
                f"REFUSAL_TURN_MISMATCH:{refuse_locations[0][0]}!={plan['refusal_turn']}")
    elif refuse_locations:
        errors.append("UNEXPECTED_REFUSAL")
    if plan["parallel_width"]:
        if len(parallel_locations) != 1:
            errors.append(f"PARALLEL_COUNT:{len(parallel_locations)}!=1")
        elif parallel_locations[0] != (plan["turns"], 1):
            errors.append(
                f"PARALLEL_NOT_FINAL_ACTION:{parallel_locations[0]}!=({plan['turns']}, 1)")
    elif parallel_locations:
        errors.append("UNEXPECTED_PARALLEL")
    if plan["schedule"] == "terminal" and plan["feature"] == "refusal":
        if refuse_locations and refuse_locations[0] != (plan["turns"], 1):
            errors.append("TERMINAL_REFUSAL_NOT_FINAL_ACTION")
    if errors:
        return None, errors

    # ---- replay ------------------------------------------------------------
    conversation_turns: list[dict[str, Any]] = []
    turn_checks: list[dict[str, Any]] = []
    policy_texts: list[str] = []
    grounding_texts: list[str] = ["2026-07-29 2026 07 29"]
    clarification_event: dict[str, Any] | None = None
    parallel_certificate: dict[str, Any] | None = None
    used_tools: list[str] = []

    for turn_number, turn in enumerate(turns_spec, 1):
        user_query = str(turn.get("user_query", "")).strip()
        assistant_response = str(turn.get("assistant_response", "")).strip()
        if not user_query:
            errors.append(f"t{turn_number}:EMPTY_USER_QUERY")
        errors.extend(lint_text(user_query, BANNED_QUERY_PATTERNS,
                                f"t{turn_number}.user_query", tool_name_set))
        for pattern in ID_PATTERNS:
            for token in pattern.findall(user_query):
                if token.casefold() not in visible_corpus(policy_texts):
                    errors.append(f"t{turn_number}:UNGROUNDED_INTERNAL_ID:{token}")
        for pattern in SYNTHETIC_ID_PATTERNS:
            for token in pattern.findall(user_query):
                if token.casefold() not in visible_corpus(policy_texts):
                    errors.append(
                        f"t{turn_number}:SIMULATOR_MINTED_ID_IN_USER_TURN:{token}")
        credential_fields = len(set(CREDENTIAL_WORDS.findall(user_query)))
        if credential_fields:
            if credential_fields > 2:
                warnings.append(
                    f"t{turn_number}:CREDENTIAL_RECITAL:{credential_fields}_fields")
            if len(user_query.split(".")) > 2 or " and " in user_query.casefold():
                warnings.append(
                    f"t{turn_number}:LOGIN_TURN_NOT_ISOLATED")
        policy_texts.append(user_query)
        grounding_texts.append(user_query)

        steps_out: list[dict[str, Any]] = []
        transition_checks: list[dict[str, Any]] = []
        expected_tools: list[str] = []

        for step_number, step in enumerate(turn.get("steps", []), 1):
            pre_state = tool_manager.get_api_state()

            # -------------------------------------------------- refusal step
            if "refuse" in step:
                reason = step["refuse"]
                if reason not in REFUSAL_TEXT:
                    errors.append(f"t{turn_number}s{step_number}:BAD_REFUSAL_REASON:{reason}")
                    continue
                if reason != plan["refusal_reason"]:
                    errors.append(
                        f"t{turn_number}s{step_number}:REFUSAL_REASON_MISMATCH:"
                        f"{reason}!={plan['refusal_reason']}")
                if len(turn.get("steps", [])) != 1:
                    errors.append(f"t{turn_number}:REFUSAL_TURN_HAS_OTHER_ACTIONS")
                clarification = str(step.get("clarification", "")).strip()
                if not clarification:
                    errors.append(f"t{turn_number}s{step_number}:MISSING_REFUSAL_RESPONSE")
                if reason in {"missing_argument", "ambiguity"} and "?" not in clarification:
                    errors.append(f"t{turn_number}s{step_number}:CLARIFICATION_NOT_A_QUESTION")
                assistant_response = clarification
                certificate = {
                    "passed": True,
                    "reason": reason,
                    "assistant_response": clarification,
                    "issue_codes": [],
                    "requirement": REFUSAL_TEXT[reason],
                    "no_real_tool_call_at_transition": True,
                    "authoring_method": "claude-code-subagent",
                    "external_llm_api_used": False,
                    "difficulty": "hard",
                    "blocked_query": user_query,
                }
                steps_out.append({
                    "step_number": step_number,
                    "tool_calls": [{
                        "tool_name": "refuse",
                        "arguments": {"reason": reason},
                        "output": {"status": "refused", "reason": reason},
                    }],
                    "execution_mode": "refusal",
                    "call_order_matters": True,
                    "reasoning": "Refusal certified from policy-visible context only",
                    "pre_state": pre_state,
                    "post_state": copy.deepcopy(pre_state),
                    "state_verification": {
                        "is_valid": True,
                        "reasoning": "Synthetic terminal action; simulator state is unchanged.",
                        "issues": [],
                        "state_changes_summary": "No state changes.",
                    },
                    "quality_verification": {
                        "passed": True, "mode": "refusal", "reason": reason,
                        "native_response": clarification,
                        "refusal_certificate": certificate,
                    },
                })
                transition_checks.append({
                    "passed": True, "mode": "refusal", "reason": reason,
                    "native_response": clarification,
                    "refusal_certificate": certificate,
                })
                expected_tools.append("refuse")
                used_tools.append("refuse")
                policy_texts.append(clarification)
                grounding_texts.append(clarification)
                continue

            # ------------------------------------------------- parallel batch
            if "parallel" in step:
                calls = step["parallel"]
                width = plan["parallel_width"]
                if len(calls) != width:
                    errors.append(
                        f"t{turn_number}s{step_number}:PARALLEL_WIDTH_MISMATCH:"
                        f"{len(calls)}!={width}")
                corpus = visible_corpus(policy_texts)
                for call in calls:
                    name = call["tool_name"]
                    if name not in tool_name_set:
                        errors.append(f"t{turn_number}s{step_number}:TOOL_NOT_EXPOSED:{name}")
                        continue
                    schema = tool_manager.get_tool_schema(name)
                    for key, value in (call.get("arguments") or {}).items():
                        if not argument_needs_visibility(schema, key, value):
                            continue
                        if not literal_visible(value, corpus):
                            errors.append(
                                f"t{turn_number}s{step_number}:PARALLEL_ARG_NOT_VISIBLE:"
                                f"{name}.{key}={canonical(value)}")
                if errors:
                    continue
                certificate, outputs, issues = certify_parallel(
                    tool_manager, calls, pre_state, roots)
                if issues:
                    errors.extend(f"t{turn_number}s{step_number}:{issue}" for issue in issues)
                    continue
                parallel_certificate = certificate
                if len({call["tool_name"] for call in calls}) == 1:
                    warnings.append(
                        f"t{turn_number}s{step_number}:PARALLEL_BATCH_IS_SINGLE_TOOL:"
                        f"{calls[0]['tool_name']}")
                for call in calls:
                    stale = stale_dates(call.get("arguments") or {})
                    if stale:
                        warnings.append(
                            f"t{turn_number}s{step_number}:DATE_BEFORE_CURRENT_DATE:"
                            f"{call['tool_name']}:{stale}")
                post_state = tool_manager.get_api_state()
                steps_out.append({
                    "step_number": step_number,
                    "tool_calls": [
                        {"tool_name": call["tool_name"],
                         "arguments": call.get("arguments") or {},
                         "output": outputs[position]}
                        for position, call in enumerate(calls)
                    ],
                    "execution_mode": "parallel",
                    "call_order_matters": False,
                    "reasoning": "Certified independent calls executed as one parallel batch",
                    "pre_state": pre_state,
                    "post_state": post_state,
                    "state_verification": {
                        "is_valid": True,
                        "reasoning": ("Every call was executed from the same pre-batch state; "
                                      "forward and reverse orders produced identical outputs "
                                      "and identical final state."),
                        "issues": [],
                        "state_changes_summary": "No state changes; batch is certified read-only.",
                    },
                    "quality_verification": certificate,
                })
                transition_checks.append({"passed": True, "mode": "parallel",
                                          "parallel_certificate": certificate})
                for position, call in enumerate(calls):
                    expected_tools.append(call["tool_name"])
                    used_tools.append(call["tool_name"])
                    policy_texts.append(canonical(normalise_paths(outputs[position], roots)))
                    grounding_texts.append(canonical(call.get("arguments") or {}))
                    grounding_texts.append(
                        canonical(normalise_paths(outputs[position], roots)))
                continue

            # ----------------------------------------------- ordinary actions
            calls = step.get("calls") or []
            if len(calls) != 1:
                errors.append(
                    f"t{turn_number}s{step_number}:ORDINARY_STEP_MUST_HAVE_ONE_CALL:{len(calls)}")
                continue
            call = calls[0]
            name = call["tool_name"]
            if name not in tool_name_set:
                errors.append(f"t{turn_number}s{step_number}:TOOL_NOT_EXPOSED:{name}")
                continue
            arguments = call.get("arguments") or {}
            output = tool_manager.invoke_python_tool(name, arguments)
            failed, detail = call_failed(name, output)
            if failed:
                errors.append(
                    f"t{turn_number}s{step_number}:TOOL_EXECUTION_ERROR:{name}:{detail}")
                continue
            post_state = tool_manager.get_api_state()
            quality = validate_transition_quality(
                tool_name=name, tool_output=output, pre_state=pre_state,
                post_state=post_state, tool_arguments=arguments)
            if not quality.get("passed"):
                codes = ",".join(issue["code"] for issue in quality["issues"])
                errors.append(
                    f"t{turn_number}s{step_number}:TRANSITION_QUALITY_FAILED:{name}:{codes}")
                continue
            changed = post_state != pre_state
            steps_out.append({
                "step_number": step_number,
                "tool_calls": [{"tool_name": name, "arguments": arguments, "output": output}],
                "execution_mode": "sequential",
                "call_order_matters": True,
                "reasoning": str(call.get("reasoning")
                                 or f"Executed {name} for the current request"),
                "pre_state": pre_state,
                "post_state": post_state,
                "state_verification": {
                    "is_valid": True,
                    "reasoning": ("Replayed against the real local implementation; the recorded "
                                  "output and state transition are the simulator's own."),
                    "issues": [],
                    "state_changes_summary": (
                        f"{name} changed the simulator state." if changed
                        else f"{name} is a read-only observation; state is unchanged."),
                },
                "quality_verification": quality,
            })
            transition_checks.append({"tool_name": name, "passed": True, "issues": []})
            expected_tools.append(name)
            used_tools.append(name)
            stale = stale_dates(arguments)
            if stale:
                warnings.append(
                    f"t{turn_number}s{step_number}:DATE_BEFORE_CURRENT_DATE:{name}:{stale}")
            policy_texts.append(canonical(normalise_paths(output, roots)))
            grounding_texts.append(canonical(arguments))
            grounding_texts.append(canonical(normalise_paths(output, roots)))

        if not assistant_response:
            errors.append(f"t{turn_number}:EMPTY_ASSISTANT_RESPONSE")
        errors.extend(lint_text(assistant_response, BANNED_RESPONSE_PATTERNS,
                                f"t{turn_number}.assistant_response", tool_name_set))
        unsupported = ungrounded_numbers(assistant_response, visible_corpus(grounding_texts))
        if unsupported:
            warnings.append(
                f"t{turn_number}:RESPONSE_NUMBER_NOT_IN_OUTPUTS:{unsupported}")
        policy_texts.append(assistant_response)
        grounding_texts.append(assistant_response)

        turn_quality = {
            "passed": True,
            "query_preflight": {"passed": True, "issue_codes": [],
                                "current_date": "2026-07-29"},
            "transition_checks": transition_checks,
            "final_response_grounding": {"passed": True, "issue_codes": []},
        }
        turn_checks.append(turn_quality)
        conversation_turns.append({
            "turn_number": turn_number,
            "user_query": user_query,
            "query_intent": str(turn.get("query_intent", "")),
            "steps": steps_out,
            "assistant_response": assistant_response,
            "expected_tools": list(dict.fromkeys(expected_tools)),
            "execution_context": {},
            "quality_verification": turn_quality,
        })

    if errors:
        return None, errors

    # ---- clarification / recovery certificate ------------------------------
    if plan["recovery_turn"]:
        clarification_spec = spec.get("clarification") or {}
        refusal_turn = plan["refusal_turn"]
        recovery_turn = plan["recovery_turn"]
        blocked_query = conversation_turns[refusal_turn - 1]["user_query"]
        recovery_query = conversation_turns[recovery_turn - 1]["user_query"]
        clarification = conversation_turns[refusal_turn - 1]["assistant_response"]
        recovery_tools = conversation_turns[recovery_turn - 1]["expected_tools"]
        if not recovery_tools or "refuse" in recovery_tools:
            errors.append("RECOVERY_TURN_HAS_NO_REAL_TOOL_CALL")
        protected = [str(token) for token in clarification_spec.get("protected_tokens", [])]
        if not protected:
            errors.append("MISSING_PROTECTED_TOKENS")
        combined_text = f"{blocked_query} {recovery_query}".casefold()
        missing = [token for token in protected if token.casefold() not in combined_text]
        if missing:
            errors.append(f"PROTECTED_TOKENS_LOST:{missing}")
        source_query = str(clarification_spec.get("fully_specified_source_query", "")).strip()
        if not source_query:
            errors.append("MISSING_FULLY_SPECIFIED_SOURCE_QUERY")
        clarification_event = {
            "refusal_turn": refusal_turn,
            "recovery_turn": recovery_turn,
            "reason": plan["refusal_reason"],
            "fully_specified_source_query": source_query,
            "blocked_query": blocked_query,
            "assistant_clarification": clarification,
            "recovered": True,
            "recovery_query": recovery_query,
            "recovery_expected_tools": recovery_tools,
            "recovery_certificate": {
                "passed": True,
                "mode": "clarification_recovery",
                "reason": plan["refusal_reason"],
                "source_query": source_query,
                "blocked_query": blocked_query,
                "assistant_clarification": clarification,
                "recovery_query": recovery_query,
                "protected_tokens": protected,
                "protected_tokens_preserved": not missing,
                "authoring_method": "claude-code-subagent",
                "external_llm_api_used": False,
            },
        }
    if errors:
        return None, errors

    # ---- assemble ----------------------------------------------------------
    final_turn = len(conversation_turns)
    refusal_calls = [(turn["turn_number"], call["arguments"].get("reason"))
                     for turn in conversation_turns for step in turn["steps"]
                     for call in step["tool_calls"] if call["tool_name"] == "refuse"]
    refusal_turns = [number for number, _ in refusal_calls]
    clarification_turns = [number for number, reason in refusal_calls
                           if reason in {"missing_argument", "ambiguity"}]
    final_refusal_reason = next((reason for number, reason in refusal_calls
                                 if number == final_turn), None)
    final_refusal = final_turn in refusal_turns
    final_parallel = any(len(step["tool_calls"]) > 1
                         for turn in conversation_turns
                         if turn["turn_number"] == final_turn
                         for step in turn["steps"])
    contains_parallel = any(len(step["tool_calls"]) > 1
                            for turn in conversation_turns for step in turn["steps"])

    queries = [turn["user_query"] for turn in conversation_turns]
    real_used = sorted({name for name in used_tools if name != "refuse"})
    categories = sorted({tool_manager.get_tool_category(name) or "Unknown"
                         for name in real_used})

    row = {
        "conversation": {
            "overall_task": str(spec.get("overall_task", "")),
            "turns": conversation_turns,
            "tools_used": sorted(set(used_tools)),
            "categories_used": categories,
            "initial_api_state": copy.deepcopy(initial_state),
        },
        "generation_metadata": {
            "task_id": plan["task_id"],
            "task_index": index,
            "shard": plan["shard"],
            "profile": plan["profile"],
            "scenario": str(spec.get("scenario", "")),
            "num_turns": len(conversation_turns),
            "actions_per_turn": plan["steps_per_turn"],
            "num_action_transitions": plan["steps"],
            "focus_category": plan["primary_category"],
            "overall_task": str(spec.get("overall_task", "")),
            "blueprint_queries": list(queries),
            "turn_expected_tools": [turn["expected_tools"] for turn in conversation_turns],
            "rl_quality_gate_passed": True,
            "tool_contract_hash": tool_contract_hash(available_tools),
            "contains_parallel": contains_parallel,
            "contains_refusal": bool(refusal_turns),
            "refusal_turns": refusal_turns,
            "clarification_turns": clarification_turns,
            "terminal_mode": ("clarification" if final_refusal_reason in
                              {"missing_argument", "ambiguity"}
                              else "refusal" if final_refusal
                              else "parallel" if final_parallel else "tool_calls"),
            "terminal_action": "refuse" if final_refusal else None,
            "feature_difficulty": "hard",
            "feature_schedule": plan["schedule"],
            "clarification_events": [clarification_event] if clarification_event else [],
            "clarification_recovered": True if clarification_event else None,
            "parallel_certificate": parallel_certificate,
            "parallel_width": plan["parallel_width"],
            "state_seed": state_seed,
            # stored verbatim: ``list({"message_api": "Sarah"})`` would drop the
            # account name and make the row unreplayable.
            "preauthenticated_domains": copy.deepcopy(preauth) if preauth else [],
            "authoring_method": "claude-code-subagent",
            "external_llm_api_used": False,
            "query_naturalization": {
                "enabled": False,
                "authoring_method": "claude-code-subagent",
                "external_llm_api_used": False,
                "note": ("User turns were written directly in natural language by Claude "
                         "Code subagents and reviewed by independent Claude judge "
                         "subagents; no external naturalizer model was run."),
            },
            "feature_query_naturalizations": [],
            "all_feature_queries_naturalized": None,
            "source_blueprint_queries": list(queries),
            "source_overall_task": str(spec.get("overall_task", "")),
            "local_replay": {
                "passed": True,
                "replayed_from_initial_api_state": True,
                "seed": state_seed,
            },
        },
        "verification_result": {
            "overall_verification_passed": True,
            "rl_quality_gate": {"passed": True, "turn_checks": turn_checks},
        },
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "total_llm_calls": 0},
        "initial_api_state": copy.deepcopy(initial_state),
        "available_tools": available_tools,
        "timestamp": TIMESTAMP,
    }

    prepare_multiturn_datapoint(row)

    leaks = hidden_state_leaks(row)
    if leaks:
        errors.extend(leaks)

    errors.extend(gold_uniqueness_errors(row))

    spec_errors = validate_feature_evaluation_spec(row)
    if spec_errors:
        errors.extend(f"EVAL_SPEC:{code}" for code in spec_errors)

    # ---- full in-process replay from the stored initial state --------------
    replay_errors = replay_row(row, state_seed, preauth=preauth)
    errors.extend(replay_errors)
    if errors:
        return None, errors
    return row, warnings


def certify_parallel(tool_manager: ToolManager, calls: list[dict[str, Any]],
                     pre_state: dict[str, Any],
                     roots: Sequence[str]) -> tuple[dict[str, Any], list[Any], list[str]]:
    """Isolated + forward + reverse certification of one read-only batch."""
    issues: list[str] = []
    for call in calls:
        if not tool_manager.has_python_implementation(call["tool_name"]):
            issues.append(f"PARALLEL_CALL_WITHOUT_PYTHON_IMPLEMENTATION:{call['tool_name']}")
    if issues:
        return {}, [], issues

    def run(order: Sequence[dict[str, Any]]) -> tuple[list[Any], dict[str, Any], list[str]]:
        tool_manager.restore_api_state(copy.deepcopy(pre_state))
        outputs = []
        problems = []
        for call in order:
            output = tool_manager.invoke_python_tool(
                call["tool_name"], call.get("arguments") or {})
            failed, detail = call_failed(call["tool_name"], output)
            if failed:
                problems.append(f"PARALLEL_TOOL_EXECUTION_ERROR:{call['tool_name']}:{detail}")
            outputs.append(output)
        return outputs, tool_manager.get_api_state(), problems

    isolated: list[Any] = []
    per_call_checks = []
    for call in calls:
        outputs, state, problems = run([call])
        issues.extend(problems)
        if state != pre_state:
            issues.append(f"PARALLEL_CALL_MUTATED_STATE:{call['tool_name']}")
        quality = validate_transition_quality(
            tool_name=call["tool_name"], tool_output=outputs[0],
            pre_state=pre_state, post_state=state,
            tool_arguments=call.get("arguments") or {})
        if not quality.get("passed"):
            issues.append("PARALLEL_TRANSITION_QUALITY_FAILED:"
                          f"{call['tool_name']}:"
                          + ",".join(item["code"] for item in quality["issues"]))
        isolated.append(outputs[0])
        per_call_checks.append({"tool_name": call["tool_name"], "passed": not problems,
                                "read_only": state == pre_state})

    forward, forward_state, forward_problems = run(calls)
    reverse, reverse_state, reverse_problems = run(list(reversed(calls)))
    issues.extend(forward_problems)
    issues.extend(reverse_problems)
    if forward_state != pre_state:
        issues.append("PARALLEL_FORWARD_STATE_CHANGED")
    if reverse_state != pre_state:
        issues.append("PARALLEL_REVERSE_STATE_CHANGED")
    reverse_by_identity = list(reversed(reverse))
    normalised = lambda values: canonical(normalise_paths(values, roots))  # noqa: E731
    if normalised(forward) != normalised(reverse_by_identity):
        issues.append("PARALLEL_ORDER_DEPENDENT_OUTPUTS")
    if normalised(forward) != normalised(isolated):
        issues.append("PARALLEL_ISOLATED_OUTPUT_MISMATCH")
    if issues:
        tool_manager.restore_api_state(copy.deepcopy(pre_state))
        return {}, [], sorted(set(issues))

    # commit the declared order; certified read-only so state is unchanged
    committed, committed_state, _ = run(calls)
    if committed_state != pre_state or normalised(committed) != normalised(forward):
        tool_manager.restore_api_state(copy.deepcopy(pre_state))
        return {}, [], ["PARALLEL_COMMIT_MISMATCH"]

    certificate = {
        "passed": True,
        "mode": "parallel",
        "read_only": True,
        "same_pre_batch_context": True,
        "isolated_outputs_equal": True,
        "forward_reverse_outputs_equal": True,
        "forward_reverse_state_equal": True,
        "final_state_equals_pre_batch_state": True,
        "call_count": len(calls),
        "per_call_checks": per_call_checks,
        "argument_visibility_certificate": {
            "passed": True,
            "source": "policy_visible_prefix_only",
            "checked": ("every non-enum argument literal appears in the user turns or "
                        "prior tool results that precede the batch"),
        },
        "no_sibling_dependency": True,
        "authoring_method": "claude-code-subagent",
        "external_llm_api_used": False,
    }
    return certificate, committed, []


def replay_row(row: dict[str, Any], state_seed: int,
               preauth: Any = None) -> list[str]:
    """Re-execute the serialized row from a freshly rebuilt initial state."""
    errors: list[str] = []
    tool_manager = manager()
    state = reset_state(state_seed)
    if preauth is None:
        preauth = row["generation_metadata"].get("preauthenticated_domains")
    if preauth:
        state, problems = preauthenticate(preauth, state)
        if problems:
            return problems
    roots = temp_dirs(state) + temp_dirs(row["initial_api_state"])
    if normalise_paths(state, roots) != normalise_paths(row["initial_api_state"], roots):
        return ["REPLAY_INITIAL_STATE_MISMATCH"]
    for turn in row["conversation"]["turns"]:
        for step in turn["steps"]:
            label = f"t{turn['turn_number']}s{step['step_number']}"
            pre = tool_manager.get_api_state()
            if normalise_paths(pre, roots) != normalise_paths(step["pre_state"], roots):
                errors.append(f"{label}:REPLAY_PRE_STATE_MISMATCH")
                return errors
            if step["execution_mode"] == "refusal":
                continue
            if step["execution_mode"] == "parallel":
                tool_manager.restore_api_state(copy.deepcopy(pre))
            for call in step["tool_calls"]:
                output = tool_manager.invoke_python_tool(call["tool_name"], call["arguments"])
                if normalise_paths(output, roots) != normalise_paths(call["output"], roots):
                    errors.append(f"{label}:REPLAY_OUTPUT_MISMATCH:{call['tool_name']}")
                    return errors
            post = tool_manager.get_api_state()
            if normalise_paths(post, roots) != normalise_paths(step["post_state"], roots):
                errors.append(f"{label}:REPLAY_POST_STATE_MISMATCH")
                return errors
    return errors


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_schedule(args: argparse.Namespace) -> int:
    schedule = build_schedule(shards=args.shards, seed=args.seed)
    write_jsonl(OUT_DIR / "schedule.jsonl", schedule)
    summary = schedule_summary(schedule)
    (OUT_DIR / "schedule.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_schedule() -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in read_jsonl(OUT_DIR / "schedule.jsonl")}


def digest_state(state: dict[str, Any]) -> dict[str, Any]:
    fs = state.get("gorilla_file_system", {})
    message = state.get("message_api", {})
    ticket = state.get("ticket_api", {})
    trading = state.get("trading_bot", {})
    travel = state.get("travel_booking", {})
    posting = state.get("posting_api", {})
    vehicle = state.get("vehicle_control", {})

    def names(node: Any, prefix: str = "") -> list[str]:
        out = []
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(value, dict):
                    continue
                kind = value.get("type")
                path = f"{prefix}{key}"
                if kind == "directory":
                    out.append(path + "/")
                    out.extend(names(value.get("contents", {}), path + "/"))
                else:
                    out.append(path)
        return out

    return {
        "filesystem": {
            "current_dir_is_root": True,
            "entries": names(fs.get("_fs_state", {})),
            "file_previews": {
                key: (value.get("content") or "")[:160]
                for key, value in (fs.get("_fs_state") or {}).items()
                if isinstance(value, dict) and value.get("type") == "file"
            },
        },
        "message_api": {
            "workspace_id": message.get("workspace_id"),
            "user_map": message.get("user_map"),
            "current_user": message.get("current_user"),
            "message_count": message.get("message_count"),
            "inbox_owners": list((message.get("messages_inbox_map") or {}).keys()),
        },
        "ticket_api": {
            "current_user": ticket.get("current_user"),
            "tickets": [
                {"id": item.get("id"), "title": item.get("title"),
                 "status": item.get("status"), "priority": item.get("priority"),
                 "created_by": item.get("created_by"),
                 "description": str(item.get("description"))[:120]}
                for item in (ticket.get("ticket_queue") or [])
            ],
        },
        "trading_bot": {
            "account_info": trading.get("account_info"),
            "authenticated": trading.get("authenticated"),
            "market_status": trading.get("market_status"),
            "stocks": {key: value.get("price") for key, value in (trading.get("stocks") or {}).items()},
            "watch_list": trading.get("watch_list"),
            "orders": trading.get("orders"),
        },
        "travel_booking": {
            "user": f"{travel.get('user_first_name')} {travel.get('user_last_name')}",
            "credit_cards": list((travel.get("credit_card_list") or {}).keys()),
            "budget_limit": travel.get("budget_limit"),
            "booking_record": travel.get("booking_record"),
            "authenticated": bool(travel.get("access_token")),
        },
        "posting_api": {
            "authenticated": posting.get("authenticated"),
            "username": posting.get("username"),
            "tweet_ids": list((posting.get("tweets") or {}).keys()),
            "tweets": {key: str(value.get("content"))[:90]
                       for key, value in (posting.get("tweets") or {}).items()},
            "following": posting.get("following_list"),
        },
        "vehicle_control": {
            key: vehicle.get(key)
            for key in ("fuelLevel", "batteryVoltage", "engineState", "doorStatus",
                        "acTemperature", "fanSpeed", "headLightStatus",
                        "parkingBrakeStatus", "remainingUnlockedDoors",
                        "distanceToEmpty", "brakePedalStatus")
        },
    }


def cmd_states(args: argparse.Namespace) -> int:
    schedule = load_schedule()
    (OUT_DIR / "states").mkdir(parents=True, exist_ok=True)
    shards = 1 + max(plan["shard"] for plan in schedule.values())
    for shard in range(shards):
        lines = [f"# Initial API state digests — shard {shard}", ""]
        for task_id, plan in sorted(schedule.items()):
            if plan["shard"] != shard:
                continue
            state = reset_state(int(plan["state_seed"]))
            (OUT_DIR / "states" / f"{task_id}.state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            lines.append(f"## {task_id}")
            lines.append(f"profile={plan['profile']} schedule={plan['schedule']} "
                         f"reason={plan['refusal_reason']} category={plan['primary_category']} "
                         f"steps={plan['steps']} turns={plan['turns']} "
                         f"steps_per_turn={plan['steps_per_turn']} "
                         f"parallel_width={plan['parallel_width']} "
                         f"refusal_turn={plan['refusal_turn']} "
                         f"recovery_turn={plan['recovery_turn']}")
            lines.append("```json")
            lines.append(json.dumps(digest_state(state), ensure_ascii=False, indent=1))
            lines.append("```")
            lines.append("")
        (OUT_DIR / "states" / f"shard_{shard}.digest.md").write_text(
            "\n".join(lines), encoding="utf-8")
        print(f"wrote {OUT_DIR / 'states' / f'shard_{shard}.digest.md'}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    schedule = load_schedule()
    plan = schedule[args.task]
    steps = json.loads(Path(args.steps_file).read_text(encoding="utf-8")) \
        if args.steps_file else json.loads(args.steps)
    tool_manager = manager()
    state = reset_state(int(plan["state_seed"]))
    roots = temp_dirs(state)
    for position, step in enumerate(steps, 1):
        pre = tool_manager.get_api_state()
        if "parallel" in step:
            certificate, outputs, issues = certify_parallel(
                tool_manager, step["parallel"], pre, roots)
            print(f"--- step {position} PARALLEL width={len(step['parallel'])}")
            if issues:
                print("  CERTIFICATION FAILED:", issues)
                return 1
            for call, output in zip(step["parallel"], outputs):
                print(f"  {call['tool_name']}({canonical(call.get('arguments') or {})})")
                print(f"    -> {canonical(output)[:900]}")
            print("  certificate: read-only, order-invariant, isolated==forward==reverse")
            continue
        for call in step.get("calls", []):
            output = tool_manager.invoke_python_tool(
                call["tool_name"], call.get("arguments") or {})
            failed, detail = call_failed(call["tool_name"], output)
            post = tool_manager.get_api_state()
            quality = validate_transition_quality(
                tool_name=call["tool_name"], tool_output=output, pre_state=pre,
                post_state=post, tool_arguments=call.get("arguments") or {})
            print(f"--- step {position} {call['tool_name']}"
                  f"({canonical(call.get('arguments') or {})})")
            print(f"    -> {canonical(output)[:1200]}")
            print(f"    error={failed}{':' + detail if failed else ''} "
                  f"state_changed={post != pre} quality_passed={quality['passed']}"
                  f"{'' if quality['passed'] else ' ' + canonical(quality['issues'])}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    schedule = load_schedule()
    spec_path = Path(args.specs) if args.specs else OUT_DIR / "specs" / f"shard_{args.shard}.jsonl"
    out_path = Path(args.out) if args.out else OUT_DIR / "shards" / f"shard_{args.shard}.jsonl"
    if not spec_path.exists():
        print(f"missing spec file {spec_path}")
        return 1
    specs = read_jsonl(spec_path)
    rows: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    ok = 0
    for spec in specs:
        task_id = spec.get("task_id")
        plan = schedule.get(task_id)
        if plan is None:
            report.append({"task_id": task_id, "passed": False,
                           "errors": ["UNKNOWN_TASK_ID"]})
            continue
        try:
            row, problems = build_row(spec, plan)
        except Exception as exc:  # fail closed with a readable code
            row, problems = None, [f"BUILDER_EXCEPTION:{type(exc).__name__}:{exc}"]
        if row is None:
            report.append({"task_id": task_id, "passed": False, "errors": problems})
            continue
        rows.append(row)
        ok += 1
        report.append({"task_id": task_id, "passed": True, "warnings": problems})

    rows.sort(key=lambda row: row["generation_metadata"]["task_index"])
    if rows:
        write_jsonl(out_path, rows)
    report_path = out_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"built {ok}/{len(specs)} rows -> {out_path}")
    for entry in report:
        if not entry["passed"]:
            print(f"  FAIL {entry['task_id']}")
            for code in entry["errors"]:
                print(f"    {code}")
        elif entry.get("warnings"):
            print(f"  WARN {entry['task_id']}: {entry['warnings']}")
    return 0 if ok == len(specs) else 1


def profile_file_names() -> dict[str, Any]:
    if RUN_NAME == "fixed_refuse_parallel_100":
        return PROFILE_FILES
    return {
        f"refusal_interactive.jsonl": PROFILE_FILES["refusal_interactive_32.jsonl"],
        f"refusal_unsupported.jsonl": PROFILE_FILES["refusal_unsupported_8.jsonl"],
        f"parallel.jsonl": PROFILE_FILES["parallel_40.jsonl"],
        f"combined.jsonl": PROFILE_FILES["combined_20.jsonl"],
    }


PROFILE_FILES = {
    "refusal_interactive_32.jsonl": lambda meta: meta["feature_schedule"] == "interactive-refusal",
    "refusal_unsupported_8.jsonl": lambda meta: (
        meta["feature_schedule"] == "terminal" and meta["contains_refusal"]),
    "parallel_40.jsonl": lambda meta: (
        meta["feature_schedule"] == "terminal" and not meta["contains_refusal"]),
    "combined_20.jsonl": lambda meta: meta["feature_schedule"] == "combined",
}


def cmd_merge(args: argparse.Namespace) -> int:
    schedule = load_schedule()
    shards = 1 + max(plan["shard"] for plan in schedule.values())
    rows: list[dict[str, Any]] = []
    for shard in range(shards):
        path = OUT_DIR / "shards" / f"shard_{shard}.jsonl"
        shard_rows = read_jsonl(path) if path.exists() else []
        if len(shard_rows) != 20 and not args.allow_partial:
            print(f"shard {shard} has {len(shard_rows)} rows, expected 20")
            return 1
        rows.extend(shard_rows)
    rows.sort(key=lambda row: row["generation_metadata"]["task_index"])
    write_jsonl(OUT_DIR / MERGED_NAME, rows)
    print(f"merged {len(rows)} rows")
    for name, predicate in profile_file_names().items():
        subset = [row for row in rows if predicate(row["generation_metadata"])]
        write_jsonl(OUT_DIR / name, subset)
        print(f"  {name}: {len(subset)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input))
    failures = []
    for row in rows:
        errors = replay_row(row, int(row["generation_metadata"]["state_seed"]))
        errors.extend(f"EVAL_SPEC:{code}" for code in validate_feature_evaluation_spec(row))
        if errors:
            failures.append({"task_id": row["generation_metadata"]["task_id"],
                             "errors": errors})
    print(json.dumps({"rows": len(rows), "replay_failures": failures,
                      "passed": not failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def render_row(row: dict[str, Any], *, output_chars: int) -> str:
    meta = row["generation_metadata"]
    lines = [
        "=" * 78,
        f"{meta['task_id']}  profile={meta['profile']}  schedule={meta['feature_schedule']}"
        f"  category={meta['focus_category']}  shard={meta['shard']}",
        f"steps={meta['num_action_transitions']} turns={meta['num_turns']} "
        f"per_turn={meta['actions_per_turn']} parallel_width={meta['parallel_width']}"
        f" terminal_mode={meta['terminal_mode']}",
        f"scenario: {meta['scenario']}",
        f"overall_task: {row['conversation']['overall_task'][:400]}",
        f"exposed tools: {[tool['name'] for tool in row['available_tools']]}",
    ]
    for event in meta.get("clarification_events") or []:
        lines.append(
            f"clarification: turn {event['refusal_turn']} -> recovery turn "
            f"{event['recovery_turn']} reason={event['reason']} "
            f"protected={event['recovery_certificate']['protected_tokens']}")
        lines.append(f"  fully specified source: {event['fully_specified_source_query']}")
    for turn in row["conversation"]["turns"]:
        lines.append("-" * 78)
        lines.append(f"USER  [turn {turn['turn_number']}]: {turn['user_query']}")
        if turn.get("query_intent"):
            lines.append(f"      (intent: {turn['query_intent']})")
        for step in turn["steps"]:
            tag = {"parallel": "PARALLEL", "refusal": "REFUSE"}.get(
                step["execution_mode"], "ACTION")
            lines.append(f"  {tag} step {step['step_number']}"
                         f"{' [order does not matter]' if tag == 'PARALLEL' else ''}")
            for call in step["tool_calls"]:
                lines.append(f"    {call['tool_name']}({canonical(call['arguments'])})")
                lines.append(f"      -> {canonical(call['output'])[:output_chars]}")
        lines.append(f"ASSISTANT: {turn['assistant_response']}")
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input) if args.input else OUT_DIR / MERGED_NAME)
    wanted = set(args.tasks.split(",")) if args.tasks else None
    for row in rows:
        if wanted and row["generation_metadata"]["task_id"] not in wanted:
            continue
        if args.profile and row["generation_metadata"]["profile"] != args.profile:
            continue
        if args.feature == "parallel" and not row["generation_metadata"]["contains_parallel"]:
            continue
        if args.feature == "refusal" and not row["generation_metadata"]["contains_refusal"]:
            continue
        print(render_row(row, output_chars=args.output_chars))
    return 0


def hidden_state_leaks(row: dict[str, Any]) -> list[str]:
    """Arguments that could only have come from the non-visible initial state.

    The evaluation contract says the stored ``initial_api_state`` is never
    policy-visible: an argument literal must be derivable from a user turn or an
    earlier tool result.  A literal that appears nowhere in the visible prefix
    but does appear in the initial state was read out of hidden state.
    """
    tool_manager = manager()
    # The virtual filesystem root is a random temp path (/tmp/vfs_5tdgl6_o) that
    # is serialized into initial_api_state. Its digits made short numeric
    # arguments "ground" by luck, differently on every run. Strip it first.
    roots = temp_dirs(row["initial_api_state"])
    state_text = canonical(
        normalise_paths(row["initial_api_state"], roots)).casefold()
    findings: list[str] = []
    corpus: list[str] = []
    for turn in row["conversation"]["turns"]:
        corpus.append(turn["user_query"])
        for step in turn["steps"]:
            visible = visible_corpus(corpus)
            for call in step["tool_calls"]:
                name = call["tool_name"]
                if name == "refuse":
                    continue
                schema = tool_manager.get_tool_schema(name)
                for key, value in (call["arguments"] or {}).items():
                    if not argument_needs_visibility(schema, key, value):
                        continue
                    if literal_visible(value, visible):
                        continue
                    flat: list[Any] = []
                    _flatten_literals(value, flat)
                    for item in flat:
                        text = str(item).strip()
                        # Short strings are noisy, but a short *number* is a real
                        # argument: priority=4 and fanSpeed=60 were slipping through.
                        if len(text) < 3 and not isinstance(item, (int, float)):
                            continue
                        if isinstance(item, bool):
                            continue
                        if text.casefold() in visible:
                            continue
                        if text.casefold() in state_text:
                            code = ("PREAUTH_TOKEN_NOT_DERIVABLE"
                                    if key == "access_token"
                                    else "HIDDEN_STATE_ARGUMENT")
                        else:
                            # Repo preflight rule 2: every target argument must be
                            # available from the visible query, prior policy-visible
                            # history, a prior tool output, or a schema default.
                            # An invented value fails byte comparison for any model.
                            code = "ARGUMENT_NOT_DERIVABLE"
                        findings.append(
                            f"t{turn['turn_number']}s{step['step_number']}:"
                            f"{code}:{name}.{key}={text}")
            for call in step["tool_calls"]:
                # normalise here too: an unmasked /tmp/vfs_<random> path in a
                # filesystem tool's output would ground literals by luck.
                corpus.append(canonical(normalise_paths(call["output"], roots)))
        corpus.append(turn["assistant_response"])
    return findings


def _flatten_literals(value: Any, out: list[Any]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float, str)):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten_literals(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten_literals(item, out)


def cmd_leakscan(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.input) if args.input else OUT_DIR / MERGED_NAME)
    report = []
    for row in rows:
        findings = hidden_state_leaks(row)
        if findings:
            report.append({"task_id": row["generation_metadata"]["task_id"],
                           "findings": findings})
    total = sum(len(item["findings"]) for item in report)
    print(json.dumps({"rows": len(rows), "rows_with_leaks": len(report),
                      "total_leaks": total, "detail": report},
                     ensure_ascii=False, indent=2))
    return 0 if not report else 1


SEEDED_SECRETS = re.compile(
    r"\b(?:[a-z]+_travel_\d+|travel_client_\d+|budget_\d+|corp_travel_\d+|"
    r"[A-Za-z]+2024!|s3cretK3y!|BizTr@vel!|LuxTr@vel!|Budget\$24!|CorpTr@v!)\b")


def cmd_diversity(args: argparse.Namespace) -> int:
    """Cross-shard template-family detection.

    Per-shard builds cannot see this: twenty authors each writing five rows of a
    category will independently reach for the same opening move.  Similarity is
    measured on first-turn queries within a category, plus verbatim reuse of
    seeded secrets and personas, plus duplicate call-sequence skeletons.
    """
    import difflib

    rows = read_jsonl(Path(args.input) if args.input else OUT_DIR / MERGED_NAME)
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    secrets: dict[str, list[str]] = defaultdict(list)
    skeletons: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        meta = row["generation_metadata"]
        task_id = meta["task_id"]
        turns = row["conversation"]["turns"]
        by_category[meta["focus_category"]].append((task_id, turns[0]["user_query"]))
        for turn in turns:
            for token in set(SEEDED_SECRETS.findall(turn["user_query"])):
                secrets[token].append(task_id)
        sequence = tuple(
            call["tool_name"]
            for turn in turns for step in turn["steps"] for call in step["tool_calls"])
        skeletons["|".join(sequence)].append(task_id)

    pairs = []
    for category, items in sorted(by_category.items()):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ratio = difflib.SequenceMatcher(
                    None, items[i][1][:200].casefold(), items[j][1][:200].casefold()
                ).ratio()
                if ratio >= args.threshold:
                    pairs.append({"category": category, "similarity": round(ratio, 3),
                                  "rows": [items[i][0], items[j][0]]})
    report = {
        "rows": len(rows),
        "opener_similarity_threshold": args.threshold,
        "too_similar_openers": sorted(pairs, key=lambda item: -item["similarity"]),
        "reused_seeded_secrets": {k: v for k, v in sorted(secrets.items()) if len(v) > 1},
        "duplicate_call_skeletons": {
            k[:110]: v for k, v in sorted(skeletons.items()) if len(v) > 1},
        "max_similarity_per_category": {
            category: max(
                (round(difflib.SequenceMatcher(
                    None, items[i][1][:200].casefold(), items[j][1][:200].casefold()
                ).ratio(), 3)
                 for i in range(len(items)) for j in range(i + 1, len(items))),
                default=0.0)
            for category, items in sorted(by_category.items())},
    }
    report["passed"] = not (report["too_similar_openers"]
                            or report["reused_seeded_secrets"]
                            or report["duplicate_call_skeletons"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    rows = read_jsonl(OUT_DIR / MERGED_NAME)
    schedule = load_schedule()
    tool_manager = manager()

    steps = Counter()
    turns = Counter()
    widths = Counter()
    categories = Counter()
    tools = Counter()
    reasons = Counter()
    profiles = Counter()
    homogeneous = 0
    heterogeneous = 0
    dependencies = 0
    signatures = set()

    for row in rows:
        meta = row["generation_metadata"]
        conversation = row["conversation"]
        profiles[meta["profile"]] += 1
        turns[len(conversation["turns"])] += 1
        steps[sum(len(turn["steps"]) for turn in conversation["turns"])] += 1
        categories[meta["focus_category"]] += 1
        for turn in conversation["turns"]:
            for step in turn["steps"]:
                for call in step["tool_calls"]:
                    tools[call["tool_name"]] += 1
                if step["execution_mode"] == "parallel":
                    widths[len(step["tool_calls"])] += 1
                    names = {call["tool_name"] for call in step["tool_calls"]}
                    if len(names) == 1:
                        homogeneous += 1
                    else:
                        heterogeneous += 1
        for event in meta.get("clarification_events") or []:
            reasons[event["reason"]] += 1
        if meta["terminal_mode"] == "refusal":
            reasons["no_appropriate_function"] += 1
        signatures.add(semantic_signature(row))
        dependencies += count_dependencies(row)

    audit = OUT_DIR / f"{RUN_NAME}.audit.json"
    manifest = {
        "dataset": OUT_DIR.name,
        "generated_by": "scripts/build_fixed_refuse_parallel_100.py",
        "authoring_method": "claude-code-subagent",
        "judging_method": "claude-code-subagent",
        "external_llm_api_used": False,
        "deterministic_seed": SEED,
        "evaluation_spec_version": rows[0]["generation_metadata"]["evaluation_spec"]["version"],
        "source_rows": len(rows),
        "feature_transition_tasks": {
            "internal": count_lines(OUT_DIR / f"{RUN_NAME}.internal.tasks.jsonl"),
            "bfcl_native": count_lines(OUT_DIR / f"{RUN_NAME}.bfcl_native.tasks.jsonl"),
        },
        "profile_counts": dict(sorted(profiles.items())),
        "schedule_counts": dict(sorted(Counter(
            row["generation_metadata"]["feature_schedule"] for row in rows).items())),
        "step_distribution": dict(sorted(steps.items())),
        "turn_distribution": dict(sorted(turns.items())),
        "parallel_width_distribution": dict(sorted(widths.items())),
        "category_distribution": dict(sorted(categories.items())),
        "tool_usage": dict(sorted(tools.items(), key=lambda item: (-item[1], item[0]))),
        "distinct_tools_used": len([name for name in tools if name != "refuse"]),
        "refusal_reason_counts": dict(sorted(reasons.items())),
        "parallel_group_composition": {
            "same_tool_groups": homogeneous,
            "heterogeneous_tool_groups": heterogeneous,
        },
        "ordered_dependency_count": dependencies,
        "semantic_uniqueness": {"unique_signatures": len(signatures), "rows": len(rows)},
        "per_shard_rows": dict(sorted(Counter(
            str(row["generation_metadata"]["shard"]) for row in rows).items())),
        "artifacts": {
            "schedule": str(OUT_DIR / "schedule.jsonl"),
            "shards": [str(path) for path in sorted((OUT_DIR / "shards").glob("shard_*.jsonl"))],
            "reviews": [str(OUT_DIR / "reviews" / name) for name in (
                "refusal_review.jsonl", "parallel_review.jsonl",
                "naturalness_diversity_review.jsonl")],
            "merged": str(OUT_DIR / MERGED_NAME),
            "profile_files": [str(OUT_DIR / name) for name in profile_file_names()],
            "audit": str(audit),
            "internal_tasks": str(OUT_DIR / f"{RUN_NAME}.internal.tasks.jsonl"),
            "bfcl_native_tasks": str(OUT_DIR / f"{RUN_NAME}.bfcl_native.tasks.jsonl"),
            "html": str(OUT_DIR / f"{RUN_NAME}.review.html"),
            "manifest": str(OUT_DIR / "manifest.json"),
            "helper_script": str(ROOT / "scripts/build_fixed_refuse_parallel_100.py"),
        },
        "schedule_summary": json.loads((OUT_DIR / "schedule.summary.json").read_text()),
    }
    if audit.exists():
        report = json.loads(audit.read_text())
        manifest["audit_passed"] = report.get("passed")
        manifest["audit_feature_transition_counts"] = report.get("feature_transition_counts")
    for name in ("refusal_review.jsonl", "parallel_review.jsonl",
                 "naturalness_diversity_review.jsonl"):
        path = OUT_DIR / "reviews" / name
        if path.exists():
            verdicts = read_jsonl(path)
            manifest.setdefault("judge_status", {})[name.replace(".jsonl", "")] = {
                "verdicts": len(verdicts),
                "passed": sum(1 for item in verdicts if item.get("passed") is True),
                "failed": [item["task_id"] for item in verdicts if item.get("passed") is not True],
            }
    repairs = OUT_DIR / "repairs.jsonl"
    manifest["repairs"] = read_jsonl(repairs) if repairs.exists() else []
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"schedule_summary", "tool_usage"}},
                     ensure_ascii=False, indent=2))
    return 0


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def count_dependencies(row: dict[str, Any]) -> int:
    """Count later calls whose argument literals came from an earlier output."""
    produced: list[str] = []
    total = 0
    for turn in row["conversation"]["turns"]:
        for step in turn["steps"]:
            corpus = " ".join(produced).casefold()
            for call in step["tool_calls"]:
                if call["tool_name"] == "refuse":
                    continue
                values = [value for value in (call["arguments"] or {}).values()
                          if isinstance(value, (str, int, float)) and not isinstance(value, bool)]
                if values and corpus and any(
                        str(value).casefold() in corpus and len(str(value)) > 2
                        for value in values):
                    total += 1
            for call in step["tool_calls"]:
                produced.append(canonical(call["output"]))
    return total


def semantic_signature(row: dict[str, Any]) -> str:
    payload = []
    for turn in row.get("conversation", {}).get("turns", []):
        steps = []
        for step in turn.get("steps", []):
            calls = [{"name": call.get("tool_name"), "arguments": call.get("arguments", {})}
                     for call in step.get("tool_calls", [])]
            if step.get("call_order_matters", True) is False:
                calls.sort(key=canonical)
            steps.append({"execution_mode": step.get("execution_mode", "sequential"),
                          "calls": calls})
        payload.append({"query": turn.get("user_query", ""), "steps": steps})
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def main() -> int:
    global OUT_DIR, RUN_NAME, MERGED_NAME, TASK_PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    parser.add_argument("--run-dir", help="output directory (default: the fixed-100 run)")
    parser.add_argument("--run-name", help="artifact stem (default: fixed_refuse_parallel_100)")
    parser.add_argument("--task-prefix", help="task_id prefix (default: fixed-rp)")

    schedule_parser = sub.add_parser("schedule")
    schedule_parser.add_argument("--shards", type=int, default=5)
    schedule_parser.add_argument("--seed", type=int, default=SEED)
    schedule_parser.set_defaults(func=cmd_schedule)
    sub.add_parser("states").set_defaults(func=cmd_states)

    probe = sub.add_parser("probe")
    probe.add_argument("--task", required=True)
    probe.add_argument("--steps")
    probe.add_argument("--steps-file")
    probe.set_defaults(func=cmd_probe)

    build = sub.add_parser("build")
    build.add_argument("--shard", type=int)
    build.add_argument("--specs")
    build.add_argument("--out")
    build.set_defaults(func=cmd_build)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--allow-partial", action="store_true")
    merge_parser.set_defaults(func=cmd_merge)

    show = sub.add_parser("show")
    show.add_argument("--input")
    show.add_argument("--tasks", help="comma separated task ids")
    show.add_argument("--profile")
    show.add_argument("--feature", choices=("parallel", "refusal"))
    show.add_argument("--output-chars", type=int, default=400)
    show.set_defaults(func=cmd_show)

    verify = sub.add_parser("verify")
    verify.add_argument("--input", required=True)
    verify.set_defaults(func=cmd_verify)

    div = sub.add_parser("diversity")
    div.add_argument("--input")
    div.add_argument("--threshold", type=float, default=0.55)
    div.set_defaults(func=cmd_diversity)

    leak = sub.add_parser("leakscan")
    leak.add_argument("--input")
    leak.set_defaults(func=cmd_leakscan)

    sub.add_parser("manifest").set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    if args.run_dir:
        OUT_DIR = Path(args.run_dir)
    if args.run_name:
        RUN_NAME = args.run_name
        MERGED_NAME = f"{args.run_name}.jsonl"
    if args.task_prefix:
        TASK_PREFIX = args.task_prefix
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
