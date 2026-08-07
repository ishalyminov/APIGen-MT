#!/usr/bin/env python3
"""Resume a matched 50-row DeepSeek/GLM teacher comparison.

DeepSeek keeps the thirteen accepted rows already paid for.  The remaining
schedule balances final category and turn-count marginals; GLM runs the same
fifty structural slots from scratch.  Every subprocess is one candidate with
a hard ten-request budget, and all failed subprocess usage is retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    ROOT / "data/generated/teacher_compare_correct_routing_50x2_20260806"
)
TOOL_POOL = ROOT / "magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
INVOCATION_EXAMPLES = (
    ROOT / "magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
)
DEDUPE_AGAINST = (
    ROOT
    / "data/generated/symbolic_episode_plan_v2_15_20_costfix3_20260806"
    / "final10_15_20.jsonl"
)
QWEN_WRITER = "Qwen/Qwen3.6-35B-A3B-FP8"
TEACHERS = {
    "deepseek": "~deepseek/deepseek-v4-flash-latest",
    "glm52": "z-ai/glm-5.2",
}
# Current OpenRouter list prices (2026-08-07).  The generator checks its 100k
# token ceiling between requests, so the final response can overshoot that
# ceiling by one bounded request. Reserve 125k tokens at the more expensive
# completion rate for every in-flight candidate. Qwen is served by the
# in-cluster proxy and contributes no OpenRouter spend.
MAX_RESERVED_TOKENS_PER_CANDIDATE = 125_000
MAX_CANDIDATE_COST_USD = {
    "deepseek": MAX_RESERVED_TOKENS_PER_CANDIDATE * 0.00000018,
    "glm52": MAX_RESERVED_TOKENS_PER_CANDIDATE * 0.000001716,
}
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


@dataclass(frozen=True)
class Spec:
    index: int
    category: str
    schedule: tuple[int, ...]
    required_tool: str | None = None
    deepseek_seed: str | None = None
    companion_category: str | None = None

    @property
    def turns(self) -> int:
        return len(self.schedule)

    @property
    def steps(self) -> int:
        return sum(self.schedule)

    @property
    def stem(self) -> str:
        category = self.category.lower().replace(" ", "_")
        vector = "-".join(map(str, self.schedule))
        return f"{self.index:03d}.{category}.t{self.turns}.s{self.steps}.v{vector}"


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _row_count(path: Path) -> int:
    return sum(1 for _ in _read_jsonl(path))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _tool_catalog() -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for tool in _read_jsonl(TOOL_POOL):
        result[str(tool["category"])].append(
            str(tool.get("name") or tool.get("api_name"))
        )
    return {key: sorted(values) for key, values in result.items()}


def _actual_tools(row: dict[str, Any]) -> list[str]:
    return [
        str(call.get("tool_name", ""))
        for turn in row.get("conversation", {}).get("turns", [])
        for step in turn.get("steps", [])
        for call in step.get("tool_calls", [])
    ]


def _balanced_vector(steps: int, turns: int, rng: random.Random) -> tuple[int, ...]:
    quotient, remainder = divmod(steps, turns)
    vector = [quotient + 1] * remainder + [quotient] * (turns - remainder)
    rng.shuffle(vector)
    if min(vector) < 1 or max(vector) > 10:
        raise ValueError(f"Unsupported {steps}-step/{turns}-turn vector {vector}")
    return tuple(vector)


def build_schedule(output_dir: Path, seed: int) -> list[Spec]:
    deepseek_root = output_dir / "deepseek_v4_flash_latest"
    seed_specs: list[Spec] = []
    index = 0
    seed_groups = (
        ("Finance", "finance.jsonl", 2),
        ("Science", "science.jsonl", 6),
        ("Storage", "storage.jsonl", 4),
    )
    for category, relative, expected in seed_groups:
        path = deepseek_root / relative
        rows = list(_read_jsonl(path))
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} seed rows in {path}, got {len(rows)}")
        for row_number in range(expected):
            seed_specs.append(
                Spec(
                    index=index,
                    category=category,
                    schedule=(3, 3, 3, 3, 3),
                    deepseek_seed=f"{relative}#{row_number}",
                )
            )
            index += 1
    smoke_relative = (
        "diverse_fill/007_t4_s16_4-4-4-4_turnwise_align_smoke.jsonl"
    )
    if _row_count(deepseek_root / smoke_relative) != 1:
        raise RuntimeError("The audited DeepSeek alignment smoke is missing")
    seed_specs.append(
        Spec(
            index=index,
            category="Science",
            schedule=(4, 4, 4, 4),
            deepseek_seed=f"{smoke_relative}#0",
        )
    )
    index += 1

    companion_categories = {
        "Communication": "Events",
        "Events": "Communication",
        "Finance": "Communication",
        "Posting Api": "Communication",
        "Science": "Communication",
        "Storage": "Communication",
        "Travel Booking": "Communication",
        "Vehicle Control": "Communication",
    }
    # Preserve all four already accepted production rows byte-for-byte. The
    # remaining marginals deliberately keep only four extreme two-turn rows;
    # 3-5 turns make 15-20 calls substantive without demanding 8-10 unrelated
    # operations in every user utterance.
    fixed_specs = {
        13: Spec(
            13, "Events", (6, 6, 6), "resolve_ticket", None, "Communication"
        ),
        15: Spec(15, "Storage", (5, 5, 4, 5), "touch", None, "Communication"),
        16: Spec(
            16,
            "Travel Booking",
            (5, 5, 5, 5),
            "list_all_airports",
            None,
            "Communication",
        ),
        30: Spec(
            30, "Posting Api", (9, 9), "get_tweet", None, "Communication"
        ),
        33: Spec(
            33,
            "Finance",
            (5, 5, 6),
            "get_transaction_history",
            None,
            "Communication",
        ),
    }
    turns = [2] * 3 + [3] * 8 + [4] * 9 + [5] * 12
    steps = [16] * 6 + [17] * 8 + [18] * 6 + [19] * 6 + [20] * 6
    categories = (
        ["Communication"] * 7
        + ["Events"] * 5
        + ["Finance"] * 3
        + ["Posting Api"] * 5
        + ["Storage"] * 1
        + ["Travel Booking"] * 5
        + ["Vehicle Control"] * 6
    )
    if not (len(turns) == len(steps) == len(categories) == 32):
        raise AssertionError("replacement schedule must contain 32 slots")
    rng = random.Random(seed)
    rng.shuffle(turns)
    rng.shuffle(steps)
    rng.shuffle(categories)

    catalog = _tool_catalog()
    existing_tools: dict[str, set[str]] = defaultdict(set)
    for spec in seed_specs:
        relative, ordinal_text = str(spec.deepseek_seed).rsplit("#", 1)
        rows = list(_read_jsonl(deepseek_root / relative))
        existing_tools[spec.category].update(_actual_tools(rows[int(ordinal_text)]))
    target_queues: dict[str, list[str]] = {}
    for category in CATEGORIES:
        unseen = [
            name for name in catalog[category] if name not in existing_tools[category]
        ]
        seen = [name for name in catalog[category] if name in existing_tools[category]]
        category_rng = random.Random(f"{seed}:{category}")
        category_rng.shuffle(unseen)
        category_rng.shuffle(seen)
        target_queues[category] = unseen + seen
    category_positions: Counter[str] = Counter()

    fill_specs: list[Spec] = []
    replacements = iter(zip(turns, steps, categories))
    for index in range(13, 50):
        if index in fixed_specs:
            fill_specs.append(fixed_specs[index])
            continue
        turn_count, step_count, category = next(replacements)
        position = category_positions[category]
        category_positions[category] += 1
        targets = target_queues[category]
        # Keep a deterministic advisory target in the manifest for later
        # coverage analysis, but do not force a rare tool into a 15-20-call
        # episode. Hard per-row targets caused otherwise coherent stateful
        # plans to contort around one operation and sharply reduced yield.
        required_tool = targets[position % len(targets)]
        fill_specs.append(
            Spec(
                index=index,
                category=category,
                schedule=_balanced_vector(step_count, turn_count, rng),
                required_tool=required_tool,
                companion_category=companion_categories[category],
            )
        )
    try:
        next(replacements)
    except StopIteration:
        pass
    else:
        raise AssertionError("replacement schedule was not exhausted")
    specs = seed_specs + fill_specs
    if len(specs) != 50:
        raise AssertionError(f"Expected 50 specs, got {len(specs)}")
    return specs


def _seed_row(spec: Spec, output_dir: Path) -> dict[str, Any]:
    if not spec.deepseek_seed:
        raise ValueError("not a seed spec")
    relative, ordinal_text = spec.deepseek_seed.rsplit("#", 1)
    rows = list(_read_jsonl(output_dir / "deepseek_v4_flash_latest" / relative))
    return rows[int(ordinal_text)]


def row_path(output_dir: Path, teacher: str, spec: Spec) -> Path:
    return output_dir / "production_50x2" / "rows" / teacher / f"{spec.stem}.jsonl"


def validate_row(
    row: dict[str, Any],
    *,
    spec: Spec,
    teacher_model: str,
) -> list[str]:
    errors: list[str] = []
    conversation = row.get("conversation", {})
    turns = conversation.get("turns", [])
    actual_vector = [
        sum(len(step.get("tool_calls", [])) for step in turn.get("steps", []))
        for turn in turns
    ]
    if actual_vector != list(spec.schedule):
        errors.append(f"CALL_VECTOR:{actual_vector}!={list(spec.schedule)}")
    actual_tools = set(_actual_tools(row))
    focus_tools = set(_tool_catalog().get(spec.category, []))
    if not actual_tools.intersection(focus_tools):
        errors.append(f"MISSING_FOCUS_CATEGORY:{spec.category}")
    if not row.get("verification_result", {}).get("overall_verification_passed"):
        errors.append("OVERALL_VERIFICATION_FAILED")
    metadata = row.get("generation_metadata", {})
    metrics = metadata.get("symbolic_plan_metrics", {})
    if int(metrics.get("hidden_argument_count", -1)) != 0:
        errors.append(f"HIDDEN_ARGUMENTS:{metrics.get('hidden_argument_count')}")
    routing = metadata.get("model_routing", {})
    expected_routing = {
        "generator": teacher_model,
        "semantic_judge": teacher_model,
        "final_response_writer": QWEN_WRITER,
        "grounding_judge": teacher_model,
    }
    if routing != expected_routing:
        errors.append(f"ROUTING:{routing}!={expected_routing}")
    if any(not str(turn.get("assistant_response", "")).strip() for turn in turns):
        errors.append("EMPTY_ASSISTANT_RESPONSE")
    if "The requested actions completed successfully." in json.dumps(row):
        errors.append("LEGACY_PLACEHOLDER")
    budget = row.get("token_usage", {})
    calls = int(budget.get("total_llm_calls", budget.get("total_calls", 0)) or 0)
    if calls > 10:
        errors.append(f"CALL_BUDGET:{calls}>10")
    return errors


def _has_valid_row(
    output_dir: Path,
    *,
    teacher: str,
    teacher_model: str,
    spec: Spec,
) -> bool:
    path = row_path(output_dir, teacher, spec)
    if _row_count(path) != 1:
        return False
    row = next(_read_jsonl(path))
    return not validate_row(row, spec=spec, teacher_model=teacher_model)


def _usage_summary(output_dir: Path, teacher: str) -> dict[str, Any]:
    reports = list(
        (output_dir / "production_50x2" / "usage" / teacher).glob("*.json")
    )
    totals: dict[str, float] = defaultdict(float)
    accepted_reports = 0
    for report in reports:
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        accepted_reports += int(payload.get("accepted_rows", 0) or 0) > 0
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_prompt_tokens",
            "cost_usd",
            "total_llm_calls",
        ):
            value = payload.get(
                key,
                payload.get("total_calls", 0) if key == "total_llm_calls" else 0,
            )
            totals[key] += float(value or 0)
    return {
        **{
            key: int(value) if key != "cost_usd" else value
            for key, value in totals.items()
        },
        "reports": len(reports),
        "accepted_reports": accepted_reports,
        "rejected_reports": len(reports) - accepted_reports,
    }


def _experiment_budget_snapshot(
    output_dir: Path,
    *,
    total_budget_usd: float,
    safety_reserve_usd: float,
) -> dict[str, Any]:
    """Conservatively account for every report and missing production report."""

    reported_cost = 0.0
    report_count = 0
    for path in output_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if not isinstance(payload.get("cost_usd"), (int, float)):
            continue
        if "total_calls" not in payload and "total_llm_calls" not in payload:
            continue
        reported_cost += float(payload["cost_usd"])
        report_count += 1

    missing_reports: list[str] = []
    missing_reserve = 0.0
    production = output_dir / "production_50x2"
    for teacher in TEACHERS:
        log_dir = production / "logs" / teacher
        usage_dir = production / "usage" / teacher
        for log in log_dir.glob("*.log"):
            expected_usage = usage_dir / f"{log.stem}.json"
            if expected_usage.exists():
                continue
            missing_reports.append(str(log.relative_to(output_dir)))
            missing_reserve += MAX_CANDIDATE_COST_USD[teacher]

    effective_spend = reported_cost + missing_reserve + safety_reserve_usd
    return {
        "hard_cap_usd": total_budget_usd,
        "reported_cost_usd": reported_cost,
        "report_count": report_count,
        "missing_report_reserve_usd": missing_reserve,
        "missing_reports": missing_reports,
        "safety_reserve_usd": safety_reserve_usd,
        "effective_spend_usd": effective_spend,
        "remaining_after_reserves_usd": max(
            0.0, total_budget_usd - effective_spend
        ),
    }


def _command(
    *,
    spec: Spec,
    teacher_model: str,
    output: Path,
    usage: Path,
    archive: Path,
    registry: Path,
) -> list[str]:
    command = [
        sys.executable,
        "src/generate_step_by_step.py",
        "--mode",
        "multi-turn",
        "--num-datapoints",
        "1",
        "--num-turns",
        str(spec.turns),
        "--num-actions",
        str(max(spec.schedule)),
        "--blueprint-max-actions-per-turn",
        str(max(spec.schedule)),
        "--blueprint-actions-per-turn",
        ",".join(map(str, spec.schedule)),
        "--min-total-steps",
        str(spec.steps),
        "--max-total-steps",
        str(spec.steps),
        "--model",
        teacher_model,
        "--judge-model",
        teacher_model,
        "--final-response-model",
        QWEN_WRITER,
        "--tool-pool",
        str(TOOL_POOL),
        "--invocation-examples",
        str(INVOCATION_EXAMPLES),
        "--category",
        spec.category,
        "--output",
        str(output),
        "--usage-report",
        str(usage),
        "--candidate-archive-dir",
        str(archive),
        "--dedupe-registry",
        str(registry),
        "--dedupe-against",
        str(DEDUPE_AGAINST),
        "--max-calls-per-candidate",
        "10",
        "--max-calls-per-accepted-row",
        "10",
        "--max-tokens-per-accepted-row",
        "100000",
        "--max-candidate-starts-per-row",
        "1",
        "--max-turn-attempts",
        "1",
        "--no-resume",
    ]
    if spec.companion_category:
        command.extend(["--context-category", spec.companion_category])
    return command


def _next_attempt(usage_dir: Path, spec: Spec) -> int:
    attempts = []
    for path in usage_dir.glob(f"{spec.stem}.a*.json"):
        try:
            attempts.append(int(path.stem.rsplit(".a", 1)[1]))
        except ValueError:
            pass
    production = usage_dir.parent.parent
    teacher = usage_dir.name
    for suffix, directory in (
        (".log", production / "logs" / teacher),
        (".jsonl", production / "traces" / teacher),
    ):
        for path in directory.glob(f"{spec.stem}.a*{suffix}"):
            try:
                attempts.append(int(path.stem.rsplit(".a", 1)[1]))
            except ValueError:
                pass
    return max(attempts, default=0) + 1


def generate_spec(
    *,
    spec: Spec,
    teacher: str,
    teacher_model: str,
    output_dir: Path,
    registry: Path,
    max_attempts: int,
    timeout: int,
) -> tuple[int, str]:
    output = row_path(output_dir, teacher, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    if _row_count(output) == 1:
        row = next(_read_jsonl(output))
        errors = validate_row(row, spec=spec, teacher_model=teacher_model)
        if not errors:
            return spec.index, "already-complete"
        quarantine = output_dir / "production_50x2" / "quarantine" / teacher
        quarantine.mkdir(parents=True, exist_ok=True)
        output.replace(quarantine / f"{spec.stem}.{time.time_ns()}.jsonl")

    base = output_dir / "production_50x2"
    usage_dir = base / "usage" / teacher
    trace_dir = base / "traces" / teacher
    log_dir = base / "logs" / teacher
    archive = base / "candidate_archive" / teacher / spec.stem
    for directory in (usage_dir, trace_dir, log_dir, archive):
        directory.mkdir(parents=True, exist_ok=True)
    first_attempt = _next_attempt(usage_dir, spec)
    statuses = []
    for offset in range(max_attempts):
        attempt = first_attempt + offset
        usage = usage_dir / f"{spec.stem}.a{attempt}.json"
        trace = trace_dir / f"{spec.stem}.a{attempt}.jsonl"
        log = log_dir / f"{spec.stem}.a{attempt}.log"
        environment = {
            **os.environ,
            "APIGEN_REASONING_EFFORT": "off",
            # Some auto-selected OpenRouter providers ignored explicit
            # reasoning-token budgets and emitted 4-6k hidden tokens per turn.
            # Use the portable low-effort control instead. Validation/writer
            # stages remain reasoning-off.
            "APIGEN_BLUEPRINT_GENERATE_REASONING_EFFORT": "low",
            "APIGEN_BLUEPRINT_TURN_COMPILE_REASONING_EFFORT": "low",
            "APIGEN_BLUEPRINT_QUERY_ALIGN_REASONING_EFFORT": "low",
            "APIGEN_BLUEPRINT_GENERATE_REASONING_MAX_TOKENS": "",
            "APIGEN_BLUEPRINT_TURN_COMPILE_REASONING_MAX_TOKENS": "",
            "APIGEN_BLUEPRINT_QUERY_ALIGN_REASONING_MAX_TOKENS": "",
            "APIGEN_BLUEPRINT_SEMANTIC_JUDGE_REASONING_EFFORT": "off",
            "APIGEN_FINAL_RESPONSE_GENERATE_REASONING_EFFORT": "off",
            "APIGEN_FINAL_RESPONSE_GROUNDING_JUDGE_REASONING_EFFORT": "off",
            "APIGEN_MAX_OUTPUT_TOKENS": "8192",
            "APIGEN_HTTP_ATTEMPTS": "1",
            "APIGEN_APPLICATION_LLM_ATTEMPTS": "1",
            # A stalled OpenRouter backend must not pin a paid candidate for
            # the library default of 15 minutes. One timeout rejects the
            # candidate and its full worst-case cost remains reserved.
            "APIGEN_LLM_TIMEOUT": "180",
            # The tested production route compiles one exact turn at a time;
            # with 2-5 turns the full alignment/judge/writer/grounding pipeline
            # still fits the hard ten-request candidate ceiling.
            "APIGEN_SYMBOLIC_TURNWISE": "1",
            "APIGEN_SYMBOLIC_TURNWISE_MIN_WIDTH": "3",
            "APIGEN_LLM_TRACE_PATH": str(trace),
        }
        command = _command(
            spec=spec,
            teacher_model=teacher_model,
            output=output,
            usage=usage,
            archive=archive,
            registry=registry,
        )
        with log.open("w", encoding="utf-8") as destination:
            destination.write(
                json.dumps(
                    {"spec": asdict(spec), "attempt": attempt, "command": command},
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=destination,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                statuses.append(f"a{attempt}:exit={completed.returncode}")
            except subprocess.TimeoutExpired:
                statuses.append(f"a{attempt}:timeout")
        if _row_count(output) == 1:
            row = next(_read_jsonl(output))
            errors = validate_row(row, spec=spec, teacher_model=teacher_model)
            if not errors:
                return spec.index, f"accepted-a{attempt}"
            quarantine = base / "quarantine" / teacher
            quarantine.mkdir(parents=True, exist_ok=True)
            output.replace(quarantine / f"{spec.stem}.a{attempt}.{time.time_ns()}.jsonl")
            statuses.append("validation=" + ",".join(errors))
    return spec.index, "failed:" + ";".join(statuses)


def _collect(
    *, output_dir: Path, teacher: str, specs: list[Spec]
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    model = TEACHERS[teacher]
    for spec in specs:
        if teacher == "deepseek" and spec.deepseek_seed:
            row = _seed_row(spec, output_dir)
        else:
            path = row_path(output_dir, teacher, spec)
            if _row_count(path) != 1:
                missing.append(spec.index)
                continue
            row = next(_read_jsonl(path))
        errors = validate_row(row, spec=spec, teacher_model=model)
        if errors:
            missing.append(spec.index)
            continue
        rows.append(row)
    return rows, missing


def _write_status(output_dir: Path, teacher: str, specs: list[Spec]) -> dict[str, Any]:
    rows, missing = _collect(output_dir=output_dir, teacher=teacher, specs=specs)
    calls = [len(_actual_tools(row)) for row in rows]
    status = {
        "teacher": teacher,
        "model": TEACHERS[teacher],
        "accepted": len(rows),
        "missing_indices": missing,
        "turn_distribution": dict(
            sorted(Counter(len(row["conversation"]["turns"]) for row in rows).items())
        ),
        "step_distribution": dict(sorted(Counter(calls).items())),
        "category_distribution": dict(
            sorted(
                Counter(
                    row.get("generation_metadata", {}).get("focus_category")
                    for row in rows
                ).items()
            )
        ),
        "usage_including_rejections": _usage_summary(output_dir, teacher),
        "updated_at": time.time(),
    }
    base = output_dir / "production_50x2"
    _atomic_json(base / f"{teacher}.status.json", status)
    _atomic_jsonl(base / f"{teacher}.partial.jsonl", rows)
    if len(rows) == 50:
        _atomic_jsonl(base / f"{teacher}.50.jsonl", rows)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--teacher", choices=("deepseek", "glm52", "both"), default="both"
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--total-budget-usd", type=float, default=10.0)
    parser.add_argument("--budget-safety-reserve-usd", type=float, default=0.75)
    parser.add_argument("--schedule-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    specs = build_schedule(output_dir, args.seed)
    production = output_dir / "production_50x2"
    production.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(production / "schedule.jsonl", [asdict(spec) for spec in specs])
    schedule_summary = {
        "rows": len(specs),
        "turn_distribution": dict(sorted(Counter(s.turns for s in specs).items())),
        "step_distribution": dict(sorted(Counter(s.steps for s in specs).items())),
        "category_distribution": dict(sorted(Counter(s.category for s in specs).items())),
        "deepseek_preserved_rows": sum(bool(s.deepseek_seed) for s in specs),
    }
    _atomic_json(production / "schedule_summary.json", schedule_summary)
    print(json.dumps(schedule_summary, indent=2), flush=True)
    if args.schedule_only:
        return 0
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_BASE"):
        raise RuntimeError("OPENAI_API_KEY and OPENAI_API_BASE must be set")
    if not os.getenv("LLM_PROXY_URL") or not os.getenv("LLM_PROXY_MASTER_KEY"):
        raise RuntimeError("LLM_PROXY_URL and LLM_PROXY_MASTER_KEY must be set")
    if args.total_budget_usd > 10.0:
        raise ValueError("The 50x2 experiment hard cap may not exceed $10")
    if args.total_budget_usd <= 0 or args.budget_safety_reserve_usd < 0:
        raise ValueError("Budget and reserve must be non-negative")

    teachers = ["deepseek", "glm52"] if args.teacher == "both" else [args.teacher]
    registry = output_dir / "dedupe_registry.txt"
    progress_lock = threading.Lock()
    failed_any = False
    for teacher in teachers:
        model = TEACHERS[teacher]
        pending = [
            spec
            for spec in specs
            if not (teacher == "deepseek" and spec.deepseek_seed)
            and not _has_valid_row(
                output_dir,
                teacher=teacher,
                teacher_model=model,
                spec=spec,
            )
        ]
        status = _write_status(output_dir, teacher, specs)
        print(
            f"{teacher}: {status['accepted']}/50 accepted; "
            f"launching {len(pending)} slots",
            flush=True,
        )
        attempt_counts: Counter[int] = Counter()
        work_queue = list(pending)
        in_flight: dict[concurrent.futures.Future, Spec] = {}
        completed_count = 0
        budget_blocked = False
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            while work_queue or in_flight:
                while work_queue and len(in_flight) < args.max_workers:
                    # Logs from already-started workers are normally included
                    # in missing_report_reserve. Also reserve every in-memory
                    # future to close the tiny race before its log is created;
                    # double reservation is intentional and conservative.
                    budget = _experiment_budget_snapshot(
                        output_dir,
                        total_budget_usd=args.total_budget_usd,
                        safety_reserve_usd=args.budget_safety_reserve_usd,
                    )
                    prospective = (
                        budget["effective_spend_usd"]
                        + (len(in_flight) + 1)
                        * MAX_CANDIDATE_COST_USD[teacher]
                    )
                    if prospective > args.total_budget_usd:
                        budget_blocked = True
                        break
                    spec = work_queue.pop(0)
                    attempt_counts[spec.index] += 1
                    future = executor.submit(
                        generate_spec,
                        spec=spec,
                        teacher=teacher,
                        teacher_model=model,
                        output_dir=output_dir,
                        registry=registry,
                        # Global admission happens after every candidate, so a
                        # worker may never hide several retries behind one check.
                        max_attempts=1,
                        timeout=args.timeout_seconds,
                    )
                    in_flight[future] = spec
                if not in_flight:
                    break
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    spec = in_flight.pop(future)
                    completed_count += 1
                    try:
                        _, result = future.result()
                    except Exception as exc:  # retain all other tasks and ledgers
                        result = f"exception:{type(exc).__name__}:{exc}"
                    accepted = result.startswith(("accepted-", "already-complete"))
                    if not accepted and attempt_counts[spec.index] < args.max_attempts:
                        work_queue.append(spec)
                    with progress_lock:
                        status = _write_status(output_dir, teacher, specs)
                        budget = _experiment_budget_snapshot(
                            output_dir,
                            total_budget_usd=args.total_budget_usd,
                            safety_reserve_usd=args.budget_safety_reserve_usd,
                        )
                        _atomic_json(production / "budget_status.json", budget)
                    print(
                        f"[{completed_count}] {teacher} {spec.stem}: {result}; "
                        f"accepted={status['accepted']}/50; "
                        f"reported_total=${budget['reported_cost_usd']:.4f}; "
                        f"reserved_total=${budget['effective_spend_usd']:.4f}",
                        flush=True,
                    )
            if budget_blocked:
                print(
                    "Hard budget admission stopped new candidates; in-flight "
                    "candidates were already fully reserved.",
                    file=sys.stderr,
                )
        status = _write_status(output_dir, teacher, specs)
        if status["accepted"] != 50:
            failed_any = True
            print(
                f"{teacher} incomplete: {status['accepted']}/50; rerun to "
                "retry only missing slots.",
                file=sys.stderr,
            )
            # Do not switch teachers after an incomplete first corpus: the
            # requested comparison is explicitly sequential.
            if teacher == "deepseek" and args.teacher == "both":
                break
    return 1 if failed_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
