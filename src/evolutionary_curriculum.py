"""Coverage, diversity, and soft-complexity curriculum for BFCL-v3 synthesis.

The scheduler shapes what the generator attempts.  It never rejects a correct
trajectory merely because a soft complexity motif was not realized.  Coverage
is updated from actual accepted calls, not from requested targets.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MOTIFS = (
    "output_chain",
    "cross_turn_reference",
    "discovery_then_action",
    "read_modify_verify",
    "branch_and_join",
    "conditional_selection",
    "stateful_followup",
    "hard_tool_disambiguation",
)
STYLE_SEEDS = (
    "concise professional",
    "casual but precise",
    "time-pressured operational",
    "careful audit-oriented",
    "non-technical end user",
    "domain-expert shorthand",
    "follow-up heavy conversation",
    "constraint-rich planning",
)
SCENARIO_SEEDS = (
    "investigation",
    "planning",
    "coordination",
    "verification",
    "recovery",
    "comparison",
    "monitoring",
    "migration",
    "cleanup",
    "reporting",
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.casefold())
        if len(token) > 2
    }


def _tool_name(tool: Dict[str, Any]) -> str:
    return str(tool.get("name") or tool.get("api_name") or "")


def _category(tool: Dict[str, Any]) -> str:
    return str(tool.get("category") or "Unknown")


def _type_compatible(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    source_type = source.get("type")
    target_type = target.get("type")
    if not source_type or not target_type:
        return True
    if source_type == target_type:
        return True
    numeric = {"integer", "number", "float", "int"}
    return source_type in numeric and target_type in numeric


@dataclass(frozen=True)
class GenerationDirective:
    directive_id: str
    target_tools: Tuple[str, ...]
    target_categories: Tuple[str, ...]
    allowed_tools: Tuple[str, ...]
    context_mode: str
    motif: str
    style_seed: str
    scenario_seed: str
    soft_requirements: Tuple[str, ...]
    lesson_ids: Tuple[str, ...] = ()
    lesson_texts: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in (
            "target_tools",
            "target_categories",
            "allowed_tools",
            "soft_requirements",
            "lesson_ids",
            "lesson_texts",
        ):
            data[key] = list(data[key])
        return data


class EvolutionLessonBank:
    def __init__(self, path: Optional[str | Path]):
        self.path = Path(path).expanduser() if path else None
        self.rules: List[Dict[str, Any]] = []
        if self.path and self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self.rules = list(payload.get("rules", []))
            except (json.JSONDecodeError, OSError, TypeError):
                self.rules = []

    def relevant(
        self,
        *,
        tools: Sequence[str],
        categories: Sequence[str],
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        tool_set = set(tools)
        category_set = set(categories)
        scored = []
        for rule in self.rules:
            applies_tools = set(rule.get("tools", []))
            applies_categories = set(rule.get("categories", []))
            score = int(bool(tool_set & applies_tools)) * 4
            score += int(bool(category_set & applies_categories)) * 2
            score += int(not applies_tools and not applies_categories)
            score += min(3, int(rule.get("evidence_count", 0)) // 5)
            if score:
                scored.append((score, str(rule.get("id", "")), rule))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [copy.deepcopy(item[2]) for item in scored[:limit]]


class CoverageState:
    VERSION = 1

    def __init__(self, path: str | Path, tool_names: Sequence[str]):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.tool_names = tuple(sorted(set(tool_names)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Initialization must use the same lock as updates. Otherwise two
        # workers starting together can overwrite the first reserved directive.
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not self.path.exists():
                    self._write(self._empty())
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _empty(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "directive_counter": 0,
            "accepted_rows": 0,
            "rejected_rows": 0,
            "target_counts": {name: 0 for name in self.tool_names},
            "used_counts": {name: 0 for name in self.tool_names},
            "target_miss_counts": {name: 0 for name in self.tool_names},
            "category_counts": {},
            "category_pair_counts": {},
            "motif_counts": {motif: 0 for motif in MOTIFS},
            "context_mode_counts": {},
            "rejection_counts": {},
        }

    def _write(self, payload: Dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def transaction(self, update=None) -> Dict[str, Any]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    payload = self._empty()
                for name in self.tool_names:
                    payload.setdefault("target_counts", {}).setdefault(name, 0)
                    payload.setdefault("used_counts", {}).setdefault(name, 0)
                    payload.setdefault("target_miss_counts", {}).setdefault(name, 0)
                if update is not None:
                    update(payload)
                    self._write(payload)
                return copy.deepcopy(payload)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class EvolutionaryCurriculum:
    def __init__(
        self,
        *,
        tools: Sequence[Dict[str, Any]],
        state_path: str | Path,
        seed: int = 0,
        all_tools_rate: float = 0.25,
        cross_domain_rate: float = 0.45,
        hard_distractor_count: int = 48,
        target_tools_per_candidate: int = 2,
        lessons_path: Optional[str | Path] = None,
    ):
        self.tools = [copy.deepcopy(tool) for tool in tools]
        self.by_name = {_tool_name(tool): tool for tool in self.tools if _tool_name(tool)}
        self.tool_names = tuple(sorted(self.by_name))
        if not self.tool_names:
            raise ValueError("evolutionary curriculum requires at least one named tool")
        self.state = CoverageState(state_path, self.tool_names)
        self.seed = int(seed)
        self.all_tools_rate = min(1.0, max(0.0, float(all_tools_rate)))
        self.cross_domain_rate = min(1.0, max(0.0, float(cross_domain_rate)))
        self.hard_distractor_count = max(4, int(hard_distractor_count))
        self.target_tools_per_candidate = max(1, min(4, int(target_tools_per_candidate)))
        self.lessons = EvolutionLessonBank(lessons_path)
        self._semantic_tokens = {
            name: _tokens(
                " ".join(
                    [
                        name,
                        str(tool.get("description", "")),
                        json.dumps(tool.get("parameters", {}), ensure_ascii=False),
                        json.dumps(tool.get("output_schema", {}), ensure_ascii=False),
                    ]
                )
            )
            for name, tool in self.by_name.items()
        }
        self._edges = self._compatible_edges()

    def _compatible_edges(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {name: [] for name in self.tool_names}
        for source_name, source_tool in self.by_name.items():
            outputs = source_tool.get("output_schema", {}).get("properties", {})
            if not outputs:
                continue
            for target_name, target_tool in self.by_name.items():
                if source_name == target_name:
                    continue
                params = target_tool.get("parameters", {}).get("properties", {})
                required = set(target_tool.get("parameters", {}).get("required", []))
                score = 0
                for output_name, output_schema in outputs.items():
                    for param_name, param_schema in params.items():
                        if not _type_compatible(output_schema, param_schema):
                            continue
                        if output_name.casefold() == param_name.casefold():
                            score += 4 + int(param_name in required)
                        elif _tokens(output_name) & _tokens(param_name):
                            score += 2
                if score:
                    result[source_name].append(target_name)
            result[source_name].sort()
        return result

    def _similarity(self, left: str, right: str) -> float:
        a = self._semantic_tokens.get(left, set())
        b = self._semantic_tokens.get(right, set())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def next_directive(self) -> GenerationDirective:
        # Reserve the serial and read the coverage frontier in one transaction.
        # A read followed by a separate increment lets concurrent workers receive
        # the same directive and defeats persistent diversity scheduling.
        def reserve(payload: Dict[str, Any]) -> None:
            payload["directive_counter"] = int(
                payload.get("directive_counter", 0)
            ) + 1

        snapshot = self.state.transaction(reserve)
        serial = int(snapshot.get("directive_counter", 1)) - 1
        rng = random.Random(self.seed + serial * 1_000_003)
        target_counts = snapshot.get("target_counts", {})
        minimum = min(target_counts.get(name, 0) for name in self.tool_names)
        frontier = [name for name in self.tool_names if target_counts.get(name, 0) <= minimum]
        primary = rng.choice(frontier)

        targets = [primary]
        candidates = list(self._edges.get(primary, []))
        candidates.sort(
            key=lambda name: (
                target_counts.get(name, 0),
                _category(self.by_name[name]) == _category(self.by_name[primary]),
                -self._similarity(primary, name),
                name,
            )
        )
        for name in candidates:
            if len(targets) >= self.target_tools_per_candidate:
                break
            if name not in targets:
                targets.append(name)
        if len(targets) < self.target_tools_per_candidate:
            remaining = sorted(
                (name for name in self.tool_names if name not in targets),
                key=lambda name: (target_counts.get(name, 0), name),
            )
            targets.extend(remaining[: self.target_tools_per_candidate - len(targets)])

        target_categories = tuple(sorted({_category(self.by_name[name]) for name in targets}))
        draw = rng.random()
        if draw < self.all_tools_rate:
            context_mode = "all_tools"
            allowed = list(self.tool_names)
        elif draw < self.all_tools_rate + self.cross_domain_rate:
            context_mode = "cross_domain"
            categories = set(target_categories)
            if len(categories) == 1:
                other_categories = sorted({_category(t) for t in self.tools} - categories)
                if other_categories:
                    categories.add(rng.choice(other_categories))
            allowed = [name for name in self.tool_names if _category(self.by_name[name]) in categories]
        else:
            context_mode = "hard_distractors"
            ranked = sorted(
                (name for name in self.tool_names if name not in targets),
                key=lambda name: (
                    -max(self._similarity(name, target) for target in targets),
                    name,
                ),
            )
            allowed = [*targets, *ranked[: max(0, self.hard_distractor_count - len(targets))]]

        for name in targets:
            if name not in allowed:
                allowed.append(name)
        allowed = sorted(set(allowed))

        motif_counts = snapshot.get("motif_counts", {})
        minimum_motif = min(motif_counts.get(motif, 0) for motif in MOTIFS)
        motif_pool = [motif for motif in MOTIFS if motif_counts.get(motif, 0) <= minimum_motif]
        motif = rng.choice(motif_pool)
        style = STYLE_SEEDS[serial % len(STYLE_SEEDS)]
        scenario = SCENARIO_SEEDS[(serial // len(STYLE_SEEDS)) % len(SCENARIO_SEEDS)]
        rules = self.lessons.relevant(tools=targets, categories=target_categories)
        directive_id = hashlib.sha256(
            f"{self.seed}|{serial}|{'|'.join(targets)}|{motif}".encode()
        ).hexdigest()[:16]

        return GenerationDirective(
            directive_id=directive_id,
            target_tools=tuple(targets),
            target_categories=target_categories,
            allowed_tools=tuple(allowed),
            context_mode=context_mode,
            motif=motif,
            style_seed=style,
            scenario_seed=scenario,
            soft_requirements=(
                "Prefer genuine prior-output bindings over copying every value into the user request.",
                "Prefer a coherent goal over a checklist of API calls.",
                "Use the selected motif only when it is semantically natural; never add filler calls.",
                "A valid simpler trajectory is acceptable and must not be rejected for complexity alone.",
            ),
            lesson_ids=tuple(str(rule.get("id", "")) for rule in rules if rule.get("id")),
            lesson_texts=tuple(str(rule.get("instruction", "")) for rule in rules if rule.get("instruction")),
        )

    def observe(
        self,
        *,
        directive: Dict[str, Any],
        row: Optional[Dict[str, Any]],
        accepted: bool,
        rejection: Optional[Dict[str, Any]] = None,
    ) -> None:
        targets = list(directive.get("target_tools", []))
        motif = str(directive.get("motif", ""))
        context_mode = str(directive.get("context_mode", ""))
        used: List[str] = []
        categories: List[str] = []
        if row:
            conversation = row.get("conversation", {})
            for turn in conversation.get("turns", []):
                for step in turn.get("steps", []):
                    for call in step.get("tool_calls", []):
                        name = str(call.get("tool_name", ""))
                        if name:
                            used.append(name)
                            if name in self.by_name:
                                categories.append(_category(self.by_name[name]))

        def update(payload: Dict[str, Any]) -> None:
            key = "accepted_rows" if accepted else "rejected_rows"
            payload[key] = int(payload.get(key, 0)) + 1
            payload.setdefault("context_mode_counts", {})[context_mode] = int(payload.setdefault("context_mode_counts", {}).get(context_mode, 0)) + 1
            if motif:
                payload.setdefault("motif_counts", {})[motif] = int(payload.setdefault("motif_counts", {}).get(motif, 0)) + 1
            if accepted:
                for name in set(used):
                    payload.setdefault("used_counts", {})[name] = int(payload.setdefault("used_counts", {}).get(name, 0)) + 1
                for name in targets:
                    if name in used:
                        payload.setdefault("target_counts", {})[name] = int(payload.setdefault("target_counts", {}).get(name, 0)) + 1
                    else:
                        payload.setdefault("target_miss_counts", {})[name] = int(payload.setdefault("target_miss_counts", {}).get(name, 0)) + 1
                for category in set(categories):
                    payload.setdefault("category_counts", {})[category] = int(payload.setdefault("category_counts", {}).get(category, 0)) + 1
                for left, right in combinations_sorted(set(categories)):
                    pair = f"{left}||{right}"
                    payload.setdefault("category_pair_counts", {})[pair] = int(payload.setdefault("category_pair_counts", {}).get(pair, 0)) + 1
            else:
                code = str((rejection or {}).get("code", "UNKNOWN"))
                payload.setdefault("rejection_counts", {})[code] = int(payload.setdefault("rejection_counts", {}).get(code, 0)) + 1

        self.state.transaction(update)

    def is_complete(self, minimum_per_tool: int = 1) -> bool:
        snapshot = self.state.transaction()
        return all(
            int(snapshot.get("used_counts", {}).get(name, 0)) >= minimum_per_tool
            for name in self.tool_names
        )

    def missing_tools(self, minimum_per_tool: int = 1) -> List[str]:
        snapshot = self.state.transaction()
        return [
            name
            for name in self.tool_names
            if int(snapshot.get("used_counts", {}).get(name, 0)) < minimum_per_tool
        ]


def combinations_sorted(values: Iterable[str]) -> Iterable[Tuple[str, str]]:
    ordered = sorted(set(values))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            yield left, right


def candidate_descriptors(row: Dict[str, Any]) -> Dict[str, Any]:
    turns = row.get("conversation", {}).get("turns", [])
    calls = []
    categories = set(row.get("conversation", {}).get("categories_used", []))
    direct_arguments = 0
    bound_arguments = 0
    mutation_steps = 0
    max_turn_calls = 0
    for turn in turns:
        turn_calls = 0
        for step in turn.get("steps", []):
            if step.get("pre_state") != step.get("post_state"):
                mutation_steps += 1
            for call in step.get("tool_calls", []):
                calls.append(str(call.get("tool_name", "")))
                turn_calls += 1
            provenance = step.get("quality_verification", {}).get("argument_provenance", {})
            stack = [provenance]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    source = value.get("source")
                    if source in {"tool_output", "history"}:
                        bound_arguments += 1
                    elif source in {"user", "literal", "visible_context"}:
                        direct_arguments += 1
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        max_turn_calls = max(max_turn_calls, turn_calls)
    total_sources = direct_arguments + bound_arguments
    return {
        "turns": len(turns),
        "tool_calls": len(calls),
        "unique_tools": len(set(calls)),
        "tools": sorted(set(calls)),
        "categories": sorted(categories),
        "output_or_history_bound_arguments": bound_arguments,
        "direct_argument_fraction": (
            direct_arguments / total_sources if total_sources else None
        ),
        "mutation_steps": mutation_steps,
        "max_calls_in_one_turn": max_turn_calls,
        "serialized_bytes": len(json.dumps(row, ensure_ascii=False, default=str).encode("utf-8")),
        "advisory_only": True,
    }
