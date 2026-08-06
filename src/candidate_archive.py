"""Lossless candidate archival and post-hoc trajectory descriptors.

This module is intentionally generation-neutral: it does not change prompts,
acceptance criteria, retry policies, or LLM call counts.  It only records what
was produced and computes descriptive metrics after generation.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return copy.deepcopy(value)


def _iter_turns(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    conversation = record.get("conversation")
    if isinstance(conversation, dict):
        turns = conversation.get("turns", [])
        if isinstance(turns, list):
            for turn in turns:
                if isinstance(turn, dict):
                    yield turn
        return

    trajectory = record.get("trajectory")
    if isinstance(trajectory, dict):
        yield {
            "turn_number": 1,
            "user_query": trajectory.get("query", ""),
            "steps": trajectory.get("steps", []),
            "assistant_response": trajectory.get("final_response", ""),
        }


def _step_calls(step: Dict[str, Any]) -> list[Dict[str, Any]]:
    calls = step.get("tool_calls", [])
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _provenance_sources(step: Dict[str, Any]) -> Iterable[str]:
    quality = step.get("quality_verification", {})
    provenance = quality.get("argument_provenance", {}) if isinstance(quality, dict) else {}
    if not isinstance(provenance, dict):
        return
    for source in provenance.values():
        if isinstance(source, dict):
            label = source.get("source") or source.get("kind")
            if label:
                yield str(label)


def _turn_dependency_depth(steps: list[Dict[str, Any]]) -> int:
    """Compute a conservative same-turn depth from cN provenance references."""

    depth_by_step: Dict[int, int] = {}
    best = 0
    for fallback_index, step in enumerate(steps, 1):
        step_index = int(step.get("step_number") or fallback_index)
        parents: list[int] = []
        quality = step.get("quality_verification", {})
        provenance = quality.get("argument_provenance", {}) if isinstance(quality, dict) else {}
        if isinstance(provenance, dict):
            for source in provenance.values():
                if not isinstance(source, dict) or source.get("source") != "tool_output":
                    continue
                call_id = str(source.get("call_id", ""))
                if call_id.startswith("c") and call_id[1:].isdigit():
                    parents.append(int(call_id[1:]))
        parent_depth = max((depth_by_step.get(parent, 1) for parent in parents), default=0)
        depth_by_step[step_index] = max(1, parent_depth + 1)
        best = max(best, depth_by_step[step_index])
    return best


def compute_posthoc_descriptors(payload: Any) -> Dict[str, Any]:
    """Describe a generated candidate without accepting or rejecting it.

    These values are diagnostics/curriculum metadata only.  No threshold in
    this module is an admission gate.
    """

    record = _jsonable(payload)
    if not isinstance(record, dict):
        return {}

    turns = list(_iter_turns(record))
    total_steps = 0
    total_calls = 0
    unique_tools: set[str] = set()
    categories: set[str] = set()
    parallel_groups = 0
    mutation_steps = 0
    source_counts: Dict[str, int] = {}
    max_same_turn_dependency_depth = 0
    state_payload_bytes = 0

    for turn in turns:
        steps = [step for step in turn.get("steps", []) if isinstance(step, dict)]
        total_steps += len(steps)
        max_same_turn_dependency_depth = max(
            max_same_turn_dependency_depth,
            _turn_dependency_depth(steps),
        )
        for step in steps:
            calls = _step_calls(step)
            total_calls += len(calls)
            if len(calls) > 1 or step.get("execution_mode") == "parallel":
                parallel_groups += 1
            for call in calls:
                tool_name = call.get("tool_name") or call.get("name")
                if tool_name:
                    unique_tools.add(str(tool_name))
            pre_state = step.get("pre_state")
            post_state = step.get("post_state")
            if pre_state is not None or post_state is not None:
                state_payload_bytes += len(
                    json.dumps(
                        {"pre": pre_state, "post": post_state},
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            if pre_state is not None and post_state is not None and pre_state != post_state:
                mutation_steps += 1
            for source in _provenance_sources(step):
                source_counts[source] = source_counts.get(source, 0) + 1

    conversation = record.get("conversation", {})
    if isinstance(conversation, dict):
        for category in conversation.get("categories_used", []) or []:
            categories.add(str(category))
    trajectory = record.get("trajectory", {})
    if isinstance(trajectory, dict):
        for category in trajectory.get("categories_used", []) or []:
            categories.add(str(category))

    sourced_arguments = sum(source_counts.values())
    direct_sources = sum(
        source_counts.get(name, 0)
        for name in ("user", "literal", "visible_context")
    )
    output_sources = source_counts.get("tool_output", 0)
    history_sources = source_counts.get("history", 0)

    return {
        "advisory_only": True,
        "turns": len(turns),
        "steps": total_steps,
        "tool_calls": total_calls,
        "unique_tools": len(unique_tools),
        "tool_names": sorted(unique_tools),
        "categories": sorted(categories),
        "parallel_groups": parallel_groups,
        "mutation_steps": mutation_steps,
        "argument_source_counts": dict(sorted(source_counts.items())),
        "same_turn_output_bound_arguments": output_sources,
        "declared_history_arguments": history_sources,
        "output_or_history_bound_arguments": output_sources + history_sources,
        "direct_argument_fraction": (
            direct_sources / sourced_arguments if sourced_arguments else None
        ),
        "max_same_turn_dependency_depth": max_same_turn_dependency_depth,
        "embedded_state_payload_bytes": state_payload_bytes,
    }


def build_partial_candidate_record(
    checkpoint_state: Dict[str, Any],
    *,
    candidate_id: str,
    usage: Dict[str, Any],
    rejection: Dict[str, Any],
    available_tools: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convert an in-memory/checkpoint snapshot into a rejected candidate row."""

    conversation = copy.deepcopy(checkpoint_state.get("partial_conversation", {}))
    completed_turns = int(checkpoint_state.get("completed_turns", 0) or 0)
    blueprint = checkpoint_state.get("blueprint", {})
    expected_turns = (
        int(blueprint.get("num_turns", 0) or 0)
        if isinstance(blueprint, dict)
        else 0
    )
    record: Dict[str, Any] = {
        "candidate_id": candidate_id,
        "disposition": "rejected",
        "candidate_complete": bool(
            expected_turns > 0 and completed_turns >= expected_turns
        ),
        "rejection": copy.deepcopy(rejection),
        "conversation": conversation,
        "generation_metadata": {
            "focus_category": checkpoint_state.get("focus_category"),
            "completed_turns": completed_turns,
            "blueprint": copy.deepcopy(blueprint),
            "generation_directive": copy.deepcopy(
                checkpoint_state.get("generation_directive", {})
            ),
        },
        "token_usage": copy.deepcopy(usage),
        "execution_context": copy.deepcopy(
            checkpoint_state.get("execution_context", {})
        ),
        "initial_api_state": copy.deepcopy(checkpoint_state.get("initial_api_state")),
        "available_tools": copy.deepcopy(available_tools or []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    record["posthoc_descriptors"] = compute_posthoc_descriptors(record)
    return record


def decorate_full_candidate_record(
    payload: Any,
    *,
    candidate_id: str,
    disposition: str,
    usage: Dict[str, Any],
    rejection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = _jsonable(payload)
    if not isinstance(record, dict):
        raise TypeError("candidate payload must serialize to a dictionary")
    record = copy.deepcopy(record)
    record["candidate_id"] = candidate_id
    record["disposition"] = disposition
    record["candidate_complete"] = True
    record["candidate_usage"] = copy.deepcopy(usage)
    if rejection is not None:
        record["rejection"] = copy.deepcopy(rejection)
    record["posthoc_descriptors"] = compute_posthoc_descriptors(record)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    return record


class CandidateArchive:
    """Append-only accepted/rejected candidate archive.

    The archive is optional.  When disabled, the generator follows the existing
    output path and behavior exactly.
    """

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser()
        self.accepted_dir = self.root / "accepted"
        self.rejected_dir = self.root / "rejected"
        self.accepted_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_path = self.accepted_dir / "candidates.jsonl"
        self.rejected_path = self.rejected_dir / "candidates.jsonl"
        self.events_path = self.root / "candidate_events.jsonl"

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def write(self, record: Dict[str, Any]) -> None:
        disposition = record.get("disposition")
        if disposition == "accepted":
            destination = self.accepted_path
        elif disposition == "rejected":
            destination = self.rejected_path
        else:
            raise ValueError(f"unknown candidate disposition: {disposition!r}")
        self._append_jsonl(destination, record)
        self._append_jsonl(
            self.events_path,
            {
                "candidate_id": record.get("candidate_id"),
                "disposition": disposition,
                "candidate_complete": record.get("candidate_complete"),
                "rejection": record.get("rejection"),
                "candidate_usage": record.get("candidate_usage") or record.get("token_usage"),
                "posthoc_descriptors": record.get("posthoc_descriptors"),
                "timestamp": record.get("timestamp"),
            },
        )
