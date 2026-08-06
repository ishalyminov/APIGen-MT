import json
from pathlib import Path

from candidate_archive import (
    CandidateArchive,
    build_partial_candidate_record,
    compute_posthoc_descriptors,
    decorate_full_candidate_record,
)


def _step(number, tool, *, source=None, mutates=False):
    quality = {"argument_provenance": {}}
    if source is not None:
        quality["argument_provenance"]["value"] = source
    pre = {"api": {"value": number}}
    post = {"api": {"value": number + 1}} if mutates else pre
    return {
        "step_number": number,
        "tool_calls": [
            {"tool_name": tool, "arguments": {}, "output": {"value": number}}
        ],
        "pre_state": pre,
        "post_state": post,
        "quality_verification": quality,
    }


def test_descriptors_are_advisory_and_count_dependencies():
    record = {
        "conversation": {
            "turns": [
                {
                    "steps": [
                        _step(1, "lookup", source={"source": "user"}),
                        _step(
                            2,
                            "consume",
                            source={
                                "source": "tool_output",
                                "call_id": "c1",
                                "path": "value",
                            },
                            mutates=True,
                        ),
                    ]
                }
            ],
            "categories_used": ["Example"],
        }
    }
    metrics = compute_posthoc_descriptors(record)
    assert metrics["advisory_only"] is True
    assert metrics["tool_calls"] == 2
    assert metrics["same_turn_output_bound_arguments"] == 1
    assert metrics["max_same_turn_dependency_depth"] == 2
    assert metrics["mutation_steps"] == 1


def test_partial_candidate_keeps_rejection_and_usage():
    checkpoint = {
        "partial_conversation": {
            "overall_task": "task",
            "turns": [{"turn_number": 1, "steps": [_step(1, "lookup")] }],
        },
        "blueprint": {"overall_task": "task", "turns": []},
        "completed_turns": 1,
        "focus_category": "Example",
        "generation_directive": {"directive_id": "directive-1"},
        "initial_api_state": {"api": {}},
    }
    row = build_partial_candidate_record(
        checkpoint,
        candidate_id="candidate-1",
        usage={"total_llm_calls": 4, "cost_usd": 0.01},
        rejection={"code": "TURN_RESPONSE_GROUNDING_FAILED"},
        available_tools=[],
    )
    assert row["disposition"] == "rejected"
    assert row["candidate_complete"] is False
    assert row["token_usage"]["total_llm_calls"] == 4
    assert row["generation_metadata"]["generation_directive"] == {
        "directive_id": "directive-1"
    }
    assert row["posthoc_descriptors"]["tool_calls"] == 1


def test_full_candidate_decoration_does_not_gate_complexity():
    payload = {
        "conversation": {
            "overall_task": "simple but valid",
            "turns": [{"turn_number": 1, "steps": [_step(1, "lookup")]}],
        }
    }
    row = decorate_full_candidate_record(
        payload,
        candidate_id="candidate-2",
        disposition="accepted",
        usage={"total_llm_calls": 2},
    )
    assert row["disposition"] == "accepted"
    assert row["posthoc_descriptors"]["advisory_only"] is True
    assert "rejection" not in row


def test_archive_separates_accepted_and_rejected(tmp_path: Path):
    archive = CandidateArchive(tmp_path)
    archive.write(
        {
            "candidate_id": "a",
            "disposition": "accepted",
            "candidate_complete": True,
            "timestamp": "now",
        }
    )
    archive.write(
        {
            "candidate_id": "r",
            "disposition": "rejected",
            "candidate_complete": False,
            "timestamp": "now",
        }
    )
    accepted = [json.loads(line) for line in archive.accepted_path.read_text().splitlines()]
    rejected = [json.loads(line) for line in archive.rejected_path.read_text().splitlines()]
    events = [json.loads(line) for line in archive.events_path.read_text().splitlines()]
    assert [row["candidate_id"] for row in accepted] == ["a"]
    assert [row["candidate_id"] for row in rejected] == ["r"]
    assert len(events) == 2
