import json

from apigen_multi_turn import MultiTurnConversation, MultiTurnGenerator, Turn
from apigen_step_by_step import TrajectoryStep, ToolCallWithOutput
from refuse_parallel_eval import build_multiturn_evaluation_spec


class QueueLLM:
    def __init__(self, responses, model="test-model"):
        self.responses = list(responses)
        self.calls = 0
        self.api_model = model

    def generate(self, messages, **kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)

    def get_token_usage(self):
        return {
            "prompt_tokens": self.calls * 10,
            "completion_tokens": self.calls * 5,
            "total_tokens": self.calls * 15,
            "total_calls": self.calls,
            "total_attempts": self.calls,
        }


class MinimalToolManager:
    python_tool_instances = {}

    def get_tool_schema(self, name):
        return {
            "name": name,
            "description": "Return a visible value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            "output_description": "The requested value.",
        }


def _step(number, value):
    return TrajectoryStep(
        step_number=number,
        tool_calls=[
            ToolCallWithOutput(
                tool_name="lookup",
                arguments={"key": value.lower()},
                output={"value": value},
            )
        ],
        quality_verification={"passed": True},
    )


def _deferred_quality():
    return {
        "passed": True,
        "query_preflight": {"passed": True},
        "transition_checks": [{"passed": True}],
        "final_response_grounding": {
            "passed": True,
            "deferred": True,
            "validator": "deferred_batched_turn_response",
        },
    }


def test_optimized_turn_response_is_deferred_without_placeholder():
    llm = QueueLLM([])
    generator = MultiTurnGenerator(
        llm_client=llm,
        tool_manager=MinimalToolManager(),
        num_turns=2,
        actions_per_turn=1,
        optimized_pipeline=True,
    )

    response, quality = generator._produce_turn_response(
        turn_index=0,
        total_turns=2,
        query="Look up alpha.",
        trajectory=[_step(1, "Alpha")],
        execution_context={},
    )

    assert response == ""
    assert quality["deferred"] is True
    assert quality["response_required"] is True
    assert llm.calls == 0


def test_hard_required_tools_are_checked_against_blueprint():
    turns = [
        {"expected_tools": ["lookup", "summarize"]},
        {"expected_tools": ["notify"]},
    ]
    assert MultiTurnGenerator._missing_hard_required_tools(
        turns,
        {"hard_required_tools": ["lookup", "notify"]},
    ) == []
    assert MultiTurnGenerator._missing_hard_required_tools(
        turns,
        {"hard_required_tools": ["lookup", "delete"]},
    ) == ["delete"]


def test_all_turn_responses_use_one_writer_and_one_grounding_call():
    main = QueueLLM([])
    writer = QueueLLM(
        [
            json.dumps(
                {
                    "responses": [
                        {"turn_number": 1, "response": "Alpha is A."},
                        {"turn_number": 2, "response": "Beta is B."},
                    ]
                }
            )
        ],
        model="writer",
    )
    judge = QueueLLM(
        [
            json.dumps(
                {
                    "turns": [
                        {
                            "turn_number": 1,
                            "is_grounded": True,
                            "issue_codes": [],
                        },
                        {
                            "turn_number": 2,
                            "is_grounded": True,
                            "issue_codes": [],
                        },
                    ]
                }
            )
        ],
        model="judge",
    )
    generator = MultiTurnGenerator(
        llm_client=main,
        tool_manager=MinimalToolManager(),
        num_turns=2,
        actions_per_turn=1,
        optimized_pipeline=True,
    )
    generator.configure_final_stage_clients(
        final_response_client=writer,
        grounding_client=judge,
    )

    conversation = MultiTurnConversation(
        overall_task="Look up two values.",
        turns=[
            Turn(
                turn_number=1,
                user_query="Look up alpha.",
                steps=[_step(1, "A")],
                assistant_response=(
                    "The requested actions completed successfully."
                ),
                quality_verification=_deferred_quality(),
            ),
            Turn(
                turn_number=2,
                user_query="Now look up beta.",
                steps=[_step(1, "B")],
                assistant_response="",
                quality_verification=_deferred_quality(),
            ),
        ],
    )

    assert generator._finalize_deferred_turn_responses(conversation) is True
    assert writer.calls == 1
    assert judge.calls == 1
    assert [turn.assistant_response for turn in conversation.turns] == [
        "Alpha is A.",
        "Beta is B.",
    ]
    assert all(
        turn.quality_verification["passed"]
        for turn in conversation.turns
    )
    assert all(
        turn.quality_verification["final_response_grounding"]["passed"]
        for turn in conversation.turns
    )

    spec = build_multiturn_evaluation_spec(
        conversation,
        available_tools=[{"name": "lookup"}],
    )
    serialized = json.dumps(spec)
    assert "The requested actions completed successfully." not in serialized
    # Turn 1 response is policy-visible to turn 2. The final response remains
    # in the conversation record rather than a future transition context.
    assert "Alpha is A." in serialized
    assert conversation.turns[-1].assistant_response == "Beta is B."


def test_grounding_failure_rejects_the_episode():
    writer = QueueLLM(
        [
            json.dumps(
                {
                    "responses": [
                        {"turn_number": 1, "response": "Alpha is Z."},
                        {"turn_number": 2, "response": "Beta is B."},
                    ]
                }
            )
        ]
    )
    judge = QueueLLM(
        [
            json.dumps(
                {
                    "turns": [
                        {
                            "turn_number": 1,
                            "is_grounded": False,
                            "issue_codes": ["CONTRADICTS_TOOL_OUTPUT"],
                        },
                        {
                            "turn_number": 2,
                            "is_grounded": True,
                            "issue_codes": [],
                        },
                    ]
                }
            )
        ]
    )
    generator = MultiTurnGenerator(
        llm_client=QueueLLM([]),
        tool_manager=MinimalToolManager(),
        num_turns=2,
        actions_per_turn=1,
        optimized_pipeline=True,
    )
    generator.configure_final_stage_clients(
        final_response_client=writer,
        grounding_client=judge,
    )
    conversation = MultiTurnConversation(
        turns=[
            Turn(
                turn_number=1,
                user_query="Look up alpha.",
                steps=[_step(1, "A")],
                quality_verification=_deferred_quality(),
            ),
            Turn(
                turn_number=2,
                user_query="Look up beta.",
                steps=[_step(1, "B")],
                quality_verification=_deferred_quality(),
            ),
        ]
    )

    assert generator._finalize_deferred_turn_responses(conversation) is False
    assert conversation.turns[0].assistant_response == "Alpha is Z."
    assert conversation.turns[0].quality_verification["passed"] is False
    assert conversation.turns[1].quality_verification["passed"] is True


def test_writer_cannot_reintroduce_the_legacy_placeholder():
    writer = QueueLLM(
        [
            json.dumps(
                {
                    "responses": [
                        {
                            "turn_number": 1,
                            "response": (
                                "The requested actions completed successfully."
                            ),
                        },
                        {"turn_number": 2, "response": "Beta is B."},
                    ]
                }
            )
        ]
    )
    judge = QueueLLM([])
    generator = MultiTurnGenerator(
        llm_client=QueueLLM([]),
        tool_manager=MinimalToolManager(),
        num_turns=2,
        actions_per_turn=1,
        optimized_pipeline=True,
    )
    generator.configure_final_stage_clients(
        final_response_client=writer,
        grounding_client=judge,
    )
    conversation = MultiTurnConversation(
        turns=[
            Turn(
                turn_number=turn_number,
                user_query=f"Look up value {turn_number}.",
                steps=[_step(1, str(turn_number))],
                quality_verification=_deferred_quality(),
            )
            for turn_number in (1, 2)
        ]
    )

    assert generator._finalize_deferred_turn_responses(conversation) is False
    assert judge.calls == 0
    assert all(not turn.assistant_response for turn in conversation.turns)


def test_empty_response_is_omitted_from_history_formatting():
    generator = MultiTurnGenerator(
        llm_client=QueueLLM([]),
        tool_manager=MinimalToolManager(),
        num_turns=1,
        actions_per_turn=1,
        optimized_pipeline=True,
    )
    conversation = MultiTurnConversation(
        turns=[
            Turn(
                turn_number=1,
                user_query="Look up alpha.",
                steps=[_step(1, "A")],
                assistant_response="",
            )
        ]
    )

    history = generator._format_conversation_history(conversation)
    assert "Assistant:" not in history
