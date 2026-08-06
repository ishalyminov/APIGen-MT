from apigen_multi_turn import MultiTurnConversation, MultiTurnGenerator, Turn
from apigen_step_by_step import ToolCallWithOutput, TrajectoryStep
from unittest.mock import MagicMock
import json


def _repeated_lookup_step():
    return TrajectoryStep(
        step_number=1,
        tool_calls=[
            ToolCallWithOutput(
                tool_name="lookup",
                arguments={"name": "first"},
                output={"id": "ID-1"},
            ),
            ToolCallWithOutput(
                tool_name="lookup",
                arguments={"name": "second"},
                output={"id": "ID-2"},
            ),
        ],
    )


def test_aggregate_turn_outputs_preserves_repeated_calls():
    aggregate = MultiTurnGenerator._aggregate_turn_outputs([_repeated_lookup_step()])

    assert [call["output"]["id"] for call in aggregate["calls"]] == ["ID-1", "ID-2"]
    assert [call["arguments"]["name"] for call in aggregate["calls"]] == [
        "first",
        "second",
    ]
    assert aggregate["by_tool"]["lookup"] == [{"id": "ID-1"}, {"id": "ID-2"}]
    assert aggregate["lookup"] == [{"id": "ID-1"}, {"id": "ID-2"}]


def test_indexed_placeholder_resolves_and_unindexed_remains_ambiguous():
    conversation = MultiTurnConversation(
        turns=[
            Turn(
                turn_number=1,
                user_query="Look up both records.",
                steps=[_repeated_lookup_step()],
                expected_tools=["lookup", "lookup"],
            )
        ]
    )
    generator = object.__new__(MultiTurnGenerator)

    indexed = generator._resolve_turn_placeholders(
        "Use {{TURN1.lookup[1].id}}.",
        turn_index=1,
        conversation=conversation,
    )
    ambiguous = generator._resolve_turn_placeholders(
        "Use {{TURN1.lookup.id}}.",
        turn_index=1,
        conversation=conversation,
    )

    assert indexed == "Use ID-2."
    assert ambiguous == "Use {{TURN1.lookup.id}}."


def test_turn_query_repair_preserves_concrete_literals():
    llm = MagicMock()
    llm.generate.return_value = (
        '{"query":"Please compare report.txt with its backup on 2026-07-29."}'
    )
    manager = MagicMock()
    manager.python_tool_instances = {}
    manager.get_tool_schema.return_value = {
        "name": "diff",
        "parameters": {"type": "object"},
    }
    generator = MultiTurnGenerator(llm, manager)

    repaired = generator._repair_turn_query(
        user_query="Compare report.txt on 2026-07-29 and also estimate its cost.",
        expected_tools=["diff"],
        policy_history=[],
        quality_feedback="TOOL_PLAN_CANNOT_FULFILL_QUERY",
    )

    assert repaired == "Please compare report.txt with its backup on 2026-07-29."


def test_cross_turn_aggregates_may_use_an_expanded_visible_dataset():
    generator = object.__new__(MultiTurnGenerator)
    current = [
        TrajectoryStep(
            step_number=1,
            tool_calls=[
                ToolCallWithOutput(
                    tool_name="standard_deviation",
                    arguments={"numbers": [12.5, 18.3, 15.7, 20.1]},
                    output={"result": 2.8},
                )
            ],
        )
    ]
    context = {
        "turn_outputs": [
            {"mean": {"result": 15.5, "input_numbers": [12.5, 18.3, 15.7]}}
        ]
    }

    assert generator._validate_cross_turn_consistency(current, context) == []


def _events_blueprint_generator(turns, action_counts):
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {"overall_task": "Manage a support ticket.", "turns": turns}
    )
    manager = MagicMock()
    manager.api_name_to_class_key = {}
    manager.get_tools_json_schema.return_value = [
        {
            "name": "create_ticket",
            "category": "Events",
            "parameters": {"type": "object"},
            "output_schema": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
            },
        },
        {
            "name": "resolve_ticket",
            "category": "Events",
            "parameters": {"type": "object"},
            "output_schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
            },
        },
    ]
    manager.get_tool_category.return_value = "Events"
    manager.tool_exists.return_value = True
    generator = MultiTurnGenerator(
        llm,
        manager,
        num_turns=2,
        actions_per_turn=2,
        blueprint_max_actions_per_turn=2,
        blueprint_actions_per_turn=action_counts,
    )
    generator._verify_blueprint_capabilities = MagicMock(
        return_value=(True, [])
    )
    generator._validate_posting_api_entities = MagicMock(return_value=[])
    return generator, llm


def test_single_prior_result_allows_natural_cross_turn_reference():
    generator, _ = _events_blueprint_generator(
        [
            {
                "user_query": "Open a ticket for the database timeout.",
                "expected_tools": ["create_ticket"],
            },
            {
                "user_query": "That issue is fixed now; mark the ticket resolved.",
                "expected_tools": ["resolve_ticket"],
            },
        ],
        [1, 1],
    )

    blueprint = generator._stage0_generate_blueprint("Events", {})

    assert blueprint is not None
    assert blueprint.turns[1]["user_query"].startswith("That issue")


def test_multiple_prior_results_require_index_and_use_all_retries():
    generator, llm = _events_blueprint_generator(
        [
            {
                "user_query": "Open separate tickets for both failures.",
                "expected_tools": ["create_ticket", "create_ticket"],
            },
            {
                "user_query": "Mark that ticket resolved.",
                "expected_tools": ["resolve_ticket"],
            },
        ],
        [2, 1],
    )

    blueprint = generator._stage0_generate_blueprint("Events", {})

    assert blueprint is None
    assert llm.generate.call_count == 2


def test_capability_judge_receives_output_schema_shape():
    llm = MagicMock()
    llm.generate.return_value = '{"is_valid": true, "issues": []}'
    manager = MagicMock()
    manager.get_tool_schema.return_value = {
        "name": "get_user_tickets",
        "description": "Get the user's tickets.",
        "parameters": {"type": "object", "properties": {}},
        "output_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        },
    }
    generator = MultiTurnGenerator(llm, manager)

    valid, issues = generator._verify_blueprint_capabilities(
        [
            {
                "user_query": "Show my ticket.",
                "expected_tools": ["get_user_tickets"],
            }
        ],
        "Events",
        {},
    )

    prompt = llm.generate.call_args.args[0][0]["content"]
    assert valid is True
    assert issues == []
    assert '"output_schema"' in prompt
    assert "single object is not a list" in prompt


def test_posting_entity_validator_ignores_followup_pronouns():
    generator = object.__new__(MultiTurnGenerator)
    state = {
        "posting_api": {
            "username": "foodie_chef",
            "users": {"food_critic": {}},
            "following_list": ["food_critic"],
            "tweets": {},
            "comments": {},
            "retweets": [],
        }
    }

    issues = generator._validate_posting_api_entities(
        [
            {
                "user_query": (
                    "Retweet the result, then check whether I am following "
                    "them already."
                ),
                "expected_tools": ["retweet", "follow_user"],
            }
        ],
        state,
    )

    assert issues == []


def test_posting_entity_validator_still_rejects_unknown_concrete_username():
    generator = object.__new__(MultiTurnGenerator)
    state = {
        "posting_api": {
            "username": "foodie_chef",
            "users": {"food_critic": {}},
            "following_list": ["food_critic"],
            "tweets": {},
            "comments": {},
            "retweets": [],
        }
    }

    issues = generator._validate_posting_api_entities(
        [
            {
                "user_query": "Follow nonexistent_handle.",
                "expected_tools": ["follow_user"],
            }
        ],
        state,
    )

    assert len(issues) == 1
    assert "nonexistent_handle" in issues[0]
