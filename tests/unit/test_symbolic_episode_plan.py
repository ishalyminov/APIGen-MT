import copy
import json
from unittest.mock import MagicMock

import pytest

from apigen_multi_turn import MultiTurnGenerator
from apigen_step_by_step import QueryGenerationResult


class NoCallLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        raise AssertionError("symbolic execution must not call an LLM compiler")

    def get_token_usage(self):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": self.calls,
        }


class SymbolicToolManager:
    def __init__(self):
        self.python_tool_instances = {"demo": object()}
        self.api_name_to_class_key = {
            "lookup": "demo",
            "save": "demo",
        }
        self.saved = []
        self.schemas = {
            "lookup": {
                "name": "lookup",
                "description": "Look up a record by its natural name.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "The record identifier.",
                "output_schema": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            },
            "save": {
                "name": "save",
                "description": "Save the record by identifier.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "Save status.",
                "output_schema": {
                    "type": "object",
                    "properties": {"success": {"type": "boolean"}},
                    "required": ["success"],
                },
            },
            "edit": {
                "name": "edit",
                "description": "Edit selected record fields.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "object",
                            "properties": {
                                "priority": {"type": "string"},
                                "owner": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                    "required": ["updates"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "Edit status.",
                "output_schema": {
                    "type": "object",
                    "properties": {"success": {"type": "boolean"}},
                },
            },
            "charge": {
                "name": "charge",
                "description": "Apply a numeric amount.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_schema": {
                    "type": "object",
                    "properties": {"success": {"type": "boolean"}},
                },
            },
        }
        self.api_name_to_class_key["edit"] = "demo"
        self.api_name_to_class_key["charge"] = "demo"

    def get_tool_schema(self, name):
        return copy.deepcopy(self.schemas[name])

    def get_tools_json_schema(self):
        return [copy.deepcopy(schema) for schema in self.schemas.values()]

    def tool_exists(self, name):
        return name in self.schemas

    def get_api_state(self):
        return {"demo": {"saved": copy.deepcopy(self.saved)}}

    def restore_api_state(self, state):
        self.saved = copy.deepcopy(state["demo"]["saved"])

    def has_python_implementation(self, name):
        return name in self.schemas

    def invoke_python_tool(self, name, arguments):
        if name == "lookup":
            return {"record_id": f"id:{arguments['name']}"}
        if name == "save":
            return {"success": True}
        if name == "edit":
            return {"success": True}
        if name == "charge":
            return {"success": True}
        raise ValueError(name)


def _generator():
    llm = NoCallLLM()
    generator = MultiTurnGenerator(
        llm,
        SymbolicToolManager(),
        num_turns=2,
        actions_per_turn=2,
        optimized_pipeline=True,
        symbolic_episode_plan=True,
    )
    return generator, llm


def test_symbolic_plan_resolves_cross_turn_output_without_compiler_call():
    generator, llm = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find the record named quarterly report.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {
                                "source": "user",
                                "value": "quarterly report",
                            }
                        },
                    }
                ],
            },
            {
                "user_query": "Great, save that record.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "tool_output",
                                "call_id": "t1c1",
                                "path": "record_id",
                            }
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "save"},
    )

    assert errors == []
    first_query = QueryGenerationResult(
        query=turns[0]["user_query"],
        intent="",
        expected_tools=turns[0]["expected_tools"],
        quality_preflight={"passed": True},
    )
    first, context = generator._execute_symbolic_blueprint_turn(
        query_result=first_query,
        turn_spec=turns[0],
        execution_context={},
    )
    assert first is not None
    context["prior_user_queries"] = [first_query.query]
    second_query = QueryGenerationResult(
        query=turns[1]["user_query"],
        intent="",
        expected_tools=turns[1]["expected_tools"],
        quality_preflight={"passed": True},
    )
    second, _ = generator._execute_symbolic_blueprint_turn(
        query_result=second_query,
        turn_spec=turns[1],
        execution_context=context,
    )

    assert llm.calls == 0
    assert second is not None
    assert second[0].tool_calls[0].arguments == {
        "record_id": "id:quarterly report"
    }
    assert second[0].quality_verification["argument_provenance"][
        "record_id"
    ] == {
        "source": "tool_output",
        "call_id": "t1c1",
        "path": "record_id",
    }


def test_symbolic_exact_action_schedule_is_prompted_and_enforced():
    wrong_schedule = {
        "overall_task": "Find two records and stop.",
        "turns": [
            {
                "user_query": "Find alpha and beta.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "alpha"}
                        },
                        "depends_on": [],
                        "parallel_group": None,
                    },
                    {
                        "call_id": "t1c2",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "beta"}
                        },
                        "depends_on": [],
                        "parallel_group": None,
                    },
                ],
            },
            {"user_query": "That is all.", "calls": []},
        ],
    }
    correct_schedule = {
        "overall_task": "Find and save one record.",
        "turns": [
            {
                "user_query": "Find alpha.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "alpha"}
                        },
                        "depends_on": [],
                        "parallel_group": None,
                    }
                ],
            },
            {
                "user_query": "Save that record.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "tool_output",
                                "call_id": "t1c1",
                                "path": "record_id",
                            }
                        },
                        "depends_on": ["t1c1"],
                        "parallel_group": None,
                    }
                ],
            },
        ],
    }
    llm = MagicMock()
    llm.generate.side_effect = [
        json.dumps(wrong_schedule),
        json.dumps(correct_schedule),
        json.dumps(
            {
                "overall_task": "Find and save one record.",
                "turns": [
                    {"turn_number": 1, "user_query": "Find alpha."},
                    {"turn_number": 2, "user_query": "Save that record."},
                ],
            }
        ),
    ]
    generator = MultiTurnGenerator(
        llm,
        SymbolicToolManager(),
        num_turns=2,
        actions_per_turn=2,
        blueprint_max_actions_per_turn=2,
        blueprint_actions_per_turn=[1, 1],
        symbolic_episode_plan=True,
        blueprint_min_total_actions=2,
        blueprint_max_total_actions=2,
    )
    generator._validate_posting_api_entities = MagicMock(return_value=[])
    generator._validate_vehicle_control_queries = MagicMock(return_value=[])
    generator._preflight_symbolic_blueprint_execution = MagicMock(
        return_value=[]
    )
    generator._verify_blueprint_capabilities = MagicMock(
        return_value=(True, [])
    )

    blueprint = generator._stage0_generate_blueprint(None, {})

    assert blueprint is not None
    assert [len(turn["calls"]) for turn in blueprint.turns] == [1, 1]
    assert llm.generate.call_count == 3
    prompt = llm.generate.call_args_list[0].args[0][0]["content"]
    assert "exactly 2 necessary calls" in prompt
    assert "turn 1=1, turn 2=1" in prompt


def test_symbolic_schedule_supports_ten_calls_in_one_turn():
    generator = MultiTurnGenerator(
        NoCallLLM(),
        SymbolicToolManager(),
        num_turns=2,
        actions_per_turn=10,
        blueprint_max_actions_per_turn=10,
        blueprint_actions_per_turn=[10, 5],
        symbolic_episode_plan=True,
        blueprint_min_total_actions=15,
        blueprint_max_total_actions=15,
    )

    assert generator.blueprint_max_actions_per_turn == 10
    assert generator.blueprint_actions_per_turn == [10, 5]


def test_symbolic_prompt_guides_high_width_turns_without_filler():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("stop after capturing prompt")
    generator = MultiTurnGenerator(
        llm,
        SymbolicToolManager(),
        num_turns=2,
        actions_per_turn=10,
        blueprint_max_actions_per_turn=10,
        blueprint_actions_per_turn=[10, 5],
        symbolic_episode_plan=True,
        blueprint_min_total_actions=15,
        blueprint_max_total_actions=15,
    )

    with pytest.raises(RuntimeError, match="LLM generate failed"):
        generator._stage0_generate_blueprint(None, {})
    prompt = llm.generate.call_args.args[0][0]["content"]
    assert "one natural compound user request" in prompt
    assert "do not omit, add, or" in prompt


def test_whole_episode_symbolic_generation_uses_strict_json_schema(monkeypatch):
    monkeypatch.setenv("APIGEN_SYMBOLIC_TURNWISE", "0")
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("stop after capturing request")
    generator = MultiTurnGenerator(
        llm,
        SymbolicToolManager(),
        num_turns=2,
        actions_per_turn=2,
        blueprint_max_actions_per_turn=2,
        blueprint_actions_per_turn=[1, 2],
        symbolic_episode_plan=True,
        blueprint_min_total_actions=3,
        blueprint_max_total_actions=3,
    )

    with pytest.raises(RuntimeError, match="LLM generate failed"):
        generator._stage0_generate_blueprint(None, {})

    response_format = llm.generate.call_args.kwargs["response_format"]
    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert schema["properties"]["turns"]["minItems"] == 2
    calls_schema = schema["properties"]["turns"]["items"]["properties"]["calls"]
    assert calls_schema["minItems"] == 1
    assert calls_schema["maxItems"] == 2


def test_high_width_symbolic_blueprint_compiles_one_turn_per_call():
    def lookup_call(turn, call, name):
        return {
            "call_id": f"t{turn}c{call}",
            "tool_name": "lookup",
            "arguments": {
                "name": {"source": "user", "value": name}
            },
            "depends_on": [],
            "parallel_group": None,
        }

    first_names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    second_names = ["zeta", "eta", "theta", "iota", "kappa"]
    llm = MagicMock()
    llm.generate.side_effect = [
        json.dumps(
            {
                "overall_task": "Look up two related batches of records.",
                "future_turn_intents": ["Look up the second batch."],
                "turn": {
                    "user_query": "Look up alpha, beta, gamma, delta, and epsilon.",
                    "intent": "Find the first five records.",
                    "calls": [
                        lookup_call(1, index, name)
                        for index, name in enumerate(first_names, 1)
                    ],
                },
            }
        ),
        json.dumps(
            {
                "turn": {
                    "user_query": "Now look up zeta, eta, theta, iota, and kappa.",
                    "intent": "Find the second five records.",
                    "calls": [
                        lookup_call(2, index, name)
                        for index, name in enumerate(second_names, 1)
                    ],
                }
            }
        ),
        json.dumps(
            {
                "overall_task": "Find two related groups of records.",
                "turns": [
                    {
                        "turn_number": 1,
                        "user_query": (
                            "Please find alpha, beta, gamma, delta, and epsilon."
                        ),
                    },
                    {
                        "turn_number": 2,
                        "user_query": (
                            "Now find zeta, eta, theta, iota, and kappa."
                        ),
                    },
                ],
            }
        ),
    ]
    generator = MultiTurnGenerator(
        llm,
        SymbolicToolManager(),
        num_turns=2,
        actions_per_turn=5,
        blueprint_max_actions_per_turn=5,
        blueprint_actions_per_turn=[5, 5],
        symbolic_episode_plan=True,
        blueprint_min_total_actions=10,
        blueprint_max_total_actions=10,
    )
    generator._verify_blueprint_capabilities = MagicMock(
        return_value=(True, [])
    )

    blueprint = generator._stage0_generate_blueprint(None, {})

    assert blueprint is not None
    assert [len(turn["calls"]) for turn in blueprint.turns] == [5, 5]
    assert llm.generate.call_count == 3
    assert all(
        call.args[0][0]["content"].startswith("Compile turn")
        for call in llm.generate.call_args_list[:2]
    )
    assert llm.generate.call_args_list[2].args[0][0]["content"].startswith(
        "Rewrite the user-facing language"
    )


def test_symbolic_query_alignment_removes_work_missing_from_fixed_graph():
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "overall_task": "Find one record.",
            "turns": [
                {
                    "turn_number": 1,
                    "user_query": "Please find the record named alpha.",
                }
            ],
        }
    )
    generator = MultiTurnGenerator(
        llm,
        SymbolicToolManager(),
        num_turns=1,
        actions_per_turn=1,
        symbolic_episode_plan=True,
    )
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find alpha and delete it.",
                "intent": "Find and delete a record.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "alpha"}
                        },
                        "depends_on": [],
                        "parallel_group": None,
                    }
                ],
            }
        ],
        available_tool_names={"lookup", "save", "edit", "charge"},
    )
    assert errors == []

    task, aligned, errors = generator._align_symbolic_blueprint_queries(
        overall_task="Find and delete one record.",
        turns=turns,
        tools_json=generator.tool_manager.get_tools_json_schema(),
    )

    assert errors == []
    assert task == "Find one record."
    assert aligned is not None
    assert aligned[0]["user_query"] == "Please find the record named alpha."
    assert "delete" not in aligned[0]["user_query"].lower()


def test_symbolic_preflight_executes_without_llm_and_restores_state():
    generator, llm = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find the record named report.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "report"}
                        },
                    }
                ],
            },
            {
                "user_query": "Save that record.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "tool_output",
                                "call_id": "t1c1",
                                "path": "record_id",
                            }
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "save"},
    )
    assert errors == []
    state_before = generator.tool_manager.get_api_state()
    assert generator._preflight_symbolic_blueprint_execution(turns) == []
    assert generator.tool_manager.get_api_state() == state_before
    assert llm.calls == 0

    broken = copy.deepcopy(turns)
    broken[1]["calls"][0]["arguments"]["record_id"]["path"] = "missing"
    issues = generator._preflight_symbolic_blueprint_execution(broken)
    assert len(issues) == 1
    assert "Turn 2" in issues[0]
    assert generator.tool_manager.get_api_state() == state_before


def test_symbolic_plan_rejects_hidden_literal_and_future_dependency():
    generator, _ = _generator()
    _, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Save my report.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "user",
                                "value": "secret report id",
                            }
                        },
                    }
                ],
            },
            {
                "user_query": "Save it.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "tool_output",
                                "call_id": "t2c2",
                                "path": "record_id",
                            }
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "save"},
    )

    assert any("not visible" in error for error in errors)
    assert any("not an earlier call" in error for error in errors)


def test_symbolic_compiler_rebinds_one_unique_prior_identifier_output():
    generator, _ = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find alpha.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "alpha"}
                        },
                    }
                ],
            },
            {
                "user_query": "Save that record.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "user",
                                "value": "invented-hidden-id",
                            }
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "save"},
    )

    assert errors == []
    assert turns[1]["calls"][0]["arguments"]["record_id"] == {
        "source": "tool_output",
        "call_id": "t1c1",
        "path": "record_id",
    }


def test_symbolic_compiler_naturally_exposes_safe_numeric_constraint():
    generator, _ = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Please apply the charge.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "charge",
                        "arguments": {
                            "amount": {"source": "user", "value": 25}
                        },
                    }
                ],
            }
        ],
        available_tool_names={"charge"},
    )

    assert errors == []
    assert turns[0]["user_query"] == (
        "Please apply the charge. Use an amount of 25."
    )
    assert turns[0]["calls"][0]["arguments"]["amount"] == {
        "source": "user",
        "value": 25,
    }


def test_invalid_file_output_path_reuses_visible_producer_input_only():
    generator, _ = _generator()
    seen_calls = {
        "t1c1": {
            "schema": {
                "output_schema": {
                    "type": "object",
                    "properties": {"success": {"type": "boolean"}},
                }
            },
            "arguments": {
                "destination": {
                    "source": "user",
                    "value": "backup/report.txt",
                },
                "name": {"source": "user", "value": "report"},
            },
        }
    }

    repaired = generator._repair_invalid_output_passthrough(
        spec={
            "source": "tool_output",
            "call_id": "t1c1",
            "path": "destination",
        },
        argument_name="file_name2",
        seen_calls=seen_calls,
    )
    unsafe = generator._repair_invalid_output_passthrough(
        spec={
            "source": "tool_output",
            "call_id": "t1c1",
            "path": "name",
        },
        argument_name="record_id",
        seen_calls=seen_calls,
    )

    assert repaired == {
        "source": "user",
        "value": "backup/report.txt",
    }
    assert unsafe["source"] == "tool_output"


def test_symbolic_plan_preserves_explicit_non_data_dependencies():
    generator, _ = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find the record and then save report.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "report"}
                        },
                        "depends_on": [],
                    },
                    {
                        "call_id": "t1c2",
                        "tool_name": "save",
                        "arguments": {
                            "record_id": {
                                "source": "user",
                                "value": "report",
                            }
                        },
                        # This edge represents sequencing rather than a
                        # tool-output argument and must not be discarded.
                        "depends_on": ["t1c1"],
                    },
                ],
            },
            {
                "user_query": "Find it again.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "history", "value": "report"}
                        },
                        "depends_on": ["t1c2"],
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "save"},
    )

    assert errors == []
    assert turns[0]["calls"][1]["depends_on"] == ["t1c1"]
    assert turns[1]["calls"][0]["depends_on"] == ["t1c2"]


def test_symbolic_plan_rejects_declared_future_order_dependency():
    generator, _ = _generator()
    _, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find report.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "report"}
                        },
                        "depends_on": ["t1c2"],
                    },
                    {
                        "call_id": "t1c2",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "report"}
                        },
                    },
                ],
            },
            {
                "user_query": "Find report once more.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "history", "value": "report"}
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup"},
    )

    assert any("dependency 't1c2' is not an earlier call" in e for e in errors)


def test_posting_entity_validation_uses_symbolic_arguments_not_query_words():
    generator, _ = _generator()
    state = {
        "posting_api": {
            "users": {"techguru": {}, "foodie_chef": {}},
            "following_list": [],
        }
    }
    valid_turns = [
        {
            "user_query": "Follow both now, then check whether I'm following them.",
            "expected_tools": ["follow_user"],
            "calls": [
                {
                    "tool_name": "follow_user",
                    "arguments": {
                        "username_to_follow": {
                            "source": "user",
                            "value": "techguru",
                        }
                    },
                }
            ],
        }
    ]
    assert generator._validate_posting_api_entities(valid_turns, state) == []

    mention_turn = copy.deepcopy(valid_turns)
    mention_turn[0]["calls"][0] = {
        "tool_name": "mention",
        "arguments": {
            "mentioned_usernames": {
                "source": "user",
                "value": ["@techguru"],
            }
        },
    }
    assert generator._validate_posting_api_entities(mention_turn, state) == []

    invalid_turns = copy.deepcopy(valid_turns)
    invalid_turns[0]["calls"][0]["arguments"]["username_to_follow"][
        "value"
    ] = "invented_user"
    issues = generator._validate_posting_api_entities(invalid_turns, state)
    assert len(issues) == 1
    assert "invented_user" in issues[0]


def test_symbolic_plan_metrics_count_policy_visible_bindings():
    metrics = MultiTurnGenerator._symbolic_plan_metrics(
        [
            {
                "calls": [
                    {
                        "call_id": "t1c1",
                        "arguments": {
                            "name": {"source": "user", "value": "report"}
                        },
                    }
                ]
            },
            {
                "calls": [
                    {
                        "call_id": "t2c1",
                        "arguments": {
                            "record_id": {
                                "source": "tool_output",
                                "call_id": "t1c1",
                                "path": "record_id",
                            }
                        },
                    }
                ]
            },
        ]
    )

    assert metrics["total_arguments"] == 2
    assert metrics["hidden_argument_count"] == 0
    assert metrics["tool_output_binding_count"] == 1
    assert metrics["cross_turn_binding_count"] == 1


def test_nested_object_fields_may_have_independent_provenance():
    generator, llm = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Find quarterly report.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {
                                "source": "user",
                                "value": "quarterly report",
                            }
                        },
                    }
                ],
            },
            {
                "user_query": "Set its priority to high and owner to Alex.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "edit",
                        "arguments": {
                            "updates": {
                                "priority": {
                                    "source": "user",
                                    "value": "high",
                                },
                                "owner": {
                                    "source": "user",
                                    "value": "Alex",
                                },
                            }
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "edit"},
    )

    assert errors == []
    query = QueryGenerationResult(
        query=turns[1]["user_query"],
        intent="",
        expected_tools=["edit"],
        quality_preflight={"passed": True},
    )
    trajectory, _ = generator._execute_symbolic_blueprint_turn(
        query_result=query,
        turn_spec=turns[1],
        execution_context={},
    )

    assert llm.calls == 0
    assert trajectory is not None
    assert trajectory[0].tool_calls[0].arguments == {
        "updates": {"priority": "high", "owner": "Alex"}
    }
    provenance = trajectory[0].quality_verification["argument_provenance"]
    assert provenance["updates"]["source"] == "composite"


def test_nested_safe_literals_are_exposed_without_an_llm_retry():
    generator, _ = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Please update the record.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "edit",
                        "arguments": {
                            "updates": {
                                "priority": {
                                    "source": "user",
                                    "value": "high",
                                },
                                "owner": {
                                    "source": "user",
                                    "value": "Alex",
                                },
                            }
                        },
                    }
                ],
            },
            {
                "user_query": "Find report.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "lookup",
                        "arguments": {
                            "name": {"source": "user", "value": "report"}
                        },
                    }
                ],
            },
        ],
        available_tool_names={"lookup", "edit"},
    )

    assert errors == []
    assert "Set the priority to high." in turns[0]["user_query"]
    assert "Assign it to Alex." in turns[0]["user_query"]


def test_raw_visible_composite_and_schema_default_are_safely_normalised():
    generator, _ = _generator()
    turns, errors = generator._normalise_symbolic_blueprint_turns(
        [
            {
                "user_query": "Set priority high and owner Alex.",
                "calls": [
                    {
                        "call_id": "t1c1",
                        "tool_name": "edit",
                        "arguments": {
                            "updates": {"priority": "high", "owner": "Alex"},
                            "labels": [],
                        },
                    }
                ],
            },
            {
                "user_query": "Find report.",
                "calls": [
                    {
                        "call_id": "t2c1",
                        "tool_name": "lookup",
                        "arguments": {"name": "report"},
                    }
                ],
            },
        ],
        available_tool_names={"edit", "lookup"},
    )

    assert errors == []
    assert turns[0]["calls"][0]["arguments"] == {
        "updates": {
            "source": "user",
            "value": {"priority": "high", "owner": "Alex"},
        },
        "labels": {"source": "schema_default"},
    }
    assert turns[1]["calls"][0]["arguments"]["name"] == {
        "source": "user",
        "value": "report",
    }


def test_tool_output_numeric_string_is_coerced_for_integer_consumer():
    generator, _ = _generator()

    value, provenance = generator._materialise_argument_source(
        spec={"source": "tool_output", "call_id": "t1c1", "path": "id"},
        schema={"type": "integer"},
        query="Delete that message.",
        policy_context={},
        call_outputs={"t1c1": {"id": "9"}},
    )

    assert value == 9
    assert provenance["coercion"] == "string_to_integer"
