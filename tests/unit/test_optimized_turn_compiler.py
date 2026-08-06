import copy
import json

import pytest

from apigen_step_by_step import (
    GenerationBudgetExceeded,
    QueryGenerationResult,
    StepByStepGenerator,
)


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return self.responses.pop(0)

    def get_token_usage(self):
        return {
            "prompt_tokens": self.calls * 100,
            "completion_tokens": self.calls * 25,
            "total_tokens": self.calls * 125,
            "total_calls": self.calls,
        }


class RetryBudgetLLM:
    def __init__(self):
        self.calls = 0
        self.http_attempts = 0
        self.transport_limits = []

    def generate(self, messages, **kwargs):
        self.calls += 1
        self.http_attempts += 1
        self.transport_limits.append(kwargs["max_retries"])
        return "{}"

    def get_token_usage(self):
        return {
            "prompt_tokens": self.calls,
            "completion_tokens": self.calls,
            "total_tokens": self.calls * 2,
            "total_calls": self.calls,
            "total_attempts": self.http_attempts,
        }


class CompilerToolManager:
    def __init__(self):
        self.value = 0
        self.python_tool_instances = {"demo": object()}
        self.api_name_to_class_key = {
            "lookup": "demo",
            "fillFuelTank": "demo",
        }
        self.schemas = {
            "lookup": {
                "name": "lookup",
                "description": "Return a value for a supplied key.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "A value.",
            },
            "fillFuelTank": {
                "name": "fillFuelTank",
                "description": "Add fuel.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"fuelAmount": {"type": "integer"}},
                    "required": ["fuelAmount"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "The fuel level.",
            },
        }

    def get_tool_schema(self, name):
        return copy.deepcopy(self.schemas[name])

    def get_api_state(self):
        return {"demo": {"value": self.value}}

    def restore_api_state(self, state):
        self.value = copy.deepcopy(state["demo"]["value"])

    def has_python_implementation(self, name):
        return name in self.schemas

    def invoke_python_tool(self, name, arguments):
        if name == "lookup":
            return {"value": f"value:{arguments['key']}"}
        if name == "fillFuelTank":
            self.value += arguments["fuelAmount"]
            return {"fuelLevel": str(self.value)}
        raise ValueError(name)


def _query(tools):
    return QueryGenerationResult(
        query="Add 1 unit of fuel, then look up the resulting level.",
        intent="",
        expected_tools=tools,
        quality_preflight={"passed": True},
    )


def test_one_llm_call_compiles_every_action_in_a_turn():
    llm = QueueLLM(
        [
            json.dumps(
                {
                    "calls": [
                        {
                            "call_id": "c1",
                            "tool_name": "lookup",
                            "arguments": {
                                "key": {"source": "user", "value": "alpha"}
                            },
                        },
                        {
                            "call_id": "c2",
                            "tool_name": "lookup",
                            "arguments": {
                                "key": {
                                    "source": "tool_output",
                                    "call_id": "c1",
                                    "path": "value",
                                }
                            },
                        },
                    ]
                }
            )
        ]
    )
    generator = StepByStepGenerator(
        llm_client=llm,
        tool_manager=CompilerToolManager(),
        num_actions=2,
        validate_outputs=True,
    )
    result = QueryGenerationResult(
        query="Look up alpha, then look up the value returned by that lookup.",
        intent="",
        expected_tools=["lookup", "lookup"],
        quality_preflight={"passed": True},
    )

    trajectory, _ = generator._stage2_generate_tools(result, 2)

    assert llm.calls == 1
    assert trajectory is not None
    assert len(trajectory) == 2
    assert trajectory[1].tool_calls[0].arguments == {"key": "value:alpha"}
    assert (
        trajectory[1]
        .quality_verification["argument_provenance"]["key"]["source"]
        == "tool_output"
    )


def test_one_turn_repair_rolls_back_the_complete_turn():
    first = {
        "calls": [
            {
                "call_id": "c1",
                "tool_name": "fillFuelTank",
                "arguments": {
                    "fuelAmount": {"source": "user", "value": 1}
                },
            },
            {
                "call_id": "c2",
                "tool_name": "lookup",
                "arguments": {
                    "key": {
                        "source": "tool_output",
                        "call_id": "c1",
                        "path": "missing",
                    }
                },
            },
        ]
    }
    repaired = copy.deepcopy(first)
    repaired["calls"][1]["arguments"]["key"]["path"] = "fuelLevel"
    llm = QueueLLM([json.dumps(first), json.dumps(repaired)])
    manager = CompilerToolManager()
    generator = StepByStepGenerator(
        llm_client=llm,
        tool_manager=manager,
        num_actions=2,
        validate_outputs=True,
    )

    trajectory, _ = generator._stage2_generate_tools(
        _query(["fillFuelTank", "lookup"]), 2
    )

    assert trajectory is not None
    assert llm.calls == 2
    assert manager.value == 1
    assert trajectory[1].tool_calls[0].arguments == {"key": "1"}


def test_candidate_budget_stops_a_nested_turn_repair():
    invalid = {
        "calls": [
            {
                "call_id": "c1",
                "tool_name": "lookup",
                "arguments": {
                    "key": {
                        "source": "tool_output",
                        "call_id": "c1",
                        "path": "value",
                    }
                },
            }
        ]
    }
    llm = QueueLLM([json.dumps(invalid), json.dumps(invalid)])
    generator = StepByStepGenerator(
        llm_client=llm,
        tool_manager=CompilerToolManager(),
        num_actions=1,
    )
    generator.max_calls_per_candidate = 1
    generator._capture_initial_usage()

    with pytest.raises(GenerationBudgetExceeded):
        generator._stage2_generate_tools(
            QueryGenerationResult(
                query="Look up alpha.",
                intent="",
                expected_tools=["lookup"],
                quality_preflight={"passed": True},
            ),
            2,
        )

    assert llm.calls == 1


def test_remaining_candidate_budget_caps_inner_http_retries(monkeypatch):
    monkeypatch.setenv("APIGEN_HTTP_ATTEMPTS", "3")
    llm = RetryBudgetLLM()
    generator = StepByStepGenerator(
        llm_client=llm,
        tool_manager=CompilerToolManager(),
        num_actions=1,
    )
    generator.max_calls_per_candidate = 3
    generator._capture_initial_usage()

    generator._safe_llm_generate([{"role": "user", "content": "one"}])
    generator._safe_llm_generate([{"role": "user", "content": "two"}])
    generator._safe_llm_generate([{"role": "user", "content": "three"}])

    assert llm.transport_limits == [3, 2, 1]
    with pytest.raises(GenerationBudgetExceeded):
        generator._safe_llm_generate([{"role": "user", "content": "blocked"}])
    assert llm.calls == 3


def test_compact_policy_context_does_not_repeat_output_aliases():
    output = {"value": "alpha"}
    context = {
        "turn_outputs": [
            {
                "calls": [
                    {
                        "call_id": "s1_c1",
                        "tool_name": "lookup",
                        "output": output,
                    }
                ],
                "by_tool": {"lookup": [output]},
                "lookup": output,
            }
        ]
    }

    compact = StepByStepGenerator._compact_policy_context(context)

    assert compact == {
        "prior_turn_outputs": [
            {
                "calls": [
                    {
                        "call_id": "s1_c1",
                        "tool_name": "lookup",
                        "output": output,
                    }
                ]
            }
        ]
    }


def test_numeric_string_is_canonicalized_before_visibility_check():
    generator = object.__new__(StepByStepGenerator)

    value, provenance = generator._materialise_argument_source(
        spec={"source": "user", "value": "300.0"},
        schema={"type": "float"},
        query="Screen those stocks for prices between 200 and 300.",
        policy_context={},
        call_outputs={},
    )

    assert value == 300.0
    assert provenance == {"source": "user"}


def test_numeric_string_still_rejects_a_value_absent_from_visible_text():
    generator = object.__new__(StepByStepGenerator)

    with pytest.raises(ValueError, match="ARGUMENT_NOT_POLICY_VISIBLE"):
        generator._materialise_argument_source(
            spec={"source": "user", "value": "301.0"},
            schema={"type": "float"},
            query="Screen those stocks for prices between 200 and 300.",
            policy_context={},
            call_outputs={},
        )
