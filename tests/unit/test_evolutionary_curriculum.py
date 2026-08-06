import json
from unittest.mock import MagicMock

from apigen_multi_turn import MultiTurnGenerator
from evolutionary_curriculum import EvolutionaryCurriculum


TOOLS = [
    {
        "name": "lookup_record",
        "category": "Records",
        "description": "Look up a record and return its identifier.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
        },
    },
    {
        "name": "update_record",
        "category": "Records",
        "description": "Update a record by identifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["record_id", "value"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"success": {"type": "boolean"}},
        },
    },
    {
        "name": "send_notice",
        "category": "Communication",
        "description": "Send a notice.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"message_id": {"type": "string"}},
        },
    },
]


def test_directives_reserve_unique_serials_and_keep_targets_visible(tmp_path):
    state_path = tmp_path / "coverage.json"
    curriculum = EvolutionaryCurriculum(
        tools=TOOLS,
        state_path=state_path,
        seed=11,
        all_tools_rate=0.0,
        cross_domain_rate=0.0,
        hard_distractor_count=4,
        target_tools_per_candidate=2,
    )

    first = curriculum.next_directive()
    second = curriculum.next_directive()

    assert first.directive_id != second.directive_id
    assert set(first.target_tools) <= set(first.allowed_tools)
    assert set(second.target_tools) <= set(second.allowed_tools)
    assert json.loads(state_path.read_text())["directive_counter"] == 2


def test_coverage_counts_only_tools_used_by_accepted_rows(tmp_path):
    curriculum = EvolutionaryCurriculum(
        tools=TOOLS,
        state_path=tmp_path / "coverage.json",
        target_tools_per_candidate=2,
    )
    directive = {
        "target_tools": ["lookup_record", "update_record"],
        "motif": "output_chain",
        "context_mode": "hard_distractors",
    }
    row = {
        "conversation": {
            "turns": [
                {
                    "steps": [
                        {
                            "tool_calls": [
                                {"tool_name": "lookup_record", "arguments": {}}
                            ]
                        }
                    ]
                }
            ]
        }
    }

    curriculum.observe(directive=directive, row=row, accepted=True)
    snapshot = curriculum.state.transaction()

    assert snapshot["used_counts"]["lookup_record"] == 1
    assert snapshot["used_counts"]["update_record"] == 0
    assert snapshot["target_counts"]["lookup_record"] == 1
    assert snapshot["target_counts"]["update_record"] == 0
    assert snapshot["target_miss_counts"]["update_record"] == 1


def test_policy_tool_context_uses_directive_allowlist_over_category():
    generator = object.__new__(MultiTurnGenerator)
    generator.tool_manager = MagicMock()
    generator.tool_manager.get_tools_json_schema.return_value = TOOLS
    generator._active_generation_directive = {
        "allowed_tools": ["lookup_record", "send_notice"]
    }

    visible = generator._get_policy_tool_schemas("Records")

    assert [tool["name"] for tool in visible] == [
        "lookup_record",
        "send_notice",
    ]


def test_fresh_generation_replaces_stale_directive_before_blueprint():
    llm = MagicMock()
    manager = MagicMock()
    manager.python_tool_instances = {}
    generator = MultiTurnGenerator(llm, manager, num_turns=1, actions_per_turn=1)
    generator._active_generation_directive = {"directive_id": "stale"}
    generator._stage0_generate_blueprint = MagicMock(return_value=None)

    result = generator.generate_multi_turn_datapoint(
        generation_directive={"directive_id": "fresh", "allowed_tools": ["x"]}
    )

    assert result is None
    assert generator._active_generation_directive == {
        "directive_id": "fresh",
        "allowed_tools": ["x"],
    }


def test_openrouter_provider_is_pinned_without_fallbacks(monkeypatch):
    class CapturingLLM:
        api_model = "example/model"

        def __init__(self):
            self.kwargs = None

        def get_token_usage(self):
            return {
                "total_calls": 0,
                "total_attempts": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        def generate(self, messages, **kwargs):
            self.kwargs = kwargs
            return "ok"

    client = CapturingLLM()
    generator = object.__new__(MultiTurnGenerator)
    generator.llm = client
    generator.max_calls_per_candidate = 10
    generator.max_tokens_per_candidate = 100_000
    generator._initial_token_usage = None
    generator._accumulated_llm_calls = 0
    monkeypatch.setenv("APIGEN_OPENROUTER_PROVIDER", "novita/fp8")
    monkeypatch.setenv("APIGEN_OPENROUTER_ALLOW_FALLBACKS", "false")

    assert generator._safe_llm_generate([{"role": "user", "content": "x"}]) == "ok"
    assert client.kwargs["provider"] == {
        "only": ["novita/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
