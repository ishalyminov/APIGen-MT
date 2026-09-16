import copy

import requests

from apigen_step_by_step import (
    QueryGenerationResult,
    StateVerificationResult,
    StepByStepGenerator,
)


class _FakeToolManager:
    def __init__(self):
        self.value = 0
        self.python_tool_instances = {"fake": object()}

    def get_api_state(self):
        return {"fake": {"value": self.value}}

    def restore_api_state(self, state):
        self.value = copy.deepcopy(state["fake"]["value"])

    def get_tool_schema(self, tool_name):
        return {
            "name": tool_name,
            "output_type": "dict",
            "output_description": "",
        }

    def get_tool_category(self, tool_name):
        return "Fake"


def test_failed_execution_is_rolled_back_before_retry():
    manager = _FakeToolManager()
    generator = StepByStepGenerator(
        llm_client=None,
        tool_manager=manager,
        num_actions=1,
        validate_outputs=False,
        optimized_pipeline=False,
    )

    generator._generate_tool_arguments = lambda **kwargs: ({"fuelAmount": 1}, None)
    generator._verify_tool_query_consistency = lambda **kwargs: (True, "")

    seen_pre_values = []

    def execute(**kwargs):
        seen_pre_values.append(manager.value)
        manager.value += 1
        if len(seen_pre_values) == 1:
            return {"error": "synthetic failure"}
        return {"fuelLevel": manager.value}

    generator._simulate_tool_execution = execute
    generator.verify_state_transition = lambda **kwargs: StateVerificationResult(
        is_valid=True,
        state_changes_summary="value incremented",
    )

    trajectory, _ = generator._stage2_generate_tools(
        QueryGenerationResult(
            query="Add one unit of fuel.",
            intent="",
            expected_tools=["fillFuelTank"],
            quality_preflight={"passed": True},
        ),
        max_retries_per_tool=2,
    )

    assert trajectory is not None
    assert seen_pre_values == [0, 0]
    assert manager.value == 1


def test_chunked_response_termination_is_retried(monkeypatch):
    manager = _FakeToolManager()
    generator = StepByStepGenerator(
        llm_client=None,
        tool_manager=manager,
        num_actions=1,
        validate_outputs=False,
    )

    class _FlakyLLM:
        calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ChunkedEncodingError(
                    "Response ended prematurely"
                )
            return '{"ok": true}'

    client = _FlakyLLM()
    monkeypatch.setattr("time.sleep", lambda _: None)

    assert generator._safe_llm_generate(
        [{"role": "user", "content": "test"}],
        llm=client,
        max_retries=2,
    ) == '{"ok": true}'
    assert client.calls == 2
