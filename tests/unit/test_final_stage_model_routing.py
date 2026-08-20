from __future__ import annotations

from types import SimpleNamespace

import pytest

from apigen_step_by_step import (
    StepByStepGenerator,
    ToolCallWithOutput,
    TrajectoryStep,
)
import generate_step_by_step as generation_cli
from generate_step_by_step import (
    QWEN_FINAL_STAGE_MODEL,
    build_final_stage_clients,
)


class FakeClient:
    def __init__(self, model: str, responses=None, url: str = "https://example/v1"):
        self.api_model = model
        self.url = url
        self.responses = list(responses or [])
        self.calls = []
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_prompt_tokens": 0,
            "cost_usd": 0.0,
            "total_calls": 0,
            "total_attempts": 0,
        }

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.usage["total_calls"] += 1
        self.usage["total_attempts"] += 1
        self.usage["prompt_tokens"] += 10
        self.usage["completion_tokens"] += 2
        self.usage["total_tokens"] += 12
        if not self.responses:
            raise AssertionError(f"No response queued for {self.api_model}")
        return self.responses.pop(0)

    def get_token_usage(self):
        return dict(self.usage)


def _fake_role_client(*, model: str, api_base: str):
    client = FakeClient(model, url=api_base.rstrip("/"))
    client.apigen_openrouter_extensions = "openrouter.ai" in api_base.casefold()
    return client


class DummyToolManager:
    python_tool_instances = {}

    def get_tool_schema(self, name):
        return {
            "name": name,
            "description": "Return a value.",
            "parameters": {"type": "object", "properties": {}},
            "output_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        }


def _args(**overrides):
    values = {
        "use_qwen_final_stages": False,
        "final_response_model": None,
        "final_response_api_base": None,
        "final_response_api_key": None,
        "grounding_model": None,
        "grounding_api_base": None,
        "grounding_api_key": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_final_stage_routing_preserves_existing_clients():
    generator = FakeClient("teacher-generator")
    judge = FakeClient("teacher-judge")

    writer, grounding = build_final_stage_clients(
        _args(),
        llm_client=generator,
        judge_client=judge,
        main_api_base=generator.url,
        main_api_key="main-key",
    )

    assert writer is generator
    assert grounding is judge


def test_qwen_flag_uses_cluster_proxy_and_reuses_one_client(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_URL", "https://176.108.242.226/v1")
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "secret")
    monkeypatch.setattr(
        generation_cli,
        "_make_role_client",
        lambda *, model, api_base, api_key: _fake_role_client(
            model=model, api_base=api_base
        ),
    )
    generator = FakeClient("teacher-generator")
    judge = FakeClient("teacher-judge")

    writer, grounding = build_final_stage_clients(
        _args(use_qwen_final_stages=True),
        llm_client=generator,
        judge_client=judge,
        main_api_base=generator.url,
        main_api_key="main-key",
    )

    assert writer is grounding
    assert writer.api_model == QWEN_FINAL_STAGE_MODEL
    assert writer.url == "https://176.108.242.226/v1"
    assert writer.apigen_openrouter_extensions is False


def test_qwen_flag_fails_closed_when_proxy_environment_is_missing(monkeypatch):
    monkeypatch.delenv("LLM_PROXY_URL", raising=False)
    monkeypatch.delenv("LLM_PROXY_MASTER_KEY", raising=False)

    with pytest.raises(RuntimeError, match="LLM_PROXY_URL"):
        build_final_stage_clients(
            _args(use_qwen_final_stages=True),
            llm_client=FakeClient("generator"),
            judge_client=FakeClient("judge"),
            main_api_base="https://main/v1",
            main_api_key="main-key",
        )


def test_final_response_and_grounding_use_configured_role_clients(monkeypatch):
    monkeypatch.delenv("APIGEN_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("APIGEN_OPENROUTER_PROVIDER", raising=False)
    generator = FakeClient("generator")
    semantic_judge = FakeClient("semantic-judge")
    writer = FakeClient("qwen-writer", ["The returned value is ok."])
    grounding = FakeClient(
        "qwen-grounding",
        ['{"is_grounded": true, "issue_codes": []}'],
    )
    tool_manager = DummyToolManager()

    pipeline = StepByStepGenerator(
        llm_client=generator,
        judge_client=semantic_judge,
        tool_manager=tool_manager,
        optimized_pipeline=True,
    )
    pipeline.configure_final_stage_clients(
        final_response_client=writer,
        grounding_client=grounding,
    )
    trajectory = [
        TrajectoryStep(
            step_number=1,
            tool_calls=[
                ToolCallWithOutput(
                    tool_name="lookup",
                    arguments={},
                    output={"value": "ok"},
                )
            ],
        )
    ]

    response = pipeline._generate_final_response(
        "Report the returned value.", trajectory, {}
    )

    assert response == "The returned value is ok."
    assert len(writer.calls) == 1
    assert len(grounding.calls) == 1
    assert generator.calls == []
    assert semantic_judge.calls == []
    assert pipeline._last_final_response_quality["passed"] is True


def test_candidate_usage_includes_distinct_final_stage_clients():
    generator = FakeClient("generator", ["unused"])
    judge = FakeClient("judge", ["unused"])
    writer = FakeClient("writer", ["unused"])
    grounding = FakeClient("grounding", ["unused"])
    pipeline = StepByStepGenerator(
        llm_client=generator,
        judge_client=judge,
        tool_manager=DummyToolManager(),
        optimized_pipeline=True,
    )
    pipeline.configure_final_stage_clients(
        final_response_client=writer,
        grounding_client=grounding,
    )
    pipeline._reset_token_tracking()
    pipeline._capture_initial_usage()

    for client in (generator, judge, writer, grounding):
        client.generate([{"role": "user", "content": "x"}])

    pipeline._update_token_usage()
    stats = pipeline._get_token_stats()
    assert stats.total_llm_calls == 4
    assert stats.total_tokens == 48


def test_local_final_stage_client_omits_openrouter_only_fields(monkeypatch):
    monkeypatch.setenv("APIGEN_REASONING_EFFORT", "high")
    monkeypatch.setenv("APIGEN_OPENROUTER_PROVIDER", "some-provider")
    local = FakeClient("local-qwen", ["ok"], url="https://cluster/v1")
    local.apigen_openrouter_extensions = False
    pipeline = StepByStepGenerator(
        llm_client=FakeClient("generator"),
        judge_client=FakeClient("judge"),
        tool_manager=DummyToolManager(),
        optimized_pipeline=True,
    )

    result = pipeline._safe_llm_generate(
        [{"role": "user", "content": "hello"}],
        llm=local,
        purpose="test_local_proxy",
    )

    assert result == "ok"
    sent_kwargs = local.calls[0][1]
    assert "provider" not in sent_kwargs
    assert "reasoning" not in sent_kwargs


def test_purpose_specific_reasoning_override_beats_global(monkeypatch):
    monkeypatch.setenv("APIGEN_REASONING_EFFORT", "low")
    monkeypatch.setenv(
        "APIGEN_BLUEPRINT_SEMANTIC_JUDGE_REASONING_EFFORT", "off"
    )
    client = FakeClient("cheap-judge", ["ok"])
    pipeline = StepByStepGenerator(
        llm_client=client,
        judge_client=client,
        tool_manager=DummyToolManager(),
        optimized_pipeline=True,
    )

    result = pipeline._safe_llm_generate(
        [{"role": "user", "content": "check"}],
        llm=client,
        purpose="blueprint_semantic_judge",
    )

    assert result == "ok"
    assert client.calls[0][1]["reasoning"] == {
        "enabled": False,
        "exclude": True,
    }


def test_purpose_specific_reasoning_token_budget_beats_effort(monkeypatch):
    monkeypatch.setenv("APIGEN_REASONING_EFFORT", "high")
    monkeypatch.setenv(
        "APIGEN_BLUEPRINT_GENERATE_REASONING_MAX_TOKENS", "1024"
    )
    client = FakeClient("cheap-teacher", ["ok"])
    pipeline = StepByStepGenerator(
        llm_client=client,
        judge_client=client,
        tool_manager=DummyToolManager(),
        optimized_pipeline=True,
    )

    result = pipeline._safe_llm_generate(
        [{"role": "user", "content": "plan"}],
        llm=client,
        purpose="blueprint_generate",
    )

    assert result == "ok"
    assert client.calls[0][1]["reasoning"] == {
        "max_tokens": 1024,
        "exclude": True,
    }
