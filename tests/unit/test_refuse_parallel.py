import copy
import json
from types import SimpleNamespace

import pytest

from apigen_step_by_step import QueryGenerationResult
from apigen_multi_turn import DialogBlueprint, MultiTurnConversation
from refuse_parallel import (
    FeatureQueryGenerationResult,
    REFUSE_TOOL_SCHEMA,
    RefusalParallelMultiTurnGenerator,
    RefusalParallelStepByStepGenerator,
)


class QueueLLM:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.prompts = []

    def get_token_usage(self):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": len(self.prompts),
        }

    def generate(self, messages, **kwargs):
        self.prompts.append(messages)
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


class ReadOnlyManager:
    def __init__(self):
        self.state = {"demo": {"value": 0}}
        self.python_tool_instances = {"demo": object()}
        self.api_name_to_class_key = {
            "lookup": "demo",
            "mutate": "demo",
        }
        self.schemas = {
            "lookup": {
                "name": "lookup",
                "description": "Look up a value for a supplied key.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "The key and looked-up value.",
            },
            "mutate": {
                "name": "mutate",
                "description": "Mutate shared state.",
                "category": "Demo",
                "parameters": {
                    "type": "object",
                    "properties": {"amount": {"type": "integer"}},
                    "required": ["amount"],
                    "additionalProperties": False,
                },
                "output_type": "dict",
                "output_description": "The new value.",
            },
        }

    def get_tools_json_schema(self):
        return [copy.deepcopy(v) for v in self.schemas.values()]

    def get_tool_schema(self, name):
        return copy.deepcopy(self.schemas[name])

    def get_tool_category(self, name):
        return self.schemas.get(name, {}).get("category")

    def get_tools_by_category(self, category):
        return [s for s in self.get_tools_json_schema() if s.get("category") == category]

    def get_categories(self):
        return ["Demo"]

    def tool_exists(self, name):
        return name in self.schemas

    def has_python_implementation(self, name):
        return name in self.schemas

    def initialize_api_state(self, force_new=False):
        self.state = {"demo": {"value": 0}}

    def get_api_state(self):
        return copy.deepcopy(self.state)

    def restore_api_state(self, state):
        self.state = copy.deepcopy(state)

    def invoke_python_tool(self, name, arguments):
        if name == "lookup":
            key = arguments["key"]
            return {"key": key, "value": f"value:{key}"}
        if name == "mutate":
            self.state["demo"]["value"] += arguments["amount"]
            return {"value": self.state["demo"]["value"]}
        raise ValueError(name)


class FixedParallelGenerator(RefusalParallelStepByStepGenerator):
    def __init__(self, *, calls, manager=None):
        self.fixed_calls = copy.deepcopy(calls)
        llm = QueueLLM()
        super().__init__(
            llm_client=llm,
            judge_client=llm,
            tool_manager=manager or ReadOnlyManager(),
            num_actions=len(calls),
            validate_outputs=False,
            allow_parallel=True,
            parallel_rate=1.0,
        )

    def _generate_parallel_arguments(self, **kwargs):
        return copy.deepcopy(self.fixed_calls)


class ConsistencyBypassGenerator(RefusalParallelStepByStepGenerator):
    def _verify_tool_query_consistency(self, **kwargs):
        return True, ""

    def _certify_parallel_arguments_policy_visible(self, **kwargs):
        return {"passed": True, "issue_codes": []}


def parallel_result(tool_names):
    return FeatureQueryGenerationResult(
        query="Look up alpha and beta together.",
        intent="Retrieve two independent values.",
        expected_tools=tool_names,
        quality_preflight={"passed": True},
        mode="parallel",
        action_plan=[list(range(len(tool_names)))],
        feature_certificate={"passed": True},
    )


def test_refuse_schema_has_only_stable_reason_argument():
    properties = REFUSE_TOOL_SCHEMA["parameters"]["properties"]
    assert set(properties) == {"reason"}
    assert REFUSE_TOOL_SCHEMA["parameters"]["additionalProperties"] is False
    assert "explanation" not in properties


def test_refusal_certifier_fails_closed_when_judge_is_unavailable():
    manager = ReadOnlyManager()
    failing = QueueLLM(exc=RuntimeError("Access denied by security policy"))
    generator = RefusalParallelStepByStepGenerator(
        llm_client=failing,
        judge_client=failing,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    certificate = generator._certify_refusal(
        query="Do the unsupported thing.",
        refusal_type="no_appropriate_function",
        real_tools=manager.get_tools_json_schema(),
        policy_history=[],
    )
    assert certificate["passed"] is False
    assert certificate["issue_codes"] == ["REFUSAL_CERTIFIER_UNAVAILABLE"]


def test_missing_argument_witness_is_deterministic_and_history_closed():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    proposal = {
        "target_tool": "lookup",
        "missing_required_argument": "key",
        "removed_value_text": "alpha",
    }
    witness = generator._validate_missing_argument_witness(
        result=proposal,
        query="Please look up the value for me.",
        original_query="Please look up alpha for me.",
        source_expected_tools=["lookup"],
        policy_history=[],
    )
    assert witness["passed"] is True
    assert witness["required_without_default"] is True
    assert witness["removed_from_source_query"] is True
    assert witness["absent_from_prior_history"] is True

    leaked = generator._validate_missing_argument_witness(
        result=proposal,
        query="Please look up the value for me.",
        original_query="Please look up alpha for me.",
        source_expected_tools=["lookup"],
        policy_history=[{"role": "user", "content": "The key is alpha."}],
    )
    assert leaked["passed"] is False
    assert "WITNESS_AVAILABLE_IN_HISTORY" in leaked["issue_codes"]


def test_recovery_only_repeats_the_proven_missing_value():
    manager = ReadOnlyManager()
    llm = QueueLLM(
        responses=[
            json.dumps(
                {"query": "Use 20-liter bucket capacity, please; go ahead."}
            )
        ]
    )
    generator = RefusalParallelMultiTurnGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_turns=5,
        actions_per_turn=2,
        allow_refusal=True,
    )
    query, certificate = generator._generate_clarification_recovery_query(
        pending={
            "source_query": (
                "Convert 12.5 liters and compare it with the "
                "20-liter bucket capacity."
            ),
            "refusal_query": "Convert 12.5 liters and compare it with the bucket.",
            "assistant_response": "What is the bucket capacity?",
            "expected_tools": ["lookup"],
            "reason": "missing_argument",
            "deterministic_missing_argument_witness": {
                "passed": True,
                "removed_value_text": "20-liter bucket capacity",
            },
        },
        history=[],
    )
    assert "20-liter bucket capacity" in query
    assert "12.5" not in query
    assert certificate["protected_tokens"] == ["20-liter bucket capacity"]
    assert certificate["protected_tokens_preserved"] is True






def test_refusal_certifier_rejects_malformed_boolean_contract():
    manager = ReadOnlyManager()
    llm = QueueLLM(responses=[json.dumps({"is_valid": "true", "issue_codes": []})])
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    certificate = generator._certify_refusal(
        query="Do the unsupported thing.",
        refusal_type="no_appropriate_function",
        real_tools=manager.get_tools_json_schema(),
        policy_history=[],
    )
    assert certificate["passed"] is False
    assert certificate["issue_codes"] == ["REFUSAL_CERTIFIER_UNAVAILABLE"]


def test_refusal_requires_adversarial_counterexample_search_to_find_no_plan():
    manager = ReadOnlyManager()
    llm = QueueLLM(
        responses=[
            json.dumps({
                "is_valid": True,
                "issue_codes": [],
                "underlying_plan_preserved": True,
                "counterexample_check": {
                    "real_tool_plan_exists": False,
                    "missing_value_is_obtainable": False,
                    "unique_safe_action_exists": False,
                },
            }),
        ]
    )
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    certificate = generator._certify_refusal(
        query="Perform an external action for which no available tool exists.",
        refusal_type="no_appropriate_function",
        real_tools=manager.get_tools_json_schema(),
        policy_history=[],
    )
    assert certificate["passed"] is True
    assert certificate["counterexample_search"]["real_tool_plan_exists"] is False
    assert len(llm.prompts) == 1


def test_refusal_is_rejected_when_counterexample_solver_finds_real_tool_plan():
    manager = ReadOnlyManager()
    llm = QueueLLM(
        responses=[
            json.dumps(
                {
                    "is_valid": True,
                    "issue_codes": [],
                    "counterexample_check": {
                    "real_tool_plan_exists": True,
                    "missing_value_is_obtainable": False,
                    "unique_safe_action_exists": False,
                    },
                }
            ),
        ]
    )
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    certificate = generator._certify_refusal(
        query="Look up alpha.",
        refusal_type="no_appropriate_function",
        real_tools=manager.get_tools_json_schema(),
        policy_history=[],
    )
    assert certificate["passed"] is False
    assert "REAL_TOOL_CAN_FULFILL" in certificate["issue_codes"]


def test_refusal_stage_has_no_hidden_or_free_form_target():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    result = FeatureQueryGenerationResult(
        query="Send this somewhere, but I did not specify where.",
        intent="Ambiguous action.",
        expected_tools=["refuse"],
        quality_preflight={"passed": True},
        mode="refusal",
        action_plan=[[0]],
        refusal_type="ambiguity",
        feature_certificate={"passed": True},
    )
    steps, context = generator._stage2_generate_tools(result, 1)
    assert len(steps) == 1
    call = steps[0].tool_calls[0]
    assert call.tool_name == "refuse"
    assert call.arguments == {"reason": "ambiguity"}
    assert steps[0].execution_mode == "refusal"
    assert steps[0].call_order_matters is True
    assert steps[0].pre_state is None
    assert steps[0].post_state is None
    assert context["refusal"]["status"] == "refused"




def test_parallel_argument_certifier_rejects_malformed_result():
    manager = ReadOnlyManager()
    llm = QueueLLM(responses=["{}"] )
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_parallel=True,
    )
    certificate = generator._certify_parallel_arguments_policy_visible(
        query="Look up alpha and beta.",
        calls=[
            {"call_id": "p1", "tool_name": "lookup", "arguments": {"key": "alpha"}},
            {"call_id": "p2", "tool_name": "lookup", "arguments": {"key": "beta"}},
        ],
        visible_history=[],
        execution_context={},
    )
    assert certificate["passed"] is False
    assert certificate["issue_codes"] == [
        "PARALLEL_ARGUMENT_CERTIFIER_UNAVAILABLE"
    ]


def test_parallel_duplicate_tool_calls_are_preserved_by_call_id():
    manager = ReadOnlyManager()
    llm = QueueLLM(
        responses=[
            json.dumps(
                {
                    "calls": [
                        {"call_id": "p1", "arguments": {"key": "alpha"}},
                        {"call_id": "p2", "arguments": {"key": "beta"}},
                    ]
                }
            )
        ]
    )
    generator = ConsistencyBypassGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        validate_outputs=False,
        allow_parallel=True,
        parallel_rate=1.0,
    )
    calls = generator._generate_parallel_arguments(
        query="Look up alpha and beta.",
        tool_names=["lookup", "lookup"],
        trajectory=[],
        execution_context={},
        max_retries=1,
    )
    assert [call["call_id"] for call in calls] == ["p1", "p2"]
    assert [call["arguments"]["key"] for call in calls] == ["alpha", "beta"]


def test_parallel_read_only_batch_is_atomic_and_order_independent():
    manager = ReadOnlyManager()
    calls = [
        {"call_id": "p1", "tool_name": "lookup", "arguments": {"key": "alpha"}},
        {"call_id": "p2", "tool_name": "lookup", "arguments": {"key": "beta"}},
    ]
    generator = FixedParallelGenerator(calls=calls, manager=manager)
    before = manager.get_api_state()
    steps, context = generator._execute_parallel_batch(
        query_result=parallel_result(["lookup", "lookup"]),
        max_retries=1,
        initial_execution_context={},
    )
    assert len(steps) == 1
    assert len(steps[0].tool_calls) == 2
    assert steps[0].execution_mode == "parallel"
    assert steps[0].call_order_matters is False
    assert steps[0].quality_verification["passed"] is True
    assert steps[0].quality_verification["forward_reverse_outputs_equal"] is True
    assert manager.get_api_state() == before
    assert context["call_p1_output"]["key"] == "alpha"
    assert context["call_p2_output"]["key"] == "beta"


def test_parallel_batch_rejects_state_mutation_and_rolls_back():
    manager = ReadOnlyManager()
    calls = [
        {"call_id": "p1", "tool_name": "mutate", "arguments": {"amount": 1}},
        {"call_id": "p2", "tool_name": "lookup", "arguments": {"key": "beta"}},
    ]
    generator = FixedParallelGenerator(calls=calls, manager=manager)
    before = manager.get_api_state()
    steps, context = generator._execute_parallel_batch(
        query_result=parallel_result(["mutate", "lookup"]),
        max_retries=1,
        initial_execution_context={},
    )
    assert steps is None
    assert context is None
    assert manager.get_api_state() == before


def test_parallel_batch_repairs_once_from_pre_turn_snapshot():
    manager = ReadOnlyManager()
    generator = FixedParallelGenerator(calls=[], manager=manager)
    attempts = iter(
        [
            [
                {
                    "call_id": "p1",
                    "tool_name": "mutate",
                    "arguments": {"amount": 1},
                },
                {
                    "call_id": "p2",
                    "tool_name": "lookup",
                    "arguments": {"key": "beta"},
                },
            ],
            [
                {
                    "call_id": "p1",
                    "tool_name": "lookup",
                    "arguments": {"key": "alpha"},
                },
                {
                    "call_id": "p2",
                    "tool_name": "lookup",
                    "arguments": {"key": "beta"},
                },
            ],
        ]
    )
    generator._generate_parallel_arguments = lambda **kwargs: next(attempts)
    before = manager.get_api_state()
    steps, _ = generator._execute_parallel_batch(
        query_result=parallel_result(["lookup", "lookup"]),
        max_retries=2,
        initial_execution_context={},
    )
    assert len(steps) == 1
    assert [call.tool_name for call in steps[0].tool_calls] == [
        "lookup",
        "lookup",
    ]
    assert manager.get_api_state() == before


def test_certified_parallel_relevance_replaces_lexical_overlap_heuristic():
    manager = ReadOnlyManager()
    calls = [
        {"call_id": "p1", "tool_name": "lookup", "arguments": {"key": "alpha"}},
        {"call_id": "p2", "tool_name": "lookup", "arguments": {"key": "beta"}},
    ]
    generator = FixedParallelGenerator(calls=calls, manager=manager)
    steps, context = generator._execute_parallel_batch(
        query_result=parallel_result(["lookup", "lookup"]),
        max_retries=1,
        initial_execution_context={},
    )
    verification = generator.run_full_verification(
        query="Please handle both of those for the comparison.",
        trajectory=steps,
        execution_context=context,
        query_quality={
            "passed": True,
            "mode": "parallel",
            "parallel_certificate": {"passed": True},
        },
    )
    assert verification.overall_verification_passed is True
    assert all(
        check["reasoning"].startswith("Certified by the episode-level")
        for check in verification.tool_relevance_checks
    )


def test_strict_argument_schema_rejects_unknown_and_wrong_type():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_parallel=True,
    )
    assert generator._validate_tool_arguments_schema("lookup", {"key": "x"}) == []
    assert generator._validate_tool_arguments_schema("lookup", {"key": 3})
    assert generator._validate_tool_arguments_schema("lookup", {"key": "x", "extra": 1})


def test_multi_turn_feature_generator_keeps_base_model_shape():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelMultiTurnGenerator(
        llm_client=llm,
        tool_manager=manager,
        num_turns=2,
        actions_per_turn=2,
        allow_refusal=True,
        allow_parallel=True,
    )
    assert generator.num_turns == 2
    assert generator.num_actions == 2
    tools = generator._get_policy_tool_schemas("Demo")
    assert [tool["name"] for tool in tools].count("refuse") == 1


def test_feature_query_result_remains_compatible_with_base_query_result():
    result = parallel_result(["lookup", "lookup"])
    assert isinstance(result, QueryGenerationResult)
    assert result.expected_tools == ["lookup", "lookup"]

class GroundedFixedParallelGenerator(FixedParallelGenerator):
    def _generate_final_response(self, query, trajectory, execution_context):
        self._last_final_response_quality = {"passed": True, "issue_codes": []}
        return "Retrieved both requested values."

    def verify_tool_relevance(self, query, tool_name, step):
        return {
            "tool_name": tool_name,
            "is_relevant": True,
            "relevance_score": 1.0,
            "reasoning": "Explicitly requested.",
        }


def test_parallel_stage3_uses_existing_main_verification_and_records_grouping():
    manager = ReadOnlyManager()
    calls = [
        {"call_id": "p1", "tool_name": "lookup", "arguments": {"key": "alpha"}},
        {"call_id": "p2", "tool_name": "lookup", "arguments": {"key": "beta"}},
    ]
    generator = GroundedFixedParallelGenerator(calls=calls, manager=manager)
    result = parallel_result(["lookup", "lookup"])
    steps, context = generator._execute_parallel_batch(
        query_result=result,
        max_retries=1,
        initial_execution_context={},
    )
    datapoint = generator._stage3_finalize(
        result,
        steps,
        context,
        "Demo",
        manager.get_api_state(),
    )
    assert datapoint is not None
    assert datapoint.verification_result["overall_verification_passed"] is True
    assert datapoint.generation_metadata["contains_parallel"] is True
    assert datapoint.generation_metadata["num_actions"] == 2
    assert len(datapoint.trajectory.steps) == 1
    assert len(datapoint.trajectory.steps[0].tool_calls) == 2


def test_refusal_stage3_is_self_contained_and_verified():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_refusal=True,
    )
    result = FeatureQueryGenerationResult(
        query="Send the report, but no recipient is specified.",
        intent="Send a report.",
        expected_tools=["refuse"],
        quality_preflight={"passed": True},
        mode="refusal",
        action_plan=[[0]],
        refusal_type="missing_argument",
        feature_certificate={"passed": True},
    )
    steps, context = generator._stage2_generate_tools(result, 1)
    datapoint = generator._stage3_finalize(
        result,
        steps,
        context,
        "Demo",
        manager.get_api_state(),
    )
    assert datapoint.verification_result["overall_verification_passed"] is True
    assert datapoint.generation_metadata["terminal_mode"] == "clarification"
    assert datapoint.trajectory.steps[0].tool_calls[0].arguments == {
        "reason": "missing_argument"
    }
    assert any(tool["name"] == "refuse" for tool in datapoint.available_tools)


def test_multi_turn_final_refusal_is_generated_without_touching_blueprint_tools(monkeypatch):
    manager = ReadOnlyManager()
    llm = QueueLLM(
        responses=[
            json.dumps({
                "query": "Look this up, but I did not provide the required key.",
                "intent": "Lookup",
                "assistant_response": "Which lookup key should I use?",
                "target_tool": "lookup",
                "missing_required_argument": "key",
                "removed_value_text": "alpha",
            }),
            json.dumps(
                {
                    "is_valid": True,
                    "issue_codes": [],
                    "underlying_plan_preserved": True,
                    "counterexample_check": {
                        "real_tool_plan_exists": False,
                        "missing_value_is_obtainable": False,
                        "unique_safe_action_exists": False,
                    },
                }
            ),
        ]
    )
    generator = RefusalParallelMultiTurnGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_turns=1,
        actions_per_turn=2,
        allow_refusal=True,
        refusal_rate=1.0,
    )
    blueprint = DialogBlueprint(
        overall_task="Retrieve information",
        num_turns=1,
        turns=[{"user_query": "Look up alpha.", "expected_tools": ["lookup"]}],
    )
    result = generator._generate_turn_query(
        blueprint,
        MultiTurnConversation(overall_task=blueprint.overall_task),
        0,
    )
    assert isinstance(result, FeatureQueryGenerationResult)
    assert result.mode == "refusal"
    assert result.expected_tools == ["refuse"]
    assert result.native_response == "Which lookup key should I use?"
    # The source blueprint remains unchanged; feature behavior is opt-in at the
    # turn-query layer and cannot corrupt the base blueprint representation.
    assert blueprint.turns[0]["expected_tools"] == ["lookup"]


def test_multi_turn_parallel_is_restricted_to_final_turn():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelMultiTurnGenerator(
        llm_client=llm,
        tool_manager=manager,
        num_turns=2,
        actions_per_turn=2,
        allow_parallel=True,
        parallel_rate=1.0,
    )
    blueprint = DialogBlueprint(
        overall_task="Retrieve information",
        num_turns=2,
        turns=[
            {"user_query": "Look up alpha.", "expected_tools": ["lookup"]},
            {"user_query": "Look up beta.", "expected_tools": ["lookup"]},
        ],
    )
    # Non-final turns use the current-main path, avoiding broken downstream
    # placeholders or dependencies caused by rewriting an earlier turn.
    generator.validate_expected_tools = lambda *args, **kwargs: (True, "")
    generator._last_query_quality = {"passed": True}
    result = generator._generate_turn_query(
        blueprint,
        MultiTurnConversation(overall_task=blueprint.overall_task),
        0,
    )
    assert not isinstance(result, FeatureQueryGenerationResult) or result.mode == "normal"
    assert result.expected_tools == ["lookup"]




def test_parallel_width_above_configured_limit_is_not_silently_truncated():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=4,
        allow_parallel=True,
        parallel_rate=1.0,
        max_parallel_width=3,
    )
    assert generator._generate_parallel_query(
        focus_category="Demo",
        num_calls=4,
        initial_api_state=manager.get_api_state(),
    ) is None
    assert llm.prompts == []


def test_cli_defaults_keep_features_disabled(monkeypatch):
    import generate_step_by_step as cli

    monkeypatch.setattr("sys.argv", ["generate_step_by_step.py"])
    args = cli.parse_args()
    assert args.allow_refusal is False
    assert args.allow_parallel is False
    assert args.refusal_rate == pytest.approx(0.12)
    assert args.parallel_rate == pytest.approx(0.25)
    assert args.require_feature is False


def test_sequential_dedupe_signature_is_backward_compatible():
    import hashlib
    import generate_step_by_step as cli

    datapoint = {
        "trajectory": {
            "steps": [
                {"tool_calls": [{"tool_name": "lookup", "arguments": {"key": "a"}}]},
                {"tool_calls": [{"tool_name": "lookup", "arguments": {"key": "b"}}]},
            ]
        }
    }
    legacy_calls = [
        {"tool_name": "lookup", "arguments": {"key": "a"}},
        {"tool_name": "lookup", "arguments": {"key": "b"}},
    ]
    payload = json.dumps(
        legacy_calls,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    legacy_signature = hashlib.sha256(payload).hexdigest()
    assert cli._trajectory_signature_from_dict(datapoint) == legacy_signature


def test_parallel_and_sequential_dedupe_signatures_differ():
    import generate_step_by_step as cli

    sequential = {
        "trajectory": {
            "steps": [
                {"tool_calls": [{"tool_name": "lookup", "arguments": {"key": "a"}}]},
                {"tool_calls": [{"tool_name": "lookup", "arguments": {"key": "b"}}]},
            ]
        }
    }
    parallel = {
        "trajectory": {
            "steps": [
                {
                    "tool_calls": [
                        {"tool_name": "lookup", "arguments": {"key": "a"}},
                        {"tool_name": "lookup", "arguments": {"key": "b"}},
                    ]
                }
            ]
        }
    }
    assert cli._trajectory_signature_from_dict(sequential) != cli._trajectory_signature_from_dict(parallel)



def test_parallel_dedupe_signature_is_order_invariant_within_batch():
    import generate_step_by_step as cli

    first = {
        "trajectory": {
            "steps": [
                {
                    "tool_calls": [
                        {"tool_name": "lookup", "arguments": {"key": "a"}},
                        {"tool_name": "lookup", "arguments": {"key": "b"}},
                    ]
                }
            ]
        }
    }
    swapped = {
        "trajectory": {
            "steps": [
                {
                    "tool_calls": [
                        {"tool_name": "lookup", "arguments": {"key": "b"}},
                        {"tool_name": "lookup", "arguments": {"key": "a"}},
                    ]
                }
            ]
        }
    }
    assert cli._trajectory_signature_from_dict(first) == cli._trajectory_signature_from_dict(swapped)


class DeterministicRefusalMulti(RefusalParallelMultiTurnGenerator):
    def _stage0_generate_blueprint(self, focus_category=None, initial_api_state=None):
        return DialogBlueprint(
            overall_task="Handle an underspecified request",
            num_turns=1,
            turns=[{"user_query": "Do the task.", "expected_tools": ["lookup"]}],
        )

    def _generate_turn_query(self, blueprint, conversation, turn_index):
        quality = {
            "passed": True,
            "mode": "refusal",
            "refusal_type": "missing_argument",
        }
        return FeatureQueryGenerationResult(
            query="Look up the record, but I did not provide the required key.",
            intent="Lookup a record.",
            expected_tools=["refuse"],
            quality_preflight=quality,
            mode="refusal",
            action_plan=[[0]],
            refusal_type="missing_argument",
            native_response="Which lookup key should I use?",
            feature_certificate={"passed": True},
        )


class DeterministicParallelMulti(RefusalParallelMultiTurnGenerator):
    def _stage0_generate_blueprint(self, focus_category=None, initial_api_state=None):
        return DialogBlueprint(
            overall_task="Retrieve two independent values",
            num_turns=1,
            turns=[
                {
                    "user_query": "Look up alpha and beta together.",
                    "expected_tools": ["lookup", "lookup"],
                }
            ],
        )

    def _generate_turn_query(self, blueprint, conversation, turn_index):
        return parallel_result(["lookup", "lookup"])

    def _generate_parallel_arguments(self, **kwargs):
        return [
            {"call_id": "p1", "tool_name": "lookup", "arguments": {"key": "alpha"}},
            {"call_id": "p2", "tool_name": "lookup", "arguments": {"key": "beta"}},
        ]

    def _generate_final_response(self, query, trajectory, execution_context):
        self._last_final_response_quality = {"passed": True, "issue_codes": []}
        return "Retrieved alpha and beta."


def test_multi_turn_refusal_runs_through_existing_conversation_loop():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = DeterministicRefusalMulti(
        llm_client=llm,
        tool_manager=manager,
        num_turns=1,
        actions_per_turn=2,
        allow_refusal=True,
    )
    datapoint = generator.generate_multi_turn_datapoint(focus_category="Demo")
    assert datapoint is not None
    assert datapoint.verification_result["overall_verification_passed"] is True
    assert datapoint.generation_metadata["contains_refusal"] is True
    assert datapoint.generation_metadata["terminal_mode"] == "clarification"
    assert datapoint.generation_metadata["terminal_action"] == "refuse"
    step = datapoint.conversation.turns[0].steps[0]
    assert step.tool_calls[0].tool_name == "refuse"
    assert step.execution_mode == "refusal"
    assert step.call_order_matters is True


def test_multi_turn_parallel_runs_through_existing_conversation_loop():
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = DeterministicParallelMulti(
        llm_client=llm,
        tool_manager=manager,
        num_turns=1,
        actions_per_turn=2,
        allow_parallel=True,
    )
    datapoint = generator.generate_multi_turn_datapoint(focus_category="Demo")
    assert datapoint is not None
    assert datapoint.verification_result["overall_verification_passed"] is True
    assert datapoint.generation_metadata["contains_parallel"] is True
    step = datapoint.conversation.turns[0].steps[0]
    assert len(step.tool_calls) == 2
    assert step.execution_mode == "parallel"
    assert step.call_order_matters is False
    assert step.quality_verification["forward_reverse_outputs_equal"] is True


def test_require_feature_resamples_instead_of_saving_normal_fallback(monkeypatch):
    manager = ReadOnlyManager()
    llm = QueueLLM()
    generator = RefusalParallelStepByStepGenerator(
        llm_client=llm,
        judge_client=llm,
        tool_manager=manager,
        num_actions=2,
        allow_parallel=True,
        parallel_rate=1.0,
        require_feature=True,
    )
    monkeypatch.setattr(generator, "_generate_parallel_query", lambda **kwargs: None)
    result = generator._stage1_generate_query(
        focus_category="Demo",
        context_hint=None,
        max_retries=1,
        initial_api_state=manager.get_api_state(),
    )
    assert result is None
