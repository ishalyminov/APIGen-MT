import json
import os
from pathlib import Path

import pytest

import check_apigen_trajectories_passk as legacy
import check_apigen_trajectories_passk_v3 as v3


def external_fixture(name):
    default_root = (
        Path(__file__).resolve().parents[4] / "tool_synth" / "APIGen-MT-main"
    )
    root = Path(os.environ.get("APIGEN_PASSK_FIXTURE_ROOT", default_root))
    path = root / "data" / "generated" / name
    if not path.is_file():
        pytest.skip(
            "large pass@k integration fixture is external; set "
            "APIGEN_PASSK_FIXTURE_ROOT"
        )
    return path


def tool_call(name, arguments, output):
    return {"name": name, "arguments": arguments, "output": output}


def make_task(step_specs, *, queries=None, responses=None):
    queries = queries or ["Do the requested work."]
    responses = responses or ["Done."] * len(queries)
    calls = []
    gold_steps = []
    user_turns = [
        {"query": query, "assistant_response": responses[index], "steps": []}
        for index, query in enumerate(queries)
    ]
    tool_names = set()
    for global_step_index, spec in enumerate(step_specs):
        turn_index = spec.get("turn_index", 0)
        step_calls = spec["calls"]
        call_indices = list(range(len(calls), len(calls) + len(step_calls)))
        calls.extend(step_calls)
        tool_names.update(call["name"] for call in step_calls)
        mode = spec["mode"]
        pre_state = {"value": spec.get("pre", global_step_index)}
        if mode in {"read", "parallel", "refusal"}:
            post_state = pre_state
        else:
            post_state = {"value": spec.get("post", global_step_index + 1)}
        raw_step = {
            "step_number": len(user_turns[turn_index]["steps"]) + 1,
            "tool_calls": [
                {
                    "tool_name": call["name"],
                    "arguments": call["arguments"],
                    "output": call["output"],
                }
                for call in step_calls
            ],
            "pre_state": pre_state,
            "post_state": post_state,
            "quality_verification": spec.get("quality_verification", {}),
            "execution_mode": "parallel" if mode == "parallel" else "sequential",
            "call_order_matters": mode != "parallel",
        }
        user_turns[turn_index]["steps"].append(raw_step)
        execution_mode = {
            "read": "sequential",
            "mutation": "sequential",
            "parallel": "parallel",
            "refusal": "refusal",
        }[mode]
        gold_steps.append(
            {
                "turn_index": turn_index,
                "step_index": len(user_turns[turn_index]["steps"]) - 1,
                "calls": step_calls,
                "call_indices": call_indices,
                "execution_mode": execution_mode,
                "call_order_matters": mode != "parallel",
                "parallel_certified": mode == "parallel",
                "refusal_certified": mode == "refusal",
                "refusal_reason": spec.get("reason"),
            }
        )
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in sorted(tool_names)
    ]
    return legacy.Task(
        position=0,
        raw={},
        query=queries[0],
        initial_state={"private": "DO_NOT_SHOW"},
        tools=tools,
        gold_calls=calls,
        gold_steps=gold_steps,
        user_turns=user_turns,
        step_order_matters=len(user_turns) > 1,
        data_issues=[],
        focus_category="test",
    )


def test_independent_read_calls_may_batch_and_reverse():
    first = tool_call("lookup", {"key": "alpha"}, {"value": 1})
    second = tool_call("lookup", {"key": "beta"}, {"value": 2})
    task = make_task(
        [
            {"mode": "read", "calls": [first]},
            {"mode": "read", "calls": [second]},
        ],
        queries=["Look up alpha and beta."],
    )
    schedule = v3.build_schedule(task)
    target = v3.scheduler_target(task, set(), schedule)
    assert target.segment.mode == "ready_read_set"
    assert target.ready_step_indices == (0, 1)
    matched = v3.match_scheduler_calls(task, target, [second, first])
    assert matched is not None
    step_indices, call_indices, gold_calls = matched
    assert step_indices == [1, 0]
    assert call_indices == [1, 0]
    assert [call["output"] for call in gold_calls] == [
        {"value": 2},
        {"value": 1},
    ]


def test_output_dependent_read_is_not_ready_or_batchable():
    producer = tool_call("find_user", {"name": "Maria"}, {"user_id": "USR-17"})
    consumer = tool_call(
        "get_profile", {"user_id": "USR-17"}, {"name": "Maria"}
    )
    task = make_task(
        [
            {"mode": "read", "calls": [producer]},
            {"mode": "read", "calls": [consumer]},
        ],
        queries=["Find Maria and show me her profile."],
    )
    schedule = v3.build_schedule(task)
    dependencies = schedule[0].dependency_map()
    assert dependencies[1] == {0}
    first_target = v3.scheduler_target(task, set(), schedule)
    assert first_target.ready_step_indices == (0,)
    assert (
        v3.match_scheduler_calls(
            task, first_target, [producer, consumer]
        )
        is None
    )
    second_target = v3.scheduler_target(task, {0}, schedule)
    assert second_target.ready_step_indices == (1,)


def test_literal_user_value_does_not_create_a_false_output_dependency():
    first = tool_call("lookup", {"key": "alpha"}, {"ticket_id": 17})
    second = tool_call("get_ticket", {"ticket_id": 17}, {"title": "Issue"})
    task = make_task(
        [
            {"mode": "read", "calls": [first]},
            {"mode": "read", "calls": [second]},
        ],
        queries=["Look up alpha and independently get ticket 17."],
    )
    target = v3.scheduler_target(task, set())
    assert target.ready_step_indices == (0, 1)


def test_echoed_input_in_tool_output_does_not_create_dependency():
    numbers = [5, 5, 3, 4, 4]
    task = make_task(
        [
            {
                "mode": "read",
                "calls": [
                    tool_call(
                        "mean",
                        {"numbers": numbers},
                        {"result": 4.2, "input_numbers": numbers},
                    )
                ],
            },
            {
                "mode": "read",
                "calls": [
                    tool_call(
                        "max_value",
                        {"numbers": numbers},
                        {"result": 5, "input_numbers": numbers},
                    )
                ],
            },
        ],
        queries=["Give me the average and maximum of the prior values."],
    )
    target = v3.scheduler_target(task, set())
    assert target.ready_step_indices == (0, 1)


def test_optional_schema_default_and_trivial_message_format_are_equivalent():
    task = make_task(
        [
            {
                "mode": "read",
                "calls": [
                    tool_call("get_user_tickets", {}, {"tickets": []})
                ],
            },
            {
                "mode": "mutation",
                "calls": [
                    tool_call(
                        "post_tweet",
                        {
                            "content": (
                                "Heads up: a storm may delay departures."
                            )
                        },
                        {"id": 1},
                    )
                ],
            },
        ]
    )
    for tool in task.tools:
        name = v3._tool_name(tool)
        if name == "get_user_tickets":
            tool["function"]["parameters"] = {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "default": "None"}
                },
            }
    read_target = v3.scheduler_target(task, set())
    assert v3.match_scheduler_calls(
        task,
        read_target,
        [
            {
                "name": "get_user_tickets",
                "arguments": {"status": "None"},
            }
        ],
    )
    mutation_target = v3.scheduler_target(task, {0})
    assert v3.match_scheduler_calls(
        task,
        mutation_target,
        [
            {
                "name": "post_tweet",
                "arguments": {
                    "content": "Heads-up: A storm may delay departures!"
                },
            }
        ],
    )


def test_commutative_math_operands_may_be_reversed():
    task = make_task(
        [
            {
                "mode": "read",
                "calls": [
                    tool_call("add", {"a": 3.79, "b": 10.7}, {"result": 14.49})
                ],
            }
        ]
    )
    target = v3.scheduler_target(task, set())
    assert v3.match_scheduler_calls(
        task,
        target,
        [{"name": "add", "arguments": {"a": 10.7, "b": 3.79}}],
    )


def test_state_change_is_an_ordered_barrier():
    read_before = tool_call("status", {}, {"running": False})
    mutation = tool_call("start", {}, {"running": True})
    read_after = tool_call("status", {}, {"running": True})
    task = make_task(
        [
            {"mode": "read", "calls": [read_before]},
            {"mode": "mutation", "calls": [mutation]},
            {"mode": "read", "calls": [read_after]},
        ]
    )
    schedule = v3.build_schedule(task)
    assert [segment.mode for segment in schedule] == [
        "ready_read_set",
        "ordered_barrier",
        "ready_read_set",
    ]
    first_target = v3.scheduler_target(task, set(), schedule)
    assert (
        v3.match_scheduler_calls(
            task, first_target, [read_before, mutation]
        )
        is None
    )
    barrier = v3.scheduler_target(task, {0}, schedule)
    assert barrier.segment.mode == "ordered_barrier"
    assert barrier.ready_step_indices == (1,)


def test_certified_parallel_is_exact_unordered_multiset_with_multiplicity():
    alpha = tool_call("lookup", {"key": "alpha"}, {"value": 1})
    beta = tool_call("lookup", {"key": "beta"}, {"value": 2})
    task = make_task(
        [{"mode": "parallel", "calls": [alpha, beta, alpha]}]
    )
    target = v3.scheduler_target(task, set())
    assert target.segment.mode == "certified_parallel"
    assert (
        v3.match_scheduler_calls(task, target, [alpha, beta])
        is None
    )
    assert (
        v3.match_scheduler_calls(task, target, [alpha, alpha, beta])
        is not None
    )


class FixedResponseClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def chat(
        self,
        messages,
        tools,
        sampling,
        seed,
        *,
        parallel_tool_calls=False,
    ):
        self.requests.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
        return self.response


class FixedJudge:
    def __init__(self, passed):
        self.passed = passed
        self.requests = []

    def judge(self, **request):
        self.requests.append(request)
        return v3.RefusalDecision(self.passed, "test_judge")


def response_with_calls(calls, *, content=""):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {
                    "content": content,
                    "reasoning_content": "private thinking",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for index, call in enumerate(calls)
                    ],
                },
            }
        ]
    }


def test_vllm_reasoning_field_is_preserved():
    lookup = tool_call("lookup", {"key": "alpha"}, {"value": 1})
    task = make_task([{"mode": "read", "calls": [lookup]}])
    response = response_with_calls([lookup])
    message = response["choices"][0]["message"]
    message["reasoning"] = message.pop("reasoning_content")
    checker = v3.InteractivePassKV3Checker(
        FixedResponseClient(response), FixedJudge(True), pass_k=1, workers=1
    )
    event = checker._generate_event(checker.build_states([task])[0])
    assert event["matched"] is True
    assert event["reasoning_content"] == "private thinking"


def test_no_tool_stop_is_empty_and_unlocks_next_turn():
    lookup = tool_call("lookup", {"key": "alpha"}, {"value": 1})
    task = make_task(
        [
            {"mode": "read", "calls": [lookup], "turn_index": 1},
        ],
        queries=["Please look it up.", "Use alpha."],
        responses=["Which key should I look up?", "The value is 1."],
    )
    task.user_turns[0].update(
        {
            "no_tool_target": True,
            "no_tool_reason": "missing_argument",
            "no_tool_certificate": {"passed": True},
            "sft_supervision": True,
        }
    )
    task.user_turns[1]["sft_supervision"] = True
    v3.prepare_next_action_tasks([task], trust_projected_parallel=True)
    client = FixedResponseClient(response_with_calls([], content=""))
    judge = FixedJudge(True)
    checker = v3.InteractivePassKV3Checker(
        client, judge, pass_k=1, workers=1
    )
    state = checker.build_states([task])[0]
    event = checker._generate_event(state)
    assert event["matched"] is True
    assert event["scheduler_mode"] == "no_tool_stop"
    assert event["tool_only_compliant"] is True
    checker.apply_event(state, event)
    assert all(
        v3._tool_name(tool) != "refuse" for tool in client.requests[0]["tools"]
    )
    assert client.requests[0]["parallel_tool_calls"] is True
    assert state.messages[-2] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "private thinking",
    }
    assert state.messages[-1] == {"role": "user", "content": "Use alpha."}
    assert not any(
        message.get("role") == "tool"
        and message.get("name") == "refuse"
        for message in state.messages
    )


def test_no_tool_target_rejects_calls_but_tracks_visible_prose_separately():
    refusal = tool_call(
        "refuse",
        {"reason": "ambiguity"},
        {"status": "refused", "reason": "ambiguity"},
    )
    task = make_task(
        [],
        queries=["Delete that file."],
        responses=["Which file did you mean?"],
    )
    task.user_turns[0].update(
        {
            "no_tool_target": True,
            "no_tool_reason": "ambiguity",
            "no_tool_certificate": {"passed": True},
            "sft_supervision": True,
        }
    )
    v3.prepare_next_action_tasks([task], trust_projected_parallel=True)
    synthetic_client = FixedResponseClient(response_with_calls([refusal]))
    checker = v3.InteractivePassKV3Checker(
        synthetic_client, FixedJudge(True), pass_k=1, workers=1
    )
    state = checker.build_states([task])[0]
    event = checker._generate_event(state)
    assert event["matched"] is False
    assert event["failure"] in {"parse_error", "tool_call_for_no_tool_target"}

    irrelevant_client = FixedResponseClient(
        response_with_calls([], content="How are you today?")
    )
    checker = v3.InteractivePassKV3Checker(
        irrelevant_client, FixedJudge(False), pass_k=1, workers=1
    )
    state = checker.build_states([task])[0]
    event = checker._generate_event(state)
    assert event["matched"] is True
    assert event["failure"] == ""
    assert event["tool_only_compliant"] is False
    checker.apply_event(state, event)
    assert state.status == "success"
    assert not checker.refusal_judge.requests


class GoldReplayClient:
    def __init__(self, task):
        self.task = task
        self.schedule = v3.build_schedule(task)
        self.matched = set()
        self.requests = []

    def chat(
        self,
        messages,
        tools,
        sampling,
        seed,
        *,
        parallel_tool_calls=False,
    ):
        target = v3.scheduler_target(self.task, self.matched, self.schedule)
        self.requests.append((list(messages), list(tools), parallel_tool_calls))
        if target.segment.mode in {"no_tool_stop", "terminal_stop"}:
            self.matched.add(target.ready_step_indices[0])
            return response_with_calls([], content="")
        if target.segment.mode == "certified_parallel":
            step_index = target.ready_step_indices[0]
            calls = list(self.task.gold_steps[step_index]["calls"])
            calls.reverse()
            self.matched.add(step_index)
            return response_with_calls(calls)
        step_index = target.ready_step_indices[0]
        self.matched.add(step_index)
        return response_with_calls(self.task.gold_steps[step_index]["calls"])


def test_full_canonical_gold_replay_and_policy_visibility():
    path = external_fixture(
        "canonical_sft_rl_corpus_565_no_claude_20260803.jsonl"
    )
    tasks, _ = legacy.load_tasks(path, tool_scope="declared")
    judge = FixedJudge(True)
    checker = v3.InteractivePassKV3Checker(
        FixedResponseClient({}), judge, pass_k=1, workers=1
    )
    failures = []
    for task in tasks:
        client = GoldReplayClient(task)
        checker.client = client
        state = checker.build_states([task])[0]
        while state.status == "active":
            event = checker._generate_event(state)
            checker.apply_event(state, event)
        if state.status != "success":
            failures.append((task.position, state.failure))
        assert all(
            v3._tool_name(tool) != "refuse"
            for _, tools, _ in client.requests
            for tool in tools
        )
        assert all(flag is True for _, _, flag in client.requests)
        assert "DO_NOT_SHOW" not in json.dumps(
            client.requests[0][0], ensure_ascii=False
        )
        revealed_queries = [
            message["content"]
            for message in client.requests[0][0]
            if message["role"] == "user"
        ]
        assert revealed_queries == [task.user_turns[0]["query"]]
    assert failures == []
