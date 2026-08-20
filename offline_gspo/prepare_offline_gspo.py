#!/usr/bin/env python3
"""Build a fail-closed offline-GSPO view of the Qwen3.5-2B pass@32 run.

The source evaluator stored structured OpenAI responses rather than raw token
strings.  This program replays every admitted rollout through the exact v8
interactive checker, canonically renders it with the signed Qwen template and
packs all assistant decisions from one user turn into one Arrow row.  It never
packs across a user-turn boundary: Qwen's template intentionally drops old
reasoning once a new user message appears.

Admission is deliberately narrow and homogeneous.  Each (task, temperature)
group starts with 32 samples and is kept iff it has 1..25 successes.  API-error
episodes and explicitly signed, non-reconstructable archive records are
removed, then the group reward mean/std and episode advantages are recomputed
over the remaining samples.  All other policy failures stay as negative
rollouts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE_ROOT = REPO_ROOT.parent
BUNDLED_PASSK_ROOT = REPO_ROOT / "evaluation" / "passk"
TOOL_SYNTH_ROOT = (
    BUNDLED_PASSK_ROOT
    if (BUNDLED_PASSK_ROOT / "check_apigen_trajectories_passk_v3.py").is_file()
    else WORKSPACE_ROOT / "tool_synth"
)

DEFAULT_PASS32_ROOT = (
    WORKSPACE_ROOT
    / "tool_synth"
    / "results/pass32_qwen35_2b_4b_reasoning_apigen1391_promptv4_20260817"
)
DEFAULT_DATASET = (
    WORKSPACE_ROOT
    / "qwen35_toolonly_sft_sweep_artifacts/data/"
    "apigen_toolonly_sft_next_action_targeted200_v1.jsonl"
)
DEFAULT_TOOL_POOL = (
    REPO_ROOT
    / "magnet_tool_extraction/"
    "bfcl_v3_tools_with_outputs.jsonl"
)
DEFAULT_PROMPT = TOOL_SYNTH_ROOT / "prompts/reasoning_next_action_system_v4.txt"
DEFAULT_TEMPLATE = REPO_ROOT / "templates/qwen35_toolonly_base.jinja"
DEFAULT_MODEL = (
    WORKSPACE_ROOT
    / "models/models--Qwen--Qwen3.5-2B/snapshots/"
    "15852e8c16360a2fea060d615a32b45270f8a8fc"
)

PROTOCOL = "apigen-semantic-reasoning-passk-v8"
CONDITIONS = (("t0p7", 0.7), ("t1p0", 1.0))
EXPECTED_MODEL = "qwen3.5-2b"
EXPECTED_TASKS = 1391
EXPECTED_SAMPLES = 32
MIN_SUCCESSES = 1
MAX_SUCCESSES = 25  # strictly below 80% of 32
FORMAT_VERSION = "offline-gspo-qwen35-interactive-v1"

# The OpenAI archive retained decoded text and token usage, not sampled token
# IDs.  Exactly one selected length-truncated response is not encode/decode
# round-trip stable: its decoded text re-encodes to 8,191 tokens while vLLM
# recorded 8,192.  Never invent the missing token.  Exclude that one signed
# source record and recompute its group's advantages.  The full-corpus builder
# requires this exact digest, so a changed source fails rather than being
# silently dropped.
ARCHIVE_SURFACE_EXCLUSIONS = {
    ("t0p7", 1082, 11): {
        "rollout_sha256": "f5ba19e33b004eafeb4a3837ef6b025b7d2de9d543a264a0098c24f63e2b3c31",
        "reason": "length_completion_decoded_text_reencodes_to_8191_not_8192",
    }
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Condition:
    tag: str
    temperature: float
    directory: Path
    config: dict[str, Any]
    summary: dict[str, Any]
    eligible_positions: tuple[int, ...]

    @property
    def rollouts_path(self) -> Path:
        return self.directory / "rollouts.jsonl"


@dataclass(frozen=True)
class SourceBundle:
    pass32_root: Path
    dataset: Path
    dataset_manifest: Path
    tool_pool: Path
    prompt: Path
    template: Path
    model: Path
    conditions: tuple[Condition, ...]
    source_hashes: dict[str, str]
    projection_manifest: dict[str, Any]


def _import_checker_modules() -> tuple[Any, Any]:
    tool_synth = str(TOOL_SYNTH_ROOT)
    if tool_synth not in sys.path:
        sys.path.insert(0, tool_synth)
    import check_apigen_trajectories_passk as legacy
    import check_apigen_trajectories_passk_v3 as checker_v8

    require(
        checker_v8.PROTOCOL_VERSION == PROTOCOL,
        f"Imported checker protocol is {checker_v8.PROTOCOL_VERSION!r}",
    )
    return legacy, checker_v8


def _task_results(summary: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = summary.get("task_results")
    require(isinstance(rows, list), "summary.task_results must be a list")
    result: dict[int, dict[str, Any]] = {}
    for item in rows:
        require(isinstance(item, dict), "task result must be an object")
        position = int(item["row_position"])
        require(position not in result, f"Duplicate task result {position}")
        result[position] = item
    return result


def select_condition_positions(summary: Mapping[str, Any]) -> tuple[int, ...]:
    """Select literal homogeneous GSPO groups: 1..25 successes out of 32."""

    rows = _task_results(summary)
    selected: list[int] = []
    for position, item in sorted(rows.items()):
        samples = int(item.get("num_samples", -1))
        successes = int(item.get("successful_rollouts", -1))
        require(samples == EXPECTED_SAMPLES, f"Row {position} has n={samples}")
        reported_rate = float(item.get("rollout_success_rate", -1.0))
        require(
            math.isclose(reported_rate, successes / samples, abs_tol=1e-12),
            f"Row {position} has inconsistent success rate",
        )
        if MIN_SUCCESSES <= successes <= MAX_SUCCESSES:
            selected.append(position)
    return tuple(selected)


def validate_sources(
    *,
    pass32_root: Path,
    dataset: Path,
    tool_pool: Path,
    prompt: Path,
    template: Path,
    model: Path,
) -> SourceBundle:
    """Validate every signed contract used to reconstruct source rollouts."""

    pass32_root = pass32_root.resolve()
    dataset = dataset.resolve()
    tool_pool = tool_pool.resolve()
    prompt = prompt.resolve()
    template = template.resolve()
    model = model.resolve()
    required_paths = (pass32_root, dataset, tool_pool, prompt, template, model)
    for path in required_paths:
        require(path.exists(), f"Required source is missing: {path}")

    legacy, checker_v8 = _import_checker_modules()
    del legacy
    projection = checker_v8.validate_projection_manifest(dataset)
    dataset_manifest = Path(projection["path"]).resolve()
    dataset_sha = sha256_file(dataset)
    prompt_sha = sha256_file(prompt)
    prompt_content = prompt.read_text(encoding="utf-8").strip()
    prompt_content_sha = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()
    template_sha = sha256_file(template)
    require(
        dataset_sha == projection["dataset_sha256"],
        "Dataset differs from its projection manifest",
    )
    require(
        template_sha
        == projection["training_contract"].get("chat_template_sha256"),
        "Chat template differs from the signed projection contract",
    )
    require(
        checker_v8.SYSTEM_PROMPT == prompt_content,
        "Imported checker prompt differs from the requested prompt",
    )

    status = read_json(pass32_root / "status.json")
    status_conditions = status.get("conditions") or {}
    conditions: list[Condition] = []
    source_hashes = {
        "dataset": dataset_sha,
        "dataset_manifest": sha256_file(dataset_manifest),
        "tool_pool": sha256_file(tool_pool),
        "system_prompt": prompt_sha,
        "system_prompt_stripped_content": prompt_content_sha,
        "chat_template": template_sha,
        "checker_v8": sha256_file(Path(checker_v8.__file__).resolve()),
        "checker_legacy": sha256_file(
            TOOL_SYNTH_ROOT / "check_apigen_trajectories_passk.py"
        ),
        "pass32_status": sha256_file(pass32_root / "status.json"),
        "model_config": sha256_file(model / "config.json"),
    }
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        path = model / name
        if path.exists():
            source_hashes[f"model/{name}"] = sha256_file(path)

    invariant_configs: list[dict[str, Any]] = []
    for tag, temperature in CONDITIONS:
        directory = pass32_root / "full/qwen35_2b" / tag
        config_path = directory / "config.json"
        summary_path = directory / "summary.json"
        rollouts_path = directory / "rollouts.jsonl"
        for path in (config_path, summary_path, rollouts_path):
            require(path.exists(), f"Missing completed condition artifact: {path}")
        config = read_json(config_path)
        summary = read_json(summary_path)
        status_item = status_conditions.get(f"qwen35_2b/{tag}") or {}
        require(status_item.get("returncode") == 0, f"{tag} did not finish cleanly")
        progress = status_item.get("progress") or {}
        require(progress.get("active") == 0, f"{tag} still has active rollouts")
        require(
            int(progress.get("total", -1)) == EXPECTED_TASKS * EXPECTED_SAMPLES,
            f"{tag} status has wrong rollout total",
        )
        require(config.get("protocol_version") == PROTOCOL, f"{tag}: protocol")
        require(Path(config["jsonl"]).resolve() == dataset, f"{tag}: dataset path")
        require(Path(config["tool_pool"]).resolve() == tool_pool, f"{tag}: tool pool")
        require(config.get("tool_scope") == "declared", f"{tag}: tool scope")
        require(int(config.get("pass_k", -1)) == EXPECTED_SAMPLES, f"{tag}: k")
        require(config.get("model") == EXPECTED_MODEL, f"{tag}: model")
        require(config.get("enable_thinking") is True, f"{tag}: thinking disabled")
        require(
            config.get("include_initial_state") is False,
            f"{tag}: initial state was policy-visible",
        )
        require(
            math.isclose(
                float((config.get("sampling") or {}).get("temperature", -1)),
                temperature,
                abs_tol=1e-12,
            ),
            f"{tag}: temperature mismatch",
        )
        require(
            Path(config["system_prompt_path"]).resolve() == prompt,
            f"{tag}: prompt path",
        )
        require(
            config.get("system_prompt_sha256") == prompt_content_sha,
            f"{tag}: prompt content SHA",
        )
        require(config.get("system_prompt") == prompt_content, f"{tag}: prompt content")
        chat = config.get("chat_template") or {}
        require(Path(chat["path"]).resolve() == template, f"{tag}: template path")
        require(chat.get("sha256") == template_sha, f"{tag}: template SHA")
        require(chat.get("matches_manifest") is True, f"{tag}: unsigned template")
        require(summary.get("config") == config, f"{tag}: summary/config mismatch")
        require(int(summary.get("num_tasks", -1)) == EXPECTED_TASKS, f"{tag}: tasks")
        require(
            int(summary.get("num_rollouts", -1))
            == EXPECTED_TASKS * EXPECTED_SAMPLES,
            f"{tag}: rollouts",
        )
        rows = _task_results(summary)
        require(set(rows) == set(range(EXPECTED_TASKS)), f"{tag}: row positions")
        selected = select_condition_positions(summary)
        conditions.append(
            Condition(tag, temperature, directory, config, summary, selected)
        )
        source_hashes[f"{tag}/config"] = sha256_file(config_path)
        source_hashes[f"{tag}/summary"] = sha256_file(summary_path)
        source_hashes[f"{tag}/rollouts"] = sha256_file(rollouts_path)

        invariant = copy.deepcopy(config)
        invariant["sampling"]["temperature"] = "<temperature>"
        invariant["vllm_url"] = "<vllm_url>"
        invariant_configs.append(invariant)
    require(
        invariant_configs[0] == invariant_configs[1],
        "2B pass@32 configs differ beyond temperature and serving URL",
    )
    require(
        tuple(len(condition.eligible_positions) for condition in conditions)
        == (629, 665),
        "Unexpected admitted group counts; source pass@32 may have changed",
    )

    return SourceBundle(
        pass32_root=pass32_root,
        dataset=dataset,
        dataset_manifest=dataset_manifest,
        tool_pool=tool_pool,
        prompt=prompt,
        template=template,
        model=model,
        conditions=tuple(conditions),
        source_hashes=source_hashes,
        projection_manifest=projection,
    )


def binary_group_advantages(rewards: Sequence[int]) -> list[float]:
    """Sample-standardize binary rewards within one behavior group.

    This intentionally matches ``torch.std`` (correction=1) in verl's GRPO
    advantage estimator.  GSPO changes the policy ratio/loss level, not the
    group-relative outcome advantage.
    """

    require(len(rewards) >= 2, "A GSPO group needs at least two episodes")
    require(all(reward in (0, 1) for reward in rewards), "Rewards must be binary")
    mean = sum(rewards) / len(rewards)
    require(0.0 < mean < 1.0, "A GSPO group must contain both rewards")
    n = len(rewards)
    std = math.sqrt(mean * (1.0 - mean) * n / (n - 1))
    result = [(reward - mean) / std for reward in rewards]
    require(abs(sum(result) / len(result)) < 1e-12, "Advantages are not centered")
    return result


def _normalise_template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI JSON argument strings to the Jinja's mapping surface."""

    result = copy.deepcopy(list(messages))
    for message in result:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                require(isinstance(parsed, dict), "Historical call arguments are not an object")
                function["arguments"] = parsed
            else:
                require(isinstance(arguments, dict), "Call arguments are not an object")
    return result


def _assistant_message(event: Mapping[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": str(event.get("raw_completion") or ""),
        "reasoning_content": str(event.get("reasoning_content") or ""),
    }
    calls = event.get("predicted_calls") or []
    if calls:
        message["tool_calls"] = [
            {
                "id": call.get("tool_call_id")
                or f"reconstructed_{event['turn']}_{index}",
                "type": "function",
                "function": {
                    "name": str(call["name"]),
                    "arguments": dict(call.get("arguments") or {}),
                },
            }
            for index, call in enumerate(calls)
        ]
    return message


def _tool_messages(
    event: Mapping[str, Any], assistant: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not event.get("matched"):
        return []
    mode = str(event.get("scheduler_mode") or "")
    if mode in {"terminal_stop", "no_tool_stop"}:
        return []
    calls = event.get("predicted_calls") or []
    outputs = event.get("tool_outputs") or []
    openai_calls = assistant.get("tool_calls") or []
    require(len(calls) == len(outputs) == len(openai_calls), "Call/output mismatch")
    return [
        {
            "role": "tool",
            "tool_call_id": openai_call["id"],
            "name": str(call["name"]),
            "content": json.dumps(output, ensure_ascii=False),
        }
        for call, output, openai_call in zip(calls, outputs, openai_calls)
    ]


def _render(tokenizer: Any, messages: Sequence[Mapping[str, Any]], tools: Any, *, generate: bool) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=generate,
        enable_thinking=True,
    )


def _ids(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    result = [int(value) for value in values]
    require(all(-(2**31) <= value < 2**31 for value in result), "Token ID overflows int32")
    return result


def _validate_event_target(
    *,
    checker_v8: Any,
    checker: Any,
    state: Any,
    event: Mapping[str, Any],
) -> list[dict[str, Any]]:
    require(int(event["turn"]) == state.next_turn, "Non-consecutive event")
    target = checker_v8.scheduler_target(
        state.task,
        state.matched_gold_step_indices,
        checker._schedule(state.task),
    )
    require(target is not None, "Event exists after the scheduler finished")
    require(
        int(event["gold_user_turn_index"]) == target.segment.turn_index,
        "Gold user-turn mismatch",
    )
    require(event.get("scheduler_mode") == target.segment.mode, "Scheduler mode mismatch")
    require(
        tuple(int(value) for value in event.get("ready_step_indices") or [])
        == target.ready_step_indices,
        "Ready-set mismatch",
    )
    if "segment_index" in event:
        require(
            int(event["segment_index"]) == target.segment.segment_index,
            "Scheduler segment mismatch",
        )
    tools = checker_v8.tools_for_turn(state.task, target.segment.turn_index)
    names = [checker_v8._tool_name(tool) for tool in tools]
    require(names == list(event.get("available_tool_names") or []), "Tool snapshot mismatch")
    return tools


def _flush_segment(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    max_length: int,
    full_text_override: str | None = None,
) -> tuple[dict[str, Any], list[int]]:
    require(targets, "Cannot flush an empty segment")
    if full_text_override is None:
        full_text_with_newline = _render(
            tokenizer, messages, tools, generate=False
        )
        require(
            full_text_with_newline.endswith("\n"),
            "Template lost its final delimiter newline",
        )
        # This newline is appended by the template after <|im_end|>; it was
        # not a sampled assistant token and must not participate in a ratio.
        full_text = full_text_with_newline[:-1]
    else:
        # A length-truncated response has no closing message to render.  Its
        # archived reasoning/content fragment is appended to the exact prompt
        # verbatim and must remain unfinished.
        full_text = full_text_override
    input_ids = _ids(tokenizer, full_text)
    require(len(input_ids) <= max_length, f"Packed segment has {len(input_ids)} > {max_length} tokens")
    target_positions: list[int] = []
    completion_deltas: list[int] = []
    previous_end = -1
    for target in targets:
        prompt_text = target["prompt_text"]
        assistant_text = target["assistant_text"]
        require(full_text.startswith(prompt_text), "Packed prompt is not a text prefix")
        require(full_text.startswith(assistant_text), "Assistant render is not a text prefix")
        prompt_ids = _ids(tokenizer, prompt_text)
        assistant_ids = _ids(tokenizer, assistant_text)
        require(input_ids[: len(prompt_ids)] == prompt_ids, "Packed prompt token prefix mismatch")
        require(
            input_ids[: len(assistant_ids)] == assistant_ids,
            "Packed assistant token prefix mismatch",
        )
        start, end = len(prompt_ids), len(assistant_ids)
        require(0 <= start < end <= len(input_ids), "Invalid assistant token span")
        require(start >= previous_end, "Assistant token spans overlap")
        previous_end = end
        require(
            start == int(target["recorded_prompt_tokens"]),
            "Rendered prompt length differs from evaluator usage",
        )
        recorded_completion = int(target["recorded_completion_tokens"])
        delta = (end - start) - recorded_completion
        if target.get("finish_reason") == "length":
            require(delta == 0, "Truncated completion is not exactly reconstructable")
        else:
            # Parsed native calls retain semantic arguments but not their raw
            # XML/whitespace bytes.  Canonical rendering can therefore be a few
            # tokens shorter *or longer*.  The signed delta is retained in the
            # manifest for a full-corpus audit; unlike a length truncation,
            # exact surface recovery is impossible from this archive.
            pass
        completion_deltas.append(delta)
        target_positions.extend(range(start, end))
    require(len(target_positions) == len(set(target_positions)), "Duplicate target token")
    return {"input_ids": input_ids, "target_positions": target_positions}, completion_deltas


def reconstruct_episode(
    *,
    legacy: Any,
    checker_v8: Any,
    checker: Any,
    tokenizer: Any,
    task: Any,
    initial_messages: Sequence[Mapping[str, Any]],
    rollout: Mapping[str, Any],
    condition_tag: str,
    temperature: float,
    advantage: float,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay, validate and pack one rollout into user-turn segments."""

    position = int(rollout["row_position"])
    sample_index = int(rollout["sample_index"])
    episode_id = f"qwen35_2b/{condition_tag}/row{position:04d}/sample{sample_index:02d}"
    group_id = f"qwen35_2b/{condition_tag}/row{position:04d}"
    state = legacy.RolloutState(
        task=task,
        sample_index=sample_index,
        messages=copy.deepcopy(list(initial_messages)),
    )
    segments: list[dict[str, Any]] = []
    segment_messages: list[dict[str, Any]] | None = None
    segment_tools: list[dict[str, Any]] | None = None
    segment_turn: int | None = None
    segment_targets: list[dict[str, Any]] = []
    segment_full_text_override: str | None = None
    all_completion_deltas: list[int] = []
    events = rollout.get("events") or []
    require(events, f"{episode_id}: rollout has no events")

    for event_index, event in enumerate(events):
        require(event.get("failure") != "api_error", f"{episode_id}: api_error was not filtered")
        tools = _validate_event_target(
            checker_v8=checker_v8,
            checker=checker,
            state=state,
            event=event,
        )
        turn_index = int(event["gold_user_turn_index"])
        current_messages = _normalise_template_messages(state.messages)
        if segment_messages is None:
            segment_messages = current_messages
            segment_tools = tools
            segment_turn = turn_index
        else:
            require(turn_index == segment_turn, "Internal segment crossed a user turn")
            require(segment_tools == tools, "Tools changed inside one user turn")
            require(current_messages == segment_messages, "Checker/message replay state mismatch")

        prompt_text = _render(tokenizer, segment_messages, segment_tools, generate=True)
        usage = event.get("usage") or {}
        require(isinstance(usage.get("prompt_tokens"), int), f"{episode_id}: missing prompt usage")
        require(
            isinstance(usage.get("completion_tokens"), int),
            f"{episode_id}: missing completion usage",
        )
        assistant = _assistant_message(event)
        finish_reason = str(event.get("finish_reason") or "")
        if finish_reason == "length":
            require(event_index + 1 == len(events), "Truncated event is not final")
            require(not event.get("predicted_calls"), "Truncated event has parsed calls")
            reasoning = str(event.get("reasoning_content") or "")
            visible = str(event.get("raw_completion") or "")
            require(reasoning, "Truncated event lost its observed reasoning")
            fragment = reasoning
            if visible:
                fragment += "\n</think>\n\n" + visible
            assistant_text = prompt_text + fragment
            segment_full_text_override = assistant_text
        else:
            assistant_render_with_newline = _render(
                tokenizer,
                segment_messages + [assistant],
                segment_tools,
                generate=False,
            )
            require(
                assistant_render_with_newline.endswith("\n"),
                "Assistant render has no template delimiter newline",
            )
            assistant_text = assistant_render_with_newline[:-1]
        require(assistant_text.startswith(prompt_text), "Assistant does not extend prompt")
        segment_targets.append(
            {
                "prompt_text": prompt_text,
                "assistant_text": assistant_text,
                "recorded_prompt_tokens": int(usage["prompt_tokens"]),
                "recorded_completion_tokens": int(usage["completion_tokens"]),
                "finish_reason": finish_reason,
            }
        )
        if finish_reason != "length":
            segment_messages.append(assistant)
            segment_messages.extend(_tool_messages(event, assistant))

        checker_v8.InteractivePassKV3Checker.apply_event(state, event)
        next_turn = (
            int(events[event_index + 1]["gold_user_turn_index"])
            if event_index + 1 < len(events)
            else None
        )
        if next_turn == turn_index:
            require(
                _normalise_template_messages(state.messages) == segment_messages,
                "Post-event checker/message state mismatch",
            )
            continue

        packed, deltas = _flush_segment(
            tokenizer=tokenizer,
            messages=segment_messages,
            tools=segment_tools,
            targets=segment_targets,
            max_length=max_length,
            full_text_override=segment_full_text_override,
        )
        segments.append(packed)
        all_completion_deltas.extend(deltas)
        segment_messages = None
        segment_tools = None
        segment_turn = None
        segment_targets = []
        segment_full_text_override = None

    replayed = checker_v8.rollout_record(state)
    for key in (
        "row_position",
        "sample_index",
        "status",
        "success",
        "failure",
        "matched_steps",
        "matched_calls",
        "matched_gold_step_indices",
        "matched_gold_indices",
        "num_gold_steps",
        "num_gold_calls",
        "data_issues",
        "calls",
        "protocol_version",
    ):
        require(replayed.get(key) == rollout.get(key), f"{episode_id}: replay mismatch in {key}")
    require(segment_messages is None, f"{episode_id}: unflushed segment")
    reward = float(bool(rollout.get("success")))
    episode_target_tokens = sum(len(segment["target_positions"]) for segment in segments)
    require(episode_target_tokens > 0, f"{episode_id}: no assistant target tokens")
    segment_count = len(segments)
    rows = [
        {
            "segment_index": index,
            "episode_id": episode_id,
            "group_id": group_id,
            "row_position": position,
            "sample_index": sample_index,
            "temperature": float(temperature),
            "reward": reward,
            "advantage": float(advantage),
            "input_ids": segment["input_ids"],
            "target_positions": segment["target_positions"],
            "episode_segment_count": segment_count,
            "episode_target_tokens": episode_target_tokens,
        }
        for index, segment in enumerate(segments)
    ]
    audit = {
        "episode_id": episode_id,
        "group_id": group_id,
        "row_position": position,
        "sample_index": sample_index,
        "temperature": temperature,
        "reward": reward,
        "advantage": advantage,
        "segment_count": segment_count,
        "segment_indices": list(range(segment_count)),
        "target_tokens": episode_target_tokens,
        "event_count": len(events),
        "status": rollout.get("status"),
        "failure": rollout.get("failure"),
        "completion_token_deltas": all_completion_deltas,
    }
    return rows, audit


def _iter_rollout_groups(path: Path) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    current_position: int | None = None
    group: list[dict[str, Any]] = []
    seen_positions = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rollout = json.loads(line)
            position = int(rollout["row_position"])
            if current_position is None:
                current_position = position
            if position != current_position:
                require(position == current_position + 1, f"Non-contiguous row at line {line_number}")
                yield current_position, group
                seen_positions += 1
                current_position = position
                group = []
            group.append(rollout)
    if current_position is not None:
        yield current_position, group
        seen_positions += 1
    require(seen_positions == EXPECTED_TASKS, f"Rollout file has {seen_positions} task groups")


def generate_rows(
    *,
    dataset: str,
    tool_pool: str,
    template: str,
    model: str,
    conditions_json: str,
    max_length: int,
    audit_path: str,
) -> Iterator[dict[str, Any]]:
    """Generator entrypoint used by ``datasets.Dataset.from_generator``."""

    # ``datasets`` treats every list-valued ``gen_kwargs`` item as a set of
    # input shards, even with ``num_proc=1``.  Pass this as one scalar JSON
    # payload so both sampling conditions share one audit stream and one
    # monotonically increasing dataset-row cursor.
    conditions = json.loads(conditions_json)
    require(
        isinstance(conditions, list)
        and all(isinstance(item, dict) for item in conditions),
        "conditions_json must encode a list of condition objects",
    )

    legacy, checker_v8 = _import_checker_modules()
    tasks, _ = legacy.load_tasks(dataset, tool_pool, tool_scope="declared")
    for task in tasks:
        task.tools = checker_v8._canonical_policy_tools(task.tools)
        for turn in task.user_turns:
            turn["assistant_response"] = ""
    preparation = checker_v8.prepare_next_action_tasks(
        tasks, trust_projected_parallel=True
    )
    require(preparation.get("tasks") == EXPECTED_TASKS, "Prepared task count changed")
    checker = checker_v8.InteractivePassKV3Checker(
        None,
        None,
        pass_k=1,
        include_initial_state=False,
        current_date="2026-08-17",
    )
    base_states = checker.build_states(tasks)
    tasks_by_position = {state.task.position: state.task for state in base_states}
    initial_messages = {
        state.task.position: copy.deepcopy(state.messages) for state in base_states
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.chat_template = Path(template).read_text(encoding="utf-8")

    dataset_row_cursor = 0
    observed_archive_exclusions: set[tuple[str, int, int]] = set()
    audit_output = Path(audit_path)
    require(
        not audit_output.exists(),
        "reconstruction generator was invoked more than once for one audit",
    )
    with audit_output.open("x", encoding="utf-8") as audit_handle:
        for condition in conditions:
            tag = str(condition["tag"])
            temperature = float(condition["temperature"])
            selected = set(int(value) for value in condition["eligible_positions"])
            expected_successes = {
                int(key): int(value)
                for key, value in condition["expected_successes"].items()
            }
            observed_selected: set[int] = set()
            for position, source_group in _iter_rollout_groups(
                Path(condition["rollouts_path"])
            ):
                require(len(source_group) == EXPECTED_SAMPLES, f"{tag}/row{position}: n != 32")
                require(
                    {int(item["sample_index"]) for item in source_group}
                    == set(range(EXPECTED_SAMPLES)),
                    f"{tag}/row{position}: sample indices are incomplete",
                )
                require(
                    all(item.get("protocol_version") == PROTOCOL for item in source_group),
                    f"{tag}/row{position}: rollout protocol mismatch",
                )
                require(
                    sum(bool(item.get("success")) for item in source_group)
                    == expected_successes[position],
                    f"{tag}/row{position}: summary/rollout reward mismatch",
                )
                if position not in selected:
                    continue
                observed_selected.add(position)
                episodes = []
                for item in sorted(
                    source_group, key=lambda value: int(value["sample_index"])
                ):
                    if item.get("failure") == "api_error" or any(
                        event.get("failure") == "api_error"
                        for event in item.get("events") or []
                    ):
                        continue
                    exclusion_key = (tag, position, int(item["sample_index"]))
                    exclusion = ARCHIVE_SURFACE_EXCLUSIONS.get(exclusion_key)
                    if exclusion is not None:
                        require(
                            sha256_json(item) == exclusion["rollout_sha256"],
                            f"{tag}/row{position}/sample{item['sample_index']}: "
                            "archive exclusion digest changed",
                        )
                        observed_archive_exclusions.add(exclusion_key)
                        continue
                    episodes.append(item)
                require(
                    len(episodes) in {31, 32},
                    f"{tag}/row{position}: too many source-integrity exclusions",
                )
                rewards = [int(bool(item.get("success"))) for item in episodes]
                advantages = binary_group_advantages(rewards)
                group_successes = sum(rewards)
                for rollout, advantage in zip(episodes, advantages):
                    rows, audit = reconstruct_episode(
                        legacy=legacy,
                        checker_v8=checker_v8,
                        checker=checker,
                        tokenizer=tokenizer,
                        task=tasks_by_position[position],
                        initial_messages=initial_messages[position],
                        rollout=rollout,
                        condition_tag=tag,
                        temperature=temperature,
                        advantage=advantage,
                        max_length=max_length,
                    )
                    audit["dataset_row_start"] = dataset_row_cursor
                    audit["dataset_row_end"] = dataset_row_cursor + len(rows)
                    audit["group_size"] = len(episodes)
                    audit["group_successes"] = group_successes
                    audit_handle.write(canonical_json(audit) + "\n")
                    dataset_row_cursor += len(rows)
                    yield from rows
            require(observed_selected == selected, f"{tag}: selected groups missing from rollouts")
        require(
            observed_archive_exclusions == set(ARCHIVE_SURFACE_EXCLUSIONS),
            "Expected archive-surface exclusions were not observed exactly once",
        )


def dataset_features() -> Any:
    from datasets import Features, Sequence as HFSequence, Value

    return Features(
        {
            "segment_index": Value("int32"),
            "episode_id": Value("string"),
            "group_id": Value("string"),
            "row_position": Value("int32"),
            "sample_index": Value("int32"),
            "temperature": Value("float32"),
            "reward": Value("float32"),
            "advantage": Value("float32"),
            "input_ids": HFSequence(Value("int32")),
            "target_positions": HFSequence(Value("int32")),
            "episode_segment_count": Value("int32"),
            "episode_target_tokens": Value("int32"),
        }
    )


def _load_audit(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def build_manifest(
    *,
    bundle: SourceBundle,
    output_dir: Path,
    audit_rows: Sequence[Mapping[str, Any]],
    dataset_rows: int,
    max_length: int,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    episodes: dict[str, Any] = {}
    delta_counts: Counter[int] = Counter()
    failure_counts: Counter[str] = Counter()
    segment_count = 0
    target_tokens = 0
    rewards = Counter()
    for episode in audit_rows:
        episode_id = str(episode["episode_id"])
        require(episode_id not in episodes, f"Duplicate episode {episode_id}")
        groups.setdefault(str(episode["group_id"]), []).append(episode)
        row_start = int(episode["dataset_row_start"])
        row_end = int(episode["dataset_row_end"])
        count = int(episode["segment_count"])
        require(row_end - row_start == count, f"{episode_id}: non-contiguous segments")
        episodes[episode_id] = {
            "dataset_row_start": row_start,
            "dataset_row_end": row_end,
            "segment_indices": list(episode["segment_indices"]),
            "segment_count": count,
            "target_tokens": int(episode["target_tokens"]),
        }
        segment_count += count
        target_tokens += int(episode["target_tokens"])
        rewards[str(int(float(episode["reward"])))] += 1
        failure_counts[str(episode.get("failure") or "success")] += 1
        delta_counts.update(int(value) for value in episode["completion_token_deltas"])
    require(segment_count == dataset_rows, "Audit and Arrow row counts disagree")

    group_stats: dict[str, Any] = {}
    for group_id, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda value: int(value["sample_index"]))
        group_rewards = [int(float(value["reward"])) for value in ordered]
        require(0 < sum(group_rewards) < len(group_rewards), f"{group_id}: no reward variance")
        advantages = [float(value["advantage"]) for value in ordered]
        require(abs(sum(advantages) / len(advantages)) < 2e-6, f"{group_id}: uncentered")
        group_stats[group_id] = {
            "episodes": len(ordered),
            "successes": sum(group_rewards),
            "temperature": float(ordered[0]["temperature"]),
            "row_position": int(ordered[0]["row_position"]),
        }

    output_hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            output_hashes[str(path.relative_to(output_dir))] = sha256_file(path)
    selected_by_condition = {
        condition.tag: {
            "temperature": condition.temperature,
            "source_groups": len(condition.eligible_positions),
            "source_episodes": len(condition.eligible_positions) * EXPECTED_SAMPLES,
        }
        for condition in bundle.conditions
    }
    return {
        "format_version": FORMAT_VERSION,
        "objective": (
            "Offline GSPO: one ratio per full interactive episode, computed as "
            "the mean token log-ratio over all target_positions in all contiguous "
            "segments sharing episode_id"
        ),
        "selection": {
            "group_key": "(row_position, temperature)",
            "source_group_size": EXPECTED_SAMPLES,
            "minimum_successes_inclusive": MIN_SUCCESSES,
            "maximum_successes_inclusive": MAX_SUCCESSES,
            "rate_interval": "[1/32, 25/32] (at least once, strictly below 80%)",
            "api_error_policy": "drop episode, recompute group advantage",
            "archive_surface_policy": (
                "drop only explicitly signed decoded-text records whose sampled "
                "token IDs cannot be reconstructed; recompute group advantage"
            ),
            "archive_surface_exclusions": [
                {
                    "condition": key[0],
                    "row_position": key[1],
                    "sample_index": key[2],
                    **value,
                }
                for key, value in sorted(ARCHIVE_SURFACE_EXCLUSIONS.items())
            ],
            "other_failure_policy": "retain as reward=0",
            "conditions": selected_by_condition,
        },
        "rendering_contract": {
            "protocol": PROTOCOL,
            "enable_thinking": True,
            "include_initial_state": False,
            "pack_unit": "all reached events in one gold_user_turn_index",
            "never_cross_user_turn": True,
            "final_template_newline_after_im_end_is_target": False,
            "prompt_token_count_validation": "exact against evaluator usage",
            "completion_token_count_validation": (
                "exact for retained length truncations; signed delta audited "
                "for parsed native calls because vLLM did not archive XML/"
                "whitespace bytes"
            ),
            "max_length": max_length,
        },
        "advantage": {
            "level": "episode",
            "normalization_group": "(row_position, temperature)",
            "standard_deviation": "sample (Bessel correction, matching torch.std/verl)",
            "epsilon": 0.0,
        },
        "counts": {
            "groups": len(groups),
            "episodes": len(episodes),
            "segments": segment_count,
            "target_tokens": target_tokens,
            "rewards": dict(sorted(rewards.items())),
            "failures": dict(sorted(failure_counts.items())),
            "completion_token_delta_canonical_minus_recorded": {
                str(key): value for key, value in sorted(delta_counts.items())
            },
        },
        "episodes": episodes,
        "groups": group_stats,
        "sources": {
            "pass32_root": str(bundle.pass32_root),
            "dataset": str(bundle.dataset),
            "dataset_manifest": str(bundle.dataset_manifest),
            "tool_pool": str(bundle.tool_pool),
            "system_prompt": str(bundle.prompt),
            "chat_template": str(bundle.template),
            "tokenizer_model": str(bundle.model),
            "hashes": bundle.source_hashes,
        },
        "output_hashes_excluding_manifest": output_hashes,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pass32-root", type=Path, default=DEFAULT_PASS32_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tool-pool", type=Path, default=DEFAULT_TOOL_POOL)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate signed inputs and print selection counts without tokenizing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.max_length > 0, "--max-length must be positive")
    bundle = validate_sources(
        pass32_root=args.pass32_root,
        dataset=args.dataset,
        tool_pool=args.tool_pool,
        prompt=args.system_prompt,
        template=args.chat_template,
        model=args.model,
    )
    validation = {
        "protocol": PROTOCOL,
        "groups": {
            condition.tag: len(condition.eligible_positions)
            for condition in bundle.conditions
        },
        "source_episodes": {
            condition.tag: len(condition.eligible_positions) * EXPECTED_SAMPLES
            for condition in bundle.conditions
        },
        "source_hashes": bundle.source_hashes,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    from datasets import Dataset

    condition_specs = [
        {
            "tag": condition.tag,
            "temperature": condition.temperature,
            "rollouts_path": str(condition.rollouts_path),
            "eligible_positions": list(condition.eligible_positions),
            "expected_successes": {
                str(position): int(item["successful_rollouts"])
                for position, item in sorted(
                    _task_results(condition.summary).items()
                )
            },
        }
        for condition in bundle.conditions
    ]
    fingerprint = sha256_json(
        {
            "format": FORMAT_VERSION,
            "sources": bundle.source_hashes,
            "max_length": args.max_length,
            "selection": condition_specs,
        }
    )[:64]
    with tempfile.TemporaryDirectory(
        prefix="offline_gspo_build_", dir=str(output_dir.parent)
    ) as temporary:
        temporary_path = Path(temporary)
        audit_path = temporary_path / "reconstruction_audit.jsonl"
        dataset = Dataset.from_generator(
            generate_rows,
            features=dataset_features(),
            cache_dir=str(temporary_path / "cache"),
            keep_in_memory=False,
            gen_kwargs={
                "dataset": str(bundle.dataset),
                "tool_pool": str(bundle.tool_pool),
                "template": str(bundle.template),
                "model": str(bundle.model),
                "conditions_json": canonical_json(condition_specs),
                "max_length": args.max_length,
                "audit_path": str(audit_path),
            },
            num_proc=1,
            fingerprint=fingerprint,
        )
        require(len(dataset) > 0, "Builder produced an empty dataset")
        staged_output = temporary_path / "dataset"
        dataset.save_to_disk(
            str(staged_output), max_shard_size=args.max_shard_size
        )
        shutil.copy2(audit_path, staged_output / "reconstruction_audit.jsonl")
        audit_rows = _load_audit(audit_path)
        manifest = build_manifest(
            bundle=bundle,
            output_dir=staged_output,
            audit_rows=audit_rows,
            dataset_rows=len(dataset),
            max_length=args.max_length,
        )
        (staged_output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged_output, output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "groups": manifest["counts"]["groups"],
                "episodes": manifest["counts"]["episodes"],
                "segments": manifest["counts"]["segments"],
                "target_tokens": manifest["counts"]["target_tokens"],
                "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
