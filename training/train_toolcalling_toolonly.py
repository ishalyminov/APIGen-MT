#!/usr/bin/env python3
"""Qwen3.5 tool-only SFT for the APIGen-MT learning-rate sweep.

This variant supervises one native next action at a time: a sequential tool
call, a complete independent parallel group, or the empty terminal response
after the completed tool loop.  Every example contains the exact golden calls
and returned tool outputs that precede that decision.  Natural-language
assistant summaries from the source data are deliberately excluded: the target
policy emits tools or stops, never prose.  The exact system prompt is loaded
from a file shared with BFCL evaluation, and ``enable_thinking=False`` is used
for every render.

What changed vs v2
------------------
* Two source schemas. The canonical corpus mixes:
    - ``trajectory`` (single task: query + steps + final_response) — like v2.
    - ``conversation`` (true multi-turn: ``turns`` each with ``user_query``,
      ``steps`` and an ``assistant_response``). 311 of these carry parallel
      tool calls (several ``tool_calls`` in one step).
  Both are rendered to native chat messages; parallel calls in a step stay in
  ONE assistant turn (the template then emits several ``<tool_call>`` blocks).
* Inline tool specs. ``available_tools`` may be a list of full function specs
  (dicts) — they are wrapped as ``{"type":"function","function":spec}`` and fed
  straight to the template, no catalog lookup. The synthetic ``refuse`` tool is
  handled this way. Legacy name-based records still fall back to the catalog.
* Robust multi-turn loss mask. v2 derived supervised spans by re-rendering
  message prefixes; that breaks for multi-turn (the Qwen3.5 template has global
  state, so ``render(messages[:i])`` is NOT a prefix of ``render(messages[:i+1])``
  once there is more than one user turn). Instead we render the full transcript
  ONCE and mask every ``<|im_start|>assistant\\n ... <|im_end|>`` segment via
  ``return_offsets_mapping``. This is immune to template global state and works
  for parallel calls, final answers and multi-turn alike (preflight-verified).
* Eligibility filter. Records with ``aggregation_metadata.eligible_for_sft_rl``
  == False are skipped (the canonical corpus ships 24 held-out rows).

Format still uses the model's SHIPPED Jinja chat template via
``tokenizer.apply_chat_template(messages, tools=tools, enable_thinking=False)``
— the ``qwen3_coder`` contract the vLLM parser expects. ``max_length`` drops
(rather than truncates) over-long rows so tool calls are never cut mid-generation.

NOTE: training saves a TEXT-only checkpoint (``Qwen3_5ForCausalLM``) which vLLM
cannot serve directly (hybrid-attention KV-cache paging bug). Run
``convert_to_vllm.py`` afterwards to emit a VL-format serving dir
(``Qwen3_5ForConditionalGeneration``) — see README/launcher.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.tool_schema import (
    TOOL_SCHEMA_PROJECTION,
    canonical_function,
)


# --------------------------------------------------------------------------- #
# Data loading + message construction
# --------------------------------------------------------------------------- #

NO_TOOL_CONTEXT_FREE_VIEW = True
SUPERVISION_CONTRACT = "next_action_group_and_terminal_stop"
PREFIX_UNIT = "one_per_next_action_or_parallel_group_with_golden_history"
BEHAVIOR_CLASSES = ("no_tool", "single_call", "parallel", "other")
DEFAULT_TRAIN_REPEAT_FACTORS = {
    # Do not overweight generic abstention: the local recovery addon already
    # supplies explicit stop-then-resume supervision for missing tools/args.
    "no_tool": 1,
    "single_call": 2,
    "parallel": 2,
    "other": 1,
}

# The source simulator accepted these redundant arguments, but the corresponding
# BFCL callables do not.  Strip them from both the rendered schema and the
# supervised target without mutating the frozen source record.
REMOVED_INPUT_PROPERTIES: dict[str, frozenset[str]] = {
    "book_flight": frozenset({"travel_cost"}),
    "delete_message": frozenset({"message_id"}),
}


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    catalog: dict[str, dict[str, Any]] = {}
    for items in raw.values():
        if not isinstance(items, list):
            continue
        for item in items:
            fn = item.get("function", {})
            name = fn.get("name")
            if isinstance(name, str) and name:
                catalog[name] = fn
    return catalog


def canonical_openai_function(fn: dict[str, Any]) -> dict[str, Any]:
    """Return the exact OpenAI function surface used at train/eval time.

    Rich APIGen records retain simulator-only fields such as ``output_schema``
    and ``category``.  They remain useful in the raw corpus, but exposing them
    to the policy creates a train/eval mismatch and leaks non-input metadata.
    """
    name = fn.get("name")
    removed = REMOVED_INPUT_PROPERTIES.get(name, frozenset())
    return canonical_function(fn, removed_input_properties=removed)


def build_tools(record: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the OpenAI ``tools`` list for ``apply_chat_template(..., tools=...)``.

    Handles two ``available_tools`` flavours:
      * dict specs (canonical corpus, incl. the synthetic ``refuse`` tool) ->
        projected to the standard OpenAI function surface with no catalog lookup.
      * name strings (legacy v2 data) -> resolved through the catalog.
    """
    tools: list[dict[str, Any]] = []
    for entry in record.get("available_tools", []):
        if isinstance(entry, dict):
            fn = entry["function"] if isinstance(entry.get("function"), dict) else entry
            tools.append(
                {"type": "function", "function": canonical_openai_function(fn)}
            )
        elif isinstance(entry, str):
            fn = catalog.get(entry)
            if fn is None:
                raise KeyError(f"Tool {entry!r} missing from catalog")
            tools.append(
                {"type": "function", "function": canonical_openai_function(fn)}
            )
        else:
            raise ValueError(f"unsupported available_tools entry: {entry!r}")
    if not tools:
        raise ValueError("record has no usable available_tools")
    return tools


def to_tool_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_target_args(tool_name: str, arguments: Any) -> dict[str, Any]:
    """Normalize one target call and remove known non-BFCL input fields."""
    normalized = normalize_args(arguments)
    for key in REMOVED_INPUT_PROPERTIES.get(tool_name, frozenset()):
        normalized.pop(key, None)
    return normalized


def _step_to_messages(
    step: dict[str, Any],
    *,
    id_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One trajectory/turn step -> (assistant tool_calls msg, tool result msgs).

    All tool_calls in the step become ONE assistant message (parallel-safe); the
    Jinja template then emits several ``<tool_call>`` blocks in a single turn.
    """
    step_number = step.get("step_number", 0)
    raw_calls = step.get("tool_calls", [])
    if not isinstance(raw_calls, list) or not raw_calls:
        return [], []

    openai_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for idx, call in enumerate(raw_calls):
        name = call.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        call_id = f"{id_prefix}_{step_number}_{idx}"
        openai_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": normalize_target_args(
                        name, call.get("arguments", {})
                    ),
                },
            }
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": to_tool_content(call.get("output", "")),
            }
        )
    if not openai_calls:
        return [], []
    assistant_msg = {"role": "assistant", "content": "", "tool_calls": openai_calls}
    return [assistant_msg], tool_messages


def _decision_snapshots(
    *,
    history: list[dict[str, Any]],
    current_prefix: list[dict[str, Any]],
    id_prefix: str,
    steps: list[dict[str, Any]],
    supervise_turn: bool,
    tool_override: list[dict[str, Any] | str] | None,
) -> tuple[
    list[
        tuple[
            list[dict[str, Any]],
            list[bool],
            list[dict[str, Any] | str] | None,
        ]
    ],
    list[dict[str, Any]],
]:
    """Expand one user turn into inference-shaped next-decision snapshots.

    Earlier gold actions and their tool results are input context only.  Exactly
    one assistant action is supervised in every snapshot; a multi-call step is
    kept intact as one unordered parallel action.  The terminal stop is a
    separate target, so a model is never trained to stop immediately after an
    intermediate tool result in the same example.
    """
    prefix = list(current_prefix)
    examples: list[
        tuple[
            list[dict[str, Any]],
            list[bool],
            list[dict[str, Any] | str] | None,
        ]
    ] = []
    for step in steps:
        assistant_messages, tool_messages = _step_to_messages(
            step, id_prefix=id_prefix
        )
        if not assistant_messages:
            continue
        assistant_message = assistant_messages[0]
        if supervise_turn:
            messages = [*history, *prefix, assistant_message]
            examples.append(
                (messages, [False] * (len(messages) - 1) + [True], tool_override)
            )
        prefix.append(assistant_message)
        prefix.extend(tool_messages)

    terminal = {"role": "assistant", "content": ""}
    if supervise_turn:
        messages = [*history, *prefix, terminal]
        examples.append(
            (messages, [False] * (len(messages) - 1) + [True], tool_override)
        )
    prefix.append(terminal)
    return examples, prefix


def _build_trajectory_examples(
    trajectory: dict[str, Any],
) -> list[
    tuple[
        list[dict[str, Any]],
        list[bool],
        list[dict[str, Any] | str] | None,
    ]
]:
    query = trajectory.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("trajectory.query must be a non-empty string")
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("trajectory.steps must be a list")
    examples, _ = _decision_snapshots(
        history=[],
        current_prefix=[{"role": "user", "content": query}],
        id_prefix="call",
        steps=steps,
        supervise_turn=True,
        tool_override=None,
    )
    return examples


def _build_conversation_examples(
    conversation: dict[str, Any],
) -> list[
    tuple[
        list[dict[str, Any]],
        list[bool],
        list[dict[str, Any] | str] | None,
    ]
]:
    """Make one golden-history training example per decision point.

    The stock Qwen3.5 template renders assistant messages after the latest real
    user with the disabled-thinking scaffold, while older assistant history is
    serialized without it.  Rendering a complete multi-turn episode once would
    therefore train every turn except the last under a prefix that can never
    occur at inference time.  Prefix snapshots preserve exact train/eval parity
    and supervise each target action exactly once.
    """
    turns = conversation.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise ValueError("conversation.turns must be a non-empty list")

    history: list[dict[str, Any]] = []
    examples: list[
        tuple[
            list[dict[str, Any]],
            list[bool],
            list[dict[str, Any] | str] | None,
        ]
    ] = []

    for turn in turns:
        user_query = turn.get("user_query")
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("turn.user_query must be a non-empty string")
        current_prefix: list[dict[str, Any]] = [
            {"role": "user", "content": user_query}
        ]

        turn_number = turn.get("turn_number", 0)
        steps = turn.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("turn.steps must be a list")
        no_tool_target = turn.get("no_tool_target") is True
        if no_tool_target and steps:
            raise ValueError("no_tool_target turn must not contain executable steps")
        turn_tools = turn.get("available_tools")
        if turn_tools is not None and not isinstance(turn_tools, list):
            raise ValueError("turn.available_tools must be a list when present")
        supervise_turn = turn.get("sft_supervision") is not False
        turn_examples, completed_turn = _decision_snapshots(
            history=history,
            current_prefix=current_prefix,
            id_prefix=f"call_t{turn_number}",
            steps=steps,
            supervise_turn=supervise_turn,
            tool_override=turn_tools,
        )
        examples.extend(turn_examples)
        if (
            supervise_turn
            and no_tool_target
            and turn.get("no_tool_reason") == "no_appropriate_function"
            and NO_TOOL_CONTEXT_FREE_VIEW
            and history
        ):
            # Also teach the same abstention from the current request alone.
            # Recoverable missing-argument/function turns must retain history:
            # deleting it changes their meaning and encourages blanket stops.
            # Standalone irrelevance remains safe to duplicate.
            # A no-tool turn has exactly one decision snapshot.  Recreate the
            # same target without older episode history for genuine standalone
            # irrelevance only.
            examples.append(
                (
                    list(completed_turn),
                    [False] * (len(completed_turn) - 1) + [True],
                    turn_tools,
                )
            )
        history.extend(completed_turn)

    return examples


def build_training_examples(
    record: dict[str, Any], system_prompt: str
) -> list[tuple[list[dict[str, Any]], list[bool]]]:
    """Backward-compatible native examples without resolved tool snapshots."""
    return [
        (messages, mask)
        for messages, mask, _tool_override in _build_training_examples_raw(
            record, system_prompt
        )
    ]


def _build_training_examples_raw(
    record: dict[str, Any], system_prompt: str
) -> list[
    tuple[
        list[dict[str, Any]],
        list[bool],
        list[dict[str, Any] | str] | None,
    ]
]:
    """Native examples plus an optional current-turn tool-list override."""
    if isinstance(record.get("conversation"), dict):
        examples = _build_conversation_examples(record["conversation"])
    elif isinstance(record.get("trajectory"), dict):
        examples = _build_trajectory_examples(record["trajectory"])
    else:
        raise ValueError("record has neither 'conversation' nor 'trajectory'")
    if not system_prompt.strip():
        raise ValueError("tool-only system prompt must not be empty")
    result = []
    for messages, mask, tool_override in examples:
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})
        mask.insert(0, False)
        result.append((messages, mask, tool_override))
    return result


def build_training_examples_with_tools(
    record: dict[str, Any],
    system_prompt: str,
    catalog: dict[str, dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], list[bool], list[dict[str, Any]]]]:
    """Resolve the exact tool snapshot visible for every inference prefix.

    Legacy records still use record-level ``available_tools``.  Recovery rows
    may override that list on a turn, matching BFCL's missing-function protocol:
    the target function is absent for one user turn and present after the tool
    list update.  ``build_tools`` remains the single schema canonicalizer.
    """
    result = []
    for messages, mask, tool_override in _build_training_examples_raw(
        record, system_prompt
    ):
        tool_record = record
        if tool_override is not None:
            tool_record = dict(record)
            tool_record["available_tools"] = tool_override
        result.append((messages, mask, build_tools(tool_record, catalog)))
    return result


def classify_training_behavior(
    messages: list[dict[str, Any]], supervise_message_mask: list[bool]
) -> str:
    """Classify the current supervised target, ignoring golden history."""
    target_widths = [
        len(message.get("tool_calls") or [])
        for message, supervised in zip(messages, supervise_message_mask)
        if supervised
        and message.get("role") == "assistant"
        and message.get("tool_calls")
    ]
    total_calls = sum(target_widths)
    if total_calls == 0:
        return "no_tool"
    if any(width > 1 for width in target_widths):
        return "parallel"
    if total_calls == 1:
        return "single_call"
    return "other"


# --------------------------------------------------------------------------- #
# Rendering + robust token-level loss masking (assistant-marker scan)
# --------------------------------------------------------------------------- #

ASSISTANT_MARKER = "<|im_start|>assistant"
IM_END_MARKER = "<|im_end|>"


def render(tokenizer, messages, tools, *, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def char_span_to_token_span(offsets, start_char: int, end_char: int) -> tuple[int, int]:
    """Token indices overlapping [start_char, end_char) — BPE-safe."""
    start_tok = None
    end_tok = None
    for i, (a, b) in enumerate(offsets):
        if b <= start_char:
            continue
        if a >= end_char:
            break
        if start_tok is None:
            start_tok = i
        end_tok = i + 1
    if start_tok is None:
        return 0, 0
    return start_tok, end_tok or start_tok


def tokenize_with_mask(
    tokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    supervise_message_mask: list[bool],
    *,
    max_length: int,
    trace_id: str = "",
) -> dict[str, Any] | None:
    """Mask every assistant turn by scanning the single rendered transcript.

    Unlike prefix re-rendering, this is stable under the Qwen3.5 template's
    global state (multi-turn safe). Returns None if nothing is supervised or the
    row exceeds ``max_length`` (dropped, never truncated).
    """
    full_text = render(tokenizer, messages, tools, add_generation_prompt=False)
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    completion_mask = [0] * len(input_ids)

    if len(messages) != len(supervise_message_mask):
        raise ValueError("message/supervision mask length mismatch")
    assistant_targets = [
        bool(supervise_message_mask[index])
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    for index, (message, supervised) in enumerate(
        zip(messages, supervise_message_mask)
    ):
        if (
            supervised
            and not message.get("tool_calls")
            and str(message.get("content") or "").strip()
        ):
            raise ValueError(
                f"message {index} supervises forbidden assistant prose"
            )
    n_assistant = len(assistant_targets)
    n_segments = 0
    search = 0
    while True:
        a = full_text.find(ASSISTANT_MARKER, search)
        if a < 0:
            break
        body_start = a + len(ASSISTANT_MARKER)
        # The template emits "<|im_start|>assistant\n" — skip the single newline.
        if body_start < len(full_text) and full_text[body_start] == "\n":
            body_start += 1
        b = full_text.find(IM_END_MARKER, body_start)
        if b < 0:
            end_char = len(full_text)
        else:
            end_char = b + len(IM_END_MARKER)  # supervise through <|im_end|> (teach stopping)
        supervise_start = body_start
        # With enable_thinking=False, Qwen3.5 pre-fills an empty think block at
        # inference time.  Keep that prefix in the input context but never put
        # loss on it; the model should learn the tool call, not learn to emit
        # the disabled-thinking scaffold itself.
        if full_text.startswith("<think>", supervise_start):
            think_end = full_text.find("</think>", supervise_start)
            if 0 <= think_end < end_char:
                supervise_start = think_end + len("</think>")
                while (
                    supervise_start < end_char
                    and full_text[supervise_start] in "\r\n"
                ):
                    supervise_start += 1
        if n_segments < len(assistant_targets) and assistant_targets[n_segments]:
            s, e = char_span_to_token_span(
                offsets, supervise_start, end_char
            )
            if e <= s:
                raise ValueError(
                    f"assistant target {n_segments} maps to zero tokens"
                )
            for pos in range(s, e):
                completion_mask[pos] = 1
        n_segments += 1
        search = end_char if end_char > a else a + 1

    n_supervised = sum(completion_mask)
    if n_supervised == 0:
        return None
    if len(input_ids) > max_length:
        return None
    if n_segments != n_assistant:
        print(
            f"[{trace_id}] WARNING: assistant segments {n_segments} != assistant "
            f"messages {n_assistant} — check template output."
        )

    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "attention_mask": [1] * len(input_ids),
        "n_supervised": n_supervised,
        "n_tokens": len(input_ids),
    }


# --------------------------------------------------------------------------- #
# Preflight visualization / verification (does NOT train)
# --------------------------------------------------------------------------- #

def run_preflight(
    tokenizer,
    records,
    catalog,
    *,
    n: int,
    max_length: int,
    system_prompt: str,
) -> None:
    """Pick the hardest prefix examples and verify the exact loss mask."""
    candidates = []
    for record_index, rec in enumerate(records):
        examples = build_training_examples_with_tools(rec, system_prompt, catalog)
        for turn_index, (messages, lm, tools) in enumerate(examples):
            supervised_calls = sum(
                len(message.get("tool_calls", []))
                for message, target in zip(messages, lm)
                if target
            )
            parallel_targets = sum(
                len(message.get("tool_calls", [])) > 1
                for message, target in zip(messages, lm)
                if target
            )
            score = len(messages) + 5 * supervised_calls + 20 * parallel_targets
            candidates.append(
                (score, record_index, turn_index, rec, tools, messages, lm)
            )
    sample = sorted(candidates, key=lambda item: item[0], reverse=True)[:n]

    ok = True
    for i, (_, record_index, turn_index, rec, tools, messages, lm) in enumerate(sample):
        feats = tokenize_with_mask(
            tokenizer,
            messages,
            tools,
            lm,
            max_length=max_length,
            trace_id=f"pre{i}",
        )
        full_text = render(tokenizer, messages, tools, add_generation_prompt=False)
        if feats is None:
            print(f"\n===== preflight {i} ===== DROPPED (no supervised tokens or >max_length={max_length})")
            ok = False
            continue
        cm = feats["completion_mask"]
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]

        expected_tc = sum(
            len(m.get("tool_calls", []))
            for j, m in enumerate(messages)
            if j < len(lm) and lm[j] and m.get("role") == "assistant" and m.get("tool_calls")
        )
        sup_tc = 0
        idx = 0
        while True:
            pos = full_text.find("<tool_call>", idx)
            if pos < 0:
                break
            s, e = char_span_to_token_span(offsets, pos, pos + len("<tool_call>"))
            if any(cm[p] for p in range(s, e)):
                sup_tc += 1
            idx = pos + 1

        terminal_targets = sum(
            bool(lm[j])
            and m.get("role") == "assistant"
            and not m.get("tool_calls")
            and not str(m.get("content") or "").strip()
            for j, m in enumerate(messages)
        )

        schema = "conversation" if "conversation" in rec else "trajectory"
        print(
            f"\n===== preflight {i} ({schema} record={record_index} "
            f"turn_prefix={turn_index}) ====="
        )
        print(f"tools={len(tools)} msgs={len(messages)} tokens={feats['n_tokens']} "
              f"supervised={feats['n_supervised']} ({100*feats['n_supervised']/feats['n_tokens']:.1f}%)")
        print(f"expected assistant <tool_call> blocks={expected_tc}  supervised={sup_tc}")
        print(f"supervised empty terminal responses={terminal_targets}")
        supervised_actions = sum(
            bool(lm[j])
            and message.get("role") == "assistant"
            for j, message in enumerate(messages)
        )
        if supervised_actions != 1:
            ok = False
            print(
                f"  !! decision snapshot has {supervised_actions} "
                "supervised assistant actions"
            )
        elif expected_tc != sup_tc:
            ok = False
            print("  !! tool-call blocks mis-masked")
        elif (expected_tc == 0) != (terminal_targets == 1):
            ok = False
            print("  !! target is neither one tool action nor one empty stop")
        else:
            print(
                "  OK: exactly one next action supervised; golden history "
                "is input-only and no assistant prose is rendered."
            )

    print("\n==== PREFLIGHT " + ("PASSED" if ok else "FAILED") + " ====")
    if not ok:
        raise SystemExit(1)


def audit_complete_dataset(
    tokenizer,
    records,
    catalog,
    *,
    max_length: int,
    system_prompt: str,
) -> None:
    """Strict streaming render/length audit without materializing a dataset."""
    count = 0
    token_sum = 0
    max_tokens = 0
    # Exact quantiles are unnecessary for this admission gate and retaining a
    # Python integer for every expanded snapshot caused avoidable memory use.
    histogram_width = 256
    histogram: Counter[int] = Counter()
    action_kinds: Counter[str] = Counter()
    for record_index, rec in enumerate(records):
        examples = build_training_examples_with_tools(rec, system_prompt, catalog)
        for turn_index, (messages, lm, tools) in enumerate(examples):
            targets = [
                message
                for message, supervised in zip(messages, lm)
                if supervised and message.get("role") == "assistant"
            ]
            if len(targets) != 1:
                raise ValueError(
                    f"line {record_index} decision {turn_index} has "
                    f"{len(targets)} supervised assistant actions"
                )
            target = targets[0]
            if str(target.get("content") or "").strip():
                raise ValueError("next-action target contains assistant prose")
            width = len(target.get("tool_calls") or [])
            action_kinds["stop" if width == 0 else f"calls:{width}"] += 1

            # Offset mappings are required while constructing the actual loss
            # mask, but retaining them during an admission-only scan wastes
            # memory. Hard preflight cases already verify exact marker masks.
            rendered = render(
                tokenizer, messages, tools, add_generation_prompt=False
            )
            input_ids = tokenizer(
                rendered,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            n_tokens = len(input_ids)
            if n_tokens > max_length:
                raise ValueError(
                    f"line {record_index} decision {turn_index} has "
                    f"{n_tokens} tokens > max_length={max_length}"
                )
            count += 1
            token_sum += n_tokens
            max_tokens = max(max_tokens, n_tokens)
            histogram[n_tokens // histogram_width] += 1
    if not count:
        raise ValueError("strict audit found no training examples")

    threshold = max(1, int(0.95 * count))
    cumulative = 0
    p95 = 0
    for bucket, bucket_count in sorted(histogram.items()):
        cumulative += bucket_count
        if cumulative >= threshold:
            p95 = min(max_tokens, (bucket + 1) * histogram_width - 1)
            break
    print(
        f"prefix_examples={count} prefix_tokens={token_sum} "
        f"mean={token_sum/count:.1f} p95_approx_le={p95} "
        f"max={max_tokens} action_kinds={dict(sorted(action_kinds.items()))}"
    )
    print("==== FULL DATASET AUDIT PASSED ====")


def audit_complete_dataset_in_subprocess(
    *,
    model: Path,
    data: Path,
    tools_catalog: Path,
    system_prompt_file: Path,
    chat_template_file: Path,
    model_tag: str,
    max_length: int,
    repeat_factors: dict[str, int],
) -> None:
    """Run the expensive admission audit in an isolated process.

    Qwen tokenizers retain substantial temporary allocations while rendering
    thousands of prefixes.  Exiting the child guarantees those allocations are
    returned before the parent loads model weights and builds the train set.
    """
    import subprocess

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        str(model),
        "--model_tag",
        model_tag,
        "--data",
        str(data),
        "--tools_catalog",
        str(tools_catalog),
        "--output_dir",
        "/tmp/q35_toolonly_audit_unused",
        "--system_prompt_file",
        str(system_prompt_file),
        "--chat_template_file",
        str(chat_template_file),
        "--max_length",
        str(max_length),
        "--no_tool_repeat",
        str(repeat_factors["no_tool"]),
        "--single_call_repeat",
        str(repeat_factors["single_call"]),
        "--parallel_repeat",
        str(repeat_factors["parallel"]),
        "--expected_supervision_contract",
        SUPERVISION_CONTRACT,
        "--expected_prefix_unit",
        PREFIX_UNIT,
        "--audit_dataset",
        "--audit_in_process",
    ]
    subprocess.run(command, check=True)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #

def load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {"input": 0, "dropped_ineligible": 0, "conversation": 0, "trajectory": 0}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        stats["input"] += 1
        rec = json.loads(line)
        agg = rec.get("aggregation_metadata", {})
        if "eligible_for_sft_rl" in agg and agg.get("eligible_for_sft_rl") is not True:
            stats["dropped_ineligible"] += 1
            continue
        records.append(rec)
        if isinstance(rec.get("conversation"), dict):
            stats["conversation"] += 1
        elif isinstance(rec.get("trajectory"), dict):
            stats["trajectory"] += 1
    return records, stats


def build_dataset(
    tokenizer,
    records,
    catalog,
    *,
    max_length: int,
    val_ratio: float,
    seed: int,
    system_prompt: str,
    require_all_rows: bool,
    repeat_factors: dict[str, int],
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    for behavior in BEHAVIOR_CLASSES:
        factor = repeat_factors.get(behavior)
        if not isinstance(factor, int) or factor < 1:
            raise ValueError(
                f"repeat factor for {behavior!r} must be an integer >= 1"
            )

    indexed = list(enumerate(records))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    n_val = max(1, int(round(len(indexed) * val_ratio)))
    val_idx = {i for i, _ in indexed[:n_val]}

    train_rows, val_rows = [], []
    behavior_counts: dict[str, Counter[str]] = {
        "train_before_repeat": Counter(),
        "train_after_repeat": Counter(),
        "validation": Counter(),
        "dropped": Counter(),
    }
    dropped = 0
    record_errors = 0
    total_examples = 0
    for j, rec in enumerate(records):
        try:
            examples = build_training_examples_with_tools(
                rec, system_prompt, catalog
            )
        except (ValueError, KeyError) as exc:
            print(f"  skipped line {j}: {exc}")
            dropped += 1
            record_errors += 1
            continue
        if not examples:
            message = f"line {j} emits no supervised next-action examples"
            if require_all_rows:
                raise ValueError(message)
            print(f"  skipped {message}")
            dropped += 1
            record_errors += 1
            continue
        for example_index, (messages, lm, tools) in enumerate(examples):
            total_examples += 1
            behavior = classify_training_behavior(messages, lm)
            try:
                feats = tokenize_with_mask(
                    tokenizer,
                    messages,
                    tools,
                    lm,
                    max_length=max_length,
                    trace_id=f"line{j}.turn{example_index}",
                )
            except (ValueError, KeyError) as exc:
                print(f"  skipped line {j} turn {example_index}: {exc}")
                dropped += 1
                behavior_counts["dropped"][behavior] += 1
                continue
            if feats is None:
                dropped += 1
                behavior_counts["dropped"][behavior] += 1
                continue
            row = {
                "input_ids": feats["input_ids"],
                "completion_mask": feats["completion_mask"],
            }
            if j in val_idx:
                val_rows.append(row)
                behavior_counts["validation"][behavior] += 1
            else:
                behavior_counts["train_before_repeat"][behavior] += 1
                factor = repeat_factors[behavior]
                train_rows.extend(dict(row) for _ in range(factor))
                behavior_counts["train_after_repeat"][behavior] += factor
    if dropped:
        print(
            f"  dropped {dropped} examples (>max_length={max_length}, "
            "no supervised tokens, or bad record)"
        )
        if require_all_rows:
            raise ValueError(
                f"strict corpus contract violated: {dropped}/{total_examples} "
                "prefix examples would be dropped"
            )
    if not train_rows:
        raise ValueError("No training examples after tokenization.")

    def ordered_counts(counter: Counter[str]) -> dict[str, int]:
        return {behavior: int(counter.get(behavior, 0)) for behavior in BEHAVIOR_CLASSES}

    behavior_stats: dict[str, Any] = {
        "record_split": {
            "train": len(records) - len(val_idx),
            "validation": len(val_idx),
        },
        "prefix_examples": {
            name: ordered_counts(counter)
            for name, counter in behavior_counts.items()
        },
        "record_errors": record_errors,
    }
    print("Behavior-balanced prefix examples (repeats are train-only):")
    for name, counts in behavior_stats["prefix_examples"].items():
        print(
            f"  {name}: total={sum(counts.values())} "
            + " ".join(f"{key}={value}" for key, value in counts.items())
        )
    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(val_rows or train_rows[:1]),
        behavior_stats,
    )


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #

class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, feats: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, attn, labels = [], [], []
        for f in feats:
            ids = f["input_ids"]
            cm = f["completion_mask"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append([tok if m else -100 for tok, m in zip(ids, cm)] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ToolCallingTrainer(Trainer):
    """Forces the Liger fused linear cross-entropy path in BOTH train and eval.

    The Liger ``qwen3_5`` patch computes ``skip_logits = self.training`` (see
    liger_kernel/.../qwen3_5.py), so the fused path (which never materializes
    the logits tensor) is only active while training. During evaluation it falls
    back to ``logits = lm_head(...)`` + ``cross_entropy`` in fp32 — and with the
    Qwen3.5 vocab of ~248k that materializes a 24-32 GiB tensor -> CUDA OOM at
    the first epoch-end eval. Passing ``skip_logits=True`` explicitly makes
    train and eval use the identical fused loss, so logits are never built.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs["skip_logits"] = True
        if getattr(self, "model_accepts_loss_kwargs", False) and num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if loss is None:
            raise RuntimeError(
                "Model returned no loss with skip_logits=True — fused CE path failed."
            )
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    data_dir = Path("/mnt/shared_ru.ml.SZ-5_000264/andriianov/training/data")
    ap.add_argument("--model", required=True, help="Path to base model (e.g. .../Qwen3.5-2B).")
    ap.add_argument(
        "--model_tag",
        required=True,
        help="Stable experiment identity recorded in the checkpoint contract.",
    )
    ap.add_argument("--data", required=True, help="canonical_sft_rl_corpus_*.jsonl (or legacy single/mixed_sft.jsonl)")
    ap.add_argument(
        "--expected_supervision_contract",
        default=SUPERVISION_CONTRACT,
    )
    ap.add_argument("--expected_prefix_unit", default=PREFIX_UNIT)
    ap.add_argument("--tools_catalog", type=Path, default=data_dir / "tools_openai_format.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--system_prompt_file",
        type=Path,
        required=True,
        help="Canonical tool-only prompt shared verbatim with BFCL evaluation.",
    )
    ap.add_argument(
        "--chat_template_file",
        type=Path,
        required=True,
        help="Canonical Jinja template shared verbatim with vLLM evaluation.",
    )
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_tool_repeat",
        type=int,
        default=DEFAULT_TRAIN_REPEAT_FACTORS["no_tool"],
        help="Train-only repeat factor for genuine no-call prefix targets.",
    )
    ap.add_argument(
        "--single_call_repeat",
        type=int,
        default=DEFAULT_TRAIN_REPEAT_FACTORS["single_call"],
        help="Train-only repeat factor for prefixes with exactly one target call.",
    )
    ap.add_argument(
        "--parallel_repeat",
        type=int,
        default=DEFAULT_TRAIN_REPEAT_FACTORS["parallel"],
        help="Train-only repeat factor for prefixes containing a parallel target.",
    )
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_strategy", default="epoch")
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--use_liger_kernel", action="store_true", default=True)
    ap.add_argument("--no_liger", dest="use_liger_kernel", action="store_false")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--preflight", type=int, default=0,
                    help="If >0: verify N examples' masks then exit (no training).")
    ap.add_argument(
        "--audit_dataset",
        action="store_true",
        help="Tokenize and strictly audit the complete dataset, then exit.",
    )
    ap.add_argument(
        "--audit_in_process",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--require_all_rows",
        action="store_true",
        help="Fail instead of silently dropping any invalid or over-length row.",
    )
    args = ap.parse_args()
    if args.expected_supervision_contract != SUPERVISION_CONTRACT:
        raise ValueError(
            "launcher/trainer supervision mismatch: "
            f"{args.expected_supervision_contract!r} != {SUPERVISION_CONTRACT!r}"
        )
    if args.expected_prefix_unit != PREFIX_UNIT:
        raise ValueError(
            "launcher/trainer prefix-unit mismatch: "
            f"{args.expected_prefix_unit!r} != {PREFIX_UNIT!r}"
        )
    base_model_path = Path(args.model).resolve()
    base_config_path = base_model_path / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(
            f"local base model has no config.json: {base_config_path}"
        )
    base_model_config_sha256 = hashlib.sha256(
        base_config_path.read_bytes()
    ).hexdigest()
    repeat_factors = {
        "no_tool": args.no_tool_repeat,
        "single_call": args.single_call_repeat,
        "parallel": args.parallel_repeat,
        "other": DEFAULT_TRAIN_REPEAT_FACTORS["other"],
    }
    invalid_repeat_factors = {
        key: value
        for key, value in repeat_factors.items()
        if not isinstance(value, int) or value < 1
    }
    if invalid_repeat_factors:
        raise ValueError(
            "all repeat factors must be integers >= 1: "
            f"{invalid_repeat_factors}"
        )

    # Admission is mandatory for real training. Run it before loading the
    # parent tokenizer/model and in a child process to release tokenizer memory.
    if args.preflight <= 0 and not args.audit_dataset:
        audit_complete_dataset_in_subprocess(
            model=base_model_path,
            data=Path(args.data).resolve(),
            tools_catalog=args.tools_catalog.resolve(),
            system_prompt_file=args.system_prompt_file.resolve(),
            chat_template_file=args.chat_template_file.resolve(),
            model_tag=args.model_tag,
            max_length=args.max_length,
            repeat_factors=repeat_factors,
        )

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n_visible)]
        print(f"[gpu] CUDA_VISIBLE_DEVICES={cuda_vis}  visible_devices={n_visible}  {names}")
        if n_visible > 1:
            print(f"[gpu] WARNING: {n_visible} GPUs visible — pin one via CUDA_VISIBLE_DEVICES.")
    else:
        print(f"[gpu] CUDA_VISIBLE_DEVICES={cuda_vis}  CUDA not available (CPU-only).")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    chat_template = args.chat_template_file.read_text(encoding="utf-8")
    if not chat_template.strip():
        raise ValueError("--chat_template_file is empty")
    tokenizer.chat_template = chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.padding_side = "right"

    catalog = load_catalog(args.tools_catalog)
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError("--system_prompt_file is empty")
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    print(
        f"Tool-only system prompt: {args.system_prompt_file} "
        f"sha256={prompt_sha256}"
    )
    print(
        f"Canonical chat template: {args.chat_template_file} "
        f"sha256={template_sha256}"
    )
    records, stats = load_records(Path(args.data))
    print(f"Loaded {len(records)} records (input={stats['input']}, "
          f"dropped_ineligible={stats['dropped_ineligible']}) from {args.data}")
    print(f"  conversation={stats['conversation']}  trajectory={stats['trajectory']}")

    if args.preflight > 0:
        run_preflight(
            tokenizer,
            records,
            catalog,
            n=args.preflight,
            max_length=args.max_length,
            system_prompt=system_prompt,
        )
        return 0

    if args.audit_dataset:
        audit_complete_dataset(
            tokenizer,
            records,
            catalog,
            max_length=args.max_length,
            system_prompt=system_prompt,
        )
        return 0

    print("Tokenizing dataset (assistant-marker loss mask)...")
    print(f"Train-only repeat factors: {repeat_factors}")
    train_ds, val_ds, behavior_stats = build_dataset(
        tokenizer, records, catalog,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        system_prompt=system_prompt,
        require_all_rows=args.require_all_rows,
        repeat_factors=repeat_factors,
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=args.use_liger_kernel,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        # load_best_model_at_end disabled: with use_liger_kernel the saved
        # checkpoint stores keys under model.language_model.*, which the
        # Trainer's internal loader cannot map back. epoch checkpoints on disk
        # remain usable via from_pretrained(); pick the best by eval_loss later.
        load_best_model_at_end=False,
        dataloader_num_workers=args.num_workers,
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    trainer = ToolCallingTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    output_path = Path(args.output_dir)
    # Do not rely on transformers' version-specific template serialization:
    # persist the exact eval template explicitly beside the trained weights.
    (output_path / "chat_template.jinja").write_text(
        chat_template, encoding="utf-8"
    )
    data_sha256 = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    metadata_path = output_path / "toolonly_contract.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_tag": args.model_tag,
                "base_model": str(base_model_path),
                "base_model_config_sha256": base_model_config_sha256,
                "system_prompt": system_prompt,
                "system_prompt_sha256": prompt_sha256,
                "chat_template_sha256": template_sha256,
                "dataset_sha256": data_sha256,
                "source_records": len(records),
                "train_prefix_examples": len(train_ds),
                "validation_prefix_examples": len(val_ds),
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "enable_thinking": False,
                "supervision": SUPERVISION_CONTRACT,
                "prefix_unit": PREFIX_UNIT,
                "tool_schema_projection": TOOL_SCHEMA_PROJECTION,
                "removed_input_properties": {
                    name: sorted(properties)
                    for name, properties in REMOVED_INPUT_PROPERTIES.items()
                },
                "no_tool_context_free_view": NO_TOOL_CONTEXT_FREE_VIEW,
                "train_repeat_factors": repeat_factors,
                "training_behavior_stats": behavior_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Done. Text-only SFT checkpoint saved to {args.output_dir}")
    print("NOTE: run convert_to_vllm.py to produce a vLLM-servable (VL-format) checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
