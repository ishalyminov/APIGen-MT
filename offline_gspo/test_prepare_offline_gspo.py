from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

from offline_gspo import prepare_offline_gspo as prep


def _summary(successes: list[int]) -> dict:
    return {
        "task_results": [
            {
                "row_position": position,
                "num_samples": 32,
                "successful_rollouts": count,
                "rollout_success_rate": count / 32,
            }
            for position, count in enumerate(successes)
        ]
    }


def test_condition_selection_is_literal_one_through_twenty_five() -> None:
    summary = _summary([0, 1, 25, 26, 31, 32])
    assert prep.select_condition_positions(summary) == (1, 2)


def test_binary_advantages_are_sample_standardized_like_verl() -> None:
    rewards = [1, 0, 0, 1]
    advantages = prep.binary_group_advantages(rewards)
    expected = math.sqrt(3.0 / 4.0)
    assert advantages == pytest.approx([expected, -expected, -expected, expected])
    assert sum(advantages) / len(advantages) == pytest.approx(0.0)
    assert math.sqrt(sum(value * value for value in advantages) / 3) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="both rewards"):
        prep.binary_group_advantages([0, 0])


class _CharacterTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


def test_truncated_target_is_kept_unclosed_and_exact() -> None:
    tokenizer = _CharacterTokenizer()
    packed, deltas = prep._flush_segment(
        tokenizer=tokenizer,
        messages=[],
        tools=[],
        targets=[
            {
                "prompt_text": "prompt<think>\n",
                "assistant_text": "prompt<think>\nunfinished",
                "recorded_prompt_tokens": len("prompt<think>\n"),
                "recorded_completion_tokens": len("unfinished"),
                "finish_reason": "length",
            }
        ],
        max_length=100,
        full_text_override="prompt<think>\nunfinished",
    )
    assert deltas == [0]
    assert packed["target_positions"] == list(
        range(len("prompt<think>\n"), len("prompt<think>\nunfinished"))
    )
    assert "</think>" not in "".join(map(chr, packed["input_ids"]))


def test_template_normalizer_rejects_non_object_arguments() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "x", "arguments": "[]"},
            }
        ],
    }
    with pytest.raises(ValueError, match="not an object"):
        prep._normalise_template_messages([message])


@pytest.mark.skipif(
    not prep.DEFAULT_MODEL.exists() or not prep.DEFAULT_DATASET.exists(),
    reason="workspace integration assets are unavailable",
)
def test_real_checker_replay_reproduces_prompt_tokens() -> None:
    """One real native-call failure exercises checker state and Jinja parity."""

    legacy, checker_v8 = prep._import_checker_modules()
    tasks, _ = legacy.load_tasks(
        prep.DEFAULT_DATASET,
        prep.DEFAULT_TOOL_POOL,
        tool_scope="declared",
        max_samples=1,
    )
    for task in tasks:
        task.tools = checker_v8._canonical_policy_tools(task.tools)
        for turn in task.user_turns:
            turn["assistant_response"] = ""
    checker_v8.prepare_next_action_tasks(tasks, trust_projected_parallel=True)
    checker = checker_v8.InteractivePassKV3Checker(
        None,
        None,
        pass_k=1,
        include_initial_state=False,
        current_date="2026-08-17",
    )
    base_state = checker.build_states(tasks)[0]

    rollout_path = (
        prep.DEFAULT_PASS32_ROOT
        / "full/qwen35_2b/t0p7/rollouts.jsonl"
    )
    with rollout_path.open(encoding="utf-8") as handle:
        rollout = json.loads(next(handle))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        prep.DEFAULT_MODEL,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.chat_template = prep.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    rows, audit = prep.reconstruct_episode(
        legacy=legacy,
        checker_v8=checker_v8,
        checker=checker,
        tokenizer=tokenizer,
        task=base_state.task,
        initial_messages=copy.deepcopy(base_state.messages),
        rollout=rollout,
        condition_tag="t0p7",
        temperature=0.7,
        advantage=-1.0,
        max_length=16384,
    )
    assert rows
    assert [row["segment_index"] for row in rows] == list(range(len(rows)))
    assert all(row["episode_id"] == audit["episode_id"] for row in rows)
    assert sum(len(row["target_positions"]) for row in rows) == audit["target_tokens"]
    assert all(delta <= 0 for delta in audit["completion_token_deltas"])
