#!/usr/bin/env python3
"""
Generate datapoints using step-by-step blueprint generation.

Supports two modes:
  - multi-turn (default): Each user turn is a separate exchange with its own query.
    Use --num-turns to control total turns and --num-actions for tools per turn.
  - step-by-step (legacy): Single user query with multiple action steps.

Checkpoint/Resume:
  Use --checkpoint to enable checkpointing. Progress is saved after each turn.
  If interrupted, running with the same --checkpoint file will resume from where
  you left off.

Usage:
    python generate_step_by_step.py [OPTIONS]

Options:
    --mode MODE             Generation mode: multi-turn (default) or step-by-step
    --num-datapoints N      Number of datapoints to generate (default: 100)
    --num-turns N            Number of user-assistant turns for multi-turn (default: 10)
    --num-actions N         Actions per turn (default: 1)
    --output FILE           Output file path (default: step_by_step_datapoints.jsonl)
    --category CATEGORY     Filter tools to a specific category
    --model MODEL           Model name (default: minimaxai/minimax-m2.7)
    --checkpoint FILE       Checkpoint file for resume support (default: none)
"""

import json
import os
import sys
import random
import argparse
import copy
import fcntl
import hashlib
import signal
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager
from apigen_step_by_step import (
    GenerationBudgetExceeded,
    StepByStepGenerator,
    StepByStepDatapoint,
    TokenUsageStats,
)
from apigen_multi_turn import MultiTurnGenerator, MultiTurnDatapoint


QWEN_FINAL_STAGE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def _unique_clients(*clients):
    """Return clients once, preserving role order."""
    result = []
    seen = set()
    for client in clients:
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        result.append(client)
    return result


def _make_role_client(*, model: str, api_base: str, api_key: str):
    if not model or not api_base or not api_key:
        raise ValueError("model, api_base and api_key are required")
    client = LocalOpenAILLMClient(
        url=api_base.rstrip("/"),
        api_key=api_key,
        api_model=model,
        hf_tokenizer_id=None,
    )
    # The in-cluster LiteLLM/vLLM proxy is OpenAI-compatible but does not need
    # OpenRouter-only ``provider``/``reasoning`` request fields.
    client.apigen_openrouter_extensions = "openrouter.ai" in api_base.casefold()
    return client


def build_final_stage_clients(
    args,
    *,
    llm_client,
    judge_client,
    main_api_base: str,
    main_api_key: str,
):
    """Resolve optional final-writer and grounding model routes.

    Defaults preserve prior behavior exactly.  ``--use-qwen-final-stages``
    routes both roles to the in-cluster Qwen proxy using LLM_PROXY_URL and
    LLM_PROXY_MASTER_KEY.  Fine-grained role flags may override either role.
    """
    use_qwen = bool(getattr(args, "use_qwen_final_stages", False))

    final_model = getattr(args, "final_response_model", None)
    final_base = getattr(args, "final_response_api_base", None)
    final_key = getattr(args, "final_response_api_key", None)
    grounding_model = getattr(args, "grounding_model", None)
    grounding_base = getattr(args, "grounding_api_base", None)
    grounding_key = getattr(args, "grounding_api_key", None)

    if use_qwen:
        final_model = final_model or QWEN_FINAL_STAGE_MODEL
        final_base = final_base or os.getenv("LLM_PROXY_URL")
        final_key = final_key or os.getenv("LLM_PROXY_MASTER_KEY")
        grounding_model = grounding_model or final_model
        grounding_base = grounding_base or final_base
        grounding_key = grounding_key or final_key
        if not final_base or not final_key:
            raise RuntimeError(
                "--use-qwen-final-stages requires LLM_PROXY_URL and "
                "LLM_PROXY_MASTER_KEY (or explicit role API arguments)"
            )

    final_response_client = llm_client
    if final_model:
        if not final_base:
            final_base = (
                os.getenv("LLM_PROXY_URL")
                if final_model == QWEN_FINAL_STAGE_MODEL
                else main_api_base
            )
        if not final_key:
            final_key = (
                os.getenv("LLM_PROXY_MASTER_KEY")
                if final_model == QWEN_FINAL_STAGE_MODEL
                else main_api_key
            )
        if not final_base or not final_key:
            raise RuntimeError("missing final-response API base/key")
        if (
            final_model == getattr(llm_client, "api_model", None)
            and final_base.rstrip("/")
            == str(getattr(llm_client, "url", "")).rstrip("/")
        ):
            final_response_client = llm_client
        elif (
            final_model == getattr(judge_client, "api_model", None)
            and final_base.rstrip("/")
            == str(getattr(judge_client, "url", "")).rstrip("/")
        ):
            final_response_client = judge_client
        else:
            final_response_client = _make_role_client(
                model=final_model, api_base=final_base, api_key=final_key
            )

    grounding_client = judge_client
    if grounding_model:
        if not grounding_base:
            grounding_base = (
                os.getenv("LLM_PROXY_URL")
                if grounding_model == QWEN_FINAL_STAGE_MODEL
                else main_api_base
            )
        if not grounding_key:
            grounding_key = (
                os.getenv("LLM_PROXY_MASTER_KEY")
                if grounding_model == QWEN_FINAL_STAGE_MODEL
                else main_api_key
            )
        if not grounding_base or not grounding_key:
            raise RuntimeError("missing grounding API base/key")
        if (
            grounding_model == getattr(final_response_client, "api_model", None)
            and grounding_base.rstrip("/")
            == str(getattr(final_response_client, "url", "")).rstrip("/")
        ):
            grounding_client = final_response_client
        elif (
            grounding_model == getattr(llm_client, "api_model", None)
            and grounding_base.rstrip("/")
            == str(getattr(llm_client, "url", "")).rstrip("/")
        ):
            grounding_client = llm_client
        elif (
            grounding_model == getattr(judge_client, "api_model", None)
            and grounding_base.rstrip("/")
            == str(getattr(judge_client, "url", "")).rstrip("/")
        ):
            grounding_client = judge_client
        else:
            grounding_client = _make_role_client(
                model=grounding_model,
                api_base=grounding_base,
                api_key=grounding_key,
            )

    return final_response_client, grounding_client


def parse_actions_per_turn(value: str) -> list[int]:
    """Parse a comma-separated exact blueprint action schedule."""

    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers, for example 1,2,3,1"
        ) from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError(
            "every per-turn action count must be a positive integer"
        )
    return result


class CheckpointManager:
    """Manages checkpoint saving and loading for resumable generation."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.state: dict = {}

    def load(self) -> dict:
        """Load checkpoint state if exists."""
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    self.state = json.load(f)
                return self.state
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save(self, state: dict) -> None:
        """Save checkpoint state to disk."""
        if not self.checkpoint_path:
            return
        self.state = state
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(state, f, default=str)
        except IOError as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def clear(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            os.remove(self.checkpoint_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate datapoints using step-by-step blueprint generation.'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='multi-turn',
        choices=['multi-turn', 'step-by-step'],
        help='Generation mode: multi-turn (default) or step-by-step (legacy)'
    )

    parser.add_argument(
        '--num-datapoints', '-n',
        type=int,
        default=100,
        help='Number of datapoints to generate (default: 100)'
    )

    parser.add_argument(
        '--num-turns', '-t',
        type=int,
        default=10,
        help='Number of user-assistant turns for multi-turn mode (default: 10)'
    )

    parser.add_argument(
        '--num-actions', '-a',
        type=int,
        default=1,
        help='Number of actions per turn (default: 1)'
    )

    parser.add_argument(
        '--blueprint-max-actions-per-turn',
        type=int,
        default=None,
        help=(
            'Maximum ordinary blueprint actions per multi-turn turn. This is '
            'separate from --num-actions, which also controls parallel width.'
        ),
    )

    parser.add_argument(
        '--blueprint-actions-per-turn',
        type=parse_actions_per_turn,
        default=None,
        metavar='N1,N2,...',
        help=(
            'Require an exact ordinary blueprint tool-count vector. Its length '
            'must equal --num-turns and every count must be within the '
            '--blueprint-max-actions-per-turn limit.'
        ),
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='step_by_step_datapoints.jsonl',
        help='Output file path (default: step_by_step_datapoints.jsonl)'
    )

    parser.add_argument(
        '--tool-pool',
        type=str,
        default='~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl',
        help='Path to tool pool file'
    )

    parser.add_argument(
        '--invocation-examples',
        type=str,
        default='~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl',
        help='Path to invocation examples file (for Python tool implementations)'
    )

    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter tools to a specific category'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='minimax/minimax-m2.7',
        help='Model to use for generation (default: minimaxai/minimax-m2.7)'
    )

    parser.add_argument(
        '--judge-model',
        type=str,
        default=None,
        help='Model to use for judge tasks (state verification, sequence validation). Defaults to --model if not set.'
    )

    parser.add_argument(
        '--judge-api-base',
        type=str,
        default=None,
        help='API base URL for judge model. Defaults to OPENAI_API_BASE if not set.'
    )

    parser.add_argument(
        '--judge-api-key',
        type=str,
        default=None,
        help='API key for judge model. Defaults to OPENAI_API_KEY if not set.'
    )

    parser.add_argument(
        '--use-qwen-final-stages',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Route final-response writing and grounding to the in-cluster '
            'Qwen/Qwen3.6-35B-A3B-FP8 proxy. Uses LLM_PROXY_URL and '
            'LLM_PROXY_MASTER_KEY. Default: disabled.'
        ),
    )
    parser.add_argument(
        '--final-response-model', type=str, default=None,
        help='Optional model override for final-response generation.'
    )
    parser.add_argument(
        '--final-response-api-base', type=str, default=None,
        help='Optional OpenAI-compatible API base for final-response generation.'
    )
    parser.add_argument(
        '--final-response-api-key', type=str, default=None,
        help='Optional API key for final-response generation.'
    )
    parser.add_argument(
        '--grounding-model', type=str, default=None,
        help='Optional model override for final-response grounding.'
    )
    parser.add_argument(
        '--grounding-api-base', type=str, default=None,
        help='Optional OpenAI-compatible API base for grounding.'
    )
    parser.add_argument(
        '--grounding-api-key', type=str, default=None,
        help='Optional API key for grounding.'
    )

    parser.add_argument(
        '--optimized-pipeline',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Compile one whole turn per LLM call and use deterministic '
            'per-action/state checks (default: enabled).'
        ),
    )

    parser.add_argument(
        '--max-calls-per-candidate',
        type=int,
        default=30,
        help='Hard LLM request budget for one candidate (default: 30).',
    )

    parser.add_argument(
        '--max-calls-per-accepted-row',
        type=int,
        default=30,
        help=(
            'Cumulative request budget across rejected candidates before one '
            'accepted row (default: 30).'
        ),
    )

    parser.add_argument(
        '--max-tokens-per-accepted-row',
        type=int,
        default=100_000,
        help=(
            'Cumulative token budget across rejected candidates before one '
            'accepted row (default: 100000).'
        ),
    )

    parser.add_argument(
        '--max-candidate-starts-per-row',
        type=int,
        default=3,
        help=(
            'Stop instead of resampling forever after this many candidate '
            'starts for one output row (default: 3).'
        ),
    )

    parser.add_argument(
        '--max-turn-attempts',
        type=int,
        choices=(1, 2),
        default=2,
        help='One turn compile plus at most one complete-turn repair.',
    )

    parser.add_argument(
        '--usage-report',
        type=str,
        default=None,
        metavar='JSON',
        help=(
            'Write provider usage even when no trajectory is accepted, so '
            'failed subprocess spend remains accounted for.'
        ),
    )

    parser.add_argument(
        '--candidate-archive-dir',
        type=str,
        default=None,
        metavar='DIR',
        help=(
            'Optional append-only archive for every completed/partial candidate. '
            'Accepted and rejected trajectories are written to separate '
            'subdirectories. This does not change prompts, retries, or gates.'
        ),
    )

    parser.add_argument(
        '--curriculum-mode',
        choices=('off', 'bfcl-v3'),
        default='off',
        help='Enable persistent all-tool BFCL-v3 coverage/diversity scheduling.',
    )
    parser.add_argument('--coverage-state', type=str, default=None)
    parser.add_argument('--curriculum-seed', type=int, default=20260730)
    parser.add_argument('--coverage-target-per-tool', type=int, default=1)
    parser.add_argument(
        '--continue-until-full-tool-coverage',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--max-extra-coverage-rows', type=int, default=500)
    parser.add_argument('--all-tools-rate', type=float, default=0.25)
    parser.add_argument('--cross-domain-rate', type=float, default=0.45)
    parser.add_argument('--hard-distractor-count', type=int, default=48)
    parser.add_argument('--target-tools-per-candidate', type=int, default=2)
    parser.add_argument('--evolution-lessons', type=str, default=None)
    parser.add_argument(
        '--required-tool',
        action='append',
        default=[],
        metavar='TOOL',
        help=(
            'Require this real tool to appear in every accepted multi-turn '
            'trajectory. May be repeated. Unlike curriculum targets, these '
            'requirements are enforced both on the blueprint and actual calls.'
        ),
    )

    parser.add_argument(
        '--num-actions-range',
        type=int,
        nargs=2,
        default=None,
        metavar=('MIN', 'MAX'),
        help='Randomize num_actions per datapoint between MIN and MAX (inclusive). Overrides -a.'
    )

    parser.add_argument(
        '--config-pool',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use diverse config pool for initial API states (default: True). Use --no-config-pool to disable.'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for resume support. If provided, progress is saved after each turn.'
    )

    parser.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Resume from checkpoint if exists (default: True). Use --no-resume to start fresh.'
    )

    parser.add_argument(
        '--allow-refusal',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Opt in to certified terminal refusal examples. Disabled by default, '
            'so the existing generator behavior is unchanged.'
        ),
    )

    parser.add_argument(
        '--refusal-rate',
        type=float,
        default=0.12,
        help='Per-episode/eligible-final-turn refusal probability when enabled (default: 0.12).',
    )

    parser.add_argument(
        '--allow-parallel',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Opt in to certified independent parallel-call batches. Disabled by '
            'default, so the existing generator behavior is unchanged.'
        ),
    )

    parser.add_argument(
        '--parallel-rate', '--parallel-prob',
        dest='parallel_rate',
        type=float,
        default=0.25,
        help='Parallel-batch probability when enabled (default: 0.25).',
    )

    parser.add_argument(
        '--max-parallel-width',
        type=int,
        default=3,
        help='Maximum number of calls in one certified parallel batch (default: 3).',
    )

    parser.add_argument(
        '--require-feature',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'When refusal/parallel generation is enabled, reject and resample '
            'instead of saving a normal fallback episode. Disabled by default.'
        ),
    )

    parser.add_argument(
        '--feature-difficulty',
        choices=('standard', 'hard'),
        default='standard',
        help='Difficulty of refusal/parallel feature turns (default: standard).',
    )

    parser.add_argument(
        '--naturalize-queries',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Run a separate style-only LLM pass over the fixed multi-turn plan '
            'and generated feature queries, then re-certify them.'
        ),
    )

    parser.add_argument(
        '--multi-turn-feature-schedule',
        choices=('terminal', 'interactive-refusal', 'combined'),
        default='terminal',
        help=(
            'terminal: one final feature; interactive-refusal: intermediate '
            'clarification plus user recovery; combined: recovery plus a final '
            'parallel batch.'
        ),
    )

    parser.add_argument(
        '--interactive-refusal-turn',
        type=int,
        default=None,
        metavar='TURN',
        help=(
            'Force the 1-based clarification/refusal turn for interactive or '
            'combined multi-turn schedules. The following turn is recovery.'
        ),
    )

    parser.add_argument(
        '--refusal-reason',
        choices=('random', 'no_appropriate_function', 'missing_argument', 'ambiguity'),
        default='random',
        help='Force a refusal reason for balanced generation shards.',
    )

    parser.add_argument(
        '--dedupe-against',
        action='append',
        default=[],
        metavar='JSONL',
        help=(
            'Reject step-by-step trajectories whose ordered tool calls and '
            'arguments already occur in this JSONL. May be repeated.'
        ),
    )

    parser.add_argument(
        '--dedupe-registry',
        type=str,
        default=None,
        metavar='FILE',
        help=(
            'Shared append-only signature registry used to deduplicate across '
            'parallel step-by-step generator processes.'
        ),
    )

    parser.add_argument(
        '--min-total-steps',
        type=int,
        default=None,
        help='Reject and resample datapoints with fewer total assistant actions.',
    )
    parser.add_argument(
        '--max-total-steps',
        type=int,
        default=None,
        help='Reject and resample datapoints with more total assistant actions.',
    )

    return parser.parse_args()


def load_tool_categories(tool_pool_path: str) -> dict:
    """Load tools and group them by category."""
    tools_by_category = {}

    with open(tool_pool_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                tool = json.loads(line.strip())
                category = tool.get('category', 'Unknown')

                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool)
            except json.JSONDecodeError:
                continue

    return tools_by_category


def _trajectory_signature_from_dict(datapoint: dict) -> str:
    """Hash the semantic action plan, preserving opt-in parallel grouping.

    For legacy/all-sequential trajectories the payload is byte-for-byte the old
    flat call list. Therefore existing signatures and default behavior are not
    changed. A trajectory with a multi-call step includes group boundaries so a
    parallel batch does not collide with the same calls executed sequentially.
    """
    if isinstance(datapoint.get('conversation'), dict):
        turns = datapoint['conversation'].get('turns', [])
        payload_turns = []
        for turn in turns:
            groups = []
            for step in turn.get('steps', []):
                group = [
                    {
                        'tool_name': call.get('tool_name'),
                        'arguments': call.get('arguments', {}),
                    }
                    for call in step.get('tool_calls', [])
                ]
                if step.get('call_order_matters', True) is False:
                    group = sorted(
                        group,
                        key=lambda call: json.dumps(
                            call,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(',', ':'),
                            default=str,
                        ),
                    )
                groups.append(
                    {
                        'execution_mode': step.get(
                            'execution_mode', 'sequential'
                        ),
                        'calls': group,
                    }
                )
            payload_turns.append(
                {
                    'query': turn.get('user_query', ''),
                    'groups': groups,
                }
            )
        payload = json.dumps(
            payload_turns,
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ':'),
            default=str,
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    steps = datapoint.get('trajectory', {}).get('steps', [])
    has_parallel_step = any(
        len(step.get('tool_calls', [])) > 1 for step in steps
    )
    if has_parallel_step:
        calls = []
        for step in steps:
            group = [
                {
                    'tool_name': tool_call.get('tool_name'),
                    'arguments': tool_call.get('arguments', {}),
                }
                for tool_call in step.get('tool_calls', [])
            ]
            if len(group) > 1:
                # A parallel batch is a multiset, not a sequence.  Sorting only
                # inside parallel steps keeps dedupe order-invariant while the
                # legacy sequential hash below remains byte-for-byte unchanged.
                group = sorted(
                    group,
                    key=lambda call: json.dumps(
                        call,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(',', ':'),
                        default=str,
                    ),
                )
            calls.append(group)
    else:
        calls = []
        for step in steps:
            for tool_call in step.get('tool_calls', []):
                calls.append({
                    'tool_name': tool_call.get('tool_name'),
                    'arguments': tool_call.get('arguments', {}),
                })
    payload = json.dumps(
        calls,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        default=str,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _trajectory_signature(datapoint: StepByStepDatapoint) -> str:
    return _trajectory_signature_from_dict(datapoint.model_dump())


def _load_trajectory_signatures(paths) -> set[str]:
    signatures = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    signatures.add(
                        _trajectory_signature_from_dict(json.loads(line))
                    )
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(
                        f'Invalid dedupe row at {path}:{line_number}: {exc}'
                    ) from exc
    return signatures


def _datapoint_total_steps(datapoint) -> int:
    trajectory = getattr(datapoint, 'trajectory', None)
    if trajectory is not None:
        return len(trajectory.steps)
    conversation = getattr(datapoint, 'conversation', None)
    if conversation is not None:
        return sum(len(turn.steps) for turn in conversation.turns)
    return 0


def _step_count_is_allowed(args, datapoint) -> bool:
    total = _datapoint_total_steps(datapoint)
    minimum = getattr(args, 'min_total_steps', None)
    maximum = getattr(args, 'max_total_steps', None)
    if minimum is not None and total < minimum:
        print(f'\n✗ Datapoint has {total} steps; need at least {minimum}')
        return False
    if maximum is not None and total > maximum:
        print(f'\n✗ Datapoint has {total} steps; maximum is {maximum}')
        return False
    return True


def run_step_by_step(
    args,
    llm_client,
    tool_manager,
    categories,
    output_path,
    judge_client=None,
    final_response_client=None,
    grounding_client=None,
):
    """Run step-by-step (legacy single-query) generation."""
    usage_clients = _unique_clients(
        llm_client, judge_client, final_response_client, grounding_client
    )
    usage_baselines = [client.get_token_usage() for client in usage_clients]

    def usage_since_last_accept() -> TokenUsageStats:
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_prompt_tokens": 0,
            "cost_usd": 0.0,
            "total_calls": 0,
        }
        for client, baseline in zip(usage_clients, usage_baselines):
            current = client.get_token_usage()
            for key in totals:
                if key == "total_calls":
                    current_value = current.get(
                        "total_attempts",
                        current.get("total_calls", 0),
                    )
                    baseline_value = baseline.get(
                        "total_attempts",
                        baseline.get("total_calls", 0),
                    )
                else:
                    current_value = current.get(key, 0)
                    baseline_value = baseline.get(key, 0)
                totals[key] += current_value - baseline_value
        return TokenUsageStats(
            prompt_tokens=int(totals["prompt_tokens"]),
            completion_tokens=int(totals["completion_tokens"]),
            total_tokens=int(totals["total_tokens"]),
            reasoning_tokens=int(totals["reasoning_tokens"]),
            cached_prompt_tokens=int(totals["cached_prompt_tokens"]),
            cost_usd=float(totals["cost_usd"]),
            total_llm_calls=int(totals["total_calls"]),
        )

    def reset_usage_baselines() -> None:
        for index, client in enumerate(usage_clients):
            usage_baselines[index] = client.get_token_usage()

    allow_refusal = bool(getattr(args, 'allow_refusal', False))
    allow_parallel = bool(getattr(args, 'allow_parallel', False))
    features_enabled = allow_refusal or allow_parallel
    if features_enabled:
        from refuse_parallel import RefusalParallelStepByStepGenerator
        generator = RefusalParallelStepByStepGenerator(
            llm_client=llm_client,
            tool_manager=tool_manager,
            num_actions=args.num_actions,
            judge_client=judge_client,
            allow_refusal=allow_refusal,
            refusal_rate=float(getattr(args, 'refusal_rate', 0.12)),
            allow_parallel=allow_parallel,
            parallel_rate=float(getattr(args, 'parallel_rate', 0.25)),
            max_parallel_width=int(getattr(args, 'max_parallel_width', 3)),
            require_feature=bool(getattr(args, 'require_feature', False)),
            feature_difficulty=str(
                getattr(args, 'feature_difficulty', 'standard')
            ),
            naturalize_queries=bool(
                getattr(args, 'naturalize_queries', False)
            ),
            forced_refusal_reason=getattr(args, 'refusal_reason', 'random'),
            optimized_pipeline=bool(args.optimized_pipeline),
        )
    else:
        # Keep the exact original class and constructor path when features are
        # disabled. This is the default and preserves current-main behavior.
        generator = StepByStepGenerator(
            llm_client=llm_client,
            tool_manager=tool_manager,
            num_actions=args.num_actions,
            judge_client=judge_client,
            optimized_pipeline=bool(args.optimized_pipeline),
        )

    generator.configure_final_stage_clients(
        final_response_client=final_response_client,
        grounding_client=grounding_client,
    )

    datapoints = []
    attempt = 0
    diversity_feedback = None
    seen_signatures = _load_trajectory_signatures(
        [output_path, *args.dedupe_against]
    )
    registry_path = (
        Path(args.dedupe_registry).expanduser()
        if args.dedupe_registry
        else None
    )
    if registry_path:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        if registry_path.exists():
            seen_signatures.update(
                line.strip()
                for line in registry_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            )
    if seen_signatures:
        print(
            f'Loaded {len(seen_signatures)} semantic trajectory signatures '
            'for deduplication'
        )

    def reserve_signature(signature: str) -> bool:
        if signature in seen_signatures:
            return False
        if registry_path:
            with registry_path.open('a+', encoding='utf-8') as registry:
                fcntl.flock(registry.fileno(), fcntl.LOCK_EX)
                registry.seek(0)
                registered = {
                    line.strip()
                    for line in registry
                    if line.strip()
                }
                if signature in registered:
                    seen_signatures.add(signature)
                    return False
                registry.seek(0, os.SEEK_END)
                registry.write(signature + '\n')
                registry.flush()
                os.fsync(registry.fileno())
        seen_signatures.add(signature)
        return True

    candidate_starts_for_row = 0
    configured_candidate_call_limit = max(
        1, int(args.max_calls_per_candidate)
    )
    configured_candidate_token_limit = max(
        1, int(args.max_tokens_per_accepted_row)
    )
    while len(datapoints) < args.num_datapoints:
        row_usage = usage_since_last_accept()
        remaining_row_calls = (
            int(args.max_calls_per_accepted_row)
            - row_usage.total_llm_calls
        )
        remaining_row_tokens = (
            int(args.max_tokens_per_accepted_row)
            - row_usage.total_tokens
        )
        if (
            candidate_starts_for_row
            >= int(args.max_candidate_starts_per_row)
            or remaining_row_calls <= 0
            or remaining_row_tokens <= 0
        ):
            print(
                "\n✗ Hard accepted-row budget reached; stopping cleanly "
                f"after {candidate_starts_for_row} candidate starts, "
                f"{row_usage.total_llm_calls} requests, and "
                f"{row_usage.total_tokens} tokens."
            )
            break
        generator.max_calls_per_candidate = min(
            configured_candidate_call_limit,
            remaining_row_calls,
        )
        generator.max_tokens_per_candidate = min(
            configured_candidate_token_limit,
            remaining_row_tokens,
        )
        remaining = args.num_datapoints - len(datapoints)

        print(f"\n{'='*70}")
        print(f"Generated: {len(datapoints)}/{args.num_datapoints} | Remaining: {remaining}")
        print("=" * 70)

        focus_category = random.choice(categories)
        print(f"Focus category: {focus_category}")

        # Randomize num_actions per accepted candidate when requested.
        if args.num_actions_range:
            num_actions = random.randint(
                args.num_actions_range[0], args.num_actions_range[1]
            )
            generator.num_actions = num_actions
            print(f"Actions for this datapoint: {num_actions}")

        # Generate datapoint
        candidate_starts_for_row += 1
        try:
            datapoint = generator.generate_datapoint(
                focus_category=focus_category,
                context_hint=diversity_feedback,
            )
        except GenerationBudgetExceeded as exc:
            print(f"\n✗ Candidate budget exhausted: {exc}")
            datapoint = None
        except RuntimeError as exc:
            if "Access denied by security policy" not in str(exc):
                raise
            print(
                "\n✗ Provider security policy rejected this candidate; "
                "resampling without writing a partial datapoint."
            )
            datapoint = None

        if datapoint and not _step_count_is_allowed(args, datapoint):
            datapoint = None

        if datapoint:
            signature = _trajectory_signature(datapoint)
            if not reserve_signature(signature):
                print(
                    '\n✗ Duplicate semantic trajectory rejected '
                    '(same ordered tools and arguments)'
                )
                duplicate_tools = list(datapoint.trajectory.tools_used)
                diversity_feedback = (
                    'DUPLICATE_TRAJECTORY: The previous candidate used the '
                    f'public tool sequence {duplicate_tools}, which duplicated '
                    'an existing trajectory. Choose a different valid tool '
                    'sequence and a substantially different scenario. Any new '
                    'argument values must be explicitly stated in the user query.'
                )
                attempt += 1
                continue

            diversity_feedback = None
            # Include all discarded candidates since the previous accepted row.
            datapoint.token_usage = usage_since_last_accept()
            datapoint_dict = datapoint.model_dump()
            datapoint_dict['timestamp'] = datetime.now().isoformat()
            datapoint_dict['generation_attempt'] = attempt

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(datapoint_dict, ensure_ascii=False) + '\n')

            datapoints.append(datapoint)
            candidate_starts_for_row = 0
            reset_usage_baselines()
            generator.max_calls_per_candidate = configured_candidate_call_limit
            generator.max_tokens_per_candidate = configured_candidate_token_limit
            print(f"\n✓ Successfully generated and verified datapoint {len(datapoints)}")
            print(f" Query: {datapoint.trajectory.query}")
            print(f" Tools used: {datapoint.trajectory.tools_used}")
        else:
            print(f"\n✗ Failed to generate datapoint")

        attempt += 1

    return datapoints


def run_multi_turn(
    args,
    llm_client,
    tool_manager,
    categories,
    output_path,
    checkpoint_manager=None,
    judge_client=None,
    final_response_client=None,
    grounding_client=None,
):
    """Run multi-turn (multiple user exchanges) generation with checkpoint support."""
    required_tools = list(
        dict.fromkeys(str(name) for name in getattr(args, 'required_tool', []))
    )
    invalid_required_tools = [
        name for name in required_tools if not tool_manager.tool_exists(name)
    ]
    if invalid_required_tools:
        raise ValueError(
            'Unknown --required-tool values: '
            + ', '.join(invalid_required_tools)
        )
    required_categories = {
        tool_manager.get_tool_category(name) for name in required_tools
    }
    required_categories.discard(None)
    if len(required_categories) > 1:
        raise ValueError(
            '--required-tool currently requires one shared category; got '
            + ', '.join(sorted(required_categories))
        )
    if required_categories and not required_categories.issubset(set(categories)):
        raise ValueError(
            '--required-tool category is outside the selected --category: '
            + ', '.join(sorted(required_categories))
        )
    if required_tools:
        configured_slots = getattr(args, 'blueprint_actions_per_turn', None)
        slot_count = (
            sum(configured_slots)
            if configured_slots is not None
            else int(args.num_turns)
            * int(
                getattr(args, 'blueprint_max_actions_per_turn', None)
                or args.num_actions
            )
        )
        if len(required_tools) > slot_count:
            raise ValueError(
                f'{len(required_tools)} required tools exceed {slot_count} '
                'available blueprint action slots'
            )
        print('Hard required tools: ' + ', '.join(required_tools))
    candidate_archive = None
    build_partial_candidate_record = None
    decorate_full_candidate_record = None
    if getattr(args, 'candidate_archive_dir', None):
        from candidate_archive import (
            CandidateArchive,
            build_partial_candidate_record as _build_partial_candidate_record,
            decorate_full_candidate_record as _decorate_full_candidate_record,
        )
        candidate_archive = CandidateArchive(args.candidate_archive_dir)
        build_partial_candidate_record = _build_partial_candidate_record
        decorate_full_candidate_record = _decorate_full_candidate_record
        print(
            "Candidate archive enabled: "
            f"{Path(args.candidate_archive_dir).expanduser()}"
        )

    curriculum = None
    curriculum_descriptors = None
    if getattr(args, 'curriculum_mode', 'off') == 'bfcl-v3':
        from evolutionary_curriculum import (
            EvolutionaryCurriculum,
            candidate_descriptors,
        )

        coverage_path = getattr(args, 'coverage_state', None)
        if not coverage_path:
            coverage_path = str(Path(output_path).with_suffix('.coverage.json'))
        curriculum = EvolutionaryCurriculum(
            tools=tool_manager.get_tools_json_schema(),
            state_path=coverage_path,
            seed=int(getattr(args, 'curriculum_seed', 20260730)),
            all_tools_rate=float(getattr(args, 'all_tools_rate', 0.25)),
            cross_domain_rate=float(getattr(args, 'cross_domain_rate', 0.45)),
            hard_distractor_count=int(
                getattr(args, 'hard_distractor_count', 48)
            ),
            target_tools_per_candidate=int(
                getattr(args, 'target_tools_per_candidate', 2)
            ),
            lessons_path=getattr(args, 'evolution_lessons', None),
        )
        curriculum_descriptors = candidate_descriptors
        print(f"BFCL-v3 curriculum enabled: {Path(coverage_path).expanduser()}")

    usage_clients = _unique_clients(
        llm_client, judge_client, final_response_client, grounding_client
    )
    usage_baselines = [client.get_token_usage() for client in usage_clients]

    def usage_since(baselines) -> TokenUsageStats:
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_prompt_tokens": 0,
            "cost_usd": 0.0,
            "total_calls": 0,
        }
        for client, baseline in zip(usage_clients, baselines):
            current = client.get_token_usage()
            for key in totals:
                if key == "total_calls":
                    current_value = current.get(
                        "total_attempts",
                        current.get("total_calls", 0),
                    )
                    baseline_value = baseline.get(
                        "total_attempts",
                        baseline.get("total_calls", 0),
                    )
                else:
                    current_value = current.get(key, 0)
                    baseline_value = baseline.get(key, 0)
                totals[key] += current_value - baseline_value
        return TokenUsageStats(
            prompt_tokens=int(totals["prompt_tokens"]),
            completion_tokens=int(totals["completion_tokens"]),
            total_tokens=int(totals["total_tokens"]),
            reasoning_tokens=int(totals["reasoning_tokens"]),
            cached_prompt_tokens=int(totals["cached_prompt_tokens"]),
            cost_usd=float(totals["cost_usd"]),
            total_llm_calls=int(totals["total_calls"]),
        )

    def usage_since_last_accept() -> TokenUsageStats:
        return usage_since(usage_baselines)

    def capture_usage_baseline():
        return [client.get_token_usage() for client in usage_clients]

    def reset_usage_baselines() -> None:
        for index, client in enumerate(usage_clients):
            usage_baselines[index] = client.get_token_usage()

    candidate_starts_for_row = 0
    candidate_sequence = 0
    configured_candidate_call_limit = max(
        1, int(args.max_calls_per_candidate)
    )
    configured_candidate_token_limit = max(
        1, int(args.max_tokens_per_accepted_row)
    )

    allow_refusal = bool(getattr(args, 'allow_refusal', False))
    allow_parallel = bool(getattr(args, 'allow_parallel', False))
    features_enabled = allow_refusal or allow_parallel
    if features_enabled:
        from refuse_parallel import RefusalParallelMultiTurnGenerator
        generator = RefusalParallelMultiTurnGenerator(
            llm_client=llm_client,
            tool_manager=tool_manager,
            num_turns=args.num_turns,
            actions_per_turn=args.num_actions,
            judge_client=judge_client,
            allow_refusal=allow_refusal,
            refusal_rate=float(getattr(args, 'refusal_rate', 0.12)),
            allow_parallel=allow_parallel,
            parallel_rate=float(getattr(args, 'parallel_rate', 0.25)),
            max_parallel_width=int(getattr(args, 'max_parallel_width', 3)),
            require_feature=bool(getattr(args, 'require_feature', False)),
            feature_difficulty=str(
                getattr(args, 'feature_difficulty', 'standard')
            ),
            naturalize_queries=bool(
                getattr(args, 'naturalize_queries', False)
            ),
            multi_turn_feature_schedule=str(
                getattr(args, 'multi_turn_feature_schedule', 'terminal')
            ),
            forced_refusal_reason=getattr(args, 'refusal_reason', 'random'),
            interactive_refusal_turn=getattr(
                args, 'interactive_refusal_turn', None
            ),
            blueprint_max_actions_per_turn=getattr(
                args, 'blueprint_max_actions_per_turn', None
            ),
            blueprint_actions_per_turn=getattr(
                args, 'blueprint_actions_per_turn', None
            ),
            optimized_pipeline=bool(args.optimized_pipeline),
        )
    else:
        # Keep the exact original class and constructor path when features are
        # disabled. This is the default and preserves current-main behavior.
        generator = MultiTurnGenerator(
            llm_client=llm_client,
            tool_manager=tool_manager,
            num_turns=args.num_turns,
            actions_per_turn=args.num_actions,
            judge_client=judge_client,
            optimized_pipeline=bool(args.optimized_pipeline),
            blueprint_max_actions_per_turn=getattr(
                args, 'blueprint_max_actions_per_turn', None
            ),
            blueprint_actions_per_turn=getattr(
                args, 'blueprint_actions_per_turn', None
            ),
        )

    generator.configure_final_stage_clients(
        final_response_client=final_response_client,
        grounding_client=grounding_client,
    )

    # Create checkpoint callback if checkpoint manager is provided
    checkpoint_callback = None
    if checkpoint_manager:
        def save_checkpoint(state):
            checkpoint_manager.save(state)
        checkpoint_callback = save_checkpoint

    datapoints = []
    seen_signatures = _load_trajectory_signatures(
        [output_path, *getattr(args, 'dedupe_against', [])]
    )
    registry_path = (
        Path(args.dedupe_registry).expanduser()
        if getattr(args, 'dedupe_registry', None)
        else None
    )
    if registry_path:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        if registry_path.exists():
            seen_signatures.update(
                line.strip()
                for line in registry_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            )

    def reserve_signature(signature: str) -> bool:
        if signature in seen_signatures:
            return False
        if registry_path:
            with registry_path.open('a+', encoding='utf-8') as registry:
                fcntl.flock(registry.fileno(), fcntl.LOCK_EX)
                registry.seek(0)
                registered = {
                    line.strip()
                    for line in registry
                    if line.strip()
                }
                if signature in registered:
                    seen_signatures.add(signature)
                    return False
                registry.seek(0, os.SEEK_END)
                registry.write(signature + '\n')
                registry.flush()
                os.fsync(registry.fileno())
        seen_signatures.add(signature)
        return True

    def _candidate_id(sequence: int) -> str:
        stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
        return f"{stamp}-p{os.getpid()}-c{sequence:06d}"

    def _archive_full_candidate(
        payload,
        *,
        candidate_id: str,
        disposition: str,
        usage: TokenUsageStats,
        rejection: dict | None = None,
    ) -> None:
        if candidate_archive is None:
            return
        try:
            record = decorate_full_candidate_record(
                payload,
                candidate_id=candidate_id,
                disposition=disposition,
                usage=usage.model_dump(),
                rejection=rejection,
            )
            candidate_archive.write(record)
        except Exception as exc:
            print(f"Warning: failed to archive {candidate_id}: {exc}")

    def _archive_partial_candidate(
        checkpoint_state: dict | None,
        *,
        candidate_id: str,
        usage: TokenUsageStats,
        rejection: dict,
        focus_category: str | None,
    ) -> None:
        if candidate_archive is None or not checkpoint_state:
            return
        partial = checkpoint_state.get('partial_conversation', {})
        turns = partial.get('turns', []) if isinstance(partial, dict) else []
        has_tool_call = any(
            step.get('tool_calls')
            for turn in turns
            if isinstance(turn, dict)
            for step in turn.get('steps', [])
            if isinstance(step, dict)
        )
        # A blueprint-only failure is not a generated trajectory.  Archive only
        # after at least one actual tool call has been produced/executed.
        if not has_tool_call:
            return
        try:
            available_tools = generator._get_policy_tool_schemas(focus_category)
        except Exception:
            available_tools = []
        try:
            record = build_partial_candidate_record(
                checkpoint_state,
                candidate_id=candidate_id,
                usage=usage.model_dump(),
                rejection=rejection,
                available_tools=available_tools,
            )
            candidate_archive.write(record)
        except Exception as exc:
            print(f"Warning: failed to archive {candidate_id}: {exc}")

    def _start_candidate_tracking():
        nonlocal candidate_sequence
        candidate_sequence += 1
        return _candidate_id(candidate_sequence), capture_usage_baseline()

    def _annotate_curriculum(dp, directive: dict | None) -> None:
        if curriculum is None:
            return
        dp.generation_metadata['generation_directive'] = copy.deepcopy(
            directive or {}
        )
        dp.generation_metadata['posthoc_descriptors'] = (
            curriculum_descriptors(dp.model_dump())
        )

    def generation_needed() -> bool:
        if len(datapoints) < args.num_datapoints:
            return True
        if (
            curriculum is None
            or not getattr(args, 'continue_until_full_tool_coverage', False)
        ):
            return False
        if curriculum.is_complete(
            int(getattr(args, 'coverage_target_per_tool', 1))
        ):
            return False
        return len(datapoints) < (
            args.num_datapoints
            + int(getattr(args, 'max_extra_coverage_rows', 500))
        )

    while generation_needed():
        remaining = max(0, args.num_datapoints - len(datapoints))

        print(f"\n{'='*70}")
        print(f"Generated: {len(datapoints)}/{args.num_datapoints} | Remaining: {remaining}")
        print("=" * 70)

        # Check for existing checkpoint to resume
        row_usage = usage_since_last_accept()
        remaining_row_calls = (
            int(args.max_calls_per_accepted_row)
            - row_usage.total_llm_calls
        )
        remaining_row_tokens = (
            int(args.max_tokens_per_accepted_row)
            - row_usage.total_tokens
        )
        if (
            candidate_starts_for_row
            >= int(args.max_candidate_starts_per_row)
            or remaining_row_calls <= 0
            or remaining_row_tokens <= 0
        ):
            print(
                "\n✗ Hard accepted-row budget reached; stopping cleanly "
                f"after {candidate_starts_for_row} candidate starts, "
                f"{row_usage.total_llm_calls} calls, and "
                f"{row_usage.total_tokens} tokens."
            )
            break
        generator.max_calls_per_candidate = min(
            configured_candidate_call_limit,
            remaining_row_calls,
        )
        generator.max_tokens_per_candidate = min(
            configured_candidate_token_limit,
            remaining_row_tokens,
        )

        if checkpoint_manager and args.resume:
            checkpoint = checkpoint_manager.load()
            if checkpoint.get('partial_conversation'):
                # Try to resume from checkpoint
                resume_directive = copy.deepcopy(
                    checkpoint.get('generation_directive', {})
                )
                candidate_starts_for_row += 1
                candidate_id, candidate_usage_baseline = _start_candidate_tracking()
                print(f"\nFound checkpoint with {checkpoint.get('completed_turns', 0)} completed turns")
                try:
                    dp = generator.continue_from_checkpoint(
                        checkpoint,
                        focus_category=checkpoint.get('focus_category'),
                        checkpoint_callback=checkpoint_callback,
                    )
                except GenerationBudgetExceeded as exc:
                    print(f"\n✗ Resume budget exhausted: {exc}")
                    generator._mark_failure(
                        code='RESUME_BUDGET_EXHAUSTED',
                        stage='checkpoint_resume',
                        details={'error': str(exc)[:500]},
                    )
                    dp = None
                except RuntimeError as exc:
                    print(f"\n✗ Resume generation failed: {exc}")
                    generator._mark_failure(
                        code='RESUME_RUNTIME_ERROR',
                        stage='checkpoint_resume',
                        details={
                            'error_type': type(exc).__name__,
                            'error': str(exc)[:500],
                        },
                    )
                    dp = None
                if dp:
                    if not _step_count_is_allowed(args, dp):
                        rejection = {
                            'code': 'STEP_COUNT_FILTER_REJECTED',
                            'stage': 'post_generation_filter',
                        }
                        _archive_full_candidate(
                            dp,
                            candidate_id=candidate_id,
                            disposition='rejected',
                            usage=usage_since(candidate_usage_baseline),
                            rejection=rejection,
                        )
                        if curriculum is not None:
                            curriculum.observe(
                                directive=resume_directive,
                                row=dp.model_dump(),
                                accepted=False,
                                rejection=rejection,
                            )
                        checkpoint_manager.clear()
                        continue
                    signature = _trajectory_signature_from_dict(dp.model_dump())
                    if not reserve_signature(signature):
                        print('\n✗ Duplicate multi-turn trajectory rejected')
                        rejection = {
                            'code': 'DUPLICATE_TRAJECTORY',
                            'stage': 'post_generation_deduplication',
                        }
                        _archive_full_candidate(
                            dp,
                            candidate_id=candidate_id,
                            disposition='rejected',
                            usage=usage_since(candidate_usage_baseline),
                            rejection=rejection,
                        )
                        if curriculum is not None:
                            curriculum.observe(
                                directive=resume_directive,
                                row=dp.model_dump(),
                                accepted=False,
                                rejection=rejection,
                            )
                        checkpoint_manager.clear()
                        continue
                    # Include every rejected attempt since the previous accepted
                    # row.  Per-datapoint generator counters intentionally reset
                    # on resampling and otherwise hide real provider spend.
                    dp.token_usage = usage_since_last_accept()
                    _annotate_curriculum(dp, resume_directive)
                    dp_dict = dp.model_dump()
                    dp_dict['timestamp'] = datetime.now().isoformat()
                    dp_dict['resumed'] = True
                    dp_dict['resumed_from_turn'] = checkpoint.get('completed_turns', 0)

                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(dp_dict, ensure_ascii=False) + '\n')

                    if curriculum is not None:
                        curriculum.observe(
                            directive=resume_directive,
                            row=dp_dict,
                            accepted=True,
                        )

                    _archive_full_candidate(
                        dp,
                        candidate_id=candidate_id,
                        disposition='accepted',
                        usage=usage_since(candidate_usage_baseline),
                    )

                    datapoints.append(dp)
                    reset_usage_baselines()
                    candidate_starts_for_row = 0
                    generator.max_calls_per_candidate = (
                        configured_candidate_call_limit
                    )
                    generator.max_tokens_per_candidate = (
                        configured_candidate_token_limit
                    )
                    checkpoint_manager.clear()
                    print(f"\n✓ Successfully resumed and completed datapoint {len(datapoints)}")
                    print(f" Task: {dp.conversation.overall_task[:80]}")
                    print(f" Turns: {len(dp.conversation.turns)}")
                    print(f" Tools: {dp.conversation.tools_used}")
                    continue
                else:
                    print(f"\n✗ Failed to resume from checkpoint, starting fresh")
                    rejection = copy.deepcopy(
                        getattr(generator, 'last_failure', None)
                        or {
                            'code': 'RESUME_RETURNED_NONE',
                            'stage': 'checkpoint_resume',
                        }
                    )
                    partial_state = (
                        copy.deepcopy(
                            getattr(generator, 'last_partial_candidate', None)
                        )
                        or checkpoint
                    )
                    _archive_partial_candidate(
                        partial_state,
                        candidate_id=candidate_id,
                        usage=usage_since(candidate_usage_baseline),
                        rejection=rejection,
                        focus_category=checkpoint.get('focus_category'),
                    )
                    if curriculum is not None:
                        curriculum.observe(
                            directive=resume_directive,
                            row=None,
                            accepted=False,
                            rejection=rejection,
                        )
                    checkpoint_manager.clear()
                    # Re-enter through the row-budget check before starting a
                    # fresh candidate. Falling through here used to bypass both
                    # max-candidate-starts and the remaining-call calculation.
                    continue

        generation_directive = None
        if curriculum is not None:
            generation_directive = curriculum.next_directive().to_dict()
            target_categories = generation_directive.get(
                'target_categories', []
            )
            focus_category = (
                target_categories[0]
                if target_categories
                else random.choice(categories)
            )
            print(
                'Curriculum directive: '
                f"targets={generation_directive.get('target_tools')} "
                f"mode={generation_directive.get('context_mode')} "
                f"motif={generation_directive.get('motif')}"
            )
        else:
            focus_category = random.choice(categories)
        if required_tools:
            generation_directive = copy.deepcopy(generation_directive or {})
            generation_directive['hard_required_tools'] = list(required_tools)
            existing_targets = list(
                generation_directive.get('target_tools', [])
            )
            generation_directive['target_tools'] = list(
                dict.fromkeys([*required_tools, *existing_targets])
            )
            generation_directive.setdefault(
                'target_categories', sorted(required_categories)
            )
            generation_directive.setdefault('context_mode', 'hard_required')
            generation_directive.setdefault('motif', 'coverage_cooccurrence')
            generation_directive.setdefault('style_seed', 'natural')
            generation_directive.setdefault('scenario_seed', 'coverage')
            generation_directive.setdefault('soft_requirements', [])
            generation_directive.setdefault('lesson_ids', [])
            generation_directive.setdefault('lesson_texts', [])
            if generation_directive.get('allowed_tools'):
                generation_directive['allowed_tools'] = sorted(
                    set(generation_directive['allowed_tools'])
                    | set(required_tools)
                )
        print(f"Focus category: {focus_category}")

        # Start fresh generation with checkpoint callback
        candidate_starts_for_row += 1
        candidate_id, candidate_usage_baseline = _start_candidate_tracking()
        latest_candidate_checkpoint = None

        def capture_candidate_checkpoint(state):
            nonlocal latest_candidate_checkpoint
            latest_candidate_checkpoint = copy.deepcopy(state)
            if checkpoint_callback:
                checkpoint_callback(state)

        try:
            dp = generator.generate_multi_turn_datapoint(
                focus_category=focus_category,
                checkpoint_callback=capture_candidate_checkpoint,
                generation_directive=generation_directive,
            )
        except GenerationBudgetExceeded as exc:
            print(f"\n✗ Candidate budget exhausted: {exc}")
            generator._mark_failure(
                code='CANDIDATE_BUDGET_EXHAUSTED',
                stage='generation_budget',
                details={'error': str(exc)[:500]},
            )
            dp = None
        except RuntimeError as exc:
            # Provider/model failures (including reasoning-only empty output)
            # are candidate rejections, not reasons to lose usage accounting by
            # crashing the whole subprocess.
            print(f"\n✗ Candidate generation failed: {exc}")
            generator._mark_failure(
                code='CANDIDATE_RUNTIME_ERROR',
                stage='llm_generation',
                details={
                    'error_type': type(exc).__name__,
                    'error': str(exc)[:500],
                },
            )
            dp = None

        if dp is None:
            print(f"\n✗ Failed to generate datapoint, retrying...")
            rejection = copy.deepcopy(
                getattr(generator, 'last_failure', None)
                or {
                    'code': 'GENERATOR_RETURNED_NONE',
                    'stage': 'unknown',
                }
            )
            partial_state = (
                copy.deepcopy(getattr(generator, 'last_partial_candidate', None))
                or latest_candidate_checkpoint
            )
            _archive_partial_candidate(
                partial_state,
                candidate_id=candidate_id,
                usage=usage_since(candidate_usage_baseline),
                rejection=rejection,
                focus_category=focus_category,
            )
            if curriculum is not None:
                curriculum.observe(
                    directive=generation_directive or {},
                    row=None,
                    accepted=False,
                    rejection=rejection,
                )
            if checkpoint_manager:
                checkpoint_manager.clear()
            continue

        hard_required = set(
            (generation_directive or {}).get('hard_required_tools', [])
        )
        actual_tools = set(dp.conversation.tools_used)
        missing_required = sorted(hard_required - actual_tools)
        if missing_required:
            print(
                '\n✗ Hard required tools missing from executed trajectory: '
                + ', '.join(missing_required)
            )
            rejection = {
                'code': 'HARD_REQUIRED_TOOL_MISSING',
                'stage': 'post_generation_filter',
                'details': {'missing_tools': missing_required},
            }
            _archive_full_candidate(
                dp,
                candidate_id=candidate_id,
                disposition='rejected',
                usage=usage_since(candidate_usage_baseline),
                rejection=rejection,
            )
            if curriculum is not None:
                curriculum.observe(
                    directive=generation_directive or {},
                    row=dp.model_dump(),
                    accepted=False,
                    rejection=rejection,
                )
            if checkpoint_manager:
                checkpoint_manager.clear()
            continue

        if not _step_count_is_allowed(args, dp):
            rejection = {
                'code': 'STEP_COUNT_FILTER_REJECTED',
                'stage': 'post_generation_filter',
            }
            _archive_full_candidate(
                dp,
                candidate_id=candidate_id,
                disposition='rejected',
                usage=usage_since(candidate_usage_baseline),
                rejection=rejection,
            )
            if curriculum is not None:
                curriculum.observe(
                    directive=generation_directive or {},
                    row=dp.model_dump(),
                    accepted=False,
                    rejection=rejection,
                )
            if checkpoint_manager:
                checkpoint_manager.clear()
            continue

        signature = _trajectory_signature_from_dict(dp.model_dump())
        if not reserve_signature(signature):
            print('\n✗ Duplicate multi-turn trajectory rejected')
            rejection = {
                'code': 'DUPLICATE_TRAJECTORY',
                'stage': 'post_generation_deduplication',
            }
            _archive_full_candidate(
                dp,
                candidate_id=candidate_id,
                disposition='rejected',
                usage=usage_since(candidate_usage_baseline),
                rejection=rejection,
            )
            if curriculum is not None:
                curriculum.observe(
                    directive=generation_directive or {},
                    row=dp.model_dump(),
                    accepted=False,
                    rejection=rejection,
                )
            if checkpoint_manager:
                checkpoint_manager.clear()
            continue

        # Include failed blueprints/turns that were discarded before this
        # accepted row, so cost reports reflect actual generation spend.
        dp.token_usage = usage_since_last_accept()
        _annotate_curriculum(dp, generation_directive)

        # Clear checkpoint on successful completion
        if checkpoint_manager:
            checkpoint_manager.clear()

        dp_dict = dp.model_dump()
        dp_dict['timestamp'] = datetime.now().isoformat()

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dp_dict, ensure_ascii=False) + '\n')

        if curriculum is not None:
            curriculum.observe(
                directive=generation_directive or {},
                row=dp_dict,
                accepted=True,
            )

        _archive_full_candidate(
            dp,
            candidate_id=candidate_id,
            disposition='accepted',
            usage=usage_since(candidate_usage_baseline),
        )

        datapoints.append(dp)
        reset_usage_baselines()
        candidate_starts_for_row = 0
        generator.max_calls_per_candidate = configured_candidate_call_limit
        generator.max_tokens_per_candidate = configured_candidate_token_limit
        print(f"\n✓ Successfully generated datapoint {len(datapoints)}")
        print(f" Task: {dp.conversation.overall_task[:80]}")
        print(f" Turns: {len(dp.conversation.turns)}")
        print(f" Tools: {dp.conversation.tools_used}")

    return datapoints


def main():
    args = parse_args()
    os.environ["APIGEN_MAX_CALLS_PER_CANDIDATE"] = str(
        max(1, args.max_calls_per_candidate)
    )
    os.environ["APIGEN_MAX_TOKENS_PER_CANDIDATE"] = str(
        max(1, args.max_tokens_per_accepted_row)
    )
    os.environ["APIGEN_MAX_TURN_ATTEMPTS"] = str(args.max_turn_attempts)
    os.environ.setdefault("APIGEN_APPLICATION_LLM_ATTEMPTS", "1")

    tool_pool_path = str(Path(args.tool_pool).expanduser())
    invocation_examples_path = str(Path(args.invocation_examples).expanduser())

    mode_label = "MULTI-TURN" if args.mode == "multi-turn" else "STEP-BY-STEP"
    print("=" * 70)
    print(f"{mode_label} DATAPOINT GENERATION")
    print("=" * 70)
    print(f"Target: {args.num_datapoints} datapoints")
    if args.mode == "multi-turn":
        print(f"Turns per conversation: {args.num_turns}")
        print(f"Actions per turn: {args.num_actions}")
    else:
        print(f"Actions per datapoint: {args.num_actions}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        sys.exit(1)

    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model=args.model,
        hf_tokenizer_id=None
    )

    print("\nLoading tools...")
    tools_by_category = load_tool_categories(tool_pool_path)

    if args.category:
        filtered = {args.category: tools_by_category.get(args.category)}
        if filtered[args.category] is None:
            print(f"Error: Category '{args.category}' not found")
            available = list(tools_by_category.keys())
            print(f"Available categories: {available}")
            return
        tools_by_category = filtered

    total_tools = sum(len(t) for t in tools_by_category.values())
    print(f"Loaded {total_tools} tools across {len(tools_by_category)} categories")

    for cat, tools in sorted(tools_by_category.items()):
        print(f"  {cat:30s}: {len(tools):3d} tools")

    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path,
        use_config_pool=args.config_pool,
    )

    # Initialize judge client if specified, otherwise reuse generator client
    if args.judge_model:
        judge_api_base = args.judge_api_base or api_base
        judge_api_key = args.judge_api_key or api_key
        judge_client = LocalOpenAILLMClient(
            url=judge_api_base,
            api_key=judge_api_key,
            api_model=args.judge_model,
            hf_tokenizer_id=None
        )
    else:
        judge_client = llm_client

    try:
        final_response_client, grounding_client = build_final_stage_clients(
            args,
            llm_client=llm_client,
            judge_client=judge_client,
            main_api_base=api_base,
            main_api_key=api_key,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR configuring final-stage models: {exc}")
        sys.exit(1)

    print(
        "Final response writer: "
        f"{getattr(final_response_client, 'api_model', None)}"
    )
    print(
        "Grounding judge: "
        f"{getattr(grounding_client, 'api_model', None)}"
    )

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager if --checkpoint is provided
    checkpoint_manager = None
    if args.checkpoint:
        checkpoint_manager = CheckpointManager(args.checkpoint)
        print(f"Checkpoint file: {args.checkpoint}")
        print(f"Resume: {'enabled' if args.resume else 'disabled'}")

    categories = list(tools_by_category.keys())

    try:
        if args.mode == "multi-turn":
            datapoints = run_multi_turn(
                args,
                llm_client,
                tool_manager,
                categories,
                args.output,
                checkpoint_manager=checkpoint_manager,
                judge_client=judge_client,
                final_response_client=final_response_client,
                grounding_client=grounding_client,
            )
        else:
            datapoints = run_step_by_step(
                args,
                llm_client,
                tool_manager,
                categories,
                args.output,
                judge_client=judge_client,
                final_response_client=final_response_client,
                grounding_client=grounding_client,
            )
    finally:
        if args.usage_report:
            usage_clients = _unique_clients(
                llm_client, judge_client, final_response_client, grounding_client
            )
            totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cached_prompt_tokens": 0,
                "cost_usd": 0.0,
                "total_calls": 0,
                "total_attempts": 0,
            }
            provider_counts = {}
            for client in usage_clients:
                current = client.get_token_usage()
                for provider, count in current.get(
                    "provider_counts", {}
                ).items():
                    provider_counts[provider] = (
                        provider_counts.get(provider, 0) + int(count)
                    )
                for key in totals:
                    if key == "total_attempts":
                        value = current.get(
                            "total_attempts",
                            current.get("total_calls", 0),
                        )
                    else:
                        value = current.get(key, 0)
                    totals[key] += value or 0
            report_path = Path(args.usage_report).expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        **totals,
                        "successful_llm_calls": int(totals["total_calls"]),
                        "total_http_attempts": int(totals["total_attempts"]),
                        "total_llm_calls": int(totals["total_attempts"]),
                        "requested_provider": os.getenv(
                            "APIGEN_OPENROUTER_PROVIDER", ""
                        ).strip() or None,
                        "provider_counts": provider_counts,
                        "accepted_rows": len(locals().get("datapoints", [])),
                        "model": args.model,
                        "judge_model": args.judge_model or args.model,
                        "final_response_model": getattr(
                            final_response_client, "api_model", None
                        ),
                        "grounding_model": getattr(
                            grounding_client, "api_model", None
                        ),
                        "output": str(Path(args.output).resolve()),
                        "updated_at": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(report_path)

    # Summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total generated: {len(datapoints)}/{args.num_datapoints}")
    print(f"Output file: {args.output}")

    if datapoints:
        from collections import Counter

        if args.mode == "multi-turn":
            tools_used_all = []
            for dp in datapoints:
                tools_used_all.extend(dp.conversation.tools_used)
            tool_counts = Counter(tools_used_all)

            print(f"\nTop 10 tools used:")
            for tool, count in tool_counts.most_common(10):
                print(f"  {tool}: {count}")

            total_calls = sum(dp.token_usage.total_llm_calls for dp in datapoints)
            total_tokens = sum(dp.token_usage.total_tokens for dp in datapoints)
        else:
            tools_used_all = []
            for dp in datapoints:
                tools_used_all.extend(dp.trajectory.tools_used)
            tool_counts = Counter(tools_used_all)

            print(f"\nTop 10 tools used:")
            for tool, count in tool_counts.most_common(10):
                print(f"  {tool}: {count}")

            total_calls = sum(dp.token_usage.total_llm_calls for dp in datapoints)
            total_tokens = sum(dp.token_usage.total_tokens for dp in datapoints)

        print(f"\nToken Usage Statistics:")
        print(f"  Total LLM calls: {total_calls}")
        print(f"  Total tokens: {total_tokens:,}")
        if datapoints:
            print(f"  Average per datapoint:")
            print(f"    - LLM calls: {total_calls / len(datapoints):.1f}")
            print(f"    - Tokens: {total_tokens // len(datapoints):,}")

    print("=" * 70)


if __name__ == "__main__":
    main()
