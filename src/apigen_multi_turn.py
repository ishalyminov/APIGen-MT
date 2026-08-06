"""Multi-turn conversation generator with step-by-step tool simulation.

Extends StepByStepGenerator to produce multi-turn conversations where
a separate LLM generates each user turn based on the dialog blueprint
and the current point in the conversation.
"""

import json
import copy
import time
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from apigen_step_by_step import (
    StepByStepGenerator,
    StepByStepDatapoint,
    ConversationTrajectory,
    TokenUsageStats,
    TrajectoryStep,
    ToolCallWithOutput,
    QueryGenerationResult,
    filter_api_state,
)
from llm_client import LLMClient
from tool_manager import ToolManager
from domain_hints import get_domain_hints


class Turn(BaseModel):
    """A single user-assistant turn in a multi-turn conversation."""

    turn_number: int
    user_query: str
    query_intent: str = ""
    steps: List[TrajectoryStep] = Field(default_factory=list)
    assistant_response: str = ""
    expected_tools: List[str] = Field(default_factory=list)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    quality_verification: Dict[str, Any] = Field(default_factory=dict)


class MultiTurnConversation(BaseModel):
    """Complete multi-turn conversation trajectory."""

    overall_task: str = ""
    turns: List[Turn] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    categories_used: List[str] = Field(default_factory=list)
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None


class MultiTurnDatapoint(BaseModel):
    """Complete multi-turn datapoint."""

    conversation: MultiTurnConversation
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    verification_result: Optional[Dict[str, Any]] = None
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)


class DialogBlueprint(BaseModel):
    """Blueprint for a multi-turn dialog."""

    overall_task: str
    num_turns: int
    turns: List[Dict[str, Any]] = Field(default_factory=list)


class MultiTurnGenerator(StepByStepGenerator):
    """Generator for multi-turn conversations with step-by-step tool simulation.

    Flow:
      1. Generate dialog blueprint (overall task + per-turn goals)
      2. For each turn:
         a. Generate user query based on blueprint + conversation history
         b. Stage 1.5: Adjust API state for expected tools
         c. Stage 2: Generate and execute tool invocations
         d. Generate assistant response
         e. Persist execution context for next turn
      3. Assemble multi-turn datapoint
    """

    _LEGACY_INTERMEDIATE_PLACEHOLDER = (
        "The requested actions completed successfully."
    )

    @classmethod
    def _is_legacy_intermediate_placeholder(cls, response: str) -> bool:
        """Recognize the old synthetic target regardless of casing/spacing."""
        return (
            str(response).strip().casefold()
            == cls._LEGACY_INTERMEDIATE_PLACEHOLDER.casefold()
        )

    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager,
        num_turns: int = 2,
        actions_per_turn: int = 2,
        validate_outputs: bool = True,
        judge_client: Optional[LLMClient] = None,
        optimized_pipeline: Optional[bool] = None,
        blueprint_max_actions_per_turn: Optional[int] = None,
        blueprint_actions_per_turn: Optional[List[int]] = None,
    ):
        super().__init__(
            llm_client,
            tool_manager,
            actions_per_turn,
            validate_outputs,
            judge_client=judge_client,
            optimized_pipeline=optimized_pipeline,
        )
        self.num_turns = num_turns
        self.blueprint_max_actions_per_turn = max(
            1,
            min(
                6,
                (
                    actions_per_turn
                    if blueprint_max_actions_per_turn is None
                    else blueprint_max_actions_per_turn
                ),
            ),
        )
        self.blueprint_actions_per_turn = (
            list(blueprint_actions_per_turn)
            if blueprint_actions_per_turn is not None
            else None
        )
        if self.blueprint_actions_per_turn is not None:
            if len(self.blueprint_actions_per_turn) != self.num_turns:
                raise ValueError(
                    "blueprint_actions_per_turn must contain exactly "
                    f"{self.num_turns} entries"
                )
            invalid = [
                value
                for value in self.blueprint_actions_per_turn
                if not (
                    isinstance(value, int)
                    and 1 <= value <= self.blueprint_max_actions_per_turn
                )
            ]
            if invalid:
                raise ValueError(
                    "blueprint_actions_per_turn entries must be integers in "
                    f"1-{self.blueprint_max_actions_per_turn}; got {invalid}"
                )

        # Diagnostics only.  The outer runner may archive a partial trajectory
        # after a post-generation rejection.  These fields do not affect
        # prompts, retry behavior, or acceptance criteria.
        self.last_failure: Optional[Dict[str, Any]] = None
        self.last_partial_candidate: Optional[Dict[str, Any]] = None
        self._active_generation_directive: Dict[str, Any] = {}

    def _mark_failure(
        self,
        *,
        code: str,
        stage: str,
        turn_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_failure = {
            "code": code,
            "stage": stage,
            "turn_number": turn_number,
            "details": copy.deepcopy(details or {}),
        }

    def _remember_partial_candidate(
        self,
        *,
        blueprint: DialogBlueprint,
        conversation: MultiTurnConversation,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]],
        focus_category: Optional[str],
        execution_context: Dict[str, Any],
        pending_turn: Optional[Turn] = None,
    ) -> None:
        snapshot_conversation = copy.deepcopy(conversation)
        if pending_turn is not None:
            snapshot_conversation.turns.append(copy.deepcopy(pending_turn))
        self.last_partial_candidate = {
            "blueprint": {
                "overall_task": blueprint.overall_task,
                "num_turns": blueprint.num_turns,
                "turns": copy.deepcopy(blueprint.turns),
            },
            "partial_conversation": snapshot_conversation.model_dump(),
            "execution_context": copy.deepcopy(execution_context),
            "completed_turns": len(snapshot_conversation.turns),
            "initial_api_state": copy.deepcopy(initial_api_state),
            "focus_category": focus_category,
            "generation_directive": copy.deepcopy(
                self._active_generation_directive
            ),
        }

    @staticmethod
    def _trajectory_contains_terminal_behavior(
        trajectory: List[TrajectoryStep],
    ) -> bool:
        return any(
            call.tool_name == "refuse"
            for step in trajectory
            for call in step.tool_calls
        )

    @staticmethod
    def _missing_hard_required_tools(
        turns: List[Dict[str, Any]],
        directive: Dict[str, Any],
    ) -> List[str]:
        """Return hard coverage targets absent from a proposed blueprint."""
        required = {
            str(name) for name in directive.get("hard_required_tools", [])
        }
        planned = {
            str(name)
            for turn in turns
            for name in turn.get("expected_tools", [])
        }
        return sorted(required - planned)

    def _produce_turn_response(
        self,
        *,
        turn_index: int,
        total_turns: int,
        query: str,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Defer ordinary optimized responses for one episode-level batch.

        BFCL ends a turn when the model emits a response with no function calls.
        We therefore keep a real grounded assistant response for every completed
        user turn, but generate all ordinary turn-ending responses together in
        one request after tool execution.  This preserves the existing request
        budget while avoiding generic placeholders in SFT/RL transcripts.

        Refusal/clarification turns remain immediate because their certified
        natural-language response is itself the target behavior.
        """
        terminal_behavior = self._trajectory_contains_terminal_behavior(
            trajectory
        )
        if self.optimized_pipeline and not terminal_behavior:
            quality = {
                "passed": True,
                "issue_codes": [],
                "validator": "deferred_batched_turn_response",
                "generated_by_llm": False,
                "deferred": True,
                "response_required": True,
            }
            self._last_final_response_quality = quality
            return "", quality

        response = self._generate_final_response(
            query, trajectory, execution_context
        )
        return response, dict(self._last_final_response_quality)

    @staticmethod
    def _response_is_deferred(response_quality: Dict[str, Any]) -> bool:
        return bool(response_quality.get("deferred", False))

    def _turn_response_evidence(self, turn: Turn) -> Dict[str, Any]:
        """Build compact, policy-visible evidence for one turn response."""
        actions: List[Dict[str, Any]] = []
        used_tool_names: List[str] = []
        for step in turn.steps:
            for call in step.tool_calls:
                actions.append(
                    {
                        "step_number": step.step_number,
                        "tool": call.tool_name,
                        "arguments": copy.deepcopy(call.arguments),
                        "output": copy.deepcopy(call.output),
                    }
                )
                if call.tool_name not in used_tool_names:
                    used_tool_names.append(call.tool_name)

        tool_definitions: List[Dict[str, Any]] = []
        for name in used_tool_names:
            schema = self.tool_manager.get_tool_schema(name)
            if not schema:
                continue
            tool_definitions.append(
                {
                    "name": schema.get("name", name),
                    "description": schema.get("description", ""),
                    "parameters": copy.deepcopy(schema.get("parameters", {})),
                    "output_schema": copy.deepcopy(
                        schema.get("output_schema", {})
                    ),
                    "output_description": schema.get(
                        "output_description", ""
                    ),
                }
            )

        return {
            "turn_number": turn.turn_number,
            "user_query": turn.user_query,
            "tool_definitions": tool_definitions,
            "actual_tool_calls_and_outputs": actions,
        }

    def _generate_batched_turn_responses(
        self,
        evidence: List[Dict[str, Any]],
    ) -> Dict[int, str]:
        """Generate real turn-ending responses for all ordinary turns once."""
        prompt = f"""Generate one concise assistant response for each completed
tool-use turn below.

Each response is the no-tool-call message that ends that user turn. It will be
inserted after that turn's tool calls and tool results, before the next user
message. Treat every turn independently: a response may use only that turn's
user query, tool definitions, actual arguments, and actual tool outputs. Never
use facts from another turn in the batch.

=== TURN EVIDENCE ===
{json.dumps(evidence, indent=2, ensure_ascii=False, default=str)}

=== RULES ===
- Answer the results requested in that turn. Do not output a generic completion
  placeholder.
- Confirm mutations only when the actual tool output/state transition supports
  success.
- Do not make another tool call or emit tool-call markup.
- Do not perform new calculations, conversions, rankings, selections, or
  lookups. Report only values returned by the tools.
- Do not invent units, IDs, fields, recommendations, or verification claims.
- If a tool failed, state the failure accurately.
- Keep each response concise but complete enough to terminate the turn.

Return ONLY one JSON object with exactly one item per supplied turn:
{{
  "responses": [
    {{"turn_number": 1, "response": "..."}}
  ]
}}
"""
        raw = self._safe_llm_generate(
            [{"role": "user", "content": prompt}],
            llm=self.final_response_llm,
            purpose="final_response_generate",
        )
        result = self._extract_json_object(raw)
        items = result.get("responses")
        if not isinstance(items, list):
            raise ValueError("turn response writer returned no responses list")

        expected = {int(item["turn_number"]) for item in evidence}
        responses: Dict[int, str] = {}
        unexpected: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                turn_number = int(item.get("turn_number"))
            except (TypeError, ValueError):
                continue
            response = str(item.get("response", "")).strip()
            if turn_number not in expected:
                unexpected.add(turn_number)
                continue
            if self._is_legacy_intermediate_placeholder(response):
                raise ValueError(
                    "turn response writer reproduced the legacy placeholder"
                )
            if response:
                if turn_number in responses:
                    raise ValueError(
                        f"duplicate response for turn {turn_number}"
                    )
                responses[turn_number] = response

        if set(responses) != expected or unexpected:
            missing = sorted(expected - set(responses))
            extra = sorted(unexpected)
            raise ValueError(
                "turn response writer returned wrong turn set: "
                f"missing={missing}, extra={extra}"
            )
        return responses

    def _verify_batched_turn_responses(
        self,
        evidence: List[Dict[str, Any]],
        responses: Dict[int, str],
    ) -> Dict[int, Dict[str, Any]]:
        """Ground every generated turn response in one fail-closed judge call."""
        allowed_codes = {
            "UNSUPPORTED_CLAIM",
            "UNCALLED_CALCULATION",
            "INVENTED_UNIT",
            "FALSE_SUCCESS_CLAIM",
            "OMITTED_REQUIRED_RESULT",
            "MISREPRESENTED_VERIFICATION",
            "CONTRADICTS_TOOL_OUTPUT",
            "CROSS_TURN_LEAKAGE",
            "TOOL_CALL_IN_RESPONSE",
            "EMPTY_RESPONSE",
            "GROUNDING_VERIFIER_UNAVAILABLE",
            "OTHER_INVALID",
        }
        candidates = [
            {
                "turn_number": item["turn_number"],
                "response": responses[int(item["turn_number"])],
            }
            for item in evidence
        ]
        prompt = f"""Certify the assistant response for every completed tool-use
turn below.

Judge each turn only against its own user query, used tool definitions, actual
tool arguments, and actual outputs. Information from another turn in the batch
is not evidence for this turn.

=== TURN EVIDENCE ===
{json.dumps(evidence, indent=2, ensure_ascii=False, default=str)}

=== CANDIDATE RESPONSES ===
{json.dumps(candidates, indent=2, ensure_ascii=False, default=str)}

Check that each response:
- answers all explicitly requested read results for that turn;
- claims a mutation succeeded only when the tool evidence supports it;
- contains no arithmetic, conversion, ranking, lookup, unit, ID, or fact not
  supported by that turn's evidence;
- does not leak a result from another turn;
- contains no new tool call or tool-call markup;
- is non-empty and suitable as the no-tool-call message that ends the turn.

Use only these issue codes:
{json.dumps(sorted(allowed_codes))}

Return ONLY JSON with exactly one item per turn:
{{
  "turns": [
    {{"turn_number": 1, "is_grounded": true, "issue_codes": []}}
  ]
}}
"""
        try:
            raw = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.grounding_judge,
                purpose="final_response_grounding_judge",
            )
            result = self._extract_json_object(raw)
            items = result.get("turns")
            if not isinstance(items, list):
                raise ValueError("grounding judge returned no turns list")

            expected = {int(item["turn_number"]) for item in evidence}
            quality_by_turn: Dict[int, Dict[str, Any]] = {}
            unexpected: set[int] = set()
            duplicate = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    turn_number = int(item.get("turn_number"))
                except (TypeError, ValueError):
                    continue
                if turn_number not in expected:
                    unexpected.add(turn_number)
                    continue
                if turn_number in quality_by_turn:
                    duplicate = True
                    continue
                raw_codes = item.get("issue_codes", [])
                if not isinstance(raw_codes, list):
                    raw_codes = ["OTHER_INVALID"]
                unknown_codes = [
                    str(code)
                    for code in raw_codes
                    if str(code) not in allowed_codes
                ]
                codes = [
                    str(code)
                    for code in raw_codes
                    if str(code) in allowed_codes
                ]
                if unknown_codes and "OTHER_INVALID" not in codes:
                    codes.append("OTHER_INVALID")
                passed = item.get("is_grounded") is True and not codes
                if not passed and not codes:
                    codes = ["OTHER_INVALID"]
                quality_by_turn[turn_number] = {
                    "passed": passed,
                    "issue_codes": codes,
                    "validator": "batched_turn_response_grounding_judge",
                    "generated_by_llm": True,
                    "deferred": False,
                    "response_required": True,
                }

            if set(quality_by_turn) != expected or unexpected or duplicate:
                raise ValueError("grounding judge returned an incomplete turn set")
            return quality_by_turn
        except Exception as exc:
            print(f"    Warning: batched turn-response judge failed: {exc}")
            return {
                int(item["turn_number"]): {
                    "passed": False,
                    "issue_codes": ["GROUNDING_VERIFIER_UNAVAILABLE"],
                    "validator": "batched_turn_response_grounding_judge",
                    "generated_by_llm": True,
                    "deferred": False,
                    "response_required": True,
                }
                for item in evidence
            }

    def _finalize_deferred_turn_responses(
        self,
        conversation: MultiTurnConversation,
    ) -> bool:
        """Replace empty/legacy placeholders with real grounded responses."""
        deferred_turns: List[Turn] = []
        for turn in conversation.turns:
            response_quality = dict(
                turn.quality_verification.get(
                    "final_response_grounding", {}
                )
            )
            if self._is_legacy_intermediate_placeholder(
                turn.assistant_response
            ):
                turn.assistant_response = ""
                response_quality = {
                    "passed": True,
                    "issue_codes": [],
                    "validator": "legacy_placeholder_migrated",
                    "generated_by_llm": False,
                    "deferred": True,
                    "response_required": True,
                }
                turn.quality_verification[
                    "final_response_grounding"
                ] = response_quality

            if response_quality.get("deferred", False):
                deferred_turns.append(turn)

        if not deferred_turns:
            return True

        # Preserve the existing single-turn final-response path.  Besides being
        # backward compatible with refusal/parallel subclasses, it already uses
        # exactly one writer call and one grounding call for ordinary turns.
        if len(deferred_turns) == 1:
            turn = deferred_turns[0]
            response = self._generate_final_response(
                turn.user_query,
                turn.steps,
                turn.execution_context,
            )
            quality = dict(self._last_final_response_quality)
            if self._is_legacy_intermediate_placeholder(response):
                response = ""
                quality = {
                    **quality,
                    "passed": False,
                    "issue_codes": ["OMITTED_REQUIRED_RESULT"],
                }
            turn.assistant_response = response
            turn.quality_verification[
                "final_response_grounding"
            ] = quality
            turn.quality_verification[
                "turn_response_grounding"
            ] = copy.deepcopy(quality)
            query_ok = bool(
                turn.quality_verification.get("query_preflight", {}).get(
                    "passed", False
                )
            )
            transitions_ok = all(
                check.get("passed", False)
                for check in turn.quality_verification.get(
                    "transition_checks", []
                )
            )
            passed = (
                bool(response)
                and query_ok
                and transitions_ok
                and bool(quality.get("passed", False))
            )
            turn.quality_verification["passed"] = passed
            return passed

        evidence = [
            self._turn_response_evidence(turn) for turn in deferred_turns
        ]
        try:
            responses = self._generate_batched_turn_responses(evidence)
        except Exception as exc:
            print(f"    Error generating batched turn responses: {exc}")
            for turn in deferred_turns:
                quality = {
                    "passed": False,
                    "issue_codes": ["OTHER_INVALID"],
                    "validator": "batched_turn_response_writer",
                    "generated_by_llm": True,
                    "deferred": False,
                    "response_required": True,
                }
                turn.quality_verification[
                    "final_response_grounding"
                ] = quality
                turn.quality_verification["passed"] = False
            return False

        qualities = self._verify_batched_turn_responses(evidence, responses)
        all_passed = True
        for turn in deferred_turns:
            turn_number = turn.turn_number
            turn.assistant_response = responses[turn_number]
            quality = qualities[turn_number]
            turn.quality_verification[
                "final_response_grounding"
            ] = quality
            turn.quality_verification[
                "turn_response_grounding"
            ] = copy.deepcopy(quality)
            query_ok = bool(
                turn.quality_verification.get("query_preflight", {}).get(
                    "passed", False
                )
            )
            transitions_ok = all(
                check.get("passed", False)
                for check in turn.quality_verification.get(
                    "transition_checks", []
                )
            )
            turn_passed = query_ok and transitions_ok and bool(
                quality.get("passed", False)
            )
            turn.quality_verification["passed"] = turn_passed
            all_passed = all_passed and turn_passed

        self._last_final_response_quality = {
            "passed": all_passed,
            "issue_codes": sorted(
                {
                    code
                    for quality in qualities.values()
                    for code in quality.get("issue_codes", [])
                }
            ),
            "validator": "batched_turn_response_grounding_judge",
            "turn_count": len(deferred_turns),
        }
        return all_passed

    def continue_from_checkpoint(
            self,
            checkpoint: dict,
            focus_category: Optional[str] = None,
            query_retries: int = 3,
            tool_retries: int = 3,
            checkpoint_callback: Optional[callable] = None,
    ) -> Optional[MultiTurnDatapoint]:
        """Continue generating a multi-turn datapoint from a checkpoint.

        Args:
            checkpoint: Dict containing:
                - 'blueprint': The dialog blueprint
                - 'partial_conversation': Partially completed MultiTurnConversation
                - 'execution_context': Dict of execution context
                - 'completed_turns': Number of turns completed
                - 'initial_api_state': API state before this datapoint started
                - 'token_usage': Accumulated token usage so far
            focus_category: Category for tool filtering
            query_retries: Max retries for query generation
            tool_retries: Max retries for tool generation

        Returns:
            MultiTurnDatapoint if successful, None otherwise
        """
        blueprint_data = checkpoint.get('blueprint')
        if not blueprint_data:
            print("✗ Checkpoint missing blueprint")
            return None
        self._reset_token_tracking()
        self._capture_initial_usage()
        self.last_failure = None
        self.last_partial_candidate = None
        self._active_generation_directive = copy.deepcopy(
            checkpoint.get('generation_directive', {})
        )

        from apigen_multi_turn import DialogBlueprint, MultiTurnConversation

        try:
            blueprint = DialogBlueprint(
                overall_task=blueprint_data.get('overall_task', ''),
                num_turns=blueprint_data.get('num_turns', self.num_turns),
                turns=blueprint_data.get('turns', [])
            )
        except Exception as e:
            print(f"✗ Failed to reconstruct blueprint: {e}")
            return None

        partial_conv_data = checkpoint.get('partial_conversation', {})
        try:
            conversation = MultiTurnConversation(
                overall_task=partial_conv_data.get('overall_task', blueprint.overall_task),
                turns=partial_conv_data.get('turns', []),
                tools_used=partial_conv_data.get('tools_used', []),
                categories_used=partial_conv_data.get('categories_used', []),
            )
        except Exception as e:
            print(f"✗ Failed to reconstruct conversation: {e}")
            return None

        completed_turns = checkpoint.get('completed_turns', len(conversation.turns))
        persisted_execution_context = copy.deepcopy(
            checkpoint.get('execution_context', {})
        )
        execution_context: Dict[str, Any] = {}
        initial_api_state = checkpoint.get('initial_api_state')

        print(f"\n{'=' * 70}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"=" * 70)
        print(f" Overall task: {blueprint.overall_task}")
        print(f" Completed turns: {completed_turns}/{blueprint.num_turns}")
        print(f" Remaining turns: {blueprint.num_turns - completed_turns}")

        # Initialize API state and replay completed turns
        if self._python_tools_available and initial_api_state:
            print("\n Restoring API state from checkpoint...")
            self.tool_manager.restore_api_state(initial_api_state)

            # Replay completed turns to restore side-effects from their tool calls
            if completed_turns > 0:
                print(f" Replaying {completed_turns} turns to restore state...")
                for turn_idx in range(completed_turns):
                    if turn_idx >= len(conversation.turns):
                        break
                    turn = conversation.turns[turn_idx]
                    for step in turn.steps:
                        for tc in step.tool_calls:
                            if self.tool_manager.has_python_implementation(tc.tool_name):
                                self.tool_manager.invoke_python_tool(tc.tool_name, tc.arguments)
                            output = tc.output
                            if isinstance(output, dict):
                                for key, value in output.items():
                                    execution_context[f"{tc.tool_name}_{key}"] = value
                                if 'access_token' in output:
                                    execution_context['access_token'] = output['access_token']
                            execution_context[f"{tc.tool_name}_output"] = output
                    execution_context.setdefault('turn_outputs', []).append(
                        self._aggregate_turn_outputs(turn.steps)
                    )
                print(f" Replayed {completed_turns} turns to restore state")

        # Persisted context can contain auxiliary aliases not derivable from the
        # public transcript. Prefer it when present, while keeping deterministic
        # replay sufficient for candidate-archive resumes.
        execution_context.update(persisted_execution_context)
        # ``turn_outputs`` is fully derivable from the public trajectory and the
        # replayed form is authoritative.  Older checkpoints stored only output
        # values, which made repeated calls to the same tool ambiguous (for
        # example, two stock quotes with no record of their input symbols).
        # Rebuild it so each visible output remains paired with the arguments
        # that produced it; those arguments were already part of the trace.
        if completed_turns > 0:
            execution_context['turn_outputs'] = [
                self._aggregate_turn_outputs(turn.steps)
                for turn in conversation.turns[:completed_turns]
            ]

        self._update_token_usage()

        tools_used = set(conversation.tools_used)
        categories_used = set(conversation.categories_used)
        for prior_turn in conversation.turns[:completed_turns]:
            for step in prior_turn.steps:
                for call in step.tool_calls:
                    tools_used.add(call.tool_name)
                    category = self.tool_manager.get_tool_category(
                        call.tool_name
                    )
                    if category:
                        categories_used.add(category)

        # Process remaining turns
        for turn_idx in range(completed_turns, blueprint.num_turns):
            print(f"\n{'=' * 70}")
            print(f"TURN {turn_idx + 1}/{blueprint.num_turns} (resumed)")
            print("=" * 70)

            turn_spec = blueprint.turns[turn_idx] if turn_idx < len(blueprint.turns) else {}

            query_result = self._generate_turn_query(
                blueprint=blueprint,
                conversation=conversation,
                turn_index=turn_idx,
            )
            if query_result is None:
                print(f"✗ Turn {turn_idx + 1} failed: Could not generate query")
                return None
            self._update_token_usage()

            trajectory, ec = self._stage2_generate_tools(
                query_result,
                min(tool_retries, self.max_turn_attempts),
                initial_execution_context=execution_context,
            )
            if trajectory is None:
                print(
                    f"✗ Turn {turn_idx + 1}: turn compile/repair failed"
                )
                return None
            errors = self._validate_tool_arguments(trajectory)
            cross_errors = self._validate_cross_turn_consistency(
                trajectory, execution_context
            )
            if errors or cross_errors:
                print(
                    f"✗ Turn {turn_idx + 1}: deterministic post-execution "
                    "validation failed"
                )
                for err in errors:
                    print(f"    arg: {err}")
                for err in cross_errors:
                    print(f"    cross: {err}")
                return None
            self._update_token_usage()

            # Merge turn context into persistent execution_context
            for k, v in ec.items():
                execution_context[k] = v

            turn_output_aggregate = self._aggregate_turn_outputs(trajectory)
            if 'turn_outputs' not in execution_context:
                execution_context['turn_outputs'] = []
            execution_context['turn_outputs'].append(turn_output_aggregate)

            assistant_response, response_quality = self._produce_turn_response(
                turn_index=turn_idx,
                total_turns=blueprint.num_turns,
                query=query_result.query,
                trajectory=trajectory,
                execution_context=execution_context,
            )
            self._update_token_usage()
            if (
                not assistant_response
                and not self._response_is_deferred(response_quality)
            ):
                print(f"✗ Turn {turn_idx + 1}: Could not produce a grounded response")
                return None

            transition_checks = [
                step.quality_verification for step in trajectory
            ]
            turn_quality = {
                "passed": (
                    bool(query_result.quality_preflight.get("passed", False))
                    and all(
                        check.get("passed", False)
                        for check in transition_checks
                    )
                    and bool(response_quality.get("passed", False))
                ),
                "query_preflight": dict(query_result.quality_preflight),
                "transition_checks": transition_checks,
                "final_response_grounding": response_quality,
            }

            for step in trajectory:
                for tc in step.tool_calls:
                    tools_used.add(tc.tool_name)
                    cat = self.tool_manager.get_tool_category(tc.tool_name)
                    if cat:
                        categories_used.add(cat)

            from apigen_multi_turn import Turn
            turn = Turn(
                turn_number=turn_idx + 1,
                user_query=query_result.query,
                query_intent=query_result.intent,
                steps=trajectory,
                assistant_response=assistant_response,
                expected_tools=query_result.expected_tools,
                execution_context=dict(execution_context),
                quality_verification=turn_quality,
            )
            conversation.turns.append(turn)
            self._remember_partial_candidate(
                blueprint=blueprint,
                conversation=conversation,
                initial_api_state=initial_api_state,
                focus_category=focus_category,
                execution_context=execution_context,
            )

            if checkpoint_callback:
                checkpoint_callback({
                    'blueprint': {
                        'overall_task': blueprint.overall_task,
                        'num_turns': blueprint.num_turns,
                        'turns': copy.deepcopy(blueprint.turns),
                    },
                    'partial_conversation': conversation.model_dump(),
                    'execution_context': copy.deepcopy(execution_context),
                    'completed_turns': turn_idx + 1,
                    'initial_api_state': copy.deepcopy(initial_api_state),
                    'focus_category': focus_category,
                    'generation_directive': copy.deepcopy(
                        self._active_generation_directive
                    ),
                })

            print(f"\n✓ Turn {turn_idx + 1} complete (resumed)")
            print(f"   Query: {query_result.query[:80]}...")

        # Generate and ground all ordinary turn-ending responses in one writer
        # call plus one judge call. Refusal/clarification responses were already
        # certified when their terminal action was created.
        if self.optimized_pipeline:
            if not self._finalize_deferred_turn_responses(conversation):
                print(
                    "✗ Resumed conversation failed batched turn-response "
                    "grounding"
                )
                self._mark_failure(
                    code="BATCHED_TURN_RESPONSE_GROUNDING_FAILED",
                    stage="turn_responses",
                )
                return None
            self._update_token_usage()

        # Finalize
        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = initial_api_state

        quality_checks = [turn.quality_verification for turn in conversation.turns]
        quality_passed = all(
            check.get("passed", False) for check in quality_checks
        )
        if not quality_passed:
            print("✗ Resumed conversation failed the positive-RL quality gate")
            return None

        available_tools = self._get_policy_tool_schemas(focus_category)
        datapoint = MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
                "actions_per_turn": self.num_actions,
                "configured_actions_per_turn": self.num_actions,
                "actual_steps_per_turn": [
                    len(turn.steps) for turn in conversation.turns
                ],
                "actual_tool_calls_per_turn": [
                    sum(len(step.tool_calls) for step in turn.steps)
                    for turn in conversation.turns
                ],
                "blueprint_actions_per_turn": (
                    list(self.blueprint_actions_per_turn)
                    if self.blueprint_actions_per_turn is not None
                    else None
                ),
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "resumed_from_turn": completed_turns,
                "blueprint_queries": [t.get("user_query", "") for t in blueprint.turns],
                "turn_expected_tools": [t.get("expected_tools", []) for t in blueprint.turns],
                "rl_quality_gate_passed": True,
                "model_routing": self._model_routing_metadata(),
                "tool_contract_hash": self._tool_contract_hash(available_tools),
                "generation_pipeline": (
                    "turn_compiler_v1_batched_turn_responses"
                    if self.optimized_pipeline
                    else "legacy_per_tool"
                ),
                "turn_response_policy": (
                    "batched_grounded_per_turn"
                    if self.optimized_pipeline
                    else "per_turn_generation"
                ),
                "generation_directive": copy.deepcopy(
                    self._active_generation_directive
                ),
                "llm_budget": {
                    "max_calls_per_candidate": self.max_calls_per_candidate,
                    "max_tokens_per_candidate": self.max_tokens_per_candidate,
                    "max_turn_attempts": self.max_turn_attempts,
                },
            },
            verification_result={
                "overall_verification_passed": True,
                "rl_quality_gate": {
                    "passed": True,
                    "turn_checks": quality_checks,
                },
            },
            token_usage=self._get_token_stats(),
            initial_api_state=conversation.initial_api_state,
            available_tools=available_tools,
        )

        print("\n" + "=" * 70)
        print("✓ RESUMED MULTI-TURN DATAPOINT GENERATION COMPLETE")
        print("=" * 70)
        print(f" Turns: {len(conversation.turns)}")
        print(f" Tools used: {conversation.tools_used}")
        print(f" Total tool calls: {sum(len(t.steps) for t in conversation.turns)}")

        return datapoint

    # ─────────────────────── Public entry point ───────────────────────

    def generate_multi_turn_datapoint(
            self,
            focus_category: Optional[str] = None,
            query_retries: int = 3,
            tool_retries: int = 3,
            checkpoint_callback: Optional[callable] = None,
            generation_directive: Optional[Dict[str, Any]] = None,
    ) -> Optional[MultiTurnDatapoint]:
        """Generate a multi-turn datapoint.

        Args:
            focus_category: Category for tool filtering
            query_retries: Max retries for query generation
            tool_retries: Max retries for tool generation
            checkpoint_callback: Optional callback(state_dict) called after each
                turn completes. The state_dict contains:
                - 'blueprint': The dialog blueprint
                - 'partial_conversation': Partially completed MultiTurnConversation
                - 'execution_context': Current execution context
                - 'completed_turns': Number of turns completed
                - 'initial_api_state': API state before this datapoint started
                - 'focus_category': The category being used
            generation_directive: Optional advisory curriculum directive. Its
                allowed_tools define the policy-visible context, while motifs
                and targets remain soft generation guidance rather than gates.
        """

        self._reset_token_tracking()
        self._capture_initial_usage()
        self.last_failure = None
        self.last_partial_candidate = None
        # Always replace the previous directive so a failed/resampled candidate
        # cannot leak its tool context or motif into the next attempt.
        self._active_generation_directive = copy.deepcopy(
            generation_directive or {}
        )

        # Initialize API state for the whole conversation
        initial_api_state = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state(force_new=True)
            initial_api_state = self.tool_manager.get_api_state()
            print(f" Captured initial API state ({len(initial_api_state)} class keys)")

        # Stage 0: Generate dialog blueprint
        print("\n" + "=" * 70)
        print("STAGE 0: Generate Dialog Blueprint")
        print("=" * 70)
        blueprint = self._stage0_generate_blueprint(focus_category, initial_api_state)
        if blueprint is None:
            print("✗ Stage 0 failed: Could not generate dialog blueprint")
            self._mark_failure(
                code="BLUEPRINT_GENERATION_FAILED",
                stage="blueprint",
            )
            return None
        self._update_token_usage()
        print(f" Overall task: {blueprint.overall_task}")
        for i, t in enumerate(blueprint.turns, 1):
            uq = t.get('user_query', '')
            print(f"   Turn {i}: {uq[:80]}...")

        conversation = MultiTurnConversation(overall_task=blueprint.overall_task)

        # Identity is validated from the blueprint and sampled state. Do not
        # rewrite account/card/user identity after generation.

        execution_context: Dict[str, Any] = {}
        tools_used = set()
        categories_used = set()

        for turn_idx in range(blueprint.num_turns):
            print(f"\n{'=' * 70}")
            print(f"TURN {turn_idx + 1}/{blueprint.num_turns}")
            print("=" * 70)

            turn_spec = blueprint.turns[turn_idx] if turn_idx < len(blueprint.turns) else {}

            # Stage 1: Generate user query for this turn
            query_result = self._generate_turn_query(
                blueprint=blueprint,
                conversation=conversation,
                turn_index=turn_idx,
            )
            if query_result is None:
                print(f"✗ Turn {turn_idx + 1} failed: Could not generate query")
                self._mark_failure(
                    code="TURN_QUERY_GENERATION_FAILED",
                    stage="turn_query",
                    turn_number=turn_idx + 1,
                )
                return None
            self._update_token_usage()

            # Stage 2: Generate and execute tool invocations (pass persistent execution_context)
            # Note: State adjustment removed - tool calls modify API state which persists,
            # and we pass current API state snapshot to the tool manager LLM
            trajectory, ec = self._stage2_generate_tools(
                query_result,
                min(tool_retries, self.max_turn_attempts),
                initial_execution_context=execution_context,
            )
            if trajectory is None:
                print(
                    f"✗ Turn {turn_idx + 1}: turn compile/repair failed"
                )
                self._mark_failure(
                    code="TURN_COMPILE_OR_EXECUTE_FAILED",
                    stage="turn_compile_execute",
                    turn_number=turn_idx + 1,
                )
                return None
            errors = self._validate_tool_arguments(trajectory)
            cross_errors = self._validate_cross_turn_consistency(
                trajectory, execution_context
            )
            if errors or cross_errors:
                print(
                    f"✗ Turn {turn_idx + 1}: deterministic post-execution "
                    "validation failed"
                )
                for err in errors:
                    print(f"    arg: {err}")
                for err in cross_errors:
                    print(f"    cross: {err}")
                pending_turn = Turn(
                    turn_number=turn_idx + 1,
                    user_query=query_result.query,
                    query_intent=query_result.intent,
                    steps=trajectory,
                    assistant_response="",
                    expected_tools=query_result.expected_tools,
                    execution_context=copy.deepcopy(execution_context),
                    quality_verification={
                        "passed": False,
                        "deterministic_argument_errors": list(errors),
                        "cross_turn_errors": list(cross_errors),
                    },
                )
                self._remember_partial_candidate(
                    blueprint=blueprint,
                    conversation=conversation,
                    initial_api_state=initial_api_state,
                    focus_category=focus_category,
                    execution_context=execution_context,
                    pending_turn=pending_turn,
                )
                self._mark_failure(
                    code="DETERMINISTIC_POST_EXECUTION_VALIDATION_FAILED",
                    stage="post_execution_validation",
                    turn_number=turn_idx + 1,
                    details={
                        "argument_errors": list(errors),
                        "cross_turn_errors": list(cross_errors),
                    },
                )
                return None
            self._update_token_usage()

            # Merge turn context into persistent execution_context
            for k, v in ec.items():
                execution_context[k] = v

            # Store turn outputs for TURN{N} placeholder resolution without
            # overwriting repeated calls to the same tool.
            turn_output_aggregate = self._aggregate_turn_outputs(trajectory)
            if 'turn_outputs' not in execution_context:
                execution_context['turn_outputs'] = []
            execution_context['turn_outputs'].append(turn_output_aggregate)

            # Generate assistant response for this turn
            assistant_response, response_quality = self._produce_turn_response(
                turn_index=turn_idx,
                total_turns=blueprint.num_turns,
                query=query_result.query,
                trajectory=trajectory,
                execution_context=execution_context,
            )
            self._update_token_usage()
            if (
                not assistant_response
                and not self._response_is_deferred(response_quality)
            ):
                print(f"✗ Turn {turn_idx + 1}: Could not produce a grounded response")
                pending_turn = Turn(
                    turn_number=turn_idx + 1,
                    user_query=query_result.query,
                    query_intent=query_result.intent,
                    steps=trajectory,
                    assistant_response="",
                    expected_tools=query_result.expected_tools,
                    execution_context=copy.deepcopy(execution_context),
                    quality_verification={
                        "passed": False,
                        "final_response_grounding": copy.deepcopy(response_quality),
                    },
                )
                self._remember_partial_candidate(
                    blueprint=blueprint,
                    conversation=conversation,
                    initial_api_state=initial_api_state,
                    focus_category=focus_category,
                    execution_context=execution_context,
                    pending_turn=pending_turn,
                )
                self._mark_failure(
                    code="TURN_RESPONSE_GROUNDING_FAILED",
                    stage="turn_response",
                    turn_number=turn_idx + 1,
                    details={"response_quality": copy.deepcopy(response_quality)},
                )
                return None

            transition_checks = [
                step.quality_verification for step in trajectory
            ]
            turn_quality = {
                "passed": (
                    bool(query_result.quality_preflight.get("passed", False))
                    and all(
                        check.get("passed", False)
                        for check in transition_checks
                    )
                    and bool(response_quality.get("passed", False))
                ),
                "query_preflight": dict(query_result.quality_preflight),
                "transition_checks": transition_checks,
                "final_response_grounding": response_quality,
            }

            # Collect tools and categories
            for step in trajectory:
                for tc in step.tool_calls:
                    tools_used.add(tc.tool_name)
                    cat = self.tool_manager.get_tool_category(tc.tool_name)
                    if cat:
                        categories_used.add(cat)

            turn = Turn(
                turn_number=turn_idx + 1,
                user_query=query_result.query,
                query_intent=query_result.intent,
                steps=trajectory,
                assistant_response=assistant_response,
                expected_tools=query_result.expected_tools,
                execution_context=dict(execution_context),
                quality_verification=turn_quality,
            )
            conversation.turns.append(turn)
            self._remember_partial_candidate(
                blueprint=blueprint,
                conversation=conversation,
                initial_api_state=initial_api_state,
                focus_category=focus_category,
                execution_context=execution_context,
            )

            print(f"\n✓ Turn {turn_idx + 1} complete")
            print(f"   Query: {query_result.query[:80]}...")
            print(f"   Steps: {len(trajectory)}")

            # Save checkpoint after each turn
            if checkpoint_callback:
                checkpoint_state = {
                    'blueprint': {
                        'overall_task': blueprint.overall_task,
                        'num_turns': blueprint.num_turns,
                        'turns': blueprint.turns,
                    },
                    'partial_conversation': conversation.model_dump(),
                    'execution_context': dict(execution_context),
                    'completed_turns': turn_idx + 1,
                    'initial_api_state': initial_api_state,
                    'focus_category': focus_category,
                    'generation_directive': copy.deepcopy(
                        self._active_generation_directive
                    ),
                }
                checkpoint_callback(checkpoint_state)
                print(f"   Checkpoint saved after turn {turn_idx + 1}")

        # Generate and ground all ordinary turn-ending responses in one writer
        # call plus one judge call. This keeps the optimized request count
        # unchanged while avoiding placeholder targets.
        if self.optimized_pipeline:
            if not self._finalize_deferred_turn_responses(conversation):
                print("✗ Conversation failed batched turn-response grounding")
                self._remember_partial_candidate(
                    blueprint=blueprint,
                    conversation=conversation,
                    initial_api_state=initial_api_state,
                    focus_category=focus_category,
                    execution_context=execution_context,
                )
                self._mark_failure(
                    code="BATCHED_TURN_RESPONSE_GROUNDING_FAILED",
                    stage="turn_responses",
                )
                return None
            self._update_token_usage()

        # Stage 3: Assemble final datapoint
        conversation.tools_used = sorted(tools_used)
        conversation.categories_used = sorted(categories_used)
        conversation.initial_api_state = filter_api_state(initial_api_state, list(tools_used)) if initial_api_state else None

        quality_checks = [turn.quality_verification for turn in conversation.turns]
        quality_passed = all(
            check.get("passed", False) for check in quality_checks
        )
        if not quality_passed:
            print("✗ Conversation failed the positive-RL quality gate")
            self._remember_partial_candidate(
                blueprint=blueprint,
                conversation=conversation,
                initial_api_state=initial_api_state,
                focus_category=focus_category,
                execution_context=execution_context,
            )
            self._mark_failure(
                code="POSITIVE_RL_QUALITY_GATE_FAILED",
                stage="episode_quality_gate",
                details={"turn_checks": copy.deepcopy(quality_checks)},
            )
            return None

        available_tools = self._get_policy_tool_schemas(focus_category)
        datapoint = MultiTurnDatapoint(
            conversation=conversation,
            generation_metadata={
                "num_turns": self.num_turns,
                "actions_per_turn": self.num_actions,
                "configured_actions_per_turn": self.num_actions,
                "actual_steps_per_turn": [
                    len(turn.steps) for turn in conversation.turns
                ],
                "actual_tool_calls_per_turn": [
                    sum(len(step.tool_calls) for step in turn.steps)
                    for turn in conversation.turns
                ],
                "blueprint_actions_per_turn": (
                    list(self.blueprint_actions_per_turn)
                    if self.blueprint_actions_per_turn is not None
                    else None
                ),
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "blueprint_queries": [t.get("user_query", "") for t in blueprint.turns],
                "turn_expected_tools": [t.get("expected_tools", []) for t in blueprint.turns],
                "rl_quality_gate_passed": True,
                "model_routing": self._model_routing_metadata(),
                "tool_contract_hash": self._tool_contract_hash(available_tools),
                "generation_pipeline": (
                    "turn_compiler_v1_batched_turn_responses"
                    if self.optimized_pipeline
                    else "legacy_per_tool"
                ),
                "turn_response_policy": (
                    "batched_grounded_per_turn"
                    if self.optimized_pipeline
                    else "per_turn_generation"
                ),
                "generation_directive": copy.deepcopy(
                    self._active_generation_directive
                ),
                "llm_budget": {
                    "max_calls_per_candidate": self.max_calls_per_candidate,
                    "max_tokens_per_candidate": self.max_tokens_per_candidate,
                    "max_turn_attempts": self.max_turn_attempts,
                },
            },
            verification_result={
                "overall_verification_passed": True,
                "rl_quality_gate": {
                    "passed": True,
                    "turn_checks": quality_checks,
                },
            },
            token_usage=self._get_token_stats(),
            initial_api_state=conversation.initial_api_state,
            available_tools=available_tools,
        )

        print("\n" + "=" * 70)
        print("✓ MULTI-TURN DATAPOINT GENERATION COMPLETE")
        print("=" * 70)
        print(f" Turns: {len(conversation.turns)}")
        print(f" Tools used: {conversation.tools_used}")
        print(f" Total tool calls: {sum(len(t.steps) for t in conversation.turns)}")

        self.last_failure = None

        return datapoint

    def _get_tool_output_fields(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Extract output field names by calling each Python tool with valid minimal inputs.

        Returns dict mapping tool_api_name -> list of output field names.
        Uses read-only calls to avoid state mutation.
        """
        if not self._python_tools_available:
            return {}

        result: Dict[str, List[str]] = {}

        # Build api_name -> class_key mapping filtered by category
        api_names = []
        for api_name, class_key in self.tool_manager.api_name_to_class_key.items():
            if category:
                tool_cat = self.tool_manager.get_tool_category(api_name)
                if tool_cat != category:
                    continue
            api_names.append((api_name, class_key))

        for api_name, class_key in api_names:
            instance = self.tool_manager.python_tool_instances.get(class_key)
            if not instance:
                continue
            method = getattr(instance, api_name, None)
            if not method or not callable(method):
                continue

            try:
                import inspect
                sig = inspect.signature(method)
                bound = []
                for pname, param in sig.parameters.items():
                    if pname == 'self':
                        continue
                    if param.annotation in (int, float) and param.default is inspect.Parameter.empty:
                        bound.append(1)
                    elif param.annotation == str and param.default is inspect.Parameter.empty:
                        if 'city' in pname.lower() or 'location' in pname.lower():
                            bound.append('New York')
                        elif 'date' in pname.lower():
                            bound.append('2025-03-15')
                        elif 'token' in pname.lower():
                            bound.append('DUMMY_TOKEN')
                        elif 'card' in pname.lower() or 'number' in pname.lower() or 'id' in pname.lower():
                            bound.append('12345')
                        elif 'message' in pname.lower() or 'name' in pname.lower():
                            bound.append('Test')
                        elif 'currency' in pname.lower():
                            bound.append('USD')
                        elif 'type' in pname.lower():
                            bound.append('basic')
                        elif 'cost' in pname.lower() or 'balance' in pname.lower() or 'value' in pname.lower() or 'limit' in pname.lower():
                            bound.append(100.0)
                        else:
                            bound.append('x')
                    elif param.annotation == bool:
                        bound.append(True)

                out = method(*bound)
                if isinstance(out, dict):
                    result[api_name] = sorted(out.keys())
                else:
                    result[api_name] = []
            except Exception:
                result[api_name] = ['success', 'message', 'id', 'result', 'error']

        return result

    def _validate_posting_api_entities(
        self,
        turns: List[Dict[str, Any]],
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Deterministically validate that entity references in PostingApi queries exist in state.

        This catches cases where the blueprint references usernames (like 'techguru')
        that don't exist in the API state, which the LLM judge might miss.

        Returns list of error messages (empty if all entities are valid).
        """
        import re

        issues = []
        if not initial_api_state:
            return issues

        # Find PostingAPI state
        posting_state = initial_api_state.get("posting_api") or initial_api_state.get("PostingAPI")
        if not posting_state:
            return issues

        # Collect valid usernames from state
        valid_usernames = set()

        # From 'users' dict keys
        if isinstance(posting_state.get("users"), dict):
            valid_usernames.update(posting_state["users"].keys())

        # From 'following_list'
        if isinstance(posting_state.get("following_list"), list):
            valid_usernames.update(posting_state["following_list"])

        # From tweets - usernames who have posted
        if isinstance(posting_state.get("tweets"), dict):
            for tweet in posting_state["tweets"].values():
                if isinstance(tweet, dict) and tweet.get("username"):
                    valid_usernames.add(tweet["username"])

        # The logged-in user's own username
        if posting_state.get("username"):
            valid_usernames.add(posting_state["username"])

        # Also add from comments
        if isinstance(posting_state.get("comments"), dict):
            for comment_list in posting_state["comments"].values():
                if isinstance(comment_list, list):
                    for comment in comment_list:
                        if isinstance(comment, dict) and comment.get("username"):
                            valid_usernames.add(comment["username"])

        # From retweets
        if isinstance(posting_state.get("retweets"), list):
            for retweet in posting_state["retweets"]:
                if isinstance(retweet, dict) and retweet.get("username"):
                    valid_usernames.add(retweet["username"])

        # Username extraction patterns from query text
        # More specific patterns to avoid false positives like "user stats" or "user and"
        username_patterns = [
            r'from\s+user\s+(\w+)',    # "from user techguru"
            r'follow\s+(\w+)',         # "follow techguru"
            r'following\s+(\w+)',      # "following techguru"
            r'tweets?\s+from\s+(\w+)', # "tweets from techguru"
            r'by\s+user\s+(\w+)',      # "by user techguru"
            r'username\s+(\w+)',       # "username techguru"
            r'@(\w+)',                 # "@techguru" mention
        ]
        # Natural follow-up phrasing such as "retweet it, then check whether
        # I'm following them" must not turn a pronoun into a state-backed
        # username requirement.  Keep rejecting concrete unknown handles.
        non_username_references = {
            'him', 'her', 'them', 'it', 'me', 'us', 'you',
            'this', 'that', 'these', 'those', 'the', 'a', 'an',
        }

        # Check each turn's query for entity references
        for turn_idx, turn in enumerate(turns, 1):
            query = turn.get("user_query", "")
            expected_tools = turn.get("expected_tools", [])

            # Only validate for PostingApi tools
            posting_tools = {
                'get_user_tweets', 'get_user_stats', 'follow_user', 'unfollow_user',
                'authenticate_twitter', 'mention', 'comment', 'retweet',
                'search_tweets', 'get_tweet', 'get_tweet_comments'
            }
            if not any(t for t in expected_tools if t in posting_tools):
                continue

            # Extract potential usernames from query
            found_usernames = set()
            for pattern in username_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                found_usernames.update(
                    match
                    for match in matches
                    if match.casefold() not in non_username_references
                )

            # Check each found username against valid usernames
            for username in found_usernames:
                if username.lower() not in {u.lower() for u in valid_usernames}:
                    issues.append(
                        f"Turn {turn_idx}: query references username '{username}' but "
                        f"'{username}' does not exist in state. "
                        f"Valid users: {', '.join(sorted(valid_usernames))[:100]}"
                    )

        return issues

    def _validate_vehicle_control_queries(
        self,
        turns: List[Dict[str, Any]],
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Validate that fuel-related queries are consistent with initial vehicle state.

        Checks queries like 'fill up the tank' or 'add fuel' against initial fuelLevel
        to ensure the scenario is coherent (e.g., not asking to fill when already full).

        Returns list of error messages (empty if all queries are valid).
        """
        import re

        issues = []
        if not initial_api_state:
            return issues

        # Find VehicleControlAPI state
        vehicle_state = initial_api_state.get("vehicle_control") or initial_api_state.get("VehicleControlAPI")
        if not vehicle_state:
            return issues

        initial_fuel = vehicle_state.get("fuelLevel")
        if initial_fuel is None:
            return issues

        # Fuel fill patterns that indicate user wants to add fuel
        fuel_fill_patterns = [
            r'fill.*tank',
            r'add.*fuel',
            r'top.*up',
            r'refuel',
            r'fuel.*fill',
        ]

        for turn_idx, turn in enumerate(turns, 1):
            query = turn.get("user_query", "")
            expected_tools = turn.get("expected_tools", [])

            # Only validate for vehicle control tools that add fuel
            fuel_tools = {'fillFuelTank', 'addFuel'}
            if not any(t for t in expected_tools if t in fuel_tools):
                continue

            # Check if query indicates intent to add fuel
            for pattern in fuel_fill_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    # Found a fuel fill request - check if tank is already full
                    if initial_fuel >= 50.0:
                        issues.append(
                            f"Turn {turn_idx}: query asks to 'fill/add fuel' but initial fuelLevel "
                            f"is {initial_fuel} (tank is at or above max capacity of 50.0). "
                            f"This scenario is incoherent - tank cannot be filled when full. "
                            f"Use a config with fuelLevel < 50 or change the query to match state."
                        )
                    break

        return issues

    def _verify_blueprint_capabilities(
        self,
        turns: List[Dict[str, Any]],
        focus_category: Optional[str] = None,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[bool, List[str]]:
        """Verify that each turn's query intent matches tool capabilities.

        Uses LLM-as-judge to check if the user query can actually be fulfilled
        by the selected tool given its description and the current API state.

        Returns (is_valid, error_list).
        """
        if not turns:
            return False, ["No turns provided"]

        # Build tool capability context
        tool_caps: List[Dict[str, Any]] = []
        seen_tool_names: set[str] = set()
        for turn in turns:
            for tool_name in turn.get("expected_tools", []):
                if tool_name not in seen_tool_names:
                    try:
                        schema = self.tool_manager.get_tool_schema(tool_name)
                        tool_caps.append(
                            {
                                "name": tool_name,
                                "description": schema.get(
                                    "description", "No description"
                                )[:500],
                                "parameters": schema.get("parameters", {}),
                                "output_schema": schema.get(
                                    "output_schema", {}
                                ),
                            }
                        )
                    except ValueError:
                        tool_caps.append(
                            {
                                "name": tool_name,
                                "description": "Tool description unavailable",
                                "parameters": {},
                                "output_schema": {},
                            }
                        )
                    seen_tool_names.add(tool_name)

        tool_cap_str = json.dumps(
            tool_caps,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        # Build state summary with actual entity values for verification
        state_summary = ""
        if initial_api_state:
            for class_key, state in initial_api_state.items():
                if isinstance(state, dict):
                    # Show full state with actual values, not just keys
                    # This allows the judge to verify entity references exist
                    state_json = json.dumps(state, indent=2, default=str)
                    state_summary += f"\n{class_key}:\n{state_json[:5000]}"

        prompt = f"""You are verifying that a dialog blueprint's user queries can be fulfilled by the selected tools.

Check each turn: does the user_query intent match what the selected tool can actually do?

=== TOOL CAPABILITIES ===
{tool_cap_str}

=== GENERATOR-ONLY CURRENT API STATE ===
This shows which entities exist so you can reject impossible requests, but it
is NOT visible to the assistant solving the task. A state value may satisfy a
tool parameter only when the user_query states it naturally or an earlier
selected tool returns it.
{state_summary if state_summary else "No specific state provided."}

=== BLUEPRINT TURNS ===
{json.dumps(turns, indent=2, default=str)}

=== VERIFICATION TASK ===
For each turn, verify:
1. Does the user_query ask for something the selected tool can actually do?
2. Does the query phrasing match tool capabilities? (e.g., "search all files" can't be done by a single-file grep)
3. For PostingApi queries, every mentioned username must exist in the state's
   users, authored tweets, or following list.
4. Reject requests outside a parameter's declared enum or the tool's actual
   capability. For example, displayCarStatus cannot report an unsupported
   option merely because it is related to vehicle state.
5. Are entity names (files, IDs, etc.) consistent with the API state?
6. If multiple tools are listed, is that realistic for one turn?
7. Treat each output_schema as authoritative. A single object is not a list:
   reject selection, counting, aggregation, or references to multiple returned
   candidates unless an output field is explicitly an array containing them.
8. Every fact requested in the answer must be present in a selected tool's
   output or be a direct report of a successful state change.
9. PARAMETER-SOURCE CLOSURE (CRITICAL): inspect the parameter schema of every
   selected tool, not just the overall intent. Every required parameter must
   have a unique policy-visible source: the current/prior user utterances, a
   field returned by an earlier selected tool, or a declared schema default.
   Reject the whole blueprint if even one required parameter would have to be
   guessed. Explicitly check commonly omitted operational details including
   units and conversion direction, rounding precision, log base, exponent,
   percentage denominator, range bounds, dates/times, sort direction,
   filenames, credentials, usernames/recipients, and free-form content.
10. Same-turn output dependencies must respect expected_tools order; a tool may
   consume only output from an earlier tool. Cross-turn references must be
   unambiguous from prior policy-visible results.

IMPORTANT: For PostingApi, inspect users, tweet authors, and following_list;
reject a username absent from all three. Also reject requests outside tool enums
or capabilities even when the prose description mentions a broader category.

Respond ONLY with valid JSON:
{{"is_valid": true/false, "issues": ["Turn N: issue description", ...]}}

If ALL turns are achievable with their selected tools, set is_valid to true with empty issues."""


        try:
            response = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.judge,
                purpose="blueprint_semantic_judge",
            )
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]

            result = json.loads(response_text)
            is_valid = result.get("is_valid", False)
            issues = result.get("issues", [])

            return is_valid, issues

        except Exception as e:
            # On error, be permissive and let execution handle it
            return True, [f"Capability check error (allowing): {str(e)[:100]}"]

    # ─────────────────────── Stage 0: Blueprint ───────────────────────

    def _stage0_generate_blueprint(
            self, focus_category: Optional[str] = None, initial_api_state: Optional[Dict[str, Any]] = None
    ) -> Optional[DialogBlueprint]:
        """Generate a highly specific dialog blueprint with concrete entities and full user queries."""
        tools_json = self.tool_manager.get_tools_json_schema()
        directive = copy.deepcopy(self._active_generation_directive or {})
        allowed_tools = set(directive.get("allowed_tools", []))
        if allowed_tools:
            tools_json = [
                tool for tool in tools_json
                if str(tool.get("name", "")) in allowed_tools
            ]
        elif focus_category:
            tools_json = [t for t in tools_json if t.get('category') == focus_category]
        # Blueprint generation needs complete parameter and output schemas, not
        # compact name/description summaries, so it can plan every required
        # argument and dependency before the trace is executed.
        tools_str = json.dumps(
            tools_json,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        # Build output fields dynamically from tool definitions for the prompt
        output_fields_str = ""
        for tool in tools_json:
            name = tool.get('name', tool.get('api_name', ''))
            schema = tool.get('output_schema', {})
            props = schema.get('properties', {}) if schema else {}
            if props:
                fields = ', '.join(props.keys())
                output_fields_str += f"- {name}: {fields}\n"
            else:
                output_fields_str += f"- {name}\n"

        # Build output_fields_map dynamically for placeholder validation
        output_fields_validation_map: Dict[str, List[str]] = {}
        for tool in tools_json:
            name = tool.get('name', tool.get('api_name', ''))
            if name:
                # Collect all known fields from tool definition's output schema if available,
                # otherwise use a generic fallback
                props = tool.get('output_schema', {}).get('properties', {}) if isinstance(tool, dict) else {}
                if props:
                    output_fields_validation_map[name] = list(props.keys())
                else:
                    output_fields_validation_map[name] = ['success', 'message', 'id', 'result', 'error']

        # Build the class-key set represented by the actual policy tool context.
        focus_class_keys = set()
        if allowed_tools:
            focus_class_keys = {
                self.tool_manager.api_name_to_class_key.get(name)
                for name in allowed_tools
            }
            focus_class_keys.discard(None)
        elif focus_category:
            for api_name, class_key in self.tool_manager.api_name_to_class_key.items():
                tool_cat = self.tool_manager.get_tool_category(api_name)
                if tool_cat == focus_category:
                    focus_class_keys.add(class_key)

        # Inject actual credentials from initial_api_state into the prompt
        credential_context = ""
        initial_state_context = ""
        if initial_api_state:
            for class_key, state in initial_api_state.items():
                # Skip APIs not in the focus category to avoid credential_context pollution
                if focus_class_keys and class_key not in focus_class_keys:
                    continue
                if not isinstance(state, dict):
                    continue
                    
                # Include full state structure for reference (filtered to focus category)
                state_summary = json.dumps(state, indent=2, default=str)
                initial_state_context += f"\n{class_key}:\n{state_summary}"
                
                # Credentials
                if 'client_id' in state and 'client_secret' in state and 'refresh_token' in state:
                    cid = state['client_id']
                    csec = state['client_secret']
                    rtok = state['refresh_token']
                    credential_context += f"\nCredential format: {cid}/{csec}/{rtok}"
                # Card IDs
                if 'credit_card_list' in state and isinstance(state['credit_card_list'], dict):
                    card_ids = list(state['credit_card_list'].keys())
                    if card_ids:
                        credential_context += f"\nAvailable card IDs: {', '.join(card_ids)}"
                # User list (messaging)
                if 'user_map' in state and isinstance(state['user_map'], dict):
                    user_ids = list(state['user_map'].keys())
                    if user_ids:
                        credential_context += f"\nAvailable user IDs: {', '.join(user_ids[:10])}"
                # Account balance
                if 'account_type' in state and 'balance' in state:
                    credential_context += f"\nAccount balance: {state.get('balance')}"
                # Username/password credentials
                if 'username' in state and 'password' in state:
                    credential_context += f"\nCredentials: {state['username']}/{state['password']}"

        max_tools_per_turn = self.blueprint_max_actions_per_turn
        exact_action_schedule = self.blueprint_actions_per_turn
        if exact_action_schedule is None:
            action_schedule_requirement = (
                f"Use 1-{max_tools_per_turn} expected tools per turn, varying "
                "the count naturally with the request."
            )
        else:
            action_schedule_requirement = (
                "Use this exact expected-tools count by turn: "
                + ", ".join(
                    f"turn {index + 1}={count}"
                    for index, count in enumerate(exact_action_schedule)
                )
                + "."
            )
        directive_section = ""
        if directive:
            hard_required_tools = directive.get('hard_required_tools', [])
            directive_section = f"""
=== EVOLUTIONARY CURRICULUM DIRECTIVE ===
This directive shapes the attempt but is NOT an acceptance test for complexity.
Never add filler calls. A correct simpler trajectory is allowed and will still
be saved.

Coverage targets: {json.dumps(directive.get('target_tools', []), ensure_ascii=False)}
Target categories: {json.dumps(directive.get('target_categories', []), ensure_ascii=False)}
Preferred structural motif: {directive.get('motif', 'none')}
Scenario framing: {directive.get('scenario_seed', '')}
Writing style: {directive.get('style_seed', '')}
Soft requirements:
{json.dumps(directive.get('soft_requirements', []), ensure_ascii=False, indent=2)}
Previously evolved lessons relevant to these tools:
{json.dumps(directive.get('lesson_texts', []), ensure_ascii=False, indent=2)}

Build the task around the coverage targets when this is semantically natural.
If a target cannot be used coherently, produce a correct task from the supplied
tools instead of forcing it into the trajectory.

=== HARD REQUIRED TOOL COVERAGE ===
{json.dumps(hard_required_tools, ensure_ascii=False)}
Every tool in this list MUST occur at least once in expected_tools across the
conversation. Unlike the soft coverage targets above, omission rejects the
blueprint. Integrate them into one coherent user goal; never add a filler call.
"""

        prompt = f"""Design a {self.num_turns}-turn user-agent conversation. Each turn: USER request → AGENT calls 1-{max_tools_per_turn} tools → AGENT responds.

=== AVAILABLE TOOLS ===
{tools_str}
{directive_section}

=== OUTPUT SCHEMAS (use these exact field names in placeholders) ===
{output_fields_str}

=== REQUIREMENTS ===
 1. Each turn: specific entities (IDs, names, dates, prices) + the required
    number of tools. {action_schedule_requirement}
 2. Conversation flows naturally, each turn builds on previous
 3. Auth persists across turns - login only in FIRST turn needing auth (don't re-login)
 4. expected_tools must obey the per-turn count requirement exactly.
 5. POLICY-CONTEXT CLOSURE: Every required argument for every expected tool must
    be available from the current/prior user queries, an earlier tool output, or
    a default declared in the tool schema.
 6. The Initial API State below is generator-only and will NOT be shown to the
    assistant that solves the task. Use it to choose valid existing values, but
    write every required state-derived value explicitly into the appropriate
    user_query unless an earlier tool call returns it.
 7. If a tool needs a value produced by another tool, include the producing tool
    first and use the exact output field. Never require the assistant to guess.
 8. Cross-turn refs: when exactly one prior tool result identifies the object,
    prefer a natural unambiguous reference such as "that ticket" or "the post
    you just created"; the assistant can recover its machine ID from the prior
    policy-visible output. When multiple same-tool results make that ambiguous,
    use the exact output field with a zero-based occurrence index, such as
    {{{{TURN1.tool_name[1].field_name}}}}.
 9. Match the exact semantic representation required by each tool. A
    human-readable label is not interchangeable with an opaque identifier, code,
    token, symbol, handle, coordinate, path, or credential.
10. General/model knowledge is not an argument source. Opaque values must be
    written in a user_query or returned by an earlier tool call.
11. Verify that each user_query plus the tool schemas and prior tool outputs is
    sufficient to determine all arguments for that turn's expected_tools.
12. State progress: mutating turns must request a change that is not already
    satisfied in the generator-only state or by an earlier turn.
13. Unique creation: create calls must add new entities and must not reuse an
    identifier already present in state.
14. If a tool returns multiple candidates and a later call consumes one, the
    relevant user_query must state a deterministic selection rule.
15. The current UTC date is {datetime.now(timezone.utc).date().isoformat()}.
    Booking and scheduling dates must not be earlier than this date.
16. Keep account, traveler, owner, cardholder, and credential identities
    coherent unless explicit authorization is stated in a user_query.
17. Every result requested in an assistant response must be directly supported
    by a called tool output; do not require uncalled calculations or lookups.
18. If multiple calls in a turn are independent, explicitly state their order in
    that turn's user_query so an ordered gold trajectory has a unique next call.
19. PLAN FROM A NATURAL USER GOAL: Prefer a human-facing name, description, or
    relation plus an earlier discovery/list/search call over asking the user for
    an opaque internal ID. The assistant should infer or retrieve machine-facing
    identifiers itself whenever the available tools make that possible.
20. An opaque value may be placed directly in user_query only when a normal user
    plausibly knows it (for example a tracking number, ticket number, username,
    filename, confirmation code, or credential) or no available tool can obtain
    it. Never phrase a request as "get the ID so you can call ...".
21. Write each turn as a realistic end-user utterance with one coherent purpose,
    not a JSON argument dump, API recipe, numbered checklist, or description of
    which function to invoke. Encode necessary sequential order through natural
    dependencies ("once that is booked...", "then send the confirmation").
22. Across the conversation, vary simple follow-ups with denser turns that require
    the assistant to combine prior results, resolve references, and choose the
    appropriate tools from intent. The user supplies goals and constraints, not
    an implementation plan.
23. Treat output_schema shape as authoritative even if prose is imprecise. A
    tool returning one object cannot support "the first two", counting,
    aggregation, or multi-candidate selection unless an explicit array field
    contains those candidates.
24. Before returning, privately audit every required parameter of every
    expected tool. For each one, identify its exact source: a value stated
    naturally in the current/prior user utterance, a named field from an
    earlier expected tool output, or a schema default. If no such source
    exists, rewrite the user utterance. Do not silently invent operational
    details such as input/output units, rounding precision, logarithm base,
    exponent, percentage denominator, date/time, range bound, sort direction,
    filename, username, recipient, or message text.
25. Every expected tool must be necessary for the user's stated goal. Do not add
    an unrelated prerequisite or state-changing action merely because it is
    available; the request must support that action naturally.
26. When one invocation accepts only one value for an option, either ask for one
    value or include one expected-tool occurrence per requested value. Do not ask
    a single call to return multiple independently selected options.

=== EXAMPLES ===
- "I’m signed in as user123; please file the network problem we discussed as high priority and let me know the ticket number." (login_action, create_item)
- "Could you let Sarah know I’ll be about ten minutes late?" (get_user_id, send_message)
- "Post 'Great day for AI!' to my feed once you’ve connected my account." (authenticate_twitter, post_tweet)
- "Book the option we found in the previous turn, then send the confirmation to the traveler." (book_flight, send_message)

=== OUTPUT ===
{{"overall_task": "scenario", "turns": [{{"user_query": "request", "expected_tools": ["t1", "t2"]}}, ...]}}"""

        if focus_category and not allowed_tools:
            prompt += f"\n\nAll available tools below are from the '{focus_category}' category."

            domain_hints = get_domain_hints(focus_category)
            if domain_hints:
                prompt += f"\n\n{domain_hints}"

        if initial_state_context:
            prompt += (
                "\n\n=== GENERATOR-ONLY INITIAL API STATE ==="
                "\nThis state is not policy-visible. Any required value selected "
                "from it must be written into a user_query unless a prior tool "
                "returns it."
                f"{initial_state_context}"
            )
        
        if credential_context:
            prompt += (
                "\n\n=== GENERATOR-ONLY CREDENTIAL VALUES ==="
                "\nIf a credential is required and no tool returns it, include "
                "the exact credential naturally in the relevant user_query."
                f"{credential_context}"
            )

        accumulated_feedback = ""
        max_blueprint_attempts = max(
            1,
            min(
                2,
                int(os.getenv("APIGEN_MAX_BLUEPRINT_ATTEMPTS", "2")),
            ),
        )
        for attempt in range(max_blueprint_attempts):
            try:
                if accumulated_feedback:
                    prompt_with_feedback = prompt + f"\n\n=== PREVIOUS ATTEMPT FEEDBACK ===\n{accumulated_feedback}\n=== END FEEDBACK ===\n"
                else:
                    prompt_with_feedback = prompt

                response = self._safe_llm_generate(
                    [{"role": "user", "content": prompt_with_feedback}],
                    purpose="blueprint_generate",
                )
                response_text = response.strip()

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

                result = json.loads(response_text)
                turns = result.get("turns", [])
                if not turns or len(turns) != self.num_turns:
                    accumulated_feedback = f"Expected {self.num_turns} turns, got {len(turns)}. Please generate exactly {self.num_turns} turns."
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                validation_errors = []
                all_tools_valid = True
                for i, t in enumerate(turns):
                    expected = t.get("expected_tools", [])
                    required_count = (
                        exact_action_schedule[i]
                        if exact_action_schedule is not None
                        else None
                    )
                    count_is_valid = (
                        len(expected) == required_count
                        if required_count is not None
                        else 1 <= len(expected) <= max_tools_per_turn
                    )
                    if not count_is_valid:
                        needed = (
                            str(required_count)
                            if required_count is not None
                            else f"1-{max_tools_per_turn}"
                        )
                        validation_errors.append(
                            f"Turn {i+1} has {len(expected)} tools, need "
                            f"{needed}: {expected}"
                        )
                        all_tools_valid = False
                        break

                    # Validate against the actual policy-visible tool context.
                    if allowed_tools:
                        for tool_name in expected:
                            if tool_name not in allowed_tools:
                                validation_errors.append(
                                    f"Turn {i+1} tool '{tool_name}' is not in the "
                                    "scheduled policy tool context."
                                )
                                all_tools_valid = False
                                break
                        if not all_tools_valid:
                            break
                    elif focus_category:
                        for tool_name in expected:
                            tool_cat = self.tool_manager.get_tool_category(tool_name)
                            if tool_cat != focus_category:
                                validation_errors.append(f"Turn {i+1} tool '{tool_name}' is from category '{tool_cat}', not '{focus_category}'. Use only {focus_category} tools.")
                                all_tools_valid = False
                                break
                        if not all_tools_valid:
                            break

                    # Validate placeholder references in user_query
                    query = t.get("user_query", "")
                    import re
                    placeholders = re.findall(
                        r'\{\{TURN(\d+)\.(\w+)(?:\[(\d+)\])?\.(\w+)\}\}',
                        query,
                    )
                    for p in placeholders:
                        ref_turn_idx = int(p[0]) - 1
                        ref_tool = p[1]
                        occurrence_text = p[2]
                        ref_field = p[3]
                        if ref_turn_idx >= i:
                            validation_errors.append(f"Turn {i+1} placeholder references future turn {p[0]}")
                            all_tools_valid = False
                            break
                        if ref_turn_idx < len(turns):
                            ref_tools = turns[ref_turn_idx].get("expected_tools", [])
                            if ref_tool not in ref_tools:
                                validation_errors.append(f"Turn {i+1} references {ref_tool} from turn {p[0]}, but that turn uses {ref_tools}")
                                all_tools_valid = False
                                break
                            occurrences = ref_tools.count(ref_tool)
                            if not occurrence_text and occurrences > 1:
                                validation_errors.append(
                                    f"Turn {i+1} has an ambiguous reference to repeated tool "
                                    f"{ref_tool}; use an occurrence index such as [0]."
                                )
                                all_tools_valid = False
                                break
                            if occurrence_text and int(occurrence_text) >= occurrences:
                                validation_errors.append(
                                    f"Turn {i+1} references occurrence [{occurrence_text}] of "
                                    f"{ref_tool}, but only {occurrences} call(s) exist."
                                )
                                all_tools_valid = False
                                break
                            # Validate that the placeholder field exists in tool output using known schema
                            known_fields = output_fields_validation_map.get(ref_tool, ['success', 'message', 'id', 'result'])
                            if ref_field not in known_fields:
                                reference = (
                                    f"TURN{p[0]}.{ref_tool}"
                                    + (f"[{occurrence_text}]" if occurrence_text else "")
                                    + f".{ref_field}"
                                )
                                validation_errors.append(f"Turn {i+1} placeholder {{{{{reference}}}}}: '{ref_field}' not in {ref_tool} output. Use: {known_fields}")
                                all_tools_valid = False
                                break
                    if not all_tools_valid:
                        break

                # Validate cross-turn entity references
                cross_turn_entity_tools = {
                    'comment': ('tweet_id', 'post_tweet'),
                    'retweet': ('tweet_id', 'post_tweet'),
                    'mention': ('tweet_id', 'post_tweet'),
                    'edit_ticket': ('ticket_id', 'create_ticket'),
                    'resolve_ticket': ('ticket_id', 'create_ticket'),
                    'close_ticket': ('ticket_id', 'create_ticket'),
                    'delete_message': ('message_id', 'send_message'),
                    'purchase_insurance': ('booking_id', 'book_flight'),
                }
                for i, t in enumerate(turns):
                    if i == 0:
                        continue
                    expected = t.get("expected_tools", [])
                    query = t.get("user_query", "")
                    for tool_name in expected:
                        if tool_name in cross_turn_entity_tools:
                            id_field, create_tool = cross_turn_entity_tools[tool_name]
                            if i > 0 and i - 1 < len(turns):
                                prior_tools = turns[i - 1].get("expected_tools", [])
                                if create_tool in prior_tools:
                                    create_count = prior_tools.count(create_tool)
                                    # A single producer is already an
                                    # unambiguous policy-visible argument
                                    # source.  Requiring its opaque ID to be
                                    # copied into the next user utterance made
                                    # conversations less natural and made the
                                    # downstream policy's task artificially
                                    # easy.  Multiple producer results still
                                    # need an indexed executable reference.
                                    if create_count > 1:
                                        valid_reference = re.search(
                                            rf'\{{\{{TURN{i}\.{re.escape(create_tool)}\[\d+\]\.{re.escape(id_field)}\}}\}}',
                                            query,
                                        ) is not None
                                        if not valid_reference:
                                            validation_errors.append(
                                                f"Turn {i+1} uses '{tool_name}' after multiple "
                                                f"'{create_tool}' results but lacks an indexed "
                                                f"placeholder for '{id_field}'."
                                            )
                                            all_tools_valid = False
                                            break
                    if not all_tools_valid:
                        break

                if not all_tools_valid:
                    accumulated_feedback = "\n".join(validation_errors) if validation_errors else "Validation failed. Please check tool categories and placeholders."
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                missing_required = self._missing_hard_required_tools(
                    turns,
                    directive,
                )
                if missing_required:
                    accumulated_feedback = (
                        "Hard required tools missing from expected_tools: "
                        + ", ".join(missing_required)
                        + ". Regenerate the same coherent task and include "
                        "every hard required tool."
                    )
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                all_tools_valid = all(
                    self.tool_manager.tool_exists(t)
                    for t_dict in turns
                    for t in t_dict.get("expected_tools", [])
                )
                if not all_tools_valid:
                    accumulated_feedback = "Some expected_tools are invalid. Please use only valid tool names from the provided list."
                    print(f"  ✗ {accumulated_feedback}")
                    continue

                # Verify tool capabilities match query intents
                print(f"  Verifying tool-query capability match...")
                cap_valid, cap_issues = self._verify_blueprint_capabilities(
                    turns, focus_category, initial_api_state
                )
                if not cap_valid:
                    cap_feedback = "\n".join(cap_issues) if cap_issues else "Tool capabilities don't match query intents"
                    accumulated_feedback = f"Capability mismatch:\n{cap_feedback}\n\nPlease regenerate with queries that match tool capabilities."
                    print(f"  ✗ {cap_feedback[:200]}...")
                    continue
                self._episode_query_quality = {
                    "passed": True,
                    "issue_codes": [],
                    "validator": "episode_blueprint_semantic_judge",
                    "episode_level": True,
                    "turn_count": len(turns),
                }

                # Deterministic entity validation for PostingApi
                entity_issues = self._validate_posting_api_entities(turns, initial_api_state)
                if entity_issues:
                    entity_feedback = "\n".join(entity_issues)
                    accumulated_feedback = f"Entity reference errors:\n{entity_feedback}\n\nPlease regenerate with valid entity references from the API state."
                    print(f"  ✗ {accumulated_feedback[:200]}...")
                    continue

                # Deterministic entity validation for VehicleControl
                vehicle_issues = self._validate_vehicle_control_queries(turns, initial_api_state)
                if vehicle_issues:
                    vehicle_feedback = "\n".join(vehicle_issues)
                    accumulated_feedback = f"Vehicle state errors:\n{vehicle_feedback}\n\nPlease regenerate with coherent vehicle state."
                    print(f"  ✗ {accumulated_feedback[:200]}...")
                    continue

                print(f" ✓ Blueprint generated: {result.get('overall_task', '')[:100]}")
                return DialogBlueprint(
                    overall_task=result.get("overall_task", ""),
                    num_turns=self.num_turns,
                    turns=turns,
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                accumulated_feedback = f"JSON parse error: {e}. Please return valid JSON."
                print(f"  ✗ Attempt {attempt + 1}: {e}")
                continue

        print(
            "  ✗ Failed to generate valid blueprint after "
            f"{max_blueprint_attempts} attempts"
        )
        return None

    # ─────────────────────── Turn query generation ───────────────────────

    def _repair_turn_query(
            self,
            *,
            user_query: str,
            expected_tools: List[str],
            policy_history: List[Dict[str, Any]],
            quality_feedback: str,
    ) -> Optional[str]:
        """Repair one synthetic utterance without replaying completed turns.

        Long conversations used to discard all completed work when a later
        blueprint utterance failed policy closure.  This bounded repair keeps
        the exact fixed tool plan and uses only prior policy-visible history.
        The caller always reruns the normal RL preflight afterward.
        """
        contracts = []
        for tool_name in expected_tools:
            schema = self.tool_manager.get_tool_schema(tool_name)
            if schema:
                contracts.append(schema)

        # Preserve concrete user-supplied literals that tend to identify the
        # intended state entity.  This guards against a style repair silently
        # changing filenames, dates, amounts, or confirmation-like values.
        import re
        protected = sorted(
            set(
                re.findall(
                    r"""(?x)
                    \b\d[\w:./-]*\b
                    |
                    \b[\w.-]+\.[A-Za-z0-9]{1,8}\b
                    """,
                    user_query,
                )
            ),
            key=lambda value: (-len(value), value),
        )
        visible_history = json.dumps(
            policy_history[-40:],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        if len(visible_history) > 16_000:
            visible_history = visible_history[-16_000:]

        prompt = f"""Repair one user utterance in a synthetic tool-use conversation.

=== CURRENT USER UTTERANCE ===
{user_query}

=== FIXED TOOL CONTRACTS AND ORDER ===
{json.dumps(contracts, ensure_ascii=False, indent=2, default=str)}

=== PRIOR POLICY-VISIBLE CONVERSATION ===
{visible_history}

=== QUALITY-GATE FEEDBACK ===
{quality_feedback}

Rewrite only the current USER utterance so that the fixed tools, in exactly the
listed order, are sufficient to fulfill everything it requests.

Rules:
1. Keep the same coherent user-facing goal and all concrete values unless the
   feedback says an unsupported extra result must be removed.
2. Use facts from PRIOR POLICY-VISIBLE CONVERSATION when relevant. Do not invent
   a hidden prior action, login, state value, lookup, or calculation.
3. Every required argument must be explicit in this utterance, available in
   prior visible history, produced by an earlier fixed call, or have a schema
   default.
4. Do not add work that the fixed calls cannot perform. Do not ask for results
   absent from their outputs.
5. Sound like a natural follow-up from a real user. Do not mention tools, APIs,
   schemas, argument names, benchmark rules, or an execution plan.
6. Preserve these concrete literals verbatim:
   {json.dumps(protected, ensure_ascii=False)}

Return only JSON: {{"query": "..."}}
"""
        try:
            response = self._safe_llm_generate(
                [{"role": "user", "content": prompt}]
            ).strip()
            if "```json" in response:
                response = response.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response:
                response = response.split("```", 1)[1].split("```", 1)[0]
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                response = response[start:end]
            repaired = str(json.loads(response).get("query", "")).strip()
        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as exc:
            print(f"  Turn-query repair failed: {exc}")
            return None
        if not repaired or repaired == user_query:
            return None
        missing = [token for token in protected if token not in repaired]
        if missing:
            print(
                "  Turn-query repair dropped protected literals: "
                + ", ".join(missing[:5])
            )
            return None
        return repaired

    def _generate_turn_query(
            self,
            blueprint: DialogBlueprint,
            conversation: MultiTurnConversation,
            turn_index: int,
    ) -> Optional[QueryGenerationResult]:
        """Use the blueprint's pre-written user query for this turn.

        The blueprint already includes a specific user_query for each turn
        with concrete entities (names, IDs, credentials). This avoids the
        inconsistency and extra LLM cost of per-turn query generation.

        Placeholders in the format {{TURN{N}.{tool_name}.{output_key}}} are
        resolved using the actual tool outputs from prior turns.
        """
        turn_spec = blueprint.turns[turn_index] if turn_index < len(blueprint.turns) else {}
        user_query = turn_spec.get("user_query", "")

        user_query = self._resolve_turn_placeholders(user_query, turn_index, conversation)

        expected_tools = turn_spec.get("expected_tools", [])

        max_tools_per_turn = self.blueprint_max_actions_per_turn
        if not user_query or not (
            1 <= len(expected_tools) <= max_tools_per_turn
        ):
            print(
                f"  ✗ Turn {turn_index + 1}: Blueprint has invalid query "
                f"({len(expected_tools)} tools, need 1-{max_tools_per_turn})"
            )
            return None

        invalid = [t for t in expected_tools if not self.tool_manager.tool_exists(t)]
        if invalid:
            print(f"  ✗ Turn {turn_index + 1}: Invalid tools in blueprint: {invalid}")
            return None

        query_result = QueryGenerationResult(
            query=user_query,
            intent="",
            expected_tools=expected_tools,
        )
        current_state = (
            self.tool_manager.get_api_state()
            if self._python_tools_available
            else None
        )
        policy_history = []
        for prior_turn in conversation.turns:
            policy_history.append({
                "role": "user",
                "content": prior_turn.user_query,
            })
            for step in prior_turn.steps:
                for call in step.tool_calls:
                    policy_history.append({
                        "role": "assistant_tool_call",
                        "name": call.tool_name,
                        "arguments": call.arguments,
                    })
                    policy_history.append({
                        "role": "tool",
                        "name": call.tool_name,
                        "content": call.output,
                    })
            if prior_turn.assistant_response:
                policy_history.append({
                    "role": "assistant",
                    "content": prior_turn.assistant_response,
                })

        if self.optimized_pipeline:
            prerequisite_rules = (
                (
                    {"add_contact", "send_message", "delete_message"},
                    "message_login",
                    "message_api",
                    "current_user",
                ),
                (
                    {
                        "follow_user", "unfollow_user", "comment", "retweet",
                        "mention", "post_tweet",
                    },
                    "authenticate_twitter",
                    "posting_api",
                    "authenticated",
                ),
                (
                    {
                        "create_ticket", "edit_ticket", "resolve_ticket",
                        "close_ticket",
                    },
                    "ticket_login",
                    "ticket_api",
                    "authenticated",
                ),
                (
                    {
                        "register_credit_card", "book_flight",
                        "cancel_booking", "purchase_insurance",
                    },
                    "authenticate_travel",
                    "travel_booking",
                    "access_token",
                ),
                (
                    {
                        "add_to_watchlist", "remove_stock_from_watchlist",
                        "place_order", "fund_account", "cancel_order",
                    },
                    "trading_login",
                    "trading_bot",
                    "authenticated",
                ),
            )
            prior_tools = {
                str(item.get("name", ""))
                for item in policy_history
                if item.get("role") == "assistant_tool_call"
            }
            issue_codes: List[str] = []
            if "{{" in user_query or "}}" in user_query:
                issue_codes.append("UNRESOLVED_CROSS_TURN_REFERENCE")
            for step_index, planned_tool in enumerate(expected_tools):
                for targets, prerequisite, class_key, state_field in (
                    prerequisite_rules
                ):
                    if planned_tool not in targets:
                        continue
                    state_ready = bool(
                        (current_state or {}).get(class_key, {}).get(state_field)
                    )
                    prerequisite_visible = (
                        prerequisite in expected_tools[:step_index]
                        or prerequisite in prior_tools
                    )
                    if not state_ready and not prerequisite_visible:
                        issue_codes.append("MISSING_PREREQUISITE")
            episode_quality = self._episode_query_quality or {
                "passed": True,
                "issue_codes": [],
                "validator": "episode_blueprint_semantic_judge",
                "episode_level": True,
            }
            query_result.quality_preflight = {
                "passed": bool(episode_quality.get("passed", False))
                and not issue_codes,
                "issue_codes": sorted(set(issue_codes)),
                "validator": "deterministic_turn_preflight",
                "episode_certificate": copy.deepcopy(episode_quality),
                "query_repaired": False,
                "query_repair_attempts": 0,
                "query_before_repair": None,
            }
            self._last_query_quality = dict(
                query_result.quality_preflight
            )
            if not query_result.quality_preflight["passed"]:
                print(
                    f"  ✗ Turn {turn_index + 1}: deterministic preflight "
                    + ", ".join(query_result.quality_preflight["issue_codes"])
                )
                return None
            print(f"  ✓ Using blueprint query for turn {turn_index + 1}")
            print(f"   Query: {user_query[:80]}...")
            print(f"   Tools: {expected_tools}")
            return query_result

        repaired_from: Optional[str] = None
        repair_attempts = 0
        for repair_attempt in range(3):
            quality_ok, quality_feedback = self.validate_expected_tools(
                user_query,
                expected_tools,
                "",
                initial_api_state=current_state,
                policy_history=policy_history,
            )
            if quality_ok:
                break
            print(
                f"  ✗ Turn {turn_index + 1}: RL quality preflight failed: "
                f"{quality_feedback}"
            )
            if repair_attempt >= 2:
                return None
            repaired = self._repair_turn_query(
                user_query=user_query,
                expected_tools=expected_tools,
                policy_history=policy_history,
                quality_feedback=quality_feedback,
            )
            if repaired is None:
                return None
            if repaired_from is None:
                repaired_from = user_query
            user_query = repaired
            repair_attempts += 1
            query_result.query = repaired
            # Persist the repaired utterance in generation metadata/checkpoints.
            turn_spec["user_query"] = repaired
            print(
                f"  ↻ Repaired turn {turn_index + 1} utterance and "
                "rerunning the RL preflight"
            )
        query_result.quality_preflight = dict(self._last_query_quality)
        query_result.quality_preflight.update(
            {
                "query_repaired": repaired_from is not None,
                "query_repair_attempts": repair_attempts,
                "query_before_repair": repaired_from,
            }
        )

        print(f"  ✓ Using blueprint query for turn {turn_index + 1}")
        print(f"   Query: {user_query[:80]}...")
        print(f"   Tools: {expected_tools}")
        return query_result

    @staticmethod
    def _aggregate_turn_outputs(trajectory: List[TrajectoryStep]) -> Dict[str, Any]:
        """Preserve every output when a turn repeats the same tool."""
        calls = []
        by_tool: Dict[str, List[Any]] = {}
        for step in trajectory:
            for call_index, tc in enumerate(step.tool_calls, 1):
                if tc.output is None:
                    continue
                call_id = f"s{step.step_number}_c{call_index}"
                calls.append({
                    "call_id": call_id,
                    "tool_name": tc.tool_name,
                    "arguments": copy.deepcopy(tc.arguments),
                    "output": tc.output,
                })
                by_tool.setdefault(tc.tool_name, []).append(tc.output)

        aggregate: Dict[str, Any] = {
            "calls": calls,
            "by_tool": by_tool,
        }
        # Backward-compatible direct lookup for existing TURN placeholders.
        for tool_name, outputs in by_tool.items():
            aggregate[tool_name] = outputs[0] if len(outputs) == 1 else outputs
        return aggregate

    def _resolve_turn_placeholders(
            self,
            query: str,
            turn_index: int,
            conversation: MultiTurnConversation,
    ) -> str:
        """Resolve {{TURN{N}.{tool_name}.{output_key}}} placeholders in a query.

        Looks up the actual output value from a prior turn's tool execution
        and substitutes it into the query.
        """
        import re
        pattern = re.compile(
            r"\{\{TURN(\d+)\.(\w+)(?:\[(\d+)\])?\.(\w+)\}\}"
        )

        def replacer(match):
            ref_turn = int(match.group(1))
            tool_name = match.group(2)
            occurrence_text = match.group(3)
            output_key = match.group(4)

            if ref_turn > turn_index:
                return match.group(0)

            ref_turn_idx = ref_turn - 1
            if ref_turn_idx >= len(conversation.turns):
                return match.group(0)

            prior_turn = conversation.turns[ref_turn_idx]
            matching_outputs = [
                tc.output
                for step in prior_turn.steps
                for tc in step.tool_calls
                if tc.tool_name == tool_name
            ]
            if not matching_outputs:
                return match.group(0)

            if occurrence_text is None:
                # An unindexed reference is valid only when the tool was called
                # exactly once; otherwise the dependency is ambiguous.
                if len(matching_outputs) != 1:
                    return match.group(0)
                output = matching_outputs[0]
            else:
                occurrence = int(occurrence_text)
                if occurrence >= len(matching_outputs):
                    return match.group(0)
                output = matching_outputs[occurrence]

            if isinstance(output, dict):
                if output_key in output:
                    return str(output[output_key])
                for key, value in output.items():
                    if output_key.lower() in key.lower() or key.lower() in output_key.lower():
                        return str(value)
                if len(output) == 1:
                    return str(next(iter(output.values())))
            return match.group(0)

        resolved = pattern.sub(replacer, query)
        if resolved != query:
            print(f"   Resolved placeholders: {query[:60]}... -> {resolved[:60]}...")
        return resolved

    # ─────────────────────── Helpers ───────────────────────

    @staticmethod
    def _validate_tool_arguments(trajectory: List[TrajectoryStep]) -> List[str]:
        """Check tool call arguments and outputs for hallucination indicators.

        Returns list of error strings (empty = valid).
        Hallucinated empty required args cause datapoint rejection + retry.
        """
        errors = []
        for step in trajectory:
            for tc in step.tool_calls:
                args = tc.arguments or {}
                out = tc.output or {}
                name = tc.tool_name

                if name == 'book_flight':
                    for field in ['travel_date', 'travel_to', 'travel_from']:
                        if not args.get(field) or str(args.get(field, '')).strip() == '':
                            errors.append(
                                f"book_flight: hallucinated empty '{field}' in arguments"
                            )
                    bh = out.get('booking_history', {})
                    if not bh.get('travel_date') or not bh.get('travel_to'):
                        errors.append(
                            f"book_flight: output booking_history missing travel_date/travel_to"
                        )
                    if not out.get('booking_id'):
                        errors.append(f"book_flight: empty booking_id in output")

                elif name == 'purchase_insurance':
                    if not args.get('booking_id'):
                        errors.append("purchase_insurance: empty booking_id in arguments")
                    ins_id = out.get('insurance_id', '')
                    ins_status = out.get('insurance_status')
                    if (ins_id == '' or ins_id is None) and ins_status is False:
                        errors.append(
                            f"purchase_insurance: failed (ins_id='{ins_id}', status={ins_status}), "
                            f"likely operating on cancelled booking"
                        )

                elif name == 'retrieve_invoice':
                    inv = out.get('invoice', {})
                    if isinstance(inv, dict) and len(inv) == 0:
                        errors.append("retrieve_invoice: empty invoice dict in output")

                elif name == 'cancel_booking':
                    if not out.get('cancel_status') and out.get('cancel_status') is not None:
                        errors.append(f"cancel_booking: cancel_status=False")

                elif name == 'authenticate_travel':
                    if not out.get('success') and out.get('success') is not None:
                        errors.append(
                            f"authenticate_travel: failed success={out.get('success')} "
                            f"error={out.get('error', '')[:60]}"
                        )

                elif name == 'get_flight_cost':
                    if out.get('error'):
                        errors.append(f"get_flight_cost: error={out['error'][:80]}")

                elif name in ('ls', 'cat', 'cd', 'mkdir', 'mv', 'rm', 'rmdir', 'touch', 'cp', 'grep', 'find', 'wc', 'tail', 'echo', 'du', 'sort'):
                    if 'calls' in args:
                        errors.append(f"{name}: LLM generated 'calls' batch format - use single tool call with direct arguments")

        return errors

    def _validate_cross_turn_consistency(
        self,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
    ) -> List[str]:
        """Validate that tool calls are consistent with prior turn outputs.

        Returns list of error strings (empty = valid).
        Cross-turn hallucination (e.g., book_flight with wrong route) causes rejection.
        """
        errors = []
        turn_outputs = execution_context.get('turn_outputs', [])

        current_tc_by_name = {}
        for step in trajectory:
            for tc in step.tool_calls:
                current_tc_by_name[tc.tool_name] = tc

        prior_tc_by_name = {}
        for turn_out in turn_outputs:
            for tool_name, output in turn_out.items():
                if tool_name not in prior_tc_by_name:
                    prior_tc_by_name[tool_name] = []
                prior_tc_by_name[tool_name].append(output)

        if 'book_flight' in current_tc_by_name:
            bf_args = current_tc_by_name['book_flight'].arguments or {}
            bf_out = current_tc_by_name['book_flight'].output or {}
            bf_from = bf_args.get('travel_from', '').upper()
            bf_to = bf_args.get('travel_to', '').upper()

            if 'get_flight_cost' in prior_tc_by_name:
                gfc_output = prior_tc_by_name['get_flight_cost'][-1]
                gfc_from = gfc_output.get('travel_from', '').upper()
                gfc_to = gfc_output.get('travel_to', '').upper()

                if gfc_from and gfc_to and (bf_from != gfc_from or bf_to != gfc_to):
                    errors.append(
                        f"book_flight: route mismatch. get_flight_cost used {gfc_from}→{gfc_to} "
                        f"but book_flight called with {bf_from}→{bf_to}"
                    )

            if 'get_nearest_airport_by_city' in prior_tc_by_name:
                airport_outputs = prior_tc_by_name['get_nearest_airport_by_city']
                prior_cities = set()
                prior_airports = set()
                for ao in airport_outputs:
                    nearest = ao.get('nearest_airport', '')
                    if nearest:
                        prior_airports.add(nearest.upper())

                if prior_airports and bf_from and bf_from.upper() not in prior_airports:
                    errors.append(
                        f"book_flight: travel_from='{bf_from}' not in prior airport lookups {prior_airports}"
                    )

        if 'purchase_insurance' in current_tc_by_name:
            pi_args = current_tc_by_name['purchase_insurance'].arguments or {}
            pi_out = current_tc_by_name['purchase_insurance'].output or {}
            pi_booking_id = pi_args.get('booking_id', '')

            if 'book_flight' in prior_tc_by_name:
                prior_booking_ids = set()
                for bo in prior_tc_by_name['book_flight']:
                    bid = bo.get('booking_id', '')
                    if bid:
                        prior_booking_ids.add(bid)

                if prior_booking_ids and pi_booking_id and pi_booking_id not in prior_booking_ids:
                    errors.append(
                        f"purchase_insurance: booking_id='{pi_booking_id}' not in prior bookings {prior_booking_ids}"
                    )

            if pi_out.get('insurance_status') is False:
                errors.append(
                    f"purchase_insurance: failed (booking_id='{pi_booking_id}', status=False)"
                )

        # Do not compare input sets of aggregate tools merely because they occur
        # in different turns. A user may validly add/remove observations before
        # asking for another statistic. The old unconditional equality heuristic
        # rejected those useful evolving-data trajectories even though it did not
        # receive the user query and therefore could not know whether "the same
        # values" were requested. Policy-visible argument provenance and the
        # episode semantic judge cover actual unsupported substitutions.

        return errors

    def _format_conversation_history(self, conversation: MultiTurnConversation) -> str:
        """Format completed turns as readable history for the LLM."""
        if not conversation.turns:
            return ""

        lines = []
        for turn in conversation.turns:
            lines.append(f"--- Turn {turn.turn_number} ---")
            lines.append(f"User: {turn.user_query}")
            for step in turn.steps:
                for tc in step.tool_calls:
                    output_preview = str(tc.output)[:100] if tc.output else ""
                    lines.append(f"  → {tc.tool_name}({json.dumps(tc.arguments, default=str)[:200]}) -> {output_preview}")
            if turn.assistant_response:
                lines.append(f"Assistant: {turn.assistant_response}")

        return "\n".join(lines)

    @staticmethod
    def _assign_tools_to_turns(blueprint: DialogBlueprint, all_tool_names: List[str]) -> Dict[int, List[str]]:
        """Distribute flat tool list across turns based on blueprint."""
        tools_by_turn: Dict[int, List[str]] = {}
        idx = 0
        for t_idx, t_spec in enumerate(blueprint.turns):
            turn_tools = t_spec.get("expected_tools", [])
            tools_by_turn[t_idx] = turn_tools
        return tools_by_turn
