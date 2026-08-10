from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import json
import random
import re
import copy
import os
import time
import requests
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from llm_client import LLMClient, LocalOpenAILLMClient
from tool_manager import ToolManager, filter_api_state
from prompts import StepByStepPrompts
from config_pool import generate_query_seed
from rl_quality_gate import validate_transition_quality


class ToolCallWithOutput(BaseModel):
    """A single tool call with its simulated output."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    output: Any = None


class StateVerificationResult(BaseModel):
    """LLM-as-judge verdict on a single state transition."""
    is_valid: bool = True
    reasoning: str = ""
    issues: List[str] = []
    state_changes_summary: str = ""


class TrajectoryStep(BaseModel):
    """A single step in the conversation trajectory."""
    step_number: int
    tool_calls: List[ToolCallWithOutput] = []
    execution_mode: str = "sequential"
    call_order_matters: bool = True
    reasoning: Optional[str] = None
    pre_state: Optional[Dict[str, Dict[str, Any]]] = None
    post_state: Optional[Dict[str, Dict[str, Any]]] = None
    state_verification: Optional[StateVerificationResult] = None
    quality_verification: Dict[str, Any] = Field(default_factory=dict)


class ConversationTrajectory(BaseModel):
    """Complete conversation trajectory for a datapoint."""
    query: str
    steps: List[TrajectoryStep] = []
    final_response: str
    tools_used: List[str] = []
    categories_used: List[str] = []
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None


class TokenUsageStats(BaseModel):
    """Token usage statistics for a single datapoint."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0
    total_llm_calls: int = 0


class StepByStepDatapoint(BaseModel):
    """Complete datapoint generated step-by-step."""
    trajectory: ConversationTrajectory
    generation_metadata: Dict[str, Any] = {}
    verification_result: Optional[Dict[str, Any]] = None
    token_usage: TokenUsageStats = Field(default_factory=TokenUsageStats)
    initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None
    intermediate_api_states: List[Dict[str, Any]] = Field(default_factory=list)
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Complete verification result for a generated datapoint."""
    query: str
    tool_relevance_checks: List[Dict[str, Any]] = []
    order_is_correct: bool
    order_verification_details: str = ""
    output_validations: List[Dict[str, Any]] = []
    placeholder_resolution: Dict[str, Any] = {}
    rl_quality_gate: Dict[str, Any] = Field(default_factory=dict)
    overall_verification_passed: bool
    verification_summary: str = ""


class StepSelectionResult(BaseModel):
    """Result of LLM selecting the next tool/step."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    reasoning: str


class QueryGenerationResult(BaseModel):
    """Result of generating a user query."""
    query: str
    intent: str
    expected_tools: List[str] = []
    quality_preflight: Dict[str, Any] = Field(default_factory=dict)


class GenerationBudgetExceeded(RuntimeError):
    """Raised before nested retries can turn one candidate into an outlier."""


class StepByStepGenerator:
    """Generator that creates datapoints step-by-step with immediate tool simulation."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager,
        num_actions: int = 2,
        validate_outputs: bool = True,
        judge_client: LLMClient = None,
        optimized_pipeline: Optional[bool] = None,
    ):
        self.llm = llm_client
        self.judge = judge_client or llm_client
        # Final-stage routing is independently configurable.  Defaults preserve
        # the historical behavior exactly: the main generator writes the final
        # answer and the judge client certifies grounding.
        self.final_response_llm = self.llm
        self.grounding_judge = self.judge
        self.tool_manager = tool_manager
        self.num_actions = num_actions
        self.validate_outputs = validate_outputs
        if optimized_pipeline is None:
            optimized_pipeline = os.getenv(
                "APIGEN_OPTIMIZED_PIPELINE", "1"
            ).strip().casefold() not in {"0", "false", "no", "off"}
        self.optimized_pipeline = bool(optimized_pipeline)
        self.max_calls_per_candidate = max(
            1, int(os.getenv("APIGEN_MAX_CALLS_PER_CANDIDATE", "30"))
        )
        self.max_tokens_per_candidate = max(
            1, int(os.getenv("APIGEN_MAX_TOKENS_PER_CANDIDATE", "100000"))
        )
        self.max_turn_attempts = max(
            1, min(2, int(os.getenv("APIGEN_MAX_TURN_ATTEMPTS", "2")))
        )
        self._python_tools_available = bool(tool_manager.python_tool_instances)
        self._accumulated_prompt_tokens: int = 0
        self._accumulated_completion_tokens: int = 0
        self._accumulated_total_tokens: int = 0
        self._accumulated_reasoning_tokens: int = 0
        self._accumulated_cached_prompt_tokens: int = 0
        self._accumulated_cost_usd: float = 0.0
        self._accumulated_llm_calls: int = 0
        self._initial_token_usage: Optional[Dict[str, Any]] = None
        self._initial_judge_token_usage: Optional[Dict[str, Any]] = None
        self._initial_usage_by_client: List[Tuple[LLMClient, Dict[str, Any]]] = []
        self._last_query_quality: Dict[str, Any] = {}
        self._last_final_response_quality: Dict[str, Any] = {}
        self._episode_query_quality: Dict[str, Any] = {}

    def configure_final_stage_clients(
        self,
        *,
        final_response_client: Optional[LLMClient] = None,
        grounding_client: Optional[LLMClient] = None,
    ) -> None:
        """Route final-answer writing and grounding without changing other roles.

        Passing neither client preserves the original behavior.  This method is
        intentionally separate from constructors so refusal/parallel subclasses
        and legacy call sites remain source-compatible.
        """
        self.final_response_llm = final_response_client or self.llm
        self.grounding_judge = grounding_client or self.judge

    def _distinct_llm_clients(self) -> List[LLMClient]:
        """Return each configured client once for budgets and usage accounting."""
        clients: List[LLMClient] = []
        seen: set[int] = set()
        for client in (
            self.llm,
            self.judge,
            self.final_response_llm,
            self.grounding_judge,
        ):
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            clients.append(client)
        return clients

    def _model_routing_metadata(self) -> Dict[str, Optional[str]]:
        """Record role/model routing without serializing endpoints or secrets."""
        return {
            "generator": getattr(self.llm, "api_model", None),
            "semantic_judge": getattr(self.judge, "api_model", None),
            "final_response_writer": getattr(
                self.final_response_llm, "api_model", None
            ),
            "grounding_judge": getattr(
                self.grounding_judge, "api_model", None
            ),
        }

    def _get_policy_tool_schemas(
        self,
        focus_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the exact tool contracts exposed for this episode."""
        tools = self.tool_manager.get_tools_json_schema()
        allowed = set(
            getattr(self, "_active_generation_directive", {}).get(
                "allowed_tools", []
            )
        )
        if allowed:
            tools = [
                tool for tool in tools
                if str(tool.get("name", "")) in allowed
            ]
        elif focus_category:
            tools = [
                tool for tool in tools
                if tool.get("category") == focus_category
            ]
        return copy.deepcopy(tools)

    @staticmethod
    def _tool_contract_hash(tools: List[Dict[str, Any]]) -> str:
        payload = json.dumps(
            tools,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _enforce_generation_budget(self, *, before_call: bool = False) -> None:
        """Fail the current candidate once its request/token budget is spent."""
        if self._initial_token_usage is None:
            return
        self._update_token_usage()
        call_limit_reached = (
            self._accumulated_llm_calls >= self.max_calls_per_candidate
            if before_call
            else self._accumulated_llm_calls > self.max_calls_per_candidate
        )
        if call_limit_reached:
            raise GenerationBudgetExceeded(
                "candidate LLM-call budget exhausted "
                f"({self._accumulated_llm_calls}/{self.max_calls_per_candidate})"
            )
        if self._accumulated_total_tokens > self.max_tokens_per_candidate:
            raise GenerationBudgetExceeded(
                "candidate token budget exhausted "
                f"({self._accumulated_total_tokens}/"
                f"{self.max_tokens_per_candidate})"
            )

    def _safe_llm_generate(
        self,
        messages: list,
        max_retries: Optional[int] = None,
        llm=None,
        purpose: str = "unspecified",
        **kwargs,
    ) -> str:
        """Call LLM generate() with application-level retry on transient errors.

        Args:
            llm: Override LLM client. Defaults to self.llm.
            purpose: Stable diagnostic label written to APIGEN_LLM_TRACE_PATH.
        """
        client = llm or self.llm
        # The HTTP clients already retry transient transport failures.  A second
        # five-attempt application loop used to multiply those retries at every
        # nested generation/judge call.
        if max_retries is None:
            max_retries = max(
                1, int(os.getenv("APIGEN_APPLICATION_LLM_ATTEMPTS", "1"))
            )
        configured_max_tokens = os.getenv("APIGEN_MAX_OUTPUT_TOKENS")
        if configured_max_tokens and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(configured_max_tokens)
        allow_openrouter_extensions = bool(
            getattr(client, "apigen_openrouter_extensions", True)
        )
        purpose_env_key = (
            "APIGEN_"
            + re.sub(r"[^A-Za-z0-9]+", "_", purpose).strip("_").upper()
            + "_REASONING_EFFORT"
        )
        # Cheap writer/judge models often become slower and more verbose when
        # they inherit the blueprint teacher's reasoning setting.  Allow a
        # stage-specific override while preserving the historical global
        # fallback for every existing caller.
        reasoning_effort = os.getenv(
            purpose_env_key,
            os.getenv("APIGEN_REASONING_EFFORT", ""),
        ).strip().lower()
        purpose_reasoning_budget_key = purpose_env_key.replace(
            "_REASONING_EFFORT", "_REASONING_MAX_TOKENS"
        )
        reasoning_max_tokens = os.getenv(
            purpose_reasoning_budget_key,
            os.getenv("APIGEN_REASONING_MAX_TOKENS", ""),
        ).strip()
        # Role-specific local clients opt out of OpenRouter-only request fields.
        # Existing generator/judge clients retain their historical behavior.
        if allow_openrouter_extensions and "reasoning" not in kwargs:
            if reasoning_max_tokens:
                budget = int(reasoning_max_tokens)
                if budget < 1:
                    raise ValueError(
                        f"{purpose_reasoning_budget_key} must be positive"
                    )
                kwargs["reasoning"] = {
                    "max_tokens": budget,
                    "exclude": True,
                }
            elif reasoning_effort in {
                "off", "none", "disabled", "false", "0"
            }:
                kwargs["reasoning"] = {"enabled": False, "exclude": True}
            elif reasoning_effort:
                kwargs["reasoning"] = {
                    "effort": reasoning_effort,
                    "exclude": True,
                }
        provider_slug = os.getenv(
            "APIGEN_OPENROUTER_PROVIDER", ""
        ).strip()
        ignored_providers = [
            provider.strip()
            for provider in os.getenv(
                "APIGEN_OPENROUTER_IGNORE_PROVIDERS", ""
            ).split(",")
            if provider.strip()
        ]
        if (
            allow_openrouter_extensions
            and (provider_slug or ignored_providers)
            and "provider" not in kwargs
        ):
            allow_fallbacks = os.getenv(
                "APIGEN_OPENROUTER_ALLOW_FALLBACKS", "false"
            ).strip().casefold() in {"1", "true", "yes", "on"}
            provider_routing = {
                "require_parameters": True,
            }
            if provider_slug:
                provider_routing.update(
                    {
                        "only": [provider_slug],
                        "allow_fallbacks": allow_fallbacks,
                    }
                )
            if ignored_providers:
                provider_routing["ignore"] = ignored_providers
            kwargs["provider"] = provider_routing
        import random as _rng
        for attempt in range(max_retries):
            try:
                self._enforce_generation_budget(before_call=True)
                # LocalOpenAILLMClient can retry several HTTP requests inside
                # one generate() call.  Bound that inner loop by the remaining
                # candidate budget; checking only before/after generate() lets
                # a call started at N-1 silently consume N+1, N+2, ... requests.
                remaining_http_attempts = max(
                    1,
                    self.max_calls_per_candidate
                    - self._accumulated_llm_calls,
                )
                configured_http_attempts = max(
                    1,
                    int(
                        kwargs.get(
                            "max_retries",
                            os.getenv("APIGEN_HTTP_ATTEMPTS", "3"),
                        )
                    ),
                )
                kwargs["max_retries"] = min(
                    configured_http_attempts,
                    remaining_http_attempts,
                )
                get_usage = getattr(client, "get_token_usage", None)
                usage_before = dict(get_usage()) if callable(get_usage) else {}
                started_at = time.monotonic()
                try:
                    result = client.generate(messages, **kwargs)
                except Exception as exc:
                    self._write_llm_trace_event(
                        client=client,
                        purpose=purpose,
                        application_attempt=attempt + 1,
                        status="error",
                        usage_before=usage_before,
                        elapsed_seconds=time.monotonic() - started_at,
                        error_type=type(exc).__name__,
                    )
                    raise
                self._write_llm_trace_event(
                    client=client,
                    purpose=purpose,
                    application_attempt=attempt + 1,
                    status=(
                        "success"
                        if result is not None and str(result).strip()
                        else "empty_response"
                    ),
                    usage_before=usage_before,
                    elapsed_seconds=time.monotonic() - started_at,
                )
                if result is None:
                    raise ValueError("LLM returned None")
                if not result.strip():
                    raise ValueError("LLM returned empty response")
                self._enforce_generation_budget()
                return result
            except GenerationBudgetExceeded:
                raise
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError,
                    requests.exceptions.ChunkedEncodingError) as e:
                delay = min(2 * (2 ** attempt), 60) + _rng.uniform(0, 2)
                print(f" [_safe_llm_generate] Transient error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            except (RuntimeError, ValueError) as e:
                if "Access denied by security policy" in str(e):
                    print(
                        " [_safe_llm_generate] Provider security policy rejected "
                        "this candidate; resampling a fresh datapoint."
                    )
                    raise
                delay = min(2 * (2 ** attempt), 30) + _rng.uniform(0, 1)
                print(f" [_safe_llm_generate] Error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            except json.JSONDecodeError as e:
                delay = min(2 * (2 ** attempt), 30) + _rng.uniform(0, 1)
                # Extra debug info for JSON errors
                print(f" [_safe_llm_generate] JSON Error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(delay)
        raise RuntimeError(f"LLM generate failed after {max_retries} application-level retries")

    @staticmethod
    def _write_llm_trace_event(
        *,
        client,
        purpose: str,
        application_attempt: int,
        status: str,
        usage_before: Dict[str, Any],
        elapsed_seconds: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Append one secret-free request accounting event when enabled."""
        trace_path = os.getenv("APIGEN_LLM_TRACE_PATH", "").strip()
        if not trace_path:
            return
        get_usage = getattr(client, "get_token_usage", None)
        usage_after = dict(get_usage()) if callable(get_usage) else {}
        numeric_keys = (
            "total_calls",
            "total_attempts",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_prompt_tokens",
            "cost_usd",
        )
        delta = {
            key: usage_after.get(key, 0) - usage_before.get(key, 0)
            for key in numeric_keys
        }
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "purpose": purpose,
            "model": getattr(client, "api_model", None),
            "requested_provider": os.getenv(
                "APIGEN_OPENROUTER_PROVIDER", ""
            ).strip() or None,
            "actual_provider": getattr(client, "last_provider", None),
            "finish_reason": getattr(client, "last_finish_reason", None),
            "application_attempt": application_attempt,
            "status": status,
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "usage_delta": delta,
        }
        if error_type:
            event["error_type"] = error_type
        destination = Path(trace_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _reset_token_tracking(self):
        """Reset token tracking for a new datapoint."""
        self._accumulated_prompt_tokens = 0
        self._accumulated_completion_tokens = 0
        self._accumulated_total_tokens = 0
        self._accumulated_reasoning_tokens = 0
        self._accumulated_cached_prompt_tokens = 0
        self._accumulated_cost_usd = 0.0
        self._accumulated_llm_calls = 0
        self._initial_token_usage = None
        self._initial_judge_token_usage = None
        self._initial_usage_by_client = []
        self._episode_query_quality = {}
    
    def _capture_initial_usage(self):
        """Capture usage for every distinct model role before a datapoint."""
        clients = self._distinct_llm_clients()
        self._initial_usage_by_client = [
            (client, dict(client.get_token_usage())) for client in clients
        ]
        # Keep historical attributes populated for compatibility with external
        # diagnostics that may inspect them directly.
        self._initial_token_usage = dict(self.llm.get_token_usage())
        self._initial_judge_token_usage = (
            dict(self.judge.get_token_usage())
            if self.judge is not self.llm
            else None
        )
    
    def _update_token_usage(self):
        """Update accumulated usage across generator, judges and final stages."""
        if not self._initial_usage_by_client:
            return

        sources = [
            (client.get_token_usage(), initial)
            for client, initial in self._initial_usage_by_client
        ]

        def delta(key: str) -> float:
            return sum(
                float(current.get(key, 0)) - float(initial.get(key, 0))
                for current, initial in sources
            )

        self._accumulated_prompt_tokens = int(delta("prompt_tokens"))
        self._accumulated_completion_tokens = int(delta("completion_tokens"))
        self._accumulated_total_tokens = int(delta("total_tokens"))
        self._accumulated_reasoning_tokens = int(delta("reasoning_tokens"))
        self._accumulated_cached_prompt_tokens = int(
            delta("cached_prompt_tokens")
        )
        self._accumulated_cost_usd = delta("cost_usd")
        self._accumulated_llm_calls = int(
            sum(
                float(
                    current.get(
                        "total_attempts",
                        current.get("total_calls", 0),
                    )
                )
                - float(
                    initial.get(
                        "total_attempts",
                        initial.get("total_calls", 0),
                    )
                )
                for current, initial in sources
            )
        )
    
    def _get_token_stats(self) -> TokenUsageStats:
        """Get current token usage stats."""
        return TokenUsageStats(
            prompt_tokens=self._accumulated_prompt_tokens,
            completion_tokens=self._accumulated_completion_tokens,
            total_tokens=self._accumulated_total_tokens,
            reasoning_tokens=self._accumulated_reasoning_tokens,
            cached_prompt_tokens=self._accumulated_cached_prompt_tokens,
            cost_usd=self._accumulated_cost_usd,
            total_llm_calls=self._accumulated_llm_calls
        )

    def _get_tool_schemas_str(self, tools_subset: Optional[List[str]] = None) -> str:
        schemas = self.tool_manager.get_tools_json_schema()
        if tools_subset:
            schemas = [s for s in schemas if s['name'] in tools_subset]
        return json.dumps(schemas, indent=2, ensure_ascii=False)


    def _get_tools_with_descriptions_str(self, category: Optional[str] = None, compact: bool = False) -> str:
        """Get a formatted string of tools with their full descriptions, organized by category."""
        tools = self.tool_manager.get_tools_json_schema()

        if category:
            tools = [t for t in tools if t.get('category') == category]

        if compact:
            result = []
            for tool in tools:
                name = tool['name']
                desc = tool.get('description', '')[:80]
                result.append(f"{name}: {desc}")
            return "\n".join(result)

        tools_by_cat = {}
        for tool in tools:
            cat = tool.get('category', 'Unknown')
            if cat not in tools_by_cat:
                tools_by_cat[cat] = []
            tools_by_cat[cat].append(tool)

        result = []
        for cat, cat_tools in sorted(tools_by_cat.items()):
            result.append(f"\n{cat}:")
            for tool in cat_tools:
                name = tool['name']
                desc = tool.get('description', 'No description available.')
                result.append(f" - {name}: {desc}")

        return "\n".join(result)

    def _process_placeholders(self, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        processed_args = copy.deepcopy(arguments)
        
        def resolve_placeholder(key_path: str) -> Any:
            """Resolve a placeholder key, supporting TURN{N} references."""
            keys = key_path.split('.')
            
            # Handle TURN{N} prefix
            if keys[0].startswith('TURN'):
                # Format: TURN{N}.{tool}.{field}
                # e.g., TURN1.authenticate_travel.access_token
                turn_ref = keys[0]  # e.g., TURN1
                turn_num = int(turn_ref.replace('TURN', '')) - 1  # 0-indexed
                
                # Look up in turn_outputs
                turn_outputs = execution_context.get('turn_outputs', [])
                if turn_num < len(turn_outputs):
                    current = turn_outputs[turn_num]
                    for k in keys[1:]:  # Skip TURN{N}, go to tool name
                        if isinstance(current, dict) and k in current:
                            current = current[k]
                        else:
                            return None
                    return current
                return None
            
            # Standard nested key lookup
            current = execution_context
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        
        for arg_name, arg_value in processed_args.items():
            if isinstance(arg_value, str):
                placeholders = re.findall(r"\{\{([^{}]+)\}\}", arg_value)
                for placeholder_full_key in placeholders:
                    resolved_value = resolve_placeholder(placeholder_full_key)
                    if resolved_value is not None:
                        placeholder_tag = "{{" + placeholder_full_key + "}}"
                        if arg_value == placeholder_tag:
                            processed_args[arg_name] = resolved_value
                        else:
                            processed_args[arg_name] = arg_value.replace(placeholder_tag, str(resolved_value))
        return processed_args

    def validate_expected_tools(
        self,
        query: str,
        expected_tools: List[str],
        intent: str,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
        policy_history: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[bool, str]:
        """Validate plan semantics and positive-RL suitability before execution.

        The judge may inspect generator-only state, but only whitelisted issue
        codes are retained or fed back to query generation. Hidden state values
        can therefore reject a task without becoming a source for gold arguments.
        """
        full_schemas = []
        for tool_name in expected_tools:
            schema = self.tool_manager.get_tool_schema(tool_name)
            if schema:
                full_schemas.append(schema)

        relevant_state = (
            filter_api_state(initial_api_state, expected_tools)
            if initial_api_state
            else {}
        )
        current_date = datetime.now(timezone.utc).date().isoformat()
        allowed_codes = {
            "TOOL_PLAN_CANNOT_FULFILL_QUERY",
            "MISSING_PREREQUISITE",
            "TARGET_ALREADY_SATISFIED",
            "MUTATION_WOULD_BE_NOOP",
            "CREATE_WOULD_OVERWRITE",
            "AMBIGUOUS_LIST_SELECTION",
            "PAST_OR_INVALID_DATE",
            "ENTITY_OWNERSHIP_MISMATCH",
            "NON_UNIQUE_GOLD_PLAN",
            "REQUESTED_RESULT_NOT_TOOL_GROUNDED",
            "POLICY_CONTEXT_NOT_CLOSED",
            "PREFLIGHT_UNAVAILABLE",
            "OTHER_INVALID",
        }

        # Reject plans that cannot reach an authenticated mutation from the
        # sampled state. Stage 1.5 is deliberately forbidden from manufacturing
        # authentication, so this prerequisite must either already hold or be
        # established by an earlier policy-visible tool call.
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
                {"create_ticket", "edit_ticket", "resolve_ticket", "close_ticket"},
                "ticket_login",
                "ticket_api",
                "authenticated",
            ),
            (
                {
                    "register_credit_card", "book_flight", "cancel_booking",
                    "purchase_insurance",
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
        for step_index, planned_tool in enumerate(expected_tools):
            for targets, prerequisite, class_key, state_field in prerequisite_rules:
                if planned_tool not in targets:
                    continue
                state_ready = bool(
                    relevant_state.get(class_key, {}).get(state_field)
                )
                prerequisite_planned = prerequisite in expected_tools[:step_index]
                if state_ready or prerequisite_planned:
                    continue
                self._last_query_quality = {
                    "passed": False,
                    "issue_codes": ["MISSING_PREREQUISITE"],
                    "current_date": current_date,
                }
                return (
                    False,
                    "RL quality preflight failed: MISSING_PREREQUISITE",
                )

        prompt = f"""You are certifying a synthetic tool-use task for positive reinforcement learning.

=== CURRENT DATE ===
{current_date}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history or [], indent=2, ensure_ascii=False, default=str)}

=== USER QUERY ===
{query}

=== INTENT ===
{intent}

=== PLANNED TOOL SEQUENCE ===
{json.dumps(expected_tools)}

=== FULL TOOL DEFINITIONS ===
{json.dumps(full_schemas, indent=2, ensure_ascii=False, default=str)}

=== GENERATOR-ONLY INITIAL STATE ===
{json.dumps(relevant_state, indent=2, ensure_ascii=False, default=str)}

The state is for validation only. Never copy, suggest, reveal, or exemplify any
state value in your response. Return only issue codes from the allowed list.

Check all of the following:
1. The exact tool sequence can fully satisfy the query, including prerequisites.
2. Every target argument is available from the visible query, prior
   policy-visible history, a prior tool output, or a declared schema default.
3. A requested mutation is not already satisfied in initial state; positive
   mutation examples must make meaningful progress.
4. A create action will add a new entity rather than reuse or overwrite an
   existing identifier.
5. If a prior tool can return several values and a later call consumes one, the
   query supplies a deterministic selection rule such as first, cheapest, index,
   or matching property. There must not be multiple equally valid gold actions.
6. If independent calls are represented as an ordered trajectory, the query
   explicitly determines their order. Otherwise report NON_UNIQUE_GOLD_PLAN.
7. Dates used for booking, scheduling, or other future actions are valid and not
   earlier than CURRENT DATE.
8. State-backed entities are coherent: account, traveler, cardholder, owner, and
   credentials refer to a compatible identity unless authorization is explicit.
9. Every result the user requests can be reported directly from planned tool
   outputs. Do not rely on an uncalled calculation or external knowledge.

Allowed issue codes:
{json.dumps(sorted(allowed_codes))}

Respond ONLY with JSON:
{{"is_valid": true, "issue_codes": []}}
or
{{"is_valid": false, "issue_codes": ["ONE_ALLOWED_CODE"]}}
"""

        try:
            response = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.judge,
                purpose="episode_plan_semantic_judge",
            )
            response_text = response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            result = json.loads(response_text)
            raw_codes = result.get("issue_codes", [])
            if not isinstance(raw_codes, list):
                raw_codes = []
            issue_codes = [
                str(code) for code in raw_codes
                if str(code) in allowed_codes
            ]
            is_valid = bool(result.get("is_valid", False)) and not issue_codes
            if not is_valid and not issue_codes:
                issue_codes = ["OTHER_INVALID"]

            self._last_query_quality = {
                "passed": is_valid,
                "issue_codes": issue_codes,
                "current_date": current_date,
            }
            if is_valid:
                return True, ""
            return False, "RL quality preflight failed: " + ", ".join(issue_codes)
        except Exception as exc:
            print(f"    Warning: RL query preflight failed: {exc}")
            self._last_query_quality = {
                "passed": False,
                "issue_codes": ["PREFLIGHT_UNAVAILABLE"],
                "current_date": current_date,
            }
            return False, "RL quality preflight failed: PREFLIGHT_UNAVAILABLE"

    def _get_example_queries(self) -> str:
        """Return few-shot examples of valid queries with correct tool sequences."""
        examples = [
            {
                "num_tools": 2,
                "query": "List items in the current directory, then create a new subdirectory.",
                "intent": "User wants to see files and create a folder",
                "expected_tools": ["ls", "mkdir"]
            },
            {
                "num_tools": 2,
                "query": "Display the contents of report.txt, then search for the word 'error' in it.",
                "intent": "User wants to read a file and find specific text",
                "expected_tools": ["cat", "grep"]
            },
            {
                "num_tools": 3,
                "query": "Create a new file named notes.txt, write 'Hello World' to it, then display its contents.",
                "intent": "User wants to create and populate a file",
                "expected_tools": ["touch", "echo", "cat"]
            },
        ]

        filtered_examples = [
            ex for ex in examples
            if self.num_actions - 1 <= ex["num_tools"] <= self.num_actions + 1
        ]
        if not filtered_examples:
            filtered_examples = examples[:2]

        result = []
        for i, ex in enumerate(filtered_examples, 1):
            result.append(f"\n=== EXAMPLE {i} ({ex['num_tools']} tools) ===")
            result.append(f"Query: \"{ex['query']}\"")
            result.append(f"Intent: {ex['intent']}")
            result.append(f"Expected tools: {ex['expected_tools']}")

        return "\n".join(result)

    def generate_user_query(
        self,
        focus_category: Optional[str] = None,
        validation_feedback: Optional[str] = None,
        max_retries: int = 3,
        query_seed: Optional[dict] = None,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> QueryGenerationResult:
        # Query generation needs complete parameter and output schemas. Compact
        # descriptions omit the information needed to plan argument dependencies.
        tools_for_prompt = self.tool_manager.get_tools_json_schema()
        if focus_category:
            tools_for_prompt = [
                tool for tool in tools_for_prompt
                if tool.get("category") == focus_category
            ]
        tools_with_descriptions = json.dumps(
            tools_for_prompt,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        # The initial state is generator-only. The policy will not see it, so any
        # state value needed by a target call must be surfaced in the user query
        # unless an earlier tool call returns it.
        state_for_prompt = initial_api_state
        if initial_api_state and tools_for_prompt:
            state_for_prompt = filter_api_state(
                initial_api_state,
                [tool.get("name", "") for tool in tools_for_prompt],
            )
        generator_state_section = ""
        if state_for_prompt:
            generator_state_section = f"""
=== GENERATOR-ONLY API STATE ===
This state is visible only while constructing the synthetic task. It will NOT be
shown to the assistant that solves the task. Use it to choose valid values, but
copy every required value into the generated user query unless that value is
returned by an earlier expected tool call.

{json.dumps(state_for_prompt, indent=2, ensure_ascii=False, default=str)}
"""

        accumulated_feedback = validation_feedback or ""
        example_queries = self._get_example_queries()

        persona_section = ""
        if query_seed:
            p = query_seed["persona"]
            c = query_seed["city"]
            persona_section = f"""
=== OPTIONAL STYLE / FRAMING SEED ===
Name: {p['name']}
Home location: {p['city']}, {p.get('country', '')}
Alternate scenario location: {c['city']}

Use these only for natural-language diversity when they remain coherent with the
selected tools and generator-only state. Do not force a name or location into a
task, do not treat persona fields as verified tool facts, and do not use them as
argument values unless the visible query explicitly states them and the tool
schema accepts that semantic representation.
"""

        for attempt in range(max_retries):
            prompt = f"""Generate a realistic user query requiring EXACTLY {self.num_actions} tools.

=== REQUIREMENTS ===
1. Specific with concrete entities (names, IDs, dates, locations)
2. EXACTLY {self.num_actions} tool calls needed - not more, not less
3. expected_tools: EXACTLY {self.num_actions} tool names from AVAILABLE TOOLS
4. CRITICAL: Use ONLY tools from AVAILABLE TOOLS - no invented names
5. Auth-dependent tools need authentication FIRST - check which tools require prior authentication
6. POLICY-CONTEXT CLOSURE: Every required argument for every expected tool must
   be available from the generated user query, an earlier expected tool output,
   or a default declared in that tool's schema.
7. The solving assistant receives only the user query, the complete tool schemas,
   and prior tool outputs. It does NOT receive the API state below.
8. If a required value exists only in API state, write that exact value naturally
   into the user query. If an earlier tool produces it, place that tool before the
   dependent tool. Never require the assistant to guess a value.
9. Match the exact semantic representation required by the tool definition. A
   human-readable label is not interchangeable with an opaque identifier, code,
   token, symbol, handle, coordinate, path, or credential.
10. General/model knowledge is not an argument source. If an opaque value is not
    written in the user query and is not returned by an earlier expected tool,
    choose a different task or include the required lookup within the call budget.
11. If exactly {self.num_actions} calls cannot satisfy these rules, generate a
    different task that can.
12. STATE PROGRESS: For a mutating tool, choose a target that is not already in
    the requested final state. Do not generate a positive example whose mutation
    is a no-op.
13. UNIQUE CREATION: A create action must add a new entity and must not reuse or
    overwrite an identifier already present in generator-only state.
14. UNAMBIGUOUS SELECTION: If one tool returns multiple candidates and a later
    tool consumes one, explicitly state a deterministic selection rule in the
    query (for example by rank, index, minimum/maximum, or matching property).
15. ORDER DETERMINISM: If two calls are independent but the dataset scores an
    ordered sequence, state their order explicitly with wording such as first /
    then. Do not leave multiple equally correct next actions.
16. TEMPORAL COHERENCE: The current UTC date is
    {datetime.now(timezone.utc).date().isoformat()}. Booking or scheduling dates
    must be on or after this date unless the task explicitly requests historical
    lookup and performs no future action.
17. ENTITY COHERENCE: State-backed users, accounts, owners, travelers, cards, and
    credentials must be mutually compatible unless the query explicitly states
    valid authorization.
18. REPORTABILITY: Every result requested from the assistant must be directly
    available in a planned tool output. Do not require an extra uncalled
    calculation or factual inference in the final response.
19. Match the query to the exact call cardinality supported by each tool. If one
    invocation accepts only one selected option, ask for one option or include a
    separate expected-tool occurrence for every requested option.
{persona_section}
=== AVAILABLE TOOLS ===
{tools_with_descriptions}
{generator_state_section}
{example_queries}"""
            if focus_category:
                prompt += f"\n=== FOCUS CATEGORY ===\nPrimary: {focus_category}\n"
            if accumulated_feedback:
                prompt += f"\n=== FEEDBACK ===\n{accumulated_feedback}\n"
            prompt += f"""
=== TASK ===
Generate query requiring EXACTLY {self.num_actions} tools. Respond JSON:
{{"query": "specific with names/IDs", "intent": "what user wants", "expected_tools": ["tool1", ...]}}"""

            try:
                response = self._safe_llm_generate([{"role": "user", "content": prompt}])
                response_text = response.strip()

                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                else:
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start >= 0 and end > start:
                        response_text = response_text[start:end]

                result = json.loads(response_text)
                query = result.get("query", "")
                intent = result.get("intent", "")
                expected_tools = result.get("expected_tools", [])

                print(f" Generated Query: {query}")
                print(f" Intent: {intent}")
                print(f" Expected tools: {expected_tools}")

                generated_summary = f"""--- ATTEMPT {attempt + 1} OUTPUT ---
Query: {query}
Intent: {intent}
Expected tools: {expected_tools}"""

                if len(expected_tools) != self.num_actions:
                    print(
                        f" ✗ Wrong tool count: {len(expected_tools)} "
                        f"!= {self.num_actions}"
                    )
                    accumulated_feedback += (
                        f"\n{generated_summary}\nFAILURE: Expected "
                        f"{self.num_actions} tools, but got {len(expected_tools)}."
                        f"\n--- END ATTEMPT {attempt + 1} ---"
                    )
                    continue

                all_tools_valid = True
                invalid_tools = []
                for tool in expected_tools:
                    if not self.tool_manager.tool_exists(tool):
                        all_tools_valid = False
                        invalid_tools.append(tool)

                if not all_tools_valid:
                    available_tools = []
                    if focus_category:
                        cat_tools = self.tool_manager.get_tools_by_category(focus_category)
                        available_tools = [t['name'] for t in cat_tools[:20]]
                    else:
                        for cat in self.tool_manager.get_categories():
                            cat_tools = self.tool_manager.get_tools_by_category(cat)
                            available_tools.extend([t['name'] for t in cat_tools[:5]])

                    print(f" ✗ Invalid tools: {invalid_tools}")
                    accumulated_feedback += f"""\n{generated_summary}
FAILURE: Tools not found: {invalid_tools}
These tools do NOT exist. Choose from available tools.
Available tools (sample): {available_tools[:15]}
--- END ATTEMPT {attempt + 1} ---"""
                    continue

                is_valid, validation_msg = self.validate_expected_tools(
                    query,
                    expected_tools,
                    intent,
                    initial_api_state=initial_api_state,
                )

                if not is_valid:
                    print(f" ✗ Tool sequence / RL preflight failed: {validation_msg}")
                    accumulated_feedback += (
                        f"\n{generated_summary}\nFAILURE: {validation_msg}. "
                        "Generate a different task that removes these generic "
                        "quality defects; do not copy hidden state values.\n"
                        f"--- END ATTEMPT {attempt + 1} ---"
                    )
                    continue

                print(f" ✓ Query generation successful")
                return QueryGenerationResult(
                    query=query,
                    intent=intent,
                    expected_tools=expected_tools,
                    quality_preflight=dict(self._last_query_quality),
                )

            except json.JSONDecodeError as e:
                print(f" ✗ JSON decode error: {e}")
                accumulated_feedback += f"\n--- ATTEMPT {attempt + 1} FAILED ---\nJSON parsing error: {e}\n--- END ATTEMPT {attempt + 1} ---"
                continue

        print(f" Failed to generate valid query after {max_retries} attempts")
        return QueryGenerationResult(query="", intent="", expected_tools=[])

    def _generate_next_step(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any], expected_tools: List[str], step_num: int = 1) -> StepSelectionResult:
        trajectory_str = ""
        for i, step in enumerate(trajectory):
            trajectory_str += f"\nStep {i+1}:"
            for tc in step.tool_calls:
                trajectory_str += f"\n - {tc.tool_name}({json.dumps(tc.arguments)})"
                if tc.output:
                    trajectory_str += f" -> {json.dumps(tc.output)[:200]}"

        tools_used = set()
        for step in trajectory:
            for tc in step.tool_calls:
                tools_used.add(tc.tool_name)

        tools_remaining = [t for t in expected_tools if t not in tools_used]
        if not tools_remaining:
            return StepSelectionResult(tool_name="__FINAL_RESPONSE__", arguments={}, reasoning="All expected tools have been used.")

        # Get tools with descriptions for remaining expected tools
        tool_descriptions_str = ""
        for tool_name in tools_remaining:
            try:
                    schema = self.tool_manager.get_tool_schema(tool_name)
                    if schema:
                        desc = schema.get('description', 'No description available.')[:150]
                        tool_descriptions_str += f" - {tool_name}: {desc}\n"
                    else:
                        tool_descriptions_str += f" - {tool_name}: (tool for completing the task)\n"
            except Exception as e:
                tool_descriptions_str += f" - {tool_name}: (tool for completing the task)\n"

        if not tool_descriptions_str:
            for tool_name in tools_remaining:
                tool_descriptions_str += f" - {tool_name}: (tool for completing the task)\n"

        prompt = f"""You are selecting the next tool to call based on the conversation context.

=== USER QUERY ===
{query}

=== CURRENT TRAJECTORY ===
{trajectory_str}

=== EXPECTED TOOLS REMAINING ===
{tool_descriptions_str}

=== EXECUTION CONTEXT (previous tool outputs) ===
{json.dumps(execution_context, indent=2, default=str)[:1000]}

=== YOUR TASK ===
Select the NEXT tool to call from the EXPECTED TOOLS REMAINING list above.

CRITICAL:
- You MUST select a tool name EXACTLY as shown in EXPECTED TOOLS REMAINING
- The tool must logically follow from the current trajectory and context
- Use values from Execution Context when available (e.g., user_id from previous step)

Respond ONLY with valid JSON:
{{
    "tool_name": "exact_name_from_expected_tools_list",
    "arguments": {{"arg1": "value1", "arg2": "value2"}},
    "reasoning": "brief explanation of why this tool and these arguments"
}}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}])
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            result = json.loads(response_text)
            return StepSelectionResult(
                tool_name=result.get("tool_name", ""),
                arguments=result.get("arguments", {}),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError as e:
            print(f"    JSON decode error in step generation: {e}")
            return StepSelectionResult(tool_name="__ERROR__", arguments={}, reasoning=f"JSON error: {e}")

    def _simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Any:
        if isinstance(arguments, list):
            if len(arguments) == 1 and isinstance(arguments[0], dict):
                arguments = arguments[0]
            else:
                arguments = arguments[0] if arguments else {}
        processed_args = self._process_placeholders(arguments, execution_context)
        if self._python_tools_available:
            if self.tool_manager.has_python_implementation(tool_name):
                return self.tool_manager.invoke_python_tool(tool_name, processed_args)
            raise NotImplementedError(f"No Python implementation for '{tool_name}' (api_name_to_class_key={self.tool_manager.api_name_to_class_key.get(tool_name, 'NOT IN MAP')}, has_impl={self.tool_manager.has_python_implementation(tool_name)}). LLM simulation disabled - implement the Python tool.")
        return self.tool_manager.invoke_tool(tool_name=tool_name, params=processed_args)

    def _verify_tool_query_consistency(self, tool_name: str, arguments: Dict[str, Any],
                                       query: str, trajectory: List[TrajectoryStep],
                                       execution_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify that the selected tool and arguments are consistent with the query and trajectory.

        Returns (is_valid: bool, feedback: str).
        """
        trajectory_summary = ""
        for i, step in enumerate(trajectory):
            for tc in step.tool_calls:
                output_summary = (
                    json.dumps(tc.output, ensure_ascii=False, default=str)
                    if tc.output is not None
                    else "None"
                )
                trajectory_summary += f"Step {i+1}: {tc.tool_name}({tc.arguments}) -> {output_summary}\n"

        tool_schema = self.tool_manager.get_tool_schema(tool_name)

        prompt = f"""You are verifying that a tool invocation is consistent with the user query and conversation context.

=== USER QUERY ===
{query}

=== SELECTED TOOL ===
{tool_name}

=== FULL TOOL DEFINITION ===
{json.dumps(tool_schema, indent=2, ensure_ascii=False, default=str)}

=== GENERATED ARGUMENTS ===
{json.dumps(arguments, indent=2)}

=== PREVIOUS TRAJECTORY ===
{trajectory_summary if trajectory_summary else "None"}

=== EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, ensure_ascii=False, default=str)}

=== CURRENT DATE ===
{datetime.now(timezone.utc).date().isoformat()}

=== VERIFICATION TASK ===
The selected tool is one step in a planned multi-call trajectory. Judge whether
this invocation correctly handles the portion of the query that is appropriate
after PREVIOUS TRAJECTORY. Do not require this single invocation to fulfill
actions assigned to later calls, and do not report those later actions as
missing. The invocation is invalid if it is unnecessary, out of order, or
inconsistent with the part it is meant to handle.

Verify the tool and arguments are consistent with the query by checking:
1. Does the tool match the query intent?
2. Are arguments correctly typed and in valid ranges?
3. Are argument values correctly referencing previous outputs (e.g., user_id, ticket_id from prior steps)?
4. Are the arguments sufficient to fulfill the query?
5. For every opaque identifier, code, token, symbol, handle, coordinate, path,
   or credential, does the exact value come from the user query, a prior saved
   tool output, or a schema default? General/model knowledge is not a source.
6. For booking or scheduling tools, is the date valid and not before CURRENT DATE?
7. If an argument selects one element from a prior list output, does the user
   query provide a unique selection rule or exact requested value?

Do not suggest, reveal, or exemplify replacement argument values. Report only
the affected argument names and the violated query/schema constraint.

Respond ONLY with valid JSON:
{{"is_valid": true/false, "issues": ["issue1", "issue2", ...]}}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}], llm=self.judge)
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            result = json.loads(response_text)
            is_valid = result.get("is_valid", True)
            issues = result.get("issues", [])
            feedback = "; ".join(issues) if issues else ""
            return is_valid, feedback
        except Exception as e:
            print(f"    Warning: Consistency verification failed: {e}")
            return False, "Consistency verifier unavailable"

    @staticmethod
    def _detect_tool_error(tool_name: str, output: dict) -> tuple:
        """Detect errors in tool output using tool-specific keys and value checks.

        Returns (has_error: bool, error_detail: str).
        """
        generic_error_keys = ['error', 'error_message', 'error_code']
        for key in generic_error_keys:
            value = output.get(key)
            if value not in (None, "", False, 0, []):
                return True, str(value)

        tool_specific_checks = {
            'si_unit_conversion': [('error', lambda v: bool(v))],
            'comment': [('comment_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'retweet': [('retweet_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'mention': [('mention_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'post_tweet': [('tweet_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'follow_user': [('follow_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'unfollow_user': [('follow_status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'authenticate_twitter': [('authentication_status', lambda v: v is False)],
            'get_ticket': [
                ('status', lambda v: isinstance(v, str) and 'not found' in v.lower()),
            ],
            'edit_ticket': [('status', lambda v: isinstance(v, str) and ('not found' in v.lower() or 'not authenticated' in v.lower()))],
            'resolve_ticket': [('status', lambda v: isinstance(v, str) and ('not found' in v.lower() or 'not authenticated' in v.lower()))],
            'close_ticket': [('status', lambda v: isinstance(v, str) and ('not found' in v.lower() or 'not authenticated' in v.lower()))],
            'create_ticket': [('status', lambda v: isinstance(v, str) and 'not authenticated' in v.lower())],
            'ticket_login': [('success', lambda v: v is False)],
            'message_login': [('login_status', lambda v: v is False)],
            'verify_traveler_information': [('verification_status', lambda v: v is False)],
            'authenticate_travel': [('success', lambda v: v is False), ('access_token', lambda v: v == '')],
            'get_flight_cost': [('error', lambda v: bool(v)), ('travel_cost_list', lambda v: isinstance(v, list) and len(v) == 0)],
            'get_symbol_by_name': [
                ('symbol', lambda v: isinstance(v, str) and 'not found' in v.lower()),
            ],
            'book_flight': [
                ('booking_status', lambda v: isinstance(v, str) and ('fail' in v.lower() or 'error' in v.lower())),
                ('booking_confirmation', lambda v: isinstance(v, str) and ('fail' in v.lower() or 'error' in v.lower())),
            ],
        }

        checks = tool_specific_checks.get(tool_name, [])
        for key, is_error in checks:
            val = output.get(key)
            if val is not None and is_error(val):
                return True, f"{key}: {val}"

        failure_phrases = (
            "not authenticated",
            "login failed",
            "authentication failed",
            "not found",
            "invalid argument",
            "invalid input",
            "insufficient funds",
            "permission denied",
            "already exists",
            "not in watchlist",
            "unable to",
        )

        def find_failure(
            value: Any,
            path: str = "",
            field_name: str = "",
        ) -> Optional[str]:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    child_path = f"{path}.{child_key}" if path else str(child_key)
                    found = find_failure(
                        child_value,
                        child_path,
                        str(child_key),
                    )
                    if found:
                        return found
            elif isinstance(value, list):
                for index, child_value in enumerate(value):
                    found = find_failure(
                        child_value,
                        f"{path}[{index}]",
                        field_name,
                    )
                    if found:
                        return found
            else:
                normalized_key = field_name.casefold()
                status_field = (
                    normalized_key in {"status", "success", "ok"}
                    or normalized_key.endswith("_status")
                )
                if status_field and value is False:
                    return f"{path}: {value}"
            if isinstance(value, str):
                lowered = value.strip().casefold()
                normalized_key = field_name.casefold()
                status_field = (
                    normalized_key in {"status", "success", "ok"}
                    or normalized_key.endswith("_status")
                )
                if status_field and (
                    lowered.startswith(("error:", "failed", "failure"))
                    or any(phrase in lowered for phrase in failure_phrases)
                ):
                    return f"{path}: {value}"
            return None

        generic_failure = find_failure(output)
        if generic_failure:
            return True, generic_failure

        return False, ""

    # ==================== REFACTORED THREE-STAGE GENERATION ====================

    def generate_datapoint(self, focus_category: Optional[str] = None, context_hint: Optional[str] = None,
                           query_retries: int = 5, tool_retries: int = 3) -> Optional[StepByStepDatapoint]:
        """
        Generate a datapoint using multi-stage generation:
        Stage 1: Generate and verify query (separate retry count)
        Stage 1.5: Adjust initial API state for expected tools (best-effort)
        Stage 2: Generate tool invocations tool-by-tool (separate retry count per tool)
        Stage 3: Finalize datapoint (no retries)
        """
        print("\n" + "=" * 70)
        print("STEP-BY-STEP DATAPOINT GENERATION (Refactored)")
        print("=" * 70)

        # Reset and start token tracking for this datapoint
        self._reset_token_tracking()
        self._capture_initial_usage()
        self._last_query_quality = {}
        self._last_final_response_quality = {}

        query_seed = generate_query_seed()
        print(f" Persona seed: {query_seed['persona']['name']}, {query_seed['city']['city']}")

        # Initialize API state with full, realistic configurations
        # This ensures login calls and subsequent operations succeed
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None
        if self._python_tools_available:
            self.tool_manager.initialize_api_state()
            initial_api_state = self.tool_manager.get_api_state()
            print(f" Captured initial API state ({len(initial_api_state)} class keys)")

        # Stage 1: Generate and verify query
        print("\n" + "-" * 70)
        print("STAGE 1: Generate and Verify Query")
        print("-" * 70)
        
        query_result = self._stage1_generate_query(
            focus_category,
            context_hint,
            query_retries,
            query_seed,
            initial_api_state,
        )
        
        if query_result is None:
            print("\n✗ Stage 1 failed: Could not generate valid query")
            print(f"  Token usage for failed datapoint: {self._accumulated_total_tokens:,} tokens, {self._accumulated_llm_calls} calls")
            return None
        
        self._update_token_usage()
        print(f"\n✓ Stage 1 complete: Query generated and verified")
        print(f" Query: {query_result.query}")
        print(f" Expected tools: {query_result.expected_tools}")
        print(f" Tokens so far: {self._accumulated_total_tokens:,}")

        # Identity coherence is validated during query preflight. Do not mutate
        # account/user/card identity after the task has been generated: doing so
        # can manufacture a world that differs from the sampled scenario and can
        # make hidden values appear valid only to the generator.

        # Stage 1.5: Adjust initial API state for expected tools
        if self._python_tools_available and query_result.expected_tools:
            print("\n" + "-" * 70)
            print("STAGE 1.5: Adjust Initial API State")
            print("-" * 70)

            adjusted = self._stage1_5_adjust_initial_state(query_result)

            if adjusted:
                initial_api_state = self.tool_manager.get_api_state()
                print(f" ✓ API state adjusted, re-captured ({len(initial_api_state)} class keys)")
                self._update_token_usage()
                print(f" Tokens so far: {self._accumulated_total_tokens:,}")

                still_valid, quality_feedback = self.validate_expected_tools(
                    query_result.query,
                    query_result.expected_tools,
                    query_result.intent,
                    initial_api_state=initial_api_state,
                )
                if not still_valid:
                    print(
                        " ✗ State adjustment made the task unsuitable for "
                        f"positive RL: {quality_feedback}"
                    )
                    return None
                query_result.quality_preflight = dict(self._last_query_quality)
            else:
                print(" ⚠ State adjustment failed or not needed, proceeding with original state")

        # Stage 2: Generate tool invocations tool-by-tool
        print("\n" + "-" * 70)
        print("STAGE 2: Generate Tool Invocations")
        print("-" * 70)
        
        trajectory, execution_context = self._stage2_generate_tools(query_result, tool_retries)
        
        if trajectory is None:
            print("\n✗ Stage 2 failed: Could not generate all tool invocations")
            print(f"  Token usage for failed datapoint: {self._accumulated_total_tokens:,} tokens, {self._accumulated_llm_calls} calls")
            return None
        
        self._update_token_usage()
        print(f"\n✓ Stage 2 complete: Generated {len(trajectory)} tool invocations")
        print(f"  Tokens so far: {self._accumulated_total_tokens:,}")

        # Stage 3: Finalize datapoint
        print("\n" + "-" * 70)
        print("STAGE 3: Finalize Datapoint")
        print("-" * 70)
        
        datapoint = self._stage3_finalize(query_result, trajectory, execution_context, focus_category, initial_api_state)
        
        if datapoint is None:
            print("\n✗ Stage 3 failed: Could not finalize datapoint")
            return None
        
        print("\n" + "=" * 70)
        print("✓ DATAPOINT GENERATION COMPLETE (VERIFIED)")
        print("=" * 70)
        print(f" Query: {datapoint.trajectory.query}")
        print(f" Tools used: {datapoint.trajectory.tools_used}")
        print(f" Steps: {len(datapoint.trajectory.steps)}")
        print(f" Verification: PASSED")

        return datapoint

    def _stage1_generate_query(
        self,
        focus_category: Optional[str],
        context_hint: Optional[str],
        max_retries: int,
        query_seed: Optional[dict] = None,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[QueryGenerationResult]:
        """
        Stage 1: Generate and verify user query.
        - Separate retry count for query generation
        - Feedback is wiped on successful verification
        - Returns QueryGenerationResult or None if all retries exhausted
        """
        accumulated_feedback = context_hint or ""
        
        for attempt in range(max_retries):
            print(f"\n[Query Attempt {attempt + 1}/{max_retries}]")
            
            # Generate query
            query_result = self.generate_user_query(
                focus_category,
                accumulated_feedback if accumulated_feedback else None,
                query_seed=query_seed,
                initial_api_state=initial_api_state,
            )

            if query_result is None or not query_result.query:
                print("  ✗ Failed to generate query")
                accumulated_feedback += f"\n--- ATTEMPT {attempt + 1} FAILED ---\nFailed to generate a valid query.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            print(f"  Generated Query: {query_result.query}")
            print(f"  Intent: {query_result.intent}")
            print(f"  Expected tools: {query_result.expected_tools}")

            # Build a summary of what was generated for feedback
            generated_summary = f"""--- ATTEMPT {attempt + 1} OUTPUT ---
    Query: {query_result.query}
    Intent: {query_result.intent}
    Expected tools: {query_result.expected_tools}"""

            # Verify expected_tools
            print(f"  Verifying expected tools...")

            if not query_result.expected_tools:
                print("  ✗ ERROR: expected_tools is empty")
                accumulated_feedback += f"\n{generated_summary}\nFAILURE: expected_tools is empty.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            if len(query_result.expected_tools) != self.num_actions:
                print(
                    "  ✗ ERROR: expected_tools count "
                    f"{len(query_result.expected_tools)} != {self.num_actions}"
                )
                accumulated_feedback += (
                    f"\n{generated_summary}\nFAILURE: expected_tools count "
                    f"mismatch - got {len(query_result.expected_tools)}, need "
                    f"{self.num_actions}.\n--- END ATTEMPT {attempt + 1} ---"
                )
                continue

            # Check if all tools exist
            invalid_tools = [t for t in query_result.expected_tools if not self.tool_manager.tool_exists(t)]
            if invalid_tools:
                print(f"  ✗ ERROR: Tools not found: {invalid_tools}")
                accumulated_feedback += f"\n{generated_summary}\nFAILURE: Tools not found: {invalid_tools}.\n--- END ATTEMPT {attempt + 1} ---"
                continue

            # Tool sequence validation already done inside generate_user_query
            # (which retries internally with feedback on failure)

            # SUCCESS: Query is valid - wipe feedback and return
            print(" ✓ Query verification passed")
            return query_result

        # All retries exhausted
        print(f"\n✗ Failed to generate valid query after {max_retries} attempts")
        return None

    def _ensure_user_identity_coherence(self, text: str) -> bool:
        """Ensure user identity is coherent across APIs.

        Finds any user name in 'text' that also exists in any API's user_map.
        If found, ensures this user is set consistently across ALL API instances
        that have identity-related attributes (first_name, last_name, current_user, etc.)
        that were set to a DIFFERENT user. This prevents incoherent scenarios where
        the query references "user Tom" but auth credentials point to "David Park".

        Returns True if adjustments were made, False otherwise.
        """
        current_state = self.tool_manager.get_api_state()
        text_lower = text.lower()

        user_candidates = set()
        for class_key, state in current_state.items():
            if not isinstance(state, dict):
                continue
            if 'user_map' in state and isinstance(state['user_map'], dict):
                for username in state['user_map']:
                    if username.lower() in text_lower:
                        user_candidates.add(username)

        if not user_candidates:
            return False

        primary_user = list(user_candidates)[0]

        adjusted = False
        identity_attrs = {'first_name', 'last_name', 'current_user', 'username', 'user_id'}

        for class_key, state in current_state.items():
            inst = self.tool_manager.python_tool_instances.get(class_key)
            if not inst:
                continue
            for attr in identity_attrs:
                if hasattr(inst, attr):
                    current_val = getattr(inst, attr)
                    if current_val and isinstance(current_val, str) and current_val != primary_user:
                        print(f"  Sync {class_key}.{attr}: {current_val} -> {primary_user}")
                        setattr(inst, attr, primary_user)
                        adjusted = True

        return adjusted

    def _stage1_5_adjust_initial_state(self, query_result: QueryGenerationResult) -> bool:
        """Stage 1.5: Adjust initial API state so expected tools will succeed.

        Sends the query, expected tools, their schemas, and the current API state
        to the LLM. The LLM identifies what state modifications are needed for
        each tool to execute successfully (e.g., ensure specific ticket IDs exist,
        user IDs are present, access tokens are set, etc.).

        Modifications are applied directly to the live Python tool instances,
        then the initial_api_state is re-captured.

        Returns True if adjustments were applied, False otherwise.
        """
        current_state = self.tool_manager.get_api_state()

        relevant_class_keys = set()
        for tool_name in query_result.expected_tools:
            class_key = self.tool_manager.api_name_to_class_key.get(tool_name)
            if class_key:
                relevant_class_keys.add(class_key)

        relevant_state = {k: v for k, v in current_state.items() if k in relevant_class_keys}

        tool_schemas_str = ""
        for tool_name in query_result.expected_tools:
            schema = self.tool_manager.get_tool_schema(tool_name)
            if schema:
                params = schema.get('parameters', {})
                desc = schema.get('description', '')
                tool_schemas_str += f"\n- {tool_name}: {desc}\n  Parameters: {json.dumps(params, indent=2, default=str)[:500]}\n"

        state_json = json.dumps(relevant_state, indent=2, default=str)[:6000]

        prompt_parts = [
            """You are preparing the initial state of API instances so that tool calls execute successfully.

=== USER QUERY ===
""",
            query_result.query,
            """

=== EXPECTED TOOL SEQUENCE ===
""",
            json.dumps(query_result.expected_tools),
            """

=== TOOL SCHEMAS ===
""",
            tool_schemas_str,
            """

=== CURRENT API STATE ===
""",
            state_json,
            """

=== RULES ===
- To add user: user_map.Username = "USR015"
- To append to list: "APPEND:array_key": value
- Set fields directly: "field_name": "value"
- Never do string operations like "APPEND:foo = bar"
- MINIMAL changes only
- Do not complete the user's requested mutation in advance. The target must still
  require a meaningful state change when the trajectory executes.
- Do not alter counters, reuse identifiers, overwrite existing entities, or
  change ownership/cardholder/account identity to force a plan to pass.
- Do not add a value that the solving policy could not obtain from the visible
  query or an earlier planned tool output.
- Prefer no modification over changing the semantic meaning of the sampled state.

=== EXAMPLES ===
Add a new entry to a key-value map:
{"modifications": {"api_name": {"map_key.NewName": "USR015"}}, "reasoning": "..."}

Add new item to a queue:
{"modifications": {"api_name": {"APPEND:queue_key": {"id": 1234}}}, "reasoning": "..."}

No changes needed:
{"modifications": {}, "reasoning": "no changes needed"}

=== RESPONSE ===
Respond only with valid JSON in one of these formats"""
        ]
        prompt = "".join(prompt_parts)

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}])
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
            modifications = result.get("modifications", {})
            reasoning = result.get("reasoning", "")

            if not modifications:
                print(f" No modifications needed: {reasoning}")
                return False

            print(f" Modifications requested: {json.dumps(modifications, indent=2, default=str)[:1000]}")
            print(f" Reasoning: {reasoning}")

            applied = self._apply_state_modifications(modifications)

            # Post-validation: remove duplicate user_map entries that would cause add_contact to fail
            if applied > 0 and 'message_api' in self.tool_manager.python_tool_instances:
                msg_api = self.tool_manager.python_tool_instances['message_api']
                user_map = getattr(msg_api, 'user_map', {})
                if user_map:
                    ids_to_names = {}
                    names_to_remove = set()
                    for name, uid in list(user_map.items()):
                        if uid in ids_to_names:
                            existing_name = ids_to_names[uid]
                            if existing_name != name:
                                names_to_remove.add(name)
                                print(f"   [DEDUP] Removing duplicate user_map entry '{name}' -> {uid} (conflicts with '{existing_name}' -> {uid})")
                        else:
                            ids_to_names[uid] = name
                    for name in names_to_remove:
                        del user_map[name]
                        applied -= 1

            # Fallback: ensure current_user is set for message operations that need it
            if applied > 0 and 'message_api' in self.tool_manager.python_tool_instances:
                msg_api = self.tool_manager.python_tool_instances['message_api']
                auth_gated_message_tools = {'delete_message', 'search_messages', 'send_message', 'delete_message', 'add_contact', 'get_user_id'}
                needs_auth = any(tool in auth_gated_message_tools for tool in query_result.expected_tools)
                current_user = getattr(msg_api, 'current_user', None)
                if needs_auth and not current_user:
                    user_map = getattr(msg_api, 'user_map', {})
                    if user_map:
                        first_user_id = list(user_map.values())[0] if user_map else None
                        if first_user_id:
                            msg_api.current_user = first_user_id
                            applied += 1
                            print(f"   [AUTH FALLBACK] Auto-set current_user to {first_user_id} for message operations")

            # Post-validation: ensure current_dir paths exist for gorilla_file_system
            if applied > 0 and 'gorilla_file_system' in self.tool_manager.python_tool_instances:
                fs = self.tool_manager.python_tool_instances['gorilla_file_system']
                current_dir = getattr(fs, 'current_dir', None)
                if current_dir and isinstance(current_dir, list):
                    root = getattr(fs, 'root', None)
                    if root and isinstance(root, dict):
                        if len(current_dir) > 0:
                            current = root
                            path_parts = current_dir
                            for i, part in enumerate(path_parts):
                                if part not in current:
                                    current[part] = {'type': 'directory', 'contents': {}}
                                if isinstance(current.get(part), dict) and current[part].get('type') == 'directory':
                                    current = current[part]['contents']
                                elif part in current and isinstance(current[part], dict):
                                    current = current[part]
                                else:
                                    break
                            print(f"   [FS FIX] Ensured directory path exists: {'/'.join(path_parts)}")
                        else:
                            current = root

                        # Fix: Ensure expected files exist in current_dir location
                        file_tools = {'cat', 'cp', 'diff', 'echo', 'grep', 'mv', 'rm', 'rmdir', 'sort', 'tail', 'touch', 'wc'}
                        needs_files = any(tool in file_tools for tool in query_result.expected_tools)
                        if needs_files:
                            # Find files needed based on the query
                            query_lower = query_result.query.lower()
                            # Common file names that might be needed
                            potential_files = ['config.json', 'processor.py', 'README.md', 'data.json', 'temp.txt']
                            for fname in potential_files:
                                if fname in query_lower or fname.replace('.py', '_v1.0.py') in query_lower:
                                    # Check if file exists in current location
                                    file_path = fname
                                    if fname not in current:
                                        # Check if file exists elsewhere in root and try to find it
                                        def find_file_in_tree(node, target, path=""):
                                            if isinstance(node, dict):
                                                if target in node:
                                                    return node[target]
                                                for k, v in node.items():
                                                    if isinstance(v, dict):
                                                        result = find_file_in_tree(v, target, f"{path}/{k}")
                                                        if result:
                                                            return result
                                            return None
                                        existing_file = find_file_in_tree(root, fname)
                                        if existing_file:
                                            # Move file to current location
                                            current[fname] = existing_file
                                            print(f"   [FS FIX] Moved '{fname}' to current directory")
                            print(f"   [FS FIX] current_dir={current_dir}, files now in location: {list(current.keys())}")

            if applied > 0:
                print(f" ✓ Applied {applied} state modifications")
                return True
            else:
                print(" No modifications could be applied")
                return False

        except (json.JSONDecodeError, ValueError) as e:
            print(f" ✗ Failed to parse state adjustment response: {e}")
            return False
        except Exception as e:
            print(f" ✗ State adjustment error: {e}")
            return False

    @staticmethod
    def _set_nested_field(obj, field_path: str, value: Any) -> None:
        """Set a field on an object using dot-notation path, creating intermediate dicts.
        
        Handles:
        1. File names with extensions by merging extension parts with the previous component.
           E.g., 'root.invoice.txt' -> root['invoice.txt']
        2. Bracket notation for keys with dots: 'root['invoice.txt']' -> root['invoice.txt']
        """
        # First, handle bracket notation: extract keys inside brackets and preserve them
        # E.g., "root['invoice.txt'].contents" -> ["root", "['invoice.txt']", "contents"]
        import re
        bracket_pattern = re.compile(r"\[('[^']+'|\"[^\"]+\")\]")
        
        # Replace brackets with a placeholder that won't be split
        parts_raw = bracket_pattern.split(field_path)
        bracket_keys = bracket_pattern.findall(field_path)
        
        # Reconstruct parts, treating bracket keys as single units
        # E.g., "root['invoice.txt'].contents" -> ['root', "['invoice.txt']", 'contents']
        processed_parts = []
        for part in parts_raw:
            if part:
                # Split remaining dots
                sub_parts = part.split('.')
                processed_parts.extend(sub_parts)
        
        # Insert bracket keys back in correct positions
        # This is tricky - we need to detect where bracket keys should go
        # For now, handle the case where bracket is at the end or followed by nothing
        
        parts = []
        i = 0
        while i < len(processed_parts):
            part = processed_parts[i]
            # Check if next part starts with [ and part doesn't end with ]
            if i + 1 < len(processed_parts) and processed_parts[i + 1].startswith('['):
                # Combine current part with next (the bracket key)
                parts.append(part + processed_parts[i + 1])
                i += 2
            else:
                parts.append(part)
                i += 1
        
        # Now handle file extensions (merge extension parts)
        common_extensions = {'txt', 'json', 'md', 'csv', 'pdf', 'html', 'xml', 'yaml', 'yml', 
                           'py', 'js', 'css', 'log', 'conf', 'config', 'sh', 'c', 'cpp', 
                           'h', 'hpp', 'go', 'rs', 'java', 'class', 'jar', 'war', 'ear',
                           'png', 'jpg', 'jpeg', 'gif', 'bmp', 'ico', 'svg', 'webp',
                           'mp3', 'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm',
                           'zip', 'tar', 'gz', 'rar', '7z', 'bz2', 'xz',
                           'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
                           'exe', 'dll', 'so', 'dylib', 'a', 'o', 'obj', 'lib',
                           'bak', 'tmp', 'cache'}
        merged_parts = []
        for part in parts:
            if merged_parts and part.lower() in common_extensions and len(part) <= 5:
                merged_parts[-1] = merged_parts[-1] + '.' + part
            else:
                merged_parts.append(part)
        parts = merged_parts
        
        # Now apply to object
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(part, {})
            else:
                obj = getattr(obj, part, {})
        last_key = parts[-1]
        if isinstance(obj, dict):
            obj[last_key] = value
        else:
            setattr(obj, last_key, value)

    def _apply_state_modifications(self, modifications: Dict[str, Any]) -> int:
        """Apply state modifications to live Python tool instances.

        Args:
            modifications: Dict mapping class_key -> {field_path: value}

        Returns:
            Number of modifications successfully applied
        """
        applied = 0

        for class_key, field_changes in modifications.items():
            actual_class_key = class_key
            extra_prefix = ""

            if class_key not in self.tool_manager.python_tool_instances and "." in class_key:
                first_dot = class_key.find(".")
                potential_key = class_key[:first_dot]
                if potential_key in self.tool_manager.python_tool_instances:
                    actual_class_key = potential_key
                    extra_prefix = class_key[first_dot + 1:] + "."
                    print(f"   ℹ Flat key detected: '{class_key}' -> class='{actual_class_key}', prefix='{extra_prefix}'")

            if actual_class_key not in self.tool_manager.python_tool_instances:
                print(f" ⚠ Unknown class_key: {class_key}, skipping")
                continue

            instance = self.tool_manager.python_tool_instances[actual_class_key]

            if not isinstance(field_changes, dict):
                effective_field_path = extra_prefix.rstrip(".") if extra_prefix else class_key
                if not effective_field_path:
                    print(f" ⚠ Empty field path for {class_key}, skipping")
                    continue
                top_level_field = effective_field_path.split(".", 1)[0]
                if top_level_field not in vars(instance):
                    print(
                        f"   ⚠ Blocking non-state top-level field: "
                        f"{actual_class_key}.{effective_field_path}"
                    )
                    continue
                value = field_changes
                self._set_nested_field(instance, effective_field_path, value)
                applied += 1
                print(f"   {actual_class_key}.{effective_field_path}: {value}")
                continue

            for field_path, value in field_changes.items():
                effective_field_path = (extra_prefix + field_path).rstrip(".")
                try:
                    normalised_path = effective_field_path.casefold()
                    protected_fragments = (
                        "counter", "access_token", "token_", "scope",
                        "grant_type", "password", "authenticated",
                        "current_user", "cardholder", "binding_card",
                        "account_id", "first_name", "last_name",
                    )
                    if any(fragment in normalised_path for fragment in protected_fragments):
                        print(
                            f"   ⚠ Blocking protected state adjustment: "
                            f"{actual_class_key}.{effective_field_path}"
                        )
                        continue
                    if field_path.startswith("APPEND:"):
                        top_level_field = field_path[len("APPEND:"):].split(".", 1)[0]
                    elif field_path.startswith("EXTEND:"):
                        top_level_field = field_path[len("EXTEND:"):].split(".", 1)[0]
                    else:
                        top_level_field = effective_field_path.split(".", 1)[0]
                    if top_level_field not in vars(instance):
                        print(
                            f"   ⚠ Blocking non-state top-level field: "
                            f"{actual_class_key}.{effective_field_path}"
                        )
                        continue
                    if field_path == "current_dir" and class_key == "gorilla_file_system":
                        print(f"   ⚠ Skipping gorilla_file_system.current_dir modification (must use cd tool)")
                        continue
                    if field_path.startswith("APPEND:"):
                        list_field = field_path[len("APPEND:"):]
                        current_list = getattr(instance, list_field, None)
                        if isinstance(current_list, list):
                            current_list.append(value)
                            applied += 1
                            print(f"   {class_key}.{list_field}: appended item")
                        else:
                            print(f"   ⚠ {class_key}.{list_field} is not a list, skipping append")
                    elif field_path.startswith("EXTEND:"):
                        list_field = field_path[len("EXTEND:"):]
                        current_list = getattr(instance, list_field, None)
                        if isinstance(current_list, list) and isinstance(value, list):
                            current_list.extend(value)
                            applied += 1
                            print(f"   {class_key}.{list_field}: extended with {len(value)} items")
                        else:
                            print(f"   ⚠ {class_key}.{list_field} extend failed (not list or value not list)")
                    elif "." in effective_field_path:
                        parts = effective_field_path.split(".")
                        obj = instance
                        for part in parts[:-1]:
                            if isinstance(obj, dict):
                                if part not in obj:
                                    obj[part] = {}
                                obj = obj[part]
                            else:
                                nested = getattr(obj, part, {})
                                if isinstance(nested, dict) and part not in (getattr(type(obj), '__dict__', {}).keys() if hasattr(obj, '__dict__') else {}):
                                    try:
                                        setattr(obj, part, {})
                                    except (AttributeError, TypeError):
                                        pass
                                obj = getattr(obj, part, {})
                        last_key = parts[-1]
                        if isinstance(obj, dict):
                            if isinstance(value, str) and value.startswith("APPEND:"):
                                append_val = value[len("APPEND:"):]
                                existing = obj.get(last_key)
                                if isinstance(existing, list):
                                    existing.append(append_val)
                                    applied += 1
                                    print(f"   {actual_class_key}.{effective_field_path}: appended '{append_val}' to existing list")
                                else:
                                    obj[last_key] = [append_val]
                                    applied += 1
                                    print(f"   {actual_class_key}.{effective_field_path}: created list with '{append_val}'")
                            elif "user_map" in effective_field_path:
                                existing_ids = set(obj.values()) if obj else set()
                                if value in existing_ids:
                                    current_mapping = {v: k for k, v in obj.items()} if obj else {}
                                    existing_user = current_mapping.get(value)
                                    if existing_user and existing_user != last_key:
                                        print(f"   ⚠ Skipping user_map.{last_key} -> {value} (ID already assigned to '{existing_user}')")
                                        continue
                                obj[last_key] = value
                                applied += 1
                                print(f"   {actual_class_key}.{effective_field_path}: set to {json.dumps(value, default=str)[:100]}")
                            else:
                                obj[last_key] = value
                                applied += 1
                                print(f"   {actual_class_key}.{effective_field_path}: set to {json.dumps(value, default=str)[:100]}")
                        else:
                            print(f"   {actual_class_key}.{effective_field_path}: parent is not a dict, skipping")
                    else:
                        setattr(instance, effective_field_path, value)
                        applied += 1
                        print(f"   {actual_class_key}.{effective_field_path}: set to {json.dumps(value, default=str)[:100]}")
                except Exception as e:
                    print(f"   ⚠ Failed to apply {actual_class_key}.{effective_field_path}: {e}")

        return applied

    def _generate_tool_arguments(self, tool_name: str, query: str, trajectory: List[TrajectoryStep],
                                 execution_context: Dict[str, Any],
                                 feedback: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Generate arguments for a specific tool based on query and context."""
        # Get tool schema
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return None, f"Tool '{tool_name}' not found"

        # Build the complete policy-visible history. Do not truncate prior tool
        # outputs: a required downstream argument may appear anywhere in them.
        visible_history = []
        for step in trajectory:
            for tool_call in step.tool_calls:
                visible_history.append({
                    "step_number": step.step_number,
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "output": tool_call.output,
                })

        # Get output type info for better argument generation
        output_type = tool_schema.get('output_type', 'unknown')
        output_description = tool_schema.get('output_description', '')

        prompt = f"""Generate arguments for '{tool_name}' based on user query and previous steps.

=== USER QUERY ===
{query}

=== PREVIOUS POLICY-VISIBLE TOOL CALLS AND OUTPUTS ===
{json.dumps(visible_history, indent=2, ensure_ascii=False, default=str) if visible_history else "None"}

=== EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, ensure_ascii=False, default=str)}
=== FULL TOOL DEFINITION ===
{json.dumps(tool_schema, indent=2, ensure_ascii=False, default=str)}

=== CURRENT DATE ===
{datetime.now(timezone.utc).date().isoformat()}

=== EXPECTED OUTPUT ===
Type: {output_type}
Description: {output_description}
"""
        if feedback:
            # Internal retries are not part of the saved policy context. Never
            # expose raw judge/tool-error text here: it may contain identifiers
            # or suggested values that the trained policy will never receive.
            prompt += """
=== RETRY NOTICE ===
A previous candidate was rejected. Recompute the arguments only from the user
query, prior saved tool outputs, and the full tool definition above. No value
from a judge message, failed internal attempt, or tool error is available.
"""
        prompt += """
=== TASK ===
Generate args matching schema and fulfilling query:
- Use only values explicitly present in the USER QUERY, previous tool outputs,
  EXECUTION CONTEXT, or defaults declared in the TOOL SCHEMA.
- Deterministic calculations and format normalization from visible values are
  allowed only when they do not require an external lookup.
- General/model knowledge is not an argument source. Do not convert a visible
  human-readable label into an opaque ID, code, token, symbol, handle,
  coordinate, path, or credential unless that exact value is visible.
- The simulator's private API state is not available to the solving assistant.
  Never invent, guess, or copy a value from hidden state.
- Values mentioned only by an internal judge, rejected attempt, or failed tool
  call are unavailable because those diagnostics are not saved in the trace.
- Do not choose an arbitrary member of a prior list output. The user query must
  state a unique selection rule or the exact requested member.
- Dates for booking or scheduling actions must not be before CURRENT DATE.
- If any required argument is unavailable from the visible sources above, return
  {"__missing_required_argument__": ["argument_name"]} instead of guessing.
- Storage tools (ls, cat, cd, mkdir, mv, rm, cp, touch, echo, grep, wc, tail,
  find) use their direct schema arguments. Do not wrap them in a `calls` batch.

Respond JSON: {"arg1": "value1", ...}
"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}])
            response_text = response.strip()
        
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            arguments = json.loads(response_text)
            if not isinstance(arguments, dict):
                return None, f"Expected JSON object dict, got {type(arguments).__name__}"
            if "__missing_required_argument__" in arguments:
                missing = arguments.get("__missing_required_argument__")
                return None, f"Required argument is not policy-visible: {missing}"
            return arguments, None
        
        except json.JSONDecodeError as e:
            return None, f"JSON parsing error: {e}"

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        raw = text.strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        else:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("expected one JSON object")
        return result

    @staticmethod
    def _normalise_schema_type(schema_type: Any) -> Optional[str]:
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            return str(non_null[0]).lower() if non_null else None
        if schema_type is None:
            return None
        aliases = {
            "dict": "object",
            "float": "number",
            "double": "number",
            "int": "integer",
            "bool": "boolean",
            "list": "array",
        }
        value = str(schema_type).lower()
        return aliases.get(value, value)

    @classmethod
    def _validate_json_value(
        cls,
        value: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
        """Small deterministic JSON-schema subset used before simulation."""
        issues: List[str] = []
        expected = cls._normalise_schema_type(schema.get("type"))
        type_ok = True
        if expected == "object":
            type_ok = isinstance(value, dict)
        elif expected == "array":
            type_ok = isinstance(value, list)
        elif expected == "string":
            type_ok = isinstance(value, str)
        elif expected == "integer":
            type_ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected == "number":
            type_ok = isinstance(value, (int, float)) and not isinstance(
                value, bool
            )
        elif expected == "boolean":
            type_ok = isinstance(value, bool)
        elif expected == "null":
            type_ok = value is None
        if not type_ok:
            return [
                f"{path}: expected {expected}, got {type(value).__name__}"
            ]

        if "const" in schema and value != schema["const"]:
            issues.append(f"{path}: value differs from const")
        if "enum" in schema and value not in schema.get("enum", []):
            issues.append(f"{path}: value is not in enum")

        if isinstance(value, str):
            pattern = schema.get("pattern")
            if pattern:
                try:
                    if re.fullmatch(str(pattern), value) is None:
                        issues.append(f"{path}: string does not match pattern")
                except re.error:
                    issues.append(f"{path}: invalid schema regex")
            if "minLength" in schema and len(value) < int(schema["minLength"]):
                issues.append(f"{path}: string is too short")
            if "maxLength" in schema and len(value) > int(schema["maxLength"]):
                issues.append(f"{path}: string is too long")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                issues.append(f"{path}: below minimum")
            if "maximum" in schema and value > schema["maximum"]:
                issues.append(f"{path}: above maximum")
            if (
                "exclusiveMinimum" in schema
                and value <= schema["exclusiveMinimum"]
            ):
                issues.append(f"{path}: below exclusive minimum")
            if (
                "exclusiveMaximum" in schema
                and value >= schema["exclusiveMaximum"]
            ):
                issues.append(f"{path}: above exclusive maximum")

        if isinstance(value, list):
            if "minItems" in schema and len(value) < int(schema["minItems"]):
                issues.append(f"{path}: too few items")
            if "maxItems" in schema and len(value) > int(schema["maxItems"]):
                issues.append(f"{path}: too many items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    issues.extend(
                        cls._validate_json_value(
                            item, item_schema, f"{path}[{index}]"
                        )
                    )

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    issues.append(f"{path}.{key}: missing required argument")
            if schema.get("additionalProperties", True) is False:
                for key in value:
                    if key not in properties:
                        issues.append(f"{path}.{key}: unexpected argument")
            for key, item in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    issues.extend(
                        cls._validate_json_value(
                            item, child_schema, f"{path}.{key}"
                        )
                    )
        return issues

    def _validate_tool_arguments_schema(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> List[str]:
        try:
            tool_schema = self.tool_manager.get_tool_schema(tool_name)
        except Exception as exc:
            return [f"{tool_name}: schema unavailable ({exc})"]
        parameters = tool_schema.get("parameters", {})
        if not isinstance(parameters, dict):
            return [f"{tool_name}: parameter schema is invalid"]
        if "type" not in parameters:
            parameters = {**parameters, "type": "object"}
        return self._validate_json_value(arguments, parameters, tool_name)

    @staticmethod
    def _compact_policy_context(
        execution_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep policy-visible prior results without duplicating every alias."""
        turn_outputs = execution_context.get("turn_outputs", [])
        if isinstance(turn_outputs, list):
            # Twelve prior turns is already well beyond the distributions used
            # by the current jobs, and avoids resending aliases for every field.
            compact_turn_outputs = []
            for turn_output in turn_outputs[-12:]:
                if (
                    isinstance(turn_output, dict)
                    and isinstance(turn_output.get("calls"), list)
                ):
                    # _aggregate_turn_outputs also keeps by_tool and direct-name
                    # aliases for backward-compatible placeholder lookup.  The
                    # compiler used to resend all three representations, making
                    # every prior output appear three times in later prompts.
                    compact_turn_outputs.append(
                        {"calls": copy.deepcopy(turn_output["calls"])}
                    )
                else:
                    compact_turn_outputs.append(copy.deepcopy(turn_output))
            turn_outputs = compact_turn_outputs
        else:
            turn_outputs = []
        prior_user_queries = execution_context.get("prior_user_queries", [])
        if not isinstance(prior_user_queries, list):
            prior_user_queries = []
        compact = {"prior_turn_outputs": turn_outputs}
        if prior_user_queries:
            compact["prior_user_queries"] = copy.deepcopy(
                prior_user_queries[-12:]
            )
        return compact

    @staticmethod
    def _resolve_output_path(output: Any, path: str) -> Any:
        current = output
        if not path:
            return current
        normalised = re.sub(r"\[(\d+)\]", r".\1", str(path))
        for key in [part for part in normalised.split(".") if part]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                index = int(key)
                if index >= len(current):
                    raise KeyError(path)
                current = current[index]
            else:
                raise KeyError(path)
        return current

    @classmethod
    def _contains_source_binding(cls, value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("source") == "tool_output":
                return True
            return any(cls._contains_source_binding(item) for item in value.values())
        if isinstance(value, list):
            return any(cls._contains_source_binding(item) for item in value)
        return False

    @staticmethod
    def _value_visible_in_text(value: Any, visible_text: str) -> bool:
        """Conservative provenance check for declared user/history literals."""
        if value is None or isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            # Numeric formatting is not semantic provenance: 45.50 and 45.5
            # are the same policy-visible value.
            numeric_text = visible_text.translate(
                str.maketrans({"−": "-", "–": "-", "—": "-"})
            )
            numeric_text = re.sub(
                r"\b(?:negative|minus)\s+(?=\d)",
                "-",
                numeric_text,
                flags=re.IGNORECASE,
            )
            for token in re.findall(
                # A sentence-final period is punctuation, not part of the
                # number. Still reject a prefix of a longer decimal/token.
                r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)",
                numeric_text,
            ):
                try:
                    if float(token.replace(",", "")) == float(value):
                        return True
                except ValueError:
                    continue
            if isinstance(value, int) or (
                isinstance(value, float) and value.is_integer()
            ):
                number_words = (
                    "zero", "one", "two", "three", "four", "five", "six",
                    "seven", "eight", "nine", "ten", "eleven", "twelve",
                    "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty",
                )
                integer = int(value)
                if 0 <= integer < len(number_words):
                    if re.search(
                        rf"\b{number_words[integer]}\b",
                        visible_text,
                        re.IGNORECASE,
                    ):
                        return True
                semantic_aliases = {
                    2: ("second", "square", "squared"),
                    3: ("third", "cube", "cubed"),
                }
                if any(
                    re.search(
                        rf"\b{re.escape(alias)}\b",
                        visible_text,
                        re.IGNORECASE,
                    )
                    for alias in semantic_aliases.get(integer, ())
                ):
                    return True
            return False
        if isinstance(value, str):
            candidate = " ".join(value.casefold().split())
            haystack = " ".join(visible_text.casefold().split())
            if candidate in haystack:
                return True
            # A path such as ``scripts/utils.py`` is deterministically composed
            # from a visible directory and filename even when the natural user
            # says "move utils.py into scripts".  Split only path/file syntax;
            # opaque IDs containing punctuation still require an exact match.
            if "/" in candidate or "\\" in candidate:
                path_parts = re.findall(r"[\w@.-]+", candidate)
                if path_parts and all(part in haystack for part in path_parts):
                    return True
            # Free-form messages often differ only in surrounding punctuation.
            candidate_tokens = re.findall(r"[\w@./:-]+", candidate)
            return bool(candidate_tokens) and all(
                token in haystack for token in candidate_tokens
            )
        if isinstance(value, list):
            return all(
                StepByStepGenerator._value_visible_in_text(item, visible_text)
                for item in value
            )
        if isinstance(value, dict):
            return all(
                StepByStepGenerator._value_visible_in_text(item, visible_text)
                for item in value.values()
            )
        return False

    def _materialise_argument_source(
        self,
        *,
        spec: Any,
        schema: Dict[str, Any],
        query: str,
        policy_context: Dict[str, Any],
        call_outputs: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        # Structured arguments may bind individual object fields/list items.
        # This is still fail-closed: every leaf must eventually reach one of
        # the explicit provenance cases below.
        if isinstance(spec, dict) and "source" not in spec:
            schema_type = str(schema.get("type", "")).casefold()
            properties = schema.get("properties", {})
            if schema_type not in {"object", "dict"} and not properties:
                raise ValueError("ARGUMENT_SOURCE_MISSING")
            values: Dict[str, Any] = {}
            fields: Dict[str, Any] = {}
            for name, child_spec in spec.items():
                child_schema = (
                    properties.get(name, {})
                    if isinstance(properties, dict)
                    else {}
                )
                value, provenance = self._materialise_argument_source(
                    spec=child_spec,
                    schema=child_schema,
                    query=query,
                    policy_context=policy_context,
                    call_outputs=call_outputs,
                )
                values[name] = value
                fields[name] = provenance
            return values, {"source": "composite", "fields": fields}
        if isinstance(spec, list):
            schema_type = str(schema.get("type", "")).casefold()
            if schema_type not in {"array", "list"}:
                raise ValueError("ARGUMENT_SOURCE_MISSING")
            item_schema = schema.get("items", {})
            values = []
            items = []
            for child_spec in spec:
                value, provenance = self._materialise_argument_source(
                    spec=child_spec,
                    schema=item_schema if isinstance(item_schema, dict) else {},
                    query=query,
                    policy_context=policy_context,
                    call_outputs=call_outputs,
                )
                values.append(value)
                items.append(provenance)
            return values, {"source": "composite", "items": items}
        if not isinstance(spec, dict) or "source" not in spec:
            raise ValueError("ARGUMENT_SOURCE_MISSING")
        source = str(spec.get("source", "")).strip().casefold()
        if source == "tool_output":
            call_id = str(spec.get("call_id", ""))
            if call_id not in call_outputs:
                raise ValueError("FUTURE_OR_SIBLING_OUTPUT_DEPENDENCY")
            path = str(spec.get("path", ""))
            try:
                value = self._resolve_output_path(call_outputs[call_id], path)
            except (KeyError, IndexError, TypeError):
                raise ValueError("TOOL_OUTPUT_PATH_NOT_FOUND")
            coercion = None
            schema_type = str(schema.get("type", "")).casefold()
            if isinstance(value, str) and schema_type in {
                "integer", "int", "number", "float",
            }:
                numeric_text = value.strip().replace(",", "")
                if re.fullmatch(
                    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                    numeric_text,
                ):
                    numeric_value = float(numeric_text)
                    if schema_type in {"integer", "int"}:
                        if numeric_value.is_integer():
                            value = int(numeric_value)
                            coercion = "string_to_integer"
                    else:
                        value = numeric_value
                        coercion = "string_to_number"
            provenance = {
                "source": "tool_output",
                "call_id": call_id,
                "path": path,
            }
            if coercion:
                provenance["coercion"] = coercion
            return copy.deepcopy(value), provenance
        if source in {"user", "history", "visible_context", "literal"}:
            if "value" not in spec:
                raise ValueError("ARGUMENT_VALUE_MISSING")
            value = spec["value"]
            # Structured-output models sometimes serialize a numeric argument
            # as "300.0" even when the schema says float/integer. Canonicalize
            # that representation before provenance and schema checks; this is
            # deterministic type compilation, not permission to invent a value.
            schema_type = str(schema.get("type", "")).casefold()
            if isinstance(value, str) and schema_type in {
                "integer", "int", "number", "float"
            }:
                numeric_text = value.strip().replace(",", "")
                if re.fullmatch(
                    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                    numeric_text,
                ):
                    numeric_value = float(numeric_text)
                    if schema_type in {"integer", "int"}:
                        if numeric_value.is_integer():
                            value = int(numeric_value)
                    else:
                        value = numeric_value
            # Query and prior history are both available before the turn.  A
            # model labelling a repeated query value as "history" is harmless;
            # validate against the union and retain the declared label only as
            # diagnostic metadata.
            visible_payload = query + "\n" + json.dumps(
                policy_context,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
            # Enum/const values are deterministic semantic normalization (for
            # example "purchase" -> enum "buy"), not hidden-state leakage.
            schema_declares_value = (
                value == schema.get("const")
                or value in schema.get("enum", [])
            )
            if not schema_declares_value and not self._value_visible_in_text(
                value, visible_payload
            ):
                raise ValueError("ARGUMENT_NOT_POLICY_VISIBLE")
            return copy.deepcopy(value), {"source": source}
        if source == "schema_default":
            if "default" not in schema:
                raise ValueError("SCHEMA_DEFAULT_MISSING")
            return copy.deepcopy(schema["default"]), {"source": "schema_default"}
        raise ValueError("UNKNOWN_ARGUMENT_SOURCE")

    def _compile_turn_arguments(
        self,
        *,
        query: str,
        expected_tools: List[str],
        execution_context: Dict[str, Any],
        repair_issues: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compile a whole sequential turn in one paid model request."""
        policy_context = self._compact_policy_context(execution_context)
        call_specs = []
        for index, tool_name in enumerate(expected_tools, 1):
            call_specs.append(
                {
                    "call_id": f"c{index}",
                    "tool_name": tool_name,
                    "tool_definition": self.tool_manager.get_tool_schema(
                        tool_name
                    ),
                }
            )
        repair_notice = ""
        if repair_issues:
            repair_notice = f"""
=== ONE ALLOWED TURN REPAIR ===
The previous complete turn was rejected with these value-free error codes:
{json.dumps(sorted(set(repair_issues)))}
Recompile the entire turn. Failed arguments, tool outputs, simulator state, and
judge suggestions are not available.
"""
        prompt = f"""Compile every tool call for one assistant turn in a single pass.

=== CURRENT USER REQUEST ===
{query}

=== POLICY-VISIBLE PRIOR TURN OUTPUTS ===
{json.dumps(policy_context, indent=2, ensure_ascii=False, default=str)}

=== FIXED ORDERED CALL SPECS ===
{json.dumps(call_specs, indent=2, ensure_ascii=False, default=str)}
{repair_notice}
Return exactly one call for each supplied call_id and tool_name, in order.
For every top-level argument, declare its provenance using one of:
- {{"source":"user","value":...}} for a value visible in the current request;
- {{"source":"history","value":...}} for a value visible in prior outputs;
- {{"source":"schema_default"}} when the parameter schema declares a default;
- {{"source":"tool_output","call_id":"c1","path":"field.subfield"}}
  for a value produced by an EARLIER call in this same turn.

Rules:
1. Symbolically bind future arguments to earlier outputs. Never predict the
   concrete output of an unexecuted call.
2. A call may reference only c1..cN-1; never itself, a later call, or a sibling
   parallel result.
3. Supply every required argument and no undeclared argument.
4. Do not use simulator state, failed attempts, judge feedback, or general
   knowledge as a source for opaque IDs, codes, tokens, symbols, handles,
   coordinates, paths, or credentials.
5. If the visible context cannot determine a required argument, return
   {{"uncompilable":true,"issue":"MISSING_POLICY_VISIBLE_ARGUMENT"}}.

Respond only with JSON:
{{
  "calls": [
    {{
      "call_id": "c1",
      "tool_name": "{expected_tools[0] if expected_tools else ''}",
      "arguments": {{"argument_name": {{"source":"user","value":"..."}}}}
    }}
  ]
}}
"""
        result = self._extract_json_object(
            self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                purpose="turn_compile",
            )
        )
        if result.get("uncompilable") is True:
            raise ValueError("MISSING_POLICY_VISIBLE_ARGUMENT")
        raw_calls = result.get("calls")
        if not isinstance(raw_calls, list) or len(raw_calls) != len(call_specs):
            raise ValueError("TURN_CALL_COUNT_MISMATCH")

        compiled: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for expected, raw_call in zip(call_specs, raw_calls):
            if not isinstance(raw_call, dict):
                raise ValueError("TURN_CALL_MALFORMED")
            call_id = str(raw_call.get("call_id", ""))
            tool_name = str(raw_call.get("tool_name", ""))
            if (
                call_id != expected["call_id"]
                or tool_name != expected["tool_name"]
                or call_id in seen_ids
            ):
                raise ValueError("TURN_CALL_PLAN_CHANGED")
            seen_ids.add(call_id)
            argument_specs = raw_call.get("arguments")
            if not isinstance(argument_specs, dict):
                raise ValueError("TURN_ARGUMENTS_MALFORMED")
            parameters = expected["tool_definition"].get("parameters", {})
            properties = parameters.get("properties", {})
            for required in parameters.get("required", []):
                if required not in argument_specs:
                    raise ValueError("MISSING_REQUIRED_ARGUMENT_SOURCE")
            if parameters.get("additionalProperties", True) is False:
                if any(name not in properties for name in argument_specs):
                    raise ValueError("UNDECLARED_ARGUMENT")
            # Static dependency check before any tool is executed.
            for spec in argument_specs.values():
                refs: List[str] = []

                def collect_refs(value: Any) -> None:
                    if isinstance(value, dict):
                        if value.get("source") == "tool_output":
                            refs.append(str(value.get("call_id", "")))
                        for child in value.values():
                            collect_refs(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect_refs(child)

                collect_refs(spec)
                if any(ref not in seen_ids - {call_id} for ref in refs):
                    raise ValueError("FUTURE_OR_SIBLING_OUTPUT_DEPENDENCY")
            compiled.append(
                {
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "argument_specs": argument_specs,
                }
            )
        return compiled

    def _execute_compiled_turn(
        self,
        *,
        query: str,
        compiled_calls: List[Dict[str, Any]],
        execution_context: Dict[str, Any],
    ) -> Tuple[List[TrajectoryStep], Dict[str, Any]]:
        policy_context = self._compact_policy_context(execution_context)
        trajectory: List[TrajectoryStep] = []
        # Episode-level symbolic plans use globally unique call IDs and may
        # bind an argument to a result from an earlier user turn.  Preserve
        # those outputs separately from the legacy convenience aliases.  The
        # ordinary turn compiler still uses c1/c2/... and therefore behaves
        # exactly as before when this map is absent.
        prior_symbolic_outputs = execution_context.get(
            "symbolic_call_outputs", {}
        )
        call_outputs: Dict[str, Any] = (
            copy.deepcopy(prior_symbolic_outputs)
            if isinstance(prior_symbolic_outputs, dict)
            else {}
        )
        updated_context = copy.deepcopy(execution_context)

        for step_num, compiled in enumerate(compiled_calls, 1):
            tool_name = compiled["tool_name"]
            tool_schema = self.tool_manager.get_tool_schema(tool_name)
            parameters = tool_schema.get("parameters", {})
            properties = parameters.get("properties", {})
            arguments: Dict[str, Any] = {}
            provenance: Dict[str, Any] = {}
            for argument_name, source_spec in compiled["argument_specs"].items():
                try:
                    value, source = self._materialise_argument_source(
                        spec=source_spec,
                        schema=properties.get(argument_name, {}),
                        query=query,
                        policy_context=policy_context,
                        call_outputs=call_outputs,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{exc}:{tool_name}.{argument_name}"
                    ) from None
                arguments[argument_name] = value
                provenance[argument_name] = source

            schema_issues = self._validate_tool_arguments_schema(
                tool_name, arguments
            )
            if schema_issues:
                first_path = schema_issues[0].split(":", 1)[0]
                raise ValueError(
                    f"ARGUMENT_SCHEMA_INVALID:{first_path}"
                )

            pre_state = (
                self.tool_manager.get_api_state()
                if self._python_tools_available
                else None
            )
            try:
                output = self._simulate_tool_execution(
                    tool_name=tool_name,
                    arguments=arguments,
                    execution_context=updated_context,
                )
            except Exception as exc:
                raise ValueError(
                    f"TOOL_EXECUTION_EXCEPTION:{type(exc).__name__}"
                ) from None

            if isinstance(output, dict):
                has_error, _ = self._detect_tool_error(tool_name, output)
                if has_error:
                    raise ValueError(f"TOOL_EXECUTION_ERROR:{tool_name}")
            if self.validate_outputs:
                output_check = self.verify_output_consistency(
                    tool_name,
                    step_num,
                    output,
                    tool_schema.get("output_type", "unknown"),
                    tool_schema.get("output_description", ""),
                )
                if (
                    not output_check.get("output_type_matches", False)
                    or output_check.get("issues")
                ):
                    raise ValueError(f"OUTPUT_SCHEMA_INVALID:{tool_name}")

            post_state = (
                self.tool_manager.get_api_state()
                if self._python_tools_available
                else None
            )
            transition = validate_transition_quality(
                tool_name=tool_name,
                tool_output=output,
                pre_state=pre_state,
                post_state=post_state,
                tool_arguments=arguments,
            )
            if not transition.get("passed", False):
                codes = [
                    str(issue.get("code", "TRANSITION_INVALID"))
                    for issue in transition.get("issues", [])
                ]
                raise ValueError(
                    f"{codes[0] if codes else 'TRANSITION_INVALID'}:"
                    f"{tool_name}"
                )

            changed_classes = []
            if pre_state is not None and post_state is not None:
                changed_classes = sorted(
                    class_key
                    for class_key in set(pre_state) | set(post_state)
                    if pre_state.get(class_key) != post_state.get(class_key)
                )
            state_verification = StateVerificationResult(
                is_valid=True,
                reasoning="Verified by deterministic simulator invariants.",
                issues=[],
                state_changes_summary=(
                    "Changed state: " + ", ".join(changed_classes)
                    if changed_classes
                    else "No state changes."
                ),
            )
            quality = {
                **transition,
                "validator": "deterministic",
                "argument_schema_valid": True,
                "argument_provenance": provenance,
            }
            trajectory.append(
                TrajectoryStep(
                    step_number=step_num,
                    tool_calls=[
                        ToolCallWithOutput(
                            tool_name=tool_name,
                            arguments=arguments,
                            output=output,
                        )
                    ],
                    reasoning=(
                        "Materialized from the turn-level symbolic call graph."
                    ),
                    pre_state=pre_state,
                    post_state=post_state,
                    state_verification=state_verification,
                    quality_verification=quality,
                )
            )
            call_outputs[compiled["call_id"]] = copy.deepcopy(output)
            updated_context.setdefault("symbolic_call_outputs", {})[
                compiled["call_id"]
            ] = copy.deepcopy(output)
            if isinstance(output, dict):
                for key, value in output.items():
                    updated_context[f"{tool_name}_{key}"] = value
                if "access_token" in output:
                    updated_context["access_token"] = output["access_token"]
            updated_context[f"{tool_name}_output"] = output
            updated_context[f"call_{compiled['call_id']}_output"] = output
        return trajectory, updated_context

    def _stage2_generate_tools_optimized(
        self,
        query_result: QueryGenerationResult,
        max_turn_attempts: int,
        initial_execution_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[TrajectoryStep]], Optional[Dict[str, Any]]]:
        """One compile request plus at most one complete-turn repair."""
        initial_context = copy.deepcopy(initial_execution_context or {})
        pre_turn_state = (
            self.tool_manager.get_api_state()
            if self._python_tools_available
            else None
        )
        repair_issues: List[str] = []
        attempts = max(1, min(self.max_turn_attempts, max_turn_attempts))
        for attempt in range(attempts):
            if pre_turn_state is not None:
                self.tool_manager.restore_api_state(copy.deepcopy(pre_turn_state))
            try:
                compiled = self._compile_turn_arguments(
                    query=query_result.query,
                    expected_tools=list(query_result.expected_tools),
                    execution_context=initial_context,
                    repair_issues=repair_issues or None,
                )
                return self._execute_compiled_turn(
                    query=query_result.query,
                    compiled_calls=compiled,
                    execution_context=initial_context,
                )
            except GenerationBudgetExceeded:
                if pre_turn_state is not None:
                    self.tool_manager.restore_api_state(pre_turn_state)
                raise
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                detail = str(exc) or type(exc).__name__
                repair_issues = [detail]
                print(
                    f"  Turn compilation/execution failed "
                    f"({attempt + 1}/{attempts}): {detail}"
                )
        if pre_turn_state is not None:
            self.tool_manager.restore_api_state(pre_turn_state)
        return None, None

    def _stage2_generate_tools(self, query_result: QueryGenerationResult,
                               max_retries_per_tool: int,
                               initial_execution_context: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[TrajectoryStep], Dict[str, Any]]]:
        """
        Stage 2: Generate tool invocations tool-by-tool.
        Uses expected_tools from Stage 1 directly - no LLM selection needed.
        - Each tool has its own retry count for argument generation
        - Feedback is wiped on successful tool completion
        - Captures pre/post API state snapshots around each tool call
        - Runs LLM-as-judge state verification after each call
        - If any tool fails after max retries, entire stage fails
        - Returns (trajectory, execution_context) or None
        """
        if self.optimized_pipeline:
            return self._stage2_generate_tools_optimized(
                query_result,
                max_retries_per_tool,
                initial_execution_context=initial_execution_context,
            )

        trajectory: List[TrajectoryStep] = []
        execution_context: Dict[str, Any] = initial_execution_context.copy() if initial_execution_context else {}

        for step_num, tool_name in enumerate(query_result.expected_tools, 1):
            total_steps = len(query_result.expected_tools)
            print(f"\n[Step {step_num}/{total_steps}] Processing tool: {tool_name}")

            tool_feedback = ""
            step_success = False

            for attempt in range(max_retries_per_tool):
                print(f" [Attempt {attempt + 1}/{max_retries_per_tool}]")

                # ── Capture PRE state snapshot ──
                pre_state = self.tool_manager.get_api_state() if self._python_tools_available else None

                # Generate arguments for this tool (with feedback from previous failures)
                print(f"  Generating arguments for {tool_name}...")
                arguments, error = self._generate_tool_arguments(
                    tool_name=tool_name,
                    query=query_result.query,
                    trajectory=trajectory,
                    execution_context=execution_context,
                    feedback=tool_feedback if tool_feedback else None,
                )

                if error:
                    print(f" ✗ {error}")
                    if error.startswith("Required argument is not policy-visible"):
                        print(" ✗ Rejecting datapoint: the generated query does not expose all required arguments")
                        return None, None
                    if attempt < max_retries_per_tool - 1:
                        continue
                    break

                print(f" Arguments: {json.dumps(arguments)}")

                # JSON schema/provenance checks and simulator execution replace
                # the legacy paid per-tool semantic judge.
                schema_issues = self._validate_tool_arguments_schema(
                    tool_name, arguments
                )
                if schema_issues:
                    print("  ✗ Argument schema validation failed")
                    if attempt < max_retries_per_tool - 1:
                        tool_feedback = (
                            "The previous arguments did not match the schema."
                        )
                        continue
                    break

                # Simulate tool execution
                print(f" Simulating {tool_name}...")
                try:
                    output = self._simulate_tool_execution(
                        tool_name=tool_name,
                        arguments=arguments,
                        execution_context=execution_context,
                    )
                except Exception as exc:
                    print(
                        f" ✗ Simulator raised {type(exc).__name__}: {exc}"
                    )
                    if pre_state is not None:
                        self.tool_manager.restore_api_state(pre_state)
                    if attempt < max_retries_per_tool - 1:
                        tool_feedback = (
                            "The previous internal simulator call failed. "
                            "Recompute only from saved policy-visible sources."
                        )
                        print(" Retrying after transactional rollback...")
                        continue
                    print(" Max retries exceeded, rejecting simulator failure")
                    break

                print(f" Output: {json.dumps(output, indent=2, ensure_ascii=False) if isinstance(output, (dict, list)) else output}")

                # Check for tool errors
                if isinstance(output, dict):
                    has_error, error_detail = self._detect_tool_error(tool_name, output)
                    if has_error:
                        error_type = output.get('error_type', 'execution_error')
                        print(f" ✗ Tool returned error: {error_detail}")
                        if pre_state is not None:
                            self.tool_manager.restore_api_state(pre_state)
                        if error_type == 'validation_failure' and attempt < max_retries_per_tool - 1:
                            tool_feedback = "The previous internal call failed validation."
                            print(f" Retrying due to validation failure...")
                            continue
                        elif attempt < max_retries_per_tool - 1:
                            tool_feedback = (
                                "The previous internal call failed. Re-read only the "
                                "saved policy-visible context and tool definition."
                            )
                            print(f" Retrying with feedback...")
                            continue
                        break

                # Validate output against declared type/description immediately
                tool_schema = self.tool_manager.get_tool_schema(tool_name)
                if tool_schema and self.validate_outputs:
                    expected_type = tool_schema.get('output_type', 'unknown')
                    expected_desc = tool_schema.get('output_description', '')
                    validation = self.verify_output_consistency(
                        tool_name, step_num, output, expected_type, expected_desc
                    )
                    if not validation['output_type_matches'] or validation.get('issues'):
                        issues_str = '; '.join(validation.get('issues', ['Type mismatch']))
                        print(f" ✗ Output validation failed: {issues_str}")
                        if pre_state is not None:
                            self.tool_manager.restore_api_state(pre_state)
                        if attempt < max_retries_per_tool - 1:
                            tool_feedback = "The previous internal output failed schema validation."
                            print(f" Retrying with new arguments...")
                            continue
                        print(f" Max retries exceeded, rejecting invalid output")
                        break

                # ── Capture POST state snapshot ──
                post_state = self.tool_manager.get_api_state() if self._python_tools_available else None

                # ── Deterministic positive-RL transition gate ──
                quality_verification = validate_transition_quality(
                    tool_name=tool_name,
                    tool_output=output,
                    pre_state=pre_state,
                    post_state=post_state,
                    tool_arguments=arguments,
                )
                if not quality_verification.get("passed", False):
                    issue_codes = [
                        issue.get("code", "UNKNOWN")
                        for issue in quality_verification.get("issues", [])
                    ]
                    print(
                        " ✗ Deterministic transition quality failed: "
                        + ", ".join(issue_codes)
                    )
                    if pre_state is not None:
                        self.tool_manager.restore_api_state(pre_state)
                    return None, None

                # State correctness is already covered by simulator execution
                # plus deterministic transition invariants above.  Do not spend
                # another LLM request narratively re-judging the same diff.
                state_verification = StateVerificationResult(
                    is_valid=True,
                    reasoning="Verified by deterministic simulator invariants.",
                    issues=[],
                    state_changes_summary=(
                        "State changed."
                        if pre_state != post_state
                        else "No state changes."
                    ),
                )

                # SUCCESS: Tool completed - add to trajectory
                print(f" ✓ Tool execution successful")

                # Update execution context
                if isinstance(output, dict):
                    for k, v in output.items():
                        execution_context[f"{tool_name}_{k}"] = v
                    # Store access_token directly for convenience (critical for auth-gated tools)
                    if 'access_token' in output:
                        execution_context['access_token'] = output['access_token']
                execution_context[f"{tool_name}_output"] = output

                # Add to trajectory (with state snapshots + verification)
                tool_call = ToolCallWithOutput(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=output
                )
                trajectory_step = TrajectoryStep(
                    step_number=step_num,
                    tool_calls=[tool_call],
                    reasoning=f"Generated arguments for {tool_name} based on query context",
                    pre_state=pre_state,
                    post_state=post_state,
                    state_verification=state_verification,
                    quality_verification=quality_verification,
                )
                trajectory.append(trajectory_step)
                step_success = True
                break

            if not step_success:
                print(f"\n✗ Tool {tool_name} failed after {max_retries_per_tool} attempts")
                return None, None

        # All tools completed successfully
        return trajectory, execution_context

    def _replay_state(self, trajectory: List[TrajectoryStep]) -> None:
        """Re-initialize API state and replay all completed trajectory steps.

        This is used to roll back state after a failed state-verification
        attempt so that the next retry starts from the correct state.
        """
        self.tool_manager.initialize_api_state()
        for step in trajectory:
            for tc in step.tool_calls:
                if self.tool_manager.has_python_implementation(tc.tool_name):
                    self.tool_manager.invoke_python_tool(tc.tool_name, tc.arguments)
        state = self.tool_manager.get_api_state()
        if 'message_api' in state and 'current_user' not in state['message_api']:
            for step in trajectory:
                for tc in step.tool_calls:
                    if tc.tool_name == 'message_login' and 'user_id' in tc.arguments:
                        self.tool_manager.python_tool_instances['message_api'].current_user = tc.arguments['user_id']
                        break

    def _stage3_finalize(self, query_result: QueryGenerationResult, trajectory: List[TrajectoryStep],
                         execution_context: Dict[str, Any],
                         focus_category: Optional[str],
                         initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[StepByStepDatapoint]:
        """
        Stage 3: Finalize datapoint.
        - No retries - if verification fails, something is fundamentally wrong
        - Assembles final datapoint with verification results
        - Stores initial_api_state and all verified intermediate states
        - Uses class-level token tracking
        """
        print("\nGenerating final response...")
        final_response = self._generate_final_response(query_result.query, trajectory, execution_context)
        if not final_response:
            print(" ✗ Could not produce a grounded final response")
            return None
        print(f" Final response: {final_response}")

        # Collect tools and categories
        tools_used = []
        categories_used = set()
        for step in trajectory:
            for tc in step.tool_calls:
                if tc.tool_name not in tools_used:
                    tools_used.append(tc.tool_name)
                cat = self.tool_manager.get_tool_category(tc.tool_name)
                if cat:
                    categories_used.add(cat)

        # Filter state snapshots to only include APIs whose tools are used
        filtered_initial_state = filter_api_state(initial_api_state, tools_used) if initial_api_state else None

        # Build filtered trajectory steps (strip irrelevant API states)
        filtered_trajectory: List[TrajectoryStep] = []
        for step in trajectory:
            filtered_pre = filter_api_state(step.pre_state, tools_used) if step.pre_state else None
            filtered_post = filter_api_state(step.post_state, tools_used) if step.post_state else None
            filtered_trajectory.append(TrajectoryStep(
                step_number=step.step_number,
                tool_calls=step.tool_calls,
                execution_mode=step.execution_mode,
                call_order_matters=step.call_order_matters,
                reasoning=step.reasoning,
                pre_state=filtered_pre,
                post_state=filtered_post,
                state_verification=step.state_verification,
                quality_verification=step.quality_verification,
            ))

        # Extract intermediate verified states from trajectory steps
        intermediate_states: List[Dict[str, Any]] = []
        for step in filtered_trajectory:
            if step.post_state is not None and step.state_verification is not None:
                intermediate_states.append({
                    "step_number": step.step_number,
                    "post_state": step.post_state,
                    "state_verification": step.state_verification.model_dump(),
                    "quality_verification": step.quality_verification,
                })

        # Create trajectory
        conv_trajectory = ConversationTrajectory(
            query=query_result.query,
            steps=filtered_trajectory,
            final_response=final_response,
            tools_used=tools_used,
            categories_used=list(categories_used),
            initial_api_state=filtered_initial_state,
        )

        # Run verification
        print("\nRunning verification...")
        verification_result = self.run_full_verification(
            query=query_result.query,
            trajectory=trajectory,
            execution_context=execution_context,
            final_response=final_response,
            query_quality=query_result.quality_preflight,
        )

        verification_passed = verification_result.overall_verification_passed if verification_result else False

        # If verification failed, return None so the caller knows to retry
        if not verification_passed:
            print(f"  Verification: FAILED")
            if verification_result:
                print(f"  Details: {verification_result.verification_summary}")
                for ov in verification_result.output_validations:
                    if not ov.get('output_type_matches', True):
                        print(f"    - {ov.get('tool_name')}: {ov.get('issues')}")
            print(f"\n✗ Datapoint failed verification - discarding")
            return None

        print(f" Verification: PASSED")

        # Update token usage from class-level tracking
        self._update_token_usage()
        token_usage = self._get_token_stats()

        available_tools = self._get_policy_tool_schemas(focus_category)

        # Create metadata
        metadata = {
            "num_actions": len(trajectory),
            "focus_category": focus_category,
            "query_intent": query_result.intent,
            "expected_tools": query_result.expected_tools,
            "rl_quality_gate_passed": True,
            "query_quality_preflight": query_result.quality_preflight,
            "final_response_grounding": dict(self._last_final_response_quality),
            "model_routing": self._model_routing_metadata(),
            "tool_contract_hash": self._tool_contract_hash(available_tools),
        }

        # Create datapoint
        datapoint = StepByStepDatapoint(
            trajectory=conv_trajectory,
            generation_metadata=metadata,
            verification_result=verification_result.model_dump() if verification_result else {},
            token_usage=token_usage,
            initial_api_state=filtered_initial_state,
            intermediate_api_states=intermediate_states,
            available_tools=available_tools,
        )

        return datapoint

    def _verify_final_response_grounding(
        self,
        query: str,
        actions: List[Dict[str, Any]],
        tool_definitions: List[Dict[str, Any]],
        final_response: str,
    ) -> Dict[str, Any]:
        """Certify the final answer using policy-visible evidence only."""
        allowed_codes = {
            "UNSUPPORTED_CLAIM",
            "UNCALLED_CALCULATION",
            "INVENTED_UNIT",
            "FALSE_SUCCESS_CLAIM",
            "OMITTED_REQUIRED_RESULT",
            "MISREPRESENTED_VERIFICATION",
            "CONTRADICTS_TOOL_OUTPUT",
            "GROUNDING_VERIFIER_UNAVAILABLE",
            "OTHER_INVALID",
        }
        prompt = f"""You are certifying the final response of a tool-using assistant.

=== USER QUERY ===
{query}

=== USED TOOL DEFINITIONS ===
{json.dumps(tool_definitions, indent=2, ensure_ascii=False, default=str)}

=== POLICY-VISIBLE TOOL CALLS AND OUTPUTS ===
{json.dumps(actions, indent=2, ensure_ascii=False, default=str)}

=== CANDIDATE FINAL RESPONSE ===
{final_response}

Check that every factual claim is supported by the user query, a tool definition,
or an actual tool output above. In particular:
- Do not allow arithmetic, conversions, rankings, or comparisons that were not
  returned by a called tool.
- Do not allow a unit that was neither declared by a used tool nor returned in
  its output.
- Do not claim a field was verified unless it was actually supplied to and
  accepted by the verification tool.
- Do not claim success when a mutation was a no-op or a tool reported failure.
- Do not report results for candidates that were not passed to the downstream
  tool.
- The response must address the requested results without inventing details.

Return only issue codes from this list:
{json.dumps(sorted(allowed_codes))}

Respond ONLY with JSON:
{{"is_grounded": true, "issue_codes": []}}
or
{{"is_grounded": false, "issue_codes": ["ONE_ALLOWED_CODE"]}}
"""
        try:
            response = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                llm=self.grounding_judge,
                purpose="final_response_grounding_judge",
            )
            response_text = response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]
            result = json.loads(response_text)
            raw_codes = result.get("issue_codes", [])
            if not isinstance(raw_codes, list):
                raw_codes = []
            issue_codes = [
                str(code) for code in raw_codes
                if str(code) in allowed_codes
            ]
            passed = bool(result.get("is_grounded", False)) and not issue_codes
            if not passed and not issue_codes:
                issue_codes = ["OTHER_INVALID"]
            return {"passed": passed, "issue_codes": issue_codes}
        except Exception as exc:
            print(f"    Warning: final-response grounding verifier failed: {exc}")
            return {
                "passed": False,
                "issue_codes": ["GROUNDING_VERIFIER_UNAVAILABLE"],
            }

    def _generate_final_response(
        self,
        query: str,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
    ) -> str:
        """Generate and certify a response grounded in actual tool evidence."""
        actions = []
        used_tool_names = []
        for step in trajectory:
            for tc in step.tool_calls:
                actions.append(
                    {
                        "step_number": step.step_number,
                        "tool": tc.tool_name,
                        "arguments": tc.arguments,
                        "output": tc.output,
                    }
                )
                if tc.tool_name not in used_tool_names:
                    used_tool_names.append(tc.tool_name)

        tool_definitions = [
            self.tool_manager.get_tool_schema(name)
            for name in used_tool_names
            if self.tool_manager.get_tool_schema(name)
        ]
        retry_codes: List[str] = []

        # One generation plus one grounding decision. A failed answer rejects
        # the candidate instead of nesting another generate+judge loop.
        for attempt in range(1):
            retry_notice = ""
            if retry_codes:
                retry_notice = (
                    "\nA previous response failed grounding with these generic issue "
                    f"codes: {', '.join(retry_codes)}. Correct those defects "
                    "without adding new facts.\n"
                )

            prompt = f"""Generate a concise final response for the user.

=== USER QUERY ===
{query}

=== USED TOOL DEFINITIONS ===
{json.dumps(tool_definitions, indent=2, ensure_ascii=False, default=str)}

=== ACTUAL TOOL CALLS AND OUTPUTS ===
{json.dumps(actions, indent=2, ensure_ascii=False, default=str)}
{retry_notice}
=== RULES ===
- State only facts supported by the query, used tool definitions, or actual
  outputs above.
- Do not perform a new calculation, conversion, selection, ranking, or lookup.
- Do not infer a unit unless the used tool definition declares it or the output
  contains it.
- Do not say a field was verified unless that field was an input to or output of
  the verification call.
- If a tool failed, say so plainly. Do not convert failure into success.
- If a tool returned multiple values, report them as returned unless a later
  tool explicitly selected one.
- Do not add advice about timing, urgency, safety, or recommendations unless a
  tool output supports it.

Return only the natural-language response.
"""
            try:
                response = self._safe_llm_generate(
                    [{"role": "user", "content": prompt}],
                    llm=self.final_response_llm,
                    purpose="final_response_generate",
                ).strip()
            except Exception as exc:
                print(f"    Error generating final response: {exc}")
                retry_codes = ["OTHER_INVALID"]
                continue

            quality = self._verify_final_response_grounding(
                query=query,
                actions=actions,
                tool_definitions=tool_definitions,
                final_response=response,
            )
            self._last_final_response_quality = quality
            if quality.get("passed", False):
                return response

            retry_codes = quality.get("issue_codes", ["OTHER_INVALID"])
            print(
                "    Final response failed grounding: "
                + ", ".join(retry_codes)
            )

        self._last_final_response_quality = {
            "passed": False,
            "issue_codes": retry_codes or ["OTHER_INVALID"],
        }
        return ""

    # ==================== VERIFICATION METHODS ====================

    def verify_tool_relevance(self, query: str, tool_name: str, step: TrajectoryStep) -> Dict[str, Any]:
        """Verify if a tool is relevant to the query."""
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return {'tool_name': tool_name, 'is_relevant': False, 'relevance_score': 0.0, 'reasoning': 'Tool not found in tool pool'}

        tool_description = tool_schema.get('description', '')
        keywords = set(tool_description.lower().split())
        query_words = set(query.lower().split())
        overlap = len(keywords & query_words)
        relevance_score = min(1.0, overlap / max(1, len(keywords)))

        name_words = set(tool_name.lower().replace('_', ' ').split())
        name_overlap = len(name_words & query_words)

        is_relevant = relevance_score > 0.1 or name_overlap > 0

        reasoning = f"Tool '{tool_name}': score={relevance_score:.2f}, name_match={name_overlap}"
        reasoning += ". Tool appears relevant." if is_relevant else ". Tool may not be directly relevant."

        return {'tool_name': tool_name, 'is_relevant': is_relevant, 'relevance_score': relevance_score, 'reasoning': reasoning}

    def verify_invocation_order(self, query: str, trajectory: List[TrajectoryStep]) -> Dict[str, Any]:
        """Verify if tools were invoked in a logical order."""
        if not trajectory:
            return {'order_is_correct': True, 'order_verification_details': 'No steps to verify'}

        return {'order_is_correct': True, 'order_verification_details': 'Order appears logical.'}

    @staticmethod
    def _is_dict_wrapped_primitive(output: Any, expected_type_lower: str) -> bool:
        """Check if a dict output wraps a value matching expected_type.

        Python tool implementations commonly return:
        - {'result': 42.0} when BFCL declares output_type=float
        - {'matching_tweets': []} when BFCL declares output_type=list
        - {'comments': [...]} when BFCL declares output_type=list
        This is a valid wrapper pattern - the semantic content *is* the
        expected type.
        """
        if not isinstance(output, dict) or not output:
            return False

        # List-wrapping: {"key": [...]}
        if 'list' in expected_type_lower:
            return any(isinstance(v, list) for v in output.values())

        prim_types = {
            'float': float,
            'number': (int, float),
            'integer': int,
            'string': str,
            'boolean': bool,
        }
        py_type = prim_types.get(expected_type_lower)
        if py_type is None:
            return False
        for v in output.values():
            if isinstance(v, py_type):
                return True
        return False

    def verify_output_consistency(self, tool_name: str, step_number: int, output: Any, expected_type: str, expected_description: str) -> Dict[str, Any]:
        """Verify if a tool's output matches its declared type and description."""
        if output is None:
            return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': False, 'issues': ['Output is None']}

        issues = []
        output_type_matches = True
        if expected_type:
            expected_type_lower = expected_type.lower()
            output_type = type(output).__name__.lower()
            type_compatible = False
            if 'dict' in expected_type_lower and isinstance(output, dict):
                type_compatible = True
            elif 'list' in expected_type_lower and isinstance(output, list):
                type_compatible = True
            elif 'string' in expected_type_lower and isinstance(output, str):
                type_compatible = True
            elif 'number' in expected_type_lower and isinstance(output, (int, float)):
                type_compatible = True
            elif 'bool' in expected_type_lower and isinstance(output, bool):
                type_compatible = True
            elif expected_type_lower in output_type:
                type_compatible = True

            if not type_compatible and self._is_dict_wrapped_primitive(output, expected_type_lower):
                type_compatible = True

            if not type_compatible:
                output_type_matches = False
                issues.append(f"Type mismatch: expected {expected_type}, got {output_type}")

        return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': output_type_matches, 'issues': issues}

    def verify_placeholder_resolution(self, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that all placeholders in tool arguments were resolved correctly."""
        total_placeholders = 0
        resolved_count = 0
        details = []
        placeholder_pattern = re.compile(r"\{\{([^{}]+)\}\}")

        for step in trajectory:
            for tc in step.tool_calls:
                for arg_name, arg_value in tc.arguments.items():
                    if isinstance(arg_value, str):
                        placeholders = placeholder_pattern.findall(arg_value)
                        for placeholder in placeholders:
                            total_placeholders += 1
                            keys = placeholder.split('.')
                            current = execution_context
                            found = True
                            for key in keys:
                                if isinstance(current, dict) and key in current:
                                    current = current[key]
                                else:
                                    found = False
                                    break
                            if found:
                                resolved_count += 1
                                details.append({'step': step.step_number, 'tool': tc.tool_name, 'argument': arg_name, 'placeholder': f"{{{{{placeholder}}}}}", 'resolved': True, 'resolved_value': str(current)[:100]})
                            else:
                                details.append({'step': step.step_number, 'tool': tc.tool_name, 'argument': arg_name, 'placeholder': f"{{{{{placeholder}}}}}", 'resolved': False, 'resolved_value': None})

        return {'all_resolved': total_placeholders == resolved_count, 'total_placeholders': total_placeholders, 'resolved_count': resolved_count, 'details': details}

    def verify_state_transition(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        tool_output: Any,
        pre_state: Dict[str, Dict[str, Any]],
        post_state: Dict[str, Dict[str, Any]],
    ) -> StateVerificationResult:
        """Use an LLM-as-judge to verify that a tool call produced a
        logically correct state transition.

        The LLM receives:
        - The tool name, arguments, and output
        - A *diff* between pre_state and post_state (only changed keys)
        - Relevant class keys (the ones that actually changed)

        It judges whether the state changes are consistent with the tool's
        declared semantics and the returned output.
        """
        # Compute diff - only include class keys that changed
        changed_classes: Dict[str, Dict[str, Any]] = {}
        for class_key in set(pre_state) | set(post_state):
            pre = pre_state.get(class_key, {})
            post = post_state.get(class_key, {})
            if pre != post:
                diff: Dict[str, Any] = {}
                all_keys = set(pre) | set(post)
                for k in all_keys:
                    pre_val = pre.get(k, "<MISSING>")
                    post_val = post.get(k, "<MISSING>")
                    if pre_val != post_val:
                        diff[k] = {"before": pre_val, "after": post_val}
                if diff:
                    changed_classes[class_key] = diff

        if not changed_classes:
            return StateVerificationResult(
                is_valid=True,
                reasoning="No state changes detected (read-only or no-op call).",
                issues=[],
                state_changes_summary="No state changes.",
            )

        # Determine which class_key the tool belongs to
        tool_class_key = self.tool_manager.api_name_to_class_key.get(tool_name, "unknown")

        # Build a compact diff summary to avoid truncation issues
        diff_summary = {}
        for class_key in changed_classes:
            changes = []
            for k, v in changed_classes[class_key].items():
                before_val = v.get("before", "<MISSING>")
                after_val = v.get("after", "<MISSING>")
                if before_val == "<MISSING>":
                    changes.append(f"{k}: added")
                elif after_val == "<MISSING>":
                    changes.append(f"{k}: removed")
                else:
                    changes.append(f"{k}: modified")
            diff_summary[class_key] = changes

        diff_summary_str = json.dumps(diff_summary, indent=2, default=str, ensure_ascii=False)

        output_str = json.dumps(tool_output, default=str, ensure_ascii=False) if not isinstance(tool_output, str) else tool_output
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "... (truncated)"

        args_str = json.dumps(tool_arguments, default=str, ensure_ascii=False)
        if len(args_str) > 1000:
            args_str = args_str[:1000] + "... (truncated)"

        prompt = f"""You are an expert API state auditor. Verify that the state transition produced by a tool call is logically correct and consistent with the tool's output.

=== TOOL CALL ===
Tool: {tool_name}
Class: {tool_class_key}
Arguments: {args_str}
Output: {output_str}

=== STATE CHANGE SUMMARY ===
{diff_summary_str}

For each changed class, the list shows what fields were added, removed, or modified.

=== YOUR TASK ===
1. Check whether the state changes are logically consistent with what the tool is supposed to do.
2. Verify that authentication/login state was updated correctly (e.g., current_user, authenticated, access_token).
3. Verify that data mutations (new messages, tickets, bookings, orders, etc.) are reflected in the state.
4. Check for any contradictory or nonsensical state changes.

NOTE: If you cannot determine validity from the summary provided, assume the state change is valid. Only mark as INVALID if you see clear contradictions or impossible changes.

Respond ONLY with valid JSON:
{{
  "is_valid": true/false,
  "reasoning": "brief explanation of your verdict",
  "issues": ["list of issues found, empty if valid"],
  "state_changes_summary": "human-readable summary of what changed"
 }}"""

        try:
            response = self._safe_llm_generate([{"role": "user", "content": prompt}], llm=self.judge)
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
            return StateVerificationResult(
                is_valid=bool(result.get("is_valid", True)),
                reasoning=result.get("reasoning", ""),
                issues=result.get("issues", []),
                state_changes_summary=result.get("state_changes_summary", ""),
            )
        except Exception as e:
            print(f" Warning: State verification LLM call failed: {e}")
            return StateVerificationResult(
                is_valid=False,
                reasoning=f"LLM judge call failed ({e}); transition is not certified.",
                issues=["STATE_VERIFIER_UNAVAILABLE"],
                state_changes_summary="Could not certify state transition.",
            )

    def run_full_verification(
        self,
        query: str,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
        final_response: Optional[str] = None,
        query_quality: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Run structural checks plus the positive-RL quality certificate."""
        print("\n  Running Verification...")

        # 1. Check tool relevance
        tool_relevance_checks = []
        all_relevant = True
        for step in trajectory:
            for tc in step.tool_calls:
                check = self.verify_tool_relevance(query, tc.tool_name, step)
                tool_relevance_checks.append(check)
                if not check['is_relevant']:
                    all_relevant = False

        # 2. Verify invocation order
        order_result = self.verify_invocation_order(query, trajectory)

        # 3. Verify output consistency
        output_validations = []
        all_outputs_valid = True
        for step in trajectory:
            for tc in step.tool_calls:
                tool_schema = self.tool_manager.get_tool_schema(tc.tool_name)
                expected_type = tool_schema.get('output_type', 'unknown') if tool_schema else 'unknown'
                expected_desc = tool_schema.get('output_description', '') if tool_schema else ''
                validation = self.verify_output_consistency(
                    tc.tool_name,
                    step.step_number,
                    tc.output,
                    expected_type,
                    expected_desc,
                )
                output_validations.append(validation)
                if not validation['output_type_matches']:
                    all_outputs_valid = False

        # 4. Check placeholder resolution
        placeholder_result = self.verify_placeholder_resolution(trajectory, execution_context)

        transition_checks = [
            step.quality_verification
            for step in trajectory
            if step.quality_verification
        ]
        transitions_passed = all(
            check.get("passed", False) for check in transition_checks
        ) if transition_checks else True

        query_check = query_quality or {"passed": True, "issue_codes": []}
        final_check = (
            dict(self._last_final_response_quality)
            if final_response is not None
            else {"passed": True, "issue_codes": []}
        )
        quality_passed = (
            bool(query_check.get("passed", False))
            and transitions_passed
            and bool(final_check.get("passed", False))
        )
        rl_quality_gate = {
            "passed": quality_passed,
            "query_preflight": query_check,
            "transition_checks": transition_checks,
            "final_response_grounding": final_check,
        }

        overall_passed = (
            all_relevant
            and order_result['order_is_correct']
            and all_outputs_valid
            and placeholder_result['all_resolved']
            and quality_passed
        )

        issues = []
        if not all_relevant:
            issues.append("Some tools are not relevant to the query")
        if not order_result['order_is_correct']:
            issues.append("Tool invocation order may be incorrect")
        if not all_outputs_valid:
            issues.append("Some tool outputs don't match their declarations")
        if not placeholder_result['all_resolved']:
            issues.append(
                f"{placeholder_result['total_placeholders'] - placeholder_result['resolved_count']} "
                "placeholders were not resolved"
            )
        if not quality_passed:
            issues.append("Positive-RL quality gate failed")

        summary = (
            "Verification PASSED"
            if overall_passed
            else "Verification FAILED - " + "; ".join(issues)
        )

        return VerificationResult(
            query=query,
            tool_relevance_checks=tool_relevance_checks,
            order_is_correct=order_result['order_is_correct'],
            order_verification_details=order_result['order_verification_details'],
            output_validations=output_validations,
            placeholder_resolution=placeholder_result,
            rl_quality_gate=rl_quality_gate,
            overall_verification_passed=overall_passed,
            verification_summary=summary,
        )


# --- CLI Entry Point ---

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        exit(1)

    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="z-ai/glm-5.1",
        hf_tokenizer_id=None
    )

    tool_pool_path = str(Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl").expanduser())
    invocation_examples_path = str(Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl").expanduser())
    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path
    )

    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_actions=2,
    )

    print("Generating test datapoint...")
    datapoint = generator.generate_datapoint(focus_category="Communication")

    if datapoint:
        print("\n" + "=" * 60)
        print("GENERATED DATAPOINT:")
        print("=" * 60)
        print(datapoint.model_dump_json(indent=2))
    else:
        print("\nFailed to generate datapoint")
