"""Multi-turn conversation generator with step-by-step tool simulation.

Extends StepByStepGenerator to produce multi-turn conversations where
a separate LLM generates each user turn based on the dialog blueprint
and the current point in the conversation.
"""

import json
import copy
import time
import os
import random
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
        symbolic_episode_plan: bool = False,
        blueprint_min_total_actions: Optional[int] = None,
        blueprint_max_total_actions: Optional[int] = None,
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
        # V2 asks the blueprint model for both natural user utterances and the
        # complete symbolic call graph.  Python then materializes and executes
        # that graph directly, eliminating one paid compiler request per turn.
        # Keep the constructor default off for callers that depend on the old
        # blueprint contract; the generation CLI enables it by default.
        self.symbolic_episode_plan = bool(symbolic_episode_plan)
        self.blueprint_max_actions_per_turn = max(
            1,
            min(
                10,
                (
                    actions_per_turn
                    if blueprint_max_actions_per_turn is None
                    else blueprint_max_actions_per_turn
                ),
            ),
        )
        self.blueprint_min_total_actions = (
            max(self.num_turns, int(blueprint_min_total_actions))
            if blueprint_min_total_actions is not None
            else None
        )
        if (
            self.blueprint_min_total_actions is not None
            and self.blueprint_min_total_actions
            > self.num_turns * self.blueprint_max_actions_per_turn
        ):
            raise ValueError(
                "blueprint_min_total_actions exceeds the available per-turn "
                "action capacity"
            )
        self.blueprint_max_total_actions = (
            min(
                self.num_turns * self.blueprint_max_actions_per_turn,
                int(blueprint_max_total_actions),
            )
            if blueprint_max_total_actions is not None
            else None
        )
        if (
            self.blueprint_max_total_actions is not None
            and self.blueprint_max_total_actions < self.num_turns
        ):
            raise ValueError(
                "blueprint_max_total_actions cannot be smaller than num_turns"
            )
        if (
            self.blueprint_min_total_actions is not None
            and self.blueprint_max_total_actions is not None
            and self.blueprint_min_total_actions
            > self.blueprint_max_total_actions
        ):
            raise ValueError(
                "blueprint_min_total_actions exceeds "
                "blueprint_max_total_actions"
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

    @staticmethod
    def _output_schema_supports_path(
        output_schema: Dict[str, Any], path: str
    ) -> bool:
        """Return whether a declared output schema contains ``path``.

        Missing/empty output schemas are intentionally permissive: runtime
        simulator output remains authoritative for legacy tool definitions.
        """
        if not isinstance(output_schema, dict) or not output_schema:
            return True
        current: Any = output_schema
        normalised = str(path or "").replace("[", ".").replace("]", "")
        for part in [item for item in normalised.split(".") if item]:
            if not isinstance(current, dict):
                return False
            schema_type = current.get("type")
            if schema_type == "array" or (
                "items" in current and str(part).isdigit()
            ):
                if not str(part).isdigit():
                    return False
                items = current.get("items")
                # Several imported BFCL contracts declare only ``array`` and
                # omit the item schema.  The nested shape is genuinely unknown,
                # not forbidden; simulator path resolution will validate it at
                # execution time.
                if not isinstance(items, dict) or not items:
                    return True
                current = items
                continue
            properties = current.get("properties", {})
            if not isinstance(properties, dict) or part not in properties:
                return False
            current = properties[part]
        return True

    def _validate_symbolic_source_tree(
        self,
        *,
        spec: Any,
        schema: Dict[str, Any],
        visible_text: str,
        seen_calls: Dict[str, Dict[str, Any]],
        label: str,
        errors: List[str],
    ) -> set[str]:
        """Validate every provenance leaf of one (possibly nested) argument."""
        dependencies: set[str] = set()
        if isinstance(spec, dict) and "source" not in spec:
            schema_type = str(schema.get("type", "")).casefold()
            properties = schema.get("properties", {})
            if schema_type not in {"object", "dict"} and not properties:
                errors.append(f"{label}: missing provenance source.")
                return dependencies
            required = schema.get("required", [])
            missing = [name for name in required if name not in spec]
            if missing:
                errors.append(
                    f"{label}: missing required nested fields {missing}."
                )
            if schema.get("additionalProperties", True) is False:
                extra = [name for name in spec if name not in properties]
                if extra:
                    errors.append(f"{label}: undeclared nested fields {extra}.")
            for name, child in spec.items():
                child_schema = (
                    properties.get(name, {})
                    if isinstance(properties, dict)
                    else {}
                )
                dependencies.update(
                    self._validate_symbolic_source_tree(
                        spec=child,
                        schema=child_schema,
                        visible_text=visible_text,
                        seen_calls=seen_calls,
                        label=f"{label}.{name}",
                        errors=errors,
                    )
                )
            return dependencies
        if isinstance(spec, list):
            schema_type = str(schema.get("type", "")).casefold()
            if schema_type not in {"array", "list"}:
                errors.append(f"{label}: list value has no array provenance schema.")
                return dependencies
            item_schema = schema.get("items", {})
            for index, child in enumerate(spec):
                dependencies.update(
                    self._validate_symbolic_source_tree(
                        spec=child,
                        schema=(
                            item_schema if isinstance(item_schema, dict) else {}
                        ),
                        visible_text=visible_text,
                        seen_calls=seen_calls,
                        label=f"{label}[{index}]",
                        errors=errors,
                    )
                )
            return dependencies
        if not isinstance(spec, dict):
            errors.append(f"{label}: missing provenance object.")
            return dependencies

        source = str(spec.get("source", "")).casefold()
        if source == "tool_output":
            producer_id = str(spec.get("call_id", ""))
            path = str(spec.get("path", ""))
            producer = seen_calls.get(producer_id)
            if producer is None:
                errors.append(
                    f"{label}: '{producer_id}' is not an earlier call."
                )
                return dependencies
            producer_output = producer["schema"].get("output_schema", {})
            if not self._output_schema_supports_path(producer_output, path):
                errors.append(
                    f"{label}: output path '{producer_id}.{path}' is not declared."
                )
            dependencies.add(producer_id)
        elif source in {"user", "history"}:
            if "value" not in spec:
                errors.append(f"{label}: literal provenance has no value.")
                return dependencies
            value = spec.get("value")
            schema_declares_value = (
                value == schema.get("const")
                or value in schema.get("enum", [])
            )
            if (
                not schema_declares_value
                and not self._value_visible_in_text(value, visible_text)
            ):
                errors.append(
                    f"{label}: declared {source} value is not visible in the "
                    "current/prior user utterances."
                )
        elif source == "schema_default":
            if "default" not in schema:
                errors.append(
                    f"{label}: schema_default is not declared by the schema."
                )
        else:
            errors.append(f"{label}: unsupported source '{source}'.")
        return dependencies

    def _normalise_policy_visible_argument_spec(
        self,
        *,
        spec: Any,
        schema: Dict[str, Any],
        current_user_query: str,
        prior_user_text: str,
    ) -> Any:
        """Recover omitted provenance only when it is deterministic and safe.

        Teachers occasionally emit a raw literal despite the requested source
        wrapper, especially ``[]`` for optional list defaults.  Rejecting the
        entire episode and regenerating it is unnecessary when Python can prove
        that the value is a declared schema default or is literally visible to
        the policy.  Anything else is left untouched and the normal fail-closed
        provenance validator rejects it.
        """
        if isinstance(spec, dict) and "source" in spec:
            return copy.deepcopy(spec)
        if "default" in schema and spec == schema.get("default"):
            return {"source": "schema_default"}
        if self._value_visible_in_text(spec, current_user_query):
            return {"source": "user", "value": copy.deepcopy(spec)}
        if prior_user_text and self._value_visible_in_text(spec, prior_user_text):
            return {"source": "history", "value": copy.deepcopy(spec)}
        return copy.deepcopy(spec)

    @staticmethod
    def _unique_prior_output_binding(
        argument_name: str,
        seen_calls: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """Bind an opaque argument to one unambiguous earlier output.

        This is a deterministic compiler repair, not a guessed value.  It is
        deliberately limited to top-level declared output fields and requires
        exactly one producer in the preceding graph.
        """
        aliases = {
            "stock": ("stock", "symbol"),
            "receiver_id": ("receiver_id", "user_id"),
        }
        candidate_fields = aliases.get(argument_name, (argument_name,))
        candidates: List[Tuple[str, str]] = []
        for call_id, call in seen_calls.items():
            output_schema = call.get("schema", {}).get("output_schema", {})
            properties = (
                output_schema.get("properties", {})
                if isinstance(output_schema, dict)
                else {}
            )
            if not isinstance(properties, dict):
                continue
            for field_name in candidate_fields:
                if field_name in properties:
                    candidates.append((call_id, field_name))
        if len(candidates) != 1:
            return None
        call_id, path = candidates[0]
        return {
            "source": "tool_output",
            "call_id": call_id,
            "path": path,
        }

    def _repair_invalid_output_passthrough(
        self,
        *,
        spec: Any,
        argument_name: str,
        seen_calls: Dict[str, Dict[str, Any]],
    ) -> Any:
        """Reuse a prior visible input when a teacher treats it as output.

        File tools commonly return only ``success`` even though the next call
        needs the destination path that the user supplied to the mutation.
        Reusing that already policy-visible input is deterministic.  The
        repair is intentionally limited to identical fields and explicit file
        path aliases; it never turns a display name into an opaque identifier.
        """
        if isinstance(spec, dict) and spec.get("source") == "tool_output":
            producer = seen_calls.get(str(spec.get("call_id", "")))
            path = str(spec.get("path", ""))
            if producer is None or not path or "." in path or "[" in path:
                return copy.deepcopy(spec)
            output_schema = producer.get("schema", {}).get(
                "output_schema", {}
            )
            if self._output_schema_supports_path(output_schema, path):
                return copy.deepcopy(spec)
            compatible = argument_name == path or (
                argument_name in {"file_name", "file_name1", "file_name2"}
                and path in {"source", "destination", "file_name", "path"}
            )
            producer_arguments = producer.get("arguments", {})
            if compatible and path in producer_arguments:
                return copy.deepcopy(producer_arguments[path])
            return copy.deepcopy(spec)
        if isinstance(spec, dict):
            return {
                name: self._repair_invalid_output_passthrough(
                    spec=child,
                    argument_name=name,
                    seen_calls=seen_calls,
                )
                for name, child in spec.items()
            }
        if isinstance(spec, list):
            return [
                self._repair_invalid_output_passthrough(
                    spec=child,
                    argument_name=argument_name,
                    seen_calls=seen_calls,
                )
                for child in spec
            ]
        return copy.deepcopy(spec)

    @staticmethod
    def _safe_literal_closure_clause(
        argument_name: str,
        value: Any,
        schema: Dict[str, Any],
    ) -> Optional[str]:
        """Return a natural clause for a safe, non-secret user constraint.

        Cheap teachers sometimes provide a perfectly ordinary numeric/date
        constraint in the symbolic arguments but omit it from the utterance.
        Python may make that proposed value policy-visible without another LLM
        call.  Opaque identifiers and credentials are never exposed this way.
        """
        lowered = argument_name.casefold()
        blocked_fragments = (
            "_id", "token", "password", "secret", "credential",
            "api_key", "account_number", "card_number", "username",
        )
        if lowered == "id" or any(item in lowered for item in blocked_fragments):
            return None
        scalar = isinstance(value, (str, int, float, bool)) or value is None
        scalar_list = isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) or item is None
            for item in value
        )
        if not (scalar or scalar_list) or isinstance(value, dict):
            return None
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, str):
            rendered = value
        elif isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        templates = {
            "order_type": f"Make it a {rendered} order.",
            "xact_type": f"Make it a {rendered} transaction.",
            "start_date": f"Use {rendered} as the start date.",
            "end_date": f"Use {rendered} as the end date.",
            "min_price": f"Use a minimum price of {rendered}.",
            "max_price": f"Use a maximum price of {rendered}.",
            "threshold": f"Use a threshold of {rendered}.",
            "amount": f"Use an amount of {rendered}.",
            "price": f"Use a price of {rendered}.",
            "symbol": f"Use the ticker symbol {rendered}.",
            "stock": f"Use {rendered} as the stock.",
            "sector": f"Focus on the {rendered} sector.",
            "destination": f"Set the destination to {rendered}.",
            "option": f"Use the {rendered} option.",
            "door": f"Apply that to the {rendered} door.",
            "priority": f"Set the priority to {rendered}.",
            "status": f"Set the status to {rendered}.",
            "owner": f"Assign it to {rendered}.",
            "description": f"Use {rendered} as the description.",
            "file_name": f"Use the file {rendered}.",
            "file_name1": f"Use {rendered} as the first file.",
            "file_name2": f"Use {rendered} as the second file.",
            "numbers": f"Use the values {rendered}.",
        }
        return templates.get(
            lowered,
            f"Use {rendered} for the {lowered.replace('_', ' ')}.",
        )

    def _close_safe_nested_literal_sources(
        self,
        *,
        spec: Any,
        schema: Dict[str, Any],
        visible_text: str,
        seen_calls: Dict[str, Dict[str, Any]],
        added_closure_values: Dict[str, Any],
        path: Tuple[str, ...],
    ) -> Tuple[Any, List[str]]:
        """Expose safe missing literals inside composite arguments.

        Top-level arguments already receive this deterministic repair in the
        normalizer.  Object arguments such as ``updates.priority`` previously
        skipped it, causing an otherwise executable episode to be regenerated.
        Opaque IDs and credentials still return no clause and remain subject to
        the fail-closed provenance validator.
        """
        if isinstance(spec, dict) and "source" in spec:
            source = str(spec.get("source", "")).casefold()
            if source not in {"user", "history"} or "value" not in spec:
                return copy.deepcopy(spec), []
            value = spec.get("value")
            schema_declares_value = (
                value == schema.get("const")
                or value in schema.get("enum", [])
            )
            if schema_declares_value or self._value_visible_in_text(
                value, visible_text
            ):
                return copy.deepcopy(spec), []
            argument_name = path[-1]
            prior_binding = self._unique_prior_output_binding(
                argument_name, seen_calls
            )
            if prior_binding is not None:
                return prior_binding, []
            closure_key = ".".join(path) + "=" + json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            clause = self._safe_literal_closure_clause(
                argument_name, value, schema
            )
            if clause is None:
                return copy.deepcopy(spec), []
            if closure_key in added_closure_values:
                clauses: List[str] = []
            else:
                added_closure_values[closure_key] = copy.deepcopy(value)
                clauses = [clause]
            return {
                "source": "user",
                "value": copy.deepcopy(value),
            }, clauses

        schema_type = str(schema.get("type", "")).casefold()
        if isinstance(spec, dict) and (
            schema_type in {"object", "dict"}
            or isinstance(schema.get("properties"), dict)
        ):
            properties = schema.get("properties", {})
            repaired: Dict[str, Any] = {}
            clauses: List[str] = []
            for name, child in spec.items():
                child_schema = (
                    properties.get(name, {})
                    if isinstance(properties, dict)
                    else {}
                )
                repaired_child, child_clauses = (
                    self._close_safe_nested_literal_sources(
                        spec=child,
                        schema=child_schema,
                        visible_text=visible_text,
                        seen_calls=seen_calls,
                        added_closure_values=added_closure_values,
                        path=(*path, name),
                    )
                )
                repaired[name] = repaired_child
                clauses.extend(child_clauses)
            return repaired, clauses

        # Safely normalize a raw nested scalar/list using the same rules as a
        # top-level argument. Composite objects are deliberately not serialized
        # into the user's utterance.
        if "default" in schema and spec == schema.get("default"):
            return {"source": "schema_default"}, []
        if self._value_visible_in_text(spec, visible_text):
            return {"source": "user", "value": copy.deepcopy(spec)}, []
        argument_name = path[-1]
        closure_key = ".".join(path) + "=" + json.dumps(
            spec,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        clause = self._safe_literal_closure_clause(
            argument_name, spec, schema
        )
        if clause is None:
            return copy.deepcopy(spec), []
        clauses = []
        if closure_key not in added_closure_values:
            added_closure_values[closure_key] = copy.deepcopy(spec)
            clauses = [clause]
        return {"source": "user", "value": copy.deepcopy(spec)}, clauses

    def _normalise_symbolic_blueprint_turns(
        self,
        turns: List[Dict[str, Any]],
        *,
        available_tool_names: set[str],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate and canonicalize an episode-wide symbolic call graph.

        This is deliberately deterministic.  It checks the graph, schemas,
        provenance declarations and user-visible literal closure before any
        simulator mutation occurs.  Concrete tool outputs are never predicted
        by the teacher; later arguments retain symbolic call-id/path bindings.
        """
        errors: List[str] = []
        canonical_turns: List[Dict[str, Any]] = []
        seen_calls: Dict[str, Dict[str, Any]] = {}
        visible_user_queries: List[str] = []

        for turn_index, raw_turn in enumerate(turns, 1):
            if not isinstance(raw_turn, dict):
                errors.append(f"Turn {turn_index} is not an object.")
                continue
            query = str(raw_turn.get("user_query", "")).strip()
            if not query:
                errors.append(f"Turn {turn_index} has no user_query.")
            raw_calls = raw_turn.get("calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                errors.append(f"Turn {turn_index} has no symbolic calls.")
                raw_calls = []
            if len(raw_calls) > self.blueprint_max_actions_per_turn:
                errors.append(
                    f"Turn {turn_index} has {len(raw_calls)} calls; maximum is "
                    f"{self.blueprint_max_actions_per_turn}."
                )

            canonical_calls: List[Dict[str, Any]] = []
            expected_tools: List[str] = []
            visible_text = "\n".join([*visible_user_queries, query])
            added_closure_values: Dict[str, Any] = {}
            for call_index, raw_call in enumerate(raw_calls, 1):
                if not isinstance(raw_call, dict):
                    errors.append(
                        f"Turn {turn_index} call {call_index} is not an object."
                    )
                    continue
                call_id = str(raw_call.get("call_id", "")).strip()
                tool_name = str(raw_call.get("tool_name", "")).strip()
                if not call_id:
                    errors.append(
                        f"Turn {turn_index} call {call_index} has no call_id."
                    )
                elif call_id in seen_calls:
                    errors.append(f"Duplicate call_id '{call_id}'.")
                if tool_name not in available_tool_names:
                    errors.append(
                        f"Turn {turn_index} call {call_id or call_index} uses "
                        f"unavailable tool '{tool_name}'."
                    )
                    continue
                if raw_call.get("parallel_group") not in (None, ""):
                    errors.append(
                        f"Turn {turn_index} call {call_id} declares a parallel "
                        "group, but symbolic_episode_plan_v2 is sequential."
                    )

                schema = self.tool_manager.get_tool_schema(tool_name)
                parameters = schema.get("parameters", {})
                properties = parameters.get("properties", {})
                argument_specs = copy.deepcopy(raw_call.get("arguments"))
                if not isinstance(argument_specs, dict):
                    errors.append(
                        f"Turn {turn_index} call {call_id}: arguments must be an object."
                    )
                    argument_specs = {}
                prior_user_text = "\n".join(visible_user_queries)
                argument_specs = {
                    argument_name: self._normalise_policy_visible_argument_spec(
                        spec=source_spec,
                        schema=(
                            properties.get(argument_name, {})
                            if isinstance(properties, dict)
                            else {}
                        ),
                        current_user_query=query,
                        prior_user_text=prior_user_text,
                    )
                    for argument_name, source_spec in argument_specs.items()
                }
                for argument_name, source_spec in list(argument_specs.items()):
                    if not (
                        isinstance(source_spec, dict)
                        and source_spec.get("source") in {"user", "history"}
                        and "value" in source_spec
                    ):
                        continue
                    value = source_spec.get("value")
                    argument_schema = (
                        properties.get(argument_name, {})
                        if isinstance(properties, dict)
                        else {}
                    )
                    schema_declares_value = (
                        value == argument_schema.get("const")
                        or value in argument_schema.get("enum", [])
                    )
                    if schema_declares_value or self._value_visible_in_text(
                        value, visible_text
                    ):
                        continue
                    prior_binding = self._unique_prior_output_binding(
                        argument_name, seen_calls
                    )
                    if prior_binding is not None:
                        argument_specs[argument_name] = prior_binding
                        continue
                    closure_key = argument_name + "=" + json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    clause = self._safe_literal_closure_clause(
                        argument_name, value, argument_schema
                    )
                    if clause is None:
                        continue
                    if closure_key not in added_closure_values:
                        query = query.rstrip()
                        if query and query[-1] not in ".!?":
                            query += "."
                        query = f"{query} {clause}".strip()
                        added_closure_values[closure_key] = copy.deepcopy(value)
                        visible_text = "\n".join(
                            [*visible_user_queries, query]
                        )
                    argument_specs[argument_name] = {
                        "source": "user",
                        "value": copy.deepcopy(value),
                    }
                for argument_name, source_spec in list(argument_specs.items()):
                    if not (
                        isinstance(source_spec, (dict, list))
                        and not (
                            isinstance(source_spec, dict)
                            and "source" in source_spec
                        )
                    ):
                        continue
                    argument_schema = (
                        properties.get(argument_name, {})
                        if isinstance(properties, dict)
                        else {}
                    )
                    repaired, clauses = self._close_safe_nested_literal_sources(
                        spec=source_spec,
                        schema=argument_schema,
                        visible_text=visible_text,
                        seen_calls=seen_calls,
                        added_closure_values=added_closure_values,
                        path=(argument_name,),
                    )
                    argument_specs[argument_name] = repaired
                    for clause in clauses:
                        query = query.rstrip()
                        if query and query[-1] not in ".!?":
                            query += "."
                        query = f"{query} {clause}".strip()
                    if clauses:
                        visible_text = "\n".join(
                            [*visible_user_queries, query]
                        )
                argument_specs = {
                    argument_name: self._repair_invalid_output_passthrough(
                        spec=source_spec,
                        argument_name=argument_name,
                        seen_calls=seen_calls,
                    )
                    for argument_name, source_spec in argument_specs.items()
                }
                missing = [
                    name
                    for name in parameters.get("required", [])
                    if name not in argument_specs
                ]
                if missing:
                    errors.append(
                        f"Turn {turn_index} call {call_id}: missing required "
                        f"argument sources {missing}."
                    )
                if parameters.get("additionalProperties", True) is False:
                    extra = [
                        name for name in argument_specs if name not in properties
                    ]
                    if extra:
                        errors.append(
                            f"Turn {turn_index} call {call_id}: undeclared "
                            f"arguments {extra}."
                        )

                dependencies: set[str] = set()
                # ``depends_on`` carries ordering constraints that are not
                # necessarily data dependencies (authenticate -> protected
                # read, mutation -> verification, "then" semantics).  V2
                # previously rebuilt this field only from tool-output
                # provenance and silently discarded the teacher's explicit
                # edges.  Besides making the archived graph inaccurate, that
                # caused the semantic judge to reject otherwise valid plans
                # and pay for a complete blueprint regeneration.
                raw_dependencies = raw_call.get("depends_on", [])
                if not isinstance(raw_dependencies, list):
                    errors.append(
                        f"Turn {turn_index} call {call_id}: depends_on must "
                        "be an array."
                    )
                    raw_dependencies = []
                for dependency in raw_dependencies:
                    dependency_id = str(dependency).strip()
                    if not dependency_id:
                        errors.append(
                            f"Turn {turn_index} call {call_id}: depends_on "
                            "contains an empty call ID."
                        )
                        continue
                    if dependency_id not in seen_calls:
                        errors.append(
                            f"Turn {turn_index} call {call_id}: dependency "
                            f"'{dependency_id}' is not an earlier call."
                        )
                        continue
                    dependencies.add(dependency_id)
                for argument_name, source_spec in argument_specs.items():
                    argument_schema = properties.get(argument_name, {})
                    dependencies.update(
                        self._validate_symbolic_source_tree(
                            spec=source_spec,
                            schema=argument_schema,
                            visible_text=visible_text,
                            seen_calls=seen_calls,
                            label=(
                                f"Turn {turn_index} call "
                                f"{call_id}.{argument_name}"
                            ),
                            errors=errors,
                        )
                    )

                canonical = {
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "arguments": copy.deepcopy(argument_specs),
                    "depends_on": sorted(dependencies),
                    "parallel_group": None,
                }
                canonical_calls.append(canonical)
                expected_tools.append(tool_name)
                if call_id:
                    seen_calls[call_id] = {
                        "turn": turn_index,
                        "schema": schema,
                        "arguments": copy.deepcopy(argument_specs),
                    }

            canonical_turns.append(
                {
                    "user_query": query,
                    "intent": str(raw_turn.get("intent", "")).strip(),
                    "calls": canonical_calls,
                    # Retain the compatibility view used by existing quality
                    # gates, schedulers and evaluation exporters.
                    "expected_tools": expected_tools,
                }
            )
            visible_user_queries.append(query)

        return canonical_turns, errors

    def _execute_symbolic_blueprint_turn(
        self,
        *,
        query_result: QueryGenerationResult,
        turn_spec: Dict[str, Any],
        execution_context: Dict[str, Any],
    ) -> Tuple[Optional[List[TrajectoryStep]], Optional[Dict[str, Any]]]:
        self._last_symbolic_execution_error = None
        compiled_calls = [
            {
                "call_id": str(call.get("call_id", "")),
                "tool_name": str(call.get("tool_name", "")),
                "argument_specs": copy.deepcopy(call.get("arguments", {})),
            }
            for call in turn_spec.get("calls", [])
        ]
        try:
            return self._execute_compiled_turn(
                query=query_result.query,
                compiled_calls=compiled_calls,
                execution_context=execution_context,
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._last_symbolic_execution_error = str(exc) or type(exc).__name__
            print(f"  Symbolic episode-plan execution failed: {exc}")
            return None, None

    def _preflight_symbolic_blueprint_execution(
        self,
        turns: List[Dict[str, Any]],
    ) -> List[str]:
        """Dry-run a symbolic episode and restore the exact simulator state.

        Structural schemas cannot prove that an array path resolves to the
        required scalar or that a planned mutation has an effect in the sampled
        state.  Catch those deterministic failures while Stage 0 can give one
        compact repair message to the teacher, before semantic judging and
        before the candidate is committed.
        """
        if not self._python_tools_available:
            return []
        initial_state = self.tool_manager.get_api_state()
        execution_context: Dict[str, Any] = {}
        try:
            for turn_index, turn in enumerate(turns, 1):
                query = str(turn.get("user_query", ""))
                query_result = QueryGenerationResult(
                    query=query,
                    intent=str(turn.get("intent", "")),
                    expected_tools=list(turn.get("expected_tools", [])),
                    quality_preflight={"passed": True},
                )
                trajectory, updated_context = (
                    self._execute_symbolic_blueprint_turn(
                        query_result=query_result,
                        turn_spec=turn,
                        execution_context=execution_context,
                    )
                )
                if trajectory is None or updated_context is None:
                    detail = getattr(
                        self,
                        "_last_symbolic_execution_error",
                        None,
                    ) or "deterministic execution failed"
                    return [f"Turn {turn_index}: {detail}"]
                execution_context = updated_context
                execution_context.setdefault("turn_outputs", []).append(
                    self._aggregate_turn_outputs(trajectory)
                )
                execution_context.setdefault("prior_user_queries", []).append(
                    query
                )
            return []
        finally:
            self.tool_manager.restore_api_state(initial_state)

    @staticmethod
    def _symbolic_plan_metrics(
        turns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        call_turn: Dict[str, int] = {}
        for turn_index, turn in enumerate(turns, 1):
            for call in turn.get("calls", []):
                call_turn[str(call.get("call_id", ""))] = turn_index

        source_counts: Dict[str, int] = {}
        total_arguments = 0
        hidden_argument_count = 0
        tool_output_bindings = 0
        cross_turn_bindings = 0
        for turn_index, turn in enumerate(turns, 1):
            for call in turn.get("calls", []):
                for spec in call.get("arguments", {}).values():
                    total_arguments += 1
                    stack = [spec]
                    argument_has_valid_source = False
                    argument_has_invalid_leaf = False
                    while stack:
                        item = stack.pop()
                        if isinstance(item, dict) and "source" in item:
                            source = str(item.get("source", "")).casefold()
                            source_counts[source] = (
                                source_counts.get(source, 0) + 1
                            )
                            if source in {
                                "user", "history", "schema_default",
                                "tool_output",
                            }:
                                argument_has_valid_source = True
                            else:
                                argument_has_invalid_leaf = True
                            if source == "tool_output":
                                tool_output_bindings += 1
                                producer_turn = call_turn.get(
                                    str(item.get("call_id", "")), turn_index
                                )
                                if producer_turn < turn_index:
                                    cross_turn_bindings += 1
                        elif isinstance(item, dict):
                            stack.extend(item.values())
                        elif isinstance(item, list):
                            stack.extend(item)
                        else:
                            argument_has_invalid_leaf = True
                    if argument_has_invalid_leaf or not argument_has_valid_source:
                        hidden_argument_count += 1
        return {
            "total_arguments": total_arguments,
            "hidden_argument_count": hidden_argument_count,
            "hidden_argument_fraction": (
                hidden_argument_count / total_arguments
                if total_arguments
                else 0.0
            ),
            "argument_source_counts": source_counts,
            "tool_output_binding_count": tool_output_bindings,
            "cross_turn_binding_count": cross_turn_bindings,
        }

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
            # This is a mechanical serialization constraint, not another
            # semantic check.  Asking the provider to close/escape the JSON
            # prevents an otherwise complete episode from being discarded
            # after all of its tools have already executed.
            response_format={"type": "json_object"},
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
        *,
        retry_on_unavailable: bool = True,
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
                response_format={"type": "json_object"},
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
            if retry_on_unavailable:
                print("    Retrying batched turn-response judge once")
                return self._verify_batched_turn_responses(
                    evidence,
                    responses,
                    retry_on_unavailable=False,
                )
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
        except Exception as first_exc:
            print(
                "    Warning: batched turn-response writer failed; "
                f"retrying once: {first_exc}"
            )
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

            if self.symbolic_episode_plan:
                trajectory, ec = self._execute_symbolic_blueprint_turn(
                    query_result=query_result,
                    turn_spec=turn_spec,
                    execution_context=execution_context,
                )
            else:
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
            execution_context.setdefault('prior_user_queries', []).append(
                query_result.query
            )

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
                "blueprint_min_total_actions": self.blueprint_min_total_actions,
                "blueprint_max_total_actions": self.blueprint_max_total_actions,
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "resumed_from_turn": completed_turns,
                "blueprint_queries": [t.get("user_query", "") for t in blueprint.turns],
                "turn_expected_tools": [t.get("expected_tools", []) for t in blueprint.turns],
                "symbolic_episode_plan": self.symbolic_episode_plan,
                "symbolic_call_graph": (
                    copy.deepcopy(blueprint.turns)
                    if self.symbolic_episode_plan
                    else None
                ),
                "symbolic_plan_metrics": (
                    self._symbolic_plan_metrics(blueprint.turns)
                    if self.symbolic_episode_plan
                    else None
                ),
                "rl_quality_gate_passed": True,
                "model_routing": self._model_routing_metadata(),
                "tool_contract_hash": self._tool_contract_hash(available_tools),
                "generation_pipeline": (
                    "symbolic_episode_plan_v2_batched_turn_responses"
                    if self.symbolic_episode_plan
                    else (
                        "turn_compiler_v1_batched_turn_responses"
                        if self.optimized_pipeline
                        else "legacy_per_tool"
                    )
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
            if self.symbolic_episode_plan:
                trajectory, ec = self._execute_symbolic_blueprint_turn(
                    query_result=query_result,
                    turn_spec=turn_spec,
                    execution_context=execution_context,
                )
            else:
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
            execution_context.setdefault('prior_user_queries', []).append(
                query_result.query
            )

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
                "blueprint_min_total_actions": self.blueprint_min_total_actions,
                "blueprint_max_total_actions": self.blueprint_max_total_actions,
                "focus_category": focus_category,
                "overall_task": blueprint.overall_task,
                "blueprint_queries": [t.get("user_query", "") for t in blueprint.turns],
                "turn_expected_tools": [t.get("expected_tools", []) for t in blueprint.turns],
                "symbolic_episode_plan": self.symbolic_episode_plan,
                "symbolic_call_graph": (
                    copy.deepcopy(blueprint.turns)
                    if self.symbolic_episode_plan
                    else None
                ),
                "symbolic_plan_metrics": (
                    self._symbolic_plan_metrics(blueprint.turns)
                    if self.symbolic_episode_plan
                    else None
                ),
                "rl_quality_gate_passed": True,
                "model_routing": self._model_routing_metadata(),
                "tool_contract_hash": self._tool_contract_hash(available_tools),
                "generation_pipeline": (
                    "symbolic_episode_plan_v2_batched_turn_responses"
                    if self.symbolic_episode_plan
                    else (
                        "turn_compiler_v1_batched_turn_responses"
                        if self.optimized_pipeline
                        else "legacy_per_tool"
                    )
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

        # Symbolic plans already expose the exact arguments that will be
        # executed.  Validate those values instead of guessing handles from
        # nearby English words (for example, "follow both now" previously
        # treated both ``both`` and ``now`` as usernames and discarded a
        # valid, already-paid-for blueprint).  Dynamic tool-output bindings
        # are checked when the simulator materializes them.
        username_parameters = {
            "authenticate_twitter": ("username",),
            "follow_user": ("username_to_follow",),
            "unfollow_user": ("username_to_unfollow",),
            "get_user_stats": ("username",),
            "get_user_tweets": ("username",),
            "mention": ("mentioned_usernames",),
        }
        if any(isinstance(turn.get("calls"), list) for turn in turns):
            valid_casefolded = {
                str(username).lstrip("@").casefold()
                for username in valid_usernames
            }
            for turn_idx, turn in enumerate(turns, 1):
                for call in turn.get("calls", []):
                    if not isinstance(call, dict):
                        continue
                    tool_name = str(call.get("tool_name", ""))
                    parameter_names = username_parameters.get(tool_name, ())
                    arguments = call.get("arguments", {})
                    if not isinstance(arguments, dict):
                        continue
                    for parameter_name in parameter_names:
                        spec = arguments.get(parameter_name)
                        if not isinstance(spec, dict):
                            continue
                        if str(spec.get("source", "")).casefold() not in {
                            "user",
                            "history",
                        }:
                            continue
                        value = spec.get("value")
                        values = value if isinstance(value, list) else [value]
                        for username in values:
                            if (
                                isinstance(username, str)
                                and username.lstrip("@").casefold()
                                not in valid_casefolded
                            ):
                                issues.append(
                                    f"Turn {turn_idx}: {tool_name}."
                                    f"{parameter_name} references username "
                                    f"'{username}' but it does not exist in "
                                    "state. Valid users: "
                                    + ", ".join(sorted(valid_usernames))[:100]
                                )
            return issues

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
            'all', 'both', 'each', 'either', 'neither', 'now',
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
        api_class_map = getattr(
            self.tool_manager, "api_name_to_class_key", {}
        )
        if not isinstance(api_class_map, dict):
            api_class_map = {}
        selected_class_keys = {
            api_class_map.get(tool_name) for tool_name in seen_tool_names
        }
        selected_class_keys.discard(None)
        if initial_api_state:
            for class_key, state in initial_api_state.items():
                if selected_class_keys and class_key not in selected_class_keys:
                    continue
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
Decision calibration: reject only a material capability, grounding,
provenance, state, necessity, or sufficiency error. Do not reject harmless
wording, redundancy, or a natural semantic equivalent of an explicit argument
(for example, “nearest integer” is decimal_places=0). A query may use normal
singular/plural or unit wording while the fixed argument uses the schema's
canonical enum token. If every requested outcome is supported and every call
is necessary, mark the episode valid even when the phrasing is not elegant.
The assistant may summarize, report, compare, or confirm facts directly from
the selected tool outputs. Those ordinary response acts do not require a
separate summarization/reporting tool; reject them only when they require an
uncalled lookup, calculation, aggregation, or unsupported fact.

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
11. A later user_query must not assert a concrete price, ID, status, sensor
    reading, search result, or other result that only an earlier planned tool
    will reveal. Natural references such as "that result" are valid; a
    pre-written outcome is hidden-state leakage. Reject especially when the
    asserted outcome conflicts with GENERATOR-ONLY CURRENT API STATE.
12. Authentication and existing-entity calls must be feasible in the supplied
    state. A display name cannot substitute for an opaque ID. If a selected
    lookup can produce the ID, require an explicit earlier symbolic binding;
    otherwise require the exact valid ID in the user utterance.
13. CALL NECESSITY: map every individual selected-tool occurrence to one
    explicit clause in the current user_query or a necessary dependency of
    that clause. Reject any extra lookup, status read, mutation, notification,
    or verification the user did not request. Reaching a target call count is
    never a justification.
14. PLAN SUFFICIENCY: map every independently requested user clause to a tool
    that can actually satisfy it. Reject a superficially related tool set that
    omits a required capability (for example, a sector request without a tool
    that filters or reports sectors).
15. If a later argument selects one element from an array output, the user must
    state an ordinal or deterministic criterion. Reject silent index-zero or
    arbitrary candidate selection.

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
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "blueprint_semantic_verdict",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "is_valid": {"type": "boolean"},
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["is_valid", "issues"],
                            "additionalProperties": False,
                        },
                    },
                },
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
            return False, [f"Capability check error: {str(e)[:100]}"]

    # ─────────────────────── Stage 0: Blueprint ───────────────────────

    def _generate_symbolic_blueprint_turnwise(
        self,
        *,
        tools_json: List[Dict[str, Any]],
        directive: Dict[str, Any],
        exact_action_schedule: List[int],
        focus_category: Optional[str],
        initial_state_context: str,
        credential_context: str,
    ) -> Optional[Dict[str, Any]]:
        """Compile a high-width episode one complete turn per request.

        A cheap teacher often loses schema fidelity when asked to emit 15-20
        calls in one JSON object.  Turn-level compilation keeps the important
        architecture unchanged (teacher proposes, Python executes/verifies)
        while fitting the complete pipeline in ten requests: at most seven
        blueprint/repair calls plus one semantic judge, one batched response
        writer and one grounding judge.
        """
        available_names = {
            str(tool.get("name") or tool.get("api_name", ""))
            for tool in tools_json
            if tool.get("name") or tool.get("api_name")
        }
        tools_str = json.dumps(
            tools_json, indent=2, ensure_ascii=False, default=str
        )
        hard_required = [
            str(name) for name in directive.get("hard_required_tools", [])
        ]
        required_by_turn: List[List[str]] = [
            [] for _ in exact_action_schedule
        ]
        # Spread hard coverage targets without exceeding any turn's exact
        # capacity.  The first-turn scaffold sees this full map and can make a
        # coherent episode arc around it.
        next_turn = 0
        for tool_name in hard_required:
            for _ in range(len(required_by_turn)):
                if (
                    len(required_by_turn[next_turn])
                    < exact_action_schedule[next_turn]
                ):
                    required_by_turn[next_turn].append(tool_name)
                    next_turn = (next_turn + 1) % len(required_by_turn)
                    break
                next_turn = (next_turn + 1) % len(required_by_turn)

        selected_categories = sorted(
            {
                str(tool.get("category", ""))
                for tool in tools_json
                if tool.get("category")
            }
        )
        domain_hints = "\n".join(
            hint
            for hint in (
                get_domain_hints(category) for category in selected_categories
            )
            if hint
        )
        total_calls = sum(exact_action_schedule)
        accepted_turns: List[Dict[str, Any]] = []
        overall_task = ""
        future_intents: List[str] = []
        # Reserve exactly four calls for batched query alignment, the episode
        # semantic judge, batched response writer and grounding judge.
        repair_budget = max(
            0,
            min(
                7 - len(exact_action_schedule),
                self.max_calls_per_candidate
                - 4
                - len(exact_action_schedule),
            ),
        )

        for turn_index, required_count in enumerate(
            exact_action_schedule, start=1
        ):
            current_intent = (
                future_intents[turn_index - 2]
                if turn_index > 1 and len(future_intents) >= turn_index - 1
                else ""
            )
            feedback = ""
            attempts = 1 + (1 if repair_budget > 0 else 0)
            turn_accepted = False
            for attempt in range(attempts):
                prior_context = json.dumps(
                    accepted_turns,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                if turn_index == 1:
                    output_contract = {
                        "overall_task": "one coherent user-facing scenario",
                        "future_turn_intents": [
                            f"natural intent for turn {index}"
                            for index in range(2, self.num_turns + 1)
                        ],
                        "turn": {
                            "user_query": "natural compound request",
                            "intent": "current semantic intent",
                            "calls": [],
                        },
                    }
                    scaffold_rule = (
                        f"Also return exactly {self.num_turns - 1} concise "
                        "future_turn_intents. They must form one realistic "
                        "conversation arc and anticipate the required tool "
                        "coverage map below without asserting future results."
                    )
                else:
                    output_contract = {
                        "turn": {
                            "user_query": "natural follow-up",
                            "intent": "current semantic intent",
                            "calls": [],
                        }
                    }
                    scaffold_rule = (
                        "Follow the supplied current intent while reacting "
                        "naturally to prior policy-visible results."
                    )

                prompt = f"""Compile turn {turn_index}/{self.num_turns} of one
coherent executable tool-use episode.

=== EPISODE CONTRACT ===
Total executed calls: exactly {total_calls}.
Exact calls by turn: {json.dumps(exact_action_schedule)}.
This turn must contain exactly {required_count} calls with IDs
t{turn_index}c1 through t{turn_index}c{required_count}; do not omit, add, or
renumber any call. Write one natural compound user request whose explicit
deliverables or necessary dependencies motivate every call. Never add filler.
{scaffold_rule}

Overall task fixed so far: {overall_task or "create it now"}
Current planned intent: {current_intent or "create it now"}
Hard required tools for this turn:
{json.dumps(required_by_turn[turn_index - 1], ensure_ascii=False)}
Hard-tool allocation for the whole episode:
{json.dumps(required_by_turn, ensure_ascii=False)}

=== PRIOR ACCEPTED SYMBOLIC TURNS ===
{prior_context if accepted_turns else "None. This is the first turn."}

=== AVAILABLE TOOL SCHEMAS ===
{tools_str}

=== SYMBOLIC RULES ===
1. Emit every required schema argument. Each scalar/array argument is exactly
   one of: {{"source":"user","value":...}},
   {{"source":"history","value":...}}, {{"source":"schema_default"}}, or
   {{"source":"tool_output","call_id":"earlier_call","path":"declared.path"}}.
   Omit optional arguments unless the user explicitly requested the behavior;
   in particular, never enable a boolean flag silently.
2. A tool_output binding may reference only an earlier call and must use an
   exact field from that tool's output_schema. Never predict its value. Put the
   producer in depends_on. Arrays may consume only actual array outputs.
3. Make every user/history literal visible verbatim in the current/prior user
   utterances. Opaque IDs/symbols should instead come from a capable lookup.
4. Include explicit ordering dependencies for authentication, mutation then
   verification, and user-requested sequencing. Set parallel_group to null.
5. Do not assert a concrete future result. Later utterances may say “that
   result”, and their calls should retain symbolic output bindings.
6. Use only feasible entities/credentials from generator state, and expose any
   state-derived required value naturally in the user utterance.
7. Audit exact call count, types, enums, output paths, dependencies and literal
   visibility before returning JSON.
8. Map every independently requested user clause to a capable call, and every
   call to an explicit clause or strictly necessary dependency. Do not mention
   an action, comparison, or report that this turn's calls cannot complete.
9. Treat output_schema as authoritative. Do not describe one returned object
   as a list, and do not promise work on “all results” when a downstream tool
   accepts only one scalar. Any array-element selection needs an explicit user
   criterion; never silently choose index zero or substitute unrelated items.
10. Every requested fact must be present in a selected tool output or be a
    direct report of a successful mutation. A related call is not sufficient.
11. Compare every mutation against generator-only state before proposing it.
    It must have a real effect: do not create an existing entity, delete a
    missing one, repeat an existing follow/watchlist membership, or set a field
    to the value it already has.
12. When REPAIR FEEDBACK reports a failed tool or state action, do not emit the
    same tool with the same arguments again. Replace it with a state-feasible
    action that still serves the intent, or rewrite the intent around a
    different necessary capability.
13. Preserve entity identity across turns. If the user says "that", "same", or
    "just created", bind the later identifier directly to the original
    create/lookup call output. Never replace it with the first result of an
    unrelated broad list/search call; a verification read must consume the
    exact identifier already established for that entity.

{domain_hints}

=== GENERATOR-ONLY API STATE (NOT POLICY-VISIBLE) ===
{initial_state_context or "No mutable state for these tools."}
{credential_context}

=== OUTPUT ===
Return only valid JSON shaped like:
{json.dumps(output_contract, ensure_ascii=False, indent=2)}
"""
                if feedback:
                    prompt += (
                        "\n=== REPAIR FEEDBACK ===\n"
                        + feedback
                        + "\nRepair this turn only; preserve the coherent goal."
                    )
                call_schema = {
                    "type": "object",
                    "properties": {
                        "call_id": {"type": "string"},
                        "tool_name": {
                            "type": "string",
                            "enum": sorted(available_names),
                        },
                        "arguments": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "parallel_group": {"type": "null"},
                    },
                    "required": [
                        "call_id",
                        "tool_name",
                        "arguments",
                        "depends_on",
                        "parallel_group",
                    ],
                    "additionalProperties": False,
                }
                turn_schema = {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string"},
                        "intent": {"type": "string"},
                        "calls": {
                            "type": "array",
                            "items": call_schema,
                            "minItems": required_count,
                            "maxItems": required_count,
                        },
                    },
                    "required": ["user_query", "intent", "calls"],
                    "additionalProperties": False,
                }
                if turn_index == 1:
                    response_schema = {
                        "type": "object",
                        "properties": {
                            "overall_task": {"type": "string"},
                            "future_turn_intents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": self.num_turns - 1,
                                "maxItems": self.num_turns - 1,
                            },
                            "turn": turn_schema,
                        },
                        "required": [
                            "overall_task",
                            "future_turn_intents",
                            "turn",
                        ],
                        "additionalProperties": False,
                    }
                else:
                    response_schema = {
                        "type": "object",
                        "properties": {"turn": turn_schema},
                        "required": ["turn"],
                        "additionalProperties": False,
                    }
                response = self._safe_llm_generate(
                    [{"role": "user", "content": prompt}],
                    purpose="blueprint_turn_compile",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"symbolic_turn_{turn_index}",
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                )
                response_text = response.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json", 1)[1].split(
                        "```", 1
                    )[0]
                elif "```" in response_text:
                    response_text = response_text.split("```", 1)[1].split(
                        "```", 1
                    )[0]
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                try:
                    parsed = json.loads(response_text[start:end])
                except (json.JSONDecodeError, ValueError) as exc:
                    feedback = f"Return valid JSON: {exc}."
                    if attempt == 0 and attempts > 1:
                        repair_budget -= 1
                    continue

                raw_turn = parsed.get("turn")
                if not isinstance(raw_turn, dict):
                    feedback = "Output must contain one object named turn."
                    if attempt == 0 and attempts > 1:
                        repair_budget -= 1
                    continue
                raw_calls = raw_turn.get("calls", [])
                if not isinstance(raw_calls, list) or len(raw_calls) != required_count:
                    feedback = (
                        f"This turn has {len(raw_calls) if isinstance(raw_calls, list) else 0} "
                        f"calls; return exactly {required_count} calls with the "
                        "prescribed IDs."
                    )
                    if attempt == 0 and attempts > 1:
                        repair_budget -= 1
                    continue

                combined, issues = self._normalise_symbolic_blueprint_turns(
                    [*accepted_turns, raw_turn],
                    available_tool_names=available_names,
                )
                current_tools = (
                    combined[-1].get("expected_tools", []) if combined else []
                )
                missing_current = [
                    name
                    for name in required_by_turn[turn_index - 1]
                    if name not in current_tools
                ]
                if missing_current:
                    issues.append(
                        "Missing hard tools for this turn: "
                        + ", ".join(missing_current)
                    )
                if not issues:
                    issues.extend(
                        self._preflight_symbolic_blueprint_execution(combined)
                    )
                if issues:
                    feedback = "\n".join(issues[:20])
                    if attempt == 0 and attempts > 1:
                        repair_budget -= 1
                    continue

                if turn_index == 1:
                    proposed_task = str(parsed.get("overall_task", "")).strip()
                    proposed_intents = parsed.get("future_turn_intents", [])
                    if (
                        not proposed_task
                        or not isinstance(proposed_intents, list)
                        or len(proposed_intents) != self.num_turns - 1
                        or not all(str(item).strip() for item in proposed_intents)
                    ):
                        feedback = (
                            "The first-turn output needs a non-empty overall_task "
                            f"and exactly {self.num_turns - 1} non-empty "
                            "future_turn_intents."
                        )
                        if attempt == 0 and attempts > 1:
                            repair_budget -= 1
                        continue
                    overall_task = proposed_task
                    future_intents = [
                        str(item).strip() for item in proposed_intents
                    ]
                accepted_turns = combined
                turn_accepted = True
                break

            if not turn_accepted:
                print(
                    f"  ✗ Turnwise symbolic compiler failed at turn "
                    f"{turn_index}: {feedback[:500]}"
                )
                return None

        return {"overall_task": overall_task, "turns": accepted_turns}

    def _align_symbolic_blueprint_queries(
        self,
        *,
        overall_task: str,
        turns: List[Dict[str, Any]],
        tools_json: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], List[str]]:
        """Rewrite all user utterances once against an immutable call graph.

        Blueprint teachers are much better at compiling a valid symbolic graph
        when they can concentrate on structure, but their prose can promise
        more work than that graph performs.  One episode-level pass makes the
        requests and fixed calls bijective without bringing back a per-turn or
        per-tool judge.  Python then repeats provenance and execution checks.
        """
        if not turns:
            return None, None, ["Cannot align an empty symbolic episode."]

        used_tool_names = {
            str(call.get("tool_name", ""))
            for turn in turns
            for call in turn.get("calls", [])
        }
        used_contracts = [
            {
                "name": tool.get("name") or tool.get("api_name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "output_schema": tool.get("output_schema", {}),
            }
            for tool in tools_json
            if str(tool.get("name") or tool.get("api_name", ""))
            in used_tool_names
        ]
        fixed_plan = [
            {
                "turn_number": turn_index,
                "current_user_query": str(turn.get("user_query", "")),
                "intent": str(turn.get("intent", "")),
                "calls": copy.deepcopy(turn.get("calls", [])),
            }
            for turn_index, turn in enumerate(turns, 1)
        ]
        prompt = f"""Rewrite the user-facing language for one fixed tool-use
episode. The symbolic calls are immutable: do not add, remove, reorder, rename,
or redesign them. Return all rewritten turns in one response.

=== CURRENT OVERALL TASK ===
{overall_task}

=== FIXED SYMBOLIC PLAN ===
{json.dumps(fixed_plan, ensure_ascii=False, indent=2, default=str)}

=== CONTRACTS FOR USED TOOLS ===
{json.dumps(used_contracts, ensure_ascii=False, indent=2, default=str)}

=== ALIGNMENT RULES ===
1. Each user utterance must request exactly the outcomes that its fixed calls
   can produce: no omitted requested result and no unsupported extra result.
2. Every fixed call must be motivated by an explicit requested outcome or be a
   strictly necessary dependency of one. Do not expose dependency mechanics.
3. Preserve every exact literal carried by a `user` source in that same turn.
   Preserve every `history` literal in an earlier user turn. Do not invent any
   literal, identifier, fact, result, calculation, mutation, or constraint.
4. A `tool_output` source is learned only by executing its producer. Refer to
   it naturally (for example, “that record” or “the result you found”); never
   put an internal call ID or predicted output value in user speech.
5. Make later turns coherent reactions or follow-ups. The user may refer to
   prior visible results but must not know future or hidden simulator state.
6. Sound like a real end user stating goals and constraints. Never mention
   tools, APIs, schemas, function names, call IDs, provenance, benchmarks, or
   the execution plan. Do not format the request as a numbered checklist.
7. If several fixed calls repeat a capability for different literal inputs,
   request every one of those items explicitly. If the graph performs only a
   subset of a calculation or collection, request only that exact subset.
8. Keep the task genuinely compound. Do not simplify away supported work, but
   do not add prose merely to make it sound harder.

Return only JSON with one concise aligned overall task and exactly
{len(turns)} turn objects in numeric order.
"""
        response_schema = {
            "type": "object",
            "properties": {
                "overall_task": {"type": "string"},
                "turns": {
                    "type": "array",
                    "minItems": len(turns),
                    "maxItems": len(turns),
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn_number": {"type": "integer"},
                            "user_query": {"type": "string"},
                        },
                        "required": ["turn_number", "user_query"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["overall_task", "turns"],
            "additionalProperties": False,
        }
        try:
            response = self._safe_llm_generate(
                [{"role": "user", "content": prompt}],
                purpose="blueprint_query_align",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "aligned_episode_queries",
                        "strict": True,
                        "schema": response_schema,
                    },
                },
            ).strip()
            if "```json" in response:
                response = response.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response:
                response = response.split("```", 1)[1].split("```", 1)[0]
            start = response.find("{")
            end = response.rfind("}") + 1
            parsed = json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as exc:
            return None, None, [f"Query alignment failed: {exc}"]

        aligned_task = str(parsed.get("overall_task", "")).strip()
        aligned_items = parsed.get("turns", [])
        if not aligned_task:
            return None, None, ["Query alignment returned an empty overall task."]
        if not isinstance(aligned_items, list) or len(aligned_items) != len(turns):
            return None, None, [
                "Query alignment returned the wrong number of turns: "
                f"expected {len(turns)}, got "
                f"{len(aligned_items) if isinstance(aligned_items, list) else 0}."
            ]

        rewritten = copy.deepcopy(turns)
        errors: List[str] = []
        import re
        internal_id_pattern = re.compile(r"\bt\d+c\d+\b", re.IGNORECASE)
        seen_turn_numbers: set[int] = set()
        for expected_number, item in enumerate(aligned_items, 1):
            if not isinstance(item, dict):
                errors.append(f"Aligned turn {expected_number} is not an object.")
                continue
            turn_number = item.get("turn_number")
            query = str(item.get("user_query", "")).strip()
            if turn_number != expected_number or turn_number in seen_turn_numbers:
                errors.append(
                    f"Aligned turn order is invalid at position {expected_number}."
                )
                continue
            seen_turn_numbers.add(turn_number)
            if not query:
                errors.append(f"Aligned turn {expected_number} is empty.")
                continue
            if internal_id_pattern.search(query):
                errors.append(
                    f"Aligned turn {expected_number} exposes an internal call ID."
                )
                continue
            rewritten[expected_number - 1]["user_query"] = query

        if errors:
            return None, None, errors
        available_names = {
            str(tool.get("name") or tool.get("api_name", ""))
            for tool in tools_json
            if tool.get("name") or tool.get("api_name")
        }
        canonical, normalisation_errors = self._normalise_symbolic_blueprint_turns(
            rewritten,
            available_tool_names=available_names,
        )
        errors.extend(normalisation_errors)
        if not errors:
            errors.extend(self._preflight_symbolic_blueprint_execution(canonical))
        if errors:
            return None, None, errors
        return aligned_task, canonical, []

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
        turnwise_symbolic = bool(
            self.symbolic_episode_plan
            and exact_action_schedule
            and max(exact_action_schedule) >= max(
                1,
                int(os.getenv("APIGEN_SYMBOLIC_TURNWISE_MIN_WIDTH", "5")),
            )
            and os.getenv("APIGEN_SYMBOLIC_TURNWISE", "1").strip().casefold()
            not in {"0", "false", "no", "off"}
        )
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

        if self.symbolic_episode_plan:
            max_total_calls = (
                self.blueprint_max_total_actions
                if self.blueprint_max_total_actions is not None
                else self.num_turns * max_tools_per_turn
            )
            min_total_calls = min(
                max_total_calls,
                (
                    self.blueprint_min_total_actions
                    if self.blueprint_min_total_actions is not None
                    else max(self.num_turns, self.num_turns * 2)
                ),
            )
            # A broad "roughly 15-20" instruction encouraged teachers to
            # return 21 calls and forced a complete paid regeneration.  Sample
            # one concrete in-range target for this episode; validation still
            # enforces the configured inclusive bounds, so this improves yield
            # without weakening the complexity floor.
            if exact_action_schedule is not None:
                target_total_calls = sum(exact_action_schedule)
                symbolic_schedule_requirement = (
                    "Use exactly this calls-array length by turn: "
                    + ", ".join(
                        f"turn {index + 1}={count}"
                        for index, count in enumerate(exact_action_schedule)
                    )
                    + "."
                )
            else:
                target_total_calls = random.randint(
                    min_total_calls,
                    max_total_calls,
                )
                symbolic_schedule_requirement = (
                    f"Use 1-{max_tools_per_turn} calls in each turn."
                )
            prompt = f"""Design one coherent, executable {self.num_turns}-turn
tool-use conversation and compile its complete symbolic call graph in the SAME
response.

=== AVAILABLE TOOLS ===
{tools_str}
{directive_section}

=== GOAL ===
Create a realistic task with exactly {target_total_calls} necessary calls across
the episode. {symbolic_schedule_requirement} Count the `calls` array items in
every turn and across the episode before returning JSON. The inclusive
{min_total_calls}-{max_total_calls} range remains a hard validity constraint;
never add filler merely to hit the sampled target.
For any turn assigned five or more calls, deliberately write one natural
compound request with several related deliverables and constraints. Build the
required call graph first, then phrase the user request so that every scheduled
call is explicitly motivated. It is fine to invoke the same capability for
different user-requested items, but never return fewer calls than the exact
schedule and never invent an unrelated check merely to fill a slot.
Prefer genuine discovery → action → verification workflows, state changes that
are used later, and follow-ups whose resolution depends on prior visible tool
results. When the supplied tools permit it, include at least two tool-output
bindings and at least one binding across user turns.

=== USER-LANGUAGE RULES ===
1. Write natural end-user utterances, not API instructions, argument dumps,
   numbered plans, benchmark prose, or requests for internal IDs.
2. The user states goals, constraints and values they plausibly know. If a
   lookup can discover an opaque ID/token/symbol, call the lookup and bind the
   later argument to its output instead of putting that opaque value in speech.
3. Each later turn must be a plausible reaction or follow-up to the prior
   visible conversation. Do not make every turn an unrelated new task. Never
   pre-write a concrete price, ID, status, balance, search result or other value
   that only a future tool execution could reveal; say "that option/result"
   and retain a symbolic tool_output binding instead.
4. Include every literal required by a call naturally in the current or an
   earlier user utterance unless it comes from an earlier call or schema default.
5. The generator-only state appended below is NOT visible to the solving model.
   Never source an argument silently from it.
6. Existing-entity and authentication calls must be simulator-feasible. Use an
   exact valid credential/entity value from generator state only after writing
   it naturally into the user utterance. If an available lookup can translate a
   natural name into the required opaque ID, use that lookup first and bind its
   output; never pass a display name where an ID is required.

=== SYMBOLIC EXECUTION CONTRACT ===
1. Use globally unique call IDs: t1c1, t1c2, ..., t2c1, and so on.
2. Emit every required argument and no argument forbidden by the schema.
3. Each argument must use exactly one provenance form:
   - {{"source":"user","value":...}} when the exact value is visible in this
     turn's utterance;
   - {{"source":"history","value":...}} when the exact value is visible in an
     earlier USER utterance (not merely in generator state or a predicted tool
     response);
   - {{"source":"schema_default"}} only if the schema declares a default;
   - {{"source":"tool_output","call_id":"t1c1","path":"field.subfield"}}
     when an earlier call returns it.
   For every ARRAY parameter, always wrap the complete array in one source
   object, for example {{"source":"user","value":["#one","#two"]}} or
   {{"source":"schema_default"}}; never emit a raw array, an `items` wrapper,
   or per-element provenance. For OBJECT parameters, either wrap the complete
   object in one source or preserve its JSON shape and put a source object at
   every scalar leaf (for example updates.priority). Never leave a leaf
   unbound.
4. Never predict or copy a future concrete tool output. A tool_output binding
   contains call_id and path only, never a guessed value.
5. A call may depend only on a call listed earlier in the episode. Use the exact
   output_schema field name and place all referenced call IDs in depends_on.
   Also include semantic ordering prerequisites even when no value is passed:
   authenticate -> protected operation, mutation -> verification read, and any
   explicit "then/after" ordering in the user request. Do not rely on array
   order alone to express these prerequisites.
6. This V2 plan is sequential: set parallel_group to null for every call.
7. Every call must be necessary for the user-visible goal. Prefer multi-hop
   dependencies and meaningful mutations/read-after-write over independent
   trivia, but do not claim results absent from tool outputs.
8. Authentication persists. Authenticate only before the first protected call.
9. Dates must be on or after {datetime.now(timezone.utc).date().isoformat()}.
10. Privately audit schema types, enums, state feasibility, argument provenance,
    output paths and dependency order before returning JSON.
11. For user/history bindings, make the exact scalar spelling visible somewhere
    in the current/prior user utterances, including enum tokens, units, dates,
    filenames and IDs. Natural surrounding prose is encouraged; hidden aliases
    are not.
12. Before returning, map every call occurrence to an explicit user request or
    a strictly necessary dependency, and map every user request to a capable
    call. Delete unrequested status/time/lookups and add the actually required
    capability instead. Never pad the graph to reach the target count.
13. Preserve entity identity across turns. A later reference to "that",
    "same", or "just created" must bind directly to the original create/lookup
    output. Never silently substitute the first item from a broad list/search
    call; use the established identifier for all later reads and mutations.

=== OUTPUT ===
Return ONLY valid JSON with exactly {self.num_turns} turns:
{{
  "overall_task": "one concise user-facing scenario",
  "turns": [
    {{
      "user_query": "natural request",
      "intent": "short semantic intent",
      "calls": [
        {{
          "call_id": "t1c1",
          "tool_name": "exact_tool_name",
          "arguments": {{
            "argument_name": {{"source":"user","value":"exact visible value"}}
          }},
          "depends_on": [],
          "parallel_group": null
        }}
      ]
    }}
  ]
}}
"""

        if focus_category and not allowed_tools:
            prompt += f"\n\nAll available tools below are from the '{focus_category}' category."

        selected_categories = sorted(
            {
                str(tool.get("category", ""))
                for tool in tools_json
                if tool.get("category")
            }
        )
        combined_domain_hints = "\n".join(
            hint
            for hint in (
                get_domain_hints(category) for category in selected_categories
            )
            if hint
        )
        if combined_domain_hints:
            prompt += f"\n\n{combined_domain_hints}"

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

        symbolic_response_format: Optional[Dict[str, Any]] = None
        if self.symbolic_episode_plan and not turnwise_symbolic:
            available_names = sorted(
                {
                    str(tool.get("name") or tool.get("api_name", ""))
                    for tool in tools_json
                    if tool.get("name") or tool.get("api_name")
                }
            )
            minimum_turn_calls = (
                min(exact_action_schedule) if exact_action_schedule else 1
            )
            maximum_turn_calls = (
                max(exact_action_schedule)
                if exact_action_schedule
                else max_tools_per_turn
            )
            call_schema = {
                "type": "object",
                "properties": {
                    "call_id": {
                        "type": "string",
                        "pattern": r"^t[1-9][0-9]*c[1-9][0-9]*$",
                    },
                    "tool_name": {"type": "string", "enum": available_names},
                    "arguments": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "parallel_group": {"type": "null"},
                },
                "required": [
                    "call_id",
                    "tool_name",
                    "arguments",
                    "depends_on",
                    "parallel_group",
                ],
                "additionalProperties": False,
            }
            turn_schema = {
                "type": "object",
                "properties": {
                    "user_query": {"type": "string"},
                    "intent": {"type": "string"},
                    "calls": {
                        "type": "array",
                        "items": call_schema,
                        "minItems": minimum_turn_calls,
                        "maxItems": maximum_turn_calls,
                    },
                },
                "required": ["user_query", "intent", "calls"],
                "additionalProperties": False,
            }
            symbolic_response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "symbolic_episode_blueprint",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "overall_task": {"type": "string"},
                            "turns": {
                                "type": "array",
                                "items": turn_schema,
                                "minItems": self.num_turns,
                                "maxItems": self.num_turns,
                            },
                        },
                        "required": ["overall_task", "turns"],
                        "additionalProperties": False,
                    },
                },
            }

        accumulated_feedback = ""
        max_blueprint_attempts = max(
            1,
            min(
                2,
                int(os.getenv("APIGEN_MAX_BLUEPRINT_ATTEMPTS", "2")),
            ),
        )
        if turnwise_symbolic:
            # Turnwise compilation already owns its bounded repair budget.
            max_blueprint_attempts = 1
        for attempt in range(max_blueprint_attempts):
            try:
                if turnwise_symbolic:
                    turnwise_result = self._generate_symbolic_blueprint_turnwise(
                        tools_json=tools_json,
                        directive=directive,
                        exact_action_schedule=exact_action_schedule,
                        focus_category=focus_category,
                        initial_state_context=initial_state_context,
                        credential_context=credential_context,
                    )
                    if turnwise_result is None:
                        continue
                    response = json.dumps(
                        turnwise_result, ensure_ascii=False, default=str
                    )
                elif accumulated_feedback:
                    prompt_with_feedback = prompt + f"\n\n=== PREVIOUS ATTEMPT FEEDBACK ===\n{accumulated_feedback}\n=== END FEEDBACK ===\n"
                    response = self._safe_llm_generate(
                        [{"role": "user", "content": prompt_with_feedback}],
                        purpose="blueprint_generate",
                        **(
                            {"response_format": symbolic_response_format}
                            if symbolic_response_format
                            else {}
                        ),
                    )
                else:
                    response = self._safe_llm_generate(
                        [{"role": "user", "content": prompt}],
                        purpose="blueprint_generate",
                        **(
                            {"response_format": symbolic_response_format}
                            if symbolic_response_format
                            else {}
                        ),
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

                if self.symbolic_episode_plan:
                    turns, symbolic_errors = (
                        self._normalise_symbolic_blueprint_turns(
                            turns,
                            available_tool_names={
                                str(
                                    tool.get("name")
                                    or tool.get("api_name", "")
                                )
                                for tool in tools_json
                                if tool.get("name") or tool.get("api_name")
                            },
                        )
                    )
                    actual_total_calls = sum(
                        len(turn.get("calls", [])) for turn in turns
                    )
                    if exact_action_schedule is not None:
                        for turn_index, (turn, required_count) in enumerate(
                            zip(turns, exact_action_schedule),
                            start=1,
                        ):
                            actual_count = len(turn.get("calls", []))
                            if actual_count != required_count:
                                symbolic_errors.append(
                                    f"Turn {turn_index} has {actual_count} "
                                    f"calls; exactly {required_count} required."
                                )
                    configured_minimum = (
                        self.blueprint_min_total_actions
                        if self.blueprint_min_total_actions is not None
                        else min(
                            self.num_turns
                            * self.blueprint_max_actions_per_turn,
                            self.num_turns * 2,
                        )
                    )
                    if actual_total_calls < configured_minimum:
                        symbolic_errors.append(
                            f"Episode has {actual_total_calls} necessary calls; "
                            f"minimum is {configured_minimum}. Build a more "
                            "substantive coherent workflow without filler."
                        )
                    configured_maximum = (
                        self.blueprint_max_total_actions
                        if self.blueprint_max_total_actions is not None
                        else self.num_turns
                        * self.blueprint_max_actions_per_turn
                    )
                    if actual_total_calls > configured_maximum:
                        symbolic_errors.append(
                            f"Episode has {actual_total_calls} calls; maximum "
                            f"is {configured_maximum}. Remove unnecessary work."
                        )
                    if symbolic_errors:
                        accumulated_feedback = (
                            "Symbolic plan validation failed:\n"
                            + "\n".join(symbolic_errors[:20])
                            + "\nRegenerate the complete episode plan; do not "
                            "guess hidden values or future outputs."
                        )
                        print(f"  ✗ {accumulated_feedback[:1000]}")
                        continue
                    result["turns"] = turns

                validation_errors = []
                all_tools_valid = True
                for i, t in enumerate(turns):
                    expected = t.get("expected_tools", [])
                    required_count = (
                        exact_action_schedule[i]
                        if (
                            exact_action_schedule is not None
                            and not self.symbolic_episode_plan
                        )
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

                    # Validate legacy placeholder references in user_query.
                    # V2 carries executable call-id/path bindings outside the
                    # natural utterance and therefore needs no template token.
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

                # Validate legacy cross-turn entity references.  V2 validates
                # these directly in the symbolic dependency graph above.
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
                    if self.symbolic_episode_plan:
                        break
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

                if self.symbolic_episode_plan:
                    execution_issues = (
                        self._preflight_symbolic_blueprint_execution(turns)
                    )
                    if execution_issues:
                        execution_feedback = "\n".join(execution_issues)
                        accumulated_feedback = (
                            "Deterministic execution preflight failed:\n"
                            f"{execution_feedback}\n\nRepair the symbolic binding "
                            "or state-dependent action while preserving the "
                            "same coherent user goal."
                        )
                        print(f"  ✗ {accumulated_feedback[:500]}...")
                        continue

                    aligned_task, aligned_turns, alignment_issues = (
                        self._align_symbolic_blueprint_queries(
                            overall_task=str(result.get("overall_task", "")),
                            turns=turns,
                            tools_json=tools_json,
                        )
                    )
                    if alignment_issues or aligned_turns is None:
                        alignment_feedback = "\n".join(alignment_issues)
                        accumulated_feedback = (
                            "Episode query/call alignment failed:\n"
                            f"{alignment_feedback}\n\nRegenerate a call graph whose "
                            "exact supported work can be stated naturally."
                        )
                        print(f"  ✗ {accumulated_feedback[:500]}...")
                        continue
                    turns = aligned_turns
                    result["turns"] = turns
                    result["overall_task"] = aligned_task

                # Run the only semantic judge after every deterministic gate,
                # so malformed or simulator-infeasible plans consume no judge
                # request and can use the bounded Stage-0 repair attempt.
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
            intent=str(turn_spec.get("intent", "")),
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
            if not isinstance(turn_out, dict):
                continue
            calls = turn_out.get("calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    tool_name = str(call.get("tool_name", ""))
                    output = call.get("output")
                    if tool_name and isinstance(output, dict):
                        prior_tc_by_name.setdefault(tool_name, []).append(
                            output
                        )
                continue
            # Compatibility with checkpoints written before call-preserving
            # turn aggregates were introduced.
            for tool_name, output in turn_out.items():
                if tool_name in {"calls", "by_tool"}:
                    continue
                outputs = output if isinstance(output, list) else [output]
                for item in outputs:
                    if isinstance(item, dict):
                        prior_tc_by_name.setdefault(tool_name, []).append(item)

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
