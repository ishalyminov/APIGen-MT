"""Opt-in refusal and certified-parallel generation.

This module deliberately does not modify the existing step-by-step or multi-turn
classes.  The original generators remain the default.  When explicitly enabled,
these subclasses add two terminal/action modes:

* ``refuse``: a fail-closed, policy-visible terminal pseudo-tool.
* ``parallel``: one batch of independent calls generated from the same visible
  context and certified by isolated, forward-order, and reverse-order execution.

``think`` is intentionally not implemented.
"""

from __future__ import annotations

import copy
import json
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from apigen_step_by_step import (
    ConversationTrajectory,
    QueryGenerationResult,
    StateVerificationResult,
    StepByStepDatapoint,
    StepByStepGenerator,
    ToolCallWithOutput,
    TrajectoryStep,
)
from apigen_multi_turn import (
    DialogBlueprint,
    MultiTurnConversation,
    MultiTurnDatapoint,
    MultiTurnGenerator,
)
from rl_quality_gate import MUTATING_TOOLS, validate_transition_quality
from refuse_parallel_eval import (
    prepare_multiturn_datapoint,
    prepare_step_by_step_datapoint,
)


REFUSAL_REASONS = (
    "no_appropriate_function",
    "missing_argument",
    "ambiguity",
)

REFUSE_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "refuse",
    "description": (
        "Use only when the request cannot be acted on with the available tools: "
        "the required capability is unavailable, a required value is missing and "
        "cannot be obtained, or the request is materially ambiguous."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": list(REFUSAL_REASONS),
                "description": "Why no real tool call can be selected safely.",
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
    "output_type": "dict",
    "output_description": "A terminal refusal status and reason.",
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["refused"]},
            "reason": {"type": "string", "enum": list(REFUSAL_REASONS)},
        },
        "required": ["status", "reason"],
        "additionalProperties": False,
    },
    "category": "Control",
    "synthetic_terminal_tool": True,
}


class FeatureQueryGenerationResult(QueryGenerationResult):
    """Query result carrying opt-in refusal/parallel metadata."""

    mode: str = "normal"  # normal | refusal | parallel
    action_plan: List[List[int]] = Field(default_factory=list)
    refusal_type: Optional[str] = None
    native_response: Optional[str] = None
    feature_certificate: Dict[str, Any] = Field(default_factory=dict)


class RefusalParallelConfig(BaseModel):
    allow_refusal: bool = False
    refusal_rate: float = 0.12
    allow_parallel: bool = False
    parallel_rate: float = 0.25
    max_parallel_width: int = 3
    require_feature: bool = False
    feature_difficulty: str = "standard"
    naturalize_queries: bool = False
    multi_turn_feature_schedule: str = "terminal"
    forced_refusal_reason: Optional[str] = None
    interactive_refusal_turn: Optional[int] = None


class _RefusalParallelSupport:
    """Shared implementation used by both opt-in generator subclasses."""

    feature_config: RefusalParallelConfig

    def _configure_refusal_parallel(
        self,
        *,
        allow_refusal: bool,
        refusal_rate: float,
        allow_parallel: bool,
        parallel_rate: float,
        max_parallel_width: int,
        require_feature: bool = False,
        feature_difficulty: str = "standard",
        naturalize_queries: bool = False,
        multi_turn_feature_schedule: str = "terminal",
        forced_refusal_reason: Optional[str] = None,
        interactive_refusal_turn: Optional[int] = None,
    ) -> None:
        difficulty = str(feature_difficulty).strip().casefold()
        if difficulty not in {"standard", "hard"}:
            raise ValueError(
                "feature_difficulty must be 'standard' or 'hard'"
            )
        schedule = str(multi_turn_feature_schedule).strip().casefold()
        if schedule not in {"terminal", "interactive-refusal", "combined"}:
            raise ValueError(
                "multi_turn_feature_schedule must be terminal, "
                "interactive-refusal, or combined"
            )
        reason = (
            str(forced_refusal_reason).strip()
            if forced_refusal_reason is not None
            else None
        )
        if reason == "random":
            reason = None
        if reason is not None and reason not in REFUSAL_REASONS:
            raise ValueError(
                "forced_refusal_reason must be random or one of "
                + ", ".join(REFUSAL_REASONS)
            )
        if schedule in {"interactive-refusal", "combined"} and reason == (
            "no_appropriate_function"
        ):
            raise ValueError(
                "interactive clarification cannot recover from "
                "no_appropriate_function"
            )
        self.feature_config = RefusalParallelConfig(
            allow_refusal=allow_refusal,
            refusal_rate=max(0.0, min(1.0, refusal_rate)),
            allow_parallel=allow_parallel,
            parallel_rate=max(0.0, min(1.0, parallel_rate)),
            max_parallel_width=max(2, max_parallel_width),
            require_feature=bool(require_feature),
            feature_difficulty=difficulty,
            naturalize_queries=bool(naturalize_queries),
            multi_turn_feature_schedule=schedule,
            forced_refusal_reason=reason,
            interactive_refusal_turn=interactive_refusal_turn,
        )

    def _required_feature_mode(self, *, parallel_eligible: bool) -> Optional[str]:
        """Choose one enabled feature when feature-only output is requested."""

        if not self.feature_config.require_feature:
            return None
        modes: List[str] = []
        weights: List[float] = []
        if self.feature_config.allow_refusal:
            modes.append("refusal")
            weights.append(max(self.feature_config.refusal_rate, 1e-6))
        if self.feature_config.allow_parallel and parallel_eligible:
            modes.append("parallel")
            weights.append(max(self.feature_config.parallel_rate, 1e-6))
        if not modes:
            return "unavailable"
        return random.choices(modes, weights=weights, k=1)[0]

    def _sample_refusal_reason(self) -> str:
        if self.feature_config.forced_refusal_reason:
            return self.feature_config.forced_refusal_reason
        if self.feature_config.feature_difficulty == "hard":
            # Near-miss clarifications force the policy to inspect schemas and
            # visible history; obvious unsupported-capability refusals are rare.
            return random.choices(
                list(REFUSAL_REASONS),
                weights=[0.10, 0.45, 0.45],
                k=1,
            )[0]
        return random.choice(REFUSAL_REASONS)

    @staticmethod
    def _protected_query_tokens(query: str) -> List[str]:
        """Values a style-only rewrite must preserve byte-for-byte."""

        pattern = re.compile(
            r"\{\{[^{}]+\}\}"
            r"|(?<![\w.])-?\d+(?:\.\d+)?(?:%|[A-Za-z]+)?"
            r"|['\"][^'\"]+['\"]"
            r"|(?:[\w.-]+/)+[\w.-]+"
            r"|[\w.-]+\.(?:txt|json|jsonl|csv|yaml|yml|py|log|md|zip)",
            re.IGNORECASE,
        )
        return sorted(set(pattern.findall(query)))

    def _naturalize_feature_query(
        self,
        *,
        query: str,
        mode: str,
        expected_tools: List[str],
        policy_history: List[Dict[str, Any]],
        refusal_type: Optional[str] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self.feature_config.naturalize_queries:
            return query, {"enabled": False, "rewritten": False}

        protected = self._protected_query_tokens(query)
        prompt = f"""Rewrite one synthetic tool-use request into a more natural, dense user
utterance. This is a STYLE-ONLY second pass after the tool plan was fixed.

=== SOURCE REQUEST ===
{query}

=== MODE ===
{mode}

=== FIXED EXPECTED TOOLS ===
{json.dumps(expected_tools)}

=== REFUSAL REASON, IF ANY ===
{refusal_type or "None"}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history, ensure_ascii=False, indent=2, default=str)}

=== TOKENS TO PRESERVE VERBATIM ===
{json.dumps(protected, ensure_ascii=False)}

Rules:
1. Preserve the exact task, entities, identifiers, numbers, dates, filenames,
   quoted content, placeholders, and all argument-determining facts.
2. Do not add a new fact, value, operation, prerequisite, or requested result.
3. Make the wording conversational and cohesive: use context-sensitive
   references, realistic motivation, subordinate clauses, and varied syntax.
4. Do not mention tool/function names or describe a tool-call plan.
5. Refusal: preserve exactly the same single blocker. Do not fill the missing
   field, resolve the ambiguity, or announce why the request is impossible.
6. Parallel: preserve all independent requested results without introducing an
   ordering/dependency and without words such as parallel, simultaneously,
   independently, in any order, or at the same time.

Few-shot examples:

SOURCE: "Get the distance for 02108 and 19103 and display the fuel status."
NATURAL: "Before I decide whether to make the trip, how far apart are 02108 and
19103, and what does the car currently report for its fuel status?"

SOURCE: "Round 8.52803 to 2 decimals and find the minimum of [8.1, 8.7, 8.4]."
NATURAL: "For the summary table, I need 8.52803 rounded to 2 decimals; alongside
that, tell me the smallest reading in [8.1, 8.7, 8.4]."

SOURCE: "Resolve ticket 654321." (the required resolution text is absent)
NATURAL: "With the other updates on ticket 654321 already settled, go ahead and
close out that ticket for me."

Respond only with JSON:
{{"query": "..."}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate(
                    [{"role": "user", "content": prompt}]
                )
            )
        except Exception as exc:
            print(f"  Feature query naturalization failed: {exc}")
            return None

        rewritten = str(result.get("query", "")).strip()
        if not rewritten:
            return None
        missing = [token for token in protected if token not in rewritten]
        if missing:
            print(
                "  Feature naturalization dropped protected tokens: "
                + ", ".join(missing[:5])
            )
            return None
        return rewritten, {
            "enabled": True,
            "rewritten": True,
            "text_changed": rewritten != query,
            "source_query": query,
            "protected_tokens": protected,
            "protected_tokens_preserved": True,
        }

    # ------------------------------------------------------------------
    # Policy tool contracts
    # ------------------------------------------------------------------

    def _get_policy_tool_schemas(
        self,
        focus_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Feature-enabled episodes expose the exact real contracts plus the
        # stable refusal action.  Legacy pseudo-tools from the feature branch
        # are deliberately filtered out; ``think`` is not part of this patch.
        tools = self._real_policy_tool_schemas(focus_category)
        if (
            self.feature_config.allow_refusal
            and not any(tool.get("name") == "refuse" for tool in tools)
        ):
            tools.append(copy.deepcopy(REFUSE_TOOL_SCHEMA))
        return tools

    def _real_policy_tool_schemas(
        self,
        focus_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            tool
            for tool in super()._get_policy_tool_schemas(focus_category)
            if tool.get("name") not in {"think", "refuse"}
        ]

    # ------------------------------------------------------------------
    # JSON and schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        else:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected one JSON object")
        return parsed

    @staticmethod
    def _normalise_schema_type(schema_type: Any) -> Optional[str]:
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            return str(non_null[0]).lower() if non_null else None
        if schema_type is None:
            return None
        value = str(schema_type).lower()
        aliases = {
            "dict": "object",
            "float": "number",
            "double": "number",
            "int": "integer",
            "bool": "boolean",
            "list": "array",
        }
        return aliases.get(value, value)

    @classmethod
    def _validate_json_value(
        cls,
        value: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
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
            type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected == "boolean":
            type_ok = isinstance(value, bool)
        elif expected == "null":
            type_ok = value is None

        if not type_ok:
            return [f"{path}: expected {expected}, got {type(value).__name__}"]

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
            if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
                issues.append(f"{path}: below exclusive minimum")
            if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
                issues.append(f"{path}: above exclusive maximum")
            if "multipleOf" in schema:
                divisor = schema["multipleOf"]
                if divisor and abs((value / divisor) - round(value / divisor)) > 1e-9:
                    issues.append(f"{path}: not a multiple of required value")

        if isinstance(value, list):
            if "minItems" in schema and len(value) < int(schema["minItems"]):
                issues.append(f"{path}: too few items")
            if "maxItems" in schema and len(value) > int(schema["maxItems"]):
                issues.append(f"{path}: too many items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    issues.extend(
                        cls._validate_json_value(item, item_schema, f"{path}[{index}]")
                    )

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    issues.append(f"{path}.{key}: missing required argument")
            additional_allowed = schema.get("additionalProperties", True)
            if additional_allowed is False:
                for key in value:
                    if key not in properties:
                        issues.append(f"{path}.{key}: unexpected argument")
            for key, item in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    issues.extend(
                        cls._validate_json_value(item, child_schema, f"{path}.{key}")
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
    def _contains_unresolved_marker(value: Any) -> bool:
        if isinstance(value, dict):
            if "__missing_required_argument__" in value:
                return True
            return any(
                _RefusalParallelSupport._contains_unresolved_marker(item)
                for item in value.values()
            )
        if isinstance(value, list):
            return any(
                _RefusalParallelSupport._contains_unresolved_marker(item)
                for item in value
            )
        if isinstance(value, str):
            return "__MISSING_ARGUMENT__" in value or "{{" in value or "}}" in value
        return False

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    # ------------------------------------------------------------------
    # Refusal generation and certification
    # ------------------------------------------------------------------

    def _generate_refusal_query(
        self,
        *,
        focus_category: Optional[str],
        refusal_type: str,
        policy_history: Optional[List[Dict[str, Any]]] = None,
        original_query: Optional[str] = None,
        source_expected_tools: Optional[List[str]] = None,
    ) -> Optional[FeatureQueryGenerationResult]:
        real_tools = self._real_policy_tool_schemas(focus_category)
        if not real_tools:
            return None
        hard_mode = self.feature_config.feature_difficulty == "hard"
        if hard_mode and not policy_history:
            return None
        hard_rules = """
=== HARD-NEGATIVE RULES ===
This must be a difficult near-miss, not an obvious refusal:
- The request must be a coherent continuation that uses at least two concrete
  entities, values, or results from PRIOR POLICY-VISIBLE HISTORY.
- Exactly one blocker determines the terminal action. Everything else needed
  for the most plausible real tool plan must already be visible.
- For missing_argument, omit a non-obvious required field while supplying or
  grounding the other required fields.
- For ambiguity, make two materially different schema-valid choices remain
  after considering the full history; avoid explicit phrases such as "either".
- For no_appropriate_function, stay immediately adjacent to a real tool's
  supported capability instead of requesting an obviously unrelated action.
- The surface wording must not announce that information is missing, ambiguous,
  unsupported, unavailable, or impossible.
""" if hard_mode else ""

        prompt = f"""Generate one realistic tool-use request whose correct terminal action is
`refuse` with reason `{refusal_type}`, plus the exact benchmark-native assistant
response that should be produced with NO real tool call.

=== REAL AVAILABLE TOOLS ===
{json.dumps(real_tools, indent=2, ensure_ascii=False, default=str)}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history or [], indent=2, ensure_ascii=False, default=str)}

=== OPTIONAL ORIGINAL TURN TO REWRITE ===
{original_query or "None"}

=== FIXED UNDERLYING PLAN WHEN THE BLOCKER IS RESOLVED ===
{json.dumps(source_expected_tools or [])}

=== DEFINITIONS ===
- no_appropriate_function: the requested external action cannot be performed by
  any real available tool. Do not use this for a request that can simply be
  answered without a tool.
- missing_argument: one or more values required by the only appropriate
  real tool are absent from all visible messages, have no schema default, and
  cannot be obtained with another available tool. Do not invent the value.
- ambiguity: at least two materially different valid tool actions or
  argument values fit the request and the visible conversation provides no safe
  selection rule.

=== RESPONSE RULES ===
- no_appropriate_function: briefly explain that the requested capability is not
  available. Do not name an invented tool.
- missing_argument: ask one direct, specific question for the actually missing
  field or fields. Do not issue a generic refusal.
- ambiguity: ask one direct question that distinguishes the material alternatives.
- Never invent or reveal a missing value, opaque identifier, credential, token,
  hidden state value, or internal judge result.

=== GENERATION RULES ===
1. The request must be natural and realistic.
2. Do not mention tool names or the word "refuse" in the user request.
3. Do not include the missing value when the reason is missing_argument.
4. Do not create fake IDs, credentials, codes, or hidden context.
5. The request must genuinely require clarification or unavailable capability;
   it must not be satisfiable by a real tool call from the visible context.
6. When OPTIONAL ORIGINAL TURN is supplied, rewrite it minimally: preserve its
   broad domain, named entities, and intended user goal. Introduce only the
   unavailable capability, missing field, or material ambiguity needed for the
   claimed reason. Do not switch to an unrelated topic, person, city, file, or
   account. The new turn must remain a plausible continuation of prior history.
7. When FIXED UNDERLYING PLAN is non-empty, the blocked request must still ask
   for exactly that operation. If the omitted/ambiguous fact were supplied, the
   same listed tool sequence—not a different available tool—would fulfill it.

{hard_rules}

Respond only with JSON:
{{
  "query": "...",
  "intent": "...",
  "assistant_response": "..."
}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate([{"role": "user", "content": prompt}])
            )
        except Exception as exc:
            print(f"  Refusal query generation failed: {exc}")
            return None

        query = str(result.get("query", "")).strip()
        intent = str(result.get("intent", "")).strip()
        native_response = str(result.get("assistant_response", "")).strip()
        if not query or not native_response:
            return None
        if refusal_type in {"missing_argument", "ambiguity"}:
            if "?" not in native_response:
                return None

        naturalized = self._naturalize_feature_query(
            query=query,
            mode="refusal",
            expected_tools=["refuse"],
            policy_history=policy_history or [],
            refusal_type=refusal_type,
        )
        if naturalized is None:
            return None
        query, naturalization = naturalized

        certificate = self._certify_refusal(
            query=query,
            refusal_type=refusal_type,
            real_tools=real_tools,
            policy_history=policy_history or [],
            proposed_response=native_response,
            original_query=original_query or "",
            source_expected_tools=source_expected_tools or [],
        )
        if not certificate.get("passed", False):
            print(
                "  Refusal certification failed: "
                + ", ".join(certificate.get("issue_codes", []))
            )
            return None
        certificate["query_naturalization"] = naturalization

        quality = {
            "passed": True,
            "mode": "refusal",
            "refusal_type": refusal_type,
            "issue_codes": [],
            "native_response": native_response,
            "refusal_certificate": certificate,
        }
        self._last_query_quality = quality
        return FeatureQueryGenerationResult(
            query=query,
            intent=intent,
            expected_tools=["refuse"],
            quality_preflight=quality,
            mode="refusal",
            action_plan=[[0]],
            refusal_type=refusal_type,
            native_response=native_response,
            feature_certificate=certificate,
        )

    def _certify_refusal(
        self,
        *,
        query: str,
        refusal_type: str,
        real_tools: List[Dict[str, Any]],
        policy_history: List[Dict[str, Any]],
        proposed_response: str = "",
        original_query: str = "",
        source_expected_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        proposed_response = proposed_response.strip() or self._refusal_response(
            refusal_type
        )
        allowed_codes = {
            "REAL_TOOL_CAN_FULFILL",
            "WRONG_REFUSAL_REASON",
            "MISSING_VALUE_IS_OBTAINABLE",
            "REQUEST_IS_NOT_MATERIALLY_AMBIGUOUS",
            "REQUEST_DOES_NOT_REQUIRE_A_TOOL",
            "RESPONSE_NOT_REASON_SPECIFIC",
            "RESPONSE_INVENTS_VALUE",
            "RESPONSE_NOT_GROUNDED",
            "RESPONSE_NOT_A_QUESTION",
            "CONVERSATION_DISCONTINUITY",
            "HARD_NEGATIVE_TOO_EASY",
            "HARD_NEGATIVE_NOT_HISTORY_DEPENDENT",
            "HARD_NEGATIVE_MULTIPLE_BLOCKERS",
            "REFUSAL_CERTIFIER_UNAVAILABLE",
            "OTHER_INVALID",
        }
        prompt = f"""Certify whether a synthetic assistant must decline to select a real tool
and whether its benchmark-native no-tool response is correct.

=== USER QUERY ===
{query}

=== OPTIONAL ORIGINAL TURN BEING REWRITTEN ===
{original_query or "None"}

=== FIXED UNDERLYING PLAN WHEN THE BLOCKER IS RESOLVED ===
{json.dumps(source_expected_tools or [])}

=== CLAIMED REASON ===
{refusal_type}

=== PROPOSED NO-TOOL ASSISTANT RESPONSE ===
{proposed_response}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history, indent=2, ensure_ascii=False, default=str)}

=== REAL AVAILABLE TOOLS ===
{json.dumps(real_tools, indent=2, ensure_ascii=False, default=str)}

The assistant sees only the query, prior visible history, and these tool schemas.
It does not see simulator state or internal judge messages.

Return valid only when both the decision and response are correct:
- no_appropriate_function: no real tool or valid sequence of real tools can
  perform the requested external action, and the response clearly explains that
  the capability is unavailable without inventing a tool;
- missing_argument: an appropriate tool exists, but at least one truly
  required value is unavailable, has no default, and cannot be retrieved by any
  available tool; the response asks specifically for that missing information;
- ambiguity: at least two materially different valid actions/values fit and
  choosing either would be unsupported; the response asks a targeted question
  that resolves the material ambiguity.

The response must not invent or reveal a missing value, identifier, credential,
token, hidden-state fact, or internal tool/judge detail. For missing_argument and
ambiguity it must be a real clarification question, not a generic refusal.

When an original turn is supplied, the generated request must remain in the same
broad domain, preserve its primary named entities and intended goal, and be a
plausible continuation of prior history. Reject an unrelated topic switch with
CONVERSATION_DISCONTINUITY.

When a fixed underlying plan is supplied, mentally resolve the claimed blocker
using the fully specified original turn. The resulting request must still map to
exactly that listed tool sequence and operation. A switch such as watchlist →
order placement, lookup → mutation, draft → send, or booking → cancellation is
CONVERSATION_DISCONTINUITY even when it stays in the same broad domain.

When feature difficulty is HARD, also reject unless the request is a subtle
near-miss that depends on at least two concrete facts/results in prior visible
history, has exactly one blocking issue, and does not announce that issue in
obvious words. Use HARD_NEGATIVE_TOO_EASY,
HARD_NEGATIVE_NOT_HISTORY_DEPENDENT, or HARD_NEGATIVE_MULTIPLE_BLOCKERS.

=== FEATURE DIFFICULTY ===
{self.feature_config.feature_difficulty}

Do not reveal or suggest replacement values. Return only allowed issue codes.
Allowed issue codes: {json.dumps(sorted(allowed_codes))}

Respond only with JSON:
{{
  "is_valid": true,
  "issue_codes": [],
  "underlying_plan_preserved": true,
  "counterexample_check": {{
    "real_tool_plan_exists": false,
    "missing_value_is_obtainable": false,
    "unique_safe_action_exists": false
  }},
  "hardness": {{
    "uses_prior_visible_context": true,
    "subtle_near_miss": true,
    "single_blocker": true
  }}
}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate(
                    [{"role": "user", "content": prompt}],
                    llm=self.judge,
                )
            )
            raw_valid = result.get("is_valid")
            raw_codes = result.get("issue_codes")
            if not isinstance(raw_valid, bool) or not isinstance(raw_codes, list):
                return {
                    "passed": False,
                    "reason": refusal_type,
                    "issue_codes": ["REFUSAL_CERTIFIER_UNAVAILABLE"],
                }
            codes = [str(code) for code in raw_codes if str(code) in allowed_codes]
            plan_preserved = result.get("underlying_plan_preserved")
            if source_expected_tools and plan_preserved is not True:
                codes.append("CONVERSATION_DISCONTINUITY")
            hardness = result.get("hardness", {})
            if self.feature_config.feature_difficulty == "hard":
                if not isinstance(hardness, dict):
                    hardness = {}
                if hardness.get("uses_prior_visible_context") is not True:
                    codes.append("HARD_NEGATIVE_NOT_HISTORY_DEPENDENT")
                if hardness.get("subtle_near_miss") is not True:
                    codes.append("HARD_NEGATIVE_TOO_EASY")
                if hardness.get("single_blocker") is not True:
                    codes.append("HARD_NEGATIVE_MULTIPLE_BLOCKERS")
                codes = sorted(set(codes))
            else:
                codes = sorted(set(codes))
            passed = raw_valid and not codes
            if not passed and not codes:
                codes = ["OTHER_INVALID"]
            if not passed:
                return {
                    "passed": False,
                    "reason": refusal_type,
                    "assistant_response": proposed_response,
                    "issue_codes": codes,
                    "primary_judge": {"passed": False, "issue_codes": codes},
                    "hardness": hardness,
                }

            # The same episode-level semantic decision includes an adversarial
            # counterexample section.  The previous implementation paid for a
            # second almost-identical full-schema judge request.
            counterexample_raw = result.get("counterexample_check", {})
            counterexample_fields = (
                "real_tool_plan_exists",
                "missing_value_is_obtainable",
                "unique_safe_action_exists",
            )
            if (
                not isinstance(counterexample_raw, dict)
                or any(
                    not isinstance(counterexample_raw.get(key), bool)
                    for key in counterexample_fields
                )
            ):
                return {
                    "passed": False,
                    "reason": refusal_type,
                    "assistant_response": proposed_response,
                    "issue_codes": ["REFUSAL_CERTIFIER_UNAVAILABLE"],
                    "primary_judge": {"passed": True, "issue_codes": []},
                }
            counterexample_codes: List[str] = []
            if counterexample_raw["real_tool_plan_exists"]:
                counterexample_codes.append("REAL_TOOL_CAN_FULFILL")
            if (
                refusal_type == "missing_argument"
                and counterexample_raw["missing_value_is_obtainable"]
            ):
                counterexample_codes.append("MISSING_VALUE_IS_OBTAINABLE")
            if (
                refusal_type == "ambiguity"
                and counterexample_raw["unique_safe_action_exists"]
            ):
                counterexample_codes.append(
                    "REQUEST_IS_NOT_MATERIALLY_AMBIGUOUS"
                )
            counterexample = {
                "passed": not counterexample_codes,
                "issue_codes": counterexample_codes,
                **counterexample_raw,
            }
            if counterexample_codes:
                return {
                    "passed": False,
                    "reason": refusal_type,
                    "assistant_response": proposed_response,
                    "issue_codes": counterexample_codes,
                    "primary_judge": {"passed": True, "issue_codes": []},
                    "counterexample_search": counterexample,
                }

            return {
                "passed": True,
                "reason": refusal_type,
                "assistant_response": proposed_response,
                "issue_codes": [],
                "primary_judge": {"passed": True, "issue_codes": []},
                "counterexample_search": counterexample,
                "difficulty": self.feature_config.feature_difficulty,
                "hardness": hardness,
                "underlying_plan_preserved": plan_preserved,
            }
        except Exception as exc:
            print(f"  Refusal certifier failed closed: {exc}")
            return {
                "passed": False,
                "reason": refusal_type,
                "assistant_response": proposed_response,
                "issue_codes": ["REFUSAL_CERTIFIER_UNAVAILABLE"],
            }

    def _search_refusal_counterexample(
        self,
        *,
        query: str,
        refusal_type: str,
        real_tools: List[Dict[str, Any]],
        policy_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Adversarially try to disprove the proposed refusal.

        Refusal has no executable positive trace, so a single permissive judge is
        not enough.  This second, differently phrased pass tries to construct a
        real-tool counterexample.  Only booleans/issue codes are retained; no
        proposed argument values are fed back into generation.
        """
        prompt = f"""Act as an adversarial solver. Try to disprove a proposed terminal
`refuse` action using only policy-visible information.

=== USER QUERY ===
{query}

=== CLAIMED REASON ===
{refusal_type}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history, indent=2, ensure_ascii=False, default=str)}

=== REAL AVAILABLE TOOLS ===
{json.dumps(real_tools, indent=2, ensure_ascii=False, default=str)}

Determine whether any executable sequence of REAL tools can complete the
requested external action without guessing an unavailable value or choosing an
unsupported interpretation. Also determine whether a supposedly missing value
can be obtained by a real tool, and whether an ambiguity actually has one unique
safe action.

Do not output a plan, tool arguments, example identifiers, or replacement
values. Respond only with JSON booleans:
{{
  "real_tool_plan_exists": false,
  "missing_value_is_obtainable": false,
  "unique_safe_action_exists": false
}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate(
                    [{"role": "user", "content": prompt}],
                    llm=self.judge,
                )
            )
            required = (
                "real_tool_plan_exists",
                "missing_value_is_obtainable",
                "unique_safe_action_exists",
            )
            if any(not isinstance(result.get(key), bool) for key in required):
                return {
                    "passed": False,
                    "issue_codes": ["REFUSAL_CERTIFIER_UNAVAILABLE"],
                }
            plan_exists = result["real_tool_plan_exists"]
            obtainable = result["missing_value_is_obtainable"]
            unique_safe = result["unique_safe_action_exists"]

            codes: List[str] = []
            if plan_exists:
                codes.append("REAL_TOOL_CAN_FULFILL")
            if refusal_type == "missing_argument" and obtainable:
                codes.append("MISSING_VALUE_IS_OBTAINABLE")
            if refusal_type == "ambiguity" and unique_safe:
                codes.append("REQUEST_IS_NOT_MATERIALLY_AMBIGUOUS")
            return {
                "passed": not codes,
                "issue_codes": codes,
                "real_tool_plan_exists": plan_exists,
                "missing_value_is_obtainable": obtainable,
                "unique_safe_action_exists": unique_safe,
            }
        except Exception as exc:
            print(f"  Refusal counterexample search failed closed: {exc}")
            return {
                "passed": False,
                "issue_codes": ["REFUSAL_CERTIFIER_UNAVAILABLE"],
            }

    @staticmethod
    def _refusal_response(refusal_type: str) -> str:
        if refusal_type == "missing_argument":
            return "I need the missing required information before I can safely complete that request."
        if refusal_type == "ambiguity":
            return "I need clarification because the request allows more than one materially different action."
        return "I can’t complete that request with the available tools."

    # ------------------------------------------------------------------
    # Parallel query generation and certification
    # ------------------------------------------------------------------

    def _generate_parallel_query(
        self,
        *,
        focus_category: Optional[str],
        num_calls: int,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]],
        policy_history: Optional[List[Dict[str, Any]]] = None,
        original_query: Optional[str] = None,
        query_seed: Optional[dict] = None,
        max_retries: int = 3,
    ) -> Optional[FeatureQueryGenerationResult]:
        if num_calls < 2:
            return None
        if num_calls > self.feature_config.max_parallel_width:
            return None
        real_tools = self._real_policy_tool_schemas(focus_category)
        if not real_tools:
            return None
        hard_mode = self.feature_config.feature_difficulty == "hard"
        if hard_mode and not policy_history:
            return None

        # The state is generator-only and may reject an unsuitable task, but it
        # is never supplied to argument generation.
        state_for_prompt = initial_api_state or {}
        style_seed = ""
        if query_seed:
            persona = query_seed.get("persona", {})
            style_seed = (
                "Optional phrasing seed only (never an argument source): "
                f"{persona.get('name', '')}."
            )

        feedback = ""
        for attempt in range(max_retries):
            prompt = f"""Generate a realistic user request requiring EXACTLY {num_calls}
real tool calls that can be issued in ONE parallel batch.

=== AVAILABLE REAL TOOLS ===
{json.dumps(real_tools, indent=2, ensure_ascii=False, default=str)}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history or [], indent=2, ensure_ascii=False, default=str)}

=== GENERATOR-ONLY STATE ===
The policy will not see this state. Use it only to avoid impossible/no-op tasks.
Never rely on it as an argument source.
{json.dumps(state_for_prompt, indent=2, ensure_ascii=False, default=str)}

=== OPTIONAL ORIGINAL TURN TO REWRITE ===
{original_query or "None"}

{style_seed}

=== STRICT PARALLEL RULES ===
1. All {num_calls} calls are independent and read-only. No call may change API
   state, authenticate, create, update, delete, navigate a working directory, or
   consume another call's output.
2. Every required argument for every call is already visible in the user query,
   prior policy-visible history, or a declared schema default.
3. General knowledge is not a source for opaque IDs, codes, tokens, symbols,
   paths, coordinates, or credentials.
4. Calling the tools in any order must produce the same per-call outputs and the
   same final state.
5. The user asks for all results together and does not impose a sequential order.
6. Duplicate calls to the same tool are allowed when their arguments differ.
7. The request is unambiguous and every requested final claim is directly
   reportable from tool outputs.
8. Do not choose any state-mutating or authentication-dependent operation.
9. When OPTIONAL ORIGINAL TURN is supplied, rewrite it minimally into independent
   read-only requests: preserve its broad domain, primary named entities, and
   intended information goal. Do not switch to unrelated people, cities, files,
   accounts, or topics.
10. The current UTC date is {datetime.now(timezone.utc).date().isoformat()}.

=== HARD PARALLEL RULES ===
{(
    f'''- Use at least two distinct tool names across the {num_calls} calls.
- Ground at least two calls in different concrete results/entities from PRIOR
  POLICY-VISIBLE HISTORY, while keeping every sibling independent.
- Require nontrivial, schema-specific arguments; avoid a batch of generic status,
  listing, display, or duplicate same-tool lookups.
- The wording must not explicitly say "parallel", "simultaneously",
  "independently", "in any order", or "at the same time".
- The request should look like one coherent analytical follow-up whose
  independent subresults must all be returned.'''
    if hard_mode
    else "Standard difficulty."
)}

Respond only with JSON:
{{
  "query": "...",
  "intent": "...",
  "expected_tools": ["tool_name", "tool_name"]
}}
{('Generic retry: ' + feedback) if feedback else ''}
"""
            try:
                result = self._extract_json_object(
                    self._safe_llm_generate([{"role": "user", "content": prompt}])
                )
            except Exception as exc:
                feedback = "Return valid JSON with the exact required fields."
                print(f"  Parallel query attempt {attempt + 1} failed: {exc}")
                continue

            query = str(result.get("query", "")).strip()
            intent = str(result.get("intent", "")).strip()
            expected_tools = result.get("expected_tools", [])
            if not isinstance(expected_tools, list):
                expected_tools = []
            expected_tools = [str(name) for name in expected_tools]

            if not query or len(expected_tools) != num_calls:
                feedback = "Use the exact requested number of calls."
                continue
            allowed_tool_names = {tool.get("name") for tool in real_tools}
            if any(name not in allowed_tool_names for name in expected_tools):
                feedback = "Use only real tool names from the supplied schemas."
                continue
            if any(name in MUTATING_TOOLS for name in expected_tools):
                feedback = "Choose read-only tools only."
                continue
            if hard_mode and len(set(expected_tools)) < 2:
                feedback = "Hard mode requires at least two distinct tools."
                continue
            if any(
                not self.tool_manager.has_python_implementation(name)
                for name in expected_tools
            ):
                feedback = "Choose tools with executable implementations."
                continue

            naturalized = self._naturalize_feature_query(
                query=query,
                mode="parallel",
                expected_tools=expected_tools,
                policy_history=policy_history or [],
            )
            if naturalized is None:
                feedback = "Rewrite the task naturally without changing its plan."
                continue
            query, naturalization = naturalized

            certificate = self._certify_parallel_query(
                query=query,
                expected_tools=expected_tools,
                real_tools=[
                    self.tool_manager.get_tool_schema(name)
                    for name in expected_tools
                ],
                initial_api_state=initial_api_state or {},
                policy_history=policy_history or [],
                original_query=original_query or "",
            )
            if not certificate.get("passed", False):
                feedback = "Generate a different independent read-only task."
                continue
            certificate["query_naturalization"] = naturalization

            quality = {
                "passed": True,
                "mode": "parallel",
                "issue_codes": [],
                "parallel_certificate": certificate,
            }
            self._last_query_quality = quality
            return FeatureQueryGenerationResult(
                query=query,
                intent=intent,
                expected_tools=expected_tools,
                quality_preflight=quality,
                mode="parallel",
                action_plan=[list(range(num_calls))],
                feature_certificate=certificate,
            )

        return None

    def _certify_parallel_query(
        self,
        *,
        query: str,
        expected_tools: List[str],
        real_tools: List[Dict[str, Any]],
        initial_api_state: Dict[str, Any],
        policy_history: List[Dict[str, Any]],
        original_query: str = "",
    ) -> Dict[str, Any]:
        allowed_codes = {
            "PARALLEL_DEPENDENCY",
            "PARALLEL_STATE_CONFLICT",
            "PARALLEL_MUTATION",
            "PARALLEL_REQUEST_ORDERED",
            "POLICY_CONTEXT_NOT_CLOSED",
            "MISSING_PREREQUISITE",
            "AMBIGUOUS_GOLD_ACTION",
            "REQUESTED_RESULT_NOT_TOOL_GROUNDED",
            "CONVERSATION_DISCONTINUITY",
            "HARD_PARALLEL_TOO_EASY",
            "HARD_PARALLEL_NOT_HISTORY_DEPENDENT",
            "HARD_PARALLEL_NOT_HETEROGENEOUS",
            "PARALLEL_CERTIFIER_UNAVAILABLE",
            "OTHER_INVALID",
        }
        prompt = f"""Certify one synthetic parallel tool-call batch for positive RL.

=== USER QUERY ===
{query}

=== OPTIONAL ORIGINAL TURN BEING REWRITTEN ===
{original_query or "None"}

=== PLANNED CALLS (all in one batch) ===
{json.dumps(expected_tools)}

=== FULL TOOL DEFINITIONS ===
{json.dumps(real_tools, indent=2, ensure_ascii=False, default=str)}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(policy_history, indent=2, ensure_ascii=False, default=str)}

=== GENERATOR-ONLY STATE ===
{json.dumps(initial_api_state, indent=2, ensure_ascii=False, default=str)}

The state is validation-only. Never reveal or suggest a state value.
A valid parallel batch must satisfy all of these:
1. Every call is read-only and has no authentication/setup prerequisite missing.
2. No call consumes an output or state mutation of a sibling call.
3. All arguments are available before the batch from visible messages/history or
   schema defaults; model knowledge cannot supply opaque values.
4. The calls can execute in any order with identical results and state.
5. The query does not require a sequence among calls.
6. The exact call multiset and argument intents are unambiguous.
7. All requested results are directly grounded in the planned outputs.
8. When an original turn is supplied, the generated request remains in the same
   broad domain, preserves the primary named entities and intended information
   goal, and is a plausible continuation. Reject an unrelated rewrite with
   CONVERSATION_DISCONTINUITY.

For HARD difficulty, also require a subtle coherent follow-up grounded in at
least two different concrete prior visible results/entities, at least two
distinct tool names, and nontrivial schema-specific arguments. The user wording
must not explicitly announce parallelism or order-invariance.

=== FEATURE DIFFICULTY ===
{self.feature_config.feature_difficulty}

Return only allowed issue codes; do not suggest replacement values.
Allowed: {json.dumps(sorted(allowed_codes))}

Respond only with JSON:
{{
  "is_valid": true,
  "issue_codes": [],
  "hardness": {{
    "uses_multiple_prior_results": true,
    "heterogeneous_tools": true,
    "nontrivial_arguments": true,
    "parallelism_not_announced": true
  }}
}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate(
                    [{"role": "user", "content": prompt}],
                    llm=self.judge,
                )
            )
            raw_valid = result.get("is_valid")
            raw_codes = result.get("issue_codes")
            if not isinstance(raw_valid, bool) or not isinstance(raw_codes, list):
                return {
                    "passed": False,
                    "issue_codes": ["PARALLEL_CERTIFIER_UNAVAILABLE"],
                }
            codes = [str(code) for code in raw_codes if str(code) in allowed_codes]
            hardness = result.get("hardness", {})
            if self.feature_config.feature_difficulty == "hard":
                if not isinstance(hardness, dict):
                    hardness = {}
                if hardness.get("uses_multiple_prior_results") is not True:
                    codes.append("HARD_PARALLEL_NOT_HISTORY_DEPENDENT")
                if (
                    hardness.get("heterogeneous_tools") is not True
                    or len(set(expected_tools)) < 2
                ):
                    codes.append("HARD_PARALLEL_NOT_HETEROGENEOUS")
                if (
                    hardness.get("nontrivial_arguments") is not True
                    or hardness.get("parallelism_not_announced") is not True
                ):
                    codes.append("HARD_PARALLEL_TOO_EASY")
                codes = sorted(set(codes))
            passed = raw_valid and not codes
            if not passed and not codes:
                codes = ["OTHER_INVALID"]
            return {
                "passed": passed,
                "issue_codes": codes,
                "difficulty": self.feature_config.feature_difficulty,
                "hardness": hardness,
            }
        except Exception as exc:
            print(f"  Parallel certifier failed closed: {exc}")
            return {
                "passed": False,
                "issue_codes": ["PARALLEL_CERTIFIER_UNAVAILABLE"],
            }

    # ------------------------------------------------------------------
    # Parallel arguments and execution
    # ------------------------------------------------------------------

    def _policy_visible_history(
        self,
        trajectory: Sequence[TrajectoryStep],
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for step in trajectory:
            for call_index, call in enumerate(step.tool_calls, 1):
                history.append(
                    {
                        "call_id": f"s{step.step_number}_c{call_index}",
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                        "output": call.output,
                    }
                )
        return history

    def _generate_parallel_arguments(
        self,
        *,
        query: str,
        tool_names: List[str],
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
        max_retries: int,
    ) -> Optional[List[Dict[str, Any]]]:
        self._last_parallel_argument_certificate = {}
        call_specs = [
            {
                "call_id": f"p{index + 1}",
                "tool_name": tool_name,
                "tool_definition": self.tool_manager.get_tool_schema(tool_name),
            }
            for index, tool_name in enumerate(tool_names)
        ]
        visible_history = self._policy_visible_history(trajectory)

        for attempt in range(max_retries):
            retry_notice = ""
            if attempt:
                retry_notice = (
                    "A previous internal candidate was rejected. Recompute from "
                    "the same visible sources; no failed attempt or judge value is "
                    "available."
                )
            prompt = f"""Generate arguments for all calls in one parallel batch.

=== USER QUERY ===
{query}

=== PRIOR SAVED TOOL CALLS AND OUTPUTS ===
{json.dumps(visible_history, indent=2, ensure_ascii=False, default=str)}

=== PRIOR-TURN EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, ensure_ascii=False, default=str)}

=== PARALLEL CALL SPECS ===
{json.dumps(call_specs, indent=2, ensure_ascii=False, default=str)}

=== RULES ===
1. Return one entry for every call_id, in the supplied order.
2. Every argument must come from the user query, prior saved outputs/context, or
   a declared schema default.
3. Do not use simulator state, persona data, internal feedback, or model knowledge
   to invent opaque IDs, codes, tokens, symbols, paths, coordinates, or credentials.
4. Sibling parallel calls cannot see or consume one another's outputs.
5. Obey the complete schema, including required fields, enum, pattern, ranges,
   arrays, and nested objects.
6. If a required value is unavailable, return
   {{"__missing_required_argument__": ["argument_name"]}} for that call rather
   than guessing.

{retry_notice}

Respond only with JSON:
{{
  "calls": [
    {{"call_id": "p1", "arguments": {{}}}},
    {{"call_id": "p2", "arguments": {{}}}}
  ]
}}
"""
            try:
                result = self._extract_json_object(
                    self._safe_llm_generate([{"role": "user", "content": prompt}])
                )
            except Exception as exc:
                print(f"  Parallel argument attempt {attempt + 1} failed: {exc}")
                continue

            raw_calls = result.get("calls", [])
            if not isinstance(raw_calls, list) or len(raw_calls) != len(call_specs):
                continue
            by_id: Dict[str, Dict[str, Any]] = {}
            malformed = False
            for item in raw_calls:
                if not isinstance(item, dict):
                    malformed = True
                    break
                call_id = str(item.get("call_id", ""))
                arguments = item.get("arguments", {})
                if call_id in by_id or not isinstance(arguments, dict):
                    malformed = True
                    break
                by_id[call_id] = arguments
            if malformed or set(by_id) != {spec["call_id"] for spec in call_specs}:
                continue

            materialised: List[Dict[str, Any]] = []
            valid = True
            for spec in call_specs:
                arguments = by_id[spec["call_id"]]
                if self._contains_unresolved_marker(arguments):
                    valid = False
                    break
                schema_issues = self._validate_tool_arguments_schema(
                    spec["tool_name"], arguments
                )
                if schema_issues:
                    valid = False
                    break
                tool_schema = self.tool_manager.get_tool_schema(
                    spec["tool_name"]
                )
                properties = tool_schema.get("parameters", {}).get(
                    "properties", {}
                )
                visible_text = query + "\n" + json.dumps(
                    {
                        "history": visible_history,
                        "prior_turn_outputs": execution_context.get(
                            "turn_outputs", []
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
                invisible = []
                for argument_name, value in arguments.items():
                    argument_schema = properties.get(argument_name, {})
                    declared = (
                        value == argument_schema.get("const")
                        or value in argument_schema.get("enum", [])
                        or value == argument_schema.get("default")
                    )
                    if not declared and not self._value_visible_in_text(
                        value, visible_text
                    ):
                        invisible.append(argument_name)
                if invisible:
                    valid = False
                    break
                materialised.append(
                    {
                        "call_id": spec["call_id"],
                        "tool_name": spec["tool_name"],
                        "arguments": arguments,
                    }
                )
            if valid:
                signatures = [
                    self._canonical(
                        {
                            "tool_name": call["tool_name"],
                            "arguments": call["arguments"],
                        }
                    )
                    for call in materialised
                ]
                if len(signatures) != len(set(signatures)):
                    # A duplicate call with identical arguments adds no useful
                    # parallel behavior and creates a non-unique/redundant target.
                    valid = False
                else:
                    self._last_parallel_argument_certificate = {
                        "passed": True,
                        "validator": "deterministic",
                        "schema_valid": True,
                        "policy_visible_literals": True,
                        "sibling_output_dependencies": False,
                    }
                    return materialised

        return None

    def _certify_parallel_arguments_policy_visible(
        self,
        *,
        query: str,
        calls: List[Dict[str, Any]],
        visible_history: List[Dict[str, Any]],
        execution_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fail-closed audit of the complete parallel target from policy context.

        This is deliberately separate from the per-call legacy consistency judge:
        it sees the whole batch, but it never sees simulator state or internal
        retry feedback.  It therefore catches sibling dependencies and hidden
        opaque arguments before execution.
        """
        allowed_codes = {
            "POLICY_CONTEXT_NOT_CLOSED",
            "SIBLING_OUTPUT_DEPENDENCY",
            "QUERY_ARGUMENT_MISMATCH",
            "AMBIGUOUS_GOLD_ACTION",
            "SCHEMA_SEMANTIC_MISMATCH",
            "PARALLEL_ARGUMENT_CERTIFIER_UNAVAILABLE",
            "OTHER_INVALID",
        }
        call_payload = [
            {
                "call_id": call["call_id"],
                "tool_name": call["tool_name"],
                "arguments": call["arguments"],
                "tool_definition": self.tool_manager.get_tool_schema(
                    call["tool_name"]
                ),
            }
            for call in calls
        ]
        prompt = f"""Certify the complete argument set for one parallel tool-call
batch intended as positive RL data.

=== USER QUERY ===
{query}

=== PRIOR POLICY-VISIBLE TOOL HISTORY ===
{json.dumps(visible_history, indent=2, ensure_ascii=False, default=str)}

=== PRIOR-TURN POLICY-VISIBLE EXECUTION CONTEXT ===
{json.dumps(execution_context, indent=2, ensure_ascii=False, default=str)}

=== PROPOSED PARALLEL CALLS AND FULL DEFINITIONS ===
{json.dumps(call_payload, indent=2, ensure_ascii=False, default=str)}

The policy sees only the query, prior visible history/context, and tool
definitions. It does not see simulator state, persona objects, internal failed
attempts, judge messages, or sibling outputs.

Require all of the following:
1. Every exact opaque ID, code, token, symbol, handle, coordinate, path, or
   credential is explicitly visible before the batch or is a schema default.
2. No argument comes from a sibling call's result.
3. Every call and argument matches an unambiguous part of the user request.
4. Human-readable labels are not substituted for schema-required opaque values.
5. No unsupported list-item choice or external/world-knowledge lookup is used.

Do not suggest replacement values. Return only allowed issue codes:
{json.dumps(sorted(allowed_codes))}

Respond only with JSON:
{{"is_valid": true, "issue_codes": []}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate(
                    [{"role": "user", "content": prompt}],
                    llm=self.judge,
                )
            )
            raw_valid = result.get("is_valid")
            raw_codes = result.get("issue_codes")
            if not isinstance(raw_valid, bool) or not isinstance(raw_codes, list):
                return {
                    "passed": False,
                    "issue_codes": [
                        "PARALLEL_ARGUMENT_CERTIFIER_UNAVAILABLE"
                    ],
                }
            codes = [
                str(code)
                for code in raw_codes
                if str(code) in allowed_codes
            ]
            passed = raw_valid and not codes
            if not passed and not codes:
                codes = ["OTHER_INVALID"]
            return {"passed": passed, "issue_codes": codes}
        except Exception as exc:
            print(f"  Parallel argument certifier failed closed: {exc}")
            return {
                "passed": False,
                "issue_codes": ["PARALLEL_ARGUMENT_CERTIFIER_UNAVAILABLE"],
            }

    def _invoke_and_validate_read_only(
        self,
        *,
        call: Dict[str, Any],
        pre_state: Dict[str, Any],
        step_number: int,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]], List[str]]:
        tool_name = call["tool_name"]
        arguments = call["arguments"]
        self.tool_manager.restore_api_state(copy.deepcopy(pre_state))
        output = self.tool_manager.invoke_python_tool(tool_name, arguments)

        issues: List[str] = []
        if not isinstance(output, dict):
            # The base verifier can handle primitive/list outputs, but the error
            # detector expects a dict.  Wrap only for error inspection.
            error_probe: Dict[str, Any] = {"result": output}
        else:
            error_probe = output
        has_error, _ = self._detect_tool_error(tool_name, error_probe)
        if has_error:
            issues.append("TOOL_EXECUTION_ERROR")

        schema = self.tool_manager.get_tool_schema(tool_name)
        if self.validate_outputs:
            validation = self.verify_output_consistency(
                tool_name,
                step_number,
                output,
                schema.get("output_type", "unknown"),
                schema.get("output_description", ""),
            )
            if not validation.get("output_type_matches", False) or validation.get("issues"):
                issues.append("OUTPUT_SCHEMA_INVALID")

        post_state = self.tool_manager.get_api_state()
        if post_state != pre_state:
            issues.append("PARALLEL_CALL_MUTATED_STATE")

        transition = validate_transition_quality(
            tool_name=tool_name,
            tool_output=output,
            pre_state=pre_state,
            post_state=post_state,
            tool_arguments=arguments,
        )
        if not transition.get("passed", False):
            issues.append("TRANSITION_QUALITY_FAILED")

        return output, post_state, issues

    def _execute_parallel_batch(
        self,
        *,
        query_result: FeatureQueryGenerationResult,
        max_retries: int,
        initial_execution_context: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[List[TrajectoryStep]], Optional[Dict[str, Any]]]:
        if not self._python_tools_available:
            print("  Parallel generation requires executable Python tools")
            return None, None

        tool_names = list(query_result.expected_tools)
        execution_context = copy.deepcopy(initial_execution_context or {})
        pre_state = self.tool_manager.get_api_state()

        calls = self._generate_parallel_arguments(
            query=query_result.query,
            tool_names=tool_names,
            trajectory=[],
            execution_context=execution_context,
            max_retries=max_retries,
        )
        if not calls:
            self.tool_manager.restore_api_state(pre_state)
            return None, None

        isolated_outputs: Dict[str, Any] = {}
        per_call_checks: List[Dict[str, Any]] = []
        for call in calls:
            output, _, issues = self._invoke_and_validate_read_only(
                call=call,
                pre_state=pre_state,
                step_number=1,
            )
            if issues:
                self.tool_manager.restore_api_state(pre_state)
                return None, None
            isolated_outputs[call["call_id"]] = output
            per_call_checks.append(
                {
                    "call_id": call["call_id"],
                    "tool_name": call["tool_name"],
                    "passed": True,
                    "read_only": True,
                }
            )

        def run_order(order: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
            self.tool_manager.restore_api_state(copy.deepcopy(pre_state))
            outputs: Dict[str, Any] = {}
            run_issues: List[str] = []
            for call in order:
                output = self.tool_manager.invoke_python_tool(
                    call["tool_name"], call["arguments"]
                )
                probe = output if isinstance(output, dict) else {"result": output}
                has_error, _ = self._detect_tool_error(call["tool_name"], probe)
                if has_error:
                    run_issues.append("TOOL_EXECUTION_ERROR")
                outputs[call["call_id"]] = output
            return outputs, self.tool_manager.get_api_state(), run_issues

        forward_outputs, forward_state, forward_issues = run_order(calls)
        reverse_outputs, reverse_state, reverse_issues = run_order(list(reversed(calls)))

        if forward_issues or reverse_issues:
            self.tool_manager.restore_api_state(pre_state)
            return None, None
        if forward_state != pre_state or reverse_state != pre_state:
            self.tool_manager.restore_api_state(pre_state)
            return None, None
        if self._canonical(forward_outputs) != self._canonical(reverse_outputs):
            self.tool_manager.restore_api_state(pre_state)
            return None, None
        if self._canonical(forward_outputs) != self._canonical(isolated_outputs):
            self.tool_manager.restore_api_state(pre_state)
            return None, None

        # Leave the live environment in the declared-order final state. For a
        # certified read-only batch this is identical to pre_state.
        self.tool_manager.restore_api_state(copy.deepcopy(pre_state))
        committed_outputs: Dict[str, Any] = {}
        for call in calls:
            committed_outputs[call["call_id"]] = self.tool_manager.invoke_python_tool(
                call["tool_name"], call["arguments"]
            )
        post_state = self.tool_manager.get_api_state()
        if post_state != pre_state or self._canonical(committed_outputs) != self._canonical(forward_outputs):
            self.tool_manager.restore_api_state(pre_state)
            return None, None

        tool_calls: List[ToolCallWithOutput] = []
        for call in calls:
            output = committed_outputs[call["call_id"]]
            tool_calls.append(
                ToolCallWithOutput(
                    tool_name=call["tool_name"],
                    arguments=call["arguments"],
                    output=output,
                )
            )

        certificate = {
            "passed": True,
            "mode": "parallel",
            "read_only": True,
            "same_pre_batch_context": True,
            "forward_reverse_outputs_equal": True,
            "forward_reverse_state_equal": True,
            "isolated_outputs_equal": True,
            "argument_visibility_certificate": copy.deepcopy(
                getattr(
                    self,
                    "_last_parallel_argument_certificate",
                    {"passed": True, "source": "prevalidated_override"},
                )
            ),
            "per_call_checks": per_call_checks,
        }
        step = TrajectoryStep(
            step_number=1,
            tool_calls=tool_calls,
            execution_mode="parallel",
            call_order_matters=False,
            reasoning="Certified independent calls executed as one parallel batch",
            pre_state=pre_state,
            post_state=post_state,
            state_verification=StateVerificationResult(
                is_valid=True,
                reasoning=(
                    "Every call was executed from the same pre-batch state; "
                    "forward and reverse orders produced identical outputs and state."
                ),
                issues=[],
                state_changes_summary="No state changes; batch is certified read-only.",
            ),
            quality_verification=certificate,
        )

        execution_context.setdefault("parallel_batches", []).append(
            {
                "step_number": 1,
                "calls": [
                    {
                        "call_id": call["call_id"],
                        "tool_name": call["tool_name"],
                        "arguments": call["arguments"],
                        "output": committed_outputs[call["call_id"]],
                    }
                    for call in calls
                ],
            }
        )
        by_tool: Dict[str, List[Any]] = {}
        for call in calls:
            output = committed_outputs[call["call_id"]]
            execution_context[f"call_{call['call_id']}_output"] = output
            by_tool.setdefault(call["tool_name"], []).append(output)
        for tool_name, outputs in by_tool.items():
            execution_context[f"{tool_name}_outputs"] = outputs
            if len(outputs) == 1:
                execution_context[f"{tool_name}_output"] = outputs[0]
                if isinstance(outputs[0], dict):
                    for key, value in outputs[0].items():
                        execution_context[f"{tool_name}_{key}"] = value

        return [step], execution_context

    # ------------------------------------------------------------------
    # Shared stage hooks
    # ------------------------------------------------------------------

    def _stage1_5_adjust_initial_state(self, query_result: QueryGenerationResult) -> bool:
        if isinstance(query_result, FeatureQueryGenerationResult) and query_result.mode in {
            "refusal",
            "parallel",
        }:
            # Refusal must remain impossible from visible context, and parallel
            # tasks are already certified against the sampled state. Mutating the
            # state after either certification would invalidate the certificate.
            return False
        return super()._stage1_5_adjust_initial_state(query_result)

    def _stage2_generate_tools(
        self,
        query_result: QueryGenerationResult,
        max_retries_per_tool: int,
        initial_execution_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[TrajectoryStep]], Optional[Dict[str, Any]]]:
        if not isinstance(query_result, FeatureQueryGenerationResult):
            return super()._stage2_generate_tools(
                query_result,
                max_retries_per_tool,
                initial_execution_context=initial_execution_context,
            )

        if query_result.mode == "refusal":
            reason = query_result.refusal_type or "no_appropriate_function"
            # A refusal has no simulator transition.  Do not attach the entire
            # hidden state to this pseudo-action, where downstream exporters could
            # accidentally treat it as policy context.
            state = None
            output = {"status": "refused", "reason": reason}
            native_response = (
                query_result.native_response
                or query_result.feature_certificate.get("assistant_response")
                or self._refusal_response(reason)
            )
            certificate = {
                "passed": True,
                "mode": "refusal",
                "reason": reason,
                "native_response": native_response,
                "refusal_certificate": query_result.feature_certificate,
            }
            step = TrajectoryStep(
                step_number=1,
                tool_calls=[
                    ToolCallWithOutput(
                        tool_name="refuse",
                        arguments={"reason": reason},
                        output=output,
                    )
                ],
                execution_mode="refusal",
                call_order_matters=True,
                reasoning="Terminal refusal certified from policy-visible context",
                pre_state=state,
                post_state=copy.deepcopy(state),
                state_verification=StateVerificationResult(
                    is_valid=True,
                    reasoning="Synthetic terminal action; simulator state is unchanged.",
                    issues=[],
                    state_changes_summary="No state changes.",
                ),
                quality_verification=certificate,
            )
            context = copy.deepcopy(initial_execution_context or {})
            context["refusal"] = output
            return [step], context

        if query_result.mode == "parallel":
            return self._execute_parallel_batch(
                query_result=query_result,
                max_retries=max_retries_per_tool,
                initial_execution_context=initial_execution_context,
            )

        return super()._stage2_generate_tools(
            query_result,
            max_retries_per_tool,
            initial_execution_context=initial_execution_context,
        )

    def _generate_final_response(
        self,
        query: str,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
    ) -> str:
        if (
            len(trajectory) == 1
            and len(trajectory[0].tool_calls) == 1
            and trajectory[0].tool_calls[0].tool_name == "refuse"
        ):
            reason = trajectory[0].tool_calls[0].arguments.get(
                "reason", "no_appropriate_function"
            )
            self._last_final_response_quality = {
                "passed": True,
                "mode": "refusal",
                "issue_codes": [],
                "reason": reason,
            }
            native_response = trajectory[0].quality_verification.get(
                "native_response"
            )
            return native_response or self._refusal_response(reason)
        return super()._generate_final_response(query, trajectory, execution_context)

    @staticmethod
    def _contains_refusal(trajectory: List[TrajectoryStep]) -> bool:
        return any(
            call.tool_name == "refuse"
            for step in trajectory
            for call in step.tool_calls
        )


class RefusalParallelStepByStepGenerator(_RefusalParallelSupport, StepByStepGenerator):
    """Opt-in step-by-step generator. Original generator is untouched."""

    def __init__(
        self,
        *args: Any,
        allow_refusal: bool = False,
        refusal_rate: float = 0.12,
        allow_parallel: bool = False,
        parallel_rate: float = 0.25,
        max_parallel_width: int = 3,
        require_feature: bool = False,
        feature_difficulty: str = "standard",
        naturalize_queries: bool = False,
        forced_refusal_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._configure_refusal_parallel(
            allow_refusal=allow_refusal,
            refusal_rate=refusal_rate,
            allow_parallel=allow_parallel,
            parallel_rate=parallel_rate,
            max_parallel_width=max_parallel_width,
            require_feature=require_feature,
            feature_difficulty=feature_difficulty,
            naturalize_queries=naturalize_queries,
            forced_refusal_reason=forced_refusal_reason,
        )

    def _stage1_generate_query(
        self,
        focus_category: Optional[str],
        context_hint: Optional[str],
        max_retries: int,
        query_seed: Optional[dict] = None,
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[QueryGenerationResult]:
        parallel_eligible = (
            2 <= self.num_actions <= self.feature_config.max_parallel_width
        )
        required_mode = self._required_feature_mode(
            parallel_eligible=parallel_eligible
        )
        if required_mode == "unavailable":
            print("  Required feature is not eligible for this configuration")
            return None

        try_refusal = (
            required_mode == "refusal"
            or (
                required_mode is None
                and self.feature_config.allow_refusal
                and random.random() < self.feature_config.refusal_rate
            )
        )
        if try_refusal:
            for _ in range(min(max_retries, self.max_turn_attempts)):
                result = self._generate_refusal_query(
                    focus_category=focus_category,
                    refusal_type=self._sample_refusal_reason(),
                )
                if result is not None:
                    return result
            if required_mode == "refusal":
                print("  Required refusal candidate failed; resampling datapoint")
                return None
            print("  Refusal feature candidate failed; falling back to current-main generation")

        try_parallel = (
            parallel_eligible
            and (
                required_mode == "parallel"
                or (
                    required_mode is None
                    and self.feature_config.allow_parallel
                    and random.random() < self.feature_config.parallel_rate
                )
            )
        )
        if try_parallel:
            for _ in range(min(max_retries, self.max_turn_attempts)):
                result = self._generate_parallel_query(
                    focus_category=focus_category,
                    num_calls=self.num_actions,
                    initial_api_state=initial_api_state,
                    query_seed=query_seed,
                    max_retries=1,
                )
                if result is not None:
                    return result
            if required_mode == "parallel":
                print("  Required parallel candidate failed; resampling datapoint")
                return None
            print("  Parallel feature candidate failed; falling back to current-main generation")

        if required_mode is not None:
            return None
        return super()._stage1_generate_query(
            focus_category,
            context_hint,
            max_retries,
            query_seed,
            initial_api_state,
        )

    def _stage3_finalize(
        self,
        query_result: QueryGenerationResult,
        trajectory: List[TrajectoryStep],
        execution_context: Dict[str, Any],
        focus_category: Optional[str],
        initial_api_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[StepByStepDatapoint]:
        if isinstance(query_result, FeatureQueryGenerationResult) and query_result.mode == "refusal":
            final_response = self._generate_final_response(
                query_result.query, trajectory, execution_context
            )
            available_tools = self._get_policy_tool_schemas(focus_category)
            token_usage = self._get_token_stats()
            reason = query_result.refusal_type or "no_appropriate_function"
            conversation = ConversationTrajectory(
                query=query_result.query,
                steps=trajectory,
                final_response=final_response,
                tools_used=["refuse"],
                categories_used=["Control"],
                initial_api_state=copy.deepcopy(initial_api_state),
            )
            quality_gate = {
                "passed": True,
                "query_preflight": query_result.quality_preflight,
                "transition_checks": [trajectory[0].quality_verification],
                "final_response_grounding": dict(self._last_final_response_quality),
            }
            verification = {
                "query": query_result.query,
                "tool_relevance_checks": [
                    {
                        "tool_name": "refuse",
                        "is_relevant": True,
                        "relevance_score": 1.0,
                        "reasoning": "Certified terminal refusal.",
                    }
                ],
                "order_is_correct": True,
                "order_verification_details": "Single terminal refusal action.",
                "output_validations": [
                    {
                        "tool_name": "refuse",
                        "step_number": 1,
                        "output_type_matches": True,
                        "issues": [],
                    }
                ],
                "placeholder_resolution": {
                    "all_resolved": True,
                    "total_placeholders": 0,
                    "resolved_count": 0,
                    "details": [],
                },
                "rl_quality_gate": quality_gate,
                "overall_verification_passed": True,
                "verification_summary": "Verification PASSED",
            }
            terminal_mode = (
                "refusal"
                if reason == "no_appropriate_function"
                else "clarification"
            )
            metadata = {
                "num_actions": 1,
                "num_steps": 1,
                "focus_category": focus_category,
                "query_intent": query_result.intent,
                "expected_tools": ["refuse"],
                "terminal_mode": terminal_mode,
                "terminal_action": "refuse",
                "refusal_type": reason,
                "feature_difficulty": self.feature_config.feature_difficulty,
                "rl_quality_gate_passed": True,
                "query_quality_preflight": query_result.quality_preflight,
                "final_response_grounding": dict(self._last_final_response_quality),
                "tool_contract_hash": self._tool_contract_hash(available_tools),
            }
            datapoint = StepByStepDatapoint(
                trajectory=conversation,
                generation_metadata=metadata,
                verification_result=verification,
                token_usage=token_usage,
                initial_api_state=None,
                intermediate_api_states=[],
                available_tools=available_tools,
            )
            return prepare_step_by_step_datapoint(datapoint)

        datapoint = super()._stage3_finalize(
            query_result,
            trajectory,
            execution_context,
            focus_category,
            initial_api_state,
        )
        if datapoint is not None and isinstance(query_result, FeatureQueryGenerationResult):
            if query_result.mode == "parallel":
                datapoint.generation_metadata.update(
                    {
                        "num_actions": sum(
                            len(step.tool_calls) for step in trajectory
                        ),
                        "num_steps": len(trajectory),
                        "contains_parallel": True,
                        "parallel_group_count": sum(
                            1 for step in trajectory if len(step.tool_calls) > 1
                        ),
                        "parallel_order_invariant": False,
                        "parallel_order_invariance_scope": "per_transition_only",
                        "action_plan": query_result.action_plan,
                        "parallel_certificate": query_result.feature_certificate,
                        "feature_difficulty": (
                            self.feature_config.feature_difficulty
                        ),
                    }
                )
        return prepare_step_by_step_datapoint(datapoint)


class RefusalParallelMultiTurnGenerator(_RefusalParallelSupport, MultiTurnGenerator):
    """Opt-in multi-turn generator with refusal and parallel turns."""

    def __init__(
        self,
        *args: Any,
        judge_client: Any = None,
        allow_refusal: bool = False,
        refusal_rate: float = 0.12,
        allow_parallel: bool = False,
        parallel_rate: float = 0.25,
        max_parallel_width: int = 3,
        require_feature: bool = False,
        feature_difficulty: str = "standard",
        naturalize_queries: bool = False,
        multi_turn_feature_schedule: str = "terminal",
        forced_refusal_reason: Optional[str] = None,
        interactive_refusal_turn: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if judge_client is not None:
            self.judge = judge_client
        self._configure_refusal_parallel(
            allow_refusal=allow_refusal,
            refusal_rate=refusal_rate,
            allow_parallel=allow_parallel,
            parallel_rate=parallel_rate,
            max_parallel_width=max_parallel_width,
            require_feature=require_feature,
            feature_difficulty=feature_difficulty,
            naturalize_queries=naturalize_queries,
            multi_turn_feature_schedule=multi_turn_feature_schedule,
            forced_refusal_reason=forced_refusal_reason,
            interactive_refusal_turn=interactive_refusal_turn,
        )
        if interactive_refusal_turn is not None:
            minimum_turn = 3
            maximum_turn = self.num_turns - 2
            if not (
                minimum_turn
                <= interactive_refusal_turn
                <= maximum_turn
            ):
                raise ValueError(
                    "interactive_refusal_turn must leave two prior turns, one "
                    "recovery turn, and one later turn; expected "
                    f"{minimum_turn}-{maximum_turn}, got "
                    f"{interactive_refusal_turn}"
                )
        self._last_blueprint_naturalization: Dict[str, Any] = {
            "enabled": False,
            "rewritten": False,
        }
        self._pending_clarification_recovery: Optional[Dict[str, Any]] = None
        self._clarification_events: List[Dict[str, Any]] = []
        self._scheduled_refusal_index: Optional[int] = None

    def _stage0_generate_blueprint(
        self,
        focus_category: Optional[str] = None,
        initial_api_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[DialogBlueprint]:
        """Plan first, then independently rewrite the fixed plan as dialogue."""

        self._pending_clarification_recovery = None
        self._clarification_events = []
        self._scheduled_refusal_index = None
        self._last_blueprint_naturalization = {
            "enabled": self.feature_config.naturalize_queries,
            "rewritten": False,
        }
        blueprint = super()._stage0_generate_blueprint(
            focus_category,
            initial_api_state,
        )
        if blueprint is None:
            return blueprint
        if self.feature_config.multi_turn_feature_schedule in {
            "interactive-refusal",
            "combined",
        }:
            forced_turn = self.feature_config.interactive_refusal_turn
            self._scheduled_refusal_index = (
                forced_turn - 1
                if forced_turn is not None
                else self._select_interactive_refusal_index(blueprint)
            )
        if not self.feature_config.naturalize_queries:
            return blueprint

        source_turns = copy.deepcopy(blueprint.turns)
        plans = [
            {
                "turn": index + 1,
                "source_query": turn.get("user_query", ""),
                "fixed_expected_tools": list(turn.get("expected_tools", [])),
            }
            for index, turn in enumerate(source_turns)
        ]
        protected_by_turn = [
            self._protected_query_tokens(str(turn.get("user_query", "")))
            for turn in source_turns
        ]
        relevant_tool_names = sorted(
            {
                name
                for turn in source_turns
                for name in turn.get("expected_tools", [])
                if self.tool_manager.tool_exists(name)
            }
        )
        tool_contracts = [
            self.tool_manager.get_tool_schema(name)
            for name in relevant_tool_names
        ]
        prompt = f"""Rewrite the user side of a synthetic multi-turn tool-use plan into
natural conversation. The plan, tool order, facts, and cross-turn dependencies
are already fixed. This is a STYLE-ONLY second pass.

=== FIXED TURN PLANS ===
{json.dumps(plans, ensure_ascii=False, indent=2, default=str)}

=== RELEVANT TOOL CONTRACTS (for semantic preservation only) ===
{json.dumps(tool_contracts, ensure_ascii=False, indent=2, default=str)}

=== TOKENS EACH TURN MUST PRESERVE VERBATIM ===
{json.dumps(protected_by_turn, ensure_ascii=False)}

Rules:
1. Return exactly {len(source_turns)} queries in the original order. Do not add,
   remove, merge, or split requested operations.
2. Preserve every argument-determining fact, exact value, placeholder, entity,
   constraint, selection rule, and dependency. Never invent a value.
3. Write like a real person continuing one coherent conversation. Use natural
   references to prior results, varied syntax, concise motivations, and implicit
   context where it remains unambiguous.
4. Never mention function/tool names, JSON fields, schemas, calls, a plan, or
   execution order jargon. Do not sound like a benchmark instruction.
5. Do not ask the user to provide an opaque internal ID when an earlier result
   already identifies the object. Refer naturally to "that booking", "the
   message you just sent", "the ticket we opened", and similar objects while
   retaining any machine placeholder needed by the fixed executable plan.
6. Opaque values may appear only when ordinary users naturally know/provide
   them (for example a ticket number, tracking code, username, filename, or
   confirmation code) or when represented by an earlier-result placeholder.
7. If a turn contains several operations, express one realistic higher-level
   goal rather than a numbered checklist.

Few-shot examples:

SOURCE: "Get user id for Sarah with get_user_id, then send a message."
NATURAL: "Could you let Sarah know I’ll be about ten minutes late?"

SOURCE: "Use {{{{TURN2.create_ticket.ticket_id}}}} to resolve the ticket."
NATURAL: "That issue we just logged is fixed now—please mark it resolved."

SOURCE: "Call get_weather for Paris and get_exchange_rate for EUR to USD."
NATURAL: "I’m budgeting for the Paris trip; what weather should I pack for,
and what’s the current EUR-to-USD rate?"

Respond only with JSON:
{{"queries": ["...", "..."]}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate([{"role": "user", "content": prompt}])
            )
        except Exception as exc:
            print(f"  Blueprint naturalization failed: {exc}")
            return None

        rewritten = result.get("queries")
        if not isinstance(rewritten, list) or len(rewritten) != len(source_turns):
            print("  Blueprint naturalization returned the wrong turn count")
            return None
        rewritten = [str(query).strip() for query in rewritten]
        if any(not query for query in rewritten):
            return None

        missing_tokens: List[Dict[str, Any]] = []
        for index, (query, protected) in enumerate(
            zip(rewritten, protected_by_turn),
            1,
        ):
            missing = [token for token in protected if token not in query]
            if missing:
                missing_tokens.append({"turn": index, "tokens": missing})
        if missing_tokens:
            print(
                "  Blueprint naturalization dropped protected values: "
                + json.dumps(missing_tokens[:3], ensure_ascii=False)
            )
            return None

        changed = [
            source_turns[index].get("user_query", "") != query
            for index, query in enumerate(rewritten)
        ]
        certification_prompt = f"""Certify a style-only rewrite of a fixed
multi-turn tool-use plan.

=== SOURCE AND REWRITTEN TURNS WITH FIXED PLANS ===
{json.dumps([
    {
        **plan,
        "rewritten_query": rewritten[index],
    }
    for index, plan in enumerate(plans)
], ensure_ascii=False, indent=2, default=str)}

The rewritten dialogue is valid only if:
- every turn keeps exactly the same executable intent, argument facts,
  requested results, selection rule, and dependency as its source;
- it adds no unsupported value, operation, or assumption;
- the turns form a coherent, natural conversation rather than tool/API syntax;
- it avoids unnecessary raw internal IDs and never mentions tool/function names;
- every fixed tool plan remains fully determined from the rewritten current
  query plus preceding rewritten dialogue and tool results.

Respond only with JSON:
{{
  "semantic_plan_preserved": true,
  "natural_conversation": true,
  "no_tool_syntax": true,
  "avoids_unnecessary_internal_ids": true
}}
"""
        try:
            certificate = self._extract_json_object(
                self._safe_llm_generate(
                    [{"role": "user", "content": certification_prompt}],
                    llm=self.judge,
                )
            )
        except Exception as exc:
            print(f"  Blueprint naturalness certifier failed: {exc}")
            return None
        required = (
            "semantic_plan_preserved",
            "natural_conversation",
            "no_tool_syntax",
            "avoids_unnecessary_internal_ids",
        )
        if any(certificate.get(key) is not True for key in required):
            print("  Blueprint naturalness certificate failed closed")
            return None

        for turn, query in zip(blueprint.turns, rewritten):
            turn["user_query"] = query
        self._last_blueprint_naturalization = {
            "enabled": True,
            "rewritten": True,
            "rewritten_turns": sum(changed),
            "total_turns": len(rewritten),
            "protected_tokens_preserved": True,
            "source_queries": [
                str(turn.get("user_query", "")) for turn in source_turns
            ],
            "rewritten_queries": rewritten,
            "certificate": certificate,
        }
        return blueprint

    def _select_interactive_refusal_index(
        self,
        blueprint: DialogBlueprint,
    ) -> int:
        if blueprint.num_turns < 4:
            raise ValueError(
                "interactive-refusal and combined schedules need at least 4 turns"
            )
        candidates = range(2, blueprint.num_turns - 2)
        desired_reason = self.feature_config.forced_refusal_reason

        def score(index: int) -> Tuple[int, int, int]:
            expected = blueprint.turns[index].get("expected_tools", [])
            required_count = 0
            enum_count = 0
            for name in expected:
                if not self.tool_manager.tool_exists(name):
                    continue
                parameters = self.tool_manager.get_tool_schema(name).get(
                    "parameters", {}
                )
                required_count += len(parameters.get("required", []))
                enum_count += sum(
                    1
                    for prop in parameters.get("properties", {}).values()
                    if isinstance(prop, dict) and prop.get("enum")
                )
            if desired_reason == "missing_argument":
                primary = required_count
            elif desired_reason == "ambiguity":
                primary = enum_count * 2 + required_count
            else:
                primary = required_count + enum_count
            # Prefer a feature late enough to depend on rich history while
            # reserving a recovery turn and at least one later task.
            return primary, len(expected), index

        return max(candidates, key=score)

    def _interactive_refusal_index(self, blueprint: DialogBlueprint) -> int:
        if self._scheduled_refusal_index is not None:
            return self._scheduled_refusal_index
        if blueprint.num_turns < 4:
            raise ValueError(
                "interactive-refusal and combined schedules need at least 4 turns"
            )
        return min(
            max(2, blueprint.num_turns // 2),
            blueprint.num_turns - 3,
        )

    def _generate_clarification_recovery_query(
        self,
        *,
        pending: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        source_query = str(pending["source_query"])
        protected = self._protected_query_tokens(source_query)
        expected_tools = list(pending["expected_tools"])
        contracts = [
            self.tool_manager.get_tool_schema(name)
            for name in expected_tools
        ]
        prompt = f"""Write the user's next message after a precise assistant
clarification question. The next message must naturally answer that question
and let the assistant complete the original fully specified request.

=== BLOCKED USER REQUEST ===
{pending["refusal_query"]}

=== ASSISTANT CLARIFICATION ===
{pending["assistant_response"]}

=== FULLY SPECIFIED SOURCE REQUEST ===
{source_query}

=== FIXED TOOL CONTRACTS (do not name them in the output) ===
{json.dumps(contracts, ensure_ascii=False, indent=2, default=str)}

=== PRIOR POLICY-VISIBLE HISTORY ===
{json.dumps(history, ensure_ascii=False, indent=2, default=str)}

=== SOURCE TOKENS TO PRESERVE VERBATIM ===
{json.dumps(protected, ensure_ascii=False)}

Return one natural conversational reply. It should directly supply the missing
fact or choose the intended interpretation, briefly confirm the original goal,
and contain everything needed for the fixed executable plan. Do not mention
tools, APIs, schemas, argument names, or that this is a benchmark. Do not add
any operation or value not present in the fully specified source request.

Few-shot examples:

ASSISTANT: "Which address should I use for the delivery?"
SOURCE: "Send it to 18 Willow Lane."
REPLY: "Use 18 Willow Lane—the same place as last time—and go ahead with it."

ASSISTANT: "Do you mean the personal or work calendar?"
SOURCE: "Put the appointment on my work calendar."
REPLY: "The work calendar, please; keep the appointment at the time we discussed."

Respond only with JSON:
{{"query": "..."}}
"""
        try:
            result = self._extract_json_object(
                self._safe_llm_generate([{"role": "user", "content": prompt}])
            )
        except Exception as exc:
            print(f"  Clarification recovery generation failed: {exc}")
            return None
        query = str(result.get("query", "")).strip()
        if not query:
            return None
        missing = [token for token in protected if token not in query]
        if missing:
            print(
                "  Clarification recovery dropped source values: "
                + ", ".join(missing[:5])
            )
            return None
        return query, {
            "passed": True,
            "mode": "clarification_recovery",
            "reason": pending["reason"],
            "source_query": source_query,
            "blocked_query": pending["refusal_query"],
            "assistant_clarification": pending["assistant_response"],
            "protected_tokens": protected,
            "protected_tokens_preserved": True,
        }

    @staticmethod
    def _conversation_policy_history(
        conversation: MultiTurnConversation,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for turn in conversation.turns:
            history.append({"role": "user", "content": turn.user_query})
            for step in turn.steps:
                for call_index, call in enumerate(step.tool_calls, 1):
                    history.append(
                        {
                            "role": "assistant_tool_call",
                            "call_id": f"t{turn.turn_number}s{step.step_number}c{call_index}",
                            "name": call.tool_name,
                            "arguments": call.arguments,
                        }
                    )
                    history.append(
                        {
                            "role": "tool",
                            "name": call.tool_name,
                            "content": call.output,
                        }
                    )
            if turn.assistant_response:
                history.append(
                    {"role": "assistant", "content": turn.assistant_response}
                )
        return history

    def _generate_turn_query(
        self,
        blueprint: DialogBlueprint,
        conversation: MultiTurnConversation,
        turn_index: int,
    ) -> Optional[QueryGenerationResult]:
        turn_spec = blueprint.turns[turn_index] if turn_index < len(blueprint.turns) else {}
        original_query = self._resolve_turn_placeholders(
            str(turn_spec.get("user_query", "")),
            turn_index,
            conversation,
        )
        history = self._conversation_policy_history(conversation)
        is_final_turn = turn_index == blueprint.num_turns - 1
        schedule = self.feature_config.multi_turn_feature_schedule
        interactive_index = (
            self._interactive_refusal_index(blueprint)
            if schedule in {"interactive-refusal", "combined"}
            else None
        )

        if (
            self._pending_clarification_recovery is not None
            and turn_index
            == self._pending_clarification_recovery.get("recovery_index")
        ):
            pending = self._pending_clarification_recovery
            recovery = self._generate_clarification_recovery_query(
                pending=pending,
                history=history,
            )
            if recovery is None:
                return None
            recovery_query, recovery_certificate = recovery
            rewritten_turns = copy.deepcopy(blueprint.turns)
            rewritten_turns[turn_index] = {
                "user_query": recovery_query,
                "expected_tools": list(pending["expected_tools"]),
            }
            recovery_blueprint = DialogBlueprint(
                overall_task=blueprint.overall_task,
                num_turns=blueprint.num_turns,
                turns=rewritten_turns,
            )
            result = super()._generate_turn_query(
                recovery_blueprint,
                conversation,
                turn_index,
            )
            if result is None:
                return None
            result.quality_preflight["clarification_recovery"] = (
                recovery_certificate
            )
            event_index = int(pending["event_index"])
            self._clarification_events[event_index].update(
                {
                    "recovery_turn": turn_index + 1,
                    "recovery_query": recovery_query,
                    "recovery_expected_tools": list(pending["expected_tools"]),
                    "recovery_certificate": recovery_certificate,
                    "recovered": True,
                }
            )
            self._pending_clarification_recovery = None
            return result

        parallel_eligible = (
            is_final_turn
            and 2 <= self.num_actions <= self.feature_config.max_parallel_width
        )
        if schedule == "combined" and is_final_turn:
            required_mode = "parallel"
        elif schedule == "interactive-refusal":
            required_mode = None
        else:
            required_mode = (
                self._required_feature_mode(parallel_eligible=parallel_eligible)
                if is_final_turn
                else None
            )
        if required_mode == "unavailable":
            print("  Required final-turn feature is not eligible")
            return None

        interactive_refusal = (
            interactive_index is not None and turn_index == interactive_index
        )
        try_refusal = (
            interactive_refusal
            or (
                is_final_turn
                and schedule == "terminal"
                and (
                    required_mode == "refusal"
                    or (
                        required_mode is None
                        and self.feature_config.allow_refusal
                        and random.random() < self.feature_config.refusal_rate
                    )
                )
            )
        )
        if try_refusal:
            for _ in range(self.max_turn_attempts):
                if interactive_refusal:
                    reason = (
                        self.feature_config.forced_refusal_reason
                        or random.choice(["missing_argument", "ambiguity"])
                    )
                else:
                    reason = self._sample_refusal_reason()
                result = self._generate_refusal_query(
                    focus_category=None,
                    refusal_type=reason,
                    policy_history=history,
                    original_query=original_query,
                    source_expected_tools=list(
                        turn_spec.get("expected_tools", [])
                    ),
                )
                if result is not None:
                    if interactive_refusal:
                        event = {
                            "refusal_turn": turn_index + 1,
                            "recovery_turn": turn_index + 2,
                            "reason": reason,
                            "fully_specified_source_query": original_query,
                            "blocked_query": result.query,
                            "assistant_clarification": (
                                result.native_response or ""
                            ),
                            "recovered": False,
                        }
                        self._clarification_events.append(event)
                        self._pending_clarification_recovery = {
                            "recovery_index": turn_index + 1,
                            "event_index": len(self._clarification_events) - 1,
                            "reason": reason,
                            "source_query": original_query,
                            "expected_tools": list(
                                turn_spec.get("expected_tools", [])
                            ),
                            "refusal_query": result.query,
                            "assistant_response": (
                                result.native_response or ""
                            ),
                        }
                    return result
            if required_mode == "refusal" or interactive_refusal:
                print("  Required refusal turn failed; resampling conversation")
                return None
            print("  Refusal turn candidate failed; keeping the original blueprint turn")

        try_parallel = (
            is_final_turn
            and parallel_eligible
            and (
                required_mode == "parallel"
                or (
                    schedule == "terminal"
                    and
                    required_mode is None
                    and self.feature_config.allow_parallel
                    and random.random() < self.feature_config.parallel_rate
                )
            )
        )
        if try_parallel:
            categories = {
                self.tool_manager.get_tool_category(name)
                for name in turn_spec.get("expected_tools", [])
                if self.tool_manager.tool_exists(name)
            }
            categories.discard(None)
            focus_category = next(iter(categories)) if len(categories) == 1 else None
            current_state = (
                self.tool_manager.get_api_state()
                if self._python_tools_available
                else None
            )
            for _ in range(self.max_turn_attempts):
                result = self._generate_parallel_query(
                    focus_category=focus_category,
                    num_calls=self.num_actions,
                    initial_api_state=current_state,
                    policy_history=history,
                    original_query=original_query,
                    max_retries=1,
                )
                if result is not None:
                    return result
            if required_mode == "parallel":
                print("  Required parallel turn failed; resampling conversation")
                return None
            print("  Parallel turn candidate failed; keeping the original blueprint turn")

        if required_mode is not None:
            return None
        return super()._generate_turn_query(blueprint, conversation, turn_index)

    def _annotate_feature_metadata(
        self,
        datapoint: Optional[MultiTurnDatapoint],
    ) -> Optional[MultiTurnDatapoint]:
        if datapoint is None:
            return None

        contains_parallel = any(
            len(step.tool_calls) > 1
            for turn in datapoint.conversation.turns
            for step in turn.steps
        )
        refusal_calls = [
            (turn.turn_number, call.arguments.get("reason"))
            for turn in datapoint.conversation.turns
            for step in turn.steps
            for call in step.tool_calls
            if call.tool_name == "refuse"
        ]
        refusal_turns = [turn_number for turn_number, _ in refusal_calls]
        clarification_turns = [
            turn_number
            for turn_number, reason in refusal_calls
            if reason in {"missing_argument", "ambiguity"}
        ]
        final_turn_number = len(datapoint.conversation.turns)
        final_refusal = final_turn_number in refusal_turns
        final_refusal_reason = next(
            (
                reason
                for turn_number, reason in refusal_calls
                if turn_number == final_turn_number
            ),
            None,
        )
        final_parallel = any(
            len(step.tool_calls) > 1
            for turn in datapoint.conversation.turns
            if turn.turn_number == final_turn_number
            for step in turn.steps
        )
        feature_naturalizations: List[Dict[str, Any]] = []
        for turn in datapoint.conversation.turns:
            preflight = turn.quality_verification.get("query_preflight", {})
            for certificate_key in (
                "refusal_certificate",
                "parallel_certificate",
            ):
                certificate = preflight.get(certificate_key, {})
                naturalization = certificate.get("query_naturalization")
                if isinstance(naturalization, dict):
                    feature_naturalizations.append(
                        {
                            "turn": turn.turn_number,
                            "mode": preflight.get("mode"),
                            **copy.deepcopy(naturalization),
                        }
                    )
        schedule = self.feature_config.multi_turn_feature_schedule
        all_recovered = bool(self._clarification_events) and all(
            event.get("recovered") is True
            for event in self._clarification_events
        )
        if schedule == "interactive-refusal" and (
            not refusal_turns or not all_recovered
        ):
            print("✗ Interactive-refusal schedule was not fully realized")
            return None
        if schedule == "combined" and (
            not refusal_turns or not contains_parallel or not all_recovered
        ):
            print("✗ Combined schedule was not fully realized")
            return None
        datapoint.generation_metadata.update(
            {
                "contains_parallel": contains_parallel,
                "contains_refusal": bool(refusal_turns),
                "refusal_turns": refusal_turns,
                "clarification_turns": clarification_turns,
                "terminal_mode": (
                    "clarification"
                    if final_refusal_reason in {
                        "missing_argument",
                        "ambiguity",
                    }
                    else "refusal"
                    if final_refusal
                    else "parallel"
                    if final_parallel
                    else "tool_calls"
                ),
                "terminal_action": "refuse" if final_refusal else None,
                "feature_difficulty": (
                    self.feature_config.feature_difficulty
                ),
                "feature_schedule": schedule,
                "clarification_events": copy.deepcopy(
                    self._clarification_events
                ),
                "clarification_recovered": (
                    all_recovered if self._clarification_events else None
                ),
                "query_naturalization": copy.deepcopy(
                    self._last_blueprint_naturalization
                ),
                "feature_query_naturalizations": feature_naturalizations,
                "all_feature_queries_naturalized": (
                    all(
                        item.get("enabled") is True
                        and item.get("rewritten") is True
                        for item in feature_naturalizations
                    )
                    if feature_naturalizations
                    else None
                ),
            }
        )
        return prepare_multiturn_datapoint(datapoint)

    def generate_multi_turn_datapoint(
        self, *args: Any, **kwargs: Any
    ) -> Optional[MultiTurnDatapoint]:
        # The base generator's checkpoint dictionary contains mutable nested
        # structures.  Deep-copy before handing it to an external callback so a
        # later turn cannot retroactively mutate a saved checkpoint.
        callback = kwargs.get("checkpoint_callback")
        if callback is not None:
            def safe_callback(state: Dict[str, Any]) -> None:
                callback(copy.deepcopy(state))

            kwargs["checkpoint_callback"] = safe_callback
        return self._annotate_feature_metadata(
            super().generate_multi_turn_datapoint(*args, **kwargs)
        )

    def continue_from_checkpoint(self, *args: Any, **kwargs: Any) -> Optional[MultiTurnDatapoint]:
        return self._annotate_feature_metadata(
            super().continue_from_checkpoint(*args, **kwargs)
        )
