"""Deterministic admission checks for positive RL tool trajectories.

These checks intentionally cover properties that should not be delegated to an
LLM judge: read calls cannot mutate state, requested mutations must change
state, and creates cannot overwrite an existing entity.  The module does not
inspect hidden state to construct arguments; it only certifies simulator
transitions after execution.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# High-confidence state-changing tools.  A no-op call is not useful as a
# positive mutation example unless the task explicitly models idempotence, which
# this generator currently does not.
MUTATING_TOOLS = frozenset(
    {
        "authenticate_twitter",
        "follow_user",
        "unfollow_user",
        "comment",
        "retweet",
        "mention",
        "post_tweet",
        "trading_login",
        "add_to_watchlist",
        "remove_stock_from_watchlist",
        "place_order",
        "fund_account",
        "cancel_order",
        "authenticate_travel",
        "book_flight",
        "cancel_booking",
        "purchase_insurance",
        "message_login",
        "add_contact",
        "send_message",
        "delete_message",
        "ticket_login",
        "create_ticket",
        "edit_ticket",
        "resolve_ticket",
        "close_ticket",
        "mkdir",
        "touch",
        "echo",
        "mv",
        "rm",
        "rmdir",
        "cp",
        "cd",
        "make_transaction",
        "update_stock_price",
        "register_credit_card",
        "set_budget_limit",
        "fillFuelTank",
        "startEngine",
        "set_navigation",
        "setCruiseControl",
        "setHeadlights",
        "lockDoors",
        "pressBrakePedal",
        "activateParkingBrake",
        "adjustClimateControl",
    }
)


# Creation rules let us prove that a returned identifier was not already in the
# target collection and that the new entity was actually persisted.
CREATE_RULES: Dict[str, Dict[str, Any]] = {
    "post_tweet": {
        "id_field": "id",
        "collection_path": ("posting_api", "tweets"),
    },
    "place_order": {
        "id_field": "order_id",
        "collection_path": ("trading_bot", "orders"),
    },
    "book_flight": {
        "id_field": "booking_id",
        "collection_path": ("travel_booking", "booking_record"),
    },
    "create_ticket": {
        "id_field": "id",
        "collection_path": ("ticket_api", "ticket_queue"),
    },
    "add_contact": {
        "id_field": "user_id",
        "collection_path": ("message_api", "user_map"),
        "ids_are_values": True,
    },
    "register_credit_card": {
        "id_field": "card_id",
        "collection_path": ("travel_booking", "credit_card_list"),
    },
}


def _get_path(root: Any, path: Sequence[str]) -> Any:
    current = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _normalise_identifier(value: Any) -> Optional[str]:
    if value is None or value is False:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        # Zero is a valid first identifier for several synthetic APIs. Failure
        # is established by state persistence/no-op checks, not by the value.
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)
    return str(value)


def _record_identifier(record: Any) -> Optional[str]:
    if not isinstance(record, Mapping):
        return None
    for key in ("id", "order_id", "booking_id", "ticket_id", "user_id"):
        if key in record:
            return _normalise_identifier(record[key])
    return None


def _collection_contains_identifier(
    collection: Any,
    identifier: str,
    *,
    ids_are_values: bool = False,
) -> bool:
    if isinstance(collection, Mapping):
        if identifier in {str(key) for key in collection.keys()}:
            return True
        if ids_are_values and identifier in {
            str(value) for value in collection.values() if not isinstance(value, (dict, list))
        }:
            return True
        return any(_record_identifier(value) == identifier for value in collection.values())

    if isinstance(collection, list):
        return any(_record_identifier(item) == identifier for item in collection)

    return False


def _collection_size(collection: Any) -> Optional[int]:
    if isinstance(collection, (Mapping, list, tuple, set)):
        return len(collection)
    return None


def validate_transition_quality(
    *,
    tool_name: str,
    tool_output: Any,
    pre_state: Optional[Dict[str, Any]],
    post_state: Optional[Dict[str, Any]],
    tool_arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a deterministic quality certificate for one tool transition."""

    issues = []
    arguments = tool_arguments or {}
    is_mutating = tool_name in MUTATING_TOOLS
    # echo without a file target is a terminal/display operation, not a state
    # mutation. With file_name it must change the filesystem state.
    if tool_name == "echo" and not arguments.get("file_name"):
        is_mutating = False

    if is_mutating:
        if pre_state is None or post_state is None:
            issues.append(
                {
                    "code": "STATE_SNAPSHOT_MISSING",
                    "message": "A mutating tool lacks pre/post state snapshots.",
                }
            )
        elif pre_state == post_state:
            issues.append(
                {
                    "code": "MUTATION_NO_EFFECT",
                    "message": "The requested mutating tool produced no state change.",
                }
            )
    elif (
        pre_state is not None
        and post_state is not None
        and pre_state != post_state
    ):
        issues.append(
            {
                "code": "READ_ONLY_MUTATED_STATE",
                "message": "A read-only tool unexpectedly changed simulator state.",
            }
        )

    rule = CREATE_RULES.get(tool_name)
    if rule:
        output = tool_output if isinstance(tool_output, Mapping) else {}
        identifier = _normalise_identifier(output.get(rule["id_field"]))
        before = _get_path(pre_state or {}, rule["collection_path"])
        after = _get_path(post_state or {}, rule["collection_path"])
        ids_are_values = bool(rule.get("ids_are_values", False))

        if identifier is None:
            issues.append(
                {
                    "code": "CREATED_ID_MISSING",
                    "message": f"Create tool did not return a usable {rule['id_field']}.",
                }
            )
        else:
            if _collection_contains_identifier(
                before,
                identifier,
                ids_are_values=ids_are_values,
            ):
                issues.append(
                    {
                        "code": "CREATED_ID_COLLISION",
                        "message": "The create tool reused an identifier that existed before the call.",
                    }
                )
            if not _collection_contains_identifier(
                after,
                identifier,
                ids_are_values=ids_are_values,
            ):
                issues.append(
                    {
                        "code": "CREATED_ID_NOT_PERSISTED",
                        "message": "The returned identifier is absent from post-state.",
                    }
                )

        before_size = _collection_size(before)
        after_size = _collection_size(after)
        if before_size is not None and after_size is not None and after_size <= before_size:
            issues.append(
                {
                    "code": "CREATE_COLLECTION_DID_NOT_GROW",
                    "message": "A create operation did not increase the target collection size.",
                }
            )

    return {
        "tool_name": tool_name,
        "passed": not issues,
        "issues": issues,
    }
