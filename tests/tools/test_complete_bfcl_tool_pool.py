from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tools.gorilla_file_system import GorillaFileSystem
from tools.message_api import MessageAPI
from tools.posting_api import PostingAPI
from tools.ticket_api import TicketAPI
from tools.trading_bot import TradingBot
from tools.travel_booking import TravelBooking
from tools.vehicle_control import VehicleControl


ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = ROOT / "magnet_tool_extraction" / "bfcl_v3_tool_definitions.jsonl"
POOL = ROOT / "magnet_tool_extraction" / "bfcl_v3_tools_with_outputs.jsonl"

CLASS_BY_KEY = {
    "gorilla_file_system": GorillaFileSystem,
    "message_api": MessageAPI,
    "posting_api": PostingAPI,
    "ticket_api": TicketAPI,
    "trading_bot": TradingBot,
    "travel_booking": TravelBooking,
    "vehicle_control": VehicleControl,
}


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_complete_pool_contains_all_129_definitions_with_output_schemas():
    definitions = _rows(DEFINITIONS)
    pool = _rows(POOL)
    assert len(definitions) == 129
    assert len(pool) == 129
    assert {row["api_name"] for row in pool} == {row["api_name"] for row in definitions}
    assert all(row.get("output_schema", {}).get("type") == "object" for row in pool)


@pytest.mark.parametrize("definition", _rows(DEFINITIONS), ids=lambda row: row["api_name"])
def test_every_bfcl_multiturn_tool_has_matching_callable_signature(definition):
    tool_key = definition["tool_name"]
    if tool_key == "math_api":
        from tools.math_api import MathAPI
        cls = MathAPI
    else:
        cls = CLASS_BY_KEY[tool_key]
    method = getattr(cls, definition["api_name"], None)
    assert callable(method)

    signature = inspect.signature(method)
    implementation_parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]
    implementation_names = {parameter.name for parameter in implementation_parameters}
    implementation_required = {
        parameter.name
        for parameter in implementation_parameters
        if parameter.default is inspect.Parameter.empty
    }
    schema_names = set(definition["parameters"].get("properties", {}))
    schema_required = set(definition["parameters"].get("required", []))
    assert implementation_names == schema_names
    assert implementation_required == schema_required


def test_new_message_read_tools():
    api = MessageAPI({
        "workspace_id": "WS1",
        "user_map": {"Alice": "USR001", "Bob": "USR002"},
        "messages_sent_map": {"USR001": {"USR002": [{"message_id": 1, "message": "hi"}]}},
        "messages_inbox_map": {"USR001": {}, "USR002": {"USR001": [{"message_id": 1, "message": "hi"}]}},
        "current_user": "USR001",
    })
    assert api.message_get_login_status() == {"logged_in": True, "current_user": "USR001"}
    assert api.get_message_stats()["sent_count"] == 1
    assert api.view_messages_sent()["messages"][0]["receiver_id"] == "USR002"
    assert api.list_users()["user_count"] == 2


def test_new_posting_and_ticket_session_tools():
    posting = PostingAPI({"authenticated": True, "username": "alice", "following_list": ["bob"]})
    assert posting.posting_get_login_status() == {"logged_in": True, "username": "alice"}
    assert posting.list_all_following()["following"] == ["bob"]

    tickets = TicketAPI({"authenticated": True, "current_user": "agent", "username": "agent"})
    assert tickets.ticket_get_login_status()["username"] == "agent"
    assert tickets.logout() == {"success": True, "username": "agent"}
    assert tickets.ticket_get_login_status()["logged_in"] is False


def test_new_trading_read_and_session_tools():
    trading = TradingBot({
        "authenticated": True,
        "current_user": "alice",
        "account_info": {"account_id": 7, "balance": 100.0, "binding_card": 1234},
        "orders": {"3": {"id": 3, "order_type": "Buy", "symbol": "AAPL", "price": 1.0, "amount": 2, "status": "Open"}},
        "watch_list": ["AAPL"],
    })
    assert trading.get_account_info()["account_id"] == 7
    assert trading.get_order_history()["order_ids"] == [3]
    assert trading.get_watchlist()["watchlist"] == ["AAPL"]
    assert trading.get_current_time()["current_time"]
    assert trading.trading_get_login_status()["logged_in"] is True
    assert trading.trading_logout()["success"] is True


def test_new_travel_catalog_and_session_tools():
    travel = TravelBooking({
        "access_token": "token",
        "token_expires_in": 3600,
        "token_scope": "read",
        "user_first_name": "Alice",
        "user_last_name": "Smith",
        "credit_card_list": {"card1": {"card_number": "4111111111111111", "balance": 10.0}},
    })
    assert travel.travel_get_login_status()["logged_in"] is True
    assert travel.get_all_credit_cards()["cards"][0]["card_number_masked"] == "****1111"
    airports = travel.list_all_airports()
    assert airports["count"] >= 80
    assert {"city": "San Francisco", "code": "SFO"} in airports["airports"]


def test_new_vehicle_read_and_brake_tools_are_deterministic():
    vehicle = VehicleControl({
        "engineState": "running",
        "currentSpeed": 42.0,
        "destination": "Denver",
        "outsideTemperatureC": 12.0,
        "frontLeftTirePressure": 29.0,
        "frontRightTirePressure": 32.0,
        "rearLeftTirePressure": 33.0,
        "rearRightTirePressure": 34.0,
        "brakePedalStatus": "pressed",
        "brakePedalForce": 100.0,
    })
    assert vehicle.check_tire_pressure()["low_tires"] == ["front_left"]
    assert vehicle.get_current_speed() == {"current_speed": 42.0, "unit": "mph"}
    assert vehicle.get_outside_temperature_from_google()["temperature"] == 12.0
    assert vehicle.get_outside_temperature_from_weather_com()["temperature"] == 12.4
    assert vehicle.find_nearest_tire_shop() == vehicle.find_nearest_tire_shop()
    assert vehicle.releaseBrakePedal() == {"brakePedalStatus": "released", "brakePedalForce": 0.0}


def test_pwd_returns_virtual_path(tmp_path):
    fs = GorillaFileSystem({"root": {"docs": {"type": "directory", "contents": {}}}})
    assert fs.pwd()["path"] == "/"
    assert fs.cd("docs").get("success") is True
    assert fs.pwd()["path"] == "/docs"


def _smoke_value(name, schema):
    type_name = schema.get("type")
    description = schema.get("description", "")
    special = {
        "b": 2.0,
        "whole": 2.0,
        "base": 2.0,
        "precision": 2,
        "date_of_birth": "1990-01-01",
        "passport_number": "P12345",
        "grant_type": "read_write",
        "ignitionMode": "START",
        "order_type": "Buy",
        "xact_type": "deposit",
        "travel_class": "economy",
        "base_currency": "USD",
        "target_currency": "EUR",
        "insurance_type": "basic",
        "cityA": "10001",
        "cityB": "20001",
        "option": "fuel",
        "door": ["driver"],
        "updates": {"title": "updated"},
    }
    if name in special:
        return special[name]
    if "date" in name:
        return "2026-12-15"
    if name == "unit_in":
        return "meter"
    if name == "unit_out":
        return "kilometer"
    if name == "mode":
        for candidate in ("engage", "auto", "on", "l"):
            if candidate in description:
                return candidate
        return "auto"
    if name == "numbers":
        return [1.0, 2.0]
    if name in {"stocks", "messages", "mentioned_usernames", "tags", "mentions"}:
        return ["AAPL"]
    if type_name in {"string", "str"}:
        return "test"
    if type_name in {"integer", "int"}:
        return 1
    if type_name in {"number", "float"}:
        return 1.0
    if type_name in {"boolean", "bool"}:
        return False
    if type_name in {"array", "list"}:
        return []
    if type_name in {"object", "dict"}:
        return {}
    return "test"


@pytest.mark.parametrize("definition", _rows(DEFINITIONS), ids=lambda row: f"runtime_{row['api_name']}")
def test_every_bfcl_multiturn_tool_is_runtime_callable(definition):
    tool_key = definition["tool_name"]
    if tool_key == "math_api":
        from tools.math_api import MathAPI
        cls = MathAPI
    else:
        cls = CLASS_BY_KEY[tool_key]
    instance = cls({})
    required = set(definition["parameters"].get("required", []))
    arguments = {
        name: _smoke_value(name, schema)
        for name, schema in definition["parameters"].get("properties", {}).items()
        if name in required
    }
    result = getattr(instance, definition["api_name"])(**arguments)
    assert isinstance(result, dict)
