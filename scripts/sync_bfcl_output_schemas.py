#!/usr/bin/env python3
"""Synchronize top-level BFCL output schemas with real local implementations.

The historical output metadata was partly LLM-generated and contains obvious
shape errors (for example, arrays labeled as strings).  This script executes
all 129 methods in deterministic success-oriented fixtures and merges observed
top-level keys/types into the tool pool.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASS_NAMES = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "PostingAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelBooking",
    "vehicle_control": "VehicleControl",
}

CONFIGS = {
    "gorilla_file_system": {
        "root": {
            "a.txt": {"type": "file", "content": "beta\nalpha"},
            "b.txt": {"type": "file", "content": "beta\ngamma"},
            "empty": {"type": "directory", "contents": {}},
            "dir": {"type": "directory", "contents": {"nested.txt": {"type": "file", "content": "needle"}}},
        }
    },
    "math_api": {},
    "message_api": {
        "workspace_id": "WS1",
        "user_map": {"Alice": "USR001", "Bob": "USR002"},
        "messages_sent_map": {"USR001": {"USR002": [{"message_id": 1, "message": "hello"}]}, "USR002": {}},
        "messages_inbox_map": {"USR001": {}, "USR002": {"USR001": [{"message_id": 1, "message": "hello"}]}},
        "message_count": 1,
        "current_user": "USR001",
    },
    "posting_api": {
        "authenticated": True,
        "username": "alice",
        "password": "secret",
        "tweet_counter": 2,
        "tweets": {"1": {"id": 1, "username": "alice", "content": "hello world", "tags": ["#hello"], "mentions": []}},
        "comments": {"1": []},
        "retweets": [],
        "following_list": ["bob"],
        "users": {"bob": {"tweet_count": 1, "following_count": 1, "retweet_count": 0}},
    },
    "ticket_api": {
        "authenticated": True,
        "current_user": "agent",
        "username": "agent",
        "password": "secret",
        "ticket_counter": 1,
        "ticket_queue": [{"id": 1, "title": "Issue", "description": "Desc", "status": "Open", "priority": 2, "created_by": "agent"}],
    },
    "trading_bot": {
        "authenticated": True,
        "current_user": "trader",
        "username": "trader",
        "password": "secret",
        "account_info": {"account_id": 7, "balance": 10000.0, "binding_card": 1234},
        "market_status": "Open",
        "current_time": "2024-11-05 10:00:00",
        "order_counter": 1,
        "orders": {"1": {"id": 1, "order_type": "Buy", "symbol": "AAPL", "price": 100.0, "amount": 1, "status": "Open"}},
        "transaction_history": [{"order_id": 1, "symbol": "AAPL", "price": 100.0, "num_shares": 1, "status": "Filled", "timestamp": "2024-11-05 10:00:00"}],
        "stocks": {"AAPL": {"price": 100.0, "percent_change": 1.0, "volume": 10.0, "MA(5)": 99.0, "MA(20)": 98.0}},
        "watch_list": ["AAPL"],
    },
    "travel_booking": {
        "access_token": "token",
        "token_type": "Bearer",
        "token_expires_in": 3600,
        "token_scope": "read_write",
        "client_id": "client",
        "client_secret": "secret",
        "refresh_token": "refresh",
        "user_first_name": "Alice",
        "user_last_name": "Smith",
        "budget_limit": 1000.0,
        "credit_card_list": {"card1": {"card_number": "4111111111111111", "expiration_date": "12/30", "cardholder_name": "Alice Smith", "card_verification_number": 123, "balance": 10000.0}},
        "booking_record": {"flight_001": {"travel_to": "LAX", "travel_from": "SFO", "insurance": "none", "travel_cost": 100.0, "travel_date": "2026-12-15", "travel_class": "economy", "transaction_id": "txn", "card_id": "card1"}},
    },
    "vehicle_control": {
        "engineState": "running",
        "fuelLevel": 20.0,
        "batteryVoltage": 12.8,
        "currentSpeed": 40.0,
        "destination": "Denver",
        "outsideTemperatureC": 20.0,
        "frontLeftTirePressure": 32.0,
        "frontRightTirePressure": 32.0,
        "rearLeftTirePressure": 32.0,
        "rearRightTirePressure": 32.0,
        "brakePedalStatus": "pressed",
        "brakePedalForce": 100.0,
    },
}

ARGS = {
    "cat": {"file_name": "a.txt"}, "cd": {"folder": "dir"}, "cp": {"source": "a.txt", "destination": "copy.txt"},
    "diff": {"file_name1": "a.txt", "file_name2": "b.txt"}, "du": {"human_readable": False},
    "echo": {"content": "hello", "file_name": "out.txt"}, "find": {"path": ".", "name": "txt"},
    "grep": {"file_name": "a.txt", "pattern": "alpha"}, "ls": {"a": True}, "mkdir": {"dir_name": "newdir"},
    "mv": {"source": "a.txt", "destination": "moved.txt"}, "rm": {"file_name": "a.txt"}, "rmdir": {"dir_name": "empty"},
    "sort": {"file_name": "a.txt"}, "tail": {"file_name": "a.txt", "lines": 1}, "touch": {"file_name": "new.txt"},
    "wc": {"file_name": "a.txt", "mode": "l"},
    "absolute_value": {"number": -2.0}, "add": {"a": 1.0, "b": 2.0}, "divide": {"a": 4.0, "b": 2.0},
    "imperial_si_conversion": {"value": 1.0, "unit_in": "foot", "unit_out": "meter"},
    "logarithm": {"value": 10.0, "base": 10.0, "precision": 2}, "max_value": {"numbers": [1.0, 2.0]},
    "mean": {"numbers": [1.0, 2.0]}, "min_value": {"numbers": [1.0, 2.0]}, "multiply": {"a": 2.0, "b": 3.0},
    "percentage": {"part": 1.0, "whole": 2.0}, "power": {"base": 2.0, "exponent": 3.0},
    "round_number": {"number": 1.234, "decimal_places": 2}, "si_unit_conversion": {"value": 1000.0, "unit_in": "meter", "unit_out": "kilometer"},
    "square_root": {"number": 4.0, "precision": 2}, "standard_deviation": {"numbers": [1.0, 2.0]},
    "subtract": {"a": 3.0, "b": 1.0}, "sum_values": {"numbers": [1.0, 2.0]},
    "add_contact": {"user_name": "Carol"}, "delete_message": {"receiver_id": "USR002", "message_id": 1},
    "get_user_id": {"user": "Alice"}, "message_login": {"user_id": "USR001"}, "search_messages": {"keyword": "hello"},
    "send_message": {"receiver_id": "USR002", "message": "hello"},
    "authenticate_twitter": {"username": "alice", "password": "secret"}, "comment": {"tweet_id": 1, "comment_content": "nice"},
    "follow_user": {"username_to_follow": "carol"}, "get_tweet": {"tweet_id": 1}, "get_tweet_comments": {"tweet_id": 1},
    "get_user_stats": {"username": "alice"}, "get_user_tweets": {"username": "alice"}, "mention": {"tweet_id": 1, "mentioned_usernames": ["bob"]},
    "post_tweet": {"content": "new", "tags": ["tag"], "mentions": ["bob"]}, "retweet": {"tweet_id": 1},
    "search_tweets": {"keyword": "hello"}, "unfollow_user": {"username_to_unfollow": "bob"},
    "close_ticket": {"ticket_id": 1}, "create_ticket": {"title": "New", "description": "Desc", "priority": 2},
    "edit_ticket": {"ticket_id": 1, "updates": {"title": "Updated"}}, "get_ticket": {"ticket_id": 1},
    "get_user_tickets": {"status": "Open"}, "resolve_ticket": {"ticket_id": 1, "resolution": "Fixed"},
    "ticket_login": {"username": "agent", "password": "secret"},
    "add_to_watchlist": {"stock": "MSFT"}, "cancel_order": {"order_id": 1}, "filter_stocks_by_price": {"stocks": ["AAPL"], "min_price": 50.0, "max_price": 150.0},
    "fund_account": {"amount": 10.0}, "get_available_stocks": {"sector": "Technology"}, "get_order_details": {"order_id": 1},
    "get_stock_info": {"symbol": "AAPL"}, "get_symbol_by_name": {"name": "apple"}, "get_transaction_history": {"start_date": "2024-11-01", "end_date": "2024-11-30"},
    "make_transaction": {"account_id": 7, "xact_type": "deposit", "amount": 10.0}, "notify_price_change": {"stocks": ["AAPL"], "threshold": 0.5},
    "place_order": {"order_type": "Buy", "symbol": "AAPL", "price": 100.0, "amount": 1}, "remove_stock_from_watchlist": {"symbol": "AAPL"},
    "trading_login": {"username": "trader", "password": "secret"}, "update_market_status": {"current_time_str": "10:00 AM"},
    "update_stock_price": {"symbol": "AAPL", "new_price": 101.0},
    "authenticate_travel": {"client_id": "client", "client_secret": "secret", "refresh_token": "refresh", "grant_type": "read_write", "user_first_name": "Alice", "user_last_name": "Smith"},
    "book_flight": {"access_token": "token", "card_id": "card1", "travel_date": "2026-12-15", "travel_from": "SFO", "travel_to": "LAX", "travel_class": "economy", "travel_cost": 100.0},
    "cancel_booking": {"access_token": "token", "booking_id": "flight_001"}, "compute_exchange_rate": {"base_currency": "USD", "target_currency": "EUR", "value": 100.0},
    "contact_customer_support": {"booking_id": "flight_001", "message": "help"}, "get_budget_fiscal_year": {},
    "get_credit_card_balance": {"access_token": "token", "card_id": "card1"}, "get_flight_cost": {"travel_from": "SFO", "travel_to": "LAX", "travel_date": "2026-12-15", "travel_class": "economy"},
    "get_nearest_airport_by_city": {"location": "San Francisco"}, "purchase_insurance": {"access_token": "token", "insurance_type": "basic", "insurance_cost": 10.0, "booking_id": "flight_001", "card_id": "card1"},
    "register_credit_card": {"access_token": "token", "card_number": "5555555555554444", "expiration_date": "12/30", "cardholder_name": "Alice Smith", "card_verification_number": 123},
    "retrieve_invoice": {"access_token": "token", "booking_id": "flight_001", "insurance_id": "None"}, "set_budget_limit": {"access_token": "token", "budget_limit": 500.0},
    "verify_traveler_information": {"first_name": "Alice", "last_name": "Smith", "date_of_birth": "1990-01-01", "passport_number": "P123"},
    "activateParkingBrake": {"mode": "engage"}, "adjustClimateControl": {"temperature": 21.0, "unit": "celsius", "fanSpeed": 50, "mode": "auto"},
    "displayCarStatus": {"option": "fuel"}, "display_log": {"messages": ["hello"]}, "estimate_distance": {"cityA": "10001", "cityB": "20001"},
    "estimate_drive_feasibility_by_mileage": {"distance": 100.0}, "fillFuelTank": {"fuelAmount": 1.0}, "gallon_to_liter": {"gallon": 1.0},
    "get_zipcode_based_on_city": {"city": "Dallas"}, "liter_to_gallon": {"liter": 3.78541}, "lockDoors": {"unlock": False, "door": ["driver"]},
    "pressBrakePedal": {"pedalPosition": 0.5}, "setCruiseControl": {"speed": 60.0, "activate": True, "distanceToNextVehicle": 100.0},
    "setHeadlights": {"mode": "on"}, "set_navigation": {"destination": "Denver"}, "startEngine": {"ignitionMode": "START"},
}


def json_type(value: Any) -> str:
    if isinstance(value, bool): return "boolean"
    if isinstance(value, int): return "integer"
    if isinstance(value, float): return "number"
    if isinstance(value, str): return "string"
    if isinstance(value, list): return "array"
    if isinstance(value, dict): return "object"
    if value is None: return "string"
    return "string"


def main() -> None:
    pool_path = ROOT / "magnet_tool_extraction" / "bfcl_v3_tools_with_outputs.jsonl"
    rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line.strip()]
    for row in rows:
        module = importlib.import_module(f"tools.{row['tool_name']}")
        cls = getattr(module, CLASS_NAMES[row["tool_name"]])
        instance = cls(CONFIGS[row["tool_name"]])
        output = getattr(instance, row["api_name"])(**ARGS.get(row["api_name"], {}))
        if not isinstance(output, dict):
            raise RuntimeError(f"{row['api_name']} returned {type(output).__name__}, expected dict")
        schema = row.setdefault("output_schema", {"type": "object", "properties": {}})
        schema["type"] = "object"
        properties = schema.setdefault("properties", {})
        for key, value in output.items():
            previous = properties.get(key, {})
            properties[key] = {
                "type": json_type(value),
                "description": previous.get("description", f"Runtime output field: {key}"),
            }
    pool_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Synchronized output schemas for {len(rows)} tools")


if __name__ == "__main__":
    main()
