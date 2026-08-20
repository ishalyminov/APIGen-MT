#!/usr/bin/env python3
"""Build the executable 129-tool BFCL V3 multi-turn pool.

The repository historically used a 105-tool output-enriched subset.  This
script merges that metadata with the complete 129-tool multi-turn definition
file and supplies deterministic output contracts for the previously omitted or
schema-less functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def field(
    type_name: str,
    description: str,
    **schema_details: Any,
) -> Dict[str, Any]:
    return {
        "type": type_name,
        "description": description,
        **schema_details,
    }


def obj(**properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties}


OUTPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "diff": obj(diff=field("array", "Line-by-line differences."), identical=field("boolean", "Whether the files are identical."), error=field("string", "Execution error, when present.")),
    "sort": obj(lines=field("array", "Sorted lines."), count=field("integer", "Number of sorted lines."), error=field("string", "Execution error, when present.")),
    "pwd": obj(path=field("string", "Virtual current directory path."), current_directory=field("string", "Virtual current directory path.")),
    "get_message_stats": obj(logged_in=field("boolean", "Whether a message user is logged in."), current_user=field("string", "Current user ID."), sent_count=field("integer", "Sent message count."), received_count=field("integer", "Received message count."), total_count=field("integer", "Total message count."), conversation_count=field("integer", "Distinct conversation count.")),
    "list_users": obj(workspace_id=field("string", "Workspace ID."), user_count=field("integer", "Number of users."), users=field("array", "Users with names and user IDs.")),
    "message_get_login_status": obj(logged_in=field("boolean", "Whether a user is logged in."), current_user=field("string", "Current user ID.")),
    "view_messages_sent": obj(logged_in=field("boolean", "Whether a user is logged in."), current_user=field("string", "Current user ID."), message_count=field("integer", "Sent message count."), messages=field("array", "Messages sent by the current user.")),
    "search_messages": obj(results=field("array", "Matching sent and received messages.", items=obj(message=field("string", "Message text."), direction=field("string", "Whether the message was sent or received.")))),
    "list_all_following": obj(authenticated=field("boolean", "Whether the posting user is authenticated."), following_count=field("integer", "Number of followed users."), following=field("array", "Followed usernames.")),
    "posting_get_login_status": obj(logged_in=field("boolean", "Whether the posting user is logged in."), username=field("string", "Authenticated username.")),
    "logout": obj(success=field("boolean", "Whether a ticket user was logged out."), username=field("string", "Logged-out username.")),
    "ticket_get_login_status": obj(logged_in=field("boolean", "Whether a ticket user is logged in."), username=field("string", "Authenticated username.")),
    "get_account_info": obj(authenticated=field("boolean", "Whether the trading user is authenticated."), account_id=field("integer", "Trading account ID."), balance=field("number", "Account balance."), binding_card=field("integer", "Bound card number.")),
    "get_current_time": obj(current_time=field("string", "Deterministic simulator timestamp."), market_status=field("string", "Current market status.")),
    "get_order_history": obj(authenticated=field("boolean", "Whether the trading user is authenticated."), order_ids=field("array", "Order IDs."), orders=field("array", "Order details.")),
    "get_watchlist": obj(authenticated=field("boolean", "Whether the trading user is authenticated."), count=field("integer", "Watchlist size."), watchlist=field("array", "Stock symbols in the watchlist.")),
    "trading_get_login_status": obj(logged_in=field("boolean", "Whether the trading user is logged in."), username=field("string", "Authenticated username.")),
    "trading_logout": obj(success=field("boolean", "Whether a trading user was logged out."), username=field("string", "Logged-out username.")),
    "get_all_credit_cards": obj(logged_in=field("boolean", "Whether a travel session is active."), count=field("integer", "Number of registered cards."), cards=field("array", "Masked registered-card details.", items=obj(card_id=field("string", "Registered card ID."), card_number_masked=field("string", "Masked card number."), expiration_date=field("string", "Card expiration date."), cardholder_name=field("string", "Cardholder name."), balance=field("number", "Available card balance.")))),
    "list_all_airports": obj(count=field("integer", "Number of airports."), airports=field("array", "City and airport-code pairs.")),
    "travel_get_login_status": obj(logged_in=field("boolean", "Whether a travel session is active."), user_first_name=field("string", "Traveler first name."), user_last_name=field("string", "Traveler last name."), token_type=field("string", "Token type."), scope=field("string", "Token scope."), expires_in=field("integer", "Token lifetime in seconds.")),
    "activateParkingBrake": obj(parkingBrakeStatus=field("string", "Parking-brake status."), _parkingBrakeForce=field("number", "Parking-brake force."), _slopeAngle=field("number", "Road slope angle.")),
    "adjustClimateControl": obj(currentTemperature=field("number", "Current cabin temperature in Celsius."), climateMode=field("string", "Climate mode."), humidityLevel=field("number", "Cabin humidity.")),
    "displayCarStatus": obj(status=field("object", "Requested vehicle-status fields.")),
    "display_log": obj(log=field("array", "Log messages.")),
    "estimate_distance": obj(distance=field("number", "Estimated distance."), unit=field("string", "Distance unit."), intermediaryCities=field("array", "Intermediate cities.")),
    "estimate_drive_feasibility_by_mileage": obj(canDrive=field("boolean", "Whether the available fuel can cover the distance.")),
    "fillFuelTank": obj(fuelLevel=field("number", "Fuel level after filling.")),
    "gallon_to_liter": obj(liter=field("number", "Converted liters.")),
    "get_zipcode_based_on_city": obj(zipcode=field("string", "Deterministic synthetic ZIP code.")),
    "liter_to_gallon": obj(gallon=field("number", "Converted gallons.")),
    "lockDoors": obj(lockStatus=field("string", "Requested lock state."), remainingUnlockedDoors=field("integer", "Number of unlocked doors.")),
    "pressBrakePedal": obj(brakePedalStatus=field("string", "Brake-pedal status."), brakePedalForce=field("number", "Applied brake force.")),
    "setCruiseControl": obj(cruiseStatus=field("string", "Cruise-control status."), currentSpeed=field("number", "Current cruise speed."), distanceToNextVehicle=field("number", "Distance to next vehicle.")),
    "setHeadlights": obj(headlightStatus=field("string", "Headlight status.")),
    "set_navigation": obj(status=field("string", "Navigation status.")),
    "startEngine": obj(engineState=field("string", "Engine state."), fuelLevel=field("number", "Fuel level."), batteryVoltage=field("number", "Battery voltage.")),
    "check_tire_pressure": obj(tire_pressure=field("object", "Pressure for each tire."), unit=field("string", "Pressure unit."), status=field("string", "Overall pressure status."), low_tires=field("array", "Tires below the safe range."), high_tires=field("array", "Tires above the safe range.")),
    "find_nearest_tire_shop": obj(shop_name=field("string", "Nearest shop name."), address=field("string", "Shop address."), distance=field("number", "Distance to the shop."), unit=field("string", "Distance unit.")),
    "get_current_speed": obj(current_speed=field("number", "Current vehicle speed."), unit=field("string", "Speed unit.")),
    "get_outside_temperature_from_google": obj(temperature=field("number", "Outside temperature."), unit=field("string", "Temperature unit."), source=field("string", "Weather provider.")),
    "get_outside_temperature_from_weather_com": obj(temperature=field("number", "Outside temperature."), unit=field("string", "Temperature unit."), source=field("string", "Weather provider.")),
    "releaseBrakePedal": obj(brakePedalStatus=field("string", "Brake-pedal status."), brakePedalForce=field("number", "Applied brake force.")),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definitions", default="magnet_tool_extraction/bfcl_v3_tool_definitions.jsonl")
    parser.add_argument("--existing", default="magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl")
    parser.add_argument("--output", default="magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl")
    args = parser.parse_args()

    definitions = load_jsonl(Path(args.definitions))
    existing_rows = load_jsonl(Path(args.existing))
    existing = {row["api_name"]: row for row in existing_rows}

    complete = []
    for definition in definitions:
        api_name = definition["api_name"]
        row = dict(definition)
        old = existing.get(api_name, {})
        row["output_type"] = old.get("output_type", "dict")
        row["output_description"] = old.get(
            "output_description",
            f"Structured deterministic result returned by {api_name}.",
        )
        # Explicit schemas describe the deterministic implementations in this
        # repository and must win on subsequent idempotent rebuilds.  Otherwise
        # an older shallow array schema is copied forever and teachers bind an
        # object (``cards.0``) where a scalar ID (``cards.0.card_id``) is needed.
        schema = OUTPUT_SCHEMAS.get(api_name) or old.get("output_schema")
        if schema is None:
            raise RuntimeError(f"No output schema available for {api_name}")
        row["output_schema"] = schema
        complete.append(row)

    names = [row["api_name"] for row in complete]
    if len(names) != 129 or len(set(names)) != 129:
        raise RuntimeError(f"Expected 129 unique tools, got {len(names)} rows / {len(set(names))} unique")
    if any("output_schema" not in row for row in complete):
        raise RuntimeError("Every output row must have an output_schema")

    output = Path(args.output)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in complete),
        encoding="utf-8",
    )
    print(f"Wrote {len(complete)} executable BFCL V3 multi-turn tools to {output}")


if __name__ == "__main__":
    main()
