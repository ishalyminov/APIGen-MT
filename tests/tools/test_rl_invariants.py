import json
import os
import subprocess
import sys
from pathlib import Path

from tools.gorilla_file_system import GorillaFileSystem
from tools.message_api import MessageAPI
from tools.posting_api import PostingAPI
from tools.ticket_api import TicketAPI
from tools.trading_bot import TradingBot
from tools.travel_booking import TravelBooking
from tools.vehicle_control import VehicleControl


def test_find_honors_documented_filename_substring_matching():
    api = GorillaFileSystem(
        {
            "paper_draft.txt": {
                "type": "file",
                "content": "draft",
            }
        }
    )
    assert api.find(path=".", name="paper")["files"] == ["paper_draft.txt"]


def test_post_tweet_never_overwrites_stale_counter():
    api = PostingAPI(
        {
            "authenticated": True,
            "tweet_counter": 25,
            "tweets": {"25": {"id": 25, "content": "existing"}},
            "username": "user",
        }
    )
    result = api.post_tweet("new")
    assert result["id"] == 26
    assert api.tweets["25"]["content"] == "existing"
    assert api.tweets["26"]["content"] == "new"


def test_place_order_never_overwrites_and_status_is_consistent():
    bot = TradingBot(
        {
            "authenticated": True,
            "order_counter": 89000,
            "transaction_history": [
                {
                    "order_id": 89001,
                    "symbol": "AAPL",
                    "price": 100.0,
                    "num_shares": 1,
                    "status": "Filled",
                }
            ],
        }
    )
    result = bot.place_order("Buy", "TSLA", 200.0, 2)
    assert result["order_id"] == 89002
    assert bot.orders[89001]["symbol"] == "AAPL"
    assert bot.orders[89002]["status"] == result["status"]
    assert bot.transaction_history[-1]["status"] == result["status"]


def test_ticket_counter_starts_after_existing_ids():
    api = TicketAPI(
        {
            "authenticated": True,
            "ticket_counter": 3,
            "ticket_queue": [{"id": 9, "title": "existing"}],
        }
    )
    result = api.create_ticket("new")
    assert result["id"] == 10


def test_message_counter_starts_after_existing_ids():
    api = MessageAPI(
        {
            "current_user": "USR001",
            "message_count": 1,
            "messages_sent_map": {
                "USR001": {
                    "USR002": [{"message_id": 7, "message": "existing"}]
                }
            },
            "messages_inbox_map": {},
        }
    )
    result = api.send_message("USR002", "new")
    assert result["message_id"] == "8"


def test_booking_id_uses_max_existing_suffix_not_collection_length():
    api = TravelBooking(
        {
            "access_token": "token",
            "credit_card_list": {"1": {"balance": 1000.0}},
            "booking_record": {
                "flight_001": {},
                "flight_010": {},
            },
        }
    )
    result = api.book_flight(
        access_token="token",
        card_id="1",
        travel_date="2099-01-01",
        travel_from="AAA",
        travel_to="BBB",
        travel_class="economy",
        travel_cost=100.0,
    )
    assert result["booking_id"] == "flight_011"


def _vehicle_output_for_seed(seed: str):
    root = Path(__file__).resolve().parents[2]
    code = """
import json
from tools.vehicle_control import VehicleControl
v = VehicleControl({})
print(json.dumps({
    'zip': v.get_zipcode_based_on_city('Lagos'),
    'distance': v.estimate_distance('12345', '900001'),
}))
"""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = f"{root}:{root / 'src'}"
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
    )
    return json.loads(output)


def test_vehicle_outputs_are_process_stable_symmetric_and_unitful():
    first = _vehicle_output_for_seed("1")
    second = _vehicle_output_for_seed("999")
    assert first == second

    vehicle = VehicleControl({})
    forward = vehicle.estimate_distance("12345", "900001")
    reverse = vehicle.estimate_distance("900001", "12345")
    assert forward == reverse
    assert forward["unit"] == "miles"


def test_register_credit_card_does_not_overwrite_same_last_four():
    api = TravelBooking(
        {
            "access_token": "token",
            "credit_card_list": {
                "card_1234": {
                    "card_number": "4111111111111234",
                    "balance": 1000.0,
                }
            },
        }
    )
    result = api.register_credit_card(
        access_token="token",
        card_number="5555555555551234",
        expiration_date="12/99",
        cardholder_name="Example User",
        card_verification_number=123,
    )
    assert result["card_id"] == "card_1234_2"
    assert api.credit_card_list["card_1234"]["card_number"] == "4111111111111234"
    assert api.credit_card_list["card_1234_2"]["card_number"] == "5555555555551234"
