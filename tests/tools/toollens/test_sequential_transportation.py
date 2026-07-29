import pytest
import json
from typing import List, Dict, Any
from tools.toollens.transportation import TransportationTools


@pytest.fixture
def transport_tools() -> TransportationTools:
    return TransportationTools()


class TestTransportationToolsSequentialCorrect:
    """
    Correct ordered sequences that build on prior results.
    """

    def test_province_list_to_cities(self, transport_tools: TransportationTools) -> None:
        """Get province list → pick first province → get its cities."""
        provinces = transport_tools.Province_List()
        assert isinstance(provinces, list), "Province_List should return a list"
        assert len(provinces) > 0, "Should have at least one province"
        first_province = provinces[0]
        cities = transport_tools.Cities(province=first_province)
        assert isinstance(cities, dict), "Cities should return a dict"
        # Expect the response to contain information about the province or cities
        assert "province" in cities or "cities" in cities, \
            "Cities response should include province or cities key"

    def test_makes_to_car_models(self, transport_tools: TransportationTools) -> None:
        """Get all makes → pick first maker → get its car models."""
        makes = transport_tools.specs_v1_getMakes()
        assert isinstance(makes, list), "specs_v1_getMakes should return a list"
        assert len(makes) > 0, "Should have at least one make"
        first_make = makes[0]
        models = transport_tools.Get_Car_Models(maker=first_make)
        assert isinstance(models, dict), "Get_Car_Models should return a dict"
        # Models response should contain the maker or models key
        assert "maker" in models or "models" in models, \
            "Car models response should include maker or models key"

    def test_airline_list_to_details(self, transport_tools: TransportationTools) -> None:
        """Get airline list → extract first airline code → get its details."""
        airlines = transport_tools.airlines_list()
        assert isinstance(airlines, dict), "airlines_list should return a dict"
        # Assume airlines dict contains an 'airlines' list of dicts with 'code'
        assert "airlines" in airlines, "airlines_list should have 'airlines' key"
        airline_list = airlines["airlines"]
        assert len(airline_list) > 0, "Should have at least one airline"
        first_code = airline_list[0].get("code", "AA")  # fallback for safety
        details = transport_tools.Get_Airline_Details(code=first_code)
        assert isinstance(details, dict), "Get_Airline_Details should return a dict"
        # Details should contain airline information
        assert "code" in details or "name" in details, \
            "Airline details should include code or name"

    def test_us_then_state(self, transport_tools: TransportationTools) -> None:
        """Get US overview → then get California details."""
        us_info = transport_tools.us()
        assert isinstance(us_info, dict), "us() should return a dict"
        # us() should return structured data about the US
        ca_info = transport_tools.us_ca()
        assert isinstance(ca_info, dict), "us_ca() should return a dict"
        # CA info should contain information about California
        assert "state" in ca_info or "name" in ca_info or "california" in str(ca_info).lower(), \
            "us_ca() should include California information"


class TestTransportationToolsSequentialProblematic:
    """
    Problematic sequences: invalid arguments, nonexistent resources, etc.
    The next method should still work without crashing.
    """

    def test_invalid_airline_details_then_alliance_list(
        self, transport_tools: TransportationTools
    ) -> None:
        """Get airline details for nonexistent code → then get alliance list (should succeed)."""
        details = transport_tools.Get_Airline_Details(code="NONEXISTENT")
        assert isinstance(details, dict), "Get_Airline_Details should return a dict"
        # The response may be empty or contain an error indicator
        # Subsequent call must not crash
        alliances = transport_tools.Get_Airline_Alliance_List()
        assert isinstance(alliances, dict), "Get_Airline_Alliance_List should return a dict"

    def test_invalid_make_then_province_list(self, transport_tools: TransportationTools) -> None:
        """Request models for invalid maker → then get province list (should succeed)."""
        models = transport_tools.Get_Car_Models(maker="InvalidMaker")
        assert isinstance(models, dict), "Get_Car_Models should return a dict"
        # Continue with another call
        provinces = transport_tools.Province_List()
        assert isinstance(provinces, list), "Province_List should return a list"

    def test_invalid_airport_routes_then_metro_iata(
        self, transport_tools: TransportationTools
    ) -> None:
        """Request nonstop routes for an invalid airport IATA → then get metro IATA codes."""
        routes = transport_tools.airports_Nonstop_routes_for_an_airport(
            airportiatacode="ZZZ"
        )
        assert isinstance(routes, dict), \
            "airports_Nonstop_routes_for_an_airport should return a dict"
        # Next call should work
        metro = transport_tools.airports_Metro_IATA_codes()
        assert isinstance(metro, dict), "airports_Metro_IATA_codes should return a dict"

    def test_invalid_city_province_then_total_aircraft(
        self, transport_tools: TransportationTools
    ) -> None:
        """Request cities for an invalid province → then get total tracked aircraft."""
        cities = transport_tools.Cities(province="NonExistentProvince")
        assert isinstance(cities, dict), "Cities should return a dict"
        # Next call should work
        total = transport_tools.Total_Live_tracked_Aircraft()
        assert isinstance(total, dict), "Total_Live_tracked_Aircraft should return a dict"

    def test_invalid_airline_alliance_then_us(
        self, transport_tools: TransportationTools
    ) -> None:
        """Get airline details with empty code → then get US info."""
        details = transport_tools.Get_Airline_Details(code="")
        assert isinstance(details, dict), "Get_Airline_Details should return a dict"
        # Proceed to another method
        us_info = transport_tools.us()
        assert isinstance(us_info, dict), "us() should return a dict"