import pytest
import json
from typing import List, Dict, Any
from tools.toollens.travel import TravelTools


class TestTravelToolsSequentialCorrect:
    """Correct ordered sequences of TravelTools API calls."""

    @pytest.fixture
    def tools(self) -> TravelTools:
        """Return a fresh TravelTools instance."""
        return TravelTools(initial_config=json.loads(json.dumps(None)))

    def test_autocomplete_then_search_place(self, tools: TravelTools) -> None:
        """
        Call Auto_complete with a partial string, then use its first result
        to perform a Search_Place query for the full place name.
        """
        # Step 1: auto-complete a partial query
        auto_results = tools.Auto_complete("New")
        assert isinstance(auto_results, list), "Auto_complete must return a list"
        # Step 2: search for a place if we got suggestions
        query = "New York"  # fallback if empty
        if auto_results:
            first = auto_results[0]
            if isinstance(first, dict) and "name" in first:
                query = first["name"]
            elif isinstance(first, str):
                query = first
        search_result = tools.Search_Place(query)
        assert isinstance(search_result, dict), "Search_Place must return a dict"
        # Ensure the search result contains expected keys (e.g., "results", "status")
        assert any(key in search_result for key in ("results", "status", "error")), \
            "Search_Place result should contain meaningful keys"

    def test_cities_list_then_distance_by_city(self, tools: TravelTools) -> None:
        """
        Retrieve the list of cities, pick two, then compute distance
        between them using Get_Distance_By_City_2.
        """
        cities_list = tools.Get_Cities_List()
        assert isinstance(cities_list, list), "Get_Cities_List must return a list"
        assert len(cities_list) >= 2, "Need at least two cities for distance test"
        # Choose the first two cities (assume each entry is a dict with city/country)
        city1 = cities_list[0]
        city2 = cities_list[1]
        # Extract city and country names; use defaults if structure unknown
        city1_name = city1.get("city", "Paris") if isinstance(city1, dict) else str(city1)
        city1_country = city1.get("country", "France") if isinstance(city1, dict) else "France"
        city2_name = city2.get("city", "London") if isinstance(city2, dict) else str(city2)
        city2_country = city2.get("country", "United Kingdom") if isinstance(city2, dict) else "United Kingdom"
        # Call Get_Distance_By_City_2 (parameters: country1, country2, state2, city2, city1, state1)
        distance_result = tools.Get_Distance_By_City_2(
            country1=city1_country,
            country2=city2_country,
            state2="",
            city2=city2_name,
            city1=city1_name,
            state1=""
        )
        assert isinstance(distance_result, dict), "Get_Distance_By_City_2 must return a dict"

    def test_webcams_bbox_then_region(self, tools: TravelTools) -> None:
        """
        Query webcams inside a bounding box, then retrieve webcams for
        a specific region.
        """
        # Bounding box around France (SW corner to NE corner)
        bbox_result = tools.webcams_list_bbox_ne_lat(
            ne_lat=51.0,
            sw_lng=-5.0,
            sw_lat=42.0,
            ne_lng=8.0
        )
        assert isinstance(bbox_result, list), "webcams_list_bbox_ne_lat must return a list"
        # Now query webcams for a region (e.g., "france")
        region_result = tools.webcams_list_region_region(region="france")
        assert isinstance(region_result, dict), "webcams_list_region_region must return a dict"
        assert "webcams" in region_result or "result" in region_result or "error" in region_result, \
            "Region result should contain expected keys"

    def test_airport_then_currencies(self, tools: TravelTools) -> None:
        """
        Retrieve airport data, then get the list of currencies.
        Both are independent but represent typical data-loading sequence.
        """
        airports = tools.Airport_data_in_json_format()
        assert isinstance(airports, list), "Airport_data_in_json_format must return a list"
        assert len(airports) > 0, "Airport list should not be empty"
        currencies = tools.Get_Currencies_List()
        assert isinstance(currencies, list), "Get_Currencies_List must return a list"
        # Verify each currency entry is a dict with 'code' or 'name'
        if currencies:
            first = currencies[0]
            assert isinstance(first, dict) and ("code" in first or "name" in first), \
                "Currency entries should have 'code' or 'name'"


class TestTravelToolsSequentialProblematic:
    """Problematic sequences that test error handling and edge cases."""

    @pytest.fixture
    def tools(self) -> TravelTools:
        """Return a fresh TravelTools instance."""
        return TravelTools(initial_config=json.loads(json.dumps(None)))

    def test_invalid_bbox_then_region(self, tools: TravelTools) -> None:
        """
        Call webcams_list_bbox_ne_lat with invalid coordinates
        (NE latitude less than SW latitude), then call a region query.
        Both should not raise exceptions.
        """
        # Invalid bbox: ne_lat < sw_lat (south of sw) → should return empty or error
        invalid_bbox = tools.webcams_list_bbox_ne_lat(
            ne_lat=40.0,   # less than sw_lat
            sw_lng=-10.0,
            sw_lat=50.0,
            ne_lng=10.0
        )
        # Should be a list (could be empty)
        assert isinstance(invalid_bbox, list), "Invalid bbox should still return a list"
        # Now call region query; must not crash regardless of previous call
        region_result = tools.webcams_list_region_region(region="invalid_region_xyz")
        assert isinstance(region_result, dict), "Region result must be a dict"

    def test_empty_search_and_autocomplete(self, tools: TravelTools) -> None:
        """
        Call Search_Place with an empty string, then Auto_complete
        with an empty string. Both must return empty or graceful error.
        """
        empty_search = tools.Search_Place("")
        assert isinstance(empty_search, dict), "Empty search should return a dict"
        # Might contain "error" or empty results list
        empty_auto = tools.Auto_complete("")
        assert isinstance(empty_auto, list), "Empty autocomplete should return a list"
        # Should be empty or contain error-like dict
        if empty_auto:
            first = empty_auto[0]
            # Could be a dict with "error" or a string suggestion
            assert isinstance(first, (dict, str))

    def test_invalid_distance_params(self, tools: TravelTools) -> None:
        """
        Call Get_Distance with out-of-range latitude/longitude values,
        then call Get_Distance_in_Km with similarly invalid values.
        Both must return a dict with error information.
        """
        # Latitude out of range (-90 to 90), longitude out of range (-180 to 180)
        distance1 = tools.Get_Distance(latB=100.0, longA=200.0, latA=-100.0, longB=-200.0)
        assert isinstance(distance1, dict), "Get_Distance must return a dict"
        # Check for some key indicating error or still structured response
        # Latitude out of range in km version (note parameter order: latB, longB, longA, latA)
        distance2 = tools.Get_Distance_in_Km(latB=100.0, longB=200.0, longA=-200.0, latA=-100.0)
        assert isinstance(distance2, dict), "Get_Distance_in_Km must return a dict"
        # If distance had 'distance' key, the error version should not raise
        # Just ensure we can call without exception

    def test_invalid_administrative_division(self, tools: TravelTools) -> None:
        """
        Call Get_administrative_divisions with a non-existent country code,
        then call Prices_and_Availability_by_administrative_divisions with
        an unlikely combination.
        """
        # Invalid country code "ZZ"
        invalid_adm = tools.Get_administrative_divisions(countrycode="ZZ")
        assert isinstance(invalid_adm, list), "Get_administrative_divisions must return a list"
        # Should be empty list on error
        # Now call prices/availability with invalid/outlandish parameters
        invalid_price = tools.Prices_and_Availability_by_administrative_divisions(
            month="InvalidMonth",
            country_code="ZZ",
            year=9999
        )
        assert isinstance(invalid_price, dict), "Prices_and_Availability must return a dict"
        # Should contain error key or empty data

    def test_invalid_state_gas_price(self, tools: TravelTools) -> None:
        """
        Call stateUsaPrice with a non-existent US state, then call allUsaPrice
        to ensure the API remains functional.
        """
        bad_state = tools.stateUsaPrice(state="NotAState")
        assert isinstance(bad_state, dict), "stateUsaPrice must return a dict"
        # Should return error message about invalid state
        all_prices = tools.allUsaPrice()
        assert isinstance(all_prices, dict), "allUsaPrice must return a dict"
        # Verify that overall API still works
        assert "prices" in all_prices or "result" in all_prices or "error" in all_prices, \
            "allUsaPrice should have expected keys"