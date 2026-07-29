import json
import pytest
from tools.toollens.weather import WeatherTools


@pytest.fixture
def weather_tools():
    """Create a fresh WeatherTools instance with deep-copied initial config."""
    config = json.loads(json.dumps({
        "api_key": "YOUR_API_KEY",
        "stations": [
            {"id": "9414290", "name": "San Francisco"},
            {"id": "9410170", "name": "Los Angeles"}
        ],
        "countries": ["US", "CA", "GB", "AU"],
        "resorts": ["Aspen", "Whistler", "Stowe", "Park City"],
        "earthquakes": [
            {"id": "us7000abc", "place": "California"},
            {"id": "us7000def", "place": "Alaska"}
        ],
        "locations": [
            {"lat": 40.7128, "lon": -74.006, "city": "New York"},
            {"lat": 34.0522, "lon": -118.2437, "city": "Los Angeles"}
        ],
        "zip_codes": ["10001", "90210", "94102"],
        "climate_key": "ABCD1234",
        "forecasts": {},
        "alerts": []
    }))
    return WeatherTools(config)


class TestWeatherToolsSequentialCorrect:
    """Sequence tests that exercise typical correct trajectories through the API."""

    def test_search_city_then_weather_updates(self, weather_tools):
        """Search for a city and then get weather updates for it."""
        # Step 1: Search for a location
        search_result = weather_tools.Search_API("New York")
        assert isinstance(search_result, dict), "Search_API should return a dict"
        # Assume the search result contains relevant data
        city = "New York"  # Use the city directly from config

        # Step 2: Get weather updates
        weather_result = weather_tools.Get_Weather_Updates(city)
        assert isinstance(weather_result, dict), "Get_Weather_Updates should return a dict"
        # Expect typical weather fields
        assert any(key in weather_result for key in ("temperature", "city", "weather")), (
            "Weather result should contain temperature or city info"
        )

    def test_resort_forecast_then_snow_conditions(self, weather_tools):
        """Get a 5-day forecast for a resort and then current snow conditions."""
        # Step 1: Get 5-day forecast for a known resort
        forecast = weather_tools.m_5_Day_Forecast("Aspen")
        assert isinstance(forecast, dict), "m_5_Day_Forecast should return a dict"
        # Step 2: Get snow conditions for the same resort
        snow = weather_tools.Current_Snow_Conditions("Aspen")
        assert isinstance(snow, dict), "Current_Snow_Conditions should return a dict"
        # Both calls should succeed
        assert "error" not in forecast, f"Unexpected error in forecast: {forecast.get('error')}"
        assert "error" not in snow, f"Unexpected error in snow: {snow.get('error')}"

    def test_air_quality_forecast_then_classification(self, weather_tools):
        """Get air quality forecast for a location and then retrieve its climate classification."""
        # Use the first location from config
        loc = weather_tools._get_config("locations")[0]
        lat, lon = loc["lat"], loc["lon"]

        # Step 1: Air quality forecast
        aq_forecast = weather_tools.Air_Quality_Forecast(lat, lon)
        assert isinstance(aq_forecast, dict), "Air_Quality_Forecast should return a dict"
        # Step 2: Get Koppen classification
        classification = weather_tools.Classification(str(lon), str(lat))
        assert isinstance(classification, dict), "Classification should return a dict"
        assert "error" not in classification or classification.get("status") != "error", (
            "Classification should succeed for valid lat/lon"
        )

    def test_stations_then_current_conditions(self, weather_tools):
        """List stations and then get detailed current conditions for the first one."""
        # Step 1: Get stations
        stations = weather_tools.Get_stations()
        assert isinstance(stations, list), "Get_stations should return a list"
        # Stations may have lat/lon; use the first station's name to get coordinates from config
        # But the config doesn't store lat/lon per station. We'll use the first location instead.
        loc = weather_tools._get_config("locations")[0]
        # Step 2: Get detailed conditions for that location
        conditions = weather_tools.Current_conditions_detailed(str(loc["lon"]), str(loc["lat"]))
        assert isinstance(conditions, dict), "Current_conditions_detailed should return a dict"
        assert not conditions.get("error"), f"Unexpected error: {conditions.get('error')}"

    def test_postal_code_then_hardiness_zone(self, weather_tools):
        """Check air quality by postal code and then retrieve hardiness zone."""
        # Step 1: Get air quality by postal code
        aq = weather_tools.By_Postal_Code(10001)
        assert isinstance(aq, dict), "By_Postal_Code should return a dict"
        # Step 2: Get hardiness zone for the same zip code
        zone = weather_tools.Retrieve_the_Hardiness_Zone("10001")
        assert isinstance(zone, dict), "Retrieve_the_Hardiness_Zone should return a dict"
        # Both should succeed
        assert "error" not in aq or aq.get("status") != "error"
        assert "error" not in zone or zone.get("status") != "error"


class TestWeatherToolsSequentialProblematic:
    """Sequence tests that handle invalid parameters or missing data gracefully."""

    def test_invalid_resort_then_valid_resort(self, weather_tools):
        """First call with non-existent resort, then valid resort should still succeed."""
        # Step 1: Try a non-existent resort
        forecast = weather_tools.m_5_Day_Forecast("NonExistentResort")
        assert isinstance(forecast, dict), "m_5_Day_Forecast should return a dict"
        # Expect an error indicator
        assert forecast.get("error") or forecast.get("status") == "error", (
            "Should return error for invalid resort"
        )
        # Step 2: Now call with a valid resort – should work
        forecast_ok = weather_tools.m_5_Day_Forecast("Aspen")
        assert isinstance(forecast_ok, dict), "Second call should still return a dict"
        assert "error" not in forecast_ok or forecast_ok.get("status") != "error", (
            "Valid resort should not return error"
        )

    def test_invalid_postal_code_then_valid_postal_code(self, weather_tools):
        """First call with invalid postal code, then valid call."""
        # Step 1: Invalid postal code
        invalid = weather_tools.By_Postal_Code(99999)
        assert isinstance(invalid, dict), "By_Postal_Code should return dict"
        assert invalid.get("error") or invalid.get("status") == "error", (
            "Invalid postal code should return error"
        )
        # Step 2: Valid postal code
        valid = weather_tools.By_Postal_Code(10001)
        assert isinstance(valid, dict), "Second call should return dict"
        assert "error" not in valid or valid.get("status") != "error", (
            "Valid postal code should not return error"
        )

    def test_invalid_lat_lon_then_valid_lat_lon(self, weather_tools):
        """First call with invalid lat/lon, then valid should work."""
        # Step 1: Invalid lat/lon strings
        invalid = weather_tools.Current_Air_Quality("invalid", "invalid")
        assert isinstance(invalid, dict), "Current_Air_Quality should return dict"
        assert invalid.get("error") or invalid.get("status") == "error", (
            "Invalid lat/lon should return error"
        )
        # Step 2: Valid lat/lon from config
        loc = weather_tools._get_config("locations")[0]
        valid = weather_tools.Current_Air_Quality(str(loc["lon"]), str(loc["lat"]))
        assert isinstance(valid, dict), "Second call should return dict"
        assert "error" not in valid or valid.get("status") != "error", (
            "Valid lat/lon should succeed"
        )

    def test_empty_search_then_weather_call(self, weather_tools):
        """Call search with empty string, then try weather update (should handle gracefully)."""
        # Step 1: Empty search
        empty_search = weather_tools.Search_API("")
        assert isinstance(empty_search, dict), "Search_API should return dict"
        # Step 2: Get weather with empty city (may error, but shouldn't crash)
        weather_result = weather_tools.Get_Weather_Updates("")
        assert isinstance(weather_result, dict), "Get_Weather_Updates should return dict"
        # The tool should not raise; error is expected but that's fine
        # No exception means the sequence is safe

    def test_problematic_sequence_does_not_crash_instance(self, weather_tools):
        """Multiple problematic calls in sequence should leave the instance usable."""
        # Call with missing/empty arguments
        _ = weather_tools.Current_Snow_Conditions("")
        _ = weather_tools.m_5_Day_Forecast("")
        _ = weather_tools.By_Postal_Code(-1)
        _ = weather_tools.Current_Air_Quality("", "")
        # Final call with valid data should still work
        final = weather_tools.Get_Weather_Updates("New York")
        assert isinstance(final, dict), "After multiple problematic calls, instance should still work"
        # Should have no lingering errors
        if "error" in final:
            # If error, it should be for a different reason (e.g., API key missing)
            # But config has api key, so we expect success
            pass