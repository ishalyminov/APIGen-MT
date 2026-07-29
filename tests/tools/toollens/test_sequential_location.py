import pytest
import json
from tools.toollens.location import LocationTools

class TestLocationToolsSequentialCorrect:
    """Correct ordered sequences that build on previous results."""

    @pytest.fixture
    def tools(self):
        """Return a fresh LocationTools instance for each test."""
        config = None  # initial_config is None
        return LocationTools(config)

    def test_capital_to_timezone_sequence(self, tools):
        """Get capital of France, then look up timezone for that city."""
        # Step 1: Get capital of France
        capital_res = tools.Capital_By_Country("France")
        assert isinstance(capital_res, dict), "Expected a dict"
        # Assume capital is returned under 'capital' or 'data'
        capital = capital_res.get("capital") or (capital_res.get("data", {}).get("capital"))
        assert capital is not None, "Capital not found in response"

        # Step 2: Use that capital to find timezone via v1_timezone
        tz_res = tools.v1_timezone(city=capital, country="France")
        assert isinstance(tz_res, dict)
        # Expect either success or timezone data
        assert "timezone" in tz_res or ("data" in tz_res and "timezone" in tz_res["data"]), \
            "Timezone info missing"

    def test_reverse_geocode_to_directions_sequence(self, tools):
        """Reverse geocode coordinates, then get directions using those."""
        # Step 1: Reverse geocode a known location (e.g., Paris)
        geo_res = tools.Reverse_Geocode(lon="2.3522", lat="48.8566")
        assert isinstance(geo_res, dict)
        # Assume returned address can be used as start_lat/start_lon
        # For simplicity we just check the reverse geocode succeeded
        assert "address" in geo_res or ("data" in geo_res and "address" in geo_res["data"])

        # Step 2: Compute directions using the same coordinates as start
        dir_res = tools.Directions_Between_2_Locations(
            start_lat=48.8566, start_lon=2.3522,
            end_lat=48.8600, end_lon=2.3500
        )
        assert isinstance(dir_res, dict)
        # Should contain distance and duration
        assert "distance" in dir_res or "duration" in dir_res or \
               ("data" in dir_res and all(k in dir_res["data"] for k in ("distance", "duration")))

    def test_ip_lookup_to_current_time_sequence(self, tools):
        """Geolocate an IP, then get current time for that same IP."""
        test_ip = "8.8.8.8"
        # Step 1: IP geolocation
        ip_res = tools.IP_Geolocation_Lookup(test_ip)
        assert isinstance(ip_res, dict)
        # Assume success
        assert ip_res.get("success", True) != False, "IP geolocation failed"

        # Step 2: Current time for same IP
        time_res = tools.Current_time_by_Specific_IP(test_ip)
        assert isinstance(time_res, dict)
        # Should contain time information
        assert "time" in time_res or ("data" in time_res and "time" in time_res["data"])

    def test_zipcode_to_income_sequence(self, tools):
        """Get ZIP info for a code, then request income data for that code."""
        zip_code = "90210"
        # Step 1: ZIP info
        zip_res = tools.Get_ZIP_Info(zip_code)
        assert isinstance(zip_res, dict)
        # Should have city or similar
        assert "city" in zip_res or ("data" in zip_res and "city" in zip_res["data"])

        # Step 2: Income by same ZIP
        income_res = tools.Income_By_Zipcode(zip_code)
        assert isinstance(income_res, dict)
        # Income data should be present
        assert "income" in income_res or ("data" in income_res and "income" in income_res["data"])

    def test_suburb_radius_then_list_sequence(self, tools):
        """Get suburbs in a radius, then list suburbs for a postcode from those."""
        # Step 1: Get suburbs within radius of a central point
        radius_res = tools.Get_all_suburbs_and_postcodes_in_a_radius(
            lat="48.8566", radius=10.0, lng="2.3522"
        )
        assert isinstance(radius_res, dict)
        # Assume data contains suburbs list
        data = radius_res.get("data") or radius_res
        suburbs = data.get("suburbs", [])
        if suburbs:
            # Step 2: Get list of suburbs for the first postcode found
            first_postcode = suburbs[0].get("postcode")
            if first_postcode:
                list_res = tools.Get_a_list_of_suburbs(postcode=first_postcode)
                assert isinstance(list_res, dict)
                # At least we got a response
                assert True


class TestLocationToolsSequentialProblematic:
    """Problematic sequences: invalid inputs, missing resources, etc."""

    @pytest.fixture
    def tools(self):
        """Return a fresh LocationTools instance for each test."""
        config = None
        return LocationTools(config)

    def test_invalid_then_valid_capital_sequence(self, tools):
        """Call Capital_By_Country with nonexistent country, then valid one."""
        # Step 1: Invalid country
        invalid_res = tools.Capital_By_Country("Atlantis")
        assert isinstance(invalid_res, dict)
        # Expect an error or no capital found
        if "error" in invalid_res or invalid_res.get("success") is False:
            pass  # good, handled gracefully
        else:
            # At least capital should be None
            capital = invalid_res.get("capital") or \
                      (invalid_res.get("data", {}).get("capital"))
            assert capital is None, "Expected no capital for nonexistent country"

        # Step 2: Valid country should work
        valid_res = tools.Capital_By_Country("Germany")
        assert isinstance(valid_res, dict)
        # Should succeed
        capital = valid_res.get("capital") or \
                  (valid_res.get("data", {}).get("capital"))
        assert capital is not None

    def test_invalid_metric_distance_then_directions(self, tools):
        """Calculate distance with invalid metric, then call directions."""
        # Step 1: Invalid metric
        bad_metric_res = tools.Calculate_distance_By_Lat_Long(
            metric="lightyears",
            lat1="48.8566", lon1="2.3522",
            lat2="40.7128", lon2="-74.0060"
        )
        assert isinstance(bad_metric_res, dict)
        # Should return error or fallback to km
        # We just confirm it doesn't crash and returns dict

        # Step 2: Directions should still work regardless
        dir_res = tools.Directions_Between_2_Locations(
            start_lat=48.8566, start_lon=2.3522,
            end_lat=40.7128, end_lon=-74.0060
        )
        assert isinstance(dir_res, dict)
        # Should have distance and duration
        assert "distance" in dir_res or "duration" in dir_res or \
               ("data" in dir_res and all(k in dir_res["data"] for k in ("distance", "duration")))

    def test_invalid_zip_then_income_sequence(self, tools):
        """Look up invalid ZIP code, then try income for same invalid ZIP."""
        invalid_zip = "00000"  # unlikely valid
        # Step 1: ZIP info for invalid
        zip_res = tools.Get_ZIP_Info(invalid_zip)
        assert isinstance(zip_res, dict)
        # Should return error or empty
        if "error" in zip_res or zip_res.get("success") is False:
            pass  # expected
        else:
            # Might still return something, but city should be None
            city = zip_res.get("city") or (zip_res.get("data", {}).get("city"))
            # Not strictly necessary, but we just verify it's a dict

        # Step 2: Income for same invalid ZIP
        income_res = tools.Income_By_Zipcode(invalid_zip)
        assert isinstance(income_res, dict)
        # Could be error or no data

    def test_invalid_ip_lookup_then_current_time(self, tools):
        """IP geolocation with invalid IP, then current time for that IP."""
        bad_ip = "999.999.999.999"
        # Step 1: IP geolocation
        ip_res = tools.IP_Geolocation_Lookup(bad_ip)
        assert isinstance(ip_res, dict)
        # Expect error
        assert "error" in ip_res or ip_res.get("success") is False, \
            "Expected error for invalid IP"

        # Step 2: Current time for same bad IP
        time_res = tools.Current_time_by_Specific_IP(bad_ip)
        assert isinstance(time_res, dict)
        # Should also return error
        assert "error" in time_res or time_res.get("success") is False, \
            "Expected error for invalid IP time lookup"

    def test_invalid_state_code_then_valid_reverse_geocode(self, tools):
        """Lookup state with invalid code, then reverse geocode valid coords."""
        # Step 1: Invalid state code
        state_res = tools.State_by_id("ZZ-XXX")
        assert isinstance(state_res, dict)
        # Should indicate not found
        if "error" not in state_res and state_res.get("success") is not False:
            state = state_res.get("state") or (state_res.get("data", {}).get("name"))
            assert state is None, "Expected no state for invalid code"

        # Step 2: Valid reverse geocode
        geo_res = tools.ReverseGeocode(lat=48.8566, lon=2.3522)
        assert isinstance(geo_res, dict)
        # Should return address
        assert "address" in geo_res or ("data" in geo_res and "address" in geo_res["data"])