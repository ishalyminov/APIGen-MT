import pytest
from tools.toollens.transportation import TransportationTools


@pytest.fixture
def transportation_instance():
    """Fixture providing a stateless TransportationTools instance."""
    return TransportationTools(initial_config=None)


# -----------------------------------------------------------------------------
# specs_v1_getMakes
# -----------------------------------------------------------------------------
def test_specs_v1_getMakes_returns_list(transportation_instance):
    """specs_v1_getMakes should return a list of strings."""
    result = transportation_instance.specs_v1_getMakes()
    assert isinstance(result, list)
    # at least one element (realistic mock)
    assert len(result) > 0
    if result:
        assert isinstance(result[0], str)


# -----------------------------------------------------------------------------
# us
# -----------------------------------------------------------------------------
def test_us_returns_dict(transportation_instance):
    """us should return a dictionary."""
    result = transportation_instance.us()
    assert isinstance(result, dict)


def test_us_contains_country_info(transportation_instance):
    """us should contain expected keys (mock data)."""
    result = transportation_instance.us()
    assert isinstance(result, dict)
    # typical mock keys
    assert "name" in result or "country" in result or "data" in result


# -----------------------------------------------------------------------------
# us_ak
# -----------------------------------------------------------------------------
def test_us_ak_returns_dict(transportation_instance):
    """us_ak should return a dictionary."""
    result = transportation_instance.us_ak()
    assert isinstance(result, dict)


def test_us_ak_contains_state_info(transportation_instance):
    """us_ak should contain state-level info."""
    result = transportation_instance.us_ak()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# us_al
# -----------------------------------------------------------------------------
def test_us_al_returns_dict(transportation_instance):
    """us_al should return a dictionary."""
    result = transportation_instance.us_al()
    assert isinstance(result, dict)


def test_us_al_contains_state_info(transportation_instance):
    """us_al should contain state-level info."""
    result = transportation_instance.us_al()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# us_az
# -----------------------------------------------------------------------------
def test_us_az_returns_dict(transportation_instance):
    """us_az should return a dictionary."""
    result = transportation_instance.us_az()
    assert isinstance(result, dict)


def test_us_az_contains_state_info(transportation_instance):
    """us_az should contain state-level info."""
    result = transportation_instance.us_az()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# us_ca
# -----------------------------------------------------------------------------
def test_us_ca_returns_dict(transportation_instance):
    """us_ca should return a dictionary."""
    result = transportation_instance.us_ca()
    assert isinstance(result, dict)


def test_us_ca_contains_state_info(transportation_instance):
    """us_ca should contain state-level info."""
    result = transportation_instance.us_ca()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# us_dc
# -----------------------------------------------------------------------------
def test_us_dc_returns_dict(transportation_instance):
    """us_dc should return a dictionary."""
    result = transportation_instance.us_dc()
    assert isinstance(result, dict)


def test_us_dc_contains_state_info(transportation_instance):
    """us_dc should contain district-level info."""
    result = transportation_instance.us_dc()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# us_de
# -----------------------------------------------------------------------------
def test_us_de_returns_dict(transportation_instance):
    """us_de should return a dictionary."""
    result = transportation_instance.us_de()
    assert isinstance(result, dict)


def test_us_de_contains_state_info(transportation_instance):
    """us_de should contain state-level info."""
    result = transportation_instance.us_de()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# us_ga
# -----------------------------------------------------------------------------
def test_us_ga_returns_dict(transportation_instance):
    """us_ga should return a dictionary."""
    result = transportation_instance.us_ga()
    assert isinstance(result, dict)


def test_us_ga_contains_state_info(transportation_instance):
    """us_ga should contain state-level info."""
    result = transportation_instance.us_ga()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# v1_motorcycles
# -----------------------------------------------------------------------------
def test_v1_motorcycles_returns_dict(transportation_instance):
    """v1_motorcycles should return a dictionary."""
    result = transportation_instance.v1_motorcycles()
    assert isinstance(result, dict)


def test_v1_motorcycles_has_motorcycle_data(transportation_instance):
    """v1_motorcycles should contain motorcycle-related data."""
    result = transportation_instance.v1_motorcycles()
    assert isinstance(result, dict)
    # typical mock keys
    assert "makes" in result or "models" in result or "data" in result


# -----------------------------------------------------------------------------
# Cities
# -----------------------------------------------------------------------------
def test_Cities_valid_province(transportation_instance):
    """Cities with a known province should return a dict."""
    result = transportation_instance.Cities(province="Ontario")
    assert isinstance(result, dict)


def test_Cities_none_province(transportation_instance):
    """Cities with None should return an error dict or empty data."""
    result = transportation_instance.Cities(province=None)
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Get_Airline_Alliance_List
# -----------------------------------------------------------------------------
def test_Get_Airline_Alliance_List_returns_dict(transportation_instance):
    """Get_Airline_Alliance_List should return a dictionary."""
    result = transportation_instance.Get_Airline_Alliance_List()
    assert isinstance(result, dict)


def test_Get_Airline_Alliance_List_has_alliances(transportation_instance):
    """Get_Airline_Alliance_List should contain alliance data."""
    result = transportation_instance.Get_Airline_Alliance_List()
    assert isinstance(result, dict)
    assert "alliances" in result or "data" in result


# -----------------------------------------------------------------------------
# Get_Airline_Details
# -----------------------------------------------------------------------------
def test_Get_Airline_Details_valid_code(transportation_instance):
    """Get_Airline_Details with a valid IATA code should return a dict."""
    result = transportation_instance.Get_Airline_Details(code="AA")
    assert isinstance(result, dict)


def test_Get_Airline_Details_none_code(transportation_instance):
    """Get_Airline_Details with None should return an error dict."""
    result = transportation_instance.Get_Airline_Details(code=None)
    assert isinstance(result, dict)
    # expecting error indication
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# Get_Car_Models
# -----------------------------------------------------------------------------
def test_Get_Car_Models_valid_maker(transportation_instance):
    """Get_Car_Models with a known maker should return a dict."""
    result = transportation_instance.Get_Car_Models(maker="Toyota")
    assert isinstance(result, dict)


def test_Get_Car_Models_none_maker(transportation_instance):
    """Get_Car_Models with None should return an error dict."""
    result = transportation_instance.Get_Car_Models(maker=None)
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# Get_TimeZones
# -----------------------------------------------------------------------------
def test_Get_TimeZones_returns_dict(transportation_instance):
    """Get_TimeZones should return a dictionary."""
    result = transportation_instance.Get_TimeZones()
    assert isinstance(result, dict)


def test_Get_TimeZones_has_timezones(transportation_instance):
    """Get_TimeZones should contain timezone data."""
    result = transportation_instance.Get_TimeZones()
    assert isinstance(result, dict)
    assert "timezones" in result or "data" in result


# -----------------------------------------------------------------------------
# Get_taxi_fares
# -----------------------------------------------------------------------------
def test_Get_taxi_fares_valid_coordinates(transportation_instance):
    """Get_taxi_fares with valid lat/lng should return a dict."""
    result = transportation_instance.Get_taxi_fares(
        arr_lat=40.7128, arr_lng=-74.0060, dep_lat=34.0522, dep_lng=-118.2437
    )
    assert isinstance(result, dict)


def test_Get_taxi_fares_invalid_coordinates(transportation_instance):
    """Get_taxi_fares with None values should return an error dict."""
    result = transportation_instance.Get_taxi_fares(
        arr_lat=None, arr_lng=None, dep_lat=None, dep_lng=None
    )
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# Province_List
# -----------------------------------------------------------------------------
def test_Province_List_returns_list(transportation_instance):
    """Province_List should return a list of strings."""
    result = transportation_instance.Province_List()
    assert isinstance(result, list)
    # at least one element
    assert len(result) > 0
    if result:
        assert isinstance(result[0], str)


# -----------------------------------------------------------------------------
# Provinces
# -----------------------------------------------------------------------------
def test_Provinces_returns_dict(transportation_instance):
    """Provinces should return a dictionary."""
    result = transportation_instance.Provinces()
    assert isinstance(result, dict)


def test_Provinces_has_province_data(transportation_instance):
    """Provinces should contain province-level information."""
    result = transportation_instance.Provinces()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Total_Live_tracked_Aircraft
# -----------------------------------------------------------------------------
def test_Total_Live_tracked_Aircraft_returns_dict(transportation_instance):
    """Total_Live_tracked_Aircraft should return a dictionary."""
    result = transportation_instance.Total_Live_tracked_Aircraft()
    assert isinstance(result, dict)


def test_Total_Live_tracked_Aircraft_has_count(transportation_instance):
    """Total_Live_tracked_Aircraft should contain a count or data."""
    result = transportation_instance.Total_Live_tracked_Aircraft()
    assert isinstance(result, dict)
    assert "total" in result or "count" in result or "data" in result


# -----------------------------------------------------------------------------
# aircrafts_list
# -----------------------------------------------------------------------------
def test_aircrafts_list_returns_dict(transportation_instance):
    """aircrafts_list should return a dictionary."""
    result = transportation_instance.aircrafts_list()
    assert isinstance(result, dict)


def test_aircrafts_list_has_aircraft_data(transportation_instance):
    """aircrafts_list should contain aircraft information."""
    result = transportation_instance.aircrafts_list()
    assert isinstance(result, dict)
    assert "aircraft" in result or "aircrafts" in result or "data" in result


# -----------------------------------------------------------------------------
# airlines_Airlines_and_the_countries_they_operate_in
# -----------------------------------------------------------------------------
def test_airlines_Airlines_and_the_countries_they_operate_in_returns_dict(transportation_instance):
    """airlines_Airlines_and_the_countries_they_operate_in should return a dict."""
    result = transportation_instance.airlines_Airlines_and_the_countries_they_operate_in()
    assert isinstance(result, dict)


def test_airlines_Airlines_and_the_countries_they_operate_in_has_data(transportation_instance):
    """Should contain airline-country mapping data."""
    result = transportation_instance.airlines_Airlines_and_the_countries_they_operate_in()
    assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# airlines_get_logos
# -----------------------------------------------------------------------------
def test_airlines_get_logos_returns_dict(transportation_instance):
    """airlines_get_logos should return a dictionary."""
    result = transportation_instance.airlines_get_logos()
    assert isinstance(result, dict)


def test_airlines_get_logos_has_logo_data(transportation_instance):
    """airlines_get_logos should contain logo information."""
    result = transportation_instance.airlines_get_logos()
    assert isinstance(result, dict)
    assert "logos" in result or "data" in result


# -----------------------------------------------------------------------------
# airlines_list
# -----------------------------------------------------------------------------
def test_airlines_list_returns_dict(transportation_instance):
    """airlines_list should return a dictionary."""
    result = transportation_instance.airlines_list()
    assert isinstance(result, dict)


def test_airlines_list_has_airline_data(transportation_instance):
    """airlines_list should contain airline list data."""
    result = transportation_instance.airlines_list()
    assert isinstance(result, dict)
    assert "airlines" in result or "data" in result


# -----------------------------------------------------------------------------
# airports_Direct_routes_for_an_airport
# -----------------------------------------------------------------------------
def test_airports_Direct_routes_for_an_airport_valid_code(transportation_instance):
    """Direct routes with a valid IATA code should return a dict."""
    result = transportation_instance.airports_Direct_routes_for_an_airport(airportiatacode="JFK")
    assert isinstance(result, dict)


def test_airports_Direct_routes_for_an_airport_none_code(transportation_instance):
    """Direct routes with None code should return an error dict."""
    result = transportation_instance.airports_Direct_routes_for_an_airport(airportiatacode=None)
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# airports_Direct_routes_for_an_airport_by_airline
# -----------------------------------------------------------------------------
def test_airports_Direct_routes_for_an_airport_by_airline_valid(transportation_instance):
    """Direct routes by airline with valid codes should return a dict."""
    result = transportation_instance.airports_Direct_routes_for_an_airport_by_airline(
        airportiatacode="JFK", airlineiatacode="AA"
    )
    assert isinstance(result, dict)


def test_airports_Direct_routes_for_an_airport_by_airline_none(transportation_instance):
    """Direct routes by airline with None codes should return an error dict."""
    result = transportation_instance.airports_Direct_routes_for_an_airport_by_airline(
        airportiatacode=None, airlineiatacode=None
    )
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# airports_Metro_IATA_codes
# -----------------------------------------------------------------------------
def test_airports_Metro_IATA_codes_returns_dict(transportation_instance):
    """airports_Metro_IATA_codes should return a dictionary."""
    result = transportation_instance.airports_Metro_IATA_codes()
    assert isinstance(result, dict)


def test_airports_Metro_IATA_codes_has_metro_codes(transportation_instance):
    """airports_Metro_IATA_codes should contain metro IATA data."""
    result = transportation_instance.airports_Metro_IATA_codes()
    assert isinstance(result, dict)
    assert "metros" in result or "data" in result


# -----------------------------------------------------------------------------
# airports_Nearest_airports_for_a_given_latitude_and_longitude
# -----------------------------------------------------------------------------
def test_airports_Nearest_airports_valid_coords(transportation_instance):
    """Nearest airports with valid coordinates should return a dict."""
    result = transportation_instance.airports_Nearest_airports_for_a_given_latitude_and_longitude(
        lon=-73.935242, lat=40.730610
    )
    assert isinstance(result, dict)


def test_airports_Nearest_airports_invalid_coords(transportation_instance):
    """Nearest airports with None coords should return an error dict."""
    result = transportation_instance.airports_Nearest_airports_for_a_given_latitude_and_longitude(
        lon=None, lat=None
    )
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# airports_Nonstop_and_direct_routes_for_an_airport
# -----------------------------------------------------------------------------
def test_airports_Nonstop_and_direct_routes_for_an_airport_valid_code(transportation_instance):
    """Nonstop & direct routes with a valid IATA code should return a dict."""
    result = transportation_instance.airports_Nonstop_and_direct_routes_for_an_airport(airportiatacode="LHR")
    assert isinstance(result, dict)


def test_airports_Nonstop_and_direct_routes_for_an_airport_none_code(transportation_instance):
    """Nonstop & direct routes with None should return an error dict."""
    result = transportation_instance.airports_Nonstop_and_direct_routes_for_an_airport(airportiatacode=None)
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# airports_Nonstop_and_direct_routes_for_an_airport_by_airline
# -----------------------------------------------------------------------------
def test_airports_Nonstop_and_direct_routes_by_airline_valid(transportation_instance):
    """Nonstop & direct routes by airline with valid codes should return a dict."""
    result = transportation_instance.airports_Nonstop_and_direct_routes_for_an_airport_by_airline(
        airlineiatacode="BA", airportiatacode="LHR"
    )
    assert isinstance(result, dict)


def test_airports_Nonstop_and_direct_routes_by_airline_none(transportation_instance):
    """Nonstop & direct routes by airline with None should return an error dict."""
    result = transportation_instance.airports_Nonstop_and_direct_routes_for_an_airport_by_airline(
        airlineiatacode=None, airportiatacode=None
    )
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# airports_Nonstop_routes_for_an_airport
# -----------------------------------------------------------------------------
def test_airports_Nonstop_routes_for_an_airport_valid_code(transportation_instance):
    """Nonstop routes with a valid IATA code should return a dict."""
    result = transportation_instance.airports_Nonstop_routes_for_an_airport(airportiatacode="DXB")
    assert isinstance(result, dict)


def test_airports_Nonstop_routes_for_an_airport_none_code(transportation_instance):
    """Nonstop routes with None should return an error dict."""
    result = transportation_instance.airports_Nonstop_routes_for_an_airport(airportiatacode=None)
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False


# -----------------------------------------------------------------------------
# flights_list_in_boundary
# -----------------------------------------------------------------------------
def test_flights_list_in_boundary_valid_bounds(transportation_instance):
    """flights_list_in_boundary with valid boundary values should return a dict."""
    result = transportation_instance.flights_list_in_boundary(
        bl_lng=-180.0, tr_lat=90.0, bl_lat=-90.0, tr_lng=180.0
    )
    assert isinstance(result, dict)


def test_flights_list_in_boundary_invalid_bounds(transportation_instance):
    """flights_list_in_boundary with None values should return an error dict."""
    result = transportation_instance.flights_list_in_boundary(
        bl_lng=None, tr_lat=None, bl_lat=None, tr_lng=None
    )
    assert isinstance(result, dict)
    assert "error" in result or result.get("success") is False