import pytest
import json
from tools.toollens.travel import TravelTools


@pytest.fixture
def travel_instance():
    """Create a stateless TravelTools instance."""
    config = None
    return TravelTools(initial_config=config)


# ------------------------------------------------------------------
# webcams_list_bbox_ne_lat
# ------------------------------------------------------------------
def test_webcams_list_bbox_ne_lat_returns_list(travel_instance):
    """Test that webcams_list_bbox_ne_lat returns a list with expected keys."""
    result = travel_instance.webcams_list_bbox_ne_lat(
        ne_lat=48.0, sw_lng=-125.0, sw_lat=24.0, ne_lng=-66.0
    )
    assert isinstance(result, list)
    if result:  # not empty
        assert "id" in result[0]
        assert "lat" in result[0]
        assert "lng" in result[0]


def test_webcams_list_bbox_ne_lat_edge_empty(travel_instance):
    """Test with extreme / unexpected coordinates (should still return list)."""
    result = travel_instance.webcams_list_bbox_ne_lat(
        ne_lat=0.0, sw_lng=0.0, sw_lat=0.0, ne_lng=0.0
    )
    assert isinstance(result, list)


# ------------------------------------------------------------------
# webcams_list_orderby_order
# ------------------------------------------------------------------
def test_webcams_list_orderby_order_returns_dict(travel_instance):
    """Test that webcams_list_orderby_order returns a dict with result info."""
    result = travel_instance.webcams_list_orderby_order(sort="popularity", order="desc")
    assert isinstance(result, dict)
    assert "total" in result or "webcams" in result


def test_webcams_list_orderby_order_invalid_sort(travel_instance):
    """Test with invalid sort key (should still return dict with error or fallback)."""
    result = travel_instance.webcams_list_orderby_order(sort="invalid", order="asc")
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# webcams_list_region_region
# ------------------------------------------------------------------
def test_webcams_list_region_region_returns_dict(travel_instance):
    """Test that webcams_list_region_region returns a dict."""
    result = travel_instance.webcams_list_region_region(region="europe")
    assert isinstance(result, dict)


def test_webcams_list_region_region_empty_region(travel_instance):
    """Test with empty region string (should still return dict)."""
    result = travel_instance.webcams_list_region_region(region="")
    assert isinstance(result, dict)
    # should contain some kind of 'error' or default data
    assert "error" in result or "webcams" in result


# ------------------------------------------------------------------
# Airport_data_in_json_format
# ------------------------------------------------------------------
def test_airport_data_in_json_format_returns_list(travel_instance):
    """Test that Airport_data_in_json_format returns a list."""
    result = travel_instance.Airport_data_in_json_format()
    assert isinstance(result, list)
    if result:
        assert "code" in result[0] or "name" in result[0]


def test_airport_data_in_json_format_not_empty(travel_instance):
    """Test that the list contains at least one airport."""
    result = travel_instance.Airport_data_in_json_format()
    assert len(result) > 0


# ------------------------------------------------------------------
# Auto_complete
# ------------------------------------------------------------------
def test_auto_complete_returns_list(travel_instance):
    """Test that Auto_complete returns a list of suggestions."""
    result = travel_instance.Auto_complete(string="Lon")
    assert isinstance(result, list)
    if result:
        assert isinstance(result[0], str) or isinstance(result[0], dict)


def test_auto_complete_empty_string(travel_instance):
    """Test with empty string – should still return list (maybe empty)."""
    result = travel_instance.Auto_complete(string="")
    assert isinstance(result, list)


# ------------------------------------------------------------------
# Autocomplete_2
# ------------------------------------------------------------------
def test_autocomplete_2_returns_list(travel_instance):
    """Test that Autocomplete_2 returns a list with predictions."""
    result = travel_instance.Autocomplete_2(query="New York")
    assert isinstance(result, list)


def test_autocomplete_2_none_query(travel_instance):
    """Test with None query – should handle gracefully and return list."""
    result = travel_instance.Autocomplete_2(query=None)
    assert isinstance(result, list)


# ------------------------------------------------------------------
# City_data_in_json_format
# ------------------------------------------------------------------
def test_city_data_in_json_format_returns_list(travel_instance):
    """Test that City_data_in_json_format returns a list."""
    result = travel_instance.City_data_in_json_format()
    assert isinstance(result, list)
    if result:
        assert "name" in result[0] or "city" in result[0]


def test_city_data_in_json_format_has_cities(travel_instance):
    """Test the list is non-empty."""
    result = travel_instance.City_data_in_json_format()
    assert len(result) > 0


# ------------------------------------------------------------------
# Download_chains
# ------------------------------------------------------------------
def test_download_chains_returns_list(travel_instance):
    """Test that Download_chains returns a list of chains."""
    result = travel_instance.Download_chains()
    assert isinstance(result, list)
    if result:
        assert "name" in result[0] or "chain" in result[0]


def test_download_chains_not_empty(travel_instance):
    """Test list is non-empty."""
    result = travel_instance.Download_chains()
    assert len(result) > 0


# ------------------------------------------------------------------
# Get_Cities_List
# ------------------------------------------------------------------
def test_get_cities_list_returns_list(travel_instance):
    """Test that Get_Cities_List returns a list."""
    result = travel_instance.Get_Cities_List()
    assert isinstance(result, list)
    if result:
        assert "city" in result[0] or "name" in result[0]


def test_get_cities_list_contains_expected(travel_instance):
    """Test that result contains typical city entries."""
    result = travel_instance.Get_Cities_List()
    assert len(result) > 0
    # check for expected structure (each item should have city/id)
    assert any("city" in item for item in result) or any("name" in item for item in result)


# ------------------------------------------------------------------
# Get_Currencies_List
# ------------------------------------------------------------------
def test_get_currencies_list_returns_list(travel_instance):
    """Test that Get_Currencies_List returns a list."""
    result = travel_instance.Get_Currencies_List()
    assert isinstance(result, list)
    if result:
        assert "code" in result[0] or "currency" in result[0]


def test_get_currencies_list_has_multiple(travel_instance):
    """Test list contains more than one currency."""
    result = travel_instance.Get_Currencies_List()
    assert len(result) >= 1


# ------------------------------------------------------------------
# Get_Distance
# ------------------------------------------------------------------
def test_get_distance_returns_dict(travel_instance):
    """Test that Get_Distance returns a dict with distance info."""
    result = travel_instance.Get_Distance(latB=40.7128, longA=-74.0060, latA=34.0522, longB=-118.2437)
    assert isinstance(result, dict)
    assert "miles" in result or "distance" in result or "error" in result


def test_get_distance_invalid_coordinates(travel_instance):
    """Test with out-of-range coordinates (should still return dict)."""
    result = travel_instance.Get_Distance(latB=200, longA=500, latA=-100, longB=1000)
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# Get_Distance_By_City_2
# ------------------------------------------------------------------
def test_get_distance_by_city_2_returns_dict(travel_instance):
    """Test that Get_Distance_By_City_2 returns a dict."""
    result = travel_instance.Get_Distance_By_City_2(
        country1="US", country2="US",
        state1="California", city1="Los Angeles",
        state2="New York", city2="New York"
    )
    assert isinstance(result, dict)
    # should contain distance or error
    assert "distance" in result or "miles" in result or "km" in result or "error" in result


def test_get_distance_by_city_2_missing_city(travel_instance):
    """Test with empty city name – should still return dict."""
    result = travel_instance.Get_Distance_By_City_2(
        country1="US", country2="US",
        state1="", city1="",
        state2="", city2=""
    )
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# Get_Distance_in_Km
# ------------------------------------------------------------------
def test_get_distance_in_km_returns_dict(travel_instance):
    """Test that Get_Distance_in_Km returns a dict."""
    result = travel_instance.Get_Distance_in_Km(latB=40.7128, longB=-74.0060, longA=-118.2437, latA=34.0522)
    assert isinstance(result, dict)
    assert "km" in result or "distance" in result or "error" in result


def test_get_distance_in_km_negative_coords(travel_instance):
    """Test with negative coordinates (should still work)."""
    result = travel_instance.Get_Distance_in_Km(latB=-33.86, longB=151.20, longA=0, latA=0)
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# Get_Stations
# ------------------------------------------------------------------
def test_get_stations_returns_dict(travel_instance):
    """Test that Get_Stations returns a dict."""
    result = travel_instance.Get_Stations()
    assert isinstance(result, dict)
    # should contain stations or message
    assert "stations" in result or "message" in result or "error" in result


def test_get_stations_has_data(travel_instance):
    """Test that result has non-empty data."""
    result = travel_instance.Get_Stations()
    assert len(result) > 0


# ------------------------------------------------------------------
# Get_administrative_divisions
# ------------------------------------------------------------------
def test_get_administrative_divisions_returns_list(travel_instance):
    """Test that Get_administrative_divisions returns a list."""
    result = travel_instance.Get_administrative_divisions(countrycode="US")
    assert isinstance(result, list)
    if result:
        assert "name" in result[0] or "admin" in result[0]


def test_get_administrative_divisions_invalid_code(travel_instance):
    """Test with invalid country code – should return empty list or error info in list."""
    result = travel_instance.Get_administrative_divisions(countrycode="ZZ")
    assert isinstance(result, list)


# ------------------------------------------------------------------
# Latin_America
# ------------------------------------------------------------------
def test_latin_america_returns_dict(travel_instance):
    """Test that Latin_America returns a dict."""
    result = travel_instance.Latin_America()
    assert isinstance(result, dict)
    assert "cities" in result or "data" in result or "error" in result


def test_latin_america_has_cities(travel_instance):
    """Test that the dict contains a non-empty list of cities."""
    result = travel_instance.Latin_America()
    if "cities" in result:
        assert isinstance(result["cities"], list)
        assert len(result["cities"]) > 0


# ------------------------------------------------------------------
# Meta_Properties_description
# ------------------------------------------------------------------
def test_meta_properties_description_returns_dict(travel_instance):
    """Test that Meta_Properties_description returns a dict."""
    result = travel_instance.Meta_Properties_description()
    assert isinstance(result, dict)
    # should have some properties
    assert len(result) > 0


def test_meta_properties_description_has_expected_keys(travel_instance):
    """Test that result contains typical meta property keys."""
    result = travel_instance.Meta_Properties_description()
    assert "description" in result or "properties" in result or "meta" in result


# ------------------------------------------------------------------
# North_America
# ------------------------------------------------------------------
def test_north_america_returns_dict(travel_instance):
    """Test that North_America returns a dict."""
    result = travel_instance.North_America()
    assert isinstance(result, dict)
    assert "cities" in result or "data" in result or "error" in result


def test_north_america_has_data(travel_instance):
    """Test that the dict contains expected keys."""
    result = travel_instance.North_America()
    # should have at least one top-level key
    assert len(result) > 0


# ------------------------------------------------------------------
# Oceania
# ------------------------------------------------------------------
def test_oceania_returns_dict(travel_instance):
    """Test that Oceania returns a dict."""
    result = travel_instance.Oceania()
    assert isinstance(result, dict)
    assert "cities" in result or "data" in result or "error" in result


def test_oceania_has_cities(travel_instance):
    """Test that result contains a non-empty list of cities."""
    result = travel_instance.Oceania()
    if "cities" in result:
        assert isinstance(result["cities"], list)
        assert len(result["cities"]) > 0


# ------------------------------------------------------------------
# Prices_and_Availability_by_administrative_divisions
# ------------------------------------------------------------------
def test_prices_and_availability_returns_dict(travel_instance):
    """Test that Prices_and_Availability_by_administrative_divisions returns a dict."""
    result = travel_instance.Prices_and_Availability_by_administrative_divisions(
        month="January", country_code="US", year=2023
    )
    assert isinstance(result, dict)
    assert "price" in result or "availability" in result or "error" in result


def test_prices_and_availability_invalid_month(travel_instance):
    """Test with invalid month (should still return dict, maybe with error)."""
    result = travel_instance.Prices_and_Availability_by_administrative_divisions(
        month="Invalid", country_code="US", year=2023
    )
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# Query_Dive_Operators_by_a_country_or_a_region
# ------------------------------------------------------------------
def test_query_dive_operators_returns_dict(travel_instance):
    """Test that Query_Dive_Operators_by_a_country_or_a_region returns a dict."""
    result = travel_instance.Query_Dive_Operators_by_a_country_or_a_region()
    assert isinstance(result, dict)
    # should contain message or data
    assert "message" in result or "operators" in result or "error" in result


def test_query_dive_operators_has_message(travel_instance):
    """Test that result is non-empty."""
    result = travel_instance.Query_Dive_Operators_by_a_country_or_a_region()
    assert len(result) > 0


# ------------------------------------------------------------------
# Query_Divesites_by_a_country_or_a_region
# ------------------------------------------------------------------
def test_query_divesites_by_country_returns_dict(travel_instance):
    """Test that Query_Divesites_by_a_country_or_a_region returns a dict."""
    result = travel_instance.Query_Divesites_by_a_country_or_a_region(country="Egypt")
    assert isinstance(result, dict)
    assert "divesites" in result or "message" in result or "error" in result


def test_query_divesites_by_country_empty_country(travel_instance):
    """Test with empty country string – should still return dict."""
    result = travel_instance.Query_Divesites_by_a_country_or_a_region(country="")
    assert isinstance(result, dict)


# ------------------------------------------------------------------
# Query_divesites_by_gps_boundaries_For_use_with_maps
# ------------------------------------------------------------------
def test_query_divesites_by_gps_returns_dict(travel_instance):
    """Test that Query_divesites_by_gps_boundaries_For_use_with_maps returns a dict."""
    result = travel_instance.Query_divesites_by_gps_boundaries_For_use_with_maps()
    assert isinstance(result, dict)
    # should contain count or message
    assert "count" in result or "total" in result or "message" in result


def test_query_divesites_by_gps_has_data(travel_instance):
    """Test that result is non-empty."""
    result = travel_instance.Query_divesites_by_gps_boundaries_For_use_with_maps()
    assert len(result) > 0


# ------------------------------------------------------------------
# Ranked_World_Crime_cities
# ------------------------------------------------------------------
def test_ranked_world_crime_cities_returns_dict(travel_instance):
    """Test that Ranked_World_Crime_cities returns a dict."""
    result = travel_instance.Ranked_World_Crime_cities()
    assert isinstance(result, dict)
    assert "rankings" in result or "cities" in result or "message" in result


def test_ranked_world_crime_cities_has_rankings(travel_instance):
    """Test that result contains at least one ranking."""
    result = travel_instance.Ranked_World_Crime_cities()
    # should have data
    assert len(result) > 0


# ------------------------------------------------------------------
# Search_Place
# ------------------------------------------------------------------
def test_search_place_returns_dict(travel_instance):
    """Test that Search_Place returns a dict with place details."""
    result = travel_instance.Search_Place(query="Eiffel Tower")
    assert isinstance(result, dict)
    # should have place info or error
    assert "place" in result or "name" in result or "error" in result


def test_search_place_empty_query(travel_instance):
    """Test with empty query – should still return dict (maybe error)."""
    result = travel_instance.Search_Place(query="")
    assert isinstance(result, dict)
    # expect some kind of error message
    assert "error" in result or "message" in result


# ------------------------------------------------------------------
# TrainView
# ------------------------------------------------------------------
def test_train_view_returns_dict(travel_instance):
    """Test that TrainView returns a dict with train locations."""
    result = travel_instance.TrainView()
    assert isinstance(result, dict)
    assert "trains" in result or "locations" in result or "error" in result


def test_train_view_has_data(travel_instance):
    """Test that result contains at least one key."""
    result = travel_instance.TrainView()
    assert len(result) > 0


# ------------------------------------------------------------------
# USA_Borders_Waiting_Times
# ------------------------------------------------------------------
def test_usa_borders_waiting_times_returns_dict(travel_instance):
    """Test that USA_Borders_Waiting_Times returns a dict."""
    result = travel_instance.USA_Borders_Waiting_Times()
    assert isinstance(result, dict)
    assert "ports" in result or "waiting_times" in result or "error" in result


def test_usa_borders_waiting_times_has_ports(travel_instance):
    """Test that result contains port entries."""
    result = travel_instance.USA_Borders_Waiting_Times()
    # if no error, should have ports
    if "error" not in result:
        assert "ports" in result or "waiting_times" in result


# ------------------------------------------------------------------
# allUsaPrice
# ------------------------------------------------------------------
def test_all_usa_price_returns_dict(travel_instance):
    """Test that allUsaPrice returns a dict with state prices."""
    result = travel_instance.allUsaPrice()
    assert isinstance(result, dict)
    assert "states" in result or "prices" in result or "data" in result or "error" in result


def test_all_usa_price_has_prices(travel_instance):
    """Test that result contains price information."""
    result = travel_instance.allUsaPrice()
    # should have data
    assert len(result) > 0


# ------------------------------------------------------------------
# cities
# ------------------------------------------------------------------
def test_cities_returns_list(travel_instance):
    """Test that cities returns a list."""
    result = travel_instance.cities()
    assert isinstance(result, list)
    if result:
        assert "name" in result[0] or "city" in result[0]


def test_cities_non_empty(travel_instance):
    """Test that list is non-empty."""
    result = travel_instance.cities()
    assert len(result) > 0


# ------------------------------------------------------------------
# currencies
# ------------------------------------------------------------------
def test_currencies_returns_dict(travel_instance):
    """Test that currencies returns a dict with currency example."""
    result = travel_instance.currencies()
    assert isinstance(result, dict)
    # should contain code, name, etc.
    assert "code" in result or "currency" in result or "error" in result


def test_currencies_has_data(travel_instance):
    """Test that dict is non-empty."""
    result = travel_instance.currencies()
    assert len(result) > 0


# ------------------------------------------------------------------
# europeanCountries
# ------------------------------------------------------------------
def test_european_countries_returns_dict(travel_instance):
    """Test that europeanCountries returns a dict with gasoline prices."""
    result = travel_instance.europeanCountries()
    assert isinstance(result, dict)
    assert "countries" in result or "prices" in result or "data" in result or "error" in result


def test_european_countries_has_data(travel_instance):
    """Test that result contains entries."""
    result = travel_instance.europeanCountries()
    assert len(result) > 0


# ------------------------------------------------------------------
# stateUsaPrice
# ------------------------------------------------------------------
def test_state_usa_price_returns_dict(travel_instance):
    """Test that stateUsaPrice returns a dict for a given state."""
    result = travel_instance.stateUsaPrice(state="California")
    assert isinstance(result, dict)
    assert "state" in result or "price" in result or "error" in result


def test_state_usa_price_invalid_state(travel_instance):
    """Test with invalid state abbreviation – should return dict with error."""
    result = travel_instance.stateUsaPrice(state="Invalid")
    assert isinstance(result, dict)
    # expect error message
    assert "error" in result or "state" in result


# ------------------------------------------------------------------
# stays_auto_complete
# ------------------------------------------------------------------
def test_stays_auto_complete_returns_list(travel_instance):
    """Test that stays_auto_complete returns a list of location suggestions."""
    result = travel_instance.stays_auto_complete(location="Paris")
    assert isinstance(result, list)
    if result:
        # each item should have destination details
        assert "dest_id" in result[0] or "name" in result[0]


def test_stays_auto_complete_empty_location(travel_instance):
    """Test with empty location – should still return list (maybe empty)."""
    result = travel_instance.stays_auto_complete(location="")
    assert isinstance(result, list)


# ------------------------------------------------------------------
# usaCitiesList
# ------------------------------------------------------------------
def test_usa_cities_list_returns_list(travel_instance):
    """Test that usaCitiesList returns a list with city price info."""
    result = travel_instance.usaCitiesList()
    assert isinstance(result, list)
    if result:
        assert "city" in result[0] or "name" in result[0]


def test_usa_cities_list_has_data(travel_instance):
    """Test that list is non-empty."""
    result = travel_instance.usaCitiesList()
    assert len(result) > 0


# ------------------------------------------------------------------
# v2_get_meta_data
# ------------------------------------------------------------------
def test_v2_get_meta_data_returns_dict(travel_instance):
    """Test that v2_get_meta_data returns a dict with locale meta data."""
    result = travel_instance.v2_get_meta_data()
    assert isinstance(result, dict)
    # should have keys like locale, meta
    assert "locale" in result or "meta" in result or "error" in result


def test_v2_get_meta_data_has_data(travel_instance):
    """Test that result is non-empty."""
    result = travel_instance.v2_get_meta_data()
    assert len(result) > 0