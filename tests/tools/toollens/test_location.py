import pytest
from tools.toollens.location import LocationTools


@pytest.fixture
def location_instance():
    """Return a stateless LocationTools instance."""
    config = None
    return LocationTools(initial_config=config)


# -----------------------------------------------------------------------------
# v1_timezone
# -----------------------------------------------------------------------------
class TestV1Timezone:
    def test_default(self, location_instance):
        result = location_instance.v1_timezone()
        assert isinstance(result, dict)

    def test_with_params(self, location_instance):
        result = location_instance.v1_timezone(lat=48.8566, lon=2.3522, city="Paris")
        assert isinstance(result, dict)

    def test_with_none_params(self, location_instance):
        result = location_instance.v1_timezone(lat=None, lon=None)
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# All
# -----------------------------------------------------------------------------
class TestAll:
    def test_normal(self, location_instance):
        result = location_instance.All()
        assert isinstance(result, dict)

    def test_edge_no_params(self, location_instance):
        result = location_instance.All()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# All_German_Cities
# -----------------------------------------------------------------------------
class TestAllGermanCities:
    def test_normal(self, location_instance):
        result = location_instance.All_German_Cities()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# All_communes
# -----------------------------------------------------------------------------
class TestAllCommunes:
    def test_normal(self, location_instance):
        result = location_instance.All_communes()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Calculate_distance_By_Lat_Long
# -----------------------------------------------------------------------------
class TestCalculateDistanceByLatLong:
    def test_normal_km(self, location_instance):
        result = location_instance.Calculate_distance_By_Lat_Long(
            metric="km",
            lat2="48.8566",
            lon2="2.3522",
            lon1="-0.1278",
            lat1="51.5074"
        )
        assert isinstance(result, dict)

    def test_normal_mi(self, location_instance):
        result = location_instance.Calculate_distance_By_Lat_Long(
            metric="mi",
            lat2="48.8566",
            lon2="2.3522",
            lon1="-0.1278",
            lat1="51.5074"
        )
        assert isinstance(result, dict)

    def test_edge_invalid_metric(self, location_instance):
        result = location_instance.Calculate_distance_By_Lat_Long(
            metric="xyz",
            lat2="48.8566",
            lon2="2.3522",
            lon1="-0.1278",
            lat1="51.5074"
        )
        assert isinstance(result, dict)
        # should contain error info
        assert "error" in result or "Error" in result


# -----------------------------------------------------------------------------
# Capital_By_Country
# -----------------------------------------------------------------------------
class TestCapitalByCountry:
    def test_normal(self, location_instance):
        result = location_instance.Capital_By_Country(country="France")
        assert isinstance(result, dict)

    def test_unknown_country(self, location_instance):
        result = location_instance.Capital_By_Country(country="Atlantis")
        assert isinstance(result, dict)
        # expect error
        assert "error" in result or "Error" in result

    def test_edge_empty_string(self, location_instance):
        result = location_instance.Capital_By_Country(country="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Countries_All_min
# -----------------------------------------------------------------------------
class TestCountriesAllMin:
    def test_normal(self, location_instance):
        result = location_instance.Countries_All_min()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Current_time_by_Specific_IP
# -----------------------------------------------------------------------------
class TestCurrentTimeBySpecificIP:
    def test_normal(self, location_instance):
        result = location_instance.Current_time_by_Specific_IP(ipv4="8.8.8.8")
        assert isinstance(result, dict)

    def test_invalid_ip(self, location_instance):
        result = location_instance.Current_time_by_Specific_IP(ipv4="not_an_ip")
        assert isinstance(result, dict)
        assert "error" in result or "Error" in result

    def test_edge_empty(self, location_instance):
        result = location_instance.Current_time_by_Specific_IP(ipv4="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Directions_Between_2_Locations
# -----------------------------------------------------------------------------
class TestDirectionsBetween2Locations:
    def test_normal(self, location_instance):
        result = location_instance.Directions_Between_2_Locations(
            end_lat=48.8566,
            end_lon=2.3522,
            start_lat=51.5074,
            start_lon=-0.1278
        )
        assert isinstance(result, dict)

    def test_same_point(self, location_instance):
        result = location_instance.Directions_Between_2_Locations(
            end_lat=40.7128,
            end_lon=-74.0060,
            start_lat=40.7128,
            start_lon=-74.0060
        )
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Filter_German_Cities
# -----------------------------------------------------------------------------
class TestFilterGermanCities:
    def test_normal(self, location_instance):
        result = location_instance.Filter_German_Cities()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Geo_Ping_Global_IP_lookup
# -----------------------------------------------------------------------------
class TestGeoPingGlobalIPLookup:
    def test_normal(self, location_instance):
        result = location_instance.Geo_Ping_Global_IP_lookup(domain="google.com")
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], dict)

    def test_invalid_domain(self, location_instance):
        result = location_instance.Geo_Ping_Global_IP_lookup(domain="invalid_domain_xyz")
        assert isinstance(result, list)
        # may be empty or contain error dict; either way, type is list

    def test_edge_empty(self, location_instance):
        result = location_instance.Geo_Ping_Global_IP_lookup(domain="")
        assert isinstance(result, list)


# -----------------------------------------------------------------------------
# Get_All_Cities_in_Vietnam
# -----------------------------------------------------------------------------
class TestGetAllCitiesInVietnam:
    def test_normal(self, location_instance):
        result = location_instance.Get_All_Cities_in_Vietnam()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Get_Time_Zones
# -----------------------------------------------------------------------------
class TestGetTimeZones:
    def test_normal(self, location_instance):
        result = location_instance.Get_Time_Zones()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Get_ZIP_Info
# -----------------------------------------------------------------------------
class TestGetZIPInfo:
    def test_normal(self, location_instance):
        result = location_instance.Get_ZIP_Info(zipcode="10001")
        assert isinstance(result, dict)

    def test_invalid(self, location_instance):
        result = location_instance.Get_ZIP_Info(zipcode="abc")
        assert isinstance(result, dict)
        # expect error
        assert "error" in result or "Error" in result

    def test_edge_empty(self, location_instance):
        result = location_instance.Get_ZIP_Info(zipcode="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Get_a_list_of_suburbs
# -----------------------------------------------------------------------------
class TestGetAListOfSuburbs:
    def test_normal(self, location_instance):
        result = location_instance.Get_a_list_of_suburbs(postcode=2010.0)
        assert isinstance(result, dict)

    def test_edge_zero(self, location_instance):
        result = location_instance.Get_a_list_of_suburbs(postcode=0.0)
        assert isinstance(result, dict)

    def test_edge_none(self, location_instance):
        result = location_instance.Get_a_list_of_suburbs(postcode=None)
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Get_all_suburbs_and_postcodes_in_a_radius
# -----------------------------------------------------------------------------
class TestGetAllSuburbsAndPostcodesInARadius:
    def test_normal(self, location_instance):
        result = location_instance.Get_all_suburbs_and_postcodes_in_a_radius(
            lat="-33.8675",
            radius=5.0,
            lng="151.2070"
        )
        assert isinstance(result, dict)

    def test_edge_large_radius(self, location_instance):
        result = location_instance.Get_all_suburbs_and_postcodes_in_a_radius(
            lat="51.5074",
            radius=100.0,
            lng="-0.1278"
        )
        assert isinstance(result, dict)

    def test_edge_empty_str(self, location_instance):
        result = location_instance.Get_all_suburbs_and_postcodes_in_a_radius(
            lat="",
            radius=5.0,
            lng=""
        )
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Get_the_cities
# -----------------------------------------------------------------------------
class TestGetTheCities:
    def test_normal(self, location_instance):
        result = location_instance.Get_the_cities()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# IP_Geolocation_Lookup
# -----------------------------------------------------------------------------
class TestIPGeolocationLookup:
    def test_normal(self, location_instance):
        result = location_instance.IP_Geolocation_Lookup(ip="8.8.8.8")
        assert isinstance(result, dict)

    def test_invalid(self, location_instance):
        result = location_instance.IP_Geolocation_Lookup(ip="999.999.999.999")
        assert isinstance(result, dict)
        assert "error" in result or "Error" in result

    def test_edge_empty(self, location_instance):
        result = location_instance.IP_Geolocation_Lookup(ip="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# IP_Locator
# -----------------------------------------------------------------------------
class TestIPLocator:
    def test_normal(self, location_instance):
        result = location_instance.IP_Locator(ip_address="8.8.8.8", format="json")
        assert isinstance(result, dict)

    def test_another_format(self, location_instance):
        result = location_instance.IP_Locator(ip_address="8.8.8.8", format="xml")
        assert isinstance(result, dict)

    def test_invalid_ip(self, location_instance):
        result = location_instance.IP_Locator(ip_address="invalid", format="json")
        assert isinstance(result, dict)
        assert "error" in result or "Error" in result

    def test_edge_empty_ip(self, location_instance):
        result = location_instance.IP_Locator(ip_address="", format="json")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Income_By_Zipcode
# -----------------------------------------------------------------------------
class TestIncomeByZipcode:
    def test_normal(self, location_instance):
        result = location_instance.Income_By_Zipcode(zip="90210")
        assert isinstance(result, dict)

    def test_invalid(self, location_instance):
        result = location_instance.Income_By_Zipcode(zip="00000")
        assert isinstance(result, dict)
        # may contain error
        # just check type

    def test_edge_empty(self, location_instance):
        result = location_instance.Income_By_Zipcode(zip="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Nearest_Metro_Station
# -----------------------------------------------------------------------------
class TestNearestMetroStation:
    def test_normal(self, location_instance):
        result = location_instance.Nearest_Metro_Station(long="2.3522", lat="48.8566")
        assert isinstance(result, dict)

    def test_edge_negative(self, location_instance):
        result = location_instance.Nearest_Metro_Station(long="-74.0060", lat="40.7128")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Reverse_Geocode
# -----------------------------------------------------------------------------
class TestReverseGeocode:
    def test_normal(self, location_instance):
        result = location_instance.Reverse_Geocode(lon="-0.1278", lat="51.5074")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.Reverse_Geocode(lon="", lat="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Reverse_Geocoding
# -----------------------------------------------------------------------------
class TestReverseGeocoding:
    def test_normal(self, location_instance):
        result = location_instance.Reverse_Geocoding(query="1600 Amphitheatre Parkway, Mountain View, CA")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.Reverse_Geocoding(query="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# ReverseGeocode
# -----------------------------------------------------------------------------
class TestReverseGeocodeAlt:
    def test_normal(self, location_instance):
        result = location_instance.ReverseGeocode(lat=51.5074, lon=-0.1278)
        assert isinstance(result, dict)

    def test_edge_zero(self, location_instance):
        result = location_instance.ReverseGeocode(lat=0.0, lon=0.0)
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# State_by_id
# -----------------------------------------------------------------------------
class TestStateById:
    def test_normal(self, location_instance):
        result = location_instance.State_by_id(code="US-CA")
        assert isinstance(result, dict)

    def test_invalid(self, location_instance):
        result = location_instance.State_by_id(code="XX-YYY")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.State_by_id(code="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# TZ_Lookup_by_Location
# -----------------------------------------------------------------------------
class TestTZLookupByLocation:
    def test_normal(self, location_instance):
        result = location_instance.TZ_Lookup_by_Location(lat=48.8566, lng=2.3522)
        assert isinstance(result, dict)

    def test_edge_zeros(self, location_instance):
        result = location_instance.TZ_Lookup_by_Location(lat=0.0, lng=0.0)
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# Timezone_for_Location
# -----------------------------------------------------------------------------
class TestTimezoneForLocation:
    def test_normal(self, location_instance):
        result = location_instance.Timezone_for_Location(location="Europe/Paris", area="Europe")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.Timezone_for_Location(location="", area="")
        assert isinstance(result, dict)

    def test_invalid(self, location_instance):
        result = location_instance.Timezone_for_Location(location="Invalid/Location", area="Invalid")
        assert isinstance(result, dict)
        # error expected
        assert "error" in result or "Error" in result


# -----------------------------------------------------------------------------
# Wilaya_Informations
# -----------------------------------------------------------------------------
class TestWilayaInformations:
    def test_normal(self, location_instance):
        result = location_instance.Wilaya_Informations()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# address_code
# -----------------------------------------------------------------------------
class TestAddressCode:
    def test_normal(self, location_instance):
        result = location_instance.address_code(code="12345")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.address_code(code="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# address_postal_code
# -----------------------------------------------------------------------------
class TestAddressPostalCode:
    def test_normal(self, location_instance):
        result = location_instance.address_postal_code(postal_code="75001")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.address_postal_code(postal_code="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# cities_By_State
# -----------------------------------------------------------------------------
class TestCitiesByState:
    def test_normal(self, location_instance):
        result = location_instance.cities_By_State(state="California")
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.cities_By_State(state="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# continents
# -----------------------------------------------------------------------------
class TestContinents:
    def test_normal(self, location_instance):
        result = location_instance.continents()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# countiesSearchInRadius
# -----------------------------------------------------------------------------
class TestCountiesSearchInRadius:
    def test_normal(self, location_instance):
        result = location_instance.countiesSearchInRadius(
            radius=10.0,
            longitude=-0.1278,
            latitude=51.5074
        )
        assert isinstance(result, dict)

    def test_edge_zero_radius(self, location_instance):
        result = location_instance.countiesSearchInRadius(
            radius=0.0,
            longitude=2.3522,
            latitude=48.8566
        )
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# datum_conversion
# -----------------------------------------------------------------------------
class TestDatumConversion:
    def test_normal(self, location_instance):
        result = location_instance.datum_conversion(
            coord="48.8566N 2.3522E",
            after_datum="WGS84"
        )
        assert isinstance(result, dict)

    def test_edge_empty(self, location_instance):
        result = location_instance.datum_conversion(coord="", after_datum="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# emoji_Flag_By_Country
# -----------------------------------------------------------------------------
class TestEmojiFlagByCountry:
    def test_normal(self, location_instance):
        result = location_instance.emoji_Flag_By_Country()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# findpincodebydistrict
# -----------------------------------------------------------------------------
class TestFindPincodeByDistrict:
    def test_normal(self, location_instance):
        result = location_instance.findpincodebydistrict()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# findpincodebysubdistrict
# -----------------------------------------------------------------------------
class TestFindPincodeBySubdistrict:
    def test_normal(self, location_instance):
        result = location_instance.findpincodebysubdistrict()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# format
# -----------------------------------------------------------------------------
class TestFormat:
    def test_normal(self, location_instance):
        result = location_instance.format()
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# get_capital
# -----------------------------------------------------------------------------
class TestGetCapital:
    def test_normal(self, location_instance):
        result = location_instance.get_capital(country_code="US")
        assert isinstance(result, dict)

    def test_invalid_code(self, location_instance):
        result = location_instance.get_capital(country_code="ZZ")
        assert isinstance(result, dict)
        # expect error
        assert "error" in result or "Error" in result

    def test_edge_empty(self, location_instance):
        result = location_instance.get_capital(country_code="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# get_geo
# -----------------------------------------------------------------------------
class TestGetGeo:
    def test_normal(self, location_instance):
        result = location_instance.get_geo(country_code="FR")
        assert isinstance(result, dict)

    def test_invalid_code(self, location_instance):
        result = location_instance.get_geo(country_code="ZZ")
        assert isinstance(result, dict)
        assert "error" in result or "Error" in result

    def test_edge_empty(self, location_instance):
        result = location_instance.get_geo(country_code="")
        assert isinstance(result, dict)


# -----------------------------------------------------------------------------
# getZIP
# -----------------------------------------------------------------------------
class TestGetZIP:
    def test_normal(self, location_instance):
        result = location_instance.getZIP(zip="10001")
        assert isinstance(result, dict)

    def test_invalid(self, location_instance):
        result = location_instance.getZIP(zip="abc")
        assert isinstance(result, dict)
        assert "error" in result or "Error" in result

    def test_edge_empty(self, location_instance):
        result = location_instance.getZIP(zip="")
        assert isinstance(result, dict)