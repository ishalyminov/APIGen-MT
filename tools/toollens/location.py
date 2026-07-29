"""Auto-generated LocationTools implementation."""

import json
import math
import re
import copy
import datetime
import random
from typing import List, Dict, Any, Optional, Tuple, Union


class LocationTools:
    """Location-related tools for geocoding, timezone, IP lookup, etc."""

    METHOD_NAME_MAP = {
        '/v1/timezone': 'v1_timezone',
        'All': 'All',
        'All German Cities': 'All_German_Cities',
        'All communes': 'All_communes',
        'Calculate distance  By Lat & Long': 'Calculate_distance_By_Lat_Long',
        'Capital By Country': 'Capital_By_Country',
        'Countries All (min)': 'Countries_All_min',
        'Current time by Specific IP': 'Current_time_by_Specific_IP',
        'Directions Between 2 Locations': 'Directions_Between_2_Locations',
        'Filter German Cities': 'Filter_German_Cities',
        'Geo Ping Global IP lookup': 'Geo_Ping_Global_IP_lookup',
        'Get All Cities in Vietnam': 'Get_All_Cities_in_Vietnam',
        'Get Time Zones': 'Get_Time_Zones',
        'Get ZIP Info': 'Get_ZIP_Info',
        'Get a list of suburbs': 'Get_a_list_of_suburbs',
        'Get all suburbs and postcodes in a radius': 'Get_all_suburbs_and_postcodes_in_a_radius',
        'Get the cities': 'Get_the_cities',
        'IP Geolocation Lookup': 'IP_Geolocation_Lookup',
        'IP-Locator': 'IP_Locator',
        'Income By Zipcode': 'Income_By_Zipcode',
        'Nearest Metro Station': 'Nearest_Metro_Station',
        'Reverse Geocode': 'Reverse_Geocode',
        'Reverse Geocoding': 'Reverse_Geocoding',
        'ReverseGeocode': 'ReverseGeocode',
        'State by id': 'State_by_id',
        'TZ Lookup by Location': 'TZ_Lookup_by_Location',
        'Timezone for Location': 'Timezone_for_Location',
        'Wilaya_Informations': 'Wilaya_Informations',
        'address_code': 'address_code',
        'address_postal_code': 'address_postal_code',
        'cities By State': 'cities_By_State',
        'continents': 'continents',
        'countiesSearchInRadius': 'countiesSearchInRadius',
        'datum_conversion': 'datum_conversion',
        'emoji Flag By Country': 'emoji_Flag_By_Country',
        'findpincodebydistrict': 'findpincodebydistrict',
        'findpincodebysubdistrict': 'findpincodebysubdistrict',
        'format': 'format',
        'get capital': 'get_capital',
        'get geo': 'get_geo',
        'getZIP': 'getZIP',
    }

    # Common data for deterministic responses
    _country_capitals = {
        'US': 'Washington, D.C.',
        'FR': 'Paris',
        'JP': 'Tokyo',
        'DE': 'Berlin',
        'VN': 'Hanoi',
        'IN': 'New Delhi',
    }

    _country_geo = {
        'US': '{"status":"ok","data":"USA"}',
        'FR': '{"status":"ok","data":"France"}',
        'JP': '{"status":"ok","data":"Japan"}',
    }

    _german_cities = [
        {'city': 'Berlin', 'district': 'Berlin', 'lat': 52.5200, 'long': 13.4050, 'postal_code': 10115, 'postal_code_5': '10115', 'state': 'Berlin'},
        {'city': 'München', 'district': 'Oberbayern', 'lat': 48.1351, 'long': 11.5820, 'postal_code': 80331, 'postal_code_5': '80331', 'state': 'Bayern'},
        {'city': 'Hamburg', 'district': 'Hamburg', 'lat': 53.5511, 'long': 9.9937, 'postal_code': 20095, 'postal_code_5': '20095', 'state': 'Hamburg'},
    ]

    _vietnam_cities = [
        {'code': 'HN', 'name': 'Hanoi', 'unit': 'Thành phố'},
        {'code': 'HCM', 'name': 'Ho Chi Minh City', 'unit': 'Thành phố'},
        {'code': 'DN', 'name': 'Da Nang', 'unit': 'Thành phố'},
    ]

    _timezones_list = [
        {"page": 1, "total_items": 2, "total_pages": 1, "total": 2},
    ]

    _all_communes = [
        {"province": "Brabant Wallon", "nom": "Wavre", "nomMinus": "wavre", "codeCom": 2512, "codePost": 1300},
        {"province": "Hainaut", "nom": "Mons", "nomMinus": "mons", "codeCom": 5305, "codePost": 7000},
    ]

    _pharmacies = [
        {"date": "2025-03-15", "nom": "Pharmacie Centrale", "type": "Garde"},
        {"date": "2025-03-15", "nom": "Pharmacie des Lilas", "type": "Garde"},
    ]

    _continents = [
        {"name": "Africa", "code": "AF", "area_km2": 30370000},
        {"name": "Europe", "code": "EU", "area_km2": 10180000},
        {"name": "Asia", "code": "AS", "area_km2": 44579000},
    ]

    _suburbs_data = {
        2000: [{"suburb": "Sydney", "state": "NSW", "latitude": "-33.8688", "longitude": "151.2093"}],
        3000: [{"suburb": "Melbourne", "state": "VIC", "latitude": "-37.8136", "longitude": "144.9631"}],
    }

    def __init__(self, initial_config: dict = None):
        """Initialize the tools with optional configuration."""
        if initial_config is None:
            self._config_data = {}
        else:
            self._config_data = copy.deepcopy(initial_config)
        # internal cache if needed
        self._cache = {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _compute_distance(self, lat1: float, lon1: float, lat2: float, lon2: float, metric: str = 'km') -> float:
        """Approximate distance using haversine formula."""
        R = 6371  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = R * c
        if metric.lower() in ('mi', 'mile', 'miles'):
            distance_km *= 0.621371
        return round(distance_km, 2)

    # ------------------------------------------------------------------
    # /v1/timezone -> v1_timezone
    # ------------------------------------------------------------------
    def v1_timezone(self, lat: float = None, lon: float = None, city: str = None, state: str = None, country: str = None) -> Dict[str, Any]:
        """Return timezone for given location."""
        # If city is provided, use it; else use lat/lon
        if city:
            if 'London' in city:
                tz = 'Europe/London'
                out_city = 'London'
            elif 'New York' in city:
                tz = 'America/New_York'
                out_city = 'New York'
            elif 'Paris' in city:
                tz = 'Europe/Paris'
                out_city = 'Paris'
            else:
                tz = 'UTC'
                out_city = city
            return {"timezone": tz, "city": out_city}
        elif lat is not None and lon is not None:
            # approximate timezone from longitude
            offset = round(lon / 15)
            tz = f'Etc/GMT{offset:+d}' if offset != 0 else 'UTC'
            return {"timezone": tz, "city": "Unknown"}
        else:
            return {"error": "Either (city/state/country) or (lat,lon) must be provided."}

    # ------------------------------------------------------------------
    # All
    # ------------------------------------------------------------------
    def All(self) -> Dict[str, Any]:
        """Return a pharmacy on duty."""
        pharm = self._pharmacies[0]
        return {"date": pharm["date"], "nom": pharm["nom"], "type": pharm["type"]}

    # ------------------------------------------------------------------
    # All German Cities
    # ------------------------------------------------------------------
    def All_German_Cities(self) -> Dict[str, Any]:
        """Return first German city from internal list."""
        if self._german_cities:
            return self._german_cities[0]
        return {"city": "", "district": "", "lat": 0, "long": 0, "postal_code": 0, "postal_code_5": "", "state": ""}

    # ------------------------------------------------------------------
    # All communes
    # ------------------------------------------------------------------
    def All_communes(self) -> Dict[str, Any]:
        """Return a commune."""
        if self._all_communes:
            return self._all_communes[0]
        return {"province": "", "nom": "", "nomMinus": "", "codeCom": 0, "codePost": 0}

    # ------------------------------------------------------------------
    # Calculate distance  By Lat & Long
    # ------------------------------------------------------------------
    def Calculate_distance_By_Lat_Long(self, metric: str, lat2: str, lon2: str, lon1: str, lat1: str) -> Dict[str, Any]:
        """Calculate distance between two coordinates."""
        try:
            lat1_f = float(lat1)
            lon1_f = float(lon1)
            lat2_f = float(lat2)
            lon2_f = float(lon2)
        except (ValueError, TypeError):
            return {"error": "Invalid coordinate values."}
        dist = self._compute_distance(lat1_f, lon1_f, lat2_f, lon2_f, metric)
        return {"distance": dist}

    # ------------------------------------------------------------------
    # Capital By Country
    # ------------------------------------------------------------------
    def Capital_By_Country(self, country: str) -> Dict[str, Any]:
        """Return capital of a country."""
        mapping = {
            'France': ('France', 'FR', 'Paris'),
            'Japan': ('Japan', 'JP', 'Tokyo'),
            'United States': ('United States', 'US', 'Washington, D.C.'),
            'Germany': ('Germany', 'DE', 'Berlin'),
            'Vietnam': ('Vietnam', 'VN', 'Hanoi'),
            'India': ('India', 'IN', 'New Delhi'),
        }
        data = mapping.get(country, (country, '??', 'Unknown'))
        return {"countryName": data[0], "CountryCode": data[1], "Capital": data[2]}

    # ------------------------------------------------------------------
    # Countries All (min)
    # ------------------------------------------------------------------
    def Countries_All_min(self) -> Dict[str, Any]:
        """Return minimized info for a country."""
        return {"code": "US", "nameEngCommon": "United States of America", "nameNativeCommon": "United States"}

    # ------------------------------------------------------------------
    # Current time by Specific IP
    # ------------------------------------------------------------------
    def Current_time_by_Specific_IP(self, ipv4: str) -> Dict[str, Any]:
        """Return current time for given IP."""
        now = datetime.datetime.now(datetime.timezone.utc)
        return {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "UTC",
            "timestamp": int(now.timestamp())
        }

    # ------------------------------------------------------------------
    # Directions Between 2 Locations
    # ------------------------------------------------------------------
    def Directions_Between_2_Locations(self, end_lat: float, end_lon: float, start_lat: float, start_lon: float) -> Dict[str, Any]:
        """Return distance and duration between two locations."""
        dist = round(self._compute_distance(start_lat, start_lon, end_lat, end_lon, 'km'), 2)
        # rough duration: assume 50 km/h average speed
        duration_hours = dist / 50 if dist > 0 else 0
        return {
            "distance": f"{dist} km",
            "duration": f"{duration_hours * 60:.1f} minutes"
        }

    # ------------------------------------------------------------------
    # Filter German Cities
    # ------------------------------------------------------------------
    def Filter_German_Cities(self) -> Dict[str, Any]:
        """Return a German city (simulate filter)."""
        if self._german_cities:
            return self._german_cities[0]
        return {"city": "", "district": "", "lat": 0, "long": 0, "postal_code": 0, "postal_code_5": "", "state": ""}

    # ------------------------------------------------------------------
    # Geo Ping Global IP lookup
    # ------------------------------------------------------------------
    def Geo_Ping_Global_IP_lookup(self, domain: str) -> List[Dict[str, Any]]:
        """Return ping results from global servers."""
        servers = [
            {"city": "New York", "country": "US", "response_time_ms": 45, "local_ip": "10.0.0.1"},
            {"city": "London", "country": "UK", "response_time_ms": 85, "local_ip": "10.0.0.2"},
            {"city": "Tokyo", "country": "JP", "response_time_ms": 150, "local_ip": "10.0.0.3"},
        ]
        return servers

    # ------------------------------------------------------------------
    # Get All Cities in Vietnam
    # ------------------------------------------------------------------
    def Get_All_Cities_in_Vietnam(self) -> Dict[str, Any]:
        """Return first city in Vietnam."""
        if self._vietnam_cities:
            return self._vietnam_cities[0]
        return {"code": "", "name": "", "unit": ""}

    # ------------------------------------------------------------------
    # Get Time Zones
    # ------------------------------------------------------------------
    def Get_Time_Zones(self) -> Dict[str, Any]:
        """Return time zone list info."""
        return self._timezones_list[0] if self._timezones_list else {"page": 0, "total_items": 0, "total_pages": 0, "total": 0}

    # ------------------------------------------------------------------
    # Get ZIP Info
    # ------------------------------------------------------------------
    def Get_ZIP_Info(self, zipcode: str) -> Dict[str, Any]:
        """Return ZIP code information."""
        data = {
            "ZipCode": zipcode,
            "City": "Sample City",
            "State": "CA",
            "County": "Sample County",
            "AreaCode": "123",
            "CityType": "D",
            "CityAliasAbbreviation": "SC",
            "CityAliasName": "SampleCity",
            "Latitude": "34.0522",
            "Longitude": "-118.2437",
            "TimeZone": "America/Los_Angeles",
            "Elevation": "100",
            "CountyFIPS": "06037",
            "DayLightSaving": "Y",
            "PreferredLastLineKey": "XXXX",
            "ClassificationCode": "U",
            "MultiCounty": "N",
            "StateFIPS": "06",
            "CityDeliveryIndicator": "Y",
            "CarrierRouteIndicator": "N",
            "CarrierRoute": "C000",
            "FinanceNumber": "12345",
            "PostNetBarCode": "90000",
            "ZipCodeType": "S",
            "AlternateZipCodes": "90000-1234",
            "IsActive": "Y",
            "StateAbbreviation": "CA",
            "CountyNumber": "37",
            "FacilityCode": "0",
            "CityMixedCase": "Sample City",
            "CityAbbreviation": "SMPL",
            "WorldRegion": "NA",
            "CountryCode": "US"
        }
        return data

    # ------------------------------------------------------------------
    # Get a list of suburbs
    # ------------------------------------------------------------------
    def Get_a_list_of_suburbs(self, postcode: float) -> Dict[str, Any]:
        """Return suburb for a given postcode."""
        pc = int(postcode)
        suburbs = self._suburbs_data.get(pc, [{"suburb": "Unknown", "state": "XX", "latitude": "0", "longitude": "0"}])
        return suburbs[0]

    # ------------------------------------------------------------------
    # Get all suburbs and postcodes in a radius
    # ------------------------------------------------------------------
    def Get_all_suburbs_and_postcodes_in_a_radius(self, lat: str, radius: float, lng: str) -> Dict[str, Any]:
        """Return suburbs within radius."""
        # Return sample
        return {
            "postcode": "2000",
            "suburb": "Sydney",
            "state": "NSW",
            "latitude": "-33.8688",
            "longitude": "151.2093",
            "distance": f"{radius} km"
        }

    # ------------------------------------------------------------------
    # Get the cities
    # ------------------------------------------------------------------
    def Get_the_cities(self) -> Dict[str, Any]:
        """Return a city with nested state and country."""
        return {
            "id": "1",
            "name": "Los Angeles",
            "state": {
                "id": "CA",
                "name": "California",
                "abbreviation": "CA"
            },
            "country": {
                "id": "US",
                "name": "United States",
                "alpha2": "US",
                "alpha3": "USA",
                "number": 840,
                "countryCode": "1"
            }
        }

    # ------------------------------------------------------------------
    # IP Geolocation Lookup
    # ------------------------------------------------------------------
    def IP_Geolocation_Lookup(self, ip: str) -> Dict[str, Any]:
        """Return geolocation for an IP."""
        return {
            "status": "success",
            "ipAddress": ip,
            "country": "United States",
            "countryCode": "US",
            "state": "Virginia",
            "stateCode": "VA",
            "city": "Ashburn",
            "postal": "20149",
            "countryCodeIso3": "USA",
            "continent": "North America",
            "continentCode": "NA",
            "capital": "Washington, D.C.",
            "currency": "USD",
            "currencySymbol": "$",
            "phoneCode": "1",
            "latitude": 39.0438,
            "longitude": -77.4874,
            "isp": "Amazon.com",
            "org": "AWS",
            "asn": "AS14618",
            "asnName": "AMAZON-AES",
            "organization": "Amazon Web Services",
            "timezone": "America/New_York",
            "zip": "20149"
        }

    # ------------------------------------------------------------------
    # IP-Locator
    # ------------------------------------------------------------------
    def IP_Locator(self, ip_address: str, format: str) -> Dict[str, Any]:
        """Return location data for IP."""
        return {
            "ip": ip_address,
            "country_code": "US",
            "country_name": "United States",
            "region_code": None,
            "region_name": "California",
            "city": "Mountain View",
            "zip_code": "94043",
            "time_zone": "America/Los_Angeles",
            "latitude": 37.4223,
            "longitude": -122.0848,
            "metro_code": 807
        }

    # ------------------------------------------------------------------
    # Income By Zipcode
    # ------------------------------------------------------------------
    def Income_By_Zipcode(self, zip: str) -> Dict[str, Any]:
        """Return income data for a zipcode."""
        return {
            "income_data": {
                "Households": 39250,
                "Income100kTo150k": 11200,
                "Income150kTo200k": 5800,
                "Income25To44Years": 14000,
                "Income25To44YearsError": 500,
                "Income25kTo50k": 8200,
                "Income45To64Years": 16500,
                "Income45To64YearsError": 600,
                "Income50kTo75k": 10500,
                "Income65YearsAndOver": 8750,
                "Income65YearsAndOverError": 400,
                "Income75kTo100k": 7200,
                "IncomeUnder10k": 2100,
                "IncomeUnder25k": 5300,
                "Median": 62000,
                "Mean": 85000,
                "PerCapita": 35000,
                "TotalIncome": 3336250000,
                "ZipCode": zip
            }
        }

    # ------------------------------------------------------------------
    # Nearest Metro Station
    # ------------------------------------------------------------------
    def Nearest_Metro_Station(self, long: str, lat: str) -> Dict[str, Any]:
        """Return nearest metro station status."""
        return {"status": "OK"}

    # ------------------------------------------------------------------
    # Reverse Geocode
    # ------------------------------------------------------------------
    def Reverse_Geocode(self, lon: str, lat: str) -> Dict[str, Any]:
        """Reverse geocode coordinates."""
        return {
            "amenity": "Eiffel Tower",
            "category": "tourism",
            "city": "Paris",
            "country": "France",
            "display_name": "Eiffel Tower, Paris, France",
            "region": "Île-de-France",
            "suburb": "7th Arrondissement"
        }

    # ------------------------------------------------------------------
    # Reverse Geocoding
    # ------------------------------------------------------------------
    def Reverse_Geocoding(self, query: str) -> Dict[str, Any]:
        """Translate query into address."""
        return {
            "formatted_address": "1600 Amphitheatre Parkway, Mountain View, CA 94043, USA",
            "street_address": "1600 Amphitheatre Parkway",
            "city": "Mountain View",
            "state": "California",
            "country": "United States",
            "postal_code": "94043",
            "latitude": 37.4223,
            "longitude": -122.0848
        }

    # ------------------------------------------------------------------
    # ReverseGeocode
    # ------------------------------------------------------------------
    def ReverseGeocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """Return text address from lat/lon."""
        return {
            "formatted_address": "350 5th Ave, New York, NY 10118, USA",
            "street_address": "350 5th Ave",
            "city": "New York",
            "state": "New York",
            "country": "United States",
            "postal_code": "10118"
        }

    # ------------------------------------------------------------------
    # State by id
    # ------------------------------------------------------------------
    def State_by_id(self, code: str) -> Dict[str, Any]:
        """Lookup state by ISO 3166-2 code."""
        return {"message": f"State for code {code} is Minnesota"}

    # ------------------------------------------------------------------
    # TZ Lookup by Location
    # ------------------------------------------------------------------
    def TZ_Lookup_by_Location(self, lat: float, lng: float) -> Dict[str, Any]:
        """Return timezone info for coordinates."""
        offset = round(lng / 15)
        return {
            "tz_id": f"Etc/GMT{offset:+d}" if offset != 0 else "Etc/GMT",
            "base_utc_offset": offset * 3600,
            "dst_offset": 0
        }

    # ------------------------------------------------------------------
    # Timezone for Location
    # ------------------------------------------------------------------
    def Timezone_for_Location(self, location: str, area: str) -> Dict[str, Any]:
        """Return timezone details for a location."""
        now = datetime.datetime.utcnow()
        return {
            "abbreviation": "GMT",
            "client_ip": "92.223.89.73",
            "datetime": now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "day_of_week": now.weekday(),
            "day_of_year": now.timetuple().tm_yday,
            "dst": False,
            "dst_from": None,
            "dst_offset": 0,
            "dst_until": None,
            "raw_offset": 0,
            "timezone": "Europe/London",
            "unixtime": int(now.timestamp()),
            "utc_datetime": now.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "utc_offset": "+00:00",
            "week_number": now.isocalendar()[1]
        }

    # ------------------------------------------------------------------
    # Wilaya_Informations
    # ------------------------------------------------------------------
    def Wilaya_Informations(self) -> Dict[str, Any]:
        """Return province information."""
        return {"provinces": ["Alger", "Oran", "Constantine"]}

    # ------------------------------------------------------------------
    # address_code
    # ------------------------------------------------------------------
    def address_code(self, code: str) -> Dict[str, Any]:
        """Return address info from code."""
        return {
            "unit": {
                "datum": "World Geodetic System 1984",
                "coord_unit": "degrees"
            }
        }

    # ------------------------------------------------------------------
    # address_postal_code
    # ------------------------------------------------------------------
    def address_postal_code(self, postal_code: str) -> Dict[str, Any]:
        """Return address info from postal code."""
        return {
            "count": {
                "total": 1,
                "offset": 0,
                "limit": 10
            },
            "unit": {
                "datum": "World Geodetic System 1984",
                "coord_unit": "degrees"
            }
        }

    # ------------------------------------------------------------------
    # cities By State
    # ------------------------------------------------------------------
    def cities_By_State(self, state: str) -> Dict[str, Any]:
        """Return a city from given state."""
        return {
            "CityName": "Sacramento",
            "StateCode": "CA",
            "CountryCode": "US",
            "latitude": "38.5816",
            "longitude": "-121.4944"
        }

    # ------------------------------------------------------------------
    # continents
    # ------------------------------------------------------------------
    def continents(self) -> List[Dict[str, Any]]:
        """Return list of continents."""
        return self._continents[:]

    # ------------------------------------------------------------------
    # countiesSearchInRadius
    # ------------------------------------------------------------------
    def countiesSearchInRadius(self, radius: float, longitude: float, latitude: float) -> Dict[str, Any]:
        """Return counties search results (mock)."""
        return {
            "timestamp": int(datetime.datetime.utcnow().timestamp()),
            "status": 200,
            "error": None,
            "message": "OK",
            "path": f"/counties?lat={latitude}&lon={longitude}&radius={radius}"
        }

    # ------------------------------------------------------------------
    # datum_conversion
    # ------------------------------------------------------------------
    def datum_conversion(self, coord: str, after_datum: str) -> Dict[str, Any]:
        """Convert coordinate datum."""
        # parse coord like "35.624822,139.742121"
        parts = coord.split(',')
        if len(parts) != 2:
            return {"error": "Invalid coord format"}
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            return {"error": "Invalid coord numbers"}
        if after_datum.lower() in ('wgs84', 'tokyo'):
            out_lat = lat
            out_lon = lon
        else:
            out_lat = lat
            out_lon = lon
        return {
            "coord": {
                "lat": out_lat,
                "lon": out_lon
            },
            "unit": {
                "datum": "World Geodetic System 1984",
                "coord_unit": "degrees"
            }
        }

    # ------------------------------------------------------------------
    # emoji Flag By Country
    # ------------------------------------------------------------------
    def emoji_Flag_By_Country(self) -> Dict[str, Any]:
        """Return emoji flag for a country."""
        return {
            "CountryName": "United States",
            "CountryCode": "US",
            "emojiU": "U+1F1FA U+1F1F8"
        }

    # ------------------------------------------------------------------
    # findpincodebydistrict
    # ------------------------------------------------------------------
    def findpincodebydistrict(self) -> Dict[str, Any]:
        """Return mock pincode search result."""
        return {
            "status": 200,
            "message": "Found 5 pincodes",
            "noOfItems": 5
        }

    # ------------------------------------------------------------------
    # findpincodebysubdistrict
    # ------------------------------------------------------------------
    def findpincodebysubdistrict(self) -> Dict[str, Any]:
        """Return mock subdistrict pincode search."""
        return {
            "status": 200,
            "message": "Found 3 pincodes",
            "noOfItems": 3
        }

    # ------------------------------------------------------------------
    # format
    # ------------------------------------------------------------------
    def format(self) -> Dict[str, Any]:
        """Return current format setting."""
        return {"format": "json"}

    # ------------------------------------------------------------------
    # get capital
    # ------------------------------------------------------------------
    def get_capital(self, country_code: str) -> Dict[str, Any]:
        """Return capital data for country."""
        cap = self._country_capitals.get(country_code, "Unknown")
        return {
            "status": "ok",
            "data": f"The capital of {country_code} is {cap}."
        }

    # ------------------------------------------------------------------
    # get geo
    # ------------------------------------------------------------------
    def get_geo(self, country_code: str) -> Dict[str, Any]:
        """Return geo data for country."""
        geo = self._country_geo.get(country_code, '{"status":"ok","data":"Unknown"}')
        return {
            "status": "ok",
            "data": geo
        }

    # ------------------------------------------------------------------
    # getZIP
    # ------------------------------------------------------------------
    def getZIP(self, zip: float) -> Dict[str, Any]:
        """Return status for zip code lookup."""
        return {
            "status": {
                "count": 1,
                "distinct": 1,
                "status": "success"
            }
        }