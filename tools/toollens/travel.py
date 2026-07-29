"""Auto-generated TravelTools implementation."""

import json
import math
import re
import copy
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple, Union


class TravelTools:
    """ToolLens Travel category – provides travel-related API methods."""

    METHOD_NAME_MAP = {
        '/webcams/list/bbox={ne_lat}': 'webcams_list_bbox_ne_lat',
        '/webcams/list/orderby={order}': 'webcams_list_orderby_order',
        '/webcams/list/region={region}[': 'webcams_list_region_region',
        'Airport data in json format': 'Airport_data_in_json_format',
        'Auto complete': 'Auto_complete',
        'Autocomplete': 'Autocomplete_2',
        'City data in json format': 'City_data_in_json_format',
        'Download chains': 'Download_chains',
        'Get Cities List': 'Get_Cities_List',
        'Get Currencies List': 'Get_Currencies_List',
        'Get Distance': 'Get_Distance',
        'Get Distance By City': 'Get_Distance_By_City_2',
        'Get Distance in Km': 'Get_Distance_in_Km',
        'Get Stations': 'Get_Stations',
        'Get administrative divisions': 'Get_administrative_divisions',
        'Latin America': 'Latin_America',
        'Meta Properties description': 'Meta_Properties_description',
        'North America': 'North_America',
        'Oceania': 'Oceania',
        'Prices and Availability by administrative divisions': 'Prices_and_Availability_by_administrative_divisions',
        'Query Dive Operators by a country or a region.': 'Query_Dive_Operators_by_a_country_or_a_region',
        'Query Divesites by a country or a region.': 'Query_Divesites_by_a_country_or_a_region',
        'Query divesites by gps boundaries (For use with maps)': 'Query_divesites_by_gps_boundaries_For_use_with_maps',
        'Ranked World Crime cities': 'Ranked_World_Crime_cities',
        'Search Place': 'Search_Place',
        'TrainView': 'TrainView',
        'USA Borders Waiting Times': 'USA_Borders_Waiting_Times',
        'allUsaPrice': 'allUsaPrice',
        'cities': 'cities',
        'currencies': 'currencies',
        'europeanCountries': 'europeanCountries',
        'stateUsaPrice': 'stateUsaPrice',
        'stays/auto-complete': 'stays_auto_complete',
        'usaCitiesList': 'usaCitiesList',
        'v2/get-meta-data': 'v2_get_meta_data',
    }

    def __init__(self, initial_config: dict = None):
        self._config_data = initial_config if initial_config is not None else {}
        # Example internal collections for methods that need state
        self._airports = [
            {"code": "CDG", "name": "Charles de Gaulle", "city": "Paris", "country": "France"},
            {"code": "LHR", "name": "Heathrow", "city": "London", "country": "UK"},
            {"code": "JFK", "name": "John F Kennedy", "city": "New York", "country": "USA"},
        ]
        self._cities = [
            {"name": "Paris", "country": "France", "code": "PAR"},
            {"name": "London", "country": "UK", "code": "LON"},
            {"name": "Tokyo", "country": "Japan", "code": "TYO"},
        ]
        self._currencies = [
            {"code": "USD", "name": "US Dollar", "symbol": "$"},
            {"code": "EUR", "name": "Euro", "symbol": "€"},
            {"code": "GBP", "name": "British Pound", "symbol": "£"},
        ]
        self._hotel_chains = [
            {"name": "Marriott", "id": "1", "logo": "https://example.com/marriott.png"},
            {"name": "Hilton", "id": "2", "logo": "https://example.com/hilton.png"},
        ]
        self._admin_divisions = [
            {"admin1": "Île-de-France", "admin2": "Paris", "admin3": ""},
            {"admin1": "England", "admin2": "Greater London", "admin3": "London"},
        ]

    # ------------------------------------------------------------------
    # Webcams
    # ------------------------------------------------------------------

    def webcams_list_bbox_ne_lat(self, ne_lat: float, sw_lng: float, sw_lat: float, ne_lng: float) -> list:
        """
        Returns a list of webcams within the specified bounding box.
        """
        # Realistic deterministic mock
        return [
            {
                "id": "wc001",
                "name": "Eiffel Tower Cam",
                "latitude": (ne_lat + sw_lat) / 2,
                "longitude": (ne_lng + sw_lng) / 2,
                "preview_url": "https://example.com/webcam/wc001.jpg",
                "city": "Paris",
                "country": "France",
                "description": "View of the Eiffel Tower"
            },
            {
                "id": "wc002",
                "name": "Big Ben Cam",
                "latitude": (ne_lat + sw_lat) / 2 + 0.01,
                "longitude": (ne_lng + sw_lng) / 2 - 0.01,
                "preview_url": "https://example.com/webcam/wc002.jpg",
                "city": "London",
                "country": "UK",
                "description": "View of Big Ben"
            }
        ]

    def webcams_list_orderby_order(self, sort: str, order: str) -> dict:
        """
        Returns metadata about the webcam list ordering.
        """
        return {
            "total_count": 150,
            "order": order,
            "sort": sort
        }

    def webcams_list_region_region(self, region: str) -> dict:
        """
        Returns metadata about webcam results for the given region.
        """
        return {
            "total_results": 42
        }

    # ------------------------------------------------------------------
    # Airport / City / Chain data (file-style endpoints)
    # ------------------------------------------------------------------

    def Airport_data_in_json_format(self) -> list:
        """Returns a list of airports."""
        return self._airports

    def City_data_in_json_format(self) -> list:
        """Returns a list of cities."""
        return self._cities

    def Download_chains(self) -> list:
        """Returns a list of hotel chains."""
        return self._hotel_chains

    # ------------------------------------------------------------------
    # Autocomplete / Search
    # ------------------------------------------------------------------

    def Auto_complete(self, string: str) -> list:
        """
        Gets airport and city ids for the air product related to words.
        """
        # Mock matching based on substring
        matches = []
        keyword = string.lower()
        for a in self._airports:
            if keyword in a["name"].lower() or keyword in a["city"].lower():
                matches.append({"id": a["code"], "name": a["name"], "type": "airport"})
        for c in self._cities:
            if keyword in c["name"].lower() or keyword in c["country"].lower():
                matches.append({"id": c["code"], "name": c["name"], "type": "city"})
        if not matches:
            matches = [{"id": "CDG", "name": "Paris Charles de Gaulle", "type": "airport"}]
        return matches

    def Autocomplete_2(self, query: str) -> list:
        """
        Search for Flixbus or train stations (unified autocomplete).
        """
        return [
            {"station": "Paris Gare de Lyon", "city": "Paris", "code": "PARGDL"},
            {"station": "London Victoria", "city": "London", "code": "LONVIC"},
            {"station": "Berlin Hauptbahnhof", "city": "Berlin", "code": "BERHBF"},
        ]

    def Search_Place(self, query: str) -> dict:
        """
        Search for a place to get the entityId needed for hotel API.
        """
        return {
            "entityId": "275440",
            "placeName": query.title(),
            "country": "France" if "paris" in query.lower() else "UK"
        }

    # ------------------------------------------------------------------
    # Lists (cities, currencies, etc.)
    # ------------------------------------------------------------------

    def Get_Cities_List(self) -> list:
        """Get a list of all available cities."""
        return [c["name"] for c in self._cities]

    def Get_Currencies_List(self) -> list:
        """Get a list of all available currencies."""
        return self._currencies

    def cities(self) -> list:
        """Get a list of cities."""
        return self._cities

    def currencies(self) -> dict:
        """Get a currency example."""
        # Return a single currency as object as per schema
        return self._currencies[0]

    # ------------------------------------------------------------------
    # Distance calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in miles between two points."""
        R = 3958.8  # Earth radius in miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points."""
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)

    def Get_Distance(self, latB: float, longA: float, latA: float, longB: float) -> dict:
        """
        Returns the circle math distance in miles between (latA, longA) and (latB, longB).
        """
        distance = self._haversine_miles(latA, longA, latB, longB)
        return {"distance": distance}

    def Get_Distance_By_City_2(self, country1: str, country2: str, state2: str, city2: str, city1: str,
                                state1: str) -> dict:
        """
        Takes city, state, country of two locations and returns latitude, longitude, and calculated miles.
        """
        # Mock coordinates for realistic output
        coords = {
            ("Paris", "France"): (48.8566, 2.3522),
            ("London", "UK"): (51.5074, -0.1278),
            ("New York", "USA"): (40.7128, -74.0060),
        }
        lat1, lon1 = coords.get((city1, country1), (40.0, -100.0))
        lat2, lon2 = coords.get((city2, country2), (30.0, -90.0))
        distance_miles = self._haversine_miles(lat1, lon1, lat2, lon2)
        return {
            "latitude1": lat1,
            "longitude1": lon1,
            "latitude2": lat2,
            "longitude2": lon2,
            "distance_miles": distance_miles
        }

    def Get_Distance_in_Km(self, latB: float, longB: float, longA: float, latA: float) -> dict:
        """
        Returns circle math distance in kilometers.
        """
        distance_km = self._haversine_km(latA, longA, latB, longB)
        return {"distance_km": distance_km}

    # ------------------------------------------------------------------
    # EV / Charging / Stations
    # ------------------------------------------------------------------

    def Get_Stations(self) -> dict:
        """Return nearest charging stations info (mock)."""
        return {"message": "Nearest charging station: 3.2 miles away at 123 Main St, Paris."}

    # ------------------------------------------------------------------
    # Administrative divisions
    # ------------------------------------------------------------------

    def Get_administrative_divisions(self, countrycode: str) -> list:
        """Retrieve geographical admin names."""
        return self._admin_divisions

    # ------------------------------------------------------------------
    # Regional cities (Latin America, North America, Oceania)
    # ------------------------------------------------------------------

    def Latin_America(self) -> dict:
        """Get Latin America cities sorted by overall score."""
        return {"message": "Top city: Buenos Aires, score: 82.5"}

    def North_America(self) -> dict:
        """Get North America cities sorted by overall score."""
        return {
            "pagination": {
                "page": 1,
                "size": 20,
                "total": 200
            }
        }

    def Oceania(self) -> dict:
        """Get Oceania cities sorted by overall score."""
        return {
            "pagination": {
                "page": 1,
                "size": 20,
                "total_results": 45,
                "total_pages": 3
            }
        }

    # ------------------------------------------------------------------
    # Prices and Availability
    # ------------------------------------------------------------------

    def Prices_and_Availability_by_administrative_divisions(self, month: str, country_code: str, year: int) -> dict:
        """Retrieve average price, availability, etc. for an administrative division."""
        return {
            "average_price": 185.50,
            "average_price_of_available_properties": 210.00,
            "availability_percent": 72.3,
            "processed_properties_count": 150
        }

    # ------------------------------------------------------------------
    # Dive operators / sites
    # ------------------------------------------------------------------

    def Query_Dive_Operators_by_a_country_or_a_region(self) -> dict:
        """Return dive operators message."""
        return {"message": "Found 12 dive operators in Thailand."}

    def Query_Divesites_by_a_country_or_a_region(self, country: str) -> dict:
        """Return dive sites message for country."""
        return {"message": f"Found 25 dive sites in {country}."}

    def Query_divesites_by_gps_boundaries_For_use_with_maps(self) -> dict:
        """Return total count of dive sites in GPS boundary."""
        return {"total_count": 34}

    # ------------------------------------------------------------------
    # Crime, Borders, Gas
    # ------------------------------------------------------------------

    def Ranked_World_Crime_cities(self) -> dict:
        """Return crime rankings message."""
        return {"message": "Top 5 crime cities: Caracas, Cape Town, San Juan, Mogadishu, Baghdad."}

    def USA_Borders_Waiting_Times(self) -> dict:
        """Return USA border ports waiting times (mock)."""
        return {"ports": [{"name": "San Ysidro", "wait_minutes": 45}, {"name": "El Paso", "wait_minutes": 20}]}

    def allUsaPrice(self) -> dict:
        """Return average current gasoline prices in US states."""
        return {"message": "National average regular: $3.45/gal, midgrade: $3.85, premium: $4.20, diesel: $4.50."}

    def europeanCountries(self) -> dict:
        """Return current gasoline prices at European countries."""
        return {"message": "Germany: 1.80 EUR/L, France: 1.75 EUR/L, UK: 1.65 GBP/L."}

    def stateUsaPrice(self, state: str) -> dict:
        """Return gasoline prices for a given US state."""
        return {
            "state": state,
            "regular_price": 3.45,
            "midgrade_price": 3.85,
            "premium_price": 4.20,
            "diesel_price": 4.50,
            "unit": "USD/gal",
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
        }

    def usaCitiesList(self) -> list:
        """Return list of cities with price information in USA."""
        return [
            {"city": "New York", "state": "NY", "regular_price": 3.50},
            {"city": "Los Angeles", "state": "CA", "regular_price": 4.20},
            {"city": "Chicago", "state": "IL", "regular_price": 3.60}
        ]

    # ------------------------------------------------------------------
    # Meta, TrainView
    # ------------------------------------------------------------------

    def Meta_Properties_description(self) -> dict:
        """Return meta properties description."""
        return {"description": "Meta properties for travel data"}

    def v2_get_meta_data(self) -> dict:
        """Get locale meta data."""
        return {"locale": "en-US", "currency": "USD", "date_format": "MM/dd/yyyy"}

    def TrainView(self) -> dict:
        """Return real-time train locations (mock)."""
        return {
            "lat": "39.9526",
            "lon": "-75.1652",
            "trainno": "383",
            "service": "Regional Rail",
            "dest": "Philadelphia",
            "currentstop": "Suburban Station",
            "nextstop": "30th Street Station",
            "line": "Paoli/Thorndale",
            "consist": "Silverliner V",
            "heading": 180,
            "late": 5,
            "SOURCE": "SEPTA",
            "TRACK": "1",
            "TRACK_CHANGE": "0"
        }

    # ------------------------------------------------------------------
    # stays auto-complete
    # ------------------------------------------------------------------

    def stays_auto_complete(self, location: str) -> list:
        """Auto complete for stays."""
        return [
            {"id": "loc_001", "name": f"Paris, France", "type": "city"},
            {"id": "loc_002", "name": f"London, UK", "type": "city"}
        ]