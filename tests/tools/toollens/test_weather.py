import pytest
import json
from tools.toollens.weather import WeatherTools


@pytest.fixture
def weather_instance():
    config = {
        'api_key': 'YOUR_API_KEY',
        'stations': [
            {'id': '9414290', 'name': 'San Francisco'},
            {'id': '9410170', 'name': 'Los Angeles'}
        ],
        'countries': ['US', 'CA', 'GB', 'AU'],
        'resorts': ['Aspen', 'Whistler', 'Stowe', 'Park City'],
        'earthquakes': [
            {'id': 'us7000abc', 'place': 'California'},
            {'id': 'us7000def', 'place': 'Alaska'}
        ],
        'locations': [
            {'lat': 40.7128, 'lon': -74.006, 'city': 'New York'},
            {'lat': 34.0522, 'lon': -118.2437, 'city': 'Los Angeles'}
        ],
        'zip_codes': ['10001', '90210', '94102'],
        'climate_key': 'ABCD1234',
        'forecasts': {},
        'alerts': []
    }
    return WeatherTools(initial_config=config)


# ----------------------------------------------------------------
# Tests for each method (sanitized names)
# ----------------------------------------------------------------

def test_v1_airquality(weather_instance):
    result = weather_instance.v1_airquality()
    assert isinstance(result, dict)


def test_v1_airquality_error(weather_instance):
    # No parameters, so just call and ensure dict returned
    result = weather_instance.v1_airquality()
    assert isinstance(result, dict)


def test_v1_weather(weather_instance):
    result = weather_instance.v1_weather()
    assert isinstance(result, dict)


def test_m_5_Day_Forecast(weather_instance):
    result = weather_instance.m_5_Day_Forecast(resort="Aspen")
    assert isinstance(result, dict)


def test_m_5_Day_Forecast_edge_none(weather_instance):
    # When resort is None, method should return error dict
    result = weather_instance.m_5_Day_Forecast(resort=None)
    assert isinstance(result, dict)
    # Expect an error key
    assert "error" in result or "status" in result


def test_m_5_Day_Forecast_edge_empty(weather_instance):
    result = weather_instance.m_5_Day_Forecast(resort="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_m_7_Day_Forecast(weather_instance):
    result = weather_instance.m_7_Day_Forecast(lat="40.7128", long="-74.006")
    assert isinstance(result, dict)


def test_m_7_Day_Forecast_edge_invalid(weather_instance):
    result = weather_instance.m_7_Day_Forecast(lat="abc", long="xyz")
    assert isinstance(result, dict)
    # Expect error or status indicating invalid input
    assert "error" in result or "status" in result


def test_Air_Quality_Forecast(weather_instance):
    result = weather_instance.Air_Quality_Forecast(lat=40.7128, lon=-74.006)
    assert isinstance(result, dict)


def test_Air_Quality_Forecast_edge_none(weather_instance):
    result = weather_instance.Air_Quality_Forecast(lat=None, lon=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Air_Quality_History(weather_instance):
    result = weather_instance.Air_Quality_History(lat=40.7128, lon=-74.006)
    assert isinstance(result, dict)


def test_Air_Quality_History_edge_invalid(weather_instance):
    result = weather_instance.Air_Quality_History(lat=111, lon=222)
    assert isinstance(result, dict)
    # The mock may return error for unreasonable coordinates
    assert isinstance(result, dict)


def test_Astronomy_API(weather_instance):
    result = weather_instance.Astronomy_API(q="London", date="2025-04-10")
    assert isinstance(result, dict)


def test_Astronomy_API_edge_missing(weather_instance):
    result = weather_instance.Astronomy_API(q=None, date=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Availability(weather_instance):
    result = weather_instance.Availability(latitude=40.7128, longitude=-74.006)
    assert isinstance(result, dict)


def test_Availability_edge_zero(weather_instance):
    result = weather_instance.Availability(latitude=0, longitude=0)
    assert isinstance(result, dict)


def test_By_Postal_Code(weather_instance):
    result = weather_instance.By_Postal_Code(postalCode=10001)
    assert isinstance(result, dict)


def test_By_Postal_Code_edge_invalid(weather_instance):
    result = weather_instance.By_Postal_Code(postalCode=-1)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Classification(weather_instance):
    result = weather_instance.Classification(lon="-74.006", lat="40.7128")
    assert isinstance(result, dict)


def test_Classification_edge_none(weather_instance):
    result = weather_instance.Classification(lon=None, lat=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Current_Air_Quality(weather_instance):
    result = weather_instance.Current_Air_Quality(lon="-74.006", lat="40.7128")
    assert isinstance(result, dict)


def test_Current_Air_Quality_edge_out_of_range(weather_instance):
    result = weather_instance.Current_Air_Quality(lon="500", lat="500")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Current_Snow_Conditions(weather_instance):
    result = weather_instance.Current_Snow_Conditions(resort="Aspen")
    assert isinstance(result, dict)


def test_Current_Snow_Conditions_edge_unknown(weather_instance):
    result = weather_instance.Current_Snow_Conditions(resort="NonExistent")
    assert isinstance(result, dict)
    # Should return error or empty data
    assert isinstance(result, dict)


def test_Current_conditions_detailed(weather_instance):
    result = weather_instance.Current_conditions_detailed(longitude="-74.006", latitude="40.7128")
    assert isinstance(result, dict)


def test_Current_conditions_detailed_edge_none(weather_instance):
    result = weather_instance.Current_conditions_detailed(longitude=None, latitude=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Get_Weather_Updates(weather_instance):
    result = weather_instance.Get_Weather_Updates(city="London")
    assert isinstance(result, dict)


def test_Get_Weather_Updates_edge_empty(weather_instance):
    result = weather_instance.Get_Weather_Updates(city="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Get_climate_data_by_lat_lon_or_Key(weather_instance):
    result = weather_instance.Get_climate_data_by_lat_lon_or_Key()
    assert isinstance(result, dict)


def test_Get_forecastdata_by_lat_lon(weather_instance):
    result = weather_instance.Get_forecastdata_by_lat_lon(LAT=40.7128, LON=-74.006)
    assert isinstance(result, dict)


def test_Get_forecastdata_by_lat_lon_edge(weather_instance):
    result = weather_instance.Get_forecastdata_by_lat_lon(LAT=999, LON=999)
    assert isinstance(result, dict)
    assert isinstance(result, dict)


def test_Get_stations(weather_instance):
    result = weather_instance.Get_stations()
    assert isinstance(result, list)
    if len(result) > 0:
        assert all(isinstance(item, dict) for item in result)


def test_Historical_daily(weather_instance):
    result = weather_instance.Historical_daily(longitude="-74.006", date="2025-04-10", latitude="40.7128")
    assert isinstance(result, dict)


def test_Historical_daily_edge_none(weather_instance):
    result = weather_instance.Historical_daily(longitude=None, date=None, latitude=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Hourly_Forecast(weather_instance):
    result = weather_instance.Hourly_Forecast(resort="Whistler")
    assert isinstance(result, dict)


def test_Hourly_Forecast_edge_empty(weather_instance):
    result = weather_instance.Hourly_Forecast(resort="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Hourly_forecast_48_hours(weather_instance):
    result = weather_instance.Hourly_forecast_48_hours(latitude="40.7128", longitude="-74.006")
    assert isinstance(result, dict)


def test_Hourly_forecast_48_hours_edge_invalid(weather_instance):
    result = weather_instance.Hourly_forecast_48_hours(latitude="abc", longitude="xyz")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_IP_Lookup_API(weather_instance):
    result = weather_instance.IP_Lookup_API(q="8.8.8.8")
    assert isinstance(result, dict)


def test_IP_Lookup_API_edge_invalid(weather_instance):
    result = weather_instance.IP_Lookup_API(q="not_an_ip")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Latest_Top_15_Earthquake_felt_by_local(weather_instance):
    result = weather_instance.Latest_Top_15_Earthquake_felt_by_local()
    assert isinstance(result, dict)


def test_List_of_all_Countries(weather_instance):
    result = weather_instance.List_of_all_Countries()
    assert isinstance(result, dict)
    # May contain 'countries' or similar key
    assert isinstance(result, dict)


def test_Predict_Feature_Forecast_1_Day(weather_instance):
    result = weather_instance.Predict_Feature_Forecast_1_Day(lat="40.7128", long="-74.006")
    assert isinstance(result, dict)


def test_Predict_Feature_Forecast_1_Day_edge_none(weather_instance):
    result = weather_instance.Predict_Feature_Forecast_1_Day(lat=None, long=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Retrieve_the_Hardiness_Zone(weather_instance):
    result = weather_instance.Retrieve_the_Hardiness_Zone(zipcode="10001")
    assert isinstance(result, dict)


def test_Retrieve_the_Hardiness_Zone_edge_invalid(weather_instance):
    result = weather_instance.Retrieve_the_Hardiness_Zone(zipcode="00000")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Search_API(weather_instance):
    result = weather_instance.Search_API(q="London")
    assert isinstance(result, dict)


def test_Search_API_edge_empty(weather_instance):
    result = weather_instance.Search_API(q="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Search_location_by_Name_or_zip_code_2(weather_instance):
    result = weather_instance.Search_location_by_Name_or_zip_code_2()
    assert isinstance(result, dict)


def test_Search_Autocomplete_API(weather_instance):
    result = weather_instance.Search_Autocomplete_API(q="New")
    assert isinstance(result, dict)


def test_Search_Autocomplete_API_edge_empty(weather_instance):
    result = weather_instance.Search_Autocomplete_API(q="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_Weather_Data(weather_instance):
    result = weather_instance.Weather_Data(start="2025-04-10", lat=40.7128, param="temperature", lon=-74.006, end="2025-04-11")
    assert isinstance(result, dict)


def test_Weather_Data_edge_none(weather_instance):
    result = weather_instance.Weather_Data(start=None, lat=None, param=None, lon=None, end=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_alerts(weather_instance):
    result = weather_instance.alerts()
    assert isinstance(result, dict)


def test_current(weather_instance):
    result = weather_instance.current()
    assert isinstance(result, dict)


def test_historical_weather(weather_instance):
    result = weather_instance.historical_weather(date="2025-04-10")
    assert isinstance(result, dict)


def test_historical_weather_edge_empty(weather_instance):
    result = weather_instance.historical_weather(date="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result


def test_nearest_place(weather_instance):
    result = weather_instance.nearest_place(lon=-74.006, lat=40.7128)
    assert isinstance(result, dict)


def test_nearest_place_edge_no_stations(weather_instance):
    result = weather_instance.nearest_place(lon=200, lat=200)
    assert isinstance(result, dict)
    # Should still return dict (maybe empty or error)
    assert isinstance(result, dict)


def test_predictions(weather_instance):
    result = weather_instance.predictions()
    assert isinstance(result, dict)


def test_sunposition(weather_instance):
    result = weather_instance.sunposition(lat=40.7128, lon=-74.006)
    assert isinstance(result, dict)


def test_sunposition_edge_zero(weather_instance):
    result = weather_instance.sunposition(lat=0, lon=0)
    assert isinstance(result, dict)


def test_weather_report(weather_instance):
    result = weather_instance.weather_report(cityName="London")
    assert isinstance(result, dict)


def test_weather_report_edge_empty(weather_instance):
    result = weather_instance.weather_report(cityName="")
    assert isinstance(result, dict)
    assert "error" in result or "status" in result