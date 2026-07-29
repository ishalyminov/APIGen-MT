import pytest
import json
import copy
from typing import Any, Dict, List
from tools.toollens.medical import MedicalTools


@pytest.fixture
def medical_config() -> dict:
    """Fixture providing a comprehensive initial config for MedicalTools."""
    config = {
        "covid19_cases": [
            {
                "country": "USA",
                "date": "2023-01-15",
                "cases": 1024,
                "deaths": 45
            },
            {
                "country": "India",
                "date": "2023-02-20",
                "cases": 856,
                "deaths": 12
            },
            {
                "country": "Brazil",
                "date": "2023-03-10",
                "cases": 432,
                "deaths": 8
            }
        ],
        "covid19_lookup_dates": [
            "2023-01-15",
            "2023-02-20",
            "2023-03-10"
        ],
        "covid19_lookup_countries": [
            "USA",
            "India",
            "Brazil",
            "Germany",
            "Japan"
        ],
        "covid19_empty": [],
        "brand_to_generic": {
            "Tylenol": "Acetaminophen",
            "Advil": "Ibuprofen",
            "Lipitor": "Atorvastatin",
            "Zoloft": "Sertraline",
            "Nexium": "Esomeprazole",
            "Amoxil": "Amoxicillin",
            "Motrin": "Ibuprofen"
        },
        "brand_lookup_keys": [
            "Tylenol",
            "Advil",
            "Lipitor",
            "Zoloft",
            "Nexium",
            "Amoxil",
            "Motrin",
            "UnknownBrand"
        ],
        "brand_empty": {},
        "tcia_collections": [
            "LIDC-IDRI",
            "TCGA-LUAD",
            "PROSTATEx",
            "QIN-BREAST",
            "REMBRANDT",
            "National Lung Screening Trial"
        ],
        "tcia_collection_values": [
            "LIDC-IDRI",
            "TCGA-LUAD",
            "PROSTATex",
            "QIN-BREAST",
            "REMBRANDT",
            "National Lung Screening Trial"
        ],
        "tcia_empty_collections": []
    }
    return json.loads(json.dumps(config))


@pytest.fixture
def empty_config() -> dict:
    """Fixture providing an empty config for edge case testing."""
    return {}


@pytest.fixture
def medical_tools_factory():
    """Factory fixture to create fresh MedicalTools instances with given config."""
    def _create(config: dict = None) -> MedicalTools:
        return MedicalTools(initial_config=copy.deepcopy(config) if config else None)
    return _create


class TestMedicalToolsSequentialCorrect:
    """Correct ordered sequences exercising typical user trajectories."""

    def test_covid19_then_genericname_then_getCollectionValues(
        self, medical_tools_factory, medical_config
    ):
        """Call all three methods in sequence to verify each returns valid data."""
        tools = medical_tools_factory(medical_config)

        result_covid = tools.v1_covid19()
        assert isinstance(result_covid, dict)
        assert "data" in result_covid or "result" in result_covid or len(result_covid) > 0

        result_generic = tools.genericname()
        assert isinstance(result_generic, dict)
        assert len(result_generic) > 0

        result_collections = tools.getCollectionValues()
        assert isinstance(result_collections, dict)
        assert len(result_collections) > 0

    def test_covid19_data_then_genericname_lookup(
        self, medical_tools_factory, medical_config
    ):
        """Fetch COVID-19 data, then use genericname to look up drug info."""
        tools = medical_tools_factory(medical_config)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        # Verify both calls returned non-error responses
        for result in [covid_result, generic_result]:
            if "error" in result:
                assert result["error"] is None or result["error"] == ""

    def test_genericname_then_getCollectionValues_sequence(
        self, medical_tools_factory, medical_config
    ):
        """Look up generic drug names, then fetch TCIA collection values."""
        tools = medical_tools_factory(medical_config)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        collection_result = tools.getCollectionValues()
        assert isinstance(collection_result, dict)

        # Both should return structured data
        assert generic_result is not None
        assert collection_result is not None

    def test_all_three_methods_repeated_calls(
        self, medical_tools_factory, medical_config
    ):
        """Call all three methods twice to verify state consistency across calls."""
        tools = medical_tools_factory(medical_config)

        covid1 = tools.v1_covid19()
        generic1 = tools.genericname()
        collections1 = tools.getCollectionValues()

        covid2 = tools.v1_covid19()
        generic2 = tools.genericname()
        collections2 = tools.getCollectionValues()

        assert isinstance(covid1, dict) and isinstance(covid2, dict)
        assert isinstance(generic1, dict) and isinstance(generic2, dict)
        assert isinstance(collections1, dict) and isinstance(collections2, dict)

        # Repeated calls should return consistent results
        assert covid1 == covid2
        assert generic1 == generic2
        assert collections1 == collections2

    def test_covid19_then_collections_then_genericname(
        self, medical_tools_factory, medical_config
    ):
        """Exercise methods in a different order to verify independence."""
        tools = medical_tools_factory(medical_config)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        collections_result = tools.getCollectionValues()
        assert isinstance(collections_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        # All results should be non-empty dicts
        assert len(covid_result) > 0
        assert len(collections_result) > 0
        assert len(generic_result) > 0


class TestMedicalToolsSequentialProblematic:
    """Problematic sequences testing error handling and edge cases."""

    def test_empty_config_all_methods_no_crash(
        self, medical_tools_factory, empty_config
    ):
        """Call all methods with empty config; none should crash."""
        tools = medical_tools_factory(empty_config)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        collection_result = tools.getCollectionValues()
        assert isinstance(collection_result, dict)

    def test_none_config_then_all_methods(
        self, medical_tools_factory
    ):
        """Pass None as initial_config; methods should handle gracefully."""
        tools = medical_tools_factory(None)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        collection_result = tools.getCollectionValues()
        assert isinstance(collection_result, dict)

    def test_covid19_empty_data_then_genericname_empty(
        self, medical_tools_factory
    ):
        """Use config with empty data arrays; verify no crashes and sensible responses."""
        config = {
            "covid19_cases": [],
            "covid19_lookup_dates": [],
            "covid19_lookup_countries": [],
            "covid19_empty": [],
            "brand_to_generic": {},
            "brand_lookup_keys": [],
            "brand_empty": {},
            "tcia_collections": [],
            "tcia_collection_values": [],
            "tcia_empty_collections": []
        }
        tools = medical_tools_factory(config)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        collection_result = tools.getCollectionValues()
        assert isinstance(collection_result, dict)

    def test_partial_config_missing_keys(
        self, medical_tools_factory
    ):
        """Config with only some keys present; methods should not crash."""
        config = {
            "covid19_cases": [
                {"country": "USA", "date": "2023-01-15", "cases": 100, "deaths": 5}
            ]
        }
        tools = medical_tools_factory(config)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        collection_result = tools.getCollectionValues()
        assert isinstance(collection_result, dict)

    def test_wrong_order_calls_with_empty_config(
        self, medical_tools_factory
    ):
        """Call methods in reverse order with empty config; verify graceful handling."""
        tools = medical_tools_factory({})

        collection_result = tools.getCollectionValues()
        assert isinstance(collection_result, dict)

        generic_result = tools.genericname()
        assert isinstance(generic_result, dict)

        covid_result = tools.v1_covid19()
        assert isinstance(covid_result, dict)

        # Verify no exceptions were raised and all returned dicts
        assert collection_result is not None
        assert generic_result is not None
        assert covid_result is not None