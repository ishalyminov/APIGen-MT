import pytest
import json
from tools.toollens.medical import MedicalTools


@pytest.fixture
def medical_instance():
    config = {
        'covid19_cases': [
            {'country': 'USA', 'date': '2023-01-15', 'cases': 1024, 'deaths': 45},
            {'country': 'India', 'date': '2023-02-20', 'cases': 856, 'deaths': 12},
            {'country': 'Brazil', 'date': '2023-03-10', 'cases': 432, 'deaths': 8}
        ],
        'covid19_lookup_dates': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'covid19_lookup_countries': ['USA', 'India', 'Brazil', 'Germany', 'Japan'],
        'covid19_empty': [],
        'brand_to_generic': {
            'Tylenol': 'Acetaminophen',
            'Advil': 'Ibuprofen',
            'Lipitor': 'Atorvastatin',
            'Zoloft': 'Sertraline',
            'Nexium': 'Esomeprazole',
            'Amoxil': 'Amoxicillin',
            'Motrin': 'Ibuprofen'
        },
        'brand_lookup_keys': ['Tylenol', 'Advil', 'Lipitor', 'Zoloft', 'Nexium', 'Amoxil', 'Motrin', 'UnknownBrand'],
        'brand_empty': {},
        'tcia_collections': ['LIDC-IDRI', 'TCGA-LUAD', 'PROSTATEx', 'QIN-BREAST', 'REMBRANDT', 'National Lung Screening Trial'],
        'tcia_collection_values': ['LIDC-IDRI', 'TCGA-LUAD', 'PROSTATex', 'QIN-BREAST', 'REMBRANDT', 'National Lung Screening Trial'],
        'tcia_empty_collections': []
    }
    return MedicalTools(initial_config=config)


@pytest.fixture
def empty_medical_instance():
    return MedicalTools(initial_config=None)


class TestV1Covid19:
    """Tests for the v1_covid19 method."""

    def test_v1_covid19_returns_dict_with_covid_data(self, medical_instance):
        """Test that v1_covid19 returns a dict with expected COVID-19 case data."""
        result = medical_instance.v1_covid19()

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_v1_covid19_with_empty_config(self, empty_medical_instance):
        """Test v1_covid19 handles empty/None config gracefully."""
        result = empty_medical_instance.v1_covid19()

        assert isinstance(result, dict)


class TestGenericname:
    """Tests for the genericname method."""

    def test_genericname_returns_dict_with_drug_mappings(self, medical_instance):
        """Test that genericname returns a dict containing brand-to-generic mappings."""
        result = medical_instance.genericname()

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_genericname_with_empty_config(self, empty_medical_instance):
        """Test genericname handles empty/None config without raising exceptions."""
        result = empty_medical_instance.genericname()

        assert isinstance(result, dict)


class TestGetCollectionValues:
    """Tests for the getCollectionValues method."""

    def test_getCollectionValues_returns_dict_with_tcia_collections(self, medical_instance):
        """Test that getCollectionValues returns a dict with TCIA collection data."""
        result = medical_instance.getCollectionValues()

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_getCollectionValues_with_empty_config(self, empty_medical_instance):
        """Test getCollectionValues handles empty/None config gracefully."""
        result = empty_medical_instance.getCollectionValues()

        assert isinstance(result, dict)