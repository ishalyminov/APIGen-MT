import pytest
import json
from tools.toollens.business import BusinessTools


@pytest.fixture
def business_instance():
    config = {
        'v1_databaseStatus': {
            'last_updated': '2024-01-15T08:30:00Z',
            'status': 'fresh',
            'version': '2.1'
        },
        'All_Exchange_Rates': {
            'base_currencies': ['USD', 'EUR', 'GBP', 'JPY'],
            'rates': {
                'USD': {'EUR': 0.92, 'GBP': 0.79, 'JPY': 149.5, 'NGN': 1480},
                'EUR': {'USD': 1.09, 'GBP': 0.86}
            },
            'last_refresh': '2024-01-20T12:00:00Z'
        },
        'Business_name': {
            'generated_names': [
                'Quantum Dynamics LLC',
                'Stellar Ventures Inc',
                'Nimbus Solutions Ltd'
            ]
        },
        'Businessplan': {
            'plans': [
                {'idea': 'SaaS platform', 'plan': 'Executive Summary...'}
            ]
        },
        'Casino_Tournaments_List': {
            'tournaments': [
                {'id': 't1', 'name': 'Weekend Slots Championship', 'prize': 5000, 'status': 'active'},
                {'id': 't2', 'name': 'Poker Masters', 'prize': 10000, 'status': 'upcoming'}
            ]
        },
        'Categorize_Job_Title': {
            'categories': ['Engineering', 'Sales', 'Marketing', 'Finance', 'Operations'],
            'results': {
                'Software Engineer': 'Engineering',
                'Account Executive': 'Sales',
                'CFO': 'Finance'
            }
        },
        'Fetch_email_of_a_person': {
            'persons': [
                {'name': 'John Doe', 'company': 'acme.com', 'email': 'john@acme.com'},
                {'name': 'Jane Smith', 'company': 'globex.com'}
            ]
        },
        'Get_All_Companies_Paginated': {
            'companies': [
                {'id': 'RC1234567', 'name': 'Dangote Industries', 'country': 'Nigeria'},
                {'id': 'RC2345678', 'name': 'Flutterwave', 'country': 'Nigeria'}
            ],
            'page': 1,
            'total': 2
        },
        'Get_a_random_self_help_quote': {
            'quotes': [
                {
                    'text': 'The only way to do great work is to love what you do',
                    'author': 'Steve Jobs',
                    'tags': ['passion', 'work'],
                    'book': 'Stanford Speech'
                }
            ]
        },
        'Get_all_available_tags_for_self_help_quotes': {
            'tags': ['motivation', 'success', 'productivity', 'mindfulness', 'leadership', 'habits']
        },
        'Get_rentals': {
            'rentals': [
                {'id': 'r1', 'city': 'Austin', 'price': 1500, 'type': 'apartment'},
                {'id': 'r2', 'city': 'Seattle', 'price': 2200, 'type': 'house'}
            ]
        },
        'Indicator_Categories': {
            'categories': ['Economic', 'Social', 'Environmental', 'Governance']
        },
        'Search_by_company_name': {
            'results': [
                {'name': 'MTN Uganda', 'registration': '80020001234567'},
                {'name': 'Stanbic Bank Uganda', 'registration': '80020002345678'}
            ],
            'registry': 'Uganda'
        },
        'auto_complete': {
            'suggestions': [
                {'term': 'bus', 'suggestions': ['business', 'business plan', 'business loan']},
                {'term': 'mar', 'suggestions': ['marketing', 'market research']}
            ]
        },
        'blogs_copy': {
            'blogs': [
                {'title': 'Startup Funding Guide', 'author': 'Sarah Chen', 'date': '2024-01-10'},
                {'title': 'Growth Hacking 101', 'author': 'Mike Ross', 'date': '2024-01-05'}
            ]
        },
        '_edge_cases': {
            'empty_collections': [],
            'sparse_records': [None],
            'missing_fields': {}
        }
    }
    return BusinessTools(initial_config=config)


@pytest.fixture
def empty_business_instance():
    return BusinessTools(initial_config=None)


# --- v1_databaseStatus ---

def test_v1_databaseStatus_normal(business_instance):
    """Test v1_databaseStatus returns a dict with expected keys."""
    result = business_instance.v1_databaseStatus()
    assert isinstance(result, dict)
    assert 'status' in result or 'last_updated' in result or 'version' in result or len(result) > 0


def test_v1_databaseStatus_no_config(empty_business_instance):
    """Test v1_databaseStatus with no initial config does not raise."""
    result = empty_business_instance.v1_databaseStatus()
    assert isinstance(result, dict)


# --- All_Exchange_Rates ---

def test_All_Exchange_Rates_normal(business_instance):
    """Test All_Exchange_Rates returns a dict of currency rates."""
    result = business_instance.All_Exchange_Rates()
    assert isinstance(result, dict)
    if result:
        for base, rates in result.items():
            assert isinstance(base, str)
            assert isinstance(rates, dict)


def test_All_Exchange_Rates_no_config(empty_business_instance):
    """Test All_Exchange_Rates with no config returns a dict."""
    result = empty_business_instance.All_Exchange_Rates()
    assert isinstance(result, dict)


# --- Business_name ---

def test_Business_name_normal(business_instance):
    """Test Business_name returns a dict with a name."""
    result = business_instance.Business_name()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_Business_name_no_config(empty_business_instance):
    """Test Business_name with no config returns a dict."""
    result = empty_business_instance.Business_name()
    assert isinstance(result, dict)


# --- Businessplan ---

def test_Businessplan_normal(business_instance):
    """Test Businessplan with a valid idea returns a plan dict."""
    result = business_instance.Businessplan(idea='SaaS platform')
    assert isinstance(result, dict)
    assert len(result) > 0


def test_Businessplan_empty_idea(business_instance):
    """Test Businessplan with empty idea string does not raise."""
    result = business_instance.Businessplan(idea='')
    assert isinstance(result, dict)


# --- Casino_Tournaments_List ---

def test_Casino_Tournaments_List_normal(business_instance):
    """Test Casino_Tournaments_List returns a dict with tournament data."""
    result = business_instance.Casino_Tournaments_List()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_Casino_Tournaments_List_no_config(empty_business_instance):
    """Test Casino_Tournaments_List with no config returns a dict."""
    result = empty_business_instance.Casino_Tournaments_List()
    assert isinstance(result, dict)


# --- Categorize_Job_Title ---

def test_Categorize_Job_Title_normal(business_instance):
    """Test Categorize_Job_Title with a known title returns a dict."""
    result = business_instance.Categorize_Job_Title(title='Software Engineer')
    assert isinstance(result, dict)


def test_Categorize_Job_Title_empty_title(business_instance):
    """Test Categorize_Job_Title with empty title does not raise."""
    result = business_instance.Categorize_Job_Title(title='')
    assert isinstance(result, dict)


# --- Fetch_email_of_a_person ---

def test_Fetch_email_of_a_person_normal(business_instance):
    """Test Fetch_email_of_a_person with valid params returns a dict."""
    result = business_instance.Fetch_email_of_a_person(
        first_name='John', domain='acme.com', last_name='Doe'
    )
    assert isinstance(result, dict)


def test_Fetch_email_of_a_person_missing_params(business_instance):
    """Test Fetch_email_of_a_person with missing params does not raise."""
    result = business_instance.Fetch_email_of_a_person(
        first_name='', domain='', last_name=''
    )
    assert isinstance(result, dict)


# --- Get_All_Companies_Paginated ---

def test_Get_All_Companies_Paginated_normal(business_instance):
    """Test Get_All_Companies_Paginated with page and limit returns a dict."""
    result = business_instance.Get_All_Companies_Paginated(page=1, limit=10)
    assert isinstance(result, dict)


def test_Get_All_Companies_Paginated_edge(business_instance):
    """Test Get_All_Companies_Paginated with zero/negative page does not raise."""
    result = business_instance.Get_All_Companies_Paginated(page=0, limit=0)
    assert isinstance(result, dict)


# --- Get_a_random_self_help_quote ---

def test_Get_a_random_self_help_quote_normal(business_instance):
    """Test Get_a_random_self_help_quote returns a dict."""
    result = business_instance.Get_a_random_self_help_quote()
    assert isinstance(result, dict)


def test_Get_a_random_self_help_quote_no_config(empty_business_instance):
    """Test Get_a_random_self_help_quote with no config returns a dict."""
    result = empty_business_instance.Get_a_random_self_help_quote()
    assert isinstance(result, dict)


# --- Get_all_available_tags_for_self_help_quotes ---

def test_Get_all_available_tags_normal(business_instance):
    """Test Get_all_available_tags_for_self_help_quotes returns a dict."""
    result = business_instance.Get_all_available_tags_for_self_help_quotes()
    assert isinstance(result, dict)


def test_Get_all_available_tags_no_config(empty_business_instance):
    """Test Get_all_available_tags_for_self_help_quotes with no config returns a dict."""
    result = empty_business_instance.Get_all_available_tags_for_self_help_quotes()
    assert isinstance(result, dict)


# --- Get_rentals ---

def test_Get_rentals_normal(business_instance):
    """Test Get_rentals returns a dict with rental data."""
    result = business_instance.Get_rentals()
    assert isinstance(result, dict)


def test_Get_rentals_no_config(empty_business_instance):
    """Test Get_rentals with no config returns a dict."""
    result = empty_business_instance.Get_rentals()
    assert isinstance(result, dict)


# --- Indicator_Categories ---

def test_Indicator_Categories_normal(business_instance):
    """Test Indicator_Categories returns a dict with categories."""
    result = business_instance.Indicator_Categories()
    assert isinstance(result, dict)


def test_Indicator_Categories_no_config(empty_business_instance):
    """Test Indicator_Categories with no config returns a dict."""
    result = empty_business_instance.Indicator_Categories()
    assert isinstance(result, dict)


# --- Search_by_company_name ---

def test_Search_by_company_name_normal(business_instance):
    """Test Search_by_company_name with a valid name returns a dict."""
    result = business_instance.Search_by_company_name(name='MTN')
    assert isinstance(result, dict)


def test_Search_by_company_name_empty(business_instance):
    """Test Search_by_company_name with empty name does not raise."""
    result = business_instance.Search_by_company_name(name='')
    assert isinstance(result, dict)


# --- auto_complete ---

def test_auto_complete_normal(business_instance):
    """Test auto_complete with a search term returns data."""
    result = business_instance.auto_complete(search_term='bus')
    assert result is not None


def test_auto_complete_empty_term(business_instance):
    """Test auto_complete with empty search term does not raise."""
    result = business_instance.auto_complete(search_term='')
    assert result is not None


# --- blogs_copy ---

def test_blogs_copy_normal(business_instance):
    """Test blogs_copy returns a dict with blog data."""
    result = business_instance.blogs_copy()
    assert isinstance(result, dict)


def test_blogs_copy_no_config(empty_business_instance):
    """Test blogs_copy with no config returns a dict."""
    result = empty_business_instance.blogs_copy()
    assert isinstance(result, dict)