import pytest
import json
from tools.toollens.business import BusinessTools


@pytest.fixture
def business_tools():
    """Fixture providing a fresh BusinessTools instance with initial config."""
    initial_config = {
        "v1_databaseStatus": {
            "last_updated": "2024-01-15T08:30:00Z",
            "status": "fresh",
            "version": "2.1"
        },
        "All_Exchange_Rates": {
            "base_currencies": ["USD", "EUR", "GBP", "JPY"],
            "rates": {
                "USD": {"EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "NGN": 1480},
                "EUR": {"USD": 1.09, "GBP": 0.86}
            },
            "last_refresh": "2024-01-20T12:00:00Z"
        },
        "Business_name": {
            "generated_names": [
                "Quantum Dynamics LLC",
                "Stellar Ventures Inc",
                "Nimbus Solutions Ltd"
            ]
        },
        "Businessplan": {
            "plans": [
                {"idea": "SaaS platform", "plan": "Executive Summary..."}
            ]
        },
        "Casino_Tournaments_List": {
            "tournaments": [
                {"id": "t1", "name": "Weekend Slots Championship", "prize": 5000, "status": "active"},
                {"id": "t2", "name": "Poker Masters", "prize": 10000, "status": "upcoming"}
            ]
        },
        "Categorize_Job_Title": {
            "categories": ["Engineering", "Sales", "Marketing", "Finance", "Operations"],
            "results": {
                "Software Engineer": "Engineering",
                "Account Executive": "Sales",
                "CFO": "Finance"
            }
        },
        "Fetch_email_of_a_person": {
            "persons": [
                {"name": "John Doe", "company": "acme.com", "email": "john@acme.com"},
                {"name": "Jane Smith", "company": "globex.com"}
            ]
        },
        "Get_All_Companies_Paginated": {
            "companies": [
                {"id": "RC1234567", "name": "Dangote Industries", "country": "Nigeria"},
                {"id": "RC2345678", "name": "Flutterwave", "country": "Nigeria"}
            ],
            "page": 1,
            "total": 2
        },
        "Get_a_random_self_help_quote": {
            "quotes": [
                {"text": "The only way to do great work is to love what you do", "author": "Steve Jobs", "tags": ["passion", "work"], "book": "Stanford Speech"}
            ]
        },
        "Get_all_available_tags_for_self_help_quotes": {
            "tags": ["motivation", "success", "productivity", "mindfulness", "leadership", "habits"]
        },
        "Get_rentals": {
            "rentals": [
                {"id": "r1", "city": "Austin", "price": 1500, "type": "apartment"},
                {"id": "r2", "city": "Seattle", "price": 2200, "type": "house"}
            ]
        },
        "Indicator_Categories": {
            "categories": ["Economic", "Social", "Environmental", "Governance"]
        },
        "Search_by_company_name": {
            "results": [
                {"name": "MTN Uganda", "registration": "80020001234567"},
                {"name": "Stanbic Bank Uganda", "registration": "80020002345678"}
            ]
        },
        "auto_complete": {
            "suggestions": [
                {"term": "marketing", "category": "business"},
                {"term": "market research", "category": "business"}
            ]
        },
        "blogs_copy": {
            "posts": [
                {"id": "b1", "title": "Starting Your Business", "author": "Admin"},
                {"id": "b2", "title": "Funding Strategies", "author": "Editor"}
            ]
        }
    }
    config = json.loads(json.dumps(initial_config))
    return BusinessTools(initial_config=config)


class TestBusinessToolsSequentialCorrect:
    """Tests exercising correct ordered sequences of BusinessTools method calls."""

    def test_database_status_then_exchange_rates(self, business_tools):
        """Check database freshness, then retrieve exchange rates."""
        status_result = business_tools.v1_databaseStatus()
        assert status_result is not None
        assert "status" in status_result or "version" in status_result

        rates_result = business_tools.All_Exchange_Rates()
        assert rates_result is not None
        # Rates should be a dict keyed by base currency
        assert isinstance(rates_result, dict)
        assert len(rates_result) > 0

    def test_generate_name_then_business_plan(self, business_tools):
        """Generate a business name, then create a business plan using that name as idea."""
        name_result = business_tools.Business_name()
        assert name_result is not None
        # Extract generated name from result
        generated_name = None
        if isinstance(name_result, dict):
            for key in ["name", "business_name", "generated_name"]:
                if key in name_result:
                    generated_name = name_result[key]
                    break
            if generated_name is None and "generated_names" in name_result:
                names = name_result["generated_names"]
                if isinstance(names, list) and len(names) > 0:
                    generated_name = names[0]
        assert generated_name is not None

        plan_result = business_tools.Businessplan(idea=generated_name)
        assert plan_result is not None
        assert isinstance(plan_result, dict)

    def test_get_tags_then_random_quote(self, business_tools):
        """List available self-help quote tags, then fetch a random quote."""
        tags_result = business_tools.Get_all_available_tags_for_self_help_quotes()
        assert tags_result is not None
        assert isinstance(tags_result, dict)
        # Should contain a tags list
        tags_list = tags_result.get("tags", [])
        assert isinstance(tags_list, list)
        assert len(tags_list) > 0

        quote_result = business_tools.Get_a_random_self_help_quote()
        assert quote_result is not None
        assert isinstance(quote_result, dict)

    def test_search_company_then_paginate_companies(self, business_tools):
        """Search for a company by name, then paginate through all companies."""
        search_result = business_tools.Search_by_company_name(name="MTN")
        assert search_result is not None
        assert isinstance(search_result, dict)

        page_result = business_tools.Get_All_Companies_Paginated(page=1, limit=10)
        assert page_result is not None
        assert isinstance(page_result, dict)
        # Should have companies list
        companies = page_result.get("companies", [])
        assert isinstance(companies, list)

    def test_categorize_job_then_fetch_email(self, business_tools):
        """Categorize a job title, then fetch email for a person in that role."""
        category_result = business_tools.Categorize_Job_Title(title="Software Engineer")
        assert category_result is not None
        assert isinstance(category_result, dict)

        email_result = business_tools.Fetch_email_of_a_person(
            first_name="John", last_name="Doe", domain="acme.com"
        )
        assert email_result is not None
        assert isinstance(email_result, dict)


class TestBusinessToolsSequentialProblematic:
    """Tests exercising problematic sequences and edge cases."""

    def test_search_nonexistent_company_then_paginate(self, business_tools):
        """Search for a nonexistent company, then paginate companies to verify no crash."""
        search_result = business_tools.Search_by_company_name(name="ZZZNonexistentCorp123")
        assert search_result is not None
        assert isinstance(search_result, dict)
        # Results should be empty or contain an error indicator
        results = search_result.get("results", [])
        assert results == [] or "error" in search_result

        page_result = business_tools.Get_All_Companies_Paginated(page=1, limit=10)
        assert page_result is not None
        assert isinstance(page_result, dict)

    def test_businessplan_empty_idea_then_generate_name(self, business_tools):
        """Call Businessplan with empty idea, then generate a name without crash."""
        plan_result = business_tools.Businessplan(idea="")
        assert plan_result is not None
        assert isinstance(plan_result, dict)

        name_result = business_tools.Business_name()
        assert name_result is not None

    def test_fetch_email_missing_params_then_categorize(self, business_tools):
        """Fetch email with missing params, then categorize a job title."""
        email_result = business_tools.Fetch_email_of_a_person()
        assert email_result is not None
        assert isinstance(email_result, dict)

        category_result = business_tools.Categorize_Job_Title(title="")
        assert category_result is not None
        assert isinstance(category_result, dict)

    def test_paginate_invalid_page_then_get_rentals(self, business_tools):
        """Paginate with invalid page number, then get rentals without crash."""
        page_result = business_tools.Get_All_Companies_Paginated(page=-1, limit=0)
        assert page_result is not None
        assert isinstance(page_result, dict)

        rentals_result = business_tools.Get_rentals()
        assert rentals_result is not None
        assert isinstance(rentals_result, dict)

    def test_auto_complete_empty_then_blogs_copy(self, business_tools):
        """Auto-complete with empty search term, then retrieve blogs."""
        auto_result = business_tools.auto_complete(search_term="")
        assert auto_result is not None

        blogs_result = business_tools.blogs_copy()
        assert blogs_result is not None
        assert isinstance(blogs_result, dict)