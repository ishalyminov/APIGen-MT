import pytest
import json
from tools.toollens.ecommerce import EcommerceTools


@pytest.fixture
def tools():
    """
    Fixture that returns a fresh EcommerceTools instance.
    Although the initial config is None, we mimic a deep copy to stay
    consistent with the requirements.
    """
    config = json.loads(json.dumps(None))
    return EcommerceTools(initial_config=config)


class TestEcommerceToolsSequentialCorrect:
    """Correct ordered sequences – typical user journeys through the API."""

    def test_autocomplete_then_search(self, tools):
        """
        User types a partial product name (autocomplete) and then uses
        the suggestion to perform a full product search.
        """
        # Step 1: get autocomplete suggestions
        autocomplete_result = tools.auto_complete(q="lap")
        assert autocomplete_result is not None
        assert isinstance(autocomplete_result, list)
        assert len(autocomplete_result) > 0
        # Use the first suggestion as the search term
        suggested_term = autocomplete_result[0] if autocomplete_result else "laptop"

        # Step 2: search for the suggested product
        search_result = tools.Search_for_a_product(q=suggested_term)
        assert search_result is not None
        assert isinstance(search_result, dict)
        # The method returns a dict (e.g. {"name": ..., "price": ...})
        assert "name" in search_result or "title" in search_result or "product" in search_result

    def test_countries_then_categories_list(self, tools):
        """
        User fetches supported countries, then uses the first country's
        code and language to list H&M categories.
        """
        # Step 1: get all supported countries
        countries_result = tools.Countries()
        assert countries_result is not None
        assert isinstance(countries_result, list) or isinstance(countries_result, dict)
        # Extract a sample country and language
        if isinstance(countries_result, list) and len(countries_result) > 0:
            sample = countries_result[0]
            country = sample.get("countryCode", "US")
            lang = sample.get("language", "en")
        else:
            # Fallback
            country, lang = "US", "en"

        # Step 2: list categories for that country & language
        categories_result = tools.categories_list(country=country, lang=lang)
        assert categories_result is not None
        assert isinstance(categories_result, dict)
        # Expect keys like "categories", "count", etc.
        assert "categories" in categories_result or "category" in categories_result

    def test_search_by_keyword_then_fetch_company_details(self, tools):
        """
        User searches for a product keyword, then looks up the company
        details of the manufacturer or a related company.
        """
        # Step 1: search by keyword
        search_result = tools.Search_by_keyword(keyword="samsung", page=1.0)
        assert search_result is not None
        assert isinstance(search_result, dict)

        # Step 2: fetch company details for the brand
        company_result = tools.Fetch_Company_Details(query="Samsung")
        assert company_result is not None
        assert isinstance(company_result, dict)
        # Expected keys: "name", "description", "website", etc.
        assert "name" in company_result

    def test_get_stores_then_stores_list(self, tools):
        """
        User first gets the generic list of stores, then uses a coordinate
        to find nearby stores.
        """
        # Step 1: get generic store list
        stores_generic = tools.Get_Stores()
        assert stores_generic is not None
        assert isinstance(stores_generic, dict)

        # Step 2: list stores near a given location (e.g., New York)
        stores_nearby = tools.stores_list(longitude=-74.006, latitude=40.7128)
        assert stores_nearby is not None
        assert isinstance(stores_nearby, dict)
        # The result should contain store information
        assert "stores" in stores_nearby or "store" in stores_nearby

    def test_categories_then_products_list(self, tools):
        """
        User browses Facebook categories, then displays products for
        the first category in a specific language and pagination.
        """
        # Step 1: fetch Facebook item categories
        categories_result = tools.categories()
        assert categories_result is not None
        assert isinstance(categories_result, dict)

        # Step 2: list H&M products with pagination (use a default category if possible)
        products_result = tools.products_list(
            lang="en", currentpage=1.0, country="US", pagesize=10.0
        )
        assert products_result is not None
        assert isinstance(products_result, dict)
        # Should contain "products" or "items"
        assert "products" in products_result or "items" in products_result or "total" in products_result


class TestEcommerceToolsSequentialProblematic:
    """Problematic sequences – invalid parameters, missing resources, wrong order."""

    def test_search_invalid_then_autocomplete(self, tools):
        """
        User provides an empty keyword to search, then autocomplete
        should still work without crashing.
        """
        # Step 1: search with empty string
        result1 = tools.Search_for_a_product(q="")
        # Even with empty query, the method should return a dict (maybe empty)
        assert isinstance(result1, dict)

        # Step 2: autocomplete should still function normally
        result2 = tools.auto_complete(q="lap")
        assert isinstance(result2, list)
        assert len(result2) > 0

    def test_nonexistent_product_then_fetch_company(self, tools):
        """
        User searches for a nonsensical product, then tries to fetch a
        company that does not exist. Neither should raise an exception.
        """
        # Step 1: search for a string that likely returns no results
        no_product = tools.Search_by_keyword(keyword="xyznonexistent12345", page=1.0)
        assert isinstance(no_product, dict)

        # Step 2: fetch company details for a nonexistent name
        no_company = tools.Fetch_Company_Details(query="ThisCompanyDoesNotExist999")
        assert isinstance(no_company, dict)

    def test_empty_keyword_sequence(self, tools):
        """
        Multiple consecutive calls with empty or null parameters.
        """
        # Step 1: autocomplete with empty string
        result1 = tools.auto_complete(q="")
        assert isinstance(result1, list)

        # Step 2: search with empty query
        result2 = tools.Search_for_a_product(q="")
        assert isinstance(result2, dict)

        # Step 3: get Amazon search results with empty query (likely returns empty)
        result3 = tools.Get_Amazon_Search_results(searchQuery="")
        assert isinstance(result3, dict)

    def test_negative_page_then_search(self, tools):
        """
        User passes an invalid (negative) page number to a paginated
        method, then performs a normal search. The first call should
        return a sensible error/empty result without breaking the instance.
        """
        # Step 1: search with negative page
        result1 = tools.Search_by_keyword(keyword="laptop", page=-1.0)
        assert isinstance(result1, dict)

        # Step 2: regular search – must still work
        result2 = tools.Search_for_a_product(q="laptop")
        assert isinstance(result2, dict)
        # The result should contain product info
        assert "name" in result2 or "title" in result2 or "product" in result2

    def test_invalid_country_then_store_list(self, tools):
        """
        User provides an invalid country code to Stores(), then
        tries to list stores for a valid country. The invalid call
        should not break subsequent calls.
        """
        # Step 1: stores with an invalid country code
        result1 = tools.Stores(countryCode="XX")
        assert result1 is not None
        # The method returns 'Any' – guarantee it is not an exception

        # Step 2: list stores for a valid country
        result2 = tools.Stores(countryCode="US")
        assert result2 is not None

        # Step 3: also test the stores_list method (if available)
        result3 = tools.stores_list(longitude=-73.985, latitude=40.748)
        assert isinstance(result3, dict)