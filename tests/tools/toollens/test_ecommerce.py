import pytest
from typing import Dict, Any, List

from tools.toollens.ecommerce import EcommerceTools


@pytest.fixture
def ecommerce_instance():
    """Create a stateless EcommerceTools instance for testing."""
    return EcommerceTools(initial_config=None)


# ---- BestBuyProductData_2 (keyword: str, page: float) -> Dict[str, Any] ----
def test_best_buy_product_data_2_normal(ecommerce_instance):
    """Normal call with valid keyword and page returns a dict."""
    result = ecommerce_instance.BestBuyProductData_2(keyword="laptop", page=1)
    assert isinstance(result, dict), "Expected dict return type"
    # realistic response should contain at least 'status' or 'products'
    assert len(result) > 0


def test_best_buy_product_data_2_edge(ecommerce_instance):
    """Passing empty keyword and zero page returns a dict (maybe error)."""
    result = ecommerce_instance.BestBuyProductData_2(keyword="", page=0)
    assert isinstance(result, dict), "Should still return a dict"
    # no exception raised


# ---- Categories_List () -> Any (likely list) ----
def test_categories_list(ecommerce_instance):
    """Fetching categories returns a list or dict (not None)."""
    result = ecommerce_instance.Categories_List()
    assert result is not None, "Categories_List should return something"
    # it could be a list or a dict; we just check it's not None
    assert isinstance(result, (list, dict))


# ---- Countries () -> Any ----
def test_countries(ecommerce_instance):
    """Countries returns a list of countries."""
    result = ecommerce_instance.Countries()
    assert result is not None
    # expected to be a list of dicts or a dict
    assert isinstance(result, (list, dict))


# ---- Fetch_Company_Details (query: str) -> Dict[str, str] ----
def test_fetch_company_details_normal(ecommerce_instance):
    """Valid company name returns a dict with details."""
    result = ecommerce_instance.Fetch_Company_Details(query="Apple")
    assert isinstance(result, dict), "Expected dict return type"
    assert len(result) > 0


def test_fetch_company_details_edge(ecommerce_instance):
    """Empty query returns a dict (fallback or error)."""
    result = ecommerce_instance.Fetch_Company_Details(query="")
    assert isinstance(result, dict)


# ---- Get_Amazon_Search_results (searchQuery: str) -> Dict[str, Any] ----
def test_get_amazon_search_results_normal(ecommerce_instance):
    """Normal search returns a dict with results."""
    result = ecommerce_instance.Get_Amazon_Search_results(searchQuery="piano")
    assert isinstance(result, dict), "Expected dict"
    assert len(result) > 0


def test_get_amazon_search_results_edge(ecommerce_instance):
    """Empty search query returns a dict (maybe error)."""
    result = ecommerce_instance.Get_Amazon_Search_results(searchQuery="")
    assert isinstance(result, dict)


# ---- Get_Stores () -> Dict[str, str] ----
def test_get_stores(ecommerce_instance):
    """Get_Stores returns a dict."""
    result = ecommerce_instance.Get_Stores()
    assert isinstance(result, dict), "Expected dict"
    assert len(result) > 0


# ---- Get_list_of_GitHub_repo_for_Ruby_web_scrapping () -> Dict[str, str] ----
def test_get_list_of_git_hub_repo_for_ruby_web_scrapping(ecommerce_instance):
    """Returns a dict with repo information."""
    result = ecommerce_instance.Get_list_of_GitHub_repo_for_Ruby_web_scrapping()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- Get_list_of_Github_repo_for_Ruby_Webscrapping () -> Dict[str, str] ----
def test_get_list_of_github_repo_for_ruby_webscrapping(ecommerce_instance):
    """Returns a dict (variant casing)."""
    result = ecommerce_instance.Get_list_of_Github_repo_for_Ruby_Webscrapping()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- Get_list_of_Github_repo_for_ruby_web_scrapping () -> Dict[str, str] ----
def test_get_list_of_github_repo_for_ruby_web_scrapping(ecommerce_instance):
    """Returns a dict (lowercase variant)."""
    result = ecommerce_instance.Get_list_of_Github_repo_for_ruby_web_scrapping()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- MarketPlace_List () -> Any ----
def test_market_place_list(ecommerce_instance):
    """MarketPlace_List returns data (likely a list)."""
    result = ecommerce_instance.MarketPlace_List()
    assert result is not None
    assert isinstance(result, (list, dict))


# ---- Search_By_Keyword_Filters (countryCode: str, keyword: str) -> Any ----
def test_search_by_keyword_filters_normal(ecommerce_instance):
    """Valid inputs return a non‑None result."""
    result = ecommerce_instance.Search_By_Keyword_Filters(countryCode="US", keyword="shirt")
    assert result is not None
    assert isinstance(result, (list, dict))


def test_search_by_keyword_filters_edge(ecommerce_instance):
    """Empty parameters should still return a result without exception."""
    result = ecommerce_instance.Search_By_Keyword_Filters(countryCode="", keyword="")
    assert result is not None


# ---- Search_by_keyword (keyword: str, page: float) -> Dict[str, str] ----
def test_search_by_keyword_normal(ecommerce_instance):
    """Normal search returns a dict."""
    result = ecommerce_instance.Search_by_keyword(keyword="laptop", page=1)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_by_keyword_edge(ecommerce_instance):
    """Empty keyword and zero page returns a dict (possibly error)."""
    result = ecommerce_instance.Search_by_keyword(keyword="", page=0)
    assert isinstance(result, dict)


# ---- Search_for_a_product (q: str) -> Dict[str, str] ----
def test_search_for_a_product_normal(ecommerce_instance):
    """Search for a product returns a dict."""
    result = ecommerce_instance.Search_for_a_product(q="sneakers")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_for_a_product_edge(ecommerce_instance):
    """Empty query returns a dict (no exception)."""
    result = ecommerce_instance.Search_for_a_product(q="")
    assert isinstance(result, dict)


# ---- Stores (countryCode: str) -> Any ----
def test_stores_normal(ecommerce_instance):
    """Countries with stores return a list or dict."""
    result = ecommerce_instance.Stores(countryCode="US")
    assert result is not None
    assert isinstance(result, (list, dict))


def test_stores_edge(ecommerce_instance):
    """Empty country code returns some data (maybe error dict)."""
    result = ecommerce_instance.Stores(countryCode="")
    assert result is not None


# ---- Tax_Rate (zipCode: str) -> Dict[str, str] ----
def test_tax_rate_normal(ecommerce_instance):
    """Valid zip code returns a dict with tax info."""
    result = ecommerce_instance.Tax_Rate(zipCode="10001")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_tax_rate_edge(ecommerce_instance):
    """Empty zip code returns a dict (fallback or error)."""
    result = ecommerce_instance.Tax_Rate(zipCode="")
    assert isinstance(result, dict)


# ---- auto_complete (q: str) -> List[str] ----
def test_auto_complete_normal(ecommerce_instance):
    """Valid term returns a list of strings."""
    result = ecommerce_instance.auto_complete(q="lap")
    assert isinstance(result, list), "Expected list"
    # all items should be strings
    if result:  # list may be empty
        assert all(isinstance(item, str) for item in result)


def test_auto_complete_edge(ecommerce_instance):
    """Empty query returns a list (empty or with suggestions)."""
    result = ecommerce_instance.auto_complete(q="")
    assert isinstance(result, list)


# ---- categories () -> Dict[str, str] ----
def test_categories(ecommerce_instance):
    """Facebook categories returns a dict."""
    result = ecommerce_instance.categories()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- categories_list (country: str, lang: str) -> Dict[str, str] ----
def test_categories_list_normal(ecommerce_instance):
    """Valid country and language return a dict of categories."""
    result = ecommerce_instance.categories_list(country="us", lang="en")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_categories_list_edge(ecommerce_instance):
    """Empty parameters return a dict (no exception)."""
    result = ecommerce_instance.categories_list(country="", lang="")
    assert isinstance(result, dict)


# ---- categories_v2_list () -> Dict[str, Any] ----
def test_categories_v2_list(ecommerce_instance):
    """v2 categories list returns a dict with metadata."""
    result = ecommerce_instance.categories_v2_list()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- countries_detail () -> Dict[str, Any] ----
def test_countries_detail(ecommerce_instance):
    """Country details returns a dict."""
    result = ecommerce_instance.countries_detail()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- products_list (lang: str, currentpage: float, country: str, pagesize: float) -> Dict[str, Any] ----
def test_products_list_normal(ecommerce_instance):
    """Valid parameters return a dict with products."""
    result = ecommerce_instance.products_list(lang="en", currentpage=1, country="us", pagesize=10)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_products_list_edge(ecommerce_instance):
    """Empty strings and zero values return a dict (may be error)."""
    result = ecommerce_instance.products_list(lang="", currentpage=0, country="", pagesize=0)
    assert isinstance(result, dict)


# ---- regions_list () -> Dict[str, str] ----
def test_regions_list(ecommerce_instance):
    """Regions list returns a dict."""
    result = ecommerce_instance.regions_list()
    assert isinstance(result, dict)
    assert len(result) > 0


# ---- search_products (keyword: str) -> Dict[str, str] ----
def test_search_products_normal(ecommerce_instance):
    """Search products with a keyword returns a dict."""
    result = ecommerce_instance.search_products(keyword="hat")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_products_edge(ecommerce_instance):
    """Empty keyword returns a dict (fallback)."""
    result = ecommerce_instance.search_products(keyword="")
    assert isinstance(result, dict)


# ---- search_autocomplete (q: str) -> Dict[str, str] ----
def test_search_autocomplete_normal(ecommerce_instance):
    """Autocomplete returns a dict with suggestions."""
    result = ecommerce_instance.search_autocomplete(q="com")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_autocomplete_edge(ecommerce_instance):
    """Empty query returns a dict (no exception)."""
    result = ecommerce_instance.search_autocomplete(q="")
    assert isinstance(result, dict)


# ---- stores_list (longitude: float, latitude: float) -> Dict[str, Any] ----
def test_stores_list_normal(ecommerce_instance):
    """Valid coordinates return a dict with nearby stores."""
    result = ecommerce_instance.stores_list(longitude=48.8566, latitude=2.3522)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_stores_list_edge(ecommerce_instance):
    """Zero coordinates return a dict (fallback or results)."""
    result = ecommerce_instance.stores_list(longitude=0.0, latitude=0.0)
    assert isinstance(result, dict)