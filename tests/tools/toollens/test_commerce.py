import pytest
import json
from tools.toollens.commerce import CommerceTools


@pytest.fixture
def commerce_instance():
    config = {
        'Laptops': [
            {'id': 1, 'brand': 'Dell', 'model': 'XPS 15', 'price': 1499, 'cpu': 'Intel i7', 'ram': 16},
            {'id': 2, 'brand': 'Lenovo', 'model': 'ThinkPad X1', 'price': 1899, 'cpu': 'Intel i9', 'ram': 32},
            {'id': 3, 'brand': 'Apple', 'model': 'MacBook Pro', 'price': 2299, 'cpu': 'M2 Pro', 'ram': 16}
        ],
        'Get_Prices': [
            {'gpu': 'RTX 4090', 'price': 1599},
            {'gpu': 'RTX 4070', 'price': 549},
            {'gpu': 'RX 7900 XTX', 'price': 999}
        ],
        'Get_Products': [
            {'id': 101, 'name': 'Wireless Mouse', 'price': 29.99, 'stock': 50},
            {'id': 102, 'name': 'USB Hub', 'price': 19.99, 'stock': 0},
            {'id': 103, 'name': 'Webcam', 'price': 89.99, 'stock': 15}
        ],
        'Get_a_specific_item': {
            'known_ids': [101, 102, 103],
            'sample_id': 101
        },
        'Get_all_the_shoes': [
            {'id': 201, 'brand': 'Nike', 'name': 'Air Max', 'size': 10, 'price': 129},
            {'id': 202, 'brand': 'Adidas', 'name': 'Ultraboost', 'size': 9, 'price': 180},
            {'id': 203, 'brand': 'Puma', 'name': 'Suede', 'size': 11, 'price': 65}
        ],
        'Search_Product': {
            'query': 'laptop',
            'expected_results': ['Dell XPS 15', 'MacBook Pro']
        },
        'Search_Products': {
            'country': 'US',
            'query': 'headphones',
            'last_page': 5
        },
        'Search_for_Creators': {
            'search_query': 'tech review',
            'sample_creators': ['Marques Brownlee', 'Linus Tech Tips']
        },
        'Search_on_ebay': {
            'query': 'vintage camera',
            'country': 'US'
        },
        'getProducts': [
            {'id': 301, 'name': 'Coffee Maker', 'price': 49.99},
            {'id': 302, 'name': 'Blender', 'price': 79.99}
        ],
        'mailcheck': {
            'test_emails': ['user@gmail.com', 'test@tempmail.com', 'admin@company.org']
        },
        'newlyRegisteredDomains': {
            'sample_domains': ['example.com', 'mynewsite.org', 'testdomain.net'],
            'days_back': 7
        },
        'sortProductsMaster': {
            'available_sorts': ['price_asc', 'price_desc', 'name_asc', 'name_desc', 'newest', 'popularity']
        },
        'empty_collections': {
            'Get_Products_empty': [],
            'Laptops_sparse': []
        }
    }
    return CommerceTools(initial_config=config)


@pytest.fixture
def empty_commerce_instance():
    return CommerceTools(initial_config=None)


# Tests for Laptops
def test_laptops_returns_dict_with_laptop_data(commerce_instance):
    """Test that Laptops returns a dict containing laptop listing data."""
    result = commerce_instance.Laptops()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_laptops_with_empty_config(empty_commerce_instance):
    """Test Laptops handles empty initial config gracefully."""
    result = empty_commerce_instance.Laptops()
    assert isinstance(result, dict)


# Tests for Get_Prices
def test_get_prices_returns_dict_with_gpu_pricing(commerce_instance):
    """Test that Get_Prices returns a dict with GPU price information."""
    result = commerce_instance.Get_Prices()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_prices_with_empty_config(empty_commerce_instance):
    """Test Get_Prices handles empty initial config gracefully."""
    result = empty_commerce_instance.Get_Prices()
    assert isinstance(result, dict)


# Tests for Get_Products
def test_get_products_returns_dict_with_product_listings(commerce_instance):
    """Test that Get_Products returns a dict with product data."""
    result = commerce_instance.Get_Products()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_products_with_empty_config(empty_commerce_instance):
    """Test Get_Products handles empty initial config gracefully."""
    result = empty_commerce_instance.Get_Products()
    assert isinstance(result, dict)


# Tests for Get_a_specific_item
def test_get_a_specific_item_returns_dict_with_item_details(commerce_instance):
    """Test that Get_a_specific_item returns a dict with item details."""
    result = commerce_instance.Get_a_specific_item()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_a_specific_item_with_empty_config(empty_commerce_instance):
    """Test Get_a_specific_item handles empty initial config gracefully."""
    result = empty_commerce_instance.Get_a_specific_item()
    assert isinstance(result, dict)


# Tests for Get_all_the_shoes
def test_get_all_the_shoes_returns_dict_with_shoe_listings(commerce_instance):
    """Test that Get_all_the_shoes returns a dict with shoe data."""
    result = commerce_instance.Get_all_the_shoes()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_all_the_shoes_with_empty_config(empty_commerce_instance):
    """Test Get_all_the_shoes handles empty initial config gracefully."""
    result = empty_commerce_instance.Get_all_the_shoes()
    assert isinstance(result, dict)


# Tests for Search_Product
def test_search_product_with_valid_query(commerce_instance):
    """Test Search_Product with a valid query string."""
    result = commerce_instance.Search_Product(query="laptop", act="search")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_product_with_empty_params(commerce_instance):
    """Test Search_Product handles empty/None params gracefully."""
    result = commerce_instance.Search_Product(query="", act="")
    assert isinstance(result, dict)


# Tests for Search_Products
def test_search_products_with_valid_query(commerce_instance):
    """Test Search_Products with a valid search query."""
    result = commerce_instance.Search_Products(search_query="headphones")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_products_with_empty_query(commerce_instance):
    """Test Search_Products handles empty search query gracefully."""
    result = commerce_instance.Search_Products(search_query="")
    assert isinstance(result, dict)


# Tests for Search_for_Creators
def test_search_for_creators_with_valid_query(commerce_instance):
    """Test Search_for_Creators with a valid search query."""
    result = commerce_instance.Search_for_Creators(search_query="tech review")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_for_creators_with_empty_query(commerce_instance):
    """Test Search_for_Creators handles empty search query gracefully."""
    result = commerce_instance.Search_for_Creators(search_query="")
    assert isinstance(result, dict)


# Tests for Search_on_ebay
def test_search_on_ebay_with_valid_query(commerce_instance):
    """Test Search_on_ebay with a valid search query."""
    result = commerce_instance.Search_on_ebay(searchQuery="vintage camera")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_on_ebay_with_empty_query(commerce_instance):
    """Test Search_on_ebay handles empty search query gracefully."""
    result = commerce_instance.Search_on_ebay(searchQuery="")
    assert isinstance(result, dict)


# Tests for getProducts
def test_get_products_returns_dict_with_product_data(commerce_instance):
    """Test that getProducts returns a dict with product information."""
    result = commerce_instance.getProducts()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_products_with_empty_config(empty_commerce_instance):
    """Test getProducts handles empty initial config gracefully."""
    result = empty_commerce_instance.getProducts()
    assert isinstance(result, dict)


# Tests for mailcheck
def test_mailcheck_with_valid_domain(commerce_instance):
    """Test mailcheck with a valid domain."""
    result = commerce_instance.mailcheck(domain="gmail.com")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_mailcheck_with_empty_domain(commerce_instance):
    """Test mailcheck handles empty domain gracefully."""
    result = commerce_instance.mailcheck(domain="")
    assert isinstance(result, dict)


# Tests for newlyRegisteredDomains
def test_newly_registered_domains_returns_dict_with_domain_data(commerce_instance):
    """Test that newlyRegisteredDomains returns a dict with domain data."""
    result = commerce_instance.newlyRegisteredDomains()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_newly_registered_domains_with_empty_config(empty_commerce_instance):
    """Test newlyRegisteredDomains handles empty initial config gracefully."""
    result = empty_commerce_instance.newlyRegisteredDomains()
    assert isinstance(result, dict)


# Tests for sortProductsMaster
def test_sort_products_master_returns_dict_with_sort_options(commerce_instance):
    """Test that sortProductsMaster returns a dict with sort options."""
    result = commerce_instance.sortProductsMaster()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_sort_products_master_with_empty_config(empty_commerce_instance):
    """Test sortProductsMaster handles empty initial config gracefully."""
    result = empty_commerce_instance.sortProductsMaster()
    assert isinstance(result, dict)