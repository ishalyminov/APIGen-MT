import pytest
import json
from tools.toollens.commerce import CommerceTools


INITIAL_CONFIG = {
    "Laptops": [
        {
            "id": 1,
            "brand": "Dell",
            "model": "XPS 15",
            "price": 1499,
            "cpu": "Intel i7",
            "ram": 16
        },
        {
            "id": 2,
            "brand": "Lenovo",
            "model": "ThinkPad X1",
            "price": 1899,
            "cpu": "Intel i9",
            "ram": 32
        },
        {
            "id": 3,
            "brand": "Apple",
            "model": "MacBook Pro",
            "price": 2299,
            "cpu": "M2 Pro",
            "ram": 16
        }
    ],
    "Get_Prices": [
        {
            "gpu": "RTX 4090",
            "price": 1599
        },
        {
            "gpu": "RTX 4070",
            "price": 549
        },
        {
            "gpu": "RX 7900 XTX",
            "price": 999
        }
    ],
    "Get_Products": [
        {
            "id": 101,
            "name": "Wireless Mouse",
            "price": 29.99,
            "stock": 50
        },
        {
            "id": 102,
            "name": "USB Hub",
            "price": 19.99,
            "stock": 0
        },
        {
            "id": 103,
            "name": "Webcam",
            "price": 89.99,
            "stock": 15
        }
    ],
    "Get_a_specific_item": {
        "known_ids": [
            101,
            102,
            103
        ],
        "sample_id": 101
    },
    "Get_all_the_shoes": [
        {
            "id": 201,
            "brand": "Nike",
            "name": "Air Max",
            "size": 10,
            "price": 129
        },
        {
            "id": 202,
            "brand": "Adidas",
            "name": "Ultraboost",
            "size": 9,
            "price": 180
        },
        {
            "id": 203,
            "brand": "Puma",
            "name": "Suede",
            "size": 11,
            "price": 65
        }
    ],
    "Search_Product": {
        "query": "laptop",
        "expected_results": [
            "Dell XPS 15",
            "MacBook Pro"
        ]
    },
    "Search_Products": {
        "country": "US",
        "query": "headphones",
        "last_page": 5
    },
    "Search_for_Creators": {
        "search_query": "tech review",
        "sample_creators": [
            "Marques Brownlee",
            "Linus Tech Tips"
        ]
    },
    "Search_on_ebay": {
        "query": "vintage camera",
        "country": "US"
    },
    "getProducts": [
        {
            "id": 301,
            "name": "Coffee Maker",
            "price": 49.99
        },
        {
            "id": 302,
            "name": "Blender",
            "price": 79.99
        }
    ],
    "mailcheck": {
        "test_emails": [
            "user@gmail.com",
            "test@tempmail.com",
            "admin@company.org"
        ]
    },
    "newlyRegisteredDomains": {
        "sample_domains": [
            "example.com",
            "mynewsite.org",
            "testdomain.net"
        ],
        "days_back": 7
    },
    "sortProductsMaster": {
        "available_sorts": [
            "price_asc",
            "price_desc",
            "name_asc",
            "name_desc",
            "newest",
            "popularity"
        ]
    },
    "empty_collections": {
        "Get_Products_empty": [],
        "Laptops_sparse": []
    }
}


@pytest.fixture
def commerce_instance():
    """Create a fresh CommerceTools instance with a deep-copied config."""
    config = json.loads(json.dumps(INITIAL_CONFIG))
    return CommerceTools(initial_config=config)


@pytest.fixture
def empty_commerce_instance():
    """Create a CommerceTools instance with empty collections."""
    config = json.loads(json.dumps(INITIAL_CONFIG))
    config["Get_Products"] = []
    config["Laptops"] = []
    config["Get_all_the_shoes"] = []
    config["getProducts"] = []
    return CommerceTools(initial_config=config)


class TestCommerceToolsSequentialCorrect:
    """Tests exercising correct ordered sequences of CommerceTools methods."""

    def test_get_products_then_get_specific_item(self, commerce_instance):
        """Get_Products followed by Get_a_specific_item for a known product id."""
        products_response = commerce_instance.Get_Products()
        assert products_response is not None
        assert isinstance(products_response, dict)

        item_response = commerce_instance.Get_a_specific_item()
        assert item_response is not None
        assert isinstance(item_response, dict)

        # Verify both calls returned data
        assert len(products_response) > 0
        assert len(item_response) > 0

    def test_laptops_then_get_prices_cross_reference(self, commerce_instance):
        """Laptops listing then Get_Prices for GPU cross-referencing."""
        laptops_response = commerce_instance.Laptops()
        assert laptops_response is not None
        assert isinstance(laptops_response, dict)

        prices_response = commerce_instance.Get_Prices()
        assert prices_response is not None
        assert isinstance(prices_response, dict)

        # Both should return valid data structures
        assert len(laptops_response) > 0
        assert len(prices_response) > 0

    def test_search_product_then_search_products(self, commerce_instance):
        """Search_Product then Search_Products for combined search workflow."""
        search_response = commerce_instance.Search_Product(query="laptop")
        assert search_response is not None
        assert isinstance(search_response, dict)

        products_search_response = commerce_instance.Search_Products(search_query="headphones")
        assert products_search_response is not None
        assert isinstance(products_search_response, dict)

        # Both searches should return results
        assert len(search_response) > 0
        assert len(products_search_response) > 0

    def test_get_all_shoes_then_sort_products_master(self, commerce_instance):
        """Get_all_the_shoes then sortProductsMaster for sorting workflow."""
        shoes_response = commerce_instance.Get_all_the_shoes()
        assert shoes_response is not None
        assert isinstance(shoes_response, dict)

        sort_response = commerce_instance.sortProductsMaster()
        assert sort_response is not None
        assert isinstance(sort_response, dict)

        # Both should return valid data
        assert len(shoes_response) > 0
        assert len(sort_response) > 0

    def test_mailcheck_then_newly_registered_domains(self, commerce_instance):
        """mailcheck then newlyRegisteredDomains for domain analysis workflow."""
        mail_response = commerce_instance.mailcheck(domain="gmail.com")
        assert mail_response is not None
        assert isinstance(mail_response, dict)

        domains_response = commerce_instance.newlyRegisteredDomains()
        assert domains_response is not None
        assert isinstance(domains_response, dict)

        # Both should return valid data
        assert len(mail_response) > 0
        assert len(domains_response) > 0

    def test_get_products_then_getproducts_cross_collection(self, commerce_instance):
        """Get_Products then getProducts for cross-collection product browsing."""
        products_response = commerce_instance.Get_Products()
        assert products_response is not None
        assert isinstance(products_response, dict)

        get_products_response = commerce_instance.getProducts()
        assert get_products_response is not None
        assert isinstance(get_products_response, dict)

        # Both collections should return data
        assert len(products_response) > 0
        assert len(get_products_response) > 0

    def test_search_for_creators_then_search_on_ebay(self, commerce_instance):
        """Search_for_Creators then Search_on_ebay for combined search workflow."""
        creators_response = commerce_instance.Search_for_Creators(search_query="tech review")
        assert creators_response is not None
        assert isinstance(creators_response, dict)

        ebay_response = commerce_instance.Search_on_ebay(searchQuery="vintage camera")
        assert ebay_response is not None
        assert isinstance(ebay_response, dict)

        # Both searches should return results
        assert len(creators_response) > 0
        assert len(ebay_response) > 0


class TestCommerceToolsSequentialProblematic:
    """Tests exercising problematic sequences of CommerceTools methods."""

    def test_get_specific_item_with_empty_collections(self, empty_commerce_instance):
        """Get_a_specific_item on empty collections should not crash."""
        products_response = empty_commerce_instance.Get_Products()
        assert products_response is not None
        assert isinstance(products_response, dict)

        item_response = empty_commerce_instance.Get_a_specific_item()
        assert item_response is not None
        assert isinstance(item_response, dict)
        # Should handle empty gracefully without raising

    def test_search_product_with_invalid_query_then_valid_search(self, commerce_instance):
        """Search_Product with invalid query then valid Search_Products."""
        invalid_response = commerce_instance.Search_Product(query="")
        assert invalid_response is not None
        assert isinstance(invalid_response, dict)

        valid_response = commerce_instance.Search_Products(search_query="headphones")
        assert valid_response is not None
        assert isinstance(valid_response, dict)
        # Second call should still work after invalid first call

    def test_mailcheck_with_invalid_domain_then_newly_registered(self, commerce_instance):
        """mailcheck with invalid domain then newlyRegisteredDomains."""
        invalid_mail_response = commerce_instance.mailcheck(domain="")
        assert invalid_mail_response is not None
        assert isinstance(invalid_mail_response, dict)

        domains_response = commerce_instance.newlyRegisteredDomains()
        assert domains_response is not None
        assert isinstance(domains_response, dict)
        # Second call should still work after invalid first call

    def test_search_on_ebay_empty_query_then_get_products(self, commerce_instance):
        """Search_on_ebay with empty query then Get_Products."""
        empty_search_response = commerce_instance.Search_on_ebay(searchQuery="")
        assert empty_search_response is not None
        assert isinstance(empty_search_response, dict)

        products_response = commerce_instance.Get_Products()
        assert products_response is not None
        assert isinstance(products_response, dict)
        # Get_Products should still work after empty search

    def test_laptops_on_empty_then_get_prices(self, empty_commerce_instance):
        """Laptops on empty collection then Get_Prices."""
        laptops_response = empty_commerce_instance.Laptops()
        assert laptops_response is not None
        assert isinstance(laptops_response, dict)

        prices_response = empty_commerce_instance.Get_Prices()
        assert prices_response is not None
        assert isinstance(prices_response, dict)
        # Get_Prices should still work after empty Laptops

    def test_get_all_shoes_empty_then_sort_products_master(self, empty_commerce_instance):
        """Get_all_the_shoes on empty then sortProductsMaster."""
        shoes_response = empty_commerce_instance.Get_all_the_shoes()
        assert shoes_response is not None
        assert isinstance(shoes_response, dict)

        sort_response = empty_commerce_instance.sortProductsMaster()
        assert sort_response is not None
        assert isinstance(sort_response, dict)
        # sortProductsMaster should still work after empty shoes collection

    def test_search_for_creators_empty_then_search_product(self, commerce_instance):
        """Search_for_Creators with empty query then Search_Product."""
        empty_creators_response = commerce_instance.Search_for_Creators(search_query="")
        assert empty_creators_response is not None
        assert isinstance(empty_creators_response, dict)

        product_response = commerce_instance.Search_Product(query="laptop")
        assert product_response is not None
        assert isinstance(product_response, dict)
        # Search_Product should still work after empty creators search