import json
import pytest
from tools.toollens.news_media import NewsMediaTools


@pytest.fixture
def tool():
    """Create a fresh NewsMediaTools instance for each test."""
    config = None
    config_copy = json.loads(json.dumps(config)) if config is not None else None
    return NewsMediaTools(initial_config=config_copy)


class TestNewsMediaToolsSequentialCorrect:
    """Correct sequential call patterns representing typical user journeys."""

    def test_sequence_search_and_filter(self, tool):
        """Search for a topic then apply a regional filter."""
        # Step 1: Basic search
        search_result = tool.Basic_Search("climate change")
        assert isinstance(search_result, dict), "Basic_Search should return dict"
        # Step 2: Apply China filter (no parameters)
        filter_result = tool.China_filter_for_all_news()
        assert isinstance(filter_result, dict), "China_filter_for_all_news should return dict"
        # No exceptions expected
        assert "error" not in search_result.get("status", ""), "Search should not error"
        assert "error" not in filter_result.get("status", ""), "Filter should not error"

    def test_sequence_latest_articles_and_articles(self, tool):
        """Get latest article list then retrieve articles for the same language."""
        # Step 1: Get latest article list for English
        latest = tool.Get_Latest_Article_List("en")
        assert isinstance(latest, dict), "Get_Latest_Article_List should return dict"
        # Step 2: Get articles in English
        articles = tool.Get_articles("en")
        assert isinstance(articles, dict), "Get_articles should return dict"

    def test_sequence_news_and_stats(self, tool):
        """Get news for a location then retrieve stats for the same location."""
        # Step 1: Get news for US
        news = tool.GetNews("US")
        assert isinstance(news, dict), "GetNews should return dict"
        # Step 2: Get stats for US
        stats = tool.GetStats("US")
        assert isinstance(stats, dict), "GetStats should return dict"

    def test_sequence_faqs_and_suggest(self, tool):
        """Get FAQs by topic then use the topic as a suggestion."""
        # Step 1: Get FAQs for technology, page 1
        faqs = tool.GetFAQsByTopic("technology", 1.0)
        assert isinstance(faqs, dict), "GetFAQsByTopic should return dict"
        # Step 2: Get suggestions for the same keyword
        suggest = tool.Suggest("technology")
        assert isinstance(suggest, dict), "Suggest should return dict"

    def test_sequence_search_title_and_climate(self, tool):
        """Search by title then retrieve all climate change news."""
        # Step 1: Find articles by title
        title_search = tool.Find_by_title("global warming")
        assert isinstance(title_search, dict), "Find_by_title should return dict"
        # Step 2: Get all climate change news
        climate_news = tool.All_Climate_Change_News()
        assert isinstance(climate_news, dict), "All_Climate_Change_News should return dict"


class TestNewsMediaToolsSequentialProblematic:
    """Problematic sequences that should be handled gracefully (no crashes)."""

    def test_invalid_page_and_empty_search(self, tool):
        """Call with invalid page then empty search query."""
        # Step 1: Invalid page number (-1)
        faqs = tool.GetFAQsByTopic("health", -1.0)
        assert isinstance(faqs, dict), "GetFAQsByTopic should return dict even with invalid page"
        # Expect error indication or empty result
        if "error" in faqs:
            pass  # expected behaviour
        # Step 2: Search with empty query
        empty_search = tool.Basic_Search("")
        assert isinstance(empty_search, dict), "Basic_Search should handle empty query"
        # No crash

    def test_invalid_nconst_and_empty_search(self, tool):
        """Call filmography with invalid nconst then empty general search."""
        # Step 1: Invalid nconst
        filmography = tool.actors_get_all_filmography("invalid_nconst_xyz")
        assert isinstance(filmography, dict), "actors_get_all_filmography should return dict"
        # Step 2: General search with empty query
        empty_search = tool.search("")
        assert isinstance(empty_search, dict), "search should handle empty query"
        # No crash

    def test_invalid_osay_state_and_suggest(self, tool):
        """Call osay with invalid state then suggest with empty keyword."""
        # Step 1: Invalid state
        osay_result = tool.osay("non_existent_state")
        assert isinstance(osay_result, dict), "osay should return dict"
        # Step 2: Empty suggest keyword
        suggest = tool.Suggest("")
        assert isinstance(suggest, dict), "Suggest should handle empty keyword"
        # No crash

    def test_empty_language_and_articles(self, tool):
        """Call language-specific methods with empty language."""
        # Step 1: Get latest article list with empty language
        latest = tool.Get_Latest_Article_List("")
        assert isinstance(latest, dict), "Get_Latest_Article_List should handle empty language"
        # Step 2: Get articles with empty language
        articles = tool.Get_articles("")
        assert isinstance(articles, dict), "Get_articles should handle empty language"
        # No crash

    def test_detik_search_invalid_params_and_find(self, tool):
        """Call detik_search with invalid parameters then find by empty title."""
        # Step 1: detik_search with negative page, zero limit, empty keyword
        detik = tool.detik_search(-1.0, 0.0, "")
        assert isinstance(detik, dict), "detik_search should handle invalid params"
        # Step 2: Find by title with empty query
        find = tool.Find_by_title("")
        assert isinstance(find, dict), "Find_by_title should handle empty query"
        # No crash