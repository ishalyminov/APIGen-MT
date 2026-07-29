import pytest
import json
from typing import Dict, Any, List
from tools.toollens.news_media import NewsMediaTools

@pytest.fixture
def tools_instance():
    """Fixture providing a NewsMediaTools instance with default config."""
    return NewsMediaTools(initial_config=None)

class TestNewsMediaTools:
    """Test suite for NewsMediaTools class."""

    # ---------- Helper to check error response ----------
    @staticmethod
    def _assert_error_response(result: dict) -> None:
        """Assert that the result is an error dict."""
        assert isinstance(result, dict)
        assert "error" in result

    # ========== All_Climate_Change_News ==========
    def test_all_climate_change_news_success(self, tools_instance):
        """Test All_Climate_Change_News returns a dict with expected structure."""
        result = tools_instance.All_Climate_Change_News()
        assert isinstance(result, dict)

    # ========== Available_Region_List ==========
    def test_available_region_list_success(self, tools_instance):
        """Test Available_Region_List returns a dict."""
        result = tools_instance.Available_Region_List()
        assert isinstance(result, dict)
        assert len(result) > 0

    # ========== Basic_Search ==========
    def test_basic_search_success(self, tools_instance):
        """Test Basic_Search with a valid query returns a dict."""
        result = tools_instance.Basic_Search(q="climate")
        assert isinstance(result, dict)

    def test_basic_search_empty_query(self, tools_instance):
        """Test Basic_Search with empty string returns error dict."""
        result = tools_instance.Basic_Search(q="")
        self._assert_error_response(result)

    def test_basic_search_none_query(self, tools_instance):
        """Test Basic_Search with None returns error dict."""
        result = tools_instance.Basic_Search(q=None)
        self._assert_error_response(result)

    # ========== China_and_US_relation_filter ==========
    def test_china_and_us_relation_filter_success(self, tools_instance):
        """Test China_and_US_relation_filter returns a dict."""
        result = tools_instance.China_and_US_relation_filter()
        assert isinstance(result, dict)

    # ========== China_filter_for_all_news ==========
    def test_china_filter_for_all_news_success(self, tools_instance):
        """Test China_filter_for_all_news returns a dict."""
        result = tools_instance.China_filter_for_all_news()
        assert isinstance(result, dict)

    # ========== Filter_Korean_news ==========
    def test_filter_korean_news_success(self, tools_instance):
        """Test Filter_Korean_news returns a dict."""
        result = tools_instance.Filter_Korean_news()
        assert isinstance(result, dict)

    # ========== Filter_for_conflict ==========
    def test_filter_for_conflict_success(self, tools_instance):
        """Test Filter_for_conflict returns a dict."""
        result = tools_instance.Filter_for_conflict()
        assert isinstance(result, dict)

    # ========== Find_by_title ==========
    def test_find_by_title_success(self, tools_instance):
        """Test Find_by_title with valid title returns a dict."""
        result = tools_instance.Find_by_title(q="Climate Crisis")
        assert isinstance(result, dict)

    def test_find_by_title_empty(self, tools_instance):
        """Test Find_by_title with empty string returns error dict."""
        result = tools_instance.Find_by_title(q="")
        self._assert_error_response(result)

    def test_find_by_title_none(self, tools_instance):
        """Test Find_by_title with None returns error dict."""
        result = tools_instance.Find_by_title(q=None)
        self._assert_error_response(result)

    # ========== General_search ==========
    def test_general_search_success(self, tools_instance):
        """Test General_search with valid searchId returns a dict."""
        result = tools_instance.General_search(searchId="12345")
        assert isinstance(result, dict)

    def test_general_search_empty_id(self, tools_instance):
        """Test General_search with empty searchId returns error dict."""
        result = tools_instance.General_search(searchId="")
        self._assert_error_response(result)

    def test_general_search_none_id(self, tools_instance):
        """Test General_search with None searchId returns error dict."""
        result = tools_instance.General_search(searchId=None)
        self._assert_error_response(result)

    # ========== Get_ALL_Feed ==========
    def test_get_all_feed_success(self, tools_instance):
        """Test Get_ALL_Feed returns a dict with an integer value."""
        result = tools_instance.Get_ALL_Feed()
        assert isinstance(result, dict)
        # At least one key should have an int value (return type Dict[str, int])
        for v in result.values():
            assert isinstance(v, int)

    # ========== Get_All_Climate_Change_News_3 ==========
    def test_get_all_climate_change_news_3_success(self, tools_instance):
        """Test Get_All_Climate_Change_News_3 returns a dict."""
        result = tools_instance.Get_All_Climate_Change_News_3()
        assert isinstance(result, dict)
        # Return type is Dict[str, str]
        for k, v in result.items():
            assert isinstance(k, str) and isinstance(v, str)

    # ========== Get_All_Climate_Change_Related_News ==========
    def test_get_all_climate_change_related_news_success(self, tools_instance):
        """Test Get_All_Climate_Change_Related_News returns a dict with int values."""
        result = tools_instance.Get_All_Climate_Change_Related_News()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, int)

    # ========== Get_All_Crypto_News ==========
    def test_get_all_crypto_news_success(self, tools_instance):
        """Test Get_All_Crypto_News returns a list of dicts."""
        result = tools_instance.Get_All_Crypto_News()
        assert isinstance(result, list)
        if result:  # may be empty list
            for item in result:
                assert isinstance(item, dict)

    # ========== Get_All_Narco_News ==========
    def test_get_all_narco_news_success(self, tools_instance):
        """Test Get_All_Narco_News returns a dict."""
        result = tools_instance.Get_All_Narco_News()
        assert isinstance(result, dict)

    # ========== Get_All_News ==========
    def test_get_all_news_success(self, tools_instance):
        """Test Get_All_News returns a dict."""
        result = tools_instance.Get_All_News()
        assert isinstance(result, dict)

    # ========== Get_All_Trump_Articles ==========
    def test_get_all_trump_articles_success(self, tools_instance):
        """Test Get_All_Trump_Articles returns a list of dicts."""
        result = tools_instance.Get_All_Trump_Articles()
        assert isinstance(result, list)
        if result:
            for item in result:
                assert isinstance(item, dict)

    # ========== Get_Latest_Article_List ==========
    def test_get_latest_article_list_success(self, tools_instance):
        """Test Get_Latest_Article_List with valid language returns a dict with int values."""
        result = tools_instance.Get_Latest_Article_List(language="en")
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, int)

    def test_get_latest_article_list_empty_language(self, tools_instance):
        """Test Get_Latest_Article_List with empty language returns error dict."""
        result = tools_instance.Get_Latest_Article_List(language="")
        self._assert_error_response(result)

    def test_get_latest_article_list_none_language(self, tools_instance):
        """Test Get_Latest_Article_List with None language returns error dict."""
        result = tools_instance.Get_Latest_Article_List(language=None)
        self._assert_error_response(result)

    # ========== Get_News ==========
    def test_get_news_success(self, tools_instance):
        """Test Get_News returns a dict."""
        result = tools_instance.Get_News()
        assert isinstance(result, dict)

    # ========== Get_all_AI_News ==========
    def test_get_all_ai_news_success(self, tools_instance):
        """Test Get_all_AI_News returns a dict."""
        result = tools_instance.Get_all_AI_News()
        assert isinstance(result, dict)

    # ========== Get_all_climate_change_news ==========
    def test_get_all_climate_change_news_success(self, tools_instance):
        """Test Get_all_climate_change_news returns a dict with int values."""
        result = tools_instance.Get_all_climate_change_news()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, int)

    # ========== Get_all_the_relevant_articles ==========
    def test_get_all_the_relevant_articles_success(self, tools_instance):
        """Test Get_all_the_relevant_articles returns a dict."""
        result = tools_instance.Get_all_the_relevant_articles()
        assert isinstance(result, dict)

    # ========== Get_all_the_top_articles_of_the_week_by_default ==========
    def test_get_all_the_top_articles_of_the_week_by_default_success(self, tools_instance):
        """Test Get_all_the_top_articles_of_the_week_by_default returns a dict."""
        result = tools_instance.Get_all_the_top_articles_of_the_week_by_default()
        assert isinstance(result, dict)

    # ========== Get_articles ==========
    def test_get_articles_success(self, tools_instance):
        """Test Get_articles with valid language returns a dict."""
        result = tools_instance.Get_articles(language="en")
        assert isinstance(result, dict)

    def test_get_articles_empty_language(self, tools_instance):
        """Test Get_articles with empty language returns error dict."""
        result = tools_instance.Get_articles(language="")
        self._assert_error_response(result)

    def test_get_articles_none_language(self, tools_instance):
        """Test Get_articles with None language returns error dict."""
        result = tools_instance.Get_articles(language=None)
        self._assert_error_response(result)

    # ========== Get_the_latest_30_News_from_IEEE_Spectrum ==========
    def test_get_the_latest_30_news_from_ieee_spectrum_success(self, tools_instance):
        """Test Get_the_latest_30_News_from_IEEE_Spectrum returns a dict."""
        result = tools_instance.Get_the_latest_30_News_from_IEEE_Spectrum()
        assert isinstance(result, dict)

    # ========== Get_the_month_s_top_articles ==========
    def test_get_the_month_s_top_articles_success(self, tools_instance):
        """Test Get_the_month_s_top_articles returns a dict."""
        result = tools_instance.Get_the_month_s_top_articles()
        assert isinstance(result, dict)

    # ========== GetFAQsByTopic ==========
    def test_get_faqs_by_topic_success(self, tools_instance):
        """Test GetFAQsByTopic with valid parameters returns a dict."""
        result = tools_instance.GetFAQsByTopic(topic="climate", page=1.0)
        assert isinstance(result, dict)

    def test_get_faqs_by_topic_invalid_page(self, tools_instance):
        """Test GetFAQsByTopic with negative page returns error dict."""
        result = tools_instance.GetFAQsByTopic(topic="climate", page=-1.0)
        self._assert_error_response(result)

    def test_get_faqs_by_topic_empty_topic(self, tools_instance):
        """Test GetFAQsByTopic with empty topic returns error dict."""
        result = tools_instance.GetFAQsByTopic(topic="", page=1.0)
        self._assert_error_response(result)

    def test_get_faqs_by_topic_none_topic(self, tools_instance):
        """Test GetFAQsByTopic with None topic returns error dict."""
        result = tools_instance.GetFAQsByTopic(topic=None, page=1.0)
        self._assert_error_response(result)

    def test_get_faqs_by_topic_none_page(self, tools_instance):
        """Test GetFAQsByTopic with None page returns error dict."""
        result = tools_instance.GetFAQsByTopic(topic="climate", page=None)
        self._assert_error_response(result)

    # ========== GetNews ==========
    def test_get_news_location_success(self, tools_instance):
        """Test GetNews with valid location returns a dict."""
        result = tools_instance.GetNews(location="us")
        assert isinstance(result, dict)

    def test_get_news_empty_location(self, tools_instance):
        """Test GetNews with empty location returns error dict."""
        result = tools_instance.GetNews(location="")
        self._assert_error_response(result)

    def test_get_news_none_location(self, tools_instance):
        """Test GetNews with None location returns error dict."""
        result = tools_instance.GetNews(location=None)
        self._assert_error_response(result)

    # ========== GetStats ==========
    def test_get_stats_success(self, tools_instance):
        """Test GetStats with valid location returns a dict."""
        result = tools_instance.GetStats(location="us")
        assert isinstance(result, dict)

    def test_get_stats_empty_location(self, tools_instance):
        """Test GetStats with empty location returns error dict."""
        result = tools_instance.GetStats(location="")
        self._assert_error_response(result)

    def test_get_stats_none_location(self, tools_instance):
        """Test GetStats with None location returns error dict."""
        result = tools_instance.GetStats(location=None)
        self._assert_error_response(result)

    # ========== Latest_News ==========
    def test_latest_news_success(self, tools_instance):
        """Test Latest_News returns a dict."""
        result = tools_instance.Latest_News()
        assert isinstance(result, dict)

    # ========== Recent_50 ==========
    def test_recent_50_success(self, tools_instance):
        """Test Recent_50 returns a dict."""
        result = tools_instance.Recent_50()
        assert isinstance(result, dict)

    # ========== Sources_List_New ==========
    def test_sources_list_new_success(self, tools_instance):
        """Test Sources_List_New returns a dict."""
        result = tools_instance.Sources_List_New()
        assert isinstance(result, dict)

    # ========== Suggest ==========
    def test_suggest_success(self, tools_instance):
        """Test Suggest with valid keyword returns a dict."""
        result = tools_instance.Suggest(keyword="climate")
        assert isinstance(result, dict)

    def test_suggest_empty_keyword(self, tools_instance):
        """Test Suggest with empty keyword returns error dict."""
        result = tools_instance.Suggest(keyword="")
        self._assert_error_response(result)

    def test_suggest_none_keyword(self, tools_instance):
        """Test Suggest with None keyword returns error dict."""
        result = tools_instance.Suggest(keyword=None)
        self._assert_error_response(result)

    # ========== actors_get_all_filmography ==========
    def test_actors_get_all_filmography_success(self, tools_instance):
        """Test actors_get_all_filmography with valid nconst returns a dict."""
        result = tools_instance.actors_get_all_filmography(nconst="nm0000102")
        assert isinstance(result, dict)

    def test_actors_get_all_filmography_empty_nconst(self, tools_instance):
        """Test actors_get_all_filmography with empty nconst returns error dict."""
        result = tools_instance.actors_get_all_filmography(nconst="")
        self._assert_error_response(result)

    def test_actors_get_all_filmography_none_nconst(self, tools_instance):
        """Test actors_get_all_filmography with None nconst returns error dict."""
        result = tools_instance.actors_get_all_filmography(nconst=None)
        self._assert_error_response(result)

    # ========== all_articles ==========
    def test_all_articles_success(self, tools_instance):
        """Test all_articles returns a dict."""
        result = tools_instance.all_articles()
        assert isinstance(result, dict)

    # ========== detik_search ==========
    def test_detik_search_success(self, tools_instance):
        """Test detik_search with valid params returns a dict."""
        result = tools_instance.detik_search(page=1.0, limit=10.0, keyword="news")
        assert isinstance(result, dict)

    def test_detik_search_invalid_page(self, tools_instance):
        """Test detik_search with negative page returns error dict."""
        result = tools_instance.detik_search(page=-1.0, limit=10.0, keyword="news")
        self._assert_error_response(result)

    def test_detik_search_invalid_limit(self, tools_instance):
        """Test detik_search with zero limit returns error dict."""
        result = tools_instance.detik_search(page=1.0, limit=0.0, keyword="news")
        self._assert_error_response(result)

    def test_detik_search_empty_keyword(self, tools_instance):
        """Test detik_search with empty keyword returns error dict."""
        result = tools_instance.detik_search(page=1.0, limit=10.0, keyword="")
        self._assert_error_response(result)

    def test_detik_search_none_params(self, tools_instance):
        """Test detik_search with None values returns error dict."""
        result = tools_instance.detik_search(page=None, limit=None, keyword=None)
        self._assert_error_response(result)

    # ========== fetch_all_mediabiasfactcheck_com_data ==========
    def test_fetch_all_mediabiasfactcheck_com_data_success(self, tools_instance):
        """Test fetch_all_mediabiasfactcheck_com_data returns a dict."""
        result = tools_instance.fetch_all_mediabiasfactcheck_com_data()
        assert isinstance(result, dict)

    # ========== filter_for_diease ==========
    def test_filter_for_diease_success(self, tools_instance):
        """Test filter_for_diease returns a dict."""
        result = tools_instance.filter_for_diease()
        assert isinstance(result, dict)

    # ========== get_all_climate_news ==========
    def test_get_all_climate_news_success(self, tools_instance):
        """Test get_all_climate_news returns a dict."""
        result = tools_instance.get_all_climate_news()
        assert isinstance(result, dict)

    # ========== getNews ==========
    def test_get_news_no_args_success(self, tools_instance):
        """Test getNews returns a dict."""
        result = tools_instance.getNews()
        assert isinstance(result, dict)

    # ========== news_list ==========
    def test_news_list_success(self, tools_instance):
        """Test news_list returns a dict with nested dict values."""
        result = tools_instance.news_list()
        assert isinstance(result, dict)
        # Return type: Dict[str, Dict[str, Any]]
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)

    # ========== osay ==========
    def test_osay_success(self, tools_instance):
        """Test osay with valid state returns a dict."""
        result = tools_instance.osay(state="California")
        assert isinstance(result, dict)

    def test_osay_empty_state(self, tools_instance):
        """Test osay with empty state returns error dict."""
        result = tools_instance.osay(state="")
        self._assert_error_response(result)

    def test_osay_none_state(self, tools_instance):
        """Test osay with None state returns error dict."""
        result = tools_instance.osay(state=None)
        self._assert_error_response(result)

    # ========== politicians ==========
    def test_politicians_success(self, tools_instance):
        """Test politicians returns a dict."""
        result = tools_instance.politicians()
        assert isinstance(result, dict)

    # ========== search ==========
    def test_search_success(self, tools_instance):
        """Test search with valid query returns a dict with nested dict values."""
        result = tools_instance.search(query="climate")
        assert isinstance(result, dict)
        # Return type: Dict[str, Dict[str, Any]]
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)

    def test_search_empty_query(self, tools_instance):
        """Test search with empty query returns error dict."""
        result = tools_instance.search(query="")
        self._assert_error_response(result)

    def test_search_none_query(self, tools_instance):
        """Test search with None query returns error dict."""
        result = tools_instance.search(query=None)
        self._assert_error_response(result)