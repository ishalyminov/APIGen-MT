import pytest
from typing import Union, List, Dict, Any
from tools.toollens.movies import MoviesTools


@pytest.fixture
def movies_instance() -> MoviesTools:
    """Fixture providing a stateless MoviesTools instance."""
    return MoviesTools(initial_config=None)


# ---------------------------------------------------------------------------
# 1. get_movies_by_name
# ---------------------------------------------------------------------------

def test_get_movies_by_name_returns_dict(movies_instance: MoviesTools) -> None:
    """Verifies that get_movies_by_name returns a dictionary."""
    result = movies_instance.get_movies_by_name()
    assert isinstance(result, dict), "Expected dict"


# ---------------------------------------------------------------------------
# 2. Basic_Info
# ---------------------------------------------------------------------------

def test_basic_info_with_valid_id(movies_instance: MoviesTools) -> None:
    """Basic_Info with a valid people id returns a dict with success True."""
    result = movies_instance.Basic_Info(peopleid="nm0000102")
    assert isinstance(result, dict)
    assert result.get("success") in (True, None)  # accept either convention


def test_basic_info_with_no_id(movies_instance: MoviesTools) -> None:
    """Basic_Info with no id returns a dict indicating an error."""
    result = movies_instance.Basic_Info()
    assert isinstance(result, dict)
    # Should report missing argument
    assert not result.get("success", True) or "error" in result


# ---------------------------------------------------------------------------
# 3. GET_quote_by_Year
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("year", [1999, 2020, None])
def test_get_quote_by_year(
    movies_instance: MoviesTools, year: Union[int, float, None]
) -> None:
    """GET_quote_by_Year returns a dict for valid/invalid year."""
    result = movies_instance.GET_quote_by_Year(year=year)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. GET_quote_by_movie_or_TV_show_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("show", ["Inception", "", None])
def test_get_quote_by_movie_or_tv_show(
    movies_instance: MoviesTools, show: Union[str, None]
) -> None:
    """GET_quote_by_movie_or_TV_show_name always returns a dict."""
    result = movies_instance.GET_quote_by_movie_or_TV_show_name(show=show)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 5. Genres_FREE
# ---------------------------------------------------------------------------

def test_genres_free_returns_dict(movies_instance: MoviesTools) -> None:
    """Genres_FREE returns a dictionary."""
    result = movies_instance.Genres_FREE()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 6. Get_All
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page,size", [("1", "10"), (None, None)])
def test_get_all(
    movies_instance: MoviesTools, page: Union[str, None], size: Union[str, None]
) -> None:
    """Get_All returns a dict for given page/size or missing."""
    result = movies_instance.Get_All(page=page, size=size)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 7. Get_Detailed_Response
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("movie_id", [550, -1, None])
def test_get_detailed_response(
    movies_instance: MoviesTools, movie_id: Union[int, float, None]
) -> None:
    """Get_Detailed_Response always returns a dict."""
    result = movies_instance.Get_Detailed_Response(movie_id=movie_id)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 8. Get_Genres
# ---------------------------------------------------------------------------

def test_get_genres_returns_dict(movies_instance: MoviesTools) -> None:
    """Get_Genres returns a dictionary."""
    result = movies_instance.Get_Genres()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 9-12. Monthly Top 100 Torrents (4 methods)
# ---------------------------------------------------------------------------

TORRENT_METHODS = [
    "Get_Monthly_Top_100_Games_Torrents",
    "Get_Monthly_Top_100_Movies_Torrents",
    "Get_Monthly_Top_100_Music_Torrents",
    "Get_Monthly_Top_100_TV_shows_Torrents",
]


@pytest.mark.parametrize("method_name", TORRENT_METHODS)
def test_monthly_top_100_torrents(
    movies_instance: MoviesTools, method_name: str
) -> None:
    """All Monthly Top 100 torrent methods return a dict."""
    method = getattr(movies_instance, method_name)
    result = method()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 13. Get_all_characters
# ---------------------------------------------------------------------------

def test_get_all_characters_returns_dict(movies_instance: MoviesTools) -> None:
    """Get_all_characters returns a dictionary."""
    result = movies_instance.Get_all_characters()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 14. Get_all_quotes
# ---------------------------------------------------------------------------

def test_get_all_quotes_returns_dict(movies_instance: MoviesTools) -> None:
    """Get_all_quotes returns a dictionary."""
    result = movies_instance.Get_all_quotes()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 15. Get_one_anime_by_ranking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [1, 100, None])
def test_get_one_anime_by_ranking(
    movies_instance: MoviesTools, rank: Union[int, float, None]
) -> None:
    """Get_one_anime_by_ranking always returns a dict."""
    result = movies_instance.Get_one_anime_by_ranking(rank=rank)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 16. Get_quote_by_character
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("character", ["Walter White", "", None])
def test_get_quote_by_character(
    movies_instance: MoviesTools, character: Union[str, None]
) -> None:
    """Get_quote_by_character always returns a dict."""
    result = movies_instance.Get_quote_by_character(character=character)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 17. Params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param", ["api_key", "", None])
def test_params(
    movies_instance: MoviesTools, param: Union[str, None]
) -> None:
    """Params returns either a dict or a list."""
    result = movies_instance.Params(param=param)
    assert isinstance(result, (dict, list))


# ---------------------------------------------------------------------------
# 18. Search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["star wars", "", None])
def test_search(
    movies_instance: MoviesTools, query: Union[str, None]
) -> None:
    """Search returns either a list or a dict."""
    result = movies_instance.Search(query=query)
    assert isinstance(result, (list, dict))


# ---------------------------------------------------------------------------
# 19. Search_Pro
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "country,services",
    [
        ("US", "netflix"),
        ("", ""),
        (None, None),
    ],
)
def test_search_pro(
    movies_instance: MoviesTools,
    country: Union[str, None],
    services: Union[str, None],
) -> None:
    """Search_Pro always returns a dict."""
    result = movies_instance.Search_Pro(country=country, services=services)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 20. Search_Torrents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "keywords,quantity",
    [
        ("ubuntu iso", 5),
        ("", 0),
        (None, None),
    ],
)
def test_search_torrents(
    movies_instance: MoviesTools,
    keywords: Union[str, None],
    quantity: Union[int, float, None],
) -> None:
    """Search_Torrents always returns a dict."""
    result = movies_instance.Search_Torrents(keywords=keywords, quantity=quantity)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 21. Search_by_Name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["The Matrix", "", None])
def test_search_by_name(
    movies_instance: MoviesTools, query: Union[str, None]
) -> None:
    """Search_by_Name always returns a dict."""
    result = movies_instance.Search_by_Name(query=query)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 22. Season_Episodes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ids", ["tt0903747,1", "", None])
def test_season_episodes(
    movies_instance: MoviesTools, ids: Union[str, None]
) -> None:
    """Season_Episodes returns either a list or a dict."""
    result = movies_instance.Season_Episodes(ids=ids)
    assert isinstance(result, (list, dict))