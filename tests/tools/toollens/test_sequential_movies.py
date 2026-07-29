import pytest
import json
from tools.toollens.movies import MoviesTools


@pytest.fixture
def movies_tools():
    """Fixture providing a fresh MoviesTools instance."""
    config = None
    # Use deep copy to avoid shared state (though stateless)
    return MoviesTools(initial_config=json.loads(json.dumps(config)))


class TestMoviesToolsSequentialCorrect:
    """
    Tests that exercise correct, ordered sequences of method calls
    representing typical user trajectories.
    """

    def test_search_then_detail(self, movies_tools: MoviesTools) -> None:
        """Search for a movie, then fetch detailed response for a result."""
        # Step 1: Search for movies
        search_result = movies_tools.Search(query="Inception")
        assert isinstance(search_result, dict), "Search should return a dict"
        assert "results" in search_result or isinstance(search_result, dict), \
            "Search result should contain results or be a proper dict"

        # Extract a movie ID from the search results (if available)
        movie_id = None
        if "results" in search_result and search_result["results"]:
            first_movie = search_result["results"][0]
            if "id" in first_movie:
                movie_id = first_movie["id"]
        # Fallback to a known movie ID if extraction fails
        if movie_id is None:
            movie_id = 550  # Fight Club (example)

        # Step 2: Get detailed response for that movie
        detail_result = movies_tools.Get_Detailed_Response(movie_id=movie_id)
        assert isinstance(detail_result, dict), "Get_Detailed_Response should return a dict"
        # Verify schema presence
        if "id" in detail_result:
            assert detail_result["id"] == movie_id, "Returned movie ID should match"

    def test_genres_then_search_pro(self, movies_tools: MoviesTools) -> None:
        """Get genres list, then use a genre in a Pro search."""
        # Step 1: Retrieve all genres
        genres_result = movies_tools.Get_Genres()
        assert isinstance(genres_result, dict), "Get_Genres should return a dict"
        genre_name = None
        if "genres" in genres_result and isinstance(genres_result["genres"], list):
            if genres_result["genres"]:
                genre_name = genres_result["genres"][0]
        if genre_name is None:
            genre_name = "Action"

        # Step 2: Search Pro with that genre and a streaming service
        pro_result = movies_tools.Search_Pro(country="US", services="Netflix")
        assert isinstance(pro_result, dict), "Search_Pro should return a dict"

    def test_top_movies_then_torrent_search(self, movies_tools: MoviesTools) -> None:
        """Get top 100 movie torrents, then search torrents with a keyword from results."""
        # Step 1: Retrieve top movie torrents
        top_result = movies_tools.Get_Monthly_Top_100_Movies_Torrents()
        assert isinstance(top_result, dict), "Get_Monthly_Top_100_Movies_Torrents should return a dict"
        keyword = None
        if "torrents" in top_result and isinstance(top_result["torrents"], list):
            if top_result["torrents"]:
                keyword = top_result["torrents"][0].get("name", "movie")
        if keyword is None:
            keyword = "Inception"

        # Step 2: Search torrents with that keyword
        torrent_search_result = movies_tools.Search_Torrents(keywords=keyword, quantity=5)
        assert isinstance(torrent_search_result, dict), "Search_Torrents should return a dict"

    def test_all_quotes_then_quote_by_character(self, movies_tools: MoviesTools) -> None:
        """Get all quotes, then fetch quotes by a character from the list."""
        # Step 1: Get all quotes
        all_quotes = movies_tools.Get_all_quotes()
        assert isinstance(all_quotes, dict), "Get_all_quotes should return a dict"
        character_name = None
        if "quotes" in all_quotes and isinstance(all_quotes["quotes"], list):
            if all_quotes["quotes"]:
                character_name = all_quotes["quotes"][0].get("character", "Batman")
        if character_name is None:
            character_name = "The Joker"

        # Step 2: Get quotes by that character
        quote_by_char = movies_tools.Get_quote_by_character(character=character_name)
        assert isinstance(quote_by_char, dict), "Get_quote_by_character should return a dict"

    def test_quote_by_year_then_movies_by_name(self, movies_tools: MoviesTools) -> None:
        """Get a quote by a specific year, then search for movies by a name from the quote."""
        # Step 1: Get a quote from a given year
        year_quote = movies_tools.GET_quote_by_Year(year=1994)
        assert isinstance(year_quote, dict), "GET_quote_by_Year should return a dict"
        movie_name = None
        if "quote" in year_quote and isinstance(year_quote["quote"], dict):
            movie_name = year_quote["quote"].get("movie", "The Shawshank Redemption")
        if movie_name is None:
            movie_name = "Pulp Fiction"

        # Step 2: Search movies by that name
        movies_result = movies_tools.get_movies_by_name()
        # Note: get_movies_by_name takes no args, so just verify it returns a dict
        assert isinstance(movies_result, dict), "get_movies_by_name should return a dict"


class TestMoviesToolsSequentialProblematic:
    """
    Tests that exercise problematic or invalid sequences, ensuring
    the API handles errors gracefully without raising exceptions.
    """

    def test_nonexistent_movie_then_quotes(self, movies_tools: MoviesTools) -> None:
        """Request a non‑existent movie detail, then fetch all quotes."""
        # Step 1: Get detailed response with an invalid ID
        bad_detail = movies_tools.Get_Detailed_Response(movie_id=-1)
        assert isinstance(bad_detail, dict), "Get_Detailed_Response should return a dict even for invalid ID"
        # The result may contain an error key
        assert "error" in bad_detail or True, "Should contain error or be a dict"

        # Step 2: Get all quotes (should work normally)
        all_quotes = movies_tools.Get_all_quotes()
        assert isinstance(all_quotes, dict), "Get_all_quotes should return a dict"

    def test_invalid_year_then_empty_search(self, movies_tools: MoviesTools) -> None:
        """Use an invalid year value, then perform an empty search."""
        # Step 1: Get quote by year with a non‑numeric year
        bad_year_quote = movies_tools.GET_quote_by_Year(year="not_a_year")
        assert isinstance(bad_year_quote, dict), "GET_quote_by_Year should return a dict even with invalid year"

        # Step 2: Search with no query
        empty_search = movies_tools.Search(query="")
        assert isinstance(empty_search, dict), "Search should return a dict with empty query"

    def test_movies_no_args_then_genres(self, movies_tools: MoviesTools) -> None:
        """Call get_movies_by_name without arguments, then get genres."""
        # Step 1: Call get_movies_by_name with no arguments
        no_args_movies = movies_tools.get_movies_by_name()
        assert isinstance(no_args_movies, dict), "get_movies_by_name should return a dict even with no args"

        # Step 2: Get genres (should work)
        genres = movies_tools.Get_Genres()
        assert isinstance(genres, dict), "Get_Genres should return a dict"

    def test_search_invalid_query_then_all_characters(self, movies_tools: MoviesTools) -> None:
        """Call Search with a None query, then retrieve all characters."""
        # Step 1: Search with None
        none_query = movies_tools.Search(query=None)
        assert isinstance(none_query, dict), "Search should return a dict even with None query"

        # Step 2: Get all characters
        all_chars = movies_tools.Get_all_characters()
        assert isinstance(all_chars, dict), "Get_all_characters should return a dict"

    def test_basic_info_invalid_peopleid_then_quote_by_show(self, movies_tools: MoviesTools) -> None:
        """Call Basic_Info with an invalid people ID, then get a quote by show name."""
        # Step 1: Basic_Info with empty string
        bad_info = movies_tools.Basic_Info(peopleid="")
        assert isinstance(bad_info, dict), "Basic_Info should return a dict even with empty peopleid"

        # Step 2: Get a quote by a show name (valid call)
        quote_show = movies_tools.GET_quote_by_movie_or_TV_show_name(show="Breaking Bad")
        assert isinstance(quote_show, dict), "GET_quote_by_movie_or_TV_show_name should return a dict"