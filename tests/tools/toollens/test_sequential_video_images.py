import pytest
import json
from typing import List, Dict, Any
from tools.toollens.video_images import VideoImagesTools


# -----------------------------------------------------------------------------
# Base configuration as provided (movies are dicts – will be transformed later)
# -----------------------------------------------------------------------------
INITIAL_CONFIG: Dict[str, Any] = {
    "available_modes": ["mask", "blur", "replace"],
    "movies": [
        {"id": 1, "title": "The Shawshank Redemption", "year": 1994, "genre": "Drama", "rating": 9.3},
        {"id": 2, "title": "The Godfather", "year": 1972, "genre": "Crime", "rating": 9.2},
        {"id": 3, "title": "Inception", "year": 2010, "genre": "Sci-Fi", "rating": 8.8},
        {"id": 4, "title": "Interstellar", "year": 2014, "genre": "Sci-Fi", "rating": 8.6},
        {"id": 5, "title": "Parasite", "year": 2019, "genre": "Thriller", "rating": 8.6},
    ],
    "empty_genre": [],
    "single_movie": [
        {"id": 6, "title": "Test Movie", "year": 2023, "genre": "Comedy", "rating": 7.0}
    ],
    "genres": ["Drama", "Crime", "Sci-Fi", "Thriller", "Comedy", "Action"],
    "sort_options": ["title", "year", "rating"],
    "default_sort": "year"
}


# -----------------------------------------------------------------------------
# Fixture – deep copies the config and adapts the movies list to strings
# so that List_Movies works correctly with the given VideoImagesTools code.
# -----------------------------------------------------------------------------
@pytest.fixture
def tools_instance() -> VideoImagesTools:
    config = json.loads(json.dumps(INITIAL_CONFIG))
    # Transform movies dicts into a flat list of titles
    if isinstance(config.get("movies"), list) and all(isinstance(m, dict) for m in config["movies"]):
        config["movies"] = [m["title"] for m in config["movies"]]
    return VideoImagesTools(initial_config=config)


# #############################################################################
# Correct sequences – methods called in a sensible, orderly fashion
# #############################################################################
class TestVideoImagesToolsSequentialCorrect:
    """Tests that exercise valid, ordered sequences of API calls."""

    def test_get_modes_then_list_movies(self, tools_instance: VideoImagesTools) -> None:
        """Get available modes, then list the movie database."""
        # Step 1: Get list of available modes
        modes = tools_instance.Get_list_of_available_modes()
        assert isinstance(modes, list)
        assert "mask" in modes
        assert "blur" in modes
        assert "replace" in modes

        # Step 2: List all movies
        result = tools_instance.List_Movies()
        assert isinstance(result, dict)
        assert "message" in result
        # The message should contain at least one movie title
        assert "The Shawshank Redemption" in result["message"]

    def test_list_movies_then_sort_by_rating(self, tools_instance: VideoImagesTools) -> None:
        """List movies first, then apply sorting by rating."""
        # Step 1: List movies
        list_result = tools_instance.List_Movies()
        assert "message" in list_result

        # Step 2: Sort by rating
        sort_result = tools_instance.Sort_By(sort_by="rating")
        assert isinstance(sort_result, dict)
        assert "message" in sort_result
        assert "sorted by 'rating'" in sort_result["message"]

    def test_get_modes_then_sort_by_year(self, tools_instance: VideoImagesTools) -> None:
        """Fetch available modes, then sort results by year."""
        # Step 1: get modes
        modes = tools_instance.Get_list_of_available_modes()
        assert isinstance(modes, list) and len(modes) > 0

        # Step 2: sort by year
        result = tools_instance.Sort_By(sort_by="year")
        assert "sorted by 'year'" in result["message"]

    def test_full_sequence_modes_movies_sort(self, tools_instance: VideoImagesTools) -> None:
        """Complete trajectory: modes → movies → sort."""
        # Step 1: available modes
        tools_instance.Get_list_of_available_modes()

        # Step 2: list movies
        list_result = tools_instance.List_Movies()
        assert "Inception" in list_result["message"]

        # Step 3: sort by rating
        sort_result = tools_instance.Sort_By(sort_by="rating")
        assert "sorted by 'rating'" in sort_result["message"]

    def test_sort_by_popularity_then_get_modes(self, tools_instance: VideoImagesTools) -> None:
        """Apply a sort, then retrieve the modes list – order does not matter."""
        # Step 1: sort by popularity
        sort_result = tools_instance.Sort_By(sort_by="popularity")
        assert "sorted by 'popularity'" in sort_result["message"]

        # Step 2: get modes – should not be affected
        modes = tools_instance.Get_list_of_available_modes()
        assert isinstance(modes, list) and "mask" in modes


# #############################################################################
# Problematic sequences – invalid calls, missing arguments, etc.
# #############################################################################
class TestVideoImagesToolsSequentialProblematic:
    """Tests where one call is problematic; subsequent calls must not crash."""

    def test_sort_by_no_param_then_list_movies(self, tools_instance: VideoImagesTools) -> None:
        """Call Sort_By without a parameter, then verify List_Movies still works."""
        # Step 1: invalid call – missing sort_by
        error = tools_instance.Sort_By()
        assert isinstance(error, dict)
        assert "error" in error["message"].lower() or "required" in error["message"].lower()

        # Step 2: list movies – should succeed
        result = tools_instance.List_Movies()
        assert "message" in result
        assert "The Godfather" in result["message"]

    def test_sort_by_invalid_option_then_get_modes(self, tools_instance: VideoImagesTools) -> None:
        """Use an unknown sort option, then fetch available modes."""
        # Step 1: invalid sort_by value
        error = tools_instance.Sort_By(sort_by="nonexistent_option")
        assert "unknown sort option" in error["message"].lower()

        # Step 2: get modes – should still return the list
        modes = tools_instance.Get_list_of_available_modes()
        assert isinstance(modes, list) and len(modes) == 3

    def test_sort_by_empty_string_then_list_movies(self, tools_instance: VideoImagesTools) -> None:
        """Pass an empty string to Sort_By, then list movies."""
        # Step 1: empty sort_by
        error = tools_instance.Sort_By(sort_by="")
        assert "cannot be empty" in error["message"]

        # Step 2: list movies – must not be corrupted
        result = tools_instance.List_Movies()
        assert "Parasite" in result["message"]

    def test_sort_by_list_then_get_modes(self, tools_instance: VideoImagesTools) -> None:
        """Pass a list (invalid type) to Sort_By, then retrieve modes."""
        # Step 1: invalid argument type – a list instead of a string
        error = tools_instance.Sort_By(sort_by=["rating"])  # type: ignore[arg-type]
        assert "unknown sort option" in error["message"].lower()
        # The method checks against allowed_sorts (which are strings), so it should reject it.

        # Step 2: get modes – should work normally
        modes = tools_instance.Get_list_of_available_modes()
        assert "blur" in modes

    def test_sort_by_none_then_list_movies(self, tools_instance: VideoImagesTools) -> None:
        """Pass None explicitly, then list movies."""
        # Step 1: sort_by is None
        error = tools_instance.Sort_By(sort_by=None)
        assert "cannot be empty" in error["message"]

        # Step 2: list movies – should still work
        result = tools_instance.List_Movies()
        assert "Interstellar" in result["message"]