import pytest
from tools.toollens.video_images import VideoImagesTools


@pytest.fixture
def video_images_instance():
    config = {
        "available_modes": [
            "mask",
            "blur",
            "replace"
        ],
        "movies": [
            {
                "id": 1,
                "title": "The Shawshank Redemption",
                "year": 1994,
                "genre": "Drama",
                "rating": 9.3
            },
            {
                "id": 2,
                "title": "The Godfather",
                "year": 1972,
                "genre": "Crime",
                "rating": 9.2
            },
            {
                "id": 3,
                "title": "Inception",
                "year": 2010,
                "genre": "Sci-Fi",
                "rating": 8.8
            },
            {
                "id": 4,
                "title": "Interstellar",
                "year": 2014,
                "genre": "Sci-Fi",
                "rating": 8.6
            },
            {
                "id": 5,
                "title": "Parasite",
                "year": 2019,
                "genre": "Thriller",
                "rating": 8.6
            }
        ],
        "empty_genre": [],
        "single_movie": [
            {
                "id": 6,
                "title": "Test Movie",
                "year": 2023,
                "genre": "Comedy",
                "rating": 7.0
            }
        ],
        "genres": [
            "Drama",
            "Crime",
            "Sci-Fi",
            "Thriller",
            "Comedy",
            "Action"
        ],
        "sort_options": [
            "title",
            "year",
            "rating"
        ],
        "default_sort": "year"
    }
    return VideoImagesTools(initial_config=config)


class TestGetListOfAvailableModes:
    """Tests for Get_list_of_available_modes method."""

    def test_returns_list(self, video_images_instance):
        result = video_images_instance.Get_list_of_available_modes()
        assert isinstance(result, list), "Return type should be list"

    def test_returns_configured_modes(self, video_images_instance):
        expected = ["mask", "blur", "replace"]
        result = video_images_instance.Get_list_of_available_modes()
        assert result == expected, "Should return the modes from config"


class TestListMovies:
    """Tests for List_Movies method."""

    def test_returns_dict_with_message_key(self, video_images_instance):
        result = video_images_instance.List_Movies()
        assert isinstance(result, dict), "Return type should be dict"
        assert "message" in result, "Result should contain 'message' key"
        assert isinstance(result["message"], str), "Message should be a string"

    def test_message_contains_available_movies(self, video_images_instance):
        result = video_images_instance.List_Movies()
        assert "Available movies" in result["message"], (
            "Message should indicate available movies"
        )
        # Check that at least one known movie title is present
        assert "The Shawshank Redemption" in result["message"], (
            "Message should include movie titles from config"
        )


class TestSortBy:
    """Tests for Sort_By method."""

    def test_valid_sort_returns_success_message(self, video_images_instance):
        result = video_images_instance.Sort_By(sort_by="rating")
        assert isinstance(result, dict), "Return type should be dict"
        assert "message" in result, "Result should contain 'message' key"
        assert "sorted by 'rating'" in result["message"], (
            "Message should confirm sorting by the given criterion"
        )

    def test_missing_sort_returns_error(self, video_images_instance):
        result = video_images_instance.Sort_By(sort_by=None)
        assert "message" in result
        assert "Error" in result["message"], (
            "Missing sort should return an error message"
        )

    def test_invalid_sort_returns_error(self, video_images_instance):
        result = video_images_instance.Sort_By(sort_by="not_valid")
        assert "message" in result
        assert "Error" in result["message"], (
            "Invalid sort option should return an error message"
        )