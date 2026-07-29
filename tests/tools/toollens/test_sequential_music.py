import pytest
import json
from copy import deepcopy
from typing import Any, Dict, List

from tools.toollens.music import MusicTools

# Initial configuration from the tool definition
INITIAL_CONFIG: Dict[str, Any] = {
    "videos": {
        "youtube_24h": ["dQw4w9WgXcQ", "3tmd-ClpJxA"],
        "youtube_trending_overall": ["9bZkp7q19f0"],
        "youtube_weekly": []
    },
    "charts": {
        "Artist_100": [
            {"rank": 1, "artist": "Taylor Swift", "weeks": 10}
        ],
        "BILLBOARD_200": [
            {"rank": 1, "album": "Midnights", "artist": "Taylor Swift", "last_week": 1}
        ],
        "Billboard_200": [
            {"rank": 1, "album": "Midnights", "artist": "Taylor Swift"}
        ],
        "Billboard_Global_Excl_US": [],
        "Billboard_Hot_100": [
            {"rank": 1, "song": "Anti-Hero", "artist": "Taylor Swift"}
        ],
        "Catalog_Albums": [
            {"rank": 1, "album": "Thriller", "artist": "Michael Jackson"}
        ],
        "Greatest_of_All_Time_Hot_100_Songs": [
            {"rank": 1, "song": "Blinding Lights", "artist": "The Weeknd"}
        ],
        "Independent_Albums": [
            {"rank": 1, "album": "RTJ4", "artist": "Run The Jewels"}
        ],
        "Year_End_Billboard_Global_200": [
            {"rank": 1, "song": "Heat Waves", "artist": "Glass Animals"}
        ],
        "Year_End_Top_Artists": [
            {"rank": 1, "artist": "Taylor Swift"}
        ]
    },
    "users": {
        "spotify_user_1": {
            "display_name": "John Doe",
            "id": "spotify_user_1",
            "followers": 100
        },
        "unknown_user": {}
    },
    "albums": {
        "List_User_Albums": {
            "spotify_user_1": ["album1", "album2"],
            "empty_user": []
        },
        "New_releases": {
            "US": [{"album": "Album1", "artist": "Artist1"}],
            "GB": []
        }
    },
    "search": {
        "auto_complete": {
            "hello": ["hello world", "hello goodbye"],
            "empty": []
        },
        "boy_groups": ["BTS", "EXO"],
        "girl_groups": ["BLACKPINK", "TWICE"],
        "random_boy_group": "BTS",
        "random_song": {
            "artist": "Taylor Swift",
            "album": "1989",
            "song": "Shake It Off"
        }
    },
    "channels": ["channel1", "channel2"]
}


@pytest.fixture
def music_instance() -> MusicTools:
    """Returns a fresh MusicTools instance with a deep copy of the initial config."""
    config = json.loads(json.dumps(INITIAL_CONFIG))
    return MusicTools(initial_config=config)


# ----------------------------------------------------------------------
# Correct Sequential Tests
# ----------------------------------------------------------------------

class TestMusicToolsSequentialCorrect:
    """Tests that exercise correct ordered sequences of method calls."""

    def test_youtube_sequence(self, music_instance: MusicTools) -> None:
        """Retrieve 24h trending, then overall trending, then weekly videos."""
        # Step 1: youtube 24h
        result_24h = music_instance.youtube_24h()
        assert isinstance(result_24h, dict)
        # Should contain the two configured video IDs
        assert "dQw4w9WgXcQ" in result_24h or "3tmd-ClpJxA" in result_24h

        # Step 2: youtube trending overall
        result_overall = music_instance.youtube_trending_overall()
        assert isinstance(result_overall, dict)
        assert "9bZkp7q19f0" in str(result_overall)

        # Step 3: youtube weekly (empty list in config)
        result_weekly = music_instance.youtube_weekly()
        assert isinstance(result_weekly, dict)

    def test_user_album_sequence(self, music_instance: MusicTools) -> None:
        """Get user details, then list albums for that user."""
        # Step 1: Get user details for known user
        user_details = music_instance.User_details("spotify_user_1")
        assert isinstance(user_details, dict)
        assert user_details.get("display_name") == "John Doe"

        # Step 2: List albums for same user
        user_albums = music_instance.List_User_Albums("spotify_user_1")
        assert isinstance(user_albums, dict)
        albums = user_albums.get("albums") or user_albums.get("List_User_Albums", [])
        assert len(albums) > 0

    def test_billboard_chart_sequence(self, music_instance: MusicTools) -> None:
        """Fetch Hot 100, Catalog Albums, and Independent Albums in sequence."""
        hot_100 = music_instance.Billboard_Hot_100()
        assert isinstance(hot_100, dict)
        # Expect a song entry
        assert "Anti-Hero" in str(hot_100)

        catalog = music_instance.Catalog_Albums()
        assert isinstance(catalog, dict)
        # Should contain Thriller
        assert "Thriller" in str(catalog)

        indep = music_instance.Independent_Albums()
        assert isinstance(indep, dict)
        assert "RTJ4" in str(indep)

    def test_search_sequence(self, music_instance: MusicTools) -> None:
        """Auto-complete a term, then search boy/girl groups."""
        # Step 1: auto complete "hel"
        auto_result = music_instance.auto_complete("hel")
        assert isinstance(auto_result, dict)
        completions = auto_result.get("auto_complete", [])
        assert len(completions) >= 2

        # Step 2: search boy groups with "B"
        boy_result = music_instance.boy_groups("B")
        assert isinstance(boy_result, dict)
        # Should contain BTS
        groups_boy = boy_result.get("boy_groups", [])
        assert "BTS" in groups_boy

        # Step 3: search girl groups with "B"
        girl_result = music_instance.girl_groups("B")
        assert isinstance(girl_result, dict)
        groups_girl = girl_result.get("girl_groups", [])
        assert "BLACKPINK" in groups_girl

    def test_chart_year_sequence(self, music_instance: MusicTools) -> None:
        """Call Artist_100 with a date, then Year_End charts."""
        # Step 1: Artist_100 for a specific date
        artist_100 = music_instance.Artist_100("2023-01-01")
        assert isinstance(artist_100, dict)
        # Verify rank 1 artist is Taylor Swift
        assert "Taylor Swift" in str(artist_100)

        # Step 2: Year-End Billboard Global 200 for 2023
        year_global = music_instance.Year_End_Billboard_Global_200(2023)
        assert isinstance(year_global, dict)
        assert "Heat Waves" in str(year_global)

        # Step 3: Year-End Top Artists for 2023
        year_artists = music_instance.Year_End_Top_Artists(2023)
        assert isinstance(year_artists, dict)
        assert "Taylor Swift" in str(year_artists)


# ----------------------------------------------------------------------
# Problematic Sequences
# ----------------------------------------------------------------------

class TestMusicToolsSequentialProblematic:
    """Tests that exercise problematic or invalid parameter sequences."""

    def test_nonexistent_user_sequence(self, music_instance: MusicTools) -> None:
        """Get details for a nonexistent user, then try to list their albums."""
        # Step 1: Request details for a user that does not exist
        unknown_details = music_instance.User_details("nonexistent_user")
        assert isinstance(unknown_details, dict)
        # The response should be empty or contain an error indicator
        # According to spec, method never raises; returns dict with error info
        assert not unknown_details or "error" in unknown_details

        # Step 2: List albums for the same nonexistent user
        empty_albums = music_instance.List_User_Albums("nonexistent_user")
        assert isinstance(empty_albums, dict)
        # Expect empty list or error
        albums = empty_albums.get("albums") or empty_albums.get("List_User_Albums", [])
        # Should be empty
        assert albums == []

    def test_invalid_chart_date_sequence(self, music_instance: MusicTools) -> None:
        """Call chart methods with invalid date strings."""
        # Step 1: Artist_100 with invalid date
        bad_artist = music_instance.Artist_100("not-a-date")
        assert isinstance(bad_artist, dict)
        # Should indicate error (e.g., empty dict or error key)
        assert not bad_artist or "error" in bad_artist

        # Step 2: Billboard_200_2 with invalid date
        bad_billboard = music_instance.Billboard_200_2("")
        assert isinstance(bad_billboard, dict)
        # Should also handle gracefully
        assert not bad_billboard or "error" in bad_billboard

        # Step 3: Billboard_Global_Excl_US with invalid date
        bad_global = music_instance.Billboard_Global_Excl_US("2023-13-01")  # invalid month
        assert isinstance(bad_global, dict)

    def test_empty_autocomplete_sequence(self, music_instance: MusicTools) -> None:
        """Call auto_complete with empty string, then with a term that returns empty."""
        # Step 1: auto_complete with empty input
        empty_term = music_instance.auto_complete("")
        assert isinstance(empty_term, dict)
        completions = empty_term.get("auto_complete", [])
        assert completions == []  # should be empty

        # Step 2: auto_complete with a term that yields no results
        no_result = music_instance.auto_complete("empty")
        assert isinstance(no_result, dict)
        completions2 = no_result.get("auto_complete", [])
        assert completions2 == []

    def test_unsupported_country_sequence(self, music_instance: MusicTools) -> None:
        """Request new releases for unsupported country, then for valid one."""
        # Step 1: Unsupported country
        unsupported = music_instance.New_releases("XX")
        assert isinstance(unsupported, list)
        # Should be empty list
        assert unsupported == []

        # Step 2: Supported country (US)
        supported = music_instance.New_releases("US")
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert supported[0]["album"] == "Album1"

    def test_random_song_invalid_artist_sequence(self, music_instance: MusicTools) -> None:
        """Call random song methods with invalid artist/album."""
        # Step 1: random song from an artist (but artist doesn't exist in config)
        bad_artist_song = music_instance.random_song_song_s_album_information_out_of_artist("NonexistentArtist")
        assert isinstance(bad_artist_song, dict)
        # Should return empty or error info
        assert not bad_artist_song or "error" in bad_artist_song

        # Step 2: random song from a specific artist and album (both invalid)
        bad_artist_album = music_instance.random_song_from_a_specific_artist_and_specified_album(
            "NonexistentArtist", "NonexistentAlbum"
        )
        assert isinstance(bad_artist_album, dict)
        # Expect empty dict or error
        assert not bad_artist_album or "error" in bad_artist_album

        # Step 3: call with valid data to ensure tool still works
        valid_song = music_instance.random_song_from_a_specific_artist_and_specified_album(
            "Taylor Swift", "1989"
        )
        assert isinstance(valid_song, dict)
        # Should contain the song
        assert "Shake It Off" in str(valid_song) or valid_song.get("song") == "Shake It Off"