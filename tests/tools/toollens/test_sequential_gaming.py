import json
import pytest
from tools.toollens.gaming import GamingTools


@pytest.fixture
def gaming_tools() -> GamingTools:
    """Provide a fresh GamingTools instance for each test."""
    return GamingTools()


class TestGamingToolsSequentialCorrect:
    """Correct ordered sequences of GamingTools API calls."""

    def test_eu_us_sequence(self, gaming_tools: GamingTools) -> None:
        """Call eu() twice, then us() – verify both return appropriate dicts."""
        eu1 = gaming_tools.eu()
        eu2 = gaming_tools.eu()
        us = gaming_tools.us()

        # EU returns a random number between 1 and 90
        assert isinstance(eu1, dict)
        assert "random" in eu1 or "number" in eu1
        assert 1 <= eu1.get("random", eu1.get("number")) <= 90

        assert isinstance(eu2, dict)
        assert "random" in eu2 or "number" in eu2
        assert 1 <= eu2.get("random", eu2.get("number")) <= 90

        # US returns a 5×5 Bingo card (list of 5 lists, each of length 5)
        assert isinstance(us, dict)
        card = us.get("card")
        assert card is not None
        assert len(card) == 5
        for row in card:
            assert isinstance(row, list)
            assert len(row) == 5
            for num in row:
                assert 1 <= num <= 75

    def test_lost_ark_classes_dungeons(self, gaming_tools: GamingTools) -> None:
        """Retrieve Lost Ark classes first, then abyssal dungeons."""
        classes = gaming_tools.get_classes()
        dungeons = gaming_tools.get_abyssal_dungeons()

        assert isinstance(classes, dict)
        assert "classes" in classes
        assert isinstance(classes["classes"], list)

        assert isinstance(dungeons, dict)
        assert "dungeons" in dungeons
        assert isinstance(dungeons["dungeons"], list)

    def test_charades_and_characters(self, gaming_tools: GamingTools) -> None:
        """Get MVC2 characters then a random charades word."""
        chars = gaming_tools.All_Characters()
        word = gaming_tools.Get_Charades_Word(difficulty="easy")

        assert isinstance(chars, dict)
        assert "characters" in chars
        assert isinstance(chars["characters"], list)

        assert isinstance(word, dict)
        assert "word" in word
        assert "difficulty" in word
        assert word["difficulty"] == "easy"

    def test_player_details_and_dice(self, gaming_tools: GamingTools) -> None:
        """Fetch player details (existing player) then roll dice."""
        player = gaming_tools.Get_Player_Details(name="JohnDoe", region="US")
        dice = gaming_tools.Regular_dice_rolls()

        assert isinstance(player, dict)
        assert "name" in player
        assert "region" in player
        assert player["name"] == "JohnDoe"
        assert player["region"] == "US"

        assert isinstance(dice, dict)
        assert "total" in dice
        assert "rolls" in dice
        assert isinstance(dice["rolls"], list)
        # A typical dice roll: 2 dice, sum between 2 and 12 (if 2 dice)
        if len(dice["rolls"]) > 0:
            min_possible = len(dice["rolls"])
            max_possible = len(dice["rolls"]) * 6
            assert min_possible <= dice["total"] <= max_possible

    def test_schedule_and_nadeo(self, gaming_tools: GamingTools) -> None:
        """Get league schedules then Nadeo players list."""
        schedule = gaming_tools.Get_Schedule()
        nadeo = gaming_tools.Nadeo_Players()

        assert isinstance(schedule, dict)
        assert "schedules" in schedule
        assert isinstance(schedule["schedules"], list)

        assert isinstance(nadeo, dict)
        assert "players" in nadeo
        assert isinstance(nadeo["players"], list)


class TestGamingToolsSequentialProblematic:
    """Problematic sequences: invalid/missing parameters, nonexistent resources."""

    def test_nonexistent_player_then_eu(self, gaming_tools: GamingTools) -> None:
        """Request a non-existing player (should return error), then eu() still works."""
        player = gaming_tools.Get_Player_Details(name="__nonexistent__", region="XX")
        # Expect an error response
        assert isinstance(player, dict)
        assert "error" in player or "message" in player

        # Subsequent call must not crash
        eu = gaming_tools.eu()
        assert isinstance(eu, dict)
        assert "random" in eu or "number" in eu

    def test_invalid_charades_difficulty_then_us(self, gaming_tools: GamingTools) -> None:
        """Attempt charades with invalid difficulty, then us() still returns a card."""
        word = gaming_tools.Get_Charades_Word(difficulty="super_hard")
        assert isinstance(word, dict)
        # Should indicate an error (e.g., invalid difficulty)
        assert "error" in word or "message" in word

        us = gaming_tools.us()
        assert isinstance(us, dict)
        card = us.get("card")
        assert card is not None and len(card) == 5
        for row in card:
            assert len(row) == 5

    def test_missing_player_params_then_schedule(self, gaming_tools: GamingTools) -> None:
        """Call Get_Player_Details with missing/invalid arguments, then Get_Schedule."""
        # Call with None parameters (method should handle gracefully)
        player = gaming_tools.Get_Player_Details(name=None, region=None)  # type: ignore
        assert isinstance(player, dict)
        # Expect an error because name/region are required
        assert "error" in player or "message" in player

        schedule = gaming_tools.Get_Schedule()
        assert isinstance(schedule, dict)
        assert "schedules" in schedule

    def test_invalid_types_then_abyssal(self, gaming_tools: GamingTools) -> None:
        """Use wrong types for Get_Player_Details, then ensure get_abyssal_dungeons works."""
        player = gaming_tools.Get_Player_Details(name=123, region=["US"])  # type: ignore
        assert isinstance(player, dict)
        # Expect an error due to type mismatch
        assert "error" in player or "message" in player

        dungeons = gaming_tools.get_abyssal_dungeons()
        assert isinstance(dungeons, dict)
        assert "dungeons" in dungeons
        assert isinstance(dungeons["dungeons"], list)

    def test_nonexistent_stronghold_recipe_then_all_islands(self, gaming_tools: GamingTools) -> None:
        """First call Get_Stronghold_Item_Recipes (assume resource exists), then call
        Get_all_island_with_dropped_items. Both should return valid responses."""
        recipes = gaming_tools.Get_Stronghold_Item_Recipes()
        assert isinstance(recipes, dict)
        # If empty, it might be a valid empty list, not an error
        if "error" in recipes:
            # Error case: still fine, no crash
            pass
        else:
            assert "recipes" in recipes
            assert isinstance(recipes["recipes"], list)

        islands = gaming_tools.Get_all_island_with_dropped_items()
        assert isinstance(islands, dict)
        if "error" in islands:
            pass
        else:
            assert "islands" in islands
            assert isinstance(islands["islands"], list)