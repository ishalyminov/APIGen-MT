import pytest
import json
from tools.toollens.sports import SportsTools


@pytest.fixture
def sports_tools():
    """Fixture providing a fresh SportsTools instance for each test."""
    # Deep copy using json roundtrip (config is None)
    config = json.loads(json.dumps(None))
    return SportsTools(initial_config=config)


# =============================================================================
# Correct sequential tests
# =============================================================================

class TestSportsToolsSequentialCorrect:
    """Correct ordered sequences of API calls."""

    def test_list_sports_then_categories(self, sports_tools):
        """Get list of sports, then request categories for the first sport."""
        # Step 1: get list of sports
        sports_resp = sports_tools.List_of_sports()
        assert isinstance(sports_resp, dict)
        # Assuming 'sports' key contains a list or 'data' key
        sports = sports_resp.get('sports', sports_resp.get('data', []))
        if not sports:
            # fallback: just use a known sport string
            sport = "soccer"
        else:
            sport = sports[0] if isinstance(sports, list) else sports

        # Step 2: get categories for the chosen sport
        categories_resp = sports_tools.categories_list(sport=sport)
        assert isinstance(categories_resp, dict)

    def test_autocomplete_then_fixtures(self, sports_tools):
        """Auto-complete a soccer query, then list upcoming fixtures."""
        # Step 1: auto-complete "soccer"
        autocomplete_resp = sports_tools.auto_complete(query="soccer")
        assert isinstance(autocomplete_resp, dict)

        # Step 2: list fixtures
        fixtures_resp = sports_tools.fixtures()
        assert isinstance(fixtures_resp, dict)

    def test_daily_calories_then_list(self, sports_tools):
        """Calculate daily calorie needs for a typical user, then list surebets."""
        # Step 1: daily calorie requirements
        cal_resp = sports_tools.Daily_calory_requirements(
            age=30,
            weight=70.0,
            height=175.0,
            gender="male",
            activityLevel="moderate"
        )
        assert isinstance(cal_resp, dict)

        # Step 2: list surebets
        list_resp = sports_tools.List()
        assert isinstance(list_resp, dict)

    def test_premier_league_then_competitions(self, sports_tools):
        """Get Premier League standings, then all competitions info."""
        # Step 1: Premier League standings
        pl_resp = sports_tools.Premier_League_Standings()
        assert isinstance(pl_resp, dict)

        # Step 2: all competitions information
        comp_resp = sports_tools.Get_all_competitions_information()
        assert isinstance(comp_resp, dict)

    def test_seasons_then_drivers_standings(self, sports_tools):
        """List F1 seasons, then drivers standings for current season."""
        # Step 1: list seasons
        seasons_resp = sports_tools.Seasons()
        assert isinstance(seasons_resp, dict)
        # Assuming seasons list is available
        seasons = seasons_resp.get('seasons', seasons_resp.get('data', []))
        if not seasons:
            # fallback: just call drivers standings directly
            pass

        # Step 2: drivers standings
        drivers_resp = sports_tools.Drivers_Standings()
        assert isinstance(drivers_resp, dict)


# =============================================================================
# Problematic sequential tests
# =============================================================================

class TestSportsToolsSequentialProblematic:
    """Problematic sequences – invalid parameters, nonexistent resources, etc."""

    def test_invalid_calorie_params_then_seasons(self, sports_tools):
        """Call daily calorie requirements with invalid (negative) weight,
        then call seasons – should not crash."""
        # Step 1: daily calory with negative weight
        cal_resp = sports_tools.Daily_calory_requirements(
            age=30,
            weight=-10.0,
            height=175.0,
            gender="male",
            activityLevel="moderate"
        )
        # Might return an error dict; we just check it's a dict
        assert isinstance(cal_resp, dict)

        # Step 2: call seasons – must still work
        seasons_resp = sports_tools.Seasons()
        assert isinstance(seasons_resp, dict)

    def test_autocomplete_empty_then_list_sports(self, sports_tools):
        """Auto-complete with empty string, then list sports –
        empty query should be handled gracefully."""
        # Step 1: empty query
        auto_resp = sports_tools.auto_complete(query="")
        assert isinstance(auto_resp, dict)

        # Step 2: list sports – must still work
        sports_resp = sports_tools.List_of_sports()
        assert isinstance(sports_resp, dict)

    def test_nonexistent_category_then_competitions(self, sports_tools):
        """Request categories for a non‑existent sport, then list competitions."""
        # Step 1: categories with unclear sport name
        cat_resp = sports_tools.categories_list(sport="non_existent_sport_xyz")
        assert isinstance(cat_resp, dict)
        # Typically an error or empty response

        # Step 2: competitions list – must work
        comp_resp = sports_tools.Competitions_2()
        assert isinstance(comp_resp, dict)

    def test_invalid_matchschedule_then_list_markets(self, sports_tools):
        """Call MatchSchedules with impossible day/month, then list available markets."""
        # Step 1: invalid date
        match_resp = sports_tools.MatchSchedules(day=32, year=2020, month=13)
        assert isinstance(match_resp, dict)

        # Step 2: list markets – should not be affected
        markets_resp = sports_tools.List_available_markets()
        assert isinstance(markets_resp, dict)

    def test_invalid_wnba_scoreboard_then_odds(self, sports_tools):
        """Request WNBA scoreboard with non‑numeric strings, then call Odds_6."""
        # Step 1: WNBA scoreboard with clearly invalid date
        wnba_resp = sports_tools.WNBA_ScoreBoard(month="Feb", day="30", year="2021")
        assert isinstance(wnba_resp, dict)

        # Step 2: odds – must work
        odds_resp = sports_tools.Odds_6()
        assert isinstance(odds_resp, dict)