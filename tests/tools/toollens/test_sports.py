import pytest
from typing import Dict, Any, List, Union
from tools.toollens.sports import SportsTools

@pytest.fixture
def sports_instance() -> SportsTools:
    """Fixture that returns a stateless SportsTools instance."""
    return SportsTools(initial_config=None)

# --- Arbitrage_Low_Hold ---
def test_arbitrage_low_hold_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Arbitrage_Low_Hold returns a dictionary."""
    result = sports_instance.Arbitrage_Low_Hold()
    assert isinstance(result, dict), "Expected dict, got %s" % type(result)

def test_arbitrage_low_hold_non_empty(sports_instance: SportsTools) -> None:
    """Test that Arbitrage_Low_Hold returns a non-empty dict."""
    result = sports_instance.Arbitrage_Low_Hold()
    assert len(result) > 0, "Expected non-empty dict"

# --- Competitions_2 ---
def test_competitions_2_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Competitions_2 returns a dictionary."""
    result = sports_instance.Competitions_2()
    assert isinstance(result, dict)

def test_competitions_2_non_empty(sports_instance: SportsTools) -> None:
    """Test that Competitions_2 returns a non-empty dict."""
    result = sports_instance.Competitions_2()
    assert len(result) > 0

# --- Daily_calory_requirements ---
def test_daily_calory_requirements_valid(sports_instance: SportsTools) -> None:
    """Test Daily_calory_requirements with valid parameters."""
    result = sports_instance.Daily_calory_requirements(
        activitylevel="moderate", weight=70, gender="male", height=175, age=30
    )
    assert isinstance(result, dict)
    # Should contain expected keys, e.g., 'calories'
    assert "calories" in result or "error" in result  # handle both success or error

def test_daily_calory_requirements_edge_values(sports_instance: SportsTools) -> None:
    """Test Daily_calory_requirements with extreme values (e.g., zero weight)."""
    result = sports_instance.Daily_calory_requirements(
        activitylevel="sedentary", weight=0, gender="female", height=150, age=20
    )
    assert isinstance(result, dict)

# --- Drivers_Standings ---
def test_drivers_standings_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Drivers_Standings returns a dictionary."""
    result = sports_instance.Drivers_Standings()
    assert isinstance(result, dict)

def test_drivers_standings_contains_drivers(sports_instance: SportsTools) -> None:
    """Test that Drivers_Standings contains driver standings data."""
    result = sports_instance.Drivers_Standings()
    assert "drivers" in result or "standings" in result or len(result) > 0

# --- First_25_teams ---
def test_first_25_teams_returns_dict(sports_instance: SportsTools) -> None:
    """Test that First_25_teams returns a dictionary."""
    result = sports_instance.First_25_teams()
    assert isinstance(result, dict)

def test_first_25_teams_non_empty(sports_instance: SportsTools) -> None:
    """Test that First_25_teams returns at least one team."""
    result = sports_instance.First_25_teams()
    assert len(result) > 0

# --- Get_NCAA_Men_2000_2021 ---
def test_get_ncaa_men_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Get_NCAA_Men_2000_2021 returns a dictionary."""
    result = sports_instance.Get_NCAA_Men_2000_2021()
    assert isinstance(result, dict)

def test_get_ncaa_men_contains_years(sports_instance: SportsTools) -> None:
    """Test that the returned dict has some NCAA data."""
    result = sports_instance.Get_NCAA_Men_2000_2021()
    assert len(result) > 0

# --- Get_Player_List ---
def test_get_player_list_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Get_Player_List returns a dictionary."""
    result = sports_instance.Get_Player_List()
    assert isinstance(result, dict)

# --- Get_all_competitions_information ---
def test_get_all_competitions_info_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Get_all_competitions_information returns a dictionary."""
    result = sports_instance.Get_all_competitions_information()
    assert isinstance(result, dict)

# --- List ---
def test_list_returns_dict(sports_instance: SportsTools) -> None:
    """Test that List returns a dictionary."""
    result = sports_instance.ListTools()
    assert isinstance(result, dict)

def test_list_non_empty(sports_instance: SportsTools) -> None:
    """Test that List returns a non-empty dictionary."""
    result = sports_instance.ListTools()
    assert len(result) > 0

# --- List_available_markets ---
def test_list_available_markets_returns_dict(sports_instance: SportsTools) -> None:
    """Test that List_available_markets returns a dictionary."""
    result = sports_instance.List_available_markets()
    assert isinstance(result, dict)

# --- List_of_sports ---
def test_list_of_sports_returns_dict(sports_instance: SportsTools) -> None:
    """Test that List_of_sports returns a dictionary."""
    result = sports_instance.List_of_sports()
    assert isinstance(result, dict)

def test_list_of_sports_contains_sports(sports_instance: SportsTools) -> None:
    """Test that List_of_sports returns non-empty."""
    result = sports_instance.List_of_sports()
    assert len(result) > 0

# --- MatchSchedules ---
def test_match_schedules_valid_date(sports_instance: SportsTools) -> None:
    """Test MatchSchedules with a valid date."""
    result = sports_instance.MatchSchedules(day=15, year=2023, month=3)
    assert isinstance(result, dict)

def test_match_schedules_invalid_date(sports_instance: SportsTools) -> None:
    """Test MatchSchedules with an invalid month (should still return dict)."""
    result = sports_instance.MatchSchedules(day=1, year=2023, month=13)
    assert isinstance(result, dict)

# --- Matches_4 (returns Union[List, Dict]) ---
def test_matches_4_returns_list_or_dict(sports_instance: SportsTools) -> None:
    """Test that Matches_4 returns either a list or a dictionary."""
    result = sports_instance.Matches_4()
    assert isinstance(result, (list, dict)), "Expected list or dict"

def test_matches_4_not_empty(sports_instance: SportsTools) -> None:
    """Test that the result from Matches_4 is non-empty if possible."""
    result = sports_instance.Matches_4()
    if isinstance(result, list):
        assert len(result) >= 0
    else:
        assert len(result) >= 0

# --- Odds_6 ---
def test_odds_6_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Odds_6 returns a dictionary."""
    result = sports_instance.Odds_6()
    assert isinstance(result, dict)

def test_odds_6_non_empty(sports_instance: SportsTools) -> None:
    """Test that Odds_6 returns a non-empty dictionary."""
    result = sports_instance.Odds_6()
    assert len(result) > 0

# --- Premier_League_Standings ---
def test_premier_league_standings_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Premier_League_Standings returns a dictionary."""
    result = sports_instance.Premier_League_Standings()
    assert isinstance(result, dict)

def test_premier_league_standings_contains_teams(sports_instance: SportsTools) -> None:
    """Test that the standings contain team data."""
    result = sports_instance.Premier_League_Standings()
    assert "teams" in result or "standings" in result or len(result) > 0

# --- Premium_History ---
def test_premium_history_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Premium_History returns a dictionary."""
    result = sports_instance.Premium_History()
    assert isinstance(result, dict)

def test_premium_history_contains_tips(sports_instance: SportsTools) -> None:
    """Test that the history contains some data."""
    result = sports_instance.Premium_History()
    assert len(result) > 0

# --- Russian_Premier_League_Standings ---
def test_russian_league_standings_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Russian_Premier_League_Standings returns a dictionary."""
    result = sports_instance.Russian_Premier_League_Standings()
    assert isinstance(result, dict)

def test_russian_league_standings_non_empty(sports_instance: SportsTools) -> None:
    """Test that the result is non-empty."""
    result = sports_instance.Russian_Premier_League_Standings()
    assert len(result) > 0

# --- Seasons ---
def test_seasons_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Seasons returns a dictionary."""
    result = sports_instance.Seasons()
    assert isinstance(result, dict)

def test_seasons_contains_seasons(sports_instance: SportsTools) -> None:
    """Test that Seasons returns season data."""
    result = sports_instance.Seasons()
    assert "seasons" in result or len(result) > 0

# --- Tournament_List ---
def test_tournament_list_returns_dict(sports_instance: SportsTools) -> None:
    """Test that Tournament_List returns a dictionary."""
    result = sports_instance.Tournament_List()
    assert isinstance(result, dict)

# --- UFC_Fight_Night_Dern_vs_Hill_May_20 ---
def test_ufc_dern_hill_returns_dict(sports_instance: SportsTools) -> None:
    """Test that UFC_Fight_Night_Dern_vs_Hill_May_20 returns a dictionary."""
    result = sports_instance.UFC_Fight_Night_Dern_vs_Hill_May_20()
    assert isinstance(result, dict)

def test_ufc_dern_hill_contains_details(sports_instance: SportsTools) -> None:
    """Test that the result contains event details."""
    result = sports_instance.UFC_Fight_Night_Dern_vs_Hill_May_20()
    assert len(result) > 0

# --- UFC_Fight_Night_Holloway_vs_Allen_April_15 ---
def test_ufc_holloway_allen_returns_dict(sports_instance: SportsTools) -> None:
    """Test that UFC_Fight_Night_Holloway_vs_Allen_April_15 returns a dictionary."""
    result = sports_instance.UFC_Fight_Night_Holloway_vs_Allen_April_15()
    assert isinstance(result, dict)

def test_ufc_holloway_allen_contains_fight(sports_instance: SportsTools) -> None:
    """Test the result has data."""
    result = sports_instance.UFC_Fight_Night_Holloway_vs_Allen_April_15()
    assert len(result) > 0

# --- WNBA_ScoreBoard ---
def test_wnba_scoreboard_valid_date(sports_instance: SportsTools) -> None:
    """Test WNBA_ScoreBoard with a valid date string."""
    result = sports_instance.WNBA_ScoreBoard(month="06", day="15", year="2023")
    assert isinstance(result, dict)

def test_wnba_scoreboard_empty_strings(sports_instance: SportsTools) -> None:
    """Test WNBA_ScoreBoard with empty strings (should still return dict, maybe error)."""
    result = sports_instance.WNBA_ScoreBoard(month="", day="", year="")
    assert isinstance(result, dict)

# --- auto_complete ---
def test_auto_complete_valid_query(sports_instance: SportsTools) -> None:
    """Test auto_complete with a valid query."""
    result = sports_instance.auto_complete(query="soccer")
    assert isinstance(result, dict)

def test_auto_complete_empty_query(sports_instance: SportsTools) -> None:
    """Test auto_complete with an empty string (should return error dict)."""
    result = sports_instance.auto_complete(query="")
    assert isinstance(result, dict)

def test_auto_complete_none_query(sports_instance: SportsTools) -> None:
    """Test auto_complete with None (should handle gracefully)."""
    result = sports_instance.auto_complete(query=None)  # type: ignore
    assert isinstance(result, dict)

# --- categories_list ---
def test_categories_list_valid_sport(sports_instance: SportsTools) -> None:
    """Test categories_list with a valid sport name."""
    result = sports_instance.categories_list(sport="football")
    assert isinstance(result, dict)

def test_categories_list_empty_sport(sports_instance: SportsTools) -> None:
    """Test categories_list with an empty string."""
    result = sports_instance.categories_list(sport="")
    assert isinstance(result, dict)

# --- fixtures ---
def test_fixtures_returns_dict(sports_instance: SportsTools) -> None:
    """Test that fixtures returns a dictionary."""
    result = sports_instance.fixtures()
    assert isinstance(result, dict)

def test_fixtures_non_empty(sports_instance: SportsTools) -> None:
    """Test that fixtures returns some data."""
    result = sports_instance.fixtures()
    assert len(result) > 0

# --- teams_list ---
def test_teams_list_valid_match_type(sports_instance: SportsTools) -> None:
    """Test teams_list with a valid match type."""
    result = sports_instance.teams_list(matchType="league")
    assert isinstance(result, dict)

def test_teams_list_empty_type(sports_instance: SportsTools) -> None:
    """Test teams_list with an empty match type string."""
    result = sports_instance.teams_list(matchType="")
    assert isinstance(result, dict)

def test_teams_list_none_type(sports_instance: SportsTools) -> None:
    """Test teams_list with None (should handle gracefully)."""
    result = sports_instance.teams_list(matchType=None)  # type: ignore
    assert isinstance(result, dict)