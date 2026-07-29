"""
Test file for FoodTools sequential API calls.
Tests correct ordered sequences and problematic sequences.
"""

import pytest
import json
from tools.toollens.food import FoodTools


@pytest.fixture
def food_tools():
    """Fixture providing a fresh FoodTools instance."""
    return FoodTools()


class TestFoodToolsSequentialCorrect:
    """Correct ordered sequences of FoodTools methods."""

    def test_states_cities_restaurants(self, food_tools):
        """Get all states, then all cities, then fetch restaurant info for a city."""
        # Step 1: Get all states
        states_resp = food_tools.GET_all_State_names()
        assert isinstance(states_resp, dict), "Expected a dict from GET_all_State_names"

        # Step 2: Get all cities
        cities_resp = food_tools.GET_all_City_names()
        assert isinstance(cities_resp, dict), "Expected a dict from GET_all_City_names"

        # Step 3: Fetch restaurant information for a city
        restaurant_resp = food_tools.Fetch_Restaurant_Information(query="Seoul")
        assert isinstance(restaurant_resp, dict), "Expected a dict from Fetch_Restaurant_Information"

    def test_random_cocktail_then_nonalcoholic(self, food_tools):
        """Get a random cocktail, then a random nonalcoholic cocktail."""
        # Step 1: Random cocktail
        random_resp = food_tools.Random()
        assert isinstance(random_resp, dict), "Expected a dict from Random"

        # Step 2: Random nonalcoholic cocktail
        nonalc_resp = food_tools.Random_Nonalcoholic()
        assert isinstance(nonalc_resp, dict), "Expected a dict from Random_Nonalcoholic"

    def test_beers_list_then_country(self, food_tools):
        """Get all beers list, then get beers by a single country."""
        # Step 1: Get all beers list
        beers_list_resp = food_tools.getAllBeersList()
        assert isinstance(beers_list_resp, dict), "Expected a dict from getAllBeersList"

        # Step 2: Get beers for a country
        country_beer_resp = food_tools.Get_beers_by_single_country(beerId="USA")
        assert isinstance(country_beer_resp, dict), "Expected a dict from Get_beers_by_single_country"

    def test_recipe_search_and_generate(self, food_tools):
        """Search recipes, then generate a recipe with ingredients."""
        # Step 1: Search recipes
        recipe_resp = food_tools.v1_recipe(query="chicken")
        assert isinstance(recipe_resp, dict), "Expected a dict from v1_recipe"

        # Step 2: Generate recipe
        gen_recipe_resp = food_tools.Generate_Recipe(ingredients="chicken, rice, beans")
        assert isinstance(gen_recipe_resp, dict), "Expected a dict from Generate_Recipe"

    def test_auto_complete_and_suggestions(self, food_tools):
        """Get auto-complete suggestions, then get suggestions."""
        # Step 1: Auto-complete
        auto_resp = food_tools.feeds_auto_complete(q="chicken")
        assert isinstance(auto_resp, dict), "Expected a dict from feeds_auto_complete"

        # Step 2: Suggestions
        sugg_resp = food_tools.Suggestions(q="chicken")
        assert isinstance(sugg_resp, dict), "Expected a dict from Suggestions"


class TestFoodToolsSequentialProblematic:
    """Problematic sequences for FoodTools methods."""

    def test_nonexistent_restaurant_then_random(self, food_tools):
        """Fetch restaurant with empty query, then call random (should not crash)."""
        # Step 1: Fetch restaurant with empty query (nonexistent)
        bad_restaurant = food_tools.Fetch_Restaurant_Information(query="")
        assert isinstance(bad_restaurant, dict), "Expected a dict from Fetch_Restaurant_Information even with empty query"

        # Step 2: Random cocktail (should succeed)
        random_resp = food_tools.Random()
        assert isinstance(random_resp, dict), "Expected a dict from Random after a problematic call"

    def test_missing_params_then_valid_call(self, food_tools):
        """Call several methods with missing parameters, then a valid call."""
        # Step 1: v1_nutrition with no param
        nutrition_resp = food_tools.v1_nutrition()
        assert isinstance(nutrition_resp, dict), "Expected a dict from v1_nutrition with no param"

        # Step 2: v1_recipe with no param
        recipe_resp = food_tools.v1_recipe()
        assert isinstance(recipe_resp, dict), "Expected a dict from v1_recipe with no param"

        # Step 3: Valid call after errors
        valid_recipe = food_tools.v1_recipe(query="pasta")
        assert isinstance(valid_recipe, dict), "Expected a dict from v1_recipe with valid query"

    def test_empty_generate_and_autocomplete_then_suggestions(self, food_tools):
        """Call Generate_Recipe and feeds_auto_complete with empty strings, then valid suggestion."""
        # Step 1: Generate recipe with empty ingredients
        gen_resp = food_tools.Generate_Recipe(ingredients="")
        assert isinstance(gen_resp, dict), "Expected a dict from Generate_Recipe with empty ingredients"

        # Step 2: Auto-complete with empty query
        auto_resp = food_tools.feeds_auto_complete(q="")
        assert isinstance(auto_resp, dict), "Expected a dict from feeds_auto_complete with empty query"

        # Step 3: Valid suggestion
        sugg_resp = food_tools.Suggestions(q="pizza")
        assert isinstance(sugg_resp, dict), "Expected a dict from Suggestions after errors"

    def test_random_nonalcoholic_then_invalid_api_calls(self, food_tools):
        """Get random nonalcoholic then call UPC_Api and v1_cocktail with no params."""
        # Step 1: Random nonalcoholic
        nonalc = food_tools.Random_Nonalcoholic()
        assert isinstance(nonalc, dict), "Expected a dict from Random_Nonalcoholic"

        # Step 2: UPC_Api with no param
        upc = food_tools.UPC_Api()
        assert isinstance(upc, dict), "Expected a dict from UPC_Api with no param"

        # Step 3: v1_cocktail with no param (needs name or ingredient)
        cocktail = food_tools.v1_cocktail()
        assert isinstance(cocktail, dict), "Expected a dict from v1_cocktail with no param"

    def test_drinks_and_restaurants_in_any_order(self, food_tools):
        """Call Drinks then Restaurants; both should work regardless of order."""
        # Step 1: Get drinks
        drinks = food_tools.Drinks()
        assert isinstance(drinks, dict), "Expected a dict from Drinks"

        # Step 2: Get restaurants (different method from Restaurants)
        restaurants_list = food_tools.restaurants()
        assert isinstance(restaurants_list, dict), "Expected a dict from restaurants"

        # Both calls should succeed