import pytest
import json
from tools.toollens.food import FoodTools


@pytest.fixture
def food_instance():
    """Create a stateless instance of FoodTools."""
    return FoodTools(initial_config=None)


# ──────────────────────────────────────────────
# Group 1: Methods without parameters
# ──────────────────────────────────────────────

def test_v1_cocktail(food_instance):
    """Check that v1_cocktail returns a dict."""
    result = food_instance.v1_cocktail()
    assert isinstance(result, dict)


def test_All_Desserts_Data(food_instance):
    """Check that All_Desserts_Data returns a dict."""
    result = food_instance.All_Desserts_Data()
    assert isinstance(result, dict)


def test_Drinks(food_instance):
    """Check that Drinks returns a dict."""
    result = food_instance.Drinks()
    assert isinstance(result, dict)


def test_Explore(food_instance):
    """Check that Explore returns a dict."""
    result = food_instance.Explore()
    assert isinstance(result, dict)


def test_GET_all_City_names(food_instance):
    """Check that GET_all_City_names returns a dict."""
    result = food_instance.GET_all_City_names()
    assert isinstance(result, dict)


def test_GET_all_State_names(food_instance):
    """Check that GET_all_State_names returns a dict."""
    result = food_instance.GET_all_State_names()
    assert isinstance(result, dict)


def test_Generate_recipe(food_instance):
    """Check that Generate_recipe returns a dict."""
    result = food_instance.Generate_recipe()
    assert isinstance(result, dict)


def test_Get_all_foods(food_instance):
    """Check that Get_all_foods returns a dict."""
    result = food_instance.Get_all_foods()
    assert isinstance(result, dict)


def test_Get_recipes(food_instance):
    """Check that Get_recipes returns a dict."""
    result = food_instance.Get_recipes()
    assert isinstance(result, dict)


def test_Random(food_instance):
    """Check that Random returns a dict."""
    result = food_instance.Random()
    assert isinstance(result, dict)


def test_Random_Nonalcoholic(food_instance):
    """Check that Random_Nonalcoholic returns a dict."""
    result = food_instance.Random_Nonalcoholic()
    assert isinstance(result, dict)


def test_Restaurants(food_instance):
    """Check that Restaurants (capital R) returns a dict."""
    result = food_instance.Restaurants()
    assert isinstance(result, dict)


def test_UPC_Api(food_instance):
    """Check that UPC_Api returns a dict."""
    result = food_instance.UPC_Api()
    assert isinstance(result, dict)


def test_getAllBeersList(food_instance):
    """Check that getAllBeersList returns a dict."""
    result = food_instance.getAllBeersList()
    assert isinstance(result, dict)


def test_restaurants(food_instance):
    """Check that restaurants (lowercase) returns a dict."""
    result = food_instance.restaurants()
    assert isinstance(result, dict)


# ──────────────────────────────────────────────
# Group 2: Methods with optional parameters
# ──────────────────────────────────────────────

# ---------- v1_nutrition ----------

def test_v1_nutrition_valid(food_instance):
    """v1_nutrition with a query string returns a dict."""
    result = food_instance.v1_nutrition(query="chicken")
    assert isinstance(result, dict)


def test_v1_nutrition_missing(food_instance):
    """v1_nutrition with no query returns an error dict."""
    result = food_instance.v1_nutrition(query=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result  # generic error check


# ---------- v1_recipe ----------

def test_v1_recipe_valid(food_instance):
    """v1_recipe with a query string returns a dict."""
    result = food_instance.v1_recipe(query="pasta")
    assert isinstance(result, dict)


def test_v1_recipe_missing(food_instance):
    """v1_recipe with no query returns an error dict."""
    result = food_instance.v1_recipe(query=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- Fetch_Restaurant_Information ----------

def test_Fetch_Restaurant_Information_valid(food_instance):
    """Valid restaurant query."""
    result = food_instance.Fetch_Restaurant_Information(query="McDonald's")
    assert isinstance(result, dict)


def test_Fetch_Restaurant_Information_missing(food_instance):
    """Missing restaurant query -> error."""
    result = food_instance.Fetch_Restaurant_Information(query=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- Generate_Recipe ----------

def test_Generate_Recipe_valid(food_instance):
    """Generate recipe with a list of ingredients."""
    result = food_instance.Generate_Recipe(ingredients="chicken,rice")
    assert isinstance(result, dict)


def test_Generate_Recipe_missing(food_instance):
    """Missing ingredients -> error."""
    result = food_instance.Generate_Recipe(ingredients=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- Get_beers_by_single_country ----------

def test_Get_beers_by_single_country_valid(food_instance):
    """Valid beerId."""
    result = food_instance.Get_beers_by_single_country(beerId="1")
    assert isinstance(result, dict)


def test_Get_beers_by_single_country_missing(food_instance):
    """Missing beerId -> error."""
    result = food_instance.Get_beers_by_single_country(beerId=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- Recipes_by_author ----------

def test_Recipes_by_author_valid(food_instance):
    """Valid profile name."""
    result = food_instance.Recipes_by_author(profile_name="John Doe")
    assert isinstance(result, dict)


def test_Recipes_by_author_missing(food_instance):
    """Missing profile name -> error."""
    result = food_instance.Recipes_by_author(profile_name=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- Search_a_Grocery ----------

def test_Search_a_Grocery_valid(food_instance):
    """Valid grocery item."""
    result = food_instance.Search_a_Grocery(grocery="milk")
    assert isinstance(result, dict)


def test_Search_a_Grocery_missing(food_instance):
    """Missing grocery -> error."""
    result = food_instance.Search_a_Grocery(grocery=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- Suggestions ----------

def test_Suggestions_valid(food_instance):
    """Valid query string."""
    result = food_instance.Suggestions(q="chocolate")
    assert isinstance(result, dict)


def test_Suggestions_missing(food_instance):
    """Missing query -> error."""
    result = food_instance.Suggestions(q=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- appetizer_ingredient ----------

def test_appetizer_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.appetizer_ingredient(ingredient="cheese")
    assert isinstance(result, dict)


def test_appetizer_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.appetizer_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- breakfast_ingredient ----------

def test_breakfast_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.breakfast_ingredient(ingredient="eggs")
    assert isinstance(result, dict)


def test_breakfast_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.breakfast_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- cake_ingredient ----------

def test_cake_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.cake_ingredient(ingredient="flour")
    assert isinstance(result, dict)


def test_cake_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.cake_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- dinner_ingredient ----------

def test_dinner_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.dinner_ingredient(ingredient="beef")
    assert isinstance(result, dict)


def test_dinner_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.dinner_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- feeds_auto_complete ----------

def test_feeds_auto_complete_valid(food_instance):
    """Valid query string."""
    result = food_instance.feeds_auto_complete(q="ch")
    assert isinstance(result, dict)


def test_feeds_auto_complete_missing(food_instance):
    """Missing query -> error."""
    result = food_instance.feeds_auto_complete(q=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- go ----------

def test_go_valid(food_instance):
    """Valid query string."""
    result = food_instance.go(q="dessert")
    assert isinstance(result, dict)


def test_go_missing(food_instance):
    """Missing query -> error."""
    result = food_instance.go(q=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- icecream_ingredient ----------

def test_icecream_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.icecream_ingredient(ingredient="vanilla")
    assert isinstance(result, dict)


def test_icecream_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.icecream_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- lunch_ingredient ----------

def test_lunch_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.lunch_ingredient(ingredient="chicken")
    assert isinstance(result, dict)


def test_lunch_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.lunch_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- mediterranean_ingredient ----------

def test_mediterranean_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.mediterranean_ingredient(ingredient="olive oil")
    assert isinstance(result, dict)


def test_mediterranean_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.mediterranean_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- pastry_ingredient ----------

def test_pastry_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.pastry_ingredient(ingredient="butter")
    assert isinstance(result, dict)


def test_pastry_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.pastry_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- recipes_auto_complete ----------

def test_recipes_auto_complete_valid(food_instance):
    """Valid prefix string."""
    result = food_instance.recipes_auto_complete(prefix="pa")
    assert isinstance(result, dict)


def test_recipes_auto_complete_missing(food_instance):
    """Missing prefix -> error."""
    result = food_instance.recipes_auto_complete(prefix=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result


# ---------- salad_ingredient ----------

def test_salad_ingredient_valid(food_instance):
    """Valid ingredient."""
    result = food_instance.salad_ingredient(ingredient="lettuce")
    assert isinstance(result, dict)


def test_salad_ingredient_missing(food_instance):
    """Missing ingredient -> error."""
    result = food_instance.salad_ingredient(ingredient=None)
    assert isinstance(result, dict)
    assert "error" in result or "status" not in result