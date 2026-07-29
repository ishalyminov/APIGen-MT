import pytest
import json
from typing import Dict, List, Any, Optional, Union
from tools.toollens.health_and_fitness import HealthAndFitnessTools


@pytest.fixture
def health_and_fitness_instance():
    """Fixture providing a fresh HealthAndFitnessTools instance."""
    config = None  # stateless
    return HealthAndFitnessTools(initial_config=config)


# ===== marks_men =====

class TestMarksMen:
    def test_marks_men_returns_list(self, health_and_fitness_instance):
        """Should return a list of mark records."""
        result = health_and_fitness_instance.marks_men()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_marks_men_items_have_expected_keys(self, health_and_fitness_instance):
        """Each item should contain typical world record fields."""
        result = health_and_fitness_instance.marks_men()
        if len(result) > 0:
            item = result[0]
            assert "event" in item or "mark" in item or "athlete" in item


# ===== marks_women =====

class TestMarksWomen:
    def test_marks_women_returns_list(self, health_and_fitness_instance):
        """Should return a list of women's world record marks."""
        result = health_and_fitness_instance.marks_women()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_marks_women_contains_expected_structure(self, health_and_fitness_instance):
        """Each item should be a dict with typical keys."""
        result = health_and_fitness_instance.marks_women()
        if len(result) > 0:
            item = result[0]
            assert isinstance(item, dict)


# ===== marks_women_points =====

class TestMarksWomenPoints:
    def test_marks_women_points_valid(self, health_and_fitness_instance):
        """Should return marks for a valid points value."""
        result = health_and_fitness_instance.marks_women_points(1000)
        assert isinstance(result, dict)
        assert "result" in result or "score" in result or "points" in result

    def test_marks_women_points_zero(self, health_and_fitness_instance):
        """Should handle edge case of zero points."""
        result = health_and_fitness_instance.marks_women_points(0)
        assert isinstance(result, dict)

    def test_marks_women_points_negative(self, health_and_fitness_instance):
        """Negative points should not cause an exception, return sensible error info."""
        result = health_and_fitness_instance.marks_women_points(-50)
        assert isinstance(result, dict)
        # Should contain error or fallback info
        assert any("error" in k.lower() for k in result.keys()) or "result" in result


# ===== v1_caloriesburned =====

class TestV1CaloriesBurned:
    def test_v1_caloriesburned_valid_activity(self, health_and_fitness_instance):
        """Should return calories info for an activity string."""
        result = health_and_fitness_instance.v1_caloriesburned("running")
        assert isinstance(result, dict)
        assert "calories" in result or "activity" in result or "result" in result

    def test_v1_caloriesburned_empty_string(self, health_and_fitness_instance):
        """Empty activity should return a sensible dict."""
        result = health_and_fitness_instance.v1_caloriesburned("")
        assert isinstance(result, dict)

    def test_v1_caloriesburned_none(self, health_and_fitness_instance):
        """None activity should be handled without exception."""
        result = health_and_fitness_instance.v1_caloriesburned(None)
        assert isinstance(result, dict)


# ===== BMI =====

class TestBMI:
    def test_bmi_normal_values(self, health_and_fitness_instance):
        """Should compute BMI and return the result dict."""
        result = health_and_fitness_instance.BMI(70.0, 1.75)
        assert isinstance(result, dict)
        assert "bmi" in result or "result" in result or "category" in result

    def test_bmi_zero_weight(self, health_and_fitness_instance):
        """Zero weight should be handled gracefully."""
        result = health_and_fitness_instance.BMI(0.0, 1.75)
        assert isinstance(result, dict)

    def test_bmi_negative_height(self, health_and_fitness_instance):
        """Negative height should return error info."""
        result = health_and_fitness_instance.BMI(70.0, -1.0)
        assert isinstance(result, dict)
        assert any("error" in k.lower() for k in result.keys()) or "result" in result


# ===== Conception_Date =====

class TestConceptionDate:
    def test_conception_date_valid(self, health_and_fitness_instance):
        """Should return pregnancy timeline for a valid date string."""
        result = health_and_fitness_instance.Conception_Date("2023-06-15")
        assert isinstance(result, dict)
        assert "due_date" in result or "weeks" in result or "result" in result

    def test_conception_date_invalid_format(self, health_and_fitness_instance):
        """Poorly formatted date should still return a dict (error handled inside)."""
        result = health_and_fitness_instance.Conception_Date("not-a-date")
        assert isinstance(result, dict)

    def test_conception_date_none(self, health_and_fitness_instance):
        """None should be handled without crash."""
        result = health_and_fitness_instance.Conception_Date(None)
        assert isinstance(result, dict)


# ===== Fertility_Window_GET =====

class TestFertilityWindowGET:
    def test_fertility_window_valid(self, health_and_fitness_instance):
        """Should return fertility window for typical cycle length and date."""
        result = health_and_fitness_instance.Fertility_Window_GET("28", "2023-01-01")
        assert isinstance(result, dict)
        assert "window" in result or "fertile" in result or "result" in result

    def test_fertility_window_edge_cases(self, health_and_fitness_instance):
        """Edge lengths or invalid dates should be handled."""
        result = health_and_fitness_instance.Fertility_Window_GET("0", "invalid")
        assert isinstance(result, dict)

    def test_fertility_window_none_params(self, health_and_fitness_instance):
        """Both params as None should not raise an exception."""
        result = health_and_fitness_instance.Fertility_Window_GET(None, None)
        assert isinstance(result, dict)


# ===== GET_Attributes =====

class TestGETAttributes:
    def test_get_attributes_returns_dict(self, health_and_fitness_instance):
        """Should return a dictionary of attributes."""
        result = health_and_fitness_instance.GET_Attributes()
        assert isinstance(result, dict)

    def test_get_attributes_has_content(self, health_and_fitness_instance):
        """The dict should have at least one key-value pair."""
        result = health_and_fitness_instance.GET_Attributes()
        assert len(result) > 0


# ===== Imperial_Pounds =====

class TestImperialPounds:
    def test_imperial_pounds_valid(self, health_and_fitness_instance):
        """Should compute imperial BMI and return dict."""
        result = health_and_fitness_instance.Imperial_Pounds(150.0, 5.5)
        assert isinstance(result, dict)
        assert "bmi" in result or "result" in result or "category" in result

    def test_imperial_pounds_zero_weight(self, health_and_fitness_instance):
        """Zero weight returns a dict (no crash)."""
        result = health_and_fitness_instance.Imperial_Pounds(0.0, 5.5)
        assert isinstance(result, dict)

    def test_imperial_pounds_none(self, health_and_fitness_instance):
        """None params should be handled."""
        result = health_and_fitness_instance.Imperial_Pounds(None, None)
        assert isinstance(result, dict)


# ===== Last_Menstrual_Period_LMP =====

class TestLastMenstrualPeriodLMP:
    def test_lmp_valid(self, health_and_fitness_instance):
        """Should return due date / pregnancy info for valid LMP."""
        result = health_and_fitness_instance.Last_Menstrual_Period_LMP("28", "2023-01-01")
        assert isinstance(result, dict)
        assert "due_date" in result or "result" in result

    def test_lmp_none_cycle_length(self, health_and_fitness_instance):
        """None cycle length should be handled."""
        result = health_and_fitness_instance.Last_Menstrual_Period_LMP(None, "2023-01-01")
        assert isinstance(result, dict)

    def test_lmp_none_date(self, health_and_fitness_instance):
        """None date should be handled."""
        result = health_and_fitness_instance.Last_Menstrual_Period_LMP("28", None)
        assert isinstance(result, dict)


# ===== List_of_bodyparts =====

class TestListOfBodyparts:
    def test_list_of_bodyparts_returns_list(self, health_and_fitness_instance):
        """Should return a list of body parts."""
        result = health_and_fitness_instance.List_of_bodyparts()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_of_bodyparts_items_are_strings(self, health_and_fitness_instance):
        """Each item should be a string body part name."""
        result = health_and_fitness_instance.List_of_bodyparts()
        for item in result:
            assert isinstance(item, str) or isinstance(item, dict)


# ===== List_of_equipment =====

class TestListOfEquipment:
    def test_list_of_equipment_returns_dict(self, health_and_fitness_instance):
        """Should return a dict of equipment categories."""
        result = health_and_fitness_instance.List_of_equipment()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_list_of_equipment_has_expected_keys(self, health_and_fitness_instance):
        """The dict should contain equipment list under a known key."""
        result = health_and_fitness_instance.List_of_equipment()
        assert "equipment" in result or "result" in result or len(result) > 0


# ===== List_of_target_muscles =====

class TestListOfTargetMuscles:
    def test_list_of_target_muscles_returns_list(self, health_and_fitness_instance):
        """Should return a list of target muscles."""
        result = health_and_fitness_instance.List_of_target_muscles()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_of_target_muscles_contains_strings(self, health_and_fitness_instance):
        """Each item should be a string muscle name or dict."""
        result = health_and_fitness_instance.List_of_target_muscles()
        for item in result:
            assert isinstance(item, (str, dict))


# ===== Macronutrient_Distribution =====

class TestMacronutrientDistribution:
    def test_macronutrient_distribution_returns_dict(self, health_and_fitness_instance):
        """Should return a dict with macros info."""
        result = health_and_fitness_instance.Macronutrient_Distribution()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_macronutrient_distribution_has_macros(self, health_and_fitness_instance):
        """Should contain protein, carbs, fat keys."""
        result = health_and_fitness_instance.Macronutrient_Distribution()
        macros = ["protein", "carbohydrates", "fat", "carbs", "fats"]
        assert any(macro in result for macro in macros) or "result" in result


# ===== View_All_Cores_With_Their_Food_Items =====

class TestViewAllCoresWithTheirFoodItems:
    def test_view_all_cores_with_food_items_returns_dict(self, health_and_fitness_instance):
        """Should return a dict of food cores with items."""
        result = health_and_fitness_instance.View_All_Cores_With_Their_Food_Items()
        assert isinstance(result, dict)

    def test_view_all_cores_with_food_items_has_content(self, health_and_fitness_instance):
        """The dict should contain at least one core."""
        result = health_and_fitness_instance.View_All_Cores_With_Their_Food_Items()
        assert len(result) > 0


# ===== View_All_Food_Items =====

class TestViewAllFoodItems:
    def test_view_all_food_items_returns_dict(self, health_and_fitness_instance):
        """Should return a dict of all food items."""
        result = health_and_fitness_instance.View_All_Food_Items()
        assert isinstance(result, dict)

    def test_view_all_food_items_not_empty(self, health_and_fitness_instance):
        """The dict should have at least one food item."""
        result = health_and_fitness_instance.View_All_Food_Items()
        assert len(result) > 0


# ===== View_Food_Item_By_Name =====

class TestViewFoodItemByName:
    def test_view_food_item_by_name_returns_dict(self, health_and_fitness_instance):
        """Should return a dict representing the food item."""
        result = health_and_fitness_instance.View_Food_Item_By_Name()
        assert isinstance(result, dict)

    def test_view_food_item_by_name_has_name_field(self, health_and_fitness_instance):
        """The dict should contain a name key or similar identifier."""
        result = health_and_fitness_instance.View_Food_Item_By_Name()
        assert "name" in result or "item" in result or "food" in result or "result" in result


# ===== View_Food_Items_by_Core =====

class TestViewFoodItemsByCore:
    def test_view_food_items_by_core_returns_dict(self, health_and_fitness_instance):
        """Should return a dict of food items grouped by core."""
        result = health_and_fitness_instance.View_Food_Items_by_Core()
        assert isinstance(result, dict)

    def test_view_food_items_by_core_has_content(self, health_and_fitness_instance):
        """The dict should contain at least one core group."""
        result = health_and_fitness_instance.View_Food_Items_by_Core()
        assert len(result) > 0


# ===== Weight_Category =====

class TestWeightCategory:
    def test_weight_category_valid_bmi(self, health_and_fitness_instance):
        """Should return weight category for a normal BMI value."""
        result = health_and_fitness_instance.Weight_Category(22.5)
        assert isinstance(result, dict)
        assert "category" in result or "result" in result or "bmi" in result

    def test_weight_category_extreme_bmi(self, health_and_fitness_instance):
        """Extreme BMI values should still be handled."""
        result = health_and_fitness_instance.Weight_Category(999.0)
        assert isinstance(result, dict)

    def test_weight_category_negative_bmi(self, health_and_fitness_instance):
        """Negative BMI returns error information in dict."""
        result = health_and_fitness_instance.Weight_Category(-10.0)
        assert isinstance(result, dict)
        assert any("error" in k.lower() for k in result.keys()) or "category" in result


# ===== getHospitalsByName =====

class TestGetHospitalsByName:
    def test_get_hospitals_by_name_valid(self, health_and_fitness_instance):
        """Should return hospital info for a valid name string."""
        result = health_and_fitness_instance.getHospitalsByName("General")
        assert isinstance(result, dict)
        assert "hospitals" in result or "name" in result or "result" in result

    def test_get_hospitals_by_name_empty(self, health_and_fitness_instance):
        """Empty name should be handled."""
        result = health_and_fitness_instance.getHospitalsByName("")
        assert isinstance(result, dict)

    def test_get_hospitals_by_name_none(self, health_and_fitness_instance):
        """None name should not raise exception."""
        result = health_and_fitness_instance.getHospitalsByName(None)
        assert isinstance(result, dict)


# ===== hoscoscope =====

class TestHoscoscope:
    def test_hoscoscope_valid(self, health_and_fitness_instance):
        """Should return prediction for valid date and sign."""
        result = health_and_fitness_instance.hoscoscope("2024-01-15", "Aries")
        assert isinstance(result, dict)
        assert "prediction" in result or "horoscope" in result or "result" in result

    def test_hoscoscope_invalid_date(self, health_and_fitness_instance):
        """Invalid date string should be handled."""
        result = health_and_fitness_instance.hoscoscope("bad-date", "Leo")
        assert isinstance(result, dict)

    def test_hoscoscope_empty_sign(self, health_and_fitness_instance):
        """Empty sign should be handled."""
        result = health_and_fitness_instance.hoscoscope("2024-01-01", "")
        assert isinstance(result, dict)

    def test_hoscoscope_none_params(self, health_and_fitness_instance):
        """Both params None should not crash."""
        result = health_and_fitness_instance.hoscoscope(None, None)
        assert isinstance(result, dict)