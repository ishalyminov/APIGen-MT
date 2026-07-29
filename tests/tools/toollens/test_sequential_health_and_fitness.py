import pytest
import json
from tools.toollens.health_and_fitness import HealthAndFitnessTools


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def tool():
    """Return a fresh HealthAndFitnessTools instance with config deep‑copied."""
    config = json.loads(json.dumps(None))
    return HealthAndFitnessTools(initial_config=config)


# =============================================================================
# Correct sequential tests
# =============================================================================

class TestHealthAndFitnessSequentialCorrect:
    """Multi‑call sequences that represent typical user flows."""

    def test_bmi_then_weight_category(self, tool):
        """Compute BMI and then use it to retrieve the weight category."""
        bmi_result = tool.BMI(weight=80.0, height=1.75)
        assert isinstance(bmi_result, dict), "BMI should return a dict"
        # The returned dict is expected to contain at least 'bmi'
        bmi_value = bmi_result.get("bmi")
        assert bmi_value is not None, "BMI dict must contain 'bmi' key"

        category_result = tool.Weight_Category(bmi=bmi_value)
        assert isinstance(category_result, dict), "Weight_Category should return a dict"
        assert "category" in category_result or "weight_category" in category_result, (
            "Weight_Category dict should contain a category key"
        )

    def test_marks_men_and_women(self, tool):
        """Retrieve marks for men and then for women (two stateless list methods)."""
        marks_men_list = tool.marks_men()
        assert isinstance(marks_men_list, list), "marks_men() should return a list"

        marks_women_result = tool.marks_women()
        # marks_women may return a list or a dict depending on implementation
        assert marks_women_result is not None, "marks_women() should not return None"

    def test_conception_date_and_fertility_window(self, tool):
        """Two menstrual/fertility related calls in sequence."""
        cd_result = tool.Conception_Date(conception_date="2024-01-15")
        assert isinstance(cd_result, dict), "Conception_Date should return a dict"
        assert "gestational_age" in cd_result or "due_date" in cd_result, (
            "Conception_Date result missing expected keys"
        )

        fw_result = tool.Fertility_Window_GET(
            cycle_length="28",
            menstrual_date="2024-01-01"
        )
        assert isinstance(fw_result, dict), "Fertility_Window_GET should return a dict"
        assert "fertility_window" in fw_result or "start_date" in fw_result, (
            "Fertility_Window_GET result missing expected keys"
        )

    def test_list_body_parts_and_equipment_and_muscles(self, tool):
        """Explore available body parts, equipment, and target muscles."""
        body_parts = tool.List_of_bodyparts()
        assert isinstance(body_parts, list), "List_of_bodyparts should return a list"

        equipment = tool.List_of_equipment()
        assert isinstance(equipment, dict), "List_of_equipment should return a dict"

        target_muscles = tool.List_of_target_muscles()
        assert isinstance(target_muscles, list), "List_of_target_muscles should return a list"

    def test_calories_burned_then_imperial_pounds(self, tool):
        """Call two calculation methods in a row (no dependency but sensible)."""
        cal_result = tool.v1_caloriesburned(activity="running")
        assert isinstance(cal_result, dict), "v1_caloriesburned should return a dict"
        assert "calories" in cal_result or "energy" in cal_result, (
            "Calories burned result missing expected keys"
        )

        imperial_result = tool.Imperial_Pounds(weight=70.0, height=1.75)
        assert isinstance(imperial_result, dict), "Imperial_Pounds should return a dict"
        assert "bmi" in imperial_result or "imperial_bmi" in imperial_result, (
            "Imperial_Pounds result missing expected keys"
        )


# =============================================================================
# Problematic sequential tests
# =============================================================================

class TestHealthAndFitnessSequentialProblematic:
    """Sequences with invalid data, missing resources, or unnatural ordering."""

    def test_nonexistent_hospital_then_get_attributes(self, tool):
        """Look for a hospital that does not exist and then fetch attributes."""
        # Search for a hospital that is unlikely to exist
        hospital_result = tool.getHospitalsByName(name="NonExistentHospitalXYZ")
        assert isinstance(hospital_result, dict), (
            "getHospitalsByName should return a dict even when no match"
        )
        # The response may contain an error flag or an empty list
        # Ensure no exception was raised
        assert "error" not in hospital_result or hospital_result.get("error") is None, (
            "Missing hospital should not produce a real error, maybe an empty result"
        )

        # Subsequent call should succeed
        attrs_result = tool.GET_Attributes()
        assert isinstance(attrs_result, dict), "GET_Attributes should return a dict"
        assert len(attrs_result) > 0, "Attributes dict should not be empty"

    def test_invalid_bmi_then_weight_category(self, tool):
        """Call BMI with negative weight, then Weight_Category with that invalid BMI."""
        bmi_result = tool.BMI(weight=-10.0, height=1.75)
        assert isinstance(bmi_result, dict), "BMI should return a dict even with invalid input"
        # Expect an error indication
        bmi_value = bmi_result.get("bmi")
        # The implementation might still return a number (negative) or an error dict
        # If it returns an error dict, the 'bmi' key may be missing
        if bmi_value is not None:
            # If a value was returned, weight category should still be callable
            cat_result = tool.Weight_Category(bmi=bmi_value)
            assert isinstance(cat_result, dict), "Weight_Category should return a dict"
        else:
            # BMI returned an error; Weight_Category with an invalid value should still work
            cat_result = tool.Weight_Category(bmi=-10.0)
            assert isinstance(cat_result, dict), (
                "Weight_Category should not raise exception with negative BMI"
            )

    def test_wrong_order_fertility_methods(self, tool):
        """Call Fertility_Window_GET before the typical Conception_Date (unnatural order)."""
        # First call: fertility window (normally preceded by last period date)
        fw_result = tool.Fertility_Window_GET(
            cycle_length="28",
            menstrual_date="2024-01-01"
        )
        assert isinstance(fw_result, dict), "Fertility_Window_GET should return a dict"

        # Second call: conception date with invalid format (problematic)
        cd_result = tool.Conception_Date(conception_date="not-a-date")
        assert isinstance(cd_result, dict), "Conception_Date should return a dict"
        # Expect an error indicator
        assert "error" in cd_result or "status" in cd_result, (
            "Invalid date should produce an error indication"
        )

        # Third call: a completely different method to ensure tool still works
        list_equipment = tool.List_of_equipment()
        assert isinstance(list_equipment, dict), "Tool should still work after invalid call"

    def test_invalid_date_formats_in_menstrual_methods(self, tool):
        """Pass bad date strings to two menstrual methods, then call a safe method."""
        lmp_result = tool.Last_Menstrual_Period_LMP(
            cycle_length="28",
            last_period_date="bad-date"
        )
        assert isinstance(lmp_result, dict), (
            "Last_Menstrual_Period_LMP should return a dict even with bad date"
        )
        # Expect an error/missing key indication
        # (we just ensure no exception)

        cd_result = tool.Conception_Date(conception_date="2024/01/15")  # wrong format
        assert isinstance(cd_result, dict), "Conception_Date should return a dict"

        # Clean call afterwards
        macros = tool.Macronutrient_Distribution()
        assert isinstance(macros, dict), "Macronutrient_Distribution should succeed afterwards"

    def test_missing_or_invalid_params_then_normal_call(self, tool):
        """Call a method with invalid types (e.g., string for float) then a normal call."""
        # Attempt BMI with string values (may be handled gracefully or raise)
        # We pass strings to see if tool handles them – if it doesn't, we still
        # expect a dict (error) and the next call should not break.
        try:
            bmi_result = tool.BMI(weight="seventy", height="tall")
        except (TypeError, ValueError):
            # If the tool raises an exception, we catch it and consider it handled
            # The subsequent call must still work.
            pass
        else:
            # If it returned a dict, check for error
            if isinstance(bmi_result, dict):
                # May contain 'error' key
                pass

        # Normal call should work regardless
        cal_result = tool.v1_caloriesburned(activity="walking")
        assert isinstance(cal_result, dict), "v1_caloriesburned should work after invalid BMI call"