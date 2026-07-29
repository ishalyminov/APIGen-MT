"""Auto-generated HealthAndFitnessTools implementation."""

import json
import math
import re
from copy import deepcopy
from datetime import datetime, timedelta
from random import seed, choice, randint, uniform
from typing import List, Dict, Any, Optional, Tuple, Union


class HealthAndFitnessTools:
    """
    Health and Fitness tools providing BMI calculation, pregnancy tracking,
    athletic performance marks, exercise information, food data, and more.
    """

    METHOD_NAME_MAP = {
        '/marks/men/': 'marks_men',
        '/marks/women/': 'marks_women',
        '/marks/women/{points}': 'marks_women_points',
        '/v1/caloriesburned': 'v1_caloriesburned',
        'BMI': 'BMI',
        'Conception Date': 'Conception_Date',
        'Fertility Window - GET': 'Fertility_Window_GET',
        'GET Attributes': 'GET_Attributes',
        'Imperial [Pounds]': 'Imperial_Pounds',
        'Last Menstrual Period (LMP)': 'Last_Menstrual_Period_LMP',
        'List of bodyparts': 'List_of_bodyparts',
        'List of equipment': 'List_of_equipment',
        'List of target muscles': 'List_of_target_muscles',
        'Macronutrient Distribution': 'Macronutrient_Distribution',
        'View All Cores With Their Food Items': 'View_All_Cores_With_Their_Food_Items',
        'View All Food Items': 'View_All_Food_Items',
        'View Food Item By Name': 'View_Food_Item_By_Name',
        'View Food Items by Core': 'View_Food_Items_by_Core',
        'Weight Category': 'Weight_Category',
        'getHospitalsByName': 'getHospitalsByName',
        'hoscoscope': 'hoscoscope',
    }

    def __init__(self, initial_config: dict = None):
        # Initialize internal state, store config data safely
        self._config_data = {}
        if initial_config:
            self._config_data.update(initial_config)
        else:
            self._init_state()

    def _init_state(self):
        """Set up default empty state (minimal since most methods are stateless)."""
        self._config_data['call_count'] = 0
        self._config_data['_seed'] = 42  # for reproducibility in random outputs

    # ---------- Utility helper ----------
    def _next_call(self):
        """Increment call counter and return current value."""
        count = self._config_data.get('call_count', 0) + 1
        self._config_data['call_count'] = count
        seed(self._config_data.get('_seed', 42) + count)
        return count

    # ===== METHODS =====

    # /marks/men/
    def marks_men(self) -> list:
        """
        Retrieve all men's marks from the World Athletics Scoring Tables.

        Returns:
            list: A collection of men's athletic marks (scoring data).
        """
        self._next_call()
        return [
            {"event": "100m", "points": 1200, "performance": "9.58s", "category": "Sprints"},
            {"event": "Long Jump", "points": 1300, "performance": "8.90m", "category": "Jumps"},
            {"event": "Shot Put", "points": 1100, "performance": "22.50m", "category": "Throws"},
        ]

    # /marks/women/
    def marks_women(self):
        """
        Retrieve all women's marks from the World Athletics Scoring Tables.

        Returns:
            list: A collection of women's athletic marks (scoring data).
        """
        self._next_call()
        return [
            {"event": "100m", "points": 1100, "performance": "10.49s", "category": "Sprints"},
            {"event": "Heptathlon", "points": 1290, "performance": "7291 pts", "category": "Combined"},
            {"event": "High Jump", "points": 1200, "performance": "2.09m", "category": "Jumps"},
        ]

    # /marks/women/{points}
    def marks_women_points(self, points: float) -> dict:
        """
        Retrieve women's marks for a given point value (0-1400).

        Args:
            points: A number between 1 and 1400.

        Returns:
            dict: Object with keys 'points' and 'marks' containing event details.
        """
        self._next_call()
        # Validate points
        if not isinstance(points, (int, float)) or points < 1 or points > 1400:
            return {
                "points": 0,
                "marks": {
                    "event": "invalid",
                    "performance": "0",
                    "category": "error"
                }
            }
        int_points = int(points)
        seed(int_points)
        events = ["100m", "200m", "400m", "800m", "1500m", "5000m", "10000m",
                  "Marathon", "100m Hurdles", "400m Hurdles", "3000m Steeplechase",
                  "High Jump", "Long Jump", "Triple Jump", "Pole Vault",
                  "Shot Put", "Discus Throw", "Hammer Throw", "Javelin Throw",
                  "Heptathlon", "Decathlon"]
        event = choice(events)
        # Generate a performance string that depends on event type
        if "m" in event and "Hurdles" not in event and "Steeplechase" not in event and "Marathon" not in event:
            performance = f"{uniform(10.0, 60.0):.2f}s"
        elif "Hurdles" in event or "Steeplechase" in event:
            performance = f"{uniform(12.0, 65.0):.2f}s"
        elif "Jump" in event or "Vault" in event:
            performance = f"{uniform(1.0, 6.0):.2f}m"
        elif "Throw" in event or "Put" in event:
            performance = f"{uniform(10.0, 80.0):.2f}m"
        elif "Marathon" in event:
            performance = f"{uniform(120.0, 180.0):.2f} min"
        else:
            performance = f"{uniform(5000, 10000):d} pts"
        return {
            "points": int_points,
            "marks": {
                "event": event,
                "performance": performance,
                "category": choice(["Sprints", "Distance", "Jumps", "Throws", "Combined"])
            }
        }

    # /v1/caloriesburned
    def v1_caloriesburned(self, activity: str) -> dict:
        """
        API Ninjas Calories Burned endpoint.

        Args:
            activity: Name of activity (can be partial, e.g., 'ski').

        Returns:
            dict: Object with keys name, calories_per_hour, duration_minutes, total_calories.
        """
        self._next_call()
        # Normalize: use the keyword to generate sensible values
        activity_lower = activity.strip().lower()
        # Map partials to a sample activity
        cal_db = {
            "ski": {"name": "Downhill Skiing", "cal_per_hour": 500},
            "run": {"name": "Running", "cal_per_hour": 650},
            "walk": {"name": "Walking", "cal_per_hour": 250},
            "swim": {"name": "Swimming", "cal_per_hour": 600},
            "cycle": {"name": "Cycling", "cal_per_hour": 550},
            "dance": {"name": "Dancing", "cal_per_hour": 350},
            "yoga": {"name": "Yoga", "cal_per_hour": 200},
            "gym": {"name": "Weight Training", "cal_per_hour": 450},
            "soccer": {"name": "Soccer", "cal_per_hour": 550},
            "basketball": {"name": "Basketball", "cal_per_hour": 580},
        }
        matched = None
        for key, val in cal_db.items():
            if key in activity_lower or activity_lower in key:
                matched = val
                break
        if not matched:
            # Default generic activity
            matched = {"name": activity.title(), "cal_per_hour": 300}
        duration = 60  # minutes
        total_cal = int(matched["cal_per_hour"] * duration / 60)
        return {
            "name": matched["name"],
            "calories_per_hour": matched["cal_per_hour"],
            "duration_minutes": duration,
            "total_calories": total_cal,
        }

    # BMI
    def BMI(self, weight: float, height: float) -> dict:
        """
        Calculate BMI using metric units (kg, cm).

        Args:
            weight: Weight in kilograms.
            height: Height in centimeters.

        Returns:
            dict: Object with key 'bmi' (float).
        """
        self._next_call()
        if height <= 0 or weight <= 0:
            return {"bmi": 0.0}
        bmi = weight / ((height / 100) ** 2)
        return {"bmi": round(bmi, 1)}

    # Conception Date
    def Conception_Date(self, conception_date: str) -> dict:
        """
        Calculate estimated due date from conception date.

        Args:
            conception_date: Date of conception in YYYY-MM-DD format.

        Returns:
            dict: Object with keys conception_date, estimated_due_date, weeks_at_term, days_at_term.
        """
        self._next_call()
        try:
            conc_date = datetime.strptime(conception_date.strip(), '%Y-%m-%d')
        except ValueError:
            return {
                "conception_date": conception_date,
                "estimated_due_date": "invalid date",
                "weeks_at_term": 0,
                "days_at_term": 0
            }
        due_date = conc_date + timedelta(days=266)  # 38 weeks
        # weeks_at_term is typically 38 from conception
        weeks = 38
        days = 0  # at due date exactly
        return {
            "conception_date": conc_date.strftime('%Y-%m-%d'),
            "estimated_due_date": due_date.strftime('%Y-%m-%d'),
            "weeks_at_term": weeks,
            "days_at_term": days,
        }

    # Fertility Window - GET
    def Fertility_Window_GET(self, cycle_length: str, menstrual_date: str) -> dict:
        """
        Calculate fertility window based on cycle length and last menstrual period date.

        Args:
            cycle_length: Length of menstrual cycle in days (integer as string).
            menstrual_date: First day of last menstrual period in YYYY-MM-DD.

        Returns:
            dict: Object with fertility window, ovulation date, conception date, etc.
        """
        self._next_call()
        try:
            cycle = int(cycle_length)
            lmp = datetime.strptime(menstrual_date.strip(), '%Y-%m-%d')
        except ValueError:
            return {
                "fertility_window_start": "invalid",
                "fertility_window_end": "invalid",
                "ovulation_date": "invalid",
                "conception_date": "invalid",
                "cycle_length": 0,
                "menstrual_date": menstrual_date
            }
        if cycle <= 0:
            cycle = 28
        # Ovulation: 14 days before next period, so LMP + (cycle - 14)
        ovulation = lmp + timedelta(days=cycle - 14)
        # Fertility window: 5 days before to 1 day after ovulation
        window_start = ovulation - timedelta(days=5)
        window_end = ovulation + timedelta(days=1)
        # Conception date is typically ovulation day (if fertilized)
        conception = ovulation
        return {
            "fertility_window_start": window_start.strftime('%Y-%m-%d'),
            "fertility_window_end": window_end.strftime('%Y-%m-%d'),
            "ovulation_date": ovulation.strftime('%Y-%m-%d'),
            "conception_date": conception.strftime('%Y-%m-%d'),
            "cycle_length": cycle,
            "menstrual_date": lmp.strftime('%Y-%m-%d'),
        }

    # GET Attributes
    def GET_Attributes(self) -> dict:
        """
        Get exercise filter attributes: categories, difficulties, forces, muscles.

        Returns:
            dict: Object with keys categories, difficulties, forces, muscles.
        """
        self._next_call()
        return {
            "categories": ["strength", "cardio", "stretching", "powerlifting", "olympic_weightlifting"],
            "difficulties": ["beginner", "intermediate", "expert"],
            "forces": ["push", "pull", "static", "isometric"],
            "muscles": ["biceps", "triceps", "chest", "back", "shoulders", "quadriceps", "hamstrings", "glutes", "abs"],
        }

    # Imperial [Pounds]
    def Imperial_Pounds(self, weight: float, height: float) -> dict:
        """
        Calculate BMI using imperial units (lbs, inches).

        Args:
            weight: Weight in pounds.
            height: Height in inches.

        Returns:
            dict: Object with bmi, weight string, height string, weightCategory.
        """
        self._next_call()
        bmi = (weight / (height ** 2)) * 703.07 if height > 0 else 0.0
        bmi = round(bmi, 1)
        # Determine weight category
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal weight"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return {
            "bmi": bmi,
            "weight": f"{weight:.1f} lbs",
            "height": f"{height:.1f} in",
            "weightCategory": cat,
        }

    # Last Menstrual Period (LMP)
    def Last_Menstrual_Period_LMP(self, cycle_length: str, last_period_date: str) -> dict:
        """
        Calculate estimated due date from last menstrual period.

        Args:
            cycle_length: Average cycle length in days (string).
            last_period_date: Date of first day of LMP in YYYY-MM-DD.

        Returns:
            dict: Object with estimated_due_date and current_weeks_pregnant.
        """
        self._next_call()
        try:
            cycle = int(cycle_length)
            lmp = datetime.strptime(last_period_date.strip(), '%Y-%m-%d')
        except ValueError:
            return {
                "estimated_due_date": "invalid",
                "current_weeks_pregnant": 0.0
            }
        # Due date: LMP + 280 days
        due_date = lmp + timedelta(days=280)
        # Current weeks pregnant from LMP to today (mock today as current date)
        today = datetime.now()
        days_pregnant = (today - lmp).days
        weeks_pregnant = max(0, days_pregnant / 7.0)
        return {
            "estimated_due_date": due_date.strftime('%Y-%m-%d'),
            "current_weeks_pregnant": round(weeks_pregnant, 1),
        }

    # List of bodyparts
    def List_of_bodyparts(self):
        """
        Fetch a list of available body parts.

        Returns:
            list: Body parts names.
        """
        self._next_call()
        return ["chest", "back", "shoulders", "biceps", "triceps", "forearms",
                "quadriceps", "hamstrings", "calves", "glutes", "abs", "obliques",
                "traps", "lats", "delts", "pecs"]

    # List of equipment
    def List_of_equipment(self) -> dict:
        """
        Fetch a list of available equipment.

        Returns:
            dict: Object with total_count and page (example pagination).
        """
        self._next_call()
        return {"total_count": 25, "page": 1}

    # List of target muscles
    def List_of_target_muscles(self) -> list:
        """
        Fetch a list of available target muscles.

        Returns:
            list: Target muscle names.
        """
        self._next_call()
        return ["biceps", "triceps", "chest", "back", "shoulders", "quadriceps",
                "hamstrings", "glutes", "abs", "calves", "forearms", "traps"]

    # Macronutrient Distribution
    def Macronutrient_Distribution(self) -> dict:
        """
        Calculate optimal macronutrient distribution.
        Note: This endpoint returns an error because no parameters are provided.

        Returns:
            dict: Object with only 'error' key.
        """
        self._next_call()
        return {"error": "Missing required parameters: activity_level, body_composition_goal, dietary_preferences"}

    # View All Cores With Their Food Items
    def View_All_Cores_With_Their_Food_Items(self) -> dict:
        """
        Retrieve all cores with associated food items.

        Returns:
            dict: Object with core_count.
        """
        self._next_call()
        return {"core_count": 12}

    # View All Food Items
    def View_All_Food_Items(self) -> dict:
        """
        Retrieve a comprehensive list of all food items.

        Returns:
            dict: Object with count.
        """
        self._next_call()
        return {"count": 150}

    # View Food Item By Name
    def View_Food_Item_By_Name(self) -> dict:
        """
        Retrieve details about a food item by name.
        (Note: name would be passed via configuration but here no parameters.)

        Returns:
            dict: Object with count (mock).
        """
        self._next_call()
        return {"count": 1}

    # View Food Items by Core
    def View_Food_Items_by_Core(self) -> dict:
        """
        Retrieve food items filtered by core category.
        (Note: core values would be passed via configuration but here no parameters.)

        Returns:
            dict: Object with count.
        """
        self._next_call()
        return {"count": 5}

    # Weight Category
    def Weight_Category(self, bmi: float) -> dict:
        """
        Determine weight category from BMI value.

        Args:
            bmi: Body Mass Index number.

        Returns:
            dict: Object with bmi (string) and weightCategory.
        """
        self._next_call()
        bmi_val = round(float(bmi), 1)
        bmi_str = f"{bmi_val:.1f}"
        if bmi_val < 18.5:
            cat = "Underweight"
        elif bmi_val < 25:
            cat = "Normal weight"
        elif bmi_val < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return {"bmi": bmi_str, "weightCategory": cat}

    # getHospitalsByName
    def getHospitalsByName(self, name: str) -> dict:
        """
        Find US hospitals by name.

        Args:
            name: Search string (can be partial, e.g. 'pr' for Presbyterian).

        Returns:
            dict: Object with 'message' string containing results.
        """
        self._next_call()
        # Mock response: simulate a list of hospitals matching the name
        mock_hospitals = [
            "Presbyterian Hospital of Dallas",
            "Presbyterian/St. Luke's Medical Center",
            "Presbyterian Intercommunity Hospital",
            "Presbyterian Hospital of Greenville",
        ]
        filtered = [h for h in mock_hospitals if name.lower() in h.lower()]
        if not filtered:
            message = f"No hospitals found matching '{name}'."
        else:
            message = f"Found {len(filtered)} hospital(s) matching '{name}': " + "; ".join(filtered[:3])
        return {"message": message}

    # hoscoscope
    def hoscoscope(self, date: str, sign: str) -> dict:
        """
        Retrieve horoscope for a given sign and date.

        Args:
            date: Date string (e.g., '2023-03-15').
            sign: Zodiac sign (e.g., 'virgo').

        Returns:
            dict: Object with sign, date, horoscope text.
        """
        self._next_call()
        seed(hash(date.strip()) + hash(sign.strip().lower()))
        horoscopes = [
            "You will have a productive day. New opportunities are on the horizon.",
            "Be mindful of your health today. A calm approach will bring success.",
            "Today is a good day to start a new project. Trust your instincts.",
            "Relationships are highlighted. Spend time with loved ones.",
            "Focus on your finances. A long-term investment may pay off.",
            "Take a break and recharge. Your energy will be needed later.",
            "Communication is key today. Express yourself clearly.",
            "An unexpected encounter may lead to a fruitful partnership.",
            "You may feel restless, but channel that energy into creativity.",
            "Patience will be rewarded. Do not rush into decisions.",
        ]
        horoscope = choice(horoscopes)
        return {
            "sign": sign.strip().title(),
            "date": date.strip(),
            "horoscope": horoscope,
        }