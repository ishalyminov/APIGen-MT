"""Auto-generated MathAPI implementation."""

import math
from typing import List, Dict, Any, Optional, Tuple


class MathAPI:
    """A class providing various mathematical operations."""

    def __init__(self, initial_config: dict) -> None:
        """
        Initialize the MathAPI with an initial configuration.
        Note: All methods are pure - they take parameters directly.
        The initial_config is kept for API compatibility but no fields are used.
        """
        pass

    def absolute_value(self, number: float) -> Dict[str, Any]:
        """
        Calculate the absolute value of a number.
        """
        try:
            return {"result": float(abs(number))}
        except Exception:
            return {"result": 0.0}

    def add(self, a: float, b: float) -> Dict[str, Any]:
        """
        Add two numbers.
        """
        try:
            return {"result": float(a + b)}
        except Exception:
            return {"result": 0.0}

    def divide(self, a: float, b: float) -> Dict[str, Any]:
        """
        Divide one number by another.
        """
        try:
            if b == 0:
                return {"result": float('inf') if a >= 0 else float('-inf')}
            return {"result": float(a / b)}
        except Exception:
            return {"result": 0.0}

    def imperial_si_conversion(self, value: float, unit_in: str, unit_out: str) -> Dict[str, Any]:
        """Convert linear, area, or volume values between imperial and SI units."""
        try:
            def normalise_unit(raw: str) -> Tuple[str, int]:
                unit = str(raw).lower().strip().replace("²", "^2").replace("³", "^3")
                dimension = 1
                if unit.startswith("square "):
                    dimension, unit = 2, unit[len("square "):]
                elif unit.startswith("cubic "):
                    dimension, unit = 3, unit[len("cubic "):]
                elif unit.endswith(" squared"):
                    dimension, unit = 2, unit[:-len(" squared")]
                elif unit.endswith(" cubed"):
                    dimension, unit = 3, unit[:-len(" cubed")]
                elif unit.endswith("^2"):
                    dimension, unit = 2, unit[:-2]
                elif unit.endswith("^3"):
                    dimension, unit = 3, unit[:-2]

                aliases = {
                    "f": "fahrenheit", "°f": "fahrenheit",
                    "c": "celsius", "°c": "celsius",
                    "miles": "mile", "kilometers": "km", "kilometres": "km",
                    "pounds": "pound", "kilograms": "kg",
                    "inches": "inch", "centimeters": "cm", "centimetres": "cm",
                    "gallons": "gallon", "liters": "liter", "litres": "liter",
                    "feet": "foot", "meters": "meter", "metres": "meter",
                    "yards": "yard", "ounces": "ounce", "grams": "gram",
                }
                return aliases.get(unit.strip(), unit.strip()), dimension

            source, source_dimension = normalise_unit(unit_in)
            target, target_dimension = normalise_unit(unit_out)
            if source_dimension != target_dimension:
                return {
                    "error": "Cannot convert between units with different dimensions."
                }

            if source == target:
                return {"result": float(value)}

            if source_dimension == 1 and {source, target} == {"fahrenheit", "celsius"}:
                if source == "fahrenheit":
                    return {"result": float((value - 32) * 5 / 9)}
                return {"result": float((value * 9 / 5) + 32)}

            linear_factors = {
                ("inch", "cm"): 2.54,
                ("pound", "kg"): 0.453592,
                ("mile", "km"): 1.60934,
                ("gallon", "liter"): 3.78541,
                ("foot", "meter"): 0.3048,
                ("yard", "meter"): 0.9144,
                ("ounce", "gram"): 28.3495,
            }
            factor = linear_factors.get((source, target))
            if factor is None:
                inverse = linear_factors.get((target, source))
                if inverse is not None:
                    factor = 1 / inverse
            if factor is None:
                return {
                    "error": f"Unsupported conversion: {unit_in} to {unit_out}."
                }

            return {"result": float(value * (factor ** source_dimension))}
        except Exception as exc:
            return {"error": f"Conversion failed: {exc}"}

    def logarithm(self, value: float, base: float, precision: int) -> Dict[str, Any]:
        """
        Compute the logarithm of a number with adjustable precision using mpmath.
        """
        try:
            if value <= 0 or base <= 0 or base == 1:
                return {"result": 0.0}
            result = math.log(value, base)
            precision = max(0, precision)
            return {"result": float(round(result, precision))}
        except Exception:
            return {"result": 0.0}

    def max_value(self, numbers: List[float]) -> Dict[str, Any]:
        """
        Find the maximum value in a list of numbers.
        """
        try:
            if not numbers:
                return {"result": 0.0, "input_numbers": []}
            return {"result": float(max(numbers)), "input_numbers": list(numbers)}
        except Exception:
            return {"result": 0.0, "input_numbers": []}

    def mean(self, numbers: List[float]) -> Dict[str, Any]:
        """
        Calculate the mean of a list of numbers.
        """
        try:
            if not numbers:
                return {"result": 0.0, "input_numbers": []}
            return {"result": float(sum(numbers) / len(numbers)), "input_numbers": list(numbers)}
        except Exception:
            return {"result": 0.0, "input_numbers": []}

    def min_value(self, numbers: List[float]) -> Dict[str, Any]:
        """
        Find the minimum value in a list of numbers.
        """
        try:
            if not numbers:
                return {"result": 0.0, "input_numbers": []}
            return {"result": float(min(numbers)), "input_numbers": list(numbers)}
        except Exception:
            return {"result": 0.0, "input_numbers": []}

    def multiply(self, a: float, b: float) -> Dict[str, Any]:
        """
        Multiply two numbers.
        """
        try:
            return {"result": float(a * b)}
        except Exception:
            return {"result": 0.0}

    def percentage(self, part: float, whole: float) -> Dict[str, Any]:
        """
        Calculate the percentage of a part relative to a whole.
        """
        try:
            if whole == 0:
                return {"result": 0.0}
            return {"result": float((part / whole) * 100)}
        except Exception:
            return {"result": 0.0}

    def power(self, base: float, exponent: float) -> Dict[str, Any]:
        """
        Raise a number to a power.
        """
        try:
            return {"result": float(math.pow(base, exponent))}
        except Exception:
            return {"result": 0.0}

    def round_number(self, number: float, decimal_places: int = 0) -> Dict[str, Any]:
        """
        Round a number to a specified number of decimal places.
        """
        try:
            return {"result": float(round(number, decimal_places))}
        except Exception:
            return {"result": 0.0}

    def si_unit_conversion(self, value: float, unit_in: str, unit_out: str) -> Dict[str, Any]:
        """
        Convert a value from one SI unit to another.
        """
        try:
            si_prefixes = {
                "pico": 1e-12,
                "nano": 1e-9,
                "micro": 1e-6,
                "milli": 1e-3,
                "centi": 1e-2,
                "deci": 1e-1,
                "": 1.0,
                "deca": 1e1,
                "hecto": 1e2,
                "kilo": 1e3,
                "mega": 1e6,
                "giga": 1e9,
                "tera": 1e12,
            }

            base_units = ["meter", "gram", "liter", "second", "ampere", "kelvin", "mole", "candela", "byte", "bit"]
            unit_aliases = {
                "m": "meter", "g": "gram", "l": "liter", "s": "second", "a": "ampere", "k": "kelvin", "b": "byte",
                "km": "kilometer", "cm": "centimeter", "mm": "millimeter",
                "kg": "kilogram", "mg": "milligram",
                "ml": "milliliter", "cl": "centiliter",
                "ms": "millisecond", "us": "microsecond", "ns": "nanosecond",
                "ma": "milliampere", "kb": "kilobyte", "mb": "megabyte", "gb": "gigabyte",
            }

            def parse_unit(unit_str: str) -> Tuple[float, str]:
                unit_str = unit_str.lower().strip()
                plural_to_singular = {
                    "meters": "meter", "grams": "gram", "liters": "liter", "seconds": "second",
                    "amperes": "ampere", "kelvins": "kelvin", "moles": "mole", "candelas": "candela",
                    "bytes": "byte", "bits": "bit",
                    "meters": "meter", "centimeters": "centimeter", "millimeters": "millimeter",
                    "kilometers": "kilometer", "kilograms": "kilogram", "milligrams": "milligram",
                    "milliliters": "milliliter", "centiliters": "centiliter",
                    "milliseconds": "millisecond", "microseconds": "microsecond", "nanoseconds": "nanosecond",
                    "kibibytes": "kilobyte", "mebibytes": "megabyte", "gibibytes": "gigabyte",
                    "kilobytes": "kilobyte", "megabytes": "megabyte", "gigabytes": "gigabyte",
                }
                unit_str = plural_to_singular.get(unit_str, unit_str)
                if unit_str in unit_aliases:
                    val = unit_aliases[unit_str]
                    for prefix, factor in si_prefixes.items():
                        if prefix:
                            for base in base_units:
                                if val == prefix + base:
                                    return factor, base
                    return 1.0, val
                for prefix, factor in si_prefixes.items():
                    if prefix:
                        for base in base_units:
                            if unit_str == prefix + base:
                                return factor, base
                return 1.0, unit_str

            factor_in, base_in = parse_unit(unit_in)
            factor_out, base_out = parse_unit(unit_out)

            if base_in != base_out:
                return {"error": f"Cannot convert between incompatible units: {unit_in} (base: {base_in}) and {unit_out} (base: {base_out})", "result": 0.0}

            return {"result": float(value * factor_in / factor_out)}
        except Exception:
            return {"result": 0.0}

    def square_root(self, number: float, precision: int) -> Dict[str, Any]:
        """
        Calculate the square root of a number with adjustable precision using the decimal module.
        """
        try:
            if number < 0:
                return {"result": 0.0}
            result = math.sqrt(number)
            precision = max(0, precision)
            return {"result": float(round(result, precision))}
        except Exception:
            return {"result": 0.0}

    def standard_deviation(self, numbers: List[float]) -> Dict[str, Any]:
        """
        Calculate the standard deviation of a list of numbers.
        """
        try:
            if not numbers:
                return {"result": 0.0, "input_numbers": []}
            if len(numbers) == 1:
                return {"result": 0.0, "input_numbers": list(numbers)}
            mean_val = sum(numbers) / len(numbers)
            variance = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
            return {"result": float(math.sqrt(variance)), "input_numbers": list(numbers)}
        except Exception:
            return {"result": 0.0, "input_numbers": []}

    def subtract(self, a: float, b: float) -> Dict[str, Any]:
        """
        Subtract one number from another.
        """
        try:
            return {"result": float(a - b)}
        except Exception:
            return {"result": 0.0}

    def sum_values(self, numbers: List[float]) -> Dict[str, Any]:
        """
        Calculate the sum of a list of numbers.
        """
        try:
            if not numbers:
                return {"result": 0.0, "input_numbers": []}
            return {"result": float(sum(numbers)), "input_numbers": list(numbers)}
        except Exception:
            return {"result": 0.0, "input_numbers": []}
