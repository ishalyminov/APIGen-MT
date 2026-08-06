import math

from tools.math_api import MathAPI


def test_area_conversion_uses_squared_linear_factor():
    api = MathAPI({})
    result = api.imperial_si_conversion(
        value=367.35,
        unit_in="square meters",
        unit_out="square feet",
    )
    assert "error" not in result
    assert math.isclose(
        result["result"],
        367.35 * (1 / 0.3048) ** 2,
        rel_tol=1e-12,
    )


def test_unsupported_conversion_returns_error_not_fake_zero():
    api = MathAPI({})
    result = api.imperial_si_conversion(
        value=1,
        unit_in="meter",
        unit_out="kilogram",
    )
    assert "error" in result
    assert "result" not in result


def test_temperature_conversion_accepts_degree_symbol_aliases():
    api = MathAPI({})
    result = api.imperial_si_conversion(
        value=98.7,
        unit_in="°F",
        unit_out="°C",
    )
    assert "error" not in result
    assert math.isclose(
        result["result"],
        (98.7 - 32) * 5 / 9,
        rel_tol=1e-12,
    )
