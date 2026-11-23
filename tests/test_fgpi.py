"""FGPI is frozen by spec section 8: these values must not drift."""

from __future__ import annotations

import pytest

from app.models.fgpi import compute_fgpi


def test_optimal_conditions_are_full_marks():
    assert compute_fgpi(22.0, 85.0, 0.0) == (100.0, "critical")
    assert compute_fgpi(20.0, 80.0, -1.0) == (100.0, "critical")
    assert compute_fgpi(25.0, 100.0, 500.0) == (100.0, "critical")


@pytest.mark.parametrize("temperature", [5.0, 4.0, 0.0, -10.0, 35.0, 36.0, 50.0])
def test_temperature_edges_are_zero(temperature):
    score, risk = compute_fgpi(temperature, 90.0, 0.0)
    assert score == 0.0
    assert risk == "low"


@pytest.mark.parametrize("rh", [0.0, 40.0, 60.0])
def test_humidity_floor_is_zero(rh):
    assert compute_fgpi(22.0, rh, 0.0) == (0.0, "low")


def test_ramp_midpoints():
    # f_T(12.5) = 0.5, f_RH(70) = 0.5 -> 25 -> "moderate" boundary
    score, risk = compute_fgpi(12.5, 70.0, 0.0)
    assert score == pytest.approx(25.0)
    assert risk == "moderate"
    # f_T(30) = 0.5, f_RH(90) = 1.0 -> 50 -> "moderate"/"high" boundary
    score, risk = compute_fgpi(30.0, 90.0, 0.0)
    assert score == pytest.approx(50.0)
    assert risk == "high"


def test_risk_thresholds():
    assert compute_fgpi(22.0, 61.0, 0.0)[1] == "low"      # 5.0
    assert compute_fgpi(22.0, 65.0, 0.0)[1] == "moderate"  # 25.0
    assert compute_fgpi(22.0, 70.0, 0.0)[1] == "high"      # 50.0 (<=75 is high)
    assert compute_fgpi(22.0, 75.0, 0.0)[1] == "high"      # 75.0
    assert compute_fgpi(22.0, 80.0, 0.0)[1] == "critical"  # 100.0


def test_light_is_neutral_and_bad_input_degrades():
    assert compute_fgpi(22.0, 90.0, 0.0)[0] == compute_fgpi(22.0, 90.0, 99999.0)[0]
    assert compute_fgpi("nonsense", 90.0) == (0.0, "low")
    assert compute_fgpi(float("nan"), 90.0) == (0.0, "low")
