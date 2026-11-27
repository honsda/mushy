"""FGPI behaviour.

Two humidity curves are supported:

* `spec` - the literal section 8 ramp. Frozen: these numbers must not drift.
* `smooth` (default) - a continuous smoothstep over 40-90 %RH, so the index
  responds to measured humidity instead of being pinned at 0 below 60 %RH.
  The endpoints keep their meaning (0 at <= 40, 1 at >= 90).

Every value below passes `curve=` explicitly so the tests do not depend on
config.yaml.
"""

from __future__ import annotations

import pytest

from app.models.fgpi import compute_fgpi, humidity_factor


# --------------------------------------------------------------------- spec

def test_spec_optimal_conditions_are_full_marks():
    assert compute_fgpi(22.0, 85.0, 0.0, curve="spec") == (100.0, "critical")
    assert compute_fgpi(20.0, 80.0, -1.0, curve="spec") == (100.0, "critical")
    assert compute_fgpi(25.0, 100.0, 500.0, curve="spec") == (100.0, "critical")


@pytest.mark.parametrize("temperature", [5.0, 4.0, 0.0, -10.0, 35.0, 36.0, 50.0])
def test_spec_temperature_edges_are_zero(temperature):
    score, risk = compute_fgpi(temperature, 90.0, 0.0, curve="spec")
    assert score == 0.0
    assert risk == "low"


@pytest.mark.parametrize("rh", [0.0, 40.0, 60.0])
def test_spec_humidity_floor_is_zero(rh):
    assert compute_fgpi(22.0, rh, 0.0, curve="spec") == (0.0, "low")


def test_spec_ramp_midpoints():
    # f_T(12.5) = 0.5, f_RH(70) = 0.5 -> 25 -> "moderate" boundary
    score, risk = compute_fgpi(12.5, 70.0, 0.0, curve="spec")
    assert score == pytest.approx(25.0)
    assert risk == "moderate"
    # f_T(30) = 0.5, f_RH(90) = 1.0 -> 50 -> "moderate"/"high" boundary
    score, risk = compute_fgpi(30.0, 90.0, 0.0, curve="spec")
    assert score == pytest.approx(50.0)
    assert risk == "high"


def test_spec_risk_thresholds():
    assert compute_fgpi(22.0, 61.0, 0.0, curve="spec")[1] == "low"      # 5.0
    assert compute_fgpi(22.0, 65.0, 0.0, curve="spec")[1] == "moderate"  # 25.0
    assert compute_fgpi(22.0, 70.0, 0.0, curve="spec")[1] == "high"      # 50.0
    assert compute_fgpi(22.0, 75.0, 0.0, curve="spec")[1] == "high"      # 75.0
    assert compute_fgpi(22.0, 80.0, 0.0, curve="spec")[1] == "critical"  # 100.0


# ------------------------------------------------------------------- smooth

def test_smooth_has_no_cliff_in_the_operating_range():
    """A 1 %RH step must never move the index by more than a few points."""
    previous = None
    for tenths in range(0, 1001):  # 0.0 .. 100.0 %RH in 0.1 steps
        score = compute_fgpi(22.0, tenths / 10.0, 0.0, curve="smooth")[0]
        if previous is not None:
            assert abs(score - previous) <= 1.0, f"cliff at {tenths / 10.0} %RH"
        previous = score


def test_smooth_endpoints_keep_their_meaning():
    # Dry air is a genuine zero, saturation is a genuine hundred.
    assert compute_fgpi(22.0, 40.0, 0.0, curve="smooth") == (0.0, "low")
    assert compute_fgpi(22.0, 20.0, 0.0, curve="smooth") == (0.0, "low")
    assert compute_fgpi(22.0, 90.0, 0.0, curve="smooth") == (100.0, "critical")
    assert compute_fgpi(22.0, 99.0, 0.0, curve="smooth") == (100.0, "critical")


def test_smooth_is_monotonic_and_strictly_between_endpoints():
    scores = [humidity_factor(rh / 2.0, "smooth") for rh in range(80, 181)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(a <= b for a, b in zip(scores, scores[1:])), "f_RH is not monotonic"
    assert scores[0] == 0.0 and scores[-1] == 1.0
    # ... and it genuinely moves in the middle instead of stepping at 60.
    assert humidity_factor(50.0, "smooth") > 0.05
    assert 0.2 < humidity_factor(60.0, "smooth") < 0.6
    assert 0.5 < humidity_factor(70.0, "smooth") < 0.9


def test_smooth_reacts_below_the_old_spec_floor():
    """The reported bug: a dry room pinned FGPI at exactly 0 for ever."""
    dry = compute_fgpi(25.0, 43.0, 0.0, curve="smooth")[0]
    assert 0.0 < dry < 5.0  # 43 %RH stays low, but it is no longer a dead zero
    wetter = compute_fgpi(25.0, 55.0, 0.0, curve="smooth")[0]
    assert wetter > 5.0  # ...but +12 %RH now visibly raises the index
    assert compute_fgpi(25.0, 60.0, 0.0, curve="smooth")[0] > 25.0


def test_default_curve_comes_from_config():
    """config.yaml ships `humidity_curve: smooth`; no config -> same default."""
    assert humidity_factor(60.0) > 0.0
    assert compute_fgpi(22.0, 70.0, 0.0)[0] > compute_fgpi(22.0, 70.0, 0.0, curve="spec")[0]


# -------------------------------------------------------------------- common

def test_light_is_neutral_and_bad_input_degrades():
    for curve in ("spec", "smooth"):
        assert compute_fgpi(22.0, 90.0, 0.0, curve=curve)[0] == compute_fgpi(
            22.0, 90.0, 99999.0, curve=curve
        )[0]
        assert compute_fgpi("nonsense", 90.0, curve=curve) == (0.0, "low")
        assert compute_fgpi(float("nan"), 90.0, curve=curve) == (0.0, "low")
        assert compute_fgpi(22.0, float("nan"), curve=curve) == (0.0, "low")
