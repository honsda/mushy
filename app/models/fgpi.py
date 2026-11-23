"""Fungal Growth Potential Index (FGPI).

Spec section 8, implemented verbatim:

    f_T(T)  = 0              T <= 5 or T >= 35
            = (T-5)/15       5 < T < 20
            = 1              20 <= T <= 25
            = (35-T)/10      25 < T < 35

    f_RH(RH) = 0             RH <= 60
             = (RH-60)/20    60 < RH < 80
             = 1             RH >= 80

    f_L(L) = 1.0             light-neutral by design

    FGPI = 100 * f_T * f_RH * f_L

Light is intentionally not weighted: these species respond primarily to
moisture and temperature, and cheap lux sensors are easy to shadow or saturate.
"""

from __future__ import annotations

RISK_CLASSES = ("low", "moderate", "high", "critical")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _temperature_factor(temperature: float) -> float:
    if temperature <= 5.0 or temperature >= 35.0:
        return 0.0
    if temperature < 20.0:
        return _clamp01((temperature - 5.0) / 15.0)
    if temperature <= 25.0:
        return 1.0
    return _clamp01((35.0 - temperature) / 10.0)


def _humidity_factor(humidity: float) -> float:
    if humidity <= 60.0:
        return 0.0
    if humidity < 80.0:
        return _clamp01((humidity - 60.0) / 20.0)
    return 1.0


def risk_class(fgpi: float) -> str:
    """Map an FGPI score to the fixed risk buckets (section 8)."""
    if fgpi < 25.0:
        return "low"
    if fgpi < 50.0:
        return "moderate"
    if fgpi <= 75.0:
        return "high"
    return "critical"


def compute_fgpi(T: float, RH: float, L: float = 0.0) -> tuple[float, str]:
    """Return (fgpi 0-100, risk_class in {low, moderate, high, critical}).

    Invalid sensor values degrade to 0 rather than raising: a dashboard that
    shows "low" beats a sampler that dies.
    """
    try:
        t = float(T)
        rh = float(RH)
    except (TypeError, ValueError):
        return 0.0, "low"

    if t != t or rh != rh:  # NaN guard
        return 0.0, "low"

    # L is accepted for contract stability; f_L is fixed at 1.0.
    score = round(100.0 * _temperature_factor(t) * _humidity_factor(rh) * 1.0, 2)
    score = max(0.0, min(100.0, score))
    return score, risk_class(score)
