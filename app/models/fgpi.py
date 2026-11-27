"""Fungal Growth Potential Index (FGPI).

    FGPI = 100 * f_T * f_RH * f_L

`f_T` and `f_L` are spec section 8, verbatim:

    f_T(T)  = 0              T <= 5 or T >= 35
            = (T-5)/15       5 < T < 20
            = 1              20 <= T <= 25
            = (35-T)/10      25 < T < 35

    f_L(L)  = 1.0           light-neutral by design

`f_RH` has two selectable shapes (`fgpi.humidity_curve` in config.yaml):

  smooth (default)  continuous smoothstep over `fgpi.humidity_range`
                    (40 -> 90 %RH by default):

                        f_RH = 0                     RH <= lo
                             = 3t^2 - 2t^3           t = (RH-lo)/(hi-lo)
                             = 1                     RH >= hi

                    The index therefore tracks measured humidity everywhere in
                    the sensor's usable range instead of being pinned at 0 for
                    every dry reading and then jumping 0 -> 100 between 60 and
                    80 %RH. Endpoints still mean what they meant: no growth
                    potential at <= lo, saturated at >= hi.

  spec              the literal section 8 ramp, kept for reproducibility:

                        f_RH = 0                     RH <= 60
                             = (RH-60)/20            60 < RH < 80
                             = 1                     RH >= 80

Both curves agree at the extremes and differ only in the middle, so the
risk-class boundaries and the spatial/heatmap domains are unchanged.

Light is intentionally not weighted: these species respond primarily to
moisture and temperature, and cheap lux sensors are easy to shadow or saturate.
"""

from __future__ import annotations

import os
from pathlib import Path

RISK_CLASSES = ("low", "moderate", "high", "critical")

CURVES = ("smooth", "spec")
DEFAULT_CURVE = "smooth"
DEFAULT_RH_RANGE = (40.0, 90.0)
SPEC_RH_RANGE = (60.0, 80.0)

# (config mtime, curve, lo, hi) - resolving this per call would deepcopy the
# whole config for every grid cell / dataset row.
_settings_cache: tuple[float, str, float, float] | None = None


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


def _humidity_factor_smooth(humidity: float, lo: float, hi: float) -> float:
    if humidity <= lo:
        return 0.0
    if humidity >= hi:
        return 1.0
    t = (humidity - lo) / (hi - lo)
    return _clamp01(t * t * (3.0 - 2.0 * t))


def _humidity_factor_spec(humidity: float) -> float:
    if humidity <= 60.0:
        return 0.0
    if humidity < 80.0:
        return _clamp01((humidity - 60.0) / 20.0)
    return 1.0


def _settings() -> tuple[str, float, float]:
    """Resolve (curve, lo, hi) from config.yaml, cached by config mtime."""
    global _settings_cache
    try:
        from app import BASE_DIR, load_config  # local: keeps this module import-free

        raw_path = os.environ.get("FUNGI_CONFIG") or str(BASE_DIR / "config.yaml")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        stamp = path.stat().st_mtime
    except Exception:  # no config file at all: defaults are fine, never raise
        return DEFAULT_CURVE, DEFAULT_RH_RANGE[0], DEFAULT_RH_RANGE[1]

    if _settings_cache is not None and _settings_cache[0] == stamp:
        return _settings_cache[1], _settings_cache[2], _settings_cache[3]

    try:
        section = (load_config() or {}).get("fgpi") or {}
    except Exception:  # a broken config must never break a sample
        section = {}

    curve = str(section.get("humidity_curve", DEFAULT_CURVE)).strip().lower()
    if curve not in CURVES:
        curve = DEFAULT_CURVE
    default_range = SPEC_RH_RANGE if curve == "spec" else DEFAULT_RH_RANGE
    try:
        lo, hi = (float(v) for v in section.get("humidity_range", default_range))
    except (TypeError, ValueError):
        lo, hi = default_range
    if not hi > lo:  # a bad range must not produce a divide-by-zero
        lo, hi = default_range

    _settings_cache = (stamp, curve, lo, hi)
    return curve, lo, hi


def humidity_factor(humidity: float, curve: str | None = None) -> float:
    """f_RH for a given RH, using `curve` or the configured curve."""
    selected, lo, hi = _settings()
    selected = (curve or selected).strip().lower()
    if selected == "spec":
        return _humidity_factor_spec(humidity)
    return _humidity_factor_smooth(humidity, lo, hi)


def risk_class(fgpi: float) -> str:
    """Map an FGPI score to the fixed risk buckets (section 8)."""
    if fgpi < 25.0:
        return "low"
    if fgpi < 50.0:
        return "moderate"
    if fgpi <= 75.0:
        return "high"
    return "critical"


def compute_fgpi(
    T: float, RH: float, L: float = 0.0, *, curve: str | None = None
) -> tuple[float, str]:
    """Return (fgpi 0-100, risk_class in {low, moderate, high, critical}).

    `curve` overrides `fgpi.humidity_curve` for this call ("smooth" | "spec");
    the positional signature stays as spec section 7 defines it.

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
    score = 100.0 * _temperature_factor(t) * humidity_factor(rh, curve) * 1.0
    score = round(max(0.0, min(100.0, score)), 2)
    return score, risk_class(score)
