"""Hardware sensor drivers and the shared calibration step."""

from __future__ import annotations


def apply_calibration(
    cfg: dict | None, temperature: float, humidity: float, lux: float
) -> tuple[float, float, float]:
    """Apply the configured offsets to one sample.

    Offsets live in `config.yaml` under `sensors` so that calibration survives
    a reinstall and is visible in one place (see docs/calibration.md). Values
    are clamped to physical ranges so a mistyped offset cannot poison FGPI.
    """
    sensors = (cfg or {}).get("sensors", {}) if cfg else {}
    try:
        t_offset = float(sensors.get("temp_offset", 0.0) or 0.0)
    except (TypeError, ValueError):
        t_offset = 0.0
    try:
        rh_offset = float(sensors.get("humidity_offset", 0.0) or 0.0)
    except (TypeError, ValueError):
        rh_offset = 0.0
    try:
        lux_scale = float(sensors.get("lux_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        lux_scale = 1.0

    adjusted_t = max(-40.0, min(80.0, float(temperature) + t_offset))
    adjusted_rh = max(0.0, min(100.0, float(humidity) + rh_offset))
    adjusted_lux = float(lux) if lux is not None else -1.0
    if adjusted_lux >= 0 and lux_scale > 0:
        adjusted_lux = adjusted_lux * lux_scale
    return round(adjusted_t, 2), round(adjusted_rh, 2), round(adjusted_lux, 1)
