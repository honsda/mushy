"""Payload validators for the JSON API.

Plain-dict validation on purpose: the Pi should not need a pydantic wheel for
four fields, and the error messages are already the ones an operator needs.
Every function either returns a fully typed dict or raises ``ValidationError``
with a message that is safe to echo to the client.
"""

from __future__ import annotations

import re
import time

NODE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
SPECIES_RE = re.compile(r"^[A-Za-z][A-Za-z .()/_+-]{0,63}$")

TEMPERATURE_RANGE = (-40.0, 80.0)
HUMIDITY_RANGE = (0.0, 100.0)
LUX_MAX = 200_000.0
LUX_UNAVAILABLE = -1.0


class ValidationError(ValueError):
    """Client-safe validation failure (mapped to HTTP 400)."""


def _number(value, field: str, low: float, high: float) -> float:
    if value is None or value == "":
        raise ValidationError(f"{field} is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number") from None
    if number != number:  # NaN
        raise ValidationError(f"{field} must be a number")
    if not (low <= number <= high):
        raise ValidationError(f"{field} outside {low:g}..{high:g}")
    return number


def _optional_number(value, field: str, low: float, high: float) -> float | None:
    if value is None or value == "":
        return None
    return _number(value, field, low, high)


def validate_reading(data, spatial: dict | None = None) -> dict:
    """Normalise an /api/ingest payload.

    Coordinates are clamped to the configured room bounds rather than rejected:
    a node with slightly stale calibration should still record its sample, and
    the map clips anyway. Out-of-range *sensor* values are still rejected,
    because those usually mean a wiring fault.
    """
    if not isinstance(data, dict):
        raise ValidationError("JSON object required")
    spatial = spatial or {}
    x_max = float(spatial.get("x_max", 10.0))
    y_max = float(spatial.get("y_max", 8.0))

    temperature = _number(
        data.get("temperature", data.get("T")), "temperature", *TEMPERATURE_RANGE
    )
    humidity = _number(data.get("humidity", data.get("RH")), "humidity", *HUMIDITY_RANGE)

    lux_raw = data.get("lux", data.get("L", LUX_UNAVAILABLE))
    if lux_raw is None or lux_raw == "":
        lux = LUX_UNAVAILABLE
    else:
        try:
            lux_value = float(lux_raw)
        except (TypeError, ValueError):
            raise ValidationError("lux must be a number") from None
        if lux_value != lux_value:
            raise ValidationError("lux must be a number")
        lux = LUX_UNAVAILABLE if lux_value < 0 else lux_value

    if lux > LUX_MAX:
        raise ValidationError(f"lux outside 0..{LUX_MAX:g}")

    x = _optional_number(data.get("x"), "x", 0.0, x_max)
    y = _optional_number(data.get("y"), "y", 0.0, y_max)

    node_id = str(data.get("node_id") or "remote").strip()
    if not NODE_ID_RE.match(node_id):
        raise ValidationError("node_id must match [A-Za-z0-9_-]{1,32}")

    if data.get("ts") is not None:
        try:
            ts = int(float(data["ts"]))
        except (TypeError, ValueError):
            raise ValidationError("ts must be an integer epoch second") from None
    else:
        ts = int(time.time())

    return {
        "temperature": temperature,
        "humidity": humidity,
        "lux": lux,
        "x": x,
        "y": y,
        "ts": ts,
        "node_id": node_id,
    }


def validate_label(data) -> dict:
    """Normalise an /api/labels payload. Manual ground truth only."""
    if not isinstance(data, dict):
        raise ValidationError("JSON object required")

    raw_id = data.get("reading_id")
    if raw_id is None or raw_id == "":
        raise ValidationError("reading_id is required")
    try:
        reading_id = int(raw_id)
    except (TypeError, ValueError):
        raise ValidationError("reading_id must be an integer") from None
    if reading_id < 1:
        raise ValidationError("reading_id must be positive")

    species = " ".join(str(data.get("species") or "").split())
    if not species:
        raise ValidationError("species is required")
    if not SPECIES_RE.match(species):
        raise ValidationError("species must be 1-64 characters of letters, spaces or .()/_+-")

    observed = data.get("observed")
    if isinstance(observed, str):
        observed = observed.strip().lower() in {"1", "true", "yes", "present"}
    elif observed is None:
        raise ValidationError("observed is required (true = present)")
    else:
        observed = bool(observed)

    confidence_raw = data.get("confidence", 3)
    try:
        confidence = int(confidence_raw)
    except (TypeError, ValueError):
        raise ValidationError("confidence must be 1..5") from None
    if not (1 <= confidence <= 5):
        raise ValidationError("confidence must be 1..5")

    note = str(data.get("note") or "").strip()[:280]
    operator = str(data.get("operator") or "operator").strip()[:32]
    image_path = data.get("image_path")
    if image_path:
        # User-supplied reference only; the Pi never captures images.
        image_path = str(image_path).strip()[:240]

    return {
        "reading_id": reading_id,
        "species": species,
        "observed": observed,
        "confidence": confidence,
        "note": note,
        "operator": operator,
        "image_path": image_path or None,
    }
