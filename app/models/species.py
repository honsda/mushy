"""Species suitability prediction — two-tier strategy (spec section 9).

Tier 1 (always available): triangular cardinal-temperature membership and a
saturating RH membership from species_curves.json, optionally modulated by a
lux preference. Values are literature-derived niche midpoints; they support
*ranking*, not detection.

Tier 2 (preferred when trained/species_rf.joblib exists): scikit-learn
RandomForestClassifier loaded with joblib, predict_proba over [T, RH, lux].

Guardrail: this module never claims detection. It returns predicted
suitability scores only.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from app import MODEL_DIR, load_config

CURVES_PATH = MODEL_DIR / "species_curves.json"
RF_PATH = MODEL_DIR / "species_rf.joblib"
# Optional East Java overlay: loaded after the base table and merged per species.
CURVES_INDONESIA_PATH = MODEL_DIR / "species_curves_indonesia.json"
REGION = "east_java"

log = logging.getLogger("fungi.species")

_lock = threading.Lock()
_curves_cache: tuple[tuple[float, float | None, str], list[dict]] | None = None
_model_cache: tuple[tuple[str, float], Any] | None = None
_meta_cache: tuple[tuple[str, float], dict] | None = None

# Last-resort curves if the JSON is missing/corrupt. Kept in sync with the
# shipped seed table so the API can always return a ranked top-5.
_FALLBACK_CURVES = [
    {"name": "Aspergillus versicolor", "T_min": 4, "T_opt": 25, "T_max": 37,
     "RH_min": 65, "RH_opt": 85, "lux_pref": {"min": 0, "opt": 20, "max": 800}},
    {"name": "Stachybotrys chartarum", "T_min": 5, "T_opt": 23, "T_max": 35,
     "RH_min": 90, "RH_opt": 97, "lux_pref": {"min": 0, "opt": 0, "max": 300}},
    {"name": "Eurotium herbariorum", "T_min": 4, "T_opt": 27, "T_max": 37,
     "RH_min": 70, "RH_opt": 85, "lux_pref": {"min": 0, "opt": 10, "max": 600}},
    {"name": "Aspergillus niger", "T_min": 6, "T_opt": 33, "T_max": 47,
     "RH_min": 75, "RH_opt": 90, "lux_pref": {"min": 0, "opt": 50, "max": 1500}},
    {"name": "Aspergillus ochraceus group", "T_min": 8, "T_opt": 28, "T_max": 37,
     "RH_min": 75, "RH_opt": 90, "lux_pref": {"min": 0, "opt": 30, "max": 1000}},
    {"name": "Fusarium spp.", "T_min": 2, "T_opt": 24, "T_max": 32,
     "RH_min": 85, "RH_opt": 97, "lux_pref": {"min": 0, "opt": 100, "max": 2000}},
]


def region() -> str:
    """Configured region: `default` (global curves) or `east_java` (overlay)."""
    return str(load_config().get("species", {}).get("region") or "default").lower()


def _apply_overlay(base: list[dict], overlay: dict) -> list[dict]:
    """Merge the regional overlay into the base curves, per species name.

    Overlay entries may be partial (only the parameters a regional study
    pins down); everything else keeps the global value. Unknown species are
    appended so a regional-only record is not silently dropped.
    """
    species_overlay = (overlay or {}).get("species") or overlay or {}
    if not isinstance(species_overlay, dict):
        return base
    merged: list[dict] = []
    seen: set[str] = set()
    for curve in base:
        name = str(curve.get("name", ""))
        override = species_overlay.get(name)
        if isinstance(override, dict):
            merged.append({**curve, **override, "name": name})
            seen.add(name)
        else:
            merged.append(curve)
    for name, override in species_overlay.items():
        if name in seen or not isinstance(override, dict):
            continue
        entry = {"name": name, **override}
        if all(key in entry for key in ("T_min", "T_opt", "T_max", "RH_min", "RH_opt")):
            merged.append(entry)
    return merged


def _load_curves() -> list[dict]:
    global _curves_cache
    current_region = region()
    try:
        mtime = CURVES_PATH.stat().st_mtime
    except OSError:
        return _FALLBACK_CURVES
    overlay_mtime: float | None = None
    if current_region == REGION:
        try:
            overlay_mtime = CURVES_INDONESIA_PATH.stat().st_mtime
        except OSError:
            overlay_mtime = None
    cache_key = (mtime, overlay_mtime, current_region)
    if _curves_cache and _curves_cache[0] == cache_key:
        return _curves_cache[1]
    try:
        with CURVES_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        curves = data["species"]
        if not isinstance(curves, list) or not curves:
            raise ValueError("species list empty")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        curves = [dict(c) for c in _FALLBACK_CURVES]
    if current_region == REGION and overlay_mtime is not None:
        try:
            with CURVES_INDONESIA_PATH.open("r", encoding="utf-8") as fh:
                curves = _apply_overlay(curves, json.load(fh))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    _curves_cache = (cache_key, curves)
    return curves


def _model_path():
    """Regional model if configured, else the default joblib path."""
    species_cfg = load_config().get("species", {})
    configured = species_cfg.get("model_path")
    if configured:
        path = Path(str(configured))
        if not path.is_absolute():
            path = MODEL_DIR.parent.parent / path if str(configured).startswith("app/") else MODEL_DIR / path
        return path
    if region() == REGION:
        regional = MODEL_DIR / "species_rf_east_java.joblib"
        if regional.exists():
            return regional
    return RF_PATH


def _meta() -> dict:
    """Training metadata (feature ranges, CV scores) for confidence scoring."""
    global _meta_cache
    path = _model_path().with_suffix(".meta.json")
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _meta_cache and _meta_cache[0] == (str(path), mtime):
        return _meta_cache[1]
    try:
        with path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError):
        meta = {}
    _meta_cache = ((str(path), mtime), meta if isinstance(meta, dict) else {})
    return _meta_cache[1]


def _load_model() -> Any | None:
    """Load the RandomForest once; hot-reload if the joblib file changes."""
    global _model_cache
    path = _model_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _model_cache and _model_cache[0] == (str(path), mtime):
        return _model_cache[1]
    try:
        import joblib  # type: ignore

        model = joblib.load(path)
        _model_cache = ((str(path), mtime), model)
        return model
    except Exception:
        return None


def _triangular(x: float, low: float, opt: float, high: float) -> float:
    if x < low or x > high:
        return 0.0
    if x == opt:
        return 1.0
    if x < opt:
        return 1.0 if opt <= low else (x - low) / (opt - low)
    return 1.0 if high <= opt else (high - x) / (high - opt)


def _rh_membership(rh: float, rh_min: float, rh_opt: float) -> float:
    if rh <= rh_min:
        return 0.0
    if rh >= rh_opt:
        return 1.0
    if rh_opt <= rh_min:
        return 1.0
    return (rh - rh_min) / (rh_opt - rh_min)


def _lux_membership(lux: float | None, pref: Any) -> float:
    """Soft light preference, floored at 0.5.

    Light is growth-neutral in FGPI, the BH1750 saturates in daylight and the
    curve table's `lux_pref.max` values are indoor-scale. A hard triangular
    cutoff therefore zeroed every species that carries a preference as soon as
    a room got bright (5000 lx). Keeping the modifier in [0.5, 1.0] ranks
    light-loving species higher without erasing them.
    """
    if lux is None or lux < 0:
        return 1.0  # no usable light reading -> neutral
    if isinstance(pref, dict):
        return max(
            0.5,
            _triangular(
                float(lux),
                float(pref.get("min", 0.0)),
                float(pref.get("opt", 50.0)),
                float(pref.get("max", 1000.0)),
            ),
        )
    if isinstance(pref, (int, float)):
        opt = float(pref)
        return _triangular(float(lux), 0.0, opt, max(1000.0, opt * 10.0))
    return 1.0


def curve_names() -> list[str]:
    """Known species names from the active curve table (base + overlay)."""
    return [str(curve.get("name", "")) for curve in _load_curves() if curve.get("name")]


def _parametric(T: float, RH: float, L: float | None) -> list[dict]:
    ranked: list[dict] = []
    for curve in _load_curves():
        try:
            t_factor = _triangular(
                T, float(curve["T_min"]), float(curve["T_opt"]), float(curve["T_max"])
            )
            rh_factor = _rh_membership(
                RH, float(curve["RH_min"]), float(curve["RH_opt"])
            )
            l_factor = _lux_membership(L, curve.get("lux_pref"))
            score = max(0.0, min(1.0, t_factor * rh_factor * l_factor))
            # Inside the niche envelope both memberships are comfortably above
            # zero; near an edge the parametric score is a guess, so say so.
            on_edge = min(t_factor, rh_factor) < 0.15
            ranked.append(
                {
                    "species": str(curve["name"]),
                    "suitability": round(score, 4),
                    "confidence": "low" if on_edge else "medium",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    ranked.sort(key=lambda item: (-item["suitability"], item["species"]))
    return ranked


def _in_training_range(T: float, RH: float, L: float | None) -> bool | None:
    """True/False when the model ships feature ranges, None when unknown."""
    ranges = (_meta().get("feature_ranges") or {}) if _meta() else {}
    if not ranges:
        return None
    # Temperature and humidity decide; lux is growth-neutral and routinely
    # saturated, so it must not downgrade confidence on its own.
    checks = [("temperature", T), ("humidity", RH)]
    for name, value in checks:
        bounds = ranges.get(name)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        if not (low <= value <= high):
            return False
    return True


def _ml(T: float, RH: float, L: float | None) -> list[dict] | None:
    model = _load_model()
    if model is None:
        return None
    try:
        lux = float(L) if L is not None and L >= 0 else 0.0
        proba = model.predict_proba([[T, RH, lux]])[0]
        classes = list(getattr(model, "classes_", []))
        in_range = _in_training_range(T, RH, L)
        # An ML model asked about conditions it never saw reports high numbers
        # with no evidence behind them; confidence must not inherit that.
        confidence = "high" if in_range is not False else "low"
        ranked = [
            {"species": str(name), "suitability": round(float(score), 4), "confidence": confidence}
            for name, score in zip(classes, proba)
        ]
        ranked.sort(key=lambda item: (-item["suitability"], item["species"]))
        return ranked or None
    except Exception:
        return None


def _sgd_proba(T: float, RH: float, L: float | None) -> dict[str, float] | None:
    """Incremental adapter probabilities, or None when disabled/absent."""
    cfg = load_config().get("species", {})
    if not (cfg.get("sgd") or {}).get("enabled", True):
        return None
    try:
        from app.models import species_sgd  # local import: optional dependency path

        proba = species_sgd.predict_proba(T, RH, L)
    except Exception:
        return None
    return proba or None


def _blend_settings(cfg: dict) -> tuple[bool, float]:
    """(enabled, sgd_weight) from `species.blend`, falling back to the flat key."""
    block = cfg.get("blend") or {}
    enabled = bool(block.get("enabled", True)) and bool(sgd_block_enabled(cfg))
    weight = block.get("sgd_weight", cfg.get("blend_sgd_weight", 0.3))
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        weight = 0.3
    return enabled, max(0.0, min(1.0, weight))


def sgd_block_enabled(cfg: dict) -> bool:
    return bool((cfg.get("sgd") or {}).get("enabled", True))


def _blend(ranked: list[dict], sgd: dict[str, float], weight: float) -> list[dict]:
    """Weighted average of the SGD adapter and the RF/parametric posterior."""
    weight = max(0.0, min(1.0, float(weight)))
    log.debug(
        "blend rf_weight=%.2f sgd_weight=%.2f rf_top=%s sgd_top=%s",
        1.0 - weight,
        weight,
        ranked[0]["species"] if ranked else None,
        max(sgd, key=sgd.get) if sgd else None,
    )
    blended: list[dict] = []
    for entry in ranked:
        name = entry["species"]
        rf_score = float(entry.get("suitability") or 0.0)
        score = (1.0 - weight) * rf_score + weight * float(sgd.get(name, 0.0))
        blended.append({**entry, "suitability": round(max(0.0, min(1.0, score)), 4)})
    for name, score in sgd.items():
        if not any(e["species"] == name for e in blended):
            blended.append(
                {
                    "species": name,
                    "suitability": round(weight * float(score), 4),
                    "confidence": "medium",
                }
            )
    blended.sort(key=lambda item: (-item["suitability"], item["species"]))
    return blended


def predict_species(T: float, RH: float, L: float = -1.0) -> list[dict]:
    """Return [{'species', 'suitability', 'confidence'}, ...] sorted desc, top-5.

    Confidence is a property of the evidence, not of the number:
      high    - trained model, input inside its training feature range
      medium  - parametric curves, input comfortably inside the niche
      low     - input near/outside the envelope, or a model extrapolating
    """
    cfg = load_config().get("species", {})
    top_k = max(1, min(10, int(cfg.get("top_k", 5))))
    mode = str(cfg.get("mode", "auto")).lower()

    try:
        t, rh = float(T), float(RH)
    except (TypeError, ValueError):
        t, rh = 0.0, 0.0
    lux: float | None
    try:
        lux = float(L)
    except (TypeError, ValueError):
        lux = None

    with _lock:
        ranked: list[dict] | None = None
        if mode in {"auto", "ml"}:
            ranked = _ml(t, rh, lux)
        if ranked is None:
            ranked = _parametric(t, rh, lux)
        ranked_for_blend = ranked
        sgd = _sgd_proba(t, rh, lux)
        blend_enabled, blend_weight = _blend_settings(cfg)
        if ranked_for_blend and sgd and blend_enabled:
            ranked = _blend(ranked_for_blend, sgd, blend_weight)

    if not ranked:
        ranked = [
            {"species": c["name"], "suitability": 0.0, "confidence": "low"}
            for c in _load_curves()
        ]
    return ranked[:top_k]


def confidence_for(species_name: str, T: float, RH: float, L: float = -1.0) -> str:
    """Confidence for an already-stored prediction (dashboard backfill)."""
    ranked = predict_species(T, RH, L)
    for entry in ranked:
        if entry["species"] == species_name:
            return str(entry.get("confidence") or "low")
    return "low"


def backend_name() -> str:
    """Human-readable backend label for the dashboard."""
    mode = str(load_config().get("species", {}).get("mode", "auto")).lower()
    label = "parametric curves"
    if mode != "parametric" and _load_model() is not None:
        label = "RandomForest (joblib)"
    sgd_enabled = bool((load_config().get("species", {}).get("sgd") or {}).get("enabled", True))
    if sgd_enabled:
        try:
            from app.models import species_sgd

            if species_sgd.available():
                label += " + SGD adapter"
        except Exception:
            pass
    if region() == REGION:
        label += " [east_java]"
    return label
