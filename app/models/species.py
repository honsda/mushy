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
import threading
from typing import Any

from app import MODEL_DIR, load_config

CURVES_PATH = MODEL_DIR / "species_curves.json"
RF_PATH = MODEL_DIR / "species_rf.joblib"

_lock = threading.Lock()
_curves_cache: tuple[float, list[dict]] | None = None
_model_cache: tuple[float, Any] | None = None

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


def _load_curves() -> list[dict]:
    global _curves_cache
    try:
        mtime = CURVES_PATH.stat().st_mtime
    except OSError:
        return _FALLBACK_CURVES
    if _curves_cache and _curves_cache[0] == mtime:
        return _curves_cache[1]
    try:
        with CURVES_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        curves = data["species"]
        if not isinstance(curves, list) or not curves:
            raise ValueError("species list empty")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        curves = _FALLBACK_CURVES
    _curves_cache = (mtime, curves)
    return curves


def _load_model() -> Any | None:
    """Load the RandomForest once; hot-reload if the joblib file changes."""
    global _model_cache
    try:
        mtime = RF_PATH.stat().st_mtime
    except OSError:
        return None
    if _model_cache and _model_cache[0] == mtime:
        return _model_cache[1]
    try:
        import joblib  # type: ignore

        model = joblib.load(RF_PATH)
        _model_cache = (mtime, model)
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
    if lux is None or lux < 0:
        return 1.0  # no usable light reading -> neutral
    if isinstance(pref, dict):
        return _triangular(
            float(lux),
            float(pref.get("min", 0.0)),
            float(pref.get("opt", 50.0)),
            float(pref.get("max", 1000.0)),
        )
    if isinstance(pref, (int, float)):
        opt = float(pref)
        return _triangular(float(lux), 0.0, opt, max(1000.0, opt * 10.0))
    return 1.0


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
            ranked.append({"species": str(curve["name"]), "suitability": round(score, 4)})
        except (KeyError, TypeError, ValueError):
            continue
    ranked.sort(key=lambda item: (-item["suitability"], item["species"]))
    return ranked


def _ml(T: float, RH: float, L: float | None) -> list[dict] | None:
    model = _load_model()
    if model is None:
        return None
    try:
        lux = float(L) if L is not None and L >= 0 else 0.0
        proba = model.predict_proba([[T, RH, lux]])[0]
        classes = list(getattr(model, "classes_", []))
        ranked = [
            {"species": str(name), "suitability": round(float(score), 4)}
            for name, score in zip(classes, proba)
        ]
        ranked.sort(key=lambda item: (-item["suitability"], item["species"]))
        return ranked or None
    except Exception:
        return None


def predict_species(T: float, RH: float, L: float = -1.0) -> list[dict]:
    """Return [{'species': str, 'suitability': float}, ...] sorted desc, top-5."""
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

    if not ranked:
        ranked = [{"species": c["name"], "suitability": 0.0} for c in _load_curves()]
    return ranked[:top_k]


def backend_name() -> str:
    """Human-readable backend label for the dashboard."""
    mode = str(load_config().get("species", {}).get("mode", "auto")).lower()
    if mode != "parametric" and _load_model() is not None:
        return "RandomForest (joblib)"
    return "parametric curves"
