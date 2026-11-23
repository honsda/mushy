"""Incremental SGD adapter: learns from labels entered on the device.

Why a second model at all
------------------------
The RandomForest is the accuracy model, but retraining it needs a dev machine.
A `SGDClassifier(loss='log_loss')` with `partial_fit` is ~100 KB, fits in a few
MB of RAM, and can absorb last week's labels on the Pi itself at 02:00. It is
blended with the RF posterior (default 30/70) rather than replacing it, so a
handful of labels nudges rankings without discarding the trained model.

Guards
------
* Aborts unless at least `species.min_new_labels` labels arrived since the last
  run, so one label never reshapes the model.
* Checks free RAM and SoC temperature before training on the Pi (both checks
  are skipped when the tooling is absent, e.g. on the dev machine).
* Never raises into the server: `available()`/`predict_proba()` degrade to
  None and the caller falls back to the RF or the parametric curves.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from app import MODEL_DIR, load_config

log = logging.getLogger("fungi.sgd")

SGD_PATH = MODEL_DIR / "species_sgd.joblib"
STATE_PATH = MODEL_DIR / "species_sgd.state.json"
_lock = threading.Lock()


def available() -> bool:
    return SGD_PATH.is_file()


def _state() -> dict:
    try:
        with STATE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + chr(10), encoding="utf-8")
    tmp.replace(STATE_PATH)


def load():
    """Return the fitted estimator, or None."""
    if not SGD_PATH.is_file():
        return None
    try:
        import joblib  # type: ignore

        return joblib.load(SGD_PATH)
    except Exception:
        return None


def predict_proba(T: float, RH: float, L: float = -1.0) -> dict[str, float] | None:
    """Species -> probability from the adapter, or None when unavailable."""
    model = load()
    if model is None:
        return None
    try:
        lux = float(L) if L is not None and float(L) >= 0 else 0.0
        proba = model.predict_proba([[float(T), float(RH), lux]])[0]
        classes = list(getattr(model, "classes_", []))
        if not classes:
            return None
        return {str(name): float(score) for name, score in zip(classes, proba)}
    except Exception:
        return None


def is_fresh(max_age_hours: float = 72.0) -> bool:
    """Stale adapters are ignored so an abandoned model cannot dominate."""
    try:
        age = time.time() - SGD_PATH.stat().st_mtime
    except OSError:
        return False
    return age <= float(max_age_hours) * 3600.0


def _all_species() -> list[str]:
    from app.models.species import _load_curves  # one source of truth

    return [str(c["name"]) for c in _load_curves()]


def resource_check(cfg: dict | None = None) -> tuple[bool, str]:
    """Cheap pre-flight for on-Pi training. Missing tooling is not a failure."""
    cfg = cfg or load_config()
    sgd_cfg = (cfg.get("species") or {}).get("sgd") or {}
    min_free_mb = float(sgd_cfg.get("min_free_mb", 300))
    max_temp_c = float(sgd_cfg.get("max_soc_temp_c", 70))

    try:
        import psutil  # type: ignore

        free_mb = psutil.virtual_memory().available / (1024 * 1024)
        if free_mb < min_free_mb:
            return False, f"only {free_mb:.0f} MB RAM free (need {min_free_mb:.0f})"
    except Exception:
        pass  # psutil not installed: skip the RAM gate

    if Path("/usr/bin/vcgencmd").exists() or Path("/opt/vc/bin/vcgencmd").exists():
        try:
            raw = subprocess.run(
                ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=5
            ).stdout
            temp_c = float(raw.split("=")[1].split("'")[0])
            if temp_c >= max_temp_c:
                return False, f"SoC at {temp_c:.0f} C (limit {max_temp_c:.0f})"
        except Exception:
            pass
    return True, "ok"


def _training_rows(labels: list[dict]) -> tuple[list[list[float]], list[str]]:
    """Supervised rows from labels that still have their sensor context."""
    X: list[list[float]] = []
    y: list[str] = []
    for row in labels:
        if not row.get("observed"):
            continue
        t, rh = row.get("temperature"), row.get("humidity")
        if t is None or rh is None:
            continue
        lux = row.get("lux")
        X.append([float(t), float(rh), float(lux) if lux is not None and lux >= 0 else 0.0])
        y.append(str(row["species"]))
    return X, y


def partial_update(label_rows: list[dict] | None = None, cfg: dict | None = None) -> dict:
    """Fit the adapter on new labels; returns a small status dict.

    Callers pass rows from `db.labels_with_readings()` (already joined to the
    sensor reading). With fewer than `min_new_labels` unseen rows the run is
    skipped, which is the common case on a quiet week.
    """
    cfg = cfg or load_config()
    sgd_cfg = (cfg.get("species") or {}).get("sgd") or {}
    min_new = int(sgd_cfg.get("min_new_labels", 20))

    if label_rows is None:
        import app.db as db

        label_rows = db.labels_with_readings(0)

    state = _state()
    seen = int(state.get("labels_seen", 0))
    new_rows = label_rows[seen:]
    if len(new_rows) < min_new:
        return {
            "ok": False,
            "reason": f"only {len(new_rows)} new labels (need {min_new})",
            "labels_seen": seen,
        }

    ok, detail = resource_check(cfg)
    if not ok:
        return {"ok": False, "reason": f"resource guard: {detail}"}

    X, y = _training_rows(new_rows)
    if len(set(y)) < 2:
        return {"ok": False, "reason": "batch needs at least two species"}

    try:
        import joblib  # type: ignore
        from sklearn.linear_model import SGDClassifier
    except ImportError:
        return {"ok": False, "reason": "scikit-learn/joblib not installed"}

    classes = _all_species()
    with _lock:
        model = load()
        if model is None:
            model = SGDClassifier(
                loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42
            )
            model.partial_fit(X, y, classes=classes)
        else:
            model.partial_fit(X, y)
        tmp = SGD_PATH.with_suffix(".joblib.tmp")
        joblib.dump(model, tmp, compress=3)
        os.replace(tmp, SGD_PATH)

    _save_state(
        {
            "labels_seen": len(label_rows),
            "updated_at": int(time.time()),
            "batch_rows": len(X),
            "classes": classes,
        }
    )
    log.info("SGD adapter updated with %d labels", len(X))
    return {"ok": True, "batch_rows": len(X), "labels_seen": len(label_rows)}
