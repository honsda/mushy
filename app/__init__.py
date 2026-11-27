"""Fungal Growth Forecast — application package.

Configuration helpers live here so every module can share one resolved,
cached view of config.yaml without introducing a second config layer.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
MODEL_DIR = BASE_DIR / "app" / "models" / "trained"

_DEFAULTS = {
    "sensors": {
        "dht22_pin": 4,
        "light_driver": "bh1750",
        "i2c_bus": 1,
        "simulate": False,
    },
    "sampling": {"interval_seconds": 30},
    "fgpi": {"humidity_curve": "smooth", "humidity_range": [40.0, 90.0]},
    "spatial": {
        "x_max": 10.0,
        "y_max": 8.0,
        "resolution": 100,
        "method": "idw",
        "fixed_node_position": [5.0, 4.0],
        "window_hours": 6,
    },
    "server": {"host": "0.0.0.0", "port": 5000},
    "species": {"mode": "auto", "top_k": 5},
    "database": {"path": "fungi.db"},
}

_config_cache: dict[str, tuple[float, dict]] = {}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml, merged over sane defaults. Cached by mtime."""
    cfg_path = Path(path or os.environ.get("FUNGI_CONFIG", BASE_DIR / "config.yaml"))
    cfg_path = cfg_path.expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (BASE_DIR / cfg_path).resolve()

    try:
        mtime = cfg_path.stat().st_mtime
    except OSError:
        mtime = 0.0

    cache_key = str(cfg_path)
    cached = _config_cache.get(cache_key)
    if cached and cached[0] == mtime:
        cfg = copy.deepcopy(cached[1])
    else:
        data: dict = {}
        if cfg_path.exists():
            import yaml  # imported lazily: not needed just to boot

            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = _merge(_DEFAULTS, data)
        _config_cache[cache_key] = (mtime, copy.deepcopy(cfg))

    # Environment overrides make deployment and bench testing predictable.
    if os.environ.get("FUNGI_DB"):
        cfg.setdefault("database", {})["path"] = os.environ["FUNGI_DB"]
    if os.environ.get("FUNGI_SIMULATE", "").lower() in {"1", "true", "yes", "on"}:
        cfg.setdefault("sensors", {})["simulate"] = True
    if os.environ.get("FUNGI_HOST"):
        cfg.setdefault("server", {})["host"] = os.environ["FUNGI_HOST"]
    if os.environ.get("FUNGI_PORT"):
        try:
            cfg.setdefault("server", {})["port"] = int(os.environ["FUNGI_PORT"])
        except ValueError:
            pass

    return cfg


def simulation_enabled(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.get("sensors", {}).get("simulate", False))


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    raw = os.environ.get("FUNGI_DB") or cfg.get("database", {}).get("path", "fungi.db")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
