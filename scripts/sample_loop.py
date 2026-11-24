#!/usr/bin/env python3
"""30-second sampling daemon: sensors -> FGPI -> species -> SQLite.

Run on the Pi as the fungi-sampler.service unit, or manually:

    python scripts/sample_loop.py                 # hardware mode
    FUNGI_SIMULATE=1 python scripts/sample_loop.py --once   # bench check
    python scripts/sample_loop.py --mobile        # random-walk mobile node

A sensor failure is logged and skipped; the loop never dies just because the
DHT22 missed a beat. The next cycle retries.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import load_config  # noqa: E402
from app.db import (  # noqa: E402
    init_db,
    insert_fgpi,
    insert_reading,
    insert_species_predictions,
)
from app.models.fgpi import compute_fgpi  # noqa: E402
from app.models.species import predict_species  # noqa: E402
from app.sensors import apply_calibration  # noqa: E402
from app.sensors.dht22 import SensorError, read_temperature_humidity  # noqa: E402
from app.sensors.light import read_lux  # noqa: E402

log = logging.getLogger("fungi.sampler")


def _initial_position(cfg: dict) -> tuple[float, float] | None:
    raw = cfg.get("spatial", {}).get("fixed_node_position")
    if not raw:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError):
        return None


def _walk(position: tuple[float, float], cfg: dict, rng: random.Random) -> tuple[float, float]:
    spatial = cfg.get("spatial", {})
    x_max = float(spatial.get("x_max", 10.0))
    y_max = float(spatial.get("y_max", 8.0))
    x = min(x_max, max(0.0, position[0] + rng.uniform(-0.6, 0.6)))
    y = min(y_max, max(0.0, position[1] + rng.uniform(-0.6, 0.6)))
    return x, y


def sample_once(cfg: dict, node_id: str, position: tuple[float, float] | None) -> dict:
    reading = read_temperature_humidity()
    lux = read_lux()
    ts = int(time.time())
    temperature, humidity, lux = apply_calibration(
        cfg, reading["temperature"], reading["humidity"], lux
    )

    score, risk = compute_fgpi(temperature, humidity, lux)
    species = predict_species(temperature, humidity, lux)

    x, y = (position if position is not None else (None, None))
    reading_id = insert_reading(ts, x, y, node_id, temperature, humidity, lux)
    fgpi_id = insert_fgpi(reading_id, ts, score, risk)
    insert_species_predictions(fgpi_id, species)

    # Alert from the process that measured it, so a restarted web server cannot
    # swallow a critical event. maybe_alert() never raises.
    try:
        from app.alerts import maybe_alert

        maybe_alert(
            {
                "fgpi": score,
                "risk_class": risk,
                "temperature": temperature,
                "humidity": humidity,
                "node_id": node_id,
                "ts": ts,
            },
            cfg,
        )
    except Exception:
        log.exception("alert dispatch failed")

    return {
        "ts": ts,
        "reading_id": reading_id,
        "fgpi_id": fgpi_id,
        "temperature": temperature,
        "humidity": humidity,
        "lux": lux,
        "fgpi": score,
        "risk_class": risk,
        "top_species": species[0]["species"] if species else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="take one sample and exit")
    parser.add_argument("--simulate", action="store_true", help="use synthetic sensor values")
    parser.add_argument("--mobile", action="store_true", help="random-walk spatial coordinates")
    parser.add_argument("--node-id", default=os.environ.get("FUNGI_NODE_ID", "pi-01"))
    parser.add_argument("--interval", type=float, default=None, help="override config interval")
    args = parser.parse_args()

    if args.simulate:
        os.environ["FUNGI_SIMULATE"] = "1"

    cfg = load_config()
    interval = float(
        args.interval
        if args.interval is not None
        else cfg.get("sampling", {}).get("interval_seconds", 30)
    )
    interval = max(1.0, interval)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    init_db()

    position = _initial_position(cfg)
    rng = random.Random()
    stop = threading.Event()

    def _stop(_signum, _frame):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):
            pass

    failures = 0
    while not stop.is_set():
        try:
            result = sample_once(cfg, args.node_id, position)
            log.info(
                "T=%.1fC RH=%.1f%% lux=%s FGPI=%.1f (%s) top=%s",
                result["temperature"],
                result["humidity"],
                f"{result['lux']:.1f}" if result["lux"] is not None and result["lux"] >= 0 else "n/a",
                result["fgpi"],
                result["risk_class"],
                result["top_species"],
            )
            failures = 0
            if args.mobile and position is not None:
                position = _walk(position, cfg, rng)
        except SensorError as exc:
            failures += 1
            log.warning("sensor failure (%d): %s", failures, exc)
        except Exception:
            failures += 1
            log.exception("sample cycle failed (%d)", failures)

        if args.once:
            return 0 if failures == 0 else 2
        stop.wait(interval)

    log.info("sampler stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
