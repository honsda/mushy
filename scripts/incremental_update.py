#!/usr/bin/env python3
"""Nightly incremental model update (Pi, 02:00, via fungi-incremental.timer).

Absorbs labels entered since the last run into the SGD adapter. Skipped unless
there are at least ``species.min_new_labels`` new labels, the Pi has free RAM,
and the SoC is cool enough — see app/models/species_sgd.py.

Logs one line per run to ``logs/incremental.log`` so a week of runs fits on a
phone screen.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "logs" / "incremental.log"


def _logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fungi.incremental")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def _rss_hint() -> str:
    try:
        import psutil  # type: ignore

        return f"{psutil.Process().memory_info().rss / (1024 * 1024):.0f} MB RSS"
    except Exception:
        return "RSS n/a"


def main() -> int:
    log = _logger()
    started = time.perf_counter()
    try:
        from app.models import species_sgd

        result = species_sgd.partial_update()
    except Exception as exc:  # never fail the timer; the adapter is optional
        log.exception("incremental update crashed: %s", exc)
        return 1

    elapsed = time.perf_counter() - started
    if result.get("ok"):
        log.info(
            "SGD adapter updated: %s labels, %s in %.1fs",
            result.get("batch_rows"),
            _rss_hint(),
            elapsed,
        )
        return 0
    log.info("skipped: %s (%.1fs)", result.get("reason"), elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
