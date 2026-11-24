#!/usr/bin/env python3
"""Nightly on-device update of the SGD adapter (Pi, 02:00, fungi-incremental.timer).

Phase 3 of the master scheme. The RandomForest stays frozen; only the small
linear adapter learns from labels an operator recorded by hand.

Guards (all must hold, otherwise the run is skipped and logged):
  * at least `species.sgd.min_new_labels` labels since the last run
  * at least `species.sgd.min_free_mb` MB RAM available
  * SoC temperature at or below `species.sgd.max_soc_temp_c`

One line per run goes to ``logs/sgd_last_run.txt`` (overwritten) and to
``logs/incremental.log`` (appended), so `cat logs/sgd_last_run.txt` answers
"did it learn last night?" in one line.
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
STATUS_PATH = ROOT / "logs" / "sgd_last_run.txt"


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


def _write_status(result: dict, elapsed: float) -> None:
    """One-line status file; timers have no console to read."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if result.get("ok"):
        line = (
            f"{stamp} updated: {result.get('batch_rows')} labels, "
            f"seen={result.get('labels_seen')}, {elapsed:.1f}s"
        )
    else:
        line = f"{stamp} skipped: {result.get('reason')}"
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".txt.tmp")
    tmp.write_text(line + "\n", encoding="utf-8")
    tmp.replace(STATUS_PATH)


def main() -> int:
    log = _logger()
    started = time.perf_counter()
    try:
        from app.models import species_sgd

        result = species_sgd.partial_update()
    except Exception as exc:  # never fail the timer; the adapter is optional
        log.exception("nightly update crashed: %s", exc)
        _write_status({"ok": False, "reason": f"crash: {exc}"}, time.perf_counter() - started)
        return 1

    elapsed = time.perf_counter() - started
    if result.get("ok"):
        log.info(
            "SGD adapter updated: %s labels, %s in %.1fs",
            result.get("batch_rows"),
            _rss_hint(),
            elapsed,
        )
    else:
        log.info("skipped: %s (%.1fs)", result.get("reason"), elapsed)
    _write_status(result, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())