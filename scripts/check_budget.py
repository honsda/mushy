#!/usr/bin/env python3
"""Measure the Pi 4 resource budget and report against the targets.

Advisory by design: the numbers depend on the host, so this never fails the
build. Run it on the Pi for the real figures, or on a dev machine to catch
regressions in render time and payload size.

    python scripts/check_budget.py           # report, exit 0
    python scripts/check_budget.py --strict  # exit 1 when a target is missed
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Targets from the master plan section 7.
TARGETS = {
    "idle_rss_mb": 120.0,
    "inference_rss_mb": 180.0,
    "heatmap_seconds": 3.0,
    "api_latest_ms": 50.0,
    "dashboard_gzip_kb": 50.0,
    "static_disk_kb": 110.0,
}
ASSETS = ["dashboard.css", "htmx.min.js", "uplot.min.js", "uplot.min.css"]


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:  # Linux without psutil
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def measure() -> list[tuple[str, float | None, float, str]]:
    findings: list[tuple[str, float | None, float, str]] = []

    idle_rss = _rss_mb()
    findings.append(("idle RSS (this process)", idle_rss, TARGETS["idle_rss_mb"], "MB"))

    static_kb = sum(
        (ROOT / "static" / name).stat().st_size for name in ASSETS if (ROOT / "static" / name).is_file()
    ) / 1024.0
    findings.append(("static assets on disk", static_kb, TARGETS["static_disk_kb"], "KB"))

    # Heatmap + API timings use a throwaway DB and simulated sensors.
    tmp = Path(tempfile.mkdtemp(prefix="mushy-budget-"))
    os.environ["FUNGI_DB"] = str(tmp / "budget.db")
    os.environ["FUNGI_SIMULATE"] = "1"
    try:
        import app.db as db
        from app.models.fgpi import compute_fgpi
        from app.spatial.heatmap import render_layer_png
        from app.spatial.interpolate import grid_data
        from app.spatial.layers import get_layer

        db.init_db()
        points = [(1.0, 1.0, 88.0), (9.0, 7.0, 95.0), (5.0, 4.0, 60.0)]
        grid = grid_data(points, 10.0, 8.0, 100)
        started = time.perf_counter()
        render_layer_png(grid, str(tmp / "probe.png"), layer=get_layer("fgpi"))
        findings.append(
            ("heatmap render", time.perf_counter() - started, TARGETS["heatmap_seconds"], "s")
        )

        from app.server import create_app

        client = create_app().test_client()
        client.get("/api/latest")
        started = time.perf_counter()
        for _ in range(10):
            client.get("/api/latest")
        findings.append(
            ("/api/latest", (time.perf_counter() - started) * 100.0, TARGETS["api_latest_ms"], "ms")
        )

        page = client.get("/", headers={"Accept-Encoding": "gzip"})
        findings.append(
            ("dashboard (gzip)", len(page.data) / 1024.0, TARGETS["dashboard_gzip_kb"], "KB")
        )

        rss = _rss_mb()
        findings.append(("RSS after app import", rss, TARGETS["inference_rss_mb"], "MB"))
        compute_fgpi(24, 90, 100)
    finally:
        os.environ.pop("FUNGI_DB", None)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit 1 if a target is missed")
    args = parser.parse_args()

    missed = 0
    print(f"{'metric':<28}{'value':>10}{'target':>10}  status")
    for name, value, target, unit in measure():
        if value is None:
            print(f"{name:<28}{'n/a':>10}{target:>9.0f}{unit:>1}  skipped")
            continue
        ok = value <= target
        missed += 0 if ok else 1
        print(f"{name:<28}{value:>9.1f}{unit:<1}{target:>9.0f}{unit:<1}  {'ok' if ok else 'OVER'}")
    if missed:
        print(f"\n{missed} metric(s) over target. Advisory only unless --strict.")
    return 1 if (args.strict and missed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
