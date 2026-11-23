#!/usr/bin/env python3
"""Weekly database maintenance: prune old samples, back up, and compact.

Run by ``fungi-maintenance.timer`` on the Pi, or by hand:

    python scripts/db_maintenance.py --prune-days 90 --backup /home/pi/backups
    python scripts/db_maintenance.py --vacuum-only

Pruning deletes readings older than the cutoff together with their FGPI and
species rows. Deletion is explicit rather than relying on ON DELETE CASCADE so
databases created before the cascade was added behave identically.

The backup uses SQLite's online backup API, so it is safe while the dashboard
and sampler are running; the resulting file is a consistent snapshot.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.db as db  # noqa: E402


def backup(destination: Path) -> Path:
    """Consistent online backup; returns the written path."""
    source = db.db_path()
    if destination.suffix == ".db":
        target = destination
    else:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"fungi-{time.strftime('%Y%m%d-%H%M%S')}.db"
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"Backup written: {target} ({size_mb:.2f} MB)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prune-days", type=int, default=0, help="delete samples older than N days")
    parser.add_argument("--backup", type=Path, default=None, help="file or directory")
    parser.add_argument("--vacuum", action="store_true", help="VACUUM after pruning")
    parser.add_argument("--vacuum-only", action="store_true")
    args = parser.parse_args()

    db.init_db()

    if args.prune_days > 0:
        before = db.db_path().stat().st_size / (1024 * 1024)
        result = db.prune(args.prune_days)
        after = db.db_path().stat().st_size / (1024 * 1024)
        print(
            f"Pruned rows older than {args.prune_days} days: "
            f"{result['readings']} readings, {result['fgpi']} fgpi, "
            f"{result['species_predictions']} species rows"
        )
        print(f"File size {before:.2f} MB -> {after:.2f} MB")
        if result["readings"] == 0:
            print("Nothing to prune; timer can stay weekly.")

    if args.backup:
        backup(args.backup)

    if args.vacuum or args.vacuum_only:
        started = time.perf_counter()
        db.vacuum()
        print(f"VACUUM finished in {time.perf_counter() - started:.1f}s")

    if not (args.prune_days or args.backup or args.vacuum or args.vacuum_only):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
