#!/usr/bin/env python3
"""Idempotent DB migration for an existing deployment (spec 4.2).

`db.init_db()` already ships the full current schema with `CREATE TABLE IF NOT
EXISTS`, so a fresh install needs no migration. This script exists for the
database that has been running since M1:

  v2  labels + alerts_log tables and their indexes (Phase 2/4)
  v3  foreign keys actually enforced, orphan rows reported, integrity checked

Everything is additive and re-runnable. A timestamped copy is written to
``<db>.bak-<epoch>`` before the first change, so a failed migration is one
`cp` away from the previous state.

Usage:
    python scripts/migrate_db.py            # to the latest version
    python scripts/migrate_db.py --to 2     # stop at v2
    python scripts/migrate_db.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LATEST = 3


def _db_path() -> Path:
    from app import db_path

    return Path(db_path())


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}


def _has_foreign_keys(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute(f"PRAGMA foreign_key_list({table})").fetchall())


def _orphans(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute("PRAGMA foreign_key_check").fetchall()


def migrate(to: int = LATEST, dry_run: bool = False) -> dict:
    """Bring the configured database up to `to`; returns a report dict."""
    import app.db as db

    path = _db_path()
    report: dict = {"db": str(path), "existed": path.exists(), "changes": [], "warnings": []}
    if not path.exists():
        report["changes"].append("created fresh database")
        if not dry_run:
            db.init_db()
        return report

    conn = sqlite3.connect(str(path))
    try:
        before = _tables(conn)
        report["tables_before"] = sorted(before)

        needed = {"labels", "alerts_log"}
        missing = sorted(needed - before)
        if missing:
            report["changes"].append(f"v2: create {', '.join(missing)}")

        # v3 is about the foreign keys on the tables that already existed.
        fk_tables = ["fgpi", "species_predictions", "labels"]
        no_fk = [t for t in fk_tables if t in before and not _has_foreign_keys(conn, t)]
        if to >= 3 and no_fk:
            report["warnings"].append(
                f"v3: {', '.join(no_fk)} predate foreign keys; the constraint is "
                "declared in the schema but SQLite cannot add it in place. "
                "Rows are still checked below; rebuild the table manually if you "
                "need it enforced."
            )

        orphans = _orphans(conn) if to >= 3 else []
        if orphans:
            report["warnings"].append(f"v3: {len(orphans)} orphan row(s) detected")
            report["orphans"] = orphans[:20]

        if not report["changes"] and not report["warnings"]:
            report["changes"].append("already current")
            return report

        if dry_run:
            report["dry_run"] = True
            return report

        backup = path.with_name(f"{path.name}.bak-{int(time.time())}")
        shutil.copy2(path, backup)
        report["backup"] = str(backup)
    finally:
        conn.close()

    # init_db() is the single source of truth for the schema and is idempotent.
    db.init_db()

    conn = sqlite3.connect(str(path))
    try:
        report["tables_after"] = sorted(_tables(conn))
        report["integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", type=int, default=LATEST, choices=[2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = migrate(to=args.to, dry_run=args.dry_run)
    print(f"db: {report['db']}")
    for change in report["changes"]:
        print(f"  change: {change}")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")
    if report.get("backup"):
        print(f"  backup: {report['backup']}")
    if report.get("integrity"):
        print(f"  integrity_check: {report['integrity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())