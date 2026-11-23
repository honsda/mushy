#!/usr/bin/env python3
"""Create the SQLite database and the three spec tables.

Usage: python scripts/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db_path  # noqa: E402
from app.db import init_db, table_names  # noqa: E402


def main() -> int:
    init_db()
    print(f"Initialised database: {db_path()}")
    print("Tables:", ", ".join(table_names()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
