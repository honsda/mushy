"""SQLite persistence helpers.

Schema is fixed by the project spec (section 6). A connection is opened per
operation: SQLite is fast enough at 30 s sampling cadence, and per-call
connections keep Flask's threaded dev server and the sampler daemon from
sharing unsafe handles.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterable

from app import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  x REAL, y REAL,
  node_id TEXT,
  temperature REAL NOT NULL,
  humidity REAL NOT NULL,
  lux REAL
);

CREATE TABLE IF NOT EXISTS fgpi (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id INTEGER REFERENCES readings(id),
  ts INTEGER NOT NULL,
  fgpi REAL NOT NULL,
  risk_class TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS species_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fgpi_id INTEGER REFERENCES fgpi(id),
  species TEXT NOT NULL,
  suitability REAL NOT NULL,
  rank INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);
CREATE INDEX IF NOT EXISTS idx_fgpi_ts ON fgpi(ts);
CREATE INDEX IF NOT EXISTS idx_species_fgpi ON species_predictions(fgpi_id);
"""


@contextmanager
def connection():
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


def insert_reading(
    ts: int,
    x: float | None,
    y: float | None,
    node_id: str | None,
    T: float,
    RH: float,
    lux: float | None,
) -> int:
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO readings (ts, x, y, node_id, temperature, humidity, lux)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (int(ts), x, y, node_id, float(T), float(RH), lux),
        )
        return int(cur.lastrowid)


def insert_fgpi(reading_id: int, ts: int, fgpi: float, risk_class: str) -> int:
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO fgpi (reading_id, ts, fgpi, risk_class) VALUES (?, ?, ?, ?)",
            (int(reading_id), int(ts), float(fgpi), str(risk_class)),
        )
        return int(cur.lastrowid)


def insert_species_predictions(fgpi_id: int, preds: list[dict]) -> None:
    rows = [
        (int(fgpi_id), str(p["species"]), float(p["suitability"]), rank)
        for rank, p in enumerate(preds, start=1)
    ]
    with connection() as conn:
        conn.executemany(
            "INSERT INTO species_predictions (fgpi_id, species, suitability, rank)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )


def latest_window(seconds: int = 3600) -> list[dict]:
    """Readings joined with their FGPI score, oldest first."""
    cutoff = int(time.time()) - int(seconds)
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id AS reading_id, r.ts, r.x, r.y, r.node_id,
                   r.temperature, r.humidity, r.lux,
                   f.id AS fgpi_id, f.fgpi, f.risk_class
            FROM readings r
            JOIN fgpi f ON f.reading_id = r.id
            WHERE f.ts >= ?
            ORDER BY f.ts ASC, f.id ASC
            """,
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def latest_snapshot() -> dict[str, Any] | None:
    """Most recent reading + FGPI + ranked species in one payload."""
    with connection() as conn:
        row = conn.execute(
            """
            SELECT r.id AS reading_id, r.ts, r.x, r.y, r.node_id,
                   r.temperature, r.humidity, r.lux,
                   f.id AS fgpi_id, f.fgpi, f.risk_class
            FROM readings r
            JOIN fgpi f ON f.reading_id = r.id
            ORDER BY f.ts DESC, f.id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        snap = dict(row)
        species = conn.execute(
            "SELECT species, suitability FROM species_predictions"
            " WHERE fgpi_id = ? ORDER BY rank ASC",
            (snap["fgpi_id"],),
        ).fetchall()
    snap["species"] = [dict(s) for s in species]
    return snap


def species_for_fgpi(fgpi_id: int) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT species, suitability FROM species_predictions"
            " WHERE fgpi_id = ? ORDER BY rank ASC",
            (int(fgpi_id),),
        ).fetchall()
    return [dict(r) for r in rows]


def spatial_points(seconds: int = 21600) -> list[tuple[float, float, float]]:
    """(x, y, fgpi) points for interpolation; fixed nodes use configured x/y."""
    cutoff = int(time.time()) - int(seconds)
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT r.x, r.y, f.fgpi
            FROM readings r
            JOIN fgpi f ON f.reading_id = r.id
            WHERE f.ts >= ? AND r.x IS NOT NULL AND r.y IS NOT NULL
            ORDER BY f.ts ASC
            """,
            (cutoff,),
        ).fetchall()
    return [(float(r["x"]), float(r["y"]), float(r["fgpi"])) for r in rows]


def latest_fgpi_ts() -> int:
    with connection() as conn:
        row = conn.execute("SELECT MAX(ts) AS ts FROM fgpi").fetchone()
    return int(row["ts"]) if row and row["ts"] is not None else 0


def latest_fgpi_id() -> int:
    """Max row id; a monotonic change signal even for out-of-order timestamps."""
    with connection() as conn:
        row = conn.execute("SELECT MAX(id) AS id FROM fgpi").fetchone()
    return int(row["id"]) if row and row["id"] is not None else 0


def rows_for_series(seconds: int = 21600) -> Iterable[dict]:
    cutoff = int(time.time()) - int(seconds)
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT f.ts, f.fgpi, r.temperature, r.humidity, r.lux
            FROM fgpi f
            JOIN readings r ON r.id = f.reading_id
            WHERE f.ts >= ?
            ORDER BY f.ts ASC
            """,
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def table_names() -> list[str]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]
