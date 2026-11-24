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

-- Ground truth entered by a human. Never populated by a camera or a model:
-- the Pi has no camera and species here are typed by an operator.
CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  reading_id INTEGER REFERENCES readings(id),
  species TEXT NOT NULL,
  observed INTEGER NOT NULL,          -- 1 = present, 0 = confirmed absent
  confidence INTEGER,                 -- 1..5 operator confidence
  note TEXT,
  operator TEXT,
  image_path TEXT                     -- optional external reference, not captured here
);

-- One row per attempted alert so a critical zone notifies once per debounce
-- window even across restarts.
CREATE TABLE IF NOT EXISTS alerts_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  zone TEXT NOT NULL,
  fgpi REAL NOT NULL,
  backend TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);
CREATE INDEX IF NOT EXISTS idx_fgpi_ts ON fgpi(ts);
CREATE INDEX IF NOT EXISTS idx_species_fgpi ON species_predictions(fgpi_id);
CREATE INDEX IF NOT EXISTS idx_labels_reading ON labels(reading_id);
CREATE INDEX IF NOT EXISTS idx_labels_ts ON labels(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_zone ON alerts_log(zone, ts);
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
            "SELECT species, suitability, rank FROM species_predictions"
            " WHERE fgpi_id = ? ORDER BY rank ASC",
            (snap["fgpi_id"],),
        ).fetchall()
    snap["species"] = [dict(s) for s in species]
    return snap


def species_for_fgpi(fgpi_id: int) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT species, suitability, rank FROM species_predictions"
            " WHERE fgpi_id = ? ORDER BY rank ASC",
            (int(fgpi_id),),
        ).fetchall()
    return [dict(r) for r in rows]


def spatial_points(seconds: int = 21600, field: str = "fgpi") -> list[tuple[float, float, float]]:
    """(x, y, value) points for interpolation; fixed nodes use configured x/y.

    field is whitelisted to a column of the joined view so the layer dropdown
    cannot become arbitrary SQL.
    """
    column = {
        "fgpi": "f.fgpi",
        "temperature": "r.temperature",
        "humidity": "r.humidity",
        "lux": "r.lux",
    }.get(field, "f.fgpi")
    cutoff = int(time.time()) - int(seconds)
    with connection() as conn:
        rows = conn.execute(
            f"""
            SELECT r.x, r.y, {column} AS value
            FROM readings r
            JOIN fgpi f ON f.reading_id = r.id
            WHERE f.ts >= ? AND r.x IS NOT NULL AND r.y IS NOT NULL
              AND {column} IS NOT NULL
            ORDER BY f.ts ASC
            """,
            (cutoff,),
        ).fetchall()
    points: list[tuple[float, float, float]] = []
    for row in rows:
        value = float(row["value"])
        # read_lux() reports -1.0 when the sensor is missing; treat as absent.
        if field == "lux" and value < 0:
            continue
        points.append((float(row["x"]), float(row["y"]), value))
    return points


def latest_reading_values() -> dict[str, float | None] | None:
    """Newest reading's raw sensor columns, for single-point map fallbacks."""
    with connection() as conn:
        row = conn.execute(
            "SELECT x, y, temperature, humidity, lux FROM readings"
            " ORDER BY ts DESC, id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def sample_locations(seconds: int = 21600) -> list[dict]:
    """Distinct sensor sites with their newest FGPI, risk and top species."""
    cutoff = int(time.time()) - int(seconds)
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT r.x, r.y, r.node_id, f.fgpi, f.risk_class, s.species
            FROM fgpi f
            JOIN readings r ON r.id = f.reading_id
            LEFT JOIN species_predictions s ON s.fgpi_id = f.id AND s.rank = 1
            WHERE f.ts >= ?
              AND r.x IS NOT NULL AND r.y IS NOT NULL
              AND f.id = (
                SELECT f2.id FROM fgpi f2 JOIN readings r2 ON r2.id = f2.reading_id
                WHERE r2.x = r.x AND r2.y = r.y
                ORDER BY f2.ts DESC, f2.id DESC LIMIT 1
              )
            ORDER BY f.ts DESC
            LIMIT 32
            """,
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


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


# --------------------------------------------------------------- labels
def insert_label(
    ts: int,
    reading_id: int | None,
    species: str,
    observed: bool,
    confidence: int | None = None,
    note: str | None = None,
    operator: str | None = None,
    image_path: str | None = None,
) -> int:
    """Record one human-observed label. Raises ValueError for unknown readings."""
    with connection() as conn:
        if reading_id is not None:
            exists = conn.execute(
                "SELECT 1 FROM readings WHERE id = ?", (int(reading_id),)
            ).fetchone()
            if exists is None:
                raise ValueError(f"reading_id {reading_id} does not exist")
        cur = conn.execute(
            "INSERT INTO labels (ts, reading_id, species, observed, confidence, note,"
            " operator, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(ts),
                int(reading_id) if reading_id is not None else None,
                str(species),
                1 if observed else 0,
                int(confidence) if confidence is not None else None,
                note,
                operator,
                image_path,
            ),
        )
        return int(cur.lastrowid)


def labels_with_readings(since_ts: int = 0) -> list[dict]:
    """Labels left-joined to the sensor context that was current when taken."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT l.id, l.ts, l.species, l.observed, l.confidence, l.note,
                   l.operator, l.image_path,
                   r.id AS reading_id, r.temperature, r.humidity, r.lux,
                   r.x, r.y, r.node_id
            FROM labels l
            LEFT JOIN readings r ON r.id = l.reading_id
            WHERE l.ts >= ?
            ORDER BY l.ts ASC, l.id ASC
            """,
            (int(since_ts),),
        ).fetchall()
    return [dict(row) for row in rows]


def label_counts() -> dict[str, int]:
    with connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN observed = 1 THEN 1 ELSE 0 END) AS present"
            " FROM labels"
        ).fetchone()
    total = int(row["total"] or 0)
    present = int(row["present"] or 0)
    return {"total": total, "present": present, "absent": total - present}


# --------------------------------------------------------------- nodes
def node_status(interval_seconds: int = 30) -> list[dict]:
    """Per-node freshness for the multi-node panel."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT r.node_id,
                   MAX(r.ts) AS last_ts,
                   COUNT(*) AS samples,
                   MAX(r.id) AS last_id
            FROM readings r
            WHERE r.ts >= ?
            GROUP BY r.node_id
            ORDER BY last_ts DESC
            """,
            (int(time.time()) - 86400,),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            latest = conn.execute(
                """
                SELECT r.node_id, r.temperature, r.humidity, r.lux, r.x, r.y,
                       f.fgpi, f.risk_class
                FROM readings r
                LEFT JOIN fgpi f ON f.reading_id = r.id
                WHERE r.id = ?
                """,
                (row["last_id"],),
            ).fetchone()
            if latest is None:
                continue
            age = max(0, int(time.time()) - int(row["last_ts"]))
            if age < 2 * interval_seconds:
                status = "green"
            elif age < 10 * interval_seconds:
                status = "yellow"
            else:
                status = "red"
            out.append(
                {
                    "node_id": latest["node_id"] or "unknown",
                    # Spec field names (green/yellow/red, last_seen/age_s,
                    # t_avg/rh_avg) plus the older aliases the dashboard and
                    # existing clients already read.
                    "last_seen": int(row["last_ts"]),
                    "age_s": age,
                    "t_avg": latest["temperature"],
                    "rh_avg": latest["humidity"],
                    "last_ts": int(row["last_ts"]),
                    "age_seconds": age,
                    "samples": int(row["samples"]),
                    "temperature": latest["temperature"],
                    "humidity": latest["humidity"],
                    "lux": latest["lux"],
                    "x": latest["x"],
                    "y": latest["y"],
                    "fgpi": latest["fgpi"],
                    "risk_class": latest["risk_class"],
                    "status": status,
                }
            )
    return out


# --------------------------------------------------------------- alerts
def record_alert(ts: int, zone: str, fgpi: float, backend: str, status: str) -> int:
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO alerts_log (ts, zone, fgpi, backend, status)"
            " VALUES (?, ?, ?, ?, ?)",
            (int(ts), str(zone), float(fgpi), str(backend), str(status)),
        )
        return int(cur.lastrowid)


def last_alert_ts(zone: str) -> int:
    with connection() as conn:
        row = conn.execute(
            "SELECT MAX(ts) AS ts FROM alerts_log WHERE zone = ?", (str(zone),)
        ).fetchone()
    return int(row["ts"]) if row and row["ts"] is not None else 0


def alerts_since(seconds: int = 86400) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, ts, zone, fgpi, backend, status FROM alerts_log"
            " WHERE ts >= ? ORDER BY ts DESC",
            (int(time.time()) - int(seconds),),
        ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------- maintenance
def prune(days: int = 90) -> dict[str, int]:
    """Delete readings older than `days` plus their FGPI/species children.

    Children are removed explicitly so the query works on databases created
    before the foreign keys gained ON DELETE CASCADE.
    """
    cutoff = int(time.time()) - int(days) * 86400
    with connection() as conn:
        fgpi_ids = [
            r["id"]
            for r in conn.execute("SELECT id FROM fgpi WHERE ts < ?", (cutoff,)).fetchall()
        ]
        species_deleted = 0
        if fgpi_ids:
            for chunk_start in range(0, len(fgpi_ids), 500):
                chunk = fgpi_ids[chunk_start : chunk_start + 500]
                placeholders = ",".join("?" * len(chunk))
                species_deleted += conn.execute(
                    f"DELETE FROM species_predictions WHERE fgpi_id IN ({placeholders})",
                    chunk,
                ).rowcount
        fgpi_deleted = conn.execute("DELETE FROM fgpi WHERE ts < ?", (cutoff,)).rowcount
        readings_deleted = conn.execute(
            "DELETE FROM readings WHERE ts < ?", (cutoff,)
        ).rowcount
    # Outside the transaction: a checkpoint inside it fails with "table is
    # locked" on the very database we just wrote to.
    try:
        with connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        pass  # a reader is active; the next run will truncate
    return {
        "cutoff": cutoff,
        "readings": int(readings_deleted),
        "fgpi": int(fgpi_deleted),
        "species_predictions": int(species_deleted),
    }


def vacuum() -> None:
    with connection() as conn:
        conn.execute("VACUUM")


def export_rows(table: str, from_ts: int, to_ts: int) -> Iterable[dict]:
    """Stream rows for CSV export. `table` is whitelisted, never interpolated."""
    queries = {
        "readings": "SELECT * FROM readings WHERE ts BETWEEN ? AND ? ORDER BY ts ASC",
        "fgpi": (
            "SELECT f.id, f.ts, f.fgpi, f.risk_class, r.temperature, r.humidity, r.lux,"
            " r.x, r.y, r.node_id FROM fgpi f LEFT JOIN readings r ON r.id = f.reading_id"
            " WHERE f.ts BETWEEN ? AND ? ORDER BY f.ts ASC"
        ),
        "labels": (
            "SELECT id, ts, reading_id, species, observed, confidence, note, operator,"
            " image_path FROM labels WHERE ts BETWEEN ? AND ? ORDER BY ts ASC"
        ),
        "alerts": "SELECT * FROM alerts_log WHERE ts BETWEEN ? AND ? ORDER BY ts ASC",
    }
    query = queries.get(table)
    if query is None:
        raise ValueError(f"unknown table '{table}'")
    with connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, (int(from_ts), int(to_ts)))
        for row in cursor:
            yield dict(row)
