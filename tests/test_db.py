"""Round-trips for the persistence helpers, on a throwaway database."""

from __future__ import annotations

import time


def test_insert_and_window_round_trip(fresh_db):
    now = int(time.time())
    reading_id = fresh_db.insert_reading(now, 1.0, 2.0, "node-a", 24.0, 88.0, 120.0)
    fgpi_id = fresh_db.insert_fgpi(reading_id, now, 100.0, "critical")
    fresh_db.insert_species_predictions(
        fgpi_id,
        [
            {"species": "Aspergillus versicolor", "suitability": 0.91},
            {"species": "Aspergillus niger", "suitability": 0.44},
        ],
    )

    window = fresh_db.latest_window(3600)
    assert len(window) == 1
    row = window[0]
    assert row["temperature"] == 24.0
    assert row["risk_class"] == "critical"
    assert row["node_id"] == "node-a"

    species = fresh_db.species_for_fgpi(fgpi_id)
    assert [s["rank"] for s in species] == [1, 2]

    snapshot = fresh_db.latest_snapshot()
    assert snapshot["reading_id"] == reading_id
    assert fresh_db.latest_fgpi_id() == fgpi_id


def test_labels_join_to_readings_and_counts(fresh_db):
    now = int(time.time())
    reading_id = fresh_db.insert_reading(now, None, None, "node-b", 25.0, 90.0, 50.0)
    fresh_db.insert_label(now, reading_id, "Aspergillus versicolor", True, 4, "wall", "tester")
    fresh_db.insert_label(now, reading_id, "Stachybotrys chartarum", False, 3, None, "tester")

    rows = fresh_db.labels_with_readings(0)
    assert len(rows) == 2
    assert rows[0]["temperature"] == 25.0
    counts = fresh_db.label_counts()
    assert counts == {"total": 2, "present": 1, "absent": 1}


def test_unknown_reading_is_rejected(fresh_db):
    import pytest

    with pytest.raises(ValueError):
        fresh_db.insert_label(int(time.time()), 9999, "Aspergillus niger", True, 3)


def test_prune_removes_children_and_keeps_recent(fresh_db):
    old = int(time.time()) - 100 * 86400
    recent = int(time.time())
    for ts in (old, recent):
        rid = fresh_db.insert_reading(ts, 1.0, 1.0, "n", 22.0, 70.0, 10.0)
        fid = fresh_db.insert_fgpi(rid, ts, 50.0, "high")
        fresh_db.insert_species_predictions(fid, [{"species": "Aspergillus niger", "suitability": 0.5}])

    result = fresh_db.prune(90)
    assert result["readings"] == 1
    assert result["fgpi"] == 1
    assert result["species_predictions"] == 1
    assert len(fresh_db.latest_window(365 * 86400)) == 1


def test_node_status_marks_staleness(fresh_db):
    now = int(time.time())
    fresh_db.insert_reading(now, 1.0, 1.0, "fresh", 22.0, 80.0, 10.0)
    fresh_db.insert_reading(now - 100, 2.0, 2.0, "stale", 22.0, 80.0, 10.0)
    rows = {row["node_id"]: row for row in fresh_db.node_status(30)}
    assert rows["fresh"]["status"] == "green"
    assert rows["stale"]["status"] in {"yellow", "red"}
    # Spec-facing names stay available alongside the older ones.
    assert rows["fresh"]["age_s"] == rows["fresh"]["age_seconds"]
    assert rows["fresh"]["t_avg"] == rows["fresh"]["temperature"]
    assert rows["fresh"]["last_seen"] == rows["fresh"]["last_ts"]
