"""HTTP surface: auth, validation, rate limit, labels, nodes, export."""

from __future__ import annotations

import time

import pytest

AUTH = {"Authorization": "Bearer test-token"}


def _latest_reading_id() -> int:
    import app.db as db

    snapshot = db.latest_snapshot()
    assert snapshot, "no reading in the test database"
    return int(snapshot["reading_id"])


def test_latest_shape(app_client):
    body = app_client.get("/api/latest").get_json()
    for key in ("ts", "fgpi", "risk_class", "species"):
        assert key in body
    assert all("confidence" in entry for entry in body["species"])
    assert all(entry["confidence"] in {"low", "medium", "high"} for entry in body["species"])


def test_ingest_requires_token_when_configured(app_client):
    payload = {"temperature": 24.0, "humidity": 85.0, "node_id": "auth-test"}
    assert app_client.post("/api/ingest", json=payload).status_code == 401
    wrong = app_client.post(
        "/api/ingest", json=payload, headers={"Authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401


def test_ingest_validates_payload(app_client):
    assert (
        app_client.post(
            "/api/ingest", json={"temperature": 200, "humidity": 50}, headers=AUTH
        ).status_code
        == 400
    )
    assert (
        app_client.post(
            "/api/ingest",
            json={"temperature": 22, "humidity": 70, "node_id": "bad node!"},
            headers=AUTH,
        ).status_code
        == 400
    )
    assert (
        app_client.post(
            "/api/ingest",
            json={"temperature": 22, "humidity": 70, "lux": 10**9, "node_id": "lux-bad"},
            headers=AUTH,
        ).status_code
        == 400
    )


def test_ingest_accepts_then_rate_limits_per_node(app_client):
    payload = {"temperature": 24.0, "humidity": 85.0, "node_id": "rlimit"}
    first = app_client.post("/api/ingest", json=payload, headers=AUTH)
    assert first.status_code == 201, first.get_data(as_text=True)[:200]
    second = app_client.post("/api/ingest", json=payload, headers=AUTH)
    assert second.status_code == 429, second.status_code
    assert int(second.headers["Retry-After"]) >= 1
    # A different node is unaffected: the limit is per node, not global.
    assert (
        app_client.post(
            "/api/ingest",
            json={**payload, "node_id": "rlimit-2"},
            headers=AUTH,
        ).status_code
        == 201
    )


def test_labels_post_list_and_export(app_client):
    reading_id = _latest_reading_id()
    created = app_client.post(
        "/api/labels",
        json={
            "reading_id": reading_id,
            "species": "Aspergillus versicolor",
            "observed": True,
            "confidence": 4,
            "note": "visible bloom on window frame",
        },
        headers=AUTH,
    )
    assert created.status_code == 201, created.get_data(as_text=True)[:300]

    listing = app_client.get("/api/labels").get_json()
    assert listing["count"] >= 1
    assert listing["present"] >= 1

    export = app_client.get("/api/labels/export.csv")
    assert export.status_code == 200
    text = export.get_data(as_text=True)
    assert text.splitlines()[0].startswith("id,ts,reading_id,species,observed")
    assert "Aspergillus versicolor" in text


def test_labels_accept_form_encoded_dashboard_post(app_client):
    reading_id = _latest_reading_id()
    response = app_client.post(
        "/api/labels",
        data={
            "reading_id": str(reading_id),
            "species": "Aspergillus niger",
            "observed": "0",
            "confidence": "2",
        },
        headers={**AUTH, "Origin": "http://localhost"},
    )
    assert response.status_code == 201, response.get_data(as_text=True)[:300]


def test_labels_reject_unknown_reading_and_bad_species(app_client):
    for payload in (
        {"reading_id": 999999, "species": "Aspergillus niger", "observed": True},
        {"reading_id": _latest_reading_id(), "species": "<script>", "observed": True},
        {"reading_id": _latest_reading_id(), "species": "A. niger", "observed": True, "confidence": 9},
    ):
        response = app_client.post("/api/labels", json=payload, headers=AUTH)
        assert response.status_code == 400, (payload, response.status_code)


def test_nodes_and_export_csv(app_client):
    nodes = app_client.get("/api/nodes").get_json()
    assert nodes["count"] >= 1
    row = nodes["nodes"][0]
    # Spec field names plus the aliases the dashboard already reads.
    assert {"node_id", "status", "age_seconds", "risk_class"} <= set(row)
    assert {"last_seen", "age_s", "t_avg", "rh_avg"} <= set(row)
    assert row["status"] in {"green", "yellow", "red"}

    now = int(time.time())
    export = app_client.get(
        f"/api/export.csv?table=readings&from={now - 3600}&to={now + 60}"
    )
    assert export.status_code == 200
    assert export.get_data(as_text=True).startswith("id,ts,x,y,node_id")

    assert app_client.get("/api/export.csv?table=sqlite_master").status_code == 400
    assert app_client.get("/api/export.csv?table=readings&from=999&to=1").status_code == 400


def test_label_modal_roundtrip(app_client):
    """The per-species widget loads a modal and posts with a session token."""
    reading_id = _latest_reading_id()
    species = "Aspergillus versicolor"

    page = app_client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'id="label-modal"' in html
    assert "partials/label_widget" not in html  # included, not referenced
    assert "present" in html

    modal = app_client.get(
        "/partials/label_modal",
        query_string={"reading_id": reading_id, "species": species, "observed": 1},
    )
    assert modal.status_code == 200
    body = modal.get_data(as_text=True)
    assert species in body
    assert "X-Label-Session" in body

    # Bad input is refused rather than stored.
    assert (
        app_client.get(
            "/partials/label_modal",
            query_string={"reading_id": 0, "species": species},
        ).status_code
        == 400
    )
    assert (
        app_client.get(
            "/partials/label_modal",
            query_string={"reading_id": reading_id, "species": "Morchella esculenta"},
        ).status_code
        == 400
    )
    assert app_client.get("/partials/label_modal/close").status_code == 204

def test_label_session_token_authorises_dashboard_post(app_client):
    """A forged session token must not bypass the ingest token."""
    reading_id = _latest_reading_id()
    payload = {
        "reading_id": reading_id,
        "species": "Aspergillus niger",
        "observed": "1",
        "confidence": "3",
    }
    forged = app_client.post(
        "/api/labels",
        data={**payload, "label_token": "0" * 32},
        headers={"X-Label-Session": "0" * 32},
    )
    assert forged.status_code == 401

    with app_client.session_transaction() as sess:
        sess["label_token"] = "a" * 32
    ok = app_client.post(
        "/api/labels", data=payload, headers={"X-Label-Session": "a" * 32}
    )
    assert ok.status_code == 201, ok.get_data(as_text=True)[:300]

def test_sse_disabled_by_default(app_client):
    assert app_client.get("/stream").status_code == 404


def test_schemas_reject_out_of_range():
    from app.schemas import ValidationError, validate_label, validate_reading

    ok = validate_reading({"temperature": 24, "humidity": 85, "node_id": "n-1"})
    assert ok["node_id"] == "n-1"
    assert ok["lux"] == -1.0

    for bad in (
        {"temperature": 120, "humidity": 50},
        {"temperature": 22, "humidity": -1},
        {"temperature": 22, "humidity": 50, "lux": 10**9},
        {"temperature": 22, "humidity": 50, "node_id": "a" * 40},
        {"temperature": "abc", "humidity": 50},
    ):
        with pytest.raises(ValidationError):
            validate_reading(bad)

    label = validate_label(
        {"reading_id": 1, "species": "A. niger", "observed": "yes"}
    )
    assert label["observed"] is True
    assert label["confidence"] == 3
