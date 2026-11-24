"""Alerting backends (spec 4.1) and the optional SSE transport (spec 4.4).

Neither path is exercised by verify.py: alerting is off by default and the SSE
branch is behind `ui.transport: sse`. Both are failure-prone precisely because
they run unattended, so they get tests of their own.
"""

from __future__ import annotations

import time
import urllib.error

import pytest

import app.alerts as alerts

# Alerts are stored with a real timestamp; `alerts_since(n)` is relative to now.
NOW = int(time.time())

CRITICAL = {
    "fgpi": 88.0,
    "risk_class": "critical",
    "temperature": 27.4,
    "humidity": 92.0,
    "node_id": "zone-a",
    "ts": NOW,
}


def _cfg(**overrides) -> dict:
    cfg = {
        "alerts": {
            "enabled": True,
            "backends": ["webhook"],
            "webhook_url": "https://example.invalid/hook",
            "debounce_minutes": 15,
        }
    }
    cfg["alerts"].update(overrides)
    return cfg


class _FakeResponse:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_alerts_do_nothing_when_disabled(fresh_db):
    sent = alerts.maybe_alert(CRITICAL, {"alerts": {"enabled": False, "backends": ["webhook"]}})
    assert sent == []
    assert fresh_db.alerts_since(3600) == []


def test_alerts_ignore_non_critical_readings(fresh_db):
    snapshot = {**CRITICAL, "risk_class": "high"}
    assert alerts.maybe_alert(snapshot, _cfg()) == []
    assert fresh_db.alerts_since(3600) == []


def test_webhook_alert_is_sent_once_per_debounce_window(fresh_db, monkeypatch):
    calls: list[dict] = []

    def fake_urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "body": request.data})
        return _FakeResponse()

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)

    first = alerts.maybe_alert(CRITICAL, _cfg())
    assert [row["status"] for row in first] == ["sent"]
    assert len(calls) == 1
    assert b"FGPI critical in zone zone-a" in calls[0]["body"]

    # Same zone, well inside the 15 minute window: debounced.
    assert alerts.maybe_alert({**CRITICAL, "ts": CRITICAL["ts"] + 60}, _cfg()) == []
    assert len(calls) == 1

    # Past the window it fires again.
    later = CRITICAL["ts"] + 16 * 60
    assert [row["status"] for row in alerts.maybe_alert({**CRITICAL, "ts": later}, _cfg())] == [
        "sent"
    ]
    assert len(calls) == 2
    assert len(fresh_db.alerts_since(3600)) == 2


def test_debounce_is_per_zone(fresh_db, monkeypatch):
    monkeypatch.setattr(
        alerts.urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse()
    )
    assert [r["status"] for r in alerts.maybe_alert(CRITICAL, _cfg())] == ["sent"]
    other = {**CRITICAL, "node_id": "zone-b"}
    assert [r["status"] for r in alerts.maybe_alert(other, _cfg())] == ["sent"]


def test_webhook_failure_is_logged_not_raised(fresh_db, monkeypatch):
    def boom(request, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(alerts.urllib.request, "urlopen", boom)
    sent = alerts.maybe_alert(CRITICAL, _cfg())
    assert [row["status"] for row in sent] == ["failed"]
    assert "no route to host" in sent[0]["detail"]
    # A failed attempt is still recorded, so it is not retried every sample.
    assert alerts.maybe_alert({**CRITICAL, "ts": CRITICAL["ts"] + 60}, _cfg()) == []


def test_unconfigured_backends_report_instead_of_crashing(fresh_db):
    no_url = alerts.maybe_alert(CRITICAL, _cfg(webhook_url=""))
    assert no_url[0]["status"] == "failed"
    assert "webhook_url" in no_url[0]["detail"]

    unknown = alerts.maybe_alert({**CRITICAL, "node_id": "z2"}, _cfg(backends=["telegram"]))
    assert unknown[0]["status"] == "failed"
    assert "telegram" in unknown[0]["detail"]


def test_smtp_without_host_reports_instead_of_connecting(fresh_db):
    sent = alerts.maybe_alert(CRITICAL, _cfg(backends=["smtp"], smtp={}))
    assert sent[0]["status"] == "failed"
    assert sent[0]["detail"] == "smtp.host not configured"


@pytest.fixture()
def sse_client(tmp_path, monkeypatch):
    """A second app instance configured for SSE instead of polling."""
    import yaml

    import app as fungi

    monkeypatch.setenv("FUNGI_DB", str(tmp_path / "sse.db"))
    monkeypatch.setenv("FUNGI_SIMULATE", "1")
    cfg = fungi.load_config()
    cfg.setdefault("ui", {})["transport"] = "sse"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from app.server import create_app

    application = create_app(str(cfg_path))
    application.config.update(TESTING=True)
    return application.test_client()


def test_sse_stream_emits_a_cards_frame(sse_client):
    import app.db as db

    db.init_db()
    db.insert_reading(1_700_000_000, 1.0, 2.0, "sse-node", 24.0, 90.0, 30.0)

    response = sse_client.get("/stream")
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"

    stream = response.response
    first = next(iter(stream))
    assert first == b"retry: 3000\n\n"
    # The generator is infinite; take the first data frame and stop.
    frame = b""
    for chunk in stream:
        frame += chunk
        if b"data: " in frame:
            break
    text = frame.decode("utf-8")
    assert text.startswith("data: ")
    assert "hx-swap-oob" in text or "FGPI" in text
    stream.close()