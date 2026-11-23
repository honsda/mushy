"""Alerting backends (webhook + SMTP) with per-zone debouncing.

Design notes
------------
* The Pi must not block sampling on a slow network: every send runs with a
  short timeout and a failure is logged, never raised.
* Dedup state lives in SQLite (``alerts_log``), so a restart does not re-page
  the operator for a condition that is still true.
* Only *transitions* alert. A zone that stays critical for hours notifies once
  per ``debounce_minutes``, not once per sample.
* This is the only module that ever makes an outbound request. With
  ``alerts.enabled: false`` (the default) nothing leaves the device.
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage

import app.db as db

log = logging.getLogger("fungi.alerts")

HTTP_TIMEOUT = 5.0


def _format_message(snapshot: dict, zone: str) -> str:
    return (
        f"FGPI critical in zone {zone}: "
        f"{float(snapshot.get('fgpi') or 0):.0f}/100 "
        f"({snapshot.get('temperature'):.1f} C, "
        f"{snapshot.get('humidity'):.0f} %RH, "
        f"{snapshot.get('node_id') or 'unknown node'}). "
        "Predicted growth suitability is critical; inspect the area."
    )


def _send_webhook(url: str, text: str, snapshot: dict) -> tuple[bool, str]:
    payload = {
        # Discord and Slack both accept `content`; Teams-style hooks read `text`.
        "content": text,
        "text": text,
        "username": "mushy",
        "fgpi": snapshot.get("fgpi"),
        "risk_class": snapshot.get("risk_class"),
        "node_id": snapshot.get("node_id"),
        "ts": snapshot.get("ts"),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return True, f"http {response.status}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"webhook failed: {exc}"


def _send_smtp(cfg: dict, text: str) -> tuple[bool, str]:
    smtp = cfg.get("smtp") or {}
    host = smtp.get("host")
    if not host:
        return False, "smtp.host not configured"
    message = EmailMessage()
    message["Subject"] = smtp.get("subject", "[mushy] FGPI critical")
    message["From"] = smtp.get("from", "mushy@localhost")
    message["To"] = ", ".join(smtp.get("to") or [])
    if not message["To"]:
        return False, "smtp.to not configured"
    message.set_content(text)
    try:
        with smtplib.SMTP(host, int(smtp.get("port", 587)), timeout=HTTP_TIMEOUT) as client:
            if smtp.get("starttls", True):
                client.starttls()
            if smtp.get("user"):
                client.login(smtp["user"], smtp.get("password", ""))
            client.send_message(message)
        return True, "smtp ok"
    except (OSError, smtplib.SMTPException) as exc:
        return False, f"smtp failed: {exc}"


def maybe_alert(snapshot: dict | None, cfg: dict) -> list[dict]:
    """Notify if the snapshot is critical and outside its debounce window.

    Returns the alert rows written (empty when nothing fired). Safe to call on
    every sample; the debounce query decides.
    """
    alerts_cfg = (cfg or {}).get("alerts") or {}
    if not alerts_cfg.get("enabled"):
        return []
    if not snapshot or snapshot.get("risk_class") != "critical":
        return []

    zone = str(snapshot.get("node_id") or "default")
    now = int(snapshot.get("ts") or time_now())
    debounce = int(alerts_cfg.get("debounce_minutes", 15)) * 60
    if debounce > 0 and now - db.last_alert_ts(zone) < debounce:
        return []

    text = _format_message(snapshot, zone)
    written: list[dict] = []
    for backend in alerts_cfg.get("backends") or []:
        if backend == "webhook":
            url = alerts_cfg.get("webhook_url") or ""
            ok, detail = (
                _send_webhook(url, text, snapshot)
                if url
                else (False, "webhook_url not configured")
            )
        elif backend == "smtp":
            ok, detail = _send_smtp(alerts_cfg, text)
        else:
            ok, detail = False, f"unknown backend '{backend}'"

        status = "sent" if ok else "failed"
        db.record_alert(now, zone, float(snapshot.get("fgpi") or 0.0), backend, status)
        written.append({"zone": zone, "backend": backend, "status": status, "detail": detail})
        log.info("alert %s via %s: %s", zone, backend, detail)
    return written


def time_now() -> int:
    import time

    return int(time.time())
