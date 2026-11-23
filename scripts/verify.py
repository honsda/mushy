#!/usr/bin/env python3
"""Acceptance harness for spec section 15 — one command, terse output.

    python scripts/verify.py        # PASS/FAIL table, exit code 1 on any failure
    python scripts/verify.py -v     # same, plus a measured detail per check

Uses a throwaway SQLite file and the simulated sensors (FUNGI_SIMULATE=1), so it
is safe on the dev machine, in CI, and on the Pi. It does write
static/heatmap.png, because that is exactly what /heatmap.png does in service.

Checks that need hardware, a reboot, or a Bluetooth peer stay manual and are
listed at the end of AGENTS.md.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS = (
    "static/dashboard.css",
    "static/htmx.min.js",
    "static/uplot.min.js",
    "static/uplot.min.css",
)
RISK_CLASSES = {"low", "moderate", "high", "critical"}

_client = None


def client():
    global _client
    if _client is None:
        from app.server import create_app

        application = create_app()
        application.config.update(TESTING=True)
        _client = application.test_client()
    return _client


def counts() -> dict[str, int]:
    con = sqlite3.connect(os.environ["FUNGI_DB"])
    try:
        return {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("readings", "fgpi", "species_predictions")
        }
    finally:
        con.close()


def check_schema():
    import app.db as db

    db.init_db()
    con = sqlite3.connect(os.environ["FUNGI_DB"])
    try:
        tables = {
            r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        indexes = {
            r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    finally:
        con.close()
    missing = {"readings", "fgpi", "species_predictions"} - tables
    assert not missing, f"missing tables: {sorted(missing)}"
    missing = {"idx_readings_ts", "idx_fgpi_ts"} - indexes
    assert not missing, f"missing indexes: {sorted(missing)}"
    return f"tables {sorted(tables - {'sqlite_sequence'})}"


def check_sampler():
    before = counts()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sample_loop.py"), "--once"],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-500:]
    after = counts()
    for table, n in before.items():
        assert after[table] >= n + 1, f"{table}: {n} -> {after[table]} (need +1)"
    return " +1 reading/fgpi/species"


def check_api_latest():
    resp = client().get("/api/latest")
    assert resp.status_code == 200, resp.status_code
    body = resp.get_json()
    assert body is not None, "not JSON"
    for key in ("ts", "fgpi", "risk_class", "species"):
        assert key in body, f"missing {key}"
    assert body["risk_class"] in RISK_CLASSES, body["risk_class"]
    species = body["species"]
    assert isinstance(species, list) and species, "no species predictions"
    suit = [p["suitability"] for p in species]
    assert suit == sorted(suit, reverse=True), "species not sorted desc"
    assert all(0.0 <= s <= 1.0 for s in suit), suit
    return f"fgpi={body['fgpi']:.1f} {body['risk_class']}, {len(species)} species"


def check_api_series():
    body = client().get("/api/series?hours=6").get_json()
    for key in ("ts", "fgpi", "humidity"):
        assert isinstance(body.get(key), list), f"missing list {key}"
    assert len(body["ts"]) == len(body["fgpi"]), "series lengths differ"
    return f"{len(body['ts'])} points"


def check_dashboard():
    resp = client().get("/")
    assert resp.status_code == 200, resp.status_code
    html = resp.get_data(as_text=True)
    for needle in ("htmx.min.js", "uplot.min.js", "dashboard.css"):
        assert needle in html, f"missing asset ref {needle}"
    for interval in ("every 5s", "every 10s", "every 30s"):
        assert interval in html, f"missing poll {interval}"
    return "assets linked, 5s/10s/30s polls"


def check_partials():
    for path in ("/partials/cards", "/partials/species", "/partials/heatmap"):
        resp = client().get(path)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}"
        assert len(resp.get_data()) > 100, f"{path} body too small"
    return "cards/species/heatmap OK"


def check_heatmap():
    started = time.perf_counter()
    resp = client().get("/heatmap.png")
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200, resp.status_code
    png = resp.get_data()
    assert png.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG"
    assert len(png) > 1000, f"{len(png)} B"
    assert elapsed <= 3.0, f"{elapsed:.2f}s > 3s budget"
    return f"{len(png)} B in {elapsed * 1000:.0f} ms"


def check_ingest():
    before = counts()["readings"]
    payload = {"temperature": 24.0, "humidity": 85.0, "lux": 100.0, "x": 7.0, "y": 3.0, "node_id": "verify"}
    resp = client().post("/api/ingest", json=payload)
    assert resp.status_code < 300, f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    assert counts()["readings"] >= before + 1, "ingest did not persist a reading"
    return f"{resp.status_code} accepted"


def check_assets():
    raw = sum((ROOT / a).stat().st_size for a in ASSETS)
    gz = sum(len(gzip.compress((ROOT / a).read_bytes(), 9)) for a in ASSETS)
    # The dashboard shows one map at a time, so the worst case is the largest
    # rendered layer, not the sum of all four.
    maps = [
        p for p in (ROOT / "static").glob("map-*.png")
    ] + [ROOT / "static" / "heatmap.png"]
    present = [p.stat().st_size for p in maps if p.exists()]
    biggest_map = max(present) if present else 0
    transferred = gz + biggest_map
    assert raw <= 100_000, f"vendor assets {raw} B > 100 KB on disk"
    assert transferred <= 100_000, f"{transferred} B transferred > 100 KB"
    return (
        f"{raw / 1000:.1f} KB on disk, {transferred / 1000:.1f} KB transferred "
        f"(incl. {biggest_map / 1000:.1f} KB map)"
    )


def check_parametric():
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    cfg.setdefault("species", {})["mode"] = "parametric"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
        cfg_path = fh.name
    env = os.environ.copy()
    env["FUNGI_CONFIG"] = cfg_path
    code = (
        "import json; from app.models.species import predict_species; "
        "print(json.dumps(predict_species(23.0, 88.0, 100.0)))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        Path(cfg_path).unlink(missing_ok=True)
    assert proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-500:]
    preds = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(preds) == 5, f"{len(preds)} predictions, want 5"
    suit = [p["suitability"] for p in preds]
    assert suit == sorted(suit, reverse=True), "not ranked desc"
    assert all(0.0 <= s <= 1.0 for s in suit), suit
    return f"no-ML top-5, best={preds[0]['species']} {suit[0]:.2f}"


def check_guardrails():
    import re

    code_pattern = re.compile(r"^\s*(?:import|from)\s+(cv2|torch|tensorflow|tflite_runtime)\b", re.M)
    template_pattern = re.compile(r"\bdetected\b", re.I)
    remote_pattern = re.compile(r"<(?:script|link)[^>]+(?:https?:)?//", re.I)
    offenders: list[str] = []
    for path in sorted((ROOT / "app").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        if code_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(f"{path.relative_to(ROOT)} imports vision/ML framework")
    for path in sorted((ROOT / "app" / "templates").rglob("*.html")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if template_pattern.search(text):
            offenders.append(f"{path.relative_to(ROOT)} says 'detected'")
        if remote_pattern.search(text):
            offenders.append(f"{path.relative_to(ROOT)} fetches a remote asset")
    assert not offenders, "; ".join(offenders)
    return "no camera/SPA/CDN, no 'detected' copy"


def check_maps():
    keys = ("fgpi", "temperature", "humidity", "lux")
    sizes = {}
    for key in keys:
        started = time.perf_counter()
        resp = client().get(f"/maps/{key}.png")
        elapsed = time.perf_counter() - started
        assert resp.status_code == 200, f"/maps/{key}.png -> {resp.status_code}"
        png = resp.get_data()
        assert png.startswith(b"\x89PNG\r\n\x1a\n"), f"{key}: not a PNG"
        assert elapsed <= 3.0, f"{key}: {elapsed:.2f}s > 3s budget"
        sizes[key] = len(png)
    # Unknown layers must 404, not render the default.
    assert client().get("/maps/nope.png").status_code == 404, "unknown layer did not 404"
    meta = client().get("/api/maps").get_json()
    assert [item["key"] for item in meta["layers"]] == list(keys), meta
    assert meta["default"] == "fgpi", meta["default"]
    largest = max(sizes.values())
    return f"{len(sizes)} layers, largest {largest / 1000:.0f} KB"


def check_layer_partial():
    for layer in ("fgpi", "temperature", "humidity", "lux"):
        html = client().get(f"/partials/heatmap?layer={layer}").get_data(as_text=True)
        assert f"/maps/{layer}.png" in html, f"layer {layer} not in img src"
        assert f'<option value="{layer}" selected' in html, f"layer {layer} not selected"
        assert "name=\"layer\"" in html, "dropdown missing"
    return "dropdown drives all 4 layers"


def check_dataset_calibration():
    """Offline sanity check: risk classes must order with reference growth."""
    fixture = ROOT / "scripts" / "fixtures" / "mould_index_sample.csv"
    assert fixture.exists(), "run: python scripts/dataset_check.py --write-fixture"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dataset_check.py"), "--csv", str(fixture)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-500:]
    report = json.loads(proc.stdout[: proc.stdout.rfind("}") + 1])
    rows = report["results"][0]["rows"]
    assert rows >= 5, f"fixture has {rows} usable rows"
    return f"fixture {rows} rows, risk classes ordered"


def check_auth():
    """Writes need a bearer token once one is configured; reads stay open."""
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    cfg.setdefault("server", {}).setdefault("auth", {})["ingest_token"] = "verify-token"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
        cfg_path = fh.name
    from app.server import create_app

    client = create_app(cfg_path).test_client()
    payload = {"temperature": 24.0, "humidity": 88.0, "node_id": "auth-check"}
    assert client.post("/api/ingest", json=payload).status_code == 401, "accepted an unauthenticated write"
    assert client.post("/api/ingest", json={**payload, "node_id": "auth-wrong"},
                       headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/api/ingest", json={**payload, "node_id": "auth-ok"},
                       headers={"Authorization": "Bearer verify-token"}).status_code == 201
    limited = client.post("/api/ingest", json={**payload, "node_id": "auth-ok"},
                          headers={"Authorization": "Bearer verify-token"})
    assert limited.status_code == 429, f"rate limit not enforced: {limited.status_code}"
    assert int(limited.headers.get("Retry-After", 0)) >= 1
    assert client.get("/api/latest").status_code == 200, "reads must stay open on the LAN"
    Path(cfg_path).unlink(missing_ok=True)
    return "401 without token, 201 with, 429 on repeat"


def check_labels():
    import app.db as db

    reading_id = int(db.latest_snapshot()["reading_id"])
    created = client().post(
        "/api/labels",
        json={
            "reading_id": reading_id,
            "species": "Aspergillus versicolor",
            "observed": True,
            "confidence": 4,
            "note": "verify",
        },
    )
    assert created.status_code == 201, created.get_data(as_text=True)[:200]
    listing = client().get("/api/labels").get_json()
    assert listing["count"] >= 1, listing
    export = client().get("/api/labels/export.csv")
    assert export.status_code == 200
    text = export.get_data(as_text=True)
    assert text.startswith("id,ts,reading_id,species,observed"), text[:80]
    return f"{listing['count']} label(s), CSV export ok"


def check_nodes_and_export():
    nodes = client().get("/api/nodes").get_json()
    assert nodes["count"] >= 1, nodes
    row = nodes["nodes"][0]
    assert row["status"] in {"green", "amber", "red"}, row
    now = int(time.time())
    export = client().get(f"/api/export.csv?table=readings&from={now - 3600}&to={now + 60}")
    assert export.status_code == 200, export.status_code
    body = export.get_data(as_text=True)
    assert body.startswith("id,ts,x,y,node_id"), body[:80]
    assert len(body.splitlines()) >= 2, "export has no rows"
    assert client().get("/api/export.csv?table=sqlite_master").status_code == 400
    return f"{nodes['count']} nodes, {len(body.splitlines()) - 1} CSV rows"


def check_confidence():
    species = client().get("/api/latest").get_json()["species"]
    assert species, "no species predictions"
    allowed = {"low", "medium", "high"}
    for entry in species:
        assert entry.get("confidence") in allowed, entry
    return f"{len(species)} predictions tagged ({species[0]['confidence']})"


def check_regional_overlay():
    import app.models.species as model

    assert model.region() == "east_java", model.region()
    names = {curve["name"] for curve in model._load_curves()}
    assert "Aspergillus flavus" in names, "overlay not merged"
    flavus = next(c for c in model._load_curves() if c["name"] == "Aspergillus flavus")
    assert flavus["T_opt"] == 30.0, flavus
    return f"east_java overlay active ({len(names)} species)"


def check_sse_off_by_default():
    assert client().get("/stream").status_code == 404, "SSE served while ui.transport=poll"
    return "poll transport (SSE opt-in)"


CHECKS = (
    ("schema", check_schema),
    ("sampler", check_sampler),
    ("api/latest", check_api_latest),
    ("api/series", check_api_series),
    ("dashboard", check_dashboard),
    ("partials", check_partials),
    ("maps: 4 layers <=3s", check_maps),
    ("map dropdown", check_layer_partial),
    ("heatmap.png <=3s", check_heatmap),
    ("api/ingest", check_ingest),
    ("static budget", check_assets),
    ("parametric fallback", check_parametric),
    ("dataset calibration", check_dataset_calibration),
    ("guardrails", check_guardrails),
    ("api auth", check_auth),
    ("labels", check_labels),
    ("nodes + CSV export", check_nodes_and_export),
    ("confidence", check_confidence),
    ("regional overlay", check_regional_overlay),
    ("transport", check_sse_off_by_default),
)


def main(argv: list[str]) -> int:
    verbose = any(a in {"-v", "--verbose"} for a in argv[1:])
    tmp = tempfile.TemporaryDirectory(prefix="fungi-verify-")
    os.environ["FUNGI_DB"] = str(Path(tmp.name) / "verify.db")
    os.environ["FUNGI_SIMULATE"] = "1"

    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            suffix = f"  [{detail}]" if verbose and detail else ""
            print(f"PASS  {name}{suffix}")

    total = len(CHECKS)
    print(f"\n{total - failures}/{total} passed")
    if failures:
        print("Manual checks left to a human: reboot persistence, Bluetooth PAN reachability.")
    tmp.cleanup()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))