"""Flask app factory and HTTP surface (spec sections 10, 11 and 12)."""

from __future__ import annotations

import gzip
import logging
import mimetypes
import threading
import time
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.security import safe_join

import app.db as db
from app import STATIC_DIR, TEMPLATE_DIR, load_config
from app.models.fgpi import compute_fgpi
from app.models.species import backend_name, predict_species
from app.spatial.heatmap import render_png
from app.spatial.interpolate import grid_data

log = logging.getLogger("fungi.server")

RISK_RINGS = {
    "low": "ring-green-500",
    "moderate": "ring-yellow-500",
    "high": "ring-orange-500",
    "critical": "ring-red-500",
}
RISK_TEXT = {
    "low": "text-green-700",
    "moderate": "text-yellow-700",
    "high": "text-orange-700",
    "critical": "text-red-700",
}

# pi-btnap's usual PAN subnet. Overridable for unusual deployments.
BT_PREFIXES = ("10.10.10.",)


def _connection_label(remote_addr: str | None) -> str:
    if not remote_addr:
        return "Unknown"
    if remote_addr.startswith(BT_PREFIXES):
        return "Bluetooth PAN"
    if remote_addr in {"127.0.0.1", "::1"} or remote_addr.startswith("::ffff:127."):
        return "Local"
    return "Wi-Fi"


def create_app(config_path: str | None = None) -> Flask:
    cfg = load_config(config_path)
    app = Flask(
        __name__,
        static_folder=None,  # custom route below serves precompressed assets
        template_folder=str(TEMPLATE_DIR),
    )
    app.config["FUNGI_CONFIG"] = cfg
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    db.init_db()
    heatmap_lock = threading.Lock()
    heatmap_path = STATIC_DIR / "heatmap.png"
    heatmap_state = {"source_id": -1}

    @app.after_request
    def _compress_dynamic(response):
        """Gzip HTML/JSON too, so devtools counts the same small payload."""
        if response.direct_passthrough or response.headers.get("Content-Encoding"):
            return response
        if "gzip" not in request.headers.get("Accept-Encoding", "").lower():
            return response
        content_type = (response.content_type or "").lower()
        if not (
            content_type.startswith("text/")
            or content_type in {"application/json", "application/javascript"}
        ):
            return response
        data = response.get_data()
        if len(data) < 500:
            return response
        response.set_data(gzip.compress(data, compresslevel=9, mtime=0))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
        return response

    # ---------------------------------------------------------------- helpers
    def _snapshot() -> dict:
        snap = db.latest_snapshot()
        if snap is None:
            return {
                "has_data": False,
                "ts": None,
                "fgpi": None,
                "risk_class": None,
                "risk_ring": "ring-slate-300",
                "risk_text": "text-slate-500",
                "reading": None,
                "species": [],
            }
        species = snap.get("species") or []
        if not species:
            # Backfill for databases written before species were persisted.
            species = predict_species(
                snap["temperature"], snap["humidity"], snap.get("lux", -1.0) or -1.0
            )
        return {
            "has_data": True,
            "ts": snap["ts"],
            "reading_id": snap["reading_id"],
            "fgpi_id": snap["fgpi_id"],
            "x": snap.get("x"),
            "y": snap.get("y"),
            "node_id": snap.get("node_id"),
            "temperature": snap["temperature"],
            "humidity": snap["humidity"],
            "lux": snap.get("lux"),
            "fgpi": round(float(snap["fgpi"]), 2),
            "risk_class": snap["risk_class"],
            "risk_ring": RISK_RINGS.get(snap["risk_class"], "ring-slate-300"),
            "risk_text": RISK_TEXT.get(snap["risk_class"], "text-slate-500"),
            "species": species,
            "reading": {
                "temperature": snap["temperature"],
                "humidity": snap["humidity"],
                "lux": snap.get("lux"),
                "x": snap.get("x"),
                "y": snap.get("y"),
                "node_id": snap.get("node_id"),
            },
        }

    def _ensure_heatmap() -> Path:
        """Render only when the newest FGPI point is newer than the PNG."""
        spatial = cfg.get("spatial", {})
        with heatmap_lock:
            latest_ts = db.latest_fgpi_ts()
            latest_id = db.latest_fgpi_id()
            try:
                fresh = heatmap_path.exists() and heatmap_path.stat().st_mtime >= latest_ts
            except OSError:
                fresh = False
            if fresh and heatmap_state["source_id"] == latest_id:
                return heatmap_path

            window = int(float(spatial.get("window_hours", 6)) * 3600)
            points = db.spatial_points(window)
            if not points:
                snap = db.latest_snapshot()
                fixed = spatial.get("fixed_node_position") or [
                    float(spatial.get("x_max", 10.0)) / 2.0,
                    float(spatial.get("y_max", 8.0)) / 2.0,
                ]
                points = [
                    (
                        float(snap["x"]) if snap and snap.get("x") is not None else float(fixed[0]),
                        float(snap["y"]) if snap and snap.get("y") is not None else float(fixed[1]),
                        float(snap["fgpi"]) if snap else 0.0,
                    )
                ]
            grid = grid_data(
                points,
                float(spatial.get("x_max", 10.0)),
                float(spatial.get("y_max", 8.0)),
                int(spatial.get("resolution", 100)),
            )
            started = time.perf_counter()
            render_png(grid, str(heatmap_path))
            heatmap_state["source_id"] = latest_id
            log.info("heatmap rendered in %.2fs (%d points)", time.perf_counter() - started, len(points))
            return heatmap_path

    def _heatmap_token() -> int:
        try:
            return int(heatmap_path.stat().st_mtime)
        except OSError:
            return int(time.time())

    def _template_context() -> dict:
        snap = _snapshot()
        return {
            "latest": snap,
            "cfg": cfg,
            "connection": _connection_label(request.remote_addr),
            "backend": backend_name(),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap.get("ts") or time.time())),
            "heatmap_token": _heatmap_token(),
        }

    # ------------------------------------------------------------- static files
    @app.route("/static/<path:filename>", endpoint="static")
    def static_files(filename: str):
        target = safe_join(str(STATIC_DIR), filename)
        if not target or not Path(target).is_file():
            abort(404)
        mime, _ = mimetypes.guess_type(filename)
        mime = mime or "application/octet-stream"
        gz = Path(target + ".gz")
        accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "").lower()
        # The .gz siblings are built artifacts kept in sync by
        # scripts/build_tailwind.sh and the release process.
        if gz.is_file() and accepts_gzip:
            response = send_file(gz, mimetype=mime, conditional=True)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        else:
            response = send_file(target, mimetype=mime, conditional=True)
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    # ------------------------------------------------------------------- pages
    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", **_template_context())

    @app.get("/partials/cards")
    def partial_cards():
        return render_template("partials/cards.html", **_template_context())

    @app.get("/partials/species")
    def partial_species():
        return render_template("partials/species.html", **_template_context())

    @app.get("/partials/heatmap")
    def partial_heatmap():
        return render_template(
            "partials/heatmap.html",
            heatmap_token=_heatmap_token(),
            cfg=cfg,
        )

    # --------------------------------------------------------------------- API
    @app.get("/api/latest")
    def api_latest():
        snap = _snapshot()
        payload = {
            "ts": snap.get("ts"),
            "fgpi": snap.get("fgpi"),
            "risk_class": snap.get("risk_class"),
            "species": snap.get("species", []),
            "reading": snap.get("reading"),
            "node_id": snap.get("node_id"),
        }
        return jsonify(payload)

    @app.get("/api/series")
    def api_series():
        try:
            hours = float(request.args.get("hours", 6))
        except ValueError:
            hours = 6.0
        hours = max(0.1, min(168.0, hours))
        rows = db.rows_for_series(int(hours * 3600))
        return jsonify(
            {
                "hours": hours,
                "ts": [int(r["ts"]) for r in rows],
                "fgpi": [round(float(r["fgpi"]), 2) for r in rows],
                "humidity": [round(float(r["humidity"]), 1) for r in rows],
                "temperature": [round(float(r["temperature"]), 1) for r in rows],
                "lux": [
                    round(float(r["lux"]), 1) if r["lux"] is not None else -1.0
                    for r in rows
                ],
            }
        )

    @app.get("/heatmap.png")
    def heatmap_png():
        path = _ensure_heatmap()
        response = send_file(path, mimetype="image/png", conditional=True)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.post("/api/ingest")
    def api_ingest():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400

        raw_t = data.get("temperature", data.get("T"))
        raw_rh = data.get("humidity", data.get("RH"))
        try:
            temperature = float(raw_t)
            humidity = float(raw_rh)
        except (TypeError, ValueError):
            return jsonify({"error": "temperature and humidity are required numbers"}), 400
        if not (-40.0 <= temperature <= 80.0):
            return jsonify({"error": "temperature outside -40..80 C"}), 400
        if not (0.0 <= humidity <= 100.0):
            return jsonify({"error": "humidity outside 0..100 %"}), 400

        lux_raw = data.get("lux", data.get("L", -1.0))
        try:
            lux = float(lux_raw)
        except (TypeError, ValueError):
            lux = -1.0
        if lux < 0:
            lux = -1.0

        def _optional_float(key: str) -> float | None:
            value = data.get(key)
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        x = _optional_float("x")
        y = _optional_float("y")
        try:
            ts = int(data.get("ts") or time.time())
        except (TypeError, ValueError):
            ts = int(time.time())
        node_id = str(data.get("node_id") or "remote")[:64]

        score, cls = compute_fgpi(temperature, humidity, lux)
        preds = predict_species(temperature, humidity, lux)

        reading_id = db.insert_reading(ts, x, y, node_id, temperature, humidity, lux)
        fgpi_id = db.insert_fgpi(reading_id, ts, score, cls)
        db.insert_species_predictions(fgpi_id, preds)
        return (
            jsonify(
                {
                    "ok": True,
                    "reading_id": reading_id,
                    "fgpi_id": fgpi_id,
                    "ts": ts,
                    "fgpi": score,
                    "risk_class": cls,
                    "species": preds,
                }
            ),
            201,
        )

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("dashboard.html", **_template_context()), 404

    return app


if __name__ == "__main__":
    configuration = load_config()
    server = configuration.get("server", {})
    flask_app = create_app()
    flask_app.run(
        host=str(server.get("host", "0.0.0.0")),
        port=int(server.get("port", 5000)),
        threaded=True,
        debug=False,
    )
