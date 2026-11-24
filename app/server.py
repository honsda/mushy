"""Flask app factory and HTTP surface (spec sections 10, 11 and 12)."""

from __future__ import annotations

import csv
import gzip
import io
import logging
import mimetypes
import os
import threading
import time
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    session,
    stream_with_context,
)
from werkzeug.security import safe_join

import app.db as db
from app import STATIC_DIR, TEMPLATE_DIR, load_config
from app.alerts import maybe_alert
from app.models.fgpi import compute_fgpi
from app.models.species import (
    backend_name,
    confidence_for,
    curve_names,
    predict_species,
)
from app.schemas import ValidationError, validate_label, validate_reading
from app.spatial.heatmap import render_layer_png
from app.spatial.interpolate import grid_data, grid_values
from app.spatial.layers import DEFAULT_LAYER, LAYERS, get_layer, layer_keys

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


# Species names are abbreviated on the map so labels stay inside the plot area.
# "Aspergillus versicolor" -> "A. versicolor"; group/family suffixes are dropped.
def _short_species(name: str) -> str:
    """Compact a binomial for map labels; never changes its meaning."""
    cleaned = " ".join(str(name or "").split())
    cleaned = cleaned.replace(" spp.", " spp").replace(" group", "")
    if not cleaned:
        return "n/a"
    parts = cleaned.split(" ")
    if len(parts) >= 2 and parts[1].lower() in {"spp", "sp.", "sp"}:
        return f"{parts[0][:1]}. {parts[1].rstrip(".")}"
    if len(parts) >= 2 and len(parts[0]) > 3:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return cleaned


def _parse_ts(value, default: int) -> int:
    """Accept epoch seconds or an ISO-ish date/datetime; fall back silently."""
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return int(time.mktime(time.strptime(str(value), fmt)))
        except ValueError:
            continue
    return int(default)


def _parse_day(value) -> int:
    return _parse_ts(value, 0)


def _csv_response(filename: str, columns: list[str], rows) -> Response:
    """Stream CSV without materialising the whole table in memory."""

    def generate():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        yield buffer.getvalue()
        for row in rows:
            buffer.seek(0)
            buffer.truncate(0)
            writer.writerow(row)
            yield buffer.getvalue()

    response = Response(generate(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Cache-Control"] = "no-store"
    return response


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
    # Only used to derive the per-session label token; regenerated on restart
    # unless the operator pins server.secret_key.
    app.secret_key = cfg.get("server", {}).get("secret_key") or os.urandom(32)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    db.init_db()
    heatmap_lock = threading.Lock()
    heatmap_path = STATIC_DIR / "heatmap.png"
    heatmap_state = {"source_id": -1}
    # One PNG per layer, refreshed lazily; key is the layer name.
    layer_paths = {key: STATIC_DIR / f"map-{key}.png" for key in layer_keys()}
    layer_state: dict[str, int] = {key: -1 for key in layer_keys()}

    # ------------------------------------------------------------------- auth
    # Writes always need a token when one is configured. Reads stay open on a
    # trusted LAN unless the operator explicitly opts in (auth_reads).
    auth_cfg = cfg.get("server", {}).get("auth", {}) or {}
    ingest_token = os.environ.get("MUSHY_INGEST_TOKEN") or str(
        auth_cfg.get("ingest_token") or ""
    )
    read_token = os.environ.get("MUSHY_READ_TOKEN") or str(auth_cfg.get("read_token") or "")
    auth_reads = bool(auth_cfg.get("auth_reads", False)) and bool(read_token)
    rate_limit_seconds = max(0.0, float(auth_cfg.get("ingest_rate_limit_seconds", 5)))
    ingest_hits: dict[str, float] = {}

    def _presented_token() -> str:
        header = request.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return (
            request.headers.get("X-API-Key")
            or request.args.get("token")
            or ""
        ).strip()

    def _token_ok(expected: str) -> bool:
        import hmac

        return hmac.compare_digest(_presented_token(), expected)

    @app.before_request
    def _guard_reads():
        if not auth_reads:
            return None
        path = request.path or ""
        if path.startswith(("/api/", "/partials/")) and not path.startswith("/api/ingest"):
            if not _token_ok(read_token):
                return jsonify({"error": "unauthorized"}), 401
        return None

    @app.after_request
    def _compress_dynamic(response):
        """Gzip HTML/JSON too, so devtools counts the same small payload."""
        if (
            response.direct_passthrough
            or response.is_streamed
            or response.headers.get("Content-Encoding")
        ):
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
        if species and any("confidence" not in entry for entry in species):
            # Rows written before confidence existed get it computed from the
            # reading they came from, so the UI never shows a blank column.
            species = [
                {
                    **entry,
                    "confidence": entry.get("confidence")
                    or confidence_for(
                        entry["species"],
                        snap["temperature"],
                        snap["humidity"],
                        snap.get("lux", -1.0) or -1.0,
                    ),
                }
                for entry in species
            ]
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

    def _site_annotations() -> list[dict]:
        """Label each measured site with node id, FGPI and predicted species.

        Only real samples are labelled; interpolated cells stay unannotated so
        the map never implies a species identification between sensors.
        """
        sites: list[dict] = []
        for row in db.sample_locations(int(float(cfg["spatial"].get("window_hours", 6)) * 3600)):
            species = _short_species(str(row.get("species") or ""))
            fgpi = float(row.get("fgpi") or 0.0)
            node = str(row.get("node_id") or "node")[:12]
            label = f"{node}\nFGPI {fgpi:.0f} {row.get('risk_class') or ''}\n{species}"
            sites.append({"x": row["x"], "y": row["y"], "label": label})
        return sites

    def _map_points(field: str, window: int) -> list[tuple[float, float, float]]:
        points = db.spatial_points(window, field=field)
        if points:
            return points
        # A fixed node with no coordinates still deserves a flat field.
        spatial = cfg.get("spatial", {})
        snap = db.latest_reading_values()
        fixed = spatial.get("fixed_node_position") or [
            float(spatial.get("x_max", 10.0)) / 2.0,
            float(spatial.get("y_max", 8.0)) / 2.0,
        ]
        if not snap:
            return []
        x = float(snap["x"]) if snap.get("x") is not None else float(fixed[0])
        y = float(snap["y"]) if snap.get("y") is not None else float(fixed[1])
        if field == "fgpi":
            latest = db.latest_snapshot()
            value = float(latest["fgpi"]) if latest else 0.0
        elif field == "lux":
            value = float(snap.get("lux") or 0.0)
            if value < 0:
                return []
        else:
            value = float(snap.get(field) or 0.0)
        return [(x, y, value)]

    def _ensure_heatmap(layer: str = DEFAULT_LAYER) -> Path:
        """Render a layer only when the newest FGPI row is newer than its PNG."""
        spec = get_layer(layer)
        spatial = cfg.get("spatial", {})
        out_path = layer_paths[spec.key]
        with heatmap_lock:
            latest_ts = db.latest_fgpi_ts()
            latest_id = db.latest_fgpi_id()
            try:
                fresh = out_path.exists() and out_path.stat().st_mtime >= latest_ts
            except OSError:
                fresh = False
            if fresh and layer_state.get(spec.key) == latest_id:
                return out_path

            window = int(float(spatial.get("window_hours", 6)) * 3600)
            resolution = int(spatial.get("resolution", 100))
            x_max = float(spatial.get("x_max", 10.0))
            y_max = float(spatial.get("y_max", 8.0))
            points = _map_points(spec.key, window)
            if not points:
                return out_path

            if spec.key == "fgpi":
                grid = grid_data(points, x_max, y_max, resolution)
            else:
                raw = grid_values(points, x_max, y_max, resolution)
                grid = spec.transform(raw)

            started = time.perf_counter()
            sites = _site_annotations()
            render_layer_png(
                grid,
                str(out_path),
                layer=spec,
                sites=sites,
                x_max=x_max,
                y_max=y_max,
            )
            layer_state[spec.key] = latest_id
            log.info(
                "%s map rendered in %.2fs (%d points, %d sites)",
                spec.key,
                time.perf_counter() - started,
                len(points),
                len(sites),
            )
            return out_path

    def _heatmap_token(layer: str = DEFAULT_LAYER) -> int:
        try:
            return int(layer_paths[get_layer(layer).key].stat().st_mtime)
        except OSError:
            return int(time.time())

    def _selected_layer() -> str:
        return get_layer(request.args.get("layer") or request.args.get("map")).key

    def _template_context() -> dict:
        snap = _snapshot()
        active = _selected_layer()
        interval = int(float(cfg.get("sampling", {}).get("interval_seconds", 30)))
        return {
            "latest": snap,
            "cfg": cfg,
            "connection": _connection_label(request.remote_addr),
            "backend": backend_name(),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap.get("ts") or time.time())),
            "heatmap_token": _heatmap_token(active),
            "active_layer": active,
            "layer": LAYERS[active],
            "layers": [LAYERS[k] for k in layer_keys()],
            "nodes": db.node_status(interval),
            "interval_seconds": interval,
            "label_token": _label_token(),
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
        # oob=True adds the header timestamp as an out-of-band swap. The
        # initial server-rendered include must not emit a second #updated.
        return render_template("partials/cards.html", oob=True, **_template_context())

    @app.get("/partials/species")
    def partial_species():
        return render_template("partials/species.html", **_template_context())

    @app.get("/partials/heatmap")
    def partial_heatmap():
        active = _selected_layer()
        return render_template(
            "partials/heatmap.html",
            heatmap_token=_heatmap_token(active),
            cfg=cfg,
            active_layer=active,
            layer=LAYERS[active],
            layers=[LAYERS[k] for k in layer_keys()],
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
        """FGPI map. Kept for the spec URL and for older bookmarks."""
        return _map_response(DEFAULT_LAYER)

    @app.get("/maps/<layer>.png")
    def map_png(layer: str):
        spec = LAYERS.get(layer.lower())
        if spec is None:
            abort(404)
        return _map_response(spec.key)

    def _map_response(layer: str):
        path = layer_paths[get_layer(layer).key]
        try:
            path = _ensure_heatmap(layer)
        except Exception:
            # A map failure must not take the dashboard down; serve the last
            # good PNG if we have one, otherwise a clear 503.
            log.exception("%s map generation failed", layer)
            if not path.exists():
                abort(503)
        response = send_file(path, mimetype="image/png", conditional=True)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/api/maps")
    def api_maps():
        """Layer metadata for the dropdown and for API consumers."""
        return jsonify(
            {
                "default": DEFAULT_LAYER,
                "layers": [
                    {
                        "key": spec.key,
                        "label": spec.label,
                        "unit": spec.unit,
                        "url": f"/maps/{spec.key}.png",
                        "log_axis": spec.log_axis,
                    }
                    for spec in (LAYERS[k] for k in layer_keys())
                ],
            }
        )

    @app.post("/api/ingest")
    def api_ingest():
        if ingest_token and not _token_ok(ingest_token):
            response = jsonify({"error": "unauthorized"})
            response.status_code = 401
            response.headers["WWW-Authenticate"] = 'Bearer realm="mushy"'
            return response

        try:
            payload = validate_reading(request.get_json(silent=True), cfg.get("spatial", {}))
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400

        node_id = payload["node_id"]
        now = time.monotonic()
        last = ingest_hits.get(node_id)
        if rate_limit_seconds and last is not None and now - last < rate_limit_seconds:
            retry_after = max(1, int(rate_limit_seconds - (now - last)))
            response = jsonify(
                {
                    "error": f"rate limited: one sample per {rate_limit_seconds:g}s per node",
                    "retry_after": retry_after,
                }
            )
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response
        ingest_hits[node_id] = now

        temperature = payload["temperature"]
        humidity = payload["humidity"]
        lux = payload["lux"]
        ts = payload["ts"]

        score, cls = compute_fgpi(temperature, humidity, lux)
        preds = predict_species(temperature, humidity, lux)

        reading_id = db.insert_reading(
            ts, payload["x"], payload["y"], node_id, temperature, humidity, lux
        )
        fgpi_id = db.insert_fgpi(reading_id, ts, score, cls)
        db.insert_species_predictions(fgpi_id, preds)
        try:
            maybe_alert(
                {
                    "fgpi": score,
                    "risk_class": cls,
                    "temperature": temperature,
                    "humidity": humidity,
                    "node_id": node_id,
                    "ts": ts,
                },
                cfg,
            )
        except Exception:  # alerting must never fail an ingest
            log.exception("alert dispatch failed")
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

    # ------------------------------------------------------------------ labels
    def _label_token() -> str:
        """Per-session token so the dashboard can label without the ingest token.

        Generated at first use and kept in the (signed) session cookie, so it
        cannot be forged and dies with the session. `server.auth.ingest_token`
        still works for scripts.
        """
        token = session.get("label_token")
        if not token:
            token = os.urandom(16).hex()
            session["label_token"] = token
        return token

    def _label_token_ok() -> bool:
        import hmac

        presented = request.headers.get("X-Label-Session") or request.form.get(
            "label_token", ""
        )
        expected = session.get("label_token")
        return bool(presented and expected and hmac.compare_digest(presented, expected))

    def _same_origin() -> bool:
        """True when the request came from the dashboard itself.

        The label form runs in the operator's browser and must not carry the
        ingest token in page source, so same-origin submissions are accepted
        when `auth.allow_dashboard_labels` is on (default). Cross-origin posts
        still need the bearer token.
        """
        origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
        if not origin:
            return False
        host = request.host
        return origin.startswith(f"http://{host}") or origin.startswith(f"https://{host}")

    @app.post("/api/labels")
    def api_labels_create():
        if ingest_token and not _token_ok(ingest_token):
            allow_ui = bool(auth_cfg.get("allow_dashboard_labels", True)) and (
                _label_token_ok() or _same_origin()
            )
            if not allow_ui:
                response = jsonify({"error": "unauthorized"})
                response.status_code = 401
                response.headers["WWW-Authenticate"] = 'Bearer realm="mushy"'
                return response
        raw = request.get_json(silent=True)
        if raw is None:
            raw = request.form.to_dict()
        try:
            label = validate_label(raw)
        except ValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            label_id = db.insert_label(
                int(time.time()),
                label["reading_id"],
                label["species"],
                label["observed"],
                label["confidence"],
                label["note"],
                label["operator"],
                label["image_path"],
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if request.headers.get("HX-Request"):
            return render_template(
                "partials/label_saved.html",
                label_id=label_id,
                species=label["species"],
                observed=label["observed"],
            ), 201
        return jsonify({"ok": True, "id": label_id, **label}), 201

    @app.get("/partials/label_modal")
    def label_modal():
        """Modal fragment for the per-species present/absent buttons."""
        try:
            reading_id = int(request.args.get("reading_id", "0"))
        except (TypeError, ValueError):
            reading_id = 0
        species_name = (request.args.get("species") or "").strip()
        observed = str(request.args.get("observed", "1")).lower() in {"1", "true", "yes"}
        known = set(curve_names())
        if reading_id <= 0 or not species_name:
            abort(400)
        if known and species_name not in known:
            # The UI only offers names it displayed, so an unknown one means a
            # hand-crafted request; refuse rather than store a typo as a class.
            abort(400)
        return render_template(
            "partials/label_modal.html",
            reading_id=reading_id,
            species=species_name,
            observed=observed,
            label_token=_label_token(),
        )

    @app.get("/partials/label_modal/close")
    def label_modal_close():
        return "", 204

    @app.get("/api/labels")
    def api_labels_list():
        since = request.args.get("since")
        since_ts = _parse_day(since) if since else 0
        rows = db.labels_with_readings(since_ts)
        return jsonify({"count": len(rows), "labels": rows, **db.label_counts()})

    @app.get("/api/labels/export.csv")
    def api_labels_export():
        since_ts = _parse_day(request.args.get("since")) if request.args.get("since") else 0
        rows = db.labels_with_readings(since_ts)
        columns = [
            "id",
            "ts",
            "reading_id",
            "species",
            "observed",
            "confidence",
            "temperature",
            "humidity",
            "lux",
            "x",
            "y",
            "node_id",
            "operator",
            "note",
        ]
        return _csv_response(
            f"labels-{time.strftime('%Y%m%d')}.csv",
            columns,
            ({key: row.get(key) for key in columns} for row in rows),
        )

    # ------------------------------------------------------------------- nodes
    @app.get("/api/nodes")
    def api_nodes():
        interval = int(float(cfg.get("sampling", {}).get("interval_seconds", 30)))
        nodes = db.node_status(interval)
        return jsonify(
            {
                "count": len(nodes),
                "interval_seconds": interval,
                "nodes": nodes,
            }
        )

    @app.get("/partials/nodes")
    def partial_nodes():
        interval = int(float(cfg.get("sampling", {}).get("interval_seconds", 30)))
        return render_template(
            "partials/nodes.html",
            nodes=db.node_status(interval),
            interval_seconds=interval,
        )

    # ------------------------------------------------------------------ export
    @app.get("/api/export.csv")
    def api_export_csv():
        table = str(request.args.get("table") or "readings").lower()
        if table not in {"readings", "fgpi", "labels", "alerts"}:
            return jsonify({"error": "table must be readings|fgpi|labels|alerts"}), 400
        now = int(time.time())
        from_ts = _parse_ts(request.args.get("from"), now - 7 * 86400)
        to_ts = _parse_ts(request.args.get("to"), now)
        if from_ts > to_ts:
            return jsonify({"error": "`from` must be before `to`"}), 400
        rows = list(db.export_rows(table, from_ts, to_ts))
        columns = list(rows[0].keys()) if rows else ["ts"]
        return _csv_response(
            f"{table}-{time.strftime('%Y%m%d')}.csv",
            columns,
            ({key: row.get(key) for key in columns} for row in rows),
        )

    # -------------------------------------------------------------------- SSE
    @app.get("/stream")
    def stream():
        """Optional SSE transport (config ui.transport: sse).

        Emits the cards fragment whenever SQLite's data_version changes. Falls
        back to polling when the operator leaves transport at `poll`.
        """
        if str(cfg.get("ui", {}).get("transport", "poll")).lower() != "sse":
            abort(404)

        def generate():
            last_version = None
            idle = 0.0
            yield "retry: 3000\n\n"
            while True:
                try:
                    with db.connection() as conn:
                        version = conn.execute("PRAGMA data_version").fetchone()[0]
                except Exception:
                    version = None
                if version is not None and version != last_version:
                    last_version = version
                    idle = 0.0
                    html = render_template(
                        "partials/cards.html", oob=True, **_template_context()
                    )
                    # SSE payloads are newline delimited; the fragment has no
                    # <pre>, so collapsing newlines is safe.
                    yield "data: " + html.replace("\n", " ") + "\n\n"
                else:
                    idle += 1.0
                    if idle >= 15.0:
                        idle = 0.0
                        yield ": keepalive\n\n"
                time.sleep(1.0)

        # Without stream_with_context the request context is torn down before
        # the generator body runs, so the first render_template() inside it
        # raises "Working outside of request context" and the stream dies.
        response = Response(
            stream_with_context(generate()), mimetype="text/event-stream"
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("dashboard.html", **_template_context()), 404

    # ------------------------------------------------------------- alert loop
    # Only runs when explicitly enabled; a daemon thread so `flask run` and
    # the systemd unit both stop cleanly.
    if (cfg.get("alerts") or {}).get("enabled"):
        poll_seconds = max(5.0, float((cfg.get("alerts") or {}).get("poll_seconds", 30)))

        def _alert_loop():
            while True:
                try:
                    maybe_alert(db.latest_snapshot(), cfg)
                except Exception:
                    log.exception("alert poll failed")
                time.sleep(poll_seconds)

        threading.Thread(target=_alert_loop, name="fungi-alerts", daemon=True).start()
        log.info("alerting enabled (poll every %.0fs)", poll_seconds)

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
