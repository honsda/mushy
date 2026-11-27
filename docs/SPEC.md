# PROJECT SPEC — Raspberry Pi Fungal Growth Forecast & Localization Dashboard

> Archived verbatim from the original brief so it never has to be re-pasted.
> Authoritative for interfaces and acceptance. Implementation notes live in
> `README.md`; working rules live in `AGENTS.md`.

Feed this whole document to the agent. It is structured as requirements + interfaces + acceptance criteria.

---

### 1. One-liner
Raspberry Pi system that reads **DHT22 (T, RH)** + **photodiode (lux)**, computes a **Fungal Growth Potential Index (FGPI)**, predicts **likely fungal species** from environmental niche models/ML, interpolates results **spatially**, and serves a **lightweight Tailwind-style dashboard** over **Wi-Fi and Bluetooth PAN**.

---

### 2. Goals / Non-goals

**Goals**
- Continuous sensing + SQLite logging
- FGPI computation per point
- Species suitability prediction (ranked list, not detection)
- Spatial interpolation → heatmap PNG
- Lightweight dashboard (Tailwind UI, HTMX, uPlot, <100 KB frontend)
- Dual access: Wi-Fi (`http://<pi-ip>:5000`) and Bluetooth PAN (same URL)

**Non-goals (hard constraints)**
- ❌ No camera, no image processing, no visual identification
- ❌ No species *detection* — only *suitability prediction*
- ❌ No cloud dependency at runtime
- ❌ No Node.js required on the Pi at runtime
- ❌ No heavy SPA framework (React/Vue/Angular)

---

### 3. Hardware BOM & Wiring

| Part | Interface | Notes |
|---|---|---|
| DHT22 | GPIO4 (data), 3.3V, GND | 10 kΩ pull-up |
| BH1750 or photodiode+op-amp+MCP3008 | I²C (BH1750) or SPI (MCP3008) | BH1750 preferred |
| Raspberry Pi 4/5 | — | 2 GB+ RAM |
| Optional: GPS / AprilTag / UWB | UART/USB | For mobile-node coordinates |

---

### 4. Software Stack (fixed)

| Layer | Choice |
|---|---|
| Runtime | Python 3.11+ |
| Web | Flask + Jinja2 |
| Templating interactivity | HTMX (~14 KB) |
| CSS | Precompiled Tailwind (single `dashboard.css`, ~15 KB) + DaisyUI optional |
| Charts | uPlot (~40 KB) |
| Icons | Inline SVG (Lucide subset) |
| Heatmap render | Matplotlib → `static/heatmap.png` |
| DB | SQLite |
| ML | scikit-learn (`RandomForestClassifier`), persisted via `joblib` |
| Process mgmt | systemd service |
| Access | Wi-Fi (default) + `pi-btnap` for Bluetooth PAN |

---

### 5. Repository Layout (agent must produce exactly this)

```
fungi-forecast/
├── app/
│   ├── __init__.py
│   ├── server.py              # Flask app factory + routes
│   ├── sensors/
│   │   ├── dht22.py           # read_temperature_humidity() -> dict
│   │   └── light.py           # read_lux() -> float
│   ├── models/
│   │   ├── fgpi.py            # compute_fgpi(T, RH, L) -> float 0-100
│   │   ├── species.py         # predict_species(T, RH, L) -> list[dict]
│   │   └── trained/           # joblib model + species_curves.json
│   ├── spatial/
│   │   ├── interpolate.py     # grid_data() -> 2D array
│   │   └── heatmap.py         # render_png() -> path
│   ├── db.py                  # SQLite helpers
│   └── templates/
│       ├── dashboard.html
│       └── partials/
│           ├── cards.html
│           ├── species.html
│           ├── heatmap.html
│           ├── nodes.html          # Phase 4: per-node status table
│           ├── label_widget.html   # Phase 2: present/absent buttons per species
│           ├── label_modal.html    # Phase 2: the form the widget opens
│           └── label_saved.html    # Phase 2: inline confirmation
├── static/
│   ├── dashboard.css          # precompiled Tailwind
│   ├── htmx.min.js
│   ├── uplot.min.js
│   ├── uplot.min.css
│   └── heatmap.png
├── scripts/
│   ├── build_tailwind.sh      # run on dev machine, not Pi
│   ├── train_species.py       # offline training (+ --region/--cv/--out)
│   ├── nightly_sgd_update.py  # Phase 3: 02:00 partial_fit, logs/sgd_last_run.txt
│   ├── compress_model.py      # Phase 3: joblib compress=3, guarded pruning
│   ├── merge_training_data.py # Phase 2: labels + negatives -> training CSV
│   ├── fetch_regional_data.py # Phase 1: GBIF/Figshare bbox filter
│   ├── migrate_db.py          # Phase 4: idempotent v2 (labels) / v3 (FK) migration
│   ├── db_maintenance.py      # Phase 4: prune/backup/vacuum
│   ├── check_budget.py        # Phase 3: Pi 4 RSS/latency/asset budget
│   ├── sample_loop.py         # 30s sampling daemon
│   └── init_db.py
├── config.yaml
├── requirements.txt
├── systemd/
│   ├── fungi-server.service
│   └── fungi-sampler.service
└── README.md
```

---

### 6. SQLite Schema (agent must implement)

```sql
CREATE TABLE readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,           -- unix epoch
  x REAL, y REAL,                -- spatial coords (nullable for fixed node)
  node_id TEXT,
  temperature REAL NOT NULL,
  humidity REAL NOT NULL,
  lux REAL
);

CREATE TABLE fgpi (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  reading_id INTEGER REFERENCES readings(id),
  ts INTEGER NOT NULL,
  fgpi REAL NOT NULL,
  risk_class TEXT NOT NULL       -- low | moderate | high | critical
);

CREATE TABLE species_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fgpi_id INTEGER REFERENCES fgpi(id),
  species TEXT NOT NULL,
  suitability REAL NOT NULL,     -- 0..1
  rank INTEGER NOT NULL
);

CREATE INDEX idx_readings_ts ON readings(ts);
CREATE INDEX idx_fgpi_ts ON fgpi(ts);
```

---

### 7. Module Contracts (agent must implement these signatures)

```python
# sensors/dht22.py
def read_temperature_humidity() -> dict:
    """Returns {'temperature': float, 'humidity': float}. Raises SensorError on failure."""

# sensors/light.py
def read_lux() -> float:
    """Returns lux. Returns -1.0 if sensor unavailable."""

# models/fgpi.py
def compute_fgpi(T: float, RH: float, L: float) -> tuple[float, str]:
    """Returns (fgpi 0-100, risk_class in {low,moderate,high,critical})."""

# models/species.py
def predict_species(T: float, RH: float, L: float) -> list[dict]:
    """Returns [{'species': str, 'suitability': float}, ...] sorted desc, top-5.
    Uses joblib model if present, else falls back to species_curves.json parametric rules."""

# spatial/interpolate.py
def grid_data(points: list[tuple[float,float,float]], x_max: float, y_max: float, resolution: int = 100) -> np.ndarray:
    """IDW or cubic griddata of FGPI over domain. Returns 2D array."""

# spatial/heatmap.py
def render_png(grid: np.ndarray, out_path: str) -> None:
    """Saves heatmap PNG using Matplotlib. Must complete in <3 s on Pi 4."""

# db.py
def init_db() -> None
def insert_reading(ts, x, y, node_id, T, RH, lux) -> int
def insert_fgpi(reading_id, ts, fgpi, risk_class) -> int
def insert_species_predictions(fgpi_id, preds: list[dict]) -> None
def latest_window(seconds: int = 3600) -> list[dict]
```

---

### 8. FGPI Model Spec

```
f_T(T)  = 0                if T <= 5 or T >= 35
        = (T-5)/15         if 5  < T < 20
        = 1                if 20 <= T <= 25
        = (35-T)/10        if 25 < T < 35

f_RH(RH) = 0               if RH <= 60
         = (RH-60)/20      if 60 < RH < 80
         = 1               if RH >= 80

f_L(L)   = 1.0             (light-neutral default; do not over-weight)

FGPI = 100 * f_T * f_RH * f_L
risk_class: <25 low | 25-50 moderate | 50-75 high | >75 critical
```

**Deviation in force (`fgpi.humidity_curve: smooth`, default).** The `f_RH`
ramp above is kept as `humidity_curve: spec` for reproducibility, but it is a
cliff: the index is pinned at exactly 0 for every reading below 60 %RH and then
climbs 0 -> 100 across 60-80 %RH, so ordinary indoor air (40-55 %RH) never moves
the number. The shipped default replaces it with a C1-continuous smoothstep
over `fgpi.humidity_range` (40 -> 90 %RH):

```
f_RH(RH) = 0                        if RH <= lo
         = 3t^2 - 2t^3   t = (RH-lo)/(hi-lo)   if lo < RH < hi
         = 1                        if RH >= hi

lo = 40 %RH, hi = 90 %RH by default
```

Endpoints keep their meaning (no growth potential at <= lo, saturated at >= hi),
`f_T`, `f_L`, the risk-class thresholds and every module signature are unchanged.
The curve is validated by `scripts/dataset_check.py` against the Hukka-Viitanen /
ASHRAE 160 reference index: risk classes still order monotonically with the
reference growth rate (`moderate +0.0008, high +0.0018, critical +0.0020` per
hour). Set `fgpi.humidity_curve: spec` to restore the literal formula.

---

### 9. Species Prediction Spec

**Two-tier strategy (must implement both):**

1. **Parametric (fallback, always available):** `species_curves.json` with per-species `T_min, T_opt, T_max, RH_min, RH_opt` and optional `lux_pref`. Suitability = product of triangular memberships.
2. **ML (preferred when model exists):** `trained/species_rf.joblib` loaded at startup. Inputs `[T, RH, lux]` → `predict_proba` → top-5.

**Seed species table (agent should ship `species_curves.json` with these):**
- *Aspergillus versicolor*
- *Stachybotrys chartarum*
- *Eurotium herbariorum*
- *Aspergillus niger*
- *Aspergillus ochraceus* group
- *Fusarium spp.*

Values sourced from published T-RH growth isopleth literature. Agent must document source in comments.

---

### 10. HTTP API (agent must implement)

| Method | Path | Returns |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/partials/cards` | HTMX fragment: T/RH/lux/FGPI cards |
| GET | `/partials/species` | HTMX fragment: ranked species table |
| GET | `/partials/heatmap` | HTMX fragment: `<img src="/heatmap.png?t=…">` |
| GET | `/api/latest` | JSON: latest reading + fgpi + species |
| GET | `/api/series?hours=6` | JSON: time series for uPlot |
| GET | `/heatmap.png` | PNG heatmap (cache-busted) |
| POST | `/api/ingest` | Accept reading from remote node (JSON) |
| POST | `/api/labels` | Record a manual observation (JSON or form-encoded) |
| GET | `/api/labels` | Labels since `?since=`, with counts |
| GET | `/api/labels/export.csv` | Label CSV for retraining |
| GET | `/api/nodes` | Per-node `last_seen`/`age_s`/`t_avg`/`rh_avg`, `status` (`green`/`yellow`/`red`) |
| GET | `/partials/nodes` | HTMX fragment: node status table |
| GET | `/partials/label_modal` | HTMX fragment: the label form (`reading_id`, `species`, `observed`) |
| GET | `/partials/label_modal/close` | `204`, empties the modal host |
| GET | `/api/export.csv?table=…` | Reading/species export with `from`/`to` |
| GET | `/stream` | SSE stream, only when `ui.transport: sse` |
| GET | `/maps/<layer>.png` | Per-layer PNG (`fgpi`/`temperature`/`humidity`/`lux`) |

All partials refresh via `hx-trigger="every 5s"` (cards) / `every 10s` (species) / `every 30s` (heatmap).

---

### 11. Dashboard UI Spec (Tailwind)

**Layout:** single page, responsive `grid-cols-1 md:grid-cols-3`, max-width `6xl`, `p-4 gap-4`.

**Components (required):**

1. **Header** — title, last-updated timestamp, connection badge (Wi-Fi/BT).
2. **Metric cards ×4** — Temperature, Humidity, Light, FGPI. Each: `rounded-2xl shadow bg-white p-4`, value in `text-3xl font-semibold`, risk color ring (`ring-green-500 / yellow-500 / orange-500 / red-500`).
3. **Species forecast panel** — ranked list with `progress` bar per species (Tailwind `h-2 rounded-full bg-emerald-500` width = suitability %).
4. **Heatmap panel** — `<img>` in `rounded-2xl shadow`, with legend + axis labels baked into PNG.
5. **Time-series chart** — uPlot line chart of FGPI + RH over last 6 h.

**HTMX pattern:** each panel is its own partial with independent `hx-get` + `hx-swap="innerHTML"` + `hx-trigger="every Ns"`. No polling the whole page.

**CSS delivery:** ship `static/dashboard.css` precompiled on dev machine via `scripts/build_tailwind.sh`. Pi never runs Tailwind build.

---

### 12. Access Layer Spec

**Wi-Fi (default):**
- Flask binds `0.0.0.0:5000`
- systemd `fungi-server.service` starts on boot

**Bluetooth PAN (fallback/standalone):**
- Install `pi-btnap` server mode (`tools/install-btnap server`)
- Pi exposes IP (e.g. `10.10.10.1`)
- Same Flask URL reachable at `http://10.10.10.1:5000`
- **No code changes** — access layer is orthogonal to app

**Connection badge** in header reflects which interface the request came in on (`request.remote_addr` prefix check).

---

### 13. Datasets (for offline training only)

| Dataset | Use |
|---|---|
| *Aspergillus* distribution (Figshare, `training_data_species.csv`) | Train RF classifier |
| *Fusarium* ecophysiology (Data INRAE) | Parametric curves for Fusarium |
| Greenhouse IoT-ML synthetic dataset | Prototype pipeline |

> ⚠️ Agent must **not** use AI-generated datasets (e.g., `PsilocybeEnvironmentalSensitivity`). Reject if encountered.

Training runs on dev machine only; output `species_rf.joblib` copied to `app/models/trained/`.

---

### 14. config.yaml (agent must scaffold)

```yaml
sensors:
  dht22_pin: 4
  light_driver: bh1750   # or mcp3008
  i2c_bus: 1
sampling:
  interval_seconds: 30
spatial:
  x_max: 10.0
  y_max: 8.0
  resolution: 100
  method: idw            # idw | cubic
server:
  host: 0.0.0.0
  port: 5000
species:
  mode: auto             # auto = ml if model exists else parametric
  top_k: 5
```

---

### 15. Acceptance Criteria (agent must verify)

- [ ] `python scripts/init_db.py` creates `readings`, `fgpi`, `species_predictions` tables
- [ ] `python scripts/sample_loop.py` writes ≥1 reading + fgpi + species every 30 s
- [ ] `curl http://localhost:5000/api/latest` returns valid JSON with `fgpi`, `risk_class`, `species` array
- [ ] `curl http://localhost:5000/heatmap.png` returns a valid PNG ≤3 s after data exists
- [ ] Dashboard loads with **all static assets <100 KB total** (verify with browser devtools)
- [ ] Dashboard updates cards every 5 s, species every 10 s, heatmap every 30 s **without full page reload**
- [ ] `predict_species()` returns ranked top-5 even with no ML model (parametric fallback)
- [ ] systemd services start on boot and survive reboot
- [ ] Same dashboard reachable over Wi-Fi and (if configured) Bluetooth PAN
- [ ] README documents: BOM, wiring, install, dataset training, BT PAN setup, limitations

> Automated coverage for these lives in `scripts/verify.py`; the reboot and
> Bluetooth items need a human on the hardware.

---

### 16. Milestones (agent executes in order)

1. **M1 — Skeleton:** repo layout, config, requirements, DB init, Flask hello-world
2. **M2 — Sensing:** DHT22 + BH1750 drivers, sampler daemon, DB writes
3. **M3 — Models:** FGPI, parametric species, optional RF training script
4. **M4 — Spatial:** interpolation + Matplotlib heatmap PNG
5. **M5 — Dashboard:** Tailwind CSS build, HTMX partials, uPlot chart
6. **M6 — Access:** systemd units, Wi-Fi bind, `pi-btnap` docs
7. **M7 — Hardening:** error handling, sensor-failure fallbacks, README

---

### 17. Guardrails for the Agent (do not violate)

- Do **not** add a camera module or any image classification
- Do **not** introduce Node/npm/build tooling on the Pi at runtime
- Do **not** add React/Vue/Svelte or any SPA framework
- Do **not** add cloud APIs, telemetry, or external runtime dependencies
- Do **not** claim detection — UI copy must say **"predicted suitability"**, never **"detected"**
- Do **not** use AI-generated datasets
- Do **not** exceed 100 KB of frontend static assets
- Preserve all module signatures in §7 exactly — downstream code depends on them

---

**End of spec.** Agent should scaffold the repo, implement M1–M7, and self-verify against §15.