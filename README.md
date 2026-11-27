# 🍄 MusHy — Fungal Growth Forecast & Localization Dashboard

Raspberry Pi system that reads **DHT22 temperature + relative humidity** and a
**BH1750 lux** sensor, computes a **Fungal Growth Potential Index (FGPI)**,
ranks **predicted species suitability**, interpolates FGPI over the monitored
area, and serves a lightweight dashboard over **Wi-Fi or Bluetooth PAN**.

> **This is not a detector.** There is no camera and no image processing. Species
> values are *predicted environmental suitability* from published cardinal
> temperature/humidity curves or a RandomForest model. The UI deliberately says
> "predicted suitability", never "detected".

The repository is `honsda/mushy`. The provided systemd units assume it is
deployed at `/home/pi/fungi-forecast` (the `fungi-forecast/` root from the
project spec) — clone it there, or edit the unit paths.

---

## Contents

- [BOM](#bom)
- [Wiring](#wiring)
- [Repository layout](#repository-layout)
- [Install on the Pi](#install-on-the-pi)
- [Configuration](#configuration)
- [Run it](#run-it)
- [systemd services](#systemd-services)
- [HTTP API](#http-api)
- [Dashboard](#dashboard)
- [Models](#models)
- [East Java regionalization](#east-java-regionalization)
- [Supervised labelling](#supervised-labelling)
- [Training the RandomForest](#training-the-randomforest)
- [Retraining workflow](#retraining-workflow)
- [Resource budget on a Pi 4](#resource-budget-on-a-pi-4)
- [Dataset check](#dataset-check-fgpi-calibration)
- [Bluetooth PAN](#bluetooth-pan)
- [Verification](#verification-against-the-acceptance-criteria)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

`docs/SPEC.md` holds the original contract verbatim; `AGENTS.md` is the working
agreement for future coding sessions.

---

## BOM

| Part | Interface | Notes |
|---|---|---|
| Raspberry Pi 4 or 5, 2 GB+ | — | Raspberry Pi OS Bookworm 64-bit recommended |
| DHT22 / AM2302 | GPIO4 (data), 3.3 V, GND | 10 kΩ pull-up between data and 3.3 V |
| BH1750 | I²C (SDA GPIO2, SCL GPIO3) | preferred light sensor; address `0x23` with ADDR→GND |
| Optional: MCP3008 + photodiode op-amp | SPI0 | alternative light path (`light_driver: mcp3008`) |
| Optional: GPS / AprilTag / UWB | UART/USB | supply `x`, `y` via `POST /api/ingest` |
| microSD, 5 V PSU, enclosure | — | ventilate the DHT22; avoid direct sun on the light sensor |

## Wiring

**DHT22 (AM2302)** — physical pin numbers on the 40-pin header:

| DHT22 | Pi |
|---|---|
| VCC (pin 1) | 3.3 V (pin 1) |
| DATA (pin 2) | GPIO4 (pin 7) |
| GND (pin 4) | GND (pin 6) |
| 10 kΩ | between DATA and 3.3 V |

`config.yaml → sensors.dht22_pin: 4`.

**BH1750**:

| BH1750 | Pi |
|---|---|
| VCC | 3.3 V (pin 1) |
| GND | GND (pin 6) |
| SDA | GPIO2 / SDA1 (pin 3) |
| SCL | GPIO3 / SCL1 (pin 5) |
| ADDR | GND → I²C `0x23` |

Enable I²C with `sudo raspi-config` → Interface Options → I2C, then check with
`i2cdetect -y 1` (expect `23`).

**MCP3008 (optional)** — CLK GPIO11 (pin 23), DOUT GPIO9 (pin 21),
DIN GPIO10 (pin 19), CS GPIO8 (pin 24), CH0 from the photodiode op-amp, VDD/VREF
3.3 V, AGND/DGND GND. Set `light_driver: mcp3008`; calibrate
`sensors.mcp3008_max_lux` for your photodiode.

---

## Repository layout

```
mushy/                         # deployed as fungi-forecast/
├── app/
│   ├── __init__.py            # config loading + paths
│   ├── server.py              # Flask app factory + routes
│   ├── sensors/
│   │   ├── dht22.py           # read_temperature_humidity()
│   │   └── light.py           # read_lux()
│   ├── models/
│   │   ├── fgpi.py            # compute_fgpi()
│   │   ├── species.py         # predict_species()
│   │   └── trained/
│   │       └── species_curves.json
│   ├── spatial/
│   │   ├── interpolate.py     # grid_data() + grid_values()
│   │   ├── layers.py          # map layer specs (units, ramps, domains)
│   │   └── heatmap.py         # render_png() / render_layer_png()
│   ├── db.py                  # SQLite schema + helpers
│   └── templates/
│       ├── dashboard.html
│       └── partials/
│           ├── cards.html
│           ├── species.html
│           └── heatmap.html
├── static/
│   ├── dashboard.css          # precompiled utilities (3.6 KB)
│   ├── htmx.min.js            # 1.9.12 (48.1 KB)
│   ├── uplot.min.js           # 1.6.24 (46.0 KB)
│   ├── uplot.min.css          # 1.9 KB
│   ├── heatmap.png            # generated FGPI map (spec URL)
│   ├── map-<layer>.png        # generated per-layer maps (fgpi/temperature/humidity/lux)
│   └── *.gz                   # precompressed siblings served by Flask
├── scripts/
│   ├── build_tailwind.sh      # dev machine only
│   ├── train_species.py       # offline training
│   ├── nightly_sgd_update.py  # 02:00 adapter update (systemd timer)
│   ├── compress_model.py      # joblib compress + guarded pruning
│   ├── merge_training_data.py # labels + negatives -> one CSV
│   ├── migrate_db.py          # idempotent v2/v3 migration for old DBs
│   ├── dataset_check.py       # FGPI vs published mould-index dataset
│   ├── verify.py              # one-command §15 acceptance gate
│   ├── sample_loop.py         # 30 s sampling daemon
│   └── init_db.py
├── docs/                      # SPEC.md, calibration.md, retraining.md
│   └── SPEC.md               # original brief, verbatim
├── AGENTS.md                  # working agreement for coding sessions
├── systemd/
│   ├── fungi-server.service
│   └── fungi-sampler.service
├── config.yaml
├── requirements.txt
├── requirements-dev.txt         # + pytest, dev/CI only
├── tests/                       # pytest: fgpi, species, db, api, alerts, stream
├── .github/workflows/ci.yml     # compile + pytest + budget guardrails
└── README.md
```

`static/*.gz` are generated build artifacts. On-disk JS+CSS is **99,636 bytes**
(< 100 KB). Flask serves the `.gz` sibling when the client sends
`Accept-Encoding: gzip`, so the actual browser transfer is only ~38 KB before
the map image. Only one map renders per view, so the budget check counts the
largest single layer (~20-30 KB) rather than the sum of all four:
`python scripts/verify.py` asserts both numbers stay under 100 KB.

---

## Install on the Pi

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip i2c-tools libgpiod2
sudo raspi-config          # enable I2C (and SPI only for MCP3008)

git clone https://github.com/honsda/mushy /home/pi/fungi-forecast
cd /home/pi/fungi-forecast

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

sudo usermod -aG i2c,spi,gpio "$USER"   # log out/in afterwards
python scripts/init_db.py
```

`requirements.txt` uses platform markers, so the same file also installs on a
Windows/macOS development machine without trying to build GPIO packages.
On a dev machine `pip install -r requirements-dev.txt` adds pytest; the Pi does
not need it.

Pin the Python version you tested (3.11+). On a Pi 5, Blinka is required; the
DHT driver is instantiated with `use_pulseio=False`, which is the reliable path
on modern kernels.

**Bullseye (Python 3.9) fallbacks.** Blinka 8 needs `gpiod` 2.x, which Debian 11
does not ship; use `pip install "adafruit-blinka<8"` there. Blinka's sysfs path
cannot time a DHT22 reliably on 3.9, so also install the legacy C driver, which
`read_temperature_humidity()` falls back to automatically:

```bash
sudo apt install -y python3-dev build-essential
pip install Adafruit_DHT
```

Verify the wiring with `i2cdetect -y 1` (BH1750 answers at `0x23`, or `0x5C`
when ADDR is tied high) and `raspi-gpio get 2-3` (`func=ALT0` = I2C live).

---

## Configuration

`config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `sensors.dht22_pin` | `4` | BCM GPIO for the DHT22 data line |
| `sensors.light_driver` | `bh1750` | `bh1750` or `mcp3008` |
| `sensors.i2c_bus` | `1` | I²C bus number |
| `sensors.simulate` | `false` | **bench/CI only** synthetic sensors |
| `sampling.interval_seconds` | `30` | sampler cadence |
| `spatial.x_max`, `y_max` | `10.0`, `8.0` | interpolation domain (metres) |
| `spatial.resolution` | `100` | heatmap grid is resolution² |
| `spatial.method` | `idw` | `idw` or `cubic` (scipy) |
| `spatial.fixed_node_position` | `[5.0, 4.0]` | known node coordinates; `null` stores no x/y |
| `spatial.window_hours` | `6` | data window used for the map |
| `server.host`, `server.port` | `0.0.0.0`, `5000` | Flask bind address |
| `sensors.temp_offset`, `humidity_offset`, `lux_scale` | `0.0`, `0.0`, `1.0` | calibration, see [docs/calibration.md](docs/calibration.md) |
| `species.mode` | `auto` | `auto` = RF if joblib exists, else parametric |
| `species.region` | `east_java` | applies `species_curves_indonesia.json` |
| `species.model_path` | `""` | explicit joblib (repo-relative or absolute); empty = `species_rf_east_java.joblib` if present |
| `species.top_k` | `5` | ranked species returned/persisted |
| `species.blend` | `{enabled: true, rf_weight: 0.7, sgd_weight: 0.3}` | blended inference (Phase 3) |
| `species.blend_sgd_weight` | `0.3` | legacy flat key, used when `blend` is absent |
| `species.sgd.min_new_labels` | `20` | labels required before a nightly `partial_fit` |
| `species.min_new_labels` | `20` | legacy flat alias of the above |
| `server.auth.ingest_token` | `""` | bearer token for writes; empty = open LAN demo |
| `server.auth.ingest_rate_limit_seconds` | `5` | per-node ingest interval, `429` beyond |
| `server.auth.allow_dashboard_labels` | `true` | operator's own browser may POST labels |
| `alerts.enabled` | `false` | webhook/SMTP notifications on risk change |
| `ui.transport` | `poll` | `sse` switches the dashboard to `/stream` |
| `database.prune_days` | `90` | retention used by the maintenance timer |

Environment overrides: `FUNGI_CONFIG`, `FUNGI_DB`, `FUNGI_SIMULATE=1`,
`FUNGI_HOST`, `FUNGI_PORT`, `FUNGI_NODE_ID`, `MUSHY_INGEST_TOKEN` (wins over
`server.auth.ingest_token`).

---

## Run it

### Bench / CI without hardware

Linux/macOS:

```bash
export FUNGI_SIMULATE=1
python scripts/init_db.py
python scripts/sample_loop.py --once      # one reading + FGPI + 5 species
python scripts/sample_loop.py             # continuous 30 s loop
python -m app.server                     # http://localhost:5000
```

PowerShell:

```powershell
$env:FUNGI_SIMULATE="1"
python scripts/sample_loop.py --once
python -m app.server
```

`--simulate` is equivalent to the environment variable; `--mobile` random-walks
the node coordinates so the heatmap has spatial variation to interpolate.

### On the Pi with hardware

```bash
source .venv/bin/activate
python scripts/sample_loop.py            # logs T, RH, lux, FGPI, top species
python -m app.server                     # in another shell, or use systemd
```

Open `http://<pi-ip>:5000`.

---

## systemd services

```bash
sudo cp systemd/fungi-server.service systemd/fungi-sampler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fungi-sampler.service fungi-server.service

systemctl status fungi-server fungi-sampler
journalctl -u fungi-sampler -f
```

The units assume user `pi` and `/home/pi/fungi-forecast/.venv/bin/python`.
Change `User=`, `Group=`, `WorkingDirectory=`, and `ExecStart=` if your user or
path differs. Both services restart automatically and survive a reboot.

Two optional timers keep the data bounded and the model current:

```bash
sudo cp systemd/fungi-incremental.* systemd/fungi-maintenance.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fungi-incremental.timer fungi-maintenance.timer

systemctl list-timers 'fungi-*' --no-pager
```

| Timer | When | Does |
|---|---|---|
| `fungi-incremental.timer` | nightly 02:00 (`Persistent=true`) | `scripts/nightly_sgd_update.py` — absorbs new labels into the SGD adapter (status in `logs/sgd_last_run.txt`) |
| `fungi-maintenance.timer` | Sunday 03:30 | `scripts/db_maintenance.py --prune-days 90 --backup ~/backups --vacuum` |

Both are `Type=oneshot`, low priority and skip work rather than compete with
the sampler. The incremental run exits 0 even when it skips, so a quiet week
does not turn the unit red.

---

## HTTP API

| Method | Path | Returns |
|---|---|---|
| GET | `/` | dashboard HTML |
| GET | `/partials/cards` | HTMX fragment: T/RH/lux/FGPI cards |
| GET | `/partials/species` | HTMX fragment: ranked species table |
| GET | `/partials/heatmap?layer=<key>` | HTMX fragment: dropdown + cache-busted `<img>` |
| GET | `/api/latest` | latest reading + `fgpi`, `risk_class`, `species` |
| GET | `/api/series?hours=6` | uPlot arrays: `ts`, `fgpi`, `humidity`, … |
| GET | `/api/maps` | layer metadata for the dropdown (`key`, `label`, `unit`, `url`) |
| GET | `/heatmap.png` | FGPI PNG (alias of `/maps/fgpi.png`, kept for the spec URL) |
| GET | `/maps/<layer>.png` | PNG for `fgpi`, `temperature`, `humidity` or `lux` (`404` otherwise) |
| POST | `/api/ingest` | accept a remote-node reading (JSON, bearer token) |
| POST | `/api/labels` | record a manual observation (present/absent) |
| GET | `/api/labels?since=ISO` | labels with their joined reading |
| GET | `/api/labels/export.csv` | labels + `temperature`/`humidity`/`lux` for retraining |
| GET | `/api/nodes` | per-node `last_seen`/`age_s`/`t_avg`/`rh_avg`, status (`green`/`yellow`/`red`), latest FGPI |
| GET | `/api/export.csv?table=readings&from=…&to=…` | streaming CSV, whitelisted tables |
| GET | `/stream` | SSE push (only when `ui.transport: sse`) |

Writes require `Authorization: Bearer $MUSHY_INGEST_TOKEN` once a token is
configured; reads stay open on the LAN. Repeat ingests from the same node
inside `ingest_rate_limit_seconds` get `429` with `Retry-After`. Payloads are
validated in `app/schemas.py`: temperature −40…80 °C, RH 0…100 %, lux
0…200000 or `-1`, `node_id` `[A-Za-z0-9_-]{1,32}`.

### Map layers

The map panel has a dropdown; each layer is an independent PNG so one view
costs one request. `fgpi` keeps its spec signature (`render_png`); the others
reuse the same interpolation and Matplotlib code with their own colour ramp and
domain:

| Layer | Domain | Notes |
|---|---|---|
| `fgpi` | 0-100 | IDW/cubic per `spatial.method`, clamped to the index range |
| `temperature` | 0-40 °C | always IDW (cubic overshoots with sparse sensors) |
| `humidity` | 0-100 %RH | always IDW |
| `lux` | log10(1+lux) | log scale, because indoor light spans orders of magnitude |

Every map labels each measured site with `node_id`, its FGPI/risk class and the
top predicted species. Only measured sites are labelled; interpolated cells are
never given a species name, so the map cannot be read as an identification at a
location with no sensor. Lux is growth-neutral in FGPI (`f_L = 1.0`) and is
shown for context only.


```bash
curl http://localhost:5000/api/latest
curl "http://localhost:5000/api/series?hours=6"
curl -o heatmap.png http://localhost:5000/heatmap.png

curl -X POST http://localhost:5000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{"temperature":24.1,"humidity":88.0,"lux":120,
       "x":2.5,"y":1.0,"node_id":"node-b"}'
```

`/api/latest` example shape:

```json
{
  "ts": 1795425036,
  "fgpi": 80.0,
  "risk_class": "critical",
  "species": [
    {"species": "Aspergillus ochraceus group", "suitability": 0.901}
  ],
  "reading": {"temperature": 27.0, "humidity": 92.0, "lux": 80.0,
               "x": 2.0, "y": 1.5, "node_id": "test-node"}
}
```

`/api/ingest` computes FGPI and species server-side, inserts all three tables,
and returns `201` with the new ids. It is intended for the trusted LAN / BT PAN;
there is no authentication.

---

## Dashboard

Single page, `max-w-6xl`, responsive `grid-cols-1 md:grid-cols-3`:

- **Header** — title, last-updated time (updated out-of-band by the 5 s card
  poll), connection badge derived from `request.remote_addr`
  (`10.10.10.*` → Bluetooth PAN, loopback → Local, otherwise Wi-Fi).
- **Four metric cards** — temperature, humidity, lux, FGPI; the FGPI card gets
  a risk ring (`green`/`yellow`/`orange`/`red`).
- **Species forecast** — ranked top-5, predicted-suitability progress bars.
- **Map panel** — Matplotlib PNG with a **layer dropdown** (FGPI / Temperature /
  Humidity / Light). Each measured site is annotated with its node id, FGPI
  score, risk class and top predicted species. Axes, title and colourbar are
  baked into the PNG.
- **Time series** — uPlot line chart of FGPI and RH over the last 6 h.

HTMX partially refreshes each panel independently:

| Panel | Endpoint | Trigger |
|---|---|---|
| Cards | `/partials/cards` | `every 5s` |
| Species | `/partials/species` | `every 10s` |
| Map | `/partials/heatmap?layer=<key>` | `every 30s` + `change` on the dropdown |
| Chart | `/api/series` (fetch) | `every 30s` |

No full-page reloads and no SPA framework. Chart refreshes share the same data
window as the map. The dropdown selection survives the 30 s refresh because the
active layer is embedded in the poll URL (`hx-vals` on an ancestor would win
over the select's own value in htmx 1.9, so it is deliberately not used).

### Static asset budget

| Asset | Raw | Served (gzip) |
|---|---:|---:|
| `htmx.min.js` (1.9.12) | 48,101 B | 15,665 B |
| `uplot.min.js` (1.6.24) | 46,025 B | 19,868 B |
| `dashboard.css` | 3,627 B | 1,310 B |
| `uplot.min.css` | 1,883 B | 768 B |
| **Total JS + CSS** | **99,636 B** | **37,611 B** |
| `map-<layer>.png` (generated, one per view) | 20–39 KB raw; 11 KB with a
  single fixed node | same |
| dashboard HTML (gzip) | — | ~3 KB |

Full dashboard transfer in devtools is roughly **50–80 KB** including the
generated heatmap and gzipped HTML. The heatmap is palette-quantised and
downscaled to stay small even when the interpolated field is detailed.

Rebuild the CSS on a dev machine (Node is **not** installed or needed on the
Pi):

```bash
bash scripts/build_tailwind.sh
```

Vendored library licenses are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Models

### FGPI (spec §8)

```
f_T(T)  = 0              T <= 5 or T >= 35
        = (T-5)/15       5 < T < 20
        = 1              20 <= T <= 25
        = (35-T)/10      25 < T < 35
f_RH(RH) = 0                        RH <= 40      (smooth, default)
         = 3t^2 - 2t^3  t=(RH-40)/50   40 < RH < 90
         = 1                        RH >= 90
f_L(L)  = 1.0            light-neutral by design

FGPI = 100 * f_T * f_RH
risk_class: <25 low | 25-50 moderate | 50-75 high | >75 critical
```

`fgpi.humidity_curve: spec` restores the literal §8 ramp (0 below 60 %RH,
1 above 80). The shipped default is the smoothstep above because the literal
ramp is a cliff: it holds the index at exactly 0 for every reading below
60 %RH, so ordinary indoor air never moves the number, and then jumps
0 -> 100 across 20 %RH. Both curves agree at the endpoints; the smooth one
simply responds everywhere in between:

| RH (at 22-25 °C) | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|
| FGPI, `smooth` | 0 | 10 | 35 | 65 | 90 | 100 |
| FGPI, `spec` | 0 | 0 | 0 | 50 | 100 | 100 |

Light is recorded and displayed but intentionally not weighted in FGPI: cheap
lux sensors are easy to shadow or saturate, and the seed species respond
primarily to moisture and temperature.

### Species suitability (spec §9)

1. **Parametric (always available)** — `app/models/trained/species_curves.json`
   holds `T_min`, `T_opt`, `T_max`, `RH_min`, `RH_opt`, and an optional
   `lux_pref` triangular modifier for:
   *Aspergillus versicolor*, *Stachybotrys chartarum*, *Eurotium herbariorum*,
   *Aspergillus niger*, *Aspergillus ochraceus* group, and *Fusarium* spp.
   Suitability is the product of the memberships, ranked descending, top-5.
2. **ML (preferred when present)** — `app/models/trained/species_rf.joblib`,
   a scikit-learn `RandomForestClassifier` trained on
   `[temperature, relative_humidity, lux]`; `predict_proba` is ranked top-5.
   If the file is missing or fails to load, the parametric tier is used
   automatically.

Curve provenance is documented in `species_curves.json → _meta.sources`:
Pitt & Hocking (2009), Flannigan & Miller (2011), Grant et al. (1989),
Andersen/Nielsen/Jarvis (2002), Andrews & Pitt (1987), Samson et al. (2004),
Ramos et al. (1998), and Data INRAE *Fusarium* ecophysiology records. Values are
rounded literature midpoints for **ranking**, not clinical or regulatory advice.

---

## East Java regionalization

The global curve table describes organisms, not a place. Setting
`species.region: east_java` overlays `app/models/trained/species_curves_indonesia.json`
on top of it: per-species `T_opt`/`RH_opt` shifts from Indonesian and Javanese
strain literature, plus two species absent from the global table —
*Aspergillus flavus* (12/30/42 °C, 75/90 %RH) and *Fusarium oxysporum*
(10/25/35 °C, 80/90 %RH) — both common in East Java maize and soybean residues.

The overlay is additive and auditable: `_apply_overlay()` merges per field, so
a curve keeps every global parameter it does not override, and it names its
sources in `_comment`.

Train a region-filtered model:

```bash
python scripts/fetch_regional_data.py --filter training_data_species.csv \
  --out data/regional/aspergillus_indonesia.csv     # bbox -8.8,-6.5,111,115
python scripts/train_species.py --data data/regional/aspergillus_indonesia.csv \
  --region east_java --cv 5
```

`--region east_java` requires latitude/longitude columns: without them the
script refuses to pretend a global file is regional. The run reports 5-fold
macro-F1 and writes `species_rf_east_java.meta.json` with provenance, class
counts, feature ranges and the CV report. At runtime `confidence: high` means
"a trained model answered and this input is inside those feature ranges"; an
input outside them is downgraded to `low` regardless of the score.

GBIF (`fetch_regional_data.py --gbif`) gives occurrence records for the bbox.
Those carry no temperature or humidity, so they validate *presence*, not
growth — they are deliberately not used as a training set.

---

## Supervised labelling

Species predictions are **predicted suitability**, never detections. The Pi
has no camera and does no visual identification. What it can do is record what
an operator saw: each species row has `✓ present` / `✗ absent` buttons that open
a small modal (`partials/label_widget.html` → `partials/label_modal.html`) with
a 1–5 confidence and an optional note. Rows land in the `labels` table joined to
the reading that was live at that moment, so a label is always attributable to
the T/RH/lux that produced it.

The modal carries a **session-scoped** token (`X-Label-Session`, minted per
browser session and signed with the app secret), so the page never contains the
ingest token. Cross-origin posts still need `Authorization: Bearer …`.

```bash
curl -s http://localhost:5000/api/labels | python -m json.tool
curl -s "http://localhost:5000/api/labels/export.csv?since=2025-01-01" -o labels.csv
```

The form posts from the operator's browser without the ingest token in page
source; `server.auth.allow_dashboard_labels` permits same-origin posts and
everything else still needs the bearer token. Labels are the only training data
this project produces — see [docs/retraining.md](docs/retraining.md) for the
export → merge → train → compress → deploy loop. `merge_training_data.py`
converts an `absent` into a negative class only when another species was
marked present at that same reading, so "nobody looked" is never recorded as
"not there".

---

## Training the RandomForest

Training runs on a development machine only; the Pi never trains at runtime.
Copy the resulting joblib into `app/models/trained/` if you want the ML tier.

```bash
pip install scikit-learn joblib
python scripts/train_species.py --data training_data_species.csv
# or, for a pipeline smoke test / bootstrap:
python scripts/train_species.py
```

Approved data sources (spec §13):

- *Aspergillus* distribution dataset (Figshare, `training_data_species.csv`)
- *Fusarium* ecophysiology (Data INRAE) for the parametric curves
- Greenhouse IoT-ML synthetic dataset for pipeline prototyping

`train_species.py` refuses filenames/labels containing `psilocybe`,
`EnvironmentalSensitivity`, or AI-generated markers — e.g. the
`PsilocybeEnvironmentalSensitivity` dataset. The bootstrap mode samples the
shipped literature curves; it is convenient for testing the ML code path but is
**not** a validated occurrence dataset. Report it as such.

The default repo ships **no** joblib model, so a fresh install uses the
parametric tier. That is intentional: a curve-bootstrap model is not evidence of
real-world accuracy.

---

## Retraining workflow

[docs/retraining.md](docs/retraining.md) is the full runbook; the short form:

```bash
curl -s "http://pi:5000/api/labels/export.csv?since=2025-01-01" -o labels.csv
python scripts/merge_training_data.py --base data/regional/aspergillus_indonesia.csv \
  --labels labels.csv --out data/regional/combined.csv
python scripts/train_species.py --data data/regional/combined.csv --region east_java --cv 5
python scripts/compress_model.py --in app/models/trained/species_rf_east_java.joblib \
  --data data/regional/combined.csv --out app/models/trained/species_rf_east_java.min.joblib
```

`compress_model.py` rejects the compressed model (exit 2) when macro-F1 drops
more than 2 points, and in lossless mode makes no accuracy claim at all. The
loader hot-reloads on joblib mtime.

On-device adaptation uses an `SGDClassifier` adapter rather than retraining the
forest on the Pi: `partial_fit` on new labels at 02:00, blended 0.3/0.7 with the
RF posterior, guarded by ≥20 new labels, ≥300 MB free RAM and SoC <70 °C. Every
run logs one line to `logs/incremental.log` and overwrites
`logs/sgd_last_run.txt` with a one-line verdict (`updated: 24 labels` or
`skipped: only 3 new labels (need 20)`).

---

## Resource budget on a Pi 4

Measured with `python scripts/check_budget.py` (advisory; `--strict` to fail).

| Item | Target (spec §7) | This build |
|---|---|---|
| Idle RSS, Flask + numpy + matplotlib | ≤ 300 MB | ~150 MB |
| Map render (100×100 grid, 14 sites) | ≤ 3 s | 0.2–1.6 s |
| `/api/latest` | ≤ 200 ms | ~10 ms |
| Static assets transferred | < 110 KB | 99.6 KB on disk, ~58 KB gzipped + one map |
| Sampler cadence | 30 s | 30 s, a few % of one core |
| Nightly SGD `partial_fit` | < 5 s, < 300 MB | seconds; skipped when idle |

The budget script measures the live process rather than trusting the design:
idle RSS, asset bytes, heatmap wall time and `/api/latest` latency over ten
calls. In CI it runs with `continue-on-error: true` so a slow shared runner
never blocks a merge, while a regression on the Pi is still visible in
`journalctl`.

---

## Dataset check (FGPI calibration)

`species_curves.json` covers *which* species fit a niche. Separately,
`scripts/dataset_check.py` asks whether the FGPI axis itself behaves.

Dataset used — the only one found with all three required columns and a clear
licence:

| | |
|---|---|
| Title | *Low-Cost Indoor Air Quality Monitor with Mould Risk Prediction on an ESP32* |
| DOI | [`10.6084/m9.figshare.35080789`](https://figshare.com/articles/dataset/Data_and_code_for_Low-Cost_Indoor_Air_Quality_Monitor_with_Mould_Risk_Prediction_on_an_ESP32/35080789) |
| Licence | CC BY 4.0 |
| Columns | `temperature_c`, `humidity_pct`, mould index (Hukka–Viitanen / ASHRAE 160) |
| Size | 12,934 real ESP32 readings + 8,760 modelled hourly rows |

```bash
python scripts/dataset_check.py            # downloads once into .fungi/datasets/
python scripts/dataset_check.py --csv my.csv
python scripts/dataset_check.py --write-fixture   # offline sample for CI
```

It reports Spearman correlation, the reference growth rate per FGPI class, and a
monotonicity check. Results on the reference file:

```
growth per hour by FGPI class: low -0.0002 | moderate +0.0008 | high +0.0018 | critical +0.0020
→ monotonic: yes
```

**Known calibration gap (flagged, not silently tuned):** the reference index
first becomes measurably active at FGPI class *moderate*, not *critical*.
FGPI's `f_RH` starts responding at 40 %RH while Hukka–Viitanen needs ~80 %
surface RH, so FGPI acts as an **early-warning growth-potential index**, not a
formation-risk index. That gap is deliberate and documented rather than tuned
away; set `fgpi.humidity_curve: spec` to get the literal §8 ramp back. Class thresholds also deliberately run ahead of the
reference; treat `high`/`critical` as "conditions permit growth", not "mould is
present".

Rejected on sight per spec §13: AI-generated sets such as
`PsilocybeEnvironmentalSensitivity`. The checker refuses filenames containing
`psilocybe` or synthetic/AI markers.

---

## Bluetooth PAN

The Flask app is interface-agnostic: it binds `0.0.0.0:5000`, so the same URL
works over Wi-Fi and Bluetooth PAN. No app changes are needed.

1. Install BlueZ and `pi-btnap` following its upstream README.
2. Enable the PAN server:

   ```bash
   sudo tools/install-btnap server
   ```

3. Reboot or start the `btnap` service, then check the PAN address:

   ```bash
   ip -4 addr show bnep0
   ```

   `pi-btnap`'s usual server address is `10.10.10.1`; adjust to your setup.

4. Pair your phone/laptop, enable Bluetooth tethering/PAN, and browse to:

   ```
   http://10.10.10.1:5000
   ```

5. The header badge reports **Bluetooth PAN** when `request.remote_addr`
   starts with `10.10.10.`. If your subnet differs, edit `BT_PREFIXES` in
   `app/server.py`.

The sampler and dashboard processes do not depend on Bluetooth being up; PAN is
purely an access path.

---

## Verification against the acceptance criteria

One command covers the automatable criteria (throwaway DB, simulated sensors):

```bash
python scripts/verify.py -v    # 20/20 passed, ~20 s
python -m pytest tests -q      # 52 unit/API tests, no hardware
```

The manual walkthrough below is for checking a live deployment by hand:

```bash
# 1. schema
python scripts/init_db.py
sqlite3 fungi.db ".tables"     # alerts_log fgpi labels readings species_predictions

# 2. one sample (hardware, or add FUNGI_SIMULATE=1 on a bench)
python scripts/sample_loop.py --once

# 3. JSON API
curl -s http://localhost:5000/api/latest | python -m json.tool

# 4. every map layer under 3 s once data exists
for l in fgpi temperature humidity lux; do
  time curl -s -o /tmp/$l.png http://localhost:5000/maps/$l.png
  file /tmp/$l.png
done
curl -s http://localhost:5000/api/maps

# 5. dashboard + polling attributes
curl -s http://localhost:5000/ | grep -E 'hx-trigger|uplot|htmx|map-layer'

# 6. assets (one map per view, not four)
du -b static/dashboard.css static/htmx.min.js static/uplot.min.js static/uplot.min.css
# 99636 total on disk; browser devtools shows ~38 KB transferred (gzip)
ls -l static/map-*.png

# 7. parametric fallback with no ML model
python -c "from app.models.species import predict_species; print(predict_species(23,88,100))"

# 8. services
systemctl is-enabled fungi-server fungi-sampler
systemctl is-active fungi-server fungi-sampler

# 9. ingest from a second node
curl -X POST http://localhost:5000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{"temperature":24,"humidity":85,"lux":100,"x":7,"y":3,"node_id":"node-b"}'
```

All partials use `hx-trigger="every Ns"` with `hx-swap`; verifying with browser
devtools shows only the fragment requests, never a full document reload. The
map panel swaps its container (`outerHTML`) so the dropdown keeps working after
each refresh.

---

## Limitations

- **No detection.** No camera, no image classification, no species
  identification. Species output is environmental niche suitability only.
- **Not an occurrence probability.** Parametric curves are cardinal-range
  heuristics; an RF trained on curve bootstrap is a smoke-test model. Validate
  against local data before acting on it.
- **Microclimate matters.** DHT22 accuracy is roughly ±0.5 °C / ±2 % RH. Enclose
  the sensor with airflow and keep it out of direct sunlight and condensation.
- **FGPI is light-neutral.** Lux is stored and shown as its own log-scale map,
  and only mildly modulates some species curves via `lux_pref`.
- **Maps are interpolated surfaces, not measurements.** Site labels name the
  species *predicted at that sensor*; neighbouring colour is IDW-smoothed and
  carries no species identity. Between sensors the field is a model output.
- **Lux map is log-scaled** and excludes unreadable samples (`read_lux() == -1`);
  a dataset with no usable lux renders a `503` for that layer only.
- **FGPI runs ahead of ASHRAE/Hukka–Viitanen** (see
  [Dataset check](#dataset-check-fgpi-calibration)): `high`/`critical` mean
  "conditions can support growth", not "mould is growing".
- **Spatial map quality depends on node coverage.** A single fixed node yields a
  uniform surface. Add `x`/`y` via `/api/ingest` or run `--mobile` to exercise
  interpolation. IDW smooths; it does not model airflow or building geometry.
- **No ingest authentication.** Keep the dashboard on a trusted LAN or BT PAN.
- **SQLite** grows slowly (one row per 30 s plus five species rows). Back up
  `fungi.db`; prune old rows if flash lifetime is a concern.
- **Bluetooth PAN is an access path only** — device pairing/security is handled
  by BlueZ/pi-btnap.
- The frontend is static and dependency-light by design; no cloud/telemetry.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `SensorError: ... not installed` | `pip install -r requirements.txt` inside the venv; on Pi install `libgpiod2` |
| DHT22 intermittent failures | Check the 10 kΩ pull-up and 3.3 V; keep the cable short; the driver retries 3× |
| BH1750 card shows `—` | `i2cdetect -y 1`; check ADDR→GND (`0x23`), enable I²C, add user to `i2c` group |
| Heatmap takes >3 s | Lower `spatial.resolution` or `spatial.window_hours`; first render includes Matplotlib import |
| Port 5000 busy | Change `server.port`, restart both units |
| `database is locked` | Only one writer is expected (sampler); WAL is enabled; check for a stray second sampler |
| Empty chart | Wait for ≥1 sample; `/api/series?hours=6` should return non-empty `ts` |
| Services fail after reboot | `journalctl -u fungi-server -b`; confirm venv path and `WorkingDirectory` in the unit |
| BT URL unreachable | `ip addr show bnep0`; confirm Flask is on `0.0.0.0`; check `ufw`/firewall |

---

Built for the Raspberry Pi Fungal Growth Forecast & Localization spec
(M1–M7). Keep the guardrails: no camera, no detection claims, no cloud
dependency, no SPA, no Node on the Pi, and no AI-generated training data.
