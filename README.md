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
- [Training the RandomForest](#training-the-randomforest)
- [Bluetooth PAN](#bluetooth-pan)
- [Verification](#verification-against-the-acceptance-criteria)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

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
│   │   ├── interpolate.py     # grid_data()
│   │   └── heatmap.py         # render_png()
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
│   ├── heatmap.png            # generated; placeholder shipped
│   └── *.gz                   # precompressed siblings served by Flask
├── scripts/
│   ├── build_tailwind.sh      # dev machine only
│   ├── train_species.py       # offline training
│   ├── sample_loop.py         # 30 s sampling daemon
│   └── init_db.py
├── systemd/
│   ├── fungi-server.service
│   └── fungi-sampler.service
├── config.yaml
├── requirements.txt
└── README.md
```

`static/*.gz` are generated build artifacts. On-disk JS+CSS is **99,636 bytes**
(< 100 KB). Flask serves the `.gz` sibling when the client sends
`Accept-Encoding: gzip`, so the actual browser transfer is only ~38 KB before
the generated heatmap image (and ~65 KB including it and the gzipped HTML).

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

Pin the Python version you tested (3.11+). On a Pi 5, Blinka is required; the
DHT driver is instantiated with `use_pulseio=False`, which is the reliable path
on modern kernels.

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
| `species.mode` | `auto` | `auto` = RF if joblib exists, else parametric |
| `species.top_k` | `5` | ranked species returned/persisted |
| `database.path` | `fungi.db` | relative to the project root |

Environment overrides: `FUNGI_CONFIG`, `FUNGI_DB`, `FUNGI_SIMULATE=1`,
`FUNGI_HOST`, `FUNGI_PORT`, `FUNGI_NODE_ID`.

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

---

## HTTP API

| Method | Path | Returns |
|---|---|---|
| GET | `/` | dashboard HTML |
| GET | `/partials/cards` | HTMX fragment: T/RH/lux/FGPI cards |
| GET | `/partials/species` | HTMX fragment: ranked species table |
| GET | `/partials/heatmap` | HTMX fragment: cache-busted `<img>` |
| GET | `/api/latest` | latest reading + `fgpi`, `risk_class`, `species` |
| GET | `/api/series?hours=6` | uPlot arrays: `ts`, `fgpi`, `humidity`, … |
| GET | `/heatmap.png` | generated PNG (no-store, cache-busted by query) |
| POST | `/api/ingest` | accept a remote-node reading (JSON) |

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
- **FGPI map** — Matplotlib PNG with baked-in axes, title and colourbar.
- **Time series** — uPlot line chart of FGPI and RH over the last 6 h.

HTMX partially refreshes each panel independently:

| Panel | Endpoint | Trigger |
|---|---|---|
| Cards | `/partials/cards` | `every 5s` |
| Species | `/partials/species` | `every 10s` |
| Heatmap | `/partials/heatmap` | `every 30s` |
| Chart | `/api/series` (fetch) | `every 30s` |

No full-page reloads and no SPA framework. Chart refreshes share the same data
window as the map.

### Static asset budget

| Asset | Raw | Served (gzip) |
|---|---:|---:|
| `htmx.min.js` (1.9.12) | 48,101 B | 15,665 B |
| `uplot.min.js` (1.6.24) | 46,025 B | 19,868 B |
| `dashboard.css` | 3,627 B | 1,310 B |
| `uplot.min.css` | 1,883 B | 768 B |
| **Total JS + CSS** | **99,636 B** | **37,611 B** |
| `heatmap.png` (generated) | 9–38 KB | same |
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
f_RH(RH) = 0             RH <= 60
         = (RH-60)/20    60 < RH < 80
         = 1             RH >= 80
f_L(L)  = 1.0            light-neutral by design

FGPI = 100 * f_T * f_RH
risk_class: <25 low | 25-50 moderate | 50-75 high | >75 critical
```

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

```bash
# 1. schema
python scripts/init_db.py
sqlite3 fungi.db ".tables"     # fgpi readings species_predictions

# 2. one sample (hardware, or add FUNGI_SIMULATE=1 on a bench)
python scripts/sample_loop.py --once

# 3. JSON API
curl -s http://localhost:5000/api/latest | python -m json.tool

# 4. PNG under 3 s once data exists
time curl -s -o /tmp/heatmap.png http://localhost:5000/heatmap.png
file /tmp/heatmap.png

# 5. dashboard + polling attributes
curl -s http://localhost:5000/ | grep -E 'hx-trigger|uplot|htmx'

# 6. assets
du -b static/dashboard.css static/htmx.min.js static/uplot.min.js static/uplot.min.css
# 99636 total on disk; browser devtools shows ~38 KB transferred (gzip)

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

All partials use `hx-trigger="every Ns"` with `hx-swap="innerHTML"`; verifying
with browser devtools shows only the fragment requests, never a full document
reload.

---

## Limitations

- **No detection.** No camera, no image classification, no species
  identification. Species output is environmental niche suitability only.
- **Not an occurrence probability.** Parametric curves are cardinal-range
  heuristics; an RF trained on curve bootstrap is a smoke-test model. Validate
  against local data before acting on it.
- **Microclimate matters.** DHT22 accuracy is roughly ±0.5 °C / ±2 % RH. Enclose
  the sensor with airflow and keep it out of direct sunlight and condensation.
- **FGPI is light-neutral.** Lux is stored and shown, and only mildly modulates
  some species curves via `lux_pref`.
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
