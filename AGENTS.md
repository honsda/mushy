# AGENTS.md — working agreement for this repo

Read this instead of re-exploring the tree or re-reading the spec. The spec is
archived verbatim in `docs/SPEC.md`; `README.md` covers BOM, wiring, install,
training, BT PAN. Load only the section you need.

## Token discipline (keeps quality, cuts cost)

1. **Verify, don't narrate.** `python scripts/verify.py` is the whole §15 gate
   (11 checks, ~10 s). Paste its output instead of hand-running curls.
2. **Grep before you read.** `rg -n "pattern" app/` beats opening whole files.
   Only read a file when you are about to edit it, and read the region you edit.
3. **One edit call per file.** Batch all changes to a file into a single `edit`
   (or one `write` for full rewrites). No read-edit-read loops.
4. **Don't restate the repo.** Reference paths and line numbers, not contents.
5. **Run once, at the end.** Batch shell checks; avoid re-running the verifier
   after cosmetic edits.
6. **Answer in the diff's shape:** what changed, where, and the verify line.
   No per-file walkthroughs unless asked.

## Layout map (do not re-derive)

| Need | File |
|---|---|
| Flask factory + all routes | `app/server.py` |
| Config load/merge/cache | `app/__init__.py` |
| SQLite helpers + schema | `app/db.py` |
| FGPI formula | `app/models/fgpi.py` |
| Species ML/parametric | `app/models/species.py`, `app/models/trained/species_curves.json` |
| IDW/cubic grid | `app/spatial/interpolate.py` |
| Matplotlib PNG | `app/spatial/heatmap.py` |
| Sensors (simulate flag) | `app/sensors/dht22.py`, `app/sensors/light.py` |
| Dashboard + HTMX partials | `app/templates/` |
| Vendored JS/CSS | `static/` (precompiled, no build on Pi) |
| Ops | `scripts/`, `systemd/` |

## Commands

```bash
python scripts/verify.py -v          # full acceptance gate
FUNGI_SIMULATE=1 python scripts/sample_loop.py --once
FUNGI_SIMULATE=1 python -m flask --app app.server:create_app run   # dev server
python scripts/train_species.py      # offline, dev machine only
```

## Invariants (break one → the change is wrong)

- Signatures in `docs/SPEC.md` §7 are frozen; downstream code depends on them.
- No camera/image ML, no SPA, no CDN/cloud at runtime, no npm on the Pi.
- Copy says "predicted suitability", never "detected".
- Frontend budget: <100 KB transferred (currently ~46.6 KB gzipped + heatmap).
- Sensors fail open: `read_lux()` returns `-1.0`; sampler logs and skips.
- Partials poll independently (5 s cards / 10 s species / 30 s heatmap); never
  reload the whole document.

## Not automatable — a human on the Pi must check

- `systemctl is-enabled/is-active fungi-server fungi-sampler` after reboot.
- Same dashboard over Wi-Fi and Bluetooth PAN (`http://10.10.10.1:5000`).
- Real DHT22/BH1750 read once with `sensors.simulate: false`.