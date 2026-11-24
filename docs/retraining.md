# Retraining workflow

Everything here runs on a dev machine, never on the Pi at runtime. The Pi only
loads the resulting `.joblib` and optionally updates the small SGD adapter in
place (see [On-device adaptation](#on-device-adaptation)).

```
labels CSV ──┐
             ├─ merge_training_data.py ─→ combined.csv ─→ train_species.py
regional CSV ┘                                                   │
                                                                 ├─→ *.joblib
                                                    compress_model.py
                                                                 └─→ *.meta.json
```

## 0. Collect labels on the dashboard

Each species row has a **record sighting** control: `✓ present` / `✗ absent`,
a 1–5 confidence slider and an optional note. That writes to the `labels`
table (manual entry only; the Pi has no camera and never infers a label).

Export them:

```bash
TOKEN=$(grep -A2 'ingest_token' config.yaml | head -1)   # or set MUSHY_INGEST_TOKEN
curl -H "Authorization: Bearer $TOKEN" \
     "http://pi:5000/api/labels/export.csv?since=2025-01-01" -o labels.csv
```

The export joins each label to the reading that was current when it was
recorded, so it carries `temperature`, `humidity`, `lux`, `x`, `y`, `node_id`.

## 1. Get a regional prior

```bash
# Filter a published occurrence extract to the East Java bbox
python scripts/fetch_regional_data.py \
  --filter ~/downloads/training_data_species.csv \
  --out data/regional/aspergillus_indonesia.csv

# Presence check for the local genera (no T/RH; not a training set)
python scripts/fetch_regional_data.py --gbif
```

`--gbif` answers "is this species actually recorded in East Java?". It cannot
answer "at what temperature does it grow" — occurrences carry no T/RH. Use it
to sanity-check the regional overlay, not to fit a model.

## 2. Merge labels with the prior

```bash
python scripts/merge_training_data.py \
  --base data/regional/aspergillus_indonesia.csv \
  --labels labels.csv \
  --out data/regional/combined.csv \
  --label-weight 5
```

Rules that keep labels honest:

- a `present` label becomes a training row for that species;
- an `absent` label becomes a negative **only if another species was marked
  present at the same reading** — otherwise "nobody looked" would be recorded
  as "not there" (censoring bias);
- labels are weighted above the prior (`--label-weight`) and rows are
  rebalanced by inverse class frequency, so a rare species is not drowned out.

## 3. Train with cross-validation

```bash
python scripts/train_species.py \
  --data data/regional/combined.csv \
  --region east_java \
  --cv 5 \
  --trees 250 \
  --out app/models/trained/species_rf_east_java.joblib
```

Prints per-class precision/recall and macro-F1. Writes
`species_rf_east_java.meta.json` next to the model with: timestamp, row count,
class distribution, feature ranges and the CV report. The feature ranges are
what `confidence: high` means at runtime — an input inside them has evidence
behind it, an input outside them does not.

## 4. Compress for the Pi

```bash
python scripts/compress_model.py \
  --in  app/models/trained/species_rf_east_java.joblib \
  --data data/regional/combined.csv \
  --out app/models/trained/species_rf_east_java.min.joblib \
  --max-depth 12 --max-leaves 4096
```

Refits a shallower forest and **rejects the result** when macro-F1 drops more
than 2 points (exit code 2). Without `--data` it only recompresses losslessly
and makes no accuracy claim — the script says which mode it used.

## 5. Deploy and reload

```bash
scp app/models/trained/species_rf_east_java*.{joblib,json} \
    pi:~/fungi-forecast/app/models/trained/
ssh pi 'sudo systemctl restart fungi-server'
# then point the config at the compressed file:
#   species.model_path: app/models/trained/species_rf_east_java.min.joblib
```

The loader hot-reloads when the joblib mtime changes, so a restart is only
needed to be certain.

## On-device adaptation

`fungi-incremental.timer` runs `scripts/nightly_sgd_update.py` nightly at
02:00. It calls `SGDClassifier.partial_fit` on labels that arrived since the
last run and blends the result with the RF posterior
(`species.blend_sgd_weight`, default 0.3).

It skips unless **all** of these hold:

| Guard | Default | Config |
|---|---|---|
| New labels since last run | ≥ 20 | `species.min_new_labels` |
| Free RAM | ≥ 300 MB | `species.sgd.min_free_mb` |
| SoC temperature | < 70 °C | `species.sgd.max_soc_temp_c` |
| At least two species in the batch | — | — |

Every run appends one line to `logs/incremental.log` with the label count,
RSS and duration. A run costs roughly 4 W for a few seconds; 02:00 keeps it
clear of the daily thermal peak.

## Runtime inference order

1. **SGD adapter**, if it exists, is younger than 72 h and blending is enabled:
   `score = 0.7 × RF + 0.3 × SGD`.
2. **Regional RF**, if `species_rf_east_java.joblib` exists (or `model_path`).
3. **Parametric overlay**, always available; this is the fallback on a fresh
   install.

`backend_name()` in the dashboard header names whichever path answered.

## What "confidence" means

| Level | Condition |
|---|---|
| `high` | a trained model answered and the input is inside its training range |
| `medium` | parametric curves, input comfortably inside the niche envelope |
| `low` | near/outside the envelope, or a model extrapolating |

Confidence describes the **evidence**, not the score. A `low` 90 % is still
just a hypothesis.
