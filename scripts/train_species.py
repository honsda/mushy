#!/usr/bin/env python3
"""Offline RandomForest training for species suitability (dev machine only).

Output: app/models/trained/species_rf.joblib
Feature order is fixed: [temperature_C, relative_humidity_percent, lux].

Two input modes
---------------
1. --data path.csv (preferred)
   CSV with columns species/T/RH/lux (common aliases accepted). Use real,
   published datasets only, e.g. the Figshare *Aspergillus* distribution
   extract (`training_data_species.csv`) or Data INRAE *Fusarium*
   ecophysiology records.

2. no --data (bootstrap)
   Samples each species' documented cardinal niche from species_curves.json.
   This produces a usable model for pipeline testing, but it is a curve
   bootstrap, not an occurrence dataset. Say so when reporting results.

Guardrail: AI-generated datasets are rejected, including the known
`PsilocybeEnvironmentalSensitivity` example. Training is never run on the Pi
at runtime; copy the resulting joblib into app/models/trained/ if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.species import CURVES_PATH, RF_PATH  # noqa: E402

# East Java bounding box (lat, lon). Used by --region to keep only local rows.
EAST_JAVA_BBOX = (-8.8, -6.5, 111.0, 115.0)

FORBIDDEN_MARKERS = (
    "psilocybe",
    "environmentalsensitivity",
    "environmental_sensitivity",
    "ai-generated",
    "ai_generated",
)


def _guard(label: str, path: str = "") -> None:
    haystack = f"{path} {label}".lower()
    for marker in FORBIDDEN_MARKERS:
        if marker in haystack:
            raise SystemExit(
                f"Refusing dataset matching forbidden marker '{marker}'. "
                "Only published, non-AI-generated datasets may be used."
            )


def _pick(header: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in header:
            return header[name]
    return None


def _in_east_java(lat, lon) -> bool:
    if lat is None or lon is None:
        return False
    lat_min, lat_max, lon_min, lon_max = EAST_JAVA_BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def load_csv(
    path: Path, region: str | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    _guard("", str(path))
    rows: list[list[float]] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row")
        header = {h.strip().lower(): h for h in reader.fieldnames}
        species_col = _pick(header, "species", "label", "name")
        temp_col = _pick(header, "temperature", "temp", "t")
        rh_col = _pick(header, "humidity", "rh", "relative_humidity")
        lux_col = _pick(header, "lux", "light")
        lat_col = _pick(header, "latitude", "lat", "decimallatitude")
        lon_col = _pick(header, "longitude", "lon", "lng", "decimalLongitude".lower())
        if not species_col or not temp_col or not rh_col:
            raise SystemExit("CSV must contain species, temperature and humidity columns")

        for row in reader:
            label = (row.get(species_col) or "").strip()
            if not label:
                continue
            _guard(label)
            try:
                t = float(row[temp_col])
                rh = float(row[rh_col])
                lux = float(row[lux_col]) if lux_col and row.get(lux_col) else 0.0
            except (TypeError, ValueError):
                continue
            if not (-20.0 <= t <= 60.0) or not (0.0 <= rh <= 100.0):
                continue
            if region == "east_java":
                if not (lat_col and lon_col):
                    raise SystemExit(
                        "--region east_java needs latitude/longitude columns in the CSV"
                    )
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                except (TypeError, ValueError):
                    # No coordinates: a label collected by this deployment is
                    # already in the region, so keep it. Only records that
                    # claim a location outside the bbox are dropped.
                    pass
                else:
                    if not _in_east_java(lat, lon):
                        continue
            rows.append([t, rh, max(0.0, lux)])
            labels.append(label)

    if not rows:
        raise SystemExit("No valid rows parsed from CSV")
    return np.asarray(rows, dtype=float), labels, sorted(set(labels))


def load_synthetic(n_per_species: int, seed: int) -> tuple[np.ndarray, list[str], list[str]]:
    with CURVES_PATH.open("r", encoding="utf-8") as fh:
        curves = json.load(fh)["species"]
    if not curves:
        raise SystemExit(f"No species curves in {CURVES_PATH}")

    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[str] = []
    for curve in curves:
        t_min, t_opt, t_max = map(float, (curve["T_min"], curve["T_opt"], curve["T_max"]))
        rh_min, rh_opt = map(float, (curve["RH_min"], curve["RH_opt"]))
        lux = curve.get("lux_pref") or {}
        lux_min = float(lux.get("min", 0.0))
        lux_opt = float(lux.get("opt", 50.0))
        lux_max = float(lux.get("max", 1000.0))
        for _ in range(n_per_species):
            # Triangular draws put more samples near the documented optimum,
            # which makes the RF posterior track the parametric niche shape.
            t = rng.triangular(t_min, t_opt, t_max)
            rh = rng.triangular(rh_min, rh_opt, min(100.0, rh_opt + 3.0))
            if lux_max <= lux_opt:
                light = rng.uniform(0.0, 1000.0)
            else:
                light = rng.triangular(max(0.0, lux_min), lux_opt, lux_max)
            rows.append([t, rh, light])
            labels.append(str(curve["name"]))
    return np.asarray(rows, dtype=float), labels, sorted(set(labels))


def _cv_report(model, X, labels, folds: int, seed: int) -> dict:
    """Stratified CV with per-class precision/recall and macro-F1."""
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, X, labels, cv=splitter, n_jobs=-1)
    report = classification_report(
        labels, predicted, output_dict=True, zero_division=0
    )
    macro_f1 = float(f1_score(labels, predicted, average="macro", zero_division=0))
    print("")
    print(f"{folds}-fold CV (macro-F1 {macro_f1:.3f})")
    for name, metrics in sorted(report.items()):
        if not isinstance(metrics, dict):
            continue
        print(
            f"  {name:<34} precision {metrics.get('precision', 0):.3f}"
            f"  recall {metrics.get('recall', 0):.3f}"
            f"  f1 {metrics.get('f1-score', 0):.3f}"
            f"  n={int(metrics.get('support', 0))}"
        )
    return {"macro_f1": round(macro_f1, 4), "per_class": {
        k: v for k, v in report.items() if isinstance(v, dict) and k not in {"accuracy"}
    }, "folds": folds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="real training CSV")
    parser.add_argument("--out", type=Path, default=RF_PATH, help="output joblib path")
    parser.add_argument("--n", type=int, default=400, help="samples per species in bootstrap mode")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument(
        "--region",
        default=None,
        choices=["default", "east_java"],
        help="east_java filters CSV rows to the East Java bbox and names the meta file",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="stratified k-fold CV (0 = legacy single score, 5 recommended)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from joblib import dump
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        raise SystemExit("Install scikit-learn and joblib on the dev machine first")

    if args.data:
        X, labels, classes = load_csv(args.data, region=args.region)
        provenance = f"CSV {args.data}"
    else:
        X, labels, classes = load_synthetic(max(50, args.n), args.seed)
        provenance = f"bootstrap from {CURVES_PATH.name} (not an occurrence dataset)"

    if len(classes) < 2:
        raise SystemExit("Need at least two species/classes to train a classifier")
    print(f"Loaded {len(labels)} samples, {len(classes)} classes ({provenance})")

    model = RandomForestClassifier(
        n_estimators=max(50, args.trees),
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    cv_summary: dict | None = None
    if args.cv and args.cv >= 2:
        folds = min(int(args.cv), min(labels.count(c) for c in classes))
        if folds >= 2:
            cv_summary = _cv_report(model, X, labels, folds, args.seed)
        else:
            print("Not enough samples per class for CV; skipping")
    else:
        folds = min(3, min(labels.count(c) for c in classes))
        if folds >= 2:
            scores = cross_val_score(
                model,
                X,
                labels,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed),
            )
            print(f"{folds}-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")

    # Companion metadata: confidence scoring reads feature_ranges at runtime,
    # and a reviewer can see exactly what the model was trained on.
    meta = {
        "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "region": args.region or "default",
        "provenance": provenance,
        "rows": int(X.shape[0]),
        "classes": classes,
        "class_counts": {c: int(labels.count(c)) for c in classes},
        "feature_order": ["temperature_C", "relative_humidity_percent", "lux"],
        "feature_ranges": {
            "temperature": [round(float(np.min(X[:, 0])), 2), round(float(np.max(X[:, 0])), 2)],
            "humidity": [round(float(np.min(X[:, 1])), 2), round(float(np.max(X[:, 1])), 2)],
            "lux": [round(float(np.min(X[:, 2])), 2), round(float(np.max(X[:, 2])), 2)],
        },
        "trees": int(args.trees),
        "cv": cv_summary,
        "warning": "predicted niche suitability, not detection",
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    print(f"Wrote {meta_path}")
    print("Feature order: [temperature_C, relative_humidity_percent, lux]")
    print("Reminder: predict_proba ranks niche suitability; it is not detection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
