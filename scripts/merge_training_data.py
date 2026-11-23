#!/usr/bin/env python3
"""Merge human labels into a regional training set (offline dev-machine step).

Two rules that matter for label quality
--------------------------------------
1. A positive label (`observed=1`) becomes a training row for that species.
2. A negative (`observed=0`) becomes a negative example for that species only
   when another species was observed positive at the same reading. Otherwise
   "we did not look for it" would be recorded as "it was absent", which is
   censoring bias and teaches the model to under-predict rare species.

The base CSV supplies the regional prior; labels carry local ground truth and
are weighted higher (``--label-weight``) so a few dozen observations can move
the model without erasing the published distribution.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

COLUMNS = ["species", "temperature", "humidity", "lux", "weight", "origin"]


def _norm(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_base(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        species_col = next(
            (header[k] for k in ("species", "label", "name") if k in header), None
        )
        t_col = next((header[k] for k in ("temperature", "temp", "t") if k in header), None)
        rh_col = next((header[k] for k in ("humidity", "rh") if k in header), None)
        lux_col = next((header[k] for k in ("lux", "light") if k in header), None)
        lat_col = next(
            (header[k] for k in ("latitude", "lat", "decimallatitude") if k in header), None
        )
        lon_col = next(
            (header[k] for k in ("longitude", "lon", "lng") if k in header), None
        )
        if not (species_col and t_col and rh_col):
            raise SystemExit(f"{path}: need species, temperature and humidity columns")
        for row in reader:
            species = (row.get(species_col) or "").strip()
            t, rh = _norm(row.get(t_col)), _norm(row.get(rh_col))
            if not species or t is None or rh is None:
                continue
            lux = _norm(row.get(lux_col)) if lux_col else None
            entry = {
                "species": species,
                "temperature": t,
                "humidity": rh,
                "lux": lux if lux is not None and lux >= 0 else 0.0,
                "weight": 1.0,
                "origin": "base",
            }
            # Coordinates survive the merge so the result can still be
            # region-filtered at training time.
            lat = _norm(row.get(lat_col)) if lat_col else None
            lon = _norm(row.get(lon_col)) if lon_col else None
            if lat is not None and lon is not None:
                entry["latitude"] = lat
                entry["longitude"] = lon
            rows.append(entry)
    return rows


def read_labels(path: Path, label_weight: float) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    if "observed" not in rows[0]:
        raise SystemExit(f"{path}: not a labels export (missing 'observed')")

    positives: dict[str, set[str]] = {}
    for row in rows:
        if str(row.get("observed", "")).strip() in {"1", "True", "true"}:
            positives.setdefault(str(row.get("reading_id")), set()).add(
                str(row.get("species") or "").strip()
            )

    out: list[dict] = []
    skipped = 0
    for row in rows:
        species = str(row.get("species") or "").strip()
        t, rh = _norm(row.get("temperature")), _norm(row.get("humidity"))
        reading_id = str(row.get("reading_id") or "")
        observed = str(row.get("observed", "")).strip() in {"1", "True", "true"}
        if not species or t is None or rh is None:
            skipped += 1
            continue
        if not observed:
            others = positives.get(reading_id, set()) - {species}
            if not others:
                skipped += 1  # uninformative negative: nobody looked
                continue
        lux = _norm(row.get("lux"))
        out.append(
            {
                "species": species,
                "temperature": t,
                "humidity": rh,
                "lux": lux if lux is not None and lux >= 0 else 0.0,
                "weight": float(label_weight) if observed else float(label_weight) * 0.5,
                "origin": "label" if observed else "label-negative",
            }
        )
    if skipped:
        print(f"Skipped {skipped} label rows without context or usable negatives")
    return out


def balanced_weights(rows: list[dict]) -> dict[str, float]:
    counts = Counter(row["species"] for row in rows)
    total = sum(counts.values()) or 1
    classes = len(counts) or 1
    return {
        species: round(total / (classes * count), 4) for species, count in counts.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", type=Path, required=True, help="regional/global prior CSV")
    parser.add_argument("--labels", type=Path, required=True, help="/api/labels/export.csv")
    parser.add_argument("--out", type=Path, required=True, help="combined training CSV")
    parser.add_argument("--label-weight", type=float, default=5.0)
    args = parser.parse_args()

    base = read_base(args.base)
    labels = read_labels(args.labels, args.label_weight)
    rows = base + labels
    if not rows:
        raise SystemExit("Nothing to write: base and labels were both empty")

    # Per-row weight = base weight * inverse class frequency, so a rare
    # species is not drowned out by whichever species dominates the export.
    multipliers = balanced_weights(rows)
    for row in rows:
        row["weight"] = round(row["weight"] * multipliers[row["species"]], 4)

    fieldnames = list(COLUMNS)
    if any("latitude" in row for row in rows):
        fieldnames += ["latitude", "longitude"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["origin"] for row in rows)
    print(f"Wrote {args.out}: {len(rows)} rows")
    print(f"  base={counts['base']} labels={counts['label']} negatives={counts['label-negative']}")
    print("  species distribution:")
    for species, count in sorted(Counter(r["species"] for r in rows).items()):
        print(f"    {species:<34} {count:5d}  weight x{multipliers[species]:.2f}")
    print("Next: python scripts/train_species.py --data", args.out, "--cv 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
