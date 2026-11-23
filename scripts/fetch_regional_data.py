#!/usr/bin/env python3
"""Fetch and filter regional data for East Java (offline-first, stdlib only).

Two subcommands, both explicit about what they can and cannot provide:

  --filter CSV --out CSV
      Apply the East Java bounding box to a published CSV that already has
      latitude/longitude (e.g. a Figshare *Aspergillus* distribution extract).
      Use this to build ``data/regional/aspergillus_indonesia.csv``.

  --gbif
      Download GBIF occurrence records for the target genera inside the East
      Java bbox, saved to ``data/regional/gbif_east_java.csv``.

GBIF records are **occurrence** data: a species was found at a place and date.
They carry no temperature or humidity, so they do not become training rows
directly. They answer a different question — "is this species actually
recorded here?" — which is what makes a regional curve defensible instead of
guessed. Use them to check the overlay, not to fit it.

Anything matching the project's AI-dataset blocklist is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "regional"

EAST_JAVA = {"lat_min": -8.8, "lat_max": -6.5, "lon_min": 111.0, "lon_max": 115.0}
TAXA = [
    "Aspergillus",
    "Fusarium",
    "Eurotium",
    "Stachybotrys",
    "Aspergillus flavus",
    "Fusarium oxysporum",
]
FORBIDDEN = ("psilocybe", "environmentalsensitivity", "ai-generated", "ai_generated")
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"


def guard(path: Path) -> None:
    haystack = str(path).lower()
    for marker in FORBIDDEN:
        if marker in haystack:
            raise SystemExit(f"Refusing '{path}': matches forbidden marker '{marker}'")


def in_bbox(lat: float, lon: float) -> bool:
    return (
        EAST_JAVA["lat_min"] <= lat <= EAST_JAVA["lat_max"]
        and EAST_JAVA["lon_min"] <= lon <= EAST_JAVA["lon_max"]
    )


def _find(header: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in header:
            return header[name]
    return None


def filter_csv(src: Path, out: Path) -> int:
    guard(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        lat_col = _find(header, "latitude", "lat", "decimallatitude")
        lon_col = _find(header, "longitude", "lon", "lng", "decimallongitude")
        if not (lat_col and lon_col):
            raise SystemExit(f"{src}: no latitude/longitude columns to filter on")
        with out.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])
                except (TypeError, ValueError):
                    continue
                if in_bbox(lat, lon):
                    writer.writerow(row)
                    kept += 1
    print(f"{src.name}: kept {kept} rows inside the East Java bbox -> {out}")
    return kept


def fetch_gbif(limit_per_taxon: int = 300) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gbif_east_java.csv"
    total = 0
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["taxon", "scientific_name", "lat", "lon", "year", "basis_of_record"])
        for taxon in TAXA:
            params = {
                "q": taxon,
                "decimalLatitude": f"{EAST_JAVA['lat_min']},{EAST_JAVA['lat_max']}",
                "decimalLongitude": f"{EAST_JAVA['lon_min']},{EAST_JAVA['lon_max']}",
                "hasCoordinate": "true",
                "limit": str(min(300, limit_per_taxon)),
            }
            url = GBIF_URL + "?" + urllib.parse.urlencode(params)
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    payload = json.load(response)
            except Exception as exc:
                print(f"  {taxon}: GBIF unavailable ({exc})", file=sys.stderr)
                continue
            results = payload.get("results") or []
            for record in results:
                try:
                    lat = float(record["decimalLatitude"])
                    lon = float(record["decimalLongitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not in_bbox(lat, lon):
                    continue
                writer.writerow(
                    [
                        taxon,
                        record.get("scientificName") or record.get("species") or taxon,
                        lat,
                        lon,
                        record.get("year") or "",
                        record.get("basisOfRecord") or "",
                    ]
                )
                total += 1
            print(f"  {taxon}: {len(results)} records returned, running total {total}")
    print(f"Wrote {out} ({total} occurrences)")
    print("Occurrences confirm presence only; they carry no T/RH. Not a training set.")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filter", type=Path, default=None, help="CSV to bbox-filter")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gbif", action="store_true", help="download GBIF occurrences")
    parser.add_argument("--limit", type=int, default=300, help="records per taxon")
    args = parser.parse_args()

    if args.filter:
        out = args.out or OUT_DIR / f"{args.filter.stem}_east_java.csv"
        filter_csv(args.filter, out)
    if args.gbif:
        fetch_gbif(args.limit)
    if not args.filter and not args.gbif:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
