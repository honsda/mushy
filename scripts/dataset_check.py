#!/usr/bin/env python3
"""Validate the FGPI index against a published mould-index dataset.

Why this exists
---------------
`species_curves.json` is literature cardinal ranges. That is defensible for
*ranking* species, but it is nice to show the FGPI axis itself is calibrated
against something independent. The only dataset found that carries the required
columns (T, RH, mould index) under a clear licence is:

  "Data and code for Low-Cost Indoor Air Quality Monitor with Mould Risk
   Prediction on an ESP32" - figshare 10.6084/m9.figshare.35080789, CC BY 4.0.
  Files: air_quality_log.csv (12,934 real ESP32 readings) and
         reference_ashrae_output.csv (8,760 hourly rows, ASHRAE 160 /
         Hukka-Viitanen mould index computed by the authors' reference model).

That dataset is a *hygrothermal formation* model (Hukka-Viitanen), not a
per-species growth curve, so it is used to check monotonic agreement
(Spearman) rather than to fit FGPI coefficients. This keeps the project honest:
FGPI predicts growth *potential*, ASHRAE predicts mould *formation risk*.

Rejected on sight (spec guardrail): anything named like
`PsilocybeEnvironmentalSensitivity` or otherwise AI-generated.

Usage
-----
    python scripts/dataset_check.py                       # download + report
    python scripts/dataset_check.py --csv path/to/file.csv
    python scripts/dataset_check.py --write-fixture       # tiny local sample

The network is dev-machine-only; the Pi never runs this.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.fgpi import compute_fgpi  # noqa: E402

RISK_ORDER = ("low", "moderate", "high", "critical")

DATASET = {
    "doi": "10.6084/m9.figshare.35080789",
    "title": "Low-Cost Indoor Air Quality Monitor with Mould Risk Prediction on an ESP32",
    "license": "CC BY 4.0",
    "landing": "https://figshare.com/articles/dataset/"
    "Data_and_code_for_Low-Cost_Indoor_Air_Quality_Monitor_with_Mould_Risk_"
    "Prediction_on_an_ESP32/35080789",
    "files": {
        "real": "https://ndownloader.figshare.com/files/71710465",  # air_quality_log.csv
        "reference": "https://ndownloader.figshare.com/files/71710534",  # ashrae rows
    },
    "note": "Hukka-Viitanen / ASHRAE 160 mould index, hourly T/RH. Used for "
    "monotonic agreement only; it is not a per-species growth table.",
}

REJECT_MARKERS = ("psilocybe", "aicurated", "ai-generated", "synthetic_gpt")


def _download(url: str, cache: Path) -> str:
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers={"User-Agent": "mushy-dataset-check/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    cache.write_text(text, encoding="utf-8")
    return text


def _rows(csv_text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))
    lower = {name.lower(): name for name in (reader.fieldnames or [])}
    t_key = next((lower[k] for k in lower if k in {"temperature_c", "temperature", "t"}), None)
    rh_key = next((lower[k] for k in lower if k in {"humidity_pct", "humidity", "rh"}), None)
    idx_key = next(
        (lower[k] for k in lower if "index" in k or "mould_score" in k or "mold_score" in k),
        None,
    )
    if not (t_key and rh_key and idx_key):
        raise SystemExit(
            f"need temperature, humidity and a mould index column; got {list(lower)}"
        )
    out: list[dict] = []
    for row in reader:
        try:
            t = float(row[t_key])
            rh = float(row[rh_key])
            idx = float(row[idx_key])
        except (TypeError, ValueError):
            continue
        out.append({"T": t, "RH": rh, "index": idx})
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation without pulling in scipy (tiny and dependency-free)."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        rank = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rank[order[k]] = average
            i = j + 1
        return rank

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def analyse(rows: list[dict], label: str) -> dict:
    """Compare FGPI against a reference mould index.

    The reference index (Hukka-Viitanen / ASHRAE 160) is cumulative: it rises
    while conditions favour growth and only decays slowly, so a raw correlation
    with an instantaneous FGPI is confounded by history. The meaningful test is
    behavioural - when FGPI says the niche is good, does the reference index
    actually climb? So we compare, per risk class, the fraction of hours in
    which the index increased. That should rise monotonically with FGPI.
    """
    order = {name: i for i, name in enumerate(RISK_ORDER)}
    pairs = []
    prev_index: float | None = None
    for row in rows:
        fgpi, risk = compute_fgpi(row["T"], row["RH"], -1.0)
        delta = None if prev_index is None else row["index"] - prev_index
        prev_index = row["index"]
        pairs.append((fgpi, row["index"], risk, delta))
    if not pairs:
        return {"label": label, "rows": 0}

    rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])

    class_level: dict[str, list[float]] = {name: [] for name in order}
    class_rate: dict[str, list[float]] = {name: [] for name in order}
    for fgpi, index, risk, delta in pairs:
        class_level.setdefault(risk, []).append(index)
        # Only measure growth where the reference index has not saturated at
        # its ceiling, otherwise "no growth" is a measurement artifact.
        if delta is not None and index < 1.5:
            class_rate.setdefault(risk, []).append(delta)

    means = {
        risk: round(sum(vals) / len(vals), 3)
        for risk, vals in class_level.items()
        if vals
    }
    growth_per_hour = {
        risk: round(sum(vals) / len(vals), 4)
        for risk, vals in class_rate.items()
        if vals
    }
    ranked = sorted(growth_per_hour, key=lambda r: order.get(r, 99))
    sequence = [growth_per_hour[r] for r in ranked]
    monotonic = all(
        a <= b + 2e-4 for a, b in zip(sequence, sequence[1:])
    ) if len(sequence) > 1 else True
    return {
        "label": label,
        "rows": len(pairs),
        "spearman_rho_cumulative": round(rho, 3),
        "means_by_class": means,
        "growth_per_hour_by_class": growth_per_hour,
        "monotonic": monotonic,
        "fgpi_mean": round(sum(p[0] for p in pairs) / len(pairs), 2),
    }


FIXTURE = """timestamp,temperature_c,humidity_pct,mould_index_06
2026-09-14 06:00,25.6,65.8,0
2026-09-14 07:00,24.1,72.0,0
2026-09-14 08:00,23.0,84.5,1
2026-09-14 09:00,22.4,91.0,2
2026-09-14 10:00,21.8,96.2,3
2026-09-14 11:00,26.0,55.0,0
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="local CSV with T, RH and a mould index column")
    parser.add_argument("--write-fixture", action="store_true", help="write the offline sample")
    args = parser.parse_args(argv[1:])

    fixture = ROOT / "scripts" / "fixtures" / "mould_index_sample.csv"
    if args.write_fixture:
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(FIXTURE, encoding="utf-8")
        print(f"wrote {fixture.relative_to(ROOT)}")
        return 0

    sources: list[tuple[str, str]] = []
    cache_dir = ROOT / ".fungi" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        name = Path(args.csv).name.lower()
        if any(marker in name for marker in REJECT_MARKERS):
            raise SystemExit(f"refusing AI-generated/unrelated dataset: {name}")
        sources.append((Path(args.csv).name, Path(args.csv).read_text(encoding="utf-8")))
    else:
        for key, url in DATASET["files"].items():
            text = _download(url, cache_dir / f"{key}.csv")
            sources.append((f"{DATASET['doi']} ({key})", text))
        fixture_text = FIXTURE if not fixture.exists() else fixture.read_text(encoding="utf-8")
        sources.append(("offline fixture", fixture_text))

    results = [analyse(_rows(text), label) for label, text in sources]
    report = {"dataset": DATASET, "results": results}
    print(json.dumps(report, indent=2))

    # Order matters: report against low -> critical, never alphabetical.
    order = {name: i for i, name in enumerate(RISK_ORDER)}
    for row in results:
        rates = row.get("growth_per_hour_by_class") or {}
        if not rates:
            continue
        ranked = sorted(rates, key=lambda r: order.get(r, 99))
        sequence = [rates[r] for r in ranked]
        if not row.get("monotonic", True):
            print(
                "\nWARNING: FGPI risk classes do not track reference growth rate "
                f"for {row['label']}: {sequence}"
            )
            continue
        # Monotonic but the class thresholds may not match the reference's
        # activation point; say so instead of silently retuning a frozen formula.
        first_active = next((r for r in ranked if rates[r] > 5e-4), None)
        if first_active and first_active != "critical":
            print(
                f"\nCALIBRATION NOTE for {row['label']}:\n"
                f"  Reference growth rate per hour by FGPI class (low..critical): {sequence}\n"
                f"  Growth first becomes measurable at FGPI class '{first_active}'.\n"
                "  FGPI's f_RH starts responding at 40 %RH (smoothstep default)\n"
                "  while Hukka-Viitanen / ASHRAE 160 needs ~80 % surface RH to\n"
                "  activate, so FGPI is an early-warning growth-potential index\n"
                "  rather than a formation-risk index. Flagged, not auto-tuned:\n"
                "  set fgpi.humidity_curve: spec for the literal section 8 ramp."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))