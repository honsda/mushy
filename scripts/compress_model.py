#!/usr/bin/env python3
"""Compress a trained species model for the Pi (offline dev-machine step).

Two modes, because they make different promises:

* ``--data train.csv``: refit a shallower forest on the same data with
  ``max_depth`` and ``max_leaf_nodes`` caps, then compare macro-F1 against the
  uncompressed model on a stratified holdout. The compression is **rejected**
  when macro-F1 drops more than ``--max-drop`` (default 2 points), so a smaller
  file can never silently mean a worse predictor.

* no ``--data``: lossless joblib recompression only (``compress=3``). This
  typically cuts the file by half and is always safe. No accuracy claim is
  made because there is nothing to validate against — the script says so.

The Pi only ever loads the result; it never runs this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_csv(path: Path):
    import csv

    rows, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        species_col = next((header[k] for k in ("species", "label", "name") if k in header), None)
        t_col = next((header[k] for k in ("temperature", "temp", "t") if k in header), None)
        rh_col = next((header[k] for k in ("humidity", "rh") if k in header), None)
        lux_col = next((header[k] for k in ("lux", "light") if k in header), None)
        weights_col = next((header[k] for k in ("weight",) if k in header), None)
        if not (species_col and t_col and rh_col):
            raise SystemExit(f"{path}: need species, temperature and humidity columns")
        weights = []
        for row in reader:
            try:
                t = float(row[t_col])
                rh = float(row[rh_col])
            except (TypeError, ValueError):
                continue
            species = (row.get(species_col) or "").strip()
            if not species:
                continue
            lux = 0.0
            if lux_col:
                try:
                    lux = max(0.0, float(row[lux_col]))
                except (TypeError, ValueError):
                    lux = 0.0
            rows.append([t, rh, lux])
            labels.append(species)
            weights.append(float(row[weights_col]) if weights_col and row.get(weights_col) else 1.0)
    return np.asarray(rows, dtype=float), labels, np.asarray(weights, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", dest="dst", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="training CSV for validation")
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-leaves", type=int, default=4096)
    parser.add_argument("--max-nodes", type=int, default=50000, help="reported budget")
    parser.add_argument("--max-drop", type=float, default=0.02, help="allowed macro-F1 loss")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise SystemExit("Install scikit-learn and joblib on the dev machine first")

    if not args.src.is_file():
        raise SystemExit(f"{args.src} not found")

    original_size = args.src.stat().st_size
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    if args.data is None:
        model = joblib.load(args.src)
        joblib.dump(model, args.dst, compress=3)
        new_size = args.dst.stat().st_size
        print(f"Recompressed {args.src.name}: {original_size} -> {new_size} bytes")
        print("Lossless mode: no accuracy claim (pass --data to validate a refit).")
        _write_meta(args.dst, original_size, new_size, None, args)
        return 0

    X, y, weights = load_csv(args.data)
    if len(set(y)) < 2:
        raise SystemExit("need at least two classes to validate")
    original = joblib.load(args.src)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.25, random_state=args.seed, stratify=y
    )

    # The reference must be trained on the same split. Evaluating the shipped
    # model directly would score it on rows it memorised, and every refit would
    # look like a regression.
    try:
        reference = RandomForestClassifier(**original.get_params())
        reference.fit(X_train, y_train, sample_weight=w_train)
    except (AttributeError, TypeError) as exc:
        raise SystemExit(
            f"{args.src.name} is not a scikit-learn RandomForest ({exc}); "
            "update --data/--in or validate it another way"
        )
    baseline = f1_score(
        y_test, reference.predict(X_test), average="macro", zero_division=0
    )

    compressed = RandomForestClassifier(
        n_estimators=max(50, len(getattr(original, "estimators_", []) or [50])),
        max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaves,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    compressed.fit(X_train, y_train, sample_weight=w_train)
    after = f1_score(y_test, compressed.predict(X_test), average="macro", zero_division=0)
    drop = baseline - after

    total_nodes = sum(int(tree.tree_.node_count) for tree in compressed.estimators_)
    print(f"Baseline macro-F1 {baseline:.4f} | compressed {after:.4f} | drop {drop:+.4f}")
    print(f"Compressed forest: {len(compressed.estimators_)} trees, {total_nodes} nodes")
    if total_nodes > args.max_nodes:
        print(f"WARNING: {total_nodes} nodes exceeds the {args.max_nodes} budget")
    if drop > args.max_drop:
        print(f"REJECTED: macro-F1 drop {drop:.4f} > allowed {args.max_drop:.4f}")
        print("Model not written. Lower --max-depth is not the answer; add data or raise --max-leaves.")
        return 2

    joblib.dump(compressed, args.dst, compress=3)
    new_size = args.dst.stat().st_size
    print(f"Wrote {args.dst.name}: {original_size} -> {new_size} bytes")
    _write_meta(
        args.dst,
        original_size,
        new_size,
        {
            "baseline_macro_f1": round(baseline, 4),
            "compressed_macro_f1": round(after, 4),
            "macro_f1_drop": round(drop, 4),
            "trees": len(compressed.estimators_),
            "nodes": total_nodes,
        },
        args,
        X=X,
    )
    return 0


def _write_meta(dst: Path, before: int, after: int, scores: dict | None, args, X=None) -> None:
    """Keep the .meta.json next to the model so confidence scoring keeps working."""
    src_meta = args.src.with_suffix(".meta.json")
    meta: dict = {}
    if src_meta.is_file():
        try:
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
    meta.update(
        {
            "compressed_from": args.src.name,
            "bytes_before": before,
            "bytes_after": after,
            "compression_ratio": round(before / after, 2) if after else None,
            "compression_scores": scores,
            "max_depth": args.max_depth,
            "max_leaves": args.max_leaves,
        }
    )
    if X is not None and len(X):
        meta["feature_ranges"] = {
            "temperature": [round(float(X[:, 0].min()), 2), round(float(X[:, 0].max()), 2)],
            "humidity": [round(float(X[:, 1].min()), 2), round(float(X[:, 1].max()), 2)],
            "lux": [round(float(X[:, 2].min()), 2), round(float(X[:, 2].max()), 2)],
        }
    dst.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
