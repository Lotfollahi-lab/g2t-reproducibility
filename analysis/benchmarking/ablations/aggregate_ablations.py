#!/usr/bin/env python3
"""aggregate_ablations.py

Read the per-run ``extended_metrics.csv`` files produced by
``compute_extended_metrics.py`` for every (ablation, seed) listed in
``ablation_manifest.csv``, and print/write one tidy comparison table:
for each ablation, the mean +/- SEM across its 5 seeds of the three
headline G2T metrics.

Columns pulled from extended_metrics.csv (see
scgg/src/scgg/evaluation/luna_metrics.py :: aggregate_slices):
    spearman_mean_of_medians   (higher better)  -> "Spearman"
    f1_mean                    (higher better)  -> "Contact F1"
    sum_rssd_mean              (lower  better)  -> "Sum RSSD (per cell class)"

Workflow:
    1. bash run_ablations.sh                         # submit jobs, write manifest
    2. python compute_extended_metrics.py ...        # score each run timestamp
    3. python aggregate_ablations.py                 # this script

Usage:
    python aggregate_ablations.py \
        [--manifest ablation_manifest.csv] \
        [--artifacts_root /nfs/team361/sb75/scgg-reproducibility/artifacts] \
        [--dataset mmc_luna] \
        [--out ablation_comparison.csv]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

# Metric column -> (display name, direction) where direction is +1 if
# higher-is-better, -1 if lower-is-better (only affects the printed arrow).
METRICS = {
    "spearman_mean_of_medians": ("Spearman (mean-of-medians)", +1),
    "f1_mean": ("Contact F1", +1),
    "sum_rssd_mean": ("Sum RSSD (per cell class)", -1),
}


def _sem(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--manifest", default=str(here / "ablation_manifest.csv"))
    ap.add_argument("--artifacts_root",
                    default="/nfs/team361/sb75/scgg-reproducibility/artifacts")
    ap.add_argument("--dataset", default="mmc_luna")
    ap.add_argument("--out", default=str(here / "ablation_comparison.csv"))
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    inf_root = Path(args.artifacts_root) / args.dataset / "scgg_inference"

    # Collect per-run metric rows.
    per_run = []
    missing = []
    for _, r in man.iterrows():
        ts = str(r["timestamp"])
        agg_csv = inf_root / ts / "extended_metrics.csv"
        if not agg_csv.exists():
            missing.append((r["ablation"], r["seed"], ts, str(agg_csv)))
            continue
        row = pd.read_csv(agg_csv).iloc[0].to_dict()
        rec = {"ablation": r["ablation"], "seed": r["seed"], "timestamp": ts}
        for col in METRICS:
            rec[col] = float(row.get(col, float("nan")))
        per_run.append(rec)

    if missing:
        print("WARNING: extended_metrics.csv missing for "
              f"{len(missing)} run(s) — did compute_extended_metrics.py run?")
        for ab, s, ts, p in missing[:10]:
            print(f"   - {ab} seed{s} ({ts}): {p}")
        print()

    if not per_run:
        print("No per-run metrics found. Run compute_extended_metrics.py first.")
        return 1

    df = pd.DataFrame(per_run)

    # Aggregate mean +/- SEM across seeds, per ablation.
    rows = []
    for ablation, g in df.groupby("ablation"):
        rec = {"ablation": ablation, "n_seeds": len(g)}
        for col in METRICS:
            vals = [v for v in g[col].tolist() if not math.isnan(v)]
            rec[f"{col}_mean"] = sum(vals) / len(vals) if vals else float("nan")
            rec[f"{col}_sem"] = _sem(vals)
        rows.append(rec)
    out = pd.DataFrame(rows)

    # Order ablations: baseline first, then alphabetical.
    out["__ord"] = out["ablation"].apply(lambda a: (0 if a == "baseline" else 1, a))
    out = out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    # ---- Pretty print ----
    print("\n" + "=" * 78)
    print(f"G2T ablation comparison  (dataset={args.dataset}, mean +/- SEM over seeds)")
    print("=" * 78)
    header = f"{'ablation':14s} {'n':>2s}"
    for col, (disp, direction) in METRICS.items():
        arrow = "↑" if direction > 0 else "↓"
        header += f"  {disp+' '+arrow:>30s}"
    print(header)
    print("-" * len(header))
    base = out[out["ablation"] == "baseline"]
    base_vals = {col: float(base[f"{col}_mean"].iloc[0]) for col in METRICS} if len(base) else {}
    for _, r in out.iterrows():
        line = f"{r['ablation']:14s} {int(r['n_seeds']):>2d}"
        for col, (disp, direction) in METRICS.items():
            m, sem = r[f"{col}_mean"], r[f"{col}_sem"]
            cell = f"{m:.4f} ± {sem:.4f}"
            # delta vs baseline (signed so + always means "better").
            if base_vals and r["ablation"] != "baseline" and not math.isnan(m):
                raw = m - base_vals[col]
                better = raw * direction  # positive => better than baseline
                cell += f" ({'+' if better >= 0 else ''}{better:+.4f})".replace("++", "+")
            line += f"  {cell:>30s}"
        print(line)
    print("=" * 78)
    print("Δ in parentheses is signed so a POSITIVE value = better than baseline")
    print("(Spearman/Contact F1 higher-is-better; Sum RSSD lower-is-better).\n")

    out.to_csv(args.out, index=False)
    print(f"Wrote per-ablation table: {args.out}")
    df.sort_values(["ablation", "seed"]).to_csv(
        Path(args.out).with_name("ablation_per_seed.csv"), index=False)
    print(f"Wrote per-seed table:     {Path(args.out).with_name('ablation_per_seed.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
