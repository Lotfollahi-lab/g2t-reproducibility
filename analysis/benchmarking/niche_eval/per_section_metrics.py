#!/usr/bin/env python3
"""per_section_metrics.py

Recompute niche NMI/ARI WITHIN each section (then macro-average across sections),
from the per-cell labels already saved by cellcharter_niche_eval.py — NO
CellCharter re-run needed.

Why: niche_eval_metrics.csv scores GLOBALLY (all 14 CNS sections pooled). Because
those sections are different tissues with largely section-specific GT regions, the
pooled score gets credit for merely separating sections, which inflates absolute
values and compresses differences between coordinate sources. Scoring within each
section removes that between-section signal and is far more sensitive to
coordinate quality.

    python per_section_metrics.py --root niche_eval_out

Reads   root/seed*/niche_labels.csv  (cols: section, gt_region, niche_<source>...)
Writes  root/niche_per_section_per_seed.csv        (source x section x seed)
        root/niche_per_section_across_seeds.csv     (macro-avg over sections, ±SD over seeds)
        root/niche_per_section_NMI_table.csv        (source x section, mean over seeds)
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (normalized_mutual_info_score as nmi,
                             adjusted_rand_score as ari)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="niche_eval_out")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(root / "seed*" / "niche_labels.csv")))
    if not files:
        raise SystemExit(f"No seed*/niche_labels.csv under {root}")

    recs = []
    for f in files:
        m = re.search(r"seed(\d+)", f)
        seed = int(m.group(1)) if m else -1
        df = pd.read_csv(f)
        if "section" not in df or "gt_region" not in df:
            raise SystemExit(f"{f} missing 'section'/'gt_region' columns")
        sources = [c[len("niche_"):] for c in df.columns if c.startswith("niche_")]
        for src in sources:
            col = f"niche_{src}"
            for sec, g in df.groupby("section"):
                # within-section scoring needs >=2 GT regions and >=2 predicted niches
                if g["gt_region"].nunique() < 2 or g[col].nunique() < 2:
                    continue
                recs.append(dict(seed=seed, source=src, section=str(sec), n=len(g),
                                 NMI=float(nmi(g["gt_region"], g[col])),
                                 ARI=float(ari(g["gt_region"], g[col]))))
    per = pd.DataFrame(recs)
    if per.empty:
        raise SystemExit("No scorable (section, source) groups (every section had "
                         "<2 GT regions or <2 niches).")
    per.to_csv(out / "niche_per_section_per_seed.csv", index=False)

    order = ["reference", "luna", "g2t", "celery", "shuffled"]

    # macro-average over sections within each (seed, source), then mean±SD over seeds
    macro = per.groupby(["seed", "source"])[["NMI", "ARI"]].mean().reset_index()
    present = [s for s in order if s in set(macro["source"])] + \
              [s for s in macro["source"].unique() if s not in order]
    rows = []
    for s in present:
        sub = macro[macro["source"] == s]
        rows.append(dict(source=s, n_seeds=int(sub.shape[0]),
                         n_sections=int(per[per["source"] == s]["section"].nunique()),
                         NMI_mean=float(sub["NMI"].mean()), NMI_std=float(sub["NMI"].std(ddof=1)),
                         ARI_mean=float(sub["ARI"].mean()), ARI_std=float(sub["ARI"].std(ddof=1))))
    agg = pd.DataFrame(rows)
    agg.to_csv(out / "niche_per_section_across_seeds.csv", index=False)

    # per (source x section) NMI, averaged over seeds — shows where methods differ
    tab = (per.groupby(["section", "source"])["NMI"].mean()
              .unstack("source").reindex(columns=present))
    tab.to_csv(out / "niche_per_section_NMI_table.csv")

    print("Per-section (within-section) NMI/ARI, macro-averaged over sections "
          f"then over seeds  [{macro['seed'].nunique()} seeds, "
          f"{per['section'].nunique()} sections]:\n")
    print(agg.to_string(index=False))
    print("\nPer-section NMI (mean over seeds), one row per section:\n")
    print(tab.round(3).to_string())
    print(f"\n[per-section] wrote 3 CSVs under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
