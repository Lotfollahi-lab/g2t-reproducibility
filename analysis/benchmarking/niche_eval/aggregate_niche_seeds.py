#!/usr/bin/env python3
"""aggregate_niche_seeds.py

Aggregate the per-seed CellCharter niche-eval outputs (one seed<S>/ dir per LSF
job from submit_cellcharter_eval.sh) into a single mean±SD table per coordinate
source, plus a paper-ready grouped bar chart with SD error bars.

    python aggregate_niche_seeds.py --root niche_eval_out

Writes, under --root (or --out):
    niche_eval_per_seed.csv       every seed's row, tidy
    niche_eval_across_seeds.csv   mean / SD of NMI & ARI per source
    niche_eval_across_seeds.png/pdf
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./niche_eval_out",
                    help="dir containing seed*/niche_eval_metrics.csv")
    ap.add_argument("--out", default=None, help="output dir (default = --root)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(root / "seed*" / "niche_eval_metrics.csv")))
    if not files:
        raise SystemExit(f"No seed*/niche_eval_metrics.csv under {root} "
                         f"(did the per-seed jobs finish?)")
    rows = []
    for f in files:
        m = re.search(r"seed(\d+)", f)
        df = pd.read_csv(f)
        df["seed"] = int(m.group(1)) if m else -1
        rows.append(df)
    alld = pd.concat(rows, ignore_index=True)
    alld.to_csv(out / "niche_eval_per_seed.csv", index=False)
    print(f"[agg] {len(files)} seed file(s); "
          f"{alld['seed'].nunique()} seeds x {alld['source'].nunique()} sources")

    order = ["reference", "luna", "g2t", "celery"]
    present = [s for s in order if s in set(alld["source"])] + \
              [s for s in alld["source"].unique() if s not in order]
    g = alld.groupby("source")
    agg = pd.DataFrame({
        "source":    present,
        "n_seeds":   [int(g.get_group(s).shape[0]) for s in present],
        "NMI_mean":  [float(g.get_group(s)["NMI"].mean()) for s in present],
        "NMI_std":   [float(g.get_group(s)["NMI"].std(ddof=1)) for s in present],
        "ARI_mean":  [float(g.get_group(s)["ARI"].mean()) for s in present],
        "ARI_std":   [float(g.get_group(s)["ARI"].std(ddof=1)) for s in present],
    })
    agg.to_csv(out / "niche_eval_across_seeds.csv", index=False)
    print("\n" + agg.to_string(index=False))
    print(f"\n[agg] wrote {out/'niche_eval_across_seeds.csv'} and "
          f"{out/'niche_eval_per_seed.csv'}")

    # ---- bar chart (mean, SD error bars); colours match the benchmark figs ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        try:
            from cellcharter_niche_eval import METHOD_COLORS
        except Exception:
            METHOD_COLORS = {"g2t": "#D55E00", "luna": "#0072B2",
                             "celery": "#009E73", "reference": "#999999"}

        x = np.arange(len(present)); w = 0.38
        col = lambda s: METHOD_COLORS.get(s, "#4C78A8")
        fig, ax = plt.subplots(figsize=(1.7 * len(present) + 2.0, 4.6))
        ax.bar(x - w / 2, agg["NMI_mean"], w, yerr=agg["NMI_std"], capsize=3,
               color=[col(s) for s in present], edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, agg["ARI_mean"], w, yerr=agg["ARI_std"], capsize=3,
               color=[col(s) for s in present], edgecolor="black", linewidth=0.4,
               hatch="//")
        for i, s in enumerate(present):
            ax.annotate(f"{agg['NMI_mean'][i]:.3f}", (x[i] - w / 2, agg["NMI_mean"][i]),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
            ax.annotate(f"{agg['ARI_mean'][i]:.3f}", (x[i] + w / 2, agg["ARI_mean"][i]),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        if float(agg[["NMI_mean", "ARI_mean"]].to_numpy().min()) < 0:
            ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(present, fontsize=10)
        ax.set_ylabel("agreement with GT tissue regions")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5); ax.set_axisbelow(True)
        ax.legend(handles=[Patch(facecolor="white", edgecolor="black", label="NMI"),
                           Patch(facecolor="white", edgecolor="black", hatch="//",
                                 label="ARI")], frameon=False, fontsize=9, loc="upper right")
        ax.set_title(f"CellCharter niche recovery vs GT tissue regions "
                     f"(CNS test; mean±SD over {int(agg['n_seeds'].max())} seeds)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "niche_eval_across_seeds.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / "niche_eval_across_seeds.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[agg] wrote {out/'niche_eval_across_seeds.png'} and .pdf")
    except Exception as e:
        print(f"[agg] plot skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
