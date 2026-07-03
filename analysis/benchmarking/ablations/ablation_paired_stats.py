#!/usr/bin/env python3
"""ablation_paired_stats.py

Turn per-seed ablation metrics (ablation_per_seed.csv) into the paper's Table 1
statistics: for each ablation, mean ± SD across seeds, and a PAIRED t-test vs
baseline on the SHARED seeds (pairing by seed controls for the seed effect, which
a two-sample test would not). Significance stars: * p<.05, ** p<.01, *** p<.001.

    python ablation_paired_stats.py                       # reads ./ablation_per_seed.csv
    python ablation_paired_stats.py --per_seed path.csv --baseline baseline

Writes ablation_stats.csv (machine-readable) and prints a human table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# metric column -> (display, +1 higher-better / -1 lower-better)
METRICS = {
    "spearman_mean_of_medians": ("Spearman", +1),
    "f1_mean": ("Contact F1", +1),
    "sum_rssd_mean": ("Sum RSSD", -1),
}


def paired_t(a, b):
    """Two-sided paired t-test on aligned lists a,b. Returns (t, p, n)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    if sd == 0:
        return (float("inf"), 0.0, n) if md != 0 else (0.0, 1.0, n)
    t = md / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), df=n - 1))
    except Exception:  # scipy missing -> normal approximation (fine at n=10)
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p, n


def stars(p):
    if p != p:  # NaN
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_seed", default=str(here / "ablation_per_seed.csv"))
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--min_seeds", type=int, default=0,
                    help="only report ablations with >= this many distinct seeds "
                         "(baseline always kept). Use 10 to restrict to the "
                         "fully-run core set.")
    ap.add_argument("--out", default=str(here / "ablation_stats.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.per_seed)
    for req in ("ablation", "seed", *METRICS):
        if req not in df.columns:
            raise SystemExit(f"{args.per_seed} missing column {req!r} "
                             f"(have {list(df.columns)})")
    # De-dup repeated (ablation, seed) rows (e.g. a run re-submitted); keep last.
    ndup = int(df.duplicated(["ablation", "seed"]).sum())
    if ndup:
        print(f"[stats] WARNING: {ndup} duplicate (ablation, seed) row(s) in "
              f"{Path(args.per_seed).name} — keeping the last of each.")
        df = df.drop_duplicates(["ablation", "seed"], keep="last")
    # Show exactly what we're working with, so a stale/short CSV is obvious.
    sc = df.groupby("ablation")["seed"].nunique()
    print(f"[stats] {Path(args.per_seed).name}: {len(df)} rows; distinct seeds per "
          f"ablation:\n" + sc.to_string())
    if int(sc.max()) < 10:
        print(f"[stats] NOTE: max {int(sc.max())} seeds found (<10). If you expected "
              f"10, re-run `python run_ablation_analysis.py` so ablation_per_seed.csv "
              f"is regenerated from the current (10-seed) manifest.", file=sys.stderr)

    # optional: keep only ablations with enough seeds (baseline always kept)
    if args.min_seeds > 0:
        keep = set(sc[sc >= args.min_seeds].index) | {args.baseline}
        dropped = sorted(set(df["ablation"]) - keep)
        if dropped:
            print(f"[stats] --min_seeds {args.min_seeds}: dropping "
                  f"{dropped} (fewer than {args.min_seeds} seeds).")
        df = df[df["ablation"].isin(keep)]

    base = df[df["ablation"] == args.baseline].set_index("seed")
    if base.empty:
        raise SystemExit(f"No '{args.baseline}' rows in {args.per_seed}")

    order = [args.baseline] + sorted(x for x in df["ablation"].unique()
                                     if x != args.baseline)
    rows = []
    for ab in order:
        g = df[df["ablation"] == ab].set_index("seed")
        rec = {"ablation": ab, "n_seeds": int(g.shape[0])}
        for col, (disp, direction) in METRICS.items():
            vals = g[col].dropna()
            rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if ab != args.baseline:
                shared = sorted(set(g.index) & set(base.index))
                pairs = [(g.loc[s, col], base.loc[s, col]) for s in shared
                         if pd.notna(g.loc[s, col]) and pd.notna(base.loc[s, col])]
                if pairs:
                    aa, bb = zip(*pairs)
                    t, p, n = paired_t(list(aa), list(bb))
                    rec[f"{col}_delta"] = rec[f"{col}_mean"] - float(base[col].mean())
                    rec[f"{col}_better"] = (rec[f"{col}_delta"] * direction) > 0
                    rec[f"{col}_t"] = t
                    rec[f"{col}_p"] = p
                    rec[f"{col}_n_pair"] = n
                    rec[f"{col}_sig"] = stars(p)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    # ---- human-readable table ------------------------------------------------
    print(f"\nAblation stats — mean ± SD over seeds; paired t-test vs "
          f"'{args.baseline}' on shared seeds (*/**/*** = p<.05/.01/.001)\n")
    for _, r in out.iterrows():
        head = f"{r['ablation']:12s} n={int(r['n_seeds']):>2d}"
        cells = []
        for col, (disp, direction) in METRICS.items():
            c = f"{r[f'{col}_mean']:.4f}±{r[f'{col}_sd']:.4f}"
            if r["ablation"] != args.baseline and f"{col}_p" in r and pd.notna(r.get(f"{col}_p")):
                c += f"{r.get(f'{col}_sig','')}(Δ{r[f'{col}_delta']:+.4f},p={r[f'{col}_p']:.3f})"
            cells.append(f"{disp}: {c}")
        print(head + "  |  " + "   ".join(cells))
    print(f"\n[stats] wrote {args.out}")
    print("[stats] Δ is (ablation − baseline); a change is 'better' when it moves "
          "the metric in the improving direction (Spearman/F1 up, Sum RSSD down).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
