#!/usr/bin/env python3
"""benchmark_paired_stats.py

Paired significance test for the HEADLINE comparison G2T vs LUNA vs CeLEry (the
one the paper actually claims), mirroring ablation_paired_stats.py so both the
benchmark table and the ablation table (Table 1) use the IDENTICAL paired t-test.

For each dataset it loads the per-seed ``extended_metrics.csv`` behind Figs 2/3
(the same timestamp lists plot_method_comparison.py plots), pairs each baseline
method to G2T BY SEED (seed = position in the method's ordered timestamp list;
all methods share the same fixed test split), and runs a two-sided paired t-test
of G2T vs LUNA and G2T vs CeLEry on the shared seeds.

    python benchmark_paired_stats.py --dataset mmc_luna
    python benchmark_paired_stats.py --dataset cns_luna

Writes benchmark_stats_<dataset>.csv and prints a human table with mean ± SD,
Delta = (G2T - comparator), the improvement %, the paired-t p-value and stars.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# metric column -> (display, +1 higher-better / -1 lower-better) -- SAME set/order
# as ablation_paired_stats.py so the two tables are directly comparable.
METRICS = {
    "spearman_mean_of_medians": ("Spearman", +1),
    "f1_mean": ("Contact F1", +1),
    "sum_rssd_mean": ("Sum RSSD", -1),
}
EXTENDED_METRICS_FILENAME = "extended_metrics.csv"

# Canonical per-dataset, per-method seed timestamps. Imported from
# plot_method_comparison.py when possible (single source of truth); the
# hardcoded fallback below is a verbatim copy for when that module's heavy
# imports (matplotlib) are unavailable. KEEP IN SYNC with that file.
ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
_FALLBACK = {
    "scgg": {
        "mmc_luna": ["20260530_165200", "20260530_165210", "20260530_165216",
                     "20260530_165223", "20260530_165229"],
        "cns_luna": ["20260602_142452", "20260602_142505", "20260602_142510",
                     "20260602_142516", "20260602_142522"],
    },
    "luna": {
        "mmc_luna": [],  # auto-discover
        "cns_luna": ["20260601_085512", "20260601_085523", "20260601_085535",
                     "20260601_085541", "20260601_085547"],
    },
    "celery": {
        "mmc_luna": ["20260602_074322", "20260602_074327", "20260602_074332",
                     "20260602_074336", "20260602_074342"],
        "cns_luna": ["20260604_101049", "20260604_141924", "20260604_141952",
                     "20260604_142006", "20260604_141959"],
    },
}
_TS_RE = re.compile(r"^\d{8}_\d{6}(?:_[A-Za-z0-9]+)?$")

try:  # single source of truth for the timestamps/roots
    import plot_method_comparison as _P
    ARTIFACTS_ROOT = _P.ARTIFACTS_ROOT
    _TS = {"scgg": _P.DEFAULT_SCGG_TIMESTAMPS, "luna": _P.DEFAULT_LUNA_TIMESTAMPS,
           "celery": _P.DEFAULT_CELERY_TIMESTAMPS}
    print("[bench-stats] using timestamps from plot_method_comparison.py", file=sys.stderr)
except Exception as e:  # matplotlib/_nature_style not importable here
    _TS = _FALLBACK
    print(f"[bench-stats] plot_method_comparison import failed ({type(e).__name__}); "
          f"using built-in fallback timestamps", file=sys.stderr)


# --- paired test (VERBATIM from ablation_paired_stats.py, so the two tables
# --- use exactly the same statistic) ------------------------------------------
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
    except Exception:
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p, n


def stars(p):
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _discover_luna_timestamps(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"LUNA inference root not found: {root}")
    ts = sorted(p.name for p in root.iterdir() if p.is_dir() and _TS_RE.match(p.name))
    if not ts:
        raise RuntimeError(f"No YYYYMMDD_HHMMSS subdirs under {root}")
    return ts


def load_method(root: Path, timestamps):
    """seed_index -> {metric: value} from each timestamp's extended_metrics.csv."""
    out = {}
    for i, ts in enumerate(timestamps):
        csv = root / ts / EXTENDED_METRICS_FILENAME
        if not csv.exists():
            print(f"[bench-stats] WARN missing {csv} (seed {i}) — skipped", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        if len(df) != 1:
            raise ValueError(f"expected single-row {csv}, got {len(df)}")
        r = df.iloc[0]
        out[i] = {m: float(r[m]) for m in METRICS if m in r.index}
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mmc_luna", choices=("mmc_luna", "cns_luna"))
    ap.add_argument("--reference", default="G2T", help="method all others are tested against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds = args.dataset
    out_csv = Path(args.out) if args.out else here / f"benchmark_stats_{ds}.csv"

    roots = {"G2T": ARTIFACTS_ROOT / ds / "scgg_inference",
             "LUNA": ARTIFACTS_ROOT / ds / "luna_inference",
             "CeLEry": ARTIFACTS_ROOT / ds / "celery_inference"}
    ts_scgg = list(_TS["scgg"].get(ds, []))
    ts_luna = list(_TS["luna"].get(ds, [])) or _discover_luna_timestamps(roots["LUNA"])
    ts_cel = list(_TS["celery"].get(ds, []))
    data = {"G2T": load_method(roots["G2T"], ts_scgg),
            "LUNA": load_method(roots["LUNA"], ts_luna),
            "CeLEry": load_method(roots["CeLEry"], ts_cel)}

    ref = args.reference
    others = [m for m in ("G2T", "LUNA", "CeLEry") if m != ref]
    shared = sorted(set(data[ref]).intersection(*[set(data[m]) for m in others]))
    print(f"[bench-stats] dataset={ds}: shared seeds across all methods = "
          f"{len(shared)} {shared}")
    if len(shared) < 2:
        raise SystemExit("Need >=2 shared seeds across all methods for a paired test.")

    def vals(method, col):
        return [data[method][s][col] for s in shared]

    order = [ref] + others
    rows = []
    for m in order:
        rec = {"method": m, "n_seeds": len(shared)}
        for col, (disp, direction) in METRICS.items():
            v = np.array(vals(m, col), float)
            rec[f"{col}_mean"] = float(v.mean())
            rec[f"{col}_sd"] = float(v.std(ddof=1))
            if m != ref:
                rv = np.array(vals(ref, col), float)          # reference (G2T)
                t, p, n = paired_t(list(rv), list(v))          # G2T vs comparator
                delta = float(rv.mean() - v.mean())            # (G2T - comparator)
                denom = abs(v.mean()) if v.mean() != 0 else float("nan")
                pct = (delta if direction > 0 else -delta) / denom * 100.0
                rec.update({f"{col}_delta": delta, f"{col}_pct": pct,
                            f"{col}_better": (delta * direction) > 0,
                            f"{col}_t": t, f"{col}_p": p, f"{col}_sig": stars(p)})
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    label = {"mmc_luna": "MMC cortex", "cns_luna": "Mouse CNS"}[ds]
    print(f"\n{label} — mean ± SD over {len(shared)} shared seeds; paired t-test of "
          f"each method vs {ref} (Delta = {ref} - method; %/stars from that)\n")
    for _, r in out.iterrows():
        head = f"{r['method']:7s} n={int(r['n_seeds']):>2d}"
        cells = []
        for col, (disp, direction) in METRICS.items():
            c = f"{r[f'{col}_mean']:.4f}±{r[f'{col}_sd']:.4f}"
            if r["method"] != ref and pd.notna(r.get(f"{col}_p")):
                c += (f"{r.get(f'{col}_sig','')}(Δ{r[f'{col}_delta']:+.4f}, "
                      f"{r[f'{col}_pct']:+.1f}%, p={r[f'{col}_p']:.3f})")
            cells.append(f"{disp}: {c}")
        print(head + "  |  " + "   ".join(cells))
    print(f"\n[bench-stats] wrote {out_csv}")
    print(f"[bench-stats] Δ>0 means {ref} better on Spearman/F1; Δ<0 means {ref} "
          f"better on Sum RSSD (lower-is-better). % is the improvement of {ref}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
