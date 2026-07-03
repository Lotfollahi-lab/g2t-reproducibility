#!/usr/bin/env python3
"""run_ablation_analysis.py

ONE-SHOT ablation analysis: score every run, build the comparison table, and
render every figure needed to compare the G2T ablations. Run this after the
LSF jobs from ``run_ablations.sh`` have all finished.

Pipeline (all steps run by default; toggle with flags):
  1. SCORE     -> calls compute_extended_metrics.py for every timestamp in the
                  manifest, writing <artifacts>/<dataset>/scgg_inference/<TS>/
                  extended_metrics.csv (Spearman / Contact F1 / Sum RSSD).
                  Reuses the SAME scorer as the main paper figures.
  2. AGGREGATE -> mean +/- SEM per ablation across seeds, signed Delta vs
                  baseline; writes ablation_comparison.csv + ablation_per_seed.csv
                  and prints a table.
  3. BARPLOT   -> grouped bar chart, one panel per metric, baseline highlighted,
                  Delta-vs-baseline annotated -> ablation_comparison.{png,pdf}.
  4. SVG (opt) -> qualitative side-by-side of the top spatially-variable genes on
                  the true vs generated layout for one section, via
                  plot_svg_comparison.py (only if --svg_section is given).

Usage:
  python run_ablation_analysis.py                          # score + aggregate + barplot
  python run_ablation_analysis.py --skip_score             # reuse existing metrics
  python run_ablation_analysis.py --workers 4              # parallelise scoring
  python run_ablation_analysis.py --svg_section mouse2_slice229 --svg_top_k 4
  python run_ablation_analysis.py --dataset cns_luna       # (if you also ablate on CNS)

Everything writes into --outdir (default: this script's directory).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Metric columns pulled from each run's extended_metrics.csv (written by
# compute_extended_metrics.py; see scgg/src/scgg/evaluation/luna_metrics.py
# :: aggregate_slices). direction = +1 higher-is-better, -1 lower-is-better.
# ---------------------------------------------------------------------------
METRICS = {
    "spearman_mean_of_medians": ("Spearman (mean-of-medians)", +1),
    "f1_mean": ("Contact F1", +1),
    "sum_rssd_mean": ("Sum RSSD (per cell class)", -1),
}

BASELINE_COLOR = "#D55E00"   # vermillion — the G2T colour in Figs 2/3
                             # (plots/plot_method_comparison.py :: METHODS["G2T"])
OTHER_COLOR = "#4C78A8"      # steel blue for the (non-method) ablation variants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sem(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def _delta_label(m: float, base: float | None, direction: int) -> str:
    """Compact vs-baseline label. Small changes -> signed % (sign-flipped so
    + always means BETTER). Extreme changes (>~10x, e.g. a collapsed layout
    on the unbounded Sum RSSD) -> a readable fold ("Nx worse/better") instead
    of an absurd percentage like -789373%."""
    if base is None or base == 0 or math.isnan(m) or math.isnan(base):
        return ""
    v = direction * (m - base) / abs(base) * 100.0   # + = better
    if abs(v) <= 999:
        return f"{v:+.1f}%"
    fold = m / base
    if fold >= 1:
        word = "worse" if direction < 0 else "better"
        return f"{fold:.0f}x {word}"
    word = "better" if direction < 0 else "worse"
    return f"{(1.0 / fold):.0f}x {word}"


# ---------------------------------------------------------------------------
# Step 1: score
# ---------------------------------------------------------------------------
def score_runs(man: pd.DataFrame, compute_script: Path, dataset: str,
               skip_existing: bool, workers: int) -> None:
    ts_list = [t for t in man["timestamp"].astype(str).tolist()
               if t and t != "UNKNOWN_PARSE_FAILED"]
    ts_list = list(dict.fromkeys(ts_list))  # dedup, preserve order
    if not ts_list:
        raise SystemExit("No usable timestamps in the manifest to score.")
    if not compute_script.exists():
        raise SystemExit(f"compute_extended_metrics.py not found: {compute_script}\n"
                         f"Pass --compute_script with the correct path.")
    cmd = [sys.executable, str(compute_script),
           "--dataset", dataset, "--methods", "g2t",
           "--scgg_timestamps", ",".join(ts_list)]
    if skip_existing:
        cmd.append("--skip_existing")
    if workers and workers > 1:
        cmd += ["--workers", str(workers)]
    print(f"[score] scoring {len(ts_list)} run(s) via compute_extended_metrics.py ...")
    print("        " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        # Partial failure is expected when the manifest lists timestamps whose run
        # dir was removed (e.g. a re-submitted batch). Don't abort — the aggregate
        # step below uses only the runs that actually have extended_metrics.csv and
        # reports the rest as 'missing'.
        print(f"[score] WARNING: compute_extended_metrics.py exited {rc}; some runs "
              f"failed to score (commonly manifest timestamps whose run dir no longer "
              f"exists). Continuing with the runs that DID score.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 2: aggregate
# ---------------------------------------------------------------------------
def load_per_run(man: pd.DataFrame, inf_root: Path) -> pd.DataFrame:
    per_run, missing = [], []
    for _, r in man.iterrows():
        ts = str(r["timestamp"])
        agg_csv = inf_root / ts / "extended_metrics.csv"
        if not agg_csv.exists():
            missing.append((r["ablation"], r["seed"], ts))
            continue
        row = pd.read_csv(agg_csv).iloc[0].to_dict()
        rec = {"ablation": r["ablation"], "seed": r["seed"], "timestamp": ts}
        for col in METRICS:
            rec[col] = float(row.get(col, float("nan")))
        per_run.append(rec)
    if missing:
        print(f"[aggregate] WARNING: extended_metrics.csv missing for "
              f"{len(missing)} run(s) (did scoring run / succeed?):")
        for ab, s, ts in missing[:10]:
            print(f"            - {ab} seed{s} ({ts})")
    if not per_run:
        raise SystemExit("No per-run metrics found. Run without --skip_score first.")
    df = pd.DataFrame(per_run)
    # De-dup (ablation, seed) — runs submitted in >1 batch (e.g. a 5-seed then a
    # 10-seed batch) record the same seed twice; keep the last SCORED run of each.
    ndup = int(df.duplicated(["ablation", "seed"]).sum())
    if ndup:
        print(f"[aggregate] {ndup} duplicate (ablation, seed) run(s) from multiple "
              f"submission batches — keeping the last scored one of each.")
        df = df.drop_duplicates(["ablation", "seed"], keep="last").reset_index(drop=True)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ablation, g in df.groupby("ablation"):
        rec = {"ablation": ablation, "n_seeds": len(g)}
        for col in METRICS:
            vals = [v for v in g[col].tolist() if not math.isnan(v)]
            rec[f"{col}_mean"] = sum(vals) / len(vals) if vals else float("nan")
            rec[f"{col}_sem"] = _sem(vals)
        rows.append(rec)
    out = pd.DataFrame(rows)
    # baseline first, then alphabetical
    out["__ord"] = out["ablation"].apply(lambda a: (0 if a == "baseline" else 1, a))
    return out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)


def print_table(out: pd.DataFrame, dataset: str) -> None:
    print("\n" + "=" * 82)
    print(f"G2T ablation comparison  (dataset={dataset}, mean +/- SEM over seeds)")
    print("=" * 82)
    base = out[out["ablation"] == "baseline"]
    base_vals = ({col: float(base[f"{col}_mean"].iloc[0]) for col in METRICS}
                 if len(base) else {})
    header = f"{'ablation':14s} {'n':>2s}"
    for col, (disp, direction) in METRICS.items():
        header += f"  {disp + (' (up)' if direction > 0 else ' (down)'):>30s}"
    print(header)
    print("-" * len(header))
    for _, r in out.iterrows():
        line = f"{r['ablation']:14s} {int(r['n_seeds']):>2d}"
        for col, (disp, direction) in METRICS.items():
            m, sem = r[f"{col}_mean"], r[f"{col}_sem"]
            cell = f"{m:.4f} +/- {sem:.4f}"
            if base_vals and r["ablation"] != "baseline":
                lbl = _delta_label(m, base_vals.get(col), direction)
                if lbl:
                    cell += f" ({lbl})"
            line += f"  {cell:>30s}"
        print(line)
    print("=" * 82)
    print("(% in parentheses is signed so POSITIVE = better than baseline;")
    print(" Spearman/Contact F1 higher-is-better, Sum RSSD lower-is-better.)\n")


# ---------------------------------------------------------------------------
# Step 3: bar chart
# ---------------------------------------------------------------------------
def make_barplot(out: pd.DataFrame, out_png: Path, dataset: str) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ablations = out["ablation"].tolist()
    x = np.arange(len(ablations))
    base = out[out["ablation"] == "baseline"]
    has_base = len(base) > 0
    colors = [BASELINE_COLOR if a == "baseline" else OTHER_COLOR for a in ablations]

    ncols = len(METRICS)
    fig, axes = plt.subplots(1, ncols, figsize=(4.7 * ncols, 4.6))
    if ncols == 1:
        axes = [axes]

    for ax, (col, (disp, direction)) in zip(axes, METRICS.items()):
        means = out[f"{col}_mean"].to_numpy(dtype=float)
        sems = out[f"{col}_sem"].to_numpy(dtype=float)
        ax.bar(x, means, yerr=sems, capsize=3, color=colors,
               edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1})
        better = "higher is better" if direction > 0 else "lower is better"
        ax.set_title(f"{disp}\n({better})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ablations, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        # Auto log-scale when one ablation degenerates and blows up the range
        # (e.g. edm_off / K2 on Sum RSSD), so the other bars stay readable.
        pos = means[np.isfinite(means) & (means > 0)]
        if pos.size >= 2 and pos.max() / pos.min() > 30:
            ax.set_yscale("log")

        base_val = float(base[f"{col}_mean"].iloc[0]) if has_base else None
        for xi, (a, m, sem) in enumerate(zip(ablations, means, sems)):
            if not math.isfinite(m):
                txt = "n/a"
            elif a == "baseline":
                txt = "ref"
            else:
                txt = _delta_label(m, base_val, direction)
            if txt:
                top = m + (sem if math.isfinite(sem) else 0.0)
                ax.annotate(txt, (xi, top), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"G2T ablations — {dataset}  (mean ± SEM over seeds; "
        f"% = change vs baseline, + is better)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")  # vector for the paper
    plt.close(fig)
    print(f"[barplot] wrote {out_png}")
    print(f"[barplot] wrote {out_png.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Step 4: optional qualitative SVG panel
# ---------------------------------------------------------------------------
def make_svg(man: pd.DataFrame, inf_root: Path, svg_script: Path, outdir: Path,
             ablation: str, seed, section: str, top_k: int) -> None:
    if not svg_script.exists():
        print(f"[svg] SKIP: {svg_script} not found (pass --svg_script).")
        return
    sel = man[(man["ablation"] == ablation)
              & (man["seed"].astype(str) == str(seed))]
    if not len(sel):
        print(f"[svg] SKIP: no manifest row for ablation={ablation} seed={seed}.")
        return
    ts = str(sel["timestamp"].iloc[0])
    run_root = inf_root / ts
    test_csv = run_root / "work" / "test.csv"
    if not test_csv.exists():
        cand = list(run_root.rglob("test.csv"))
        if cand:
            test_csv = cand[0]
        else:
            print(f"[svg] SKIP: no test.csv found under {run_root}.")
            return
    out_png = outdir / f"svg_{ablation}_seed{seed}_{section}.png"
    cmd = [sys.executable, str(svg_script),
           "--test_csv", str(test_csv), "--pred_dir", str(run_root),
           "--section", section, "--top_k", str(top_k), "--out", str(out_png)]
    print(f"[svg] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[svg] wrote {out_png}")


# ---------------------------------------------------------------------------
# Optional: top-2 MDS variance fraction (how 2-D-embeddable D_hat is).
# NOTE: this cannot be computed from the metric CSVs (they don't contain the
# predicted distance matrix). It aggregates the per-slice
# "[mds_var] top2_frac=<x>" lines that the EDM head prints at eval when run
# with SCGG_LOG_MDS_VAR=1 (see scgg/src/models/edm_head.py).
# ---------------------------------------------------------------------------
def report_mds_variance(pattern: str, outdir: Path) -> None:
    import glob as _glob
    import re as _re
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"[mds-var] no files matched {pattern!r}")
        return
    fracs = []
    for f in files:
        try:
            txt = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        for tok in _re.findall(r"top2_frac=([0-9]*\.?[0-9]+)", txt):
            v = float(tok)
            if v == v:  # drop NaN
                fracs.append(v)
    if not fracs:
        print(f"[mds-var] matched {len(files)} file(s) but found no "
              f"'top2_frac=' lines. Did the eval run with SCGG_LOG_MDS_VAR=1?")
        return
    n = len(fracs)
    mean = sum(fracs) / n
    sd = (sum((x - mean) ** 2 for x in fracs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    print("\n" + "=" * 72)
    print(f"[mds-var] top-2 MDS variance fraction over {n} slice(s): "
          f"{100*mean:.1f} +/- {100*sd:.1f}%  "
          f"(min {100*min(fracs):.1f}%, max {100*max(fracs):.1f}%)")
    print("[mds-var] paper sentence (paste into Section 2.4):")
    print(f'   "the top two MDS eigenvalues capture {100*mean:.1f}\\% (SD '
          f'{100*sd:.1f}) of the positive-eigenvalue variance of '
          f'$\\widehat{{\\mathbf{{D}}}}$ on the held-out slices, confirming the '
          f'predicted matrices are near-2-D-embeddable."')
    print("=" * 72)
    out_csv = outdir / "mds_variance_summary.csv"
    out_csv.write_text("n_slices,mean_top2_frac,sd_top2_frac\n"
                       f"{n},{mean:.6f},{sd:.6f}\n")
    print(f"[mds-var] wrote {out_csv}")


# ---------------------------------------------------------------------------
def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(here / "ablation_manifest.csv"))
    ap.add_argument("--artifacts_root",
                    default="/nfs/team361/sb75/scgg-reproducibility/artifacts")
    ap.add_argument("--dataset", default="mmc_luna")
    ap.add_argument("--outdir", default=str(here))
    ap.add_argument("--compute_script",
                    default=str(here.parent / "plots" / "compute_extended_metrics.py"))
    ap.add_argument("--svg_script", default=str(here / "plot_svg_comparison.py"))
    # step toggles
    ap.add_argument("--skip_score", action="store_true",
                    help="Reuse existing extended_metrics.csv; don't re-score.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Passed to compute_extended_metrics.py: skip already-scored runs.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Scoring workers (passed to compute_extended_metrics.py).")
    ap.add_argument("--no_barplot", action="store_true")
    # optional SVG panel
    ap.add_argument("--svg_section", default=None,
                    help="Section name to render the true-vs-generated SVG panel for.")
    ap.add_argument("--svg_ablation", default="baseline")
    ap.add_argument("--svg_seed", default="0")
    ap.add_argument("--svg_top_k", type=int, default=4)
    # optional: top-2 MDS variance fraction (reviewer request)
    ap.add_argument("--mds_var_logs", default=None,
                    help="Glob of eval log files with '[mds_var] top2_frac=' lines "
                         "(from an SCGG_LOG_MDS_VAR=1 eval run); aggregates them.")
    ap.add_argument("--mds_var_only", action="store_true",
                    help="With --mds_var_logs, only run that aggregation and exit.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional: aggregate the top-2 MDS variance fraction emitted by the EDM
    # head under SCGG_LOG_MDS_VAR=1. Can run standalone via --mds_var_only.
    if args.mds_var_logs:
        report_mds_variance(args.mds_var_logs, outdir)
        if args.mds_var_only:
            return 0

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")
    inf_root = Path(args.artifacts_root) / args.dataset / "scgg_inference"
    man = pd.read_csv(manifest)

    # Manifest sanity: warn on duplicate timestamps (the collision bug).
    dups = man["timestamp"].astype(str)
    dups = dups[dups != "UNKNOWN_PARSE_FAILED"]
    dup_ts = dups[dups.duplicated(keep=False)].unique().tolist()
    if dup_ts:
        print(f"[warn] {len(dup_ts)} timestamp(s) appear on >1 manifest row — "
              f"those runs shared an output dir and clobbered each other: {dup_ts}\n"
              f"       Results for them are unreliable; consider re-running "
              f"(SUBMIT_DELAY spacing in run_ablations.sh).")

    # 1) score
    if not args.skip_score:
        score_runs(man, Path(args.compute_script), args.dataset,
                   args.skip_existing, args.workers)
    else:
        print("[score] skipped (--skip_score); using existing extended_metrics.csv.")

    # 2) aggregate
    df = load_per_run(man, inf_root)
    out = aggregate(df)
    print_table(out, args.dataset)
    comp_csv = outdir / "ablation_comparison.csv"
    seed_csv = outdir / "ablation_per_seed.csv"
    out.to_csv(comp_csv, index=False)
    df.sort_values(["ablation", "seed"]).to_csv(seed_csv, index=False)
    print(f"[aggregate] wrote {comp_csv}")
    print(f"[aggregate] wrote {seed_csv}")

    # 3) bar chart
    if not args.no_barplot:
        make_barplot(out, outdir / "ablation_comparison.png", args.dataset)

    # 4) optional qualitative SVG
    if args.svg_section:
        make_svg(man, inf_root, Path(args.svg_script), outdir,
                 args.svg_ablation, args.svg_seed, args.svg_section, args.svg_top_k)

    print("\n[done] Outputs in:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
