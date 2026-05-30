"""compare_per_slice_metrics.py

Compare per-slice metrics between a LUNA inference run and a G2T (scGG)
inference run, then rank slices by the per-slice improvement
(G2T − LUNA) with the largest gains on top.

Reads:
    <ARTIFACTS_ROOT>/<dataset>/luna_inference/<LUNA_TS>/per_slice_metrics.csv
    <ARTIFACTS_ROOT>/<dataset>/scgg_inference/<SCGG_TS>/per_slice_metrics.csv

Writes (into OUT_DIR, default
    <ARTIFACTS_ROOT>/<dataset>/comparison_plots/per_slice_comparisons/):

    per_slice_delta_<LUNA_TS>_vs_<SCGG_TS>.csv
        Tidy long form: section_label, n_cells, luna, g2t, delta, abs_delta.
        Rows sorted by delta descending (biggest G2T wins first).

    per_slice_delta_<LUNA_TS>_vs_<SCGG_TS>.txt
        Human-readable ranking — same content as stdout, captured to
        disk so the ranking survives even if you lose the LSF stdout.

Per-slice metrics column choice:
    The pipeline writes ``per_slice_metrics.csv`` with at least:
        section_label, n_cells, spearman_per_cell_median,
        spearman_per_cell_mean
    There is NO "pearson" column in this CSV today — Spearman is the
    only correlation flavour the LUNA paper / scgg pipeline emit. The
    user request asked for "pearson delta"; we default to Spearman
    (per-cell median, the LUNA Fig. 3 building block) because that's
    what's actually on disk. Override with --metric_col if you want
    the mean variant or a future-added pearson column.

LSF: tiny CPU-only pandas job (~seconds). Submit with
``submit_compare_per_slice.sh`` in the sibling lsf/ dir.

Usage:
    python compare_per_slice_metrics.py \\
        --luna_timestamp 20260526_085131 \\
        --scgg_timestamp 20260530_165229
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Inputs (edit if you point this at a different artifacts root)
# ──────────────────────────────────────────────────────────────────────

ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
DATASET = "mmc_luna"

LUNA_INFERENCE_ROOT = ARTIFACTS_ROOT / DATASET / "luna_inference"
SCGG_INFERENCE_ROOT = ARTIFACTS_ROOT / DATASET / "scgg_inference"

OUT_DIR = ARTIFACTS_ROOT / DATASET / "comparison_plots" / "per_slice_comparisons"

# Default metric column to compute delta on. The CSV has
# spearman_per_cell_median + spearman_per_cell_mean. The median variant
# is the LUNA paper Fig. 3 building block (their headline is mean of
# medians across slices), so it's the most directly-comparable single
# number per slice. Override with --metric_col on the CLI.
DEFAULT_METRIC_COL = "spearman_per_cell_median"

# Optional: a higher value is better for these metrics (so positive
# delta = G2T win). Set to False for any metric where lower is better
# (e.g. RSSD, NLL). Detected from a small known-direction table; falls
# back to True with a warning if the column isn't in the table.
HIGHER_IS_BETTER: dict[str, bool] = {
    "spearman_per_cell_median": True,
    "spearman_per_cell_mean": True,
    "pearson_per_cell_median": True,
    "pearson_per_cell_mean": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "absolute_rssd": False,
    "mean_rssd": False,
    "sum_rssd": False,
}

_TS_RE = re.compile(r"^\d{8}_\d{6}(?:_[A-Za-z0-9]+)?$")


def _validate_ts(ts: str, name: str) -> None:
    if not _TS_RE.match(ts):
        raise ValueError(
            f"{name}={ts!r} does not match YYYYMMDD_HHMMSS[_suffix]"
        )


def _load_per_slice(csv_path: Path, metric_col: str) -> pd.DataFrame:
    """Read one per_slice_metrics.csv and return [section_label, n_cells, value].

    Raises clearly when the file is missing or the requested column
    isn't present — partial pipeline runs sometimes write metrics.csv
    but skip per_slice_metrics.csv, which would silently produce an
    empty comparison if we returned an empty df.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"per_slice_metrics.csv missing: {csv_path}. "
            f"Either the inference run hasn't finished or it crashed "
            f"before the per-slice CSV was written."
        )
    df = pd.read_csv(csv_path)
    if "section_label" not in df.columns:
        raise KeyError(
            f"No 'section_label' column in {csv_path} "
            f"(columns: {list(df.columns)})."
        )
    if metric_col not in df.columns:
        raise KeyError(
            f"Column {metric_col!r} not in {csv_path} "
            f"(available: {list(df.columns)}). "
            f"Pass --metric_col=<one of those> to override."
        )
    keep = ["section_label", metric_col]
    if "n_cells" in df.columns:
        keep.insert(1, "n_cells")
    return df[keep].rename(columns={metric_col: "value"})


def compare(
    luna_csv: Path,
    scgg_csv: Path,
    metric_col: str,
) -> pd.DataFrame:
    """Inner-join LUNA + G2T on section_label and compute the delta.

    Returns a sorted DataFrame with one row per slice:
        section_label, n_cells, luna, g2t, delta, abs_delta
    Sorted by delta DESCENDING when higher-is-better, ASCENDING when
    lower-is-better — either way, the "G2T won by the most" rows are
    on top.
    """
    higher_is_better = HIGHER_IS_BETTER.get(metric_col, True)
    if metric_col not in HIGHER_IS_BETTER:
        print(
            f"[compare_per_slice] WARN: direction unknown for "
            f"{metric_col!r}; assuming higher-is-better. Add the "
            f"column to HIGHER_IS_BETTER at the top of this script "
            f"if that's wrong."
        )

    luna = _load_per_slice(luna_csv, metric_col).rename(columns={"value": "luna"})
    scgg = _load_per_slice(scgg_csv, metric_col).rename(columns={"value": "g2t"})

    # Inner-join — drop slices that aren't present in both runs (a
    # full LUNA + G2T MMC run should produce the same set, but if
    # --exclude_test_files was set on one side and not the other, the
    # join naturally restricts to the intersection).
    merged_cols = ["section_label"]
    if "n_cells" in luna.columns:
        merged_cols.append("n_cells")
    df = luna.merge(scgg, on=merged_cols, how="inner")

    n_luna_only = set(luna["section_label"]) - set(df["section_label"])
    n_g2t_only = set(scgg["section_label"]) - set(df["section_label"])
    if n_luna_only:
        print(f"[compare_per_slice] WARN: in LUNA but not G2T: "
              f"{sorted(n_luna_only)}")
    if n_g2t_only:
        print(f"[compare_per_slice] WARN: in G2T but not LUNA: "
              f"{sorted(n_g2t_only)}")
    if df.empty:
        raise RuntimeError(
            "No overlapping section_labels between the two CSVs — "
            "nothing to compare. Are these two inference runs of the "
            "same dataset?"
        )

    # Delta: G2T − LUNA, oriented so positive means "G2T did better".
    # When the metric is lower-is-better (e.g. RSSD), flip the sign
    # so the ranking still puts G2T's biggest wins on top.
    df["delta"] = df["g2t"] - df["luna"]
    if not higher_is_better:
        df["delta"] = -df["delta"]
    df["abs_delta"] = df["delta"].abs()

    # Sort: biggest G2T-win first.
    df = df.sort_values("delta", ascending=False).reset_index(drop=True)
    return df


def format_ranking(
    df: pd.DataFrame,
    metric_col: str,
    luna_ts: str,
    scgg_ts: str,
) -> str:
    """Render the ranking as a fixed-width text block."""
    higher_is_better = HIGHER_IS_BETTER.get(metric_col, True)
    arrow = "↑" if higher_is_better else "↓"

    n_total = len(df)
    n_g2t_wins = int((df["delta"] > 0).sum())
    n_ties = int((df["delta"] == 0).sum())
    n_luna_wins = int((df["delta"] < 0).sum())
    g2t_share = n_g2t_wins / n_total if n_total else 0.0

    lines = [
        "=" * 84,
        f"  Per-slice metric comparison: G2T vs LUNA",
        "=" * 84,
        f"  metric           : {metric_col}  ({arrow} = better)",
        f"  LUNA timestamp   : {luna_ts}",
        f"  G2T  timestamp   : {scgg_ts}",
        f"  slices compared  : {n_total}",
        f"  G2T wins         : {n_g2t_wins}  ({g2t_share*100:.1f}%)",
        f"  ties             : {n_ties}",
        f"  LUNA wins        : {n_luna_wins}",
        "-" * 84,
        f"  mean delta       : {df['delta'].mean():+.4f}",
        f"  median delta     : {df['delta'].median():+.4f}",
        f"  std delta        : {df['delta'].std():.4f}",
        f"  best slice gain  : {df['delta'].max():+.4f}  ({df.iloc[0]['section_label']})",
        f"  worst slice loss : {df['delta'].min():+.4f}  ({df.iloc[-1]['section_label']})",
        "=" * 84,
        "",
        "  Ranking (biggest G2T-vs-LUNA improvement on top):",
        "",
    ]

    has_ncells = "n_cells" in df.columns
    if has_ncells:
        header = f"  {'rank':>4}  {'section_label':40s}  {'n_cells':>7s}  {'LUNA':>8s}  {'G2T':>8s}  {'delta':>9s}"
    else:
        header = f"  {'rank':>4}  {'section_label':40s}  {'LUNA':>8s}  {'G2T':>8s}  {'delta':>9s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        section = str(row["section_label"])[:40]
        luna = float(row["luna"])
        g2t = float(row["g2t"])
        delta = float(row["delta"])
        if has_ncells:
            n = int(row["n_cells"]) if not pd.isna(row["n_cells"]) else 0
            lines.append(
                f"  {rank:>4d}  {section:40s}  {n:>7d}  "
                f"{luna:>8.4f}  {g2t:>8.4f}  {delta:>+9.4f}"
            )
        else:
            lines.append(
                f"  {rank:>4d}  {section:40s}  "
                f"{luna:>8.4f}  {g2t:>8.4f}  {delta:>+9.4f}"
            )

    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--luna_timestamp", required=True,
        help="YYYYMMDD_HHMMSS timestamp of the LUNA inference run "
             "to compare against. Resolves to "
             f"{LUNA_INFERENCE_ROOT}/<TS>/per_slice_metrics.csv.",
    )
    p.add_argument(
        "--scgg_timestamp", required=True,
        help="YYYYMMDD_HHMMSS timestamp of the G2T (scGG) inference "
             "run. Resolves to "
             f"{SCGG_INFERENCE_ROOT}/<TS>/per_slice_metrics.csv.",
    )
    p.add_argument(
        "--metric_col", default=DEFAULT_METRIC_COL,
        help=f"Column name in per_slice_metrics.csv to compute delta on. "
             f"Default: {DEFAULT_METRIC_COL}. Available in MMC runs: "
             f"spearman_per_cell_median, spearman_per_cell_mean. "
             f"Note: there is NO 'pearson' column in this CSV today; "
             f"the LUNA pipeline only emits Spearman.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help=f"Where to write the ranking. Default: {OUT_DIR}.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_ts(args.luna_timestamp, "--luna_timestamp")
    _validate_ts(args.scgg_timestamp, "--scgg_timestamp")

    luna_csv = LUNA_INFERENCE_ROOT / args.luna_timestamp / "per_slice_metrics.csv"
    scgg_csv = SCGG_INFERENCE_ROOT / args.scgg_timestamp / "per_slice_metrics.csv"

    df = compare(luna_csv, scgg_csv, args.metric_col)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"per_slice_delta_{args.metric_col}_"
        f"luna_{args.luna_timestamp}_vs_g2t_{args.scgg_timestamp}"
    )
    csv_path = out_dir / f"{stem}.csv"
    txt_path = out_dir / f"{stem}.txt"

    df.to_csv(csv_path, index=False)
    text = format_ranking(df, args.metric_col, args.luna_timestamp, args.scgg_timestamp)
    print(text)
    txt_path.write_text(text)

    print(f"\n[compare_per_slice] wrote {csv_path}")
    print(f"[compare_per_slice] wrote {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
