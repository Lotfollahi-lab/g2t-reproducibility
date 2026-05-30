"""plot_spearman_g2t_vs_luna.py

Compare G2T (formerly scGG) vs LUNA on the MMC cortex benchmark by the
``spearman_mean_of_medians`` metric reported in each run's
``metrics.csv``. Five seeds per method; bar = mean across seeds, error
bar = SEM, jittered dots = individual seeds. Single-panel figure in
the same Nature-journal style as the MintFlow benchmarking notebook
this script was adapted from.

The G2T (scGG) timestamps are HARDCODED in ``SCGG_TIMESTAMPS`` below
because the user has run more than five training jobs in the scgg_inference
tree and wants to pin which five seeds form the published comparison.
Edit that list to swap seeds in/out. The LUNA timestamps are
AUTO-DISCOVERED from ``LUNA_INFERENCE_ROOT``: every immediate
subdirectory matching ``YYYYMMDD_HHMMSS`` is treated as a seed run and
the metrics.csv inside is loaded. If you want to restrict LUNA the
same way, replace the ``_discover_luna_timestamps`` call with an
explicit list (same shape as SCGG_TIMESTAMPS).

Outputs ``g2t_vs_luna_spearman.{svg,pdf,png}`` + a processed CSV
(``g2t_vs_luna_spearman_processed.csv``) into OUT_DIR. SVG is the
primary deliverable per the user request; PDF/PNG are written
alongside so the same script feeds the supplemental and the slide
decks without re-running.

LSF: this is a tiny CPU-only matplotlib job (~seconds). Submit with
``bash submit_plot_spearman_g2t_vs_luna.sh`` in the sibling lsf/ dir,
which fires it on the ``normal`` queue with 4GB / 1 core / 10min.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ──────────────────────────────────────────────────────────────────────
# Inputs (edit these paths / timestamps to point at your runs)
# ──────────────────────────────────────────────────────────────────────

ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
DATASET = "mmc_luna"

LUNA_INFERENCE_ROOT = ARTIFACTS_ROOT / DATASET / "luna_inference"
SCGG_INFERENCE_ROOT = ARTIFACTS_ROOT / DATASET / "scgg_inference"

# G2T (scGG) seeds — hardcoded because the inference tree contains more
# than five runs and the user picks the canonical five by timestamp.
SCGG_TIMESTAMPS = [
    "20260530_133432",
    "20260530_133426",
    "20260530_133419",
    "20260530_133412",
    "20260530_133954",
]

OUT_DIR = ARTIFACTS_ROOT / DATASET / "comparison_plots"

METRIC_COL = "spearman_mean_of_medians"
METRIC_LABEL = "Spearman's Rank Correlation"
METRIC_HIGHER_IS_BETTER = True

# Display name + colour per method. Order in this dict = top-to-bottom
# order on the y-axis (top row plotted first). G2T on top because it's
# the proposed method.
METHODS: dict[str, str] = {
    "G2T": "#FF006E",   # magenta — matches MintFlow's "proposed" colour
    "LUNA": "#3A86FF",  # blue — matches MEFISTO's spot from the notebook
}


# ──────────────────────────────────────────────────────────────────────
# Style (verbatim from the notebook template)
# ──────────────────────────────────────────────────────────────────────

mm = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

_TS_RE = re.compile(r"^\d{8}_\d{6}(?:_[A-Za-z0-9]+)?$")


def _discover_luna_timestamps(root: Path) -> list[str]:
    """Return all immediate-subdir basenames matching YYYYMMDD_HHMMSS.

    Sorted lexicographically so the seed→run mapping is stable across
    re-runs of this script. Raises if the directory doesn't exist or
    contains no matching subdirs (better to fail loudly than silently
    plot an empty bar).
    """
    if not root.exists():
        raise FileNotFoundError(
            f"LUNA inference root not found: {root}. "
            f"Either no runs exist yet or the artifacts path moved."
        )
    timestamps = sorted(p.name for p in root.iterdir() if p.is_dir() and _TS_RE.match(p.name))
    if not timestamps:
        raise RuntimeError(
            f"No YYYYMMDD_HHMMSS subdirectories found under {root}. "
            f"Did the inference runs finish?"
        )
    return timestamps


def _load_metric(metrics_csv: Path, metric: str) -> float:
    """Read one row's metric value from a pipeline metrics.csv.

    metrics.csv is the single-row summary written by run_*_pipeline.py;
    each method's pipeline writes the same schema so the read pattern
    is shared. Raises with the offending path on any failure mode
    (missing file, missing column, multi-row file) so partial-run
    artifacts don't silently degrade the plot.
    """
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv missing: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    if metric not in df.columns:
        raise KeyError(
            f"Column {metric!r} not in {metrics_csv} "
            f"(columns: {list(df.columns)})"
        )
    if len(df) != 1:
        raise ValueError(
            f"Expected single-row metrics.csv, got {len(df)} rows at {metrics_csv}"
        )
    return float(df[metric].iloc[0])


def _collect(
    method: str,
    inference_root: Path,
    timestamps: list[str],
) -> pd.DataFrame:
    """Build a tidy DataFrame: one row per (method, timestamp, value)."""
    rows = []
    for ts in timestamps:
        csv_path = inference_root / ts / "metrics.csv"
        value = _load_metric(csv_path, METRIC_COL)
        rows.append({"method": method, "timestamp": ts, "value": value})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def plot_panel(ax, df: pd.DataFrame, methods: list[str]) -> None:
    """Bars + SEM + jittered dots, mirroring the notebook's AUC panel.

    df has columns method/timestamp/value. ``methods`` controls the
    y-order (top→bottom).
    """
    y = np.arange(len(methods))

    by_method = {m: df.loc[df["method"] == m, "value"].dropna().values for m in methods}
    means = {m: float(np.mean(v)) if len(v) else np.nan for m, v in by_method.items()}
    sems = {
        m: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        for m, v in by_method.items()
    }

    for i, method in enumerate(methods):
        vals = by_method[method]
        if len(vals) == 0:
            ax.text(
                0.5, i, "n/a",
                va="center", ha="center",
                fontsize=5.5, fontstyle="italic", color="0.5",
                transform=ax.get_yaxis_transform(),
            )
            continue

        color = METHODS[method]
        # Bar (mean)
        ax.barh(
            i, means[method], height=0.7,
            color=color, edgecolor=color,
            linewidth=0.5, alpha=0.35, zorder=2,
        )
        # Error bar (SEM) — only if more than one seed
        if len(vals) > 1:
            ax.errorbar(
                means[method], i, xerr=sems[method],
                fmt="none", ecolor="0.3", elinewidth=0.5,
                capsize=1.2, capthick=0.5, zorder=3,
            )
        # Individual seed dots, light vertical jitter so overlaps
        # are still readable
        rng = np.random.default_rng(42 + i)
        yj = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            vals, i + yj, s=14, color=color,
            edgecolors="white", linewidths=0.3, zorder=4, alpha=0.92,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontweight="medium")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.4, len(methods) - 0.6)
    ax.invert_yaxis()

    ax.xaxis.grid(True, linewidth=0.2, alpha=0.4, color="0.65", linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))

    ax.set_title("MMC cortex", fontsize=6, fontweight="medium", pad=3)


def main() -> int:
    # ── Discover / load ────────────────────────────────────────────────
    luna_timestamps = _discover_luna_timestamps(LUNA_INFERENCE_ROOT)
    print(f"[plot_spearman] LUNA timestamps ({len(luna_timestamps)}):")
    for ts in luna_timestamps:
        print(f"    {ts}")
    print(f"[plot_spearman] G2T (scgg) timestamps ({len(SCGG_TIMESTAMPS)}):")
    for ts in SCGG_TIMESTAMPS:
        print(f"    {ts}")

    luna_df = _collect("LUNA", LUNA_INFERENCE_ROOT, luna_timestamps)
    g2t_df = _collect("G2T", SCGG_INFERENCE_ROOT, SCGG_TIMESTAMPS)
    df = pd.concat([g2t_df, luna_df], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = OUT_DIR / "g2t_vs_luna_spearman_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"[plot_spearman] saved processed CSV → {processed_path}")

    # Quick numeric summary to stdout — useful in LSF logs.
    summary = (
        df.groupby("method")["value"]
          .agg(["mean", "std", "count", "min", "max"])
          .reindex(list(METHODS.keys()))
    )
    print("[plot_spearman] summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Figure ─────────────────────────────────────────────────────────
    methods = list(METHODS.keys())
    n_methods = len(methods)

    # Single panel, sized to roughly match one panel of the multi-panel
    # template figures so the aesthetic is consistent.
    panel_width = 32 * mm   # a hair wider than the notebook's 18 mm
                            # because there are only two rows and the
                            # ticks need breathing room
    row_height = 0.18       # taller than notebook (0.12) because we
                            # only have 2 rows so the figure would
                            # otherwise be uncomfortably squat
    fig_height = n_methods * row_height + 0.45
    fig_width = panel_width + 0.65

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plot_panel(ax, df, methods)

    # Per-panel x-limits: tight around observed range with padding,
    # but always include the (0,1) span Spearman lives in so the
    # bar's relation to "0 = no correlation" stays interpretable.
    valid = df["value"].dropna().values
    lo = min(valid.min(), 0.0)
    hi = max(valid.max(), 1.0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo - pad, hi + pad)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)

    arrow = r" ($\uparrow$)" if METRIC_HIGHER_IS_BETTER else r" ($\downarrow$)"
    fig.text(
        0.5, 0.07,
        METRIC_LABEL + arrow,
        ha="center", va="center", fontsize=7, fontweight="medium",
    )

    for ext in ("svg", "pdf", "png"):
        out_path = OUT_DIR / f"g2t_vs_luna_spearman.{ext}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        print(f"[plot_spearman] saved → {out_path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
