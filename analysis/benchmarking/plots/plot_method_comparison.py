"""plot_method_comparison.py

Compare G2T (formerly scGG) vs LUNA vs CeLEry on the MMC cortex
benchmark across the full LUNA-paper metric battery — Spearman (mean
of medians AND mean of means), contact precision / recall / F1, and
three flavours of RSSD (absolute, mean-of-per-class,
sum-of-per-class). One panel per metric, all in the same
Nature-journal style as the MintFlow benchmarking notebook this script
was adapted from.

Reads pre-computed ``extended_metrics.csv`` files produced by
``compute_extended_metrics.py`` (one per inference timestamp directory).
That script loops over each per-slice ``metadata_pred.csv`` +
``metadata_true.csv`` pair, calls the byte-faithful LUNA-metric
reimplementations in ``scgg.evaluation.luna_metrics``, and aggregates
across slices. Run it FIRST against the same timestamps you want to
plot, or this script will fail loudly because ``extended_metrics.csv``
won't exist yet.

The G2T (scGG) AND CeLEry timestamps are HARDCODED in
``DEFAULT_SCGG_TIMESTAMPS`` / ``DEFAULT_CELERY_TIMESTAMPS`` below
because each inference tree typically holds more runs than the
canonical N seeds we want in the published comparison. Override via
``--scgg_timestamps ts1,...`` / ``--celery_timestamps ts1,...`` on the
CLI. The LUNA timestamps are AUTO-DISCOVERED from
``LUNA_INFERENCE_ROOT`` (they're the only method without a multi-mode
ambiguity, so a glob-all default is safe there).

Outputs (into OUT_DIR):
    g2t_vs_luna_vs_celery_extended_metrics.{svg,pdf,png}  ← 2×4 panels
    g2t_vs_luna_vs_celery_extended_metrics_processed.csv  ← long-form data

Per-panel design (mirrors the notebook AUC cell):
    - bar         = mean across seeds
    - error bar   = SEM
    - jittered    = individual seed dots
    - right-side  = "0.463 ± 0.012  (+3.2% ↑ vs LUNA)" on the best method

LSF: tiny CPU-only matplotlib job (~seconds). Submit with
``bash submit_plot_method_comparison.sh`` in the sibling lsf/ dir,
which fires it on the ``normal`` queue with 4 GB / 1 core / 10 min.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import _nature_style as ns


# ──────────────────────────────────────────────────────────────────────
# Inputs (edit these paths / timestamps to point at your runs)
# ──────────────────────────────────────────────────────────────────────

ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
DEFAULT_DATASET = "mmc_luna"
VALID_DATASETS = ("mmc_luna", "cns_luna")

# Per-dataset default per-method timestamp lists. Keys are dataset
# names; values are the canonical N seeds to plot. MUST match the
# ones in compute_extended_metrics.py so the compute step scores the
# same seeds the plot step pulls (change in one place, mirror the
# other — or factor out to a shared config module if this drifts).
#
# Refreshed 2026-06-02 (MMC) and 2026-06-04 (CNS).
DEFAULT_SCGG_TIMESTAMPS: dict[str, list[str]] = {
    # MMC: 2026-05-30 heads32_fastmds seed0-4 sweep.
    # Headline spearman_per_cell_median: 0.469 / 0.469 / 0.477 /
    # 0.469 / 0.478.
    "mmc_luna": [
        "20260530_165200",  # seed 0
        "20260530_165210",  # seed 1
        "20260530_165216",  # seed 2
        "20260530_165223",  # seed 3
        "20260530_165229",  # seed 4
    ],
    # CNS: 2026-06-02 heads32_fastmds seed0-4 sweep on cns_luna.
    "cns_luna": [
        "20260602_142452",  # seed 0
        "20260602_142505",  # seed 1
        "20260602_142510",  # seed 2
        "20260602_142516",  # seed 3
        "20260602_142522",  # seed 4
    ],
}

# CeLEry default per-dataset — both lists are the PER-REFERENCE runs
# (per_reference outperformed multi_slice on MMC, so it's the
# canonical CeLEry protocol for this comparison). Override via
# ``--celery_timestamps`` if you want to swap in multi_slice.
DEFAULT_CELERY_TIMESTAMPS: dict[str, list[str]] = {
    # MMC: 2026-06-02 per_reference seed0-4.
    "mmc_luna": [
        "20260602_074322",  # seed 0
        "20260602_074327",  # seed 1
        "20260602_074332",  # seed 2
        "20260602_074336",  # seed 3
        "20260602_074342",  # seed 4
    ],
    # CNS: 2026-06-04 per_reference seed0-4 (sagittal1/2/3 + spinalcord
    # excluded; basement queue; bs=256; --cores 32 + OMP/MKL threads).
    "cns_luna": [
        "20260604_101049",  # seed 0
        "20260604_141924",  # seed 1
        "20260604_141952",  # seed 2
        "20260604_142006",  # seed 3
        "20260604_141959",  # seed 4
    ],
}

# LUNA default per-dataset: MMC = auto-discover (empty list falls
# back to _discover_luna_timestamps); CNS = pin the 5 we want in the
# figure (the cns_luna/luna_inference tree holds extra exploratory
# runs we don't want).
DEFAULT_LUNA_TIMESTAMPS: dict[str, list[str]] = {
    "mmc_luna": [],
    "cns_luna": [
        "20260601_085512",  # seed 0
        "20260601_085523",  # seed 1
        "20260601_085535",  # seed 2
        "20260601_085541",  # seed 3
        "20260601_085547",  # seed 4
    ],
}

# Filename of the per-timestamp aggregated CSV written by
# compute_extended_metrics.py — single-row schema mirroring the
# pipeline's metrics.csv convention.
EXTENDED_METRICS_FILENAME = "extended_metrics.csv"


class MetricSpec(NamedTuple):
    """Description of one metric → one panel.

    Attributes:
        column: Column name in extended_metrics.csv. Single value per
            seed (the aggregate across that seed's test slices, computed
            by ``aggregate_slices`` in luna_metrics.py).
        label: Human-readable panel title.
        higher_is_better: Controls (a) which method gets the
            "+X.X% vs other" tag in the panel annotation, (b) the
            arrow shown next to the percent (↑ if True, ↓ if False).
    """
    column: str
    label: str
    higher_is_better: bool


# Metrics to render — one panel per entry, laid out 2 rows × 4 cols. The
# row groupings cluster semantically related metrics (Spearman together,
# contact together, RSSD together) so the figure reads top-to-bottom as
# "rank correlation / graph quality / geometric error".
#
# The keys here MUST match what ``aggregate_slices`` writes:
#   - ``spearman_mean_of_medians`` — LUNA paper Fig. 3 headline.
#   - ``spearman_mean_of_means``   — companion aggregation.
#   - ``{precision,recall,f1}_mean`` — mean across slices.
#   - ``{absolute,mean,sum}_rssd_mean`` — mean across slices.
METRICS: list[MetricSpec] = [
    MetricSpec("spearman_mean_of_medians", "Spearman (mean of medians)", True),
    MetricSpec("spearman_mean_of_means",   "Spearman (mean of means)",   True),
    MetricSpec("precision_mean",           "Contact precision",          True),
    MetricSpec("recall_mean",              "Contact recall",             True),
    MetricSpec("f1_mean",                  "Contact F1",                 True),
    MetricSpec("absolute_rssd_mean",       "Absolute RSSD",              False),
    MetricSpec("mean_rssd_mean",           "Mean RSSD (per cell class)", False),
    MetricSpec("sum_rssd_mean",            "Sum RSSD (per cell class)",  False),
]

# 2 rows × 4 cols grid. Change to (1, 8) for a single-row strip or
# (4, 2) for a tall narrow layout; the figure-sizing math below
# auto-adapts.
GRID_ROWS = 2
GRID_COLS = 4
assert GRID_ROWS * GRID_COLS >= len(METRICS), \
    f"GRID_ROWS*GRID_COLS={GRID_ROWS*GRID_COLS} < len(METRICS)={len(METRICS)}"

# Display name + colour per method. Order in this dict = top-to-bottom
# order on the y-axis (top row plotted first). G2T on top because it's
# the proposed method; LUNA next (the headline diffusion baseline);
# CeLEry last (supervised-MLP baseline from the LUNA paper's Supp
# Note 2).
#
# Colour-blind-friendly palette: subset of Okabe-Ito (the de facto
# standard for scientific figures — see Wong, "Points of view: Color
# blindness", Nature Methods 8, 441 (2011); also recommended by Nature
# for accessibility). The full 8-colour palette is reproduced for
# reference in the comment block below; we picked three high-contrast
# colours that stay distinguishable under deuteranopia (red-green),
# protanopia, tritanopia, AND grayscale conversion. Verified with
# the Coblis simulator.
#
#   Okabe-Ito (HEX):
#     #000000 black          #E69F00 orange         #56B4E9 sky blue
#     #009E73 bluish green   #F0E442 yellow         #0072B2 blue
#     #D55E00 vermillion     #CC79A7 reddish purple
#
# Trade-off vs the previous palette: this no longer matches the
# magenta/blue/teal used in the G2T architecture diagrams
# (draw_g2t_internals.py etc.) — accessibility wins over visual
# coherence with the schematic figures. The schematic figures don't
# need to be CB-safe because they convey shape, not data values.
METHODS: dict[str, str] = {
    "G2T":    "#D55E00",   # vermillion — proposed method (stands out)
    "LUNA":   "#0072B2",   # blue       — headline diffusion baseline
    "CeLEry": "#009E73",   # bluish green — supervised-MLP baseline
}


# ──────────────────────────────────────────────────────────────────────
# Style — shared with all plot scripts in this directory via
# _nature_style.py. ``mm`` is the millimetre-to-inch conversion used
# for sizing the figure in Nature's print units; ``ns.apply()`` installs
# the journal's font / size / fonttype rcParams globally.
# ──────────────────────────────────────────────────────────────────────

mm = ns.mm
ns.apply()


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

_TS_RE = re.compile(r"^\d{8}_\d{6}(?:_[A-Za-z0-9]+)?$")


def _discover_luna_timestamps(root: Path) -> list[str]:
    """Return all immediate-subdir basenames matching YYYYMMDD_HHMMSS.

    Sorted lexicographically so the seed→run mapping is stable across
    re-runs. Raises if the directory doesn't exist or contains no
    matching subdirs (better to fail loudly than silently plot empty).
    """
    if not root.exists():
        raise FileNotFoundError(
            f"LUNA inference root not found: {root}. "
            f"Either no runs exist yet or the artifacts path moved."
        )
    timestamps = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and _TS_RE.match(p.name)
    )
    if not timestamps:
        raise RuntimeError(
            f"No YYYYMMDD_HHMMSS subdirectories found under {root}. "
            f"Did the inference runs finish?"
        )
    return timestamps


def _load_extended_row(extended_csv: Path) -> pd.Series:
    """Read the single-row extended_metrics.csv into a Series.

    Raises with the offending path on any failure mode (missing file,
    multi-row file) so partial-run artifacts don't silently degrade
    the plot. Missing METRIC columns are tolerated here — the plot
    layer handles per-panel NaNs by drawing 'n/a'.
    """
    if not extended_csv.exists():
        raise FileNotFoundError(
            f"{EXTENDED_METRICS_FILENAME} missing: {extended_csv}. "
            f"Run compute_extended_metrics.py first for the matching "
            f"timestamps."
        )
    df = pd.read_csv(extended_csv)
    if len(df) != 1:
        raise ValueError(
            f"Expected single-row {EXTENDED_METRICS_FILENAME}, got "
            f"{len(df)} rows at {extended_csv}"
        )
    return df.iloc[0]


def _collect(
    method: str,
    inference_root: Path,
    timestamps: list[str],
) -> pd.DataFrame:
    """Long-form DataFrame: one row per (method, timestamp, metric, value).

    Loads each timestamp's extended_metrics.csv once, then unpacks all
    METRICS columns. Missing columns become NaN so the plot can still
    render the panels for metrics that DO have data.
    """
    rows: list[dict] = []
    for ts in timestamps:
        csv_path = inference_root / ts / EXTENDED_METRICS_FILENAME
        row = _load_extended_row(csv_path)
        for spec in METRICS:
            value = float(row[spec.column]) if spec.column in row.index else float("nan")
            rows.append({
                "method": method,
                "timestamp": ts,
                "metric": spec.column,
                "value": value,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def plot_panel(
    ax,
    df_metric: pd.DataFrame,
    methods: list[str],
    spec: MetricSpec,
    show_ylabels: bool = True,
) -> None:
    """Render one metric's panel: bars + SEM + jittered dots + annotations.

    ``df_metric`` has columns method/timestamp/value, ALREADY filtered
    to a single metric. ``methods`` controls the y-order (top→bottom).
    """
    y = np.arange(len(methods))

    by_method = {
        m: df_metric.loc[df_metric["method"] == m, "value"].dropna().values
        for m in methods
    }
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
        ax.barh(
            i, means[method], height=0.7,
            color=color, edgecolor=color,
            linewidth=0.5, alpha=0.35, zorder=2,
        )
        if len(vals) > 1:
            ax.errorbar(
                means[method], i, xerr=sems[method],
                fmt="none", ecolor="0.3", elinewidth=0.5,
                capsize=1.2, capthick=0.5, zorder=3,
            )
        rng = np.random.default_rng(42 + i)
        yj = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            vals, i + yj, s=14, color=color,
            edgecolors="white", linewidths=0.3, zorder=4, alpha=0.92,
        )

    # ── Right-side annotations: per-method mean (± SEM) + a "+X.X% vs
    # <other>" tag on the best method. "Best" is decided strictly by
    # mean: higher = better when spec.higher_is_better is True, lower =
    # better otherwise. The gap is always reported as a POSITIVE
    # percentage of the second-best method's mean — i.e. how much
    # *better* the winner is, never a negative number.
    valid_means = {m: means[m] for m in methods if not np.isnan(means[m])}
    if len(valid_means) >= 2:
        ranked = sorted(
            valid_means.items(),
            key=lambda kv: -kv[1] if spec.higher_is_better else kv[1],
        )
        best_method, best_val = ranked[0]
        second_method, second_val = ranked[1]
        denom = abs(second_val) if second_val != 0 else float("nan")
        if spec.higher_is_better:
            pct_change = (best_val - second_val) / denom * 100.0
        else:
            pct_change = (second_val - best_val) / denom * 100.0
    else:
        best_method = None
        second_method = None
        pct_change = None

    # Format helper: pick decimals based on the magnitude of the value
    # so we get "0.463" for Spearman/F1 but "12.4" for RSSD without
    # hand-tuning per metric.
    def _fmt(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.3f}"

    for i, method in enumerate(methods):
        m_val = means.get(method, np.nan)
        if np.isnan(m_val):
            continue
        n_seeds = len(by_method[method])
        if n_seeds > 1:
            label = f"{_fmt(m_val)} ± {_fmt(sems[method])}"
        else:
            label = _fmt(m_val)
        if method == best_method and pct_change is not None:
            arrow = "↑" if spec.higher_is_better else "↓"
            label = f"{label}  ({pct_change:+.1f}% {arrow} vs {second_method})"
        ax.text(
            m_val, i, "  " + label,
            va="center", ha="left",
            fontsize=5.5, fontweight="medium",
            color=METHODS[method],
            clip_on=False,
        )

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(methods, fontweight="medium")
    else:
        ax.set_yticklabels([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.4, len(methods) - 0.6)
    ax.invert_yaxis()

    ax.xaxis.grid(True, linewidth=0.2, alpha=0.4, color="0.65", linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))

    arrow = "↑" if spec.higher_is_better else "↓"
    ax.set_title(
        f"{spec.label} ({arrow})",
        fontsize=6.5, fontweight="medium", pad=3,
    )


def _set_panel_xlim(ax, df_metric: pd.DataFrame, spec: MetricSpec) -> None:
    """Set the bar-region xlim per panel so different metric scales
    (Spearman in [0,1] vs RSSD in [10, 100]) each get tight, readable
    panels. The annotation text overflows past xlim into the figure
    margin reserved by ``annotation_room`` (clip_on=False).
    """
    valid = df_metric["value"].dropna().values
    if len(valid) == 0:
        return
    lo = float(valid.min())
    hi = float(valid.max())
    span = hi - lo if hi > lo else max(abs(hi), 1e-6)
    # Tight padding on the left; smaller on the right (the annotation
    # text lives outside the data area, in the reserved margin).
    pad = span * 0.05
    if spec.column.startswith("spearman_") or spec.column.endswith("_mean"):
        # For bounded-[0,1]-ish metrics, anchor xlim to 0 on the left
        # so bar lengths are visually comparable to the "no signal"
        # baseline. RSSD doesn't have that interpretation so we let it
        # float to data range.
        if spec.column.startswith(("spearman_", "precision_", "recall_", "f1_")):
            lo = min(lo, 0.0)
    ax.set_xlim(lo - pad, hi + pad)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=VALID_DATASETS,
        help=(
            f"Dataset slug (subdir of {ARTIFACTS_ROOT}). Default "
            f"{DEFAULT_DATASET!r}. Selects both the inference roots "
            f"to read AND the DEFAULT_*_TIMESTAMPS dicts' keys. The "
            f"output figure lands in <ARTIFACTS_ROOT>/<dataset>/"
            f"comparison_plots/."
        ),
    )
    p.add_argument(
        "--scgg_timestamps",
        default=None,
        help=(
            "Comma-separated YYYYMMDD_HHMMSS timestamps to load from "
            "scgg_inference/ for G2T. Overrides DEFAULT_SCGG_TIMESTAMPS. "
            "Each timestamp must contain extended_metrics.csv "
            "(produced by compute_extended_metrics.py)."
        ),
    )
    p.add_argument(
        "--luna_timestamps",
        default=None,
        help=(
            "Comma-separated YYYYMMDD_HHMMSS timestamps to load from "
            "luna_inference/ for LUNA. When omitted, auto-discovers "
            "every matching subdir under LUNA_INFERENCE_ROOT."
        ),
    )
    p.add_argument(
        "--celery_timestamps",
        default=None,
        help=(
            "Comma-separated YYYYMMDD_HHMMSS timestamps to load from "
            "celery_inference/ for CeLEry. Overrides "
            "DEFAULT_CELERY_TIMESTAMPS. The default points at the "
            "PER-REFERENCE sweep (one model per test slice). To "
            "use the multi-slice sweep instead pass those TS here."
        ),
    )
    return p.parse_args(argv)


def _parse_ts_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    items = [t.strip() for t in s.split(",") if t.strip()]
    if not items:
        raise ValueError("timestamp list parsed to empty")
    for ts in items:
        if not _TS_RE.match(ts):
            raise ValueError(
                f"timestamp {ts!r} does not match YYYYMMDD_HHMMSS[_suffix]"
            )
    return items


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve dataset-derived paths + default timestamp lists.
    dataset = args.dataset
    luna_root   = ARTIFACTS_ROOT / dataset / "luna_inference"
    scgg_root   = ARTIFACTS_ROOT / dataset / "scgg_inference"
    celery_root = ARTIFACTS_ROOT / dataset / "celery_inference"
    out_dir     = ARTIFACTS_ROOT / dataset / "comparison_plots"
    print(f"[plot_metrics] dataset = {dataset}")

    scgg_timestamps = (
        _parse_ts_list(args.scgg_timestamps)
        or list(DEFAULT_SCGG_TIMESTAMPS.get(dataset, []))
    )
    if not scgg_timestamps:
        raise RuntimeError(
            f"No G2T timestamps for dataset={dataset!r}. Add an entry to "
            f"DEFAULT_SCGG_TIMESTAMPS or pass --scgg_timestamps."
        )
    celery_timestamps = (
        _parse_ts_list(args.celery_timestamps)
        or list(DEFAULT_CELERY_TIMESTAMPS.get(dataset, []))
    )
    if not celery_timestamps:
        raise RuntimeError(
            f"No CeLEry timestamps for dataset={dataset!r}. Add an entry "
            f"to DEFAULT_CELERY_TIMESTAMPS or pass --celery_timestamps."
        )
    luna_timestamps = (
        _parse_ts_list(args.luna_timestamps)
        or list(DEFAULT_LUNA_TIMESTAMPS.get(dataset, []))
        or _discover_luna_timestamps(luna_root)
    )

    print(f"[plot_metrics] LUNA timestamps ({len(luna_timestamps)}):")
    for ts in luna_timestamps:
        print(f"    {ts}")
    print(f"[plot_metrics] G2T (scgg) timestamps ({len(scgg_timestamps)}):")
    for ts in scgg_timestamps:
        print(f"    {ts}")
    print(f"[plot_metrics] CeLEry timestamps ({len(celery_timestamps)}):")
    for ts in celery_timestamps:
        print(f"    {ts}")

    g2t_df    = _collect("G2T",    scgg_root,   scgg_timestamps)
    luna_df   = _collect("LUNA",   luna_root,   luna_timestamps)
    celery_df = _collect("CeLEry", celery_root, celery_timestamps)
    df = pd.concat([g2t_df, luna_df, celery_df], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    processed_path = out_dir / "g2t_vs_luna_vs_celery_extended_metrics_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"[plot_metrics] saved processed CSV → {processed_path}")

    # Numeric summary per metric — useful in LSF logs for quick sanity-check.
    print("[plot_metrics] summary:")
    summary = (
        df.groupby(["metric", "method"])["value"]
          .agg(["mean", "std", "count"])
          .unstack("method")
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Figure ─────────────────────────────────────────────────────────
    methods = list(METHODS.keys())
    n_methods = len(methods)

    # Per-panel sizes — tightened so multiple comparison figures can
    # sit side-by-side in the final manuscript figure. Each panel is
    # now ~30 mm of bars + ~30 mm of annotation room, so the full 2×4
    # figure lands around 240 mm wide × ~60 mm tall (vs the previous
    # ~320 mm wide). The annotation text uses clip_on=False so it can
    # spill rightward into the next panel's wspace — keep wspace
    # generous (subplots_adjust below) so it doesn't actually collide.
    panel_width      = 28 * mm           # was 42 — narrower bar region
    annotation_room  = 30 * mm           # was 38 — shorter "± SEM (% ↑)"
    row_height       = 0.20              # was 0.22 — one extra method
                                         # (CeLEry) per panel; offset by
                                         # slightly shorter row height
                                         # so the figure doesn't grow
    fig_width  = (panel_width + annotation_room) * GRID_COLS + 0.85
    fig_height = (n_methods * row_height + 0.55) * GRID_ROWS + 0.45

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(fig_width, fig_height),
        sharey=True,
    )
    # Flatten for easy iteration regardless of GRID_ROWS/COLS shape.
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, spec in enumerate(METRICS):
        ax = axes_flat[idx]
        col, row = idx % GRID_COLS, idx // GRID_COLS
        df_metric = df.loc[df["metric"] == spec.column].copy()
        plot_panel(ax, df_metric, methods, spec, show_ylabels=(col == 0))
        _set_panel_xlim(ax, df_metric, spec)

    # Hide any leftover axes (if METRICS doesn't exactly fill the grid).
    for k in range(len(METRICS), len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    # Bottom margin sized so the per-method colour legend (drawn next)
    # doesn't overlap the bottom row's tick labels.
    plt.subplots_adjust(wspace=0.55, hspace=0.55, bottom=0.10)

    # Single shared figure footer caption: identify the dataset + the
    # number of seeds powering each method's bars.
    n_g2t    = df.loc[df["method"] == "G2T",    "timestamp"].nunique()
    n_luna   = df.loc[df["method"] == "LUNA",   "timestamp"].nunique()
    n_celery = df.loc[df["method"] == "CeLEry", "timestamp"].nunique()
    # Human-friendly dataset label for the caption — fall back to the
    # raw slug if we ever add a dataset the lookup doesn't know about.
    _DATASET_DISPLAY = {
        "mmc_luna": "MMC cortex",
        "cns_luna": "Mouse CNS",
    }
    dataset_label = _DATASET_DISPLAY.get(dataset, dataset)
    fig.text(
        0.5, 0.03,
        f"{dataset_label} — G2T ({n_g2t} seeds) vs LUNA ({n_luna} seeds) vs "
        f"CeLEry ({n_celery} seeds). "
        f"Bar = mean across seeds, error bar = SEM, dots = individual seeds.",
        ha="center", va="center", fontsize=6.5, fontweight="medium",
    )

    out_stem = "g2t_vs_luna_vs_celery_extended_metrics"
    for ext in ("svg", "pdf", "png"):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        print(f"[plot_metrics] saved → {out_path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
