"""compute_extended_metrics.py

Loop over per-slice ``metadata_pred.csv`` + ``metadata_true.csv`` files
written by a LUNA or scgg inference run, compute the full LUNA-paper
metric battery (Spearman, contact precision/recall/F1, RSSD) per slice,
aggregate across slices, and write the result as ``extended_metrics.csv``
into the same timestamp folder where ``metrics.csv`` lives. Also writes
``per_slice_extended_metrics.csv`` with the raw per-slice rows so the
plot script can show distribution + best/worst slices later if needed.

Why a separate script: the pipeline's headline ``metrics.csv`` only
carries Spearman aggregates today, but the per-slice
``metadata_*.csv`` pairs contain everything we need to recompute the
LUNA paper's full metric set offline. Re-running inference is
multi-hour; this script is seconds.

Per-slice layout (identical for LUNA + scgg pipelines):

    <ARTIFACTS_ROOT>/<dataset>/<method>_inference/<TS>/
        metrics.csv                              ← single-row pipeline summary
        luna_run/test_results/
            <run_name>/model_<...>_epoch_<N>/
                <slice_name>_0/
                    metadata_pred.csv            ← cell_class, coord_X, coord_Y
                    metadata_true.csv            ← cell_class, coord_X, coord_Y

This script writes:

    <ARTIFACTS_ROOT>/<dataset>/<method>_inference/<TS>/
        extended_metrics.csv                     ← single-row aggregated
        per_slice_extended_metrics.csv           ← one row per slice

Aggregation follows ``aggregate_slices`` in
``scgg/src/scgg/evaluation/luna_metrics.py``: the LUNA-paper headline
is ``spearman_mean_of_medians`` (Fig. 3); we additionally write
``spearman_mean_of_means`` plus mean/median/std of every other metric.

Usage:
    # Score every method (default — uses DEFAULT_SCGG_TIMESTAMPS and
    # DEFAULT_CELERY_TIMESTAMPS hard-coded below, plus auto-discovers
    # every LUNA timestamp):
    python compute_extended_metrics.py

    # Score ONLY celery (luna + g2t already done in a prior run):
    python compute_extended_metrics.py --methods celery

    # Score luna + g2t but skip celery:
    python compute_extended_metrics.py --methods luna,g2t

    # Override the default timestamp lists for one or more methods:
    python compute_extended_metrics.py \\
        --luna_timestamps   20260526_072208,20260526_093724,... \\
        --scgg_timestamps   20260530_133432,20260530_133426,... \\
        --celery_timestamps 20260602_074322,20260602_074327,...

Default per-method timestamp lists are HARDCODED below to mirror
``plot_method_comparison.py`` — change them in one place and both
scripts pick up the new canonical seed set.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ── Import LUNA-faithful metric implementations ─────────────────────
# The vendored scgg package contains scgg.evaluation.luna_metrics with
# byte-faithful reimplementations of the paper's three metrics. Add
# scgg/src/ to sys.path so we can import it without needing the
# package installed.
_SCGG_SRC = Path("/nfs/team361/sb75/scgg/src").resolve()
if _SCGG_SRC.exists() and str(_SCGG_SRC) not in sys.path:
    sys.path.insert(0, str(_SCGG_SRC))

from scgg.evaluation.luna_metrics import (  # noqa: E402
    aggregate_slices,
    evaluate_slice,
)


# ──────────────────────────────────────────────────────────────────────
# Inputs (edit if you point this at a different artifacts root)
# ──────────────────────────────────────────────────────────────────────

ARTIFACTS_ROOT = Path("/nfs/team361/sb75/scgg-reproducibility/artifacts")
DEFAULT_DATASET = "mmc_luna"
VALID_DATASETS = ("mmc_luna", "cns_luna")

# Per-dataset default per-method timestamp lists. Keys are dataset
# names; values are the canonical N seeds to score / plot for that
# method+dataset combo. MUST match the ones in
# plot_method_comparison.py so a default-flags run of this script
# scores exactly the seeds the plot will pull. Change in one place,
# remember to mirror in the other (or factor both out into a shared
# config module later if this drifts often).
#
# LUNA: in the MMC case we auto-discover (the tree only ever held
# the 5 canonical seeds). For CNS the tree holds extra exploratory
# runs we don't want in the figure, so we pin 5 explicit seeds.
DEFAULT_SCGG_TIMESTAMPS: dict[str, list[str]] = {
    # 2026-05-30 heads32_fastmds seed0-4 sweep — MMC headline run set.
    "mmc_luna": [
        "20260530_165200",  # seed 0
        "20260530_165210",  # seed 1
        "20260530_165216",  # seed 2
        "20260530_165223",  # seed 3
        "20260530_165229",  # seed 4
    ],
    # 2026-06-02 heads32_fastmds seed0-4 sweep on CNS.
    "cns_luna": [
        "20260602_142452",  # seed 0
        "20260602_142505",  # seed 1
        "20260602_142510",  # seed 2
        "20260602_142516",  # seed 3
        "20260602_142522",  # seed 4
    ],
}
DEFAULT_CELERY_TIMESTAMPS: dict[str, list[str]] = {
    # 2026-06-02 per_reference seed0-4 sweep on MMC.
    "mmc_luna": [
        "20260602_074322",  # seed 0
        "20260602_074327",  # seed 1
        "20260602_074332",  # seed 2
        "20260602_074336",  # seed 3
        "20260602_074342",  # seed 4
    ],
    # 2026-06-04 per_reference seed0-4 sweep on CNS
    # (sagittal1/2/3 + spinalcord excluded; basement queue; bs=256;
    #  --cores 32 + OMP/MKL thread propagation).
    "cns_luna": [
        "20260604_101049",  # seed 0
        "20260604_141924",  # seed 1
        "20260604_141952",  # seed 2
        "20260604_142006",  # seed 3
        "20260604_141959",  # seed 4
    ],
}
# LUNA defaults: MMC = auto-discover (empty list = fall back to
# _discover_timestamps); CNS = pin the 5 we want in the figure.
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

# Valid --methods tokens. "g2t" and "scgg" are aliases for the same
# inference tree (the package is named scgg internally but published
# as G2T) — accept both so old shell-snippets keep working.
_METHOD_ALIASES = {
    "luna":   "LUNA",
    "g2t":    "G2T",
    "scgg":   "G2T",
    "celery": "CeLEry",
}
_DEFAULT_METHODS = ["luna", "g2t", "celery"]

# Columns written by both pipelines (scgg + luna utils/data/load.py
# both call ``metadata.to_csv(...)`` with this fixed schema).
COL_CLASS = "cell_class"
COL_X = "coord_X"
COL_Y = "coord_Y"

_TS_RE = re.compile(r"^\d{8}_\d{6}(?:_[A-Za-z0-9]+)?$")


# ──────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ──────────────────────────────────────────────────────────────────────

def _discover_timestamps(root: Path) -> list[str]:
    """List immediate subdirs matching YYYYMMDD_HHMMSS[_suffix]."""
    if not root.exists():
        raise FileNotFoundError(f"inference root not found: {root}")
    timestamps = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and _TS_RE.match(p.name)
    )
    if not timestamps:
        raise RuntimeError(f"no YYYYMMDD_HHMMSS subdirs under {root}")
    return timestamps


def _find_slice_dirs(ts_root: Path) -> list[Path]:
    """Return every per-slice subdir under <TS>/luna_run/test_results/...

    A per-slice subdir is one that contains BOTH metadata_pred.csv AND
    metadata_true.csv. We rglob for ``metadata_pred.csv`` and infer
    the slice dir as its parent; the matching ``metadata_true.csv``
    must exist alongside or the slice is skipped with a warning.

    Both LUNA and scgg pipelines use the same intermediate path
    (``luna_run/test_results/<run_name>/model_<...>_epoch_<N>/<slice>_0/``)
    because scgg vendors LUNA's test loop verbatim. We don't anchor on
    that path explicitly — rglob handles any future layout change.
    """
    if not ts_root.exists():
        raise FileNotFoundError(f"timestamp dir not found: {ts_root}")
    pred_csvs = sorted(ts_root.rglob("metadata_pred.csv"))
    slice_dirs = []
    for p in pred_csvs:
        true_csv = p.parent / "metadata_true.csv"
        if true_csv.exists():
            slice_dirs.append(p.parent)
        else:
            print(
                f"[compute_extended] WARN: metadata_pred.csv at {p} "
                f"has no sibling metadata_true.csv; skipping."
            )
    if not slice_dirs:
        raise RuntimeError(
            f"no slice subdirs with both metadata_pred.csv and "
            f"metadata_true.csv found under {ts_root}"
        )
    return slice_dirs


# ──────────────────────────────────────────────────────────────────────
# Per-slice loading + scoring
# ──────────────────────────────────────────────────────────────────────

def _load_pair(slice_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read (coords_true, coords_pred, cell_class, slice_name) for one slice.

    Both CSVs are indexed by cell_ID; we sort BOTH by cell_ID before
    extracting arrays so a row-order mismatch between the two files
    (which would silently scramble the per-cell pairing and tank the
    Spearman score) is impossible. The cell-id column comes through as
    the unnamed first column when pd.read_csv sees ``cell_ID`` in the
    header; ``index_col=0`` handles both the "explicitly named" and
    "unnamed-first-column" cases.
    """
    pred = pd.read_csv(slice_dir / "metadata_pred.csv", index_col=0)
    true = pd.read_csv(slice_dir / "metadata_true.csv", index_col=0)

    # Reindex pred onto true's cell-id order so per-row arithmetic is
    # well-defined regardless of how either file was written.
    if not pred.index.equals(true.index):
        common = true.index.intersection(pred.index)
        if len(common) == 0:
            raise ValueError(
                f"{slice_dir}: metadata_pred and metadata_true have "
                f"NO overlapping cell IDs"
            )
        if len(common) < len(true) or len(common) < len(pred):
            print(
                f"[compute_extended] WARN: {slice_dir.name}: "
                f"pred/true index mismatch — keeping the "
                f"{len(common)}-cell intersection "
                f"(pred has {len(pred)}, true has {len(true)})."
            )
        true = true.loc[common]
        pred = pred.loc[common]
    else:
        true = true.sort_index()
        pred = pred.loc[true.index]

    coords_true = true[[COL_X, COL_Y]].to_numpy(dtype=np.float64)
    coords_pred = pred[[COL_X, COL_Y]].to_numpy(dtype=np.float64)
    # cell_class is taken from TRUE (predictions don't change cell ids).
    cell_class = true[COL_CLASS].to_numpy()
    return coords_true, coords_pred, cell_class, slice_dir.name


def _score_timestamp(
    ts: str,
    inference_root: Path,
    method_label: str,
    *,
    vectorized: bool = True,
    workers: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Score every slice under one <TS> dir; return per-slice df + aggregate.

    Speedup knobs (both opt-in to non-default behaviour where they
    matter):
      vectorized: forwarded to evaluate_slice as ``spearman_vectorized``.
                  Default True (the vectorised path is 10-50× faster on
                  CNS-sized slices and produces identical per-cell rho).
                  Set False to fall back to the original
                  scipy.stats.spearmanr per-cell loop.
      workers:    if > 1, parallelise the per-slice loop over
                  ``min(workers, n_slices)`` worker processes via
                  ProcessPoolExecutor. Default 1 keeps the original
                  serial execution semantics.
    """
    ts_root = inference_root / ts
    slice_dirs = _find_slice_dirs(ts_root)
    print(f"[compute_extended] {method_label} {ts}: "
          f"{len(slice_dirs)} slice(s)"
          f"{'  [vectorized]' if vectorized else '  [loop]'}"
          f"{'' if workers == 1 else f'  [workers={workers}]'}",
          flush=True)

    # Each work unit is one slice. _score_one_slice is pure (reads its
    # CSVs from disk, returns a dict) so it parallelises cleanly via
    # ProcessPoolExecutor when workers > 1.
    work_items = [(sd, ts, method_label, vectorized) for sd in slice_dirs]

    per_slice_rows: list[Dict[str, float]] = []
    if workers <= 1:
        # Serial path — preserves the original execution semantics
        # exactly (same print order, same memory peak) so a default
        # run with workers=1 vectorized=False is byte-faithful to
        # the old code.
        for item in work_items:
            row = _score_one_slice(*item)
            per_slice_rows.append(row)
            _print_slice_row(row)
    else:
        # Parallel path. We use ProcessPoolExecutor (not threads)
        # because the per-cell Spearman is CPU-bound — even the
        # vectorised path holds the GIL during numpy reductions on
        # the big N×N matrices, so threads wouldn't help.
        # ``max_workers`` is clamped against the work-unit count so
        # a 32-core box scoring 8 slices doesn't spin up 32 idle
        # workers.
        max_workers = min(workers, len(work_items))
        print(f"[compute_extended] launching {max_workers} parallel "
              f"workers across {len(work_items)} slice(s)", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            # submit + as_completed so the print order tracks
            # completion order (helpful for live progress visibility);
            # the final per_slice_rows order is then resorted to
            # the original slice_dirs ordering for deterministic CSV
            # output.
            future_to_idx = {
                ex.submit(_score_one_slice, *item): idx
                for idx, item in enumerate(work_items)
            }
            results: dict[int, Dict[str, float]] = {}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                row = fut.result()
                results[idx] = row
                _print_slice_row(row)
        # Restore deterministic per-slice order.
        per_slice_rows = [results[i] for i in range(len(work_items))]

    per_slice_df = pd.DataFrame(per_slice_rows)
    aggregate = aggregate_slices(per_slice_rows)
    # Tag the aggregate row with provenance.
    aggregate["timestamp"] = ts
    aggregate["method"] = method_label
    return per_slice_df, aggregate


def _score_one_slice(
    slice_dir: Path,
    ts: str,
    method_label: str,
    vectorized: bool,
) -> Dict[str, float]:
    """Score one slice. Pure function — picklable for multiprocessing.

    Lives at module scope (not nested inside _score_timestamp) because
    ProcessPoolExecutor pickles the function by qualified name; nested
    functions are not picklable.
    """
    coords_true, coords_pred, cell_class, slice_name = _load_pair(slice_dir)
    row = evaluate_slice(
        coords_true=coords_true,
        coords_pred=coords_pred,
        cell_class=cell_class,
        contact_percentile=0.01,
        compute_rssd=True,
        rssd_projection="pca",   # ignored here because coords_pred is already 2-D
        spearman_vectorized=vectorized,
    )
    row["slice_name"] = slice_name
    row["timestamp"] = ts
    row["method"] = method_label
    return row


def _print_slice_row(row: Dict[str, float]) -> None:
    """One-line per-slice progress log line. Factored out of the
    serial / parallel branches so the format stays in lockstep."""
    print(
        f"    {row['slice_name']:40s} "
        f"n={int(row['n_cells']):5d}  "
        f"spr_med={row['spearman_per_cell_median']:.3f}  "
        f"f1={row['f1']:.3f}  "
        f"rssd={row['absolute_rssd']:.1f}",
        flush=True,
    )


def _write_outputs(
    ts: str,
    inference_root: Path,
    per_slice_df: pd.DataFrame,
    aggregate: Dict[str, float],
) -> Tuple[Path, Path]:
    """Write per_slice_extended_metrics.csv + extended_metrics.csv into <TS>/."""
    ts_root = inference_root / ts
    per_slice_path = ts_root / "per_slice_extended_metrics.csv"
    agg_path = ts_root / "extended_metrics.csv"

    # Stable column order: identifiers first, then metrics.
    id_cols = ["slice_name", "timestamp", "method"]
    metric_cols = [c for c in per_slice_df.columns if c not in id_cols]
    per_slice_df = per_slice_df[id_cols + metric_cols]
    per_slice_df.to_csv(per_slice_path, index=False)

    # Aggregate as a single-row dataframe so consumers can pd.read_csv
    # then iloc[0] same as they do for metrics.csv.
    agg_df = pd.DataFrame([aggregate])
    # Put identifiers first here too.
    cols = ["timestamp", "method"] + [c for c in agg_df.columns if c not in ("timestamp", "method")]
    agg_df = agg_df[cols]
    agg_df.to_csv(agg_path, index=False)

    print(f"[compute_extended] wrote {per_slice_path}")
    print(f"[compute_extended] wrote {agg_path}")
    return per_slice_path, agg_path


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

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
            f"to score AND the DEFAULT_*_TIMESTAMPS dicts' keys."
        ),
    )
    p.add_argument(
        "--methods",
        default=",".join(_DEFAULT_METHODS),
        help=(
            "Comma-separated subset of methods to score. Tokens: "
            "luna, g2t (alias scgg), celery. Default scores all three. "
            "Use this to skip methods you've already scored — e.g. "
            "``--methods celery`` re-scores ONLY celery_inference/, "
            "leaving the existing luna_inference/extended_metrics.csv "
            "and scgg_inference/extended_metrics.csv untouched."
        ),
    )
    p.add_argument(
        "--luna_timestamps",
        default=None,
        help=(
            "Comma-separated YYYYMMDD_HHMMSS timestamps under "
            "<luna_inference>. If omitted, auto-discovers every "
            "matching subdir."
        ),
    )
    p.add_argument(
        "--scgg_timestamps",
        default=None,
        help=(
            "Comma-separated YYYYMMDD_HHMMSS timestamps under "
            "<scgg_inference>. If omitted, uses DEFAULT_SCGG_TIMESTAMPS "
            "(the canonical N seeds — same list the plot script pulls). "
            "The legacy behaviour of auto-discovering every subdir was "
            "removed because the scgg tree typically holds many more "
            "runs than we want in the published comparison; running the "
            "compute step over all of them would score dozens of "
            "obsolete experiments unnecessarily."
        ),
    )
    p.add_argument(
        "--celery_timestamps",
        default=None,
        help=(
            "Comma-separated YYYYMMDD_HHMMSS timestamps under "
            "<celery_inference>. If omitted, uses "
            "DEFAULT_CELERY_TIMESTAMPS (the per_reference sweep, "
            "same 5 seeds the plot script pulls). Pass the multi_slice "
            "TS here to score those instead."
        ),
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip timestamps whose extended_metrics.csv already exists. "
            "Off by default — by default we re-score and overwrite to "
            "pick up metric-implementation changes."
        ),
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Process-pool size for the per-slice loop. Default 1 "
            "(serial). >1 enables ProcessPoolExecutor parallelism — "
            "use 8-16 on CNS for a 4-8× speedup. Capped per timestamp "
            "to ``min(workers, n_slices)`` so you don't spin up idle "
            "workers. Memory note: each worker independently builds "
            "two N×N float32 distance matrices for its current slice "
            "(~180 GB peak for N=150k), so set workers conservatively "
            "if multiple workers might score the largest slices at "
            "the same time."
        ),
    )
    p.add_argument(
        "--no_vectorize",
        action="store_true",
        help=(
            "Disable the vectorised per-cell Spearman path; fall back "
            "to the original scipy.stats.spearmanr per-cell loop. The "
            "two paths produce identical per-cell rho (see the "
            "regression test in scgg/tests/test_per_cell_spearman_"
            "vectorized.py) — this flag exists only as a debugging "
            "escape hatch / for reproducing the pre-vectorisation "
            "metric values exactly."
        ),
    )
    return p.parse_args(argv)


def _parse_ts_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    items = [t.strip() for t in s.split(",") if t.strip()]
    for ts in items:
        if not _TS_RE.match(ts):
            raise ValueError(
                f"timestamp {ts!r} does not match YYYYMMDD_HHMMSS[_suffix]"
            )
    return items


def _parse_methods(s: str) -> list[str]:
    """Validate + canonicalise --methods. Returns a list of *canonical*
    labels (``LUNA``, ``G2T``, ``CeLEry``) preserving the user's order so
    log output reads in the order they asked for."""
    tokens = [t.strip().lower() for t in s.split(",") if t.strip()]
    if not tokens:
        raise ValueError("--methods parsed to empty list")
    bad = [t for t in tokens if t not in _METHOD_ALIASES]
    if bad:
        raise ValueError(
            f"unknown --methods token(s): {bad}. "
            f"valid: {sorted(_METHOD_ALIASES.keys())}"
        )
    # de-dup while preserving order (g2t + scgg are aliases — keep first)
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        canonical = _METHOD_ALIASES[t]
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    selected = _parse_methods(args.methods)

    # Resolve dataset-derived paths + default timestamp lists.
    dataset = args.dataset
    luna_root   = ARTIFACTS_ROOT / dataset / "luna_inference"
    scgg_root   = ARTIFACTS_ROOT / dataset / "scgg_inference"
    celery_root = ARTIFACTS_ROOT / dataset / "celery_inference"
    print(f"[compute_extended] dataset = {dataset}")

    # Build the (method_label, root, timestamps) work list, scoping each
    # method's timestamp source independently:
    #   - LUNA: explicit list, else DEFAULT_LUNA_TIMESTAMPS[dataset];
    #     if that's empty, fall back to auto-discovering the tree.
    #   - G2T:  explicit list, else DEFAULT_SCGG_TIMESTAMPS[dataset].
    #   - CeLEry: explicit list, else DEFAULT_CELERY_TIMESTAMPS[dataset].
    # Only methods in ``selected`` are added to the work list — others
    # are silently skipped (their already-existing extended_metrics.csv
    # is left untouched).
    work: list[tuple[str, Path, list[str]]] = []
    if "LUNA" in selected:
        luna_ts = (
            _parse_ts_list(args.luna_timestamps)
            or list(DEFAULT_LUNA_TIMESTAMPS.get(dataset, []))
            or _discover_timestamps(luna_root)
        )
        work.append(("LUNA", luna_root, luna_ts))
    if "G2T" in selected:
        scgg_ts = (
            _parse_ts_list(args.scgg_timestamps)
            or list(DEFAULT_SCGG_TIMESTAMPS.get(dataset, []))
        )
        if not scgg_ts:
            raise RuntimeError(
                f"No G2T timestamps for dataset={dataset!r}. Add an entry "
                f"to DEFAULT_SCGG_TIMESTAMPS or pass --scgg_timestamps."
            )
        work.append(("G2T", scgg_root, scgg_ts))
    if "CeLEry" in selected:
        celery_ts = (
            _parse_ts_list(args.celery_timestamps)
            or list(DEFAULT_CELERY_TIMESTAMPS.get(dataset, []))
        )
        if not celery_ts:
            raise RuntimeError(
                f"No CeLEry timestamps for dataset={dataset!r}. Add an "
                f"entry to DEFAULT_CELERY_TIMESTAMPS or pass "
                f"--celery_timestamps."
            )
        work.append(("CeLEry", celery_root, celery_ts))

    if not work:
        # Defensive: _parse_methods enforces non-empty, but if the
        # canonical labels somehow collapse to zero (impossible today)
        # we'd silently do nothing — fail loudly instead.
        print("[compute_extended] ERROR: no methods selected after parsing "
              f"--methods={args.methods!r}", file=sys.stderr)
        return 2

    print(f"[compute_extended] methods to score: {[m for m, _, _ in work]}")
    for label, _, ts_list in work:
        print(f"[compute_extended] {label} ({len(ts_list)}):")
        for t in ts_list:
            print(f"    {t}")

    # Resolve speedup knobs once and log them so it's obvious in the
    # LSF stdout which path is running.
    vectorized = not args.no_vectorize
    workers = max(1, int(args.workers))
    print(
        f"[compute_extended] speedup knobs: "
        f"vectorized={vectorized} workers={workers}"
    )
    # When workers > 1 the per-process BLAS threading would multiply
    # against the worker count and oversubscribe the box. Pin each
    # worker to a single BLAS thread so the total thread count stays
    # roughly equal to ``workers``.  Honour the user's explicit
    # OMP_NUM_THREADS if it was set by the caller (e.g. submit_pipeline.sh
    # exports OMP_NUM_THREADS=$LSF_CORES_FINAL).
    if workers > 1 and "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        print(
            "[compute_extended] pinned OMP/MKL/OPENBLAS to 1 thread per "
            "worker to avoid CPU oversubscription"
        )

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for method_label, root, ts_list in work:
        for ts in ts_list:
            ts_root = root / ts
            agg_path = ts_root / "extended_metrics.csv"
            if args.skip_existing and agg_path.exists():
                print(f"[compute_extended] {method_label} {ts}: skip "
                      f"(extended_metrics.csv exists)")
                n_skipped += 1
                continue
            try:
                per_slice_df, aggregate = _score_timestamp(
                    ts, root, method_label,
                    vectorized=vectorized, workers=workers,
                )
                _write_outputs(ts, root, per_slice_df, aggregate)
                n_done += 1
            except Exception as exc:
                # Don't let a single bad run abort the whole sweep —
                # report and continue so the others still get scored.
                n_failed += 1
                print(f"[compute_extended] ERROR scoring {method_label} {ts}: "
                      f"{type(exc).__name__}: {exc}")

    print(f"[compute_extended] done. scored={n_done}  "
          f"skipped={n_skipped}  failed={n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
