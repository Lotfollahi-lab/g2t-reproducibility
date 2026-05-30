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
    python compute_extended_metrics.py \\
        --luna_timestamps 20260526_072208,20260526_093724,... \\
        --scgg_timestamps 20260530_133432,20260530_133426,...

If --luna_timestamps is omitted, auto-discovers every YYYYMMDD_HHMMSS
subdir under LUNA_INFERENCE_ROOT (mirrors the plot script's behaviour).
--scgg_timestamps must be supplied explicitly (the scgg tree typically
holds more than the canonical N seeds).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
DATASET = "mmc_luna"

LUNA_INFERENCE_ROOT = ARTIFACTS_ROOT / DATASET / "luna_inference"
SCGG_INFERENCE_ROOT = ARTIFACTS_ROOT / DATASET / "scgg_inference"

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
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Score every slice under one <TS> dir; return per-slice df + aggregate."""
    ts_root = inference_root / ts
    slice_dirs = _find_slice_dirs(ts_root)
    print(f"[compute_extended] {method_label} {ts}: "
          f"{len(slice_dirs)} slice(s)")

    per_slice_rows: list[Dict[str, float]] = []
    for sd in slice_dirs:
        coords_true, coords_pred, cell_class, slice_name = _load_pair(sd)
        row = evaluate_slice(
            coords_true=coords_true,
            coords_pred=coords_pred,
            cell_class=cell_class,
            contact_percentile=0.01,
            compute_rssd=True,
            rssd_projection="pca",   # ignored here because coords_pred is already 2-D
        )
        row["slice_name"] = slice_name
        row["timestamp"] = ts
        row["method"] = method_label
        per_slice_rows.append(row)
        print(
            f"    {slice_name:40s} "
            f"n={row['n_cells']:5d}  "
            f"spr_med={row['spearman_per_cell_median']:.3f}  "
            f"f1={row['f1']:.3f}  "
            f"rssd={row['absolute_rssd']:.1f}"
        )

    per_slice_df = pd.DataFrame(per_slice_rows)
    aggregate = aggregate_slices(per_slice_rows)
    # Tag the aggregate row with provenance.
    aggregate["timestamp"] = ts
    aggregate["method"] = method_label
    return per_slice_df, aggregate


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
            "<scgg_inference>. If omitted, auto-discovers every "
            "matching subdir (typically NOT what you want for scgg — "
            "the tree usually holds more runs than the canonical N "
            "seeds, so prefer to specify explicitly)."
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    luna_ts = _parse_ts_list(args.luna_timestamps) or _discover_timestamps(LUNA_INFERENCE_ROOT)
    if args.scgg_timestamps is None:
        print("[compute_extended] WARN: --scgg_timestamps not provided; "
              "auto-discovering ALL scgg_inference timestamps. "
              "Pass --scgg_timestamps=ts1,ts2,... to restrict.")
        scgg_ts = _discover_timestamps(SCGG_INFERENCE_ROOT)
    else:
        scgg_ts = _parse_ts_list(args.scgg_timestamps)

    print(f"[compute_extended] LUNA  ({len(luna_ts)}):")
    for t in luna_ts:
        print(f"    {t}")
    print(f"[compute_extended] G2T   ({len(scgg_ts)}):")
    for t in scgg_ts:
        print(f"    {t}")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for method_label, root, ts_list in [
        ("LUNA", LUNA_INFERENCE_ROOT, luna_ts),
        ("G2T", SCGG_INFERENCE_ROOT, scgg_ts),
    ]:
        for ts in ts_list:
            ts_root = root / ts
            agg_path = ts_root / "extended_metrics.csv"
            if args.skip_existing and agg_path.exists():
                print(f"[compute_extended] {method_label} {ts}: skip "
                      f"(extended_metrics.csv exists)")
                n_skipped += 1
                continue
            try:
                per_slice_df, aggregate = _score_timestamp(ts, root, method_label)
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
