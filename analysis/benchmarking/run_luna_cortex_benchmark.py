#!/usr/bin/env python
"""
Run LUNA on the MERFISH mouse cortex silver h5ads and report the same
metrics as scgg, so the two methods can be compared head-to-head.

Pipeline (all I/O paths configurable via CLI):

  1.  Discover Mouse-1 / Mouse-2 silver h5ad files (LUNA cortex naming
      convention: merfish_mouse_cortex_mouse{1,2}_slice{N}.h5ad).
  2.  Write LUNA-format CSVs into a work directory — gene columns first,
      then cell_section / cell_class / coord_X / coord_Y.
  3.  Subprocess into the LUNA conda/uv venv and invoke LUNA's main.py
      with Hydra overrides:
          general.name=<run-name>
          general.mode=train_and_test
          dataset.train_data_path=<train.csv>
          dataset.test_data_path=<test.csv>
          dataset.gene_columns_start=0
          dataset.gene_columns_end=<n_genes>
          train.batch_size=<...>
          train.n_epochs=<...>
          test.save_dir=<absolute path inside output_dir>
          hydra.run.dir=<absolute path inside output_dir>
  4.  After training, read LUNA's per-section outputs
      ({section}/metadata_pred.csv + metadata_true.csv) and compute
      Spearman / contact F1 / Kabsch RSSD via scgg.evaluation.luna_metrics.
  5.  Write the same artifacts as scgg's benchmark
      (per_slice_metrics.csv, aggregate_metrics.json, config.yaml).

The script itself runs in the scgg env (it imports scgg.data.luna_cortex
and scgg.evaluation.luna_metrics). LUNA's training subprocess runs in the
LUNA venv created by setup_luna_env.sh — they don't share Python versions
or torch builds.

Usage:
    python scgg-reproducibility/analysis/benchmarking/run_luna_cortex_benchmark.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/merfish_mouse_cortex_luna \\
        --output_dir ./results/luna_baseline_v1 \\
        --luna_venv /nfs/team361/sb75/.venvs/luna \\
        --luna_repo /nfs/team361/sb75/code/LUNA
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("luna_benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SLICE_RE = re.compile(
    r"^merfish_mouse_cortex_mouse(?P<mouse>\d+)_slice(?P<slice>\d+)\.h5ad$"
)


def _enumerate_slice_files(root: Path) -> List[Tuple[int, int, Path]]:
    out = []
    for p in sorted(root.iterdir()):
        m = _SLICE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m["mouse"]), int(m["slice"]), p))
    return out


def _build_luna_csv(
    files: List[Tuple[int, int, Path]],
    out_csv: Path,
    log2_normalize: bool,
) -> Dict[str, int]:
    """Concatenate per-slice h5ads into a single LUNA-format CSV.

    LUNA wants:
      * gene columns first (positions 0..n_genes-1)
      * then `coord_X`, `coord_Y`, `cell_section`, `cell_class`

    Returns a small dict of summary stats.
    """
    import anndata as ad
    import scipy.sparse as sp

    rows_total = 0
    n_genes = None
    gene_names: Optional[List[str]] = None
    first = True

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    for mouse, slice_id, path in files:
        adata = ad.read_h5ad(path)
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        if log2_normalize:
            X = np.log2(X + 1.0)

        if gene_names is None:
            gene_names = list(adata.var_names)
            n_genes = len(gene_names)
        else:
            if list(adata.var_names) != gene_names:
                raise ValueError(
                    f"Gene panel mismatch in {path.name}: expected "
                    f"{len(gene_names)} genes, got {adata.n_vars}"
                )

        section_label = f"mouse{mouse}_slice{slice_id}"
        cell_class = (
            adata.obs["cell_class"].astype(str).values
            if "cell_class" in adata.obs.columns
            else np.full(adata.n_obs, "unknown")
        )
        # Coordinates: prefer obsm['spatial'], fall back to obs columns.
        if "spatial" in adata.obsm:
            xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
        else:
            xy = np.column_stack([
                adata.obs["coord_X"].to_numpy(dtype=np.float32),
                adata.obs["coord_Y"].to_numpy(dtype=np.float32),
            ])

        df = pd.DataFrame(X, columns=gene_names)
        df["coord_X"] = xy[:, 0]
        df["coord_Y"] = xy[:, 1]
        df["cell_section"] = section_label
        df["cell_class"] = cell_class
        # LUNA reads with index_col=0 — we write the original cell barcodes.
        df.index = adata.obs_names
        df.index.name = "cell_id"

        df.to_csv(out_csv, mode="a", header=first)
        rows_total += len(df)
        first = False
        logger.info(f"    wrote {len(df):>6,} cells from {section_label}")

    return {
        "n_rows": rows_total,
        "n_genes": int(n_genes) if n_genes is not None else 0,
        "n_sections": len(files),
    }


# ---------------------------------------------------------------------------
# LUNA invocation
# ---------------------------------------------------------------------------


def _run_luna_training(
    luna_venv: Path,
    luna_repo: Path,
    train_csv: Path,
    test_csv: Path,
    n_genes: int,
    output_dir: Path,
    run_name: str,
    epochs: int,
    batch_size: int,
    extra_overrides: List[str],
) -> Path:
    """Invoke LUNA's main.py via the LUNA venv's Python.

    Returns the path to LUNA's run output directory (where per-section
    metadata_pred.csv / metadata_true.csv files end up).
    """
    luna_python = luna_venv / "bin" / "python"
    main_py = luna_repo / "main.py"
    if not luna_python.exists():
        raise FileNotFoundError(f"LUNA venv python not found: {luna_python}")
    if not main_py.exists():
        raise FileNotFoundError(f"LUNA main.py not found: {main_py}")

    # Absolute paths for LUNA's outputs. We chdir LUNA into its own repo dir
    # so its `hydra.run.dir=../outputs/...` relative path resolves predictably
    # (or we override hydra.run.dir ourselves to point inside output_dir).
    luna_outputs = output_dir / "luna_outputs"
    luna_outputs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    run_dir = luna_outputs / timestamp.replace("/", "_")
    run_dir = run_dir.with_name(f"{run_dir.name}-{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # LUNA's test outputs land under `test.save_dir`; we put them in a
    # subdirectory of run_dir we control.
    test_save_dir = run_dir / "test_results"
    test_save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(luna_python),
        str(main_py),
        f"general.name={run_name}",
        "general.mode=train_and_test",
        f"dataset.train_data_path={train_csv.resolve()}",
        f"dataset.test_data_path={test_csv.resolve()}",
        "dataset.gene_columns_start=0",
        f"dataset.gene_columns_end={n_genes}",
        f"train.batch_size={batch_size}",
        f"train.n_epochs={epochs}",
        f"test.save_dir={test_save_dir.resolve()}",
        f"hydra.run.dir={run_dir.resolve()}",
        *extra_overrides,
    ]
    logger.info("Invoking LUNA:")
    for arg in cmd:
        logger.info(f"    {arg}")

    log_path = run_dir / "luna_stdout.log"
    t0 = time.time()
    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(luna_repo),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = (time.time() - t0) / 60.0
    logger.info(
        f"LUNA exited with code {proc.returncode} after {elapsed:.1f} min "
        f"(log: {log_path})"
    )
    if proc.returncode != 0:
        # Surface the last bit of the log so the user can see what failed.
        with open(log_path) as f:
            tail = f.read().splitlines()[-50:]
        for line in tail:
            logger.error(f"  | {line}")
        raise RuntimeError(
            f"LUNA training failed (exit {proc.returncode}). See {log_path}"
        )
    return test_save_dir


# ---------------------------------------------------------------------------
# Metric computation (delegates to scgg.evaluation.luna_metrics)
# ---------------------------------------------------------------------------


def _evaluate_luna_outputs(
    test_save_dir: Path,
    contact_percentile: float,
    rssd: bool,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Walk LUNA's per-section outputs and compute scgg metrics."""
    from scgg.evaluation.luna_metrics import evaluate_slice, aggregate_slices

    # LUNA writes one subdirectory per section (or sometimes one extra level
    # for the checkpoint name). We recursively find any pair of
    # metadata_pred.csv / metadata_true.csv that live in the same directory.
    pred_files = list(test_save_dir.rglob("metadata_pred.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No metadata_pred.csv files under {test_save_dir}. "
            "Did LUNA finish successfully?"
        )
    logger.info(f"Found {len(pred_files)} LUNA prediction files under {test_save_dir}")

    per_slice: List[Dict[str, float]] = []
    for pred_path in sorted(pred_files):
        true_path = pred_path.parent / "metadata_true.csv"
        if not true_path.exists():
            logger.warning(f"  {pred_path.parent.name}: no metadata_true.csv; skipping")
            continue
        section_label = pred_path.parent.name

        pred = pd.read_csv(pred_path, index_col=0)
        true = pd.read_csv(true_path, index_col=0)

        # Align by index if both have one; otherwise assume row order matches.
        if not pred.index.equals(true.index):
            common = pred.index.intersection(true.index)
            pred = pred.loc[common]
            true = true.loc[common]
            logger.info(f"  {section_label}: aligned to {len(common)} common cells")

        coords_pred = pred[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        coords_true = true[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
        cell_class = (
            true["cell_class"].astype(str).to_numpy()
            if "cell_class" in true.columns
            else None
        )

        if len(coords_true) < 10:
            logger.info(f"  {section_label}: only {len(coords_true)} cells; skipping")
            continue

        row = evaluate_slice(
            coords_true, coords_pred, cell_class,
            contact_percentile=contact_percentile,
            compute_rssd=rssd,
        )
        row["section_label"] = section_label
        per_slice.append(row)
        logger.info(
            f"  {section_label:30s}  spr_median={row['spearman_per_cell_median']:.4f}  "
            f"spr_mean={row['spearman_per_cell_mean']:.4f}  "
            f"prec={row['precision']:.4f}  "
            f"rssd={row.get('absolute_rssd', float('nan')):.2f}"
        )

    agg = aggregate_slices(per_slice)
    return per_slice, agg


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    silver_dir: str,
    output_dir: str,
    luna_venv: str,
    luna_repo: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    log2_normalize: bool,
    contact_percentile: float,
    compute_rssd: bool,
    skip_training: bool,
    test_save_dir_override: Optional[str],
    extra_overrides: List[str],
) -> Dict[str, float]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / "benchmark.log", mode="w"),
        ],
        force=True,
    )

    silver = Path(silver_dir)
    luna_venv_p = Path(luna_venv)
    luna_repo_p = Path(luna_repo)

    files = _enumerate_slice_files(silver)
    train_files = [f for f in files if f[0] == 1]
    test_files = [f for f in files if f[0] == 2]
    logger.info(
        f"Silver dir: {silver}  ({len(train_files)} train [Mouse 1], "
        f"{len(test_files)} test [Mouse 2])"
    )
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Need Mouse 1 (train) and Mouse 2 (test) slices under {silver}. "
            f"Found: train={len(train_files)}, test={len(test_files)}"
        )

    # ---- 1. Build LUNA CSVs ----------------------------------------------
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    train_csv = work / "MERFISH_mouse_cortex_train.csv"
    test_csv = work / "MERFISH_mouse_cortex_test.csv"

    if skip_training and test_save_dir_override:
        logger.info("--skip_training set: jumping straight to metrics")
    else:
        if train_csv.exists() and test_csv.exists():
            logger.info("LUNA CSVs already exist; reusing")
        else:
            logger.info(f"Writing train CSV -> {train_csv}")
            train_stats = _build_luna_csv(train_files, train_csv, log2_normalize)
            logger.info(f"  train: {train_stats}")
            logger.info(f"Writing test CSV -> {test_csv}")
            test_stats = _build_luna_csv(test_files, test_csv, log2_normalize)
            logger.info(f"  test : {test_stats}")
        n_genes = train_stats["n_genes"] if "train_stats" in locals() else (
            len(pd.read_csv(train_csv, nrows=1, index_col=0).columns) - 4
        )
    # ---- 2. Run LUNA training/test ---------------------------------------
    if skip_training:
        if not test_save_dir_override:
            raise ValueError(
                "--skip_training requires --test_save_dir to point to an "
                "existing LUNA output directory."
            )
        test_save_dir = Path(test_save_dir_override)
    else:
        test_save_dir = _run_luna_training(
            luna_venv=luna_venv_p,
            luna_repo=luna_repo_p,
            train_csv=train_csv,
            test_csv=test_csv,
            n_genes=n_genes,
            output_dir=out,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            extra_overrides=extra_overrides,
        )

    # ---- 3. Compute scgg-style metrics on LUNA's outputs -----------------
    per_slice, agg = _evaluate_luna_outputs(
        test_save_dir,
        contact_percentile=contact_percentile,
        rssd=compute_rssd,
    )

    luna_paper = 0.448
    headline = agg.get("spearman_mean_of_medians", float("nan"))
    logger.info("=" * 72)
    logger.info("LUNA (this run) — aggregated metrics across test slices")
    logger.info("=" * 72)
    for k, v in agg.items():
        logger.info(f"  {k:34s} = {v}")
    logger.info("-" * 72)
    logger.info(
        f"Headline (mean-of-per-slice-median Spearman): {headline:.4f}   |   "
        f"LUNA paper: {luna_paper:.4f}"
    )
    if not np.isnan(headline):
        logger.info(f"Delta vs LUNA paper: {(headline - luna_paper)*100:+.2f} pp")

    # ---- 4. Write outputs (matching scgg's benchmark) --------------------
    fieldnames = sorted({k for r in per_slice for k in r.keys()})
    with open(out / "per_slice_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_slice:
            w.writerow(r)
    with open(out / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    cfg_snap = {
        "silver_dir": str(silver),
        "luna_venv": str(luna_venv_p),
        "luna_repo": str(luna_repo_p),
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "log2_normalize": log2_normalize,
        "contact_percentile": contact_percentile,
        "compute_rssd": compute_rssd,
        "extra_overrides": extra_overrides,
        "test_save_dir": str(test_save_dir),
    }
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_snap, f, sort_keys=False)

    logger.info(f"Wrote benchmark artifacts to {out}")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--silver_dir", required=True,
        help="Per-slice h5ad directory (LUNA cortex split). "
             "Mouse 1 slices => train, Mouse 2 => test.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write LUNA's outputs + scgg-format metrics.",
    )
    p.add_argument(
        "--luna_venv", default="/nfs/team361/sb75/.venvs/luna",
        help="Path to the uv venv created by setup_luna_env.sh.",
    )
    p.add_argument(
        "--luna_repo", default="/nfs/team361/sb75/code/LUNA",
        help="Path to the cloned LUNA repository.",
    )
    p.add_argument(
        "--run_name", default="MERFISH_mouse_cortex",
        help="Sets general.name in LUNA's Hydra config.",
    )
    p.add_argument("--epochs", type=int, default=1000,
                   help="train.n_epochs override (LUNA paper default: 1000).")
    p.add_argument("--batch_size", type=int, default=6,
                   help="train.batch_size override (LUNA paper default: 6 sections).")
    p.add_argument(
        "--no_log2_normalize", action="store_true",
        help="Skip log2(x+1) normalization when writing the LUNA CSVs. "
             "LUNA expects log2-normalized expression; only flip this off "
             "if your silver h5ads are ALREADY log2-normalized.",
    )
    p.add_argument(
        "--contact_percentile", type=float, default=0.01,
        help="Percentile for the contact F1 metric (matches scgg default).",
    )
    p.add_argument("--no_rssd", action="store_true",
                   help="Skip the Kabsch RSSD computation.")
    p.add_argument(
        "--skip_training", action="store_true",
        help="Re-use an existing LUNA output dir (set via --test_save_dir).",
    )
    p.add_argument(
        "--test_save_dir", default=None,
        help="When --skip_training: LUNA's pre-computed test save dir.",
    )
    p.add_argument(
        "--luna_override", action="append", default=[],
        help="Extra Hydra overrides to pass to LUNA, e.g. "
             "'--luna_override train.lr=1e-4'. Repeatable.",
    )
    args = p.parse_args()

    return 0 if run_benchmark(
        silver_dir=args.silver_dir,
        output_dir=args.output_dir,
        luna_venv=args.luna_venv,
        luna_repo=args.luna_repo,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log2_normalize=not args.no_log2_normalize,
        contact_percentile=args.contact_percentile,
        compute_rssd=not args.no_rssd,
        skip_training=args.skip_training,
        test_save_dir_override=args.test_save_dir,
        extra_overrides=args.luna_override,
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
