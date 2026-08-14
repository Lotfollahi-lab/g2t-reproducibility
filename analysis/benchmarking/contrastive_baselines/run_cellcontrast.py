#!/usr/bin/env python3
"""run_cellcontrast.py — run the AUTHORS' published CellContrast on our benchmark.

This is a WRAPPER, not a reimplementation. All modelling is done by the upstream
MIT-licensed package (https://github.com/HKU-BAL/CellContrast, Li et al.,
Patterns 5(8):101022, 2024) at the authors' own default hyperparameters. Our code
only (a) converts our silver h5ad slices into the exact AnnData layout the
package requires, (b) shells out to its documented CLI, and (c) converts its
output into the artifact schema our metric harness already reads. Anything that
would change the method itself is deliberately NOT done here.

Verified against the upstream source before writing (not guessed):
  * train CLI      : --train_data_path --save_folder --parameter_file_path [-sc]
  * inference CLI  : --query_data_path --model_folder --parameter_file_path
                     --ref_data_path --save_path [--enable_denovo]
  * coordinates    : written to uns['referenced x'] / uns['referenced y'];
                     each query cell COPIES the (x, y) of its single most
                     cosine-similar reference cell (top-1 argmax in
                     inference.map_to_ST).
  * coords are read from obs['x'] / obs['y'] ONLY (loadData.py) — there is no
    obsm['spatial'] fallback, so those two columns must exist.
  * slices are grouped by obs['embryo'] (hardcoded sample_field_name). IF THAT
    COLUMN IS ABSENT the package silently invents a single sample, and the
    spatial kNN graph then spans different tissue sections — a wrong result that
    still looks plausible. We therefore always set it and assert it is set.
  * the package performs NO normalisation; the paper used log-normalised input,
    so we pre-normalise (see --expression_mode).
  * defaults from parameters_singleCell.json: 3000 epochs, lr 0.1, batch 64,
    temperature 0.05, k_nearest_positives 80.

PROTOCOL DECISION (pre-registered, do not change silently)
----------------------------------------------------------
At inference CellContrast copies coordinates out of a REFERENCE object. We pass
the TRAINING donor's slices as that reference. Using the held-out test slice
would leak its coordinate set and inflate every metric. This puts CellContrast
in structurally the same position as CeLEry, which also predicts into the
training frame.

Because each training slice has its own micron frame, we min-max normalise every
slice's coordinates into a shared [-0.5, 0.5] box before training, and invert
predictions with the TEST slice's own scaler so the written artifacts are in
original microns (matching CeLEry and novosparc).

COORDINATE FRAME (``--coord_frame``, default ISOTROPIC).
Upstream builds its k=80 positive graph with a KDTree over exactly these
coordinates (``loadData.checkNeighbors``), so the metric we hand it decides which
cells are positives.
  * ``isotropic`` (default): both axes divided by the LARGER span, box centred.
    Aspect preserved, so the neighbour ranking is IDENTICAL to raw microns —
    i.e. the authors' positive graph, unchanged. This is why it is the default.
  * ``per_axis``: each axis divided by its own span (what LUNA/G2T's
    ``position_normalize`` does). NOT a similarity transform: measured on the MMC
    train split it alters ~7% of the k=80 positive set at the median slice aspect
    (1.25) and ~14% at the worst (1.59), and it perturbs the Spearman ranks and
    the Contact F1 threshold too — not Sum RSSD alone. Kept only for
    reproducing earlier runs.
Note this is INDEPENDENT of the frame the artifacts are written in (always
microns) and of the cross-method RSSD frame problem, which is scorer-side
(``compute_extended_metrics.py --rssd_frame``).

Usage (cortex):
    python run_cellcontrast.py \
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \
        --cellcontrast_repo /nfs/team361/sb75/CellContrast \
        --out_root /nfs/team361/sb75/scgg-reproducibility/artifacts \
        --dataset mmc_luna --seed 0

Usage (CNS). The shared 600-d cross-platform latent lives in ``adata.X`` ITSELF,
not in any obsm key (the only obsm key on those h5ads is ``spatial``), so do NOT
pass --use_obsm; hand the latent over untouched. Passing --use_obsm spatial would
feed GROUND-TRUTH COORDINATES as features and is refused.
    python run_cellcontrast.py --data_dir .../cns_luna --dataset cns_luna \
        --expression_mode silver_raw --max_train_cells 150000 \
        --exclude_test_files sagittal1_test.h5ad,sagittal2_test.h5ad,\
sagittal3_test.h5ad,spinalcord_test.h5ad ...

Always start with --smoke_test (minutes, proves the install) before a real run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_adapter import (  # noqa: E402
    COL_CLASS,
    SliceCoordScaler,
    discover_split_files,
    section_label_from_filename,
    write_run_manifest,
    write_slice_artifacts,
)

__version__ = "2026-08-13-run-cellcontrast-v1"

LOG = logging.getLogger("cellcontrast")
UNS_X, UNS_Y = "referenced x", "referenced y"
SAMPLE_FIELD = "embryo"                # hardcoded upstream


# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="silver dir holding *_train.h5ad / *_test.h5ad")
    p.add_argument("--cellcontrast_repo", required=True,
                   help="path to the cloned CellContrast repo (contains cellContrast.py)")
    p.add_argument("--out_root", default="/nfs/team361/sb75/scgg-reproducibility/artifacts")
    p.add_argument("--dataset", default=None,
                   help="dataset name for the artifact path; default = data_dir basename")
    p.add_argument("--run_timestamp", default=None, help="pin the artifact timestamp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None,
                   help="override training_epoch (upstream default 3000; the paper "
                        "reports >1000 is needed). Lower values are a DEVIATION and "
                        "are recorded in the manifest.")
    p.add_argument("--single_cell", action="store_true", default=True,
                   help="select the authors' IMAGING-ST parameters "
                        "(parameters_singleCell.json, k_nearest_positives=80). "
                        "Default True: MERFISH/STARmap/Xenium.")
    p.add_argument("--no_single_cell", dest="single_cell", action="store_false",
                   help="select the authors' SPOT-ST parameters "
                        "(parameters_spot.json, k_nearest_positives=20). REQUIRED "
                        "for Visium/spot data -- the paper's k is platform-specific "
                        "(80 for SeqFISH/MERSCOPE, 20 for Stereo-seq/10x Visium).")
    p.add_argument("--force_platform", action="store_true",
                   help="bypass the dataset-name vs --single_cell consistency check.")
    p.add_argument("--expression_mode", default="log2", choices=("log2", "silver_raw"),
                   help="log2 = log2(1+x) (default). This is NOT parity with the "
                        "other baselines -- LUNA/G2T/novosparc apply the identity "
                        "and CeLEry adds a z-score; log2_norm has zero call sites "
                        "in either repo. It is chosen for fidelity to the "
                        "CellContrast paper ('log-normalized ... by scran'; scran "
                        "logNormCounts = size-factor normalise then log base 2), "
                        "which composes correctly on count-magnitude silver X. "
                        "silver_raw hands the matrix over untouched -- REQUIRED for "
                        "the both-sign cns_luna latent and for raw-count silver dirs.")
    p.add_argument("--coord_frame", default="isotropic",
                   choices=("isotropic", "per_axis"),
                   help="how per-slice coordinates are put in a shared box. "
                        "isotropic (default) preserves aspect, so upstream's k-NN "
                        "positive graph is IDENTICAL to raw microns. per_axis "
                        "matches LUNA/G2T position_normalize but changes ~7-14%% of "
                        "the positive set. See the module docstring.")
    p.add_argument("--use_obsm", default=None,
                   help="use adata.obsm[KEY] as the feature matrix instead of .X. "
                        "NOT needed for cns_luna: its 600-d shared latent is in .X "
                        "itself (only obsm key there is 'spatial'). Passing "
                        "'spatial' is refused -- it would feed ground-truth "
                        "coordinates as features.")
    p.add_argument("--query_chunk", type=int, default=8000,
                   help="split each test slice into chunks of this many cells for "
                        "inference and concatenate. Upstream builds dense "
                        "query x query AND query x reference matrices (plus a full "
                        "argsort index), so an unchunked large slice needs hundreds "
                        "of GB. top-1 retrieval is per-query-row independent, so "
                        "chunking is BIT-IDENTICAL. Slices at or below this size "
                        "take the original single-call path. 0 disables chunking.")
    p.add_argument("--max_mem_gb", type=float, default=None,
                   help="refuse to start if the estimated inference peak exceeds "
                        "this (GB). Estimate = max(24*c^2, 16*c^2 + 24*c*R) bytes, "
                        "measured constants for float32 sim + float32 sorted + "
                        "int64 argsort. Set it to the LSF -M value.")
    p.add_argument("--max_train_cells", type=int, default=None,
                   help="cap total training cells by per-slice stratified subsampling. "
                        "REQUIRED in practice for cns_luna (2.85M cells is ~9 days). "
                        "Recorded in the manifest as a deviation.")
    p.add_argument("--max_ref_cells", type=int, default=None,
                   help="cap the reference used at inference (memory: inference builds "
                        "dense query x ref). Subsamples the training object.")
    p.add_argument("--exclude_test_files", default="",
                   help="comma-separated *_test.h5ad basenames to skip")
    p.add_argument("--smoke_test", action="store_true",
                   help="tiny end-to-end run (2 train slices, 1 test slice, 5 epochs) "
                        "to prove the install works. Never report these numbers.")
    p.add_argument("--dry_run", action="store_true", help="print commands, do not run")
    return p


# ---------------------------------------------------------------------------
def _import_anndata():
    try:
        import anndata as ad
        import scanpy as sc  # noqa: F401  (upstream imports it; fail early if absent)
        return ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "needs anndata + scanpy (present in the cellcontrast env, not locally)"
        ) from exc


def _assert_finite(M: np.ndarray, what: str) -> np.ndarray:
    """Refuse non-finite features.

    Upstream's only NaN handling is ``np.nan_to_num`` inside loadTrainData
    (loadData.py:79) -- it is TRAIN-ONLY. At inference a NaN feature row makes
    every cosine similarity NaN; NaN sorts last ascending, so upstream's
    ``argsort(...)[::-1][0]`` lands ON the NaN and the cell is silently assigned
    reference cell N-1's coordinate. That survives the frame check, the
    pair-membership check and ``np.isfinite`` on the output, so it can only be
    caught here.
    """
    if not np.isfinite(M).all():
        n_bad = int((~np.isfinite(M)).any(axis=1).sum())
        raise ValueError(
            f"{what}: {n_bad} row(s) contain NaN/Inf features. Upstream only "
            f"nan_to_num's the TRAINING matrix, so at inference these cells would "
            f"be silently assigned the last reference cell's coordinates.")
    return M


def _feature_matrix(adata, use_obsm: Optional[str], expression_mode: str,
                    what: str = "features") -> np.ndarray:
    """Feature matrix for the encoder, matching our other baselines' input."""
    if use_obsm:
        if use_obsm == "spatial":
            raise ValueError(
                "--use_obsm spatial would hand GROUND-TRUTH COORDINATES to the "
                "encoder as features, defeating the coordinate withholding in "
                "build_query and invalidating every metric. Refused. (For "
                "cns_luna the shared latent is in .X: drop --use_obsm and pass "
                "--expression_mode silver_raw.)")
        if use_obsm not in adata.obsm:
            raise KeyError(f"obsm[{use_obsm!r}] absent; have {list(adata.obsm.keys())}")
        M = np.asarray(adata.obsm[use_obsm], dtype=np.float32)
        # An embedding is already normalised; log2 on a latent would be nonsense.
        return _assert_finite(M, what)
    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    if expression_mode == "log2":
        # log2(1+x) is only defined for x > -1. Silver X holds non-negative
        # per-cell-normalised counts, so a negative here means we were handed
        # something else (e.g. an embedding) — fail loudly rather than emit NaN
        # and train on it.
        mn = float(X.min()) if X.size else 0.0
        if mn <= -1.0:
            raise ValueError(
                f"expression_mode='log2' but the matrix has min {mn:.4g} <= -1; "
                "log2(1+x) would be NaN. Use --expression_mode silver_raw, or "
                "--use_obsm for an embedding.")
        if mn < 0.0:
            LOG.warning("matrix has negative values (min %.4g); log2(1+x) is "
                        "defined but this is unexpected for counts", mn)
        X = np.log2(1.0 + X, dtype=np.float32)
    return _assert_finite(X, what)


def _coords_of(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        s = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if s.ndim == 2 and s.shape[1] >= 2:
            return s[:, :2]
    if "coord_X" in adata.obs and "coord_Y" in adata.obs:
        return np.column_stack([adata.obs["coord_X"].to_numpy(dtype=np.float64),
                                adata.obs["coord_Y"].to_numpy(dtype=np.float64)])
    raise ValueError("no coordinates in obsm['spatial'] or obs['coord_X'/'coord_Y']")


def _var_names_for(adata, use_obsm: Optional[str]) -> np.ndarray:
    if use_obsm:
        d = int(np.asarray(adata.obsm[use_obsm]).shape[1])
        return np.array([f"{use_obsm}_{i}" for i in range(d)], dtype=object)
    return np.asarray(adata.var_names, dtype=object)


# ---------------------------------------------------------------------------
def build_reference(
    ad_mod, train_files: List[Path], args, rng: np.random.Generator, work: Path,
    k_pos: int = 0,
):
    """Concatenate training slices into the single reference object upstream wants.

    Sets obs['x'], obs['y'] to PER-SLICE NORMALISED coordinates and obs['embryo']
    to the slice label so the spatial kNN never crosses sections.

    Returns ``(path, var_names, n_obs, xmin, xmax)`` so the caller can record
    provenance and cross-check the test panel.
    """
    per_slice_cap = None
    if args.max_train_cells:
        per_slice_cap = max(1, args.max_train_cells // max(1, len(train_files)))
        # loadData.checkNeighbors queries the KDTree with k = k_pos + 1 and
        # raises if a sample has fewer cells than that -- AFTER we have built and
        # written the whole reference. Catch it before doing that work.
        if k_pos and per_slice_cap < k_pos + 1:
            raise ValueError(
                f"--max_train_cells {args.max_train_cells} over {len(train_files)} "
                f"slices gives {per_slice_cap} cells/slice, but upstream needs at "
                f"least k_nearest_positives+1 = {k_pos + 1} per slice "
                f"(loadData.checkNeighbors KDTree). Raise --max_train_cells to "
                f"at least {(k_pos + 1) * len(train_files)}.")

    isotropic = (getattr(args, "coord_frame", "isotropic") == "isotropic")
    blocks, xs, ys, embryo, classes, var_ref = [], [], [], [], [], None
    for f in train_files:
        a = ad_mod.read_h5ad(f)
        label = section_label_from_filename(f)
        feats = _feature_matrix(a, args.use_obsm, args.expression_mode,
                               what=f"train slice {f.name}")
        coords = _coords_of(a)
        vn = _var_names_for(a, args.use_obsm)
        if var_ref is None:
            var_ref = vn
        elif not np.array_equal(var_ref, vn):
            raise ValueError(f"{f.name}: feature names differ from the first slice")

        if per_slice_cap and feats.shape[0] > per_slice_cap:
            keep = rng.choice(feats.shape[0], per_slice_cap, replace=False)
            keep.sort()
            feats, coords = feats[keep], coords[keep]
            cls = (a.obs[COL_CLASS].astype(str).to_numpy()[keep]
                   if COL_CLASS in a.obs else np.full(len(keep), "NA"))
        else:
            cls = (a.obs[COL_CLASS].astype(str).to_numpy()
                   if COL_CLASS in a.obs else np.full(feats.shape[0], "NA"))

        # per-slice normalisation puts every section in a shared frame
        cn = SliceCoordScaler(isotropic=isotropic).fit(coords).transform(coords)
        blocks.append(feats)
        xs.append(cn[:, 0]); ys.append(cn[:, 1])
        embryo.append(np.full(feats.shape[0], label, dtype=object))
        classes.append(cls)
        LOG.info("  train %-22s n=%6d", label, feats.shape[0])

    X = np.vstack(blocks)
    obs = {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        SAMPLE_FIELD: np.concatenate(embryo),
        COL_CLASS: np.concatenate(classes),
    }
    ref = _make_adata(ad_mod, X, obs, var_ref)

    if args.max_ref_cells and ref.n_obs > args.max_ref_cells:
        keep = rng.choice(ref.n_obs, args.max_ref_cells, replace=False)
        keep.sort()
        ref = ref[keep].copy()
        LOG.info("  reference subsampled to %d cells (--max_ref_cells)", ref.n_obs)

    # Fail loudly rather than let upstream invent a single sample.
    if SAMPLE_FIELD not in ref.obs or ref.obs[SAMPLE_FIELD].isna().any():
        raise RuntimeError(f"obs['{SAMPLE_FIELD}'] missing — kNN would cross slices")
    n_samples = ref.obs[SAMPLE_FIELD].nunique()
    LOG.info("reference: %d cells, %d features, %d sections",
             ref.n_obs, ref.n_vars, n_samples)
    if n_samples != len(train_files) and not args.max_ref_cells:
        LOG.warning("  expected %d sections, got %d", len(train_files), n_samples)

    out = work / "reference_train.h5ad"
    ref.write_h5ad(out)
    Xr = np.asarray(ref.X)
    return (out, [str(v) for v in ref.var_names], int(ref.n_obs),
            float(Xr.min()), float(Xr.max()))


def _make_adata(ad_mod, X: np.ndarray, obs: Dict[str, np.ndarray], var_names):
    import pandas as pd
    a = ad_mod.AnnData(X=np.asarray(X, dtype=np.float32),
                       obs=pd.DataFrame(obs).reset_index(drop=True))
    a.var_names = [str(v) for v in var_names]
    a.obs_names = [str(i) for i in range(a.n_obs)]
    return a


def load_query(ad_mod, test_file: Path, args):
    """Read one test slice into arrays: features, truth coords, labels, obs_names.

    Coordinates are loaded for SCORING only and are never written into the query
    object handed to upstream (see write_query_chunk) — inference needs only .X
    and .var_names, so the truth cannot leak into the prediction.
    """
    a = ad_mod.read_h5ad(test_file)
    feats = _feature_matrix(a, args.use_obsm, args.expression_mode,
                           what=f"test slice {test_file.name}")
    vn = _var_names_for(a, args.use_obsm)
    coords_true = _coords_of(a)
    if COL_CLASS not in a.obs:
        # write_slice_artifacts omits the column entirely when cell_class is
        # None, and compute_extended_metrics indexes it unconditionally -- the
        # run would score as NaN rather than fail. Refuse here instead.
        raise ValueError(
            f"{test_file.name}: obs[{COL_CLASS!r}] missing. The scorer requires "
            f"the cell_class column in metadata_*.csv (a 3-column file yields "
            f"NaN sum_rssd by construction).")
    cls = a.obs[COL_CLASS].astype(str).to_numpy()
    obs_names = np.asarray(a.obs_names, dtype=object)
    return feats, coords_true, cls, obs_names, vn


def write_query_chunk(ad_mod, feats: np.ndarray, var_names, work: Path,
                      tag: str) -> Path:
    """Write one query h5ad (features only, no coordinates)."""
    q = _make_adata(ad_mod, feats, {"placeholder": np.zeros(feats.shape[0])},
                    var_names)
    path = work / f"query_{tag}.h5ad"
    q.write_h5ad(path)
    return path


def chunk_bounds(n: int, chunk: int) -> List[tuple]:
    """Contiguous [start, stop) row ranges covering n rows.

    Chunking is BIT-IDENTICAL for this method: ``inference.map_to_ST`` takes,
    for each query row independently, the argmax over the FULL reference, and
    the discarded query x query similarity matrix feeds only runMDS/eval, which
    we never invoke. A slice at or below ``chunk`` yields a single range, i.e.
    exactly the original single-call path.
    """
    if chunk and 0 < chunk < n:
        return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    return [(0, n)]


def estimate_peak_bytes(n_query_chunk: int, n_ref: int) -> int:
    """Peak resident bytes of one upstream inference call.

    ``utils.getModelSimMat`` builds a float32 similarity matrix, a float32
    sorted copy and an int64 argsort index (16 B/element resident, ~24 B/element
    at the transient doubling inside the row loop). It runs UNCONDITIONALLY on
    query x query (inference.py:120, before the enable_denovo guard) and those
    arrays stay bound while map_to_ST builds query x reference. Hence
    ``max(24*Q^2, 16*Q^2 + 24*Q*R)``.
    """
    q, r = int(n_query_chunk), int(n_ref)
    return max(24 * q * q, 16 * q * q + 24 * q * r)


# ---------------------------------------------------------------------------
def write_parameters(repo: Path, work: Path, args) -> Path:
    """Copy the authors' parameter file, overriding ONLY what we were asked to."""
    src = repo / "parameters" / (
        "parameters_singleCell.json" if args.single_cell else "parameters_spot.json")
    if not src.exists():
        raise FileNotFoundError(f"upstream parameter file missing: {src}")
    params = json.loads(src.read_text())
    upstream_epochs = params.get("training_epoch")
    if args.smoke_test:
        params["training_epoch"] = 5
    elif args.epochs:
        params["training_epoch"] = int(args.epochs)
    out = work / "parameters.json"
    out.write_text(json.dumps(params, indent=2))
    LOG.info("parameters: %s (training_epoch %s -> %s)",
             src.name, upstream_epochs, params["training_epoch"])
    return out


def run_cmd(cmd: List[str], cwd: Path, dry: bool) -> None:
    LOG.info("$ %s", " ".join(str(c) for c in cmd))
    if dry:
        return
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(map(str, cmd))}")
    LOG.info("  ok (%.1f s)", time.time() - t0)


def find_checkpoint(model_dir: Path) -> Path:
    cks = sorted(model_dir.glob("epoch_*.pt"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    if not cks:
        raise FileNotFoundError(f"no epoch_*.pt in {model_dir}")
    return cks[-1]


def read_predicted_coords(ad_mod, path: Path, n_expected: int) -> np.ndarray:
    """Read uns['referenced x'/'y'] WITHOUT materialising the N x N matrix.

    Upstream stores the dense query x query float32 similarity matrix in
    ``uns['cosine sim of rep']`` of this same file (16 GB for a 63k-cell slice),
    and anndata materialises ``uns`` in full EVEN WITH ``backed="r"`` (verified),
    so a plain read_h5ad here can OOM on exactly the slices we just spent hours
    predicting. h5py reads only the two coordinate vectors.
    """
    try:
        import h5py
    except ImportError:                                   # pragma: no cover
        h5py = None
    if h5py is not None:
        with h5py.File(str(path), "r") as fh:
            uns = fh.get("uns")
            if uns is None:
                raise KeyError(f"{path}: no /uns group")
            missing = [k for k in (UNS_X, UNS_Y) if k not in uns]
            if missing:
                raise KeyError(f"{path}: uns{missing} missing; "
                               f"have {list(uns.keys())}")
            xy = np.column_stack([
                np.asarray(uns[UNS_X][...], dtype=np.float64).ravel(),
                np.asarray(uns[UNS_Y][...], dtype=np.float64).ravel(),
            ])
    else:
        a = ad_mod.read_h5ad(path)
        for k in (UNS_X, UNS_Y):
            if k not in a.uns:
                raise KeyError(f"{path}: uns[{k!r}] missing; "
                               f"have {list(a.uns.keys())}")
        xy = np.column_stack([np.asarray(a.uns[UNS_X], dtype=np.float64).ravel(),
                              np.asarray(a.uns[UNS_Y], dtype=np.float64).ravel()])
    if xy.shape[0] != n_expected:
        raise ValueError(f"{path}: got {xy.shape[0]} coords for {n_expected} cells")
    if not np.isfinite(xy).all():
        raise ValueError(f"{path}: non-finite predicted coordinates")
    return xy


def reference_coord_pairs(ad_mod, ref_path: Path) -> set:
    """The set of (x, y) pairs present in the reference.

    ``inference.map_to_ST`` assigns ``ref_coors[ind[0]]`` -- a VERBATIM copy of
    one reference cell's obs['x'/'y'] -- so every predicted position must be a
    member of this set. Read backed: we want two obs columns, not the matrix.
    """
    a = ad_mod.read_h5ad(ref_path, backed="r")
    xs = np.asarray(a.obs["x"], dtype=np.float64)
    ys = np.asarray(a.obs["y"], dtype=np.float64)
    return {(round(float(x), 9), round(float(y), 9)) for x, y in zip(xs, ys)}


def check_predictions(pred: np.ndarray, ref_pairs: set, label: str,
                      lo: float = -0.5, hi: float = 0.5) -> int:
    """Reject predictions that cannot be a top-1 reference copy.

    Both failure modes below are invisible in the metrics -- they produce
    finite, plausible, merely-poor numbers that read as "weak baseline", which
    is exactly the conclusion we must not reach by accident.

      * out of frame: predictions are copied from the normalised reference, so
        anything outside [lo, hi] means the readout is not what we think it is
        (microns? de-novo MDS output? row indices? similarity scores?).
      * total collapse: every cell mapped to one position. Downstream this
        yields NaN contact F1 only by luck; fail here instead.

    The pair-membership check catches an x/y transposition, which the range
    check cannot -- these slices are nearly square, so a swap stays in range.
    That matters because Sum RSSD fits a rotation only: a reflection inflates
    it while the isometry-invariant Spearman and Contact F1 stay healthy. It is
    a WARNING not a raise, because a float32 round-trip through h5ad would also
    trip it and that would be a false abort mid-sweep.
    """
    eps = 1e-6
    if pred.min() < lo - eps or pred.max() > hi + eps:
        raise ValueError(
            f"{label}: predictions leave the normalised frame [{lo}, {hi}] "
            f"(min={pred.min():.6g}, max={pred.max():.6g}). Upstream copies "
            f"reference obs['x'/'y'] verbatim, so this is not a top-1 copy.")
    pairs = [(round(float(a), 9), round(float(b), 9)) for a, b in pred]
    uniq = len(set(pairs))
    if uniq == 1:
        raise ValueError(
            f"{label}: all {len(pairs)} predictions collapsed onto a single "
            f"position. Scoring this would report a degenerate model as a "
            f"method result.")
    missing = sum(1 for p in pairs if p not in ref_pairs)
    if missing:
        LOG.warning("  %s: %d/%d predicted positions are NOT reference pairs "
                    "(expected 0 for a verbatim copy; an x/y swap looks like "
                    "this)", label, missing, len(pairs))
    if uniq <= 2 or uniq < 0.02 * len(pairs):
        LOG.warning("  %s: only %d distinct positions for %d cells "
                    "(near-collapse)", label, uniq, len(pairs))
    return uniq


# ---------------------------------------------------------------------------
_SPOT_HINTS = ("visium", "dlpfc", "spot", "cytassist", "stereo", "slide")
_IMAGING_HINTS = ("mmc", "merfish", "merscope", "xenium", "starmap", "cns",
                  "breast", "cosmx", "seqfish")


def platform_check(dataset: str, single_cell: bool, force: bool) -> None:
    """Refuse an imaging/spot mismatch between the dataset and the parameter file.

    The paper's ``k_nearest_positives`` is platform-specific -- 80 for
    imaging-resolution ST (SeqFISH/MERSCOPE/Xenium/STARmap) and 20 for
    spot-resolution ST (Stereo-seq/10x Visium) -- and that is the ONLY difference
    between the authors' two parameter files. Running Visium with k=80 is not the
    published method, and it is a silent misconfiguration: nothing downstream
    would look wrong.
    """
    if force:
        LOG.warning("--force_platform: skipping the dataset/parameter-file "
                    "consistency check (k_nearest_positives=%s)",
                    "80 (imaging)" if single_cell else "20 (spot)")
        return
    d = dataset.lower()
    is_spot = any(h in d for h in _SPOT_HINTS)
    is_imaging = any(h in d for h in _IMAGING_HINTS)
    if is_spot and is_imaging:          # ambiguous name; let the user decide
        return
    if is_spot and single_cell:
        raise SystemExit(
            f"[cellcontrast] dataset {dataset!r} looks SPOT-based (Visium), but "
            f"--single_cell is set, which selects parameters_singleCell.json with "
            f"k_nearest_positives=80. The paper uses k=20 for spot ST. Pass "
            f"--no_single_cell (or --force_platform to override).")
    if is_imaging and not single_cell:
        raise SystemExit(
            f"[cellcontrast] dataset {dataset!r} looks IMAGING-based, but "
            f"--no_single_cell selects parameters_spot.json with "
            f"k_nearest_positives=20. The paper uses k=80 for imaging ST. Drop "
            f"--no_single_cell (or --force_platform to override).")


def write_seeded_launcher(work: Path, repo: Path, seed: int,
                          sub_argv: List[str]) -> Path:
    """Generate a launcher that seeds stdlib ``random`` before upstream runs.

    Upstream shuffles the epoch's cell order and picks each anchor's ONE positive
    with an unseeded module-level ``random.shuffle`` (loadData.py:107 and :129)
    and exposes no seed argument, so two runs of an identical command differ.

    torch's RNG is deliberately LEFT ALONE at upstream's ``torch.manual_seed(0)``
    (model.py:6). Re-seeding it would move the weight initialisation off the
    authors' published value for 4 of 5 replicates, which changes the method and
    would have to be pre-registered. Seeding only stdlib random makes a replicate
    reproducible without changing the distribution it is drawn from.
    """
    launcher = work / "seeded_launch.py"
    launcher.write_text(
        "# generated by run_cellcontrast.py -- seeds stdlib random, then hands\n"
        "# control to the UNMODIFIED upstream dispatcher.\n"
        "import random, sys, runpy\n"
        f"random.seed({int(seed)})\n"
        "try:\n"
        "    import numpy as _np\n"
        f"    _np.random.seed({int(seed)} % (2 ** 32))\n"
        "except Exception:\n"
        "    pass\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        f"sys.argv = {list(sub_argv)!r}\n"
        f"runpy.run_path({str(repo / 'cellContrast.py')!r}, run_name='__main__')\n"
    )
    return launcher


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)

    repo = Path(args.cellcontrast_repo).resolve()
    entry = repo / "cellContrast.py"
    if not entry.exists():
        raise FileNotFoundError(f"cellContrast.py not found under {repo}")

    data_dir = Path(args.data_dir).resolve()
    dataset = args.dataset or data_dir.name
    # Fail on a platform/parameter mismatch BEFORE doing hours of work.
    platform_check(dataset, args.single_cell, args.force_platform)
    if args.use_obsm == "spatial":
        raise SystemExit(
            "[cellcontrast] --use_obsm spatial would feed ground-truth "
            "coordinates to the encoder as features. Refused. For cns_luna the "
            "shared latent is already in .X: drop --use_obsm and pass "
            "--expression_mode silver_raw.")
    ts = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / dataset / "cellcontrast_inference" / ts
    results = out_dir / "test_results"
    work = out_dir / "work"
    for d in (results, work):
        d.mkdir(parents=True, exist_ok=True)
    LOG.info("CellContrast wrapper %s\n  dataset=%s seed=%d\n  out=%s",
             __version__, dataset, args.seed, out_dir)

    train_files = discover_split_files(data_dir, "train")
    test_files = discover_split_files(data_dir, "test")
    excl = {s.strip() for s in args.exclude_test_files.split(",") if s.strip()}
    if excl:
        test_files = [p for p in test_files if p.name not in excl]
    if not train_files or not test_files:
        raise RuntimeError(f"need both splits in {data_dir}; "
                           f"found {len(train_files)} train / {len(test_files)} test")
    if args.smoke_test:
        train_files, test_files = train_files[:2], test_files[:1]
        LOG.warning("SMOKE TEST: 2 train / 1 test slice, 5 epochs — do not report")
    LOG.info("splits: %d train / %d test", len(train_files), len(test_files))

    ad_mod = _import_anndata()
    params_path = write_parameters(repo, work, args)
    params_all = json.loads(params_path.read_text())
    k_pos = int(params_all.get("k_nearest_positives", 0))

    # ---- reference (also the coordinate source at inference) ----------------
    ref_path, ref_var, ref_n_obs, ref_xmin, ref_xmax = build_reference(
        ad_mod, train_files, args, rng, work, k_pos=k_pos)

    # ---- train once --------------------------------------------------------
    model_dir = work / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: we deliberately do NOT pass -sc/--single_cell.
    #
    # Upstream train.py contains
    #     if(args.single_cell):
    #         args.parameter_file_path = "./parameters/parameters_singleCell.json"
    # which HARD-OVERWRITES whatever --parameter_file_path we passed with the
    # repo's own relative path, silently discarding our file (so --epochs and
    # --smoke_test had no effect and every run did the full 3000 epochs).
    # Verified against the source that `single_cell` is used for nothing else —
    # it never appears again after that reassignment — so omitting the flag and
    # passing our own copy of parameters_singleCell.json is EXACTLY equivalent
    # while letting our overrides actually apply. ``--single_cell`` on OUR CLI
    # still selects which upstream file write_parameters() copies.
    #
    # We invoke the dispatcher through a generated launcher that seeds stdlib
    # ``random`` first (see write_seeded_launcher). Upstream's positive sampling
    # and epoch shuffling are otherwise unseeded, so runs were not reproducible.
    # The upstream code itself is untouched, and torch's RNG is left at the
    # authors' hardcoded manual_seed(0).
    train_argv = ["cellContrast.py", "train",
                  "--train_data_path", str(ref_path),
                  "--save_folder", str(model_dir),
                  "--parameter_file_path", str(params_path)]
    cmd = [sys.executable, str(write_seeded_launcher(work, repo, args.seed,
                                                    train_argv))]
    # An earlier version warned here that upstream orders feature columns via a
    # Python set, so PYTHONHASHSEED had to be pinned for reproducibility. That
    # was WRONG, and it misled two independent code audits into reporting a
    # top-severity bug that does not exist. inference.format_query reindexes
    # through the ORDERED train_genes list saved in the checkpoint --
    # `[query_adata.var_names.get_loc(g) for g in train_genes]` -- and uses a
    # set only for the membership test that triggers sys.exit. Feature column
    # order is therefore deterministic regardless of hash seed, for both the
    # query and the reference (format_query is applied to both).
    run_cmd(cmd, cwd=repo, dry=args.dry_run)

    # Fail fast here rather than letting every inference call fail confusingly.
    # Also verify the epoch count we ASKED for is the one that actually ran:
    # upstream names checkpoints epoch_<N>.pt, so a mismatch means our parameter
    # file was ignored (this is exactly how the -sc override above was caught).
    if not args.dry_run:
        ckpt = find_checkpoint(model_dir)
        LOG.info("trained checkpoint: %s (%.1f MB)",
                 ckpt.name, ckpt.stat().st_size / 1e6)
        # Upstream inference.load_model builds the checkpoint path as
        #     <model_folder>/epoch_<params['training_epoch']>.pt
        # so we assert THAT EXACT path exists — the same string inference will
        # construct. Checking only "some epoch_*.pt exists" is not enough: if
        # training silently used different parameters than we passed (the -sc
        # bug above), training writes epoch_3000.pt while inference looks for
        # epoch_5.pt and dies with a bare FileNotFoundError 50 minutes later.
        requested = int(json.loads(params_path.read_text())["training_epoch"])
        expected = model_dir / f"epoch_{requested}.pt"
        if not expected.exists():
            raise RuntimeError(
                f"checkpoint {expected.name} is missing (found {ckpt.name}). "
                f"Inference derives the filename from training_epoch={requested} "
                f"in our parameter file, so training must have used DIFFERENT "
                f"parameters than we passed — the recorded hyperparameters would "
                f"not describe this model. Do not report this run. "
                f"(Known cause: passing -sc makes upstream train.py overwrite "
                f"--parameter_file_path with its own default.)")

    # ---- inference per test slice ------------------------------------------
    per_slice: List[Dict[str, object]] = []
    ref_pairs: set = set()
    manifest: Dict[str, object] = {}
    if not args.dry_run:
        ref_pairs = reference_coord_pairs(ad_mod, ref_path)
        # Write the manifest BEFORE the first slice, not after the last. Slice
        # artifacts are written inside the loop and are independently
        # scoreable, so a run that dies partway (wall clock, OOM, upstream
        # exception) used to leave scoreable CSVs with NO manifest -- and the
        # scorer's smoke-test filter deliberately KEEPS manifest-less runs, so
        # a crashed smoke test would be auto-discovered and averaged in as a
        # real replicate. "status" is flipped to complete at the end.
        manifest = {
            "wrapper_version": __version__,
            "method": "cellcontrast",
            "upstream": "https://github.com/HKU-BAL/CellContrast (MIT)",
            "upstream_commit": _git_commit(repo),
            "dataset": dataset, "seed": args.seed, "timestamp": ts,
            "n_train_slices": len(train_files), "n_test_slices": len(test_files),
            "parameters": json.loads(params_path.read_text()),
            "expression_mode": args.expression_mode,
            "use_obsm": args.use_obsm,
            "max_train_cells": args.max_train_cells,
            "max_ref_cells": args.max_ref_cells,
            "single_cell_mode": args.single_cell,
            "k_nearest_positives": k_pos,
            "smoke_test": args.smoke_test,
            # --seed now DOES reach training: we launch the upstream dispatcher
            # through a generated shim that seeds stdlib random first (upstream
            # shuffles the epoch order and samples each anchor's positive with an
            # unseeded random.shuffle and takes no seed argument). torch is left
            # at upstream's hardcoded manual_seed(0), so the weight init is the
            # authors' for every replicate; replicates differ in positive
            # sampling / batch composition only. Still NOT a torch-init sweep.
            "seed_is_effective": True,
            "seed_scope": "stdlib random (positive sampling + epoch shuffle) and "
                          "our subsampling RNG; torch init left at upstream's "
                          "manual_seed(0)",
            "inference_reference": "training-donor slices (pre-registered; test slice "
                                   "would leak its coordinate set)",
            "coordinate_frame": (
                f"per-slice min-max [-0.5,0.5], {args.coord_frame}; inverted with "
                f"the test slice's own scaler; artifacts in ORIGINAL microns"),
            "coord_frame": args.coord_frame,
            "artifact_frame": "original_microns",
            "data_dir": str(data_dir),
            "exclude_test_files": sorted(excl),
            "feature_source": (f"obsm[{args.use_obsm!r}]" if args.use_obsm else ".X"),
            "n_features": len(ref_var),
            "reference_n_obs": ref_n_obs,
            "reference_X_min": ref_xmin,
            "reference_X_max": ref_xmax,
            "query_chunk": args.query_chunk,
            "torch_version": _torch_version(),
            "status": "incomplete",
            "per_slice": per_slice,
        }
        write_run_manifest(out_dir, manifest)
    for i, tf in enumerate(test_files, 1):
        label = section_label_from_filename(tf)
        LOG.info("[%d/%d] %s", i, len(test_files), label)
        if args.dry_run:
            # Don't read slices or write h5ads on a dry run — just show the shape
            # of the command that would be issued.
            LOG.info("$ (dry) %s inference --query_data_path %s ... --ref_data_path %s",
                     sys.executable, work / f"query_{label}[_chunkK].h5ad", ref_path)
            continue

        feats, coords_true, cls, obs_names, vn = load_query(ad_mod, tf, args)
        n_cells = feats.shape[0]
        # Cross-split panel check: upstream sys.exit()s if any train gene is
        # absent from the query, and for the obsm path it synthesises positional
        # names, so a WIDER test embedding would pass a subset test and upstream
        # would silently use the first d_train columns.
        if args.use_obsm:
            if len(vn) != len(ref_var):
                raise ValueError(
                    f"{tf.name}: obsm[{args.use_obsm!r}] width {len(vn)} != train "
                    f"width {len(ref_var)}; positional names would silently "
                    f"misalign the features.")
        else:
            missing = [g for g in ref_var if g not in set(map(str, vn))]
            if missing:
                raise ValueError(
                    f"{tf.name}: {len(missing)} training feature(s) absent from the "
                    f"test panel (e.g. {missing[:5]}); upstream format_query would "
                    f"sys.exit mid-run.")

        bounds = chunk_bounds(n_cells, args.query_chunk)
        peak = estimate_peak_bytes(max(b - a for a, b in bounds), ref_n_obs)
        LOG.info("  n=%d  ref=%d  chunks=%d  est. peak %.1f GB",
                 n_cells, ref_n_obs, len(bounds), peak / 1e9)
        if args.max_mem_gb and peak > args.max_mem_gb * 1e9:
            raise SystemExit(
                f"[cellcontrast] {label}: estimated inference peak "
                f"{peak/1e9:.1f} GB exceeds --max_mem_gb {args.max_mem_gb:.1f}. "
                f"Lower --query_chunk (currently {args.query_chunk}) — the "
                f"dominant term is 24*chunk*n_ref. Do NOT use --max_ref_cells "
                f"for this: shrinking the reference changes the candidate "
                f"position set, i.e. the method.")

        preds = []
        for ci, (lo_i, hi_i) in enumerate(bounds):
            tag = label if len(bounds) == 1 else f"{label}_chunk{ci:03d}"
            q_path = write_query_chunk(ad_mod, feats[lo_i:hi_i], vn, work, tag)
            recon = work / f"recon_{tag}.h5ad"
            cmd = [sys.executable, str(entry), "inference",
                   "--query_data_path", str(q_path),
                   "--model_folder", str(model_dir),
                   "--parameter_file_path", str(params_path),
                   "--ref_data_path", str(ref_path),
                   "--save_path", str(recon)]
            if len(bounds) > 1:
                LOG.info("  chunk %d/%d rows [%d, %d)",
                         ci + 1, len(bounds), lo_i, hi_i)
            run_cmd(cmd, cwd=repo, dry=False)
            preds.append(read_predicted_coords(ad_mod, recon, hi_i - lo_i))
            for tmp in (recon, q_path):
                try:
                    tmp.unlink()     # recon holds a dense chunk x chunk matrix
                except OSError:
                    pass
        pred_norm = np.vstack(preds)
        if pred_norm.shape[0] != n_cells:
            raise ValueError(f"{label}: concatenated {pred_norm.shape[0]} "
                             f"predictions for {n_cells} cells")

        # Validate BEFORE writing artifacts: a bad readout must not reach the
        # scorer, where it is indistinguishable from a weak baseline. Also
        # reports the duplicate rate that top-1 copying necessarily produces
        # (never jitter it away — that would alter the method).
        uniq = check_predictions(pred_norm, ref_pairs, label)
        # predictions are in the shared normalised frame -> back to this slice's
        # microns, using the SAME frame convention the reference was built with.
        scaler = SliceCoordScaler(
            isotropic=(args.coord_frame == "isotropic")).fit(coords_true)
        pred_orig = scaler.inverse_transform(pred_norm)
        write_slice_artifacts(results / label, pred_orig, coords_true, cls,
                              index=obs_names)
        LOG.info("  n=%d  distinct predicted positions=%d (%.1f%%)",
                 n_cells, uniq, 100.0 * uniq / n_cells)
        per_slice.append({"section": label, "n_cells": n_cells,
                          "n_chunks": len(bounds),
                          "est_peak_gb": round(peak / 1e9, 2),
                          "distinct_predicted_positions": uniq})

    if args.dry_run:
        LOG.info("dry run complete")
        return 0

    # Same dict, now with every slice recorded; flip status so a partial run is
    # distinguishable from a finished one without counting directories.
    manifest["per_slice"] = per_slice
    manifest["status"] = ("complete" if len(per_slice) == len(test_files)
                          else f"incomplete ({len(per_slice)}/{len(test_files)} slices)")
    write_run_manifest(out_dir, manifest)
    LOG.info("\nwrote %d slice(s) to %s", len(per_slice), results)
    LOG.info("next: score with compute_extended_metrics.py --methods cellcontrast")
    return 0


def _torch_version() -> Optional[str]:
    """Record which torch actually ran. Not cosmetic: the documented CPU escape
    hatch TORCH_SPEC="torch" resolves to >=2.6 where torch.load defaults to
    weights_only=True, and upstream stores a pandas Index in the checkpoint
    (train.py:38) and loads it bare (inference.py:26-33) — that combination
    fails at INFERENCE, after training has completed."""
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return None


def _git_commit(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
