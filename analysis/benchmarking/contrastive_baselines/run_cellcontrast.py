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
slice's coordinates to [-0.5, 0.5] before training. Predictions therefore come
back in that shared normalised frame, and we invert them with the TEST slice's
own scaler so Sum RSSD is computed against original-scale truth (Spearman and
Contact F1 are rank/threshold based and unaffected).

Usage (cortex):
    python run_cellcontrast.py \
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \
        --cellcontrast_repo /nfs/team361/sb75/CellContrast \
        --out_root /nfs/team361/sb75/scgg-reproducibility/artifacts \
        --dataset mmc_luna --seed 0

Usage (CNS — needs subsampling and the Harmony latent; see --help):
    python run_cellcontrast.py --data_dir .../cns_luna --dataset cns_luna \
        --use_obsm X_harmony --max_train_cells 150000 ...

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
UNS_SIM = "cosine sim of rep"          # dense N x N; strip before saving
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
                   help="pass -sc (imaging-resolution ST). Default True for MERFISH.")
    p.add_argument("--no_single_cell", dest="single_cell", action="store_false")
    p.add_argument("--expression_mode", default="log2", choices=("log2", "silver_raw"),
                   help="log2 = log2(1+x), parity with LUNA/G2T/CeLEry (default). "
                        "silver_raw hands over the silver matrix untouched.")
    p.add_argument("--use_obsm", default=None,
                   help="use adata.obsm[KEY] as the feature matrix instead of .X "
                        "(cns_luna: the 600-d Harmony latent, matching our protocol)")
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


def _feature_matrix(adata, use_obsm: Optional[str], expression_mode: str) -> np.ndarray:
    """Feature matrix for the encoder, matching our other baselines' input."""
    if use_obsm:
        if use_obsm not in adata.obsm:
            raise KeyError(f"obsm[{use_obsm!r}] absent; have {list(adata.obsm.keys())}")
        M = np.asarray(adata.obsm[use_obsm], dtype=np.float32)
        # An embedding is already normalised; log2 on a latent would be nonsense.
        return M
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
    return X


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
) -> Path:
    """Concatenate training slices into the single reference object upstream wants.

    Sets obs['x'], obs['y'] to PER-SLICE NORMALISED coordinates and obs['embryo']
    to the slice label so the spatial kNN never crosses sections.
    """
    per_slice_cap = None
    if args.max_train_cells:
        per_slice_cap = max(1, args.max_train_cells // max(1, len(train_files)))

    blocks, xs, ys, embryo, classes, var_ref = [], [], [], [], [], None
    for f in train_files:
        a = ad_mod.read_h5ad(f)
        label = section_label_from_filename(f)
        feats = _feature_matrix(a, args.use_obsm, args.expression_mode)
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
        cn = SliceCoordScaler().fit(coords).transform(coords)
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
    return out


def _make_adata(ad_mod, X: np.ndarray, obs: Dict[str, np.ndarray], var_names):
    import pandas as pd
    a = ad_mod.AnnData(X=np.asarray(X, dtype=np.float32),
                       obs=pd.DataFrame(obs).reset_index(drop=True))
    a.var_names = [str(v) for v in var_names]
    a.obs_names = [str(i) for i in range(a.n_obs)]
    return a


def build_query(ad_mod, test_file: Path, args, work: Path):
    """Query object for one test slice: features only, no coordinates.

    Withholding coordinates is deliberate — inference needs only .X and
    .var_names, so the truth cannot leak into the prediction even accidentally.
    """
    a = ad_mod.read_h5ad(test_file)
    feats = _feature_matrix(a, args.use_obsm, args.expression_mode)
    vn = _var_names_for(a, args.use_obsm)
    coords_true = _coords_of(a)
    cls = (a.obs[COL_CLASS].astype(str).to_numpy()
           if COL_CLASS in a.obs else None)
    q = _make_adata(ad_mod, feats, {"placeholder": np.zeros(feats.shape[0])}, vn)
    path = work / f"query_{section_label_from_filename(test_file)}.h5ad"
    q.write_h5ad(path)
    return path, coords_true, cls, feats.shape[0]


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
    a = ad_mod.read_h5ad(path)
    for k in (UNS_X, UNS_Y):
        if k not in a.uns:
            raise KeyError(f"{path}: uns[{k!r}] missing; have {list(a.uns.keys())}")
    xy = np.column_stack([np.asarray(a.uns[UNS_X], dtype=np.float64).ravel(),
                          np.asarray(a.uns[UNS_Y], dtype=np.float64).ravel()])
    if xy.shape[0] != n_expected:
        raise ValueError(f"{path}: got {xy.shape[0]} coords for {n_expected} cells")
    if not np.isfinite(xy).all():
        raise ValueError(f"{path}: non-finite predicted coordinates")
    return xy


# ---------------------------------------------------------------------------
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

    # ---- reference (also the coordinate source at inference) ----------------
    ref_path = build_reference(ad_mod, train_files, args, rng, work)

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
    cmd = [sys.executable, str(entry), "train",
           "--train_data_path", str(ref_path),
           "--save_folder", str(model_dir),
           "--parameter_file_path", str(params_path)]
    env_note = os.environ.get("PYTHONHASHSEED")
    if env_note is None:
        LOG.warning("PYTHONHASHSEED unset — upstream intersects genes via a Python "
                    "set, so column order can vary between runs. Export "
                    "PYTHONHASHSEED=0 for byte-reproducibility.")
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
    for i, tf in enumerate(test_files, 1):
        label = section_label_from_filename(tf)
        LOG.info("[%d/%d] %s", i, len(test_files), label)
        q_path, coords_true, cls, n_cells = build_query(ad_mod, tf, args, work)
        recon = work / f"recon_{label}.h5ad"
        cmd = [sys.executable, str(entry), "inference",
               "--query_data_path", str(q_path),
               "--model_folder", str(model_dir),
               "--parameter_file_path", str(params_path),
               "--ref_data_path", str(ref_path),
               "--save_path", str(recon)]
        run_cmd(cmd, cwd=repo, dry=args.dry_run)
        if args.dry_run:
            continue

        pred_norm = read_predicted_coords(ad_mod, recon, n_cells)
        # predictions are in the shared normalised frame -> back to this slice's microns
        pred_orig = SliceCoordScaler().fit(coords_true).inverse_transform(pred_norm)
        write_slice_artifacts(results / label, pred_orig, coords_true, cls)

        # top-1 copying quantises predictions onto reference positions; report the
        # duplicate rate rather than hiding it (and never jitter it away — that
        # would alter the method).
        uniq = len({(round(a, 9), round(b, 9)) for a, b in pred_norm})
        LOG.info("  n=%d  distinct predicted positions=%d (%.1f%%)",
                 n_cells, uniq, 100.0 * uniq / n_cells)
        per_slice.append({"section": label, "n_cells": n_cells,
                          "distinct_predicted_positions": uniq})
        try:
            recon.unlink()          # each holds a dense NxN similarity matrix
        except OSError:
            pass

    if args.dry_run:
        LOG.info("dry run complete")
        return 0

    write_run_manifest(out_dir, {
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
        "smoke_test": args.smoke_test,
        "inference_reference": "training-donor slices (pre-registered; test slice "
                               "would leak its coordinate set)",
        "coordinate_frame": "per-slice min-max [-0.5,0.5]; inverted with the test "
                            "slice's own scaler",
        "per_slice": per_slice,
    })
    LOG.info("\nwrote %d slice(s) to %s", len(per_slice), results)
    LOG.info("next: score with compute_extended_metrics.py --methods cellcontrast")
    return 0


def _git_commit(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
