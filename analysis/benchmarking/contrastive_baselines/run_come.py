#!/usr/bin/env python3
r"""run_come.py — run the AUTHORS' published COME on our benchmark.

COME: "contrastive mapping learning for spatial reconstruction of single-cell RNA
sequencing data", Wei, Chen, Wang, Shen, Liu, Wu, Wong, Bioinformatics 41(3):
btaf083, 2025 (doi:10.1093/bioinformatics/btaf083). Code: github.com/cindyway/COME
(cited in our manuscript as ``come2025``).

WHAT WE REUSE VERBATIM vs WHAT IS OURS
-------------------------------------
All modelling is the authors' code, imported unmodified from their repo:
``model.Model`` (AutoEncoder + MapNet + the full five-term loss),
``model.ContrastiveLoss``, ``utils.normalize_type`` (their preprocessing),
``utils.cell_type_2_martix`` (their cell-type contrastive mask),
``configure.get_default_config`` (their per-platform hyperparameters), and their
own ``train_eval.pretrain_ae`` / ``train_eval.train`` loops, with the two Adam
optimizers constructed exactly as their ``main()`` constructs them.

What is OURS is only (a) data plumbing, because their ``datasets.load_data``
hardcodes five dataset names and five file paths, and (b) the coordinate
read-out, because THEIR SHIPPED ``main()`` PRODUCES NO COORDINATES AT ALL --
it is a 10-fold cross-validation over GENES that scores gene-expression
imputation (PCC/SSIM/RMSE/JSD). Our task is coordinates, so their evaluation
harness is the one part that cannot be reused.

Verified against the upstream source before writing (not guessed):
  * ``MapNet.Coefficient`` is an ``nn.Parameter`` of shape (n_spots, n_cells) --
    COME is TRANSDUCTIVE. There is no encoder-only path that can be applied to
    unseen cells, so the model is re-fitted for every (reference, test slice)
    pair. It never sees the test coordinates, only the test EXPRESSION, which is
    the same position novoSpaRc/Tangram/SpaOTsc occupy.
  * COME NEVER USES SPATIAL COORDINATES DURING TRAINING. ``loss_fn`` consumes
    only x1/x2 expression, the AE reconstructions, the cell-type mask and the
    cross mask. Coordinates enter ONLY at read-out, so the per-slice coordinate
    frame (isotropic vs per-axis) is a no-op for COME, unlike CellContrast where
    it changes the k-NN positive graph.
  * READ-OUT: upstream's own ``Model.cross_mask`` does
    ``Coefficient.max(dim=0)`` -> for each cell, the single highest-coefficient
    SPOT. We use exactly that spot's (x, y). This is the authors' own notion of
    the spot a cell maps to, and it mirrors CellContrast's top-1 copy. A
    "weighted average over spots" is NOT offered: ``Coefficient`` is a free
    parameter (initialised to 1/(n*m), L2-regularised, never softmaxed or
    clamped), so it is not a distribution and a weighted mean is not well
    defined.
  * ``--pretrain`` / ``--train`` upstream are ``action='store_false'``, so the
    README's own ``python train_val.py --pretrain --train`` DISABLES both
    training stages. We call their loops directly with both enabled.
  * PREPROCESSING is theirs and depends on ``--sttype``:
      image    -> per-gene ``MinMaxScaler`` on both matrices (NOT the "uniform
                  total counts" the paper's text describes -- the code is
                  authoritative for reproduction)
      sequence -> ``normalize_total(1e4)`` + ``log1p`` on both
    then the gene sets are intersected and SORTED. We therefore hand over the
    silver matrix UNTOUCHED and let COME normalise as its authors intended --
    which also sidesteps the log2-vs-identity question entirely.
  * ``datasets.load_data`` calls ``sc.pp.filter_cells(min_genes=1)``, which
    DROPS cells. We do not use load_data, but we replicate the check explicitly
    because a dropped query cell would silently break the row correspondence
    between predictions and truth (see --on_empty_cells).
  * ``utils.cell_type_2_martix`` reads ``obs['cell_type']`` and, if absent,
    falls back to ``obs_names`` -- which makes every cell its own type and
    silently reduces the cell-type contrastive term to an identity mask. Our
    silver stores labels in ``obs['cell_class']``, so we copy them across and
    assert the column is present.
  * that function also WRITES and ``train_eval`` CACHES
    ``data/<data_name>_type_mask.mat``, reloading it when present. A stale mask
    from a different slice or seed would be silently reused, so every fit runs
    in its own work directory with its own ``data/`` and a unique data_name.

SCALE -- READ THIS BEFORE POINTING IT AT A NEW DATASET
------------------------------------------------------
Peak memory is O((n_spots + n_cells)^2), because ``loss_fn`` builds
``full_mask`` of shape (n1+n2, n1+n2) every epoch and ``ContrastiveLoss`` then
materialises four more matrices of that size, on top of the (n1, n2)
``Coefficient`` with its Adam state and the (n2, n2) cell-type mask:

    peak ~= 16*n1*n2  (Coefficient + grad + 2 Adam states)
          +  4*n1*n2  (cross_mask)
          +  4*n2*n2  (type_mask)
          + 20*(n1+n2)^2   (full_mask + sim + sim_exp + pos/neg masks)

Measured against that model: COME's own largest published run (VISp, 15,413
cells) is ~6.6 GB. Our mmc_luna at full reference (158,379 train cells + a
5,235-cell test slice) would need ~552 GB, and cns_luna ~1,116 GB. The
envelope is n_spots + n_cells <~ 20k at 8 GB, <~ 80k at 128 GB.

Consequences, both deliberate:
  * mmc_luna IS runnable with the FULL test slice (<=5,235 cells) against a
    SUBSAMPLED reference (--max_ref_cells, e.g. 20,000 -> ~15 GB). The evaluated
    cell population is then identical to LUNA/G2T/CeLEry/CellContrast, and the
    only deviation is the reference size -- the same deviation CellContrast's
    --max_ref_cells makes, recorded in the manifest.
  * cns_luna is effectively NOT runnable: its 63,343-cell test slice needs
    ~96 GB with an EMPTY reference, ~136 GB with even a 10,000-spot reference,
    and ~1,116 GB at the reference the other methods use. Because the query term
    dominates, lowering --max_ref_cells cannot rescue it. We do not subsample the
    query (that would change the evaluated population and make the number
    incomparable with the other methods) and we do not chunk it (each chunk would
    get its own mapping matrix and its own contrastive negatives -- a real method
    change, unlike CellContrast's provably bit-identical chunking). Report the
    dataset as outside COME's envelope, with this arithmetic.

Usage (cortex):
    python run_come.py \
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \
        --come_repo /nfs/team361/sb75/COME \
        --out_root /nfs/team361/sb75/scgg-reproducibility/artifacts \
        --dataset mmc_luna --max_ref_cells 20000 --seed 0

Always start with --smoke_test before a real run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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

__version__ = "2026-08-14-run-come-v1"

LOG = logging.getLogger("come")

# Upstream's five per-platform presets (configure.py). Only k / dims / epochs
# differ; dims[0] is overwritten with the real gene count, exactly as their
# main() does at train_eval.py:91.
COME_CONFIGS = ("dro", "smFISH", "MERFISH", "STARmap", "PDAC")

# Hard feasibility ceiling. Nothing in our cluster has this much, so a fit whose
# estimated peak exceeds it cannot succeed; refusing up front beats discovering
# it after hours of queue time. Overridable with --force_scale.
HARD_CEILING_GB = 256.0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="silver dir holding *_train.h5ad / *_test.h5ad")
    p.add_argument("--come_repo", required=True,
                   help="path to the cloned COME repo (contains model.py, "
                        "configure.py, train_eval.py, utils.py)")
    p.add_argument("--out_root",
                   default="/nfs/team361/sb75/scgg-reproducibility/artifacts")
    p.add_argument("--dataset", default=None,
                   help="dataset name for the artifact path; default = data_dir basename")
    p.add_argument("--run_timestamp", default=None, help="pin the artifact timestamp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--come_config", default="MERFISH", choices=COME_CONFIGS,
                   help="which of the authors' per-platform presets to use "
                        "(configure.py). MERFISH is theirs for mouse visual "
                        "cortex imaging ST and is the right preset for mmc_luna.")
    p.add_argument("--sttype", default="image", choices=("image", "sequence"),
                   help="upstream's preprocessing switch (utils.normalize_type): "
                        "image = per-gene MinMaxScaler (imaging ST); sequence = "
                        "normalize_total(1e4)+log1p (spot ST). Default image.")
    p.add_argument("--max_ref_cells", type=int, default=20000,
                   help="cap the ST reference by per-slice stratified subsampling. "
                        "REQUIRED in practice: peak memory is O((n_ref+n_query)^2). "
                        "Recorded in the manifest as a deviation.")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the preset's training epochs. A DEVIATION; recorded.")
    p.add_argument("--pretrain_epochs", type=int, default=None,
                   help="override the preset's pretrain epochs. A DEVIATION; recorded.")
    p.add_argument("--max_mem_gb", type=float, default=None,
                   help="refuse to start a fit whose estimated peak exceeds this (GB).")
    p.add_argument("--force_scale", action="store_true",
                   help=f"bypass the hard {HARD_CEILING_GB} GB feasibility refusal. "
                        "COME is O((n_ref+n_query)^2); the refusal exists so a run "
                        "cannot burn a queue slot for hours before dying.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu"),
                   help="'auto' uses CUDA when available. Upstream builds its "
                        "contrastive full_mask on the CPU while the embeddings "
                        "live on the model's device, which makes its own GPU path "
                        "raise a device mismatch; we patch that at import "
                        "(device placement only -- see "
                        "_patch_contrastive_device), so 'auto' works. 'cpu' "
                        "remains available as a fallback and is bit-comparable, "
                        "just far slower.")
    p.add_argument("--on_empty_cells", default="fail",
                   choices=("fail", "keep", "drop"),
                   help="what to do if a query slice contains cells with zero "
                        "detected genes (upstream's load_data would DROP them, "
                        "breaking row alignment with the truth coordinates). "
                        "fail (default) refuses; keep retains them; drop removes "
                        "them AND records that the evaluated population shrank.")
    p.add_argument("--exclude_test_files", default="",
                   help="comma-separated *_test.h5ad basenames to skip")
    p.add_argument("--smoke_test", action="store_true",
                   help="tiny end-to-end run (2 train slices, 1 test slice, tiny "
                        "reference, few epochs) to prove the install works. "
                        "Never report these numbers.")
    p.add_argument("--dry_run", action="store_true",
                   help="print the plan and the memory estimate; fit nothing")
    return p


# ---------------------------------------------------------------------------
# Memory model (see the module docstring for the derivation)
# ---------------------------------------------------------------------------
def estimate_peak_bytes(n_ref: int, n_query: int) -> int:
    n1, n2 = int(n_ref), int(n_query)
    return (16 * n1 * n2          # Coefficient + grad + 2 Adam states
            + 4 * n1 * n2         # cross_mask
            + 4 * n2 * n2         # type_mask
            + 20 * (n1 + n2) ** 2)  # full_mask + sim + sim_exp + pos/neg


def max_total_for_budget(budget_gb: float) -> int:
    """Largest n_ref + n_query that fits ``budget_gb`` under the dominant term."""
    return int((budget_gb * 1e9 / 20.0) ** 0.5)


# ---------------------------------------------------------------------------
def _import_come(repo: Path):
    """Import the authors' modules unmodified.

    Their modules use top-level imports (``from utils import cal_ssim``), so the
    repo root must be on sys.path. ``train_eval`` also runs argparse AT MODULE
    LEVEL and stores the result in a global its train loops read, so sys.argv is
    stubbed to a bare vector first -- otherwise it would try to parse OUR flags
    and exit. The stub leaves every upstream default in place, including
    ``patience=10``, which is what their loops use.
    """
    repo = Path(repo).resolve()
    for f in ("model.py", "configure.py", "utils.py", "train_eval.py"):
        if not (repo / f).exists():
            raise FileNotFoundError(f"{f} not found under {repo}")
    sys.path.insert(0, str(repo))
    saved_argv = list(sys.argv)
    try:
        sys.argv = ["train_eval.py"]
        import configure          # noqa: E402
        import model as come_model  # noqa: E402
        import utils as come_utils  # noqa: E402
        import train_eval          # noqa: E402
    finally:
        sys.argv = saved_argv
    if _patch_contrastive_device(come_model):
        LOG.info("patched ContrastiveLoss.forward to move the mask to the "
                 "embeddings' device (upstream builds full_mask on the CPU; "
                 "device placement only, objective unchanged)")
    return configure, come_model, come_utils, train_eval


def _patch_contrastive_device(come_model) -> bool:
    """Let ContrastiveLoss accept the mask upstream hands it on the WRONG device.

    Upstream ``Model.loss_fn`` (model.py:119) builds

        full_mask = torch.zeros(self._n1 + self._n2, self._n1 + self._n2)

    with no ``device=``, so it lands on the CPU, while ``z1``/``z2`` and
    ``cross_mask`` (``zeros_like(Coefficient)``) live on the model's device. The
    slice-assignments that follow SURVIVE, because ``Tensor.__setitem__`` copies
    cross-device -- which is why the failure is deferred -- but
    ``ContrastiveLoss.forward`` then does
    ``torch.mul(sim_exp, positive_mask)`` (model.py:170) across devices and
    raises

        RuntimeError: Expected all tensors to be on the same device,
                      but found at least two devices, cuda:0 and cpu!

    on EVERY CUDA run. Upstream's own experiments were evidently CPU-only here,
    or on a torch old enough to be permissive.

    Moving a float32 mask between devices is bit-preserving, so this is a
    DEVICE-PLACEMENT fix and provably not a change to the objective: the mask
    values, the similarity matrix and the reduction are all untouched. We patch
    the imported class rather than editing the clone so the checkout stays
    byte-identical to the pinned commit (setup_come_env.sh verifies that).

    Two details that make this safe:
      * upstream REBUILDS ``full_mask`` at the top of every ``loss_fn`` call and
        never reads it again afterwards, and
      * ``mask_pos_and_neg`` mutates the mask in place (``fill_diagonal_``).
        After this patch that in-place write lands on our device copy instead of
        the caller's tensor -- immaterial given the point above, and strictly
        safer than mutating a tensor the caller still holds.

    Returns True if the patch was applied (False if already patched).
    """
    orig = come_model.ContrastiveLoss.forward
    if getattr(orig, "_scgg_device_patched", False):
        return False

    def forward(self, h1, h2, mask=None):
        if mask is not None and getattr(mask, "device", None) is not None \
                and mask.device != h1.device:
            mask = mask.to(h1.device)
        return orig(self, h1, h2, mask=mask)

    forward._scgg_device_patched = True
    come_model.ContrastiveLoss.forward = forward
    return True


def _make_adata(ad_mod, X: np.ndarray, obs: Dict[str, np.ndarray], var_names):
    import pandas as pd
    a = ad_mod.AnnData(X=np.asarray(X, dtype=np.float32),
                       obs=pd.DataFrame(obs).reset_index(drop=True))
    a.var_names = [str(v) for v in var_names]
    a.obs_names = [str(i) for i in range(a.n_obs)]
    return a


def _dense(X) -> np.ndarray:
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)


def _coords_of(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        s = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if s.ndim == 2 and s.shape[1] >= 2:
            return s[:, :2]
    if "coord_X" in adata.obs and "coord_Y" in adata.obs:
        return np.column_stack([adata.obs["coord_X"].to_numpy(dtype=np.float64),
                                adata.obs["coord_Y"].to_numpy(dtype=np.float64)])
    raise ValueError("no coordinates in obsm['spatial'] or obs['coord_X'/'coord_Y']")


# ---------------------------------------------------------------------------
def build_reference(ad_mod, train_files: List[Path], args,
                    rng: np.random.Generator):
    """Concatenate the training slices into COME's ST ("spot") side.

    Coordinates are min-max normalised PER SLICE into a shared box purely so the
    read-out is in one frame; COME's training never touches them (see the module
    docstring), so the frame choice cannot affect the fit.
    """
    blocks, coords, classes, var_ref = [], [], [], None
    for f in train_files:
        a = ad_mod.read_h5ad(f)
        feats = _dense(a.X)
        vn = np.asarray(a.var_names, dtype=object)
        if var_ref is None:
            var_ref = vn
        elif not np.array_equal(var_ref, vn):
            raise ValueError(f"{f.name}: gene panel differs from the first slice")
        cn = SliceCoordScaler(isotropic=True).fit(_coords_of(a)).transform(_coords_of(a))
        cls = (a.obs[COL_CLASS].astype(str).to_numpy()
               if COL_CLASS in a.obs else np.full(feats.shape[0], "NA"))
        blocks.append(feats)
        coords.append(cn)
        classes.append(cls)
        LOG.info("  train %-22s n=%6d", section_label_from_filename(f), feats.shape[0])

    X = np.vstack(blocks)
    XY = np.vstack(coords)
    CLS = np.concatenate(classes)
    if args.max_ref_cells and X.shape[0] > args.max_ref_cells:
        keep = rng.choice(X.shape[0], args.max_ref_cells, replace=False)
        keep.sort()
        X, XY, CLS = X[keep], XY[keep], CLS[keep]
        LOG.info("  reference subsampled to %d cells (--max_ref_cells)", X.shape[0])
    LOG.info("reference: %d spots x %d genes", X.shape[0], X.shape[1])
    return X, XY, CLS, [str(v) for v in var_ref]


def load_query(ad_mod, test_file: Path, args, ref_var: List[str]):
    """One test slice -> (features, truth coords, labels, obs_names, var_names)."""
    a = ad_mod.read_h5ad(test_file)
    feats = _dense(a.X)
    vn = [str(v) for v in a.var_names]
    if vn != ref_var:
        missing = [g for g in ref_var if g not in set(vn)]
        if missing:
            raise ValueError(
                f"{test_file.name}: {len(missing)} reference gene(s) absent from "
                f"the test panel (e.g. {missing[:5]}). COME intersects gene sets, "
                f"so this would silently change the feature space.")
    coords_true = _coords_of(a)
    if COL_CLASS not in a.obs:
        raise ValueError(
            f"{test_file.name}: obs[{COL_CLASS!r}] missing. It is required both "
            f"for the artifact schema the scorer reads and for COME's cell-type "
            f"contrastive mask (utils.cell_type_2_martix).")
    cls = a.obs[COL_CLASS].astype(str).to_numpy()
    obs_names = np.asarray(a.obs_names, dtype=object)

    # Upstream's load_data would drop these; we refuse to do it silently.
    n_empty = int((feats.sum(axis=1) <= 0).sum())
    if n_empty:
        if args.on_empty_cells == "fail":
            raise ValueError(
                f"{test_file.name}: {n_empty} cell(s) have zero detected genes. "
                f"Upstream's load_data calls sc.pp.filter_cells(min_genes=1) and "
                f"would DROP them, breaking the row correspondence with the truth "
                f"coordinates. Choose --on_empty_cells keep (retain them, "
                f"population matches the other methods) or drop (remove them and "
                f"record the shrunken population).")
        if args.on_empty_cells == "drop":
            keep = feats.sum(axis=1) > 0
            feats, coords_true, cls, obs_names = (feats[keep], coords_true[keep],
                                                  cls[keep], obs_names[keep])
            LOG.warning("  dropped %d empty cell(s); evaluated population now %d",
                        n_empty, feats.shape[0])
        else:
            LOG.warning("  %d empty cell(s) retained (--on_empty_cells keep)",
                        n_empty)
    return feats, coords_true, cls, obs_names, n_empty


# ---------------------------------------------------------------------------
def fit_come(come_mods, ref_X, ref_cls, qry_X, qry_cls, args, work: Path,
             tag: str) -> np.ndarray:
    """Fit COME on (reference spots, query cells); return Coefficient (n1, n2).

    Every modelling call here is the authors' own. The two optimizers are
    constructed exactly as their ``main()`` constructs them (train_eval.py:103).
    """
    import torch
    configure, come_model, come_utils, train_eval = come_mods

    # Their loops and their device global; allow forcing CPU because of the
    # CPU/GPU mask mismatch documented on --device.
    if args.device == "cpu":
        train_eval.device = torch.device("cpu")
    device = train_eval.device
    LOG.info("  device: %s", device)

    # A private cwd per fit: cell_type_2_martix WRITES data/<name>_type_mask.mat
    # and train_eval CACHES it, so a shared cwd would silently reuse another
    # slice's mask.
    fit_dir = work / f"fit_{tag}"
    (fit_dir / "data").mkdir(parents=True, exist_ok=True)
    (fit_dir / "pretrain").mkdir(parents=True, exist_ok=True)
    (fit_dir / "result").mkdir(parents=True, exist_ok=True)

    ad_mod = _import_anndata()
    n_genes = ref_X.shape[1]
    spot = _make_adata(ad_mod, ref_X, {"cell_type": ref_cls},
                       [f"g{i}" for i in range(n_genes)])
    rna = _make_adata(ad_mod, qry_X, {"cell_type": qry_cls},
                      [f"g{i}" for i in range(n_genes)])

    # THEIR preprocessing, THEIR gene intersection/sort.
    spot_n, rna_n = come_utils.normalize_type(spot, rna, type=args.sttype)
    x1 = _dense(spot_n.X)
    x2 = _dense(rna_n.X)
    LOG.info("  after normalize_type(%s): spots %s, cells %s",
             args.sttype, x1.shape, x2.shape)

    config = configure.get_default_config(args.come_config)
    config["num_sample1"] = x1.shape[0]
    config["num_sample2"] = x2.shape[0]
    config["dims"] = list(config["dims"])
    config["dims"][0] = x1.shape[1]          # as their main() does
    if args.smoke_test:
        config["pretrain_epochs"], config["epochs"] = 2, 2
    if args.pretrain_epochs:
        config["pretrain_epochs"] = int(args.pretrain_epochs)
    if args.epochs:
        config["epochs"] = int(args.epochs)

    # THEIR cell-type contrastive mask, written inside our private cwd.
    cwd0 = os.getcwd()
    os.chdir(fit_dir)
    try:
        type_mask = come_utils.cell_type_2_martix(rna_n, data_name=tag)
        model = come_model.Model(config)
        model.to(device)
        opt_pre = torch.optim.Adam(model.ae.parameters(), lr=config["pre_lr"])
        opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
        pretrain_path = f"pretrain/{tag}.pkl"
        t0 = time.time()
        train_eval.pretrain_ae(model.ae, opt_pre,
                               np.concatenate((x1, x2), axis=0), config,
                               pretrain_path)
        model.ae.load_state_dict(torch.load(pretrain_path))
        model.train()
        train_eval.train(model, opt, x1, x2, type_mask, config,
                         f"result/{tag}_model.pkl")
        model.eval()
        LOG.info("  fit done (%.1f s)", time.time() - t0)
        C = model.map.Coefficient.detach().cpu().numpy()
    finally:
        os.chdir(cwd0)
        shutil.rmtree(fit_dir, ignore_errors=True)
    return C


def coords_from_coefficient(C: np.ndarray, ref_xy: np.ndarray,
                            allow_degenerate: bool = False):
    """Assign each cell the coordinates of its highest-coefficient spot.

    This is upstream's own assignment rule: ``Model.cross_mask`` selects
    ``Coefficient.max(dim=0)`` -- per cell (column), the argmax over spots
    (rows). Returns (coords, n_distinct).

    ``allow_degenerate`` downgrades the total-collapse check from a raise to a
    warning. Set it ONLY for smoke tests: upstream initialises
    ``MapNet.Coefficient`` to a UNIFORM constant (``torch.ones((n, m)) / (n*m)``,
    model.py:48), so after the handful of epochs a smoke test runs, every column
    is still near-constant and the argmax degenerates to one spot for all cells.
    That is the expected outcome of a 2-epoch fit, not a defect, and blocking on
    it prevents the smoke test from exercising the artifact-writing path it
    exists to test. For a real run the collapse check MUST stay fatal -- scoring
    a collapsed mapping would report a degenerate model as a method result.
    """
    if C.ndim != 2:
        raise ValueError(f"Coefficient must be 2-D; got {C.shape}")
    n_spots, n_cells = C.shape
    if n_spots != ref_xy.shape[0]:
        raise ValueError(f"Coefficient has {n_spots} spot rows but the reference "
                         f"has {ref_xy.shape[0]} spots")
    if not np.isfinite(C).all():
        raise ValueError("Coefficient contains NaN/Inf — the fit diverged; do "
                         "not score this run")
    idx = np.argmax(C, axis=0)               # per cell, the best spot
    xy = ref_xy[idx]
    n_distinct = len({(round(float(a), 9), round(float(b), 9)) for a, b in xy})
    if n_distinct == 1:
        msg = (f"all {n_cells} cells mapped to a single spot — a degenerate "
               f"fit. Coefficient is initialised UNIFORM (model.py:48), so this "
               f"is the expected result of a very short fit and a real signal "
               f"of non-convergence in a full one.")
        if not allow_degenerate:
            raise ValueError(
                msg + " Scoring it would report a collapsed model as a method "
                "result. (If this IS a smoke test, the wrapper downgrades this "
                "to a warning automatically.)")
        LOG.warning("  %s Continuing because this is a smoke test — its numbers "
                    "must never be reported.", msg)
    return xy, n_distinct


def _import_anndata():
    try:
        import anndata as ad
        import scanpy as sc  # noqa: F401  (upstream imports it)
        return ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("needs anndata + scanpy (present in the come env)") from exc


def _torch_version() -> Optional[str]:
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return None


def _git_commit(repo: Path) -> Optional[str]:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)

    repo = Path(args.come_repo).resolve()
    data_dir = Path(args.data_dir).resolve()
    dataset = args.dataset or data_dir.name
    ts = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / dataset / "come_inference" / ts
    results = out_dir / "test_results"
    work = out_dir / "work"
    for d in (results, work):
        d.mkdir(parents=True, exist_ok=True)
    LOG.info("COME wrapper %s\n  dataset=%s seed=%d\n  out=%s",
             __version__, dataset, args.seed, out_dir)

    train_files = discover_split_files(data_dir, "train")
    test_files = discover_split_files(data_dir, "test")
    excl = {s.strip() for s in args.exclude_test_files.split(",") if s.strip()}
    if excl:
        test_files = [p for p in test_files if p.name not in excl]
    if not train_files or not test_files:
        raise RuntimeError(f"need both splits in {data_dir}; found "
                           f"{len(train_files)} train / {len(test_files)} test")
    if args.smoke_test:
        train_files, test_files = train_files[:2], test_files[:1]
        args.max_ref_cells = min(args.max_ref_cells or 500, 500)
        LOG.warning("SMOKE TEST: 2 train / 1 test slice, ref<=500, 2 epochs — "
                    "do not report")
    LOG.info("splits: %d train / %d test", len(train_files), len(test_files))

    ad_mod = _import_anndata()
    ref_X, ref_xy, ref_cls, ref_var = build_reference(ad_mod, train_files, args, rng)

    manifest: Dict[str, object] = {
        "wrapper_version": __version__,
        "method": "come",
        "upstream": "https://github.com/cindyway/COME",
        "upstream_commit": _git_commit(repo),
        "upstream_paper": "Wei et al., Bioinformatics 41(3):btaf083, 2025",
        "dataset": dataset, "seed": args.seed, "timestamp": ts,
        "n_train_slices": len(train_files), "n_test_slices": len(test_files),
        "come_config": args.come_config,
        "sttype": args.sttype,
        "expression_mode": "silver_raw (COME normalises internally via "
                           "utils.normalize_type)",
        "reference_n_spots": int(ref_X.shape[0]),
        "n_features": int(ref_X.shape[1]),
        "max_ref_cells": args.max_ref_cells,
        "transductive": True,
        "fit_per_test_slice": True,
        "readout": "argmax over spots of MapNet.Coefficient (upstream's own "
                   "cross_mask rule); coordinates copied verbatim from that spot",
        "coordinate_frame": "per-slice isotropic min-max [-0.5,0.5] for the "
                            "read-out only; COME's training never uses coords",
        "artifact_frame": "original_microns",
        "on_empty_cells": args.on_empty_cells,
        "device": args.device,
        # The upstream checkout is byte-identical to the pinned commit; this is
        # applied to the IMPORTED class at runtime and is a device placement
        # only, so it cannot change the objective or the predictions.
        "upstream_patches": [
            "ContrastiveLoss.forward: move the caller's mask to the embeddings' "
            "device (upstream builds full_mask via torch.zeros with no device=, "
            "so its own CUDA path raises in torch.mul at model.py:170)"
        ],
        "torch_version": _torch_version(),
        "smoke_test": args.smoke_test,
        "status": "incomplete",
        "per_slice": [],
    }
    if not args.dry_run:
        write_run_manifest(out_dir, manifest)

    come_mods = None
    per_slice: List[Dict[str, object]] = []
    for i, tf in enumerate(test_files, 1):
        label = section_label_from_filename(tf)
        LOG.info("[%d/%d] %s", i, len(test_files), label)
        feats, coords_true, cls, obs_names, n_empty = load_query(
            ad_mod, tf, args, ref_var)
        n_cells = feats.shape[0]
        peak = estimate_peak_bytes(ref_X.shape[0], n_cells)
        LOG.info("  n_query=%d  n_ref=%d  est. peak %.1f GB",
                 n_cells, ref_X.shape[0], peak / 1e9)
        if peak > HARD_CEILING_GB * 1e9 and not args.force_scale:
            q_only = 20 * n_cells * n_cells
            raise SystemExit(
                f"[come] {label}: this slice is outside COME's feasible envelope.\n"
                f"  estimated peak            {peak/1e9:>10,.0f} GB\n"
                f"  of which the query alone  {q_only/1e9:>10,.0f} GB "
                f"(20 * n_query^2, before a single reference cell)\n"
                f"  hard ceiling              {HARD_CEILING_GB:>10,.0f} GB\n"
                f"COME learns a dense (n_ref x n_query) mapping matrix as a free\n"
                f"parameter and rebuilds several dense (n_ref+n_query)^2 masks every\n"
                f"epoch, so peak memory is O((n_ref+n_query)^2). At this ceiling\n"
                f"n_ref+n_query must be <= ~{max_total_for_budget(HARD_CEILING_GB):,}, "
                f"but this query alone is {n_cells:,}.\n"
                f"Lowering --max_ref_cells cannot fix it: the query term dominates.\n"
                f"For reference, COME's own largest published run (VISp, 15,413\n"
                f"cells) is ~6.6 GB. Subsampling the query would change the\n"
                f"evaluated population and make the number incomparable with the\n"
                f"other methods; chunking it would give each chunk its own mapping\n"
                f"matrix and its own contrastive negatives, i.e. a different method.\n"
                f"Report this dataset as outside COME's envelope, with this\n"
                f"arithmetic, rather than as a poor COME score.")
        if args.max_mem_gb and peak > args.max_mem_gb * 1e9:
            raise SystemExit(
                f"[come] {label}: estimated peak {peak/1e9:.1f} GB exceeds "
                f"--max_mem_gb {args.max_mem_gb:.1f}. COME is O((n_ref+n_query)^2); "
                f"at this budget n_ref+n_query must be <= "
                f"~{max_total_for_budget(args.max_mem_gb):,} and the query alone is "
                f"{n_cells:,}. Lower --max_ref_cells, or accept that this dataset "
                f"is outside COME's feasible envelope (see the module docstring).")
        if args.dry_run:
            per_slice.append({"section": label, "n_cells": n_cells,
                              "est_peak_gb": round(peak / 1e9, 2)})
            continue

        if come_mods is None:
            come_mods = _import_come(repo)
        C = fit_come(come_mods, ref_X, ref_cls, feats, cls, args, work,
                     tag=f"{dataset}_{label}_s{args.seed}")
        pred_norm, n_distinct = coords_from_coefficient(
            C, ref_xy, allow_degenerate=args.smoke_test)
        # Read-out is in the shared normalised frame -> this slice's microns.
        pred_orig = (SliceCoordScaler(isotropic=True).fit(coords_true)
                     .inverse_transform(pred_norm))
        write_slice_artifacts(results / label, pred_orig, coords_true, cls,
                              index=obs_names)
        LOG.info("  n=%d  distinct predicted positions=%d (%.1f%%)",
                 n_cells, n_distinct, 100.0 * n_distinct / n_cells)
        per_slice.append({"section": label, "n_cells": n_cells,
                          "n_empty_cells": n_empty,
                          "est_peak_gb": round(peak / 1e9, 2),
                          "distinct_predicted_positions": n_distinct})

    if args.dry_run:
        LOG.info("dry run complete; per-slice plan: %s",
                 json.dumps(per_slice, indent=2))
        return 0

    manifest["per_slice"] = per_slice
    manifest["status"] = ("complete" if len(per_slice) == len(test_files)
                          else f"incomplete ({len(per_slice)}/{len(test_files)} slices)")
    write_run_manifest(out_dir, manifest)
    LOG.info("\nwrote %d slice(s) to %s", len(per_slice), results)
    LOG.info("next: score with compute_extended_metrics.py --methods come")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
