#!/usr/bin/env python3
"""harness_adapter.py — the seam between the published CellContrast / COME code
and our existing evaluation harness.

Everything here is fixed by OUR side of the interface, so it is correct
independently of what the two upstream packages expect internally. It gives:

  * the SAME split discovery the other baselines use  (``*_{train,test}.h5ad``)
  * the SAME expression preprocessing                  (log2(1+x), see below)
  * the SAME per-slice coordinate normalisation        ([-0.5, 0.5])
  * the EXACT artifact schema the metric code reads    (metadata_pred/true.csv)

so that any numbers produced are directly comparable with LUNA, G2T and CeLEry
rather than merely similar-looking.

Data provenance (verified against the repo, not assumed):
  * silver ``adata.X``          — LUNA's CSV values as-is, i.e. non-integer
                                  PER-CELL-NORMALISED counts (not raw integers).
  * ``adata.obsm['spatial']``   — raw micron coordinates (fallback:
                                  ``obs['coord_X'|'coord_Y']``).
  * ``adata.obs['cell_class']`` — string cell-class labels.
  * the model data path applies ``log2(1 + x)``
    (``scgg/src/utils/data/load.py::log2_norm``) — that is the "log2-normalised
    following LUNA" step in the paper.

FAIRNESS NOTE — read before changing ``expression_mode``.
LUNA, G2T and CeLEry all consume log2(1+x) of the silver matrix. If an upstream
package performs its own normalisation and expects counts, feeding it
already-log2'd values would handicap it, and the resulting number would not be a
fair report of that method. Hence ``expression_mode``:
  * "log2"  (default) — parity with our other baselines.
  * "silver_raw"      — hand over the silver matrix untouched and let the
                        upstream package normalise as its authors intended.
Whichever is used MUST be recorded in the run manifest and stated in any
write-up. When in doubt, run both and report the better one for the baseline —
never the worse.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__version__ = "2026-08-13-harness-adapter-v1"

COL_CLASS = "cell_class"
COL_X = "coord_X"
COL_Y = "coord_Y"


# ---------------------------------------------------------------------------
# Split discovery — identical convention to run_{scgg,luna}_train.py
# ---------------------------------------------------------------------------
def discover_split_files(silver_dir: Path, split: str) -> List[Path]:
    """Return sorted ``*_{split}.h5ad`` paths (same rule as the other pipelines)."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    return sorted(Path(silver_dir).glob(f"*_{split}.h5ad"))


def section_label_from_filename(path: Path) -> str:
    """Strip the ``_train`` / ``_test`` suffix from the stem."""
    stem = Path(path).stem
    for suf in ("_train", "_test"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


# ---------------------------------------------------------------------------
# Per-slice coordinate normalisation
# ---------------------------------------------------------------------------
class SliceCoordScaler:
    """Per-slice min-max scaling of 2-D coordinates into ``[lo, hi]``.

    Each slice has its own micron-scale bounding box, so coordinates are only
    comparable across slices after per-slice normalisation. Predictions are
    inverse-transformed back to the slice's own original scale before being
    written out, because Sum RSSD is computed against original-scale truth
    (Spearman and Contact F1 are rank/threshold based and unaffected).

    This mirrors CeLEry's handling, which is the closest analogue among our
    baselines (it too predicts coordinates and must invert the transform).
    """

    def __init__(self, lo: float = -0.5, hi: float = 0.5) -> None:
        if not hi > lo:
            raise ValueError("hi must exceed lo")
        self.lo, self.hi = float(lo), float(hi)
        self.min_: Optional[np.ndarray] = None
        self.span_: Optional[np.ndarray] = None

    def fit(self, coords: np.ndarray) -> "SliceCoordScaler":
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError(f"coords must be (n, 2); got {c.shape}")
        self.min_ = c.min(axis=0)
        span = c.max(axis=0) - self.min_
        # A degenerate axis (all cells share a value) would divide by zero;
        # map it to the midpoint instead of producing inf/nan.
        self.span_ = np.where(span > 0, span, 1.0)
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        self._check()
        c = np.asarray(coords, dtype=np.float64)
        unit = (c - self.min_) / self.span_          # -> [0, 1]
        return unit * (self.hi - self.lo) + self.lo  # -> [lo, hi]

    def inverse_transform(self, coords_scaled: np.ndarray) -> np.ndarray:
        self._check()
        c = np.asarray(coords_scaled, dtype=np.float64)
        unit = (c - self.lo) / (self.hi - self.lo)
        return unit * self.span_ + self.min_

    def _check(self) -> None:
        if self.min_ is None or self.span_ is None:
            raise RuntimeError("SliceCoordScaler.fit must be called first")


# ---------------------------------------------------------------------------
# Slice loading (anndata imported lazily so this module imports without it)
# ---------------------------------------------------------------------------
def load_slice(
    h5ad_path: Path,
    expression_mode: str = "log2",
) -> Dict[str, object]:
    """Load one silver slice into plain arrays.

    Returns a dict with keys: ``expr`` (n, g) float64, ``coords`` (n, 2) float64
    ORIGINAL micron scale, ``cell_class`` (n,) str or None, ``obs_names`` (n,)
    str, ``label`` str, ``var_names`` (g,) str.
    """
    if expression_mode not in ("log2", "silver_raw"):
        raise ValueError("expression_mode must be 'log2' or 'silver_raw'")
    try:
        import anndata as ad
    except Exception as exc:                          # pragma: no cover
        raise RuntimeError(
            "load_slice needs anndata; it is only available in the method envs "
            "on the farm, not in the local checkout"
        ) from exc

    path = Path(h5ad_path)
    adata = ad.read_h5ad(path)

    X = adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    if expression_mode == "log2":
        # matches scgg/src/utils/data/load.py::log2_norm
        X = np.log2(1.0 + X)

    obs = adata.obs
    if "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(f"{path}: obsm['spatial'] has shape {spatial.shape}")
        coords = spatial[:, :2]
    elif COL_X in obs.columns and COL_Y in obs.columns:
        coords = np.column_stack([obs[COL_X].to_numpy(dtype=np.float64),
                                  obs[COL_Y].to_numpy(dtype=np.float64)])
    else:
        raise ValueError(
            f"{path}: no coordinates. Expected obsm['spatial'] or "
            f"obs[{COL_X!r}/{COL_Y!r}]. obs cols: {list(obs.columns)[:10]}; "
            f"obsm keys: {list(adata.obsm.keys())}"
        )

    cell_class = (obs[COL_CLASS].astype(str).to_numpy()
                  if COL_CLASS in obs.columns else None)
    label = (str(obs["cell_section"].iloc[0])
             if "cell_section" in obs.columns and len(obs) else
             section_label_from_filename(path))

    return {
        "expr": X,
        "coords": coords,
        "cell_class": cell_class,
        "obs_names": np.asarray(adata.obs_names, dtype=object),
        "label": label,
        "var_names": np.asarray(adata.var_names, dtype=object),
    }


# ---------------------------------------------------------------------------
# Artifact writing — the exact schema compute_extended_metrics.py reads
# ---------------------------------------------------------------------------
def write_slice_artifacts(
    out_slice_dir: Path,
    coords_pred_original: np.ndarray,
    coords_true_original: np.ndarray,
    cell_class: Optional[np.ndarray],
    index: Optional[np.ndarray] = None,
) -> Tuple[Path, Path]:
    """Write ``metadata_pred.csv`` and ``metadata_true.csv`` for one slice.

    Schema (matching run_celery_inference.py so the metric code needs no
    changes): an index column, then ``coord_X``, ``coord_Y``, ``cell_class``.
    Coordinates MUST already be in the slice's ORIGINAL scale.

    Written with the csv module rather than pandas so this is usable in any env.
    """
    pred = np.asarray(coords_pred_original, dtype=np.float64)
    true = np.asarray(coords_true_original, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} != true {true.shape}")
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2); got {pred.shape}")
    n = pred.shape[0]
    if not np.isfinite(pred).all():
        raise ValueError("predicted coordinates contain NaN/Inf")
    if cell_class is not None and len(cell_class) != n:
        raise ValueError(f"cell_class has {len(cell_class)} entries for n={n}")
    idx = (np.arange(n) if index is None else np.asarray(index))
    if len(idx) != n:
        raise ValueError("index length mismatch")

    out_slice_dir = Path(out_slice_dir)
    out_slice_dir.mkdir(parents=True, exist_ok=True)
    header = ["", COL_X, COL_Y] + ([COL_CLASS] if cell_class is not None else [])

    def _dump(path: Path, coords: np.ndarray) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for i in range(n):
                row = [idx[i], repr(float(coords[i, 0])), repr(float(coords[i, 1]))]
                if cell_class is not None:
                    row.append(str(cell_class[i]))
                w.writerow(row)

    p_pred = out_slice_dir / "metadata_pred.csv"
    p_true = out_slice_dir / "metadata_true.csv"
    _dump(p_pred, pred)
    _dump(p_true, true)
    return p_pred, p_true


def write_run_manifest(out_dir: Path, manifest: Dict[str, object]) -> Path:
    """Record exactly how a run was produced.

    Non-negotiable for a baseline we did not author: which upstream commit, which
    ``expression_mode``, which seed, and any deviation from the authors' defaults
    must be recoverable months later.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "run_manifest.json"
    payload = dict(manifest)
    payload.setdefault("adapter_version", __version__)
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return p
