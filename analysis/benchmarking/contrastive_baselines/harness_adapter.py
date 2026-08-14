#!/usr/bin/env python3
"""harness_adapter.py — the seam between the published CellContrast / COME code
and our existing evaluation harness.

Everything here is fixed by OUR side of the interface, so it is correct
independently of what the two upstream packages expect internally. It gives:

  * the SAME split discovery the other baselines use  (``*_{train,test}.h5ad``)
  * per-slice coordinate normalisation into a shared frame (see SliceCoordScaler)
  * the EXACT artifact schema the metric code reads    (metadata_pred/true.csv)

Data provenance (verified against the repo, not assumed):
  * ``adata.obsm['spatial']``   — raw micron coordinates (fallback:
                                  ``obs['coord_X'|'coord_Y']``).
  * ``adata.obs['cell_class']`` — string cell-class labels.
  * silver ``adata.X`` differs BY DATASET:
      - ``mmc_luna``  : LUNA's published CSV gene block copied verbatim
                        (``build_h5ad_from_luna_csv.py:211``). Non-integer,
                        per-cell-scaled, count MAGNITUDE (max ~250) — i.e.
                        linear space, NOT log space and NOT raw integers.
      - ``cns_luna``  : the shared 600-d cross-platform latent, sitting in
                        ``.X`` itself (both-sign). It is NOT in any ``obsm``
                        key; the only ``obsm`` key on those h5ads is
                        ``spatial``. Use ``--expression_mode silver_raw`` and
                        do NOT pass ``--use_obsm``.
      - ``dlpfc_visium`` / ``breast_janesick`` / ``cns_luna_raw`` : RAW integer
                        counts. ``silver_raw`` is the correct mode there.

EXPRESSION MODE — the reasoning, because the previous version of this file got
it backwards and the error propagated into the manuscript.
``scgg/src/utils/data/load.py::log2_norm`` exists but has ZERO call sites in
either repo. The actual transform each method applies to silver ``.X`` is:

  | method        | transform on mmc_luna .X                                  |
  |---------------|-----------------------------------------------------------|
  | LUNA          | IDENTITY (``--log2_normalize`` defaults False)             |
  | G2T / scgg    | IDENTITY (``prep.normalize`` defaults "none")             |
  | CeLEry        | identity, then per-slice per-gene z-score (cel.get_zscore) |
  | novosparc     | IDENTITY                                                  |
  | CellContrast  | ``log2(1+x)``  (upstream itself normalises NOTHING)        |

So ``log2`` is NOT "parity with the other baselines" — none of them log. It is
chosen for FIDELITY TO CELLCONTRAST'S PAPER, whose methods state the input was
"log-normalized gene expressions, calculated by scran"; ``scran::logNormCounts``
is size-factor normalisation followed by log base 2, and silver mmc_luna ``.X``
is already per-cell-scaled, so ``log2(1+x)`` composes to that same functional
form. It is therefore the first and only log applied — NOT double-normalisation.

  * "log2"       — the paper-faithful default for count-magnitude ``.X``.
  * "silver_raw" — hand the matrix over untouched. REQUIRED for the CNS latent
                   (both-sign) and correct for raw-count silver dirs.

Report the log2 run as primary and any silver_raw run as a pre-registered
ablation. Do NOT pick whichever scores higher after the fact — choosing
preprocessing on the outcome is cherry-picking, even when it favours a baseline.
Whichever is used MUST be recorded in the run manifest and stated in any write-up.
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
    written out, matching what CeLEry and novosparc write.

    NOTE on frames, because an earlier version of this docstring was wrong:
    LUNA/G2T write BOTH metadata CSVs in the per-slice [-0.5, 0.5] frame
    (``data_module.py:51`` normalises truth at load; ``test.py:274``
    re-normalises the prediction), whereas CeLEry, novosparc and CellContrast
    write original microns. ``compute_kabsch_rssd`` fits a ROTATION ONLY
    (``luna_metrics.py:662`` ``R.align_vectors``), so it is homogeneous of
    degree 1 in coordinate scale and Sum RSSD is NOT comparable across those two
    groups. That is a scorer-side issue, resolved in
    ``compute_extended_metrics.py`` (``--rssd_frame``), not here.

    ``isotropic`` selects HOW the box is normalised. For CellContrast this is
    not cosmetic: upstream builds its k-nearest-neighbour positive graph with a
    KDTree over exactly these coordinates (``loadData.checkNeighbors``), so the
    metric we hand it decides which cells become positives.

      * ``isotropic=False`` (default; matches LUNA/G2T ``position_normalize``,
        which is per-axis min-max grouped by ``cell_section``): each axis is
        divided by its OWN span, so a non-square slice is squashed
        anisotropically and Euclidean neighbour ranking CHANGES. Measured on
        the MMC train split (median aspect 1.25, max 1.59) this alters ~7% of
        the k=80 positive set at the median slice and ~14% at the worst.
      * ``isotropic=True``: both axes are divided by the LARGER span and the
        box is centred, so aspect is preserved and neighbour ranking is
        IDENTICAL to raw microns. Prefer this whenever the consumer depends on
        the metric and not merely on a shared frame.

    Only a similarity transform preserves pairwise-distance ranks, so under
    ``isotropic=False`` the per-cell Spearman and the Contact F1 percentile
    threshold are affected too -- not Sum RSSD alone. (An earlier version of
    this docstring claimed otherwise.)

    This mirrors CeLEry's handling, which is the closest analogue among our
    baselines (it too predicts coordinates and must invert the transform).
    """

    def __init__(self, lo: float = -0.5, hi: float = 0.5,
                 isotropic: bool = False) -> None:
        if not hi > lo:
            raise ValueError("hi must exceed lo")
        self.lo, self.hi = float(lo), float(hi)
        self.isotropic = bool(isotropic)
        self.min_: Optional[np.ndarray] = None
        self.span_: Optional[np.ndarray] = None

    def fit(self, coords: np.ndarray) -> "SliceCoordScaler":
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError(f"coords must be (n, 2); got {c.shape}")
        self.min_ = c.min(axis=0)
        span = c.max(axis=0) - self.min_
        if self.isotropic:
            # One common scale for both axes. Re-seat ``min_`` on the corner of
            # the enlarged square so the narrow axis ends up CENTRED in
            # [lo, hi] rather than pinned to the low edge; inverse_transform
            # then inverts exactly, since it only ever uses min_/span_.
            s = float(span.max())
            if s <= 0.0:
                s = 1.0
            self.min_ = (c.max(axis=0) + self.min_) / 2.0 - s / 2.0
            span = np.full(2, s, dtype=np.float64)
        # A degenerate axis (all cells share a value) would divide by zero.
        # Substituting span 1.0 sends that axis to ``lo`` (not the midpoint, as
        # an earlier comment claimed) — a constant offset, which is harmless for
        # every metric we compute but is worth stating accurately.
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
