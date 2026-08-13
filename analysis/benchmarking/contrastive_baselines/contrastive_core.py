#!/usr/bin/env python3
"""contrastive_core.py — pure-NumPy reference implementation of the maths behind
the CellContrast / COME family of contrastive spatial-reconstruction methods.

WHY THIS FILE EXISTS
--------------------
The deployed baselines are written in PyTorch and can only be run on the farm.
This module re-implements the three mathematical cores in plain NumPy so that
they can be unit-tested anywhere, and so the Torch implementation can be
differentially tested against a second, independent implementation
(``test_contrastive_core.py`` locally; ``--selftest`` on the farm).

The three cores:

1. ``spatial_knn_positives``  — which pairs count as POSITIVES. Both methods
   define positives by *physical* proximity in the reference (spatial) data:
   cell i's positives are its k nearest neighbours in (x, y) within the SAME
   slice. Cross-slice pairs are never positives (different coordinate frames).

2. ``info_nce``               — the contrastive objective on L2-normalised
   embeddings with a temperature. Standard InfoNCE / NT-Xent form.

3. ``knn_coord_readout``      — how an embedding becomes a COORDINATE. A query
   cell is embedded, matched to its k nearest REFERENCE cells in embedding
   space, and assigned a similarity-weighted (softmax/temperature) average of
   those reference cells' known coordinates. This is a convex combination, so
   predictions necessarily lie inside the convex hull of the retrieved
   reference coordinates — a property the tests check explicitly.

All functions are deliberately written for clarity over speed; they are
reference semantics, not the production path.
"""
from __future__ import annotations

import numpy as np

__version__ = "2026-08-12-contrastive-core-v1"


# ---------------------------------------------------------------------------
# 1. Positives: k nearest spatial neighbours, within a slice
# ---------------------------------------------------------------------------
def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n, m) matrix of squared Euclidean distances between rows of A and B.

    Uses the factored identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b and clamps
    at 0, mirroring the production code (and, deliberately, its round-off
    behaviour).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    d = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(d, 0.0)


def spatial_knn_positives(coords: np.ndarray, k: int) -> np.ndarray:
    """For each cell, the indices of its ``k`` nearest spatial neighbours.

    Self is always excluded. Returns an (n, k) integer array. Ties are broken by
    index order (argsort is stable via kind="stable").
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n - 1:
        raise ValueError(f"k={k} needs at least {k + 1} cells; got n={n}")
    d = pairwise_sq_dists(coords, coords)
    np.fill_diagonal(d, np.inf)          # never pick self
    order = np.argsort(d, axis=1, kind="stable")
    return order[:, :k]


# ---------------------------------------------------------------------------
# 2. InfoNCE on L2-normalised embeddings
# ---------------------------------------------------------------------------
def l2_normalise(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    n = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(n, eps)


def info_nce(
    z: np.ndarray,
    pos_idx: np.ndarray,
    temperature: float = 0.1,
    already_normalised: bool = False,
) -> float:
    """InfoNCE with (possibly several) positives per anchor.

    For anchor i with positive set P(i), the loss is the mean over p in P(i) of

        -log( exp(s_ip / T) / sum_{j != i} exp(s_ij / T) )

    where s_ij is the cosine similarity between embeddings i and j. The
    denominator runs over all other cells in the batch (in-batch negatives) and
    EXCLUDES the anchor itself, but deliberately still includes the other
    positives — this is the standard "multi-positive InfoNCE" convention used by
    SupCon (Khosla et al. 2020, Eq. 2, the L_out form).

    Returned value is the mean over anchors. Computed in a
    log-sum-exp-stable way.
    """
    z = np.asarray(z, dtype=np.float64)
    if not already_normalised:
        z = l2_normalise(z)
    n = z.shape[0]
    pos_idx = np.asarray(pos_idx)
    if pos_idx.ndim != 2 or pos_idx.shape[0] != n:
        raise ValueError(f"pos_idx must be (n, k); got {pos_idx.shape} for n={n}")

    sim = (z @ z.T) / float(temperature)
    # mask self in the denominator
    logits = sim.copy()
    np.fill_diagonal(logits, -np.inf)
    # log-sum-exp over j != i
    m = logits.max(axis=1, keepdims=True)
    lse = (m.squeeze(1)
           + np.log(np.exp(logits - m).sum(axis=1)))

    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        pos = pos_idx[i]
        losses[i] = float(np.mean(-(sim[i, pos] - lse[i])))
    return float(losses.mean())


# ---------------------------------------------------------------------------
# 3. Embedding -> coordinate read-out
# ---------------------------------------------------------------------------
def knn_coord_readout(
    z_query: np.ndarray,
    z_ref: np.ndarray,
    coords_ref: np.ndarray,
    k: int = 30,
    temperature: float = 0.1,
    already_normalised: bool = False,
):
    """Assign coordinates to query cells by softmax-weighted retrieval.

    For each query cell: take its ``k`` most cosine-similar reference cells,
    turn those similarities into weights with a softmax at ``temperature``, and
    return the weighted average of the retrieved reference coordinates.

    Returns ``(coords_pred, nbr_idx, weights)`` so callers/tests can inspect
    the retrieval, not just the output.

    NOTE: because the weights are non-negative and sum to 1, each prediction is
    a CONVEX COMBINATION of the k retrieved reference coordinates. Two direct
    consequences, both asserted in the tests: predictions can never leave the
    convex hull of the reference cloud, and as T -> 0 the prediction converges
    to the single nearest reference cell's coordinate.
    """
    zq = np.asarray(z_query, dtype=np.float64)
    zr = np.asarray(z_ref, dtype=np.float64)
    if not already_normalised:
        zq, zr = l2_normalise(zq), l2_normalise(zr)
    coords_ref = np.asarray(coords_ref, dtype=np.float64)
    if coords_ref.shape[0] != zr.shape[0]:
        raise ValueError("coords_ref and z_ref disagree on n_ref")
    n_ref = zr.shape[0]
    if not 1 <= k <= n_ref:
        raise ValueError(f"k={k} out of range for n_ref={n_ref}")

    sim = zq @ zr.T                                    # (nq, n_ref) cosine
    # top-k by similarity (descending); stable for reproducible tie-breaking
    nbr_idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    rows = np.arange(zq.shape[0])[:, None]
    top_sim = sim[rows, nbr_idx]                       # (nq, k)

    logits = top_sim / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)        # stabilise
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)                  # (nq, k), rows sum to 1

    coords_pred = np.einsum("qk,qkd->qd", w, coords_ref[nbr_idx])
    return coords_pred, nbr_idx, w


# ---------------------------------------------------------------------------
# Metric helper used only by the self-tests (the real harness is authoritative)
# ---------------------------------------------------------------------------
def spearman_pairwise(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Spearman correlation between the two clouds' pairwise-distance vectors.

    This mirrors the *spirit* of the paper's Spearman metric (rank correlation
    of pairwise distances) closely enough to serve as a correctness signal in
    tests. It is NOT the harness metric and must not be reported.
    """
    da = pairwise_sq_dists(coords_a, coords_a)
    db = pairwise_sq_dists(coords_b, coords_b)
    iu = np.triu_indices(da.shape[0], k=1)
    x, y = np.sqrt(da[iu]), np.sqrt(db[iu])

    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")
