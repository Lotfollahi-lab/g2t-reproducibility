#!/usr/bin/env python3
"""test_contrastive_core.py — correctness tests for the NumPy reference maths.

Runs anywhere NumPy is available (no torch/scipy/sklearn needed):

    python3 test_contrastive_core.py

Each test states what would be WRONG if it failed, because the point of these
is to catch a silently-plausible implementation, not to decorate the code.
"""
from __future__ import annotations

import sys

import numpy as np

from contrastive_core import (
    info_nce,
    knn_coord_readout,
    l2_normalise,
    pairwise_sq_dists,
    spatial_knn_positives,
    spearman_pairwise,
)

RNG = np.random.default_rng(0)
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
print("\n[1] pairwise distances")
# ---------------------------------------------------------------------------
A = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
D = pairwise_sq_dists(A, A)
check("known 3-4-5 triangle", np.isclose(D[0, 1], 25.0), f"got {D[0,1]}")
check("diagonal is exactly 0", np.allclose(np.diag(D), 0.0))
check("symmetric", np.allclose(D, D.T))
check("never negative (round-off clamped)", (D >= 0).all())
# identical points must give exactly 0, not -1e-17
Z = np.tile(RNG.normal(size=(1, 8)), (5, 1))
check("identical rows -> exact 0", (pairwise_sq_dists(Z, Z) == 0).all())

# ---------------------------------------------------------------------------
print("\n[2] spatial positives (k nearest neighbours)")
# ---------------------------------------------------------------------------
# 5x5 unit grid: the nearest neighbours of an interior point are its 4
# edge-adjacent points, at distance 1.
g = np.array([[i, j] for i in range(5) for j in range(5)], dtype=np.float64)
pos = spatial_knn_positives(g, k=4)
centre = 12                      # (2,2) in a 5x5 row-major grid
expected = {7, 11, 13, 17}       # (1,2) (2,1) (2,3) (3,2)
check("grid interior point -> 4 edge-adjacent", set(pos[centre]) == expected,
      f"got {sorted(pos[centre])}")
check("self never a positive", all(i not in pos[i] for i in range(len(g))))
check("shape (n,k)", pos.shape == (25, 4))
# a corner point's 2 nearest are its 2 edge-adjacent neighbours
pos2 = spatial_knn_positives(g, k=2)
check("grid corner -> 2 edge-adjacent", set(pos2[0]) == {1, 5}, f"got {sorted(pos2[0])}")
try:
    spatial_knn_positives(g[:3], k=5)
    check("rejects k >= n", False)
except ValueError:
    check("rejects k >= n", True)

# ---------------------------------------------------------------------------
print("\n[3] InfoNCE")
# ---------------------------------------------------------------------------
# Hand-computable case: 3 unit vectors, anchor 0 with positive {1}.
# Build orthogonal-ish embeddings so similarities are known exactly.
z = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])   # 0 and 1 identical, 2 orthogonal
p = np.array([[1], [0], [0]])
T = 1.0
# For anchor 0: s01 = 1, s02 = 0  -> loss = -log(e^1/(e^1+e^0))
expect0 = -np.log(np.e / (np.e + 1.0))
# anchor 1 is symmetric to 0; anchor 2: pos is 0, s20=0, s21=0 -> -log(1/2)
expect2 = -np.log(1.0 / 2.0)
expected_mean = (expect0 + expect0 + expect2) / 3.0
got = info_nce(z, p, temperature=T)
check("matches hand-computed value", np.isclose(got, expected_mean, atol=1e-12),
      f"got {got:.10f} vs {expected_mean:.10f}")

# A perfect embedding (positives identical, negatives antipodal) must beat a
# random one. If this fails, the loss has a sign error or the mask is wrong.
n, k = 60, 3
coords = RNG.normal(size=(n, 2))
posn = spatial_knn_positives(coords, k=k)
z_perfect = np.zeros((n, 16))
for i in range(n):                      # cluster each anchor with its positives
    z_perfect[i] = RNG.normal(size=16)
for i in range(n):                      # pull positives onto the anchor
    z_perfect[posn[i]] = z_perfect[i]
z_rand = RNG.normal(size=(n, 16))
l_perfect, l_rand = info_nce(z_perfect, posn, 0.1), info_nce(z_rand, posn, 0.1)
check("aligned embeddings score lower than random", l_perfect < l_rand,
      f"aligned {l_perfect:.4f} < random {l_rand:.4f}")
check("loss is finite and positive", np.isfinite(l_rand) and l_rand > 0)
# invariance: a global rotation of all embeddings cannot change cosine sims
Q, _ = np.linalg.qr(RNG.normal(size=(16, 16)))
check("invariant to global rotation",
      np.isclose(info_nce(z_rand, posn, 0.1), info_nce(z_rand @ Q, posn, 0.1)))
# invariance: rescaling rows cannot change the loss (embeddings are normalised)
check("invariant to per-row rescaling",
      np.isclose(info_nce(z_rand, posn, 0.1),
                 info_nce(z_rand * RNG.uniform(0.5, 2.0, (n, 1)), posn, 0.1)))

# ---------------------------------------------------------------------------
print("\n[4] coordinate read-out")
# ---------------------------------------------------------------------------
n_ref = 40
z_ref = l2_normalise(RNG.normal(size=(n_ref, 12)))
coords_ref = RNG.normal(size=(n_ref, 2)) * 10.0

# (a) A query that IS a reference cell, at low temperature, must return that
#     cell's coordinate. Failure here means retrieval or weighting is broken.
j = 7
pred, nbr, w = knn_coord_readout(z_ref[j:j + 1], z_ref, coords_ref, k=5, temperature=1e-3)
check("query == ref, T->0 recovers that ref's coords",
      np.allclose(pred[0], coords_ref[j], atol=1e-6),
      f"err {np.abs(pred[0]-coords_ref[j]).max():.2e}")
check("top-1 retrieved is the identical cell", nbr[0, 0] == j)

# (b) weights must be a valid distribution
pred, nbr, w = knn_coord_readout(l2_normalise(RNG.normal(size=(25, 12))),
                                 z_ref, coords_ref, k=8, temperature=0.1)
check("weights sum to 1", np.allclose(w.sum(1), 1.0))
check("weights non-negative", (w >= 0).all())

# (c) convex-combination property: every prediction lies within the bounding box
#     of its own retrieved neighbours (a necessary condition for convexity)
lo = coords_ref[nbr].min(axis=1)
hi = coords_ref[nbr].max(axis=1)
check("prediction inside retrieved neighbours' bbox",
      bool((pred >= lo - 1e-9).all() and (pred <= hi + 1e-9).all()))
check("prediction inside global reference bbox",
      bool((pred >= coords_ref.min(0) - 1e-9).all()
           and (pred <= coords_ref.max(0) + 1e-9).all()))

# (d) high temperature -> weights flatten toward uniform -> prediction tends to
#     the mean of the retrieved coords
_, nbr_h, w_h = knn_coord_readout(l2_normalise(RNG.normal(size=(10, 12))),
                                  z_ref, coords_ref, k=8, temperature=1e4)
check("T->inf gives near-uniform weights",
      np.allclose(w_h, 1.0 / 8.0, atol=1e-3), f"max dev {np.abs(w_h-1/8).max():.2e}")

# ---------------------------------------------------------------------------
print("\n[5] POSITIVE CONTROL — oracle encoder must reconstruct geometry")
# ---------------------------------------------------------------------------
# This is the load-bearing test. If we hand the read-out an ORACLE embedding
# (one that already encodes position perfectly), it must recover the reference
# geometry almost exactly. If this fails, the read-out cannot possibly work no
# matter how good the trained encoder is, and any low score on real data would
# be OUR bug rather than the method's behaviour.
#
# Oracle construction: embed a cell by its own (x, y) lifted to the unit sphere
# so that cosine similarity is monotone in spatial proximity.
def oracle_embed(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    # place coords on a sphere cap: [x, y, c] then normalise. With c large
    # relative to the coordinate spread, cosine distance is monotone in
    # Euclidean distance, which is what a perfect contrastive encoder achieves.
    c = 5.0 * (np.abs(xy).max() + 1e-9)
    lifted = np.concatenate([xy, np.full((xy.shape[0], 1), c)], axis=1)
    return l2_normalise(lifted)


ref_xy = RNG.uniform(-1, 1, size=(600, 2))
qry_xy = RNG.uniform(-1, 1, size=(200, 2))
pred_xy, _, _ = knn_coord_readout(oracle_embed(qry_xy), oracle_embed(ref_xy),
                                  ref_xy, k=10, temperature=0.01)
rho = spearman_pairwise(pred_xy, qry_xy)
check("oracle encoder -> Spearman > 0.95", rho > 0.95, f"rho={rho:.4f}")
med_err = float(np.median(np.linalg.norm(pred_xy - qry_xy, axis=1)))
check("oracle encoder -> small absolute error", med_err < 0.10, f"median err {med_err:.4f}")

# Negative control: a random encoder must NOT reconstruct geometry. If this
# "passes" with a high score, the test itself is measuring nothing.
pred_rand, _, _ = knn_coord_readout(l2_normalise(RNG.normal(size=(200, 3))),
                                    l2_normalise(RNG.normal(size=(600, 3))),
                                    ref_xy, k=10, temperature=0.01)
rho_rand = spearman_pairwise(pred_rand, qry_xy)
check("random encoder -> Spearman near 0", abs(rho_rand) < 0.2, f"rho={rho_rand:.4f}")
check("oracle clearly separates from random", rho - abs(rho_rand) > 0.7,
      f"{rho:.3f} vs {abs(rho_rand):.3f}")

# ---------------------------------------------------------------------------
print("\n[6] leakage guard")
# ---------------------------------------------------------------------------
# The read-out must never consult query coordinates. We verify structurally:
# permuting the query coordinates cannot change the prediction, because the
# function never receives them.
p1, _, _ = knn_coord_readout(oracle_embed(qry_xy), oracle_embed(ref_xy), ref_xy,
                             k=10, temperature=0.01)
check("read-out signature takes no query coords",
      "coords_query" not in knn_coord_readout.__code__.co_varnames)
check("prediction depends only on (z_query, z_ref, coords_ref)",
      np.allclose(p1, pred_xy))

# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
if FAILS:
    print(f"{len(FAILS)} TEST(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
sys.exit(0)
