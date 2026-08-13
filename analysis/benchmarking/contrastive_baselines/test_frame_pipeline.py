#!/usr/bin/env python3
"""test_frame_pipeline.py — end-to-end test of the COORDINATE-FRAME chain that
run_cellcontrast.py implements, using a mock that reproduces upstream's exact
behaviour (top-1 argmax copy of a reference cell's coordinates).

    python3 test_frame_pipeline.py

Why this test exists
--------------------
This is the one place a silent bug would produce plausible-but-wrong numbers.
The chain is:

  train slices (each in its OWN micron frame)
    -> per-slice min-max to [-0.5, 0.5]        (build_reference)
    -> obs['x'], obs['y'] of the reference object
    -> CellContrast copies ref coords verbatim (inference.map_to_ST, top-1)
    -> predictions therefore arrive in the SHARED normalised frame
    -> invert with the TEST slice's own scaler  (main loop)
    -> metadata_pred.csv, in the test slice's microns

Every step is individually reasonable and the whole thing still fails if any one
is wrong — e.g. forgetting to normalise the training slices makes predictions
arrive in some training slice's microns, and the inverse-transform then produces
garbage. Test [3] deliberately introduces exactly that bug and asserts we catch
it, which is what gives the passing result its meaning.
"""
from __future__ import annotations

import sys

import numpy as np

from contrastive_core import knn_coord_readout, l2_normalise, spearman_pairwise
from harness_adapter import SliceCoordScaler

RNG = np.random.default_rng(7)
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# Synthetic tissue: a shared latent "anatomy" observed in per-slice frames.
# Each slice has its own offset and scale, exactly like real micron frames.
# ---------------------------------------------------------------------------
def make_slice(n: int, offset: np.ndarray, scale: float, rng) -> np.ndarray:
    """Coordinates for one slice, in that slice's own arbitrary micron frame."""
    anat = rng.uniform(-1.0, 1.0, size=(n, 2))       # shared anatomical layout
    return anat * scale + offset


def oracle_embed(xy_normalised: np.ndarray) -> np.ndarray:
    """A perfect encoder: cosine similarity monotone in normalised-frame distance.

    Stands in for CellContrast's trained encoder. Using an ORACLE isolates the
    frame logic under test from model quality — the question here is only
    'does the frame chain preserve geometry', not 'does contrastive learning work'.
    """
    c = 5.0 * (np.abs(xy_normalised).max() + 1e-9)
    lifted = np.concatenate([xy_normalised, np.full((len(xy_normalised), 1), c)], axis=1)
    return l2_normalise(lifted)


def cellcontrast_mock(feat_query, feat_ref, ref_xy_normalised):
    """Upstream's map_to_ST: copy the argmax-similarity reference cell's (x, y).

    k=1 with any temperature reduces the read-out to a verbatim top-1 copy, which
    is what inference.py does (`cur_st_coor = ref_coors[ind[0]]`).
    """
    pred, nbr, w = knn_coord_readout(feat_query, feat_ref, ref_xy_normalised,
                                     k=1, temperature=1.0)
    return pred, nbr


# ---------------------------------------------------------------------------
print("\n[1] the pipeline as implemented (normalise train slices -> copy -> invert)")
# ---------------------------------------------------------------------------
train_specs = [(400, np.array([1000.0, -500.0]), 300.0),
               (350, np.array([-8000.0, 200.0]), 120.0),
               (500, np.array([50.0, 9000.0]), 800.0)]
train_slices = [make_slice(n, off, sc, RNG) for n, off, sc in train_specs]

# build_reference: per-slice min-max, then concatenate
ref_norm = np.vstack([SliceCoordScaler().fit(c).transform(c) for c in train_slices])
check("reference lands in [-0.5, 0.5]",
      ref_norm.min() >= -0.5 - 1e-12 and ref_norm.max() <= 0.5 + 1e-12,
      f"[{ref_norm.min():.3f}, {ref_norm.max():.3f}]")
check("frames are genuinely different pre-normalisation",
      max(c.max() for c in train_slices) - min(c.min() for c in train_slices) > 1e4)

# a held-out test slice in its own frame
test_xy = make_slice(300, np.array([4242.0, -777.0]), 55.0, RNG)
test_scaler = SliceCoordScaler().fit(test_xy)

# the encoder sees NORMALISED geometry (that is what it was trained on)
feat_ref = oracle_embed(ref_norm)
feat_qry = oracle_embed(test_scaler.transform(test_xy))

pred_norm, nbr = cellcontrast_mock(feat_qry, feat_ref, ref_norm)
check("predictions are copies of reference positions",
      all(np.allclose(pred_norm[i], ref_norm[nbr[i, 0]]) for i in range(len(pred_norm))))
check("predictions stay inside the normalised frame",
      pred_norm.min() >= -0.5 - 1e-9 and pred_norm.max() <= 0.5 + 1e-9)

pred_microns = test_scaler.inverse_transform(pred_norm)
check("inverted predictions land in the test slice's own range",
      pred_microns.min() >= test_xy.min() - 1e-6
      and pred_microns.max() <= test_xy.max() + 1e-6,
      f"pred [{pred_microns.min():.0f},{pred_microns.max():.0f}] vs "
      f"true [{test_xy.min():.0f},{test_xy.max():.0f}]")

rho = spearman_pairwise(pred_microns, test_xy)
check("geometry is recovered (Spearman > 0.95)", rho > 0.95, f"rho={rho:.4f}")
# Spearman is scale-free, so it must be identical in either frame — a cheap check
# that the inverse transform is a pure affine map and not distorting anything.
rho_norm = spearman_pairwise(pred_norm, test_scaler.transform(test_xy))
check("Spearman identical in normalised and micron frames",
      np.isclose(rho, rho_norm, atol=1e-9), f"{rho:.6f} vs {rho_norm:.6f}")

# ---------------------------------------------------------------------------
print("\n[2] expected top-1 quantisation (a property of the method, not a bug)")
# ---------------------------------------------------------------------------
uniq = len({(round(a, 9), round(b, 9)) for a, b in pred_norm})
check("distinct predicted positions < n_query (clumping occurs)", uniq < len(pred_norm),
      f"{uniq}/{len(pred_norm)} distinct")
check("clumping is not total", uniq > len(pred_norm) * 0.2, f"{uniq} distinct")
print(f"       note: {100*uniq/len(pred_norm):.1f}% distinct — the runner logs this "
      f"per slice; do NOT jitter it away, it is the method's behaviour")

# ---------------------------------------------------------------------------
print("\n[3] NEGATIVE CONTROL — the bug this test is designed to catch")
# ---------------------------------------------------------------------------
# Break the chain exactly as a plausible implementation slip would: concatenate
# RAW training coordinates instead of per-slice-normalised ones. Everything
# downstream still 'works' and produces finite numbers.
ref_raw = np.vstack(train_slices)
feat_ref_bad = oracle_embed(ref_raw)                  # encoder sees raw microns
pred_bad_raw, _ = cellcontrast_mock(feat_qry, feat_ref_bad, ref_raw)
pred_bad = test_scaler.inverse_transform(pred_bad_raw)
rho_bad = spearman_pairwise(pred_bad, test_xy)
check("un-normalised reference produces a clearly worse result", rho_bad < 0.8,
      f"rho={rho_bad:.4f} (correct pipeline: {rho:.4f})")
check("and lands outside the test slice's range (detectable)",
      pred_bad.min() < test_xy.min() - 1e-6 or pred_bad.max() > test_xy.max() + 1e-6,
      f"pred [{pred_bad.min():.0f},{pred_bad.max():.0f}]")
print("       -> so a passing test [1] is informative: the same harness detects "
      "the frame bug")

# ---------------------------------------------------------------------------
print("\n[4] leakage check on the frame calibration")
# ---------------------------------------------------------------------------
# The inverse transform uses the TEST slice's min/max (as CeLEry does). That is an
# affine calibration, so it cannot change rank-based metrics. Verify: perturbing
# the test coords' scale/offset leaves Spearman untouched.
shifted = test_xy * 3.7 + np.array([1e5, -2e5])
rho_shift = spearman_pairwise(
    SliceCoordScaler().fit(shifted).inverse_transform(pred_norm), shifted)
check("Spearman invariant to the test frame's scale/offset",
      np.isclose(rho, rho_shift, atol=1e-9), f"{rho:.6f} vs {rho_shift:.6f}")
print("       -> the calibration affects Sum RSSD only; it carries no per-cell "
      "information (documented in the runner)")

print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} TEST(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL FRAME-PIPELINE TESTS PASSED")
sys.exit(0)
