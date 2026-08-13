#!/usr/bin/env python3
"""test_harness_adapter.py — tests for the harness seam (numpy + stdlib only).

    python3 test_harness_adapter.py

Covers the parts of harness_adapter.py that do NOT need anndata: the per-slice
coordinate scaler and the artifact writer. h5ad loading is exercised on the farm
(``--selftest`` in the runner), since anndata is not installed locally.

Why these specific tests: a silent bug in either piece would corrupt every
reported number while leaving plausible-looking CSVs. The scaler round-trip
protects Sum RSSD (computed against original-scale truth); the lossless-CSV
check protects all three metrics from text-rounding drift.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

from harness_adapter import (
    SliceCoordScaler,
    section_label_from_filename,
    write_slice_artifacts,
)

RNG = np.random.default_rng(0)
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


print("\n[1] SliceCoordScaler")
coords = RNG.uniform(-3000, 5000, size=(500, 2))
sc = SliceCoordScaler().fit(coords)
t = sc.transform(coords)
check("maps into [-0.5, 0.5]", t.min() >= -0.5 - 1e-12 and t.max() <= 0.5 + 1e-12,
      f"[{t.min():.3f},{t.max():.3f}]")
check("both axes span the full range",
      np.allclose(t.min(0), -0.5) and np.allclose(t.max(0), 0.5))
check("round-trip is exact", np.allclose(sc.inverse_transform(t), coords, atol=1e-9),
      f"max err {np.abs(sc.inverse_transform(t) - coords).max():.2e}")

# A slice where every cell shares an x value must not divide by zero.
deg = np.column_stack([np.full(50, 7.0), RNG.uniform(0, 1, 50)])
sdg = SliceCoordScaler().fit(deg)
tdg = sdg.transform(deg)
check("degenerate axis stays finite", np.isfinite(tdg).all())
check("degenerate axis round-trips", np.allclose(sdg.inverse_transform(tdg), deg, atol=1e-9))

# Predictions are made in scaled space and must invert into the slice's own range.
pred_scaled = np.clip(t + RNG.normal(0, 0.01, t.shape), -0.5, 0.5)
back = sc.inverse_transform(pred_scaled)
check("inverse maps predictions into the original range",
      back.min() >= coords.min() - 1e-6 and back.max() <= coords.max() + 1e-6)
try:
    SliceCoordScaler().transform(coords)
    check("errors if used unfitted", False)
except RuntimeError:
    check("errors if used unfitted", True)

print("\n[2] artifact writer schema")
with tempfile.TemporaryDirectory() as td:
    n = 17
    c_true = RNG.uniform(0, 100, (n, 2))
    c_pred = c_true + RNG.normal(0, 1, (n, 2))
    cls = np.array([f"class{i % 3}" for i in range(n)])
    p_pred, p_true = write_slice_artifacts(Path(td) / "well01", c_pred, c_true, cls)

    rows = list(csv.reader(open(p_pred)))
    check("header matches the harness", rows[0] == ["", "coord_X", "coord_Y", "cell_class"],
          str(rows[0]))
    check("row count is n+1", len(rows) == n + 1, str(len(rows)))
    check("both files written", p_pred.exists() and p_true.exists())

    got = np.array([[float(r[1]), float(r[2])] for r in rows[1:]])
    check("pred coords lossless through CSV", np.array_equal(got, c_pred),
          f"max err {np.abs(got - c_pred).max():.2e}")
    tr = np.array([[float(r[1]), float(r[2])] for r in list(csv.reader(open(p_true)))[1:]])
    check("true coords lossless through CSV", np.array_equal(tr, c_true))
    check("cell_class preserved", [r[3] for r in rows[1:]] == list(cls))

    for args, msg in [
        ((np.array([[np.nan, 0.0]]), np.array([[0.0, 0.0]])), "rejects NaN predictions"),
        ((RNG.normal(size=(3, 2)), RNG.normal(size=(4, 2))), "rejects shape mismatch"),
    ]:
        try:
            write_slice_artifacts(Path(td) / "bad", args[0], args[1], None)
            check(msg, False)
        except ValueError:
            check(msg, True)

    p2, _ = write_slice_artifacts(Path(td) / "nocls", c_pred, c_true, None)
    check("omits cell_class when absent",
          list(csv.reader(open(p2)))[0] == ["", "coord_X", "coord_Y"])

print("\n[3] section labels")
for fname, expected in [("well06_test.h5ad", "well06"),
                        ("well1_5_train.h5ad", "well1_5"),
                        ("sagittal3_test.h5ad", "sagittal3")]:
    check(f"{fname} -> {expected}",
          section_label_from_filename(Path(fname)) == expected,
          section_label_from_filename(Path(fname)))

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} TEST(S) FAILED: {FAILS}")
    sys.exit(1)
print("ALL ADAPTER TESTS PASSED")
sys.exit(0)
