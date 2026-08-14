#!/usr/bin/env python3
"""test_come_wrapper.py — tests for run_come.py.

numpy-only, so it runs in the local checkout (COME itself needs torch, which is
only in the farm env; the pure functions and the read-out are what we test here).

Covers, worst-consequence first:
  1. coords_from_coefficient — the read-out. Must implement UPSTREAM's rule
     (argmax over SPOTS per cell, i.e. Coefficient.max(dim=0)); getting the axis
     wrong silently produces plausible-but-wrong coordinates.
  2. degenerate / NaN fits must RAISE, not be scored as a weak baseline.
  3. estimate_peak_bytes / max_total_for_budget — the feasibility gate that
     decides whether a job is launched at all.
  4. load_query — mandatory cell_class, and the empty-cell policy that protects
     row alignment against upstream's filter_cells(min_genes=1).

Usage: python test_come_wrapper.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_come as C  # noqa: E402

FAILED: list = []


def check(cond: bool, name: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}" + (f"   [{detail}]" if detail else ""))
        FAILED.append(name)


def expect_raises(fn, exc, name: str) -> None:
    try:
        fn()
    except exc as e:
        print(f"  PASS  {name}   [{type(e).__name__}]")
        return
    except Exception as e:
        print(f"  FAIL  {name}   [raised {type(e).__name__}, wanted {exc.__name__}]")
        FAILED.append(name)
        return
    print(f"  FAIL  {name}   [no exception]")
    FAILED.append(name)


# ---------------------------------------------------------------------------
print("\n[1] coords_from_coefficient — must use UPSTREAM's axis (argmax over spots)")
# 4 spots, 3 cells. Coefficient is (n_spots, n_cells).
ref_xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
C_mat = np.array([
    [0.1, 0.9, 0.1],     # spot 0
    [0.2, 0.1, 0.1],     # spot 1
    [0.9, 0.2, 0.1],     # spot 2
    [0.1, 0.1, 0.9],     # spot 3
])
xy, nd = C.coords_from_coefficient(C_mat, ref_xy)
# cell0 -> spot2, cell1 -> spot0, cell2 -> spot3
check(np.array_equal(xy, ref_xy[[2, 0, 3]]),
      "each cell gets its highest-coefficient SPOT's coords", str(xy.tolist()))
check(xy.shape == (3, 2), "output is (n_cells, 2)", str(xy.shape))
check(nd == 3, "distinct-position count is right", str(nd))
# The wrong axis (argmax over cells per spot) would give 4 rows, not 3 — assert
# we did not do that.
check(xy.shape[0] == C_mat.shape[1],
      "row count follows CELLS (columns of Coefficient), not spots")
# Predictions must be a verbatim copy of reference positions.
ref_set = {tuple(r) for r in ref_xy.tolist()}
check(all(tuple(r) in ref_set for r in xy.tolist()),
      "every predicted position is a verbatim reference spot position")

print("\n[2] degenerate and invalid fits must raise")
collapsed = np.zeros((4, 5)); collapsed[1, :] = 1.0        # every cell -> spot 1
expect_raises(lambda: C.coords_from_coefficient(collapsed, ref_xy), ValueError,
              "total collapse onto one spot raises for a REAL run")
# ...but a smoke test must be allowed through: Coefficient starts UNIFORM
# (model.py:48), so a 2-epoch fit collapses by construction and blocking would
# stop the smoke test before it exercises artifact writing.
xy_c, nd_c = C.coords_from_coefficient(collapsed, ref_xy, allow_degenerate=True)
check(nd_c == 1 and xy_c.shape == (5, 2),
      "collapse is a WARNING under allow_degenerate (smoke path continues)",
      f"n_distinct={nd_c}")
# A uniform Coefficient is exactly what upstream initialises, and it must
# reproduce the collapse we saw on the farm.
uniform = np.full((4, 5), 1.0 / 20)
xy_u, nd_u = C.coords_from_coefficient(uniform, ref_xy, allow_degenerate=True)
check(nd_u == 1, "uniform Coefficient (upstream's init) collapses to one spot",
      f"n_distinct={nd_u}")
check(np.array_equal(xy_u, np.repeat(ref_xy[:1], 5, axis=0)),
      "uniform init maps every cell to spot 0 (argmax first-maximal tie-break)")
nan_mat = C_mat.copy(); nan_mat[0, 0] = np.nan
expect_raises(lambda: C.coords_from_coefficient(nan_mat, ref_xy), ValueError,
              "NaN in Coefficient raises (diverged fit)")
expect_raises(lambda: C.coords_from_coefficient(C_mat, ref_xy[:2]), ValueError,
              "spot-count mismatch between Coefficient and reference raises")
expect_raises(lambda: C.coords_from_coefficient(np.zeros(5), ref_xy), ValueError,
              "non-2D Coefficient raises")
# A 2-distinct-position fit is degenerate-ish but legal (only 1 raises).
two = np.zeros((4, 6)); two[0, :3] = 1.0; two[3, 3:] = 1.0
xy2, nd2 = C.coords_from_coefficient(two, ref_xy)
check(nd2 == 2, "two distinct positions is allowed (only total collapse raises)")

print("\n[3] estimate_peak_bytes — the feasibility gate")
# Reproduce the documented figures.
cases = {
    "COME paper VISp (1k spots, 15,413 cells)": (1000, 15413, 6.6),
    "mmc_luna full reference":                  (158379, 5235, 552.1),
    "mmc_luna ref 20k (the planned protocol)":  (20000, 5235, 14.9),
    "cns_luna as the other methods run it":     (150000, 63343, 1116.4),
    "cns_luna query ALONE, empty reference":    (0, 63343, 96.3),
}
for label, (n1, n2, want) in cases.items():
    got = C.estimate_peak_bytes(n1, n2) / 1e9
    check(abs(got - want) < 0.15, f"{label}", f"{got:.1f} GB (doc says {want})")
check(C.estimate_peak_bytes(0, 0) == 0, "empty inputs -> 0 bytes")
check(C.estimate_peak_bytes(10, 20) == C.estimate_peak_bytes(10, 20),
      "estimate is deterministic")
# monotone in both arguments
check(C.estimate_peak_bytes(1000, 500) < C.estimate_peak_bytes(2000, 500),
      "monotone in n_ref")
check(C.estimate_peak_bytes(500, 1000) < C.estimate_peak_bytes(500, 2000),
      "monotone in n_query")
# the query term dominates -> lowering the reference cannot rescue a big query
big_q = C.estimate_peak_bytes(0, 63343)
check(big_q > 90e9,
      "with a 63k query even an EMPTY reference is >90 GB (so --max_ref_cells "
      "cannot rescue cns_luna)", f"{big_q/1e9:.0f} GB")
for b in (8, 32, 128, 256):
    n = C.max_total_for_budget(b)
    check(20 * n * n <= b * 1e9 < 20 * (n + 2) ** 2,
          f"max_total_for_budget({b}) is the tight bound", f"{n:,}")

print("\n[4] load_query — mandatory labels and the empty-cell policy")


class _Col:
    def __init__(self, v): self.v = v
    def astype(self, _): return self
    def to_numpy(self): return np.asarray(self.v, dtype=object)


class _Obs(dict):
    def __getitem__(self, k): return _Col(dict.__getitem__(self, k))


class _Ad:
    def __init__(self, X, xy, cls, names, with_cls=True):
        self.X = X
        self.obsm = {"spatial": xy}
        self.obs = _Obs({"cell_class": cls} if with_cls else {})
        self.var_names = [f"g{i}" for i in range(X.shape[1])]
        self.obs_names = names
        self.n_obs = X.shape[0]


class _Mod:
    def __init__(self, a): self.a = a
    def read_h5ad(self, *_x, **_k): return self.a


class _Args:
    on_empty_cells = "fail"


rng = np.random.default_rng(0)
n, g = 12, 4
X = rng.random((n, g)).astype(np.float32) + 0.1        # no empty rows
xy = np.column_stack([rng.uniform(0, 100, n), rng.uniform(0, 50, n)])
cls = np.array([f"t{i%3}" for i in range(n)], dtype=object)
names = np.array([f"c{i}" for i in range(n)], dtype=object)
ref_var = [f"g{i}" for i in range(g)]

feats, ct, cl, on, n_empty = C.load_query(_Mod(_Ad(X, xy, cls, names)),
                                         Path("s_test.h5ad"), _Args(), ref_var)
check(np.array_equal(feats, X), "features returned unpermuted")
check(np.array_equal(ct, xy), "truth coords returned unpermuted")
check(list(on) == list(names), "obs_names returned in row order")
check(n_empty == 0, "no empty cells reported for a dense matrix")

# missing cell_class -> refuse (needed by BOTH the scorer and COME's type mask)
expect_raises(lambda: C.load_query(_Mod(_Ad(X, xy, cls, names, with_cls=False)),
                                   Path("s_test.h5ad"), _Args(), ref_var),
              ValueError, "missing obs['cell_class'] refused")

# empty cells: fail / keep / drop
Xe = X.copy(); Xe[3, :] = 0.0; Xe[7, :] = 0.0
expect_raises(lambda: C.load_query(_Mod(_Ad(Xe, xy, cls, names)),
                                   Path("s_test.h5ad"), _Args(), ref_var),
              ValueError, "empty cells refused by default (protects row alignment)")


class _Keep(_Args):
    on_empty_cells = "keep"


f2, c2, l2, o2, ne2 = C.load_query(_Mod(_Ad(Xe, xy, cls, names)),
                                   Path("s_test.h5ad"), _Keep(), ref_var)
check(f2.shape[0] == n and ne2 == 2,
      "keep: population preserved and the count is reported", f"n_empty={ne2}")


class _Drop(_Args):
    on_empty_cells = "drop"


f3, c3, l3, o3, ne3 = C.load_query(_Mod(_Ad(Xe, xy, cls, names)),
                                   Path("s_test.h5ad"), _Drop(), ref_var)
check(f3.shape[0] == n - 2 and ne3 == 2, "drop: two cells removed", f"n={f3.shape[0]}")
check(list(o3) == [f"c{i}" for i in range(n) if i not in (3, 7)],
      "drop: coords/labels/obs_names stay aligned with the surviving features")
check(np.array_equal(c3, xy[[i for i in range(n) if i not in (3, 7)]]),
      "drop: truth coords filtered by the SAME mask as the features")

# gene-panel mismatch must be refused (COME intersects gene sets silently)
expect_raises(lambda: C.load_query(_Mod(_Ad(X, xy, cls, names)),
                                   Path("s_test.h5ad"), _Args(),
                                   ref_var + ["g_absent"]),
              ValueError, "reference gene absent from the test panel refused")

print("\n[4b] coefficient_diagnostics — must make a collapse self-explaining")
# The exact situation observed on the farm: Coefficient still constant.
const = np.full((500, 5180), 1.0 / (500 * 5180), dtype=np.float32)
d_const = C.coefficient_diagnostics(const)
check(d_const["col_std_median"] > 0.0,
      "a float32 constant matrix does NOT give col_std_median == 0 "
      "(why the absolute test was wrong)", f"{d_const['col_std_median']:.3g}")
check(d_const["col_std_rel"] <= C.COL_STD_REL_DEGENERATE,
      "constant Coefficient IS flagged by the RELATIVE criterion",
      f"col_std_rel={d_const['col_std_rel']:.3g} <= {C.COL_STD_REL_DEGENERATE:.0e}")
check(d_const["distinct_argmax_spots"] == 1,
      "constant Coefficient -> 1 distinct spot")
check(d_const["frac_cells_on_spot0"] == 1.0,
      "constant Coefficient -> every cell lands on spot 0 (tie-break)")
# A healthy, differentiated matrix
rngd = np.random.default_rng(1)
healthy = rngd.random((200, 900)).astype(np.float32)
d_ok = C.coefficient_diagnostics(healthy)
check(d_ok["col_std_rel"] > C.COL_STD_REL_DEGENERATE * 100,
      "differentiated Coefficient is far above the threshold (wide margin)",
      f"col_std_rel={d_ok['col_std_rel']:.3g}")
check(d_ok["distinct_argmax_spots"] > 50,
      "differentiated Coefficient -> many distinct spots",
      str(d_ok["distinct_argmax_spots"]))
check(set(d_ok) == {"n_spots", "n_cells", "distinct_argmax_spots",
                    "frac_cells_on_spot0", "C_min", "C_max", "C_mean", "C_std",
                    "col_std_median", "col_std_min", "col_std_rel"},
      "diagnostics dict has the documented keys (it goes into the manifest)")
check(all(isinstance(v, (int, float)) for v in d_ok.values()),
      "all diagnostics are JSON-serialisable scalars")

print("\n[5] _patch_contrastive_device — fixes upstream's CUDA crash, changes no maths")
# Fake the two-device situation with plain Python objects: no torch needed. A
# "tensor" here records its device and whether it was moved.


class _T:
    def __init__(self, device, tag):
        self.device = device
        self.tag = tag
        self.moved_to = None

    def to(self, device):
        out = _T(device, self.tag)
        out.moved_to = device
        return out


class _FakeCL:
    """Stand-in for upstream ContrastiveLoss.forward: records what it received
    and raises if the mask device differs from the embeddings', exactly as
    torch.mul does at model.py:170."""
    seen = None

    def forward(self, h1, h2, mask=None):
        if mask is not None and mask.device != h1.device:
            raise RuntimeError("Expected all tensors to be on the same device, "
                               f"but found at least two devices, {h1.device} "
                               f"and {mask.device}!")
        _FakeCL.seen = (h1.device, mask.device if mask is not None else None)
        return "loss"


class _FakeModule:
    ContrastiveLoss = _FakeCL


# Before patching, upstream's own behaviour must reproduce the reported error.
h1, h2 = _T("cuda:0", "z1"), _T("cuda:0", "z2")
cpu_mask = _T("cpu", "full_mask")
expect_raises(lambda: _FakeCL().forward(h1, h2, mask=cpu_mask), RuntimeError,
              "unpatched: CPU mask + CUDA embeddings raises (the reported bug)")

mod = _FakeModule()
applied = C._patch_contrastive_device(mod)
check(applied is True, "patch reports it was applied")
res = mod.ContrastiveLoss().forward(h1, h2, mask=cpu_mask)
check(res == "loss", "patched: the call now succeeds")
check(_FakeCL.seen == ("cuda:0", "cuda:0"),
      "patched: the mask arrives on the EMBEDDINGS' device", str(_FakeCL.seen))
check(cpu_mask.device == "cpu",
      "the caller's own mask object is left on its original device (we pass a copy)")
# idempotent: importing twice must not wrap twice
check(C._patch_contrastive_device(mod) is False,
      "patch is idempotent (a second import does not double-wrap)")
# a same-device mask must be passed through UNTOUCHED (no needless copy)
gpu_mask = _T("cuda:0", "m")
mod.ContrastiveLoss().forward(h1, h2, mask=gpu_mask)
check(gpu_mask.moved_to is None,
      "a mask already on the right device is not copied")
# mask=None must still work (upstream's mask_correlated_samples path)
_FakeCL.seen = None
check(mod.ContrastiveLoss().forward(h1, h2) == "loss",
      "mask=None still reaches upstream unchanged")

print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
    raise SystemExit(1)
print("ALL COME-WRAPPER TESTS PASSED")
