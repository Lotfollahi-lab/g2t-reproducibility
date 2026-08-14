#!/usr/bin/env python3
"""test_wrapper_fixes.py — tests for the CellContrast wrapper's correctness fixes.

Runs with numpy ONLY (no torch/anndata/scanpy/pandas), so it is executable in the
local checkout rather than only on the farm. Where a function needs anndata, we
inject a stub module — every such function takes ``ad_mod`` as a parameter
precisely so this is possible.

Covers, in order of how badly a regression would hurt:
  1. chunk_bounds        — must cover every row exactly once, in order, and must
                           degenerate to the original single-call path.
  2. estimate_peak_bytes — the memory model that decides whether a job is
                           launched at all.
  3. isotropic frame     — the property the default now rests on: the k-NN
                           positive graph handed to upstream must be IDENTICAL to
                           raw microns.
  4. platform_check      — k=80 vs k=20 is silent if it goes wrong.
  5. _assert_finite      — a NaN row is otherwise assigned reference cell N-1.
  6. seeded launcher     — actually executes it and proves stdlib random is
                           seeded and argv is handed over correctly.
  7. row pairing         — prediction/truth/label/index must stay aligned; a
                           permutation here is invisible in every metric.
  8. use_obsm spatial    — must be refused (it would feed ground truth).

Usage: python test_wrapper_fixes.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_cellcontrast as R                                    # noqa: E402
from harness_adapter import SliceCoordScaler, write_slice_artifacts  # noqa: E402

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
    except Exception as e:  # wrong type
        print(f"  FAIL  {name}   [raised {type(e).__name__}, wanted {exc.__name__}]")
        FAILED.append(name)
        return
    print(f"  FAIL  {name}   [no exception]")
    FAILED.append(name)


# ---------------------------------------------------------------------------
print("\n[1] chunk_bounds — exact cover, ordered, degenerates to one call")
for n, c in [(10, 3), (10, 10), (10, 100), (1, 5), (5235, 8000), (63343, 8000)]:
    b = R.chunk_bounds(n, c)
    covered = [i for lo, hi in b for i in range(lo, hi)]
    check(covered == list(range(n)), f"n={n} chunk={c} covers 0..n-1 once, in order",
          f"{len(b)} chunk(s)")
    check(all(lo < hi for lo, hi in b), f"n={n} chunk={c} no empty range")
check(R.chunk_bounds(5235, 8000) == [(0, 5235)],
      "slice smaller than chunk -> single range (original path, bit-identical)")
check(R.chunk_bounds(100, 0) == [(0, 100)], "chunk=0 disables chunking")
check(len(R.chunk_bounds(63343, 8000)) == 8, "63,343 cells at chunk 8000 -> 8 chunks")

# ---------------------------------------------------------------------------
print("\n[2] estimate_peak_bytes — the launch gate")
# Formula: max(24 Q^2, 16 Q^2 + 24 Q R)
q, r = 63343, 150000
unchunked = R.estimate_peak_bytes(q, r)
check(abs(unchunked - (16 * q * q + 24 * q * r)) < 1,
      "well10 unchunked uses the Q x R term", f"{unchunked/1e9:.1f} GB")
check(unchunked / 1e9 > 250, "well10 unchunked exceeds any sane LSF cap",
      f"{unchunked/1e9:.1f} GB")
chunked = R.estimate_peak_bytes(8000, r)
check(chunked / 1e9 < 40, "chunk=8000 brings it under 40 GB",
      f"{chunked/1e9:.1f} GB")
check(chunked < unchunked, "chunking strictly reduces the estimate")
# mmc_luna must be unaffected
mmc = R.estimate_peak_bytes(5235, 158379)
check(mmc / 1e9 < 25, "mmc_luna single-chunk peak is small", f"{mmc/1e9:.1f} GB")
# Q^2 term dominates only when R is small
check(R.estimate_peak_bytes(1000, 10) == 24 * 1000 * 1000,
      "small reference -> the 24*Q^2 query x query term dominates")
check(R.estimate_peak_bytes(2 * 8000, r) > 2 * R.estimate_peak_bytes(8000, r) - 1,
      "peak is superlinear in chunk size (halving the chunk more than halves it)")

# ---------------------------------------------------------------------------
print("\n[3] isotropic frame — upstream's positive graph must be preserved")


def knn(c, k):
    d = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


rng = np.random.default_rng(0)
K = 20
for (W, H, name) in [(5000, 2000, "aspect 2.50"), (7521, 5475, "aspect 1.37"),
                     (3000, 3000, "aspect 1.00")]:
    coords = np.column_stack([rng.uniform(0, W, 400), rng.uniform(0, H, 400)])
    raw = knn(coords, K)
    iso = knn(SliceCoordScaler(isotropic=True).fit(coords).transform(coords), K)
    ani = knn(SliceCoordScaler(isotropic=False).fit(coords).transform(coords), K)
    same_iso = np.mean([set(raw[i].tolist()) == set(iso[i].tolist())
                        for i in range(len(raw))])
    ov_ani = np.mean([len(set(raw[i].tolist()) & set(ani[i].tolist())) / K
                      for i in range(len(raw))])
    check(same_iso == 1.0, f"{name}: isotropic preserves EVERY positive set",
          f"{100*same_iso:.0f}% identical")
    if W != H:
        check(ov_ani < 0.999, f"{name}: per_axis measurably perturbs positives",
              f"overlap {ov_ani:.4f}")
# round-trip exactness in both modes (RSSD depends on it)
for iso in (True, False):
    coords = np.column_stack([rng.uniform(-4000, 9000, 300),
                              rng.uniform(500, 2000, 300)])
    s = SliceCoordScaler(isotropic=iso).fit(coords)
    back = s.inverse_transform(s.transform(coords))
    check(np.allclose(back, coords, rtol=0, atol=1e-9),
          f"isotropic={iso}: inverse_transform is an exact inverse",
          f"max|d|={np.abs(back-coords).max():.2e}")
# isotropic must stay INSIDE the box (predictions are range-checked at [-0.5,0.5])
coords = np.column_stack([rng.uniform(0, 9000, 500), rng.uniform(0, 1500, 500)])
t = SliceCoordScaler(isotropic=True).fit(coords).transform(coords)
check(t.min() >= -0.5 - 1e-12 and t.max() <= 0.5 + 1e-12,
      "isotropic output stays within [-0.5, 0.5] (check_predictions would raise)",
      f"[{t.min():.4f}, {t.max():.4f}]")
check(abs(t[:, 1].mean()) < 0.05,
      "isotropic centres the narrow axis rather than pinning it to the low edge",
      f"mean y {t[:,1].mean():+.4f}")

# ---------------------------------------------------------------------------
print("\n[4] platform_check — k=80 vs k=20 must not be silent")
expect_raises(lambda: R.platform_check("dlpfc_visium", True, False), SystemExit,
              "Visium + --single_cell (k=80) refused")
expect_raises(lambda: R.platform_check("mmc_luna", False, False), SystemExit,
              "MERFISH + --no_single_cell (k=20) refused")
R.platform_check("dlpfc_visium", False, False)
print("  PASS  Visium + --no_single_cell accepted")
R.platform_check("mmc_luna", True, False)
print("  PASS  mmc_luna + --single_cell accepted")
R.platform_check("breast_janesick", True, False)
print("  PASS  breast (Xenium, imaging) + --single_cell accepted")
R.platform_check("dlpfc_visium", True, True)
print("  PASS  --force_platform bypasses the check")
R.platform_check("some_unknown_tissue", True, False)
print("  PASS  unrecognised dataset name does not block")

# ---------------------------------------------------------------------------
print("\n[5] _assert_finite — a NaN row silently lands on reference cell N-1")
good = np.arange(12, dtype=np.float32).reshape(4, 3)
check(R._assert_finite(good, "ok") is good, "finite matrix passes through")
bad = good.copy(); bad[2, 1] = np.nan
expect_raises(lambda: R._assert_finite(bad, "q"), ValueError, "NaN raises")
bad2 = good.copy(); bad2[0, 0] = np.inf
expect_raises(lambda: R._assert_finite(bad2, "q"), ValueError, "Inf raises")

# ---------------------------------------------------------------------------
print("\n[6] seeded launcher — execute it and prove random is seeded")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    fake_repo = td / "repo"
    fake_repo.mkdir()
    # A stand-in for the upstream dispatcher: prints argv and a random draw.
    (fake_repo / "cellContrast.py").write_text(
        "import random, sys\n"
        "if __name__ == '__main__':\n"
        "    print('ARGV=' + '|'.join(sys.argv))\n"
        "    print('RAND=%.12f' % random.random())\n"
    )
    work = td / "work"; work.mkdir()
    argv = ["cellContrast.py", "train", "--train_data_path", "ref.h5ad"]
    outs = []
    for seed in (0, 0, 7):
        lp = R.write_seeded_launcher(work, fake_repo, seed, argv)
        p = subprocess.run([sys.executable, str(lp)], capture_output=True,
                           text=True, cwd=str(fake_repo))
        check(p.returncode == 0, f"launcher runs (seed={seed})", p.stderr.strip()[:80])
        outs.append(p.stdout)
    a0, a1, a7 = outs
    # runpy.run_path sets argv[0] to the script's full path — which is exactly
    # what `python cellContrast.py train ...` also does — and the dispatcher only
    # reads argv[1] (the submodule) before slicing argv[1:] for the submodule's
    # argparse. So assert on argv[0]'s basename and argv[1:] verbatim.
    got = a0.split("ARGV=")[1].splitlines()[0].split("|")
    check(Path(got[0]).name == "cellContrast.py",
          "argv[0] is the dispatcher path (as with a normal python invocation)",
          got[0])
    check(got[1:] == ["train", "--train_data_path", "ref.h5ad"],
          "argv[1:] reaches the dispatcher verbatim", "|".join(got[1:]))
    r0 = a0.split("RAND=")[1].strip()
    r1 = a1.split("RAND=")[1].strip()
    r7 = a7.split("RAND=")[1].strip()
    check(r0 == r1, "same seed -> identical random stream (reproducible replicate)",
          r0)
    check(r0 != r7, "different seed -> different random stream (genuine replicate)",
          f"{r0} vs {r7}")
    check("torch" not in (fake_repo / "cellContrast.py").read_text()
          and "manual_seed" not in R.write_seeded_launcher(
              work, fake_repo, 3, argv).read_text(),
          "launcher does NOT touch torch's RNG (would change the method)")

# ---------------------------------------------------------------------------
print("\n[7] row pairing — permutation is invisible to every metric, so guard it")


class _StubObs(dict):
    """Minimal stand-in for adata.obs supporting `in` and [col].astype(str)."""
    def __getitem__(self, k):
        v = dict.__getitem__(self, k)

        class _Col:
            def astype(self, _):
                return self

            def to_numpy(self):
                return np.asarray(v, dtype=object)
        return _Col()


class _StubAdata:
    def __init__(self, X, coords, cls, names):
        self.X = X
        self.obsm = {"spatial": coords}
        self.obs = _StubObs({"cell_class": cls})
        self.var_names = [f"g{i}" for i in range(X.shape[1])]
        self.obs_names = names
        self.n_obs = X.shape[0]


class _StubAd:
    def __init__(self, adata):
        self._a = adata

    def read_h5ad(self, *_a, **_k):
        return self._a


class _Args:
    use_obsm = None
    expression_mode = "silver_raw"
    coord_frame = "isotropic"


n = 25
X = rng.random((n, 4), dtype=np.float64).astype(np.float32)
coords = np.column_stack([rng.uniform(0, 100, n), rng.uniform(0, 50, n)])
cls = np.array([f"t{i%3}" for i in range(n)], dtype=object)
names = np.array([f"cell{i}" for i in range(n)], dtype=object)
stub = _StubAd(_StubAdata(X, coords, cls, names))
feats, ct, cl, on, vn = R.load_query(stub, Path("x_test.h5ad"), _Args())
check(np.array_equal(feats, X), "load_query returns features unpermuted")
check(np.array_equal(ct, coords), "load_query returns truth coords unpermuted")
check(list(cl) == list(cls.astype(str)), "load_query returns labels in row order")
check(list(on) == list(names), "load_query returns obs_names in row order")
check(len(vn) == X.shape[1], "load_query returns the feature names")

# cell_class is mandatory: without it the scorer yields NaN rather than failing
stub_nocls = _StubAd(_StubAdata(X, coords, cls, names))
stub_nocls._a.obs = _StubObs({})
expect_raises(lambda: R.load_query(stub_nocls, Path("x_test.h5ad"), _Args()),
              ValueError, "missing obs['cell_class'] is refused")

# write_slice_artifacts must key both files on the SAME index so a permutation
# cannot hide: the scorer aligns on that index.
with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "slice"
    pp, pt = write_slice_artifacts(d, coords + 1.0, coords, cls, index=on)
    ip = [l.split(",")[0] for l in pp.read_text().splitlines()[1:]]
    it = [l.split(",")[0] for l in pt.read_text().splitlines()[1:]]
    check(ip == it == list(names),
          "pred and truth CSVs share the real obs_names index (joinable per cell)")
    hdr = pt.read_text().splitlines()[0]
    check(hdr == ",coord_X,coord_Y,cell_class",
          "CSV header matches CeLEry's schema", hdr)

# ---------------------------------------------------------------------------
print("\n[8] --use_obsm spatial must be refused (it would feed ground truth)")


class _A2(_Args):
    use_obsm = "spatial"


expect_raises(lambda: R._feature_matrix(_StubAdata(X, coords, cls, names),
                                        "spatial", "silver_raw"),
              ValueError, "_feature_matrix refuses use_obsm='spatial'")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
    raise SystemExit(1)
print("ALL WRAPPER-FIX TESTS PASSED")
