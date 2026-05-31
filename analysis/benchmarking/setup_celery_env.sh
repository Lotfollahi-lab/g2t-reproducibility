#!/usr/bin/env bash
# Set up an isolated uv environment for the CeLEry baseline.
#
# CeLEry (Zhang et al. 2023, Nat Commun 14:4050) is the supervised
# coordinate-regression baseline LUNA benchmarks against in Fig 3.
# It's a small MLP that maps cells×genes → (x, y) ∈ [0,1]² with a
# sigmoid output, trained per-reference-slice with MSE.
#
# Why a dedicated venv:
#  * Package name on PyPI is ``CeLEryPy`` but the import name is
#    ``CeLEry`` — easy to mis-pin, so we install once into a clean
#    env that we control.
#  * CeLEry's checkpoints are whole-object pickles (not state_dict),
#    so torch version drift breaks loads. Pinning torch in this env
#    keeps the checkpoints reproducible.
#  * CeLEry doesn't move tensors to CUDA in its public API — it's
#    CPU-only by default. We don't need pytorch+cuda; a minimal CPU
#    torch is fine and avoids dragging in the scgg env's heavy
#    GPU stack.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/celery   uv-managed venv
#   PYTHON_VERSION  = 3.10
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_celery_env.sh
#
# Rebuild from scratch:
#   rm -rf /nfs/team361/sb75/.venvs/celery
#   bash scgg-reproducibility/analysis/benchmarking/setup_celery_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/celery}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# CeLEry version. Latest as of writing is 1.1.3 on PyPI (CeLEryPy).
# Pin loosely to a known-good major. If a future release breaks the
# Fit_cord / Predict_cord signature this script depends on, bump
# after re-testing.
CELERY_VERSION="${CELERY_VERSION:-1.1.3}"

# Torch version. CeLEry pickles whole-model objects (not state_dicts),
# so the load-time torch version must agree with the save-time
# torch version. Pin to a recent CPU-only torch so we get the
# wheel-only install (no CUDA compile-from-source). 2.2 works with
# Python 3.10 + scanpy.
TORCH_VERSION="${TORCH_VERSION:-2.2.*}"

log() { printf '[setup_celery_env] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight: uv must be on PATH
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: uv is not installed.
Install it once via:
    curl -LsSf https://astral.sh/uv/install.sh | sh
Then re-run this script.
EOF
    exit 1
fi
log "uv $(uv --version)"

# ---------------------------------------------------------------------------
# 1. Create / refresh the uv venv with the pinned Python
# ---------------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    log "venv already exists: $VENV_DIR"
    log "  (delete this directory if you want a clean reinstall:"
    log "     rm -rf $VENV_DIR  &&  bash $0  )"
else
    log "creating venv at $VENV_DIR (Python $PYTHON_VERSION)..."
    mkdir -p "$(dirname "$VENV_DIR")"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "active python: $(python --version) -> $(which python)"

UV_PIP=(uv pip install --python "$VENV_DIR/bin/python")

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
# Torch first, CPU-only build. CeLEry pulls in torch as a dep but
# doesn't pin a version; installing it up-front pins the exact one
# we use across all CeLEry runs.
log "installing torch (CPU build) ==$TORCH_VERSION ..."
"${UV_PIP[@]}" --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"

# Install scikit-learn BEFORE CeLEry. Reason: CeLEryPy's setup.py
# declares the deprecated ``sklearn`` PyPI shim as a dependency, not
# the modern ``scikit-learn`` name. The shim package's own setup.py
# refuses to install (it just prints "use scikit-learn instead")
# UNLESS the SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# env var is set at install time. We set that env var below for the
# CeLEry install AND pre-install scikit-learn here so the shim's
# transitive resolution finds an already-installed scikit-learn
# instead of pulling in a second copy. This is the cleanest workaround
# until CeLEry updates its setup.py.
log "installing scikit-learn (pre-CeLEry; works around CeLEry's deprecated sklearn shim)..."
"${UV_PIP[@]}" "scikit-learn>=1.2"

# CeLEry itself. PyPI name is ``CeLEryPy``; import as ``CeLEry``.
# Set the deprecation-bypass env var so the sklearn-pypi-package
# shim that CeLEry depends on can install.
log "installing CeLEryPy==$CELERY_VERSION ..."
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    "${UV_PIP[@]}" "CeLEryPy==${CELERY_VERSION}"

# Data IO + scientific stack. Same versions as the other envs so the
# metric computation reads exactly the same h5ads.
log "installing scanpy / anndata / scipy / numpy / pandas / h5py ..."
"${UV_PIP[@]}" \
    "scanpy>=1.9" \
    "anndata>=0.9" \
    "scipy>=1.10,<1.11" \
    "numpy>=1.22" \
    "pandas>=1.5" \
    "h5py>=3.7"

# scipy <1.11 matches the LUNA / scgg / novosparc envs: 1.11+
# tightened Rotation.align_vectors's preconditions (rejects 1-pair
# inputs when return_sensitivity=True), which the per-cell-class
# Kabsch RSSD path in scgg/src/scgg/evaluation/luna_metrics.py
# exercises. We pin here so the celery env produces metric numbers
# byte-equal to LUNA's / scgg's on the same predictions.

# Plot + logging tooling for parity with the LUNA/scgg/novosparc
# pipelines: matplotlib for the GT-vs-prediction plots, pyyaml for
# the config.yaml snapshot, wandb for run-level metrics logging,
# psutil for the _RuntimeTracker's process-tree RSS sampling.
log "installing matplotlib / pyyaml / wandb / psutil ..."
"${UV_PIP[@]}" \
    "matplotlib>=3.5" \
    "pyyaml>=6.0" \
    "wandb>=0.15" \
    "psutil>=5.9"

# tqdm — CeLEry uses it for progress bars during training.
# (scikit-learn was installed earlier; see the CeLEry block above
# for the deprecated-shim workaround details.)
log "installing tqdm ..."
"${UV_PIP[@]}" "tqdm>=4.64"

# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------
log "smoke-checking the environment..."
"$VENV_DIR/bin/python" - <<'PY'
import sys
print("python    :", sys.version.split()[0])

import numpy as np
import scipy
import pandas as pd
import anndata
import scanpy
import matplotlib
print("numpy     :", np.__version__)
print("scipy     :", scipy.__version__)
print("pandas    :", pd.__version__)
print("anndata   :", anndata.__version__)
print("scanpy    :", scanpy.__version__)
print("matplotlib:", matplotlib.__version__)

import torch
print("torch     :", torch.__version__, "(cuda:", torch.cuda.is_available(), ")")

# CeLEry import smoke test — package is CeLEryPy on PyPI but
# imported as CeLEry. Verify all the public functions we'll
# actually call from the pipeline are importable.
import CeLEry as cel
print("CeLEry    :", getattr(cel, "__version__", "?"))
for fn in ("Fit_cord", "Predict_cord", "get_zscore"):
    assert hasattr(cel, fn), f"CeLEry missing {fn}"
    print(f"  cel.{fn} : OK")

# Tiny synthetic-data end-to-end smoke test: train a 1-epoch CeLEry
# model on 20 cells × 10 genes, then predict and confirm the output
# shape matches expectations. If this fails the env is broken.
print()
print("running synthetic smoke test (20 cells, 10 genes, 1 epoch)...")
import tempfile, os
n_cells, n_genes = 20, 10
rng = np.random.default_rng(42)
X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
# CeLEry reads coords positionally from the first two .obs columns,
# so we put them there explicitly.
obs = pd.DataFrame({
    "coord_X": rng.uniform(0, 100, n_cells),
    "coord_Y": rng.uniform(0, 100, n_cells),
}, index=[f"cell_{i}" for i in range(n_cells)])
var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
adata_ref = anndata.AnnData(X=X, obs=obs, var=var)
adata_qry = adata_ref.copy()  # query is the same data here

cel.get_zscore(adata_ref)
cel.get_zscore(adata_qry)

with tempfile.TemporaryDirectory() as tmp:
    # NB: CeLEry's DNN class hardcodes a 3-layer MLP — it indexes
    # hidden_dims[0], [1], [2] directly. Passing fewer than 3
    # widths raises IndexError. The pipeline default is the paper's
    # [30, 25, 15]; for the smoke test we just need 3 small ints.
    cel.Fit_cord(
        data_train=adata_ref,
        hidden_dims=[8, 8, 8],
        num_epochs_max=1,
        batch_size=4,
        num_workers=0,  # 0 to avoid multiproc on small/CI runs
        path=tmp,
        filename="smoke",
        seednum=42,
    )
    ckpt = os.path.join(tmp, "smoke.obj")
    assert os.path.exists(ckpt), f"checkpoint not written at {ckpt}"
    pred = cel.Predict_cord(
        data_test=adata_qry,
        path=tmp,
        filename="smoke",
    )
    assert hasattr(pred, "shape"), f"Predict_cord returned non-array: {type(pred)}"
    assert pred.shape == (n_cells, 2), f"unexpected pred shape: {pred.shape}"
    # Output is sigmoid-bounded.
    assert (pred >= 0).all() and (pred <= 1).all(), \
        f"pred out of [0,1] range: min={pred.min()}, max={pred.max()}"
    print("  smoke test passed: pred shape", pred.shape, "range", (pred.min(), pred.max()))
PY

log "done. venv at: $VENV_DIR"
log "activate with:  source $VENV_DIR/bin/activate"
