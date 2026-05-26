#!/usr/bin/env bash
# Set up an isolated uv environment for the novosparc baseline.
#
# novosparc (Nitzan et al. 2019; Moriel et al. 2021) solves the
# scRNA-seq → spatial-position task via optimal transport. It's
# considerably lighter than LUNA / scgg: pure-Python numpy + scipy +
# POT, no torch, no pyg, no Lightning. So this env is small.
#
# We keep it in its OWN venv (not the scgg one) because: (a) novosparc
# has no torch / pyg dependency so dragging it into the scgg env is
# wasteful; (b) running each baseline in its dedicated venv keeps
# their dependency graphs from interfering, which is going to matter
# once we also wire up CeLEry / STALocator / scSpace.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/novosparc   uv-managed venv
#   PYTHON_VERSION  = 3.10
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_novosparc_env.sh
#
# Rebuild from scratch:
#   rm -rf /nfs/team361/sb75/.venvs/novosparc
#   bash scgg-reproducibility/analysis/benchmarking/setup_novosparc_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/novosparc}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Pinned novosparc version. Latest as of writing is 0.4.4 (PyPI). Pin
# explicitly so a future PyPI release that changes the Tissue API
# doesn't silently break the pipeline. Bump after re-testing.
NOVOSPARC_VERSION="${NOVOSPARC_VERSION:-0.4.4}"
# Python Optimal Transport — novosparc's OT solver dependency. Pin
# loosely to a known-compatible major (>=0.9 introduced a few API
# tweaks novosparc already accommodates).
POT_VERSION="${POT_VERSION:->=0.9,<1.0}"

log() { printf '[setup_novosparc_env] %s\n' "$*"; }

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

# uv pip is explicit about which interpreter to install into; safer
# than relying on shell activation when this script is invoked from
# subshells / makefiles.
UV_PIP=(uv pip install --python "$VENV_DIR/bin/python")

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
# novosparc itself; pinned.
log "installing novosparc==$NOVOSPARC_VERSION ..."
"${UV_PIP[@]}" "novosparc==${NOVOSPARC_VERSION}"

# Python Optimal Transport. novosparc declares POT as a dep but
# doesn't pin a version. Pin here to a major we know works.
log "installing POT $POT_VERSION ..."
"${UV_PIP[@]}" "POT${POT_VERSION}"

# Data IO + scientific stack. scanpy is needed because the silver
# layer is in h5ad; novosparc itself doesn't require it but our
# pipeline does.
log "installing scanpy / anndata / scipy / numpy / pandas / h5py ..."
"${UV_PIP[@]}" \
    "scanpy>=1.9" \
    "anndata>=0.9" \
    "scipy>=1.10,<1.11" \
    "numpy>=1.22" \
    "pandas>=1.5" \
    "h5py>=3.7"

# Plot + logging tooling for parity with the LUNA/scgg pipelines:
# matplotlib for the GT-vs-prediction plots, pyyaml for the
# config.yaml snapshot, wandb for run-level metrics logging, psutil
# for the _RuntimeTracker's process-tree RSS sampling (peak_rss_mib
# in runtime.csv).
log "installing matplotlib / pyyaml / wandb / psutil ..."
"${UV_PIP[@]}" \
    "matplotlib>=3.5" \
    "pyyaml>=6.0" \
    "wandb>=0.15" \
    "psutil>=5.9"

# scipy <1.11 to match the LUNA/scgg envs: 1.11+ tightened
# Rotation.align_vectors's preconditions (rejects 1-pair inputs when
# return_sensitivity=True), which the per-cell-class Kabsch RSSD
# path in scgg/src/scgg/evaluation/luna_metrics.py exercises. We
# pin here so the novosparc env produces metric numbers byte-equal
# to LUNA's / scgg's on the same predictions.

# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------
log "smoke-checking the environment..."
"$VENV_DIR/bin/python" - <<'PY'
import sys
print("python   :", sys.version.split()[0])

import numpy as np
import scipy
import pandas as pd
import anndata
import scanpy
import matplotlib
print("numpy    :", np.__version__)
print("scipy    :", scipy.__version__)
print("pandas   :", pd.__version__)
print("anndata  :", anndata.__version__)
print("scanpy   :", scanpy.__version__)
print("matplotlib:", matplotlib.__version__)

import ot
print("POT      :", ot.__version__)

import novosparc
from novosparc.common._tissue import Tissue
print("novosparc:", novosparc.__version__)
print("  Tissue class importable: OK")

# Tiny end-to-end smoke test on a 20-cell × 10-gene synthetic
# tissue. If this trains without raising, the install is healthy.
np.random.seed(0)
n_cells, n_locations, n_genes = 20, 20, 10
locations = np.random.rand(n_locations, 2)
expr = np.random.rand(n_cells, n_genes)
atlas = np.random.rand(n_locations, n_genes)

adata = anndata.AnnData(X=expr)
tissue = Tissue(dataset=adata, locations=locations)
markers = np.arange(n_genes)
tissue.setup_reconstruction(
    markers_to_use=markers,
    atlas_matrix=atlas,
    num_neighbors_s=3,
    num_neighbors_t=3,
)
tissue.reconstruct(alpha_linear=0.5, epsilon=5e-4)
assert tissue.gw is not None, "novosparc reconstruct() did not populate gw"
print(f"end-to-end OT solve: OK (gw shape = {tissue.gw.shape})")

# scipy Rotation.align_vectors 1-pair test — same as the LUNA env.
from scipy.spatial.transform import Rotation
a = np.array([[1.0, 0.0, 0.0]])
b = np.array([[0.0, 1.0, 0.0]])
_, _, _ = Rotation.align_vectors(a, b, return_sensitivity=True)
print("scipy Rotation.align_vectors(1 pair, sensitivity=True): OK")
PY

log ""
log "DONE."
log "  venv : $VENV_DIR"
log ""
log "To run the novosparc benchmark on a silver h5ad dir:"
log "    source $VENV_DIR/bin/activate"
log "    python scgg/scripts/run_novosparc_pipeline.py \\"
log "        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\"
log "        --wandb_run_name novosparc_mmc_baseline"
