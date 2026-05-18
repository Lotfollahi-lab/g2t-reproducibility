#!/usr/bin/env bash
# Set up an isolated environment to run LUNA from
# https://github.com/mlbio-epfl/LUNA.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/luna     uv-managed venv
#   LUNA_CODE_DIR   = /nfs/team361/sb75/code/LUNA       LUNA repo clone
#   PYTHON_VERSION  = 3.9                               per LUNA README
#
# Notes:
#   * LUNA requires Python 3.9 (their README). We use uv to materialize that
#     interpreter independently of system Python.
#   * The pinned PyTorch is 2.0.1 + cu118. NVIDIA driver 12.4 (your cluster)
#     is backward-compatible with CUDA 11.8 runtimes, so this works on your
#     hardware even though your driver is newer.
#   * scipy is pinned to 1.9.1 because LUNA's README warns about RSSD-metric
#     incompatibilities with newer scipy versions.
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_luna_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/luna}"
LUNA_CODE_DIR="${LUNA_CODE_DIR:-/nfs/team361/sb75/code/LUNA}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
LUNA_GIT_URL="${LUNA_GIT_URL:-https://github.com/mlbio-epfl/LUNA.git}"
LUNA_GIT_REF="${LUNA_GIT_REF:-main}"

log() { printf '[setup_luna_env] %s\n' "$*"; }

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
# 1. Create / refresh the uv venv with Python 3.9
# ---------------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    log "venv already exists: $VENV_DIR"
    log "  (delete this directory if you want a clean reinstall)"
else
    log "creating venv at $VENV_DIR (Python $PYTHON_VERSION)..."
    mkdir -p "$(dirname "$VENV_DIR")"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "active python: $(python --version) -> $(which python)"

# uv pip works against the active venv. Use --python to be explicit.
UV_PIP=(uv pip install --python "$VENV_DIR/bin/python")

# ---------------------------------------------------------------------------
# 2. Clone (or update) the LUNA repository
# ---------------------------------------------------------------------------
if [[ -d "$LUNA_CODE_DIR/.git" ]]; then
    log "LUNA repo present at $LUNA_CODE_DIR; pulling latest..."
    (cd "$LUNA_CODE_DIR" && git fetch && git checkout "$LUNA_GIT_REF" && git pull --ff-only)
else
    log "cloning LUNA into $LUNA_CODE_DIR..."
    mkdir -p "$(dirname "$LUNA_CODE_DIR")"
    git clone "$LUNA_GIT_URL" "$LUNA_CODE_DIR"
    (cd "$LUNA_CODE_DIR" && git checkout "$LUNA_GIT_REF")
fi

# ---------------------------------------------------------------------------
# 3. Install LUNA's pinned dependencies (verbatim from their README)
# ---------------------------------------------------------------------------
# 3a. PyTorch + CUDA 11.8 wheels
log "installing torch 2.0.1 / torchvision 0.15.2 / torchaudio 2.0.2 (cu118)..."
"${UV_PIP[@]}" \
    "torch==2.0.1" "torchvision==0.15.2" "torchaudio==2.0.2" \
    --index-url https://download.pytorch.org/whl/cu118

# 3b. PyG extensions (pyg_lib / torch_scatter / ... )
log "installing PyG extensions..."
"${UV_PIP[@]}" \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 3c. The rest (torch_geometric pulls itself in)
log "installing torch_geometric, lightning, scanpy, hydra, et al..."
"${UV_PIP[@]}" \
    torch_geometric \
    lightning \
    scanpy \
    wandb \
    colorcet \
    squidpy \
    hydra-core \
    linear_attention_transformer

# 3d. scipy 1.9.1 pin per LUNA's README (avoids RSSD-metric incompatibility)
log "pinning scipy==1.9.1 (LUNA README requirement for RSSD)..."
"${UV_PIP[@]}" "scipy==1.9.1"

# 3e. Tiny extras useful for the benchmark adapter
"${UV_PIP[@]}" pandas anndata h5py

# ---------------------------------------------------------------------------
# 4. Smoke check: import everything LUNA needs
# ---------------------------------------------------------------------------
log "smoke-checking the environment..."
"$VENV_DIR/bin/python" - <<'PY'
import sys
print("python   :", sys.version.split()[0])
import torch;             print("torch    :", torch.__version__, "cuda?", torch.cuda.is_available())
import torchvision;       print("torchvision:", torchvision.__version__)
import torch_geometric;   print("pyg      :", torch_geometric.__version__)
import lightning;         print("lightning:", lightning.__version__)
import scanpy;            print("scanpy   :", scanpy.__version__)
import hydra;             print("hydra    :", hydra.__version__)
import scipy;             print("scipy    :", scipy.__version__)
import linear_attention_transformer; print("linear_attn: OK")
PY

log ""
log "DONE."
log "  venv      : $VENV_DIR"
log "  LUNA repo : $LUNA_CODE_DIR"
log ""
log "To run LUNA training directly:"
log "    source $VENV_DIR/bin/activate"
log "    cd $LUNA_CODE_DIR"
log "    python main.py general.name=... dataset.train_data_path=... dataset.test_data_path=..."
