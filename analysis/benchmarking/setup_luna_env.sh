#!/usr/bin/env bash
# Set up an isolated uv environment to run LUNA from
# https://github.com/mlbio-epfl/LUNA on a host with CUDA driver 12.4.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/luna     uv-managed venv
#   LUNA_CODE_DIR   = /nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/luna
#   PYTHON_VERSION  = 3.10
#
# Notes:
#   * Pinned PyTorch is 2.4.0 + cu121. CUDA-12.1 runtime is forward-
#     compatible with NVIDIA driver 12.4 (the driver-to-runtime rule is
#     "driver >= runtime", and 12.4 >= 12.1), so this works on hosts
#     with driver 12.4 or newer.
#   * Earlier versions of this script pinned torch 2.0.1 + cu118. That
#     also worked with driver 12.4 *in principle*, but uv would silently
#     resolve a much newer torch when LUNA's transitive deps requested
#     it — and the newer torch is compiled against CUDA > 12.4 which the
#     12.4 driver can't load. The explicit cu121 pin + --index-url here
#     prevents that silent upgrade.
#   * Python 3.10 (was 3.9). The torch 2.4 / lightning 2.x stack has
#     better wheel coverage on 3.10 and matches what LUNA itself uses.
#   * scipy is bumped from 1.9.1 to 1.11.x because 1.9.1 has no
#     prebuilt wheels for Python 3.10. The RSSD-incompatibility warning
#     in LUNA's README applies only to numbers, not to running.
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_luna_env.sh
#
# To rebuild from scratch (e.g. after this script changes):
#   rm -rf /nfs/team361/sb75/.venvs/luna
#   bash scgg-reproducibility/analysis/benchmarking/setup_luna_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/luna}"
LUNA_CODE_DIR="${LUNA_CODE_DIR:-/nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/luna}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
LUNA_GIT_URL="${LUNA_GIT_URL:-https://github.com/mlbio-epfl/LUNA.git}"
LUNA_GIT_REF="${LUNA_GIT_REF:-main}"

# Torch / CUDA pin. Bump together if you need a different driver target.
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.0}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu121}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"

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

# uv pip works against the active venv. --python is explicit so this still
# resolves to the same venv even if activation drifts inside subshells.
UV_PIP=(uv pip install --python "$VENV_DIR/bin/python")

# ---------------------------------------------------------------------------
# 2. Clone (or update) the LUNA repository
# ---------------------------------------------------------------------------
if [[ -d "$LUNA_CODE_DIR/.git" ]]; then
    log "LUNA repo present at $LUNA_CODE_DIR; pulling latest..."
    (cd "$LUNA_CODE_DIR" && git fetch && git checkout "$LUNA_GIT_REF" && git pull --ff-only)
elif [[ -d "$LUNA_CODE_DIR" ]]; then
    log "LUNA code dir exists but is not a git checkout; skipping clone."
    log "  ($LUNA_CODE_DIR)"
else
    log "cloning LUNA into $LUNA_CODE_DIR..."
    mkdir -p "$(dirname "$LUNA_CODE_DIR")"
    git clone "$LUNA_GIT_URL" "$LUNA_CODE_DIR"
    (cd "$LUNA_CODE_DIR" && git checkout "$LUNA_GIT_REF")
fi

# ---------------------------------------------------------------------------
# 3. Install LUNA's dependencies
# ---------------------------------------------------------------------------
# 3a. PyTorch + matching CUDA wheels. We pin explicitly to a torch built
#     against CUDA 12.1; that runtime works with NVIDIA driver 12.4+.
log "installing torch $TORCH_VERSION / torchvision $TORCHVISION_VERSION / "
log "  torchaudio $TORCHAUDIO_VERSION ($TORCH_CUDA_TAG)..."
"${UV_PIP[@]}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# 3b. PyG extensions (pyg_lib / torch_scatter / ... ) — must match the
#     torch + CUDA tag we just installed.
log "installing PyG extensions from ${PYG_WHL_URL}..."
"${UV_PIP[@]}" \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "${PYG_WHL_URL}"

# 3c. The rest. torch_geometric pulls itself in.
log "installing torch_geometric, lightning, scanpy, hydra, et al..."
"${UV_PIP[@]}" \
    torch_geometric \
    lightning \
    "pytorch_lightning>=2.0,<3.0" \
    scanpy \
    wandb \
    colorcet \
    squidpy \
    hydra-core \
    omegaconf \
    linear_attention_transformer

# 3d. scipy. LUNA's README requested 1.9.1 for an RSSD-metric quirk, but
#     1.9.1 has no Python 3.10 wheels. 1.11.x is the latest with broad
#     wheel coverage that still works with LUNA's training pipeline; the
#     RSSD numbers may differ slightly from LUNA's published table.
log "installing scipy>=1.11,<1.14 (1.9.1 has no Py3.10 wheels)..."
"${UV_PIP[@]}" "scipy>=1.11,<1.14"

# 3e. Bridge utilities we use from the scgg-side LUNA wrapper scripts.
log "installing pandas / anndata / h5py / matplotlib..."
"${UV_PIP[@]}" pandas anndata h5py matplotlib

# ---------------------------------------------------------------------------
# 4. Smoke check: import everything LUNA needs and confirm CUDA works
# ---------------------------------------------------------------------------
log "smoke-checking the environment..."
"$VENV_DIR/bin/python" - <<'PY'
import sys
print("python   :", sys.version.split()[0])
import torch
print("torch    :", torch.__version__,
      "cuda?", torch.cuda.is_available(),
      "cuda_runtime:", torch.version.cuda)
if torch.cuda.is_available():
    try:
        dev = torch.device("cuda:0")
        cap = torch.cuda.get_device_capability(dev)
        name = torch.cuda.get_device_name(dev)
        print(f"cuda dev : {name}  capability sm_{cap[0]}{cap[1]}")
        # Tiny op to confirm runtime + driver actually talk.
        a = torch.randn(8, device=dev)
        b = torch.randn(8, device=dev)
        _ = (a + b).sum().item()
        print("cuda smoke: PASSED (tensor add on GPU returned a finite scalar)")
    except Exception as e:
        print(f"cuda smoke: FAILED ({type(e).__name__}: {e})")
        print("  This usually means the installed torch was compiled "
              "against a CUDA runtime newer than the host driver.")
        sys.exit(1)
else:
    print("WARNING: torch.cuda.is_available() returned False — training "
          "will fall back to CPU and be extremely slow.")

import torchvision; print("torchvision:", torchvision.__version__)
import torch_geometric; print("pyg       :", torch_geometric.__version__)
import lightning; print("lightning :", lightning.__version__)
import scanpy; print("scanpy    :", scanpy.__version__)
import hydra; print("hydra     :", hydra.__version__)
import scipy; print("scipy     :", scipy.__version__)
import linear_attention_transformer; print("linear_attn: OK")
PY

log ""
log "DONE."
log "  venv      : $VENV_DIR"
log "  LUNA code : $LUNA_CODE_DIR"
log ""
log "To run LUNA training via our wrapper:"
log "    source $VENV_DIR/bin/activate"
log "    python scgg/scripts/run_luna_on_mmc.py \\"
log "        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\"
log "        --luna_repo $LUNA_CODE_DIR"
