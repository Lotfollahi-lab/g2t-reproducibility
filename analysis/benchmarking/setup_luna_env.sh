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
#
# pytorch_lightning is pinned to <2.5 because LUNA's DataModule does NOT
# inherit from pl.LightningDataModule directly. PL >=2.5 introduced a
# stricter check in `is_overridden` that raises `ValueError: Expected a
# parent` when the datamodule isn't a proper subclass. The 2.0–2.4 range
# uses a more permissive check that LUNA's code path works against.
#
# We deliberately do NOT install the `lightning` meta-package — it pins
# the unified Lightning ecosystem to the latest, which would yank
# pytorch_lightning back to 2.5+ via transitive deps. Pin
# torchmetrics too since some 1.x ↔ 2.x compatibility breaks Lightning's
# logging when versions drift.
log "installing torch_geometric, pytorch_lightning, scanpy, hydra, et al..."
"${UV_PIP[@]}" \
    torch_geometric \
    "pytorch_lightning>=2.0,<2.5" \
    "torchmetrics>=1.0,<1.5" \
    scanpy \
    wandb \
    colorcet \
    squidpy \
    hydra-core \
    omegaconf \
    linear_attention_transformer

# 3d. scipy. LUNA's README requested 1.9.1 for an RSSD-metric quirk, but
#     1.9.1 has no Python 3.10 wheels. scipy 1.10.x is the sweet spot:
#       * has Py3.10 wheels (1.9.1 doesn't)
#       * still has the permissive `scipy.spatial.transform.Rotation
#         .align_vectors(a, b, return_sensitivity=True)` API that LUNA's
#         `metrics/evaluation_statistics.compute_kabsch_rotation` uses.
#     scipy 1.11+ added a stricter check that raises
#         ValueError: Cannot return sensitivity matrix with an infinite
#         weight or one vector pair
#     when a per-cell-class alignment has only one vector (which happens
#     during LUNA's per-class RSSD on test slices that contain a rare
#     class with a single cell). Pinning to <1.11 keeps the original
#     LUNA-1.9.1 behaviour.
log "installing scipy>=1.10,<1.11 (>=1.10 for Py3.10 wheels, <1.11 to "
log "  avoid the align_vectors 'one vector pair' regression LUNA hits)..."
"${UV_PIP[@]}" "scipy>=1.10,<1.11"

# 3e. Bridge utilities we use from the scgg-side LUNA wrapper scripts.
log "installing pandas / anndata / h5py / matplotlib..."
"${UV_PIP[@]}" pandas anndata h5py matplotlib

# 3f. Uninstall the `lightning` meta-package if any transitive dep dragged
#     it in (scanpy / squidpy / torch_geometric all sometimes do). Why:
#     torch_geometric's LightningDataset has a try/except import:
#         try: from lightning.pytorch import LightningDataModule  # if `lightning` is present
#         except ImportError: from pytorch_lightning import LightningDataModule
#     If `lightning` is installed, the FIRST branch runs and LUNA's
#     DataModule ends up inheriting from `lightning.pytorch.LightningDataModule`.
#     But LUNA's Trainer uses `pytorch_lightning.Trainer`, which checks
#     `isinstance(instance, pytorch_lightning.LightningDataModule)`. These
#     are two distinct classes in different namespaces — the isinstance
#     check returns False and `is_overridden` raises:
#         ValueError: Expected a parent
#     Removing `lightning` forces the except branch in torch_geometric, so
#     everyone agrees on the same `pytorch_lightning.LightningDataModule`.
log "removing 'lightning' meta-package if present (forces "
log "  torch_geometric's LightningDataset to use pytorch_lightning's base)..."
uv pip uninstall --python "$VENV_DIR/bin/python" lightning 2>/dev/null || true
uv pip uninstall --python "$VENV_DIR/bin/python" lightning-fabric 2>/dev/null || true
uv pip uninstall --python "$VENV_DIR/bin/python" lightning-utilities 2>/dev/null || true
# Reinstall lightning-utilities — pytorch_lightning needs it directly.
"${UV_PIP[@]}" "lightning-utilities>=0.10"

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
import pytorch_lightning as pl; print("pl        :", pl.__version__)
import torchmetrics; print("torchmetrics:", torchmetrics.__version__)
import scanpy; print("scanpy    :", scanpy.__version__)
import hydra; print("hydra     :", hydra.__version__)
import scipy; print("scipy     :", scipy.__version__)
import linear_attention_transformer; print("linear_attn: OK")

# The actual root-cause check for the "Expected a parent" error LUNA hit:
# torch_geometric's LightningDataset must inherit from THIS env's
# `pytorch_lightning.LightningDataModule`, not from the `lightning`
# meta-package's version. If both packages are installed, torch_geometric
# picks `lightning.pytorch.LightningDataModule`, and LUNA's Trainer (which
# uses `pytorch_lightning.Trainer`) hits an isinstance mismatch.
from torch_geometric.data.lightning import LightningDataset as _PyGLightningDS
from pytorch_lightning import LightningDataModule as _PLLDM
mro_modules = [c.__module__ + "." + c.__name__ for c in _PyGLightningDS.__mro__]
print("pyg.LightningDataset MRO[:4]:")
for m in mro_modules[:4]:
    print(f"    {m}")
if any("lightning.pytorch" in m for m in mro_modules):
    print("ERROR: torch_geometric.LightningDataset inherits from "
          "`lightning.pytorch.LightningDataModule`, but LUNA's Trainer "
          "uses `pytorch_lightning.LightningDataModule`. Mismatch will "
          "cause `ValueError: Expected a parent` at trainer.fit() time.")
    print("  Fix: remove the `lightning` meta-package from this venv.")
    raise SystemExit(1)
if _PLLDM not in _PyGLightningDS.__mro__:
    print("ERROR: torch_geometric.LightningDataset does not inherit from "
          "pytorch_lightning.LightningDataModule at all. is_overridden "
          "will fail.")
    raise SystemExit(1)
print("namespace check: OK — pyg & pytorch_lightning agree on the same "
      "LightningDataModule class")

# scipy Rotation.align_vectors API check. LUNA's test phase computes
# per-cell-class Kabsch alignments, which fail when a rare class has
# only one cell. scipy 1.11+ raises:
#   ValueError: Cannot return sensitivity matrix with an infinite weight
#   or one vector pair
# We pinned scipy<1.11 to dodge this, but verify the call actually works.
from scipy.spatial.transform import Rotation
import numpy as np
try:
    a = np.array([[1.0, 0.0, 0.0]])  # ONE vector pair (the failure case)
    b = np.array([[0.0, 1.0, 0.0]])
    _, _, _ = Rotation.align_vectors(a, b, return_sensitivity=True)
    print("scipy Rotation.align_vectors(1 pair, sensitivity=True): OK")
except ValueError as e:
    print(f"scipy Rotation.align_vectors(1 pair): FAILED ({e})")
    print("  LUNA's per-class RSSD computation will crash on test slices "
          "containing a rare class with a single cell. Pin scipy < 1.11.")
    raise SystemExit(1)
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
