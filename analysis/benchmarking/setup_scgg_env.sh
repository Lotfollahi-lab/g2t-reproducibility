#!/usr/bin/env bash
# Set up an isolated uv environment to run scgg from this monorepo on a
# host with CUDA driver 12.4. scgg currently vendors the LUNA codebase
# verbatim under scgg/src/{models,utils,metrics,datasets,configs}, so
# this env mirrors the LUNA env exactly (same torch / pyg / lightning
# pins, same scipy quirks) plus an editable install of the scgg
# package itself.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/scgg     uv-managed venv
#   SCGG_CODE_DIR   = /nfs/team361/sb75/scgg            this monorepo
#   PYTHON_VERSION  = 3.10
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_scgg_env.sh
#
# To rebuild from scratch (e.g. after this script changes):
#   rm -rf /nfs/team361/sb75/.venvs/scgg
#   bash scgg-reproducibility/analysis/benchmarking/setup_scgg_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/scgg}"
SCGG_CODE_DIR="${SCGG_CODE_DIR:-/nfs/team361/sb75/scgg}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Torch / CUDA pin. Kept identical to setup_luna_env.sh so a scgg run
# is bit-equivalent to the upstream LUNA baseline when nothing under
# scgg/src/ has been edited yet.
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.0}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu121}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"

log() { printf '[setup_scgg_env] %s\n' "$*"; }

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
# Pre-flight: scgg monorepo must exist
# ---------------------------------------------------------------------------
if [[ ! -f "$SCGG_CODE_DIR/pyproject.toml" ]]; then
    cat >&2 <<EOF
ERROR: SCGG_CODE_DIR=$SCGG_CODE_DIR does not look like the scgg monorepo.
Expected pyproject.toml at: $SCGG_CODE_DIR/pyproject.toml

Either clone the repo to that path or override SCGG_CODE_DIR:
    SCGG_CODE_DIR=/path/to/scgg bash $0
EOF
    exit 1
fi

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
# 2. Install scgg's dependencies (same pins as setup_luna_env.sh)
# ---------------------------------------------------------------------------
# 2a. PyTorch + matching CUDA wheels.
log "installing torch $TORCH_VERSION / torchvision $TORCHVISION_VERSION / "
log "  torchaudio $TORCHAUDIO_VERSION ($TORCH_CUDA_TAG)..."
"${UV_PIP[@]}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# 2b. PyG extensions — must match the torch + CUDA tag we just installed.
log "installing PyG extensions from ${PYG_WHL_URL}..."
"${UV_PIP[@]}" \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "${PYG_WHL_URL}"

# 2c. The rest. torch_geometric pulls itself in.
#
# pytorch_lightning is pinned to <2.5 because LUNA's DataModule does NOT
# inherit from pl.LightningDataModule directly. PL >=2.5 introduced a
# stricter check in `is_overridden` that raises `ValueError: Expected a
# parent` when the datamodule isn't a proper subclass.
#
# We deliberately do NOT install the `lightning` meta-package — it
# yanks pytorch_lightning to the latest. Pin torchmetrics too since
# some 1.x ↔ 2.x compatibility breaks Lightning's logging.
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

# 2d. scipy. <1.11 to keep LUNA's per-class Kabsch RSSD working when a
#     rare class has only one cell (1.11+ raises on the 1-pair case).
log "installing scipy>=1.10,<1.11 (>=1.10 for Py3.10 wheels, <1.11 to "
log "  avoid the align_vectors 'one vector pair' regression LUNA hits)..."
"${UV_PIP[@]}" "scipy>=1.10,<1.11"

# 2e. Bridge utilities used by the scgg-side wrapper scripts.
log "installing pandas / anndata / h5py / matplotlib..."
"${UV_PIP[@]}" pandas anndata h5py matplotlib

# 2f. Uninstall the `lightning` meta-package if any transitive dep
#     dragged it in (scanpy / squidpy / torch_geometric all sometimes
#     do). Why: torch_geometric's LightningDataset has a try/except
#     import where, if `lightning` is present, it inherits from
#     `lightning.pytorch.LightningDataModule` — but LUNA's Trainer is
#     `pytorch_lightning.Trainer`. The isinstance check fails and
#     `is_overridden` raises `ValueError: Expected a parent` at
#     trainer.fit() time. Removing `lightning` forces the except
#     branch in pyg, so everyone agrees on the same base class.
log "removing 'lightning' meta-package if present (forces "
log "  torch_geometric's LightningDataset to use pytorch_lightning's base)..."
uv pip uninstall --python "$VENV_DIR/bin/python" lightning 2>/dev/null || true
uv pip uninstall --python "$VENV_DIR/bin/python" lightning-fabric 2>/dev/null || true
uv pip uninstall --python "$VENV_DIR/bin/python" lightning-utilities 2>/dev/null || true
# Reinstall lightning-utilities — pytorch_lightning needs it directly.
"${UV_PIP[@]}" "lightning-utilities>=0.10"

# ---------------------------------------------------------------------------
# 3. Install scgg in editable mode (after deps, so resolution is stable)
# ---------------------------------------------------------------------------
# We install with --no-deps because every dep is already pinned above —
# letting pip re-resolve from scgg's pyproject would override our pins
# (e.g. yank pytorch_lightning back to >=2.5 via transitive resolution).
#
# Nuke any stale egg-info first: a previous install can cache a package
# list that doesn't include packages added since (or, worse, includes
# packages we've removed). On the next install setuptools rebuilds it
# from scratch using the current pyproject.
log "removing stale egg-info under $SCGG_CODE_DIR/src/ (if any)..."
rm -rf "$SCGG_CODE_DIR"/src/*.egg-info "$SCGG_CODE_DIR"/*.egg-info 2>/dev/null || true

log "installing scgg (editable) from $SCGG_CODE_DIR..."
uv pip install --python "$VENV_DIR/bin/python" --no-deps -e "$SCGG_CODE_DIR"

# ---------------------------------------------------------------------------
# 4. Smoke check: import everything scgg needs and confirm CUDA works
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
        a = torch.randn(8, device=dev)
        b = torch.randn(8, device=dev)
        _ = (a + b).sum().item()
        print("cuda smoke: PASSED (tensor add on GPU returned a finite scalar)")
    except Exception as e:
        print(f"cuda smoke: FAILED ({type(e).__name__}: {e})")
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

# scgg package itself + its vendored LUNA top-level modules.
import scgg; print("scgg     :", scgg.__version__, "->", scgg.__file__)
import models, utils, metrics, datasets, configs
import main, diffusion_model
print("scgg's vendored LUNA top-level modules: models, utils, metrics, "
      "datasets, configs, main, diffusion_model -> all import OK")

# Same is_overridden / pyg-vs-pl namespace check the LUNA env does.
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
          "pytorch_lightning.LightningDataModule at all.")
    raise SystemExit(1)
print("namespace check: OK — pyg & pytorch_lightning agree on the same "
      "LightningDataModule class")

# scipy Rotation.align_vectors 1-pair test (LUNA's per-class RSSD path).
from scipy.spatial.transform import Rotation
import numpy as np
try:
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    _, _, _ = Rotation.align_vectors(a, b, return_sensitivity=True)
    print("scipy Rotation.align_vectors(1 pair, sensitivity=True): OK")
except ValueError as e:
    print(f"scipy Rotation.align_vectors(1 pair): FAILED ({e})")
    raise SystemExit(1)
PY

log ""
log "DONE."
log "  venv      : $VENV_DIR"
log "  scgg code : $SCGG_CODE_DIR"
log ""
log "To run the scgg cortex benchmark (LUNA Figure 3 reproduction, with"
log "scgg's vendored copy of LUNA under the hood):"
log "    source $VENV_DIR/bin/activate"
log "    cd $SCGG_CODE_DIR"
log "    python scripts/run_luna_cortex_benchmark.py \\"
log "        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\"
log "        --wandb_run_name my_run"
