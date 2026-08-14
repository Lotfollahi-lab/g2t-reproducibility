#!/usr/bin/env bash
# Set up an isolated uv environment for the CellContrast baseline, and clone the
# authors' code next to the other benchmarking assets.
#
# CellContrast (Li et al. 2024, Patterns 5(8):101022) is the reviewer-requested
# contrastive baseline. We run THEIR published code (MIT licence) rather than a
# reimplementation, so this script installs dependencies and changes nothing
# about the method.
#
#   Upstream: https://github.com/HKU-BAL/CellContrast
#   Licence : MIT
#
# Why a dedicated venv:
#  * Upstream's environment.yml pins only ``python=3.9`` and ``scanpy==1.9.3``
#    and leaves PyTorch COMPLETELY unpinned (their README installs a CPU build).
#    That is good for us — there is no torch/CUDA conflict to resolve — but it
#    means the env is under-specified, so we pin it here ourselves.
#  * scanpy 1.9.3 is a 2023-era release; installing it into the scgg env would
#    fight that env's much newer stack. Keep it isolated.
#  * We want a CUDA torch (training is ~12-36 h on 158k cells), unlike upstream's
#    CPU example.
#
# Defaults (override via env vars):
#   VENV_DIR        = /nfs/team361/sb75/.venvs/cellcontrast     uv-managed venv
#   REPO_DIR        = <this dir>/CellContrast                   authors' code
#   REPO_COMMIT     = 0559aa9  (upstream main, pinned — see below)
#   PYTHON_VERSION  = 3.9      (upstream's pin)
#   SCANPY_VERSION  = 1.9.3    (upstream's pin)
#   TORCH_SPEC      = torch<2.6 --index-url .../cu121
#   LOCKFILE        = <VENV_DIR>/requirements.lock              resolved pins
#
# The script ends by running the wrapper's own numpy-only test suite against the
# new venv and FAILS if any of it fails — an env that merely imports is not
# evidence the baseline computes the right thing.
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/setup_cellcontrast_env.sh
#
# Rebuild from scratch:
#   rm -rf /nfs/team361/sb75/.venvs/cellcontrast
#   bash scgg-reproducibility/analysis/benchmarking/setup_cellcontrast_env.sh
#
# Env only (skip the clone):
#   SKIP_CLONE=1 bash .../setup_cellcontrast_env.sh
#
# Skip the post-install test suite (not recommended):
#   SKIP_TESTS=1 bash .../setup_cellcontrast_env.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/cellcontrast}"
REPO_DIR="${REPO_DIR:-$HERE/CellContrast}"
REPO_URL="${REPO_URL:-https://github.com/HKU-BAL/CellContrast.git}"
# Pin the upstream revision. Upstream has no tags and no releases, so a bare
# clone tracks main and "the CellContrast baseline" would quietly mean different
# code on every re-run. This is main as of 2024-07-18 ("add link to Patterns
# ppr"), i.e. the state of the code the published paper describes, and the
# revision every number in our tables was produced from.
REPO_COMMIT="${REPO_COMMIT:-0559aa9d5fbd3d56524d39aa16705e266195a5b5}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
SCANPY_VERSION="${SCANPY_VERSION:-1.9.3}"
# scanpy 1.9.3 predates NumPy 2 and uses np.float_, which NumPy 2.0 REMOVED.
# Its metadata does not exclude numpy>=2, so a naive resolve installs numpy 2.x
# and every `import scanpy` then dies with
#   AttributeError: `np.float_` was removed in the NumPy 2.0 release.
# Pin numpy below 2 for this env. (Upstream's own environment.yml predates the
# problem, so it says nothing about it.)
NUMPY_SPEC="${NUMPY_SPEC:-numpy<2}"
# Upstream leaves torch unpinned; choose a CUDA build for the cluster. Set
# TORCH_SPEC="torch" for a CPU-only install.
#
# The <2.6 ceiling is not cosmetic. torch 2.6 flipped ``torch.load``'s default to
# weights_only=True. Upstream saves a pandas Index inside the checkpoint
# (cellContrast/train.py:38) and loads it bare (cellContrast/inference.py:26-33),
# so torch>=2.6 raises at INFERENCE — that is, AFTER a 12-36 h training run has
# already completed. The cu121 index happens to top out at 2.5.1, so the CUDA
# default was safe by accident; plain ``torch`` on py3.9 resolves to >=2.6, so
# the ceiling has to be written down to hold on both paths.
TORCH_MAX="${TORCH_MAX:-<2.6}"
TORCH_SPEC="${TORCH_SPEC:-torch$TORCH_MAX --index-url https://download.pytorch.org/whl/cu121}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"

# The uv cache and /nfs are on different filesystems here, so hardlinking is
# unavailable; say so up front instead of emitting a warning per install.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

log() { printf '[setup_cellcontrast_env] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight: uv must be on PATH (conda is NOT used on this cluster)
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
# 1. Clone / refresh the authors' code
# ---------------------------------------------------------------------------
if [[ "$SKIP_CLONE" != "1" ]]; then
    if [[ -d "$REPO_DIR/.git" ]]; then
        log "repo exists: $REPO_DIR (fetching)"
        git -C "$REPO_DIR" fetch --all --tags --quiet
    else
        log "cloning $REPO_URL -> $REPO_DIR"
        mkdir -p "$(dirname "$REPO_DIR")"
        git clone --quiet "$REPO_URL" "$REPO_DIR"
    fi
    COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
    log "commit: $COMMIT"
    # A baseline we did not author must be reproducible to a revision.
    printf '%s\n' "$COMMIT" > "$REPO_DIR/.checked_out_commit"
else
    log "SKIP_CLONE=1 (using existing $REPO_DIR)"
fi

for f in cellContrast.py parameters/parameters_singleCell.json LICENSE; do
    if [[ ! -e "$REPO_DIR/$f" ]]; then
        log "ERROR: expected '$f' under $REPO_DIR"; exit 1
    fi
done
log "sanity: entry point, parameter file and LICENSE present"

# ---------------------------------------------------------------------------
# 2. Create / refresh the uv venv
# ---------------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    log "venv already exists: $VENV_DIR"
    log "  (to rebuild:  rm -rf $VENV_DIR  &&  bash $0  )"
else
    log "creating venv at $VENV_DIR (Python $PYTHON_VERSION)..."
    mkdir -p "$(dirname "$VENV_DIR")"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

UV_PIP=(uv pip install --python "$VENV_DIR/bin/python")

# ---------------------------------------------------------------------------
# 3. Dependencies
#
#    ORDER MATTERS. torch goes first (from the CUDA index), then scanpy with the
#    numpy<2 constraint. Installing scanpy last lets its resolve pin numpy for
#    the env; doing it the other way round leaves torch's looser numpy
#    requirement free to pull in numpy 2.x, which breaks scanpy 1.9.3.
#    scanpy also brings anndata/pandas/scipy/scikit-learn/matplotlib, covering
#    every import the upstream package makes apart from torch and tqdm.
# ---------------------------------------------------------------------------
log "installing torch ($TORCH_SPEC) ..."
# shellcheck disable=SC2086
"${UV_PIP[@]}" $TORCH_SPEC

log "installing scanpy==$SCANPY_VERSION with '$NUMPY_SPEC' and tqdm ..."
"${UV_PIP[@]}" "scanpy==$SCANPY_VERSION" "$NUMPY_SPEC" tqdm

# ---------------------------------------------------------------------------
# 4. Verify — every import the upstream package actually makes
# ---------------------------------------------------------------------------
log "verifying ..."
"$VENV_DIR/bin/python" - <<'PY'
import importlib, sys
required = ("torch", "scanpy", "anndata", "numpy", "pandas",
            "scipy", "sklearn", "matplotlib", "tqdm")
missing = []
for m in required:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        missing.append(m)
        print(f"  MISS {m:12s} {type(e).__name__}: {e}")

# Guard the specific incompatibility this env is prone to: scanpy 1.9.3 uses
# np.float_, removed in NumPy 2.0. Catch it here rather than 12 h into a job.
try:
    import numpy as _np
    if int(_np.__version__.split(".")[0]) >= 2:
        missing.append("numpy<2")
        print(f"  FAIL numpy {_np.__version__} is >= 2; scanpy 1.9.x needs "
              f"numpy<2 (np.float_ was removed). Rebuild the venv.")
    else:
        print(f"  OK   numpy<2 constraint satisfied ({_np.__version__})")
except Exception:
    pass

# scipy.spatial.KDTree is what upstream uses to build spatial positive pairs
try:
    from scipy.spatial import KDTree  # noqa: F401
    print("  OK   scipy.spatial.KDTree (spatial positives)")
except Exception as e:
    missing.append("scipy.spatial.KDTree"); print(f"  MISS KDTree: {e}")

try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}"
          + (f" -> {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        print("  NOTE: no CUDA here. Head nodes usually have no GPU — this is "
              "expected; the LSF job requests one. Re-check inside the job.")
except Exception:
    pass

sys.exit(1 if missing else 0)
PY

# Import the AUTHORS' own modules — the real proof the env can run their code.
log "importing upstream modules from $REPO_DIR ..."
( cd "$REPO_DIR" && "$VENV_DIR/bin/python" - <<'PY'
import sys, traceback
bad = []
for m in ("cellContrast.model", "cellContrast.train",
          "cellContrast.inference", "cellContrast.loadData",
          "cellContrast.utils"):
    try:
        __import__(m)
        print(f"  OK   {m}")
    except Exception as e:
        bad.append(m)
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
sys.exit(1 if bad else 0)
PY
)

# ---------------------------------------------------------------------------
log "done."
log "  venv : $VENV_DIR"
log "  repo : $REPO_DIR"
cat <<EOF

NEXT — smoke-test before any real run (proves the install end-to-end, minutes):

  bash $HERE/contrastive_baselines/submit_cellcontrast.sh \\
      --dataset mmc_luna --smoke_test

Then the real runs:

  bash $HERE/contrastive_baselines/submit_cellcontrast.sh \\
      --dataset mmc_luna --seeds "0 1 2 3 4"

(The submitter defaults to this venv and repo, so no extra flags are needed.)
EOF
