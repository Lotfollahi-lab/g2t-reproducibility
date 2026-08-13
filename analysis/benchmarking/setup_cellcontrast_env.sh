#!/usr/bin/env bash
# setup_cellcontrast_env.sh
# ------------------------------------------------------------------------------
# Clone the authors' CellContrast and build an env to run it, for the
# reviewer-requested baseline. We run THEIR code (MIT licence) rather than a
# reimplementation, so this script deliberately changes nothing about the method.
#
#   Upstream : https://github.com/HKU-BAL/CellContrast   (Li et al., Patterns 2024)
#   Licence  : MIT
#
# Upstream's environment.yml pins only `python=3.9` and `scanpy==1.9.3` and leaves
# PyTorch entirely unpinned (their README installs a CPU build). That is good news
# for us: there is no CUDA/torch version conflict to resolve, so we install a GPU
# torch of our choosing. The rest of the imports (anndata, numpy, pandas, scipy,
# scikit-learn, matplotlib, tqdm) come in with scanpy.
#
# Usage:
#   bash setup_cellcontrast_env.sh                     # clone + create env
#   REPO_DIR=/path CONDA_ENV=cellcontrast bash setup_cellcontrast_env.sh
#   SKIP_CLONE=1 bash setup_cellcontrast_env.sh        # env only
#
# After it finishes, ALWAYS run the smoke test before a real run:
#   conda activate cellcontrast
#   export PYTHONHASHSEED=0
#   python .../contrastive_baselines/run_cellcontrast.py \
#       --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \
#       --cellcontrast_repo "$REPO_DIR" --smoke_test
# ------------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/HKU-BAL/CellContrast.git}"
REPO_DIR="${REPO_DIR:-/nfs/team361/sb75/CellContrast}"
CONDA_ENV="${CONDA_ENV:-cellcontrast}"
PY_VERSION="${PY_VERSION:-3.9}"
SCANPY_VERSION="${SCANPY_VERSION:-1.9.3}"
# Upstream leaves torch unpinned; pick a CUDA build that matches the cluster.
TORCH_SPEC="${TORCH_SPEC:-torch --index-url https://download.pytorch.org/whl/cu121}"
SKIP_CLONE="${SKIP_CLONE:-0}"

echo "== setup_cellcontrast_env.sh =="
echo "repo dir : $REPO_DIR"
echo "conda env: $CONDA_ENV (python $PY_VERSION, scanpy $SCANPY_VERSION)"
echo "torch    : $TORCH_SPEC"
echo

# --- 1. clone / update -------------------------------------------------------
if [[ "$SKIP_CLONE" != "1" ]]; then
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[1/3] repo exists; fetching"
    git -C "$REPO_DIR" fetch --all --tags
  else
    echo "[1/3] cloning $REPO_URL"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  echo "      commit: $(git -C "$REPO_DIR" rev-parse HEAD)"
  # Record the exact commit next to the artifacts — a baseline we did not author
  # must be reproducible to a revision.
  git -C "$REPO_DIR" rev-parse HEAD > "$REPO_DIR/.checked_out_commit"
else
  echo "[1/3] SKIP_CLONE=1"
fi

for f in cellContrast.py parameters/parameters_singleCell.json LICENSE; do
  [[ -e "$REPO_DIR/$f" ]] || { echo "ERROR: expected $f in $REPO_DIR" >&2; exit 1; }
done
echo "      sanity: entry point, parameter file and LICENSE present"

# --- 2. conda env ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH." >&2; exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[2/3] env '$CONDA_ENV' exists; reusing"
else
  echo "[2/3] creating env '$CONDA_ENV'"
  conda create -y -n "$CONDA_ENV" "python=$PY_VERSION"
fi
conda activate "$CONDA_ENV"

echo "[3/3] installing dependencies"
python -m pip install --upgrade pip
# scanpy first (it pulls anndata/numpy/pandas/scipy/sklearn/matplotlib), then torch
python -m pip install "scanpy==$SCANPY_VERSION"
# shellcheck disable=SC2086
python -m pip install $TORCH_SPEC
python -m pip install tqdm

# --- verify ------------------------------------------------------------------
echo
echo "== verification =="
python - <<'PY'
import importlib, sys
ok = True
for m in ("torch", "scanpy", "anndata", "numpy", "pandas", "scipy", "sklearn", "tqdm"):
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:8s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  MISS {m:8s} {type(e).__name__}: {e}")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
except Exception:
    pass
sys.exit(0 if ok else 1)
PY

echo
echo "Done. NEXT — smoke test before any real run:"
echo "  conda activate $CONDA_ENV"
echo "  export PYTHONHASHSEED=0   # upstream intersects genes via a Python set"
echo "  python \$(dirname \$0)/contrastive_baselines/run_cellcontrast.py \\"
echo "      --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\"
echo "      --cellcontrast_repo $REPO_DIR --smoke_test"
