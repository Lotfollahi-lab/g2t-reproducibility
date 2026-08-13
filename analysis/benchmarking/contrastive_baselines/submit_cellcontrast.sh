#!/usr/bin/env bash
# submit_cellcontrast.sh
# ------------------------------------------------------------------------------
# Submit the reviewer-requested CellContrast baseline to LSF, one job per seed,
# mirroring how the other baselines are launched (submit_pipeline.sh conventions:
# same group/queue defaults, a generated per-job script, GPU request, logs beside
# the artifacts).
#
# Runs the AUTHORS' published code (https://github.com/HKU-BAL/CellContrast, MIT)
# at their default hyperparameters via run_cellcontrast.py. Nothing here changes
# the method.
#
# Usage:
#   # 0) once: clone + env, then ALWAYS smoke-test before a real run
#   bash ../setup_cellcontrast_env.sh
#   bash submit_cellcontrast.sh --dataset mmc_luna --smoke_test
#
#   # 1) cortex, 5 seeds
#   bash submit_cellcontrast.sh --dataset mmc_luna --seeds "0 1 2 3 4"
#
#   # 2) CNS — needs the embedding key and a training cap (2.85M cells is ~9 days)
#   bash submit_cellcontrast.sh --dataset cns_luna --use_obsm <OBSM_KEY> \
#        --max_train_cells 150000 --seeds "0 1 2 3 4"
#
# Options:
#   --dataset NAME        mmc_luna | cns_luna   (required)
#   --seeds "0 1 2"       seeds, one job each   (default "0 1 2 3 4")
#   --data_dir DIR        override the silver dir
#   --repo DIR            CellContrast checkout  (default $CELLCONTRAST_REPO or
#                         /nfs/team361/sb75/CellContrast)
#   --env NAME            conda env             (default cellcontrast)
#   --epochs N            override training_epoch. OMIT to use the authors'
#                         default (3000) — that is the defensible choice for a
#                         baseline. 1000 is ~3x faster and the paper says >1000
#                         suffices, but it IS a deviation and is recorded.
#   --use_obsm KEY        feature matrix from adata.obsm[KEY] instead of genes
#                         (REQUIRED for cns_luna: the Harmony latent, matching
#                         our protocol for G2T/LUNA on that dataset)
#   --max_train_cells N   cap training cells by per-slice subsampling
#   --max_ref_cells N     cap the inference reference (memory)
#   --smoke_test          tiny 1-job run to prove the install; never reported
#   --mem MB / --wall HH:MM / --queue Q / --group G / --gpu SPEC
#   --dry_run             print the bsub commands without submitting
# ------------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_cellcontrast.py"

DATASET=""
SEEDS="0 1 2 3 4"
DATA_DIR=""
REPO="${CELLCONTRAST_REPO:-/nfs/team361/sb75/CellContrast}"
CONDA_ENV="${CONDA_ENV:-cellcontrast}"
EPOCHS=""
USE_OBSM=""
MAX_TRAIN_CELLS=""
MAX_REF_CELLS=""
SMOKE=""
DRY_RUN=""
ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
MEM_MB="${MEM_MB:-128000}"
WALL="${WALL:-48:00}"
CORES="${CORES:-8}"
GPU_SPEC="${GPU_SPEC:-num=1:mode=shared:j_exclusive=no}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)          DATASET="${2:?}"; shift 2 ;;
    --seeds)            SEEDS="${2:?}"; shift 2 ;;
    --data_dir)         DATA_DIR="${2:?}"; shift 2 ;;
    --repo)             REPO="${2:?}"; shift 2 ;;
    --env)              CONDA_ENV="${2:?}"; shift 2 ;;
    --epochs)           EPOCHS="${2:?}"; shift 2 ;;
    --use_obsm)         USE_OBSM="${2:?}"; shift 2 ;;
    --max_train_cells)  MAX_TRAIN_CELLS="${2:?}"; shift 2 ;;
    --max_ref_cells)    MAX_REF_CELLS="${2:?}"; shift 2 ;;
    --smoke_test)       SMOKE=1; shift ;;
    --mem)              MEM_MB="${2:?}"; shift 2 ;;
    --wall)             WALL="${2:?}"; shift 2 ;;
    --queue)            LSF_QUEUE="${2:?}"; shift 2 ;;
    --group)            LSF_GROUP="${2:?}"; shift 2 ;;
    --gpu)              GPU_SPEC="${2:?}"; shift 2 ;;
    --dry_run)          DRY_RUN=1; shift ;;
    -h|--help)          sed -n '2,52p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- validation --------------------------------------------------------------
case "$DATASET" in
  mmc_luna|cns_luna) ;;
  "") echo "ERROR: --dataset is required (mmc_luna | cns_luna)." >&2; exit 2 ;;
  *)  echo "ERROR: unknown --dataset '$DATASET'." >&2; exit 2 ;;
esac
[[ -f "$RUNNER" ]] || { echo "ERROR: runner missing: $RUNNER" >&2; exit 1; }
[[ -f "$REPO/cellContrast.py" ]] || {
  echo "ERROR: CellContrast not found at $REPO (expected cellContrast.py)." >&2
  echo "       Run: bash $HERE/../setup_cellcontrast_env.sh" >&2; exit 1; }
[[ -n "$DATA_DIR" ]] || DATA_DIR="/nfs/team361/sb75/DATASETS/silver/$DATASET"
[[ -d "$DATA_DIR" ]] || { echo "ERROR: data dir not found: $DATA_DIR" >&2; exit 1; }

# CNS needs the embedding, exactly as G2T/LUNA use it there. Refuse to run on raw
# genes by accident — that would silently be a different experiment.
if [[ "$DATASET" == "cns_luna" && -z "$USE_OBSM" ]]; then
  cat >&2 <<'EOF'
ERROR: --use_obsm is required for cns_luna.
  On that dataset our other methods consume the shared 600-d Harmony latent, not
  raw genes. Running CellContrast on genes would be a different experiment and
  not comparable. Find the key with:
    python -c "import anndata; a=anndata.read_h5ad('<a *_test.h5ad>'); print(list(a.obsm.keys()))"
  then pass --use_obsm <that key>.
EOF
  exit 2
fi
if [[ "$DATASET" == "cns_luna" && -z "$MAX_TRAIN_CELLS" && -z "$SMOKE" ]]; then
  echo "WARNING: cns_luna without --max_train_cells will try to train on ~2.85M" >&2
  echo "         cells (order of a week). Consider --max_train_cells 150000." >&2
fi
if [[ -n "$SMOKE" ]]; then
  SEEDS="0"; WALL="1:00"; MEM_MB="32000"
  echo "SMOKE TEST: 1 job, 1h wall — proves the install; do not report the numbers."
fi

LOGDIR="$ARTIFACTS_ROOT/$DATASET/cellcontrast_inference/lsf"
JOBDIR="$LOGDIR/jobs"
mkdir -p "$JOBDIR"

echo "== submit_cellcontrast.sh =="
echo "dataset : $DATASET   ($DATA_DIR)"
echo "repo    : $REPO  (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "env     : $CONDA_ENV"
echo "seeds   : $SEEDS"
echo "epochs  : ${EPOCHS:-<upstream default 3000>}"
echo "obsm    : ${USE_OBSM:-<gene expression, log2(1+x)>}"
echo "lsf     : $LSF_QUEUE / $LSF_GROUP / ${MEM_MB}MB / $WALL / gpu $GPU_SPEC"
echo

for SEED in $SEEDS; do
  TS="$(date +%Y%m%d_%H%M%S)"
  JOB="$JOBDIR/cellcontrast_${DATASET}_seed${SEED}_${TS}.sh"

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\""
    printf 'conda activate %q\n' "$CONDA_ENV"
    # upstream intersects genes through a Python set; pin the hash seed so the
    # feature column order is reproducible across runs
    echo "export PYTHONHASHSEED=0"
    echo 'echo "python: $(which python)"; python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"'
    printf 'exec python %q \\\n' "$RUNNER"
    printf '    --data_dir %q \\\n'           "$DATA_DIR"
    printf '    --cellcontrast_repo %q \\\n'  "$REPO"
    printf '    --out_root %q \\\n'           "$ARTIFACTS_ROOT"
    printf '    --dataset %q \\\n'            "$DATASET"
    printf '    --run_timestamp %q \\\n'      "$TS"
    printf '    --seed %q'                    "$SEED"
    [[ -n "$EPOCHS" ]]          && printf ' \\\n    --epochs %q' "$EPOCHS"
    [[ -n "$USE_OBSM" ]]        && printf ' \\\n    --use_obsm %q' "$USE_OBSM"
    [[ -n "$MAX_TRAIN_CELLS" ]] && printf ' \\\n    --max_train_cells %q' "$MAX_TRAIN_CELLS"
    [[ -n "$MAX_REF_CELLS" ]]   && printf ' \\\n    --max_ref_cells %q' "$MAX_REF_CELLS"
    [[ -n "$SMOKE" ]]           && printf ' \\\n    --smoke_test'
    echo
  } > "$JOB"
  chmod +x "$JOB"

  BSUB=(bsub
    -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$CORES"
    -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB]"
    -R "span[ptile=$CORES]"
    -gpu "$GPU_SPEC" -W "$WALL"
    -J "cc_${DATASET}_s${SEED}"
    -o "$LOGDIR/cellcontrast_${DATASET}_seed${SEED}_${TS}.%J.out"
    -e "$LOGDIR/cellcontrast_${DATASET}_seed${SEED}_${TS}.%J.err"
    bash "$JOB")

  echo "seed $SEED -> $TS"
  echo "  job script: $JOB"
  if [[ -n "$DRY_RUN" ]]; then
    printf '  [dry_run] '; printf '%q ' "${BSUB[@]}"; echo
  else
    "${BSUB[@]}"
  fi
  sleep 2   # distinct timestamps: artifact dirs are keyed to the second
done

echo
echo "logs      : $LOGDIR"
echo "artifacts : $ARTIFACTS_ROOT/$DATASET/cellcontrast_inference/<TS>/test_results/"
echo
echo "when the jobs finish, score with the SAME harness as the other methods:"
echo "  python $HERE/../plots/compute_extended_metrics.py \\"
echo "      --dataset $DATASET --methods cellcontrast"
