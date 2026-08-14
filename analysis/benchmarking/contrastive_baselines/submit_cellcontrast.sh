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
# The imaging-vs-spot parameter choice (the paper's k_nearest_positives: 80 vs
# 20) and the normalisation mode are DERIVED from --dataset here, not left to the
# runner's defaults — see the per-dataset table below. The runner independently
# re-checks both and refuses a mismatch; that duplication is deliberate.
#
# Usage:
#   # 0) once: clone + env, then ALWAYS smoke-test before a real run
#   bash ../setup_cellcontrast_env.sh
#   bash submit_cellcontrast.sh --dataset mmc_luna --smoke_test
#
#   # 1) cortex, 5 seeds
#   bash submit_cellcontrast.sh --dataset mmc_luna --seeds "0 1 2 3 4"
#
#   # 2) CNS. The shared 600-d cross-platform latent lives in adata.X ITSELF —
#   #    the only obsm key on those h5ads is 'spatial' — so do NOT pass
#   #    --use_obsm; the latent is handed over untouched via
#   #    --expression_mode silver_raw (the default for this dataset). Needs a
#   #    training cap: 2.85M cells is ~9 days.
#   bash submit_cellcontrast.sh --dataset cns_luna --max_train_cells 150000 \
#        --exclude_test_files sagittal1_test.h5ad,sagittal2_test.h5ad,sagittal3_test.h5ad,spinalcord_test.h5ad \
#        --seeds "0 1 2 3 4"
#
#   # 3) 10x Xenium breast (imaging, k=80) and 10x Visium DLPFC (spot, k=20);
#   #    both are raw integer counts, so both default to silver_raw
#   bash submit_cellcontrast.sh --dataset breast_janesick --seeds "0 1 2 3 4"
#   bash submit_cellcontrast.sh --dataset dlpfc_visium    --seeds "0 1 2 3 4"
#
# Options:
#   --dataset NAME        mmc_luna | cns_luna | breast_janesick | dlpfc_visium
#                         (required)
#   --seeds "0 1 2"       seeds, one job each   (default "0 1 2 3 4")
#   --data_dir DIR        override the silver dir
#   --repo DIR            CellContrast checkout  (default $CELLCONTRAST_REPO, else
#                         <this dir>/../CellContrast, i.e. the clone
#                         setup_cellcontrast_env.sh makes beside the other
#                         benchmarking assets)
#   --venv DIR            uv venv                (default
#                         /nfs/team361/sb75/.venvs/cellcontrast)
#   --epochs N            override training_epoch. OMIT to use the authors'
#                         default (3000) — that is the defensible choice for a
#                         baseline. 1000 is ~3x faster and the paper says >1000
#                         suffices, but it IS a deviation and is recorded.
#   --expression_mode M   log2 | silver_raw. Default is PER DATASET (table below):
#                         log2(1+x) where silver X holds count-magnitude values,
#                         silver_raw where the matrix must be handed over
#                         untouched (the both-sign cns_luna latent, raw counts).
#   --coord_frame F       isotropic (default) | per_axis. isotropic preserves the
#                         aspect ratio, so upstream's k-NN positive graph is
#                         identical to raw microns; per_axis matches LUNA/G2T's
#                         position_normalize but changes 7-14% of the positive set.
#   --query_chunk N       cells per inference call (default per dataset, 8000).
#                         Upstream builds dense query x query AND query x ref
#                         matrices, so a 63k-cell CNS slice unchunked peaks near
#                         292 GB. top-1 retrieval is per-row independent, so
#                         chunking is bit-identical. 0 disables it.
#   --max_mem_gb G        make the runner REFUSE before inference if its estimated
#                         peak exceeds this (default: 85% of the LSF reservation,
#                         leaving headroom for torch and the loaded objects), so a
#                         bad memory plan fails in seconds, not 6 hours in.
#   --force_platform      pass through the runner's dataset-vs-parameter-file
#                         check. Only needed for a dataset whose name does not
#                         advertise its platform; it does NOT change the flag
#                         derived here.
#   --use_obsm KEY        feature matrix from adata.obsm[KEY] instead of .X.
#                         'spatial' is REFUSED on every dataset (it would feed
#                         ground-truth coordinates in as features), and any
#                         --use_obsm is refused on cns_luna (its latent is in .X).
#   --max_train_cells N   cap training cells by per-slice subsampling
#   --max_ref_cells N     cap the inference reference (memory)
#   --exclude_test_files  comma-separated *_test.h5ad basenames to skip, so the
#                         mean is over the SAME sections as G2T/LUNA/CeLEry.
#                         Required for cns_luna (4 of 18 sections are excluded
#                         there). Basenames are matched exactly; a name that is
#                         not in the data dir is warned about, not guessed.
#   --smoke_test          tiny 1-job run to prove the install; never reported
#   --mem MB / --wall HH:MM / --queue Q / --group G / --gpu SPEC
#   --dry_run             print the bsub commands without submitting
#
# Env overrides: CELLCONTRAST_REPO, VENV_DIR, SCGG_ARTIFACTS_ROOT, LSF_GROUP,
# LSF_QUEUE, MEM_MB, WALL, GPU_SPEC, EXCLUDE_TEST_FILES, and CORES (default 8:
# the LSF slot count, also exported as OMP/MKL/OPENBLAS_NUM_THREADS in the job so
# upstream's per-row argsort does not oversubscribe a shared node).
# ------------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_cellcontrast.py"

DATASET=""
SEEDS="0 1 2 3 4"
DATA_DIR=""
# Defaults match setup_cellcontrast_env.sh: the authors' code is cloned next to
# the other benchmarking assets, and the env is a uv venv (this cluster has no
# conda on PATH).
REPO="${CELLCONTRAST_REPO:-$HERE/../CellContrast}"
VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/cellcontrast}"
EPOCHS=""
USE_OBSM=""
MAX_TRAIN_CELLS=""
MAX_REF_CELLS=""
# Empty means "take the per-dataset default from the table below". Anything set
# here (flag or env) wins, except where a gate proves it wrong.
EXPRESSION_MODE=""
COORD_FRAME="${COORD_FRAME:-isotropic}"
QUERY_CHUNK=""
MAX_MEM_GB=""
FORCE_PLATFORM=""
# Comma-separated *_test.h5ad basenames to skip. The runner supports this but the
# submitter never forwarded it, so a CNS run scored all 18 test sections while
# G2T/LUNA/CeLEry score 14 (sagittal1/2/3 + spinalcord excluded) — a mean over a
# different population of slices, i.e. not a comparable number.
EXCLUDE_TEST_FILES="${EXCLUDE_TEST_FILES:-}"
SMOKE=""
DRY_RUN=""
ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
# Empty means "per-dataset default" (see the table); the old flat 128000 was set
# for mmc-sized slices and is not the right reservation for the CNS/Xenium runs.
MEM_MB="${MEM_MB:-}"
WALL="${WALL:-48:00}"
CORES="${CORES:-8}"
GPU_SPEC="${GPU_SPEC:-num=1:mode=shared:j_exclusive=no}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)          DATASET="${2:?}"; shift 2 ;;
    --seeds)            SEEDS="${2:?}"; shift 2 ;;
    --data_dir)         DATA_DIR="${2:?}"; shift 2 ;;
    --repo)             REPO="${2:?}"; shift 2 ;;
    --venv)             VENV_DIR="${2:?}"; shift 2 ;;
    --epochs)           EPOCHS="${2:?}"; shift 2 ;;
    --use_obsm)         USE_OBSM="${2:?}"; shift 2 ;;
    --max_train_cells)  MAX_TRAIN_CELLS="${2:?}"; shift 2 ;;
    --max_ref_cells)    MAX_REF_CELLS="${2:?}"; shift 2 ;;
    --exclude_test_files) EXCLUDE_TEST_FILES="${2:?}"; shift 2 ;;
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
[[ -f "$VENV_DIR/bin/activate" ]] || {
  echo "ERROR: uv venv not found at $VENV_DIR" >&2
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
# Same reasoning as --use_obsm: scoring a different set of test sections than the
# other three methods is not a comparable number, so refuse rather than warn.
if [[ "$DATASET" == "cns_luna" && -z "$EXCLUDE_TEST_FILES" && -z "$SMOKE" ]]; then
  cat >&2 <<'EOF'
ERROR: --exclude_test_files is required for cns_luna.
  G2T/LUNA/CeLEry score 14 of the 18 CNS test sections (sagittal1/2/3 and
  spinalcord excluded); scoring all 18 averages over a different population of
  slices and is not comparable. List the exact basenames used by the other
  methods (see plots/compute_extended_metrics.py and the scgg/luna submitters),
  then pass them comma-separated, e.g.
    --exclude_test_files sagittal1_test.h5ad,sagittal2_test.h5ad,sagittal3_test.h5ad,spinalcord_test.h5ad
  Verify the basenames against `ls "$DATA_DIR"` before submitting.
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
echo "venv    : $VENV_DIR"
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
    printf 'source %q/bin/activate\n' "$VENV_DIR"
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
    [[ -n "$EXCLUDE_TEST_FILES" ]] && printf ' \\\n    --exclude_test_files %q' "$EXCLUDE_TEST_FILES"
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
