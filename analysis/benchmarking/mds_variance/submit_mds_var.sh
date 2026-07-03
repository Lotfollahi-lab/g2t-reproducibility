#!/usr/bin/env bash
# submit_mds_var.sh
# ------------------------------------------------------------------------------
# One-off diagnostic: how 2-D-embeddable is G2T's learned squared-distance
# matrix D_hat?  Runs INFERENCE ONLY (single test pass -> no validation-epoch
# pollution) on a trained G2T cortex checkpoint with SCGG_LOG_MDS_VAR set, so
# edm_head.py logs the top-2 MDS eigenvalue variance-explained per (reverse
# step, slice) to a CSV. Then:
#
#     python analyse_mds_var.py <csv>
#
# reports the converged (final reverse step) fraction per section + the mean
# to quote in the paper. The diagnostic does its OWN full fp64 eigh, so it is
# independent of model.edm.mds_solver (no override needed); cortex slices are
# ~7k cells so this is cheap.
#
# Usage:
#   bash submit_mds_var.sh --checkpoint /nfs/.../scgg_model/<TS>/best_model.ckpt
#
# Options:
#   --checkpoint PATH   (required) trained G2T cortex .ckpt
#   --data_dir DIR      dataset dir (default: cortex mmc_luna silver)
#   --seed N            inference seed (default: 0)
#   --out CSV           where to write the diagnostic CSV
#                       (default: ./mds_var_out/mds_var_cortex_seed<N>.csv)
#   --venv PATH         python venv (default: /nfs/team361/sb75/.venvs/scgg)
#   --repo PATH         scgg repo root (default: /nfs/team361/sb75/scgg)
#   --dry_run           print the bsub command without submitting
# ------------------------------------------------------------------------------
set -euo pipefail

CHECKPOINT=""
DATA_DIR="/nfs/team361/sb75/DATASETS/silver/mmc_luna"
SEED=0
OUT=""
VENV="${SCGG_VENV:-/nfs/team361/sb75/.venvs/scgg}"
REPO="${SCGG_REPO:-/nfs/team361/sb75/scgg}"
DRY_RUN=""
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
MEM_MB="${MEM_MB:-128000}"
WALL="${WALL:-4:00}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
        --data_dir)   DATA_DIR="${2:?}"; shift 2 ;;
        --seed)       SEED="${2:?}"; shift 2 ;;
        --out)        OUT="${2:?}"; shift 2 ;;
        --venv)       VENV="${2:?}"; shift 2 ;;
        --repo)       REPO="${2:?}"; shift 2 ;;
        --dry_run)    DRY_RUN=1; shift ;;
        -h|--help)    sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required (a trained G2T cortex .ckpt)." >&2
    echo "       Find it under <ARTIFACTS_ROOT>/mmc_luna/scgg_model/<TS>/." >&2
    exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "$OUT" ]]; then
    OUT="$HERE/mds_var_out/mds_var_cortex_seed${SEED}.csv"
fi
mkdir -p "$(dirname "$OUT")"
# Start from a clean CSV so a re-run doesn't append to stale rows.
rm -f "$OUT"

LOGDIR="$HERE/mds_var_out/lsf"
mkdir -p "$LOGDIR"

# Build the job body: activate venv, cd repo, export the diagnostic env var,
# run inference only. run_scgg_inference.py does a single test pass.
JOB_SCRIPT="$(mktemp "$HERE/mds_var_out/job.XXXXXX.sh")"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'source %q/bin/activate\n' "$VENV"
    printf 'cd %q\n' "$REPO"
    printf 'export SCGG_LOG_MDS_VAR=%q\n' "$OUT"
    echo 'echo "python: $(which python) $(python --version)"'
    echo 'echo "SCGG_LOG_MDS_VAR=$SCGG_LOG_MDS_VAR"'
    printf 'exec python scripts/run_scgg_inference.py --data_dir %q --checkpoint %q --seed %q\n' \
        "$DATA_DIR" "$CHECKPOINT" "$SEED"
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

BSUB_CMD=(
    bsub
    -G "$LSF_GROUP"
    -q "$LSF_QUEUE"
    -n 4
    -M "$MEM_MB"
    -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB]"
    -gpu "num=1:mode=shared:j_exclusive=no"
    -W "$WALL"
    -J "mds_var_seed${SEED}"
    -o "$LOGDIR/mds_var_seed${SEED}.%J.out"
    -e "$LOGDIR/mds_var_seed${SEED}.%J.err"
    bash "$JOB_SCRIPT"
)

echo "== submit_mds_var.sh =="
echo "checkpoint : $CHECKPOINT"
echo "data_dir   : $DATA_DIR"
echo "seed       : $SEED"
echo "diag CSV   : $OUT"
echo "job script : $JOB_SCRIPT"
echo "command    :"
printf '  %q ' "${BSUB_CMD[@]}"; echo
echo
if [[ -n "$DRY_RUN" ]]; then
    echo "[dry_run] not submitting."
    exit 0
fi
"${BSUB_CMD[@]}"
echo
echo "After it finishes:  python $HERE/analyse_mds_var.py $OUT"
