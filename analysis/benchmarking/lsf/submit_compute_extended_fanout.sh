#!/usr/bin/env bash
# submit_compute_extended_fanout.sh
# -----------------------------------------------------------------------------
# Fan out ``compute_extended_metrics.py`` over the cluster: one LSF job
# per (method, timestamp) pair. With 5 LUNA + 5 G2T + 5 CeLEry that's
# 15 jobs running in parallel — vs the single-job ``submit_plot_method_
# comparison.sh`` workflow which scores all 15 sequentially.
#
# Why this exists:
#   - Each (method, timestamp) job only holds ONE timestamp's
#     ~30 slice CSVs at a time, so memory pressure is small. With
#     --workers 4-8 per job and chunked vectorisation, peak per job is
#     ~250 GB even on huge CNS slices — easy to fit in standard nodes.
#   - The cluster can land 15 jobs in parallel almost immediately on
#     long/basement queues, so wall-clock is dominated by the SLOWEST
#     single (method, timestamp), not their sum. For CNS that's
#     ~10x faster than the serial wrapper.
#   - Each job logs to its own ``submit.out`` so progress visibility
#     is per-(method, ts) rather than one giant stream.
#
# After all 15 jobs finish (watch with ``bjobs -J '<job_prefix>*'``),
# run the plot step separately:
#     bash submit_plot_method_comparison.sh --dataset cns_luna --skip_compute
#
# Usage:
#     bash submit_compute_extended_fanout.sh [OPTIONS]
#
# Options:
#   --dataset SLUG           mmc_luna (default) | cns_luna. Forwarded
#                            verbatim as compute_extended_metrics.py's
#                            --dataset, also controls the work-plan
#                            lookup.
#   --methods LIST           Comma-separated subset of methods to fan
#                            out: luna, g2t, celery. Default = all
#                            three. Use this to redo only one method
#                            (e.g. ``--methods celery``).
#   --luna_timestamps  LIST  Per-method timestamp override. Comma-
#   --scgg_timestamps  LIST  separated list; only those timestamps
#   --celery_timestamps LIST get fanned out for the matching method.
#                            Use this to recover specific failed jobs
#                            (e.g. wall-time exceeded) without
#                            re-running the ones that finished. May
#                            be combined freely — e.g.
#                            ``--celery_timestamps a,b --scgg_timestamps c``
#                            fires 3 jobs total.
#   --vectorize              Legacy alias for ``--backend vectorized``.
#                            Ignored if --backend is also given.
#   --backend BACKEND        Per-cell Spearman backend forwarded to
#                            each compute job. One of:
#                              scipy       — original loop, baseline
#                                            (default; bit-faithful)
#                              vectorized  — batched rankdata + einsum
#                                            (5-20× faster, CPU)
#                              numba       — JIT'd CPU with average-tie
#                                            ranking (5-15× faster,
#                                            needs `numba`)
#                              gpu         — chunked torch on CUDA
#                                            (30-100× faster, needs
#                                            GPU; ties drift ~1e-3)
#                            When --backend gpu is used, this wrapper
#                            AUTO-adds ``-gpu num=1:gmem=20000`` to
#                            the bsub. Override with --gpu if you need
#                            a different request string. You still
#                            need to pass an appropriate --queue
#                            (e.g. ``--queue gpu-inference``).
#   --gpu STR                bsub -gpu request string. Default empty
#                            (no GPU request) for CPU backends; auto-
#                            set to ``num=1:gmem=20000`` when
#                            --backend gpu. Common overrides:
#                              num=1:gmem=80000   (require ≥80 GB GPU,
#                                                  pins to A100/H100/H200)
#                              num=1              (any visible GPU)
#   --suffix SUFFIX          Optional CSV filename tag forwarded to
#                            every per-job python invocation. Writes
#                            ``extended_metrics_<SUFFIX>.csv`` per
#                            timestamp instead of the canonical
#                            ``extended_metrics.csv``. Use this to
#                            A/B-compare backends without overwriting
#                            the existing scipy-scored CSVs — e.g.
#                            ``--backend gpu --suffix gpu`` writes
#                            extended_metrics_gpu.csv side-by-side
#                            with the original. The plot wrapper has
#                            a matching --suffix flag; pass the same
#                            value there to render from those CSVs.
#   --workers N              Per-job ProcessPoolExecutor size for the
#                            slice loop. Default 4. Each job's peak
#                            memory ≈ N_workers × ~180 GB worst slice.
#   --skip_existing          Skip (method, timestamp) pairs whose
#                            extended_metrics.csv already exists.
#                            Useful for resuming a partial fanout.
#   (no --auto_plot)         The plot step is intentionally NOT
#                            chained here. submit_plot_method_compa-
#                            rison.sh fires its own bsub internally,
#                            which makes adding a ``-w 'done(...)'``
#                            dependency from the outside awkward.
#                            Instead, two-step the workflow:
#                              1. bash submit_compute_extended_fanout.sh ...
#                              2. wait for `bjobs -J '<job_prefix>*'` to drain
#                              3. bash submit_plot_method_comparison.sh \\
#                                     --dataset <DS> --skip_compute
#   --queue NAME             LSF queue. Default basement.
#   --wall HH:MM             Per-job wall cap. Default 12:00.
#   --mem MB                 Per-job memory cap. Default 256000.
#   --cores N                Per-job LSF -n. Default = ``workers``.
#   --dry_run                Print the bsub commands without
#                            submitting.
#
# Env knobs (rarely needed):
#   VENV_PATH        venv to activate. Default: /nfs/team361/sb75/.venvs/scgg
#   LSF_GROUP        cost-code group. Default: team361
#   SCGG_REPRO_ROOT  path to scgg-reproducibility checkout. Default:
#                    derived from this script's path.
# -----------------------------------------------------------------------------

set -euo pipefail

DATASET="mmc_luna"
METHODS=""
# Per-method timestamp overrides — recovery use-case: re-fire only the
# specific (method, ts) pairs that failed in a previous fanout (e.g.
# wall-time limit). Each forwards verbatim to both --list_work and the
# matching per-job ``--<method>_timestamps`` flag.
LUNA_TIMESTAMPS=""
SCGG_TIMESTAMPS=""
CELERY_TIMESTAMPS=""
VECTORIZE=0
BACKEND=""              # one of scipy|vectorized|numba|gpu (default = python script's default 'scipy')
SUFFIX=""               # per-CSV basename suffix, forwarded to each
                        # per-job python invocation. Default empty =
                        # canonical extended_metrics.csv filename.
WORKERS=4
SKIP_EXISTING=0
QUEUE_ARG="basement"
WALL_ARG="12:00"
MEM_ARG="256000"
CORES_ARG=""
GPU_ARG=""              # bsub -gpu request string. Auto-defaults to
                        # ``num=1:gmem=20000`` when --backend gpu and
                        # no explicit --gpu given. Empty for CPU
                        # backends (the default).
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="${2:?--dataset requires a value}"; shift 2 ;;
        --methods)       METHODS="${2:?--methods requires a value}"; shift 2 ;;
        --luna_timestamps)
            LUNA_TIMESTAMPS="${2:?--luna_timestamps requires a value}"; shift 2 ;;
        --scgg_timestamps)
            SCGG_TIMESTAMPS="${2:?--scgg_timestamps requires a value}"; shift 2 ;;
        --celery_timestamps)
            CELERY_TIMESTAMPS="${2:?--celery_timestamps requires a value}"; shift 2 ;;
        --vectorize)     VECTORIZE=1; shift ;;
        --backend)
            BACKEND="${2:?--backend requires a value (scipy|vectorized|numba|gpu)}"; shift 2 ;;
        --suffix)
            SUFFIX="${2:?--suffix requires a value (e.g. gpu, numba)}"; shift 2 ;;
        --workers)       WORKERS="${2:?--workers requires a value}"; shift 2 ;;
        --skip_existing) SKIP_EXISTING=1; shift ;;
        --queue|-q)      QUEUE_ARG="${2:?--queue requires a value}"; shift 2 ;;
        --wall)          WALL_ARG="${2:?--wall requires a value}"; shift 2 ;;
        --mem)           MEM_ARG="${2:?--mem requires a value}"; shift 2 ;;
        --cores)         CORES_ARG="${2:?--cores requires a value}"; shift 2 ;;
        --gpu)           GPU_ARG="${2:?--gpu requires a bsub -gpu request string}"; shift 2 ;;
        --dry_run)       DRY_RUN=1; shift ;;
        --help|-h)
            awk 'NR>1 && /^[^#]/{exit} {print}' "$0"
            exit 0 ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2 ;;
    esac
done

CORES_FINAL="${CORES_ARG:-$WORKERS}"

# Auto-request a GPU when backend=gpu and the user didn't override.
# Conservative gmem=20000 (20 GB) covers the chunked GPU path's peak
# transient (~5 GB at chunk=2048, N=150k) with plenty of safety, so
# any reasonably modern GPU (24 GB+) will satisfy the constraint.
# Bump via --gpu "num=1:gmem=80000" if you need to pin to A100/H100.
if [[ "$BACKEND" == "gpu" && -z "$GPU_ARG" ]]; then
    GPU_ARG="num=1:gmem=20000"
fi

# Resolve paths. We assume the repro repo is the grandparent of this
# script's dir (analysis/benchmarking/lsf/ → repro root). Override
# with SCGG_REPRO_ROOT if it's anywhere else.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPRO_ROOT="${SCGG_REPRO_ROOT:-$( cd -- "$SCRIPT_DIR/../../.." &> /dev/null && pwd )}"
COMPUTE_SCRIPT="$REPRO_ROOT/analysis/benchmarking/plots/compute_extended_metrics.py"
PLOT_WRAPPER="$REPRO_ROOT/analysis/benchmarking/lsf/submit_plot_method_comparison.sh"

if [[ ! -f "$COMPUTE_SCRIPT" ]]; then
    echo "ERROR: compute script not found: $COMPUTE_SCRIPT" >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/scgg}"
LSF_GROUP_FINAL="${LSF_GROUP:-team361}"

# Resolve artifacts root for log placement. Mirrors
# submit_plot_method_comparison.sh's lookup so fanout + plot logs end
# up in the same comparison_plots/lsf_logs/ tree.
ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"
OUT_DIR="$ARTIFACTS_ROOT/$DATASET/comparison_plots"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_DIR/lsf_logs/fanout_$RUN_TS"
mkdir -p "$LOG_DIR"

# Ask the python script which (method, timestamp) pairs to launch.
# Single source of truth for the default timestamp lists — they live
# in compute_extended_metrics.py, not duplicated here.
LIST_ARGS=( --dataset "$DATASET" --list_work )
if [[ -n "$METHODS" ]]; then
    LIST_ARGS+=( --methods "$METHODS" )
fi
# Per-method timestamp overrides — these make the work plan
# specifically target a subset (e.g. only the timestamps that failed
# in a previous run). Forwarded ONLY to --list_work; the per-job
# bsubs below already pass --<method>_timestamps=<single_ts> built
# from the work-plan rows.
if [[ -n "$LUNA_TIMESTAMPS" ]]; then
    LIST_ARGS+=( --luna_timestamps "$LUNA_TIMESTAMPS" )
fi
if [[ -n "$SCGG_TIMESTAMPS" ]]; then
    LIST_ARGS+=( --scgg_timestamps "$SCGG_TIMESTAMPS" )
fi
if [[ -n "$CELERY_TIMESTAMPS" ]]; then
    LIST_ARGS+=( --celery_timestamps "$CELERY_TIMESTAMPS" )
fi
echo "================================================================"
echo "  submit_compute_extended_fanout.sh"
echo "================================================================"
echo "compute script  : $COMPUTE_SCRIPT"
echo "venv            : $VENV_PATH"
echo "dataset         : $DATASET"
echo "methods         : ${METHODS:-(all)}"
[[ -n "$LUNA_TIMESTAMPS"   ]] && echo "luna_timestamps : $LUNA_TIMESTAMPS"
[[ -n "$SCGG_TIMESTAMPS"   ]] && echo "scgg_timestamps : $SCGG_TIMESTAMPS"
[[ -n "$CELERY_TIMESTAMPS" ]] && echo "celery_timestamps: $CELERY_TIMESTAMPS"
echo "vectorize       : $VECTORIZE"
echo "backend         : ${BACKEND:-(script default)}"
echo "suffix          : ${SUFFIX:-(none)}"
echo "workers per job : $WORKERS"
echo "skip_existing   : $SKIP_EXISTING"
echo "queue           : $QUEUE_ARG"
echo "group           : $LSF_GROUP_FINAL"
echo "wall            : $WALL_ARG"
echo "mem (MB)        : $MEM_ARG"
echo "cores per job   : $CORES_FINAL"
echo "gpu request     : ${GPU_ARG:-(none)}"
echo "log dir         : $LOG_DIR"
echo "dry run         : $DRY_RUN"
echo "================================================================"

WORK_PLAN_FILE="$LOG_DIR/work_plan.tsv"
# shellcheck disable=SC2086  # we want word-splitting on LIST_ARGS
"$VENV_PATH/bin/python" "$COMPUTE_SCRIPT" "${LIST_ARGS[@]}" > "$WORK_PLAN_FILE"
echo "work plan -> $WORK_PLAN_FILE"
echo
N_JOBS=$(wc -l < "$WORK_PLAN_FILE")
echo "submitting $N_JOBS compute job(s) ..."
echo

# All compute jobs share a job-name PREFIX so the optional plot
# dependency can target them with one ``-w 'done(prefix*)'`` clause.
JOB_PREFIX="compute-extended-$DATASET-$RUN_TS"

# Per-method timestamp flag mapping. Each compute job passes exactly
# one timestamp via the matching method-specific flag — that's how the
# python script targets a single TS instead of its default list.
declare -A TS_FLAG=(
    [LUNA]="--luna_timestamps"
    [G2T]="--scgg_timestamps"
    [CeLEry]="--celery_timestamps"
)
declare -A METHODS_FLAG=(
    [LUNA]="luna"
    [G2T]="g2t"
    [CeLEry]="celery"
)

JOB_IDS=()

while IFS=$'\t' read -r METHOD TS; do
    [[ -z "$METHOD" || -z "$TS" ]] && continue
    METHOD_LOWER="${METHODS_FLAG[$METHOD]:-}"
    TS_FLAG_NAME="${TS_FLAG[$METHOD]:-}"
    if [[ -z "$METHOD_LOWER" || -z "$TS_FLAG_NAME" ]]; then
        echo "ERROR: unknown method '$METHOD' in work plan" >&2
        exit 3
    fi

    JOB_NAME="$JOB_PREFIX-$METHOD_LOWER-$TS"
    PER_JOB_LOG_DIR="$LOG_DIR/$METHOD_LOWER-$TS"
    mkdir -p "$PER_JOB_LOG_DIR"
    JOB_SCRIPT="$PER_JOB_LOG_DIR/job.sh"
    LOG_OUT="$PER_JOB_LOG_DIR/submit.out"
    LOG_ERR="$PER_JOB_LOG_DIR/submit.err"

    {
        echo "#!/usr/bin/env bash"
        echo "# Auto-generated by submit_compute_extended_fanout.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "# Re-runnable: bash $JOB_SCRIPT"
        echo "set -euo pipefail"
        echo
        # PYTHONUNBUFFERED=1 so per-slice progress lines flush in real
        # time (otherwise CPython block-buffers stdout under bsub and
        # the log file looks empty for hours — same trap we hit on the
        # plot wrapper, hence the matching fix here).
        echo "export PYTHONUNBUFFERED=1"
        # BLAS / OpenMP thread count. Two regimes:
        #   - workers == 1: serial Python — let the inner numpy einsum
        #     / rankdata calls use ALL the LSF cores.
        #   - workers >  1: ProcessPoolExecutor — each worker should be
        #     single-threaded, otherwise workers × cores threads
        #     oversubscribe the box and PESSIMIZE the wall time
        #     (this was the bug that contributed to the original
        #     wall-time blowups: cores=16, workers=4, OMP=16 →
        #     64 threads thrashing on a 16-core box).
        if [[ "$WORKERS" -le 1 ]]; then
            printf 'export OMP_NUM_THREADS=%q\n'      "$CORES_FINAL"
            printf 'export MKL_NUM_THREADS=%q\n'      "$CORES_FINAL"
            printf 'export OPENBLAS_NUM_THREADS=%q\n' "$CORES_FINAL"
        else
            echo 'export OMP_NUM_THREADS=1'
            echo 'export MKL_NUM_THREADS=1'
            echo 'export OPENBLAS_NUM_THREADS=1'
        fi
        echo
        printf 'source %q\n' "$VENV_PATH/bin/activate"
        echo
        echo "# Compute extended_metrics.csv for ONE (method, timestamp) pair."
        printf 'exec python %q --dataset %q --methods %q %s %q --workers %q' \
            "$COMPUTE_SCRIPT" "$DATASET" "$METHOD_LOWER" \
            "$TS_FLAG_NAME" "$TS" "$WORKERS"
        if [[ -n "$BACKEND" ]]; then
            printf ' --backend %q' "$BACKEND"
        elif [[ "$VECTORIZE" == "1" ]]; then
            # Legacy alias path: only forward --vectorize if --backend
            # wasn't given. (--backend takes precedence.)
            printf ' --vectorize'
        fi
        if [[ -n "$SUFFIX" ]]; then
            printf ' --suffix %q' "$SUFFIX"
        fi
        if [[ "$SKIP_EXISTING" == "1" ]]; then
            printf ' --skip_existing'
        fi
        printf '\n'
    } > "$JOB_SCRIPT"
    chmod +x "$JOB_SCRIPT"

    BSUB_CMD=(
        bsub
        -G "$LSF_GROUP_FINAL"
        -q "$QUEUE_ARG"
        -n "$CORES_FINAL"
        -M "$MEM_ARG"
        -R "select[mem>$MEM_ARG] rusage[mem=$MEM_ARG]"
        -R "span[ptile=$CORES_FINAL]"
        -W "$WALL_ARG"
        -J "$JOB_NAME"
        -o "$LOG_OUT"
        -e "$LOG_ERR"
    )
    # Tack on the -gpu request when present. Conditional so CPU jobs
    # don't accidentally land on a GPU queue's esub validator that
    # rejects ``no -gpu request given``.
    if [[ -n "$GPU_ARG" ]]; then
        BSUB_CMD+=( -gpu "$GPU_ARG" )
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  [dry] %s\n' "$JOB_NAME"
        printf '          ' ; printf '%q ' "${BSUB_CMD[@]}"
        printf '< %q\n' "$JOB_SCRIPT"
        continue
    fi

    BSUB_OUT=$("${BSUB_CMD[@]}" < "$JOB_SCRIPT")
    echo "  $BSUB_OUT  ($JOB_NAME)"
    JID=$(echo "$BSUB_OUT" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' || true)
    [[ -n "$JID" ]] && JOB_IDS+=("$JID")
done < "$WORK_PLAN_FILE"

echo
echo "================================================================"
echo "submitted $N_JOBS compute job(s) — prefix: $JOB_PREFIX"
echo "tail any: tail -f $LOG_DIR/<method>-<ts>/submit.out"
echo "watch:    bjobs -J '$JOB_PREFIX*'"
echo
echo "After all $N_JOBS compute jobs finish, render the plot with:"
echo "    bash $PLOT_WRAPPER \\"
echo "        --dataset $DATASET \\"
echo "        --skip_compute --wall 00:30 --mem 8000"
echo "================================================================"
