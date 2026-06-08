#!/usr/bin/env bash
# submit_plot_method_comparison.sh
# -----------------------------------------------------------------------------
# Submit a tiny CPU-only LSF job that (optionally) recomputes
# extended_metrics.csv for each inference timestamp and then renders
# the G2T-vs-LUNA-vs-CeLEry multi-metric comparison figure. Wraps:
#
#     analysis/benchmarking/plots/compute_extended_metrics.py
#     analysis/benchmarking/plots/plot_method_comparison.py
#
# The plot script reads the per-timestamp extended_metrics.csv written
# by the computation step; if those files already exist (from a prior
# run) you can skip recomputation with ``--skip_compute``.
#
# Sized for the normal queue (no GPU, 32 GB, 1 core, 12 h wall). The
# computation step is the bulk of the wall-clock — for each slice
# we build TWO N×N pairwise distance matrices (true + predicted),
# then run scipy.stats.spearmanr per cell over the N-long rows;
# that's O(N² log N) per slice in Python and adds up fast across
# 30+ MMC slices × 15 timestamps (5 LUNA + 5 G2T + 5 CeLEry) seeds.
# The 30 min default we shipped first was too tight; bumped to 12 h,
# which comfortably covers MMC and leaves headroom for CNS slices.
# Override with the --wall flag below.
#
# Usage:
#     bash submit_plot_method_comparison.sh [OPTIONS]
#
# Options:
#   --dry_run                Print the bsub command without submitting.
#   --skip_compute           Skip the extended_metrics.csv computation
#                            step; go straight to plotting. Use when the
#                            files already exist and you just want to
#                            re-render the figure with new style /
#                            different timestamps.
#   --dataset SLUG           Dataset to score + plot. One of:
#                            mmc_luna (default), cns_luna. Forwarded
#                            as --dataset to both compute + plot.
#                            Also rebases OUT_DIR / log dir under
#                            <artifacts>/<dataset>/comparison_plots/.
#   --methods LIST           Comma-separated subset of methods to
#                            (re-)score in the compute step:
#                            luna, g2t (alias scgg), celery. Default
#                            scores all three. Use this to incrementally
#                            score a new method without redoing the
#                            others — e.g. ``--methods celery`` re-scores
#                            ONLY celery_inference and leaves the
#                            existing luna + scgg extended_metrics.csv
#                            files alone. The plot step ALWAYS reads
#                            all three methods, regardless of --methods.
#   --scgg_timestamps LIST   Comma-separated YYYYMMDD_HHMMSS timestamps
#                            to load from scgg_inference/ for G2T.
#                            Forwarded to BOTH the compute step and the
#                            plot step. Defaults to the hardcoded list
#                            in the .py.
#   --luna_timestamps LIST   Comma-separated YYYYMMDD_HHMMSS timestamps
#                            to load from luna_inference/. Defaults to
#                            auto-discovery (every matching subdir).
#   --celery_timestamps LIST Comma-separated YYYYMMDD_HHMMSS timestamps
#                            to load from celery_inference/. Defaults
#                            to the hardcoded per_reference seeds list.
#   --suffix SUFFIX          Optional CSV basename suffix forwarded to
#                            BOTH compute (write) and plot (read). Use
#                            this to A/B-compare backends — e.g. pair
#                            with the fanout wrapper's
#                            ``--backend gpu --suffix gpu`` then plot
#                            with ``--suffix gpu`` to render from
#                            ``extended_metrics_gpu.csv``. Output
#                            figure filename also gets the suffix
#                            (``g2t_vs_luna_vs_celery_extended_metrics
#                            _<SUFFIX>.{svg,pdf,png}``) so multiple
#                            backend renders coexist in the
#                            comparison_plots/ dir.
#   --workers N              Process-pool size for the per-slice loop in
#                            the compute step (passed through as
#                            --workers N). >1 enables multiprocessing —
#                            biggest win on CNS where each slice is slow.
#                            Default unset = serial (1 worker).
#   --vectorize              Enable the vectorised per-cell Spearman
#                            path in the compute step. OFF by default
#                            so re-scoring already-completed timestamps
#                            (e.g. MMC) cannot change their metric
#                            values. Pass this flag for fresh CNS runs
#                            to get the 10-50× speedup (mathematically
#                            identical within float tolerance — see
#                            test_per_cell_spearman_vectorized.py).
#   --wall HH:MM             LSF wall-clock cap. Default: 12:00 (12 h).
#                            Drop to 00:30 if you've already run with
#                            --skip_compute and just want a fast
#                            re-plot.
#   --mem MB                 LSF memory cap in MB. Default: 32000 (32 GB).
#
# The figure (svg + pdf + png), the tidy long-form processed CSV, and
# the LSF logs all land under:
#
#     /nfs/team361/sb75/scgg-reproducibility/artifacts/
#         mmc_luna/comparison_plots/
#             g2t_vs_luna_extended_metrics.{svg,pdf,png}
#             g2t_vs_luna_extended_metrics_processed.csv
#             lsf_logs/<TS>/{submit.out,submit.err,job.sh}
#
# The per-timestamp extended_metrics.csv + per_slice_extended_metrics.csv
# files (one per LUNA/scgg seed) land next to the existing metrics.csv:
#     /nfs/team361/sb75/scgg-reproducibility/artifacts/mmc_luna/
#         {luna,scgg}_inference/<TS>/
#             metrics.csv                         (already there)
#             extended_metrics.csv                (new, single-row)
#             per_slice_extended_metrics.csv      (new, one row per slice)
#
# Env knobs (all optional):
#     SCGG_REPO        absolute path to the scgg monorepo (defaults to
#                      /nfs/team361/sb75/scgg). The plotting venv is
#                      reused from there.
#     VENV_PATH        venv to activate. Default:
#                      /nfs/team361/sb75/.venvs/scgg (matches the
#                      scgg pipeline's venv — same pandas + matplotlib).
#     LSF_QUEUE        override the queue. Default: normal.
#     LSF_GROUP        cost-code group. Default: $LSF_GROUP env or team361.
#                      NOTE: the cluster's ``normal`` queue REJECTS the
#                      sXXXX AI-acceleration groups with
#                      "this is not an AI acceleration queue. Please
#                       submit with an LSF group (-G) that is a non
#                       sXXXX project cost code group."
#                      The pipeline's submit_pipeline.sh uses s10396
#                      because it targets training-parallel (a GPU/AI
#                      queue) — don't cross-paste defaults between the two.
# -----------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=0
SKIP_COMPUTE=0
DATASET=""              # e.g. "cns_luna" → forwarded as --dataset to both
                        # compute + plot. Empty = use the script default
                        # (mmc_luna).
METHODS=""              # e.g. "celery" → forwarded as compute's --methods
SCGG_TIMESTAMPS=""
LUNA_TIMESTAMPS=""
CELERY_TIMESTAMPS=""
SUFFIX=""               # per-CSV basename suffix, forwarded to both
                        # compute (write) + plot (read). Empty = default
                        # extended_metrics.csv path; non-empty pairs
                        # the wrapper with a backend run from the
                        # fanout (e.g. --suffix gpu).
WORKERS=""              # >1 → ProcessPoolExecutor in the compute step.
                        # Combined with --vectorize, this is the main
                        # CNS speedup knob.
VECTORIZE_ON=0          # 1 → forward --vectorize to compute step.
                        # Default 0 = original scipy.stats.spearmanr
                        # per-cell loop (preserves byte-identical metric
                        # values for already-scored timestamps). Pass
                        # --vectorize to opt INTO the 10-50× faster
                        # batched-rankdata path.
WALL_ARG=""
MEM_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)
            DRY_RUN=1; shift ;;
        --skip_compute)
            SKIP_COMPUTE=1; shift ;;
        --dataset)
            DATASET="${2:?--dataset requires a value (mmc_luna|cns_luna)}"; shift 2 ;;
        --methods)
            # Subset to compute extended_metrics for; e.g. "celery" to
            # re-score ONLY the celery_inference tree (leaving the
            # luna + scgg extended_metrics.csv files untouched). Plot
            # step is unaffected — it always reads all three trees.
            METHODS="${2:?--methods requires a value, e.g. 'celery' or 'luna,g2t,celery'}"; shift 2 ;;
        --scgg_timestamps)
            SCGG_TIMESTAMPS="${2:?--scgg_timestamps requires a value}"; shift 2 ;;
        --luna_timestamps)
            LUNA_TIMESTAMPS="${2:?--luna_timestamps requires a value}"; shift 2 ;;
        --celery_timestamps)
            CELERY_TIMESTAMPS="${2:?--celery_timestamps requires a value}"; shift 2 ;;
        --suffix)
            SUFFIX="${2:?--suffix requires a value (e.g. gpu, numba)}"; shift 2 ;;
        --workers)
            WORKERS="${2:?--workers requires a positive integer}"; shift 2 ;;
        --vectorize)
            VECTORIZE_ON=1; shift ;;
        --wall)
            WALL_ARG="${2:?--wall requires a value, e.g. 24:00}"; shift 2 ;;
        --mem)
            MEM_ARG="${2:?--mem requires a value in MB}"; shift 2 ;;
        --help|-h)
            # Print the docstring (top comment block) up to the first non-comment line.
            awk 'NR>1 && /^[^#]/{exit} {print}' "$0"
            exit 0 ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2 ;;
    esac
done

# Default wall-clock + memory caps. Bumped 2026-05-30 from 00:30 /
# 8000 MB after the first MMC submission hit the wall-clock limit:
# the Spearman-per-cell loop over O(N²) distance matrices is slow in
# pure-Python (no vectorisation across cells in scipy.stats.spearmanr).
# Override per-submission with --wall / --mem.
WALL_FINAL="${WALL_ARG:-12:00}"
MEM_FINAL="${MEM_ARG:-32000}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPRO_ROOT="$( cd -- "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
PLOT_SCRIPT="$REPRO_ROOT/analysis/benchmarking/plots/plot_method_comparison.py"
COMPUTE_SCRIPT="$REPRO_ROOT/analysis/benchmarking/plots/compute_extended_metrics.py"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "ERROR: plot script not found: $PLOT_SCRIPT" >&2
    exit 1
fi
if [[ "$SKIP_COMPUTE" == "0" ]] && [[ ! -f "$COMPUTE_SCRIPT" ]]; then
    echo "ERROR: compute script not found: $COMPUTE_SCRIPT" >&2
    echo "Pass --skip_compute if the extended_metrics.csv files already exist." >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/scgg}"
LSF_QUEUE_FINAL="${LSF_QUEUE:-normal}"
# normal queue rejects sXXXX (AI-acceleration) cost-code groups; use the
# project group instead. Set LSF_GROUP=<other> in the env to override.
LSF_GROUP_FINAL="${LSF_GROUP:-team361}"

ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"
# OUT_DIR matches whatever --dataset selects (default: mmc_luna). Both
# the python plot script and this wrapper compute the same dataset-aware
# path — keep these in lockstep with plot_method_comparison.py's
# DEFAULT_DATASET.
DATASET_FINAL="${DATASET:-mmc_luna}"
OUT_DIR="$ARTIFACTS_ROOT/$DATASET_FINAL/comparison_plots"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_DIR/lsf_logs/$RUN_TS"
mkdir -p "$LOG_DIR"

JOB_NAME="plot-g2t-vs-luna-$RUN_TS"
LOG_OUT="$LOG_DIR/submit.out"
LOG_ERR="$LOG_DIR/submit.err"

# Self-contained job script — mirrors the submit_pipeline.sh convention
# (kept on disk so the configuration is re-runnable by hand).
JOB_SCRIPT="$LOG_DIR/job.sh"
{
    echo "#!/usr/bin/env bash"
    echo "# Auto-generated by submit_plot_method_comparison.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "# Re-runnable: bash $JOB_SCRIPT"
    echo "set -euo pipefail"
    echo
    # Force unbuffered stdout/stderr so the compute step's per-slice
    # progress lines reach the LSF log file in real time. Without this
    # the CPython runtime block-buffers stdout when it's redirected to
    # a file (which is what bsub does), and you can wait HOURS without
    # seeing any output even though the job is making progress — and
    # then think the job is stuck. PYTHONUNBUFFERED=1 is the env-var
    # equivalent of running ``python -u``.
    echo "export PYTHONUNBUFFERED=1"
    echo
    printf 'source %q\n' "$VENV_PATH/bin/activate"
    echo
    if [[ "$SKIP_COMPUTE" == "0" ]]; then
        echo "# Step 1: compute extended_metrics.csv per inference timestamp."
        # Build the compute-step argv. Pass timestamps when present so
        # the same scgg/luna/celery seeds get scored that the plot will
        # pull. ``--methods`` lets the caller score only a subset
        # (e.g. ``--methods celery`` to re-score ONLY celery while
        # leaving luna + scgg extended_metrics.csv untouched).
        # printf %q safely shell-quotes every value.
        printf 'python %q' "$COMPUTE_SCRIPT"
        if [[ -n "$DATASET" ]]; then
            printf ' --dataset %q' "$DATASET"
        fi
        if [[ -n "$METHODS" ]]; then
            printf ' --methods %q' "$METHODS"
        fi
        if [[ -n "$SCGG_TIMESTAMPS" ]]; then
            printf ' --scgg_timestamps %q' "$SCGG_TIMESTAMPS"
        fi
        if [[ -n "$LUNA_TIMESTAMPS" ]]; then
            printf ' --luna_timestamps %q' "$LUNA_TIMESTAMPS"
        fi
        if [[ -n "$CELERY_TIMESTAMPS" ]]; then
            printf ' --celery_timestamps %q' "$CELERY_TIMESTAMPS"
        fi
        if [[ -n "$WORKERS" ]]; then
            printf ' --workers %q' "$WORKERS"
        fi
        if [[ "$VECTORIZE_ON" == "1" ]]; then
            printf ' --vectorize'
        fi
        if [[ -n "$SUFFIX" ]]; then
            printf ' --suffix %q' "$SUFFIX"
        fi
        printf '\n'
        echo
    fi
    echo "# Step 2: render the multi-metric comparison figure."
    # The plot step ALWAYS pulls all three methods (no --methods knob)
    # — the figure wouldn't make sense with only one bar. Forward any
    # explicit timestamp lists so plot + compute agree on which seeds.
    printf 'exec python %q' "$PLOT_SCRIPT"
    if [[ -n "$DATASET" ]]; then
        printf ' --dataset %q' "$DATASET"
    fi
    if [[ -n "$SCGG_TIMESTAMPS" ]]; then
        printf ' --scgg_timestamps %q' "$SCGG_TIMESTAMPS"
    fi
    if [[ -n "$LUNA_TIMESTAMPS" ]]; then
        printf ' --luna_timestamps %q' "$LUNA_TIMESTAMPS"
    fi
    if [[ -n "$CELERY_TIMESTAMPS" ]]; then
        printf ' --celery_timestamps %q' "$CELERY_TIMESTAMPS"
    fi
    if [[ -n "$SUFFIX" ]]; then
        printf ' --suffix %q' "$SUFFIX"
    fi
    printf '\n'
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

echo "================================================================"
echo "  submit_plot_method_comparison.sh"
echo "================================================================"
echo "compute script  : $COMPUTE_SCRIPT"
echo "plot script     : $PLOT_SCRIPT"
echo "venv            : $VENV_PATH"
echo "queue           : $LSF_QUEUE_FINAL"
echo "group           : $LSF_GROUP_FINAL"
echo "out dir         : $OUT_DIR"
echo "log dir         : $LOG_DIR"
echo "job script      : $JOB_SCRIPT"
echo "skip compute    : $SKIP_COMPUTE"
echo "dataset         : $DATASET_FINAL"
[[ -n "$METHODS"           ]] && echo "methods         : $METHODS"
[[ -n "$SCGG_TIMESTAMPS"   ]] && echo "scgg_timestamps : $SCGG_TIMESTAMPS"
[[ -n "$LUNA_TIMESTAMPS"   ]] && echo "luna_timestamps : $LUNA_TIMESTAMPS"
[[ -n "$CELERY_TIMESTAMPS" ]] && echo "celery_timestamps: $CELERY_TIMESTAMPS"
[[ -n "$SUFFIX"            ]] && echo "suffix          : $SUFFIX"
[[ -n "$WORKERS"           ]] && echo "workers         : $WORKERS"
[[ "$VECTORIZE_ON" == "1" ]] && echo "vectorize       : ON (batched rankdata path)"
echo "wall            : $WALL_FINAL"
echo "mem (MB)        : $MEM_FINAL"
echo "dry run         : $DRY_RUN"
echo "================================================================"

BSUB_CMD=(
    bsub
    -G "$LSF_GROUP_FINAL"
    -q "$LSF_QUEUE_FINAL"
    -n 1
    -M "$MEM_FINAL"
    -R "select[mem>$MEM_FINAL] rusage[mem=$MEM_FINAL]"
    -W "$WALL_FINAL"
    -J "$JOB_NAME"
    -o "$LOG_OUT"
    -e "$LOG_ERR"
)

if [[ "$DRY_RUN" == "1" ]]; then
    printf '    %q ' "${BSUB_CMD[@]}"
    printf '< %q\n' "$JOB_SCRIPT"
    exit 0
fi

"${BSUB_CMD[@]}" < "$JOB_SCRIPT"
echo "submitted       : $JOB_NAME"
echo "tail logs       :"
echo "    tail -f $LOG_OUT"
echo "    tail -f $LOG_ERR"
