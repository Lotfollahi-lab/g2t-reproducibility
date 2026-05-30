#!/usr/bin/env bash
# submit_plot_spearman_g2t_vs_luna.sh
# -----------------------------------------------------------------------------
# Submit a tiny CPU-only LSF job that (optionally) recomputes
# extended_metrics.csv for each inference timestamp and then renders
# the G2T-vs-LUNA multi-metric comparison figure. Wraps:
#
#     analysis/benchmarking/plots/compute_extended_metrics.py
#     analysis/benchmarking/plots/plot_spearman_g2t_vs_luna.py
#
# The plot script reads the per-timestamp extended_metrics.csv written
# by the computation step; if those files already exist (from a prior
# run) you can skip recomputation with ``--skip_compute``.
#
# Sized for the normal queue (no GPU, 8 GB, 1 core, 30 min wall). The
# computation step is the bulk of the wall-clock — Spearman on the
# full N×N pairwise distance matrix per slice scales O(N²) and the
# MMC cortex has 30+ slices ≲ 5k cells each. Still finishes in
# seconds-to-minutes; bump --wall if you point this at CNS-scale data.
#
# Usage:
#     bash submit_plot_spearman_g2t_vs_luna.sh [OPTIONS]
#
# Options:
#   --dry_run                Print the bsub command without submitting.
#   --skip_compute           Skip the extended_metrics.csv computation
#                            step; go straight to plotting. Use when the
#                            files already exist and you just want to
#                            re-render the figure with new style /
#                            different timestamps.
#   --scgg_timestamps LIST   Comma-separated YYYYMMDD_HHMMSS timestamps
#                            to load from scgg_inference/ for G2T.
#                            Forwarded to BOTH the compute step and the
#                            plot step. Defaults to the hardcoded list
#                            in the .py.
#   --luna_timestamps LIST   Comma-separated YYYYMMDD_HHMMSS timestamps
#                            to load from luna_inference/. Defaults to
#                            auto-discovery (every matching subdir).
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
SCGG_TIMESTAMPS=""
LUNA_TIMESTAMPS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)
            DRY_RUN=1; shift ;;
        --skip_compute)
            SKIP_COMPUTE=1; shift ;;
        --scgg_timestamps)
            SCGG_TIMESTAMPS="${2:?--scgg_timestamps requires a value}"; shift 2 ;;
        --luna_timestamps)
            LUNA_TIMESTAMPS="${2:?--luna_timestamps requires a value}"; shift 2 ;;
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

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPRO_ROOT="$( cd -- "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
PLOT_SCRIPT="$REPRO_ROOT/analysis/benchmarking/plots/plot_spearman_g2t_vs_luna.py"
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
OUT_DIR="$ARTIFACTS_ROOT/mmc_luna/comparison_plots"
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
    echo "# Auto-generated by submit_plot_spearman_g2t_vs_luna.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "# Re-runnable: bash $JOB_SCRIPT"
    echo "set -euo pipefail"
    echo
    printf 'source %q\n' "$VENV_PATH/bin/activate"
    echo
    if [[ "$SKIP_COMPUTE" == "0" ]]; then
        echo "# Step 1: compute extended_metrics.csv per inference timestamp."
        # Build the compute-step argv. Pass timestamps when present so
        # the same scgg/luna seeds get scored that the plot will pull.
        # printf %q safely shell-quotes every value.
        printf 'python %q' "$COMPUTE_SCRIPT"
        if [[ -n "$SCGG_TIMESTAMPS" ]]; then
            printf ' --scgg_timestamps %q' "$SCGG_TIMESTAMPS"
        fi
        if [[ -n "$LUNA_TIMESTAMPS" ]]; then
            printf ' --luna_timestamps %q' "$LUNA_TIMESTAMPS"
        fi
        printf '\n'
        echo
    fi
    echo "# Step 2: render the multi-metric comparison figure."
    printf 'exec python %q' "$PLOT_SCRIPT"
    if [[ -n "$SCGG_TIMESTAMPS" ]]; then
        printf ' --scgg_timestamps %q' "$SCGG_TIMESTAMPS"
    fi
    if [[ -n "$LUNA_TIMESTAMPS" ]]; then
        printf ' --luna_timestamps %q' "$LUNA_TIMESTAMPS"
    fi
    printf '\n'
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

echo "================================================================"
echo "  submit_plot_spearman_g2t_vs_luna.sh"
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
[[ -n "$SCGG_TIMESTAMPS" ]] && echo "scgg_timestamps : $SCGG_TIMESTAMPS"
[[ -n "$LUNA_TIMESTAMPS" ]] && echo "luna_timestamps : $LUNA_TIMESTAMPS"
echo "dry run         : $DRY_RUN"
echo "================================================================"

BSUB_CMD=(
    bsub
    -G "$LSF_GROUP_FINAL"
    -q "$LSF_QUEUE_FINAL"
    -n 1
    -M 8000
    -R "select[mem>8000] rusage[mem=8000]"
    -W "00:30"
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
