#!/usr/bin/env bash
# submit_pipeline.sh
# -----------------------------------------------------------------------------
# Submit a SINGLE LSF job that runs one of scgg's three benchmarking
# pipelines (scgg / luna / novosparc) end to end. Wraps:
#
#   scripts/run_scgg_pipeline.py
#   scripts/run_luna_pipeline.py
#   scripts/run_novosparc_pipeline.py
#
# under a single bsub command so a one-off ablation is one line.
#
# Usage:
#   bash submit_pipeline.sh --method <scgg|luna|novosparc|celery> \
#                           --wandb_run_name <NAME> \
#                           [OPTIONS]
#
# Required:
#   --method M            scgg | luna | novosparc.
#   --wandb_run_name N    The wandb run name (also threaded into output
#                         dirs and used for the LSF job name).
#
# Data source (one is required):
#   --data_dir DIR        Silver h5ad directory (the usual silver path).
#   --train_csv FILE      Pre-built LUNA-format train CSV (pair w/ --test_csv).
#   --test_csv  FILE      Pre-built LUNA-format test CSV.
#
# Pipeline knobs (all optional; defaults inherited from the python
# pipeline scripts):
#   --epochs N            train.n_epochs override. Default: pipeline's
#                         own default (1000). Ignored for novosparc.
#   --seed S              general.seed override. Default: 0.
#   --n_inference_samples N  multi-sample inference budget. Ignored for
#                         luna (single-sample DDPM only).
#   --override "STR"      Single quoted string of space-separated
#                         key=value tokens for the training-time
#                         pipeline ``--override``. Example:
#                           --override "model.edm.embed_dim=16 train.lr=1e-4"
#                         Repeat the flag to append more tokens.
#   --inference_override "STR"  same, for inference-only overrides.
#                         Ignored for novosparc (single-process).
#   --wandb_project P     Optional wandb project override.
#   --wandb_mode M        disabled|online|offline|dryrun. Default: online.
#   --embedding_field F   adata.obsm key for pretrained gene embeddings
#                         (scgg only).
#   --skip_training       Skip the train block, run inference only on
#                         an existing checkpoint (luna only for now).
#                         REQUIRES --checkpoint=<path>. Common use:
#                         re-running inference with different
#                         dataset.num_workers when training succeeded
#                         but inference OOMed on a large dataset.
#   --checkpoint PATH     Existing .ckpt path to reuse. Paired with
#                         --skip_training.
#
# LSF resources (sensible defaults baked in; override per submission):
#   --queue / -q Q        LSF queue. Default: training-parallel for all
#                         three methods. Override via $LSF_QUEUE env
#                         var too.
#   --group / -g G        LSF cost-code group. Default: \$LSF_GROUP or
#                         the value baked in below.
#   --mem MB              memory cap in MB (rusage + select). Defaults
#                         per-method: 256000 for scgg/luna, 96000 for
#                         novosparc. Bumped 2026-05-28 from 128/64 GB
#                         because CNS-scale LUNA training peaked at
#                         ~516 GB RSS with 32 DataLoader workers; the
#                         128 GB cap routinely SIGTERM'd inference.
#                         For small datasets (MMC cortex) you can
#                         override down via --mem 64000 to schedule
#                         faster on shared queues.
#   --cores N             cores. Defaults: 24 for scgg/luna, 8 for novosparc.
#   --wall HH:MM          wall-clock cap. Defaults: 48:00 for scgg/luna,
#                         08:00 for novosparc. Bumped 2026-05-28 from
#                         24:00 / 04:00 to accommodate 2000-epoch
#                         training runs and full inference on
#                         CNS-scale data. Override down for small
#                         datasets via --wall.
#   --gpu STR             -gpu argument. Default for scgg/luna:
#                         mode=exclusive_process:num=1:block=yes.
#                         novosparc defaults to no GPU.
#   --gpu_model M         Restrict the scheduler to a specific GPU
#                         model family. M ∈ H200|H100|A100. Appends
#                         a ``gmem=N`` constraint to the -gpu spec:
#                           H200 → gmem=120000 (only H200 has ≥120 GB
#                                  on this cluster, so this uniquely
#                                  pins to H200 — the ~141 GB card).
#                           H100 → gmem=70000  (matches H100 AND H200
#                                  since gmem is a MINIMUM; for
#                                  H100-only use a manual --gpu spec
#                                  with an LSF -R "select[...]"
#                                  expression).
#                           A100 → gmem=35000  (matches A100-40,
#                                  A100-80, H100, H200 — same
#                                  minimum-only caveat).
#                         Use this when a known config peaks above
#                         80 GB GPU and you need to guarantee H200
#                         placement (e.g. CNS scgg with big slices).
#                         Trade-off: H200 queue is typically smaller
#                         than H100, so dispatch can be slower.
#
# Misc:
#   --dry_run             Print the bsub command WITHOUT submitting.
#   --help / -h           Print this usage block.
#
# Per-method default resources (each can be overridden via --mem etc):
#
#   method      queue              mem    cores  wall   gpu
#   ---------- ------------------ ------ ------ ------ -----
#   scgg        training-parallel  256000   24   48:00  1
#   luna        training-parallel  256000   24   48:00  1
#   novosparc   normal              96000    8   08:00  none
#
# NOTE: ``training-parallel`` is a GPU queue; LSF's ``esub`` validator
# rejects jobs that target it without a ``-gpu`` request. novosparc
# is CPU-only OT and doesn't use a GPU, so its default queue is
# ``normal`` instead. Override per-submission with ``--queue <name>``
# if your cluster's CPU queue has a different name.
#
# Examples:
#   # scgg with anisotropic gating on top of new defaults
#   bash submit_pipeline.sh \\
#       --method scgg \\
#       --wandb_run_name scgg_mmc_new_aniso \\
#       --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
#       --n_inference_samples 10 \\
#       --override "model.edm.anisotropic_gating=true"
#
#   # scgg with shape-matching loss + DiT backbone
#   bash submit_pipeline.sh \\
#       --method scgg \\
#       --wandb_run_name scgg_mmc_new_shape_dit \\
#       --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
#       --n_inference_samples 10 \\
#       --override "model.loss.shape_matching.enabled=true \\
#                   model.loss.shape_matching.weight=0.1 \\
#                   model.backbone=dit"
#
#   # LUNA baseline
#   bash submit_pipeline.sh \\
#       --method luna \\
#       --wandb_run_name luna_mmc_baseline \\
#       --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna
#
#   # novosparc baseline (lighter resources, no GPU)
#   bash submit_pipeline.sh \\
#       --method novosparc \\
#       --wandb_run_name novosparc_mmc_baseline \\
#       --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna
#
#   # Dry-run to inspect the bsub command:
#   bash submit_pipeline.sh --method scgg --wandb_run_name foo \\
#       --data_dir /nfs/... --dry_run
#
#   # Multi-seed sweep (run 5 jobs):
#   for s in 0 1 2 3 4; do
#     bash submit_pipeline.sh \\
#         --method scgg \\
#         --wandb_run_name scgg_mmc_new_baseline_seed${s} \\
#         --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
#         --seed $s \\
#         --n_inference_samples 10
#   done
#
# Outputs:
#   All run artifacts (checkpoints, plots, metrics, runtime.csv,
#   compute_requirements.csv) AND the LSF stdout/err for this job
#   land under the per-run dir:
#
#       <ARTIFACTS_ROOT>/<dataset>/<method>_model/<TS>/
#           best_model.ckpt
#           runtime.csv
#           compute_requirements.csv
#           per_slice_metrics.csv
#           aggregate_metrics.json
#           metrics.csv
#           plots/  (training-side, if --train_plots)
#           lsf_logs/
#               submit.out
#               submit.err
#
#       <ARTIFACTS_ROOT>/<dataset>/<method>_inference/<TS>/
#           runtime.csv
#           compute_requirements.csv
#           per_slice_metrics.csv
#           aggregate_metrics.json
#           metrics.csv
#           plots/
#           <section>/metadata_pred.csv
#
#   Both subtrees share the same <TS> so a training run and its
#   inference pair up by eye. For novosparc (no training subprocess),
#   everything including LSF logs lands under
#   <ARTIFACTS_ROOT>/<dataset>/novosparc_inference/<TS>/.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults shared across methods ---------------------------------------
LSF_GROUP_DEFAULT="${LSF_GROUP:-s10396}"

# Per-method default resources. The submitter picks one row based on
# --method, but each cell is overridable via the matching --flag.
DEFAULT_QUEUE_GPU="${LSF_QUEUE:-training-parallel}"
# CPU queue MUST be a non-GPU queue. The cluster's esub validator
# rejects jobs that target a GPU queue (e.g. training-parallel)
# without a ``-gpu`` request, and the novosparc method legitimately
# doesn't use a GPU (it's a CPU-only OT solver). ``normal`` is the
# most common non-GPU queue name; if your cluster uses a different
# name (research / batch / cpu-normal / etc.) override per-submission
# with ``--queue <name>`` or globally with ``LSF_QUEUE=<name>``.
DEFAULT_QUEUE_CPU="${LSF_QUEUE:-normal}"

# --- Help -----------------------------------------------------------------
print_help() {
    # Print the docstring (top comment block) up to the first non-comment line.
    awk 'NR>1 && /^[^#]/{exit} {print}' "$0"
}

# --- Parse args -----------------------------------------------------------
METHOD=""
RUN_NAME=""
DATA_DIR=""
TRAIN_CSV=""
TEST_CSV=""
EPOCHS=""
SEED=""
N_SAMPLES=""
OVERRIDE_STRING=""
INFERENCE_OVERRIDE_STRING=""
WANDB_PROJECT=""
WANDB_MODE=""
EMBEDDING_FIELD=""
TRAINING_MODE=""    # --training_mode multi_slice|per_reference (celery only)
HIDDEN_DIMS=""      # --hidden_dims "256 128 64" (celery only — 3 widths, space-sep)
BATCH_SIZE=""       # --batch_size N (celery only — overrides Fit_cord default of 4)
SKIP_TRAINING=""
CHECKPOINT=""
EXCLUDE_TEST_FILES=""

QUEUE_ARG=""
GROUP_ARG="$LSF_GROUP_DEFAULT"
MEM_ARG=""
CORES_ARG=""
WALL_ARG=""
GPU_ARG=""
GPU_MODEL_ARG=""    # --gpu_model H200|H100|A100 → adds gmem= constraint
DRY_RUN_ARG="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="${2:?--method requires a value}"; shift 2 ;;
        --wandb_run_name|--run_name)
            RUN_NAME="${2:?--wandb_run_name requires a value}"; shift 2 ;;
        --data_dir)
            DATA_DIR="${2:?--data_dir requires a value}"; shift 2 ;;
        --train_csv)
            TRAIN_CSV="${2:?--train_csv requires a value}"; shift 2 ;;
        --test_csv)
            TEST_CSV="${2:?--test_csv requires a value}"; shift 2 ;;
        --epochs)
            EPOCHS="${2:?--epochs requires a value}"; shift 2 ;;
        --seed)
            SEED="${2:?--seed requires a value}"; shift 2 ;;
        --n_inference_samples)
            N_SAMPLES="${2:?--n_inference_samples requires a value}"; shift 2 ;;
        --override)
            # Allow repeated --override; concatenate with a space so they
            # all end up in the single space-separated env var the runner
            # word-splits on.
            if [[ -z "$OVERRIDE_STRING" ]]; then
                OVERRIDE_STRING="${2:?--override requires a value}"
            else
                OVERRIDE_STRING="$OVERRIDE_STRING ${2:?--override requires a value}"
            fi
            shift 2 ;;
        --inference_override)
            if [[ -z "$INFERENCE_OVERRIDE_STRING" ]]; then
                INFERENCE_OVERRIDE_STRING="${2:?--inference_override requires a value}"
            else
                INFERENCE_OVERRIDE_STRING="$INFERENCE_OVERRIDE_STRING ${2:?--inference_override requires a value}"
            fi
            shift 2 ;;
        --wandb_project)
            WANDB_PROJECT="${2:?--wandb_project requires a value}"; shift 2 ;;
        --wandb_mode)
            WANDB_MODE="${2:?--wandb_mode requires a value}"; shift 2 ;;
        --embedding_field)
            EMBEDDING_FIELD="${2:?--embedding_field requires a value}"; shift 2 ;;
        --training_mode)
            TRAINING_MODE="${2:?--training_mode requires a value (multi_slice|per_reference)}"; shift 2 ;;
        --hidden_dims)
            HIDDEN_DIMS="${2:?--hidden_dims requires 3 space-separated widths, e.g. '256 128 64'}"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="${2:?--batch_size requires a positive integer (celery only)}"; shift 2 ;;
        --skip_training)
            SKIP_TRAINING=1; shift ;;
        --checkpoint)
            CHECKPOINT="${2:?--checkpoint requires a value}"; shift 2 ;;
        --exclude_test_files)
            EXCLUDE_TEST_FILES="${2:?--exclude_test_files requires a value}"; shift 2 ;;
        --queue|-q)
            QUEUE_ARG="${2:?--queue requires a value}"; shift 2 ;;
        --group|-g)
            GROUP_ARG="${2:?--group requires a value}"; shift 2 ;;
        --mem)
            MEM_ARG="${2:?--mem requires a value}"; shift 2 ;;
        --cores)
            CORES_ARG="${2:?--cores requires a value}"; shift 2 ;;
        --wall)
            WALL_ARG="${2:?--wall requires a value}"; shift 2 ;;
        --gpu)
            GPU_ARG="${2:?--gpu requires a value}"; shift 2 ;;
        --gpu_model)
            GPU_MODEL_ARG="${2:?--gpu_model requires a value (H200|H100|A100)}"; shift 2 ;;
        --dry_run)
            DRY_RUN_ARG=1; shift ;;
        --help|-h)
            print_help; exit 0 ;;
        *)
            echo "ERROR: unrecognised arg '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 2 ;;
    esac
done

# --- Validate -------------------------------------------------------------
if [[ -z "$METHOD" ]]; then
    echo "ERROR: --method is required." >&2; print_help; exit 2
fi
case "$METHOD" in
    scgg|luna|novosparc|celery) ;;
    *) echo "ERROR: --method must be scgg|luna|novosparc|celery; got '$METHOD'." >&2; exit 2 ;;
esac
if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --wandb_run_name is required." >&2; exit 2
fi

# Data-source validation (matches the python pipelines' own rule):
# exactly one of (--data_dir) or (--train_csv + --test_csv).
if [[ -z "$DATA_DIR" ]] && [[ -z "$TRAIN_CSV" && -z "$TEST_CSV" ]]; then
    echo "ERROR: must pass either --data_dir, or both --train_csv and --test_csv." >&2
    exit 2
fi
if [[ -n "$DATA_DIR" ]] && [[ -n "$TRAIN_CSV" || -n "$TEST_CSV" ]]; then
    echo "ERROR: --data_dir is mutually exclusive with --train_csv / --test_csv." >&2
    exit 2
fi
if [[ -n "$TRAIN_CSV" && -z "$TEST_CSV" ]] || [[ -z "$TRAIN_CSV" && -n "$TEST_CSV" ]]; then
    echo "ERROR: --train_csv and --test_csv must be passed together." >&2
    exit 2
fi
if [[ "$METHOD" == "novosparc" ]] && [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: novosparc pipeline only supports --data_dir (silver h5ad input)." >&2
    exit 2
fi

# Inference-only validation. Both the LUNA and scgg pipelines accept
# --skip_training + --checkpoint; novosparc has no training phase, so
# the flag is meaningless there. Flag confusion gets caught here so
# the user gets a clear error rather than a python-side argparse fail
# inside the LSF job.
if [[ -n "$SKIP_TRAINING" ]] && [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --skip_training requires --checkpoint=<path>." >&2
    exit 2
fi
if [[ -n "$SKIP_TRAINING" ]] && [[ "$METHOD" == "novosparc" ]]; then
    echo "ERROR: --skip_training is not applicable to --method=novosparc (no training phase). Use --method=scgg or --method=luna." >&2
    exit 2
fi
if [[ -n "$CHECKPOINT" ]] && [[ -z "$SKIP_TRAINING" ]]; then
    echo "ERROR: --checkpoint is only meaningful with --skip_training (otherwise the pipeline auto-discovers the trained ckpt)." >&2
    exit 2
fi

# --- Per-method defaults --------------------------------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Defaults sized for the LARGEST realistic dataset (CNS scRNA, ABCA
# Animal-1) so a single bsub-default invocation succeeds without
# per-run --mem / --wall tuning. Override down via --mem / --wall
# for small datasets (e.g. MMC cortex) if you care about scheduler
# priority — small jobs schedule faster on lightly-loaded queues
# when their resource request is small. Bumped 2026-05-28 from
# 128 GB / 24h to 256 GB / 48h after a CNS LUNA inference hit
# LSF SIGTERM with peak RSS ~500 GB at default 32 workers — see
# the matching ``dataset.num_workers`` fix in
# benchmarking/luna/utils/data/abstract_datatype.py.
#
# Default venv paths follow the setup_*_env.sh convention.
case "$METHOD" in
    scgg)
        DEFAULT_VENV="/nfs/team361/sb75/.venvs/scgg"
        DEFAULT_QUEUE="$DEFAULT_QUEUE_GPU"
        DEFAULT_MEM=256000
        DEFAULT_CORES=24
        DEFAULT_WALL="48:00"
        DEFAULT_GPU="mode=exclusive_process:num=1:block=yes"
        ;;
    luna)
        DEFAULT_VENV="/nfs/team361/sb75/.venvs/luna"
        DEFAULT_QUEUE="$DEFAULT_QUEUE_GPU"
        DEFAULT_MEM=256000
        DEFAULT_CORES=24
        DEFAULT_WALL="48:00"
        DEFAULT_GPU="mode=exclusive_process:num=1:block=yes"
        ;;
    novosparc)
        DEFAULT_VENV="/nfs/team361/sb75/.venvs/novosparc"
        DEFAULT_QUEUE="$DEFAULT_QUEUE_CPU"
        DEFAULT_MEM=96000
        DEFAULT_CORES=8
        DEFAULT_WALL="08:00"
        DEFAULT_GPU=""   # CPU-only; no -gpu arg
        ;;
    celery)
        # CeLEry does not call .to(device) in its public API and is
        # effectively CPU-only — see notes in setup_celery_env.sh.
        # Bigger MEM default than novosparc because CeLEry trains
        # N independent MLPs (one per test slice), so the working
        # set of in-memory AnnData copies adds up; with 31 MMC
        # test slices × ~5k cells × ~258 genes the RSS peak is
        # well under 32 GB. Bumped to 64 GB to give a safety margin.
        # Wall: per-slice training is ~3-5 min on CPU (500 epochs,
        # batch_size=4, 250-gene input), so 31 slices × 5 min ≈
        # 2.5 hours. 12 h is generous; bump for CNS if you point
        # this at the cns_luna silver dir.
        DEFAULT_VENV="/nfs/team361/sb75/.venvs/celery"
        DEFAULT_QUEUE="$DEFAULT_QUEUE_CPU"
        DEFAULT_MEM=64000
        DEFAULT_CORES=8
        DEFAULT_WALL="12:00"
        DEFAULT_GPU=""   # CPU-only — CeLEry doesn't move tensors to CUDA
        ;;
esac

# Resolve final values: CLI > matching env var > per-method default.
VENV_PATH="${VENV_PATH:-$DEFAULT_VENV}"
LSF_QUEUE_FINAL="${QUEUE_ARG:-$DEFAULT_QUEUE}"
LSF_GROUP_FINAL="${GROUP_ARG:-$LSF_GROUP_DEFAULT}"
LSF_MEM_FINAL="${MEM_ARG:-$DEFAULT_MEM}"
LSF_CORES_FINAL="${CORES_ARG:-$DEFAULT_CORES}"
LSF_WALL_FINAL="${WALL_ARG:-$DEFAULT_WALL}"
LSF_GPU_FINAL="${GPU_ARG:-$DEFAULT_GPU}"

# --gpu_model H200|H100|A100 → append the right ``gmem=N`` constraint
# to the -gpu spec. ``gmem=N`` means "give me a GPU with at least N
# MiB of memory" — LSF then filters its GPU pool accordingly. We use
# memory as the proxy because LSF on this cluster reports it
# reliably; the gmodel= field (the proper "model name" constraint)
# isn't populated consistently across hosts.
#
# Mapping (per-MiB):
#   H200 = 141 GB total → require gmem=120000 (only H200 has ≥120 GB,
#                         so this effectively pins to H200).
#   H100 = 80  GB total → require gmem=70000 (matches H100 AND H200
#                         since gmem is a MINIMUM; if you specifically
#                         need H100-only, use a manual --gpu string
#                         with an LSF -R "select[...]" expression
#                         instead).
#   A100 = 40 or 80 GB  → require gmem=35000 (matches A100-40, A100-80,
#                         H100, H200 — same minimum-only caveat).
#
# Skip silently if GPU_MODEL_ARG is empty (default behaviour =
# whichever GPU the scheduler hands out, same as before this flag).
# Skip with a warning if LSF_GPU_FINAL is empty (CPU-only methods
# like novosparc / celery don't have a -gpu spec to augment).
if [[ -n "$GPU_MODEL_ARG" ]]; then
    if [[ -z "$LSF_GPU_FINAL" ]]; then
        echo "WARN  : --gpu_model=$GPU_MODEL_ARG ignored — this method is CPU-only (no -gpu spec to constrain)." >&2
    else
        case "$GPU_MODEL_ARG" in
            H200|h200) LSF_GPU_FINAL="${LSF_GPU_FINAL}:gmem=120000" ;;
            H100|h100) LSF_GPU_FINAL="${LSF_GPU_FINAL}:gmem=70000"  ;;
            A100|a100) LSF_GPU_FINAL="${LSF_GPU_FINAL}:gmem=35000"  ;;
            *)
                echo "ERROR: --gpu_model must be one of H200|H100|A100. Got '$GPU_MODEL_ARG'." >&2
                exit 2
                ;;
        esac
    fi
fi

# Repo root: the lsf/ dir is at <repo_root>/scgg-reproducibility/analysis/benchmarking/lsf/,
# and the scgg monorepo (the one holding scripts/run_*_pipeline.py) is the
# sibling. Resolve to absolute path so the runner can `cd` reliably.
SCGG_REPO_DEFAULT="${SCGG_REPO:-/nfs/team361/sb75/scgg}"
SCGG_REPO_FINAL="$SCGG_REPO_DEFAULT"

# Pre-compute the per-run TIMESTAMP and forward it to the pipeline
# via --run_timestamp. This pins the training artifacts dir AND the
# inference artifacts dir to the same TS — so a whole run lives in
# ONE directory. The LSF log dir tracks the same TS in the normal
# train→infer flow.
#
# --skip_training recovery path: regex-extract the TS from
# --checkpoint so inference artifacts land under
# scgg_inference/<old_TS>/, pairing with the original training's
# scgg_model/<old_TS>/. But the LSF logs need a SEPARATE fresh TS
# to avoid clobbering the original training's lsf_logs/submit.{out,err}
# — handled below where LOG_DIR is computed.
FRESH_TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$SKIP_TRAINING" ]] && [[ -n "$CHECKPOINT" ]]; then
    # The first YYYYMMDD_HHMMSS-shaped token in the path is the
    # right one — the pipeline writes
    # <ARTIFACTS>/<dataset>/scgg_model/<TS>/best_model.ckpt, so the
    # TS is the directory immediately above the .ckpt.
    INHERITED_TS="$(echo "$CHECKPOINT" | grep -oE '[0-9]{8}_[0-9]{6}(_[A-Za-z0-9]+)?' | head -1)"
    if [[ -n "$INHERITED_TS" ]]; then
        RUN_TS="$INHERITED_TS"
        IS_RECOVERY=1
        echo "INFO  : --skip_training set; inheriting RUN_TS=$RUN_TS from --checkpoint path"
        echo "        Inference artifacts → scgg_inference/$RUN_TS/  (pairs with original training)"
        echo "        Recovery LSF logs    → scgg_model/$RUN_TS/lsf_logs/recovery_$FRESH_TS/"
    else
        RUN_TS="$FRESH_TS"
        IS_RECOVERY=0
        echo "WARN  : --skip_training set but no YYYYMMDD_HHMMSS in --checkpoint path; using fresh RUN_TS=$RUN_TS" >&2
    fi
else
    RUN_TS="$FRESH_TS"
    IS_RECOVERY=0
fi
# Sanitise run name for filesystem use (forward-slashes in particular).
RUN_NAME_SAFE="${RUN_NAME//\//_}"

# Compute the artifacts root (matches the python pipeline's default).
# Honour the same env var the pipeline reads — so if the user sets
# SCGG_ARTIFACTS_ROOT for the pipeline, the submitter's log dir
# follows automatically.
ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"

# Derive the dataset name the same way the python pipeline does
# (run_scgg_pipeline._dataset_name_from_args).
if [[ -n "$DATA_DIR" ]]; then
    DATASET_NAME="$(basename "$DATA_DIR")"
elif [[ -n "$TRAIN_CSV" ]]; then
    DATASET_NAME="$(basename "$(dirname "$(realpath "$TRAIN_CSV")")")"
    [[ -z "$DATASET_NAME" ]] && DATASET_NAME="luna_paper_csvs"
else
    DATASET_NAME="unknown"
fi

# Per-method subdir name on disk. Mirrors the python pipelines'
# ENGINE_OUTPUT_SUBDIR convention.
case "$METHOD" in
    scgg)       PIPELINE_SUBDIR="scgg_model" ;;
    luna)       PIPELINE_SUBDIR="luna_model" ;;
    novosparc)  PIPELINE_SUBDIR="novosparc_inference" ;;
    celery)     PIPELINE_SUBDIR="celery_model" ;;
esac

# LSF stdout/err. Always co-located with the pipeline output for the
# same run — outputs ONLY land in <dataset>/<method>_(model|inference)/<TS>/
# and never in a separate top-level tree. PIPELINE_SUBDIR is set
# above to ``<method>_model`` for scgg+luna (the training subprocess
# is the "main" output dir) and ``novosparc_inference`` for novosparc
# (which has no separate training phase).
#
# Recovery flow (--skip_training + inherited TS): put the LSF logs
# under a ``recovery_<FRESH_TS>/`` subfolder of the inherited TS's
# lsf_logs dir, so they don't clobber the original training run's
# submit.out / submit.err. The artifacts themselves still land
# under scgg_inference/<inherited_TS>/ via SCGG_RUN_TIMESTAMP below.
if [[ "$IS_RECOVERY" == "1" ]]; then
    LOG_DIR="$ARTIFACTS_ROOT/$DATASET_NAME/$PIPELINE_SUBDIR/$RUN_TS/lsf_logs/recovery_$FRESH_TS"
else
    LOG_DIR="$ARTIFACTS_ROOT/$DATASET_NAME/$PIPELINE_SUBDIR/$RUN_TS/lsf_logs"
fi
mkdir -p "$LOG_DIR"

# Verify runner is present.
RUNNER="$SCRIPT_DIR/_run_pipeline.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing runner at $RUNNER" >&2
    exit 1
fi

# --- Print summary --------------------------------------------------------
echo "================================================================"
echo "  submit_pipeline.sh"
echo "================================================================"
echo "method        : $METHOD"
echo "wandb_run     : $RUN_NAME"
[[ -n "$DATA_DIR" ]] && echo "data_dir      : $DATA_DIR"
[[ -n "$TRAIN_CSV" ]] && echo "train_csv     : $TRAIN_CSV"
[[ -n "$TEST_CSV" ]] && echo "test_csv      : $TEST_CSV"
[[ -n "$EPOCHS" ]] && echo "epochs        : $EPOCHS"
[[ -n "$SEED" ]] && echo "seed          : $SEED"
[[ -n "$N_SAMPLES" ]] && echo "n_inference_samples : $N_SAMPLES"
[[ -n "$OVERRIDE_STRING" ]] && echo "override      : $OVERRIDE_STRING"
[[ -n "$INFERENCE_OVERRIDE_STRING" ]] && echo "inference_override : $INFERENCE_OVERRIDE_STRING"
[[ -n "$WANDB_PROJECT" ]] && echo "wandb_project : $WANDB_PROJECT"
[[ -n "$WANDB_MODE" ]] && echo "wandb_mode    : $WANDB_MODE"
[[ -n "$EMBEDDING_FIELD" ]] && echo "embedding_field : $EMBEDDING_FIELD"
[[ -n "$TRAINING_MODE" ]] && echo "training_mode : $TRAINING_MODE"
[[ -n "$HIDDEN_DIMS" ]] && echo "hidden_dims : $HIDDEN_DIMS"
[[ -n "$SKIP_TRAINING" ]] && echo "skip_training : 1 (inference-only)"
[[ -n "$CHECKPOINT" ]] && echo "checkpoint    : $CHECKPOINT"
[[ -n "$EXCLUDE_TEST_FILES" ]] && echo "exclude_test_files : $EXCLUDE_TEST_FILES"
[[ -n "$BATCH_SIZE"          ]] && echo "batch_size         : $BATCH_SIZE"
echo "----------------------------------------------------------------"
echo "repo          : $SCGG_REPO_FINAL"
echo "venv          : $VENV_PATH"
echo "run timestamp : $RUN_TS"
echo "per-run dir   : $ARTIFACTS_ROOT/$DATASET_NAME/$PIPELINE_SUBDIR/$RUN_TS/"
echo "log dir       : $LOG_DIR"
echo "----------------------------------------------------------------"
echo "LSF resources :"
echo "  queue       : $LSF_QUEUE_FINAL"
echo "  group       : $LSF_GROUP_FINAL"
echo "  cores       : $LSF_CORES_FINAL"
echo "  mem (MB)    : $LSF_MEM_FINAL"
echo "  wall        : $LSF_WALL_FINAL"
if [[ -n "$LSF_GPU_FINAL" ]]; then
    echo "  gpu         : $LSF_GPU_FINAL"
    [[ -n "$GPU_MODEL_ARG" ]] && echo "  gpu_model   : $GPU_MODEL_ARG (forces gmem= constraint above)"
else
    echo "  gpu         : (none — CPU-only)"
fi
echo "dry run       : $DRY_RUN_ARG"
echo "================================================================"

# --- Build the bsub command ----------------------------------------------
JOB_NAME="${METHOD}-${RUN_NAME_SAFE}-${RUN_TS}"
LOG_OUT="$LOG_DIR/submit.out"
LOG_ERR="$LOG_DIR/submit.err"

# Per-job script that sets env vars and execs the runner. We DON'T
# use ``bsub -env`` because LSF's -env parser rejects values that
# contain embedded = signs or spaces — and our SCGG_OVERRIDE values
# regularly look like "model.edm.embed_dim=16 train.lr=1e-4", which
# has both. ``printf %q`` shell-quotes every value safely, so a
# job.sh file is the bullet-proof transport.
#
# The job.sh file is retained at lsf_logs/job.sh so you can re-run
# this exact configuration manually by sourcing it.
JOB_SCRIPT="$LOG_DIR/job.sh"
{
    echo "#!/usr/bin/env bash"
    echo "# Auto-generated by submit_pipeline.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "# Re-runnable: bash $JOB_SCRIPT"
    echo "set -euo pipefail"
    echo
    # Always-set vars. Empty values are still exported (the runner
    # checks ``-n \"\${VAR:-}\"`` so empties become no-ops downstream
    # — they're not equivalent to "unset" but the runner treats them
    # the same way).
    printf 'export SCGG_RUN_NAME=%q\n'           "$RUN_NAME"
    printf 'export SCGG_RUN_TIMESTAMP=%q\n'      "$RUN_TS"
    printf 'export SCGG_DATA_DIR=%q\n'           "$DATA_DIR"
    printf 'export SCGG_TRAIN_CSV=%q\n'          "$TRAIN_CSV"
    printf 'export SCGG_TEST_CSV=%q\n'           "$TEST_CSV"
    printf 'export SCGG_EPOCHS=%q\n'             "$EPOCHS"
    printf 'export SCGG_SEED=%q\n'               "$SEED"
    printf 'export SCGG_N_INFERENCE_SAMPLES=%q\n' "$N_SAMPLES"
    printf 'export SCGG_OVERRIDE=%q\n'           "$OVERRIDE_STRING"
    printf 'export SCGG_INFERENCE_OVERRIDE=%q\n' "$INFERENCE_OVERRIDE_STRING"
    printf 'export SCGG_WANDB_PROJECT=%q\n'      "$WANDB_PROJECT"
    printf 'export SCGG_WANDB_MODE=%q\n'         "$WANDB_MODE"
    printf 'export SCGG_EMBEDDING_FIELD=%q\n'    "$EMBEDDING_FIELD"
    printf 'export SCGG_TRAINING_MODE=%q\n'      "$TRAINING_MODE"
    printf 'export SCGG_HIDDEN_DIMS=%q\n'        "$HIDDEN_DIMS"
    printf 'export SCGG_BATCH_SIZE=%q\n'         "$BATCH_SIZE"
    # Inference-only mode: when SKIP_TRAINING is set, the runner
    # forwards --skip_training + --checkpoint to the python pipeline,
    # which then skips the train block and reuses the provided ckpt.
    # Only currently supported by run_luna_pipeline.py; scgg/novosparc
    # will reject the flag if added later there.
    printf 'export SCGG_SKIP_TRAINING=%q\n'      "$SKIP_TRAINING"
    printf 'export SCGG_CHECKPOINT=%q\n'         "$CHECKPOINT"
    printf 'export SCGG_EXCLUDE_TEST_FILES=%q\n' "$EXCLUDE_TEST_FILES"
    # Also forward the artifacts root so the runner / pipeline reads
    # the same place the submitter wrote logs to.
    printf 'export SCGG_ARTIFACTS_ROOT=%q\n'     "$ARTIFACTS_ROOT"
    # Optional diagnostic passthrough: when the submitter exports
    # SCGG_LOG_MDS_VAR (a CSV path), forward it so edm_head.py logs the
    # top-2 MDS eigenvalue variance-explained per (reverse step, slice).
    # Used for the one-off "how 2-D-embeddable is D_hat?" measurement
    # (analyse_mds_var.py). Off by default — no effect on normal runs.
    if [[ -n "${SCGG_LOG_MDS_VAR:-}" ]]; then
        printf 'export SCGG_LOG_MDS_VAR=%q\n'    "$SCGG_LOG_MDS_VAR"
    fi

    # Match BLAS/MKL/OpenMP thread counts to the LSF -n value so
    # PyTorch's CPU matmul ACTUALLY uses all the cores we requested.
    # Without this, PyTorch picks its own default (often 1 or
    # ``physical_cores/2``) and ignores the LSF allocation — bumping
    # ``--cores 32`` produces no speedup. Especially important for
    # CeLEry (CPU-only) on CNS where the first MLP layer is large
    # enough that the matmul actually parallelises across threads.
    # OMP_NUM_THREADS covers PyTorch's intra-op threads;
    # MKL_NUM_THREADS covers Intel MKL (numpy / scikit-learn);
    # OPENBLAS_NUM_THREADS covers the OpenBLAS path.
    printf 'export OMP_NUM_THREADS=%q\n'         "$LSF_CORES_FINAL"
    printf 'export MKL_NUM_THREADS=%q\n'         "$LSF_CORES_FINAL"
    printf 'export OPENBLAS_NUM_THREADS=%q\n'    "$LSF_CORES_FINAL"
    echo
    echo "# Hand off to the runner, which activates the venv and"
    echo "# launches the python pipeline."
    printf 'exec bash %q %q %q %q\n' \
        "$RUNNER" "$METHOD" "$VENV_PATH" "$SCGG_REPO_FINAL"
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

BSUB_CMD=(
    bsub
    -G "$LSF_GROUP_FINAL"
    -q "$LSF_QUEUE_FINAL"
    -n "$LSF_CORES_FINAL"
    -M "$LSF_MEM_FINAL"
    -R "select[mem>$LSF_MEM_FINAL] rusage[mem=$LSF_MEM_FINAL]"
    -R "span[ptile=$LSF_CORES_FINAL]"
    -W "$LSF_WALL_FINAL"
    -J "$JOB_NAME"
    -o "$LOG_OUT"
    -e "$LOG_ERR"
)
# Only add -gpu if we have one (novosparc doesn't).
if [[ -n "$LSF_GPU_FINAL" ]]; then
    BSUB_CMD+=(-gpu "$LSF_GPU_FINAL")
fi

# --- Submit -----------------------------------------------------------------
echo "submitting    :"
echo "    job script  : $JOB_SCRIPT"
if [[ "$DRY_RUN_ARG" == "1" ]]; then
    # Pretty-print the command without actually submitting.
    printf '    %q ' "${BSUB_CMD[@]}"
    printf '%q %q\n' "bash" "$JOB_SCRIPT"
    echo
    echo "(dry-run; not submitted.)"
    echo
    echo "--- job.sh contents (first 30 lines) ---"
    head -30 "$JOB_SCRIPT"
    exit 0
fi

"${BSUB_CMD[@]}" bash "$JOB_SCRIPT"

echo
echo "================================================================"
echo "submitted job : $JOB_NAME"
echo "watch with    : bjobs"
echo "kill with     : bkill -J '$JOB_NAME'"
echo "logs:"
echo "  $LOG_OUT"
echo "  $LOG_ERR"
echo "per-run dir   : <ARTIFACTS>/<dataset>/${METHOD}_(model|inference)/<TS>/"
echo "                (timestamped inside the python pipeline)"
echo "================================================================"
