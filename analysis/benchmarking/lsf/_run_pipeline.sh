#!/usr/bin/env bash
# _run_pipeline.sh
# -----------------------------------------------------------------------------
# Inside-the-LSF-job runner. NOT invoked directly by humans — used by
# ``submit_pipeline.sh`` as the body of the ``bsub ... bash _run_pipeline.sh``
# command.
#
# Reads its configuration from environment variables (set by the
# submitter via bsub -env) so the positional-args surface stays
# minimal:
#
#   $1  METHOD        scgg | luna | novosparc
#   $2  VENV_PATH     /nfs/team361/sb75/.venvs/<method>
#   $3  REPO_ROOT     /nfs/team361/sb75/scgg
#
# Env vars (all optional; forwarded by the submitter):
#
#   SCGG_RUN_NAME             passed to --wandb_run_name (required for
#                             a meaningful wandb label; submit script
#                             defaults it before we get here)
#   SCGG_DATA_DIR             passed to --data_dir
#   SCGG_TRAIN_CSV            passed to --train_csv (mutually exclusive
#                             with SCGG_DATA_DIR; submit script enforces)
#   SCGG_TEST_CSV             passed to --test_csv
#   SCGG_EPOCHS               passed to --epochs (ignored for novosparc;
#                             it has no training step)
#   SCGG_SEED                 passed to --seed
#   SCGG_N_INFERENCE_SAMPLES  passed to --n_inference_samples (ignored
#                             for luna — single-sample DDPM only)
#   SCGG_OVERRIDE             space-separated key=value tokens for the
#                             pipeline's ``--override`` flag. Forwarded
#                             as multiple args via word-splitting.
#                             Example: "model.edm.embed_dim=16 train.lr=1e-4"
#   SCGG_INFERENCE_OVERRIDE   same, for ``--inference_override``
#                             (ignored for novosparc — single-process)
#   SCGG_WANDB_PROJECT        passed to --wandb_project
#   SCGG_WANDB_MODE           passed to --wandb_mode
#   SCGG_EMBEDDING_FIELD      passed to --embedding_field (scgg only)
#   SCGG_TRAINING_MODE        passed to --training_mode (celery only).
#                             multi_slice (LUNA Supp Note 2 protocol) or
#                             per_reference (per-test-slice ref-based
#                             training).
#   SCGG_SKIP_TRAINING        if non-empty, append --skip_training
#                             (scgg + luna; novosparc has no training
#                             phase so the flag is rejected upstream)
#   SCGG_CHECKPOINT           paired with SCGG_SKIP_TRAINING — the .ckpt
#                             the pipeline reuses instead of training a
#                             fresh model (scgg + luna)
#   SCGG_EXCLUDE_TEST_FILES   comma-separated *_test.h5ad basenames to
#                             drop from the assembled test set (scgg
#                             + luna; novosparc has no equivalent).
#                             Useful for skipping too-large slices
#                             that GPU-OOM during inference.
#
# The runner activates the venv, switches to the repo root, and execs
# the appropriate ``run_<method>_pipeline.py`` with the assembled args.
# ``exec`` so the python process replaces the shell — LSF reports the
# pipeline's exit code as the job exit code without a wrapper layer.
# -----------------------------------------------------------------------------

set -euo pipefail

METHOD="${1:?ERROR: METHOD (positional arg 1) required}"
VENV_PATH="${2:?ERROR: VENV_PATH (positional arg 2) required}"
REPO_ROOT="${3:?ERROR: REPO_ROOT (positional arg 3) required}"

echo "================================================================"
echo "  _run_pipeline.sh"
echo "================================================================"
echo "host          : $(hostname)"
echo "date          : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "method        : $METHOD"
echo "venv          : $VENV_PATH"
echo "repo          : $REPO_ROOT"
echo "----------------------------------------------------------------"

# --- Activate venv -----------------------------------------------------------
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: venv activate script not found at $VENV_PATH/bin/activate" >&2
    echo "ERROR: run setup_${METHOD}_env.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
echo "python        : $(python --version) -> $(which python)"

# --- cd into the repo --------------------------------------------------------
if [[ ! -d "$REPO_ROOT/scripts" ]]; then
    echo "ERROR: REPO_ROOT=$REPO_ROOT does not contain scripts/" >&2
    exit 1
fi
cd "$REPO_ROOT"

# --- Pick pipeline script ----------------------------------------------------
case "$METHOD" in
    scgg)
        SCRIPT="scripts/run_scgg_pipeline.py"
        ;;
    luna)
        SCRIPT="scripts/run_luna_pipeline.py"
        ;;
    novosparc)
        SCRIPT="scripts/run_novosparc_pipeline.py"
        ;;
    celery)
        SCRIPT="scripts/run_celery_pipeline.py"
        ;;
    *)
        echo "ERROR: unknown METHOD='$METHOD'. Expected scgg|luna|novosparc|celery." >&2
        exit 1
        ;;
esac

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: pipeline script missing: $REPO_ROOT/$SCRIPT" >&2
    exit 1
fi

# --- Assemble pipeline args from env -----------------------------------------
ARGS=()

# Data source — one of (--data_dir) or (--train_csv + --test_csv). The
# submitter validates that exactly one mode is supplied; here we just
# forward whichever variables are set.
if [[ -n "${SCGG_DATA_DIR:-}" ]]; then
    ARGS+=(--data_dir "$SCGG_DATA_DIR")
fi
if [[ -n "${SCGG_TRAIN_CSV:-}" ]]; then
    ARGS+=(--train_csv "$SCGG_TRAIN_CSV")
fi
if [[ -n "${SCGG_TEST_CSV:-}" ]]; then
    ARGS+=(--test_csv "$SCGG_TEST_CSV")
fi

# Run name (the only flag we strongly recommend always setting).
if [[ -n "${SCGG_RUN_NAME:-}" ]]; then
    ARGS+=(--wandb_run_name "$SCGG_RUN_NAME")
fi

# Pinned timestamp from the submitter (so LSF logs + on-disk artifacts
# + wandb config all share the same per-run directory and label). All
# three pipelines now accept --run_timestamp (scgg + luna + novosparc).
# When SCGG_RUN_TIMESTAMP is empty, each pipeline falls back to its own
# wall-clock generation — so the runner stays usable for ad-hoc local
# runs that bypass the submitter.
if [[ -n "${SCGG_RUN_TIMESTAMP:-}" ]]; then
    ARGS+=(--run_timestamp "$SCGG_RUN_TIMESTAMP")
fi

# wandb project / mode.
if [[ -n "${SCGG_WANDB_PROJECT:-}" ]]; then
    ARGS+=(--wandb_project "$SCGG_WANDB_PROJECT")
fi
if [[ -n "${SCGG_WANDB_MODE:-}" ]]; then
    ARGS+=(--wandb_mode "$SCGG_WANDB_MODE")
fi

# Method-specific arg gating: novosparc has no training-time flags;
# luna has no multi-sample inference.
if [[ -n "${SCGG_SEED:-}" ]]; then
    ARGS+=(--seed "$SCGG_SEED")
fi
if [[ "$METHOD" != "novosparc" ]] && [[ -n "${SCGG_EPOCHS:-}" ]]; then
    ARGS+=(--epochs "$SCGG_EPOCHS")
fi
if [[ "$METHOD" != "luna" ]] && [[ -n "${SCGG_N_INFERENCE_SAMPLES:-}" ]]; then
    ARGS+=(--n_inference_samples "$SCGG_N_INFERENCE_SAMPLES")
fi

# scgg-only: pretrained-embedding obsm field.
if [[ "$METHOD" == "scgg" ]] && [[ -n "${SCGG_EMBEDDING_FIELD:-}" ]]; then
    ARGS+=(--embedding_field "$SCGG_EMBEDDING_FIELD")
fi

# celery-only: training mode (multi_slice vs per_reference). The
# python pipeline's argparse rejects --training_mode for any other
# method, so we gate on METHOD here.
if [[ "$METHOD" == "celery" ]] && [[ -n "${SCGG_TRAINING_MODE:-}" ]]; then
    ARGS+=(--training_mode "$SCGG_TRAINING_MODE")
fi

# Inference-only mode. Both LUNA and scgg pipelines accept
# --skip_training + --checkpoint; novosparc is gated out because it
# has no training phase to skip. Upstream validator in
# submit_pipeline.sh enforces the same rule.
if [[ "$METHOD" != "novosparc" ]] && [[ -n "${SCGG_SKIP_TRAINING:-}" ]]; then
    ARGS+=(--skip_training)
fi
if [[ "$METHOD" != "novosparc" ]] && [[ -n "${SCGG_CHECKPOINT:-}" ]]; then
    ARGS+=(--checkpoint "$SCGG_CHECKPOINT")
fi
# Per-file test-set exclusion. Consumed by BOTH the LUNA pipeline
# (→ run_luna_inference → run_benchmark) AND the scgg pipeline
# (→ run_scgg_inference → run_scgg_train.run_benchmark), which
# filter *_test.h5ad files by basename before building test.csv.
# novosparc has no equivalent flag — its pipeline is single-process
# and doesn't go through a *_test.h5ad CSV-building step the same way.
if [[ "$METHOD" != "novosparc" ]] && [[ -n "${SCGG_EXCLUDE_TEST_FILES:-}" ]]; then
    ARGS+=(--exclude_test_files "$SCGG_EXCLUDE_TEST_FILES")
fi

# Override strings — split on whitespace into multiple args. The user
# passed something like SCGG_OVERRIDE="model.edm.embed_dim=16 train.lr=1e-4";
# the pipeline's --override is action="extend" nargs="+" so it accepts
# the multiple tokens directly.
#
# Gated to scgg|luna only — these are Hydra-config overrides
# routed into the vendored LUNA engine. CeLEry has no Hydra config;
# novosparc has no Hydra config either. If the user accidentally
# passes --override for those methods, argparse would reject the
# unknown flag here. Better to silently drop with a warning so a
# pasted-from-another-method command doesn't crash the job.
if [[ -n "${SCGG_OVERRIDE:-}" ]]; then
    case "$METHOD" in
        scgg|luna)
            # shellcheck disable=SC2206  # intentional word-splitting
            OVERRIDE_TOKENS=( $SCGG_OVERRIDE )
            ARGS+=(--override "${OVERRIDE_TOKENS[@]}")
            ;;
        *)
            echo "WARN  : --override is only supported by scgg/luna; dropping for $METHOD: $SCGG_OVERRIDE" >&2
            ;;
    esac
fi
if [[ -n "${SCGG_INFERENCE_OVERRIDE:-}" ]]; then
    case "$METHOD" in
        scgg|luna)
            # shellcheck disable=SC2206
            INFER_TOKENS=( $SCGG_INFERENCE_OVERRIDE )
            ARGS+=(--inference_override "${INFER_TOKENS[@]}")
            ;;
        *)
            echo "WARN  : --inference_override is only supported by scgg/luna; dropping for $METHOD: $SCGG_INFERENCE_OVERRIDE" >&2
            ;;
    esac
fi

# --- Print + run -------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "command       : python $SCRIPT \\"
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == --* ]]; then
        echo "    ${ARGS[$i]} \\"
    else
        # value for the prior flag — combine on one line for readability
        # (this is purely for the log).
        echo "        ${ARGS[$i]} \\"
    fi
done
echo "================================================================"

# `exec` so the LSF job's exit code is whatever the pipeline returns.
exec python "$SCRIPT" "${ARGS[@]}"
