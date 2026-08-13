#!/usr/bin/env bash
# submit_subsample_robustness.sh
# -----------------------------------------------------------------------------
# Submit the cell-subsampling robustness conditions (reviewer R2: how robust is
# G2T to the number and relative abundance of profiled cells?) as LSF jobs.
#
# Inference-only against an EXISTING checkpoint -- no retraining. One job per
# (condition, seed); each job is a normal `submit_pipeline.sh --method scgg
# --skip_training` run, so environment activation, GPU spec and logging all come
# from the house submitter rather than being reinvented here.
#
# WHY EACH JOB GETS ITS OWN SCGG_ARTIFACTS_ROOT
#   With --skip_training the run timestamp is derived from the CHECKPOINT path
#   (submit_pipeline.sh ~:505-512). Every condition uses the same checkpoint, so
#   all of them would resolve to the SAME <ARTIFACTS>/<ds>/scgg_inference/<TS>/
#   and silently overwrite each other. Worse, run_scgg_train.py:1896-1910 REUSES
#   work/{train,test}.csv when present, so a second condition landing in a
#   populated directory would re-score the FIRST condition's cells and report it
#   as a new result. Pinning SCGG_ARTIFACTS_ROOT per (condition, seed) makes
#   both failure modes impossible.
#
# The scorer takes a per-run root and rglobs for metadata_pred.csv, so it does
# not need to know the timestamp -- but that also means one root must contain
# exactly ONE run, hence per-seed roots too.
#
# USAGE
#   bash submit_subsample_robustness.sh \
#       --cond_root  /nfs/.../robustness/arm_number \
#       --checkpoint /nfs/.../scgg_model/<TS>/luna_run/checkpoints/epoch=999.ckpt \
#       --train_csv  /nfs/.../scgg_model/<TS>/work/train.csv \
#       --seeds "0"
#
#   Reference and thinnest conditions want >=3 seeds (per-cell noise cannot be
#   held fixed across conditions -- N differs, so the RNG stream diverges), e.g.
#       --only uniform_rest100,uniform_rest010 --seeds "0 1 2"
#
#   DRY_RUN=1 bash submit_subsample_robustness.sh ...   # print, do not submit
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-$HERE/../lsf/submit_pipeline.sh}"
RUNS_ROOT="${RUNS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts/robustness/runs}"

COND_ROOT=""; CHECKPOINT=""; TRAIN_CSV=""; ONLY=""
SEEDS="0"
SUBMIT_DELAY="${SUBMIT_DELAY:-2}"
DRY_RUN="${DRY_RUN:-0}"
PASSTHRU=()          # --mem/--cores/--wall/--queue/--gpu/... forwarded verbatim

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cond_root)  COND_ROOT="${2:?}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
    --train_csv)  TRAIN_CSV="${2:?}"; shift 2 ;;
    --seeds)      SEEDS="${2:?}"; shift 2 ;;
    --only)       ONLY="${2:?}"; shift 2 ;;
    --runs_root)  RUNS_ROOT="${2:?}"; shift 2 ;;
    --dry_run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,44p' "$0"; exit 0 ;;
    *)            PASSTHRU+=( "$1" ); shift ;;
  esac
done

# --- validation --------------------------------------------------------------
[[ -n "$COND_ROOT"  ]] || { echo "ERROR: --cond_root is required."  >&2; exit 2; }
[[ -n "$CHECKPOINT" ]] || { echo "ERROR: --checkpoint is required." >&2; exit 2; }
[[ -n "$TRAIN_CSV"  ]] || { echo "ERROR: --train_csv is required."  >&2; exit 2; }
[[ -f "$SUBMIT"     ]] || { echo "ERROR: submitter not found: $SUBMIT" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$TRAIN_CSV"  ]] || { echo "ERROR: train_csv not found: $TRAIN_CSV" >&2; exit 2; }
[[ -f "$COND_ROOT/E_manifest.csv" ]] || {
  echo "ERROR: $COND_ROOT/E_manifest.csv missing." >&2
  echo "       Run make_subsample_conditions.py first — the scorer needs the" >&2
  echo "       fixed evaluation set, and without it these runs are unusable." >&2
  exit 2; }

# The checkpoint's epoch index is parsed from its FILENAME (scgg src/main.py:110
# int(path.split("=")[-1].split(".")[0])), so a name without epoch=<int>.ckpt
# fails deep inside the job rather than here.
[[ "$(basename "$CHECKPOINT")" =~ epoch=[0-9]+\.ckpt$ ]] || {
  echo "ERROR: checkpoint basename must match 'epoch=<int>.ckpt'; got" >&2
  echo "       $(basename "$CHECKPOINT")" >&2
  echo "       (a best_model.ckpt symlink is fine only if you resolve it first:" >&2
  echo "        readlink -f <path>)" >&2
  exit 2; }

# Model config is restored from <ckpt>/../../.hydra/config.yaml; without it the
# run silently falls back to the DEFAULT model config, which diverges from
# training if any model.* override was used.
HYDRA_CFG="$(dirname "$CHECKPOINT")/../.hydra/config.yaml"
if [[ -f "$HYDRA_CFG" ]]; then
  if grep -qE "batch_covariates:[[:space:]]*true" "$HYDRA_CFG"; then
    cat >&2 <<'EOF'
ERROR: this checkpoint was trained with prep.batch_covariates=true.
  That appends covar_slice_log_n_cells -- literally log1p(number of presented
  cells) -- as an input FEATURE. Thinning the input would then move the model
  through an out-of-distribution covariate, so the experiment would measure the
  response to a scalar, not to spatial density. The result would be invalid and
  the failure is invisible in the output. Use a checkpoint trained without it.
EOF
    exit 2
  fi
  grep -qE "normalize:[[:space:]]*(zscore|lognorm_zscore)" "$HYDRA_CFG" && {
    echo "WARNING: checkpoint used prep.normalize=zscore — feature stats are" >&2
    echo "         recomputed from the train split. Verify work/feature_norm_stats.json" >&2
    echo "         is reused, or every condition's features shift." >&2; }
else
  echo "WARNING: no .hydra/config.yaml beside the checkpoint ($HYDRA_CFG)." >&2
  echo "         The run will fall back to the DEFAULT model config, which may" >&2
  echo "         not match training. Keep the original training run dir intact." >&2
fi

# --- discover conditions -----------------------------------------------------
CONDS=()
for d in "$COND_ROOT"/*/; do
  [[ -f "$d/test.csv" ]] || continue
  nm="$(basename "$d")"
  if [[ -n "$ONLY" ]]; then
    case ",$ONLY," in *",$nm,"*) ;; *) continue ;; esac
  fi
  CONDS+=( "$nm" )
done
[[ ${#CONDS[@]} -gt 0 ]] && :
if [[ ${#CONDS[@]} -eq 0 ]]; then
  echo "ERROR: no conditions with test.csv under $COND_ROOT" >&2
  [[ -n "$ONLY" ]] && echo "       (--only '$ONLY' matched nothing)" >&2
  exit 2
fi

read -r -a SEED_ARR <<< "$SEEDS" || true
N_GENES="$(python3 -c "
import json,sys
print(json.load(open('$COND_ROOT/spec.json'))['n_genes'])" 2>/dev/null || echo "")"

echo "== submit_subsample_robustness.sh =="
echo "conditions  : ${#CONDS[@]}  (${CONDS[*]})"
echo "seeds       : ${SEED_ARR[*]}"
echo "jobs        : $(( ${#CONDS[@]} * ${#SEED_ARR[@]} ))"
echo "checkpoint  : $CHECKPOINT"
echo "runs root   : $RUNS_ROOT"
[[ -n "$N_GENES" ]] && echo "n_genes     : $N_GENES"
echo

MANIFEST="$COND_ROOT/submitted_runs.csv"
[[ -f "$MANIFEST" ]] || echo "condition,seed,artifacts_root" > "$MANIFEST"

for cond in "${CONDS[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    ROOT="$RUNS_ROOT/$(basename "$COND_ROOT")/${cond}__seed${seed}"
    # Never submit into a populated root: the work/test.csv cache would make the
    # job re-score the previous condition and report it as this one.
    if [[ -d "$ROOT" ]] && find "$ROOT" -name 'metadata_pred.csv' -print -quit | grep -q .; then
      echo "  SKIP $cond seed$seed — $ROOT already holds results" >&2
      continue
    fi
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$ROOT"

    CMD=( bash "$SUBMIT"
          --method scgg
          --skip_training
          --checkpoint "$CHECKPOINT"
          --train_csv  "$TRAIN_CSV"
          --test_csv   "$COND_ROOT/$cond/test.csv"
          --seed       "$seed"
          --wandb_mode disabled
          --override   "dataset.num_workers=0" )
    [[ ${#PASSTHRU[@]} -gt 0 ]] && CMD+=( "${PASSTHRU[@]}" )
    [[ "$DRY_RUN" == "1" ]] && CMD+=( --dry_run )

    echo "-- $cond seed$seed -> $ROOT"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "   SCGG_ARTIFACTS_ROOT=$ROOT ${CMD[*]}"
    else
      SCGG_ARTIFACTS_ROOT="$ROOT" "${CMD[@]}"
      echo "$cond,$seed,$ROOT" >> "$MANIFEST"
      sleep "$SUBMIT_DELAY"
    fi
  done
done

echo
echo "Submitted run roots recorded in: $MANIFEST"
echo
echo "When the jobs finish, score with (one --runs entry per condition):"
echo "  python $HERE/score_subsample_robustness.py \\"
echo "      --cond_root $COND_ROOT \\"
echo "      --runs \"\$(tail -n +2 $MANIFEST | awk -F, '{printf \"%s_seed%s=%s,\", \$1, \$2, \$3}' | sed 's/,\$//')\" \\"
echo "      --scgg_src /nfs/team361/sb75/scgg/src"
