#!/usr/bin/env bash
# submit_resolution_sweep.sh
# -----------------------------------------------------------------------------
# CellCharter niche eval, RESOLUTION-SWEEP variant: ONE timestamp per method,
# sweeping several niche counts K (the CellCharter "resolution"), in a single LSF
# job. Complements (does not replace) submit_cellcharter_eval.sh, which sweeps
# SEEDS across 5 jobs.
#
#   bash submit_resolution_sweep.sh           # submit
#   DRY_RUN=1 bash submit_resolution_sweep.sh # preview command only
# -----------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG =========================================
ENV_ACTIVATE="source /nfs/team361/sb75/.venvs/cellcharter/bin/activate"
EXPR_H5AD="/nfs/team361/sb75/DATASETS/silver/cns_luna"
ART="/nfs/team361/sb75/scgg-reproducibility/artifacts/cns_luna"

# ONE run per method (the timestamps you chose).
COORDS_G2T="$ART/scgg_inference/20260602_142505"
COORDS_LUNA="$ART/luna_inference/20260601_085512"
COORDS_CELERY="$ART/celery_inference/20260604_141952"

GT_METADATA="/nfs/team361/sb75/DATASETS/bronze/cns_luna_raw/metadata.csv"
GT_COL="Sub_molecular_tissue_region"
GT_ID_COL="NAME"
EXPR_ID_COL="cell_id"

# Resolution sweep: multipliers of the #GT-regions -> 5 niche counts K.
RESOLUTION_FACTORS="0.5,0.75,1.0,1.5,2.0"
N_LATENT=50
N_LAYERS=3
SEED=0
OUT="$HERE/niche_eval_out/resolution_sweep"

QUEUE="training-parallel"
GROUP="s10396"
GPU_SPEC="num=1"
MEM_MB=256000
CORES=8
WALL="12:00"
LOGDIR="$HERE/lsf_logs"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

mkdir -p "$LOGDIR" "$OUT"

for d in "$COORDS_G2T" "$COORDS_LUNA" "$COORDS_CELERY"; do
  [[ -d "$d" ]] || { echo "[submit] MISSING dir: $d"; exit 1; }
done

CMD="$ENV_ACTIVATE && python $HERE/cellcharter_niche_eval.py \
  --expr_h5ad '$EXPR_H5AD' \
  --coords_luna '$COORDS_LUNA' --coords_g2t '$COORDS_G2T' --coords_celery '$COORDS_CELERY' \
  --gt_metadata '$GT_METADATA' --gt_col '$GT_COL' --gt_id_col '$GT_ID_COL' \
  --expr_id_col '$EXPR_ID_COL' --controls \
  --resolution_factors '$RESOLUTION_FACTORS' \
  --n_latent $N_LATENT --n_layers $N_LAYERS --seed $SEED --out '$OUT'"

echo "[submit] resolution-sweep niche eval (single job)"
echo "[submit]   g2t   = $(basename "$COORDS_G2T")"
echo "[submit]   luna  = $(basename "$COORDS_LUNA")"
echo "[submit]   celery= $(basename "$COORDS_CELERY")"
echo "[submit]   K factors = $RESOLUTION_FACTORS  -> out: $OUT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $CMD"
  exit 0
fi

bsub -J "ccniche_ressweep" \
     -q "$QUEUE" -G "$GROUP" -gpu "$GPU_SPEC" -n "$CORES" \
     -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB] span[hosts=1]" \
     -W "$WALL" \
     -o "$LOGDIR/ccniche_ressweep.%J.out" -e "$LOGDIR/ccniche_ressweep.%J.err" \
     "bash -lc \"$CMD\""

echo "[submit] submitted -> $OUT (niche_eval_metrics.csv with a 'resolution' column,"
echo "         niche_eval_resolution.png/pdf, niche_labels.csv, niche_maps.png)"
