#!/usr/bin/env bash
# submit_cellcharter_eval.sh
# -----------------------------------------------------------------------------
# CellCharter niche-recovery evaluation on the CNS test set for four coordinate
# sources (imputed reference / LUNA / G2T / CeLEry), scored against the
# ground-truth `Sub_molecular_tissue_region` labels (NMI + ARI).
#
# One LSF job. The shared representation is a PCA of the 600-dim Harmony latent
# already stored in the silver AnnData (no scVI), so this is light — a GPU only
# accelerates CellCharter's GMM.
#
# Edit the CONFIG block, then:  bash submit_cellcharter_eval.sh
# -----------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG (verify paths) ==========================
ENV_ACTIVATE="source /nfs/team361/sb75/.venvs/cellcharter/bin/activate"

# Shared expression (silver cns_luna: X=600-d latent, obsm['spatial']=reference
# coords, obs['cell_id'], obs['cell_section']). Dir of per-section .h5ad or one file.
EXPR_H5AD="/nfs/team361/sb75/DATASETS/silver/cns_luna"

# Reconstructed coords = each run's inference dir (globs metadata_pred.csv per
# section). Timestamps are paired with the model checkpoints you gave.
# NOTE: if these *_inference/<TS> dirs don't contain metadata_pred.csv, re-run
# the pipeline in test_only mode against each best_model.ckpt to regenerate them.
ART="/nfs/team361/sb75/scgg-reproducibility/artifacts/cns_luna"
COORDS_LUNA="$ART/luna_inference/20260601_085512"
COORDS_G2T="$ART/scgg_inference/20260602_142452"
COORDS_CELERY="$ART/celery_inference/20260604_141952"
# Reference coords default to the AnnData's obsm['spatial']; leave empty to use it.
COORDS_REF=""

GT_METADATA="/nfs/team361/sb75/DATASETS/bronze/cns_luna_raw/metadata.csv"
GT_COL="Sub_molecular_tissue_region"     # or Main_molecular_tissue_region for a coarser (faster) comparison
GT_ID_COL="NAME"                          # SCP metadata id column
EXPR_ID_COL="cell_id"                     # AnnData obs id -> joins to metadata NAME and metadata_pred index

# Method knobs (applied IDENTICALLY to all four sources).
N_LATENT=50        # PCA dim of the 600-d latent (shared representation)
N_LAYERS=3         # CellCharter aggregation hops
N_CLUSTERS=0       # 0 = number of unique GT regions
SEED=0
OUT="$HERE/niche_eval_out"

# LSF (matches the other scripts on this cluster).
QUEUE="training-parallel"
GROUP="s10396"
GPU_SPEC="num=1"
MEM_MB=256000
CORES=8
WALL="12:00"
LOGDIR="$HERE/lsf_logs"
# =============================================================================

mkdir -p "$LOGDIR" "$OUT"

REF_ARG=""; [[ -n "$COORDS_REF" ]] && REF_ARG="--coords_ref '$COORDS_REF'"
CEL_ARG=""; [[ -n "$COORDS_CELERY" ]] && CEL_ARG="--coords_celery '$COORDS_CELERY'"

CMD="$ENV_ACTIVATE && python $HERE/cellcharter_niche_eval.py \
  --expr_h5ad '$EXPR_H5AD' \
  --coords_luna '$COORDS_LUNA' --coords_g2t '$COORDS_G2T' $CEL_ARG $REF_ARG \
  --gt_metadata '$GT_METADATA' --gt_col '$GT_COL' --gt_id_col '$GT_ID_COL' \
  --expr_id_col '$EXPR_ID_COL' \
  --n_latent $N_LATENT --n_layers $N_LAYERS --n_clusters $N_CLUSTERS --seed $SEED \
  --out '$OUT'"

echo "Submitting CellCharter niche eval:"; echo "  $CMD"

bsub -J cc_niche_eval \
     -q "$QUEUE" -G "$GROUP" -gpu "$GPU_SPEC" \
     -n "$CORES" \
     -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB] span[hosts=1]" \
     -W "$WALL" \
     -o "$LOGDIR/cc_niche_eval.%J.out" -e "$LOGDIR/cc_niche_eval.%J.err" \
     "bash -lc \"$CMD\""

echo "Submitted. Outputs -> $OUT (niche_eval_metrics.csv, niche_labels.csv, niche_maps.png)"
