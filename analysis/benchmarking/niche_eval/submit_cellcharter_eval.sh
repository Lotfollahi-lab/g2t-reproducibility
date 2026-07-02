#!/usr/bin/env bash
# submit_cellcharter_eval.sh
# -----------------------------------------------------------------------------
# CellCharter niche-recovery evaluation on the CNS test set, across 5 SEEDS,
# for four coordinate sources (imputed reference / LUNA / G2T / CeLEry), scored
# against the ground-truth `Sub_molecular_tissue_region` labels (NMI + ARI).
#
# ONE LSF JOB PER SEED (parallel, faster). Each job compares reference/luna/g2t/
# celery for that seed's reconstructions and its own CellCharter GMM seed, and
# writes niche_eval_out/seed<S>/. Aggregate to mean±SD afterwards with
#   python aggregate_niche_seeds.py --root niche_eval_out
#
# Seeds are paired by the MODEL seed embedded in each run's inner folder name
# (`..._seed<S>_inference`), so luna/g2t/celery line up per seed even if their
# timestamps don't sort the same way (falls back to array index if not found).
#
# Edit the CONFIG block, then:  bash submit_cellcharter_eval.sh
#                          or:  DRY_RUN=1 bash submit_cellcharter_eval.sh
# -----------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG (verify paths) ==========================
ENV_ACTIVATE="source /nfs/team361/sb75/.venvs/cellcharter/bin/activate"

# Shared expression (silver cns_luna dir of per-section *_test.h5ad).
EXPR_H5AD="/nfs/team361/sb75/DATASETS/silver/cns_luna"

ART="/nfs/team361/sb75/scgg-reproducibility/artifacts/cns_luna"
LUNA_PARENT="$ART/luna_inference"
G2T_PARENT="$ART/scgg_inference"
CELERY_PARENT="$ART/celery_inference"

# The 5 seed timestamps per method (from the inference dirs). Order does NOT
# matter — jobs are paired by the seed discovered inside each run — but list all
# five for each method. Each entry is a timestamp under the *_PARENT above.
LUNA_TS=(20260601_085512 20260601_085523 20260601_085535 20260601_085541 20260601_085547)
G2T_TS=(20260530_165200 20260530_165210 20260530_165216 20260530_165223 20260530_165229)
CELERY_TS=(20260604_101049 20260604_141924 20260604_141952 20260604_141959 20260604_142006)

GT_METADATA="/nfs/team361/sb75/DATASETS/bronze/cns_luna_raw/metadata.csv"
GT_COL="Sub_molecular_tissue_region"     # or Main_molecular_tissue_region (coarser)
GT_ID_COL="NAME"
EXPR_ID_COL="cell_id"

# Method knobs (applied IDENTICALLY to all sources and all seeds).
N_LATENT=50        # PCA dim of the 600-d latent (shared representation)
N_LAYERS=3         # CellCharter aggregation hops
N_CLUSTERS=0       # 0 = number of unique GT regions
OUT="$HERE/niche_eval_out"

# LSF (matches the other scripts on this cluster).
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

# Discover the model seed embedded in a run dir (…_seed<S>_inference); empty if absent.
seed_of() {
  find "$1" -maxdepth 6 -type d -name '*_seed*_inference' 2>/dev/null \
    | head -1 | sed -E 's#.*_seed([0-9]+)_inference.*#\1#'
}

# Build seed -> run-dir maps (fall back to array index when no embedded seed).
declare -A LMAP GMAP CMAP
i=0; for ts in "${LUNA_TS[@]}";   do d="$LUNA_PARENT/$ts";   s=$(seed_of "$d"); LMAP[${s:-$i}]="$d"; i=$((i+1)); done
i=0; for ts in "${G2T_TS[@]}";    do d="$G2T_PARENT/$ts";    s=$(seed_of "$d"); GMAP[${s:-$i}]="$d"; i=$((i+1)); done
i=0; for ts in "${CELERY_TS[@]}"; do d="$CELERY_PARENT/$ts"; s=$(seed_of "$d"); CMAP[${s:-$i}]="$d"; i=$((i+1)); done

echo "Seed -> run-dir pairing (luna | g2t | celery):"
n_sub=0
for s in $(printf '%s\n' "${!LMAP[@]}" | sort -n); do
  L="${LMAP[$s]:-}"; G="${GMAP[$s]:-}"; C="${CMAP[$s]:-}"
  if [[ -z "$G" || -z "$C" ]]; then
    echo "  seed $s: SKIP (missing in $([[ -z $G ]] && echo g2t) $([[ -z $C ]] && echo celery))"
    continue
  fi
  for d in "$L" "$G" "$C"; do [[ -d "$d" ]] || { echo "  seed $s: MISSING dir $d"; exit 1; }; done
  echo "  seed $s: $(basename "$L") | $(basename "$G") | $(basename "$C")"

  CMD="$ENV_ACTIVATE && python $HERE/cellcharter_niche_eval.py \
    --expr_h5ad '$EXPR_H5AD' \
    --coords_luna '$L' --coords_g2t '$G' --coords_celery '$C' \
    --gt_metadata '$GT_METADATA' --gt_col '$GT_COL' --gt_id_col '$GT_ID_COL' \
    --expr_id_col '$EXPR_ID_COL' \
    --n_latent $N_LATENT --n_layers $N_LAYERS --n_clusters $N_CLUSTERS \
    --seed $s --out '$OUT/seed$s'"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    [dry-run] $CMD"
  else
    bsub -J "ccniche_s$s" \
         -q "$QUEUE" -G "$GROUP" -gpu "$GPU_SPEC" -n "$CORES" \
         -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB] span[hosts=1]" \
         -W "$WALL" \
         -o "$LOGDIR/ccniche_s$s.%J.out" -e "$LOGDIR/ccniche_s$s.%J.err" \
         "bash -lc \"$CMD\""
  fi
  n_sub=$((n_sub+1))
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: would submit $n_sub job(s). Re-run without DRY_RUN=1 to submit."
else
  echo "Submitted $n_sub job(s) -> $OUT/seed<S>/ (each: niche_eval_metrics.csv, niche_labels.csv, *.png/pdf)."
  echo "When all finish, aggregate to mean±SD:"
  echo "  $ENV_ACTIVATE && python $HERE/aggregate_niche_seeds.py --root '$OUT'"
fi
