#!/usr/bin/env bash
# submit_cellcharter_eval.sh
# -----------------------------------------------------------------------------
# CellCharter niche-recovery evaluation on the CNS test set, across 5 SEEDS,
# for four coordinate sources (imputed reference / LUNA / G2T / CeLEry), scored
# against the ground-truth `Sub_molecular_tissue_region` labels (NMI + ARI).
#
# ONE LSF JOB PER SEED (parallel). Each job compares reference/luna/g2t/celery
# for that seed and its own CellCharter GMM seed, writing niche_eval_out/seed<S>/.
# Aggregate afterwards:  python aggregate_niche_seeds.py --root niche_eval_out
#
# Seeds are paired by the MODEL seed embedded in each run's inner folder name
# (`..._seed<S>_inference`); falls back to the array index if not found.
#
#   bash submit_cellcharter_eval.sh          # submit
#   DRY_RUN=1 bash submit_cellcharter_eval.sh # preview pairing + commands only
#
# NOTE: intentionally NOT using `set -e` — dir discovery is best-effort and must
# never make the launcher exit silently. Real problems are reported as WARN/errors.
# -----------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ((BASH_VERSINFO[0] < 4)); then
  echo "ERROR: need bash >= 4 for associative arrays (have $BASH_VERSION)"; exit 1
fi

# ============================ CONFIG (verify paths) ==========================
ENV_ACTIVATE="source /nfs/team361/sb75/.venvs/cellcharter/bin/activate"

EXPR_H5AD="/nfs/team361/sb75/DATASETS/silver/cns_luna"   # dir of per-section *_test.h5ad

ART="/nfs/team361/sb75/scgg-reproducibility/artifacts/cns_luna"
LUNA_PARENT="$ART/luna_inference"
G2T_PARENT="$ART/scgg_inference"
CELERY_PARENT="$ART/celery_inference"

# 5 seed timestamps per method (each a dir under the matching *_PARENT). Order
# does not matter — jobs are paired by the seed discovered inside each run.
LUNA_TS=(20260601_085512 20260601_085523 20260601_085535 20260601_085541 20260601_085547)
G2T_TS=(20260602_142452 20260602_142505 20260602_142510 20260602_142516 20260602_142522)
CELERY_TS=(20260604_101049 20260604_141924 20260604_141952 20260604_141959 20260604_142006)

GT_METADATA="/nfs/team361/sb75/DATASETS/bronze/cns_luna_raw/metadata.csv"
GT_COL="Sub_molecular_tissue_region"
GT_ID_COL="NAME"
EXPR_ID_COL="cell_id"

N_LATENT=50
N_LAYERS=3
N_CLUSTERS=0        # 0 = number of unique GT regions
OUT="$HERE/niche_eval_out"

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

echo "[submit] multi-seed CellCharter niche eval  (DRY_RUN=$DRY_RUN)"
echo "[submit]   EXPR_H5AD    = $EXPR_H5AD"
echo "[submit]   LUNA_PARENT  = $LUNA_PARENT"
echo "[submit]   G2T_PARENT   = $G2T_PARENT"
echo "[submit]   CELERY_PARENT= $CELERY_PARENT"
echo

# Discover the model seed embedded in a run dir (…_seed<S>_inference); prints
# nothing (and returns 0) if not found — never fails the pipeline.
seed_of() {
  local d
  d=$(find "$1" -maxdepth 6 -type d -name '*_seed*_inference' 2>/dev/null | head -n1)
  if [[ "$d" =~ _seed([0-9]+)_inference ]]; then printf '%s' "${BASH_REMATCH[1]}"; fi
  return 0
}

declare -A LMAP GMAP CMAP

# Inline (no bash-4.3 namerefs) so this works on any bash 4.x on the farm.
i=0
for ts in "${LUNA_TS[@]}"; do
  d="$LUNA_PARENT/$ts"
  if [[ ! -d "$d" ]]; then echo "[submit] WARN missing luna dir (skipped): $d"; i=$((i+1)); continue; fi
  s=$(seed_of "$d"); s="${s:-$i}"; LMAP["$s"]="$d"; echo "[submit]   luna   seed $s <- $ts"; i=$((i+1))
done
i=0
for ts in "${G2T_TS[@]}"; do
  d="$G2T_PARENT/$ts"
  if [[ ! -d "$d" ]]; then echo "[submit] WARN missing g2t dir (skipped): $d"; i=$((i+1)); continue; fi
  s=$(seed_of "$d"); s="${s:-$i}"; GMAP["$s"]="$d"; echo "[submit]   g2t    seed $s <- $ts"; i=$((i+1))
done
i=0
for ts in "${CELERY_TS[@]}"; do
  d="$CELERY_PARENT/$ts"
  if [[ ! -d "$d" ]]; then echo "[submit] WARN missing celery dir (skipped): $d"; i=$((i+1)); continue; fi
  s=$(seed_of "$d"); s="${s:-$i}"; CMAP["$s"]="$d"; echo "[submit]   celery seed $s <- $ts"; i=$((i+1))
done

echo
echo "[submit] discovered seeds -> luna:[${!LMAP[*]}] g2t:[${!GMAP[*]}] celery:[${!CMAP[*]}]"
echo

n_sub=0
for s in $(printf '%s\n' "${!LMAP[@]}" | sort -n); do
  L="${LMAP[$s]:-}"; G="${GMAP[$s]:-}"; C="${CMAP[$s]:-}"
  if [[ -z "$G" || -z "$C" ]]; then
    echo "[submit] seed $s: SKIP (missing $([[ -z $G ]] && echo g2t) $([[ -z $C ]] && echo celery))"
    continue
  fi
  echo "[submit] seed $s: $(basename "$L") | $(basename "$G") | $(basename "$C")"

  CMD="$ENV_ACTIVATE && python $HERE/cellcharter_niche_eval.py \
    --expr_h5ad '$EXPR_H5AD' \
    --coords_luna '$L' --coords_g2t '$G' --coords_celery '$C' \
    --gt_metadata '$GT_METADATA' --gt_col '$GT_COL' --gt_id_col '$GT_ID_COL' \
    --expr_id_col '$EXPR_ID_COL' \
    --n_latent $N_LATENT --n_layers $N_LAYERS --n_clusters $N_CLUSTERS \
    --controls --seed $s --out '$OUT/seed$s'"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "   [dry-run] $CMD"
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
if ((n_sub == 0)); then
  echo "[submit] NO JOBS submitted. Check the WARN lines / timestamps / *_PARENT dirs above."
  exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[submit] DRY-RUN ok: would submit $n_sub job(s). Re-run without DRY_RUN=1 to submit."
else
  echo "[submit] submitted $n_sub job(s) -> $OUT/seed<S>/"
  echo "[submit] when all finish: $ENV_ACTIVATE && python $HERE/aggregate_niche_seeds.py --root '$OUT'"
fi
