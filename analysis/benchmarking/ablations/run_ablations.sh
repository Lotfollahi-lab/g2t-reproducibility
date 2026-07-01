#!/usr/bin/env bash
# run_ablations.sh
# -----------------------------------------------------------------------------
# Launch the G2T ablation grid on the MERFISH mouse cortex (MMC) split,
# 5 seeds each, via the existing LSF pipeline submitter
# (analysis/benchmarking/lsf/submit_pipeline.sh, --method scgg).
#
# Each ablation = the G2T baseline override string with ONE factor changed,
# isolating that factor's contribution. MMC is used because it is the cheap
# benchmark (1000 epochs, ~6 train slices) — the "quick" ablation bed.
#
# This script SUBMITS jobs and records, per (ablation, seed), the run
# timestamp that submit_pipeline.sh prints, into ablation_manifest.csv.
# After all jobs finish, run:  python aggregate_ablations.py
#
# USAGE (edit the paths in the CONFIG block first, then):
#   bash run_ablations.sh                     # submit the CORE ablations
#   ABLATION_SET=all bash run_ablations.sh    # submit CORE + EXTENDED
#   ONLY=diffusion bash run_ablations.sh      # submit only the named ablation(s)
#   DRY_RUN=1 bash run_ablations.sh           # print submit commands, don't submit
# -----------------------------------------------------------------------------
set -euo pipefail

# ============================ CONFIG (EDIT ME) ===============================
SUBMIT="/nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/lsf/submit_pipeline.sh"
DATA_DIR="/nfs/team361/sb75/DATASETS/silver/mmc_luna"   # MMC silver h5ads
EPOCHS=1000
SEEDS=(0 1 2 3 4)
MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ablation_manifest.csv"

# G2T baseline override string (your scgg_mmc_edm_fm_heads32_fastmds config).
# edm.enabled=true and framework=flow_matching are config DEFAULTS; the four
# tokens below are the explicit overrides from your run config.
BASELINE_OVERRIDE="model.hidden_dims.num_heads=32 model.edm.mds_align_train=false model.edm.mds_dtype=fp32 model.edm.mds_solver=lobpcg dataset.num_workers=0"
# =============================================================================

# Ablation table as "name|extra_override" entries (pipe-delimited; empty
# override = unmodified baseline). Indexed arrays keep this portable across
# bash 3.2 (macOS) and 4+ (Linux cluster).
CORE=(
  "baseline|"                                  # reference G2T (you may already have these 5 seeds)
  "edm_off|model.edm.enabled=false"            # *** headline: drop EDM head -> predict 2-D coords directly
  "diffusion|model.framework=diffusion"        # LUNA-style DDPM instead of flow matching (EDM kept)
  "heads16|model.hidden_dims.num_heads=16"     # default heads instead of 32 (justifies heads32)
  "K2|model.edm.embed_dim=2"                   # collapse embedding to 2-D (no extra room)
  "K16|model.edm.embed_dim=16"                 # more room than K=8
)
EXTENDED=(
  "K4|model.edm.embed_dim=4"
  "K32|model.edm.embed_dim=32"
  "mds_eigh|model.edm.mds_solver=eigh"         # exact MDS vs LOBPCG (accuracy parity / runtime check)
)

ABLATION_SET="${ABLATION_SET:-core}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"          # comma-separated ablation names to restrict to
                          # (e.g. ONLY=diffusion or ONLY=diffusion,K2).
                          # Empty = the whole selected set.
SUBMIT_DELAY="${SUBMIT_DELAY:-2}"   # seconds to wait between submissions.
                          # submit_pipeline.sh keys every run's artifact dir on
                          # `date +%Y%m%d_%H%M%S` (1-SECOND resolution), so two
                          # back-to-back submits landing in the same second get
                          # the SAME timestamp -> same output dir -> they clobber
                          # each other (duplicate rows + missing seeds). A >=1s
                          # gap guarantees distinct seconds; 2s adds margin.

entries=( "${CORE[@]}" )
if [[ "$ABLATION_SET" == "all" ]]; then
  entries+=( "${EXTENDED[@]}" )
fi

# Optional ONLY filter: keep just the named ablation(s). Handy to re-run a
# single ablation (e.g. after fixing the diffusion path) without
# resubmitting the ones that already succeeded. Portable (bash 3.2+).
if [[ -n "$ONLY" ]]; then
  filtered=()
  for entry in "${entries[@]}"; do
    nm="${entry%%|*}"
    case ",$ONLY," in
      *",$nm,"*) filtered+=( "$entry" ) ;;
    esac
  done
  if [[ ${#filtered[@]} -eq 0 ]]; then
    echo "ERROR: ONLY='$ONLY' matched no ablations in set '$ABLATION_SET'." >&2
    exit 1
  fi
  entries=( "${filtered[@]}" )
fi

# Manifest header (created once).
if [[ ! -f "$MANIFEST" ]]; then
  echo "ablation,seed,run_name,timestamp" > "$MANIFEST"
fi

seen_ts=""   # timestamps recorded so far THIS invocation (collision guard)

for entry in "${entries[@]}"; do
  name="${entry%%|*}"      # text before the first '|'
  extra="${entry#*|}"      # text after  the first '|'
  if [[ -n "$extra" ]]; then ovr="$BASELINE_OVERRIDE $extra"; else ovr="$BASELINE_OVERRIDE"; fi
  for s in "${SEEDS[@]}"; do
    run_name="scgg_mmc_abl_${name}_seed${s}"
    echo "============================================================"
    echo "ABLATION=$name  SEED=$s"
    echo "  run_name : $run_name"
    echo "  override : $ovr"
    cmd=( bash "$SUBMIT"
          --method scgg
          --wandb_run_name "$run_name"
          --data_dir "$DATA_DIR"
          --epochs "$EPOCHS"
          --seed "$s"
          --override "$ovr" )
    if [[ "$DRY_RUN" == "1" ]]; then
      cmd+=( --dry_run )
    fi
    # Capture submit output so we can extract the run timestamp it prints
    # ("run timestamp : YYYYMMDD_HHMMSS").
    out="$( "${cmd[@]}" 2>&1 )"
    echo "$out" | sed -n '1,40p'
    ts="$(echo "$out" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1)"
    if [[ -z "$ts" ]]; then ts="UNKNOWN_PARSE_FAILED"; fi
    # Collision guard: submit_pipeline.sh keys every artifact dir on this
    # 1-second timestamp, so two runs sharing one clobber each other. The
    # SUBMIT_DELAY below is meant to prevent this; warn loudly if one slips
    # through anyway (e.g. SUBMIT_DELAY=0).
    if [[ "$ts" != "UNKNOWN_PARSE_FAILED" ]]; then
      case " $seen_ts " in
        *" $ts "*)
          echo "  !! WARNING: timestamp $ts already used this run -> COLLISION;" >&2
          echo "     this run will overwrite the earlier one. Increase SUBMIT_DELAY and re-run." >&2 ;;
        *) seen_ts="$seen_ts $ts" ;;
      esac
    fi
    echo "$name,$s,$run_name,$ts" >> "$MANIFEST"
    echo "  -> recorded timestamp: $ts"
    # Space submissions apart so consecutive runs land in DIFFERENT seconds
    # (distinct timestamps -> distinct artifact dirs). Skipped for dry runs.
    [[ "$DRY_RUN" == "1" ]] || sleep "$SUBMIT_DELAY"
  done
done

echo
echo "Done submitting. Manifest written to: $MANIFEST"
echo "When all LSF jobs finish:"
echo "  1) compute metrics:"
echo "       python /nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/plots/compute_extended_metrics.py \\"
echo "           --dataset mmc_luna --methods g2t \\"
echo "           --scgg_timestamps \$(tail -n +2 $MANIFEST | cut -d, -f4 | paste -sd, -)"
echo "  2) aggregate:"
echo "       python aggregate_ablations.py"
