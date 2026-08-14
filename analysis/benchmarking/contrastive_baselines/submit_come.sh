#!/usr/bin/env bash
# submit_come.sh
# ------------------------------------------------------------------------------
# Submit the reviewer-requested COME baseline to LSF, one job per seed, mirroring
# submit_cellcontrast.sh (same group/queue defaults, a generated per-job script,
# GPU request, logs beside the artifacts).
#
# Runs the AUTHORS' published code (https://github.com/cindyway/COME) at their
# own per-platform preset via run_come.py: COME = "contrastive mapping learning
# for spatial reconstruction of single-cell RNA sequencing data", Wei et al.,
# Bioinformatics 41(3):btaf083, 2025 (cited as come2025). Nothing here changes
# the method.
#
# WHY THIS SUBMITTER ONLY ACCEPTS mmc_luna
# ----------------------------------------
# COME is TRANSDUCTIVE: MapNet.Coefficient is a dense nn.Parameter of shape
# (n_spots x n_cells), so the model is refitted for every (reference, test slice)
# pair and peak memory is O((n_ref + n_query)^2):
#
#     peak = 16*n_ref*n_query      (Coefficient + grad + 2 Adam states)
#          +  4*n_ref*n_query      (cross_mask)
#          +  4*n_query^2          (type_mask)
#          + 20*(n_ref+n_query)^2  (full_mask + sim + sim_exp + pos/neg masks)
#
# That is the same formula run_come.py's estimate_peak_bytes() uses; this script
# evaluates it in shell BEFORE submitting, so an infeasible plan costs seconds
# instead of a queue slot. Budget -> envelope (n_ref + n_query):
# 8GB ~ 20,000; 32GB ~ 40,000; 128GB ~ 80,000; 256GB ~ 113,000.
#
# The DECIDED protocol is therefore: mmc_luna only, with the FULL test slice
# (<=5,235 cells, so the evaluated cell population is identical to
# LUNA/G2T/CeLEry/CellContrast) against a SUBSAMPLED reference
# (--max_ref_cells, default 20,000 -> ~15GB). The reference size is the only
# deviation, and run_come.py records it in the manifest. cns_luna is OUT OF
# SCOPE: its 63,343-cell test slice needs ~96GB with an EMPTY reference, so the
# query term dominates and no --max_ref_cells can rescue it — pass --dataset
# cns_luna to get the full arithmetic printed in a form you can paste into a
# reviewer response.
#
# COME never uses spatial coordinates during training (they enter only at the
# read-out, argmax over spots of Coefficient — upstream's own cross_mask rule),
# and it normalises internally via utils.normalize_type, so the silver matrix is
# handed over UNTOUCHED. There is no --expression_mode and no --coord_frame knob
# here, unlike the CellContrast submitter: for COME neither would mean anything.
#
# GPU: upstream builds full_mask on the CPU while cross_mask lives on the
# model's device, so the CUDA path can raise a device mismatch inside
# ContrastiveLoss. If a job dies that way, resubmit with --device cpu (the
# reservation then drops the -gpu request unless you passed --gpu yourself).
# The fits here are small enough (~15GB, a few thousand spots x a few thousand
# cells) that CPU is a viable fallback, not a disaster.
#
# Usage:
#   # 0) once: clone + env, then ALWAYS smoke-test before a real run
#   bash ../setup_come_env.sh
#   bash submit_come.sh --dataset mmc_luna --smoke_test
#
#   # 1) cortex, 5 seeds, the decided protocol (ref 20k, full test slices)
#   bash submit_come.sh --dataset mmc_luna --seeds "0 1 2 3 4"
#
#   # 2) same, but on the CPU path after a CUDA mask-device failure
#   bash submit_come.sh --dataset mmc_luna --device cpu --seeds "0 1 2 3 4"
#
#   # 3) bigger reference (memory grows ~quadratically; the reservation and the
#   #    runner budget are resized from the estimate automatically)
#   bash submit_come.sh --dataset mmc_luna --max_ref_cells 40000 --seeds "0"
#
# Options:
#   --dataset NAME        mmc_luna ONLY (required). Any other value is refused;
#                         cns_luna is refused with the feasibility arithmetic.
#   --seeds "0 1 2"       seeds, one job each   (default "0 1 2 3 4")
#   --data_dir DIR        override the silver dir
#   --repo DIR            COME checkout -> the runner's --come_repo (default
#                         $COME_REPO, else <this dir>/../COME, i.e. the clone
#                         setup_come_env.sh makes beside the other benchmarking
#                         assets)
#   --venv DIR            uv venv                (default
#                         /nfs/team361/sb75/.venvs/come)
#   --come_config C       dro | smFISH | MERFISH | STARmap | PDAC — which of the
#                         authors' per-platform presets (configure.py) to use.
#                         Default MERFISH: theirs for mouse visual cortex imaging
#                         ST, i.e. the right preset for mmc_luna. Changing it
#                         changes k, the layer dims and the epoch counts.
#   --sttype T            image (default) | sequence — upstream's preprocessing
#                         switch (utils.normalize_type): image = per-gene
#                         MinMaxScaler (imaging ST), sequence =
#                         normalize_total(1e4)+log1p (spot ST). mmc_luna is
#                         imaging, so 'image' is the reproducing choice.
#   --max_ref_cells N     cap the ST reference by per-slice stratified
#                         subsampling (default 20000). This is THE memory knob:
#                         the estimate below, the LSF reservation and the runner
#                         budget are all derived from it. Must be a positive
#                         integer.
#   --epochs N            override the preset's training epochs. OMIT to use the
#                         authors' preset — that is the defensible choice for a
#                         baseline. A deviation, and recorded as one.
#   --pretrain_epochs N   override the preset's AE pretraining epochs. Same.
#   --max_mem_gb G        make the runner REFUSE a fit whose estimated peak
#                         exceeds this (default: 85% of the LSF reservation,
#                         leaving headroom for torch and the loaded objects), so
#                         a bad memory plan fails in seconds, not hours in.
#   --device D            auto (default) | cpu — see the GPU note above.
#   --on_empty_cells P    fail (default) | keep | drop — what to do if a query
#                         slice has cells with zero detected genes. Upstream's
#                         load_data would DROP them, which breaks the row
#                         correspondence between predictions and truth; 'fail'
#                         refuses, 'drop' removes them and records that the
#                         evaluated population shrank.
#   --exclude_test_files  comma-separated *_test.h5ad basenames to skip, so the
#                         mean is over the SAME sections as G2T/LUNA/CeLEry.
#                         Not needed for mmc_luna (no sections are excluded
#                         there), kept because the runner supports it.
#   --smoke_test          tiny 1-job run to prove the install; never reported.
#                         The runner caps the reference at min(--max_ref_cells,
#                         500) and runs 2 epochs, so the estimate here uses 500
#                         as the upper bound.
#   --dry_run             print the bsub commands without submitting (this
#                         script only touches the job-script dir)
#   --plan_only           forward the RUNNER's --dry_run: really submit, but the
#                         job prints its per-slice plan and memory estimate and
#                         fits nothing. It still creates the timestamped
#                         come_inference/<TS>/ dir and leaves it EMPTY (no
#                         manifest), which the scorer skips — it requires
#                         run_manifest.json. (--runner_dry_run is an accepted
#                         alias. Two names exist because --dry_run already means
#                         "do not submit" in every submitter here, and silently
#                         overloading it would be the worse surprise.)
#   --mem MB / --wall HH:MM / --queue Q / --group G / --gpu SPEC / --cores N
#
# NOT exposed on purpose: the runner's --force_scale, which bypasses its 256GB
# hard feasibility refusal. Within the decided protocol nothing here comes near
# that ceiling, so needing it means the plan is outside the protocol — call
# run_come.py directly and say so in the write-up.
#
# Env overrides: COME_REPO, VENV_DIR, SCGG_ARTIFACTS_ROOT, LSF_GROUP, LSF_QUEUE,
# MEM_MB, WALL, GPU_SPEC, EXCLUDE_TEST_FILES, and CORES (default 8: the LSF slot
# count, also exported as OMP/MKL/OPENBLAS_NUM_THREADS in the job so the dense
# torch/BLAS ops do not oversubscribe a shared node — which matters more here
# than for CellContrast, because --device cpu is a realistic path).
# ------------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_come.py"

# --- shared arithmetic --------------------------------------------------------
# run_come.py's estimate_peak_bytes(), in shell. Kept literally term-by-term so
# the two can be diffed by eye. Unlike the runner's Python bigints this is 64-bit,
# so the caller must keep n1 + n2 below ~6.8e8 or the dominant term wraps NEGATIVE
# (--max_ref_cells is bounded to 8 digits below for exactly that reason); within
# that bound the largest value here, 20*(213,343)^2 ~ 9.1e11, is far inside range.
est_peak_bytes() {  # n_ref n_query -> bytes
  local n1="$1" n2="$2"
  echo $(( 16 * n1 * n2 + 4 * n1 * n2 + 4 * n2 * n2 + 20 * (n1 + n2) * (n1 + n2) ))
}
# Tenths of a GB, rounded, so the printed figures match the ones in the wrapper
# docstring and the tests (6.6 / 14.9 / 552.1 / 1116.4 / 96.3 GB) instead of
# being whole-GB-rounded lookalikes.
gb1() { local b="$1"; local d=$(( (b + 50000000) / 100000000 )); printf '%d.%d' $((d/10)) $((d%10)); }
gb_ceil() { echo $(( ($1 + 999999999) / 1000000000 )); }
# Largest n_ref+n_query that fits a budget under the dominant 20*(n1+n2)^2 term
# — run_come.py's max_total_for_budget(). awk only for the square root.
envelope() { awk -v b="$1" 'BEGIN{printf "%d", sqrt(b*1e9/20)}'; }
commafy() { printf '%s' "$1" | sed -e :a -e 's/\(.*[0-9]\)\([0-9]\{3\}\)/\1,\2/;ta'; }

DATASET=""
SEEDS="0 1 2 3 4"
DATA_DIR=""
# Defaults match setup_come_env.sh: the authors' code is cloned next to the other
# benchmarking assets, and the env is a uv venv (this cluster has no conda on
# PATH).
REPO="${COME_REPO:-$HERE/../COME}"
VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/come}"
COME_CONFIG="MERFISH"
STTYPE="image"
MAX_REF_CELLS="20000"
EPOCHS=""
PRETRAIN_EPOCHS=""
MAX_MEM_GB=""
DEVICE="auto"
ON_EMPTY_CELLS="fail"
EXCLUDE_TEST_FILES="${EXCLUDE_TEST_FILES:-}"
SMOKE=""
DRY_RUN=""
PLAN_ONLY=""
ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
# Empty means "derive it from the memory estimate below" — COME's footprint is a
# function of --max_ref_cells, so a flat default would be either wasteful or
# wrong as soon as that changes.
MEM_MB="${MEM_MB:-}"
WALL="${WALL:-48:00}"
CORES="${CORES:-8}"
GPU_SPEC="${GPU_SPEC:-num=1:mode=shared:j_exclusive=no}"
GPU_EXPLICIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)          DATASET="${2:?}"; shift 2 ;;
    --seeds)            SEEDS="${2:?}"; shift 2 ;;
    --data_dir)         DATA_DIR="${2:?}"; shift 2 ;;
    --repo)             REPO="${2:?}"; shift 2 ;;
    --venv)             VENV_DIR="${2:?}"; shift 2 ;;
    --come_config)      COME_CONFIG="${2:?}"; shift 2 ;;
    --sttype)           STTYPE="${2:?}"; shift 2 ;;
    --max_ref_cells)    MAX_REF_CELLS="${2:?}"; shift 2 ;;
    --epochs)           EPOCHS="${2:?}"; shift 2 ;;
    --pretrain_epochs)  PRETRAIN_EPOCHS="${2:?}"; shift 2 ;;
    --max_mem_gb)       MAX_MEM_GB="${2:?}"; shift 2 ;;
    --device)           DEVICE="${2:?}"; shift 2 ;;
    --on_empty_cells)   ON_EMPTY_CELLS="${2:?}"; shift 2 ;;
    --exclude_test_files) EXCLUDE_TEST_FILES="${2:?}"; shift 2 ;;
    --smoke_test)       SMOKE=1; shift ;;
    --dry_run)          DRY_RUN=1; shift ;;
    --plan_only|--runner_dry_run) PLAN_ONLY=1; shift ;;
    --mem)              MEM_MB="${2:?}"; shift 2 ;;
    --wall)             WALL="${2:?}"; shift 2 ;;
    --queue)            LSF_QUEUE="${2:?}"; shift 2 ;;
    --group)            LSF_GROUP="${2:?}"; shift 2 ;;
    --gpu)              GPU_SPEC="${2:?}"; GPU_EXPLICIT=1; shift 2 ;;
    --cores)            CORES="${2:?}"; shift 2 ;;
    # Print the header comment block, whatever its length: a fixed line range
    # stops being the header the moment it grows and then prints code.
    -h|--help)          awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next}
                             {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- dataset allow-list -------------------------------------------------------
# mmc_luna is the ONLY feasible dataset for COME (see the header). cns_luna gets
# the arithmetic rather than a bare refusal, because "we could not run COME on
# the CNS data" is a claim we have to justify to a reviewer, not just to
# ourselves.
if [[ "$DATASET" == "cns_luna" ]]; then
  CNS_Q=63343            # largest cns_luna test section (cells)
  CNS_REF_OTHER=150000   # reference size the other methods use on this dataset
  CNS_REF_SMALL=10000    # a deliberately tiny reference, to show it does not help
  _full=$(est_peak_bytes "$CNS_REF_OTHER" "$CNS_Q")
  _small=$(est_peak_bytes "$CNS_REF_SMALL" "$CNS_Q")
  _qonly=$(est_peak_bytes 0 "$CNS_Q")
  _come=$(est_peak_bytes 1000 15413)   # COME's own largest published run (VISp)
  # One printf per row instead of hand-spaced heredoc text, so the columns line
  # up whatever the numbers turn out to be — this message is meant to be pasted
  # into a reviewer response.
  _row() { printf '      %-46s %11s' "$1" "$2"; }
  cat >&2 <<EOF
ERROR: cns_luna is outside COME's feasible envelope; this submitter accepts
       --dataset mmc_luna only.

  COME is transductive: MapNet.Coefficient is a dense (n_spots x n_cells)
  nn.Parameter, and its loss rebuilds several dense (n_spots+n_cells)^2 masks
  every epoch, so peak memory is O((n_ref + n_query)^2):

      peak = 16*n_ref*n_query + 4*n_ref*n_query + 4*n_query^2
             + 20*(n_ref+n_query)^2

  For the largest cns_luna test section ($(commafy $CNS_Q) cells):

$(_row "reference $(commafy $CNS_REF_OTHER) (as LUNA/G2T/CeLEry run it)" "~$(gb1 $_full) GB")
$(_row "reference $(commafy $CNS_REF_SMALL) (deliberately tiny)" "~$(gb1 $_small) GB")
$(_row "reference 0 (EMPTY: the query term alone)" "~$(gb1 $_qonly) GB")
$(_row "run_come.py's hard feasibility ceiling" "256.0 GB")
      -> at that ceiling n_ref+n_query must be <= ~$(commafy $(envelope 256))

  The query term dominates, so --max_ref_cells CANNOT rescue this dataset: even
  with no reference at all the fit needs ~$(gb1 $_qonly) GB. For scale, COME's own
  largest published run (VISp: 1,000 spots + 15,413 cells) is ~$(gb1 $_come) GB, i.e.
  cns_luna is ~$(( ( _full + _come / 2 ) / _come ))x larger than anything its authors report.

  The two ways to make it fit are both disqualifying:
    * subsampling the QUERY changes the evaluated cell population, so the number
      would no longer be comparable with LUNA/G2T/CeLEry/CellContrast, which all
      score the full sections;
    * chunking the query gives each chunk its own mapping matrix and its own
      contrastive negatives — a different method, not the published one (unlike
      CellContrast's chunking, which is provably bit-identical).

  So cns_luna is reported as outside COME's envelope, with this arithmetic,
  rather than as a poor COME score. COME runs on mmc_luna with the FULL test
  slices (<=5,235 cells) against a subsampled reference:

      bash $(basename "$0") --dataset mmc_luna --max_ref_cells 20000
EOF
  exit 2
fi
DATASETS="mmc_luna"
case "$DATASET" in
  mmc_luna) ;;
  "") echo "ERROR: --dataset is required ($DATASETS)." >&2; exit 2 ;;
  *)  echo "ERROR: unknown --dataset '$DATASET' (this submitter accepts $DATASETS" >&2
      echo "       only; COME is O((n_ref+n_query)^2) and mmc_luna is the one" >&2
      echo "       benchmark dataset whose test sections fit — see the header," >&2
      echo "       and '--dataset cns_luna' for the worked arithmetic)." >&2
      exit 2 ;;
esac

[[ -f "$RUNNER" ]] || { echo "ERROR: runner missing: $RUNNER" >&2; exit 1; }
# The four modules run_come.py imports unmodified from the authors' repo
# (_import_come checks the same list; failing here saves the queue wait).
for _f in model.py configure.py train_eval.py utils.py; do
  [[ -f "$REPO/$_f" ]] || {
    echo "ERROR: COME not found at $REPO (missing $_f)." >&2
    echo "       Run: bash $HERE/../setup_come_env.sh" >&2; exit 1; }
done
[[ -f "$VENV_DIR/bin/activate" ]] || {
  echo "ERROR: uv venv not found at $VENV_DIR" >&2
  echo "       Run: bash $HERE/../setup_come_env.sh" >&2; exit 1; }
[[ -n "$DATA_DIR" ]] || DATA_DIR="/nfs/team361/sb75/DATASETS/silver/$DATASET"
[[ -d "$DATA_DIR" ]] || { echo "ERROR: data dir not found: $DATA_DIR" >&2; exit 1; }
# The runner needs BOTH splits (the reference is built from *_train.h5ad, the
# queries are the *_test.h5ad); it raises on a missing one, but only after the
# queue wait and the venv start-up.
shopt -s nullglob
_TRAIN_FILES=("$DATA_DIR"/*_train.h5ad)
_TEST_FILES=("$DATA_DIR"/*_test.h5ad)
shopt -u nullglob
if [[ ${#_TRAIN_FILES[@]} -eq 0 || ${#_TEST_FILES[@]} -eq 0 ]]; then
  echo "ERROR: $DATA_DIR has ${#_TRAIN_FILES[@]} *_train.h5ad and ${#_TEST_FILES[@]} *_test.h5ad;" >&2
  echo "       the runner needs at least one of each (reference from train," >&2
  echo "       queries from test). Check: ls $DATA_DIR" >&2
  exit 1
fi

# --- enum / numeric validation ------------------------------------------------
# Every one of these is an argparse 'choices' or a typed value in run_come.py, so
# a typo would otherwise surface as an argparse exit INSIDE the LSF job, i.e.
# after the queue wait, for every seed.
case "$COME_CONFIG" in
  dro|smFISH|MERFISH|STARmap|PDAC) ;;
  *) echo "ERROR: --come_config must be one of dro smFISH MERFISH STARmap PDAC" >&2
     echo "       (the authors' presets in configure.py); got '$COME_CONFIG'." >&2
     exit 2 ;;
esac
case "$STTYPE" in
  image|sequence) ;;
  *) echo "ERROR: --sttype must be image or sequence (got '$STTYPE')." >&2; exit 2 ;;
esac
case "$DEVICE" in
  auto|cpu) ;;
  *) echo "ERROR: --device must be auto or cpu (got '$DEVICE')." >&2; exit 2 ;;
esac
case "$ON_EMPTY_CELLS" in
  fail|keep|drop) ;;
  *) echo "ERROR: --on_empty_cells must be fail, keep or drop (got '$ON_EMPTY_CELLS')." >&2
     exit 2 ;;
esac
# --max_ref_cells is n_ref in the estimate below, so it feeds shell arithmetic
# AND the LSF reservation. The CellContrast submitter learned this the hard way:
# a non-numeric value became a bare bash arithmetic error, and '0' produced a
# nonsense estimate and submitted anyway.
[[ "$MAX_REF_CELLS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --max_ref_cells must be a positive integer (got '$MAX_REF_CELLS')." >&2
  echo "       It is the reference size n_ref in the O((n_ref+n_query)^2) memory" >&2
  echo "       estimate that sizes the LSF reservation, so 0, a negative number" >&2
  echo "       or a non-number cannot be turned into a memory plan." >&2
  exit 2; }
# Upper bound, purely so the estimate stays inside 64-bit shell arithmetic: at
# n_ref ~ 6.8e8 the dominant 20*(n_ref+n_query)^2 term exceeds 2^63-1 and WRAPS
# NEGATIVE, and a negative estimate sails through BOTH feasibility guards below
# ("> 256" and "> budget" are false for it) — i.e. the script would print a
# nonsense figure like "-4466498436.-3GB" and submit every seed anyway. Eight
# digits keeps the worst term at 20*(1e8)^2 = 2e17, well inside the range, and
# every value above it is refused by the 256GB ceiling regardless (that ceiling
# caps n_ref+n_query at ~113,137).
if [[ "${#MAX_REF_CELLS}" -gt 8 ]]; then
  echo "ERROR: --max_ref_cells $MAX_REF_CELLS is out of range (max 99999999)." >&2
  echo "       Above that the 20*(n_ref+n_query)^2 term overflows 64-bit shell" >&2
  echo "       arithmetic and wraps negative, which would silently pass the" >&2
  echo "       feasibility checks below. Nothing that large is runnable anyway:" >&2
  echo "       at run_come.py's 256GB ceiling n_ref+n_query must be" >&2
  echo "       <= ~$(commafy $(envelope 256)), and the decided protocol is 20000." >&2
  exit 2
fi
for _pair in "--epochs:$EPOCHS" "--pretrain_epochs:$PRETRAIN_EPOCHS" "--cores:$CORES"; do
  _name="${_pair%%:*}"; _val="${_pair#*:}"
  [[ -z "$_val" || "$_val" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: $_name must be a positive integer (got '$_val')." >&2; exit 2; }
done
[[ -z "$MEM_MB" || "$MEM_MB" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --mem must be an integer number of MB (got '$MEM_MB')." >&2; exit 2; }
[[ -z "$MAX_MEM_GB" || "$MAX_MEM_GB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || {
  echo "ERROR: --max_mem_gb must be a number of GB (got '$MAX_MEM_GB')." >&2; exit 2; }
# Each seed becomes the runner's --seed (argparse type=int, and the RNG that
# subsamples the reference), plus the job name, the log names and the per-seed
# artifact timestamp. An unvalidated token therefore costs one argparse exit per
# seed AFTER the queue wait, and an empty list submits nothing at all while the
# closing "when the jobs finish" block still prints. ('set -f' for the loop only:
# we want the word SPLITTING but not globbing, as in the exclude loop below.)
set -f
N_SEEDS=0
for _s in $SEEDS; do
  [[ "$_s" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "ERROR: --seeds must be whitespace-separated non-negative integers," >&2
    echo "       without leading zeros; got '$_s' in '$SEEDS'. Each seed is both" >&2
    echo "       passed to the runner's --seed and used in the job, log and" >&2
    echo "       artifact names, so e.g. '07' would name a job seed07 while the" >&2
    echo "       manifest recorded seed 7." >&2
    exit 2; }
  N_SEEDS=$(( N_SEEDS + 1 ))
done
set +f
[[ "$N_SEEDS" -gt 0 ]] || {
  echo "ERROR: --seeds is empty, so there is nothing to submit." >&2
  echo "       Pass e.g. --seeds \"0 1 2 3 4\" (the default)." >&2; exit 2; }

# --- per-dataset table --------------------------------------------------------
# WORST_Q is the largest *_test.h5ad section, in cells: the query size that sets
# the peak, since COME refits per test slice and every slice is run whole. The
# runner recomputes the estimate from the REAL slice and refuses against
# --max_mem_gb, so this is the up-front upper bound, not the authority.
# CFG_EXPECT/STTYPE_EXPECT are the presets that reproduce the authors' setting
# for this platform; departing from them is allowed but is a deviation.
case "$DATASET" in
  # Imaging-resolution mouse cortex (MERFISH-like). Sections max out at 5,235
  # cells; the reference is subsampled, the query never is.
  mmc_luna)
    WORST_Q=5235; CFG_EXPECT="MERFISH"; STTYPE_EXPECT="image" ;;
  # Unreachable via the allow-list above; here so that adding a dataset there
  # and forgetting this table fails with a sentence instead of "unbound variable".
  *) echo "ERROR: '$DATASET' has no entry in the per-dataset table; add one." >&2
     exit 3 ;;
esac
if [[ "$COME_CONFIG" != "$CFG_EXPECT" ]]; then
  echo "WARNING: --come_config $COME_CONFIG on $DATASET, whose platform matches the" >&2
  echo "         authors' $CFG_EXPECT preset. The presets differ in k, the layer" >&2
  echo "         dims and the epoch counts, so this is a hyperparameter deviation." >&2
fi
if [[ "$STTYPE" != "$STTYPE_EXPECT" ]]; then
  echo "WARNING: --sttype $STTYPE on $DATASET, which is $STTYPE_EXPECT-resolution ST." >&2
  echo "         This switches upstream's preprocessing (MinMaxScaler vs" >&2
  echo "         normalize_total+log1p) and is a deviation from the authors'" >&2
  echo "         setting for this platform." >&2
fi
# Basenames are matched EXACTLY by the runner, so a typo excludes nothing and the
# mean would silently cover a different set of sections than the other methods.
# (Unquoted expansion on purpose: it splits on the commas we substitute and on
# stray whitespace, matching the runner's per-entry .strip(). 'set -f' for the
# loop only — we want the SPLITTING but not globbing, or an entry like
# '*_test.h5ad' would expand against the current directory and the warning would
# name a file the user never typed.)
if [[ -n "$EXCLUDE_TEST_FILES" ]]; then
  set -f
  for _x in ${EXCLUDE_TEST_FILES//,/ }; do
    if [[ ! -f "$DATA_DIR/$_x" ]]; then
      echo "WARNING: --exclude_test_files entry '$_x' is not in $DATA_DIR." >&2
      echo "         Basenames are matched exactly, so a typo excludes NOTHING" >&2
      echo "         and the mean would cover a different set of sections than" >&2
      echo "         LUNA/G2T/CeLEry. Check: ls $DATA_DIR/*_test.h5ad" >&2
    fi
  done
  set +f
fi

if [[ -n "$SMOKE" ]]; then
  # The runner caps the reference at min(--max_ref_cells, 500) and runs 2 epochs
  # in smoke mode, so the fit is ~1GB or less whatever --max_ref_cells says:
  # reserve small, and base the estimate below on the CAPPED reference so the
  # printed number is an upper bound on the one the job will report (exact unless
  # --max_ref_cells is itself below 500).
  SEEDS="0"; WALL="1:00"; MEM_MB="16000"; SMOKE_REF=500
  REF_NOTE=" (smoke: the runner caps it at min(--max_ref_cells, 500))"
  echo "SMOKE TEST: 1 job, 1h wall — proves the install; do not report the numbers."
fi

# --- memory plan --------------------------------------------------------------
# Same formula as the runner, evaluated here so an impossible plan never reaches
# the queue. n_ref is exact (--max_ref_cells is always passed on), n_query is the
# worst-case section, so EST is an upper bound on the runner's own estimate; the
# runner then recomputes it per slice from the REAL reference and query and
# refuses against --max_mem_gb.
REF_EST="${SMOKE_REF:-$MAX_REF_CELLS}"
EST_BYTES="$(est_peak_bytes "$REF_EST" "$WORST_Q")"
EST_GB1="$(gb1 "$EST_BYTES")"
EST_GB="$(gb_ceil "$EST_BYTES")"

# Refuse what the runner would refuse anyway (its HARD_CEILING_GB is 256 and
# --force_scale is deliberately not exposed here), before we ask LSF for a
# reservation no node can satisfy.
if [[ "$EST_GB" -gt 256 ]]; then
  echo "ERROR: estimated peak ${EST_GB1}GB (reference $REF_EST, worst-case query" >&2
  echo "       $WORST_Q) exceeds run_come.py's 256GB hard feasibility ceiling," >&2
  echo "       so the job would refuse to start. At 256GB n_ref+n_query must be" >&2
  echo "       <= ~$(commafy $(envelope 256)), and the query alone is $(commafy $WORST_Q)." >&2
  echo "       Lower --max_ref_cells (the decided protocol is 20000, ~15GB)." >&2
  exit 2
fi

# Size the reservation FROM the estimate: reserve enough that 85% of it still
# covers the estimate, keeping the other 15% for torch, the loaded AnnData
# objects and the interpreter, which the formula does not count. Rounded up to a
# whole 4GB, with a 16GB floor (below that the venv + torch import dominate).
if [[ -z "$MEM_MB" ]]; then
  _res_gb=$(( (EST_GB * 100 + 84) / 85 ))
  _res_gb=$(( ( (_res_gb + 3) / 4 ) * 4 ))
  [[ "$_res_gb" -ge 16 ]] || _res_gb=16
  MEM_MB=$(( _res_gb * 1000 ))
  MEM_SRC="derived from the ${EST_GB1}GB estimate"
elif [[ -n "$SMOKE" ]]; then
  # The smoke block above overwrites MEM_MB unconditionally, so naming --mem here
  # would point at a knob the user may never have touched.
  MEM_SRC="smoke-test override"
else
  MEM_SRC="--mem"
fi
# BUDGET_SRC exists so the messages below name where the budget actually came
# from instead of asserting "85% of the reservation" even when it was an explicit
# --max_mem_gb, which sends the reader to the wrong knob.
if [[ -n "$MAX_MEM_GB" ]]; then
  BUDGET_SRC="--max_mem_gb"
else
  MAX_MEM_GB=$(( MEM_MB * 85 / 100000 ))
  BUDGET_SRC="85% of the ${MEM_MB}MB LSF reservation"
fi
# Whole GB for the shell-side comparison only: [[ -gt ]] is integer arithmetic,
# and a fractional --max_mem_gb there is a syntax error that quietly evaluates
# FALSE, i.e. it would disable the check it looks like it performs. The runner
# still receives the exact value.
BUDGET_GB="${MAX_MEM_GB%%.*}"; BUDGET_GB="${BUDGET_GB:-0}"
if [[ "$BUDGET_GB" -lt 1 ]]; then
  echo "ERROR: the fit memory budget rounds down to ${BUDGET_GB}GB" >&2
  echo "       (${MAX_MEM_GB}GB, from ${BUDGET_SRC})." >&2
  echo "       The runner treats 0 as 'no budget' and skips the check, and" >&2
  echo "       nothing here runs in under 1GB. Raise --mem (or --max_mem_gb)." >&2
  exit 2
fi
if [[ "$EST_GB" -gt "$BUDGET_GB" ]]; then
  echo "ERROR: estimated peak ${EST_GB1}GB exceeds the ${MAX_MEM_GB}GB budget" >&2
  echo "       (${BUDGET_SRC}), with reference $REF_EST" >&2
  echo "       (exact, --max_ref_cells) and worst-case query $WORST_Q." >&2
  echo "       The runner would refuse this fit in its first seconds. At this" >&2
  echo "       budget n_ref+n_query must be <= ~$(commafy $(envelope $BUDGET_GB))." >&2
  echo "       Lower --max_ref_cells (peak grows ~quadratically in it), raise" >&2
  echo "       --mem, or — if you set --max_mem_gb by hand — drop it and let it" >&2
  echo "       be derived from the reservation." >&2
  exit 2
fi
if [[ "$MEM_MB" -gt 128000 ]]; then
  echo "WARNING: reserving ${MEM_MB}MB (${MEM_SRC}); not every" >&2
  echo "         $LSF_QUEUE node has that much, so the job may pend for a" >&2
  echo "         long time. The decided protocol (--max_ref_cells 20000) needs" >&2
  echo "         ~15GB." >&2
fi

# COME's GPU path can die in ContrastiveLoss (upstream builds full_mask on the
# CPU while cross_mask is on the model's device), so --device cpu is a real
# fallback — and a CPU job should not hold a GPU that another job could use.
# An explicit --gpu still wins, for the case where you want the allocation.
if [[ "$DEVICE" == "cpu" && -z "$GPU_EXPLICIT" ]]; then
  GPU_SPEC=""
fi

LOGDIR="$ARTIFACTS_ROOT/$DATASET/come_inference/lsf"
JOBDIR="$LOGDIR/jobs"
mkdir -p "$JOBDIR"

echo "== submit_come.sh =="
echo "dataset : $DATASET   ($DATA_DIR)"
echo "repo    : $REPO  (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "venv    : $VENV_DIR"
echo "seeds   : $SEEDS"
echo "preset  : --come_config $COME_CONFIG  --sttype $STTYPE  (authors' $CFG_EXPECT/$STTYPE_EXPECT for this platform)"
echo "epochs  : ${EPOCHS:-<preset>} train / ${PRETRAIN_EPOCHS:-<preset>} pretrain"
echo "protocol: transductive, refit per test slice; FULL query, reference capped at ${REF_EST}${REF_NOTE:-}"
echo "memory  : est peak ${EST_GB1}GB (n_ref $REF_EST + worst-case n_query $WORST_Q), runner budget ${MAX_MEM_GB}GB (${BUDGET_SRC})"
echo "device  : $DEVICE   (empty cells: $ON_EMPTY_CELLS)"
echo "excl    : ${EXCLUDE_TEST_FILES:-<none: all *_test.h5ad scored>}"
echo "lsf     : $LSF_QUEUE / $LSF_GROUP / ${MEM_MB}MB (${MEM_SRC}) / $WALL / ${CORES} cores / gpu ${GPU_SPEC:-<none>}"
[[ -n "$PLAN_ONLY" ]] && echo "plan_only: jobs print their per-slice plan and estimate, and fit NOTHING"
echo

PREV_TS=""
for SEED in $SEEDS; do
  TS="$(date +%Y%m%d_%H%M%S)"
  # Artifact dirs are keyed to the second, so two seeds sharing a timestamp would
  # write into the SAME <out_root>/.../come_inference/<TS> tree and overwrite each
  # other's per-slice CSVs and manifest. The sleep at the end of the loop is what
  # normally guarantees distinct values; this waits it out if it somehow did not.
  while [[ "$TS" == "$PREV_TS" ]]; do sleep 1; TS="$(date +%Y%m%d_%H%M%S)"; done
  PREV_TS="$TS"
  JOB="$JOBDIR/come_${DATASET}_seed${SEED}_${TS}.sh"

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'source %q/bin/activate\n' "$VENV_DIR"
    # Cheap general determinism belt, and it keeps the env identical to the runs
    # already recorded. (The seeding that matters is inside the runner: it seeds
    # numpy and torch itself.)
    echo "export PYTHONHASHSEED=0"
    # COME's fit is dense torch/BLAS work on (n_ref+n_query)^2 masks — and with
    # --device cpu it is ALL on the CPU. Without these the libraries size their
    # pools from the HOST core count, oversubscribing a shared node and stealing
    # slots we did not reserve.
    printf 'export OMP_NUM_THREADS=%q\n'      "$CORES"
    printf 'export MKL_NUM_THREADS=%q\n'      "$CORES"
    printf 'export OPENBLAS_NUM_THREADS=%q\n' "$CORES"
    echo 'echo "python: $(which python)"; python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"'
    printf 'exec python %q \\\n' "$RUNNER"
    printf '    --data_dir %q \\\n'        "$DATA_DIR"
    printf '    --come_repo %q \\\n'       "$REPO"
    printf '    --out_root %q \\\n'        "$ARTIFACTS_ROOT"
    printf '    --dataset %q \\\n'         "$DATASET"
    printf '    --run_timestamp %q \\\n'   "$TS"
    printf '    --seed %q \\\n'            "$SEED"
    # Always explicit, never inherited: the preset, the preprocessing switch, the
    # reference cap, the memory budget, the device and the empty-cell policy are
    # the settings whose wrong value is either silent or expensive.
    printf '    --come_config %q \\\n'     "$COME_CONFIG"
    printf '    --sttype %q \\\n'          "$STTYPE"
    printf '    --max_ref_cells %q \\\n'   "$MAX_REF_CELLS"
    printf '    --device %q \\\n'          "$DEVICE"
    printf '    --on_empty_cells %q \\\n'  "$ON_EMPTY_CELLS"
    printf '    --max_mem_gb %q'           "$MAX_MEM_GB"
    [[ -n "$EPOCHS" ]]             && printf ' \\\n    --epochs %q' "$EPOCHS"
    [[ -n "$PRETRAIN_EPOCHS" ]]    && printf ' \\\n    --pretrain_epochs %q' "$PRETRAIN_EPOCHS"
    [[ -n "$EXCLUDE_TEST_FILES" ]] && printf ' \\\n    --exclude_test_files %q' "$EXCLUDE_TEST_FILES"
    [[ -n "$SMOKE" ]]              && printf ' \\\n    --smoke_test'
    [[ -n "$PLAN_ONLY" ]]          && printf ' \\\n    --dry_run'
    echo
  } > "$JOB"
  chmod +x "$JOB"

  BSUB=(bsub
    -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$CORES"
    -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB]"
    -R "span[ptile=$CORES]")
  # No -gpu at all on the CPU path (see above): an empty -gpu argument is not the
  # same thing as omitting it.
  [[ -n "$GPU_SPEC" ]] && BSUB+=(-gpu "$GPU_SPEC")
  BSUB+=(-W "$WALL"
    -J "come_${DATASET}_s${SEED}"
    -o "$LOGDIR/come_${DATASET}_seed${SEED}_${TS}.%J.out"
    -e "$LOGDIR/come_${DATASET}_seed${SEED}_${TS}.%J.err"
    bash "$JOB")

  echo "seed $SEED -> $TS"
  echo "  job script: $JOB"
  if [[ -n "$DRY_RUN" ]]; then
    printf '  [dry_run] '; printf '%q ' "${BSUB[@]}"; echo
  else
    "${BSUB[@]}"
  fi
  sleep 2   # distinct timestamps: artifact dirs are keyed to the second
done

echo
echo "logs      : $LOGDIR"
echo "artifacts : $ARTIFACTS_ROOT/$DATASET/come_inference/<TS>/test_results/"
echo "            (run_manifest.json flips \"status\" to \"complete\" only at the end,"
echo "             so an in-flight or crashed run is distinguishable from a finished one)"
echo
echo "when the jobs finish, score with the SAME harness as the other methods:"
echo "  python $HERE/../plots/compute_extended_metrics.py \\"
echo "      --dataset mmc_luna --methods come"
