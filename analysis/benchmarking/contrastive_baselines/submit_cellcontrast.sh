#!/usr/bin/env bash
# submit_cellcontrast.sh
# ------------------------------------------------------------------------------
# Submit the reviewer-requested CellContrast baseline to LSF, one job per seed,
# mirroring how the other baselines are launched (submit_pipeline.sh conventions:
# same group/queue defaults, a generated per-job script, GPU request, logs beside
# the artifacts).
#
# Runs the AUTHORS' published code (https://github.com/HKU-BAL/CellContrast, MIT)
# at their default hyperparameters via run_cellcontrast.py. Nothing here changes
# the method.
#
# The imaging-vs-spot parameter choice (the paper's k_nearest_positives: 80 vs
# 20) and the normalisation mode are DERIVED from --dataset here, not left to the
# runner's defaults — see the per-dataset table in this script. The runner independently
# re-checks both and refuses a mismatch; that duplication is deliberate.
#
# Usage:
#   # 0) once: clone + env, then ALWAYS smoke-test before a real run
#   bash ../setup_cellcontrast_env.sh
#   bash submit_cellcontrast.sh --dataset mmc_luna --smoke_test
#
#   # 1) cortex, 5 seeds
#   bash submit_cellcontrast.sh --dataset mmc_luna --seeds "0 1 2 3 4"
#
#   # 2) CNS. The shared 600-d cross-platform latent lives in adata.X ITSELF —
#   #    the only obsm key on those h5ads is 'spatial' — so do NOT pass
#   #    --use_obsm; the latent is handed over untouched via
#   #    --expression_mode silver_raw (the default for this dataset). Needs a
#   #    training cap: 2.85M cells is ~9 days.
#   bash submit_cellcontrast.sh --dataset cns_luna --max_train_cells 150000 \
#        --exclude_test_files sagittal1_test.h5ad,sagittal2_test.h5ad,sagittal3_test.h5ad,spinalcord_test.h5ad \
#        --seeds "0 1 2 3 4"
#
#   # 3) 10x Xenium breast (imaging, k=80) and 10x Visium DLPFC (spot, k=20);
#   #    both are raw integer counts, so both default to silver_raw
#   bash submit_cellcontrast.sh --dataset breast_janesick --seeds "0 1 2 3 4"
#   bash submit_cellcontrast.sh --dataset dlpfc_visium    --seeds "0 1 2 3 4"
#
# Options:
#   --dataset NAME        mmc_luna | cns_luna | breast_janesick | dlpfc_visium
#                         (required)
#   --seeds "0 1 2"       seeds, one job each   (default "0 1 2 3 4")
#   --data_dir DIR        override the silver dir
#   --repo DIR            CellContrast checkout  (default $CELLCONTRAST_REPO, else
#                         <this dir>/../CellContrast, i.e. the clone
#                         setup_cellcontrast_env.sh makes beside the other
#                         benchmarking assets)
#   --venv DIR            uv venv                (default
#                         /nfs/team361/sb75/.venvs/cellcontrast)
#   --epochs N            override training_epoch. OMIT to use the authors'
#                         default (3000) — that is the defensible choice for a
#                         baseline. 1000 is ~3x faster and the paper says >1000
#                         suffices, but it IS a deviation and is recorded.
#   --expression_mode M   log2 | silver_raw. Default is PER DATASET (see table):
#                         log2(1+x) where silver X holds count-magnitude values,
#                         silver_raw where the matrix must be handed over
#                         untouched (the both-sign cns_luna latent, raw counts).
#   --coord_frame F       isotropic (default) | per_axis. isotropic preserves the
#                         aspect ratio, so upstream's k-NN positive graph is
#                         identical to raw microns; per_axis matches LUNA/G2T's
#                         position_normalize but changes 7-14% of the positive set.
#   --query_chunk N       cells per inference call (default per dataset, 8000).
#                         Upstream builds dense query x query AND query x ref
#                         matrices, so a 63k-cell CNS slice unchunked peaks near
#                         292 GB. top-1 retrieval is per-row independent, so
#                         chunking is bit-identical. 0 disables it.
#   --max_mem_gb G        make the runner REFUSE before inference if its estimated
#                         peak exceeds this (default: 85% of the LSF reservation,
#                         leaving headroom for torch and the loaded objects), so a
#                         bad memory plan fails in seconds, not 6 hours in.
#   --force_platform      pass through the runner's dataset-vs-parameter-file
#                         check. Only needed for a dataset whose name does not
#                         advertise its platform; it does NOT change the flag
#                         derived here.
#   --use_obsm KEY        feature matrix from adata.obsm[KEY] instead of .X.
#                         'spatial' is REFUSED on every dataset (it would feed
#                         ground-truth coordinates in as features), and any
#                         --use_obsm is refused on cns_luna (its latent is in .X).
#   --max_train_cells N   cap training cells by per-slice subsampling
#   --max_ref_cells N     cap the inference reference (memory)
#   --exclude_test_files  comma-separated *_test.h5ad basenames to skip, so the
#                         mean is over the SAME sections as G2T/LUNA/CeLEry.
#                         Required for cns_luna (4 of 18 sections are excluded
#                         there). Basenames are matched exactly; a name that is
#                         not in the data dir is warned about, not guessed.
#   --smoke_test          tiny 1-job run to prove the install; never reported
#   --mem MB / --wall HH:MM / --queue Q / --group G / --gpu SPEC
#   --dry_run             print the bsub commands without submitting
#
# Env overrides: CELLCONTRAST_REPO, VENV_DIR, SCGG_ARTIFACTS_ROOT, LSF_GROUP,
# LSF_QUEUE, MEM_MB, WALL, GPU_SPEC, EXCLUDE_TEST_FILES, and CORES (default 8:
# the LSF slot count, also exported as OMP/MKL/OPENBLAS_NUM_THREADS in the job so
# upstream's per-row argsort does not oversubscribe a shared node).
# ------------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_cellcontrast.py"

DATASET=""
SEEDS="0 1 2 3 4"
DATA_DIR=""
# Defaults match setup_cellcontrast_env.sh: the authors' code is cloned next to
# the other benchmarking assets, and the env is a uv venv (this cluster has no
# conda on PATH).
REPO="${CELLCONTRAST_REPO:-$HERE/../CellContrast}"
VENV_DIR="${VENV_DIR:-/nfs/team361/sb75/.venvs/cellcontrast}"
EPOCHS=""
USE_OBSM=""
MAX_TRAIN_CELLS=""
MAX_REF_CELLS=""
# Empty means "take the per-dataset default from the table below". Anything set
# here (flag or env) wins, except where a gate proves it wrong.
EXPRESSION_MODE=""
COORD_FRAME="${COORD_FRAME:-isotropic}"
QUERY_CHUNK=""
MAX_MEM_GB=""
FORCE_PLATFORM=""
# Comma-separated *_test.h5ad basenames to skip. The runner supports this but the
# submitter never forwarded it, so a CNS run scored all 18 test sections while
# G2T/LUNA/CeLEry score 14 (sagittal1/2/3 + spinalcord excluded) — a mean over a
# different population of slices, i.e. not a comparable number.
EXCLUDE_TEST_FILES="${EXCLUDE_TEST_FILES:-}"
SMOKE=""
DRY_RUN=""
ARTIFACTS_ROOT="${SCGG_ARTIFACTS_ROOT:-/nfs/team361/sb75/scgg-reproducibility/artifacts}"
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
# Empty means "per-dataset default" (see the table); the old flat 128000 was set
# for mmc-sized slices and is not the right reservation for the CNS/Xenium runs.
MEM_MB="${MEM_MB:-}"
WALL="${WALL:-48:00}"
CORES="${CORES:-8}"
GPU_SPEC="${GPU_SPEC:-num=1:mode=shared:j_exclusive=no}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)          DATASET="${2:?}"; shift 2 ;;
    --seeds)            SEEDS="${2:?}"; shift 2 ;;
    --data_dir)         DATA_DIR="${2:?}"; shift 2 ;;
    --repo)             REPO="${2:?}"; shift 2 ;;
    --venv)             VENV_DIR="${2:?}"; shift 2 ;;
    --epochs)           EPOCHS="${2:?}"; shift 2 ;;
    --expression_mode)  EXPRESSION_MODE="${2:?}"; shift 2 ;;
    --coord_frame)      COORD_FRAME="${2:?}"; shift 2 ;;
    --query_chunk)      QUERY_CHUNK="${2:?}"; shift 2 ;;
    --max_mem_gb)       MAX_MEM_GB="${2:?}"; shift 2 ;;
    --force_platform)   FORCE_PLATFORM=1; shift ;;
    --use_obsm)         USE_OBSM="${2:?}"; shift 2 ;;
    --max_train_cells)  MAX_TRAIN_CELLS="${2:?}"; shift 2 ;;
    --max_ref_cells)    MAX_REF_CELLS="${2:?}"; shift 2 ;;
    --exclude_test_files) EXCLUDE_TEST_FILES="${2:?}"; shift 2 ;;
    --smoke_test)       SMOKE=1; shift ;;
    --mem)              MEM_MB="${2:?}"; shift 2 ;;
    --wall)             WALL="${2:?}"; shift 2 ;;
    --queue)            LSF_QUEUE="${2:?}"; shift 2 ;;
    --group)            LSF_GROUP="${2:?}"; shift 2 ;;
    --gpu)              GPU_SPEC="${2:?}"; shift 2 ;;
    --dry_run)          DRY_RUN=1; shift ;;
    # Print the header comment block, whatever its length: the old fixed range
    # ('2,52p') stopped being the header the moment it grew and printed code.
    -h|--help)          awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next}
                             {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- validation --------------------------------------------------------------
DATASETS="mmc_luna | cns_luna | breast_janesick | dlpfc_visium"
case "$DATASET" in
  mmc_luna|cns_luna|breast_janesick|dlpfc_visium) ;;
  "") echo "ERROR: --dataset is required ($DATASETS)." >&2; exit 2 ;;
  *)  echo "ERROR: unknown --dataset '$DATASET' (expected $DATASETS)." >&2; exit 2 ;;
esac
[[ -f "$RUNNER" ]] || { echo "ERROR: runner missing: $RUNNER" >&2; exit 1; }
[[ -f "$REPO/cellContrast.py" ]] || {
  echo "ERROR: CellContrast not found at $REPO (expected cellContrast.py)." >&2
  echo "       Run: bash $HERE/../setup_cellcontrast_env.sh" >&2; exit 1; }
[[ -f "$VENV_DIR/bin/activate" ]] || {
  echo "ERROR: uv venv not found at $VENV_DIR" >&2
  echo "       Run: bash $HERE/../setup_cellcontrast_env.sh" >&2; exit 1; }
[[ -n "$DATA_DIR" ]] || DATA_DIR="/nfs/team361/sb75/DATASETS/silver/$DATASET"
[[ -d "$DATA_DIR" ]] || { echo "ERROR: data dir not found: $DATA_DIR" >&2; exit 1; }

# --- per-dataset platform / normalisation / memory table ---------------------
# The paper's k_nearest_positives is platform-specific — 80 for imaging ST
# (parameters_singleCell.json), 20 for spot ST (parameters_spot.json) — and that
# is the ONLY difference between the authors' two parameter files. Getting it
# wrong is silent: nothing downstream looks broken. So the flag is DERIVED here
# per dataset instead of inheriting the runner's default (which is imaging).
# Likewise the normalisation mode, which depends on what the silver .X holds.
# REF_HINT is only used for the memory estimate printed below (the reference size
# we expect at inference); the runner recomputes the estimate from the REAL
# reference and refuses if it busts --max_mem_gb.
case "$DATASET" in
  # MERFISH cortex. Silver .X holds per-cell-normalised count-magnitude values,
  # so log2(1+x) composes into the paper's scran logNormCounts. Slices max out
  # at ~5,235 cells — below any sane chunk size, so inference keeps taking the
  # original single unchunked call.
  mmc_luna)
    SC_FLAG="--single_cell";    K_POS=80; DEF_EXPR="log2"
    DEF_CHUNK=8000; DEF_MEM=128000; REF_HINT=160000 ;;
  # Cross-platform CNS. The shared 600-d latent is in .X itself, and it is
  # both-sign, so log2(1+x) is undefined there: silver_raw hands it over
  # untouched (the runner raises on log2 for exactly this reason).
  cns_luna)
    SC_FLAG="--single_cell";    K_POS=80; DEF_EXPR="silver_raw"
    DEF_CHUNK=8000; DEF_MEM=192000; REF_HINT=150000 ;;
  # 10x Xenium breast, 313-gene panel, imaging resolution. .X is RAW INTEGER
  # counts with no size-factor step, so log2(1+x) alone would NOT reproduce
  # scran logNormCounts (it is the log without the normalisation) — a
  # half-transform is worse than none, so hand the counts over untouched and
  # record that. Test slices are ~118,752 and ~27,472 cells: chunking is what
  # makes them fit at all.
  breast_janesick)
    SC_FLAG="--single_cell";    K_POS=80; DEF_EXPR="silver_raw"
    DEF_CHUNK=8000; DEF_MEM=192000; REF_HINT=200000 ;;
  # 10x Visium DLPFC: SPOT ST, so k=20 and parameters_spot.json — this is the
  # one dataset here where --no_single_cell is mandatory. ~3.6k spots/section,
  # raw integer counts (same argument as breast_janesick).
  dlpfc_visium)
    SC_FLAG="--no_single_cell"; K_POS=20; DEF_EXPR="silver_raw"
    DEF_CHUNK=8000; DEF_MEM=64000;  REF_HINT=40000 ;;
  # Unreachable via the allow-list above; here so that adding a dataset there
  # and forgetting this table fails with a sentence instead of "unbound variable".
  *) echo "ERROR: '$DATASET' has no entry in the per-dataset table; add one." >&2
     exit 3 ;;
esac
EXPRESSION_MODE="${EXPRESSION_MODE:-$DEF_EXPR}"
QUERY_CHUNK="${QUERY_CHUNK:-$DEF_CHUNK}"
MEM_MB="${MEM_MB:-$DEF_MEM}"

# Typos here would otherwise surface as an argparse error inside the LSF job,
# i.e. after the queue wait.
case "$EXPRESSION_MODE" in
  log2|silver_raw) ;;
  *) echo "ERROR: --expression_mode must be log2 or silver_raw (got '$EXPRESSION_MODE')." >&2
     exit 2 ;;
esac
case "$COORD_FRAME" in
  isotropic|per_axis) ;;
  *) echo "ERROR: --coord_frame must be isotropic or per_axis (got '$COORD_FRAME')." >&2
     exit 2 ;;
esac
[[ "$QUERY_CHUNK" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --query_chunk must be a non-negative integer (got '$QUERY_CHUNK')." >&2
  exit 2; }

# The CNS latent is in .X, NOT in an obsm key: the only obsm key on those h5ads
# is 'spatial'. The previous gate here demanded --use_obsm for cns_luna and so
# refused every correct submission, while its own hint (print the obsm keys)
# pointed straight at 'spatial' — i.e. at feeding ground-truth coordinates in as
# features. Both halves are inverted below.
if [[ "$DATASET" == "cns_luna" && -n "$USE_OBSM" ]]; then
  cat >&2 <<'EOF'
ERROR: do not pass --use_obsm for cns_luna.
  The shared 600-d cross-platform latent that G2T/LUNA consume on this dataset
  IS adata.X on these h5ads; the only obsm key present is 'spatial' (the
  ground-truth coordinates). Drop --use_obsm — the latent is handed to the
  encoder untouched by the dataset default --expression_mode silver_raw.
EOF
  exit 2
fi
# obsm['spatial'] is the ground truth on every silver dir, so this is a truth
# leak on every dataset, not just CNS: the encoder would be handed the answer as
# a feature and every metric would be meaningless while looking excellent. The
# runner refuses this too; refusing here saves the queue wait.
if [[ "$USE_OBSM" == "spatial" ]]; then
  cat >&2 <<'EOF'
ERROR: --use_obsm spatial is refused on every dataset.
  obsm['spatial'] holds the GROUND-TRUTH coordinates. Handing them to the
  encoder as features defeats the coordinate withholding at inference and
  invalidates every reported metric. If you meant the CNS latent: that is in
  .X, so drop --use_obsm entirely.
EOF
  exit 2
fi
# The latent is both-sign, so log2(1+x) is not merely wrong here, it is
# undefined; the runner raises on min <= -1 after loading the first slice.
if [[ "$DATASET" == "cns_luna" && "$EXPRESSION_MODE" != "silver_raw" ]]; then
  cat >&2 <<'EOF'
ERROR: cns_luna requires --expression_mode silver_raw.
  .X is the shared 600-d latent, which has negative values; log2(1+x) is
  undefined there (the runner raises rather than emit NaN and train on it).
  Drop --expression_mode and take the dataset default.
EOF
  exit 2
fi
# Raw integer counts: log2 without a size-factor step is not the paper's
# preprocessing, but it is well defined, so warn instead of refusing.
if [[ "$EXPRESSION_MODE" == "log2" && "$DEF_EXPR" == "silver_raw" ]]; then
  echo "WARNING: --expression_mode log2 on $DATASET, whose .X is raw integer" >&2
  echo "         counts. log2(1+x) alone is NOT scran logNormCounts (no" >&2
  echo "         size-factor normalisation), so this is a deviation from both" >&2
  echo "         the paper and the dataset default (silver_raw)." >&2
fi
# Same reasoning as the obsm gates: scoring a different set of test sections than the
# other three methods is not a comparable number, so refuse rather than warn.
if [[ "$DATASET" == "cns_luna" && -z "$EXCLUDE_TEST_FILES" && -z "$SMOKE" ]]; then
  cat >&2 <<'EOF'
ERROR: --exclude_test_files is required for cns_luna.
  G2T/LUNA/CeLEry score 14 of the 18 CNS test sections (sagittal1/2/3 and
  spinalcord excluded); scoring all 18 averages over a different population of
  slices and is not comparable. List the exact basenames used by the other
  methods (see plots/compute_extended_metrics.py and the scgg/luna submitters),
  then pass them comma-separated, e.g.
    --exclude_test_files sagittal1_test.h5ad,sagittal2_test.h5ad,sagittal3_test.h5ad,spinalcord_test.h5ad
  Verify the basenames against `ls "$DATA_DIR"` before submitting.
EOF
  exit 2
fi
if [[ "$DATASET" == "cns_luna" && -z "$MAX_TRAIN_CELLS" && -z "$SMOKE" ]]; then
  echo "WARNING: cns_luna without --max_train_cells will try to train on ~2.85M" >&2
  echo "         cells (order of a week). Consider --max_train_cells 150000." >&2
fi
# The runner drops test files by EXACT basename, so a typo excludes nothing and
# the run silently scores more sections than the other methods — the very thing
# the gate above exists to prevent. Warn rather than fail: --data_dir may point
# at a staging copy, and the runner is the authority on what it finds.
# (Unquoted expansion on purpose: it splits on the commas we substitute AND on
# stray whitespace, matching the runner's per-entry .strip().)
if [[ -n "$EXCLUDE_TEST_FILES" ]]; then
  for _x in ${EXCLUDE_TEST_FILES//,/ }; do
    if [[ ! -f "$DATA_DIR/$_x" ]]; then
      echo "WARNING: --exclude_test_files entry '$_x' is not in $DATA_DIR." >&2
      echo "         Basenames are matched exactly, so a typo excludes NOTHING" >&2
      echo "         and the mean would cover a different set of sections than" >&2
      echo "         G2T/LUNA/CeLEry. Check: ls $DATA_DIR/*_test.h5ad" >&2
    fi
  done
fi
if [[ -n "$SMOKE" ]]; then
  # Small chunks as well as small resources: the runner's smoke path takes 2
  # whole train slices as the reference (no --max_train_cells), so on Xenium/CNS
  # the query x reference matrix would bust a 32 GB reservation at the normal
  # 8000-cell chunk and the smoke test would die on a memory refusal instead of
  # proving the install. Chunking is bit-identical and cheaper per call, so this
  # costs nothing but call count.
  SEEDS="0"; WALL="1:00"; MEM_MB="32000"; QUERY_CHUNK=2000
  echo "SMOKE TEST: 1 job, 1h wall — proves the install; do not report the numbers."
fi

# --- inference memory plan ----------------------------------------------------
# The runner's own formula (estimate_peak_bytes): one upstream inference call
# holds a float32 query x query similarity matrix, a float32 sorted copy and an
# int64 argsort index — and the query x query matrix is built UNCONDITIONALLY,
# before the enable_denovo guard, so it stays resident while map_to_ST builds
# query x reference:
#     peak = max(24*Q^2, 16*Q^2 + 24*Q*R)  bytes
# Q here is the chunk size, which is an UPPER BOUND on the real Q (a slice
# smaller than the chunk peaks lower — that is why mmc_luna is unaffected), and R
# is the reference size. --max_mem_gb hands the budget to the runner so it
# refuses in seconds instead of being OOM-killed hours into inference; we keep
# 15% of the LSF reservation back for torch, the loaded AnnData objects and the
# interpreter, which the formula does not count.
REF_EST="${MAX_REF_CELLS:-$REF_HINT}"
MAX_MEM_GB="${MAX_MEM_GB:-$(( MEM_MB * 85 / 100000 ))}"
if [[ "$QUERY_CHUNK" -gt 0 ]]; then
  _A=$(( 24 * QUERY_CHUNK * QUERY_CHUNK ))
  _B=$(( 16 * QUERY_CHUNK * QUERY_CHUNK + 24 * QUERY_CHUNK * REF_EST ))
  EST_GB=$(( ( (_A > _B ? _A : _B) + 999999999 ) / 1000000000 ))
  EST_NOTE="<=${EST_GB}GB (ref ${REF_EST})"
  if [[ "$EST_GB" -gt "$MAX_MEM_GB" ]]; then
    # Hard error only when R is exact (--max_ref_cells given); with REF_HINT the
    # number is our guess, so warn and let the runner do the authoritative check
    # against the reference it actually builds.
    if [[ -n "$MAX_REF_CELLS" ]]; then _LVL=ERROR; else _LVL=WARNING; fi
    echo "$_LVL: estimated inference peak ${EST_GB}GB exceeds the ${MAX_MEM_GB}GB" >&2
    echo "    budget (85% of the ${MEM_MB}MB LSF reservation), with reference" >&2
    echo "    size ${REF_EST}${MAX_REF_CELLS:+ (exact, --max_ref_cells)}." >&2
    echo "    Lower --query_chunk (the peak is linear in it), cap the reference" >&2
    echo "    with --max_ref_cells, or raise --mem." >&2
    if [[ "$_LVL" == ERROR ]]; then exit 2; fi
  fi
else
  EST_NOTE="UNKNOWN (chunking disabled)"
  echo "WARNING: --query_chunk 0 disables chunking, so the peak is set by the" >&2
  echo "         largest test slice: a 63,343-cell CNS slice needs ~292GB and a" >&2
  echo "         118,752-cell Xenium slice far more. The runner will refuse" >&2
  echo "         against --max_mem_gb ${MAX_MEM_GB}." >&2
fi

LOGDIR="$ARTIFACTS_ROOT/$DATASET/cellcontrast_inference/lsf"
JOBDIR="$LOGDIR/jobs"
mkdir -p "$JOBDIR"

echo "== submit_cellcontrast.sh =="
echo "dataset : $DATASET   ($DATA_DIR)"
echo "repo    : $REPO  (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "venv    : $VENV_DIR"
echo "seeds   : $SEEDS"
echo "epochs  : ${EPOCHS:-<upstream default 3000>}"
echo "platform: $SC_FLAG  (k_nearest_positives=$K_POS, derived from the dataset)"
echo "features: ${USE_OBSM:+obsm['$USE_OBSM']}${USE_OBSM:-.X}, expression_mode $EXPRESSION_MODE"
echo "frame   : $COORD_FRAME"
echo "memory  : chunk $QUERY_CHUNK, est peak $EST_NOTE, runner budget ${MAX_MEM_GB}GB"
echo "excl    : ${EXCLUDE_TEST_FILES:-<none: all *_test.h5ad scored>}"
echo "lsf     : $LSF_QUEUE / $LSF_GROUP / ${MEM_MB}MB / $WALL / ${CORES} cores / gpu $GPU_SPEC"
echo

for SEED in $SEEDS; do
  TS="$(date +%Y%m%d_%H%M%S)"
  JOB="$JOBDIR/cellcontrast_${DATASET}_seed${SEED}_${TS}.sh"

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'source %q/bin/activate\n' "$VENV_DIR"
    # PYTHONHASHSEED is NOT what makes the feature column order reproducible —
    # an earlier comment here claimed that and was wrong. inference.format_query
    # reindexes through the ORDERED train_genes list saved in the checkpoint and
    # uses a set only for the membership test, so column order is deterministic
    # at any hash seed. The real nondeterminism was upstream's unseeded stdlib
    # random.shuffle (loadData.py:107, :129), which the runner now seeds itself
    # via a generated launcher. Kept only as a cheap general determinism belt and
    # so the env matches the runs already recorded.
    echo "export PYTHONHASHSEED=0"
    # Upstream's inference sorts each row of the query x reference similarity
    # matrix (numpy/BLAS bound). Without these, the libraries size their pools
    # from the HOST core count and oversubscribe a shared node — slower, and it
    # steals slots we did not reserve.
    printf 'export OMP_NUM_THREADS=%q\n'      "$CORES"
    printf 'export MKL_NUM_THREADS=%q\n'      "$CORES"
    printf 'export OPENBLAS_NUM_THREADS=%q\n' "$CORES"
    echo 'echo "python: $(which python)"; python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"'
    printf 'exec python %q \\\n' "$RUNNER"
    printf '    --data_dir %q \\\n'           "$DATA_DIR"
    printf '    --cellcontrast_repo %q \\\n'  "$REPO"
    printf '    --out_root %q \\\n'           "$ARTIFACTS_ROOT"
    printf '    --dataset %q \\\n'            "$DATASET"
    printf '    --run_timestamp %q \\\n'      "$TS"
    printf '    --seed %q \\\n'               "$SEED"
    # Always explicit, never inherited: the platform flag, the normalisation and
    # the memory plan are the three settings whose wrong value is silent.
    printf '    %s \\\n'                       "$SC_FLAG"
    printf '    --expression_mode %q \\\n'     "$EXPRESSION_MODE"
    printf '    --coord_frame %q \\\n'         "$COORD_FRAME"
    printf '    --query_chunk %q \\\n'         "$QUERY_CHUNK"
    printf '    --max_mem_gb %q'               "$MAX_MEM_GB"
    [[ -n "$FORCE_PLATFORM" ]]  && printf ' \\\n    --force_platform'
    [[ -n "$EPOCHS" ]]          && printf ' \\\n    --epochs %q' "$EPOCHS"
    [[ -n "$USE_OBSM" ]]        && printf ' \\\n    --use_obsm %q' "$USE_OBSM"
    [[ -n "$MAX_TRAIN_CELLS" ]] && printf ' \\\n    --max_train_cells %q' "$MAX_TRAIN_CELLS"
    [[ -n "$MAX_REF_CELLS" ]]   && printf ' \\\n    --max_ref_cells %q' "$MAX_REF_CELLS"
    [[ -n "$EXCLUDE_TEST_FILES" ]] && printf ' \\\n    --exclude_test_files %q' "$EXCLUDE_TEST_FILES"
    [[ -n "$SMOKE" ]]           && printf ' \\\n    --smoke_test'
    echo
  } > "$JOB"
  chmod +x "$JOB"

  BSUB=(bsub
    -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$CORES"
    -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB]"
    -R "span[ptile=$CORES]"
    -gpu "$GPU_SPEC" -W "$WALL"
    -J "cc_${DATASET}_s${SEED}"
    -o "$LOGDIR/cellcontrast_${DATASET}_seed${SEED}_${TS}.%J.out"
    -e "$LOGDIR/cellcontrast_${DATASET}_seed${SEED}_${TS}.%J.err"
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
echo "artifacts : $ARTIFACTS_ROOT/$DATASET/cellcontrast_inference/<TS>/test_results/"
echo
echo "when the jobs finish, score with the SAME harness as the other methods:"
echo "  python $HERE/../plots/compute_extended_metrics.py \\"
echo "      --dataset $DATASET --methods cellcontrast"
