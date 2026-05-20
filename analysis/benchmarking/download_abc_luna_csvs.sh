#!/usr/bin/env bash
# Download LUNA's ABCA-spatial preprocessed CSVs from Google Drive.
# This is the LUNA Figure 4 train-side dataset (ABC Zhuang ABCA1, Animal 1).
#
# Source URL (LUNA-team Google Drive, single-file link):
#   https://drive.google.com/file/d/193bRkYCsqy2b3d2pqUENYbCr4IEJxzJC/view?usp=drive_link
#
# Expected output (under $DEST_DIR after extraction):
#   MERFISH_ABCA_animal1_train.csv
#   MERFISH_ABCA_animal1_test.csv
#
# Defaults (override via env vars):
#   DEST_DIR = /nfs/team361/sb75/DATASETS/bronze/abc_luna
#   FILE_ID  = 193bRkYCsqy2b3d2pqUENYbCr4IEJxzJC
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/download_abc_luna_csvs.sh
#   DEST_DIR=/some/other/path bash .../download_abc_luna_csvs.sh

set -euo pipefail

DEST_DIR="${DEST_DIR:-/nfs/team361/sb75/DATASETS/bronze/abc_luna}"
FILE_ID="${FILE_ID:-193bRkYCsqy2b3d2pqUENYbCr4IEJxzJC}"

log() { printf '[download_abc_luna_csvs] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Ensure gdown is available.
# ---------------------------------------------------------------------------
if ! command -v gdown >/dev/null 2>&1; then
    log "gdown is not on PATH. Installing into the active Python env..."
    pip install --quiet gdown || pip install --quiet --user gdown
fi
log "gdown: $(gdown --version 2>&1 | head -1)"

mkdir -p "$DEST_DIR"
log "destination: $DEST_DIR"

# ---------------------------------------------------------------------------
# 2. Download into a temp staging area. gdown autodetects the filename and
#    server-suggested extension, so we let it write a name we don't know
#    in advance and then move/extract.
# ---------------------------------------------------------------------------
STAGE_DIR="$(mktemp -d -t abc_luna_dl.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

log ""
log "downloading file id ${FILE_ID}..."
log "  https://drive.google.com/file/d/${FILE_ID}/view"
# gdown 6.x dropped the `--id` flag in favour of a positional argument.
# The `uc?id=…` URL form works in BOTH old (5.x) and new (6.x) versions,
# so we use it for portability.
( cd "$STAGE_DIR" && gdown "https://drive.google.com/uc?id=${FILE_ID}" )

# What actually landed?
DOWNLOADED=( "$STAGE_DIR"/* )
if [[ ${#DOWNLOADED[@]} -eq 0 ]]; then
    log "ERROR: download produced no files in $STAGE_DIR"
    exit 1
fi
log "  raw download(s):"
for f in "${DOWNLOADED[@]}"; do
    log "    $(basename "$f")  ($(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") bytes)"
done

# ---------------------------------------------------------------------------
# 3. Identify what we got. The download may be:
#      - a zip / tar.gz / tar archive (most likely for a 2-CSV bundle)
#      - a single .csv (if the URL points at one CSV)
#      - something else (fail loudly)
# ---------------------------------------------------------------------------
extract_one() {
    local src="$1" dst="$2"
    case "$src" in
        *.zip)
            log "  extracting zip → $dst"
            unzip -o -q -d "$dst" "$src"
            ;;
        *.tar.gz|*.tgz)
            log "  extracting tar.gz → $dst"
            tar -xzf "$src" -C "$dst"
            ;;
        *.tar)
            log "  extracting tar → $dst"
            tar -xf "$src" -C "$dst"
            ;;
        *.csv)
            log "  moving CSV → $dst"
            mv -f "$src" "$dst/"
            ;;
        *)
            # Sometimes gdown drops a generic name; sniff by `file`.
            local kind
            kind="$(file -b "$src" 2>/dev/null || echo unknown)"
            case "$kind" in
                *Zip*|*"Zip archive"*)
                    log "  detected zip (no extension) → $dst"
                    unzip -o -q -d "$dst" "$src"
                    ;;
                *"gzip compressed"*)
                    log "  detected tar.gz (no extension) → $dst"
                    tar -xzf "$src" -C "$dst"
                    ;;
                *"POSIX tar"*)
                    log "  detected tar (no extension) → $dst"
                    tar -xf "$src" -C "$dst"
                    ;;
                *"CSV"*|*"ASCII text"*|*"UTF-8 Unicode text"*)
                    log "  treating as raw CSV/text → $dst"
                    mv -f "$src" "$dst/"
                    ;;
                *)
                    log "ERROR: cannot identify file type for $src"
                    log "  file says: $kind"
                    log "  leaving it at $src for manual inspection."
                    return 1
                    ;;
            esac
            ;;
    esac
}

for f in "${DOWNLOADED[@]}"; do
    extract_one "$f" "$DEST_DIR"
done

# ---------------------------------------------------------------------------
# 4. Flatten any nested directory that the archive may have created so the
#    CSVs land directly under $DEST_DIR (matches the expected layout the
#    build_h5ad_from_luna_csv.py command consumes).
# ---------------------------------------------------------------------------
mapfile -t NESTED_CSVS < <(
    find "$DEST_DIR" -mindepth 2 -maxdepth 4 -type f -name "*.csv" 2>/dev/null
)
for nc in "${NESTED_CSVS[@]:-}"; do
    [[ -z "$nc" ]] && continue
    target="$DEST_DIR/$(basename "$nc")"
    if [[ -e "$target" ]]; then
        log "  skip (already at top level): $(basename "$nc")"
    else
        log "  flattening: $nc → $target"
        mv -f "$nc" "$target"
    fi
done
# Tidy up empty subdirs left behind by the archive.
find "$DEST_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
log ""
log "files under $DEST_DIR:"
find "$DEST_DIR" -maxdepth 2 -type f -printf "  %p  (%s bytes)\n" 2>/dev/null || \
    find "$DEST_DIR" -maxdepth 2 -type f -print

# Sanity check: warn if the two expected CSVs aren't both present.
WANT_TRAIN="$DEST_DIR/MERFISH_ABCA_animal1_train.csv"
WANT_TEST="$DEST_DIR/MERFISH_ABCA_animal1_test.csv"
log ""
if [[ -f "$WANT_TRAIN" && -f "$WANT_TEST" ]]; then
    log "DONE. Both expected CSVs are in place."
else
    log "WARNING: expected files not both found:"
    [[ -f "$WANT_TRAIN" ]] || log "    missing: $WANT_TRAIN"
    [[ -f "$WANT_TEST"  ]] || log "    missing: $WANT_TEST"
    log "Check the file listing above — names may differ slightly."
fi

log ""
log "Next step: build per-section h5ads"
log "    python /nfs/team361/sb75/scgg/scripts/build_h5ad_from_luna_csv.py \\"
log "        --train_csv $WANT_TRAIN \\"
log "        --test_csv  $WANT_TEST \\"
log "        --out_dir   /nfs/team361/sb75/DATASETS/silver/abc_luna \\"
log "        --prefix    abc_zhuang_abca1 --overwrite"
