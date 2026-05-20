#!/usr/bin/env bash
# Download LUNA's published preprocessed CSVs (the data their paper
# experiments actually ran on) from Google Drive.
#
# URLs are listed in LUNA's README:
#   https://github.com/mlbio-epfl/LUNA
#
# These are the closest thing LUNA releases to a "reference checkpoint" —
# training on these exact CSVs is the best way to bit-exact reproduce
# the paper's reported numbers without guessing at their preprocessing.
#
# Defaults (override via env vars):
#   DEST_DIR  = /nfs/team361/sb75/DATASETS/luna_paper_csvs
#
# Usage:
#   bash scgg-reproducibility/analysis/benchmarking/download_luna_paper_csvs.sh
#   DEST_DIR=/some/other/path bash scgg-reproducibility/analysis/benchmarking/download_luna_paper_csvs.sh

set -euo pipefail

DEST_DIR="${DEST_DIR:-/nfs/team361/sb75/DATASETS/luna_paper_csvs}"

# Google Drive IDs (folder + single sample file). If the LUNA team rotates
# them, update the URLs here.
PREPROCESSED_FOLDER_ID="1vWxVUSuQzRDF1o9Vw_cnm-wbEYw_e1Gu"
SAMPLE_CORTEX_FILE_ID="1j6W5NZV56_W3kO_UxXEtFv1nIlwUksLi"

log() { printf '[download_luna_paper_csvs] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Make sure gdown is on PATH (the only reliable Google-Drive downloader).
# ---------------------------------------------------------------------------
if ! command -v gdown >/dev/null 2>&1; then
    log "gdown is not on PATH. Installing into the active Python env..."
    # Use --user as a fallback if the active env is unwritable.
    pip install --quiet gdown || pip install --quiet --user gdown
fi
log "gdown: $(gdown --version 2>&1 | head -1)"

mkdir -p "$DEST_DIR"
log "destination: $DEST_DIR"

# ---------------------------------------------------------------------------
# 2. Try the FOLDER first (all paper-experiment CSVs).
# ---------------------------------------------------------------------------
log ""
log "downloading the full 'preprocessed datasets' folder..."
log "  https://drive.google.com/drive/folders/${PREPROCESSED_FOLDER_ID}"
if ! (cd "$DEST_DIR" && gdown --folder "https://drive.google.com/drive/folders/${PREPROCESSED_FOLDER_ID}"); then
    log "WARNING: folder download failed (quota / auth / size limit)."
    log "Falling back to just the MERFISH cortex sample dataset."
fi

# ---------------------------------------------------------------------------
# 3. Always grab the smaller sample file too (it's tiny and fast).
# ---------------------------------------------------------------------------
log ""
log "downloading the MERFISH cortex sample file..."
log "  https://drive.google.com/file/d/${SAMPLE_CORTEX_FILE_ID}/view"
SAMPLE_DEST="$DEST_DIR/MERFISH_mouse_cortex_sample.zip"
if [[ -f "$SAMPLE_DEST" ]]; then
    log "  sample already present at $SAMPLE_DEST; skipping"
else
    # gdown 6.x dropped `--id`; the uc?id=… URL form works in both 5.x and 6.x.
    gdown "https://drive.google.com/uc?id=${SAMPLE_CORTEX_FILE_ID}" -O "$SAMPLE_DEST" || \
        log "WARNING: sample-file download failed."
fi

# ---------------------------------------------------------------------------
# 4. Summarise what landed
# ---------------------------------------------------------------------------
log ""
log "downloaded files under $DEST_DIR:"
find "$DEST_DIR" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.zip" -o -name "*.tar.gz" \) -printf "  %p  (%s bytes)\n" 2>/dev/null || \
    find "$DEST_DIR" -maxdepth 3 -type f -print

log ""
log "DONE."
log ""
log "To train LUNA on these:"
log "    source /nfs/team361/sb75/.venvs/luna/bin/activate"
log "    python /nfs/team361/sb75/scgg/scripts/run_luna_on_mmc.py \\"
log "        --train_csv $DEST_DIR/MERFISH_mouse_cortex_train.csv \\"
log "        --test_csv  $DEST_DIR/MERFISH_mouse_cortex_test.csv \\"
log "        --run_name luna_paper_csvs_v1"
log ""
log "(Adjust the filenames if LUNA's folder uses different ones — the"
log " download will show what's available.)"
