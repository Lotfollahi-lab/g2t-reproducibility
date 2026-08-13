#!/usr/bin/env bash
set -euo pipefail
source /nfs/team361/sb75/.venvs/scgg/bin/activate
cd /nfs/team361/sb75/scgg
export SCGG_LOG_MDS_VAR=/nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/mds_variance/mds_var_out/mds_var_cortex_seed0.csv
echo "python: $(which python) $(python --version)"
echo "SCGG_LOG_MDS_VAR=$SCGG_LOG_MDS_VAR"
exec python scripts/run_scgg_inference.py --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna --checkpoint /nfs/team361/sb75/scgg-reproducibility/artifacts/mmc_luna/scgg_model/20260602_142505/best_model.ckpt --seed 0
