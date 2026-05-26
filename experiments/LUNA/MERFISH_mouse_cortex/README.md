# LUNA Figure 3 reproduction with ScGG

Reproduces the LUNA mouse primary motor cortex benchmark (Yu, Brbić et al. 2025,
bioRxiv 10.1101/2025.02.13.638045, **Figure 3**) using the ScGG model.

## Dataset

- **Source**: Zhang et al., *Nature* 598:137–143 (2021), MERFISH on the mouse
  primary motor cortex. 254 genes.
- **Split (exactly as in the LUNA paper):**
  - **Train** — Mouse 1, **33 slices**, **158,379 cells**.
  - **Test**  — Mouse 2, **31 slices**, **118,036 cells**.
- **Local path on the cluster (sb75 area):**
  - `/nfs/team361/sb75/DATASETS/silver/mmc_luna/` — 64 per-slice h5ad files
    named `mmc_mouse{1,2}_slice{N}.h5ad` (the legacy
    `merfish_mouse_cortex_mouse{1,2}_slice{N}.h5ad` naming is also still
    recognised by the loader).
- The loader (`scgg.data.luna_cortex.load_luna_cortex`) auto-discovers
  Mouse 1 → TRAIN and Mouse 2 → TEST from the filename pattern, so the split
  does not need to be defined explicitly anywhere.

## Metrics

Implemented in `scgg.evaluation.luna_metrics` and bit-equivalent to LUNA's
`metrics/evaluation_statistics.py`:

| Metric | Definition | LUNA in Fig.3 |
|---|---|---|
| **Spearman (per-slice median)** | For each cell *i*, Spearman correlation between row *i* of the predicted N×N pairwise-Euclidean distance matrix and row *i* of the true distance matrix. Per-slice value is the median over cells. **Headline = mean over 31 test slices.** | **44.8 %** |
| **Contact precision / F1** | Flatten both pairwise-distance matrices, threshold each at the same percentile of its own distribution (default 1 %), compute F1 on binary "contact" labels. Precision == F1 by construction. | Sup. Fig. 10 |
| **RSSD** | Pad coords with Z=0 to 3-D, Kabsch-align prediction to ground truth via `scipy.spatial.transform.Rotation.align_vectors`, return Root Sum Squared Deviation. Reported as `absolute_rssd` (over all cells) and `mean_rssd` (mean over per-cell-class Kabsch alignments). | Sup. Fig. 10 |

Spearman and contact are dimension-agnostic — they work directly on ScGG's
d-dim metric embedding (no projection to 2-D required). RSSD needs 2-D, so
the benchmark script projects the metric embedding to 2-D with PCA before
the Kabsch step.

## How to run

From a shell on a GPU node:

```bash
cd /path/to/scgg_claude_project/scgg
pip install -e .

python scripts/run_scgg.py \
    --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \
    --epochs 200 \
    --batch_size 8192 \
    --wandb_run_name luna_cortex_v1
```

`--output_dir` defaults to
`/nfs/team361/sb75/scgg-reproducibility/artifacts/<data_dir_name>/model/<YYYYMMDD_HHMMSS>/`
(i.e. `.../mmc_luna/model/20260518_213045/`). Each training run gets its
own timestamped directory; inference outputs land under the matching
`.../inference/<YYYYMMDD_HHMMSS>/` so plots stay pinned to the model
that produced them. Pass `--output_dir` explicitly to override.

From a notebook (see `notebooks/scgg_luna_cortex_benchmark.ipynb`):

```python
from scripts.run_scgg import run_benchmark

agg = run_benchmark(
    data_dir="/nfs/team361/sb75/DATASETS/silver/mmc_luna",
    epochs=200,
    batch_size=8192,
)
```

Outputs land in `output_dir` (default
`/nfs/team361/sb75/scgg-reproducibility/artifacts/mmc_luna/model/<YYYYMMDD_HHMMSS>/`):

- `per_slice_metrics.csv` — one row per test slice (31 rows).
- `aggregate_metrics.json` — the LUNA-style aggregated numbers (mean of
  per-slice-medians is the headline).
- `config.yaml` — frozen config snapshot for the run.
- `benchmark.log` — full run log.
- `checkpoints/` — model checkpoints.

## How to compare against LUNA

LUNA reports (Figure 3, Supplementary Fig. 10):

| Metric | LUNA | CeLEry | novoSpaRc | CytoSPACE | Tangram |
|---|---|---|---|---|---|
| Spearman (median per slice, mean across slices) | **0.448** | 0.224 | 0.139 | 0.117 | 0.099 |
| Precision/F1 (contact, 1 %) | higher = better | | | | |
| RSSD | lower = better | | | | |

The benchmark script's logged "Headline" line directly compares against
LUNA's 44.8 %.

## Caveats

- **Spearman aggregation choice.** LUNA's `compute_spearman_correlation`
  returns both `spr_avg` (mean over cells) and `spr_median` (median over
  cells). The Figure 3 caption is specific: "**median** Spearman's rank
  correlation". So per-slice we take the median, then average across 31
  slices for the headline. The `aggregate_slices` helper reports both
  aggregations so reviewers can cross-check.

- **Precision == F1.** This is a known property of LUNA's `compute_contact`:
  both labels and predictions are thresholded at the same percentile of their
  own distribution, so predicted-positive count == true-positive count and
  precision = recall = F1. LUNA's published CSVs confirm this to 15 decimals.

- **RSSD on 32-d metric embeddings.** ScGG (contrastive mode) outputs a
  d=32 metric embedding, but RSSD is a 2-D-coord metric. The benchmark
  projects the embedding via PCA to 2-D before Kabsch. This is a fair
  comparison provided ScGG's spatial information is largely captured by
  the top-2 PCs. If not, the flow_matching mode (which outputs 2-D directly)
  is a more apples-to-apples comparator on RSSD.

- **OOD note.** LUNA's Figure 3 is *within-modality* (ST → ST cross-animal),
  not a true OOD test to scRNA-seq. The reported numbers here apply to that
  same cross-animal setting. For ST → scRNA OOD evaluation, enable the
  cross-modality components in `config.training.ood.*`.
