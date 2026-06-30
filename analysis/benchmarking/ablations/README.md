# G2T ablations + SVG visualizations (for the MLCB full paper)

These scripts run on **your cluster** (where the silver h5ads, checkpoints, and
GPUs live). They plug into your existing pipeline (`submit_pipeline.sh`,
`compute_extended_metrics.py`). All ablations use the **MMC cortex** split
(1000 epochs, the cheap benchmark) and **5 seeds (0–4)**.

## Why these ablations (what each isolates)

Each ablation changes **one** factor relative to the G2T baseline
(`scgg_mmc_edm_fm_heads32_fastmds`: EDM head on, K=8, flow matching, 32 heads).
Reviewers will want to see *which* design choice earns the gains.

| Ablation | Override (on top of baseline) | Isolates | Tier |
|---|---|---|---|
| `baseline` | — | reference G2T | core |
| `edm_off` | `model.edm.enabled=false` | **the EDM head itself** (→ predict 2-D coords directly, LUNA-style). *The headline ablation.* | core |
| `diffusion` | `model.framework=diffusion` | flow matching vs LUNA-style DDPM (EDM kept) | core |
| `heads16` | `model.hidden_dims.num_heads=16` | the `heads32` choice | core |
| `K2` | `model.edm.embed_dim=2` | the "more room than 2-D" claim (minimal case) | core |
| `K16` | `model.edm.embed_dim=16` | more embedding room than K=8 | core |
| `K4` | `model.edm.embed_dim=4` | finer K sweep | extended |
| `K32` | `model.edm.embed_dim=32` | finer K sweep | extended |
| `mds_eigh` | `model.edm.mds_solver=eigh` | exact MDS vs LOBPCG (accuracy parity / runtime) | extended |

The **K sweep** (`K2, K4, 8(baseline), K16, K32`) directly substantiates the
paper's central claim that lifting the prediction out of the 2-D bottleneck
helps; `edm_off` is the strongest single result a reviewer will look for.

> Tip: if you already have the 5 baseline seeds from the paper run
> (`20260530_165200…165229`), you can drop `baseline` from the grid and just
> reuse those timestamps in the manifest to save 5 runs.

## Workflow

1. **Submit the runs** (edit the CONFIG paths at the top of `run_ablations.sh` first):
   ```bash
   bash run_ablations.sh                 # CORE ablations (6 × 5 = 30 runs)
   ABLATION_SET=all bash run_ablations.sh # CORE + EXTENDED (9 × 5 = 45 runs)
   DRY_RUN=1 bash run_ablations.sh        # preview submit commands only
   ```
   This submits one LSF job per (ablation, seed) and records each run's
   timestamp into `ablation_manifest.csv`.

2. **Score each run** once the jobs finish (computes Spearman / Contact F1 /
   Sum RSSD per run):
   ```bash
   python /nfs/team361/sb75/scgg-reproducibility/analysis/benchmarking/plots/compute_extended_metrics.py \
       --dataset mmc_luna --methods g2t \
       --scgg_timestamps $(tail -n +2 ablation_manifest.csv | cut -d, -f4 | paste -sd, -)
   ```

3. **Aggregate into one comparison table** (mean ± SEM across seeds, with Δ vs baseline):
   ```bash
   python aggregate_ablations.py
   ```
   Writes `ablation_comparison.csv` (per-ablation) and `ablation_per_seed.csv`.

## SVG visualizations (ground-truth vs generated coordinates)

For a slice, rank genes by Moran's I on the true coordinates, then plot each
top gene's expression on the **true** vs **generated** layout, side by side:
```bash
python plot_svg_comparison.py \
    --test_csv  /nfs/.../mmc_luna/scgg_inference/<TS>/work/test.csv \
    --pred_dir  /nfs/.../mmc_luna/scgg_inference/<TS>/luna_run/test_results \
    --section   mouse2_slice229 \
    --top_k 4 --out svg_mouse2_slice229.png
```
(Predicted coords are Procrustes-aligned to the true frame for display; pass
`--no_align` to disable.) This is the figure that turns "+X% Spearman" into a
visible biological result, which MLCB reviewers value.

## Notes / caveats
- Paths in `run_ablations.sh` and the examples are the ones from your run
  configs — double-check them on your cluster before launching.
- The `diffusion` ablation is heavier per step (DDPM many-step sampling); the
  rest are MMC-cheap.
- `mds_eigh` should match `lobpcg` on accuracy within numerical tolerance — its
  value is a runtime/exactness sanity check, not an accuracy ablation.
