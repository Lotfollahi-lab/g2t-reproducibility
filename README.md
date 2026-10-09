# G2T reproducibility

Code to reproduce the experiments of

> **G2T: Tissue Reconstruction from Gene Expression via Embedding-Distance Flow Matching**
> Sebastian Birk, Fabian J. Theis, Mohammad Lotfollahi.
> Proceedings of the Machine Learning in Computational Biology (MLCB) 2026, PMLR.

The model itself lives in [Lotfollahi-lab/g2t](https://github.com/Lotfollahi-lab/g2t)
(Python package `scgg`). This repository holds everything around it: environment
set-up, data download, the LSF launchers, wrappers for the baselines, scoring,
statistics and the benchmark figure panels.

## Code version

- The tag **`mlcb2026-camera-ready`** marks the camera-ready state of both
  repositories and contains every script referenced below.
- The reported G2T runs (Figs. 2–3) used the model code (`src/`) of g2t commit
  **`9dbe98f`** (30 May 2026); the later commits used for those runs (up to early June)
  changed only `scripts/`, not `src/`. Ablations, controls and diagnostics were run
  from later commits on the same history. Re-training the reported cortex configuration in
  July (the Table 1 baseline runs, seeds 0–4) reproduced the five headline cortex runs seed
  by seed to within 1e-4 in Spearman.
- **MDS read-out.** The current g2t default is `model.edm.mds_solver=svd`: the top-2
  principal components of the centred embeddings, which is algebraically equivalent to
  classical MDS of the predicted distances (test: `tests/test_mds_svd_readout.py` in g2t).
  All reported G2T runs used `mds_solver=lobpcg` (fp32), which adds a small Tikhonov
  regularisation, so the two read-outs agree up to solver and regularisation tolerances. To
  reproduce the reported runs, keep `model.edm.mds_solver=lobpcg` in the overrides below.
- **Trained checkpoints are not distributed.** Each training run writes its own checkpoints to
  `<run>/luna_run/checkpoints/`.

## Before you start: adapt the cluster paths

The scripts were written for the authors' LSF cluster and default to paths under
`/nfs/team361/...`. Set the following before running anything:

```bash
export SCGG_REPO=/path/to/g2t                  # clone of Lotfollahi-lab/g2t (read by the launchers)
export SCGG_ARTIFACTS_ROOT=/path/to/artifacts  # all run outputs (read by the launchers and g2t's pipelines)
export LUNA_ARTIFACTS_ROOT=$SCGG_ARTIFACTS_ROOT # g2t's run_luna_pipeline.py reads only this variable
export LSF_GROUP=<your LSF group>              # or pass --group; queues via LSF_QUEUE or --queue
# used only in this README's commands:
REPRO=/path/to/g2t-reproducibility   # this repository
DATA=/path/to/datasets               # downloaded CSVs (bronze/) and per-section h5ads (silver/)
VENVS=/path/to/venvs                 # one virtual environment per method
```

Hard-coded paths that have no environment variable and must be edited:

| File | Line(s) | Set to |
|---|---|---|
| `analysis/benchmarking/plots/compute_extended_metrics.py` | `ARTIFACTS_ROOT` (l. 100) | `$SCGG_ARTIFACTS_ROOT` |
| `analysis/benchmarking/plots/plot_method_comparison.py` | `ARTIFACTS_ROOT` (l. 79); `benchmark_paired_stats.py` reads it from here | `$SCGG_ARTIFACTS_ROOT` |
| `analysis/benchmarking/ablations/run_ablations.sh` | `SUBMIT`, `DATA_DIR` (ll. 26–27) | `lsf/submit_pipeline.sh` in your clone; `$DATA/silver/mmc_luna` |
| g2t `scripts/run_luna_train.py` | `_ENGINE_REPO_DEFAULT` (ll. 521–523) | your LUNA checkout (see below) |

Other launchers take the paths as options (`--data_dir`, `--venv`, `--repo`,
`--artifacts_root`, `--scgg_src`, `--runs_root`); the venv used by `submit_pipeline.sh` is
set per call with `VENV_PATH=...`.

**Without LSF.** `submit_pipeline.sh ... --dry_run` still writes the job script (printed
as `job script`) under `<artifacts>/<dataset>/<method>_model/<timestamp>/lsf_logs/job.sh`;
`bash` that file to run the same pipeline directly.

The paper's models were trained on a single NVIDIA H200 GPU (Supplement Sec. 1); CeLEry
runs on CPU.

## 1. Environments

Every environment is a [uv](https://github.com/astral-sh/uv) venv; install uv first.

```bash
cd $REPRO/analysis/benchmarking
VENV_DIR=$VENVS/scgg   SCGG_CODE_DIR=$SCGG_REPO bash setup_scgg_env.sh   # G2T: Python 3.10, torch 2.4.0+cu121, editable install of g2t
VENV_DIR=$VENVS/luna   LUNA_CODE_DIR=$PWD/luna LUNA_GIT_REF=v1 bash setup_luna_env.sh   # LUNA: clones github.com/mlbio-epfl/LUNA into luna/
VENV_DIR=$VENVS/celery bash setup_celery_env.sh                          # CeLEry (CeLEryPy 1.1.3, CPU torch)
VENV_DIR=$VENVS/cellcontrast bash setup_cellcontrast_env.sh              # CellContrast, pinned upstream commit 0559aa9; runs the wrapper tests
VENV_DIR=$VENVS/come   bash setup_come_env.sh                            # COME, pinned upstream commit 2005812; runs a toy fit and the wrapper tests
```

LUNA is run from its upstream repository. `LUNA_GIT_REF=v1` pins upstream tag `v1`, commit
`64f10eb`: the commit recorded at `analysis/benchmarking/luna` in this repository, and
upstream `main` throughout this work (the script's default ref is `main`). g2t's wrapper
adds train-only/test-only modes at runtime without editing LUNA's files. The authors'
cluster copy of LUNA additionally carried small local patches (e.g. a configurable
DataLoader worker count for CNS-scale memory, experiment-config extensions), which are not
distributed. Point `_ENGINE_REPO_DEFAULT` in g2t `scripts/run_luna_train.py` at
`LUNA_CODE_DIR` (the launchers do not forward its `--luna_repo` option).

`setup_novosparc_env.sh` and `--method novosparc` are not used in the paper.

## 2. Data

All benchmark inputs are the preprocessed data released with LUNA (Google Drive, fetched
with `gdown`). Run the download scripts on Linux (they need bash ≥ 4).

```bash
cd $REPRO/analysis/benchmarking
uv pip install --python $VENVS/scgg/bin/python gdown   # uv venvs have no pip, so install gdown up front
source $VENVS/scgg/bin/activate
DEST_DIR=$DATA/bronze/mmc_luna bash download_mmc_luna_csvs.sh
DEST_DIR=$DATA/bronze/cns_luna bash download_cns_luna_csvs.sh

# one h5ad per section: <section>_train.h5ad / <section>_test.h5ad
python $SCGG_REPO/scripts/build_h5ad_from_luna_csv.py \
    --train_csv $DATA/bronze/mmc_luna/MERFISH_mouse_cortex_train.csv \
    --test_csv  $DATA/bronze/mmc_luna/MERFISH_mouse_cortex_test.csv \
    --out_dir   $DATA/silver/mmc_luna --overwrite
python $SCGG_REPO/scripts/build_h5ad_from_luna_csv.py \
    --train_csv $DATA/bronze/cns_luna/ABCA_harmonized_train.csv \
    --test_csv  $DATA/bronze/cns_luna/scRNA_harmonized_test.csv \
    --out_dir   $DATA/silver/cns_luna --overwrite
```

Keep the silver directory names `mmc_luna` and `cns_luna`: the dataset name, and with it
the output folder `$SCGG_ARTIFACTS_ROOT/<dataset>/` that the scoring scripts read, is the
basename of `--data_dir`.

| Benchmark | Script (default `DEST_DIR`) | Files | Content |
|---|---|---|---|
| MERFISH mouse primary motor cortex (Fig. 2) | `download_mmc_luna_csvs.sh` (`.../bronze/mmc_luna`; zip of ~81 MB) | `MERFISH_mouse_cortex_train.csv`, `MERFISH_mouse_cortex_test.csv` | train: Donor 1, 33 sections, 158,379 cells; test: Donor 2, 31 sections, 118,036 cells; 254 genes. md5 of the CSVs used: train `4103fb5de5ee9f80dfdb00af2c32539d`, test `4b053f883757dc8c07ddd32c34e780f3` |
| Mouse CNS scRNA-seq atlas (Fig. 3) | `download_cns_luna_csvs.sh` (`.../bronze/cns_luna`) | `ABCA_harmonized_train.csv`, `scRNA_harmonized_test.csv` | train: MERFISH ABC atlas, Animal 1 (147 sections, 2.85 M cells); test: CNS scRNA-seq atlas (1.08 M cells), scored against the STARmap PLUS-imputed reference, not measured coordinates. Both as LUNA's 600-d Harmony latent over the 804 shared genes |
| ABC atlas, gene panel (optional) | `download_abc_luna_csvs.sh` (`.../bronze/abc_luna`) | `MERFISH_ABCA_animal1_train.csv`, `MERFISH_ABCA_animal1_test.csv` | not used for the reported numbers |

G2T and LUNA receive identical, untransformed inputs (the pipeline-built
`work/{train,test}.csv` are bitwise identical to LUNA's released cortex CSVs).
For CNS, the sagittal and spinal-cord test sections are held out for every method
(`--exclude_test_files` below), leaving the 14 scored sections.

**Xenium human skin (Supplement Sec. 2, Table S1).** These data are not publicly
available: they come from the atopic-eczema cohort of Steele et al. (2026, bioRxiv), also used
in TERRA, and were used with the data owners' permission. Only the scripts are provided (Section 3.5).

## 3. Experiments

| Paper element | Experiment | Section |
|---|---|---|
| Fig. 2b, §3.1 | Cortex: G2T, LUNA, CeLEry, CellContrast, COME (seeds 0–4) | 3.1 |
| Fig. 3b, §3.2 | CNS: G2T, LUNA, CeLEry (seeds 0–4) | 3.1 |
| Table 1, §3.3 | Ablations, seeds 0–9 | 3.2 |
| §3.1 "Controls and robustness" | Expression controls, uniform subsampling, L5 depletion | 3.3 |
| §2.4, Supplement Sec. 1 | MDS spectral-mass diagnostic (99.2%) | 3.4 |
| §3.1, Supplement Sec. 2, Table S1 | Xenium human skin (data not public) | 3.5 |

Each `submit_pipeline.sh` run writes `$SCGG_ARTIFACTS_ROOT/<dataset>/<method>_model/<timestamp>/`
(checkpoint, `work/{train,test}.csv`, `lsf_logs/job.sh`) and
`.../<method>_inference/<timestamp>/` (per-section `metadata_pred.csv` /
`metadata_true.csv`, `plots/<section>.svg`); the contrastive submitters write
`.../{cellcontrast,come}_inference/<timestamp>/`. The submitters print the run timestamp;
keep it, because scoring takes timestamps as input. Add `--wandb_mode disabled` to
`submit_pipeline.sh` if you do not use Weights & Biases (the default is `online`;
`run_ablations.sh` does not pass the flag, so add it to its `cmd` array). Optionally,
`--gpu_model H200` adds an LSF GPU-memory constraint (authors' cluster only).
The Python helpers in Sections 3.2–3.4 and 4 run in the `scgg` venv
(`source $VENVS/scgg/bin/activate`).

### 3.1 Headline benchmarks (Figs. 2b, 3b)

G2T run configuration (identical to `BASELINE_OVERRIDE` in `ablations/run_ablations.sh`):
1,000 epochs on cortex, 3,500 on CNS; on CNS, G2T and LUNA train on one slice per batch
(`train.batch_size=1`, as in the recorded launch scripts). The CeLEry CNS wall time below is a generous
suggestion; adjust it to your cluster.

```bash
cd $REPRO/analysis/benchmarking/lsf
OVR="model.hidden_dims.num_heads=32 model.edm.mds_align_train=false model.edm.mds_dtype=fp32 model.edm.mds_solver=lobpcg dataset.num_workers=0"
CNS_EXCL=sagittal1_test.h5ad,sagittal2_test.h5ad,sagittal3_test.h5ad,spinalcord_test.h5ad

for s in 0 1 2 3 4; do
  # G2T
  VENV_PATH=$VENVS/scgg bash submit_pipeline.sh --method scgg --wandb_run_name g2t_mmc_seed$s \
      --data_dir $DATA/silver/mmc_luna --epochs 1000 --seed $s --override "$OVR"
  VENV_PATH=$VENVS/scgg bash submit_pipeline.sh --method scgg --wandb_run_name g2t_cns_seed$s \
      --data_dir $DATA/silver/cns_luna --epochs 3500 --seed $s --override "train.batch_size=1 $OVR" \
      --exclude_test_files $CNS_EXCL
  # LUNA (released settings)
  VENV_PATH=$VENVS/luna bash submit_pipeline.sh --method luna --wandb_run_name luna_mmc_seed$s \
      --data_dir $DATA/silver/mmc_luna --epochs 1000 --seed $s
  VENV_PATH=$VENVS/luna bash submit_pipeline.sh --method luna --wandb_run_name luna_cns_seed$s \
      --data_dir $DATA/silver/cns_luna --epochs 3500 --seed $s --override "train.batch_size=1 dataset.num_workers=0" \
      --exclude_test_files $CNS_EXCL
  # CeLEry (one model per test section, CPU)
  VENV_PATH=$VENVS/celery bash submit_pipeline.sh --method celery --wandb_run_name celery_mmc_seed$s \
      --data_dir $DATA/silver/mmc_luna --seed $s --training_mode per_reference
  VENV_PATH=$VENVS/celery bash submit_pipeline.sh --method celery --wandb_run_name celery_cns_seed$s \
      --data_dir $DATA/silver/cns_luna --seed $s --training_mode per_reference \
      --batch_size 256 --cores 32 --wall 168:00 --exclude_test_files $CNS_EXCL
  sleep 2   # run directories are keyed on a 1-second timestamp
done
```

Contrastive baselines (cortex only), run from the authors' released code. CellContrast uses
its imaging-ST parameters and the full training reference; its code fixes the random seed
internally, so its five runs differ only through data shuffling. COME is a scale-limited
adaptation: all test cells against a 20,000-cell reference subsample (`--max_ref_cells`,
default 20000).

```bash
cd $REPRO/analysis/benchmarking/contrastive_baselines
bash submit_cellcontrast.sh --dataset mmc_luna --data_dir $DATA/silver/mmc_luna --venv $VENVS/cellcontrast --smoke_test
bash submit_cellcontrast.sh --dataset mmc_luna --data_dir $DATA/silver/mmc_luna --venv $VENVS/cellcontrast --seeds "0 1 2 3 4"
bash submit_come.sh --dataset mmc_luna --data_dir $DATA/silver/mmc_luna --venv $VENVS/come --smoke_test
bash submit_come.sh --dataset mmc_luna --data_dir $DATA/silver/mmc_luna --venv $VENVS/come --seeds "0 1 2 3 4"
```

### 3.2 Ablations (Table 1)

Each variant changes one factor relative to the G2T configuration above and is trained
with seeds 0–9 on the cortex benchmark.

| Table 1 row | `run_ablations.sh` name | Override added |
|---|---|---|
| G2T (full) | `baseline` | — |
| −EDM head (predict 2-D) | `edm_off` | `model.edm.enabled=false` |
| Coordinate L2 | `coord_raw` | `model.edm.enabled=false model.loss.pairwise_distance_mse.enabled=false model.loss.coord_mse_raw.enabled=true` |
| Diffusion | `diffusion` | `model.framework=diffusion` |
| 16 heads | `heads16` | `model.hidden_dims.num_heads=16` |
| K=2 / K=16 | `K2` / `K16` | `model.edm.embed_dim=2` / `=16` |

```bash
cd $REPRO/analysis/benchmarking/ablations
# run_ablations.sh appends every submission to ablation_manifest.csv in this folder,
# which ships with the authors' run IDs: move it aside first.
mv ablation_manifest.csv ablation_manifest.paper.csv
export VENV_PATH=$VENVS/scgg
SEEDS="0 1 2 3 4 5 6 7 8 9" bash run_ablations.sh                  # baseline, edm_off, diffusion, heads16, K2, K16
ONLY=coord_raw SEEDS="0 1 2 3 4 5 6 7 8 9" bash run_ablations.sh   # Coordinate L2

# when all jobs have finished: score, aggregate, then Table 1 statistics
python run_ablation_analysis.py --artifacts_root $SCGG_ARTIFACTS_ROOT --workers 4 --outdir results
python ablation_paired_stats.py --per_seed results/ablation_per_seed.csv --out results/ablation_stats.csv
```

`ablation_paired_stats.py` reports mean ± SD over seeds and a paired t-test against
`baseline` on the shared seeds. `DRY_RUN=1` previews the submissions but still appends rows
to the manifest. `ABLATION_SET=all` submits the extended set as well: `coord_raw` plus
variants that are not in Table 1 (`K4`, `K32`, `mds_eigh`, `coord_raw_canon`).

### 3.3 Expression controls and robustness (§3.1)

Inference-only experiments on one trained cortex G2T model; nothing is retrained.

```bash
cd $REPRO/analysis/benchmarking/robustness
RUN=$SCGG_ARTIFACTS_ROOT/mmc_luna/scgg_model/<timestamp>   # a cortex G2T run from 3.1
CKPT=$(readlink -f $RUN/best_model.ckpt)                    # must resolve to .../epoch=<N>.ckpt
ROB=$SCGG_ARTIFACTS_ROOT/robustness
export VENV_PATH=$VENVS/scgg RUNS_ROOT=$ROB/runs

# expression controls: real, class_mean, shuffle_type (within cell type), shuffle_all
python make_shuffle_controls.py --test_csv $RUN/work/test.csv --out_dir $ROB/arm_shuffle --seed 0
# uniform subsampling: every presented cell is scored
python make_subsample_conditions.py --test_csv $RUN/work/test.csv --out_dir $ROB/arm_depth \
    --slice_fracs 1.0,0.9,0.8,0.7,0.6,0.5,0.25,0.10,0.05 --seed 0
# depletion of the L5 glutamatergic block
python make_subsample_conditions.py --test_csv $RUN/work/test.csv --out_dir $ROB/arm_deplete \
    --eval_frac 0 --deplete_class "L5 IT,L5 ET" \
    --deplete_levels 1.0,0.9,0.8,0.7,0.6,0.5,0.25,0.1,0.05,0.0 --seed 0

for arm in arm_shuffle arm_depth arm_deplete; do
  bash submit_subsample_robustness.sh --cond_root $ROB/$arm --checkpoint $CKPT \
      --train_csv $RUN/work/train.csv --seeds "0"
done
```

Each submission prints the matching `score_subsample_robustness.py` command (pass
`--scgg_src $SCGG_REPO/src`). For `arm_depth` and `arm_deplete` also pass
`--control_from <run root of the condition that presents all cells>`
(`$ROB/runs/arm_depth/slice100__seed0`, `$ROB/runs/arm_deplete/deplete_100__seed0`): it rescores that run's own predictions on
each condition's cell set, separating the model's response from the change in which cells
are scored. The scorer reports Spearman only, in its own frame, so its values are compared
between conditions, not with the 31-section benchmark.

### 3.4 MDS spectral-mass diagnostic (§2.4, Supplement Sec. 1)

Logs, at every reverse step, the share of the double-centred Gram matrix's positive
spectral mass carried by its top two eigenvalues (own fp64 `eigh`, independent of
`mds_solver`).

```bash
cd $REPRO/analysis/benchmarking/mds_variance
bash submit_mds_var.sh --checkpoint $SCGG_ARTIFACTS_ROOT/mmc_luna/scgg_model/<timestamp>/best_model.ckpt \
    --data_dir $DATA/silver/mmc_luna --venv $VENVS/scgg --repo $SCGG_REPO \
    --out $SCGG_ARTIFACTS_ROOT/mds_var/mds_var.csv
python analyse_mds_var.py $SCGG_ARTIFACTS_ROOT/mds_var/mds_var.csv
python analyse_mds_var.py mds_var_out/mds_var_cortex_seed0.csv   # the paper's log (cortex seed-4 model 20260530_165229)
```

Always pass `--out`: the default is the committed log, which the script deletes as soon as
it starts, even with `--dry_run`. The diagnostic's predictions are written into that run's
`scgg_inference/<timestamp>/` next to its benchmark predictions, so score the run (Section 4)
first or use a copy of the run directory.

### 3.5 Xenium human skin (Supplement Sec. 2, Table S1; scripts only)

The data are not public, so this cannot be re-run from this repository. The scripts used
are in g2t: `scripts/prepare_xenium_silver.py` (silver directory; each cell scaled to the
dataset's median total count, linear space, no log) and `scripts/make_donor_split.py`
(donor-aware split: four training sections from one donor; held-out sections from the
training donor and from an unseen donor). Both methods were trained with
`submit_pipeline.sh --epochs 1000`, seeds 0–4: G2T with `--override "$OVR dataset.maximum_graph_size.train=16000"`,
LUNA with `--override "dataset.maximum_graph_size.train=16000"` (sections above 16,000
cells are split into sub-graphs). Scoring used
`compute_extended_metrics.py --dataset <split> --methods g2t,luna --scgg_timestamps ... --luna_timestamps ... --rssd_frame luna`,
and G2T and LUNA were compared per held-out section with Welch t-tests across seeds.

## 4. Scoring, statistics and figures

Run these in the `scgg` venv after editing `ARTIFACTS_ROOT` (see above).
`compute_extended_metrics.py` computes Spearman (mean of per-section medians), Contact F1
(closest 1% of pairs) and Sum RSSD for every run and writes `extended_metrics.csv` and
`per_slice_extended_metrics.csv` into each `<method>_inference/<timestamp>/`.
`--rssd_frame luna` (the default) scores every method in LUNA's frame: truth min–max
normalised per section and mean-centred, prediction min–max normalised per section; methods
that write microns (CeLEry, CellContrast, COME) are converted.

```bash
cd $REPRO/analysis/benchmarking/plots
python compute_extended_metrics.py --dataset mmc_luna --methods luna,g2t,celery,cellcontrast,come \
    --luna_timestamps <ts,...> --scgg_timestamps <ts,...> --celery_timestamps <ts,...> \
    --cellcontrast_timestamps <ts,...> --come_timestamps <ts,...> \
    --rssd_frame luna --backend vectorized --workers 8
python compute_extended_metrics.py --dataset cns_luna --methods luna,g2t,celery \
    --luna_timestamps <ts,...> --scgg_timestamps <ts,...> --celery_timestamps <ts,...> \
    --rssd_frame luna --backend vectorized --workers 2

python plot_method_comparison.py --dataset mmc_luna --luna_timestamps <ts,...> --scgg_timestamps <ts,...> \
    --celery_timestamps <ts,...> --cellcontrast_timestamps <ts,...> --come_timestamps <ts,...>   # Fig. 2b
python plot_method_comparison.py --dataset cns_luna --luna_timestamps <ts,...> --scgg_timestamps <ts,...> \
    --celery_timestamps <ts,...>                                                                  # Fig. 3b
python benchmark_paired_stats.py --dataset mmc_luna
python benchmark_paired_stats.py --dataset cns_luna
```

- The default timestamp lists in `compute_extended_metrics.py` and
  `plot_method_comparison.py` (`DEFAULT_*_TIMESTAMPS`) pin the authors' reported G2T and
  CeLEry runs and the CNS LUNA runs; cortex LUNA, CellContrast and COME are auto-discovered
  from the tree. For your runs, pass `--<method>_timestamps` or replace those lists.
  `benchmark_paired_stats.py` (seed-paired t-tests of G2T against each method, written to
  `benchmark_stats_<dataset>.csv`) has no timestamp options and reads the lists in
  `plot_method_comparison.py`.
- `--backend vectorized` matches the default `scipy` backend to within 1e-7 and is much
  faster. Peak memory is about 16·N² bytes per worker (≈64 GB for the largest CNS section,
  63k cells), so keep `--workers` low on CNS.
- Figure panels: `$SCGG_ARTIFACTS_ROOT/<dataset>/comparison_plots/method_comparison_extended_metrics.{svg,pdf,png}`
  plus the per-run values in `method_comparison_extended_metrics_processed.csv`.
- Per-section reconstructions (the kind shown in Figs. 2c, 3c) are written by the G2T,
  LUNA and CeLEry pipelines to `<method>_inference/<timestamp>/plots/<section>.svg`; re-render them with
  `python $SCGG_REPO/scripts/replot_predictions.py <inference dir> ...`.
- Section-level comparison (§3.1: G2T vs LUNA paired by held-out section). From
  `per_slice_extended_metrics.csv` (column `slice_name` carries a `_<k>` sample suffix):
  average each run's rows within a section, average the runs per section, then pair the
  two methods by section and report the mean difference with its 95% t-interval. There is
  no packaged script for this step.

## 5. Expected results

Mean over five runs (seeds 0–4), all methods scored with `--rssd_frame luna`:

| Benchmark | Method | Spearman ↑ | Contact F1 ↑ | Sum RSSD ↓ |
|---|---|---|---|---|
| Cortex (31 test sections) | **G2T** | **0.4722** | **0.0625** | **76.10** |
| | LUNA | 0.4534 | 0.0597 | 77.30 |
| | CeLEry | 0.233 | 0.0296 | 92.37 |
| | CellContrast | 0.061 | 0.0169 | 114.6 |
| | COME | 0.030 | 0.0104 | 108.4 |
| CNS (14 sections, imputed reference) | **G2T** | **0.1797** | **0.04100** | 622.0 |
| | LUNA | 0.1567 | 0.03815 | 666.3 |
| | CeLEry | 0.055 | 0.0155 | 600.1 |

On CNS, CeLEry's lower Sum RSSD largely reflects its compressed layouts; the paper discusses
this. Other outputs to compare against:

- **Section-level, cortex:** G2T − LUNA Spearman +0.019 (95% CI [0.016, 0.022]); G2T ahead
  in 30 of 31 sections.
- **Table 1:** mean ± SD over seeds 0–9 as printed in the paper. `ablations/ablation_stats.csv`
  holds the paper's values for every row, including Coordinate L2 (0.150 ± 0.144 / 0.0253 ± 0.0137 /
  128.3 ± 36.7).
- **MDS diagnostic:** top-2 share 99.2% at the final step, mean over the 31 cortex test
  sections (SD 0.1%, range 98.8–99.4%); reproducible from the committed
  `mds_variance/mds_var_out/mds_var_cortex_seed0.csv`.
- **Controls (one cortex model, scorer frame of 3.3):** Spearman 0.486 (real) → 0.212
  (`class_mean`), 0.129 (`shuffle_type`), 0.001 (`shuffle_all`).
- **Robustness:** uniform subsampling changes Spearman by at most 0.002 between 25% and 100%
  of cells; full L5 depletion lowers it by 0.019, of which 0.002 is model-attributable.
- **Xenium:** Supplementary Table S1 (data not public).

Re-runs are not guaranteed to be bit-identical across hardware or library versions;
compare at the level of the seed-to-seed variation (Table 1 SDs).

## Repository layout

| Path | Contents |
|---|---|
| `analysis/benchmarking/setup_*_env.sh` | one uv environment per method |
| `analysis/benchmarking/download_*.sh` | LUNA's released data (`download_luna_paper_csvs.sh` fetches LUNA's whole data folder; not needed) |
| `analysis/benchmarking/lsf/` | `submit_pipeline.sh` (+ `_run_pipeline.sh`) for G2T, LUNA and CeLEry; LSF fan-out wrappers for scoring and plotting |
| `analysis/benchmarking/contrastive_baselines/` | CellContrast and COME wrappers, submitters and tests |
| `analysis/benchmarking/ablations/` | Table 1: submitter, analysis, paired statistics, committed results |
| `analysis/benchmarking/robustness/` | expression controls and subsampling/depletion |
| `analysis/benchmarking/mds_variance/` | spectral-mass diagnostic and the paper's log |
| `analysis/benchmarking/plots/` | scoring, statistics, Fig. 2b/3b panels |
| `analysis/benchmarking/niche_eval/` | exploratory niche evaluation; not reported in the paper |
| `analysis/benchmarking/luna/` | empty placeholder (records LUNA commit `64f10eb`); `setup_luna_env.sh` clones LUNA into it |
| `analysis/benchmarking/{CellContrast,COME}/` | created by `setup_cellcontrast_env.sh` / `setup_come_env.sh` (pinned upstream clones) |
| `experiments/` | early exploratory notebooks; not used for the paper |

## Citation

```bibtex
@inproceedings{birk2026g2t,
  title     = {{G2T}: Tissue Reconstruction from Gene Expression via Embedding-Distance Flow Matching},
  author    = {Birk, Sebastian and Theis, Fabian J. and Lotfollahi, Mohammad},
  booktitle = {Proceedings of the Machine Learning in Computational Biology (MLCB) 2026},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  note      = {Volume and pages to be assigned}
}
```

## Licence

BSD 3-Clause (see `LICENSE`). The baselines are run from their authors' repositories
under their own terms; COME's repository ships no licence file, so it is cloned and
imported, not redistributed.
