# MDS variance-explained diagnostic

**Question (reviewer):** how often is G2T's learned squared-distance matrix
`D_hat` poorly approximated by a 2-D embedding? Report the variance explained by
the top-two MDS eigenvalues.

**Metric.** For each slice we double-centre the predicted squared-distance
matrix, `B = -1/2 J D_hat J`, and report

```
var_explained = (lambda_1 + lambda_2) / sum_{lambda > 0} lambda
```

the fraction of `B`'s positive spectral mass captured by its top two
eigenvalues. `var_explained -> 1` means `D_hat` is (almost) exactly
2-D-embeddable; a lower value means the learned distances genuinely need more
than two dimensions and the MDS read-out discards structure.

## How it works

`scgg/src/models/edm_head.py` (`_mds_align_positions`) has an env-gated,
eval-only hook. When `SCGG_LOG_MDS_VAR` is set it computes the fraction with its
own full **fp64 eigendecomposition** — so it is independent of
`model.edm.mds_solver` (works for both the `eigh` and `lobpcg` paths) and needs
no override.

`forward()` (hence MDS) runs at **every reverse-diffusion step**, and each test
slice is sampled independently, so the log holds `n_steps x n_slices` rows.
Within one slice's reverse chain `n_valid` is constant and the per-pass `call`
index increases; the **converged** value is the last row of each maximal
contiguous constant-`n_valid` run. `analyse_mds_var.py` extracts exactly that.

`SCGG_LOG_MDS_VAR` values:
- unset/empty -> off (zero cost on normal runs)
- `1`/`true`/`yes`/`on` -> print one line per (step, slice), no file
- any other value -> treated as a **CSV path** to append to (recommended)

## Run it (farm)

> Sync the local edits to the farm first (edm_head.py hook +
> submit_pipeline.sh passthrough are local until synced).

Inference-only on a trained G2T **cortex** checkpoint (single test pass -> no
validation-epoch pollution; cortex slices ~7k cells so the extra fp64 eigh is
cheap):

```bash
bash submit_mds_var.sh \
    --checkpoint /nfs/team361/sb75/scgg-reproducibility/artifacts/mmc_luna/scgg_model/<TS>/best_model.ckpt
```

Or, to attach the diagnostic to any run you launch through the LSF pipeline,
just export the CSV path before submitting (submit_pipeline.sh forwards it):

```bash
SCGG_LOG_MDS_VAR=/abs/path/mds_var.csv  bash submit_pipeline.sh --method scgg ...
```

## Summarise

```bash
python analyse_mds_var.py mds_var_out/mds_var_cortex_seed0.csv
```

prints the converged var-explained per section, the mean ± SD across sections
(the number for the paper), and the all-steps mean for context. One cortex seed
suffices — it is a geometric property of the trained model, low-variance across
seeds.
