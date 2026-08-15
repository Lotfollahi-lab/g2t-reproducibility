#!/usr/bin/env python
"""Negative controls: does G2T use expression, or only cell-type identity?

A reviewer asks for "simple controls such as a cell-type-only predictor and
within-cell-type expression shuffling". This builds the shuffling arm. It is
inference-only -- the checkpoint is untouched; only the expression handed to it
at test time changes.

CONDITIONS
  real            untouched test set (the reference point)
  shuffle_type    gene block permuted among cells OF THE SAME CLASS within each
                  section. Cell-type identity is preserved EXACTLY; within-type
                  expression variation is destroyed. If G2T's reconstruction is
                  really a cell-type -> position lookup, this barely moves.
  shuffle_all     gene block permuted among ALL cells within each section,
                  destroying type identity too. The floor: whatever survives
                  here is not coming from expression at all.

Coordinates, cell_section and cell_class are never touched, so the scored cells
and their ground truth are IDENTICAL across conditions. That is what makes this
comparison clean: unlike the subsampling arms, the scored set does not change,
so no matched control is needed -- score the conditions against each other
directly.

Permuting rows (rather than shuffling each gene independently) preserves each
cell's full expression profile, its library size, and all gene-gene covariance.
The only thing destroyed is the ASSIGNMENT of profiles to positions. A per-gene
shuffle would also destroy the profile's internal structure and would therefore
test a weaker, less interesting hypothesis.

USAGE
    python make_shuffle_controls.py \
        --test_csv <train_run>/work/test.csv \
        --out_dir  <ARTIFACTS>/robustness/arm_shuffle --seed 0

Then submit with submit_subsample_robustness.sh (it discovers any directory
holding a test.csv) and score with score_subsample_robustness.py; --control_from
is NOT needed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["coord_X", "coord_Y", "cell_section", "cell_class"]


def permute_within(df: pd.DataFrame, gene_cols: list, by: list, rng) -> pd.DataFrame:
    """Permute the gene block among rows sharing the ``by`` key.

    Values are moved as whole rows, so profiles stay intact; only which cell
    carries which profile changes.
    """
    out = df.copy()
    block = df[gene_cols].to_numpy()
    new = block.copy()
    for _, idx in df.groupby(by, sort=True, observed=True).indices.items():
        if len(idx) < 2:
            continue                     # nothing to permute; leave as-is
        perm = rng.permutation(len(idx))
        # A derangement is not enforced: with n>=2 a fixed point is possible and
        # is the honest behaviour of a random shuffle. Report the fixed-point
        # rate so the control's strength is auditable.
        new[idx] = block[idx[perm]]
    out[gene_cols] = new
    return out


def class_mean_profiles(df: pd.DataFrame, gene_cols: list) -> pd.DataFrame:
    """Replace every cell's profile with the mean profile of its class.

    THE cell-type-only predictor, built as an input ablation rather than as a
    separate baseline. The model then holds nothing but cell-type identity, yet
    runs through the same architecture, read-out and metric -- so there is no
    strawman to design, and no argument about whether the baseline was made
    deliberately weak.

    Means are taken WITHIN each test section. A class mean cannot encode any
    individual cell's position (it is averaged over every cell of that class
    wherever it sits), so no positional information leaks; taking means within
    the section instead of from the training split simply avoids introducing a
    train/test batch difference on top of the ablation.
    """
    out = df.copy()
    out[gene_cols] = (df.groupby(["cell_section", "cell_class"], observed=True)[gene_cols]
                        .transform("mean"))
    return out


def fixed_point_rate(a: pd.DataFrame, b: pd.DataFrame, gene_cols: list) -> float:
    """Fraction of cells whose profile is unchanged after permutation."""
    same = np.all(a[gene_cols].to_numpy() == b[gene_cols].to_numpy(), axis=1)
    return float(same.mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.test_csv, index_col=0)
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.test_csv} lacks required columns: {missing}")
    if df.index.duplicated().any():
        raise SystemExit("duplicate cell_id values in the index")
    gene_cols = [c for c in df.columns if c not in META_COLS]
    sections = sorted(df["cell_section"].astype(str).unique())
    print(f"{len(df)} cells, {len(gene_cols)} gene columns, {len(sections)} sections")

    df[["cell_section", "cell_class"]].to_csv(out / "cell_index.csv")

    conds = {
        "real": df,
        "class_mean": class_mean_profiles(df, gene_cols),
        "shuffle_type": permute_within(df, gene_cols, ["cell_section", "cell_class"], rng),
        "shuffle_all": permute_within(df, gene_cols, ["cell_section"], rng),
    }

    summary = []
    for name, cdf in conds.items():
        # Coordinates and labels must be byte-identical to the real set, or the
        # comparison is not controlled.
        for c in META_COLS:
            if not cdf[c].equals(df[c]):
                raise SystemExit(f"{name}: column {c} changed — not a pure shuffle")
        if len(cdf) != len(df):
            raise SystemExit(f"{name}: {len(cdf)} rows, expected {len(df)}")
        fp = fixed_point_rate(df, cdf, gene_cols)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        cdf.to_csv(d / "test.csv")
        summary.append({"condition": name, "n_cells": len(cdf),
                        "frac_of_slice_presented": 1.0,
                        "unchanged_profiles": round(fp, 5)})
        print(f"  {name:14s} n={len(cdf):7d}  profiles left in place: {100*fp:5.2f}%")

    pd.DataFrame(summary).to_csv(out / "conditions.csv", index=False)
    (out / "spec.json").write_text(json.dumps({
        "test_csv": str(Path(args.test_csv).resolve()),
        "mode": "shuffle_controls", "seed": args.seed,
        "n_genes": len(gene_cols), "n_sections": len(sections),
        "n_cells_total": int(len(df)), "conditions": summary,
        "note": "Coordinates/labels identical across conditions, so the scored "
                "set is identical and no matched control is needed. Compare "
                "shuffle_type and shuffle_all against real directly.",
    }, indent=2))
    print(f"\nwrote {len(conds)} condition(s) to {out}")
    print(f"n_genes for --n_genes: {len(gene_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
