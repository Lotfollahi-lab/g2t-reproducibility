#!/usr/bin/env python3
"""analyse_mds_var.py

Summarise the top-2 MDS eigenvalue variance-explained diagnostic emitted by
``scgg/src/models/edm_head.py`` when inference is run with
``SCGG_LOG_MDS_VAR=<csv path>``.

Reviewer request: "how often is the learned distance matrix poorly approximated
by a 2-D embedding? Report the variance explained by the top-two MDS
eigenvalues." The diagnostic logs, for every reverse-diffusion step and every
slice, the fraction

    var_explained = (lambda_1 + lambda_2) / sum_{lambda > 0} lambda

of the double-centred Gram matrix B = -1/2 J D_hat J (D_hat = the model's
predicted squared-distance matrix). var_explained -> 1 means D_hat is (almost)
exactly 2-D-embeddable; lower means the learned distances need >2 dimensions.

forward() runs at EVERY reverse-diffusion step, so the CSV holds
n_steps x n_slices rows. The CONVERGED (final-step) value is what the paper
should quote. Within one slice's independently-sampled reverse chain
``n_valid`` is constant and ``call`` increases monotonically; a new slice
begins when ``n_valid`` changes. So the converged value per slice = the last
row of each MAXIMAL CONTIGUOUS run of constant ``n_valid`` (robust even if two
non-adjacent slices share a cell count).

    python analyse_mds_var.py mds_var_cortex_seed0.csv

Prints the per-slice converged fraction and the mean +- SD across slices (the
number to put in the paper), plus the mean over ALL steps for context.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def converged_per_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Last row of each maximal contiguous constant-``n_valid`` run.

    Sort by ``call`` (reverse-step order) first so contiguity reflects the
    actual sampling order, then cut a new group whenever ``n_valid`` changes
    from the previous row. Returns one row per slice (its final reverse step).
    """
    d = df.sort_values("call", kind="stable").reset_index(drop=True)
    grp = (d["n_valid"] != d["n_valid"].shift()).cumsum()
    # last row within each contiguous run
    last = d.groupby(grp, sort=False).tail(1).reset_index(drop=True)
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV written by SCGG_LOG_MDS_VAR=<path>")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    df = pd.read_csv(path)
    need = {"call", "slice_b", "n_valid", "top2_frac"}
    if not need.issubset(df.columns):
        print(f"error: CSV must have columns {sorted(need)}; got "
              f"{list(df.columns)}", file=sys.stderr)
        return 1
    df = df.dropna(subset=["top2_frac"])
    if df.empty:
        print("error: no valid rows", file=sys.stderr)
        return 1

    final = converged_per_slice(df)
    fr = final["top2_frac"].to_numpy(float)

    print(f"\nfile: {path}")
    print(f"total rows (step x slice): {len(df)}   slices detected: {len(final)}\n")
    print("converged (final reverse step) var-explained per slice:")
    for _, r in final.iterrows():
        print(f"  n_valid={int(r['n_valid']):>7d}   var_explained={r['top2_frac']:.4f}")

    print("\n--- summary ---")
    print(f"per-slice converged var-explained: mean {fr.mean():.4f}  "
          f"SD {fr.std(ddof=1) if len(fr) > 1 else 0.0:.4f}  "
          f"min {fr.min():.4f}  max {fr.max():.4f}   (n_slices={len(fr)})")
    print(f"  -> paper number: {100*fr.mean():.1f}% "
          f"(range {100*fr.min():.1f}-{100*fr.max():.1f}%)")
    allsteps = df["top2_frac"].to_numpy(float)
    print(f"all steps (for context): mean {allsteps.mean():.4f}  "
          f"min {allsteps.min():.4f}  max {allsteps.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
