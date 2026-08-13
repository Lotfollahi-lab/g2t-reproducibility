#!/usr/bin/env python
"""Score the cell-subsampling robustness conditions on the fixed evaluation set E.

Spearman only, by design: Sum RSSD is a root-SUM so it scales with cell count and
is not comparable across conditions, and Contact F1's 0.01 percentile changes
physical meaning when the cloud is thinned.

WHY THIS SCRIPT EXISTS RATHER THAN compute_extended_metrics.py
--------------------------------------------------------------
Reading metadata_pred.csv as written would manufacture a result. The prediction
is min-max normalised per axis over the WHOLE presented predicted cloud
(scgg/src/utils/diffusion_model/test/test.py:273-274 -> position_normalize, which
takes its else-branch because to_dataframe emits no cell_section column). Per-axis
min-max is set by the single most extreme predicted cell -- usually a non-E cell --
and thinning drops it. With the model held FROZEN, that alone moved an E-only
Spearman 0.6469 -> 0.6494 -> 0.6318 -> 0.7604 across reference/50/25/10%: a +0.13
non-monotonic swing, larger than any robustness effect worth reporting and shaped
like the flattering conclusion "G2T improves when you profile fewer cells".

The truth side has the same structural problem (data_module.py:51) but is benign:
E is present in every condition and pins the bounding box, so measured aspect-ratio
drift is <=0.6%, worth ~1e-4 on the metric.

FIX: subset both frames to E FIRST on one shared index, then per-axis standardise
each over E, then score. For written = (raw - m)/R - 0.5, standardising over E
gives (raw - mean_E(raw))/std_E(raw): the presented-set constants m and R cancel
identically. In the frozen-model scenario above this is stable to four decimals.

Consequence to state in any write-up: the E-only reference value will NOT equal the
published 31-slice number, because E is a subset and the frame differs. Read the
conditions against each other, never against the headline.

USAGE
    python score_subsample_robustness.py \
        --cond_root <ARTIFACTS>/robustness/arm_number \
        --runs uniform_rest100=<out>/uniform_rest100,uniform_rest050=<out>/... \
        --scgg_src /nfs/team361/sb75/scgg/src
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COL_X, COL_Y, COL_CLASS = "coord_X", "coord_Y", "cell_class"


def load_metric_fn(scgg_src: Path):
    """Import the SAME Spearman implementation the paper reports."""
    sys.path.insert(0, str(scgg_src))
    from scgg.evaluation.luna_metrics import compute_spearman_correlation
    return compute_spearman_correlation


def standardise_over_E(a: np.ndarray) -> np.ndarray:
    """Per-axis z-score. Cancels position_normalize's presented-set constants."""
    mu, sd = a.mean(axis=0), a.std(axis=0)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(f"degenerate axis in E (std={sd}); cannot standardise")
    return (a - mu) / sd


def resolve_slice(idx: pd.Index, e_by_sec: dict[str, np.ndarray]) -> str:
    """Recover section identity from cell ids, never from the directory name.

    test.py:242-249 looks the slice name up by cell COUNT
    (mapping_dict[positions_pred.shape[0]]), so under subsampling the written
    directory name is not trustworthy: a collision silently gives two slices the
    same name with no warning.
    """
    hits = [s for s, e in e_by_sec.items() if pd.Index(e).isin(idx).all()]
    if len(hits) != 1:
        raise RuntimeError(
            f"could not identify slice from its cell ids: {len(hits)} candidate "
            f"section(s) fully contained ({hits[:4]}). Either E is incomplete in "
            f"this output or two sections share an E subset.")
    return hits[0]


def score_run(run_root: Path, e_by_sec: dict[str, np.ndarray],
              cls_by_id: pd.Series, metric_fn) -> tuple[dict, pd.DataFrame]:
    per_slice, per_cell_rows = [], []
    preds = sorted(run_root.rglob("metadata_pred.csv"))
    if not preds:
        raise SystemExit(f"no metadata_pred.csv under {run_root}")

    for pth in preds:
        true_p = pth.parent / "metadata_true.csv"
        if not true_p.exists():
            raise SystemExit(f"{pth.parent}: metadata_true.csv missing")
        pred = pd.read_csv(pth, index_col=0)
        true = pd.read_csv(true_p, index_col=0)
        if not pred.index.equals(true.index):
            raise SystemExit(f"{pth.parent}: pred/true indices differ")

        sec = resolve_slice(pred.index, e_by_sec)
        e = pd.Index(e_by_sec[sec])
        # One shared index object for both frames -> identical row order.
        t = standardise_over_E(true.loc[e, [COL_X, COL_Y]].to_numpy(float))
        p = standardise_over_E(pred.loc[e, [COL_X, COL_Y]].to_numpy(float))

        spr = metric_fn(t, p, backend="scipy")
        rho = np.asarray(spr["per_cell"], dtype=float)
        # luna_metrics.py:196 silently drops NaN rho before the median, and
        # aggregate_slices drops NaN slices while n_slices comes from a
        # different stack -- so "mean of 31" can quietly become "mean of 30".
        if np.isnan(rho).any():
            raise SystemExit(f"{sec}: {int(np.isnan(rho).sum())} NaN per-cell rho")
        if int(spr["n"]) != len(e):
            raise SystemExit(f"{sec}: scored {spr['n']} cells, expected {len(e)}")

        per_slice.append({"section": sec, "n_eval": len(e),
                          "median_rho": float(np.median(rho)),
                          "mean_rho": float(rho.mean())})
        per_cell_rows.append(pd.DataFrame({
            "section": sec, "cell_id": e.to_numpy(),
            "cell_class": cls_by_id.reindex(e).to_numpy(), "rho": rho}))

    ps = pd.DataFrame(per_slice).sort_values("section")
    if len(ps) != len(e_by_sec):
        raise SystemExit(f"scored {len(ps)} slices, expected {len(e_by_sec)}")
    if ps["section"].duplicated().any():
        raise SystemExit("two output directories resolved to the same section")
    agg = {"spearman_mean_of_medians": float(ps["median_rho"].mean()),
           "n_slices": int(len(ps)), "n_eval_cells": int(ps["n_eval"].sum())}
    return agg, pd.concat(per_cell_rows, ignore_index=True), ps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond_root", required=True,
                   help="dir written by make_subsample_conditions.py (has E_manifest.csv)")
    p.add_argument("--runs", required=True,
                   help="comma-separated name=/path/to/inference/output pairs")
    p.add_argument("--scgg_src", default="/nfs/team361/sb75/scgg/src")
    p.add_argument("--out_csv", default="")
    args = p.parse_args()

    root = Path(args.cond_root)
    man = pd.read_csv(root / "E_manifest.csv", index_col=0)
    e_by_sec = {str(s): np.sort(g.index.to_numpy())
                for s, g in man.groupby(man["cell_section"].astype(str))}
    cls_by_id = man[COL_CLASS].astype(str)
    print(f"E: {len(man)} cells across {len(e_by_sec)} sections")

    metric_fn = load_metric_fn(Path(args.scgg_src))

    rows, per_class_all, class_sets = [], [], {}
    for item in args.runs.split(","):
        if "=" not in item:
            raise SystemExit(f"--runs entry must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        agg, pc, _ = score_run(Path(path.strip()), e_by_sec, cls_by_id, metric_fn)
        rows.append({"condition": name.strip(), **agg})
        pc["condition"] = name.strip()
        per_class_all.append(pc)
        class_sets[name.strip()] = frozenset(pc["cell_class"].unique())
        print(f"  {name.strip():22s} Spearman(mean-of-medians)="
              f"{agg['spearman_mean_of_medians']:.4f}  slices={agg['n_slices']}")

    if len(set(class_sets.values())) != 1:
        print("\n[warn] the cell_class set in E differs between conditions; "
              "per-class rows are not directly comparable", file=sys.stderr)

    res = pd.DataFrame(rows)
    ref = res.iloc[0]["spearman_mean_of_medians"]
    res["delta_vs_first"] = res["spearman_mean_of_medians"] - ref
    res["pct_vs_first"] = 100.0 * res["delta_vs_first"] / ref

    print("\n=== E-only Spearman by condition (first row = comparator) ===")
    print(res.to_string(index=False))

    # The flat median over E is nearly blind to a depleted class: those cells are
    # a handful of rows and cannot move it. Per-class is what R2 actually asks about.
    pca = pd.concat(per_class_all, ignore_index=True)
    tab = (pca.groupby(["condition", "cell_class"])["rho"]
             .agg(median="median", n="size").reset_index())
    print("\n=== per-class E-only Spearman (median over that class's cells) ===")
    piv = tab.pivot(index="cell_class", columns="condition", values="median")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nNOTE: these values are NOT comparable to the published 31-slice "
          "Spearman — E is a subset and is scored in an E-standardised frame. "
          "Compare conditions to each other only.")

    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
        tab.to_csv(str(Path(args.out_csv).with_suffix("")) + "_per_class.csv",
                   index=False)
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
