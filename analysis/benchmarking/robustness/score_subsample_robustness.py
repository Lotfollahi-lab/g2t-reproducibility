#!/usr/bin/env python
"""Score the cell-subsampling robustness conditions. Spearman only.

Sum RSSD is excluded because it is a root-SUM and scales with cell count; Contact
F1 because its 0.01 percentile changes physical meaning when the cloud is thinned.

WHY NOT compute_extended_metrics.py
-----------------------------------
Reading metadata_pred.csv as written would manufacture a result. The prediction is
min-max normalised PER AXIS over the whole presented predicted cloud
(scgg/src/utils/diffusion_model/test/test.py:273-274 -> position_normalize, else
branch, because to_dataframe emits no cell_section column). That scale is set by
the single most extreme predicted cell, which thinning removes. With the model
held FROZEN this alone moved a subset Spearman 0.6469 -> 0.6494 -> 0.6318 ->
0.7604 across reference/50/25/10%: a +0.13 non-monotonic swing, larger than any
real effect and shaped like the flattering conclusion "G2T improves when you
profile fewer cells".

FIX: subset both frames to the scored cells FIRST on one shared index, then
per-axis standardise each over those cells, then score. For
written = (raw - m)/R - 0.5, standardising gives (raw - mean(raw))/std(raw): the
presented-set constants m and R cancel identically (verified to 3e-15).

THE METRIC-vs-N CONTROL (--control_from)
---------------------------------------
Scoring every presented cell means the scored set differs between conditions. Per
cell Spearman of pairwise-distance ranks has N-invariant endpoints (perfect -> 1,
random -> ~0 at any N), so this adds variance rather than much bias -- but that is
an argument, and a number is better. --control_from rescores the 100% run's OWN
predictions on random subsets matching each condition's size. The model is fixed,
so any movement there is PURE METRIC ARTIFACT, and

    model effect = actual(f) - control(f)

is the quantity to report. Needs no GPU: it reuses artifacts already on disk.

Values here are NOT comparable to the published 31-slice Spearman: the scored set
and the frame both differ. Compare conditions to each other only.

USAGE
    python score_subsample_robustness.py \
        --cond_root <ART>/robustness/arm_depth \
        --runs slice100=<runs>/slice100__seed0,slice050=<runs>/slice050__seed0 \
        --control_from <runs>/slice100__seed0 \
        --scgg_src /nfs/team361/sb75/scgg/src
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COL_X, COL_Y, COL_CLASS, COL_SEC = "coord_X", "coord_Y", "cell_class", "cell_section"


def load_metric_fn(scgg_src: Path):
    """Import the SAME Spearman implementation the paper reports."""
    sys.path.insert(0, str(scgg_src))
    from scgg.evaluation.luna_metrics import compute_spearman_correlation
    return compute_spearman_correlation


def standardise(a: np.ndarray) -> np.ndarray:
    """Per-axis z-score over the scored cells; cancels position_normalize."""
    mu, sd = a.mean(axis=0), a.std(axis=0)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(f"degenerate axis (std={sd}); cannot standardise")
    return (a - mu) / sd


def collect_slices(run_root: Path, sec_of: pd.Series) -> dict:
    """{section: (true_df, pred_df)} for one run.

    Section identity comes from the CELL IDS, never the directory name:
    test.py:241-249 resolves the name via mapping_dict[n_cells] built by
    inverting {section: n_cells}, so under subsampling a count collision
    silently merges two sections and a miss yields "unknown".
    """
    out = {}
    preds = sorted(run_root.rglob("metadata_pred.csv"))
    if not preds:
        raise SystemExit(f"no metadata_pred.csv under {run_root}")
    for pth in preds:
        tp = pth.parent / "metadata_true.csv"
        if not tp.exists():
            raise SystemExit(f"{pth.parent}: metadata_true.csv missing")
        pred = pd.read_csv(pth, index_col=0)
        true = pd.read_csv(tp, index_col=0)
        if not pred.index.equals(true.index):
            raise SystemExit(f"{pth.parent}: pred/true indices differ")
        secs = sec_of.reindex(pred.index)
        if secs.isna().any():
            raise SystemExit(f"{pth.parent}: {int(secs.isna().sum())} cell id(s) "
                             f"absent from cell_index.csv — wrong --cond_root?")
        uniq = secs.unique()
        if len(uniq) != 1:
            raise SystemExit(f"{pth.parent}: cells span {len(uniq)} sections "
                             f"({list(uniq)[:4]}) — output directories were merged")
        sec = str(uniq[0])
        if sec in out:
            raise SystemExit(f"two output directories resolved to section {sec}")
        out[sec] = (true, pred)
    return out


def score_index(true: pd.DataFrame, pred: pd.DataFrame, idx: pd.Index,
                metric_fn) -> np.ndarray:
    t = standardise(true.loc[idx, [COL_X, COL_Y]].to_numpy(float))
    p = standardise(pred.loc[idx, [COL_X, COL_Y]].to_numpy(float))
    spr = metric_fn(t, p, backend="scipy")
    rho = np.asarray(spr["per_cell"], dtype=float)
    # luna_metrics.py:196 drops NaN rho before the median, and aggregate_slices
    # drops NaN slices while n_slices comes from a different stack -- so "mean of
    # 31" can quietly become "mean of 30". Refuse instead.
    if np.isnan(rho).any():
        raise SystemExit(f"{int(np.isnan(rho).sum())} NaN per-cell rho")
    if int(spr["n"]) != len(idx):
        raise SystemExit(f"scored {spr['n']} cells, expected {len(idx)}")
    return rho


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond_root", required=True,
                   help="dir from make_subsample_conditions.py (has cell_index.csv)")
    p.add_argument("--runs", required=True, help="comma-separated name=path pairs")
    p.add_argument("--eval", default="all", choices=("all", "E"),
                   help="'all' scores every presented cell (primary); 'E' scores "
                        "only the fixed evaluation set from E_manifest.csv")
    p.add_argument("--control_from", default="",
                   help="run root of the 100%% condition; rescores ITS predictions "
                        "on random subsets to measure the metric-vs-N artifact")
    p.add_argument("--control_reps", type=int, default=3)
    p.add_argument("--control_seed", type=int, default=0)
    p.add_argument("--scgg_src", default="/nfs/team361/sb75/scgg/src")
    p.add_argument("--out_csv", default="")
    args = p.parse_args()

    root = Path(args.cond_root)
    ci = pd.read_csv(root / "cell_index.csv", index_col=0)
    sec_of, cls_of = ci[COL_SEC].astype(str), ci[COL_CLASS].astype(str)
    metric_fn = load_metric_fn(Path(args.scgg_src))

    e_idx = None
    if args.eval == "E":
        man = pd.read_csv(root / "E_manifest.csv", index_col=0)
        e_idx = {s: g.index for s, g in man.groupby(man[COL_SEC].astype(str))}
        print(f"scoring the fixed evaluation set: {len(man)} cells")
    else:
        print("scoring ALL presented cells in each condition")

    rows, per_class = [], []
    n_expected = None
    for item in args.runs.split(","):
        if "=" not in item:
            raise SystemExit(f"--runs entry must be name=path, got {item!r}")
        name, path = (s.strip() for s in item.split("=", 1))
        slices = collect_slices(Path(path), sec_of)
        if n_expected is None:
            n_expected = len(slices)
        elif len(slices) != n_expected:
            raise SystemExit(f"{name}: {len(slices)} slices, others had {n_expected}")
        meds, n_tot = [], 0
        for sec, (true, pred) in sorted(slices.items()):
            idx = pred.index if e_idx is None else e_idx[sec]
            rho = score_index(true, pred, idx, metric_fn)
            meds.append(float(np.median(rho)))
            n_tot += len(idx)
            per_class.append(pd.DataFrame({
                "condition": name, "cell_class": cls_of.reindex(idx).to_numpy(),
                "rho": rho}))
        rows.append({"condition": name, "spearman_mean_of_medians": float(np.mean(meds)),
                     "n_slices": len(meds), "n_scored_cells": n_tot})
        print(f"  {name:14s} Spearman={rows[-1]['spearman_mean_of_medians']:.4f}  "
              f"slices={len(meds)}  cells={n_tot}")

    res = pd.DataFrame(rows)
    frac = None
    cpath = root / "conditions.csv"
    if cpath.exists():
        cf = pd.read_csv(cpath).set_index("condition")["frac_of_slice_presented"]
        res["frac_presented"] = [cf.get(n.split("_seed")[0], np.nan)
                                 for n in res["condition"]]
        frac = res["frac_presented"]

    # ---- metric-vs-N control ------------------------------------------------
    if args.control_from:
        ref = collect_slices(Path(args.control_from), sec_of)
        ref_n = {s: len(pr.index) for s, (_, pr) in ref.items()}
        ctl = []
        for i, r in res.iterrows():
            f = r.get("frac_presented", np.nan)
            if not np.isfinite(f):
                ctl.append((np.nan, np.nan)); continue
            vals = []
            for rep in range(args.control_reps):
                rng = np.random.default_rng(args.control_seed + 1000 * rep
                                            + int(round(f * 1e4)))
                meds = []
                for sec, (true, pred) in sorted(ref.items()):
                    k = max(3, int(round(f * ref_n[sec])))
                    sub = pd.Index(rng.choice(pred.index.to_numpy(),
                                              size=min(k, ref_n[sec]),
                                              replace=False))
                    meds.append(float(np.median(
                        score_index(true, pred, sub, metric_fn))))
                vals.append(float(np.mean(meds)))
            ctl.append((float(np.mean(vals)), float(np.std(vals))))
        res["control_metric_only"] = [c[0] for c in ctl]
        res["control_sd"] = [c[1] for c in ctl]
        res["model_effect"] = res["spearman_mean_of_medians"] - res["control_metric_only"]

    print("\n=== Spearman by condition ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "model_effect" in res:
        print("\ncontrol_metric_only = the 100% run's OWN predictions rescored on "
              "random subsets of the same size (model fixed).")
        print("model_effect = actual - control. THIS is the robustness result; "
              "anything within control_sd is not resolvable.")

    pc = pd.concat(per_class, ignore_index=True)
    tab = (pc.groupby(["condition", "cell_class"])["rho"]
             .agg(median="median", n="size").reset_index())
    print("\n=== per-class Spearman (median over that class's scored cells) ===")
    print(tab.pivot(index="cell_class", columns="condition", values="median")
             .to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nNOTE: not comparable to the published 31-slice Spearman — different "
          "scored set and frame. Compare conditions to each other only.")
    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
        tab.to_csv(str(Path(args.out_csv).with_suffix("")) + "_per_class.csv",
                   index=False)
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
