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

THE CONTROL (--control_from) -- REQUIRED, not optional
------------------------------------------------------
Scoring every presented cell means the scored SET differs between conditions, and
that alone moves the aggregate for two reasons: (i) each cell's rho is computed
against fewer other cells, and (ii) for a CLASS-TARGETED removal the scored
population changes composition, which matters because per-class medians span
0.28 to 0.72 in this benchmark -- depleting a class that scores above the median
lowers the aggregate with no model effect whatsoever.

Both are removed by the MATCHED control: rescore the reference run's OWN
predictions on EXACTLY the cells this condition scored. The reference model saw
the full input, the scored set is identical, so

    model_effect = actual - control_matched

isolates the model's response to the changed input. This is what makes it valid to
score all presented cells instead of holding a fixed evaluation set, and it is why
the fixed-set design (with its hard floor at eval_frac) is no longer needed.
Needs no GPU -- it reuses artifacts already on disk. Every scored cell must exist
in the reference run, so --control_from must be the condition presenting ALL cells;
the code refuses otherwise.

``resolution_sd`` additionally rescores random subsets of the same SIZE over a few
draws. It is a resolution estimate only -- treat any |model_effect| below it as
unresolvable -- not the anchor.

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

    rows, per_class, scored_sets = [], [], {}
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
        meds, n_tot, scored = [], 0, {}
        for sec, (true, pred) in sorted(slices.items()):
            idx = pred.index if e_idx is None else e_idx[sec]
            rho = score_index(true, pred, idx, metric_fn)
            meds.append(float(np.median(rho)))
            n_tot += len(idx)
            scored[sec] = idx          # for the MATCHED control below
            per_class.append(pd.DataFrame({
                "condition": name, "cell_class": cls_of.reindex(idx).to_numpy(),
                "rho": rho}))
        scored_sets[name] = scored
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

    # ---- controls -----------------------------------------------------------
    # MATCHED (primary): score the reference run's OWN predictions on exactly the
    # cells this condition scored. Model saw the full input; scored set identical.
    # So actual - matched isolates the model's response to the changed input, and
    # it is correct even when removal is class-targeted (which shifts the scored
    # population and would otherwise move the aggregate by composition alone).
    # RANDOM (resolution only): random subsets of the same SIZE, repeated, giving
    # an SD. Use it to judge what size of effect is resolvable, not as the anchor.
    if args.control_from:
        ref = collect_slices(Path(args.control_from), sec_of)
        matched, rnd, per_class_ctl = [], [], []
        for _, r in res.iterrows():
            name = r["condition"]
            sc = scored_sets[name]
            missing = [s for s in sc if s not in ref]
            if missing:
                raise SystemExit(f"{name}: sections {missing[:3]} absent from the "
                                 f"reference run — is --control_from the 100% run?")
            meds = []
            for sec, idx in sorted(sc.items()):
                rt, rp = ref[sec]
                extra = idx.difference(rp.index)
                if len(extra):
                    raise SystemExit(
                        f"{name}/{sec}: {len(extra)} scored cell(s) absent from the "
                        f"reference run; --control_from must be the condition that "
                        f"presents ALL cells.")
                rho_c = score_index(rt, rp, idx, metric_fn)
                meds.append(float(np.median(rho_c)))
                # Per-class control. Essential, not decorative: a class-targeted
                # removal thins the neighbourhood of nearby classes MOST, so the
                # metric artifact is largest exactly where a "neighbouring cell
                # types suffer" signal would appear. Only the corrected
                # difference can distinguish the two.
                per_class_ctl.append(pd.DataFrame({
                    "condition": name,
                    "cell_class": cls_of.reindex(idx).to_numpy(),
                    "rho_ctl": rho_c}))
            matched.append(float(np.mean(meds)))

            f = r.get("frac_presented", np.nan)
            if not np.isfinite(f):
                rnd.append(np.nan); continue
            vals = []
            for rep in range(args.control_reps):
                rng = np.random.default_rng(args.control_seed + 1000 * rep
                                            + int(round(f * 1e4)))
                m2 = []
                for sec, (rt, rp) in sorted(ref.items()):
                    k = min(len(rp.index), max(3, int(round(f * len(rp.index)))))
                    sub = pd.Index(rng.choice(rp.index.to_numpy(), size=k,
                                              replace=False))
                    m2.append(float(np.median(score_index(rt, rp, sub, metric_fn))))
                vals.append(float(np.mean(m2)))
            rnd.append(float(np.std(vals)))
        res["control_matched"] = matched
        res["resolution_sd"] = rnd
        res["model_effect"] = res["spearman_mean_of_medians"] - res["control_matched"]

    print("\n=== Spearman by condition ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "model_effect" in res:  # noqa
        print("\ncontrol_matched = the reference run's OWN predictions rescored on "
              "EXACTLY the cells this condition scored. Model saw the full input, "
              "scored set identical, so this absorbs both the metric-vs-N effect "
              "and (crucially, for class-targeted removal) the change in scored-set "
              "COMPOSITION.")
        print("model_effect = actual - control_matched. THIS is the robustness "
              "result. resolution_sd is the spread over random subsets of the same "
              "size; treat any |model_effect| below it as unresolvable.")

    pc = pd.concat(per_class, ignore_index=True)
    tab = (pc.groupby(["condition", "cell_class"])["rho"]
             .agg(median="median", n="size").reset_index())
    print("\n=== per-class Spearman, RAW (median over that class's scored cells) ===")
    print(tab.pivot(index="cell_class", columns="condition", values="median")
             .to_string(float_format=lambda v: f"{v:.4f}"))

    if args.control_from and per_class_ctl:
        ct = (pd.concat(per_class_ctl, ignore_index=True)
                .groupby(["condition", "cell_class"])["rho_ctl"].median()
                .reset_index())
        eff = tab.merge(ct, on=["condition", "cell_class"], how="inner")
        eff["model_effect"] = eff["median"] - eff["rho_ctl"]
        print("\n=== per-class MODEL EFFECT (raw - matched control) ===")
        print("Do NOT read the raw table above for a per-class claim: removing a "
              "class thins its neighbours' local neighbourhoods most, so the metric "
              "artifact peaks exactly where a 'neighbouring types suffer' signal "
              "would. Only these corrected values separate the two.")
        print(eff.pivot(index="cell_class", columns="condition",
                        values="model_effect")
                 .to_string(float_format=lambda v: f"{v:+.4f}"))
        tab = tab.merge(eff[["condition", "cell_class", "rho_ctl", "model_effect"]],
                        on=["condition", "cell_class"], how="left")

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
