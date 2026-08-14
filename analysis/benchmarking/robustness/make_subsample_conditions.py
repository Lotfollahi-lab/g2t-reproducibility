#!/usr/bin/env python
"""Build per-condition test CSVs for the cell-subsampling robustness experiment.

Reviewer R2 asks how robust G2T is to the number and relative abundance of
profiled cells, since dissociated assays recover only a fraction of the cells
present with cell-type-specific bias.

Inference-only: reuse a trained checkpoint and vary the cell set presented.

TWO MODES
---------
``--slice_fracs`` (PRIMARY, all-presented). Subsample each slice directly to a
fraction of its cells and score EVERY presented cell. The condition label is the
fraction, with no arithmetic to misread, and there is NO FLOOR -- 5% recovery is
reachable, which is the regime a real dissociated assay operates in.

  Why this is primary. An earlier design held a fixed 20% evaluation set E
  present in every condition and scored only E, to avoid comparing metrics
  computed on different cell sets. That over-corrected. Per-cell Spearman of
  pairwise-distance ranks has N-INVARIANT ENDPOINTS -- a perfect model scores 1
  and a random one ~0 at any N -- so thinning the scored set adds variance, not
  much bias, at N in the thousands. Meanwhile fixed-E cost a hard 20% floor,
  answered a context question rather than the practical one, and diluted the
  perturbation so much that the reconstruction plots looked unchanged.
  The residual "metric vs N" concern is measured directly and for free by
  score_subsample_robustness.py --control_from, which rescores the 100% run's
  OWN predictions on random subsets: model fixed, so any movement there is pure
  metric artifact, and the difference from the real conditions is the model
  effect.

``--uniform_fracs`` / ``--deplete_class`` (fixed-E, retained). Holds a
class-stratified evaluation set E present in every condition and scores only E.
Retained because it isolates a different question -- does surrounding context
help place a FIXED set of cells -- and because earlier runs used it. Note the
floor: keeping f of the non-E cells presents ``eval_frac + f*(1-eval_frac)`` of
the slice, so at eval_frac=0.2 "rest010" presents 28%, not 10%. Conditions are
labelled by fraction ACTUALLY presented in conditions.csv either way.

Subsampling is plain uniform (not class-stratified), which is what unbiased
recovery of a fraction of cells actually looks like, and matches the control.

USAGE (primary)
    python make_subsample_conditions.py \
        --test_csv <train_run>/work/test.csv \
        --out_dir  <ARTIFACTS>/robustness/arm_depth \
        --slice_fracs 1.0,0.9,0.8,0.7,0.6,0.5,0.25,0.10,0.05 --seed 0

USAGE (composition arm; target can reach 0% because E excludes it)
    python make_subsample_conditions.py \
        --test_csv <train_run>/work/test.csv \
        --out_dir  <ARTIFACTS>/robustness/arm_deplete \
        --eval_exclude_classes "L2/3 IT" --deplete_class "L2/3 IT" \
        --deplete_levels 1.0,0.5,0.25,0.0 --uniform_fracs 1.0 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# LUNA CSV contract (scgg/scripts/run_scgg_train.py::_build_luna_csv): gene
# columns first, then these four, with a numeric cell_id index.
META_COLS = ["coord_X", "coord_Y", "cell_section", "cell_class"]


def stratified_eval_set(sub: pd.DataFrame, frac: float,
                        exclude_classes: set, rng) -> np.ndarray:
    """Class-proportional random subset of one slice (fixed-E mode only)."""
    picks = []
    for cls, g in sub.groupby("cell_class", sort=True):
        if str(cls) in exclude_classes:
            continue
        n_take = int(round(frac * len(g)))
        if n_take <= 0:
            continue
        picks.append(rng.choice(g.index.to_numpy(), size=n_take, replace=False))
    if not picks:
        raise RuntimeError("empty evaluation set — check --eval_frac / "
                           "--eval_exclude_classes")
    return np.sort(np.concatenate(picks))


def write_condition(df: pd.DataFrame, keep: np.ndarray, out: Path, name: str,
                    n_sections: int) -> dict:
    """Write one condition's test.csv, with the guards that matter."""
    cond_df = df.loc[np.sort(keep)]
    per_slice = cond_df["cell_section"].astype(str).value_counts()
    # data_module._generate_slice_indices has an off-by-one on the LAST section:
    # a trailing section left with exactly one cell is absorbed into the previous
    # graph, silently corrupting two slices' counts.
    if int(per_slice.min()) <= 1:
        raise SystemExit(f"{name}: a section has {per_slice.min()} cell(s); "
                         f"raise the fraction or drop this condition")
    if len(per_slice) != n_sections:
        raise SystemExit(f"{name}: {len(per_slice)} sections, expected "
                         f"{n_sections} — a slice vanished")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    cond_df.to_csv(d / "test.csv")
    frac = len(cond_df) / len(df)
    print(f"  {name:20s} n={len(cond_df):7d}  presented={100 * frac:5.1f}%  "
          f"min_slice={int(per_slice.min())}")
    return {"condition": name, "n_cells": len(cond_df),
            "frac_of_slice_presented": round(frac, 4),
            "min_slice_cells": int(per_slice.min())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_csv", required=True,
                   help="work/test.csv from the training run (all sections)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--slice_fracs", default="",
                   help="PRIMARY MODE: fractions of each slice to present; every "
                        "presented cell is scored. No evaluation set, no floor.")
    p.add_argument("--eval_frac", type=float, default=0.20,
                   help="fixed-E mode only: share of each slice held as E")
    p.add_argument("--eval_exclude_classes", default="",
                   help="fixed-E mode: comma-separated classes kept OUT of E")
    p.add_argument("--uniform_fracs", default="",
                   help="fixed-E mode: fractions of the NON-E cells to retain")
    p.add_argument("--deplete_class", "--deplete_classes", dest="deplete_class",
                   default="",
                   help="fixed-E mode: comma-separated cell_class values to "
                        "deplete TOGETHER as one block. A block is often what the "
                        "literature actually supports: Tasic et al. 2018 (Nature "
                        "563:72-78, Methods) report poor isolation survival for "
                        "'L5 types' via Rbp4-Cre, which labels L5 IT and L5 ET "
                        "together, not L5 ET alone. Depleting the block also gives "
                        "a spatially structured (layer-restricted) perturbation, "
                        "which uniform thinning cannot produce.")
    p.add_argument("--deplete_levels", default="1.0,0.5,0.25,0.0")
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
        raise SystemExit("duplicate cell_id values in the index — cannot key cells")
    n_genes = df.shape[1] - len(META_COLS)
    sections = sorted(df["cell_section"].astype(str).unique())
    print(f"{len(df)} cells, {n_genes} gene columns, {len(sections)} sections")

    # Written in BOTH modes. The scorer needs it to recover a slice's identity
    # from its cell ids: the written output directory name comes from a
    # cell-COUNT lookup (test.py:242-249 mapping_dict[positions_pred.shape[0]]),
    # which is not trustworthy once we change cell counts, and a collision would
    # silently give two slices the same name. Also supplies cell_class for the
    # per-class breakdown.
    df[["cell_section", "cell_class"]].to_csv(out / "cell_index.csv")

    slice_fracs = [float(s) for s in args.slice_fracs.split(",") if s.strip()]
    uniform_fracs = [float(s) for s in args.uniform_fracs.split(",") if s.strip()]
    if not slice_fracs and not uniform_fracs and not args.deplete_class:
        raise SystemExit("give --slice_fracs (primary) or --uniform_fracs / "
                         "--deplete_class (fixed-E mode)")

    summary, mode = [], "all_presented" if slice_fracs else "fixed_eval"

    # ---- PRIMARY: all-presented -------------------------------------------
    for f in slice_fracs:
        keep = []
        for sec in sections:
            idx = df.index[df["cell_section"].astype(str) == sec].to_numpy()
            n_take = int(round(f * len(idx)))
            keep.append(rng.choice(idx, size=n_take, replace=False)
                        if n_take > 0 else np.array([], dtype=idx.dtype))
        summary.append(write_condition(
            df, np.concatenate(keep), out,
            f"slice{int(round(f * 100)):03d}", len(sections)))

    # ---- fixed-E mode ------------------------------------------------------
    if uniform_fracs or args.deplete_class:
        excl = {s.strip() for s in args.eval_exclude_classes.split(",") if s.strip()}
        if excl:
            print(f"classes excluded from E: {sorted(excl)}")
        e_idx = {sec: stratified_eval_set(
            df[df["cell_section"].astype(str) == sec], args.eval_frac, excl, rng)
            for sec in sections}
        e_rows = pd.concat([df.loc[i, ["cell_section", "cell_class"]]
                            for i in e_idx.values()])
        e_rows.index.name = df.index.name or "cell_id"
        e_rows.to_csv(out / "E_manifest.csv")
        print(f"E: {len(e_rows)} cells ({100 * len(e_rows) / len(df):.1f}% of split)")

        targets = {s.strip() for s in args.deplete_class.split(",") if s.strip()}
        if targets:
            unknown = targets - set(df["cell_class"].astype(str).unique())
            if unknown:
                raise SystemExit(f"--deplete_class names absent from the data: "
                                 f"{sorted(unknown)}")
            n_t = int(df["cell_class"].astype(str).isin(targets).sum())
            print(f"depletion target(s) {sorted(targets)}: {n_t} cells "
                  f"({100 * n_t / len(df):.1f}% of the split)")
            if not targets <= excl:
                raise SystemExit(
                    "every --deplete_class must also be in --eval_exclude_classes, "
                    "or E will contain cells of the class you are depleting and the "
                    "target can never fall below E's share.")
        specs = [(f"uniform_rest{int(round(f * 100)):03d}", "uniform", f)
                 for f in uniform_fracs]
        if targets:
            specs += [(f"deplete_{int(round(f * 100)):03d}", "deplete", f)
                      for f in [float(s) for s in args.deplete_levels.split(",")
                                if s.strip()]]
        for name, kind, f in specs:
            keep = []
            for sec in sections:
                sub = df[df["cell_section"].astype(str) == sec]
                e = e_idx[sec]
                rest = sub.index.difference(pd.Index(e))
                if kind == "uniform":
                    k = int(round(f * len(rest)))
                    sel = (rng.choice(rest.to_numpy(), size=k, replace=False)
                           if k > 0 else np.array([], dtype=rest.dtype))
                else:
                    is_t = sub.loc[rest, "cell_class"].astype(str).isin(targets)
                    tgt, oth = rest[is_t.to_numpy()], rest[~is_t.to_numpy()]
                    k = int(round(f * len(tgt)))
                    sel = np.concatenate([
                        oth.to_numpy(),
                        rng.choice(tgt.to_numpy(), size=k, replace=False)
                        if k > 0 else np.array([], dtype=tgt.dtype)])
                keep.append(np.concatenate([e, sel]))
            summary.append(write_condition(df, np.concatenate(keep), out,
                                           name, len(sections)))

    pd.DataFrame(summary).to_csv(out / "conditions.csv", index=False)
    (out / "spec.json").write_text(json.dumps({
        "test_csv": str(Path(args.test_csv).resolve()),
        "mode": mode, "seed": args.seed, "n_genes": n_genes,
        "n_sections": len(sections), "n_cells_total": int(len(df)),
        "eval_frac": args.eval_frac if mode == "fixed_eval" else None,
        "conditions": summary,
        "note": ("all_presented: every presented cell is scored; the condition "
                 "label IS the presented fraction." if mode == "all_presented"
                 else "fixed_eval: only E is scored; retention is floored at "
                      "eval_frac, so report frac_of_slice_presented, not the label."),
    }, indent=2))
    print(f"\nmode={mode}; wrote {len(summary)} condition(s) to {out}")
    print(f"n_genes for --n_genes: {n_genes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
