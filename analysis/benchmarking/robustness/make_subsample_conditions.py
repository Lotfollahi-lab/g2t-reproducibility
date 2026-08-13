#!/usr/bin/env python
"""Build per-condition test CSVs for the cell-subsampling robustness experiment.

Reviewer R2 asks how robust G2T is to the number and relative abundance of
profiled cells, since dissociated assays recover only a fraction of the cells
present with cell-type-specific bias.

DESIGN
------
Inference-only: no retraining. We reuse a trained checkpoint and vary only the
cell set presented to the model.

A fixed evaluation set E is chosen per slice and is present in EVERY condition;
only the *other* cells vary. All metrics are computed on E alone, so the
evaluation set is identical across conditions and any change is attributable to
the model's input rather than to measuring different cells.

We emit LUNA-format test CSVs rather than subsampled h5ads. run_scgg_inference.py
accepts --train_csv/--test_csv directly, and in test_only mode the train CSV is
never loaded (only stat()ed, and not even that with --n_genes), so pointing
--train_csv at the training run's existing work/train.csv skips all h5ad reading.
One test.csv holds all 31 sections, so one inference run covers one condition.

TWO LABELLING TRAPS THIS SCRIPT AVOIDS
--------------------------------------
1. Because E is in every condition, retention has a FLOOR at E's share. Keeping
   25% of "the rest" presents 0.2 + 0.25*0.8 = 40% of the slice, not 25%. Every
   condition is therefore labelled by the fraction of the slice ACTUALLY
   presented, and the summary prints both numbers. Nothing below eval_frac is
   reachable in the uniform arm.
2. For the composition arm, E is drawn ONLY from non-target classes
   (--eval_exclude_classes), so the target class can be taken to 0% without the
   floor. Use one invocation per target class, each with its own E manifest, and
   compare conditions within an arm only.

USAGE (cell-number arm)
    python make_subsample_conditions.py \
        --test_csv  <train_run>/work/test.csv \
        --out_dir   <ARTIFACTS>/robustness/arm_number \
        --uniform_fracs 1.0,0.75,0.5,0.25,0.10 --seed 0

USAGE (composition arm, one target class)
    python make_subsample_conditions.py \
        --test_csv  <train_run>/work/test.csv \
        --out_dir   <ARTIFACTS>/robustness/arm_deplete_L5ET \
        --eval_exclude_classes "L5 ET" \
        --deplete_class "L5 ET" --deplete_levels 1.0,0.5,0.25,0.0 --seed 0
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
    """Class-proportional random subset of one slice, as an index array.

    Proportional (not min-1) so E's composition matches the slice: forcing rare
    classes in would over-represent them. Classes in ``exclude_classes`` are
    never eligible, which is what lets the composition arm deplete a class to 0.
    """
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_csv", required=True,
                   help="work/test.csv from the training run (all sections)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--eval_frac", type=float, default=0.20)
    p.add_argument("--eval_exclude_classes", default="",
                   help="comma-separated cell_class values kept OUT of E")
    p.add_argument("--uniform_fracs", default="1.0,0.75,0.5,0.25,0.10",
                   help="fractions of the NON-E cells to retain, uniformly")
    p.add_argument("--deplete_class", default="",
                   help="if set, also emit conditions depleting this class")
    p.add_argument("--deplete_levels", default="1.0,0.5,0.25,0.0",
                   help="fractions of --deplete_class to retain")
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
        raise SystemExit("duplicate cell_id values in the index — cannot key E")
    n_genes = df.shape[1] - len(META_COLS)
    excl = {s.strip() for s in args.eval_exclude_classes.split(",") if s.strip()}
    sections = sorted(df["cell_section"].astype(str).unique())
    print(f"{len(df)} cells, {n_genes} gene columns, {len(sections)} sections")
    if excl:
        print(f"classes excluded from E: {sorted(excl)}")

    # ---- fixed evaluation set, one per slice -------------------------------
    e_idx: dict[str, np.ndarray] = {}
    for sec in sections:
        sub = df[df["cell_section"].astype(str) == sec]
        e_idx[sec] = stratified_eval_set(sub, args.eval_frac, excl, rng)

    e_rows = pd.concat([
        df.loc[idx, ["cell_section", "cell_class"]] for idx in e_idx.values()
    ])
    e_rows.index.name = df.index.name or "cell_id"
    e_rows.to_csv(out / "E_manifest.csv")
    print(f"E: {len(e_rows)} cells "
          f"({100.0 * len(e_rows) / len(df):.1f}% of the test split)")

    # ---- condition specs ---------------------------------------------------
    conds: list[tuple[str, dict]] = []
    for f in [float(s) for s in args.uniform_fracs.split(",") if s.strip()]:
        conds.append((f"uniform_rest{int(round(f * 100)):03d}",
                      {"kind": "uniform", "rest_frac": f}))
    if args.deplete_class:
        for f in [float(s) for s in args.deplete_levels.split(",") if s.strip()]:
            conds.append((f"deplete_{int(round(f * 100)):03d}",
                          {"kind": "deplete", "target": args.deplete_class,
                           "target_frac": f}))

    summary = []
    for name, spec in conds:
        keep_all = []
        for sec in sections:
            sub = df[df["cell_section"].astype(str) == sec]
            e = e_idx[sec]
            rest = sub.index.difference(pd.Index(e))
            if spec["kind"] == "uniform":
                n_take = int(round(spec["rest_frac"] * len(rest)))
                sel = rng.choice(rest.to_numpy(), size=n_take, replace=False) \
                    if n_take > 0 else np.array([], dtype=rest.dtype)
            else:
                is_t = sub.loc[rest, "cell_class"].astype(str) == spec["target"]
                tgt, oth = rest[is_t.to_numpy()], rest[~is_t.to_numpy()]
                n_take = int(round(spec["target_frac"] * len(tgt)))
                sel = np.concatenate([
                    oth.to_numpy(),
                    rng.choice(tgt.to_numpy(), size=n_take, replace=False)
                    if n_take > 0 else np.array([], dtype=tgt.dtype)])
            keep_all.append(np.concatenate([e, sel]))

        keep = np.sort(np.concatenate(keep_all))
        cond_df = df.loc[keep]
        # Guard the last-section merge bug in data_module._generate_slice_indices:
        # a trailing section with exactly one cell is absorbed into the previous
        # graph, silently corrupting two slices' counts.
        per_slice = cond_df["cell_section"].astype(str).value_counts()
        if int(per_slice.min()) <= 1:
            raise SystemExit(f"{name}: a section has {per_slice.min()} cell(s); "
                             f"raise --eval_frac or drop this condition")
        if len(per_slice) != len(sections):
            raise SystemExit(f"{name}: {len(per_slice)} sections, expected "
                             f"{len(sections)} — a slice vanished")

        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        cond_df.to_csv(d / "test.csv")
        frac_presented = len(cond_df) / len(df)
        summary.append({"condition": name, **spec, "n_cells": len(cond_df),
                        "frac_of_slice_presented": round(frac_presented, 4),
                        "min_slice_cells": int(per_slice.min())})
        print(f"  {name:22s} n={len(cond_df):7d}  "
              f"presented={100 * frac_presented:5.1f}% of the split  "
              f"min_slice={int(per_slice.min())}")

    pd.DataFrame(summary).to_csv(out / "conditions.csv", index=False)
    (out / "spec.json").write_text(json.dumps({
        "test_csv": str(Path(args.test_csv).resolve()),
        "eval_frac": args.eval_frac,
        "eval_exclude_classes": sorted(excl),
        "seed": args.seed, "n_genes": n_genes,
        "n_sections": len(sections), "conditions": summary,
        "note": "Report frac_of_slice_presented, not rest_frac: E is present in "
                "every condition so retention is floored at eval_frac.",
    }, indent=2))
    print(f"\nwrote {len(conds)} condition(s) + E_manifest.csv to {out}")
    print(f"n_genes for --n_genes: {n_genes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
