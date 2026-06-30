#!/usr/bin/env python3
"""plot_svg_comparison.py

Side-by-side visualization of the most spatially-variable genes (SVGs) on
the GROUND-TRUTH coordinates vs the G2T-GENERATED coordinates for one slice.

For a chosen section we:
  1. rank genes by Moran's I (spatial autocorrelation) on the TRUE
     coordinates and take the top-k SVGs;
  2. for each SVG, draw two scatter panels coloured by that gene's
     expression — left = true coords, right = generated coords —
     so you can see whether G2T's reconstruction preserves the gene's
     spatial pattern.

Inputs:
  * --test_csv : the LUNA-format CSV the run was evaluated on
                 (gene columns, then coord_X, coord_Y, cell_section,
                 cell_class; indexed by cell_id). This carries gene
                 expression + true coordinates.
  * --pred_dir : a run's inference dir (we rglob metadata_pred.csv and
                 pick the sub-dir for --section). Predicted coords are
                 read from metadata_pred.csv (index cell_id; coord_X/Y).

Dependencies: numpy, pandas, scikit-learn, matplotlib (all in the scgg env).

Example:
  python plot_svg_comparison.py \
      --test_csv  /nfs/.../mmc_luna/scgg_inference/<TS>/work/test.csv \
      --pred_dir  /nfs/.../mmc_luna/scgg_inference/<TS>/luna_run/test_results \
      --section   mouse2_slice229 \
      --top_k 4 --out svg_mouse2_slice229.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = {"coord_X", "coord_Y", "cell_section", "cell_class"}


# ---------------------------------------------------------------------------
# Moran's I (self-contained: kNN graph + row-normalised weights)
# ---------------------------------------------------------------------------
def morans_i(coords: np.ndarray, expr: np.ndarray, k: int = 6) -> np.ndarray:
    """Per-gene Moran's I on a kNN spatial graph.

    coords: (N, 2) true coordinates.
    expr:   (N, G) expression matrix.
    Returns (G,) Moran's I per gene (higher = more spatially structured).
    """
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]  # drop self

    z = expr - expr.mean(axis=0, keepdims=True)          # (N, G) centred
    denom = (z ** 2).sum(axis=0) + 1e-12                 # (G,)
    # Row-normalised weights => S0 = N, so I = sum_i z_i * mean_j(z_neighbours) / denom.
    neigh_mean = z[idx].mean(axis=1)                     # (N, G): mean of neighbours
    num = (z * neigh_mean).sum(axis=0)                   # (G,)
    return num / denom


# ---------------------------------------------------------------------------
# Procrustes alignment (display only): rotate/reflect/scale pred onto true
# ---------------------------------------------------------------------------
def procrustes_align(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = pred - pred.mean(0)
    t = true - true.mean(0)
    u, _, vt = np.linalg.svd(p.T @ t)
    R = u @ vt                                           # optimal rotation+reflection
    p_rot = p @ R
    # isotropic scale to match spread
    s = (p_rot * t).sum() / ((p_rot ** 2).sum() + 1e-12)
    return p_rot * s + true.mean(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_csv", required=True, help="LUNA-format test CSV (genes+coords+meta).")
    ap.add_argument("--pred_dir", required=True,
                    help="Inference dir to rglob metadata_pred.csv from (a run's test_results).")
    ap.add_argument("--section", default=None,
                    help="cell_section to plot (e.g. mouse2_slice229). Default: first section.")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--knn", type=int, default=6, help="k for Moran's I graph.")
    ap.add_argument("--no_align", action="store_true",
                    help="Do NOT Procrustes-align predicted coords for display.")
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--out", default="svg_comparison.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.test_csv, index_col=0)
    gene_cols = [c for c in df.columns if c not in META_COLS]
    if not gene_cols:
        raise SystemExit("No gene columns found (all columns are metadata?).")

    section = args.section or str(df["cell_section"].iloc[0])
    sub = df[df["cell_section"].astype(str) == section]
    if len(sub) == 0:
        raise SystemExit(f"section {section!r} not found. Available: "
                         f"{sorted(df['cell_section'].astype(str).unique())[:10]} ...")

    true_xy = sub[["coord_X", "coord_Y"]].to_numpy(float)
    expr = sub[gene_cols].to_numpy(float)

    # Find this section's predicted-coords file.
    pred_csvs = sorted(Path(args.pred_dir).rglob("metadata_pred.csv"))
    match = [p for p in pred_csvs if section in p.parent.name]
    if not match:
        raise SystemExit(f"No metadata_pred.csv under {args.pred_dir} whose "
                         f"folder name contains {section!r}. Found: "
                         f"{[p.parent.name for p in pred_csvs][:10]}")
    pred = pd.read_csv(match[0], index_col=0)
    # Align predicted rows to the test-CSV cell order (join on cell_id).
    common = sub.index.intersection(pred.index)
    if len(common) == 0:
        raise SystemExit("No overlapping cell IDs between test CSV and metadata_pred.csv.")
    sub = sub.loc[common]
    true_xy = sub[["coord_X", "coord_Y"]].to_numpy(float)
    expr = sub[gene_cols].to_numpy(float)
    pred_xy = pred.loc[common][["coord_X", "coord_Y"]].to_numpy(float)
    if not args.no_align:
        pred_xy = procrustes_align(pred_xy, true_xy)

    # Rank SVGs by Moran's I on true coords.
    mi = morans_i(true_xy, expr, k=args.knn)
    order = np.argsort(mi)[::-1][:args.top_k]
    top_genes = [(gene_cols[i], float(mi[i])) for i in order]
    print(f"[svg] section={section}  N={len(common)} cells  top-{args.top_k} SVGs:")
    for g, v in top_genes:
        print(f"    {g:15s}  Moran's I = {v:.3f}")

    # Plot: top_k rows x 2 cols (true | generated), coloured by each SVG.
    k = len(top_genes)
    fig, axes = plt.subplots(k, 2, figsize=(8, 3.6 * k), squeeze=False)
    for r, (gi, (gene, mival)) in enumerate(zip(order, top_genes)):
        vals = expr[:, gi]
        vmin, vmax = np.percentile(vals, [2, 98])
        for c, (xy, title) in enumerate([(true_xy, "Ground truth"),
                                         (pred_xy, "G2T generated")]):
            ax = axes[r][c]
            sca = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=args.point_size,
                             cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.invert_yaxis()
            if c == 0:
                ax.set_ylabel(f"{gene}\n(Moran's I={mival:.2f})", fontsize=10)
            ax.set_title(title, fontsize=10)
        fig.colorbar(sca, ax=axes[r], fraction=0.025, pad=0.02, label="expression")

    fig.suptitle(f"Top spatially-variable genes — section {section}", fontsize=12)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[svg] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
