#!/usr/bin/env python3
"""cellcharter_niche_eval.py

Downstream biological validation for G2T (MLCB reviewer request): do the
reconstructed neighbourhoods support recovery of the true tissue niches?

We run the SAME CellCharter niche pipeline on four coordinate sources for the
CNS scRNA-seq test cells and score each against an INDEPENDENT ground-truth
annotation (``Sub_molecular_tissue_region`` from the STARmap PLUS metadata):

    reference — the spatially-imputed coords (silver AnnData ``obsm['spatial']``)
    luna      — LUNA-reconstructed coords     (test-set ``metadata_pred.csv``)
    g2t       — G2T-reconstructed coords       (test-set ``metadata_pred.csv``)
    celery    — CeLEry-reconstructed coords    (test-set ``metadata_pred.csv``)

Fairness (the point): the cell representation is the SAME for every source — a
PCA of the shared 600-dim Harmony latent that all reconstruction models were
given as input. Only the SPATIAL GRAPH differs between runs (built from each
source's coordinates), so any difference in niche recovery is due to coordinate
quality, not expression. Identical n_clusters / graph / aggregation / seed
across all four.

Metrics: NMI and ARI between CellCharter niches and the GT regions, per source.
Expected story: g2t ~ reference > luna (and vs celery).

Data facts this script assumes (verified against the codebase):
  * silver AnnData: X = 600-dim latent, obsm['spatial'] = reference coords,
    obs['cell_id'] = the join id (e.g. "well06_0"), obs['cell_section'].
  * metadata.csv (Single Cell Portal format): TAB-separated, a second "TYPE"
    header row to skip; id column "NAME"; label "Sub_molecular_tissue_region".
  * metadata_pred.csv: index = cell_ID; columns cell_class, coord_X, coord_Y.

Environment: cellcharter + squidpy + scanpy + anndata + scikit-learn + matplotlib.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Per-method colours — MUST match the benchmark figures
# (scgg-reproducibility/analysis/benchmarking/plots/plot_method_comparison.py
# :: METHODS, Okabe-Ito). 'reference' is the imputed ceiling (not a benchmarked
# method) -> neutral grey.
METHOD_COLORS = {
    "g2t":       "#D55E00",   # vermillion (proposed method)
    "luna":      "#0072B2",   # blue       (diffusion baseline)
    "celery":    "#009E73",   # bluish green (supervised-MLP baseline)
    "reference": "#999999",   # grey       (imputed ceiling)
}


def load_pred_coords(path: str, x_col: str, y_col: str, label: str = "") -> pd.DataFrame:
    """metadata_pred.csv (index=cell_ID, cols coord_X/coord_Y). Accepts a file
    or a directory, in which case it RECURSIVELY globs every metadata_pred.csv
    at any depth (one per section, e.g. .../model_*/well1_5_0/metadata_pred.csv).
    Returns a DataFrame indexed by cell id with columns [x, y]."""
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "**" / "metadata_pred.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No metadata_pred.csv under {p} (run inference first?)")
        df = pd.concat([pd.read_csv(f, index_col=0) for f in files])
    else:
        files = [str(p)]
        df = pd.read_csv(p, index_col=0)
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"{c!r} not in {path} (cols: {list(df.columns)})")
    out = df[[x_col, y_col]].copy()
    out.index = out.index.astype(str)
    n_raw = len(out)
    out = out[~out.index.duplicated(keep="first")]
    out.columns = ["x", "y"]
    dup = n_raw - len(out)
    msg = f"[niche]   {label or path}: {len(files)} file(s), {len(out)} cells"
    if dup:
        msg += (f"  (! {dup} duplicate ids dropped — multiple epoch/seed dirs may "
                f"have been globbed; point at a single results dir)")
    print(msg, flush=True)
    return out


def load_expr(path: str, id_col: str, section_col: str):
    """Silver AnnData(s): single .h5ad or a directory of per-section .h5ads."""
    import anndata as ad
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.h5ad")))
        if not files:
            raise FileNotFoundError(f"No .h5ad under {p}")
        adata = ad.concat([ad.read_h5ad(f) for f in files], join="outer", merge="same")
    else:
        adata = ad.read_h5ad(p)
    if id_col not in adata.obs:
        raise KeyError(f"obs[{id_col!r}] missing (have: {list(adata.obs.columns)})")
    if "spatial" not in adata.obsm:
        raise KeyError("obsm['spatial'] (reference coords) missing from the AnnData")
    adata.obs["_key"] = adata.obs[id_col].astype(str).values
    if section_col not in adata.obs:
        adata.obs[section_col] = "all"
    return adata


def run_cellcharter(adata, rep_key, sample_key, n_layers, n_clusters, seed):
    import squidpy as sq
    import cellcharter as cc
    sq.gr.spatial_neighbors(adata, library_key=sample_key,
                            coord_type="generic", delaunay=True)
    cc.gr.remove_long_links(adata)
    cc.gr.aggregate_neighbors(adata, n_layers=n_layers, use_rep=rep_key,
                              out_key="X_cellcharter")
    gmm = cc.tl.Cluster(n_clusters=n_clusters, random_state=seed)
    gmm.fit(adata, use_rep="X_cellcharter")
    return np.asarray(gmm.predict(adata, use_rep="X_cellcharter"))


def make_metrics_barplot(res: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart of NMI + ARI per coordinate source (paper-ready)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = res["source"].tolist()
    x = np.arange(len(sources))
    w = 0.38
    # per-method colours consistent with the benchmark figures (see METHOD_COLORS).
    def col(s):
        return METHOD_COLORS.get(s, "#4C78A8")
    fig, ax = plt.subplots(figsize=(1.7 * len(sources) + 2.0, 4.4))
    b_nmi = ax.bar(x - w / 2, res["NMI"], w, label="NMI",
                   color=[col(s) for s in sources], edgecolor="black", linewidth=0.4)
    b_ari = ax.bar(x + w / 2, res["ARI"], w, label="ARI",
                   color=[col(s) for s in sources], edgecolor="black", linewidth=0.4,
                   hatch="//")
    for bars in (b_nmi, b_ari):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=7)
    if float(res[["NMI", "ARI"]].to_numpy().min()) < 0:
        ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=10)
    ax.set_ylabel("agreement with GT tissue regions")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    # legend: solid = NMI, hatched = ARI (colour encodes source, not metric)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", label="NMI"),
                       Patch(facecolor="white", edgecolor="black", hatch="//", label="ARI")],
              frameon=False, fontsize=9, loc="upper right")
    ax.set_title("CellCharter niche recovery vs ground-truth tissue regions (CNS test)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "niche_eval_metrics.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "niche_eval_metrics.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[niche] wrote {out/'niche_eval_metrics.png'} and .pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expr_h5ad", required=True,
                    help="silver cns_luna test AnnData (file or dir of per-section .h5ad)")
    ap.add_argument("--coords_luna", required=True, help="luna_inference/<TS> dir")
    ap.add_argument("--coords_g2t", required=True, help="scgg_inference/<TS> dir")
    ap.add_argument("--coords_celery", default=None, help="celery_inference/<TS> dir (optional)")
    ap.add_argument("--coords_ref", default=None,
                    help="reference coords; default = the AnnData's obsm['spatial']")
    ap.add_argument("--gt_metadata",
                    default="/nfs/team361/sb75/DATASETS/bronze/cns_luna_raw/metadata.csv")
    # column / format knobs
    ap.add_argument("--gt_col", default="Sub_molecular_tissue_region")
    ap.add_argument("--gt_id_col", default="NAME")
    ap.add_argument("--gt_sep", default="auto",
                    help="'auto' sniffs comma vs tab from the header; or force ',' / '\\t'")
    ap.add_argument("--gt_skiprows", default="1",
                    help="rows to skip after the header (SCP 'TYPE' row); '' for none")
    ap.add_argument("--expr_id_col", default="cell_id")
    ap.add_argument("--expr_section_col", default="cell_section")
    ap.add_argument("--coord_x_col", default="coord_X")
    ap.add_argument("--coord_y_col", default="coord_Y")
    # method knobs (identical across all sources)
    ap.add_argument("--n_latent", type=int, default=50, help="PCA dim of the 600-d latent")
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_clusters", type=int, default=0, help="0 = #unique GT regions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot_section", default=None)
    ap.add_argument("--out", default="./niche_eval_out")
    args = ap.parse_args()

    import scanpy as sc
    from sklearn.metrics import (normalized_mutual_info_score as nmi,
                                 adjusted_rand_score as ari)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- 1. expression AnnData (representation + reference coords) ---------
    print(f"[niche] loading expression: {args.expr_h5ad}", flush=True)
    adata = load_expr(args.expr_h5ad, args.expr_id_col, args.expr_section_col)

    # ---- 2. ground-truth region labels (SCP tab file, skip TYPE row) -------
    skip = [int(x) for x in args.gt_skiprows.split(",")] if args.gt_skiprows.strip() else None
    if args.gt_sep == "auto":
        with open(args.gt_metadata) as _f:
            _hdr = _f.readline()
        gt_sep = "\t" if _hdr.count("\t") > _hdr.count(",") else ","
        print(f"[niche] gt_metadata delimiter auto-detected: {gt_sep!r}", flush=True)
    else:
        gt_sep = args.gt_sep
    gt = pd.read_csv(args.gt_metadata, sep=gt_sep, skiprows=skip)
    for c in (args.gt_id_col, args.gt_col):
        if c not in gt.columns:
            raise KeyError(f"gt_metadata needs {c!r}; have {list(gt.columns)[:20]}")
    gt[args.gt_id_col] = gt[args.gt_id_col].astype(str)
    gt_region = gt.drop_duplicates(args.gt_id_col).set_index(args.gt_id_col)[args.gt_col]

    # ---- 3. coordinate sources --------------------------------------------
    coord_sources = {"luna": args.coords_luna, "g2t": args.coords_g2t}
    if args.coords_celery:
        coord_sources["celery"] = args.coords_celery
    preds = {name: load_pred_coords(path, args.coord_x_col, args.coord_y_col, label=name)
             for name, path in coord_sources.items()}
    if args.coords_ref:  # optional external reference; else use obsm['spatial']
        preds["reference"] = load_pred_coords(args.coords_ref, args.coord_x_col,
                                              args.coord_y_col, label="reference")

    # ---- 4. common cells across expr + GT + every predicted source --------
    common = set(adata.obs["_key"]) & set(gt_region.index)
    for c in preds.values():
        common &= set(c.index)
    common = sorted(common)
    if len(common) < 1000:
        raise RuntimeError(
            f"Only {len(common)} cells matched across expression, GT labels and all coord "
            f"sources. Check the id spaces: AnnData obs['{args.expr_id_col}'], "
            f"metadata '{args.gt_id_col}', and metadata_pred.csv index must be the SAME ids.")
    adata = adata[adata.obs["_key"].isin(common)].copy()
    adata.obs_names = adata.obs["_key"].astype(str).values
    adata = adata[common].copy()  # subset + align to the `common` order
    print(f"[niche] {len(common)} cells matched across all sources", flush=True)

    adata.obs["gt_region"] = gt_region.reindex(common).astype("category").values
    K = args.n_clusters or int(adata.obs["gt_region"].nunique())
    print(f"[niche] K={K} (GT has {adata.obs['gt_region'].nunique()} unique "
          f"{args.gt_col!r} regions)", flush=True)

    # ---- 5. shared representation: PCA of the 600-dim latent (once) --------
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    ncomp = min(args.n_latent, X.shape[1], X.shape[0] - 1)
    from sklearn.decomposition import PCA
    adata.obsm["X_rep"] = PCA(n_components=ncomp, random_state=args.seed).fit_transform(X)
    print(f"[niche] shared representation: PCA({ncomp}) of the {X.shape[1]}-dim latent",
          flush=True)

    # reference coords default to the AnnData's imputed obsm['spatial']
    ref_xy = adata.obsm["spatial"].copy()
    order = ["reference", "luna", "g2t"] + (["celery"] if "celery" in preds else [])

    # ---- 6. CellCharter per coordinate source -----------------------------
    rows = []
    labels = pd.DataFrame(index=common)
    labels["section"] = adata.obs[args.expr_section_col].values
    labels["gt_region"] = adata.obs["gt_region"].values
    for name in order:
        if name == "reference" and "reference" not in preds:
            adata.obsm["spatial"] = ref_xy
        else:
            adata.obsm["spatial"] = preds[name].reindex(common)[["x", "y"]].to_numpy(float)
        print(f"[niche] CellCharter on '{name}' coords ...", flush=True)
        niche = run_cellcharter(adata, "X_rep", args.expr_section_col,
                                args.n_layers, K, args.seed)
        labels[f"niche_{name}"] = niche
        rows.append({"source": name, "n_cells": len(common), "n_clusters": K,
                     "NMI": nmi(adata.obs["gt_region"].values, niche),
                     "ARI": ari(adata.obs["gt_region"].values, niche)})
        print(f"[niche]   {name}: NMI={rows[-1]['NMI']:.4f}  ARI={rows[-1]['ARI']:.4f}",
              flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "niche_eval_metrics.csv", index=False)
    labels.to_csv(out / "niche_labels.csv")
    print("\n" + res.to_string(index=False))
    print(f"\n[niche] wrote {out/'niche_eval_metrics.csv'} and {out/'niche_labels.csv'}")

    # ---- 7a. quantitative metrics bar chart (NMI/ARI per source) ----------
    try:
        make_metrics_barplot(res, out)
    except Exception as e:  # never fail the run over a plot; CSV is already saved
        print(f"[niche] metrics plot skipped: {type(e).__name__}: {e}", file=sys.stderr)

    # ---- 7b. niche-map figure (one section) -------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sec = args.plot_section or labels["section"].value_counts().idxmax()
        m = (labels["section"] == sec).to_numpy()
        xy = ref_xy[m]
        cols = ["gt_region"] + [f"niche_{n}" for n in order]
        titles = ["Ground-truth regions"] + [f"Niches: {n}" for n in order]
        fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.6))
        for ax, col, title in zip(np.atleast_1d(axes), cols, titles):
            codes = pd.Categorical(labels.loc[m, col]).codes
            ax.scatter(xy[:, 0], xy[:, 1], c=codes, s=3, cmap="tab20", linewidths=0)
            ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.axis("off")
        fig.suptitle(f"CNS section {sec}: niche recovery vs ground-truth regions", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / "niche_maps.png", dpi=200, bbox_inches="tight")
        print(f"[niche] wrote {out/'niche_maps.png'} (section {sec})")
    except Exception as e:
        print(f"[niche] plot skipped: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
