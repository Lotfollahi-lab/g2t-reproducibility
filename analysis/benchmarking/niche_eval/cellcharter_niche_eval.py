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
    obs['cell_id'] = the join id (e.g. "well06_0"), obs['cell_section'], and a
    per-section row position in obs['_bronze_row_pos'] when available. Only
    *_test.h5ad files are loaded (train objects may use another id scheme).
  * metadata.csv (Single Cell Portal format): a second "TYPE" header row to
    skip; id column "NAME"; label "Sub_molecular_tissue_region".
  * metadata_pred.csv / metadata_true.csv: index = a per-section INTEGER row
    position (NOT a biological id); columns cell_class, coord_X, coord_Y. The
    section is the parent folder name; the integer is mapped back to a cell_id
    through the test AnnData (obs['_bronze_row_pos'] if present, else the
    positional 0..n-1 order).

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


def _read_xy(path, x_col, y_col) -> pd.DataFrame:
    """Read a metadata_{pred,true}.csv -> DataFrame with columns [x, y] and the
    file's own (integer, per-section) index preserved."""
    df = pd.read_csv(path, index_col=0)
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"{c!r} not in {path} (cols: {list(df.columns)})")
    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]
    return out


def _section_of(path: Path, known) -> "str | None":
    """Map a metadata_pred.csv path to a test-section stem. Inference folders are
    named ``<section>_<tile>`` (e.g. ``well01OB_0``, ``well1_5_0``), so an exact
    path-component match is tried first, then a trailing ``_<int>`` tile suffix is
    peeled off progressively until a known section matches (returning on the
    first/longest hit, so ``well1_5_0`` -> ``well1_5``, not ``well1``)."""
    import re
    parts = list(path.parts)
    hits = [s for s in known if s in set(parts)]
    if hits:
        return max(hits, key=len)
    for comp in reversed(parts):                       # deepest folder first
        c = comp
        while True:
            if c in known:
                return c
            m = re.match(r"^(.*)_\d+$", c)
            if not m:
                break
            c = m.group(1)
    return None


def _map_ints(entry, ints):
    """Map a section's metadata_pred integer index -> cell_ids. The index is the
    section-local position 0..n-1 (verified: well1_5 pred index 0..19869 == its
    19870 cells), so positional ``ids[i]`` is primary; ``_bronze_row_pos`` is only
    a fallback for runs that happened to write the global bronze position."""
    ids = entry["ids"]; n = len(ids); bronze = entry["bronze"]
    ints = np.asarray(ints, dtype=int)
    if ints.size and ints.min() >= 0 and ints.max() < n:
        return [ids[int(i)] for i in ints], "positional"
    if bronze is not None:
        mapped = [bronze.get(int(i)) for i in ints]
        if sum(v is not None for v in mapped) >= 0.5 * max(len(mapped), 1):
            return mapped, "_bronze_row_pos"
    return [ids[int(i)] if 0 <= int(i) < n else None for i in ints], "positional(clipped)"


def _resolve_str_ids(index, name2id):
    """Reconcile a string-indexed pred file to biological cell_ids. CeLEry writes
    the AnnData obs_names (``{cell_id}_{section}``, e.g. ``well01OB_0_well01OB``),
    so we map obs_name -> cell_id; ids that are already cell_ids pass through."""
    strs = [str(v) for v in index]
    if name2id is None:
        return strs, "cell_id"
    valid = set(name2id.values())
    return ([v if v in valid else name2id.get(v) for v in strs], "obs_name->cell_id")


def _axis_corr(cids, T, spatial_by_id):
    """Scale-free per-axis |corr| between mapped true coords T and the AnnData
    reference coords, under the best axis pairing (min-max norm can swap axes but
    not rotate). ~1.0 confirms the row->cell_id mapping is not scrambled."""
    keep = [(c, T[j]) for j, c in enumerate(cids) if c is not None and c in spatial_by_id]
    if len(keep) < 10:
        return None
    Tk = np.array([v for _, v in keep], float)
    S = np.array([spatial_by_id[c] for c, _ in keep], float)
    cxx = abs(np.corrcoef(Tk[:, 0], S[:, 0])[0, 1])
    cyy = abs(np.corrcoef(Tk[:, 1], S[:, 1])[0, 1])
    cxy = abs(np.corrcoef(Tk[:, 0], S[:, 1])[0, 1])
    cyx = abs(np.corrcoef(Tk[:, 1], S[:, 0])[0, 1])
    return max(min(cxx, cyy), min(cxy, cyx))


def load_test_expr(path: str, id_col: str, section_col: str):
    """Load ONLY ``*_test.h5ad`` (train objects may use a different id scheme and
    are not needed). Records, per cell, its section (from the file name) and the
    per-section integer row position that ``metadata_pred.csv`` is indexed by
    (``obs['_bronze_row_pos']`` if present, else positional 0..n-1).

    Returns ``(adata, sec2map, name2id)``: ``sec2map[section]`` bridges a (folder,
    integer) pair back to a cell id (positional / _bronze_row_pos), and
    ``name2id`` maps obs_names -> cell_id for sources (e.g. CeLEry) that index by
    the ``{cell_id}_{section}`` obs_name instead of the section-local integer."""
    import anndata as ad
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*_test.h5ad")))
        if not files:
            files = sorted(glob.glob(str(p / "*.h5ad")))
            if not files:
                raise FileNotFoundError(f"No *_test.h5ad under {p}")
            print(f"[niche] WARNING: no *_test.h5ad matched; falling back to all "
                  f"{len(files)} .h5ad in {p}", file=sys.stderr)
    else:
        files = [str(p)]

    parts, sec2map = [], {}
    for f in files:
        a = ad.read_h5ad(f)
        sec = Path(f).name
        for suf in ("_test.h5ad", ".h5ad"):
            if sec.endswith(suf):
                sec = sec[: -len(suf)]
                break
        if id_col not in a.obs:
            raise KeyError(f"obs[{id_col!r}] missing in {f} (have: {list(a.obs.columns)})")
        if "spatial" not in a.obsm:
            raise KeyError(f"obsm['spatial'] (reference coords) missing in {f}")
        n = a.n_obs
        a.obs["_section"] = sec
        a.obs["_key"] = a.obs[id_col].astype(str).values
        ids = a.obs["_key"].to_numpy()               # positional order (0..n-1)
        bronze = None
        if "_bronze_row_pos" in a.obs:
            rp = pd.to_numeric(a.obs["_bronze_row_pos"], errors="coerce").to_numpy()
            if not np.isnan(rp).any():
                bronze = dict(zip(rp.astype(int).tolist(), ids.tolist()))
        sec2map[sec] = {"ids": ids, "bronze": bronze}
        parts.append(a)
        print(f"[niche]   {Path(f).name}: section={sec!r}, {n} cells "
              f"(pred index expected 0..{n-1}; _bronze_row_pos "
              f"{'available' if bronze is not None else 'absent'})", flush=True)

    adata = parts[0] if len(parts) == 1 else ad.concat(parts, join="outer", merge="same")
    if section_col not in adata.obs:
        adata.obs[section_col] = adata.obs["_section"].values
    # obs_name -> cell_id (built BEFORE main overwrites obs_names with cell_id)
    name2id = dict(zip(adata.obs_names.astype(str), adata.obs["_key"].astype(str)))
    print(f"[niche] loaded {adata.n_obs} test cells across {len(sec2map)} sections",
          flush=True)
    return adata, sec2map, name2id


def load_pred_coords_by_section(path: str, sec2map, x_col: str, y_col: str,
                                label: str = "", spatial_by_id=None,
                                name2id=None) -> pd.DataFrame:
    """Directory holding one ``metadata_pred.csv`` per section (at any depth).
    The section is read from the folder name and the file's integer index is
    mapped back to a biological ``cell_id`` via ``sec2map``. (If a file is already
    indexed by non-numeric cell ids, it is used as-is.) Returns a DataFrame
    indexed by ``cell_id`` with columns [x, y].

    When ``spatial_by_id`` is given, each section's ``metadata_true.csv`` is
    cross-checked against the AnnData reference coords with a scale-free per-axis
    correlation — a wrong (but complete) integer->cell_id mapping would scramble
    coordinates while still "matching" every id, so this guard is what catches it."""
    p = Path(path)
    files = [str(p)] if p.is_file() else sorted(
        glob.glob(str(p / "**" / "metadata_pred.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No metadata_pred.csv under {p} (run inference first?)")
    known = set(sec2map)
    frames, n_unmapped, secs_seen, verify, hows = [], 0, [], [], set()
    for f in files:
        fp = Path(f)
        xy = _read_xy(f, x_col, y_col)
        idx_num = pd.to_numeric(xy.index, errors="coerce")
        if idx_num.notna().all():                    # per-section integer index
            sec = _section_of(fp, known)
            if sec is None:
                print(f"[niche] WARNING: no test section matches {f} "
                      f"(parent={fp.parent.name!r}); skipping", file=sys.stderr)
                continue
            entry = sec2map[sec]
            ints = idx_num.astype(int).to_numpy()
            ids_list, how = _map_ints(entry, ints)
            hows.add(how)
            ok = np.array([v is not None for v in ids_list])
            n_unmapped += int((~ok).sum())
            sub = xy.iloc[ok].copy()
            sub.index = pd.Index([v for v, k in zip(ids_list, ok) if k], name="cell_id")
            secs_seen.append(sec)
            if spatial_by_id is not None:            # sanity-check vs metadata_true
                tf = fp.with_name("metadata_true.csv")
                if tf.exists():
                    try:
                        t = _read_xy(tf, x_col, y_col)
                        tnum = pd.to_numeric(t.index, errors="coerce")
                        tk = tnum.notna().to_numpy()
                        tids, _ = _map_ints(entry, tnum[tk].astype(int).to_numpy())
                        v = _axis_corr(tids, t.iloc[tk][["x", "y"]].to_numpy(float),
                                       spatial_by_id)
                        if v is not None:
                            verify.append(v)
                    except Exception:
                        pass
        else:                                        # string-indexed (e.g. CeLEry obs_names)
            ids_list, how = _resolve_str_ids(xy.index, name2id)
            hows.add(how)
            ok = np.array([v is not None for v in ids_list])
            n_unmapped += int((~ok).sum())
            sub = xy.iloc[ok].copy()
            sub.index = pd.Index([v for v, k in zip(ids_list, ok) if k], name="cell_id")
            sec = _section_of(fp, known)
            if sec is not None:
                secs_seen.append(sec)
            if spatial_by_id is not None:            # sanity-check vs metadata_true
                tf = fp.with_name("metadata_true.csv")
                if tf.exists():
                    try:
                        t = _read_xy(tf, x_col, y_col)
                        tids, _ = _resolve_str_ids(t.index, name2id)
                        v = _axis_corr(tids, t[["x", "y"]].to_numpy(float), spatial_by_id)
                        if v is not None:
                            verify.append(v)
                    except Exception:
                        pass
        frames.append(sub)
    if not frames:
        raise RuntimeError(f"No metadata_pred.csv could be matched to a test "
                           f"section under {p} (section folder names must match "
                           f"the *_test.h5ad stems).")
    df = pd.concat(frames)
    n_raw = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dup = n_raw - len(df)
    msg = (f"[niche]   {label or path}: {len(files)} file(s), "
           f"{len(set(secs_seen)) or 'n/a'} sections, {len(df)} cells mapped"
           f" [{'/'.join(sorted(hows)) or 'id'}]")
    if n_unmapped:
        msg += f"  (! {n_unmapped} rows had no cell_id in the test AnnData — dropped)"
    if dup:
        msg += f"  (! {dup} duplicate cell_ids dropped)"
    print(msg, flush=True)
    if verify:
        v = float(np.mean(verify))
        ok = v > 0.9
        print(f"[niche]   {label or path}: mapping check vs metadata_true "
              f"(per-axis |corr| with AnnData spatial) = {v:.3f} "
              f"[{'OK' if ok else 'LOW'}]"
              + ("" if ok else "  <-- integer->cell_id mapping may be misaligned, OR "
                               "obsm['spatial'] is imputed rather than the true test "
                               "coords; inspect before trusting the metrics"),
              file=sys.stderr)
    return df


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

    # ---- 1. TEST expression AnnData (representation + reference coords) ----
    print(f"[niche] loading TEST expression: {args.expr_h5ad}", flush=True)
    adata, sec2map, name2id = load_test_expr(args.expr_h5ad, args.expr_id_col,
                                             args.expr_section_col)
    _sp = np.asarray(adata.obsm["spatial"], dtype=float)
    spatial_by_id = {k: _sp[i] for i, k in enumerate(adata.obs["_key"].astype(str).values)}

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
    preds = {name: load_pred_coords_by_section(path, sec2map, args.coord_x_col,
                                               args.coord_y_col, label=name,
                                               spatial_by_id=spatial_by_id,
                                               name2id=name2id)
             for name, path in coord_sources.items()}
    if args.coords_ref:  # optional external reference; else use obsm['spatial']
        preds["reference"] = load_pred_coords_by_section(
            args.coords_ref, sec2map, args.coord_x_col, args.coord_y_col,
            label="reference", spatial_by_id=spatial_by_id, name2id=name2id)

    # ---- 4. common cells across expr + GT + every predicted source --------
    # Any section absent from ANY source is dropped for ALL (the comparison is
    # only meaningful where every method produced coordinates). For this CNS run
    # g2t covers 14 of the 18 test sections (the sagittal*/spinalcord sections
    # were not generated), so those cells fall out here automatically.
    aid = set(adata.obs["_key"].astype(str))
    gid = set(gt_region.index.astype(str))
    id2sec = dict(zip(adata.obs["_key"].astype(str), adata.obs["_section"].astype(str)))
    _nsec = lambda ids: len({id2sec[i] for i in ids if i in id2sec})
    print(f"[niche] coverage: expr(test)={adata.n_obs} cells / {_nsec(aid)} sections; "
          f"GT∩expr={len(aid & gid)} cells", flush=True)
    for name, c in preds.items():
        pid = set(c.index.astype(str)) & aid
        print(f"[niche]   {name}: {len(pid)} cells / {_nsec(pid)} sections", flush=True)
    common = aid & gid
    for c in preds.values():
        common &= set(c.index.astype(str))
    common = sorted(common)
    if len(common) < 1000:
        import itertools
        ex = lambda s: list(itertools.islice(sorted(s), 5))
        print("[niche] DIAGNOSTIC — id spaces do not line up:", file=sys.stderr)
        print(f"  AnnData obs['{args.expr_id_col}'] (n={len(aid)}) e.g. {ex(aid)}", file=sys.stderr)
        print(f"  AnnData obs_names                 e.g. {list(map(str, adata.obs_names[:5]))}", file=sys.stderr)
        print(f"  GT '{args.gt_id_col}' (n={len(gid)}) e.g. {ex(gid)}"
              f"   | cell_id∩GT = {len(aid & gid)}", file=sys.stderr)
        for name, c in preds.items():
            pid = set(c.index.astype(str))
            print(f"  {name} pred cell_id (n={len(pid)}) e.g. {ex(pid)}"
                  f"   | cell_id∩{name} = {len(aid & pid)}", file=sys.stderr)
        raise RuntimeError(
            f"Only {len(common)} cells matched. See the DIAGNOSTIC above — the id spaces "
            f"(AnnData obs['{args.expr_id_col}'], metadata '{args.gt_id_col}', metadata_pred "
            f"index) must share the same cell ids.")
    adata = adata[adata.obs["_key"].isin(common)].copy()
    adata.obs_names = adata.obs["_key"].astype(str).values
    adata = adata[common].copy()  # subset + align to the `common` order
    surv = sorted({id2sec[i] for i in common})
    print(f"[niche] {len(common)} cells matched across all sources, "
          f"spanning {len(surv)} sections: {surv}", flush=True)

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
