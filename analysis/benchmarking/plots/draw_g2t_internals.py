"""draw_g2t_internals.py

Compact internal-architecture diagram for the G2T (scgg) model.
Designed as panel (c) of the "G2T Overview" figure — sized to slot
alongside the Training (a) and Inference (b) flow panels.

Shows only what's INSIDE the G2T box: from gene expression input to
the per-cell embedding h that feeds the pairwise distance matrix
(which is already drawn in panels a/b). Stops at the EDM head's
linear projection — the distance matrix / MDS / Procrustes happen
downstream in the surrounding flow panels.

Run:
    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python analysis/benchmarking/plots/draw_g2t_internals.py

Outputs (next to this script):
    g2t_internals.svg
    g2t_internals.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import _nature_style as ns


# Shared phase palette with the other G2T figures
PHASE = {
    "data":    {"fill": "#EEF3F8", "edge": "#3A86FF"},
    "network": {"fill": "#FFF0F5", "edge": "#FF006E"},
    "decode":  {"fill": "#F0F8F0", "edge": "#06D6A0"},
    "output":  {"fill": "#FFF8E5", "edge": "#FFB703"},
}
COL_ARROW = "#333333"
COL_NOTE = "#666666"


def rounded_box(
    ax, x, y, w, h, label, phase, *,
    subtext=None,
    title_fontsize=6.5, sub_fontsize=5.0,
    boxstyle="round,pad=0.35,rounding_size=0.8",
    lw=0.7,
):
    """Centred labelled rounded box with optional subtext below the label."""
    fill = PHASE[phase]["fill"]
    edge = PHASE[phase]["edge"]
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle, fc=fill, ec=edge, lw=lw,
    ))
    if subtext:
        ax.text(
            x + w / 2, y + h * 0.66, label,
            ha="center", va="center",
            fontsize=title_fontsize, fontweight="medium",
        )
        ax.text(
            x + w / 2, y + h * 0.30, subtext,
            ha="center", va="center",
            fontsize=sub_fontsize, color=COL_NOTE, style="italic",
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=title_fontsize, fontweight="medium",
        )


def shape_arrow(ax, x, y_top, y_bot, *, shape_label, lw=0.75):
    """Vertical arrow with a small tensor-shape label to the right."""
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw, color=COL_ARROW,
            shrinkA=1, shrinkB=1,
        ),
    )
    ax.text(
        x + 1.2, (y_top + y_bot) / 2, shape_label,
        ha="left", va="center",
        fontsize=5.0, color=COL_NOTE, style="italic",
        family="monospace",
    )


# Layout — compact single column to fit as a sub-panel. Auxiliary
# inputs (diffusion time t, noisy positions x_t) are NOT shown as
# separate boxes — they'd cross-cut the main flow with extra arrows.
# Instead the transformer's subtext mentions the conditioning,
# keeping the diagram tall+thin and easy to read top-to-bottom.
FIG_W_MM = 75.0
FIG_H_MM = 120.0

COL_X = 12.0
COL_W = 46.0
BOX_H = 10.0

# Per-box Y positions (top of each box). Top → bottom, tightly packed.
Y_INPUT  = 104.0
Y_INMLP  = 88.0
Y_TRANS  = 48.0    # taller block (32 mm), goes up to y=80
Y_OUTMLP = 32.0
Y_EDM    = 18.0
Y_HOUT   = 4.0     # the h embedding output (feeds into panel a/b)


def main(out_dir: Path | None = None) -> int:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ns.apply()

    fig, ax = plt.subplots(figsize=(FIG_W_MM * ns.mm, FIG_H_MM * ns.mm))
    ax.set_xlim(0, FIG_W_MM)
    ax.set_ylim(0, FIG_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title ----
    ax.text(
        FIG_W_MM / 2, FIG_H_MM - 4,
        "G2T internals",
        ha="center", va="top",
        fontsize=8, fontweight="bold",
    )

    # ---- 1. Gene expression input ----
    rounded_box(
        ax, COL_X, Y_INPUT, COL_W, BOX_H,
        label="Gene expression",
        subtext="(N, G)",
        phase="data",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y_INPUT, Y_INMLP + BOX_H,
                shape_label="(N, G)")

    # ---- 2. Input MLP ----
    rounded_box(
        ax, COL_X, Y_INMLP, COL_W, BOX_H,
        label="Input MLP",
        subtext="G → 256 → 256, ReLU",
        phase="data",
    )
    # Arrow into the top of the transformer block (Y_TRANS + trans_h).
    shape_arrow(ax, COL_X + COL_W / 2, Y_INMLP, Y_TRANS + 32,
                shape_label="(N, 256)")

    # ---- 3. Transformer ×8 stack (visualised as overlapping rectangles).
    # Conditioning inputs (diffusion time t, noisy positions x_t)
    # are mentioned in the subtext rather than drawn as separate
    # boxes — keeps the column clean and arrow-free.
    trans_h = 32.0
    for off_x, off_y, alpha in [(1.2, -1.2, 0.45), (0.6, -0.6, 0.7)]:
        ax.add_patch(patches.FancyBboxPatch(
            (COL_X + off_x, Y_TRANS + off_y), COL_W, trans_h,
            boxstyle="round,pad=0.35,rounding_size=0.8",
            fc=PHASE["data"]["fill"], ec=PHASE["data"]["edge"],
            lw=0.4, alpha=alpha,
        ))
    rounded_box(
        ax, COL_X, Y_TRANS, COL_W, trans_h,
        label="Transformer  ×8",
        subtext=("16-head linear self-attention\n"
                 "+ FFN(256→256→256) + LN + residuals\n"
                 r"conditioned on $t$ and noisy $\mathbf{x}_t$"),
        phase="data",
        title_fontsize=6.5, sub_fontsize=5.0,
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y_TRANS, Y_OUTMLP + BOX_H,
                shape_label="(N, 256)")

    # ---- 5. Output bottleneck MLP ----
    rounded_box(
        ax, COL_X, Y_OUTMLP, COL_W, BOX_H,
        label="Output MLP",
        subtext="256 → 256 → 32",
        phase="data",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y_OUTMLP, Y_EDM + BOX_H,
                shape_label="(N, 32)")

    # ---- 6. EDM head linear projection (the only EDM-head piece
    # that lives INSIDE G2T; the distance matrix + MDS + Procrustes
    # live in panels a/b's surrounding flow). ----
    rounded_box(
        ax, COL_X, Y_EDM, COL_W, BOX_H,
        label="EDM head",
        subtext=r"Linear  32 → $k$ (= 8)",
        phase="network",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y_EDM, Y_HOUT + BOX_H,
                shape_label=r"h: (N, k)")

    # ---- 7. Output: per-cell embedding h (feeds the pairwise
    # distance matrix in panels a/b) ----
    rounded_box(
        ax, COL_X, Y_HOUT, COL_W, BOX_H,
        label=r"Per-cell embedding  $\mathbf{h}$",
        subtext="→ pairwise distance matrix",
        phase="network",
    )

    # ---- Save ----
    out_svg = out_dir / "g2t_internals.svg"
    out_pdf = out_dir / "g2t_internals.pdf"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"wrote {out_svg}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
