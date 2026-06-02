"""draw_scgg_layers.py

Vertical layer-by-layer architecture diagram for scgg / G2T. Detailed
view: each module is a labelled box, tensor shapes are annotated on
the arrows between boxes, and the EDM head's four substages are
grouped inside a coloured frame to mark the novel decoder.

Companion to ``draw_scgg_architecture.py`` (which gives the high-level
horizontal overview). This one is the methods-section's "Fig 1c" or
"Extended Data Fig 1a" — single-column tall figure showing the full
forward pass shape-by-shape.

Run:
    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python analysis/benchmarking/plots/draw_scgg_layers.py

Outputs (next to this script):
    scgg_layers.svg
    scgg_layers.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import _nature_style as ns


# ----------------------------------------------------------------------
# Colour palette — shares the phase-keyed dict from the horizontal
# diagram so the two figures look like a matched pair.
# ----------------------------------------------------------------------
PHASE = {
    "data":    {"fill": "#EEF3F8", "edge": "#3A86FF"},
    "network": {"fill": "#FFF0F5", "edge": "#FF006E"},
    "decode":  {"fill": "#F0F8F0", "edge": "#06D6A0"},
    "output":  {"fill": "#FFF8E5", "edge": "#FFB703"},
}
COL_ARROW = "#333333"
COL_NOTE = "#666666"
COL_EDM_FRAME = "#FFC0D8"   # soft pink frame around the EDM head group


# ----------------------------------------------------------------------
# Helpers (mirrors draw_scgg_architecture.py — kept private here to
# avoid coupling the two scripts; one-off plotting code is allowed to
# duplicate trivial drawing primitives across files).
# ----------------------------------------------------------------------

def rounded_box(
    ax, x, y, w, h, label, phase, *,
    subtext=None,
    title_fontsize=6.5, sub_fontsize=5.5,
    boxstyle="round,pad=0.4,rounding_size=1.0",
    lw=0.7,
):
    """Centred labelled rounded box. Subtext is rendered below the label."""
    fill = PHASE[phase]["fill"]
    edge = PHASE[phase]["edge"]
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle, fc=fill, ec=edge, lw=lw,
    ))
    if subtext:
        ax.text(
            x + w / 2, y + h * 0.65, label,
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


def shape_arrow(ax, x, y_top, y_bot, *, shape_label, lw=0.8):
    """Vertical arrow with the tensor shape printed beside it."""
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw, color=COL_ARROW,
            shrinkA=1.5, shrinkB=1.5,
        ),
    )
    # Shape label to the right of the arrow midpoint.
    ax.text(
        x + 1.5, (y_top + y_bot) / 2, shape_label,
        ha="left", va="center",
        fontsize=5.5, color=COL_NOTE, style="italic",
        family="monospace",
    )


# ----------------------------------------------------------------------
# Layout — single tall column of boxes.
# ----------------------------------------------------------------------
# Width is set so this fits as a single-column figure (≤89 mm Nature
# column). Height is generous because we're showing 10+ stages.
FIG_W_MM = 100.0
FIG_H_MM = 230.0

# Common geometry
COL_X = 22.0          # left edge of the main column
COL_W = 56.0          # box width
BOX_H = 11.0          # height of a standard box
GAP   = 5.5           # space between consecutive boxes (= where shape labels go)

# Per-box Y positions (top of each box). Layout flows top-to-bottom.
Y = {
    "input":      210.0,
    "input_mlp":  192.0,
    "time_mlp":   174.0,    # rendered side-by-side with pos_mlp
    "pos_mlp":    174.0,
    "transformer": 152.0,   # taller block, ×8 stack
    "out_mlp":    126.0,
    # EDM head (4 substages inside a coloured frame)
    "edm_proj":    98.0,
    "edm_dist":    80.0,
    "edm_mds":     62.0,
    "edm_align":   44.0,
    # Output
    "output":      22.0,
}


def main(out_dir: Path | None = None) -> int:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ns.apply()

    fig, ax = plt.subplots(figsize=(FIG_W_MM * ns.mm, FIG_H_MM * ns.mm))
    ax.set_xlim(0, FIG_W_MM)
    ax.set_ylim(0, FIG_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # 1. Input — gene expression
    # ------------------------------------------------------------------
    rounded_box(
        ax, COL_X, Y["input"], COL_W, BOX_H,
        label="Gene expression",
        subtext="silver h5ad input",
        phase="data",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y["input"], Y["input_mlp"] + BOX_H,
                shape_label="(N, G)")

    # ------------------------------------------------------------------
    # 2. Input MLP
    # ------------------------------------------------------------------
    rounded_box(
        ax, COL_X, Y["input_mlp"], COL_W, BOX_H,
        label="Input MLP",
        subtext="G → 256 → 256, ReLU",
        phase="data",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y["input_mlp"], Y["transformer"] + 22,
                shape_label="(N, 256)")

    # ------------------------------------------------------------------
    # 3. Time MLP + Position MLP (side-by-side at same y)
    # ------------------------------------------------------------------
    # Move them slightly off-axis: time MLP at left, pos MLP at right.
    # Both feed into the transformer alongside the node features.
    half_w = COL_W * 0.46
    gap_mid = (COL_W - 2 * half_w) / 2
    time_x = COL_X + gap_mid
    pos_x  = COL_X + COL_W - half_w - gap_mid
    rounded_box(
        ax, time_x, Y["time_mlp"], half_w, BOX_H,
        label="Time MLP",
        subtext="t ∈ (0,1)\n1→256→1",
        phase="data",
        title_fontsize=6, sub_fontsize=5,
    )
    rounded_box(
        ax, pos_x, Y["pos_mlp"], half_w, BOX_H,
        label="Position MLP",
        subtext="x_t ∈ R²\nnorm-aware",
        phase="data",
        title_fontsize=6, sub_fontsize=5,
    )
    # Small arrows from each into the transformer block
    ax.annotate(
        "", xy=(COL_X + COL_W * 0.30, Y["transformer"] + 22),
        xytext=(time_x + half_w / 2, Y["time_mlp"]),
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=COL_ARROW,
                        shrinkA=1, shrinkB=1),
    )
    ax.annotate(
        "", xy=(COL_X + COL_W * 0.70, Y["transformer"] + 22),
        xytext=(pos_x + half_w / 2, Y["pos_mlp"]),
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=COL_ARROW,
                        shrinkA=1, shrinkB=1),
    )

    # ------------------------------------------------------------------
    # 4. 8× Transformer block
    # ------------------------------------------------------------------
    # Render as a stack-of-three-rectangles to suggest depth, with the
    # frontmost rectangle holding the label. The ×8 multiplier appears
    # in the subtext.
    trans_x, trans_y, trans_w, trans_h = COL_X, Y["transformer"], COL_W, 22.0
    # Back layers (faded)
    for offset_x, offset_y, alpha in [(1.4, -1.4, 0.45), (0.7, -0.7, 0.7)]:
        ax.add_patch(patches.FancyBboxPatch(
            (trans_x + offset_x, trans_y + offset_y), trans_w, trans_h,
            boxstyle="round,pad=0.4,rounding_size=1.0",
            fc=PHASE["data"]["fill"], ec=PHASE["data"]["edge"],
            lw=0.5, alpha=alpha,
        ))
    # Front layer (full opacity, holds the label)
    rounded_box(
        ax, trans_x, trans_y, trans_w, trans_h,
        label="Transformer  ×8",
        subtext=("self-attention (16 heads) +\n"
                 "FFN(256→256→256) + LN + residuals"),
        phase="data",
        title_fontsize=7,
    )
    shape_arrow(ax, COL_X + COL_W / 2, trans_y, Y["out_mlp"] + BOX_H,
                shape_label="(N, 256)")

    # ------------------------------------------------------------------
    # 5. Output projection (256 → 32 bottleneck)
    # ------------------------------------------------------------------
    rounded_box(
        ax, COL_X, Y["out_mlp"], COL_W, BOX_H,
        label="Output MLP",
        subtext="256 → 256 → 32",
        phase="data",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y["out_mlp"], Y["edm_proj"] + BOX_H,
                shape_label="(N, 32)")

    # ------------------------------------------------------------------
    # 6-9. EDM HEAD GROUP — drawn inside a soft pink frame
    # ------------------------------------------------------------------
    # Frame: encloses all 4 substages with a 3 mm pad.
    edm_top = Y["edm_proj"] + BOX_H + 4
    edm_bot = Y["edm_align"] - 4
    frame_x = COL_X - 3
    frame_w = COL_W + 6
    ax.add_patch(patches.FancyBboxPatch(
        (frame_x, edm_bot), frame_w, edm_top - edm_bot,
        boxstyle="round,pad=0.6,rounding_size=2.0",
        fc="none", ec=COL_EDM_FRAME, lw=1.2,
        linestyle=(0, (4, 2)),   # dashed
    ))
    # EDM head label at the top-left of the frame.
    ax.text(
        frame_x + 1.5, edm_top - 1.5,
        "EDM head",
        ha="left", va="top",
        fontsize=6.5, fontweight="bold", color=PHASE["network"]["edge"],
        style="italic",
    )

    # 6. Linear 32 → 8
    rounded_box(
        ax, COL_X, Y["edm_proj"], COL_W, BOX_H,
        label="Linear projection",
        subtext="32 → k (= 8)",
        phase="network",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y["edm_proj"], Y["edm_dist"] + BOX_H,
                shape_label=r"h: (N, k)")

    # 7. Pairwise distance matrix
    rounded_box(
        ax, COL_X, Y["edm_dist"], COL_W, BOX_H,
        label="Pairwise distance",
        subtext=r"$D_{ij}=\|\mathbf{h}_i - \mathbf{h}_j\|^2$",
        phase="network",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y["edm_dist"], Y["edm_mds"] + BOX_H,
                shape_label="D: (N, N)")

    # 8. Classical MDS
    rounded_box(
        ax, COL_X, Y["edm_mds"], COL_W, BOX_H,
        label="Classical MDS",
        subtext="top-2 eigenpairs of −½ J·D·J",
        phase="decode",
    )
    shape_arrow(ax, COL_X + COL_W / 2, Y["edm_mds"], Y["edm_align"] + BOX_H,
                shape_label="(N, 2)")

    # 9. Procrustes alignment to x_t
    rounded_box(
        ax, COL_X, Y["edm_align"], COL_W, BOX_H,
        label="Procrustes alignment",
        subtext=r"rotate to $\mathbf{x}_t$ frame (SVD, detached)",
        phase="decode",
    )

    # Arrow from end of EDM frame down to output
    shape_arrow(ax, COL_X + COL_W / 2, Y["edm_align"], Y["output"] + BOX_H,
                shape_label="pred.positions")

    # ------------------------------------------------------------------
    # 10. Output coords
    # ------------------------------------------------------------------
    rounded_box(
        ax, COL_X, Y["output"], COL_W, BOX_H,
        label=r"$(\mathbf{x}, \mathbf{y})$  coordinates",
        subtext="(N, 2) per slice",
        phase="output",
    )

    # ------------------------------------------------------------------
    # Side annotations:
    #   (a) Flow Matching ODE — vertical bracket on the right covering
    #       the whole forward pass, indicating "iterate this 25-50 times"
    #   (b) Training loss — pointer from the EDM frame to a side note
    # ------------------------------------------------------------------
    # (a) FM bracket on the RIGHT side
    bracket_x = COL_X + COL_W + 9
    bracket_top = Y["input_mlp"] + BOX_H
    bracket_bot = Y["edm_align"]
    # Vertical line
    ax.plot([bracket_x, bracket_x], [bracket_bot, bracket_top],
            color=PHASE["network"]["edge"], lw=1.0)
    # Top + bottom tick marks
    ax.plot([bracket_x, bracket_x + 1.5], [bracket_top, bracket_top],
            color=PHASE["network"]["edge"], lw=1.0)
    ax.plot([bracket_x, bracket_x + 1.5], [bracket_bot, bracket_bot],
            color=PHASE["network"]["edge"], lw=1.0)
    # Label rotated 90° alongside the bracket
    ax.text(
        bracket_x + 3, (bracket_top + bracket_bot) / 2,
        "Flow Matching ODE\nHeun, ~25 steps from\n"
        r"$\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$",
        ha="left", va="center",
        fontsize=6, color=PHASE["network"]["edge"], fontweight="medium",
    )

    # (b) Training loss callout — pointed from the EDM frame
    loss_x = COL_X - 5
    loss_y = (Y["edm_dist"] + Y["edm_mds"]) / 2 + BOX_H / 2
    ax.annotate(
        "Training loss:\n" + r"$\mathrm{MSE}\bigl(\sqrt{D^{\mathrm{pred}}},$" +
        "\n" + r"$\|\mathbf{x}^{\mathrm{true}}_i-\mathbf{x}^{\mathrm{true}}_j\|\bigr)$",
        xy=(COL_X, loss_y),
        xytext=(loss_x, loss_y),
        ha="right", va="center",
        fontsize=5.5, color=COL_NOTE, style="italic",
        arrowprops=dict(
            arrowstyle="-", lw=0.5, color=COL_NOTE,
            shrinkA=2, shrinkB=2,
        ),
    )

    # ------------------------------------------------------------------
    # Top title
    # ------------------------------------------------------------------
    ax.text(
        FIG_W_MM / 2, FIG_H_MM - 4,
        "scgg / G2T — layer-by-layer forward pass",
        ha="center", va="top",
        fontsize=8, fontweight="bold",
    )
    ax.text(
        FIG_W_MM / 2, FIG_H_MM - 9,
        r"$N$ = cells per slice,  $G$ = genes,  $k$ = EDM embed dim (=8)",
        ha="center", va="top",
        fontsize=5.5, color=COL_NOTE, style="italic",
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_svg = out_dir / "scgg_layers.svg"
    out_pdf = out_dir / "scgg_layers.pdf"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"wrote {out_svg}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
