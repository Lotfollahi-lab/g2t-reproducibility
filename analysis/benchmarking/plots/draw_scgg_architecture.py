"""draw_scgg_architecture.py

Generate a Nature-journal-style architecture diagram for scgg.
Saves an EDITABLE SVG (text-as-text, not paths) so co-authors can
re-style labels in Illustrator/Inkscape post-export.

Layout: single horizontal flow ~170 mm × 55 mm (full Nature page
width × ~1/4 figure height). The FM ODE callout sits above the
embedding block to show "the network is iteratively applied to
denoise h from Gaussian noise to clean h".

Run:
    source /nfs/team361/sb75/.venvs/scgg/bin/activate
    python analysis/benchmarking/plots/draw_scgg_architecture.py

Outputs (next to this script):
    scgg_architecture.svg
    scgg_architecture.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import _nature_style as ns


# ----------------------------------------------------------------------
# Colour palette — muted, print-friendly, colour-blind-safe pairs
# ----------------------------------------------------------------------
# Phases group the diagram's blocks by role: input data → network →
# decode math → output. A nested dict keeps the fill/edge pair for each
# phase together, so adding or recolouring a phase touches one place.
PHASE = {
    "data":    {"fill": "#EEF3F8", "edge": "#3A86FF"},
    "network": {"fill": "#FFF0F5", "edge": "#FF006E"},
    "decode":  {"fill": "#F0F8F0", "edge": "#06D6A0"},
    "output":  {"fill": "#FFF8E5", "edge": "#FFB703"},
}
COL_ARROW = "#333333"
COL_NOTE = "#666666"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def rounded_box(
    ax, x, y, w, h, label, phase, *,
    subtext=None,
    title_fontsize=7, sub_fontsize=5.5,
    boxstyle="round,pad=0.6,rounding_size=1.2",
    lw=0.8,
):
    """Rounded rectangle with a centred multi-line label.

    ``phase`` selects a fill+edge colour pair from the PHASE palette.
    ``label`` is the heading text; optional ``subtext`` is placed below
    the heading inside the box (italic, smaller, muted colour).
    """
    fill = PHASE[phase]["fill"]
    edge = PHASE[phase]["edge"]
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle, fc=fill, ec=edge, lw=lw,
    ))
    if subtext:
        ax.text(
            x + w / 2, y + h * 0.62, label,
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


def arrow(ax, x1, y1, x2, y2, *, lw=0.9, style="-|>"):
    """Black arrow with adjustable head."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            lw=lw, color=COL_ARROW,
            shrinkA=2, shrinkB=2,
        ),
    )


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
# Figure is sized in mm so the SVG ships at exact print dimensions.
# Coordinates are in millimetres; ax.set_xlim/ylim use the same unit.
# All block + arrow positions live here in one place so it's easy
# to nudge the layout without hunting through draw calls.

FIG_W_MM = 170.0      # full Nature page width
FIG_H_MM = 55.0       # short, leaves room for caption

# Block positions: (x, y, width, height)
B_INPUT  = (5.0,  18.0, 21.0, 18.0)
B_TRANS  = (32.0, 14.0, 30.0, 26.0)
B_HPRED  = (68.0, 18.0, 25.0, 18.0)
B_FMODE  = (68.0, 42.0, 25.0, 10.0)   # callout above HPRED
B_DMAT   = (99.0, 18.0, 22.0, 18.0)
B_MDS    = (127.0, 21.0, 16.0, 12.0)
B_OUTPUT = (149.0, 18.0, 16.0, 18.0)


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
    # 1. Input block — cells × genes matrix
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_INPUT,
        label="Gene expression",
        subtext="N cells × G genes",
        phase="data",
    )

    # ------------------------------------------------------------------
    # 2. Transformer block (the shared substrate with LUNA)
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_TRANS,
        label="Transformer",
        subtext="cell-cell self-attention",
        phase="data",
    )

    # ------------------------------------------------------------------
    # 3. Per-cell embedding h ∈ R^k (the EDM head's prediction)
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_HPRED,
        label=r"Per-cell embedding",
        subtext=r"$\mathbf{h}_i \in \mathbb{R}^k$",
        phase="network",
    )

    # ------------------------------------------------------------------
    # 4. Flow-matching ODE callout above the h-prediction block
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_FMODE,
        label="Flow Matching",
        subtext=r"$\mathbf{h}_t \to \mathbf{h}_0$, Heun ODE (~25 steps)",
        phase="network",
        title_fontsize=6.5,
    )
    # Downward arrow from FM callout INTO the h-prediction block.
    fm_x, fm_y, fm_w, _ = B_FMODE
    h_x, h_y, h_w, h_h = B_HPRED
    arrow(
        ax,
        fm_x + fm_w / 2, fm_y,
        h_x + h_w / 2, h_y + h_h,
    )
    # Tiny "Gaussian noise" annotation pointing into the FM callout.
    ax.text(
        fm_x - 1.0, fm_y + B_FMODE[3] / 2,
        r"$\mathbf{h}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$",
        ha="right", va="center",
        fontsize=5.5, color=COL_NOTE, style="italic",
    )

    # ------------------------------------------------------------------
    # 5. Pairwise distance matrix D
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_DMAT,
        label="Pairwise dist.",
        subtext=r"$D_{ij} = \|\mathbf{h}_i - \mathbf{h}_j\|^2$",
        phase="decode",
    )

    # ------------------------------------------------------------------
    # 6. MDS (classical multi-dim scaling, top-2 eigenvectors)
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_MDS,
        label="MDS",
        subtext="top-2",
        phase="decode",
    )

    # ------------------------------------------------------------------
    # 7. Output coords
    # ------------------------------------------------------------------
    rounded_box(
        ax, *B_OUTPUT,
        label=r"$(\mathbf{x}, \mathbf{y})$",
        subtext="coords",
        phase="output",
    )

    # ------------------------------------------------------------------
    # Inter-block arrows (left-to-right flow)
    # ------------------------------------------------------------------
    def right_arrow(b_from, b_to):
        fx, fy, fw, fh = b_from
        tx, ty, tw, th = b_to
        arrow(
            ax,
            fx + fw, fy + fh / 2,
            tx, ty + th / 2,
        )
    right_arrow(B_INPUT, B_TRANS)
    right_arrow(B_TRANS, B_HPRED)
    right_arrow(B_HPRED, B_DMAT)
    right_arrow(B_DMAT, B_MDS)
    right_arrow(B_MDS, B_OUTPUT)

    # ------------------------------------------------------------------
    # Training-loss annotation under the distance matrix
    # ------------------------------------------------------------------
    dm_x, dm_y, dm_w, _ = B_DMAT
    ax.annotate(
        "Training loss:\nMSE on $D$",
        xy=(dm_x + dm_w / 2, dm_y),
        xytext=(dm_x + dm_w / 2, dm_y - 8),
        ha="center", va="top",
        fontsize=5.5, color=COL_NOTE, style="italic",
        arrowprops=dict(
            arrowstyle="-", lw=0.5, color=COL_NOTE,
            shrinkA=1, shrinkB=1,
        ),
    )

    # ------------------------------------------------------------------
    # Phase labels along the top (gives the reader a mental grouping)
    # ------------------------------------------------------------------
    phase_y = FIG_H_MM - 2.0
    ax.text(
        (B_INPUT[0] + B_TRANS[0] + B_TRANS[2]) / 2, phase_y,
        "Encoder",
        ha="center", va="top",
        fontsize=6, fontweight="bold", color=PHASE["data"]["edge"],
    )
    ax.text(
        B_FMODE[0] + B_FMODE[2] / 2, phase_y,
        "EDM head + Flow Matching",
        ha="center", va="top",
        fontsize=6, fontweight="bold", color=PHASE["network"]["edge"],
    )
    ax.text(
        (B_DMAT[0] + B_MDS[0] + B_MDS[2]) / 2, phase_y,
        "Decode to 2D",
        ha="center", va="top",
        fontsize=6, fontweight="bold", color=PHASE["decode"]["edge"],
    )

    # ------------------------------------------------------------------
    # Save — both SVG (editable text) and PDF (print-ready)
    # ------------------------------------------------------------------
    out_svg = out_dir / "scgg_architecture.svg"
    out_pdf = out_dir / "scgg_architecture.pdf"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"wrote {out_svg}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
