"""Shared Nature-journal matplotlib style.

Used by every plot/diagram script in this directory so font, size, and
fontype settings stay in one place. Bump a single value here and every
downstream figure picks it up on next render — no copy-paste drift.

Usage:
    import _nature_style as ns
    ns.apply()
    fig, ax = plt.subplots(figsize=(170 * ns.mm, 55 * ns.mm))
"""

from __future__ import annotations

import matplotlib as mpl


# Millimetre-to-inch conversion. matplotlib's figsize is in inches, but
# Nature specifies print sizes in millimetres (full-width = 180 mm,
# column = 89 mm), so authors think in mm. Multiply by ``mm`` to convert.
mm = 1 / 25.4


# Single source of truth for the Nature-journal style. Includes axis +
# tick settings even for scripts that hide their axes — harmless when
# unused, avoids surprise gaps when a future caller does draw axes.
_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,        # embedded TrueType (Nature submission req)
    "ps.fonttype": 42,
    "svg.fonttype": "none",    # text stays as <text> elements (editable)
    "figure.dpi": 300,
    "savefig.dpi": 300,
}


def apply() -> None:
    """Install the Nature-journal rcParams globally."""
    mpl.rcParams.update(_RCPARAMS)
