"""Shared plot style for the FieldVote paper figures.

One place for the palette, fonts and axis defaults so every figure in the
paper reads as part of the same system. Figures are drawn at their final
printed size (IEEE two-column: 3.45 in per column, 7.16 in full width) and
included with no LaTeX rescaling, so the type sizes here are the type sizes
on the page.
"""

import matplotlib
import matplotlib.pyplot as plt

# IEEEtran conference geometry, in inches.
COL = 3.45
FULL = 7.16

# Taylor Geospatial palette.
C_FV = "#D9422A"  # FieldVote / ours
C_TUNED = "#5A7AB8"  # tuned watershed / uniform vote
C_DEL = "#A9C4CE"  # DelineateAnything (weakest baseline)
C_BROWN = "#3B1E1C"  # text, secondary series
C_GREY = "#8C8C86"  # de-emphasised rules and losses
C_GRID = "#D8D8D0"


def use():
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.labelcolor": C_BROWN,
        "text.color": C_BROWN,
        "xtick.color": C_BROWN,
        "ytick.color": C_BROWN,
        "axes.edgecolor": C_BROWN,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": C_GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderpad": 0.0,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })


def nice(name):
    """belgium -> Belgium, south_africa -> South Africa."""
    return name.replace("_", " ").title()
