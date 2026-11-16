"""Common figure style.

The report is typeset with roffin.cls, whose text block is 30pc = 5.0in wide.  Figures are drawn
at exactly that width (FIGW) so that they are included at scale 1 and the font sizes chosen here
(8pt labels) are the ones that appear in print.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C

FIGW = 5.0                # inches = \\textwidth of the report class
COL = {"M1": "#1b6ca8", "M2": "#c0392b", "M3": "#2e8b57", "IRB": "#555555"}
LAB = {"M1": "M1 empirical", "M2": "M2 Gaussian", "M3": "M3 $t$-copula"}

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.linewidth": 0.6,
                     "legend.frameon": False, "figure.constrained_layout.use": True, "figure.dpi": 120, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})


def save(fig, name):
    """Write the figure as PDF (for the report) and PNG (for a quick look)."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.pdf")
    fig.savefig(C.FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
