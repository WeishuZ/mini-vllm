"""Shared matplotlib styling — a dark 'terminal' theme so the plots match the
project's aesthetic and read well on dark and light backgrounds alike."""
from __future__ import annotations

BG = "#0b0e14"
FG = "#c9d1d9"
MUTED = "#6b7689"
GRID = "#1c2230"
# accent palette
GREEN = "#3fb950"
CYAN = "#39c5cf"
AMBER = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"


def apply():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "monospace",
        "font.size": 10,
        "figure.dpi": 130,
    })
    return plt
