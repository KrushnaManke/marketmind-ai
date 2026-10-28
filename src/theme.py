"""Shared chart palette (validated categorical slots, fixed order - never cycled).

Slots 1-4 of the reference palette. Colour follows the ENTITY, not its rank: a given
segment / model keeps its colour wherever it appears, whatever the filter state.
Dashboard (Plotly) and static report figures (matplotlib) both read from here.
"""
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # blue, orange, aqua, yellow
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]                # Low / Medium / High risk (ordered)
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

SEGMENT_COLORS = {"Champions": CAT[0], "Promising": CAT[2], "At Risk": CAT[1], "Hibernating": CAT[3]}
RISK_COLORS = {"Low": ORDINAL_3[0], "Medium": ORDINAL_3[1], "High": ORDINAL_3[2]}
MODEL_COLORS = {}      # filled lazily so each model keeps one colour everywhere


def model_color(name: str) -> str:
    if name not in MODEL_COLORS:
        MODEL_COLORS[name] = CAT[len(MODEL_COLORS) % len(CAT)]
    return MODEL_COLORS[name]


def mpl_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 110,
    })
