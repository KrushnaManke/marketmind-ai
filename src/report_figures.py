"""Static PNG figures for the README (the interactive versions live in the Streamlit app)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, PROC
from .theme import CAT, INK_2, MUTED, SEGMENT_COLORS, model_color, mpl_style


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130)
    plt.close(fig)


def churn_curves():
    c = pd.read_csv(PROC / "churn_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kind, title, xl, yl in ((axes[0], "roc", "ROC curve (test set)", "False positive rate", "True positive rate"),
                                    (axes[1], "pr", "Precision-recall curve (test set)", "Recall", "Precision")):
        for m, g in c[c.kind == kind].groupby("model"):
            ax.plot(g.x, g.y, color=model_color(m), label=m.replace("_", " "))
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.set(title=title, xlabel=xl, ylabel=yl)
        ax.legend(loc="lower right" if kind == "roc" else "upper right")
    _save(fig, "churn_roc_pr.png")


def churn_importance():
    imp = pd.read_csv(PROC / "churn_importance.csv")
    g = imp.groupby("base_feature")["importance"].sum().nlargest(10).sort_values()
    method = imp["method"].iloc[0].replace("_", " ")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(g.index, g.values, color=CAT[0], height=0.6)
    ax.set(title=f"Why customers leave: top drivers ({method})", xlabel="Mean absolute contribution to churn score")
    ax.grid(axis="y", visible=False)
    _save(fig, "churn_drivers.png")


def churn_gain():
    g = pd.read_csv(PROC / "churn_gain_table.csv")
    x = np.r_[0, g.decile * 10]
    y = np.r_[0, g.cum_churners_captured_pct]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(x, y, color=CAT[0], marker="o", ms=5, label="Model")
    ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1, label="Random targeting")
    ax.set(title="Cumulative gain: churners captured", xlabel="% of customers contacted (highest risk first)", ylabel="% of churners captured")
    ax.legend(loc="lower right")
    _save(fig, "churn_gain.png")


def revenue_at_risk():
    d = pd.read_csv(PROC / "telco_scored.csv")
    d = d[d.is_active]
    g = d.groupby("Contract")["revenue_at_risk"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(g.index, g.values / 1e3, color=CAT[0], width=0.55)
    ax.bar_label(bars, labels=[f"${v / 1e3:,.0f}k" for v in g.values], color=INK_2, padding=3)
    ax.set(title="Annual revenue at risk by contract (active customers)", ylabel="$ thousands")
    ax.grid(axis="x", visible=False)
    _save(fig, "revenue_at_risk.png")


def segments():
    r = pd.read_csv(PROC / "rfm_segments.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, (seg, col) in zip(axes.ravel(), SEGMENT_COLORS.items()):
        ax.scatter(np.log10(r.Frequency), np.log10(r.Monetary), s=6, color=MUTED, alpha=.25, linewidths=0)
        s = r[r.segment == seg]
        ax.scatter(np.log10(s.Frequency), np.log10(s.Monetary), s=9, color=col, alpha=.8, linewidths=0)
        ax.set_title(f"{seg} (n={len(s):,})")
    fig.supxlabel("log10 orders (frequency)", color=INK_2)
    fig.supylabel("log10 revenue (monetary)", color=INK_2)
    _save(fig, "segments_rfm.png")


def forecast():
    h = pd.read_csv(PROC / "forecast_history.csv", parse_dates=["date"]).tail(120)
    f = pd.read_csv(PROC / "forecast_future.csv", parse_dates=["date"])
    bt = pd.read_csv(PROC / "forecast_backtest_last_fold.csv", parse_dates=["date"])
    best = f["model"].iloc[0]
    bt = bt[bt.model == best]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(h.date, h.revenue, color=INK_2, lw=1.2, label="Actual daily revenue")
    ax.plot(bt.date, bt.predicted, color=CAT[2], lw=1.6, label="Backtest (held-out last fold)")
    ax.fill_between(f.date, f.lower_80, f.upper_80, color=CAT[0], alpha=.2, lw=0, label="80% interval (empirical)")
    ax.plot(f.date, f.forecast, color=CAT[0], lw=2, label=f"Forecast: {best.replace('_', ' ')}")
    ax.set(title="Daily revenue: history, backtest and 30-day forecast", ylabel="Revenue")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, "forecast.png")


def sentiment_terms():
    t = pd.read_csv(PROC / "sentiment_top_terms.csv")
    pos, neg = t[t.sentiment == "positive"].head(10), t[t.sentiment == "negative"].head(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(list(pos.term)[::-1], list(pos.weight)[::-1], color=CAT[0], height=.6, label="Positive")
    ax.barh(list(neg.term), list(neg.weight), color=CAT[1], height=.6, label="Negative")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(title="Most informative review terms", xlabel="Logistic-regression weight")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, "sentiment_terms.png")


def main():
    mpl_style()
    for fn in (churn_curves, churn_importance, churn_gain, revenue_at_risk, segments, forecast, sentiment_terms):
        fn()
        print("  figure:", fn.__name__)


if __name__ == "__main__":
    main()
