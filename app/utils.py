"""Shared helpers for every dashboard page: paths, loaders, styling, and report framing."""
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.theme import BLUE_SEQ, CAT, GRID, INK, INK_2, MUTED, RISK_COLORS, SEGMENT_COLORS  # noqa: E402,F401

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DB_PATH = ROOT / "data" / "marketmind.db"


def setup(title: str, icon: str = "📊", subtitle: Optional[str] = None):
    """Configure and frame a dashboard page."""
    st.set_page_config(
        page_title=f"MarketMind AI | {title}", page_icon=icon, layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
      :root { --mm-navy: #102a43; --mm-blue: #256abf; --mm-slate: #64748b; --mm-border: #dfe6ef; }
      html, body, [class*="css"] { font-family: Inter, "Segoe UI", Arial, sans-serif; }
      header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"], [data-testid="stStatusWidget"] { display: none; }
      [data-testid="stAppViewContainer"] { background: #f4f7fb; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2742 0%, #173654 100%);
        border-right: none;
      }
      [data-testid="stSidebar"] * { color: #dce8f5; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #fff; font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 1.02rem; letter-spacing: .01em;
      }
      [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        border-radius: 8px; margin: 2px 8px; padding: 7px 10px; color: #dbe8f5 !important;
      }
      [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(255,255,255,.12); color: #fff !important; font-weight: 700;
      }
      [data-testid="stSidebar"] [data-testid="stElementContainer"] { margin-bottom: .2rem; }
      [data-testid="stMain"] .block-container { background: #fff; border-left: 1px solid #edf1f6; border-right: 1px solid #edf1f6; }
      [data-testid="stMetric"] {
        background: linear-gradient(180deg, #fff 0%, #fbfdff 100%);
        border: 1px solid var(--mm-border); border-top: 3px solid #3f7fbd;
        border-radius: 10px; padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15,42,67,.055); min-height: 118px;
      }
      [data-testid="stMetricLabel"] { color: #5d7186; font-size: .72rem; font-weight: 700; letter-spacing: .045em; text-transform: uppercase; }
      [data-testid="stMetricValue"] { color: var(--mm-navy); font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 1.72rem; }
      [data-testid="stMetricDelta"] { font-weight: 600; }
      [data-testid="stDataFrame"] { border: 1px solid var(--mm-border); border-radius: 8px; overflow: hidden; }
      .block-container { max-width: 1500px; padding-top: 1.35rem; padding-bottom: 2.5rem; }
      [data-testid="stHeading"] a { display: none; }
      [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { margin-bottom: .18rem; }
      [data-testid="stExpander"] { border: 1px solid var(--mm-border); border-radius: 8px; box-shadow: 0 1px 3px rgba(15,42,67,.025); }
      [data-testid="stPlotlyChart"] { border: 1px solid #edf1f6; border-radius: 10px; background: #fff; padding: 6px 8px 0; box-shadow: 0 2px 8px rgba(15,42,67,.035); }
      [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        border-radius: 7px; border: 1px solid #cbd8e6; color: #173654; font-weight: 700;
      }
      [data-testid="stButton"] button[kind="primary"] { background: #256abf; color: #fff; border-color: #256abf; }
      [data-baseweb="input"] { border-radius: 7px; }
      .mm-page-title { margin: 0 0 .12rem 0; font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 2.05rem; line-height: 1.12; letter-spacing: -.035em; color: var(--mm-navy); }
      .mm-page-subtitle { margin: 0 0 .75rem 0; color: var(--mm-slate); font-size: .94rem; }
      .mm-eyebrow { color: var(--mm-blue); font-weight: 700; font-size: .69rem; letter-spacing: .14em; text-transform: uppercase; }
      .mm-section { margin: 1.35rem 0 .45rem; color: var(--mm-navy); font-family: Inter, "Segoe UI", Arial, sans-serif; font-weight: 700; font-size: 1.18rem; }
      .mm-method { border-left: 3px solid var(--mm-blue); background: #f5f8fc; padding: .85rem 1.1rem; border-radius: 6px; }
      .mm-footer { border-top: 1px solid #dfe6ef; margin-top: 2.5rem; padding-top: 1rem; color: #7d8ba0; font-size: .78rem; }
      .mm-source-real { color: #047857; } .mm-source-demo { color: #a65f14; }
      div[data-testid="stAlert"] { border-radius: 8px; }
      [data-testid="stException"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    if not (PROC / "churn_summary.json").exists():
        st.error("No results found. Run `python -m src.pipeline` first (see README).")
        st.stop()
    srcs = js("data_sources.json")
    synthetic = [k for k in ("telco", "retail", "reviews") if srcs.get(k) == "synthetic"]
    status = "Mixed-source" if "telco" in synthetic or "retail" in synthetic or "reviews" in synthetic else "Validated run"
    source_class = "mm-source-demo" if synthetic else "mm-source-real"
    st.markdown(f'<div class="mm-eyebrow">MarketMind Intelligence · {status}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="mm-page-title">{icon} {title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="mm-page-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span class="{source_class}" style="font-size:.78rem;font-weight:700">● {status}</span>'
        '<span style="color:#94a3b8;font-size:.78rem">Interactive decision-support workspace</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)


def section(title: str, description: Optional[str] = None):
    st.markdown(f'<div class="mm-section">{title}</div>', unsafe_allow_html=True)
    if description:
        st.caption(description)


def insight(text: str, label: str = "Executive insight"):
    st.markdown(
        f'<div class="mm-method"><strong>{label}</strong><br>{text}</div>', unsafe_allow_html=True
    )


def methodology(text: str, title: str = "Methodology & caveats"):
    with st.expander(title):
        st.markdown(text)


def footer():
    sources = js("data_sources.json")
    synthetic = [name for name in ("telco", "retail", "reviews") if sources.get(name) == "synthetic"]
    demo_note = (
        f" · Demo data: {', '.join(synthetic)} · Replace raw files and rerun the pipeline for real findings."
        if synthetic else ""
    )
    st.markdown(
        '<div class="mm-footer">MarketMind AI · Analytics generated from the project pipeline. '
        f'Insights are decision support, not guaranteed outcomes.{demo_note}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(PROC / name, **kw)


@st.cache_data
def js(name: str) -> dict:
    return json.loads((PROC / name).read_text())


def exists(name: str) -> bool:
    return (PROC / name).exists()


def money(x: float, decimals: int = 0) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e4:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.{decimals}f}"


def style(fig, height: int = 380, legend: bool = True):
    """Recessive grid, thin marks, legend on top for >= 2 series, transparent background."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=56, b=10), showlegend=legend,
        font=dict(color=INK_2, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        title=dict(x=0, font=dict(size=15, color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
