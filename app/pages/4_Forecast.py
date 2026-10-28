"""What might happen next month? Revenue forecast with honest backtests."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import CAT, INK_2, csv, exists, footer, insight, js, methodology, money, section, setup, style

setup("Revenue Forecast", "📈", "Compare forecast candidates and plan ranges with explicit backtest evidence. Error is measured with Weighted Absolute Percentage Error (WAPE).")
fc = js("forecast_summary.json")
hist = csv("forecast_history.csv", parse_dates=["date"])
fut = csv("forecast_future.csv", parse_dates=["date"])
bt = csv("forecast_backtest_last_fold.csv", parse_dates=["date"])
cmp_ = csv("forecast_model_comparison.csv")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Next 30 days (forecast)", money(fc["next_30d_total"]),
          delta=f"{100 * (fc['next_30d_total'] / fc['last_30d_actual_total'] - 1):+.1f}% vs last 30 days")
k2.metric("Last 30 days (actual)", money(fc["last_30d_actual_total"]))
k3.metric("Winning model", fc["best_model"].replace("_", " "))
k4.metric("Backtest error (Weighted Absolute Percentage Error)", f"{fc['backtest_wape_best_pct']:.0f}%",
          help="Weighted absolute % error of DAILY predictions over three rolling 28-day holdouts. Robust to zero-sales days.")

insight(
    f"The selected {fc['best_model'].replace('_', ' ')} model forecasts {money(fc['next_30d_total'])} over the next 30 days, "
    f"{'above' if fc['next_30d_total'] >= fc['last_30d_actual_total'] else 'below'} the prior 30-day actual.",
    "Forward outlook",
)
section("Forecast trajectory", "Actual history, held-out backtest and 80% empirical interval")
c1, c2 = st.columns([1, 3])
window = c1.selectbox("History shown", ["Last 60 days", "Last 120 days", "All"], index=1)
show_bt = c1.checkbox("Show backtest", value=True)
n = {"Last 60 days": 60, "Last 120 days": 120, "All": len(hist)}[window]
h = hist.tail(n)

fig = go.Figure()
fig.add_trace(go.Scatter(x=h["date"], y=h["revenue"], name="Actual", mode="lines", line=dict(color=INK_2, width=1.5)))
if show_bt:
    b = bt[bt["model"] == fc["best_model"]]
    fig.add_trace(go.Scatter(x=b["date"], y=b["predicted"], name="Backtest (held out)", mode="lines", line=dict(color=CAT[2], width=2)))
fig.add_trace(go.Scatter(x=pd.concat([fut["date"], fut["date"][::-1]]), y=pd.concat([fut["upper_80"], fut["lower_80"][::-1]]),
                         fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), name="80% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast"], name="Forecast", mode="lines", line=dict(color=CAT[0], width=2.5)))
fig.update_layout(title="Daily revenue: history, backtest and 30-day forecast", hovermode="x unified")
fig.update_yaxes(title="Revenue")
c2.plotly_chart(style(fig, 440), use_container_width=True)

section("Model comparison", "Rolling-origin backtest across 3 folds × 28 days")
st.dataframe(cmp_.rename(columns={"model": "Model", "WAPE_pct": "Weighted Absolute Percentage Error %", "total_error_pct": "28-day total error %"}),
             hide_index=True, use_container_width=True,
             column_config={"MAE": st.column_config.NumberColumn("Mean Absolute Error", format="%.0f"),
                            "RMSE": st.column_config.NumberColumn("Root Mean Squared Error", format="%.0f"),
                            "Weighted Absolute Percentage Error %": st.column_config.NumberColumn(format="%.1f"),
                            "28-day total error %": st.column_config.NumberColumn(format="%.1f")})
st.caption(f"Models tried: {', '.join(fc['models_tried'])}. "
           + ("" if fc["uses_statsmodels"] else "Install `statsmodels` to add Holt-Winters and Seasonal Auto-Regressive Integrated Moving Average with eXogenous regressors (SARIMAX); ")
           + ("" if fc["uses_prophet"] else "install `prophet` to add Prophet. ")
           + "The interval is the 10th-90th percentile of pooled backtest residuals added to the forecast.")

if exists("forecast_monthly.csv"):
    m = csv("forecast_monthly.csv", parse_dates=["month"])
    fig = go.Figure(go.Bar(x=m["month"], y=m["revenue"], marker=dict(color=CAT[0], cornerradius=4), name="Monthly revenue"))
    fig.update_layout(title="Monthly revenue (complete months only)")
    st.plotly_chart(style(fig, 320, legend=False), use_container_width=True)

st.info(fc["caveat"] + (f" Augmented Dickey-Fuller stationarity p-value of the daily series: {fc['adf_pvalue']:.3f}." if fc.get("adf_pvalue") is not None else ""))
methodology(
    "The 80% interval is built from pooled empirical residual quantiles in backtesting; it is an operating range, not a "
    "validated predictive interval with coverage guarantees. Weighted Absolute Percentage Error (WAPE) is calculated across three rolling 28-day holdouts. "
    "Forecast totals do not include plan-adjusted assumptions or known future campaigns.")
footer()
