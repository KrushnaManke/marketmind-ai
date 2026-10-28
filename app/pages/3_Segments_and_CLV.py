"""Retail customer segments (RFM + K-Means / DBSCAN), cohorts and lifetime value."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import CAT, SEGMENT_COLORS, csv, exists, footer, insight, js, methodology, money, section, setup, style

setup("Segments & Customer Lifetime Value", "🧩", "Turn customer behavior into actionable groups and 90-day value estimates. RFM means Recency, Frequency, Monetary.")
seg = js("segmentation_summary.json")
rfm = csv("rfm_segments.csv")
prof = csv("segment_profile.csv")
order = list(SEGMENT_COLORS)

st.caption(f"{seg['n_customers']:,} retail customers with an ID. RFM (Recency, Frequency, Monetary) values are log-transformed and standardised "
           f"before clustering. The retail data has **no churn label**, so 'lapsed' is a derived flag: no purchase in the last "
           f"{seg['lapsed_definition_days']} days ({seg['lapsed_rate']:.0%} of customers).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Segments (K-Means clustering)", seg["k"], help=f"Silhouette at k={seg['k']}: {seg['silhouette_at_k']:.2f}; "
          f"best silhouette was at k={seg['best_silhouette_k']}. k=4 was chosen for interpretability.")
k2.metric("Lapsed customers", f"{seg['lapsed_rate']:.0%}")
k3.metric("Density-Based Spatial Clustering outliers", f"{seg['dbscan']['outliers']:,}", help="DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.")
k4.metric("Top-1% customers' revenue share", f"{seg['dbscan']['whale_revenue_share_pct']:.0f}%")

insight(
    f"Champions and high-value segments should receive differentiated service, while lapsed and at-risk groups require a "
    f"measured re-engagement strategy. DBSCAN identifies {seg['dbscan']['outliers']} unusual customers.",
    "Segmentation recommendation",
)
section("Segment profiles", "Comparative customer, spend and inactivity characteristics")
show = prof.set_index("segment").loc[[s for s in order if s in set(prof["segment"])]].reset_index()
st.dataframe(show, hide_index=True, use_container_width=True, column_config={
    "segment": "Segment", "customers": "Customers", "avg_recency_days": st.column_config.NumberColumn("Avg recency (days)", format="%.0f"),
    "avg_orders": st.column_config.NumberColumn("Avg orders", format="%.1f"),
    "avg_spend": st.column_config.NumberColumn("Avg spend", format="$%.0f"),
    "total_revenue": st.column_config.NumberColumn("Total revenue", format="$%.0f"),
    "lapsed_pct": st.column_config.NumberColumn("Lapsed %", format="%.0f"),
    "customer_share_pct": st.column_config.NumberColumn("% customers", format="%.1f"),
    "revenue_share_pct": st.column_config.NumberColumn("% revenue", format="%.1f")})

sample = rfm.sample(min(len(rfm), 2000), random_state=1).copy()
sample["log10 orders"] = np.log10(sample["Frequency"])
sample["log10 revenue"] = np.log10(sample["Monetary"])
fig = px.scatter(sample, x="log10 orders", y="log10 revenue", color="segment", facet_col="segment", facet_col_wrap=2,
                 color_discrete_map=SEGMENT_COLORS, category_orders={"segment": order}, opacity=0.7,
                 title="Customers by segment (one panel per segment, so colours never overlap)")
fig.update_traces(marker=dict(size=5))
st.plotly_chart(style(fig, 560, legend=False), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    d = csv("kmeans_diagnostics.csv")
    fig = px.line(d, x="k", y="silhouette", markers=True, color_discrete_sequence=[CAT[0]], title="Choosing k: silhouette score")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 300, legend=False), use_container_width=True)
with c2:
    fig = px.line(d, x="k", y="inertia", markers=True, color_discrete_sequence=[CAT[0]], title="Choosing k: elbow (inertia)")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 300, legend=False), use_container_width=True)
st.caption(f"DBSCAN—Density-Based Spatial Clustering of Applications with Noise—(eps={seg['dbscan']['eps']:.2f}, min_samples={seg['dbscan']['min_samples']}) flags unusual customers "
           f"that K-Means would force into a cluster; the top 1% by spend ({seg['dbscan']['whales_p99']} customers) are the 'whales'.")

if exists("sql/retail_cohort_retention.csv"):
    section("Cohort retention", "Percentage of each acquisition cohort still purchasing")
    co = csv("sql/retail_cohort_retention.csv")
    piv = co.pivot(index="cohort_month", columns="months_since", values="retention_pct")
    fig = px.imshow(piv, aspect="auto", color_continuous_scale=["#cde2fb", "#6da7ec", "#256abf", "#0d366b"], zmin=0, zmax=100,
                    labels=dict(x="Months since first purchase", y="Cohort", color="% active"),
                    title="Share of each monthly cohort still buying")
    st.plotly_chart(style(fig, 420, legend=False), use_container_width=True)

section("Customer lifetime value", "Next-90-day spend estimate with a non-look-ahead test")
meta = js("retail_clv_metrics.json")
tbl = pd.DataFrame(meta["metrics"]).T.reset_index().rename(columns={"index": "Model", "top_decile_revenue_capture": "Top-decile revenue captured"})
st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
    "MAE": st.column_config.NumberColumn("Mean Absolute Error", format="$%.0f"),
    "RMSE": st.column_config.NumberColumn("Root Mean Squared Error", format="$%.0f"),
    "Top-decile revenue captured": st.column_config.NumberColumn(format="%.2f")})
st.caption(f"Trained on features up to {meta['cutoff']}, tested on the following {meta['window_days']} days (no look-ahead). "
           f"Selected by Mean Absolute Error: **{meta['best_model']}**. If the naive run-rate wins on Root Mean Squared Error, that is because a few very large "
           f"customers dominate squared error. Seasonality caveat: the training window and the scoring window are different seasons.")
clv = csv("retail_clv.csv").merge(rfm[["CustomerID", "segment"]], on="CustomerID", how="left")
a, b = st.columns(2)
a.metric("Predicted next-90-day revenue (all customers)", money(clv["predicted_90d_value"].sum()))
b.metric("Share from top 10% of customers", f"{clv['predicted_90d_value'].nlargest(len(clv) // 10).sum() / clv['predicted_90d_value'].sum():.0%}")
st.dataframe(clv.nlargest(25, "predicted_90d_value")[["CustomerID", "segment", "recency_days", "frequency", "monetary", "predicted_90d_value"]],
             hide_index=True, use_container_width=True, column_config={
                 "monetary": st.column_config.NumberColumn("Spend to date", format="$%.0f"),
                 "predicted_90d_value": st.column_config.NumberColumn("Predicted 90-day value", format="$%.0f")})
methodology(
    f"The Customer Lifetime Value model is trained before {meta['cutoff']} and validated on the following {meta['window_days']} days. "
    "The value is an estimate of future spend—not net profit or a guaranteed customer commitment. Segment labels reflect "
    "transaction behavior and are descriptive rather than churn outcomes.")
footer()
