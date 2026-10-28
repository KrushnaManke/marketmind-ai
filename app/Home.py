"""Executive Overview: the one page a manager reads."""
import plotly.express as px
import streamlit as st
from utils import CAT, csv, exists, footer, insight, js, methodology, money, section, setup, style

setup(
    "Executive Overview",
    "📊",
    "A connected view of customer retention, revenue exposure and growth opportunities.",
)

s = js("churn_summary.json")
fc = js("forecast_summary.json")
tel = csv("telco_scored.csv")
act = tel[tel["is_active"]]

insight(
    "The churn and retail programs are intentionally analysed as separate datasets. Results are combined for portfolio-level "
    "monitoring, but no customer-level claim crosses the two datasets.",
    "Scope & interpretation",
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active customers", f"{s['n_active']:,}")
c2.metric("Historical churn rate", f"{s['churn_rate']:.1%}")
c3.metric("Annual revenue at risk", money(s["revenue_at_risk_active"]),
          help="Sum over active customers of churn probability x MonthlyCharges x 12. Probabilities are calibrated.")
c4.metric("Customers worth contacting", f"{s['n_targeted_active']:,}",
          help="Expected-value rule: p x save rate x margin x annual revenue > cost of a retention offer.")
c5.metric("Next-30-day retail revenue", money(fc["next_30d_total"]),
          delta=f"{100 * (fc['next_30d_total'] / fc['last_30d_actual_total'] - 1):+.1f}% vs last 30 days")

section("Portfolio overview", "Key indicators refreshed by the latest pipeline run")
left, right = st.columns(2)
with left:
    g = act.groupby("Contract", as_index=False)["revenue_at_risk"].sum().sort_values("revenue_at_risk", ascending=False)
    fig = px.bar(g, x="Contract", y="revenue_at_risk", text=g["revenue_at_risk"].map(money),
                 color_discrete_sequence=[CAT[0]], title="Where the revenue at risk sits (annual, by contract)")
    fig.update_traces(textposition="outside", marker_cornerradius=4)
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, legend=False), use_container_width=True)
with right:
    if exists("sql/churn_by_tenure_band.csv"):
        t = csv("sql/churn_by_tenure_band.csv")
        fig = px.bar(t, x="tenure_band", y="churn_pct", text=t["churn_pct"].map(lambda v: f"{v:.0f}%"),
                     color_discrete_sequence=[CAT[0]], title="Churn rate by tenure (SQL analysis)")
        fig.update_traces(textposition="outside", marker_cornerradius=4)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        st.plotly_chart(style(fig, legend=False), use_container_width=True)

section("Retention economics", "Commercial value of acting on calibrated churn risk")
pol = csv("churn_policies.csv")
pol["net_benefit_usd"] = pol["net_benefit_usd"].round(0)
best = pol.loc[pol["net_benefit_usd"].idxmax()]
a = js("churn_summary.json")["assumptions"]
st.write(f"On held-out customers, **{best['policy']}** earns the most: net **{money(best['net_benefit_usd'])}**, "
         f"reaching {best['recall']:.0%} of churners with {int(best['customers_contacted']):,} contacts. "
         f"Contacting everyone is {'unprofitable' if pol.loc[pol.policy == 'contact everyone', 'net_benefit_usd'].iloc[0] < 0 else 'less efficient'}.")
show = pol.assign(recall=(100 * pol["recall"]).round(0).astype(int).astype(str) + "%",
                  net_benefit_usd=pol["net_benefit_usd"].map(lambda v: f"${v:,.0f}"))
st.dataframe(show.rename(columns={"policy": "Policy", "customers_contacted": "Contacted", "churners_reached": "Churners reached",
                                  "recall": "Recall", "net_benefit_usd": "Net benefit"}),
             hide_index=True, use_container_width=True)
st.caption(f"Assumptions (edit in `src/config.py`): gross margin {a['gross_margin']:.0%}, offer cost ${a['offer_cost']:.0f}, "
           f"save rate {a['save_rate']:.0%} of contacted churners.")

section("Retail customer base", "Revenue concentration and customer value distribution")
if exists("segment_profile.csv"):
    p = csv("segment_profile.csv")
    long = p.melt(id_vars="segment", value_vars=["customer_share_pct", "revenue_share_pct"],
                  var_name="measure", value_name="pct")
    long["measure"] = long["measure"].map({"customer_share_pct": "% of customers", "revenue_share_pct": "% of revenue"})
    fig = px.bar(long, x="segment", y="pct", color="measure", barmode="group",
                 color_discrete_sequence=[CAT[0], CAT[1]], title="Customer share vs revenue share by segment")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="%")
    st.plotly_chart(style(fig), use_container_width=True)

methodology(
    "Active customers are scored using a calibrated held-out churn model. Revenue at risk equals estimated churn probability "
    "times annualized charges. Segment and revenue figures come from the retail dataset. These are observed and modelled "
    "estimates—not causal effects—and should be validated against the policy and economics of a real campaign.")
footer()
