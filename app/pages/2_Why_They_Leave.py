"""Why are customers leaving? Model drivers + SQL breakdowns + review themes."""
import plotly.express as px
import streamlit as st
from utils import CAT, csv, exists, footer, insight, js, methodology, section, setup, style

setup("Why They Leave", "🔎", "Understand model associations and operational risks without overstating causal conclusions. SHAP means SHapley Additive exPlanations.")
m = js("churn_metrics.json")
imp = csv("churn_importance.csv")
method = imp["method"].iloc[0]
st.caption({"shap_tree": "Explanations: SHapley Additive exPlanations TreeExplainer on the best tree model.",
            "logistic_contributions": "Explanations: exact additive contributions of the logistic-regression model "
                                      "(install `shap` + `xgboost` to switch to SHAP on a tree model). SHAP means SHapley Additive exPlanations."}[method])

c1, c2 = st.columns(2)
with c1:
    g = imp.groupby("base_feature", as_index=False)["importance"].sum().nlargest(12, "importance").sort_values("importance")
    fig = px.bar(g, x="importance", y="base_feature", orientation="h", color_discrete_sequence=[CAT[0]],
                 title="Global drivers of churn score")
    fig.update_traces(marker_cornerradius=4)
    fig.update_xaxes(title="Mean |contribution|")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 460, legend=False), use_container_width=True)
with c2:
    if exists("churn_permutation_importance.csv"):
        p = csv("churn_permutation_importance.csv").head(12).sort_values("perm_importance")
        fig = px.bar(p, x="perm_importance", y="feature", orientation="h", color_discrete_sequence=[CAT[0]],
                     title=f"Cross-check: permutation importance ({m['best_model'].replace('_', ' ')})")
        fig.update_traces(marker_cornerradius=4)
        fig.update_xaxes(title="Drop in Precision-Recall Area Under the Curve when shuffled")
        fig.update_yaxes(title="")
        st.plotly_chart(style(fig, 460, legend=False), use_container_width=True)

section("Specific model patterns", "Encoded feature values and their relative global contribution")
top = imp.nlargest(14, "importance").copy()
top["effect"] = top["direction_when_present_or_high"]
st.dataframe(top[["encoded_feature", "importance", "effect"]].rename(columns={"encoded_feature": "Feature = value"}),
             hide_index=True, use_container_width=True,
             column_config={"importance": st.column_config.NumberColumn("Importance", format="%.2f"), "effect": "Effect"})

section("Observed churn by customer segment", "Raw Structured Query Language summaries provide context, not causal proof")
tabs = st.tabs(["By contract", "By payment method", "By internet + tech support"])
spec = [("churn_by_contract.csv", "Contract"), ("churn_by_payment_method.csv", "PaymentMethod")]
for tab, (name, col) in zip(tabs[:2], spec):
    with tab:
        if exists(f"sql/{name}"):
            d = csv(f"sql/{name}")
            fig = px.bar(d, x=col, y="churn_pct", text=d["churn_pct"].map(lambda v: f"{v:.0f}%"), color_discrete_sequence=[CAT[0]])
            fig.update_traces(textposition="outside", marker_cornerradius=4)
            fig.update_xaxes(title="")
            fig.update_yaxes(title="Churn %")
            st.plotly_chart(style(fig, 320, legend=False), use_container_width=True)
with tabs[2]:
    if exists("sql/churn_by_internet_and_support.csv"):
        d = csv("sql/churn_by_internet_and_support.csv")
        d["TechSupport"] = d["TechSupport"].map({"Yes": "With tech support", "No": "No tech support"})
        fig = px.bar(d, x="InternetService", y="churn_pct", color="TechSupport", barmode="group",
                     color_discrete_sequence=[CAT[0], CAT[1]])
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Churn %")
        st.plotly_chart(style(fig, 320), use_container_width=True)

section("Customer feedback themes", "Negative review topics from a separate review dataset")
st.caption("Reviews come from a separate dataset and are NOT linked to these customers. The themes show which topics dominate "
           "negative feedback; they do not prove why a particular Telco customer left.")
if exists("sentiment_themes.csv"):
    t = csv("sentiment_themes.csv").sort_values("negative_reviews")
    fig = px.bar(t, x="negative_reviews", y="theme", orientation="h", color_discrete_sequence=[CAT[1]],
                 title="Negative reviews mentioning each theme")
    fig.update_traces(marker_cornerradius=4)
    fig.update_xaxes(title="Reviews")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 320, legend=False), use_container_width=True)

section("Candidate retention playbook", "Test hypotheses and local economics before changing policy")
PLAYBOOK = {
    "Contract": "Offer 12-month incentives (discount, free add-on) during the first 90 days; month-to-month is the highest-risk plan.",
    "tenure": "Run an onboarding programme: check-in calls at months 1, 3 and 6. Early-tenure customers churn most.",
    "MonthlyCharges": "Review price/value for high-bill customers; offer bundles rather than blanket discounts.",
    "TechSupport": "Bundle or trial tech support / online security for at-risk customers without them.",
    "OnlineSecurity": "Bundle or trial tech support / online security for at-risk customers without them.",
    "PaymentMethod": "Nudge electronic-check payers towards automatic payment with a small credit.",
    "InternetService": "Investigate fibre service quality and pricing; fibre customers show elevated churn.",
    "num_addons": "Cross-sell add-on services: more services means stickier customers.",
}
shown = 0
for feat in g.sort_values("importance", ascending=False)["base_feature"]:
    if feat in PLAYBOOK and shown < 5:
        st.markdown(f"- **Test {feat}**: {PLAYBOOK[feat]}")
        shown += 1
methodology(
    "Feature importance shows predictive association on the selected model. SQL breakdowns are descriptive. Retention actions "
    "remain hypotheses until tested with an experiment or accepted as policy based on the organization's domain knowledge.")
footer()
