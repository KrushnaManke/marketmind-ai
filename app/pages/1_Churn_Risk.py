"""Which customers are likely to leave, why, and what a new customer's score would be."""
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import MODELS, RISK_COLORS, ROOT, csv, footer, insight, methodology, money, section, setup, style

setup("Churn Risk", "⚠️", "Prioritize customers using calibrated risk, projected exposure and explainable drivers. Customer Lifetime Value (CLV) estimates expected future revenue.")
df = csv("telco_scored.csv")

# ---- sidebar filters ------------------------------------------------------------
sb = st.sidebar
sb.header("Filters")
only_active = sb.checkbox("Active customers only", value=True,
                          help="Customers who already churned in the data are history, not a call list.")
contracts = sb.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
internet = sb.multiselect("Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
tmin, tmax = int(df["tenure"].min()), int(df["tenure"].max())
tenure = sb.slider("Tenure (months)", tmin, tmax, (tmin, tmax))
bands = sb.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
only_target = sb.checkbox("Only customers worth contacting", value=False)

f = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)
       & df["tenure"].between(*tenure) & df["risk_band"].isin(bands)]
if only_active:
    f = f[f["is_active"]]
if only_target:
    f = f[f["target_flag"]]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers in view", f"{len(f):,}")
k2.metric("Average churn probability", f"{f['churn_prob'].mean():.1%}" if len(f) else "n/a")
k3.metric("Annual revenue at risk", money(f["revenue_at_risk"].sum()))
k4.metric("Worth contacting", f"{int(f['target_flag'].sum()):,}")

insight(
    f"{len(f):,} customers match the active view filters. Prioritize by expected commercial value, not probability alone.",
    "Current view",
)
methodology(
    "Probabilities are calibrated from the held-out customer score. A customer is flagged for contact only when the available "
    "profit estimate exceeds the configured campaign cost. Reason strings are model associations, not proven causes.")

section("Risk distribution", "Probability mix and exposure in the filtered population")
fig = px.histogram(f, x="churn_prob", color="risk_band", nbins=40, color_discrete_map=RISK_COLORS,
                   category_orders={"risk_band": ["Low", "Medium", "High"]}, title="Distribution of churn probability")
fig.update_xaxes(title="Churn probability", tickformat=".0%")
fig.update_yaxes(title="Customers")
st.plotly_chart(style(fig, 320), use_container_width=True)

section("Prioritized customer queue", "Highest-risk customers in this filtered view; the Comma-Separated Values file includes the complete filtered queue")
st.subheader("Highest-risk customers")
cols = ["customerID", "Contract", "InternetService", "tenure", "MonthlyCharges", "churn_prob", "revenue_at_risk", "clv",
        "target_flag", "reason_1", "reason_2", "reason_3"]
top = f.sort_values("churn_prob", ascending=False)[cols]
st.dataframe(top.head(300), hide_index=True, use_container_width=True,
             column_config={
                 "churn_prob": st.column_config.ProgressColumn("Churn prob.", min_value=0.0, max_value=1.0, format="%.2f"),
                 "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.0f"),
                 "clv": st.column_config.NumberColumn("Customer Lifetime Value", format="$%.0f"),
                 "MonthlyCharges": st.column_config.NumberColumn("Monthly", format="$%.2f"),
                 "target_flag": st.column_config.CheckboxColumn("Contact?"),
                 "reason_1": "Top reason", "reason_2": "2nd", "reason_3": "3rd"})
st.download_button("Download this list (Comma-Separated Values)", top.to_csv(index=False).encode(), "high_risk_customers.csv", "text/csv")

section("Customer explanation", "Individual factors, projected loss and expected campaign value")
st.subheader("Why is this customer at risk?")
if len(top):
    pick = st.selectbox("Customer", top["customerID"].head(100).tolist())
    r = f[f["customerID"] == pick].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{r['churn_prob']:.0%}")
    b.metric("Revenue at risk (12m)", money(r["revenue_at_risk"]))
    c.metric("Expected saved profit if contacted", money(r["expected_save_value"]))
    reasons = [r[k] for k in ("reason_1", "reason_2", "reason_3") if isinstance(r[k], str) and r[k]]
    st.write("**Top factors pushing this customer's risk up:** " + (", ".join(f"`{x}`" for x in reasons) or "none stand out"))
else:
    st.info("No customers match the filters.")

# ---- what-if scorer -----------------------------------------------------------------
with st.expander("Score a new / hypothetical customer"):
    @st.cache_resource
    def load_model():
        return joblib.load(MODELS / "churn_model.joblib")

    try:
        bundle = load_model()
        from src.features import add_features
        with st.form("what_if"):
            c1, c2, c3 = st.columns(3)
            contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet_s = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            pay = c1.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))
            tenure_m = c2.slider("Tenure (months)", 0, 72, 6)
            monthly = c2.slider("Monthly charges ($)", 18, 120, 75)
            paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
            support = c3.selectbox("Tech support", ["No", "Yes"])
            security = c3.selectbox("Online security", ["No", "Yes"])
            senior = c3.selectbox("Senior citizen", [0, 1])
            go = st.form_submit_button("Score customer")
        if go:
            row = df.iloc[[0]].copy()
            ns = "No internet service"
            row = row.assign(Contract=contract, InternetService=internet_s, PaymentMethod=pay, tenure=tenure_m,
                             MonthlyCharges=float(monthly), TotalCharges=float(monthly * tenure_m), PaperlessBilling=paperless,
                             TechSupport=ns if internet_s == "No" else support,
                             OnlineSecurity=ns if internet_s == "No" else security, SeniorCitizen=senior)
            p = float(bundle["model"].predict_proba(add_features(row)[bundle["columns"]])[:, 1][0])
            st.metric("Predicted churn probability", f"{p:.0%}")
            st.caption("Other attributes are held at the first customer's values; the score is indicative.")
    except Exception as e:      # model file missing or library mismatch
        st.info(f"Scoring form unavailable ({type(e).__name__}). Re-run `python -m src.churn` in this environment.")
footer()
