"""Customer lifetime value.

Telco   : contract-level hazard (events / exposure) -> discounted expected revenue.
Retail  : supervised regression - predict each customer's spend in the next 90 days
          from RFM-style features measured BEFORE a cutoff date (no look-ahead).
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .config import (ANNUAL_DISCOUNT_RATE, CLV_HORIZON_MONTHS, CLV_WINDOW_DAYS, PROC, SEED)
from .data_sources import clean_retail_sales, load_retail


# --------------------------------------------------------------- Telco ----------
def telco_clv(df: pd.DataFrame) -> pd.Series:
    """Revenue CLV = sum_{t=1..H} MonthlyCharges * S^t / (1+d)^t,  S = 1 - monthly hazard.

    Hazard per contract type = churn events / customer-months of exposure
    (the maximum-likelihood estimate for an exponential lifetime with censoring).
    """
    g = df.groupby("Contract").agg(events=("Churn", "sum"), exposure=("tenure", "sum"), n=("Churn", "size"))
    g["hazard"] = (g["events"] + 0.5) / (g["exposure"] + g["n"])      # +0.5 / +n: avoids a zero hazard
    d_m = (1 + ANNUAL_DISCOUNT_RATE) ** (1 / 12) - 1
    r = (1 - g["hazard"]) / (1 + d_m)
    annuity = r * (1 - r ** CLV_HORIZON_MONTHS) / (1 - r)             # geometric series
    out = df["MonthlyCharges"] * df["Contract"].map(annuity)
    out.name = "clv"
    return out


# -------------------------------------------------------------- Retail ----------
def _features_at(sales: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    h = sales[sales["InvoiceDate"] <= asof]
    inv = h.groupby(["CustomerID", "InvoiceNo"]).agg(date=("InvoiceDate", "max"), rev=("Revenue", "sum")).reset_index()
    g = inv.groupby("CustomerID")
    f = pd.DataFrame({
        "recency_days": (asof - g["date"].max()).dt.days,
        "frequency": g["InvoiceNo"].nunique(),
        "monetary": g["rev"].sum(),
        "avg_order_value": g["rev"].mean(),
        "customer_age_days": (asof - g["date"].min()).dt.days,
    })
    last90 = inv[inv["date"] > asof - pd.Timedelta(days=90)].groupby("CustomerID")
    f["orders_last_90d"] = last90["InvoiceNo"].nunique().reindex(f.index).fillna(0)
    f["spend_last_90d"] = last90["rev"].sum().reindex(f.index).fillna(0)
    f["n_products"] = h.groupby("CustomerID")["StockCode"].nunique()
    f["baseline_run_rate"] = f["monetary"] / np.maximum(f["customer_age_days"], CLV_WINDOW_DAYS) * CLV_WINDOW_DAYS
    return f


def retail_clv():
    raw, source = load_retail()
    sales = clean_retail_sales(raw, customers_only=True)
    last = sales["InvoiceDate"].max().normalize()
    cutoff = last - pd.Timedelta(days=CLV_WINDOW_DAYS)

    X = _features_at(sales, cutoff)
    fut = sales[(sales["InvoiceDate"] > cutoff) & (sales["InvoiceDate"] <= cutoff + pd.Timedelta(days=CLV_WINDOW_DAYS))]
    y = fut.groupby("CustomerID")["Revenue"].sum().reindex(X.index).fillna(0.0)

    feat_cols = [c for c in X.columns if c != "baseline_run_rate"]
    tr, te = train_test_split(X.index, test_size=0.25, random_state=SEED)

    def report(pred, idx):
        yt = y.loc[idx]
        top = pd.Series(pred, index=idx).nlargest(max(1, len(idx) // 10)).index
        return {"MAE": float(mean_absolute_error(yt, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(yt, pred))),
                "top_decile_revenue_capture": float(yt.loc[top].sum() / max(yt.sum(), 1e-9))}

    results = {"naive_run_rate": report(X.loc[te, "baseline_run_rate"].to_numpy(), te)}
    models = {
        "ridge_log": Ridge(alpha=3.0),
        "hist_gb": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED),
    }
    fitted = {}
    for name, m in models.items():
        Xtr = np.log1p(X.loc[tr, feat_cols].clip(lower=0))
        m.fit(Xtr, np.log1p(y.loc[tr]))
        pred = np.expm1(m.predict(np.log1p(X.loc[te, feat_cols].clip(lower=0)))).clip(min=0)
        results[name] = report(pred, te)
        fitted[name] = m
    # select by MAE (RMSE is dominated by a handful of wholesale "whale" customers);
    # the naive run-rate baseline is a legitimate candidate - if ML cannot beat it, say so.
    best = min(results, key=lambda k: results[k]["MAE"])

    # score today's customers with the best model (features measured at the last date)
    X_now = _features_at(sales, last)
    out = X_now.copy()
    if best == "naive_run_rate":
        out["predicted_90d_value"] = X_now["baseline_run_rate"]
    else:
        out["predicted_90d_value"] = np.expm1(fitted[best].predict(np.log1p(X_now[feat_cols].clip(lower=0)))).clip(min=0)
    out = out.reset_index().rename(columns={"index": "CustomerID"})
    out.to_csv(PROC / "retail_clv.csv", index=False)

    meta = {"data_source": source, "cutoff": str(cutoff.date()), "window_days": CLV_WINDOW_DAYS,
            "n_customers": int(len(X)), "metrics": results, "best_model": best,
            "note": "Features use only data up to the cutoff; target is spend in the next window."}
    (PROC / "retail_clv_metrics.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    m = retail_clv()
    print(json.dumps(m["metrics"], indent=2), "\nbest:", m["best_model"])
