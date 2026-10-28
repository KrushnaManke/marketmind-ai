"""Phase 4: RFM customer segmentation on the retail data (K-Means + DBSCAN).

Retail has NO churn label, so we also add a clearly-labelled DERIVED flag:
`lapsed` = no purchase in the last LAPSED_DAYS days.
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import KMEANS_K, LAPSED_DAYS, PROC, SEED
from .data_sources import clean_retail_sales, load_retail

def build_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    snap = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    inv = sales.groupby(["CustomerID", "InvoiceNo"]).agg(date=("InvoiceDate", "max"), rev=("Revenue", "sum")).reset_index()
    g = inv.groupby("CustomerID")
    rfm = pd.DataFrame({"Recency": (snap - g["date"].max()).dt.days,
                        "Frequency": g["InvoiceNo"].nunique(),
                        "Monetary": g["rev"].sum()})
    rfm["lapsed"] = rfm["Recency"] > LAPSED_DAYS
    return rfm


def main():
    raw, source = load_retail()
    sales = clean_retail_sales(raw, customers_only=True)
    rfm = build_rfm(sales)

    # log-transform (RFM is heavily right-skewed) then standardise
    Z = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))

    # k-selection diagnostics (elbow + silhouette)
    diag = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        diag.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(Z, km.labels_, sample_size=3000, random_state=SEED))})
    diag = pd.DataFrame(diag)
    diag.to_csv(PROC / "kmeans_diagnostics.csv", index=False)

    km = KMeans(KMEANS_K, n_init=10, random_state=SEED).fit(Z)
    rfm["cluster"] = km.labels_
    # name clusters by a composite "goodness" score of the centroids (low recency, high F and M)
    cen = pd.DataFrame(km.cluster_centers_, columns=["R", "F", "M"])
    cen["score"] = -cen["R"] + cen["F"] + cen["M"]
    order = cen["score"].sort_values(ascending=False).index.tolist()
    if KMEANS_K == 4:
        # best score = Champions, worst = Hibernating; of the two middle clusters the MORE RECENT
        # one is "Promising" and the one that has gone quiet (higher recency) is "At Risk".
        mid = sorted(order[1:3], key=lambda c: cen.loc[c, "R"])
        mapping = {order[0]: "Champions", mid[0]: "Promising", mid[1]: "At Risk", order[3]: "Hibernating"}
    else:
        mapping = {c: f"Segment {i + 1}" for i, c in enumerate(order)}
    rfm["segment"] = rfm["cluster"].map(mapping)

    # DBSCAN: eps from the k-distance curve, noise points = unusual customers
    min_samples = 10
    kd = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1])
    eps = float(np.quantile(kd, 0.93))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    rfm["dbscan_outlier"] = db.labels_ == -1
    pd.DataFrame({"rank": np.arange(len(kd)), "k_distance": kd}).iloc[::max(1, len(kd) // 400)].to_csv(PROC / "dbscan_kdistance.csv", index=False)
    whale_cut = rfm["Monetary"].quantile(0.99)
    rfm["whale"] = rfm["Monetary"] >= whale_cut

    rfm.reset_index().to_csv(PROC / "rfm_segments.csv", index=False)

    prof = rfm.groupby("segment").agg(customers=("Recency", "size"), avg_recency_days=("Recency", "mean"),
                                      avg_orders=("Frequency", "mean"), avg_spend=("Monetary", "mean"),
                                      total_revenue=("Monetary", "sum"), lapsed_pct=("lapsed", "mean")).reset_index()
    prof["customer_share_pct"] = 100 * prof["customers"] / prof["customers"].sum()
    prof["revenue_share_pct"] = 100 * prof["total_revenue"] / prof["total_revenue"].sum()
    prof["lapsed_pct"] *= 100
    prof = prof.sort_values("avg_spend", ascending=False)
    prof.to_csv(PROC / "segment_profile.csv", index=False)

    meta = {"data_source": source, "n_customers": int(len(rfm)), "k": KMEANS_K,
            "silhouette_at_k": float(diag.loc[diag.k == KMEANS_K, "silhouette"].iloc[0]),
            "best_silhouette_k": int(diag.loc[diag.silhouette.idxmax(), "k"]),
            "dbscan": {"eps": eps, "min_samples": min_samples, "outliers": int(rfm["dbscan_outlier"].sum()),
                       "whales_p99": int(rfm["whale"].sum()),
                       "whale_revenue_share_pct": float(100 * rfm.loc[rfm["whale"], "Monetary"].sum() / rfm["Monetary"].sum())},
            "lapsed_definition_days": LAPSED_DAYS, "lapsed_rate": float(rfm["lapsed"].mean())}
    (PROC / "segmentation_summary.json").write_text(json.dumps(meta, indent=2))
    print(prof.round(1).to_string(index=False))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
