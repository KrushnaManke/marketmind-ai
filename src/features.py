"""Feature engineering for the Telco churn model.

All features are deterministic row-wise transforms (no statistics learned from the
data), so computing them before the train/test split does not leak anything.
Anything that LEARNS from data (scaling, encoding) lives inside the sklearn Pipeline.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_addons", "avg_charge_per_month", "SeniorCitizen"]
ID_AND_TARGET = ["customerID", "Churn"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["num_addons"] = sum((d[c] == "Yes").astype(int) for c in ADDON_COLS)
    d["avg_charge_per_month"] = d["TotalCharges"] / (d["tenure"] + 1)
    d["is_autopay"] = d["PaymentMethod"].str.contains("automatic", case=False).map({True: "Yes", False: "No"})
    d["has_protection"] = ((d["OnlineSecurity"] == "Yes") | (d["TechSupport"] == "Yes")).map({True: "Yes", False: "No"})
    d["tenure_group"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 100],
                               labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    return d


def model_columns(df: pd.DataFrame):
    feats = [c for c in df.columns if c not in ID_AND_TARGET]
    cats = [c for c in feats if c not in NUMERIC]
    return NUMERIC, cats


def _combine(feature, category):
    """Module-level (picklable) name combiner: 'Contract=Month-to-month'."""
    return f"{feature}={category}"


def build_preprocessor(cats):
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              feature_name_combiner=_combine), cats),
    ])


def clean_feature_names(names):
    return [n.split("__", 1)[1] if "__" in n else n for n in names]
