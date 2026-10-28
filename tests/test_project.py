"""Unit + output-sanity tests.  Run:  python tests/test_project.py   (or: pytest, if installed)
Output tests need the pipeline to have been run first (python -m src.pipeline)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import make_synthetic as syn                      # noqa: E402
from src.churn import net_benefit, value_rule              # noqa: E402
from src.clv import telco_clv                              # noqa: E402
from src.config import PROC                                # noqa: E402
from src.data_sources import clean_retail_sales            # noqa: E402
from src.features import add_features, build_preprocessor, model_columns   # noqa: E402
from src.forecast import metrics, seasonal_level, seasonal_naive            # noqa: E402
from src.run_sql import split_queries                      # noqa: E402


# ---------------------------------------------------------------- unit tests --------
def _telco_small():
    d = syn.make_telco(800, seed=3)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    d["Churn"] = (d["Churn"] == "Yes").astype(int)
    return d


def test_synthetic_telco_has_real_schema_and_blank_total_charges():
    d = syn.make_telco(500, seed=1)
    assert {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"} <= set(d.columns)
    blanks = d.loc[d["tenure"] == 0, "TotalCharges"]
    assert (blanks == " ").all()


def test_engineered_features():
    d = add_features(_telco_small())
    assert d["num_addons"].between(0, 6).all()
    assert set(d["is_autopay"]) <= {"Yes", "No"}
    assert not d["tenure_group"].isin(["nan"]).any()


def test_preprocessor_learns_only_from_the_data_it_is_fitted_on():
    d = add_features(_telco_small())
    _, cats = model_columns(d)
    X = d.drop(columns=["customerID", "Churn"])
    train, full = X.iloc[:200], X
    p = build_preprocessor(cats).fit(train)
    scaler = p.named_transformers_["num"]
    assert np.isclose(scaler.mean_[0], train["tenure"].mean())
    assert not np.isclose(scaler.mean_[0], full["tenure"].mean())


def test_retail_cleaning_removes_cancellations_and_bad_rows():
    raw = pd.DataFrame({
        "InvoiceNo": ["1", "C2", "3", "4", "5"], "StockCode": ["A1", "A1", "A1", "POST", "A2"],
        "Quantity": [2, -2, -1, 1, 3], "UnitPrice": [1.0, 1.0, 1.0, 5.0, 0.0], "CustomerID": [1.0, 1.0, 2.0, 2.0, np.nan],
        "InvoiceDate": pd.to_datetime(["2011-01-01"] * 5)})
    out = clean_retail_sales(raw)
    assert list(out["InvoiceNo"]) == ["1"]           # cancellation, negative qty, POST fee and zero price all dropped
    assert out["Revenue"].iloc[0] == 2.0


def test_telco_clv_is_higher_for_more_loyal_contracts():
    d = _telco_small()
    clv = telco_clv(d)
    per_dollar = (clv / d["MonthlyCharges"]).groupby(d["Contract"]).mean()
    assert per_dollar["Two year"] > per_dollar["One year"] > per_dollar["Month-to-month"]
    assert (clv > 0).all()


def test_business_rules():
    assert net_benefit([1, 0], [1, 1], [1200, 1200]) > 0          # contacting a churner with $1.2k revenue pays off
    assert net_benefit([1], [0], [1200]) < 0                       # contacting a loyal customer only costs money
    assert bool(value_rule([0.9], [1200])[0]) and not bool(value_rule([0.01], [300])[0])


def test_forecast_helpers():
    idx = pd.date_range("2011-01-01", periods=70)
    s = pd.Series(np.tile([10, 12, 14, 16, 0, 5, 8], 10).astype(float), index=idx)
    assert len(seasonal_naive(s, 30)) == 30 and len(seasonal_level(s, 30)) == 30
    m = metrics([10, 10], [12, 8])
    assert m["MAE"] == 2 and m["total_error_pct"] == 0


def test_sql_file_parses_into_named_queries():
    q = split_queries((ROOT / "sql" / "analysis.sql").read_text())
    def first_statement_word(sql):
        body = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--")).strip()
        return body.split()[0].upper()
    assert len(q) >= 10 and "churn_by_contract" in q
    assert all(first_statement_word(v) in ("SELECT", "WITH") for v in q.values())


# ------------------------------------------------------------- output sanity ---------
def _need(name):
    p = PROC / name
    if not p.exists():
        raise AssertionError(f"{name} missing - run `python -m src.pipeline` first")
    return p


def test_scored_customers_are_valid_and_calibrated():
    d = pd.read_csv(_need("telco_scored.csv"))
    assert d["churn_prob"].between(0, 1).all()
    assert (d["revenue_at_risk"] <= d["annual_revenue"] + 1e-6).all()
    s = json.loads(_need("churn_summary.json").read_text())
    pred, actual = s["calibration_check"]["sum_predicted"], s["calibration_check"]["sum_actual"]
    assert abs(pred - actual) / actual < 0.05, "predicted churner count should match actual within 5% (calibration)"


def test_models_beat_chance_and_selection_used_train_cv():
    m = json.loads(_need("churn_metrics.json").read_text())
    assert m["best_calibrated"]["roc_auc"] > 0.7
    assert all(r["roc_auc"] > 0.7 for r in m["models"])


def test_forecast_intervals_are_ordered_and_non_negative():
    f = pd.read_csv(_need("forecast_future.csv"))
    assert (f["lower_80"] <= f["forecast"] + 1e-6).all() and (f["forecast"] <= f["upper_80"] + 1e-6).all()
    assert (f["lower_80"] >= 0).all() and len(f) == 30


def test_segments_cover_all_customers():
    r = pd.read_csv(_need("rfm_segments.csv"))
    assert set(r["segment"]) == {"Champions", "Promising", "At Risk", "Hibernating"}
    assert r["segment"].notna().all() and r["CustomerID"].is_unique


def test_sql_exports_exist():
    assert len(list((PROC / "sql").glob("*.csv"))) >= 10


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS", name)
        except Exception as e:
            bad += 1
            print("  FAIL", name, "->", type(e).__name__, e)
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
