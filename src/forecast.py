"""Phase 5: revenue forecasting with rolling-origin backtests.

Models (always available): seasonal-naive, seasonal-level.
Optional (used automatically when installed): Holt-Winters & SARIMAX (statsmodels), Prophet.

Rules: time-ordered splits only; every model is judged on the SAME backtest folds;
prediction intervals are empirical quantiles of the pooled backtest residuals so
that all models get comparable, honest bands.
"""
import json
import warnings

import numpy as np
import pandas as pd

from .config import BACKTEST_FOLDS, BACKTEST_HORIZON, FORECAST_HORIZON, PROC
from .data_sources import clean_retail_sales, load_retail

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    HAS_SM = True
except Exception:
    HAS_SM = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False


# ----------------------------------------------------------------- models ------
def seasonal_naive(train: pd.Series, h: int) -> np.ndarray:
    last = train.iloc[-7:].to_numpy()
    return np.resize(last, h).astype(float)


def seasonal_level(train: pd.Series, h: int) -> np.ndarray:
    """Recent level x day-of-week index (index estimated on the last 8 weeks)."""
    recent = train.iloc[-56:]
    dow_idx = recent.groupby(recent.index.dayofweek).mean() / max(recent.mean(), 1e-9)
    level = train.iloc[-28:].mean()
    future_idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h)
    return np.array([level * dow_idx.get(d.dayofweek, 1.0) for d in future_idx])


def holt_winters(train: pd.Series, h: int) -> np.ndarray:
    m = ExponentialSmoothing(train.clip(lower=0) + 1, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
    return np.asarray(m.forecast(h)) - 1


def sarimax(train: pd.Series, h: int) -> np.ndarray:
    m = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False, maxiter=100)
    return np.asarray(m.forecast(h))


def prophet_model(train: pd.Series, h: int) -> np.ndarray:
    d = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.fit(d)
    return m.predict(m.make_future_dataframe(h))["yhat"].to_numpy()[-h:]


MODELS = {"seasonal_naive": seasonal_naive, "seasonal_level": seasonal_level}
if HAS_SM:
    MODELS.update({"holt_winters": holt_winters, "sarimax": sarimax})
if HAS_PROPHET:
    MODELS["prophet"] = prophet_model


# ----------------------------------------------------------------- metrics -----
def metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = p - a
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "WAPE_pct": float(100 * np.abs(err).sum() / max(a.sum(), 1e-9)),     # robust to zero-sales days
            "total_error_pct": float(100 * (p.sum() - a.sum()) / max(a.sum(), 1e-9))}


def main():
    raw, source = load_retail()
    sales = clean_retail_sales(raw, customers_only=False)          # company revenue incl. guest checkouts
    daily = sales.set_index("InvoiceDate")["Revenue"].resample("D").sum()
    last_day = daily.index.max()
    partial_last_month = last_day.day < last_day.days_in_month
    monthly = daily.resample("MS").sum()
    if partial_last_month:
        monthly = monthly.iloc[:-1]            # drop the incomplete final month

    H = BACKTEST_HORIZON
    rows, resid, bt_pred = [], {k: [] for k in MODELS}, []
    for f in range(BACKTEST_FOLDS, 0, -1):
        cut = len(daily) - f * H
        train, test = daily.iloc[:cut], daily.iloc[cut:cut + H]
        for name, fn in MODELS.items():
            try:
                pred = np.clip(fn(train, len(test)), 0, None)
            except Exception as e:                       # one failing optional model must not kill the run
                print(f"  ! {name} failed on fold {f}: {e}")
                continue
            rows.append({"model": name, "fold": BACKTEST_FOLDS - f + 1, "train_end": str(train.index[-1].date()), **metrics(test, pred)})
            resid[name] += list(test.to_numpy() - pred)
            if f == 1:
                bt_pred.append(pd.DataFrame({"date": test.index, "actual": test.to_numpy(), "predicted": pred, "model": name}))
    bt = pd.DataFrame(rows)
    bt.to_csv(PROC / "forecast_backtest.csv", index=False)
    pd.concat(bt_pred).to_csv(PROC / "forecast_backtest_last_fold.csv", index=False)
    summary = bt.groupby("model")[["MAE", "RMSE", "WAPE_pct", "total_error_pct"]].mean().sort_values("WAPE_pct")
    summary.to_csv(PROC / "forecast_model_comparison.csv")
    best = summary.index[0]
    print(summary.round(1).to_string())

    # final forecast with the winner trained on ALL history; empirical interval from its backtest residuals
    try:
        pred = np.clip(MODELS[best](daily, FORECAST_HORIZON), 0, None)
    except Exception as e:                      # optional model failed on the full history: fall back, but say so
        print(f"  ! {best} failed on full history ({e}); falling back to seasonal_level")
        best = "seasonal_level"
        pred = np.clip(MODELS[best](daily, FORECAST_HORIZON), 0, None)
        resid[best] = resid.get(best) or [0.0]
    q_lo, q_hi = np.quantile(resid[best], [0.10, 0.90])
    idx = pd.date_range(last_day + pd.Timedelta(days=1), periods=FORECAST_HORIZON)
    fut = pd.DataFrame({"date": idx, "forecast": pred, "lower_80": np.clip(pred + q_lo, 0, None), "upper_80": pred + q_hi, "model": best})
    fut.to_csv(PROC / "forecast_future.csv", index=False)
    daily.rename("revenue").rename_axis("date").reset_index().to_csv(PROC / "forecast_history.csv", index=False)
    monthly.rename("revenue").rename_axis("month").reset_index().to_csv(PROC / "forecast_monthly.csv", index=False)

    adf_p = float(adfuller(daily.iloc[:-FORECAST_HORIZON])[1]) if HAS_SM else None
    meta = {"data_source": source, "best_model": best, "models_tried": list(MODELS),
            "uses_statsmodels": HAS_SM, "uses_prophet": HAS_PROPHET,
            "history_days": int(len(daily)), "history_start": str(daily.index[0].date()), "history_end": str(last_day.date()),
            "partial_last_month_dropped_from_monthly": bool(partial_last_month),
            "next_30d_total": float(pred.sum()),
            "last_30d_actual_total": float(daily.iloc[-30:].sum()),
            "backtest_wape_best_pct": float(summary.loc[best, "WAPE_pct"]),
            "backtest_total_error_best_pct": float(summary.loc[best, "total_error_pct"]),
            "adf_pvalue": adf_p,
            "caveat": "About one year of history cannot reveal annual seasonality; treat the horizon as short-term."}
    (PROC / "forecast_summary.json").write_text(json.dumps(meta, indent=2))
    print(f"  best={best}  next-30d total={pred.sum():,.0f}  last-30d actual={meta['last_30d_actual_total']:,.0f}")


if __name__ == "__main__":
    main()
