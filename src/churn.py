"""Phase 2-3: churn prediction, explainability, revenue at risk and business impact.

Design decisions (each one prevents a common mistake):
  * Preprocessing lives inside a Pipeline          -> no leakage from scaling/encoding.
  * Models are chosen by CV on the TRAIN split      -> the test set is touched once.
  * class_weight / scale_pos_weight distort probabilities, so the final model is
    wrapped in CalibratedClassifierCV              -> revenue-at-risk sums are meaningful.
  * Customer scores are OUT-OF-FOLD predictions    -> no customer is scored by a model that saw them.
  * Targeting uses expected value, not a fixed 0.5 -> contact when  p * save_rate * margin * annual_revenue > offer_cost.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .clv import telco_clv
from .config import (GROSS_MARGIN, HIGH_RISK_THRESHOLD, MODELS, PROC, RETENTION_OFFER_COST, SAVE_RATE, SEED)
from .data_sources import load_telco
from .features import NUMERIC, add_features, build_preprocessor, clean_feature_names, model_columns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                      # xgboost not installed -> sklearn's gradient boosting
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore")


def get_models(spw: float) -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                          random_state=SEED, n_jobs=-1)
    else:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250, class_weight="balanced", random_state=SEED)
    return models


def make_pipe(model, cats):
    return Pipeline([("prep", build_preprocessor(cats)), ("clf", clone(model))])


def net_benefit(flag, y, annual_rev):
    """Realised profit of a targeting rule: saved margin on true churners minus offer costs."""
    flag = np.asarray(flag, dtype=float)
    return float((flag * (np.asarray(y) * SAVE_RATE * np.asarray(annual_rev) * GROSS_MARGIN - RETENTION_OFFER_COST)).sum())


def value_rule(p, annual_rev):
    return (np.asarray(p) * SAVE_RATE * np.asarray(annual_rev) * GROSS_MARGIN) > RETENTION_OFFER_COST


def downsample(x, y, n=150):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return np.asarray(x)[idx], np.asarray(y)[idx]


# ----------------------------------------------------------------- explain ------
def _tree_shap(clf, Xt):
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    return np.asarray(sv)


def explain(best_name, fitted, X_all, cats):
    """Return (contribution matrix, feature names, scaled feature matrix, method).
    SHAP for tree models when shap is installed; otherwise exact additive contributions
    (coefficient x standardised value) from the logistic-regression model."""
    pipe = fitted[best_name]
    if HAS_SHAP and best_name in ("xgboost", "random_forest"):
        prep = pipe.named_steps["prep"]
        Xt = prep.transform(X_all)
        return _tree_shap(pipe.named_steps["clf"], Xt), clean_feature_names(prep.get_feature_names_out()), Xt, "shap_tree"
    lr = fitted["logistic_regression"]
    prep = lr.named_steps["prep"]
    Xt = prep.transform(X_all)
    return Xt * lr.named_steps["clf"].coef_[0], clean_feature_names(prep.get_feature_names_out()), Xt, "logistic_contributions"


def top_reasons(contrib, names, Xt, k=3):
    names = np.array(names)
    out = []
    for i in range(contrib.shape[0]):
        order = np.argsort(-contrib[i])[:k]
        row = []
        for j in order:
            if contrib[i, j] <= 0.02:
                row.append("")
                continue
            n = names[j]
            if n in NUMERIC:
                n = f"{n} ({'high' if Xt[i, j] > 0 else 'low'})"
            row.append(n)
        out.append(row + [""] * (k - len(row)))
    return pd.DataFrame(out, columns=[f"reason_{i + 1}" for i in range(k)])


# -------------------------------------------------------------------- main -----
def main():
    raw, source = load_telco()
    df = add_features(raw)
    _, cats = model_columns(df)
    X, y = df.drop(columns=["customerID", "Churn"]), df["Churn"]
    annual_rev = (df["MonthlyCharges"] * 12).to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # 1) model selection by CV on the training split only
    models = get_models(spw)
    cv_ap, fitted, test_rows, curves = {}, {}, [], []
    for name, m in models.items():
        pipe = make_pipe(m, cats)
        cv_ap[name] = float(cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="average_precision", n_jobs=1).mean())
        fitted[name] = pipe.fit(X_tr, y_tr)
        p = fitted[name].predict_proba(X_te)[:, 1]
        test_rows.append({"model": name, "cv_pr_auc": cv_ap[name], "roc_auc": roc_auc_score(y_te, p),
                          "pr_auc": average_precision_score(y_te, p),
                          "precision@0.5": precision_score(y_te, p >= .5), "recall@0.5": recall_score(y_te, p >= .5),
                          "brier_uncalibrated": brier_score_loss(y_te, p)})
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        for kind, a, b in (("roc", *downsample(fpr, tpr)), ("pr", *downsample(rc, pr))):
            curves += [{"kind": kind, "model": name, "x": float(i), "y": float(j)} for i, j in zip(a, b)]
        print(f"  {name:24s} cv_ap={cv_ap[name]:.3f} test_auc={test_rows[-1]['roc_auc']:.3f} test_ap={test_rows[-1]['pr_auc']:.3f}")
    best = max(cv_ap, key=cv_ap.get)
    print("  best model (by CV PR-AUC on train):", best)

    # 2) calibrate the best model; choose the probability threshold on TRAIN out-of-fold scores
    cal = CalibratedClassifierCV(make_pipe(models[best], cats), method="sigmoid", cv=5)
    p_tr_oof = cross_val_predict(cal, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    rev_tr, rev_te = annual_rev[X_tr.index], annual_rev[X_te.index]
    grid = np.arange(0.05, 0.95, 0.01)
    best_t = float(max(grid, key=lambda t: net_benefit(p_tr_oof >= t, y_tr, rev_tr)))
    cal.fit(X_tr, y_tr)
    p_te = cal.predict_proba(X_te)[:, 1]
    cal_auc, cal_ap, cal_brier = roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te), brier_score_loss(y_te, p_te)

    top20 = p_te >= np.quantile(p_te, 0.8)
    policies = {
        "contact nobody": np.zeros(len(y_te), bool),
        "contact everyone": np.ones(len(y_te), bool),
        "top 20% by risk": top20,
        f"p >= {best_t:.2f} (tuned on train)": p_te >= best_t,
        "expected-value rule": value_rule(p_te, rev_te),
    }
    policy_rows = [{"policy": k, "customers_contacted": int(v.sum()),
                    "churners_reached": int((v & (y_te.to_numpy() == 1)).sum()),
                    "recall": float((v & (y_te.to_numpy() == 1)).sum() / max(1, y_te.sum())),
                    "net_benefit_usd": net_benefit(v, y_te, rev_te)} for k, v in policies.items()]
    pd.DataFrame(policy_rows).to_csv(PROC / "churn_policies.csv", index=False)

    # 3) out-of-fold probabilities for EVERY customer (calibrated)
    p_all = cross_val_predict(cal, X, y, cv=skf, method="predict_proba")[:, 1]
    frac, mean_p = calibration_curve(y, p_all, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_p, "observed_rate": frac}).to_csv(PROC / "churn_calibration.csv", index=False)

    # 4) explanations (global + per customer)
    contrib, names, Xt, method = explain(best, fitted, X, cats)
    imp = pd.DataFrame({"encoded_feature": names, "importance": np.abs(contrib).mean(0)})
    imp["base_feature"] = imp["encoded_feature"].str.split("=").str[0]
    def _dir(j):
        a, b = Xt[:, j], contrib[:, j]
        return "raises risk" if a.std() > 0 and b.std() > 0 and np.corrcoef(a, b)[0, 1] > 0 else "lowers risk"
    imp["direction_when_present_or_high"] = [_dir(j) for j in range(len(names))]
    imp["method"] = method
    imp.sort_values("importance", ascending=False).to_csv(PROC / "churn_importance.csv", index=False)
    pi = permutation_importance(fitted[best], X_te, y_te, scoring="average_precision", n_repeats=8,
                                random_state=SEED, n_jobs=1)
    pd.DataFrame({"feature": X_te.columns, "perm_importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("perm_importance", ascending=False).to_csv(PROC / "churn_permutation_importance.csv", index=False)
    reasons = top_reasons(contrib, names, Xt)

    # 5) scored customer table
    scored = df.copy().reset_index(drop=True)
    scored["churn_prob"] = p_all
    scored["risk_band"] = pd.cut(p_all, [-.01, .30, HIGH_RISK_THRESHOLD, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    scored["annual_revenue"] = annual_rev
    scored["revenue_at_risk"] = p_all * annual_rev
    scored["clv"] = telco_clv(df).to_numpy()
    scored["clv_at_risk"] = p_all * scored["clv"]
    scored["expected_save_value"] = p_all * SAVE_RATE * annual_rev * GROSS_MARGIN
    scored["target_flag"] = value_rule(p_all, annual_rev)
    scored["is_active"] = (df["Churn"] == 0).to_numpy()
    scored = pd.concat([scored, reasons], axis=1)
    scored.to_csv(PROC / "telco_scored.csv", index=False)

    # 6) gain / lift table (decile 1 = highest predicted risk)
    g = pd.DataFrame({"p": p_all, "y": y.to_numpy(), "rev": annual_rev})
    g["decile"] = pd.qcut(g["p"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    gt = g.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_prob=("p", "mean"),
                                 annual_revenue=("rev", "sum")).reset_index()
    gt["cum_churners_captured_pct"] = 100 * gt["churners"].cumsum() / gt["churners"].sum()
    gt["lift"] = (gt["churners"] / gt["customers"]) / g["y"].mean()
    gt.to_csv(PROC / "churn_gain_table.csv", index=False)

    # 7) summary + business impact (all customers, out-of-fold)
    act = scored[scored["is_active"]]
    nb_all = {k: net_benefit(v, y, annual_rev) for k, v in {
        "everyone": np.ones(len(y), bool), "expected_value_rule": scored["target_flag"].to_numpy(),
        "top_20pct": p_all >= np.quantile(p_all, 0.8)}.items()}
    summary = {
        "data_source": source, "n_customers": int(len(df)), "n_active": int(len(act)),
        "churn_rate": float(y.mean()),
        "annual_revenue_active": float(act["annual_revenue"].sum()),
        "revenue_at_risk_active": float(act["revenue_at_risk"].sum()),
        "clv_at_risk_active": float(act["clv_at_risk"].sum()),
        "revenue_already_lost_annualised": float(scored.loc[~scored["is_active"], "annual_revenue"].sum()),
        "mrr_active": float(act["MonthlyCharges"].sum()),
        "avg_clv": float(scored["clv"].mean()),
        "n_high_risk_active": int((act["churn_prob"] >= HIGH_RISK_THRESHOLD).sum()),
        "n_targeted_active": int(act["target_flag"].sum()),
        "expected_saved_profit_targeted_active": float(act.loc[act["target_flag"], "expected_save_value"].sum()
                                                       - RETENTION_OFFER_COST * act["target_flag"].sum()),
        "net_benefit_all_customers": nb_all,
        "calibration_check": {"sum_predicted": float(p_all.sum()), "sum_actual": int(y.sum())},
        "assumptions": {"gross_margin": GROSS_MARGIN, "offer_cost": RETENTION_OFFER_COST, "save_rate": SAVE_RATE},
    }
    (PROC / "churn_summary.json").write_text(json.dumps(summary, indent=2))
    metrics = {"data_source": source, "best_model": best, "uses_xgboost": HAS_XGB, "explanation_method": method,
               "models": test_rows,
               "best_calibrated": {"roc_auc": cal_auc, "pr_auc": cal_ap, "brier": cal_brier,
                                   "threshold_tuned_on_train": best_t},
               "train_size": int(len(X_tr)), "test_size": int(len(X_te)), "churn_rate_test": float(y_te.mean())}
    (PROC / "churn_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(curves).to_csv(PROC / "churn_curves.csv", index=False)

    # 8) final model on all data for the dashboard's "score a customer" form
    final = clone(cal).fit(X, y)
    joblib.dump({"model": final, "columns": list(X.columns), "threshold": best_t, "best": best}, MODELS / "churn_model.joblib")
    print(f"  calibrated test ROC-AUC={cal_auc:.3f} PR-AUC={cal_ap:.3f} Brier={cal_brier:.3f}")
    print(f"  revenue at risk (active): ${summary['revenue_at_risk_active']:,.0f}   explanation={method}")
    return summary


if __name__ == "__main__":
    main()
