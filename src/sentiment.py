"""Phase 6: review sentiment (TF-IDF + Logistic Regression) and theme analysis.

Reviews are NOT linked to Telco customers (no shared IDs). Themes are used only to
show which *topics* dominate negative feedback, alongside the churn drivers.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import MODELS, PROC, SEED
from .data_sources import load_reviews

THEME_KEYWORDS = {
    "Price / billing": r"price|cost|expensive|cheap|afford|value|bill|fee|overpriced",
    "Customer support": r"support|service|agent|staff|helpdesk|response|rude|helpful|polite",
    "Delivery / shipping": r"deliver|shipping|package|arrived|courier|late",
    "Product quality": r"quality|broke|broken|durable|defect|faulty|reliable|works|stopped working",
    "Network / speed": r"network|signal|outage|connection|speed|internet",
}


def tag_themes(text: str) -> list:
    t = text.lower()
    return [k for k, rx in THEME_KEYWORDS.items() if re.search(rx, t)]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=60000)),
        ("lr", LogisticRegression(max_iter=2000, C=2.0)),
    ])


def main():
    df, source = load_reviews()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    pipe = build_pipeline().fit(tr["text"], tr["label"])
    p_te = pipe.predict_proba(te["text"])[:, 1]
    metrics = {"data_source": source, "model": "TF-IDF(1-2gram) + LogisticRegression",
               "n_train": int(len(tr)), "n_test": int(len(te)),
               "accuracy": float(accuracy_score(te["label"], p_te >= .5)),
               "f1": float(f1_score(te["label"], p_te >= .5)),
               "roc_auc": float(roc_auc_score(te["label"], p_te)),
               "positive_share": float(df["label"].mean())}
    (PROC / "sentiment_metrics.json").write_text(json.dumps(metrics, indent=2))

    # most informative terms
    vec, lr = pipe.named_steps["tfidf"], pipe.named_steps["lr"]
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    top = pd.concat([
        pd.DataFrame({"term": terms[order[-20:][::-1]], "weight": lr.coef_[0][order[-20:][::-1]], "sentiment": "positive"}),
        pd.DataFrame({"term": terms[order[:20]], "weight": lr.coef_[0][order[:20]], "sentiment": "negative"}),
    ])
    top.to_csv(PROC / "sentiment_top_terms.csv", index=False)

    # score all reviews with an out-of-sample flavour: use predictions from the model (fine for descriptive themes)
    df = df.copy()
    df["prob_positive"] = pipe.predict_proba(df["text"])[:, 1]
    df["predicted"] = (df["prob_positive"] >= .5).astype(int)
    df["themes"] = df["text"].map(tag_themes)

    ex = df.explode("themes").dropna(subset=["themes"])
    theme = ex.groupby("themes").agg(reviews=("label", "size"), negative_share=("predicted", lambda s: 1 - s.mean()),
                                     avg_prob_positive=("prob_positive", "mean")).reset_index().rename(columns={"themes": "theme"})
    theme["negative_reviews"] = (theme["reviews"] * theme["negative_share"]).round().astype(int)
    theme["negative_share"] *= 100
    theme.sort_values("negative_reviews", ascending=False).to_csv(PROC / "sentiment_themes.csv", index=False)

    out = df.sample(min(len(df), 3000), random_state=SEED).assign(themes=lambda d: d["themes"].map(lambda x: "; ".join(x)))
    out.to_csv(PROC / "sentiment_scored_sample.csv", index=False)
    joblib.dump(pipe, MODELS / "sentiment_tfidf_lr.joblib")
    print(json.dumps(metrics, indent=1))
    print(theme.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
