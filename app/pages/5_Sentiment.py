"""Customer review sentiment: model quality, informative terms, themes, live scoring."""
import joblib
import plotly.express as px
import streamlit as st
from utils import CAT, MODELS, csv, exists, footer, insight, js, methodology, section, setup, style

setup("Review Sentiment", "💬", "Measure customer voice, recurring themes and the confidence of live classifications. TF-IDF means Term Frequency–Inverse Document Frequency.")
m = js("sentiment_metrics.json")
st.caption("Reviews are a separate dataset; they are not matched to individual Telco customers.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("F1 (Term Frequency–Inverse Document Frequency + Logistic Regression)", f"{m['f1']:.3f}")
k2.metric("Accuracy", f"{m['accuracy']:.3f}")
k3.metric("Receiver Operating Characteristic — Area Under the Curve", f"{m['roc_auc']:.3f}")
k4.metric("Positive reviews", f"{m['positive_share']:.0%}")
st.caption(f"Evaluation: {m['n_test']:,} held-out reviews · {m['n_train']:,} training reviews · pooled F1 (harmonic mean of precision and recall) test-set estimate")
insight("Treat sentiment and themes as a voice-of-customer signal, not evidence of a specific customer relationship or cause.", "Interpretation")
if exists("sentiment_bert_metrics.json"):
    b = js("sentiment_bert_metrics.json")
    st.success(f"Transformer comparison: **{b['model']}** F1 = {b['f1']:.3f} on {b['n_test']:,} test reviews "
               f"(Term Frequency–Inverse Document Frequency baseline: {m['f1']:.3f}).")
else:
    st.info("Optional: run `python -m src.sentiment_bert` (needs transformers + torch) to compare the Distilled Bidirectional Encoder Representations from Transformers model (DistilBERT) with the Term Frequency–Inverse Document Frequency baseline.")

terms = csv("sentiment_top_terms.csv")
terms["sentiment"] = terms["sentiment"].str.capitalize()
fig = px.bar(terms.sort_values("weight"), x="weight", y="term", color="sentiment", orientation="h",
             color_discrete_map={"Positive": CAT[0], "Negative": CAT[1]}, title="Most informative words and phrases")
fig.update_traces(marker_cornerradius=3)
fig.update_xaxes(title="Model weight")
fig.update_yaxes(title="")
st.plotly_chart(style(fig, 760), use_container_width=True)

if exists("sentiment_themes.csv"):
    t = csv("sentiment_themes.csv")
    fig = px.bar(t.sort_values("negative_share"), x="negative_share", y="theme", orientation="h",
                 color_discrete_sequence=[CAT[1]], title="Share of reviews per theme that are negative")
    fig.update_traces(marker_cornerradius=4)
    fig.update_xaxes(title="% negative")
    fig.update_yaxes(title="")
    st.plotly_chart(style(fig, 320, legend=False), use_container_width=True)

section("Live review classifier", "Enter unseen text and inspect the calibrated model output")
text = st.text_area("Type a review", "Support was rude and the bill is way too expensive.")
@st.cache_resource
def load_sentiment_model():
    return joblib.load(MODELS / "sentiment_tfidf_lr.joblib")


try:
    model = load_sentiment_model()
    p = float(model.predict_proba([text])[:, 1][0])
    st.metric("Probability positive", f"{p:.0%}", delta="positive" if p >= 0.5 else "negative")
except Exception as e:
    st.info(f"Live scoring unavailable ({type(e).__name__}); run `python -m src.sentiment`.")

section("Review explorer", "Inspect representative scored reviews and recurring themes")
rev = csv("sentiment_scored_sample.csv")
theme = st.selectbox("Theme", ["All"] + sorted({x for s in rev["themes"].dropna() for x in s.split("; ")}))
view = rev if theme == "All" else rev[rev["themes"].fillna("").str.contains(theme, regex=False)]
st.dataframe(view[["text", "label", "prob_positive", "themes"]].head(200), hide_index=True, use_container_width=True,
             column_config={"prob_positive": st.column_config.ProgressColumn("P(positive)", min_value=0.0, max_value=1.0, format="%.2f")})
methodology(
    "The displayed model uses Term Frequency–Inverse Document Frequency features with logistic regression. Positive probability is a model estimate, not a guarantee of "
    "sentiment. Short, ironic, domain-specific or translated text may be misclassified. Theme keywords are descriptive and "
    "do not establish customer intent or causality.")
footer()
