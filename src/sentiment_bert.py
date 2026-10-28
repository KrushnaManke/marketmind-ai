"""OPTIONAL upgrade: transformer sentiment (needs `pip install transformers torch`; GPU recommended).

    python -m src.sentiment_bert                 # zero-shot DistilBERT on 1,000 test reviews
    python -m src.sentiment_bert --finetune      # fine-tune DistilBERT on 4,000 reviews (Colab GPU: ~5 min)

Writes data/processed/sentiment_bert_metrics.json so the dashboard can show
TF-IDF vs BERT side by side. The dashboard itself never imports torch.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .config import PROC, SEED
from .data_sources import load_reviews


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=4000)
    a = ap.parse_args()

    from transformers import pipeline            # imported lazily: optional dependency

    df, source = load_reviews()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])   # same split as TF-IDF
    te = te.head(a.n_test)

    if a.finetune:
        import torch
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)
        name = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        enc = lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=128)
        ds = Dataset.from_pandas(tr.head(a.n_train)[["text", "label"]]).map(enc, batched=True)
        args = TrainingArguments("models/distilbert", num_train_epochs=2, per_device_train_batch_size=16,
                                 learning_rate=3e-5, report_to=[], save_strategy="no", seed=SEED)
        Trainer(model=model, args=args, train_dataset=ds).train()
        clf = pipeline("sentiment-analysis", model=model, tokenizer=tok, truncation=True, max_length=128,
                       device=0 if torch.cuda.is_available() else -1)
        positive = lambda lab: lab in ("LABEL_1", "POSITIVE")
        kind = "distilbert fine-tuned"
    else:
        clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                       truncation=True, max_length=128)
        positive = lambda lab: lab == "POSITIVE"
        kind = "distilbert zero-shot (SST-2)"

    out = clf(te["text"].tolist(), batch_size=32)
    pred = np.array([int(positive(o["label"])) for o in out])
    m = {"data_source": source, "model": kind, "n_test": int(len(te)),
         "accuracy": float(accuracy_score(te["label"], pred)), "f1": float(f1_score(te["label"], pred))}
    (PROC / "sentiment_bert_metrics.json").write_text(json.dumps(m, indent=2))
    print(m)


if __name__ == "__main__":
    main()
