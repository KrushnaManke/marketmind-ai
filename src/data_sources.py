"""Load the three datasets: real files from data/raw/ if present, else synthetic.

Real files this module understands (put them in data/raw/):
  Telco   : telco.csv  or  WA_Fn-UseC_-Telco-Customer-Churn.csv
  Retail  : Online Retail.xlsx / online_retail.xlsx / online_retail.csv / "Online Retail.csv"
            (the newer UCI "Online Retail II" column names are normalised too)
  Reviews : reviews.csv with columns  text,label   (label 1 = positive, 0 = negative;
            the Kaggle IMDB file with review,sentiment columns also works)
"""
import re

import numpy as np
import pandas as pd

from . import make_synthetic as syn
from .config import RAW

NON_PRODUCT_CODES = r"^(POST|D|M|DOT|CRUK|BANK CHARGES|AMAZONFEE|S|B|PADS|C2|ADJUST2?|TEST\d+)$"
TELCO_REQUIRED_COLUMNS = {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"}


def _first_existing(*names):
    for n in names:
        if (RAW / n).exists():
            return RAW / n
    return None


def _read_csv_any_encoding(path, expected_cols=None):
    """Try a few encodings and reject files that clearly are not the expected dataset."""
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="skip")
            if expected_cols is not None and not expected_cols.issubset(df.columns):
                continue
            return df
        except Exception as exc:  # pragma: no cover - exercised via real-world bad input files
            last_error = exc
    if last_error is not None:
        return None
    return None


# ------------------------------------------------------------------ Telco -------
def load_telco():
    p = _first_existing("telco.csv", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if p is not None:
        df = _read_csv_any_encoding(p, TELCO_REQUIRED_COLUMNS)
        if df is None:
            df, source = syn.make_telco(), "synthetic"
        else:
            source = "real"
    else:
        df, source = syn.make_telco(), "synthetic"
    # TotalCharges contains blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if df["Churn"].dtype == object or str(df["Churn"].dtype).startswith("str"):
        df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["Churn"] = df["Churn"].astype(int)
    return df, source


# ----------------------------------------------------------- Online Retail ------
def load_retail():
    p = _first_existing("Online Retail.xlsx", "online_retail.xlsx", "online_retail.csv", "Online Retail.csv")
    if p is None:
        return syn.make_retail(), "synthetic"
    if p.suffix == ".xlsx":
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, encoding_errors="replace")
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df, "real"


def clean_retail_sales(df: pd.DataFrame, customers_only: bool = False) -> pd.DataFrame:
    """Valid sale lines only: no cancellations ('C...' invoices), positive qty/price,
    no postage/fee pseudo-products. Adds a Revenue column."""
    d = df.copy()
    inv = d["InvoiceNo"].astype(str)
    d = d[~inv.str.upper().str.startswith("C")]
    d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]
    d = d[~d["StockCode"].astype(str).str.upper().str.match(NON_PRODUCT_CODES)]
    if customers_only:
        d = d.dropna(subset=["CustomerID"])
        d["CustomerID"] = d["CustomerID"].astype(int)
    d["Revenue"] = d["Quantity"] * d["UnitPrice"]
    return d


# ----------------------------------------------------------------- Reviews ------
def load_reviews():
    p = _first_existing("reviews.csv", "IMDB Dataset.csv")
    if p is None:
        return syn.make_reviews(), "synthetic"
    df = pd.read_csv(p)
    df = df.rename(columns={"review": "text", "sentiment": "label", "Review": "text", "Sentiment": "label"})
    if df["label"].dtype == object or str(df["label"].dtype).startswith("str"):
        df["label"] = df["label"].astype(str).str.lower().map({"positive": 1, "negative": 0, "pos": 1, "neg": 0})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).map(lambda s: re.sub(r"<[^>]+>", " ", s))   # strip IMDB <br />
    if len(df) > 20000:            # keep CPU training quick; BERT script has its own subsampling
        df = df.sample(20000, random_state=42)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True), "real"
