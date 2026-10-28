"""OPTIONAL: fetch the real datasets (run on your own machine, with internet).

    python -m src.download_data

Telco and Online Retail come from public URLs; if a download fails the script
prints where to get the file manually. Reviews use the IMDB set from Hugging Face.
"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import RAW

TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def fetch(url: str, dest: Path):
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)


def main():
    try:
        fetch(TELCO_URL, RAW / "telco.csv")
    except Exception as e:  # noqa: BLE001
        print("  telco failed:", e, "\n  -> download 'Telco Customer Churn' from Kaggle, save as data/raw/telco.csv")
    try:
        z = RAW / "online_retail.zip"
        fetch(RETAIL_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(RAW)
        z.unlink()
    except Exception as e:  # noqa: BLE001
        print("  retail failed:", e, "\n  -> download 'Online Retail' from UCI, save as data/raw/Online Retail.xlsx")
    try:
        from datasets import load_dataset
        ds = load_dataset("imdb", split="train").shuffle(seed=42).select(range(20000))
        pd.DataFrame({"text": ds["text"], "label": ds["label"]}).to_csv(RAW / "reviews.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print("  reviews failed:", e, "\n  -> any CSV with columns text,label (1=positive) works")


if __name__ == "__main__":
    main()
