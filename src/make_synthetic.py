"""Synthetic stand-ins for the three public datasets.

They have the SAME COLUMNS as the real files so the whole pipeline runs without
downloads. Replace them by dropping the real files in data/raw/ (see README).
The dashboard shows a banner whenever synthetic data is in use, because
conclusions drawn from synthetic data are NOT real findings.
"""
import numpy as np
import pandas as pd

from .config import SEED


# ------------------------------------------------------------------ Telco -------
def make_telco(n: int = 7043, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    # tenure depends on contract (month-to-month customers are newer / leave sooner)
    tenure = np.where(
        contract == "Month-to-month", rng.gamma(1.3, 14, n),
        np.where(contract == "One year", rng.normal(42, 18, n), rng.normal(56, 16, n)),
    )
    tenure = np.clip(np.round(tenure), 0, 72).astype(int)

    def yn(p):
        return rng.choice(["Yes", "No"], n, p=[p, 1 - p])

    def addon(p):
        v = yn(p)
        return np.where(internet == "No", "No internet service", v)

    phone = yn(0.90)
    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": (rng.random(n) < 0.16).astype(int),
        "Partner": yn(0.48),
        "Dependents": yn(0.30),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", yn(0.42)),
        "InternetService": internet,
        "OnlineSecurity": addon(0.29),
        "OnlineBackup": addon(0.34),
        "DeviceProtection": addon(0.34),
        "TechSupport": addon(0.29),
        "StreamingTV": addon(0.38),
        "StreamingMovies": addon(0.39),
        "Contract": contract,
        "PaperlessBilling": yn(0.59),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.23, 0.22, 0.21]),
    })
    base = df["InternetService"].map({"No": 20.0, "DSL": 45.0, "Fiber optic": 72.0})
    n_addons = sum((df[c] == "Yes").astype(int) for c in
                   ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"])
    monthly = base + 5.5 * n_addons + (df["PhoneService"] == "Yes") * 3 + rng.normal(0, 3, n)
    df["MonthlyCharges"] = np.round(np.clip(monthly, 18.25, 118.75), 2)
    total = df["MonthlyCharges"] * df["tenure"] * rng.normal(1.0, 0.03, n)
    df["TotalCharges"] = np.round(total, 2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "      # the real file has blank strings here

    # latent churn model (intercept solved so that overall churn rate ~ 26.5%)
    z = (
        1.4 * (contract == "Month-to-month") - 1.1 * (contract == "Two year")
        + 0.9 * (internet == "Fiber optic") - 0.8 * (internet == "No")
        - 0.8 * (df["TechSupport"] == "Yes") - 0.7 * (df["OnlineSecurity"] == "Yes")
        + 0.55 * (df["PaymentMethod"] == "Electronic check")
        + 0.25 * df["SeniorCitizen"] + 0.25 * (df["PaperlessBilling"] == "Yes")
        - 0.035 * tenure + 0.012 * (df["MonthlyCharges"] - 65) + rng.normal(0, 0.6, n)
    ).to_numpy()
    lo, hi = -6.0, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z + mid)))).mean()
        lo, hi = (mid, hi) if rate < 0.265 else (lo, mid)
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")
    return df


# ----------------------------------------------------------- Online Retail ------
def make_retail(n_customers: int = 2600, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    start, end = pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-09")
    days = pd.date_range(start, end, freq="D")

    # day weights: no Saturdays (as in the real data), quiet Sundays, Q4 peak
    month_w = {1: .8, 2: .8, 3: .95, 4: .9, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.3, 10: 1.4, 11: 1.8, 12: 1.1}
    dow_w = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.15, 4: 0.9, 5: 0.0, 6: 0.55}
    w = np.array([month_w[d.month] * dow_w[d.dayofweek] for d in days])

    n_products = 450
    price = np.round(np.exp(rng.normal(0.9, 0.8, n_products)), 2).clip(0.29, 120)
    stock = [f"{rng.integers(10000, 99999)}" for _ in range(n_products)]
    desc = [f"PRODUCT {i:03d}" for i in range(n_products)]

    rows, inv_no = [], 536365
    def add_invoice(cust, day, wholesale, cancel=False, guest=False):
        nonlocal inv_no
        inv_no += 1
        k = int(min(40, 1 + rng.geometric(0.15 if not wholesale else 0.07)))
        prods = rng.choice(n_products, k, replace=False)
        ts = pd.Timestamp(day) + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
        for pi in prods:
            qty = int(rng.geometric(0.25)) * (12 if wholesale else 1) if not cancel else -int(rng.geometric(0.4))
            up = price[pi] * (0.0 if rng.random() < 0.003 else 1.0)
            country = "United Kingdom" if rng.random() < .9 else str(rng.choice(["Germany", "France", "EIRE", "Spain"]))
            rows.append((f"C{inv_no}" if cancel else str(inv_no), stock[pi], desc[pi], qty, ts,
                         round(float(up), 2), np.nan if guest else float(cust), country))

    for c in range(n_customers):
        cust = 12347 + c
        wholesale = rng.random() < 0.10
        first = int(rng.choice(len(days), p=np.r_[np.full(30, 0.6 / 30), np.full(len(days) - 30, 0.4 / (len(days) - 30))]))
        life = int(rng.exponential(420 if wholesale else 220)) + 14
        window = np.arange(first, min(len(days), first + life))
        rate = np.exp(rng.normal(0.35, 0.55)) * (2.5 if wholesale else 1.0)   # invoices per 30 days
        n_inv = max(1, rng.poisson(rate * len(window) / 30))
        pw = w[window] / w[window].sum()
        for di in rng.choice(window, n_inv, p=pw):
            add_invoice(cust, days[di], wholesale)
            if rng.random() < 0.02:                         # cancellation shortly after
                add_invoice(cust, days[min(len(days) - 1, di + int(rng.integers(0, 6)))], wholesale, cancel=True)
    # guest checkouts (no CustomerID) ~ 20% of lines in the real data
    for di in rng.choice(len(days), 1500, p=w / w.sum()):
        add_invoice(0, days[di], False, guest=True)

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    return df.sort_values("InvoiceDate").reset_index(drop=True)


# ----------------------------------------------------------------- Reviews ------
THEMES = {
    "price": (["great value for the price", "very affordable and fair billing", "cheap compared to others", "good deals and low cost"],
              ["way too expensive for what you get", "hidden fees on every bill", "overpriced and the price keeps rising", "terrible value, costs far too much"]),
    "support": (["customer support was friendly and helpful", "the agent solved my issue quickly", "excellent service and polite staff", "support responded fast"],
                ["support was rude and unhelpful", "waited hours for a response from the agent", "terrible customer service, nobody helped", "the helpdesk kept transferring me"]),
    "delivery": (["delivery was fast and the package arrived early", "shipping was quick and well packed", "arrived on time"],
                 ["delivery was late and the package was damaged", "shipping took weeks", "the courier lost my order"]),
    "quality": (["excellent quality and very reliable", "works perfectly and feels durable", "high quality, no problems at all"],
                ["poor quality, it broke after a week", "defective and unreliable", "faulty product, stopped working quickly"]),
    "network": (["the network is fast and the signal is strong", "stable connection and great speed", "never had an outage"],
                ["constant outages and a slow connection", "the signal drops all the time", "internet speed is terrible"]),
}
FILLER = ["", "Overall,", "To be honest,", "I have been a customer for a while.", "Just my experience:", "Not sure what else to say."]


def make_reviews(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    themes = list(THEMES)
    out = []
    for _ in range(n):
        positive = rng.random() < 0.55
        k = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
        chosen = rng.choice(themes, k, replace=False)
        parts = []
        for t in chosen:
            # 15% of sentences contradict the overall sentiment (mixed reviews)
            pos_sentence = positive if rng.random() > 0.15 else not positive
            parts.append(str(rng.choice(THEMES[t][0 if pos_sentence else 1])))
        text = (str(rng.choice(FILLER)) + " " + ". ".join(parts) + ".").strip()
        label = int(positive)
        if rng.random() < 0.05:           # label noise (sarcasm / mislabelled)
            label = 1 - label
        out.append((text, label))
    return pd.DataFrame(out, columns=["text", "label"])
