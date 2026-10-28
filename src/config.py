"""Central configuration: paths, random seed and business assumptions.

Every number in the "business impact" outputs depends on the assumptions below.
Change them to match your (real or imagined) company and re-run the pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SQL_OUT = PROC / "sql"
MODELS = ROOT / "models"
SQL_DIR = ROOT / "sql"
FIGURES = ROOT / "reports" / "figures"
DB_PATH = ROOT / "data" / "marketmind.db"

SEED = 42

# ---- Business assumptions (Telco churn module) --------------------------------
GROSS_MARGIN = 0.60          # share of revenue that is profit
RETENTION_OFFER_COST = 60.0  # cost of one retention offer / call (USD)
SAVE_RATE = 0.30             # share of contacted would-be churners we actually keep
CLV_HORIZON_MONTHS = 60
ANNUAL_DISCOUNT_RATE = 0.10
HIGH_RISK_THRESHOLD = 0.60   # used for the "high-risk" KPI label only

# ---- Retail module ------------------------------------------------------------
LAPSED_DAYS = 90             # DERIVED churn definition: no purchase in 90 days
CLV_WINDOW_DAYS = 90         # horizon for the retail CLV regression
KMEANS_K = 4                 # chosen for interpretability; diagnostics for k=2..8 are saved
FORECAST_HORIZON = 30
BACKTEST_HORIZON = 28
BACKTEST_FOLDS = 3

for _p in (RAW, PROC, SQL_OUT, MODELS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
