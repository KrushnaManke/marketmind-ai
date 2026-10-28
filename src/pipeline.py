"""Run the whole project end to end:   python -m src.pipeline

Order matters only in that load_db comes first. Use --only to run a subset, e.g.
    python -m src.pipeline --only churn forecast
"""
import argparse
import time

from . import churn, clv, forecast, load_db, report_figures, run_sql, segmentation, sentiment

STEPS = {
    "load_db": load_db.main,
    "sql": run_sql.main,
    "churn": churn.main,
    "retail_clv": clv.retail_clv,
    "segmentation": segmentation.main,
    "forecast": forecast.main,
    "sentiment": sentiment.main,
    "figures": report_figures.main,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(STEPS), help="run only these steps")
    args = ap.parse_args()
    for name, fn in STEPS.items():
        if args.only and name not in args.only:
            continue
        t = time.time()
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        fn()
        print(f"--- {name} done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
