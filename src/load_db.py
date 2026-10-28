"""Phase 1: load datasets into SQLite and record where each one came from."""
import json
import sqlite3

from .config import DB_PATH, PROC
from .data_sources import load_retail, load_reviews, load_telco


def main():
    telco, s1 = load_telco()
    retail, s2 = load_retail()
    reviews, s3 = load_reviews()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        telco.to_sql("telco", con, if_exists="replace", index=False)
        retail.assign(InvoiceDate=retail["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")) \
              .to_sql("retail", con, if_exists="replace", index=False)
        reviews.to_sql("reviews", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_retail_cust ON retail(CustomerID)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_telco_contract ON telco(Contract)")

    meta = {"telco": s1, "retail": s2, "reviews": s3,
            "rows": {"telco": len(telco), "retail": len(retail), "reviews": len(reviews)}}
    (PROC / "data_sources.json").write_text(json.dumps(meta, indent=2))
    print("Loaded:", meta)


if __name__ == "__main__":
    main()
