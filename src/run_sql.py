"""Run every named query in sql/analysis.sql and export results to CSV."""
import re
import sqlite3

import pandas as pd

from .config import DB_PATH, SQL_DIR, SQL_OUT


def split_queries(text: str) -> dict:
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main():
    queries = split_queries((SQL_DIR / "analysis.sql").read_text())
    with sqlite3.connect(DB_PATH) as con:
        for name, q in queries.items():
            df = pd.read_sql_query(q, con)
            df.to_csv(SQL_OUT / f"{name}.csv", index=False)
            print(f"  {name:32s} {len(df):4d} rows")


if __name__ == "__main__":
    main()
