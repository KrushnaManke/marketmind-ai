"""SQL analysis results plus a read-only query box on the project database."""
import re
import sqlite3

import pandas as pd
import streamlit as st
from utils import DB_PATH, PROC, ROOT, csv, footer, insight, methodology, section, setup

setup("Structured Query Language Insights", "🗄️", "Trace dashboard metrics to governed Structured Query Language (SQL) statements and safely explore the underlying database.")
insight("Every dashboard export is reproducible from a named SQL statement. Use this page to audit the calculation behind a metric.", "Data lineage")
methodology("Saved queries in `sql/analysis.sql` are executed against SQLite and exported to the processed data folder. Saved results are not regenerated automatically here.")

files = sorted(p.stem for p in (PROC / "sql").glob("*.csv"))
name = st.selectbox("Saved analysis", files)
sql_text = (ROOT / "sql" / "analysis.sql").read_text()
m = re.search(rf"-- name:\s*{name}\s*\n(.*?)(?=\n-- name:|\Z)", sql_text, flags=re.S)
if m:
    with st.expander("Show Structured Query Language", expanded=True):
        st.code(m.group(1).strip(), language="sql")
st.dataframe(csv(f"sql/{name}.csv"), hide_index=True, use_container_width=True)

section("Run a read-only query", "Explore the database with a single SELECT or WITH statement")
if not DB_PATH.exists():
    st.info("The SQLite database is not in this deployment; run `python -m src.load_db` locally to use this box.")
else:
    st.caption("Tables: `telco`, `retail`, `reviews`. Only a single SELECT (choose rows) / WITH (define a temporary query) statement is accepted.")
    q = st.text_area("Structured Query Language", "SELECT Contract, COUNT(*) AS customers, ROUND(AVG(Churn)*100,1) AS churn_pct\nFROM telco GROUP BY Contract")
    if st.button("Run"):
        stmt = q.strip().rstrip(";")
        if ";" in stmt or not re.match(r"(?is)^(select|with)\b", stmt):
            st.error("Only a single SELECT (choose rows) / WITH (define a temporary query) statement is allowed.")
        else:
            try:
                con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
                out = pd.read_sql_query(stmt + " LIMIT 1000" if " limit " not in stmt.lower() else stmt, con)
                con.close()
                st.dataframe(out, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(str(e))
footer()
