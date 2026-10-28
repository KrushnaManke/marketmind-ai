-- MarketMind AI: business questions answered in SQL (SQLite dialect).
-- Each block starts with "-- name: <query_name>" and is exported to
-- data/processed/sql/<query_name>.csv by `python -m src.run_sql`.

-- name: churn_by_contract
SELECT Contract,
       COUNT(*)                              AS customers,
       ROUND(100.0 * AVG(Churn), 1)          AS churn_pct,
       ROUND(AVG(MonthlyCharges), 2)         AS avg_monthly_charge,
       ROUND(SUM(Churn * MonthlyCharges), 0) AS monthly_revenue_lost
FROM telco
GROUP BY Contract
ORDER BY churn_pct DESC;

-- name: churn_by_tenure_band
SELECT CASE WHEN tenure <= 12 THEN '1) 0-12 months'
            WHEN tenure <= 24 THEN '2) 13-24 months'
            WHEN tenure <= 48 THEN '3) 25-48 months'
            ELSE '4) 49+ months' END        AS tenure_band,
       COUNT(*)                             AS customers,
       ROUND(100.0 * AVG(Churn), 1)         AS churn_pct
FROM telco
GROUP BY 1
ORDER BY 1;

-- name: churn_by_payment_method
SELECT PaymentMethod,
       COUNT(*)                     AS customers,
       ROUND(100.0 * AVG(Churn), 1) AS churn_pct
FROM telco
GROUP BY PaymentMethod
ORDER BY churn_pct DESC;

-- name: churn_by_internet_and_support
SELECT InternetService, TechSupport,
       COUNT(*)                     AS customers,
       ROUND(100.0 * AVG(Churn), 1) AS churn_pct
FROM telco
WHERE InternetService <> 'No'
GROUP BY InternetService, TechSupport
ORDER BY churn_pct DESC;

-- name: telco_revenue_concentration
-- Share of monthly recurring revenue (MRR) by charge decile (window function NTILE).
WITH ranked AS (
  SELECT MonthlyCharges, Churn,
         NTILE(10) OVER (ORDER BY MonthlyCharges DESC) AS decile
  FROM telco)
SELECT decile,
       COUNT(*)                                                    AS customers,
       ROUND(SUM(MonthlyCharges), 0)                               AS mrr,
       ROUND(100.0 * SUM(MonthlyCharges) / SUM(SUM(MonthlyCharges)) OVER (), 1) AS mrr_share_pct,
       ROUND(100.0 * AVG(Churn), 1)                                AS churn_pct
FROM ranked
GROUP BY decile
ORDER BY decile;

-- name: retail_monthly_revenue
-- Month-over-month growth with LAG (cancellations / non-positive rows excluded).
WITH monthly AS (
  SELECT strftime('%Y-%m', InvoiceDate) AS month,
         SUM(Quantity * UnitPrice)      AS revenue,
         COUNT(DISTINCT InvoiceNo)      AS orders,
         COUNT(DISTINCT CustomerID)     AS customers
  FROM retail
  WHERE InvoiceNo NOT LIKE 'C%' AND Quantity > 0 AND UnitPrice > 0
  GROUP BY 1)
SELECT month,
       ROUND(revenue, 0) AS revenue,
       orders, customers,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1)         AS mom_growth_pct,
       ROUND(SUM(revenue) OVER (ORDER BY month), 0)           AS cumulative_revenue
FROM monthly
ORDER BY month;

-- name: retail_top_customers
SELECT CAST(CustomerID AS INTEGER)           AS customer_id,
       COUNT(DISTINCT InvoiceNo)             AS orders,
       ROUND(SUM(Quantity * UnitPrice), 0)   AS revenue,
       MAX(InvoiceDate)                      AS last_purchase
FROM retail
WHERE CustomerID IS NOT NULL AND InvoiceNo NOT LIKE 'C%' AND Quantity > 0 AND UnitPrice > 0
GROUP BY 1
ORDER BY revenue DESC
LIMIT 20;

-- name: retail_country_revenue
SELECT Country,
       ROUND(SUM(Quantity * UnitPrice), 0) AS revenue,
       COUNT(DISTINCT InvoiceNo)           AS orders
FROM retail
WHERE InvoiceNo NOT LIKE 'C%' AND Quantity > 0 AND UnitPrice > 0
GROUP BY Country
ORDER BY revenue DESC
LIMIT 10;

-- name: retail_cohort_retention
-- Cohort = month of a customer's first purchase. months_since = whole months after that.
WITH purchases AS (
  SELECT DISTINCT CAST(CustomerID AS INTEGER) AS cid,
         strftime('%Y-%m', InvoiceDate)       AS month
  FROM retail
  WHERE CustomerID IS NOT NULL AND InvoiceNo NOT LIKE 'C%' AND Quantity > 0 AND UnitPrice > 0),
cohort AS (
  SELECT cid, MIN(month) AS cohort_month FROM purchases GROUP BY cid),
activity AS (
  SELECT c.cohort_month,
         (CAST(substr(p.month, 1, 4) AS INTEGER) - CAST(substr(c.cohort_month, 1, 4) AS INTEGER)) * 12
         + CAST(substr(p.month, 6, 2) AS INTEGER) - CAST(substr(c.cohort_month, 6, 2) AS INTEGER) AS months_since,
         p.cid
  FROM purchases p JOIN cohort c ON c.cid = p.cid)
SELECT cohort_month, months_since,
       COUNT(DISTINCT cid) AS active_customers,
       ROUND(100.0 * COUNT(DISTINCT cid) /
             FIRST_VALUE(COUNT(DISTINCT cid)) OVER (PARTITION BY cohort_month ORDER BY months_since), 1) AS retention_pct
FROM activity
GROUP BY cohort_month, months_since
ORDER BY cohort_month, months_since;

-- name: retail_repeat_purchase_rate
WITH orders_per_customer AS (
  SELECT CustomerID, COUNT(DISTINCT InvoiceNo) AS n_orders
  FROM retail
  WHERE CustomerID IS NOT NULL AND InvoiceNo NOT LIKE 'C%' AND Quantity > 0 AND UnitPrice > 0
  GROUP BY CustomerID)
SELECT COUNT(*)                                              AS customers,
       SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END)         AS repeat_customers,
       ROUND(100.0 * AVG(CASE WHEN n_orders > 1 THEN 1.0 ELSE 0 END), 1) AS repeat_rate_pct,
       ROUND(AVG(n_orders), 2)                               AS avg_orders
FROM orders_per_customer;
