-- ============================================================
-- FILE: sales_analysis.sql
-- PROJECT: Sales Performance & Revenue Analytics Dashboard
-- AUTHOR: Ajinkya Wathore
-- DESCRIPTION: Advanced SQL analysis of 10,000+ sales records
--              using CTEs, Window Functions, Subqueries, and Aggregations
-- ============================================================


-- ============================================================
-- SECTION 1: DATABASE SETUP
-- ============================================================

CREATE DATABASE IF NOT EXISTS sales_analytics;
USE sales_analytics;

-- Core sales table (matches Power BI dataset structure)
CREATE TABLE IF NOT EXISTS sales (
    order_id       VARCHAR(20),
    order_date     DATE,
    ship_date      DATE,
    ship_mode      VARCHAR(30),
    customer_id    VARCHAR(20),
    customer_name  VARCHAR(100),
    segment        VARCHAR(30),
    region         VARCHAR(20),
    state          VARCHAR(50),
    category       VARCHAR(50),
    sub_category   VARCHAR(50),
    product_name   VARCHAR(200),
    sales          DECIMAL(10,2),
    quantity       INT,
    discount       DECIMAL(4,2),
    profit         DECIMAL(10,2)
);


-- ============================================================
-- SECTION 2: OVERALL BUSINESS KPIs
-- ============================================================

-- Total Revenue, Profit, Quantity and Profit Margin
SELECT
    COUNT(DISTINCT order_id)                         AS total_orders,
    COUNT(DISTINCT customer_id)                      AS total_customers,
    ROUND(SUM(sales), 2)                             AS total_revenue,
    ROUND(SUM(profit), 2)                            AS total_profit,
    SUM(quantity)                                    AS total_units_sold,
    ROUND((SUM(profit) / SUM(sales)) * 100, 2)      AS profit_margin_pct
FROM sales;


-- ============================================================
-- SECTION 3: REVENUE TREND ANALYSIS (CTE + Window Functions)
-- ============================================================

-- Monthly Revenue with Month-over-Month Growth
WITH monthly_revenue AS (
    SELECT
        YEAR(order_date)                             AS yr,
        MONTH(order_date)                            AS mth,
        DATE_FORMAT(order_date, '%Y-%m')             AS yr_month,
        ROUND(SUM(sales), 2)                         AS monthly_sales,
        ROUND(SUM(profit), 2)                        AS monthly_profit
    FROM sales
    GROUP BY YEAR(order_date), MONTH(order_date), DATE_FORMAT(order_date, '%Y-%m')
),
revenue_with_growth AS (
    SELECT
        yr,
        mth,
        yr_month,
        monthly_sales,
        monthly_profit,
        LAG(monthly_sales) OVER (ORDER BY yr, mth)  AS prev_month_sales,
        ROUND(
            ((monthly_sales - LAG(monthly_sales) OVER (ORDER BY yr, mth))
            / NULLIF(LAG(monthly_sales) OVER (ORDER BY yr, mth), 0)) * 100,
        2)                                           AS mom_growth_pct
    FROM monthly_revenue
)
SELECT * FROM revenue_with_growth
ORDER BY yr, mth;


-- ============================================================
-- SECTION 4: REGIONAL PERFORMANCE ANALYSIS
-- ============================================================

-- Revenue, Profit and Margin by Region with Rank
SELECT
    region,
    ROUND(SUM(sales), 2)                                    AS total_sales,
    ROUND(SUM(profit), 2)                                   AS total_profit,
    ROUND((SUM(profit) / SUM(sales)) * 100, 2)             AS profit_margin_pct,
    COUNT(DISTINCT customer_id)                             AS unique_customers,
    RANK() OVER (ORDER BY SUM(sales) DESC)                  AS revenue_rank
FROM sales
GROUP BY region
ORDER BY total_sales DESC;


-- Top 10 States by Revenue
SELECT
    state,
    region,
    ROUND(SUM(sales), 2)    AS total_sales,
    ROUND(SUM(profit), 2)   AS total_profit,
    COUNT(DISTINCT order_id) AS order_count,
    RANK() OVER (ORDER BY SUM(sales) DESC) AS state_rank
FROM sales
GROUP BY state, region
ORDER BY total_sales DESC
LIMIT 10;


-- ============================================================
-- SECTION 5: PRODUCT CATEGORY ANALYSIS
-- ============================================================

-- Revenue contribution % per Category
WITH total AS (
    SELECT SUM(sales) AS grand_total FROM sales
)
SELECT
    s.category,
    ROUND(SUM(s.sales), 2)                          AS category_sales,
    ROUND((SUM(s.sales) / t.grand_total) * 100, 2) AS revenue_contribution_pct,
    ROUND(SUM(s.profit), 2)                         AS category_profit,
    ROUND((SUM(s.profit) / SUM(s.sales)) * 100, 2) AS profit_margin_pct
FROM sales s, total t
GROUP BY s.category, t.grand_total
ORDER BY category_sales DESC;


-- Top 10 Sub-Categories by Revenue
SELECT
    sub_category,
    category,
    ROUND(SUM(sales), 2)                                        AS total_sales,
    ROUND(SUM(profit), 2)                                       AS total_profit,
    SUM(quantity)                                               AS units_sold,
    ROUND(AVG(discount) * 100, 2)                               AS avg_discount_pct,
    RANK() OVER (ORDER BY SUM(sales) DESC)                      AS sales_rank
FROM sales
GROUP BY sub_category, category
ORDER BY total_sales DESC
LIMIT 10;


-- ============================================================
-- SECTION 6: CUSTOMER SEGMENT ANALYSIS
-- ============================================================

SELECT
    segment,
    COUNT(DISTINCT customer_id)                             AS customer_count,
    COUNT(DISTINCT order_id)                                AS order_count,
    ROUND(SUM(sales), 2)                                    AS total_sales,
    ROUND(AVG(sales), 2)                                    AS avg_order_value,
    ROUND((SUM(profit) / SUM(sales)) * 100, 2)             AS profit_margin_pct
FROM sales
GROUP BY segment
ORDER BY total_sales DESC;


-- ============================================================
-- SECTION 7: CUSTOMER PURCHASE PATTERNS (Window Functions)
-- ============================================================

-- Customer Lifetime Value + Purchase Frequency
WITH customer_metrics AS (
    SELECT
        customer_id,
        customer_name,
        segment,
        region,
        COUNT(DISTINCT order_id)    AS total_orders,
        ROUND(SUM(sales), 2)        AS lifetime_value,
        ROUND(AVG(sales), 2)        AS avg_order_value,
        MIN(order_date)             AS first_purchase,
        MAX(order_date)             AS last_purchase,
        DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_tenure_days
    FROM sales
    GROUP BY customer_id, customer_name, segment, region
)
SELECT
    *,
    NTILE(4) OVER (ORDER BY lifetime_value DESC)   AS value_quartile,
    RANK() OVER (ORDER BY lifetime_value DESC)      AS value_rank
FROM customer_metrics
ORDER BY lifetime_value DESC
LIMIT 20;


-- ============================================================
-- SECTION 8: YEAR-OVER-YEAR COMPARISON (Subquery)
-- ============================================================

SELECT
    current_year.category,
    current_year.sales_2020,
    prior_year.sales_2019,
    ROUND(current_year.sales_2020 - prior_year.sales_2019, 2)          AS yoy_change,
    ROUND(
        ((current_year.sales_2020 - prior_year.sales_2019)
        / NULLIF(prior_year.sales_2019, 0)) * 100,
    2)                                                                  AS yoy_growth_pct
FROM
    (SELECT category, ROUND(SUM(sales),2) AS sales_2020
     FROM sales WHERE YEAR(order_date) = 2020
     GROUP BY category) AS current_year
JOIN
    (SELECT category, ROUND(SUM(sales),2) AS sales_2019
     FROM sales WHERE YEAR(order_date) = 2019
     GROUP BY category) AS prior_year
ON current_year.category = prior_year.category
ORDER BY yoy_change DESC;


-- ============================================================
-- SECTION 9: SEASONAL DEMAND ANALYSIS
-- ============================================================

SELECT
    QUARTER(order_date)            AS quarter,
    YEAR(order_date)               AS year,
    ROUND(SUM(sales), 2)           AS quarterly_sales,
    ROUND(SUM(profit), 2)          AS quarterly_profit,
    COUNT(DISTINCT order_id)       AS order_count,
    ROUND(AVG(sales), 2)           AS avg_order_value
FROM sales
GROUP BY QUARTER(order_date), YEAR(order_date)
ORDER BY year, quarter;


-- ============================================================
-- SECTION 10: SHIPPING & DISCOUNT IMPACT ANALYSIS
-- ============================================================

-- Profitability by Shipping Mode
SELECT
    ship_mode,
    COUNT(DISTINCT order_id)                           AS order_count,
    ROUND(SUM(sales), 2)                               AS total_sales,
    ROUND(SUM(profit), 2)                              AS total_profit,
    ROUND((SUM(profit) / SUM(sales)) * 100, 2)        AS profit_margin_pct,
    ROUND(AVG(DATEDIFF(ship_date, order_date)), 1)     AS avg_days_to_ship
FROM sales
GROUP BY ship_mode
ORDER BY total_sales DESC;


-- Discount Impact on Profit (Bucketised)
SELECT
    CASE
        WHEN discount = 0              THEN 'No Discount'
        WHEN discount BETWEEN 0 AND 0.1 THEN '1-10%'
        WHEN discount BETWEEN 0.1 AND 0.2 THEN '11-20%'
        WHEN discount BETWEEN 0.2 AND 0.3 THEN '21-30%'
        ELSE '30%+'
    END                                                AS discount_bucket,
    COUNT(*)                                           AS order_count,
    ROUND(AVG(profit), 2)                              AS avg_profit,
    ROUND(SUM(profit), 2)                              AS total_profit,
    ROUND(AVG(sales), 2)                               AS avg_order_value
FROM sales
GROUP BY discount_bucket
ORDER BY avg_profit DESC;
