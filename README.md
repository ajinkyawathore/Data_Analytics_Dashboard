# 📊 Sales Performance & Revenue Analytics Dashboard

> **End-to-end sales analytics project** — SQL data extraction → Python EDA & ML forecasting → Power BI interactive dashboard

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)
![Excel](https://img.shields.io/badge/Excel-217346?style=flat-square&logo=microsoft-excel&logoColor=white)

---

## 🎯 Business Problem

A retail Super Store needed a centralised analytics solution to monitor sales performance across 5+ regions, identify revenue drivers, and forecast future demand to support quarterly planning.

## 🔍 Project Overview

This project covers the **complete data analytics lifecycle**:

1. **SQL** — Advanced querying to extract and analyse 10,000+ multi-year sales records
2. **Python** — Data cleaning, EDA, and regression-based sales forecasting
3. **Power BI** — Interactive executive dashboard with 8+ KPI cards and drill-through filters

---

## 📁 Repository Structure

```
Data_Analytics_Dashboard/
│
├── sql/
│   └── sales_analysis.sql          # Advanced SQL: CTEs, window functions, revenue analysis
│
├── python/
│   ├── data_cleaning.py            # Full ETL: missing values, duplicates, standardisation
│   ├── eda_analysis.py             # EDA: seasonal trends, product performance, correlations
│   └── sales_forecasting.py        # Regression model: 87% accuracy sales forecast
│
├── dashboard/
│   ├── Sales_dashboard.pbix        # Power BI dashboard file
│   ├── Sales_Dashboard_Insights.png
│   └── Sales_Forcasting.png
│
├── requirements.txt
└── README.md
```

---

## 📌 Key Results

| Metric | Result |
|--------|--------|
| Dataset Size | 10,000+ multi-year sales records |
| Data Quality Improvement | +25% (after cleaning) |
| Forecast Accuracy | **87%** (Linear Regression, Scikit-learn) |
| Top Product Contribution | 60% of total revenue from top categories |
| Dashboard KPIs | 8+ cards: Revenue, Profit Margin, MoM Growth, Regional Sales |
| Total Sales (Dashboard) | **$1.6M** across 4 regions |

---

## 🛠️ Technical Highlights

### SQL Analysis
- Window functions (`ROW_NUMBER`, `RANK`, `LAG`, `LEAD`) for MoM growth calculation
- CTEs for multi-step revenue trend analysis
- Subqueries for regional performance ranking
- `GROUP BY` + aggregations for product category breakdowns

### Python EDA
- **Pandas** — data cleaning, wrangling, feature engineering
- **Matplotlib & Seaborn** — 10+ visualisations: heatmaps, trend lines, bar charts
- Identified **seasonal demand peaks** in Q4 across all regions
- Top 3 product categories account for **60% of total revenue**

### Sales Forecasting (ML)
- Linear Regression model using **Scikit-learn**
- Features: month, region, product category, historical lag values
- Achieved **87% prediction accuracy** (R² = 0.87)
- Supports inventory planning and quarterly target-setting

### Power BI Dashboard
- 8+ KPI cards with conditional formatting
- Drill-through filters by Region, Category, Segment
- Monthly sales comparison (2019–2020)
- Sales forecasting visual with confidence bands

---

## 🖼️ Dashboard Preview

![Sales Dashboard](dashboard/Sales_Dashboard_Insights.png)

![Sales Forecasting](dashboard/Sales_Forcasting.png)

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ajinkyawathore/Data_Analytics_Dashboard.git
cd Data_Analytics_Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data cleaning
python python/data_cleaning.py

# 4. Run EDA
python python/eda_analysis.py

# 5. Run forecasting model
python python/sales_forecasting.py
```

---

## 📦 Requirements

```
pandas==2.1.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scikit-learn==1.3.0
openpyxl==3.1.0
```

---

## 👤 Author

**Ajinkya Wathore** — [LinkedIn](https://linkedin.com/in/ajinkya-wathore-a94405245) | [GitHub](https://github.com/ajinkyawathore)
