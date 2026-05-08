# Scuderia Quant 🏎️📈
**Integrated Real-Time Commodity Market Intelligence Engine**

> Institutional-grade commodity analytics for everyone — built entirely on free, open-source tools and public data.

---

## Overview

Scuderia Quant is an end-to-end commodity market monitoring and risk analysis system covering **6 major commodities**: Gold, Silver, Crude Oil, Natural Gas, Wheat, and Copper.

The goal: bridge the gap between retail investors and institutional traders who have Bloomberg terminals and quant teams. This system delivers trend forecasting, risk classification, volatility forecasting, and anomaly detection — all from publicly available data and no paid subscriptions.

---

## Features

| Module | Description |
|---|---|
| **Trend Prediction** | 5-day price direction forecasts using Logistic Regression, Gradient Boosting, and XGBoost |
| **Risk Classification** | Daily risk level (Low / Medium / High) via Random Forest and XGBoost |
| **Anomaly Detection** | Isolation Forest flags unusual market behaviour — identified March 2020 crash and 2022 energy spike unsupervised |
| **Volatility Forecasting** | GARCH(1,1) conditional volatility with 5/10/20-day confidence intervals |
| **Macro Context** | CPI, USD Index, Fed Funds Rate, GDP, Treasury Yield, Unemployment from FRED; Crude Oil inventory and Natural Gas storage from EIA |
| **Dashboard** | 8-tab interactive Streamlit dashboard with commodity selector, date range picker, and INR/USD toggle |

---

## Data Sources

| Source | Data | Volume |
|---|---|---|
| **Yahoo Finance** | Daily OHLCV (Jan 2010 – present) | ~24,000 rows |
| **Yahoo Finance** | 15-min intraday (last 60 trading days) | ~20,000 rows |
| **FRED API** | CPI, USD Index, Fed Funds Rate, GDP, Unemployment, 10Y Treasury | Monthly/Daily |
| **EIA API** | Weekly Crude Oil inventories, Natural Gas storage | Weekly |

**Total: 44,000+ rows of price and macro data per run.**

---

## Models

| Task | Commodity | Algorithm |
|---|---|---|
| Trend Direction | Gold | Logistic Regression |
| Trend Direction | Crude Oil, Natural Gas | Gradient Boosting |
| Trend Direction | Silver, Copper, Wheat | XGBoost |
| Risk Classification | Gold, Silver, Natural Gas | Random Forest |
| Risk Classification | Copper, Crude Oil, Wheat | XGBoost |
| Anomaly Detection | All 6 | Isolation Forest |
| Volatility Forecast | All 6 | GARCH(1,1) |

All supervised models use **walk-forward validation** — trained on past data only, tested on unseen future data. No data leakage.

---

## Project Structure

```
scuderia-quant/
├── download_macro_data.py        # Pulls FRED and EIA data
├── feature_engineering.py        # 30+ technical indicators per commodity
├── merge_macro_features.py       # Joins macro data into feature dataset
├── trend_model.py                # Trains trend direction classifiers
├── risk_classification_model.py  # Trains risk level classifiers
├── anomaly_detection_model.py    # Trains Isolation Forest models
├── garch_model.py                # Fits GARCH(1,1) volatility models
└── dashboard/                    # Streamlit app (8 tabs)
```

---

## Setup

```bash
git clone https://github.com/<your-username>/scuderia-quant.git
cd scuderia-quant
pip install -r requirements.txt
```

Set your API keys (free):
```bash
export FRED_API_KEY="your_fred_key"
export EIA_API_KEY="your_eia_key"
```

Run the pipeline in order:
```bash
python download_macro_data.py
python feature_engineering.py
python merge_macro_features.py
python trend_model.py
python risk_classification_model.py
python anomaly_detection_model.py
python garch_model.py
```

Launch dashboard:
```bash
streamlit run dashboard/app.py
```

---

## Results Highlights

- **Gold trend accuracy: 60.8%** on out-of-sample test set (Logistic Regression) — above the ~55% threshold for practical trading value
- **Isolation Forest** correctly flagged March 2020 COVID crash, Feb–Mar 2022 Russia-Ukraine energy spike, and January 2026 precious metals surge — with zero supervision
- **GARCH(1,1)** fitted across all 6 commodities; α + β between 0.979–0.998, confirming near-integrated volatility persistence; validated via AIC/BIC

---

## Technical Indicators Computed (30+)

Moving Averages (SMA/EMA), RSI (Wilder's EMA method), Bollinger Bands + Band Width, ATR, Stochastic Oscillator, MACD, Log Returns, Lagged Returns, Volume signals, Rolling correlations with USD Index and Treasury Yield, and more.

---

## Academic Context

Submitted as Bachelor of Science dissertation in Data Science  
**Somaiya Vidyavihar University**, Mumbai — 2025–26  
**Author:** Bhavya Bharat Sarvaiya  
**Guide:** Ms. Minal Dive, Dept. of Information Technology and Computer Science

---

## License

This project is open-source. Free to use, modify, and build upon with attribution.
