import requests
import pandas as pd
from fredapi import Fred
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Safe for both direct run and Streamlit Cloud
try:
    import streamlit as st
    FRED_KEY = os.getenv("FRED_API_KEY") or st.secrets.get("FRED_API_KEY")
    EIA_KEY  = os.getenv("EIA_API_KEY")  or st.secrets.get("EIA_API_KEY")
except Exception:
    FRED_KEY = os.getenv("FRED_API_KEY")
    EIA_KEY  = os.getenv("EIA_API_KEY")

if not FRED_KEY:
    raise ValueError("FRED_API_KEY not set. Add to .env or Streamlit secrets.")
if not EIA_KEY:
    raise ValueError("EIA_API_KEY not set. Add to .env or Streamlit secrets.")

START = "2010-01-01"
END   = datetime.today().strftime("%Y-%m-%d")

# Create output folder
os.makedirs("data/macro", exist_ok=True)

print("Downloading FRED data...")

fred = Fred(api_key=FRED_KEY)

fred_series = {
    "CPI":            "CPIAUCSL",
    "USD_Index":      "DTWEXBGS",
    "Fed_Funds_Rate": "FEDFUNDS",
    "GDP":            "GDP",
    "Unemployment":   "UNRATE",
    "Treasury_10Y":   "DGS10",
}

fred_data = {}
for name, series_id in fred_series.items():
    series = fred.get_series(series_id, observation_start=START, observation_end=END)
    series = series.dropna()
    fred_data[name] = series
    print(f"  {name}: {len(series)} rows")

fred_df = pd.DataFrame(fred_data)
fred_df.index = pd.to_datetime(fred_df.index)

daily_index = pd.bdate_range(start=START, end=END)
fred_df = fred_df.reindex(daily_index).ffill().dropna(how="all")
fred_df.index.name = "Date"

fred_df.to_csv("data/macro/fred_data.csv")
print(f"\nfred_data.csv saved — {fred_df.shape[0]} rows, {fred_df.shape[1]} columns")
print(f"Nulls: {fred_df.isnull().sum().sum()}")

print("\nDownloading EIA data...")

eia_series = {
    "Crude_Oil_Inventory": "PET.WCRSTUS1.W",
    "NatGas_Storage":      "NG.NW2_EPG0_SWO_R48_BCF.W",
}

eia_data = {}
for name, series_id in eia_series.items():
    url = (
        f"https://api.eia.gov/v2/seriesid/{series_id}"
        f"?api_key={EIA_KEY}&data[]=value"
        f"&start={START}&end={END}"
        f"&sort[0][column]=period&sort[0][direction]=asc&length=5000"
    )
    resp = requests.get(url, timeout=30)
    rows = resp.json()["response"]["data"]
    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns = ["Date", name]
    df["Date"] = pd.to_datetime(df["Date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna().set_index("Date").sort_index()
    eia_data[name] = df[name]
    print(f"  {name}: {len(df)} rows")

eia_df = pd.DataFrame(eia_data)
eia_df.index = pd.to_datetime(eia_df.index)
eia_df = eia_df.reindex(daily_index).ffill().dropna(how="all")
eia_df.index.name = "Date"

eia_df.to_csv("data/macro/eia_data.csv")
print(f"\neia_data.csv saved — {eia_df.shape[0]} rows, {eia_df.shape[1]} columns")
print(f"Nulls: {eia_df.isnull().sum().sum()}")

print("\nDone. Next step: run feature_engineering.py")