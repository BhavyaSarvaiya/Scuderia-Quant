import pandas as pd
import os

hist  = pd.read_csv("data/historical/featured_historical_data.csv")
fred  = pd.read_csv("data/macro/fred_data.csv",  index_col="Date", parse_dates=True)
eia   = pd.read_csv("data/macro/eia_data.csv",   index_col="Date", parse_dates=True)

print("Historical:", hist.shape)
print("FRED:", fred.shape)
print("EIA:", eia.shape)

hist["Date"] = pd.to_datetime(hist["Date"], format="mixed", dayfirst=True)
hist = hist.sort_values(["Commodity", "Date"]).reset_index(drop=True)

macro  = pd.concat([fred, eia], axis=1)
merged = hist.merge(macro, left_on="Date", right_index=True, how="left")

macro_cols = list(fred.columns) + list(eia.columns)
merged[macro_cols] = merged[macro_cols].ffill()
merged = merged.sort_values(["Commodity", "Date"]).reset_index(drop=True)

print("\nAfter merge:", merged.shape)
print("Macro nulls:", merged[macro_cols].isnull().sum().sum())
print("Total nulls:", merged.isnull().sum().sum())
print("Gold sorted:", merged[merged["Commodity"]=="Gold"]["Date"].is_monotonic_increasing)

merged.to_csv("data/historical/featured_macro_merged_data.csv", index=False)
print("\nSaved featured_macro_merged_data.csv")
print("Done. Next step: run trend_model.py")