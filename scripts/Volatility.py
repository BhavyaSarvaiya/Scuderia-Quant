import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
from arch import arch_model

df = pd.read_csv("data/historical/featured_macro_merged_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)

os.makedirs("app/models", exist_ok=True)

CAP_COMMODITIES = ["Crude Oil", "Natural Gas"]
CAP_LIMIT       = 0.20

# Use last 2 years for all commodities — reflects current volatility regime
# not distorted by historical extreme events from 2020-2022
TRAIN_START = "2023-01-01"

results = {}

for commodity in df["Commodity"].unique():
    print(f"\n--- {commodity} ---")
    c = df[df["Commodity"] == commodity].copy().reset_index(drop=True)
    c = c[c["Date"] >= TRAIN_START]
    c = c.dropna(subset=["Log_Return"])

    returns = c["Log_Return"].copy()

    if commodity in CAP_COMMODITIES:
        before  = (returns.abs() > CAP_LIMIT).sum()
        returns = returns.clip(-CAP_LIMIT, CAP_LIMIT)
        print(f"  Capped {before} extreme returns at ±{CAP_LIMIT*100:.0f}%")

    print(f"  Training rows: {len(returns)}")
    returns_pct = returns * 100

    model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    res   = model.fit(disp="off", options={"maxiter": 500})

    print(f"  AIC: {res.aic:.2f} | alpha+beta: {res.params['alpha[1]']+res.params['beta[1]']:.4f}")

    cond_vol = np.sqrt(res.conditional_volatility.values)
    fore     = res.forecast(horizon=5, reindex=False)
    vol_arr  = np.sqrt(fore.variance.values[-1])
    print(f"  Last hist vol: {cond_vol[-1]:.3f}% | 1D forecast: {vol_arr[0]:.3f}% | 5D forecast: {vol_arr[-1]:.3f}%")

    pickle.dump({"model": res, "returns": returns_pct},
                open(f"app/models/garch_{commodity.replace(' ','_')}.pkl", "wb"))
    results[commodity] = {
        "alpha_beta": round(res.params["alpha[1]"] + res.params["beta[1]"], 4),
        "last_vol":   round(float(cond_vol[-1]), 3),
        "1d_vol":     round(float(vol_arr[0]), 3),
        "5d_vol":     round(float(vol_arr[-1]), 3),
    }

print("\n=== SUMMARY ===")
for c, v in results.items():
    diff = abs(v["1d_vol"] - v["last_vol"])
    flag = "OK" if diff < 0.5 else "CHECK"
    print(f"{c:15} | Last: {v['last_vol']}% | 1D: {v['1d_vol']}% | 5D: {v['5d_vol']}% | {flag}")

print("\nDone.")