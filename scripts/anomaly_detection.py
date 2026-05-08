import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/historical/featured_macro_merged_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)

os.makedirs("app/models", exist_ok=True)

FEATURES = [
    "Return_1d", "Return_5d", "ROC_10",
    "RSI_14", "Volatility_20", "Volatility_10",
    "BB_Width", "ATR_14", "Volume_Ratio",
    "Stoch_K", "Stoch_D",
]

results = {}

for commodity in df["Commodity"].unique():
    print(f"\n--- {commodity} ---")
    c = df[df["Commodity"] == commodity].copy().reset_index(drop=True)
    c = c.dropna(subset=FEATURES)

    X      = c[FEATURES].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_sc)

    c["Anomaly_Score"] = -model.score_samples(X_sc)
    c["Is_Anomaly"]    = (model.predict(X_sc) == -1).astype(int)

    n_anomalies = c["Is_Anomaly"].sum()
    anom_rate   = n_anomalies / len(c) * 100
    print(f"  Total rows: {len(c)}")
    print(f"  Anomalies detected: {n_anomalies} ({anom_rate:.1f}%)")

    top5 = c[c["Is_Anomaly"] == 1].nlargest(5, "Anomaly_Score")[["Date", "Close", "Return_1d", "Anomaly_Score"]]
    print("  Top 5 anomalies:")
    print(top5.to_string(index=False))

    pickle.dump({"model": model, "scaler": scaler, "features": FEATURES},
                open(f"app/models/iso_{commodity.replace(' ','_')}.pkl", "wb"))
    results[commodity] = n_anomalies

print("\n=== SUMMARY ===")
for c, n in results.items():
    print(f"{c:15} | Anomalies detected: {n}")
print("\nDone. Next step: run garch_model.py")