import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/historical/featured_macro_merged_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
df["Crude_Inv_Change"]   = df["Crude_Oil_Inventory"].diff()
df["NatGas_Stor_Change"] = df["NatGas_Storage"].diff()
os.makedirs("app/models", exist_ok=True)

FEATURES = [
    "Return_1d", "Return_5d", "ROC_10", "RSI_14", "Stoch_K", "Stoch_D",
    "Volatility_20", "Volatility_10", "BB_Width", "BB_Squeeze",
    "Volume_Ratio", "Volume_Spike", "CPI", "USD_Index", "Fed_Funds_Rate",
    "Treasury_10Y", "Unemployment", "Crude_Inv_Change", "NatGas_Stor_Change",
    "Lag_1", "Lag_2", "Lag_3", "Lag_5",
]

# Best model per commodity based on evaluation
MODEL_MAP = {
    "Gold":        "logistic",
    "Silver":      "xgboost",
    "Copper":      "xgboost",
    "Wheat":       "xgboost",
    "Crude Oil":   "gbdt",
    "Natural Gas": "gbdt",
}

results = {}

for commodity in df["Commodity"].unique():
    print(f"\n--- {commodity} ---")
    c = df[df["Commodity"] == commodity].copy().reset_index(drop=True)

    c["Lag_1"] = c["Return_1d"].shift(1)
    c["Lag_2"] = c["Return_1d"].shift(2)
    c["Lag_3"] = c["Return_1d"].shift(3)
    c["Lag_5"] = c["Return_1d"].shift(5)

    c["future_return"] = c["Close"].shift(-5) / c["Close"] - 1
    c["target"]        = (c["future_return"] > 0).astype(int)
    c = c.iloc[:-5].dropna(subset=FEATURES)

    split   = int(len(c) * 0.8)
    X_train = c[FEATURES].values[:split]
    y_train = c["target"].values[:split].astype(int)
    X_test  = c[FEATURES].values[split:]
    y_test  = c["target"].values[split:].astype(int)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    mtype = MODEL_MAP[commodity]
    if mtype == "logistic":
        model = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)
    elif mtype == "gbdt":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    elif mtype == "xgboost":
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7,
                              eval_metric="logloss", random_state=42, verbosity=0)
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, pred) * 100
    print(f"  Model: {mtype} | Accuracy: {acc:.1f}%")

    pickle.dump(model,  open(f"app/models/lr_{commodity.replace(' ','_')}.pkl", "wb"))
    pickle.dump(scaler, open(f"app/models/lr_scaler_{commodity.replace(' ','_')}.pkl", "wb"))
    results[commodity] = round(acc, 1)

print("\n=== SUMMARY ===")
for c, acc in results.items():
    print(f"{c:15} | Direction Accuracy: {acc}%")
print("\nDone. Next step: run anomaly_detection_model.py")