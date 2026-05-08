import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/historical/featured_macro_merged_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
df["Crude_Inv_Change"]   = df["Crude_Oil_Inventory"].diff()
df["NatGas_Stor_Change"] = df["NatGas_Storage"].diff()
os.makedirs("app/models", exist_ok=True)

USE_XGBOOST  = ["Copper", "Crude Oil", "Wheat"]
LABEL_MAP    = {"Low": 0, "Medium": 1, "High": 2}
REVERSE_MAP  = {0: "Low", 1: "Medium", 2: "High"}
results      = {}

for commodity in df["Commodity"].unique():
    print(f"\n--- {commodity} ---")
    c = df[df["Commodity"] == commodity].copy().reset_index(drop=True)

    # Features
    c["Lag_1"]          = c["Return_1d"].shift(1)
    c["Lag_2"]          = c["Return_1d"].shift(2)
    c["Lag_3"]          = c["Return_1d"].shift(3)
    c["MACD"]           = c["Close"].ewm(span=12, adjust=False).mean() - c["Close"].ewm(span=26, adjust=False).mean()
    c["Corr_USD"]       = c["Return_1d"].rolling(20).corr(c["USD_Index"])
    c["Corr_Treasury"]  = c["Return_1d"].rolling(20).corr(c["Treasury_10Y"])
    c["Dist_SMA50"]     = (c["Close"] - c["Close"].rolling(50).mean()) / c["Close"].rolling(50).mean()
    c["Dist_20d_High"]  = (c["Close"] - c["Close"].rolling(20).max()) / c["Close"].rolling(20).max()
    c["Drawdown_120"]   = (c["Close"] - c["Close"].rolling(120).max()) / c["Close"].rolling(120).max()

    FEATURES = [
        "Return_1d", "Return_5d", "ROC_10", "RSI_14", "Stoch_K", "Stoch_D",
        "Volatility_20", "Volatility_10", "BB_Width", "BB_Squeeze",
        "Volume_Ratio", "Volume_Spike", "CPI", "USD_Index", "Fed_Funds_Rate",
        "Treasury_10Y", "Unemployment", "Crude_Inv_Change", "NatGas_Stor_Change",
        "Lag_1", "Lag_2", "Lag_3", "MACD", "Corr_USD", "Corr_Treasury",
        "Dist_SMA50", "Dist_20d_High", "Drawdown_120",
    ]

    # Target — drawdown based risk label
    rolling_high    = c["Close"].rolling(60).max()
    c["Drawdown"]   = (c["Close"] - rolling_high) / rolling_high
    c["Risk_Label"] = np.where(c["Drawdown"] > -0.03, "Low",
                      np.where(c["Drawdown"] > -0.08, "Medium", "High"))

    c = c.dropna(subset=FEATURES + ["Risk_Label"])
    X, y  = c[FEATURES].values, c["Risk_Label"].values
    split = int(len(c) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    if commodity in USE_XGBOOST:
        model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7,
                              eval_metric="mlogloss", random_state=42, verbosity=0)
        model.fit(X_train, [LABEL_MAP[l] for l in y_train])
        pred = np.array([REVERSE_MAP[p] for p in model.predict(X_test)])
    else:
        model = RandomForestClassifier(n_estimators=500, max_depth=20,
                                       min_samples_leaf=2, max_features="sqrt",
                                       class_weight="balanced_subsample",
                                       random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred) * 100
    print(f"  Model: {'XGBoost' if commodity in USE_XGBOOST else 'Random Forest'} | Accuracy: {acc:.1f}%")
    print(classification_report(y_test, pred, zero_division=0))

    pickle.dump({"model": model, "features": FEATURES},
                open(f"app/models/rf_{commodity.replace(' ','_')}.pkl", "wb"))
    results[commodity] = round(acc, 1)

print("\n=== SUMMARY ===")
for c, acc in results.items():
    print(f"{c:15} | Risk Classification Accuracy: {acc}%")
print("\nDone. Next step: run anomaly_detection_model.py")