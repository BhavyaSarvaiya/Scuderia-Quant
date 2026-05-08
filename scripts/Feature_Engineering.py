import pandas as pd
import numpy as np
import os

hist  = pd.read_csv("data/historical/historical_data.csv")
intra = pd.read_csv("data/intraday/intraday_data.csv")

hist["Date"]  = pd.to_datetime(hist["Date"],  format="mixed", dayfirst=True)
intra["Date"] = pd.to_datetime(intra["Date"], format="mixed", dayfirst=True)

print("Historical:", hist.shape)
print("Intraday:",   intra.shape)

def add_features(df):
    df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
    out = []

    for commodity, g in df.groupby("Commodity", sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        c, h, l, v = g["Close"], g["High"], g["Low"], g["Volume"]

        g["Log_Return"]    = np.log(c / c.shift(1))
        g["Return_1d"]     = c.pct_change(1)
        g["Return_5d"]     = c.pct_change(5)
        g["ROC_10"]        = c.pct_change(10)

        g["SMA_20"]        = c.rolling(20).mean()
        g["SMA_50"]        = c.rolling(50).mean()
        g["EMA_20"]        = c.ewm(span=20, adjust=False).mean()

        g["Volatility_20"] = g["Log_Return"].rolling(20).std()
        g["Volatility_10"] = g["Log_Return"].rolling(10).std()

        tr                 = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        g["ATR_14"]        = tr.rolling(14).mean()

        bb_mid             = c.rolling(20).mean()
        bb_std             = c.rolling(20).std()
        g["BB_Upper"]      = bb_mid + 2 * bb_std
        g["BB_Lower"]      = bb_mid - 2 * bb_std
        g["BB_Width"]      = (g["BB_Upper"] - g["BB_Lower"]) / bb_mid
        g["BB_Squeeze"]    = (g["BB_Width"] < g["BB_Width"].rolling(50).mean()).astype(int)

        gain               = c.diff().clip(lower=0)
        loss               = (-c.diff()).clip(lower=0)
        g["RSI_14"]        = 100 - (100 / (1 + gain.ewm(alpha=1/14, adjust=False).mean() /
                                           (loss.ewm(alpha=1/14, adjust=False).mean() + 1e-10)))

        low14              = l.rolling(14).min()
        high14             = h.rolling(14).max()
        g["Stoch_K"]       = 100 * (c - low14) / (high14 - low14 + 1e-10)
        g["Stoch_D"]       = g["Stoch_K"].rolling(3).mean()

        g["Volume_SMA_20"] = v.rolling(20).mean()
        g["Volume_Ratio"]  = v / (g["Volume_SMA_20"] + 1e-10)
        g["Volume_Spike"]  = (g["Volume_Ratio"] > 2.0).astype(int)

        g["Forward_Return_5d"]  = c.shift(-5)  / c - 1
        g["Forward_Return_10d"] = c.shift(-10) / c - 1
        g["Direction_5d"]       = (g["Forward_Return_5d"] > 0).astype(int)

        p33 = g["Volatility_20"].quantile(0.33)
        p67 = g["Volatility_20"].quantile(0.67)
        g["Risk_Label"] = pd.cut(g["Volatility_20"],
                                 bins=[-np.inf, p33, p67, np.inf],
                                 labels=["Low", "Medium", "High"])
        out.append(g)

    return pd.concat(out, ignore_index=True).sort_values(["Commodity", "Date"]).reset_index(drop=True)

print("\nProcessing historical...")
hist_out = add_features(hist)
hist_out.to_csv("data/historical/featured_historical_data.csv", index=False)
print(f"Saved — {hist_out.shape} | Sorted: {hist_out[hist_out['Commodity']=='Gold']['Date'].is_monotonic_increasing}")

print("\nProcessing intraday...")
intra_out = add_features(intra)
intra_out.to_csv("data/intraday/featured_intraday_data.csv", index=False)
print(f"Saved — {intra_out.shape}")

print("\nDone. Next step: run merge_macro_features.py")