import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
import warnings
import os
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Scuderia Quant", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
COMMODITIES = ["Gold", "Silver", "Crude Oil", "Natural Gas", "Wheat", "Copper"]
TICKERS     = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F",
               "Natural Gas": "NG=F", "Wheat": "ZW=F", "Copper": "HG=F"}
USD_UNITS   = {"Gold": "USD/oz", "Silver": "USD/oz", "Crude Oil": "USD/bbl",
               "Natural Gas": "USD/MMBtu", "Wheat": "USD/bu", "Copper": "USD/lb"}
INR_UNITS   = {"Gold": "INR/10g", "Silver": "INR/kg", "Crude Oil": "INR/bbl",
               "Natural Gas": "INR/MMBtu", "Wheat": "INR/quintal", "Copper": "INR/kg"}
# Fetch live USD/INR rate — fallback to 92.27 if fetch fails
try:
    _inr_data = yf.download("INR=X", period="1d", interval="1m", progress=False)
    if isinstance(_inr_data.columns, pd.MultiIndex):
        _inr_data.columns = _inr_data.columns.get_level_values(0)
    USD_INR = float(_inr_data["Close"].iloc[-1]) if not _inr_data.empty else 92.27
except:
    USD_INR = 92.27
INR_CONV    = {
    "Gold":        lambda p: p * USD_INR / 3.11035,
    "Silver":      lambda p: p * USD_INR * 32.1507,
    "Crude Oil":   lambda p: p * USD_INR,
    "Natural Gas": lambda p: p * USD_INR,
    "Wheat":       lambda p: p * USD_INR * 3.6744,
    "Copper":      lambda p: p * USD_INR * 2.20462,
}
MODEL_DIR   = "app/models"
HIST_PATH   = "data/historical/featured_macro_merged_data.csv"
CHART_CFG   = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d",
        "autoScale2d","resetScale2d","hoverClosestCartesian",
        "hoverCompareCartesian","toggleSpikelines","toImage"
    ],
    "displaylogo": False
}
TREND_FEATURES = [
    "Return_1d","Return_5d","ROC_10","RSI_14","Stoch_K","Stoch_D",
    "Volatility_20","Volatility_10","BB_Width","BB_Squeeze",
    "Volume_Ratio","Volume_Spike","CPI","USD_Index","Fed_Funds_Rate",
    "Treasury_10Y","Unemployment","Crude_Inv_Change","NatGas_Stor_Change",
    "Lag_1","Lag_2","Lag_3","Lag_5",
]
RF_FEATURES = [
    "Return_1d","Return_5d","ROC_10","RSI_14","Stoch_K","Stoch_D",
    "Volatility_20","Volatility_10","BB_Width","BB_Squeeze",
    "Volume_Ratio","Volume_Spike","CPI","USD_Index","Fed_Funds_Rate",
    "Treasury_10Y","Unemployment","Crude_Inv_Change","NatGas_Stor_Change",
    "Lag_1","Lag_2","Lag_3","MACD","Corr_USD","Corr_Treasury",
    "Dist_SMA50","Dist_20d_High","Drawdown_120",
]
ISO_FEATURES = [
    "Return_1d","Return_5d","ROC_10","RSI_14",
    "Volatility_20","Volatility_10","BB_Width",
    "ATR_14","Volume_Ratio","Stoch_K","Stoch_D",
]
MACRO_COLS = [
    "CPI","USD_Index","Fed_Funds_Rate","GDP",
    "Unemployment","Treasury_10Y","Crude_Oil_Inventory","NatGas_Storage"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(price, currency, commodity):
    if currency == "INR ₹":
        return f"₹{INR_CONV[commodity](price):,.2f}"
    return f"${price:,.2f}"

def get_unit(currency, commodity):
    return INR_UNITS[commodity] if currency == "INR ₹" else USD_UNITS[commodity]

def to_display(series, currency, commodity):
    if currency == "INR ₹":
        return series.apply(INR_CONV[commodity])
    return series

def compute_features_for_group(g):
    """Compute all technical features for a sorted commodity group."""
    g = g.copy().reset_index(drop=True)
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
    tr                 = pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    g["ATR_14"]        = tr.rolling(14).mean()
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std()
    g["BB_Upper"]      = bb_mid + 2*bb_std
    g["BB_Lower"]      = bb_mid - 2*bb_std
    g["BB_Width"]      = (g["BB_Upper"] - g["BB_Lower"]) / bb_mid
    g["BB_Squeeze"]    = (g["BB_Width"] < g["BB_Width"].rolling(50).mean()).astype(int)
    delta              = c.diff()
    gain               = delta.clip(lower=0)
    loss               = (-delta).clip(lower=0)
    avg_gain           = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss           = loss.ewm(alpha=1/14, adjust=False).mean()
    g["RSI_14"]        = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
    low14              = l.rolling(14).min()
    high14             = h.rolling(14).max()
    g["Stoch_K"]       = 100 * (c - low14) / (high14 - low14 + 1e-10)
    g["Stoch_D"]       = g["Stoch_K"].rolling(3).mean()
    g["Volume_SMA_20"] = v.rolling(20).mean()
    g["Volume_Ratio"]  = v / (g["Volume_SMA_20"] + 1e-10)
    g["Volume_Spike"]  = (g["Volume_Ratio"] > 2.0).astype(int)
    g["Forward_Return_5d"]  = c.shift(-5) / c - 1
    g["Forward_Return_10d"] = c.shift(-10) / c - 1
    g["Direction_5d"]       = (g["Forward_Return_5d"] > 0).astype(int)
    p33 = g["Volatility_20"].quantile(0.33)
    p67 = g["Volatility_20"].quantile(0.67)
    g["Risk_Label"] = pd.cut(g["Volatility_20"],
                             bins=[-np.inf, p33, p67, np.inf],
                             labels=["Low","Medium","High"])
    return g

def update_historical_data():
    """
    Fetch new daily data from yfinance for all commodities,
    append to existing CSV without duplicates, return updated dataframe.
    """
    df = pd.read_csv(HIST_PATH)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df.sort_values(["Commodity","Date"]).reset_index(drop=True)

    last_date = df["Date"].max()
    today     = pd.Timestamp(datetime.today().date())

    if last_date >= today:
        return df  # Already up to date

    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = today.strftime("%Y-%m-%d")

    new_rows = []
    for commodity in COMMODITIES:
        ticker = TICKERS[commodity]
        try:
            raw = yf.download(ticker, start=fetch_start, end=fetch_end,
                              interval="1d", progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if raw.empty:
                continue
            raw = raw.reset_index()
            raw.columns = [c if c != "index" else "Date" for c in raw.columns]
            if "Date" not in raw.columns and "Datetime" in raw.columns:
                raw = raw.rename(columns={"Datetime":"Date"})
            raw["Date"]      = pd.to_datetime(raw["Date"]).dt.date.astype(str)
            raw["Date"]      = pd.to_datetime(raw["Date"])
            raw["Commodity"] = commodity
            new_rows.append(raw[["Date","Commodity","Open","High","Low","Close","Volume"]])
        except Exception:
            continue

    if not new_rows:
        return df  # No new data available

    new_df = pd.concat(new_rows, ignore_index=True)

    # Get full history per commodity for feature computation context
    all_groups = []
    for commodity in COMMODITIES:
        hist_group = df[df["Commodity"] == commodity].copy()
        new_group  = new_df[new_df["Commodity"] == commodity].copy() if commodity in new_df["Commodity"].values else pd.DataFrame()

        if new_group.empty:
            all_groups.append(hist_group)
            continue

        # Get last macro values to forward fill into new rows
        last_macro = hist_group[MACRO_COLS].iloc[-1].to_dict()
        for col in MACRO_COLS:
            new_group[col] = last_macro.get(col, np.nan)

        # Combine old + new for feature computation (need history for rolling windows)
        base_cols   = ["Date","Commodity","Open","High","Low","Close","Volume"] + MACRO_COLS
        hist_simple = hist_group[base_cols].copy()
        combined    = pd.concat([hist_simple, new_group], ignore_index=True)
        combined    = combined.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

        # Compute features on full combined data
        featured = compute_features_for_group(combined)

        # Add macro-derived columns
        featured["Crude_Inv_Change"]   = featured["Crude_Oil_Inventory"].diff() if "Crude_Oil_Inventory" in featured.columns else 0
        featured["NatGas_Stor_Change"] = featured["NatGas_Storage"].diff()      if "NatGas_Storage"      in featured.columns else 0

        # Keep only new rows (those not in original hist)
        new_dates    = new_group["Date"].values
        featured_new = featured[featured["Date"].isin(new_dates)]

        # Append new featured rows to original hist_group
        updated = pd.concat([hist_group, featured_new], ignore_index=True)
        updated = updated.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
        all_groups.append(updated)

    final_df = pd.concat(all_groups, ignore_index=True)
    final_df = final_df.sort_values(["Commodity","Date"]).reset_index(drop=True)
    final_df.to_csv(HIST_PATH, index=False)
    return final_df

def prep_rf_cols(cdf):
    c = cdf.copy()
    c["Lag_1"]         = c["Return_1d"].shift(1)
    c["Lag_2"]         = c["Return_1d"].shift(2)
    c["Lag_3"]         = c["Return_1d"].shift(3)
    c["MACD"]          = c["Close"].ewm(span=12,adjust=False).mean() - c["Close"].ewm(span=26,adjust=False).mean()
    c["Corr_USD"]      = c["Return_1d"].rolling(20).corr(c["USD_Index"])
    c["Corr_Treasury"] = c["Return_1d"].rolling(20).corr(c["Treasury_10Y"])
    c["Dist_SMA50"]    = (c["Close"] - c["Close"].rolling(50).mean()) / c["Close"].rolling(50).mean()
    c["Dist_20d_High"] = (c["Close"] - c["Close"].rolling(20).max()) / c["Close"].rolling(20).max()
    c["Drawdown_120"]  = (c["Close"] - c["Close"].rolling(120).max()) / c["Close"].rolling(120).max()
    c["Crude_Inv_Change"]   = c["Crude_Oil_Inventory"].diff()
    c["NatGas_Stor_Change"] = c["NatGas_Storage"].diff()
    return c

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    for c in COMMODITIES:
        k = c.replace(" ","_")
        m[c] = {
            "lr":    pickle.load(open(f"{MODEL_DIR}/lr_{k}.pkl",        "rb")),
            "lr_sc": pickle.load(open(f"{MODEL_DIR}/lr_scaler_{k}.pkl", "rb")),
            "rf":    pickle.load(open(f"{MODEL_DIR}/rf_{k}.pkl",        "rb")),
            "iso":   pickle.load(open(f"{MODEL_DIR}/iso_{k}.pkl",       "rb")),
            "garch": pickle.load(open(f"{MODEL_DIR}/garch_{k}.pkl",     "rb")),
        }
    return m

@st.cache_data(ttl=300)
def get_live_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return None, None
        return float(data["Close"].iloc[-1]), float(data["Close"].iloc[0])
    except:
        return None, None

@st.cache_data(ttl=300)
def get_intraday(ticker, date_str):
    try:
        data = yf.download(ticker, period="60d", interval="15m", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)
        sel = pd.Timestamp(date_str).date()
        day = data[data.index.date == sel]
        return day if not day.empty else None
    except:
        return None

# ── Update + load data ────────────────────────────────────────────────────────
models = load_models()

with st.spinner("Updating market data..."):
    try:
        df = update_historical_data()
    except Exception:
        df = pd.read_csv(HIST_PATH)
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)

df["Crude_Inv_Change"]   = df.groupby("Commodity")["Crude_Oil_Inventory"].diff()
df["NatGas_Stor_Change"] = df.groupby("Commodity")["NatGas_Storage"].diff()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("Scuderia Quant")
st.sidebar.markdown("---")
commodity = st.sidebar.selectbox("Select Commodity", COMMODITIES)
currency  = st.sidebar.radio("Currency", ["USD $", "INR ₹"])
st.sidebar.markdown("---")
st.sidebar.caption("Data Sources")
st.sidebar.caption("• Yahoo Finance — Price Data")
st.sidebar.caption("• FRED API — Macro Indicators")
st.sidebar.caption("• EIA API — Energy Inventory")
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ── Per-commodity data ────────────────────────────────────────────────────────
ticker = TICKERS[commodity]
unit   = get_unit(currency, commodity)
cdf    = df[df["Commodity"] == commodity].sort_values("Date").reset_index(drop=True)

live_price, open_price = get_live_price(ticker)
if live_price:
    curr_price = live_price
    chg_pct    = (live_price - open_price) / open_price * 100 if open_price else 0.0
else:
    cdf_valid  = cdf.dropna(subset=["Close","Return_1d"])
    curr_price = float(cdf_valid["Close"].iloc[-1])
    chg_pct    = float(cdf_valid["Return_1d"].iloc[-1]) * 100

last_data_date = cdf["Date"].max().strftime("%Y-%m-%d")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("Scuderia Quant")
st.markdown("---")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Current Price", fmt(curr_price, currency, commodity), f"{chg_pct:+.2f}%")
c2.metric("52W High",      fmt(float(cdf["Close"].tail(252).max()), currency, commodity))
c3.metric("52W Low",       fmt(float(cdf["Close"].tail(252).min()), currency, commodity))
c4.metric("Unit",          unit)
c5.metric("Last Updated",  last_data_date)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
    "Overview","Trend Analysis","Risk Classification",
    "Volatility Forecast","Anomaly Detection",
    "Macro & EIA","Correlation","Intraday"
])

# ══ TAB 1 — OVERVIEW ══════════════════════════════════════════════════════════
with t1:
    st.subheader("All Commodities — Snapshot")
    rows = []
    for com in COMMODITIES:
        cd       = df[df["Commodity"] == com].sort_values("Date")
        cd_valid = cd.dropna(subset=["Return_1d","RSI_14","Volatility_20"])
        if cd_valid.empty:
            continue
        last_row = cd_valid.iloc[-1]
        # Use live price from yfinance for consistency with header
        lp, op = get_live_price(TICKERS[com])
        live_close = lp if lp else float(cd.dropna(subset=["Close"])["Close"].iloc[-1])
        live_ret   = (lp - op) / op * 100 if (lp and op) else float(last_row["Return_1d"]) * 100
        rows.append({
            "Commodity":      com,
            "Latest Price":   fmt(live_close, currency, com),
            "1D Return":      f"{live_ret:+.2f}%",
            "RSI-14":         f"{float(last_row['RSI_14']):.1f}",
            "20D Volatility": f"{float(last_row['Volatility_20'])*100:.2f}%",
            "Unit":           get_unit(currency, com),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Normalised Returns — All Commodities (Last 2 Years)")
    colors = ["#e6a817","#999","#2196f3","#00bcd4","#4caf50","#ff7043"]
    fig    = go.Figure()
    for i, com in enumerate(COMMODITIES):
        cd   = df[df["Commodity"] == com].sort_values("Date").tail(504).dropna(subset=["Close"])
        cd   = cd[cd["Date"].dt.dayofweek < 5]  # weekdays only
        norm = cd["Close"] / cd["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(x=cd["Date"], y=norm, name=com,
                                  line=dict(color=colors[i], width=1.5)))
    fig.update_layout(height=380, xaxis_title="Date",
                      yaxis_title="Normalised Price (Base = 100)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

# ══ TAB 2 — TREND ANALYSIS ════════════════════════════════════════════════════
with t2:
    st.subheader(f"Trend Analysis — {commodity}")

    tcdf = cdf.copy()
    tcdf["Lag_1"] = tcdf["Return_1d"].shift(1)
    tcdf["Lag_2"] = tcdf["Return_1d"].shift(2)
    tcdf["Lag_3"] = tcdf["Return_1d"].shift(3)
    tcdf["Lag_5"] = tcdf["Return_1d"].shift(5)
    tcdf["Crude_Inv_Change"]   = tcdf["Crude_Oil_Inventory"].diff()
    tcdf["NatGas_Stor_Change"] = tcdf["NatGas_Storage"].diff()
    last = tcdf.dropna(subset=TREND_FEATURES).iloc[-1]

    X_input = last[TREND_FEATURES].values.reshape(1,-1).astype(float)
    X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc   = models[commodity]["lr_sc"].transform(X_input)
    lr     = models[commodity]["lr"]
    if hasattr(lr, "predict_proba"):
        proba   = lr.predict_proba(X_sc)[0]
        prob_up = proba[list(lr.classes_).index(1)] * 100 if 1 in lr.classes_ else 50.0
    else:
        prob_up = 50.0
    signal = "UP" if prob_up >= 50 else "DOWN"

    c1,c2,c3 = st.columns(3)
    c1.metric("5-Day Signal", signal)
    c2.metric("Confidence",   f"{prob_up:.1f}%")
    c3.metric("RSI-14",       f"{float(last['RSI_14']):.1f}")

    plot_df = cdf.tail(252).dropna(subset=["Close","SMA_20","SMA_50"])
    p_close = to_display(plot_df["Close"], currency, commodity)
    p_sma20 = to_display(plot_df["SMA_20"], currency, commodity)
    p_sma50 = to_display(plot_df["SMA_50"], currency, commodity)
    y_min   = float(min(p_close.min(), p_sma20.min(), p_sma50.min())) * 0.98
    y_max   = float(max(p_close.max(), p_sma20.max(), p_sma50.max())) * 1.02
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=p_close,
                              name="Price", line=dict(color="#2196f3", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=p_sma20,
                              name="SMA 20", line=dict(color="#ff7043", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=p_sma50,
                              name="SMA 50", line=dict(color="#4caf50", width=1.5, dash="dash")))
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title=unit,
                      yaxis=dict(range=[y_min, y_max]),
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    rsi_df = plot_df.dropna(subset=["RSI_14"])
    fig2 = go.Figure()
    fig2.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig2.add_trace(go.Scatter(x=rsi_df["Date"], y=rsi_df["RSI_14"], name="RSI-14",
                               line=dict(color="#9c27b0", width=1.8)))
    fig2.update_layout(height=220, yaxis=dict(range=[0,100]),
                       xaxis_title="Date", yaxis_title="RSI", margin=dict(t=10))
    st.plotly_chart(fig2, width="stretch", config=CHART_CFG)

# ══ TAB 3 — RISK CLASSIFICATION ═══════════════════════════════════════════════
with t3:
    st.subheader(f"Risk Classification — {commodity}")

    rf_data = models[commodity]["rf"]
    rf_mod  = rf_data["model"]
    rf_feat = rf_data["features"]
    rf_cdf  = prep_rf_cols(cdf)
    last_rf = rf_cdf.dropna(subset=rf_feat).iloc[-1]
    X_rf    = last_rf[rf_feat].values.reshape(1,-1).astype(float)
    X_rf    = np.nan_to_num(X_rf, nan=0.0, posinf=0.0, neginf=0.0)

    risk_label_raw = rf_mod.predict(X_rf)[0]
    # Convert numeric labels from XGBoost back to string
    num_to_label = {0: "Low", 1: "Medium", 2: "High"}
    risk_label   = num_to_label.get(risk_label_raw, str(risk_label_raw))
    if hasattr(rf_mod, "predict_proba"):
        classes   = list(rf_mod.classes_)
        try:
            risk_conf = rf_mod.predict_proba(X_rf)[0][list(classes).index(risk_label_raw)] * 100
        except:
            risk_conf = 0.0
    else:
        risk_conf = 0.0

    c1,c2 = st.columns(2)
    c1.metric("Risk Label",  risk_label)
    c2.metric("Confidence",  f"{risk_conf:.1f}%")

    rc_map = {"Low": "green", "Medium": "orange", "High": "red"}
    re_map = {"Low": "🟢",    "Medium": "🟡",      "High": "🔴"}
    rc = rc_map.get(risk_label, "gray")
    re = re_map.get(risk_label, "")
    st.markdown(f"""
    <div style="background-color:rgba(128,128,128,0.1);border-left:5px solid {rc};
    padding:12px 16px;border-radius:4px;margin:8px 0">
    <b style="font-size:16px">{re} Current Risk: {risk_label}</b><br>
    <span style="font-size:13px;color:gray">Confidence: {risk_conf:.1f}% &nbsp;|&nbsp;
    Based on 60-day drawdown analysis &nbsp;|&nbsp; As of {last_data_date}</span>
    </div>
    """, unsafe_allow_html=True)

    rf_cdf["Drawdown"] = (rf_cdf["Close"] - rf_cdf["Close"].rolling(60).max()) / \
                          rf_cdf["Close"].rolling(60).max() * 100
    fig = go.Figure()
    fig.add_hrect(y0=-3, y1=5,   fillcolor="green",  opacity=0.10, line_width=0,
                  annotation_text="LOW RISK",    annotation_position="top right")
    fig.add_hrect(y0=-8, y1=-3,  fillcolor="orange", opacity=0.12, line_width=0,
                  annotation_text="MEDIUM RISK", annotation_position="top right")
    fig.add_hrect(y0=-60,y1=-8,  fillcolor="red",    opacity=0.10, line_width=0,
                  annotation_text="HIGH RISK",   annotation_position="top right")
    fig.add_hline(y=-3, line_dash="dash", line_color="orange", line_width=1)
    fig.add_hline(y=-8, line_dash="dash", line_color="red",    line_width=1)
    fig.add_trace(go.Scatter(x=rf_cdf["Date"], y=rf_cdf["Drawdown"],
                              fill="tozeroy", name="Drawdown %",
                              line=dict(color="#1565c0", width=1.5),
                              fillcolor="rgba(21,101,192,0.15)"))
    fig.update_layout(height=320, xaxis_title="Date",
                      yaxis_title="Drawdown from 60D High (%)",
                      yaxis=dict(range=[-60,5]), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    bb_df   = cdf.tail(252).dropna(subset=["BB_Upper","BB_Lower","Close"])
    bb_up   = to_display(bb_df["BB_Upper"], currency, commodity)
    bb_lo   = to_display(bb_df["BB_Lower"], currency, commodity)
    bb_cl   = to_display(bb_df["Close"],    currency, commodity)
    bb_ymin = float(bb_lo.min()) * 0.98
    bb_ymax = float(bb_up.max()) * 1.02
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bb_df["Date"], y=bb_up,
                               name="Upper Band", line=dict(color="lightblue", width=1)))
    fig2.add_trace(go.Scatter(x=bb_df["Date"], y=bb_lo,
                               name="Lower Band", fill="tonexty",
                               line=dict(color="lightblue", width=1),
                               fillcolor="rgba(173,216,230,0.1)"))
    fig2.add_trace(go.Scatter(x=bb_df["Date"], y=bb_cl,
                               name="Price", line=dict(color="#2196f3", width=2)))
    fig2.update_layout(height=300, xaxis_title="Date", yaxis_title=unit,
                       yaxis=dict(range=[bb_ymin, bb_ymax]),
                       legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig2, width="stretch", config=CHART_CFG)

# ══ TAB 4 — VOLATILITY FORECAST ═══════════════════════════════════════════════
with t4:
    st.subheader(f"GARCH(1,1) Volatility Forecast — {commodity}")

    horizon = st.selectbox("Forecast Horizon", ["5 Days","10 Days","20 Days"])
    h       = int(horizon.split()[0])
    res     = models[commodity]["garch"]["model"]
    fore    = res.forecast(horizon=h, reindex=False)
    vol_arr = np.sqrt(fore.variance.values[-1])
    alpha   = res.params["alpha[1]"]
    beta    = res.params["beta[1]"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AIC",            f"{res.aic:.1f}")
    c2.metric("BIC",            f"{res.bic:.1f}")
    c3.metric("alpha + beta",   f"{alpha+beta:.4f}")
    c4.metric(f"{h}D Forecast", f"{vol_arr[-1]:.2f}%")

    cond_vol   = np.sqrt(res.conditional_volatility.values)
    n_hist     = min(252, len(cond_vol))
    hist_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in cdf["Date"].values[-n_hist:]]
    hist_vol   = list(cond_vol[-n_hist:])

    last_date = pd.Timestamp(hist_dates[-1])
    fc_dates  = [d.strftime("%Y-%m-%d") for d in
                 pd.bdate_range(start=last_date + timedelta(days=1), periods=h)]
    fc_vol    = list(vol_arr)
    fc_upper  = [v * 1.25 for v in fc_vol]
    fc_lower  = [v * 0.75 for v in fc_vol]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_vol, name="Historical Volatility",
                              line=dict(color="#2196f3", width=2)))
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_upper,
                              line=dict(width=0), showlegend=False, mode="lines"))
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_lower, name="95% CI",
                              fill="tonexty", line=dict(width=0),
                              fillcolor="rgba(255,152,0,0.25)", mode="lines"))
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_vol, name=f"{h}-Day Forecast",
                              line=dict(color="#ff9800", width=2.5, dash="dash"),
                              mode="lines+markers", marker=dict(size=8, color="#ff9800")))
    fig.update_layout(height=420, xaxis=dict(title="Date", tickangle=-30, nticks=15),
                      yaxis_title="Volatility (%)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    rows = [{"Day": f"Day {i+1}", "Date": d,
             "Forecast Vol (%)": f"{v:.3f}",
             "Upper CI (%)": f"{u:.3f}", "Lower CI (%)": f"{l:.3f}"}
            for i,(d,v,u,l) in enumerate(zip(fc_dates,fc_vol,fc_upper,fc_lower))]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ══ TAB 5 — ANOMALY DETECTION ═════════════════════════════════════════════════
with t5:
    st.subheader(f"Anomaly Detection — {commodity}")

    iso_data = models[commodity]["iso"]
    iso_mod  = iso_data["model"]
    iso_sc   = iso_data["scaler"]
    iso_feat = iso_data["features"]

    adf   = cdf.dropna(subset=iso_feat).copy()
    iso_vals = np.nan_to_num(adf[iso_feat].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X_iso = iso_sc.transform(iso_vals)
    adf["Anomaly_Score"] = -iso_mod.score_samples(X_iso)
    adf["Is_Anomaly"]    = (iso_mod.predict(X_iso) == -1).astype(int)

    n_anom = int(adf["Is_Anomaly"].sum())
    latest = "Anomaly" if adf["Is_Anomaly"].iloc[-1] == 1 else "Normal"

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Anomalies", n_anom)
    c2.metric("Anomaly Rate",    f"{n_anom/len(adf)*100:.1f}%")
    c3.metric("Latest Status",   latest)

    anom   = adf[adf["Is_Anomaly"] == 1]
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=adf["Date"],
                              y=to_display(adf["Close"], currency, commodity),
                              name="Price", line=dict(color="#2196f3", width=1.5)))
    fig.add_trace(go.Scatter(x=anom["Date"],
                              y=to_display(anom["Close"], currency, commodity),
                              mode="markers", name="Anomaly",
                              marker=dict(color="red", size=8,
                                          line=dict(width=1.5, color="darkred"))))
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title=unit,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    st.subheader("Top 10 Anomalies")
    top10 = adf[adf["Is_Anomaly"]==1].nlargest(10,"Anomaly_Score")[
        ["Date","Close","Return_1d","Anomaly_Score"]].copy()
    top10["Date"]          = top10["Date"].dt.strftime("%Y-%m-%d")
    top10["Close"]         = top10["Close"].apply(lambda x: fmt(x, currency, commodity))
    top10["Return_1d"]     = top10["Return_1d"].apply(lambda x: f"{x*100:+.2f}%")
    top10["Anomaly_Score"] = top10["Anomaly_Score"].apply(lambda x: f"{x:.4f}")
    st.dataframe(top10, width="stretch", hide_index=True)

# ══ TAB 6 — MACRO & EIA ═══════════════════════════════════════════════════════
with t6:
    st.subheader(f"Macroeconomic Context — {commodity}")
    last = cdf.iloc[-1]

    c1,c2,c3 = st.columns(3)
    c1.metric("CPI",            f"{float(last['CPI']):.1f}")
    c2.metric("USD Index",      f"{float(last['USD_Index']):.1f}")
    c3.metric("Fed Funds Rate", f"{float(last['Fed_Funds_Rate']):.2f}%")
    c1,c2,c3 = st.columns(3)
    c1.metric("10Y Treasury",   f"{float(last['Treasury_10Y']):.2f}%")
    c2.metric("Unemployment",   f"{float(last['Unemployment']):.1f}%")
    c3.metric("GDP",            f"{float(last['GDP']):.1f}")

    st.subheader(f"USD Index vs {commodity} Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["Date"],
                              y=to_display(cdf["Close"], currency, commodity),
                              name=commodity, line=dict(color="#2196f3", width=2)))
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["USD_Index"], name="USD Index",
                              line=dict(color="#ff7043", width=1.5, dash="dash"),
                              yaxis="y2"))
    fig.update_layout(height=320, xaxis_title="Date",
                      yaxis=dict(title=unit),
                      yaxis2=dict(title="USD Index", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    st.subheader("EIA Inventory")
    if commodity in ["Crude Oil", "Natural Gas"]:
        inv_col  = "Crude_Oil_Inventory" if commodity == "Crude Oil" else "NatGas_Storage"
        unit_lbl = "Million Barrels"     if commodity == "Crude Oil" else "Billion Cubic Feet"
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=cdf["Date"], y=cdf[inv_col],
                               name="Inventory", marker_color="rgba(33,150,243,0.5)"))
        fig2.add_trace(go.Scatter(x=cdf["Date"],
                                   y=to_display(cdf["Close"], currency, commodity),
                                   name=f"{commodity} Price", yaxis="y2",
                                   line=dict(color="#ff7043", width=2)))
        fig2.update_layout(height=320, xaxis_title="Date",
                           yaxis=dict(title=unit_lbl),
                           yaxis2=dict(title=unit, overlaying="y", side="right"),
                           legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
        st.plotly_chart(fig2, width="stretch", config=CHART_CFG)
    else:
        st.info(f"EIA inventory data is only available for Crude Oil and Natural Gas. Currently selected: {commodity}")

# ══ TAB 7 — CORRELATION ═══════════════════════════════════════════════════════
with t7:
    st.subheader("Cross-Commodity Correlation")
    pivot = df.pivot_table(index="Date", columns="Commodity", values="Return_1d")
    corr  = pivot.corr().round(3)
    fig   = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}", textfont={"size":11}
    ))
    fig.update_layout(height=420, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    st.subheader("Rolling 90-Day Correlation")
    c1,c2 = st.columns(2)
    com1  = c1.selectbox("Commodity 1", COMMODITIES, index=0)
    com2  = c2.selectbox("Commodity 2", COMMODITIES, index=1)
    roll  = pivot[[com1,com2]].dropna().rolling(90).corr().unstack()[com2][com1].dropna()
    fig2  = go.Figure()
    fig2.add_hline(y=0, line_dash="dot", line_color="gray")
    fig2.add_trace(go.Scatter(x=roll.index, y=roll.values, fill="tozeroy",
                               name=f"{com1} vs {com2}",
                               line=dict(color="#2196f3", width=2),
                               fillcolor="rgba(33,150,243,0.1)"))
    fig2.update_layout(height=300, yaxis=dict(range=[-1,1]),
                       xaxis_title="Date", yaxis_title="Correlation", margin=dict(t=10))
    st.plotly_chart(fig2, width="stretch", config=CHART_CFG)

# ══ TAB 8 — INTRADAY ══════════════════════════════════════════════════════════
with t8:
    st.subheader(f"Intraday View — {commodity} (15-Min Bars)")
    sel_date = st.date_input("Select Date", value=datetime.today())
    intra    = get_intraday(ticker, str(sel_date))

    if intra is not None and not intra.empty:
        opens  = intra["Open"].values.flatten().astype(float)
        highs  = intra["High"].values.flatten().astype(float)
        lows   = intra["Low"].values.flatten().astype(float)
        closes = intra["Close"].values.flatten().astype(float)

        fig = go.Figure(go.Candlestick(
            x=intra.index,
            open=opens, high=highs, low=lows, close=closes,
            increasing_line_color="green", decreasing_line_color="red"
        ))
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(height=380, xaxis_title="Time",
                          yaxis_title=unit, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Open",  fmt(opens[0],    currency, commodity))
        c2.metric("High",  fmt(highs.max(), currency, commodity))
        c3.metric("Low",   fmt(lows.min(),  currency, commodity))
        c4.metric("Close", fmt(closes[-1],  currency, commodity))
    else:
        st.warning(f"No intraday data for {sel_date}. Yahoo Finance supports last 60 days of 15-min data.")
#& ".\venv\Scripts\streamlit.exe" run "C:\Users\bhavy\Commodity Market Project\dashboard.py"