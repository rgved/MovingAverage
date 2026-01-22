# This file is for feature extraction. We will compute Simple, Exponential and Weighted averages

import pandas as pd
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- Moving Averages ----------
def compute_sma(df, column="Close", window=20):
    return df[column].rolling(window=window).mean()

def compute_ema(df, column="Close", span=20):
    return df[column].ewm(span=span, adjust=False).mean()

def compute_wma(df, column="Close", window=20):
    weights = np.arange(1, window + 1)
    return (
        df[column]
        .rolling(window)
        .apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    )

# ---------- Feature Builder ----------
def add_moving_averages(df, ma_type="SMA", fast=10, slow=20):
    df = df.copy()
    if ma_type.upper() == "SMA":
        df[f"MA_Fast"] = compute_sma(df, "Close", fast)
        df[f"MA_Slow"] = compute_sma(df, "Close", slow)
    elif ma_type.upper() == "EMA":
        df[f"MA_Fast"] = compute_ema(df, "Close", fast)
        df[f"MA_Slow"] = compute_ema(df, "Close", slow)
    elif ma_type.upper() == "WMA":
        df[f"MA_Fast"] = compute_wma(df, "Close", fast)
        df[f"MA_Slow"] = compute_wma(df, "Close", slow)
    else:
        raise ValueError("ma_type must be SMA, EMA, or WMA")
    return df

# ---------- Signal Detection ----------
def generate_signals(df):
    df = df.copy()
    df["Signal"] = 0
    df.loc[df["MA_Fast"] > df["MA_Slow"], "Signal"] = 1   # Bullish
    df.loc[df["MA_Fast"] < df["MA_Slow"], "Signal"] = -1  # Bearish

    # Identify crossover points
    df["Crossover"] = df["Signal"].diff()  # +2 → bullish cross, -2 → bearish cross
    return df

# ---------- Master Function ----------
def process_file(filepath, ma_type="SMA", fast=10, slow=20):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Add MAs
    df = add_moving_averages(df, ma_type, fast, slow)
    # Add signals
    df = generate_signals(df)

    return df

# ---------- Batch Processor ----------
def process_all(data_dir = os.path.join(BASE_DIR, "data", "raw"), out_dir = os.path.join(BASE_DIR, "data", "processed"),
                ma_type="SMA", fast=10, slow=20):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            path = os.path.join(data_dir, file)
            df = process_file(path, ma_type, fast, slow)
            out_path = os.path.join(out_dir, file)
            df.to_csv(out_path, index=False)
            print(f"✅ Processed {file} → {out_path}")

if __name__ == "__main__":
    # You can change these as needed
    data_dir = os.path.join(BASE_DIR, "data", "raw")
    out_dir  = os.path.join(BASE_DIR, "data", "processed")


    ma_type = "EMA"    # choose: "SMA", "EMA", or "WMA"
    fast = 10
    slow = 20

    process_all(data_dir=data_dir, out_dir=out_dir, ma_type=ma_type, fast=fast, slow=slow)
    print("🎉 Feature extraction complete! Files saved in data/processed/")

