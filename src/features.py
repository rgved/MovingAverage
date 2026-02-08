# This file is for feature extraction. We will compute Simple, Exponential and Weighted averages

import pandas as pd
import numpy as np
import os

# ---------- PATH SETUP ----------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                # project root

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ---------- IMPORTS ----------
try:
    from constants import FNO_STOCKS, INDICES
except ImportError:
    # Fallback to local path if running directly in src
    import sys
    sys.path.append(SRC_DIR)
    from constants import FNO_STOCKS, INDICES

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
        df["MA_Fast"] = compute_sma(df, "Close", fast)
        df["MA_Slow"] = compute_sma(df, "Close", slow)
    elif ma_type.upper() == "EMA":
        df["MA_Fast"] = compute_ema(df, "Close", fast)
        df["MA_Slow"] = compute_ema(df, "Close", slow)
    elif ma_type.upper() == "WMA":
        df["MA_Fast"] = compute_wma(df, "Close", fast)
        df["MA_Slow"] = compute_wma(df, "Close", slow)
    else:
        raise ValueError("ma_type must be SMA, EMA, or WMA")
    return df

# ---------- Signal Detection ----------
def generate_signals(df):
    df = df.copy()
    # Use binary signal to avoid neutral state during exact equality
    # 1 = Bullish (Fast > Slow), -1 = Bearish (Fast <= Slow)
    df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
    df["Crossover"] = df["Signal"].diff()
    return df

# ---------- Master Function ----------
def process_file(filepath, ma_type="SMA", fast=10, slow=20):
    df = pd.read_csv(filepath)
    # Ensure Date is parsed as UTC first if it has offset, then convert to IST
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.sort_values("Date")

    df = add_moving_averages(df, ma_type, fast, slow)
    df = generate_signals(df)
    return df

# ---------- Batch Processor ----------
def process_all(data_dir=RAW_DATA_DIR,
                out_dir=PROCESSED_DATA_DIR,
                ma_type="SMA", fast=10, slow=20):

    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "") # Assumes filename is symbol.csv
            base_symbol = symbol.split(".")[0] # Remove .NS or similar extensions if present for check, or assume FNO_STOCKS has clean names
            
            # F&O and Indices Filter
            # Check if it's F&O (base_symbol) OR an Index (full symbol)
            if (base_symbol not in FNO_STOCKS) and (symbol not in INDICES):
                print(f"Skipping {file} (Not F&O/Index)")
                continue

            path = os.path.join(data_dir, file)
            df = process_file(path, ma_type, fast, slow)
            out_path = os.path.join(out_dir, file)
            df.to_csv(out_path, index=False)
            print(f"OK Processed {file} -> {out_path}")

# ---------- RUN ----------
if __name__ == "__main__":
    ma_type = "EMA"    # SMA | EMA | WMA
    fast = 10
    slow = 20

    process_all(ma_type=ma_type, fast=fast, slow=slow)
    print("Feature extraction complete! Files saved in data/processed/")
