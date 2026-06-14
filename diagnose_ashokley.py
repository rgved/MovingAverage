"""Check ALL MA pairs for ASHOKLEY to see which one fires on June 11."""
import os, math
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

_FAST_MAS = [5, 10, 12, 20, 50]
_SLOW_MAS = [20, 26, 50, 100, 200]
_ALL_MA_PAIRS = [(f, s) for f in _FAST_MAS for s in _SLOW_MAS if f < s]
_ALL_MA_TYPES = ["EMA", "SMA"]

def _ema_min_periods(span):
    alpha = 2.0 / (span + 1)
    k = math.ceil(math.log(0.01) / math.log(1.0 - alpha))
    return max(span, min(k, span * 3))

def _compute_ma(close, span, ma_type):
    if ma_type == "EMA":
        return close.ewm(span=span, min_periods=_ema_min_periods(span), adjust=False).mean()
    else:
        return close.rolling(span, min_periods=span).mean()

def compute_crossover_series(ma_fast, ma_slow):
    both_valid = ma_fast.notna() & ma_slow.notna()
    prev_valid = both_valid.shift(1).fillna(False).astype(bool)
    diff = (ma_fast - ma_slow).astype(float)
    prev_diff = diff.shift(1)
    crossover = pd.Series(0.0, index=ma_fast.index)
    bull = both_valid & prev_valid & (prev_diff < 0) & (diff > 0)
    bear = both_valid & prev_valid & (prev_diff > 0) & (diff < 0)
    crossover[bull] = 2.0
    crossover[bear] = -2.0
    return crossover

# Load ASHOKLEY
fpath = os.path.join(DATA_DIR, "ASHOKLEY.NS.parquet")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df = pd.read_parquet(fpath, engine="pyarrow")
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
df = df.sort_values("Date").reset_index(drop=True)

print(f"Data: {df['Date'].min().date()} -> {df['Date'].max().date()} ({len(df)} rows)")
print()
print(f"{'Pair':<10} {'Type':<5} {'Last Crossover Date':<22} {'Signal'}")
print("-"*55)

best_date = pd.Timestamp.min
best_info = None

for fast, slow in _ALL_MA_PAIRS:
    for mtype in _ALL_MA_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ma_f = _compute_ma(df["Close"], fast, mtype)
            ma_s = _compute_ma(df["Close"], slow, mtype)
            crossover = compute_crossover_series(ma_f, ma_s)
        crosses = df[crossover.abs() == 2].copy()
        crosses["Crossover"] = crossover[crossover.abs() == 2].values
        if not crosses.empty:
            last = crosses.iloc[-1]
            last_date = last["Date"]
            ctype = "BUY" if last["Crossover"] == 2 else "SELL"
            print(f"{fast}/{slow:<6} {mtype:<5} {str(last_date.date()):<22} {ctype}")
            if last_date > best_date:
                best_date = last_date
                best_info = (fast, slow, mtype, ctype, last_date)
        else:
            print(f"{fast}/{slow:<6} {mtype:<5} {'(no crossovers)':<22}")

print()
print(f"==> calculate_crossovers() would pick: {best_info}")
print(f"==> Because it selects the MOST RECENT crossover from ANY pair")
print()
print("The screener table shows the date from the MOST RECENT pair, NOT from 50/200 SMA.")
