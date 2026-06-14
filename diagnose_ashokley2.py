"""Check which pairs fire on June 11 for ASHOKLEY."""
import os, math, datetime, warnings
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

_FAST_MAS = [5, 10, 12, 20, 50]
_SLOW_MAS = [20, 26, 50, 100, 200]
_ALL_MA_PAIRS = [(f, s) for f in _FAST_MAS for s in _SLOW_MAS if f < s]

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

fpath = os.path.join(DATA_DIR, "ASHOKLEY.NS.parquet")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df = pd.read_parquet(fpath, engine="pyarrow")
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
df = df.sort_values("Date").reset_index(drop=True)

target = datetime.date(2026, 6, 11)

print("Pairs that fire on June 11 for ASHOKLEY:")
found = False
for fast, slow in _ALL_MA_PAIRS:
    for mtype in ["EMA", "SMA"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ma_f = _compute_ma(df["Close"], fast, mtype)
            ma_s = _compute_ma(df["Close"], slow, mtype)
            crossover = compute_crossover_series(ma_f, ma_s)
        matches = df.copy()
        matches["Crossover"] = crossover.values
        hits = matches[matches["Crossover"].abs() == 2]
        for _, row in hits.iterrows():
            if row["Date"].date() == target:
                ctype = "BUY" if row["Crossover"] == 2 else "SELL"
                print(f"  {fast}/{slow} {mtype}: [{ctype}] {row['Date'].date()}")
                found = True
if not found:
    print("  (NONE -- ASHOKLEY has NO crossover on June 11 for any pair)")

print()
print("All recent crossovers for ASHOKLEY (any pair, after May 1):")
for fast, slow in _ALL_MA_PAIRS:
    for mtype in ["EMA", "SMA"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ma_f = _compute_ma(df["Close"], fast, mtype)
            ma_s = _compute_ma(df["Close"], slow, mtype)
            crossover = compute_crossover_series(ma_f, ma_s)
        matches = df.copy()
        matches["Crossover"] = crossover.values
        hits = matches[matches["Crossover"].abs() == 2]
        may1 = datetime.date(2026, 5, 1)
        recent = hits[hits["Date"].dt.date >= may1]
        for _, row in recent.iterrows():
            ctype = "BUY" if row["Crossover"] == 2 else "SELL"
            print(f"  {fast}/{slow} {mtype}: [{ctype}] {row['Date'].date()}")
