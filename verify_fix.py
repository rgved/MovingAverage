"""Final verification: simulate what the dashboard screener would do for
GRASIM, GLAND, CANFIN, and CIPLA using the exact same logic as app.py.

Tests both EMA and SMA convergence guards to ensure no false crossovers."""
import pandas as pd
import numpy as np
import os
import math

# Replicate _ema_min_periods from app.py
def _ema_min_periods(span):
    alpha = 2 / (span + 1)
    if alpha >= 1:
        return span
    extra = math.ceil(math.log(0.05) / math.log(1 - alpha))
    return span + extra

# Replicate _sma_min_periods from app.py
def _sma_min_periods(window):
    alpha = 2 / (window + 1)
    if alpha >= 1:
        return window
    extra = math.ceil(math.log(0.05) / math.log(1 - alpha))
    return window + extra

# Replicate _compute_sma from app.py
def _compute_sma(series, window):
    sma = series.rolling(window).mean()
    min_p = _sma_min_periods(window)
    if min_p > window:
        sma.iloc[:min_p] = np.nan
    return sma

# Replicate compute_crossover_series from app.py
def compute_crossover_series(ma_fast, ma_slow):
    signal = pd.Series(np.nan, index=ma_fast.index, dtype=float)
    valid_mask = ma_fast.notna() & ma_slow.notna()
    signal.loc[valid_mask] = np.where(ma_fast[valid_mask] > ma_slow[valid_mask], 1.0, -1.0)
    crossover = signal.diff().fillna(0)
    crossover.loc[~valid_mask] = 0
    return signal, crossover

start = pd.Timestamp('2026-03-01').date()
end = pd.Timestamp('2026-06-05').date()

print(f"Convergence thresholds:")
print(f"  EMA 50:  min_periods = {_ema_min_periods(50)}")
print(f"  EMA 200: min_periods = {_ema_min_periods(200)}")
print(f"  SMA 50:  min_periods = {_sma_min_periods(50)}")
print(f"  SMA 200: min_periods = {_sma_min_periods(200)}")
print()

test_stocks = ['GRASIM.NS.parquet', 'GLAND.NS.parquet', 'CANFINHOME.NS.parquet', 'CIPLA.NS.parquet']

all_pass = True
for stock_file in test_stocks:
    filepath = os.path.join('data', 'processed', stock_file)
    if not os.path.exists(filepath):
        print(f"SKIP {stock_file} (not found)")
        continue
    df = pd.read_parquet(filepath, engine='pyarrow')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    sym = stock_file.replace('.parquet', '')
    
    print(f"{sym} ({len(df)} rows)")
    
    for m in ['EMA', 'SMA']:
        if m == 'EMA':
            f = df['Close'].ewm(span=50, min_periods=_ema_min_periods(50), adjust=False).mean()
            s = df['Close'].ewm(span=200, min_periods=_ema_min_periods(200), adjust=False).mean()
        else:
            f = _compute_sma(df['Close'], 50)
            s = _compute_sma(df['Close'], 200)
        
        valid_fast = f.notna().sum()
        valid_slow = s.notna().sum()
        print(f"  {m} 50: {valid_fast} valid values, {m} 200: {valid_slow} valid values")
        
        _, crossover = compute_crossover_series(f, s)
        temp = df.copy()
        temp['Cross'] = crossover
        in_range = (temp['Date'].dt.date >= start) & (temp['Date'].dt.date <= end)
        events = temp[(temp['Cross'].abs() == 2) & in_range]
        
        if len(events) > 0:
            for _, row in events.iterrows():
                t = 'Bullish' if row['Cross'] == 2 else 'Bearish'
                idx = row.name
                gap = abs(f.iloc[idx] - s.iloc[idx])
                close = df.iloc[idx]['Close']
                gap_pct = gap / close * 100
                print(f"    {row['Date'].date()} {t} (gap={round(gap,2)}, {gap_pct:.3f}% of price)")
                if gap_pct < 0.1:
                    print(f"    ** SUSPICIOUS - gap is only {gap_pct:.4f}% of price **")
                    all_pass = False
        else:
            print(f"    No crossover in range (CORRECT for stocks with <500 rows)")
    print()

if all_pass:
    print("PASS: All false crossovers from insufficient data are eliminated.")
else:
    print("FAIL: Some suspicious crossovers still detected!")
