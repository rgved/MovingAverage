import pandas as pd
import numpy as np
import os

df = pd.read_parquet('data/processed/CIPLA.NS.parquet', engine='pyarrow')
df = df.sort_values('Date').reset_index(drop=True)
print('Total rows:', len(df))
print()

# SMA 50/200
sma50 = df['Close'].rolling(50).mean()
sma200 = df['Close'].rolling(200).mean()
print('SMA 50 valid values:', sma50.dropna().shape[0])
print('SMA 200 valid values:', sma200.dropna().shape[0])
if pd.notna(sma50.iloc[-1]):
    print('SMA 50 last value:', round(sma50.iloc[-1], 2))
else:
    print('SMA 50 last value: NaN (not enough data!)')
if pd.notna(sma200.iloc[-1]):
    print('SMA 200 last value:', round(sma200.iloc[-1], 2))
else:
    print('SMA 200 last value: NaN (not enough data - need 200 rows, only have', len(df), ')')
print()

# EMA 50/200 - EWM always produces values from row 0 even with insufficient data
ema50 = df['Close'].ewm(span=50, adjust=False).mean()
ema200 = df['Close'].ewm(span=200, adjust=False).mean()
print('EMA 50 last value:', round(ema50.iloc[-1], 2))
print('EMA 200 last value:', round(ema200.iloc[-1], 2))
print('EMA 50 FIRST value:', round(ema50.iloc[0], 2))
print('EMA 200 FIRST value:', round(ema200.iloc[0], 2))
print()
print('Close price FIRST value:', round(df['Close'].iloc[0], 2))
print()

# Key insight: EWM starts from row 0 using the first value as seed
# With only 70 rows, the 200 EMA is heavily biased toward initial values
# and has NOT converged to a true 200-period EMA

# Find EMA crossover
signal = pd.Series(np.nan, index=df.index)
valid = ema50.notna() & ema200.notna()
signal[valid] = np.where(ema50[valid] > ema200[valid], 1.0, -1.0)
crossover = signal.diff().fillna(0)

cross_idx = df.index[crossover.abs() == 2].tolist()
print('EMA 50/200 crossover dates:')
for i in cross_idx:
    ctype = "Bullish" if crossover.iloc[i] == 2 else "Bearish"
    print(f"  Row {i}: Date={df.iloc[i]['Date']}, Close={df.iloc[i]['Close']}, EMA50={round(ema50.iloc[i],2)}, EMA200={round(ema200.iloc[i],2)}, Type={ctype}")

print()
print("=" * 80)
print("ROOT CAUSE ANALYSIS:")
print("=" * 80)
print(f"Data has only {len(df)} rows (from {df['Date'].min()} to {df['Date'].max()})")
print()
print("SMA 200 requires 200 data points to produce ANY value.")
print(f"  -> SMA 200 is completely NaN (0 valid values) - CORRECTLY shows no crossover")
print()
print("EMA 200 uses ewm(span=200, adjust=False) which starts from row 0,")
print("seeding with the FIRST closing price and applying a tiny smoothing factor (2/201).")
print("With only 70 rows, the EMA 200 barely moves from the initial seed price,")
print("creating a FAKE flat line that can falsely cross with the 50 EMA.")
print("This is NOT a real 200-period EMA - it hasn't seen enough data to converge.")
print()

# Show what TradingView would show
# TradingView requires min_periods worth of data for EMA too
print("TradingView/Broker behavior:")
print("  - Requires sufficient historical data (200+ bars) before plotting 200 EMA/SMA")
print("  - Would NOT show a 200 EMA with only 70 bars of data")
print("  - The 50 SMA and 200 SMA on the broker chart are at 1533 and 1513 respectively")
print("  - These are far apart because they use the FULL history, not just 70 bars")
