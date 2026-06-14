"""Simulate EXACTLY what the dashboard does: load archive + processed, 
combine, compute 50/200 EMA, and find crossovers. Compare with what
a TradingView-style approach would show."""
import pandas as pd
import numpy as np
import os

def _normalize_date_series(series, normalize=True):
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return parsed.dt.normalize() if normalize else parsed

def compute_crossover_series(ma_fast, ma_slow):
    signal = pd.Series(np.nan, index=ma_fast.index, dtype=float)
    valid_mask = ma_fast.notna() & ma_slow.notna()
    signal.loc[valid_mask] = np.where(ma_fast[valid_mask] > ma_slow[valid_mask], 1.0, -1.0)
    crossover = signal.diff().fillna(0)
    crossover.loc[~valid_mask] = 0
    return signal, crossover

def _ema_min_periods(span):
    return span  # current app.py logic

needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Simulate load_all_price_data for RELIANCE
archive_file = 'archive_csv/processed/RELIANCE.NS.csv'
processed_file = 'data/processed/RELIANCE.NS.parquet'

# Load archive
archive_df = pd.read_csv(archive_file, usecols=needed_cols)
archive_df["Date"] = _normalize_date_series(archive_df["Date"])
archive_df = archive_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Load processed
processed_df = pd.read_parquet(processed_file, engine="pyarrow", columns=needed_cols)
processed_df["Date"] = _normalize_date_series(processed_df["Date"])
processed_df = processed_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

print(f"Archive: {len(archive_df)} rows, {archive_df['Date'].min().date()} to {archive_df['Date'].max().date()}")
print(f"Processed: {len(processed_df)} rows, {processed_df['Date'].min().date()} to {processed_df['Date'].max().date()}")

# Merge like load_all_price_data does
latest_loaded_date = archive_df["Date"].max()
newer_rows = processed_df[processed_df["Date"] > latest_loaded_date]
print(f"Newer rows from processed: {len(newer_rows)}")

combined_df = pd.concat([archive_df, newer_rows], ignore_index=True)
combined_df = combined_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

print(f"Combined: {len(combined_df)} rows, {combined_df['Date'].min().date()} to {combined_df['Date'].max().date()}")
print()

# Compute EMA 50/200 on combined data
ema50 = combined_df["Close"].ewm(span=50, min_periods=_ema_min_periods(50), adjust=False).mean()
ema200 = combined_df["Close"].ewm(span=200, min_periods=_ema_min_periods(200), adjust=False).mean()

signal, crossover = compute_crossover_series(ema50, ema200)

combined_df["EMA50"] = ema50
combined_df["EMA200"] = ema200
combined_df["Signal"] = signal
combined_df["Crossover"] = crossover

# Print recent crossovers
cross_mask = crossover.abs() == 2
cross_rows = combined_df[cross_mask]

print("EMA 50/200 Crossovers (all):")
for _, row in cross_rows.iterrows():
    ctype = "BUY " if row["Crossover"] == 2 else "SELL"
    idx = row.name
    print(f"  [{ctype}] {row['Date'].date()} | Close={row['Close']:.2f} | EMA50={row['EMA50']:.2f} | EMA200={row['EMA200']:.2f} | diff={row['EMA50']-row['EMA200']:.4f}")

print()

# SMA 50/200
sma50 = combined_df["Close"].rolling(50).mean()
sma200 = combined_df["Close"].rolling(200).mean()

signal_s, crossover_s = compute_crossover_series(sma50, sma200)
combined_df["SMA50"] = sma50
combined_df["SMA200"] = sma200
combined_df["SMA_Crossover"] = crossover_s

cross_mask_s = crossover_s.abs() == 2
cross_rows_s = combined_df[cross_mask_s]

print("SMA 50/200 Crossovers (all):")
for _, row in cross_rows_s.iterrows():
    ctype = "BUY " if row["SMA_Crossover"] == 2 else "SELL"
    print(f"  [{ctype}] {row['Date'].date()} | Close={row['Close']:.2f} | SMA50={row['SMA50']:.2f} | SMA200={row['SMA200']:.2f} | diff={row['SMA50']-row['SMA200']:.4f}")

print()

# Now simulate the GRAPH: what happens when we slice for 3-month lookback?
three_months_ago = combined_df["Date"].max() - pd.DateOffset(months=3)
chart_df = combined_df[combined_df["Date"] >= three_months_ago].copy()
chart_df = chart_df.reset_index(drop=True)

# This is what the graph does:
buy_indices = chart_df.index[chart_df["Crossover"] == 2].to_numpy()
sell_indices = chart_df.index[chart_df["Crossover"] == -2].to_numpy()

print(f"EMA 50/200 - Chart (3-month lookback):")
print(f"  Buy signals shown: {len(buy_indices)}")
for i in buy_indices:
    print(f"    BUY at {chart_df.iloc[i]['Date'].date()}")
print(f"  Sell signals shown: {len(sell_indices)}")
for i in sell_indices:
    print(f"    SELL at {chart_df.iloc[i]['Date'].date()}")

print()
print("=" * 80)
print("KEY CHECK: Does the crossover detection produce duplicate same-direction signals?")
print("=" * 80)

# Look at the raw Crossover column for consecutive same-sign values
all_cross_events = combined_df[combined_df["Crossover"].abs() == 2]["Crossover"]
for idx in range(1, len(all_cross_events)):
    if all_cross_events.iloc[idx] == all_cross_events.iloc[idx-1]:
        i = all_cross_events.index[idx]
        prev_i = all_cross_events.index[idx-1]
        ctype = "BUY" if all_cross_events.iloc[idx] == 2 else "SELL"
        print(f"  CONSECUTIVE DUPLICATE {ctype}: {combined_df.iloc[prev_i]['Date'].date()} -> {combined_df.iloc[i]['Date'].date()}")

print()
print("Now checking if the graph code could show arrows on wrong days...")
print("The graph uses chart_df['Crossover'] == 2 / -2 after reset_index.")
print("Since crossover was computed on FULL data before slicing, the signals")
print("should be correct - no recomputation happens after slice.")
print()

# Check if there's an issue with chart_mode being different from data timeframe
# In the graph code, if chart_mode != timeframe, a new weekly resample happens
# BUT the crossover is computed AGAIN on chart_df (line 1358)
# This is where the bug might be!
print("=" * 80)
print("IMPORTANT: The graph recomputes crossovers on the chart data (line 1358)!")
print("If chart_mode is the same as timeframe, chart_df = df (full data)")
print("Then MAs and crossovers are computed on full data - CORRECT")
print("But if chart_mode is Weekly, chart_df is a fresh weekly resample")
print("Then crossovers are computed on the weekly resampled data - could differ")
print("=" * 80)
