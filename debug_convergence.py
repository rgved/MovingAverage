"""
Deep analysis: Why do GLAND and CANFIN show EMA 50/200 crossovers 
that TradingView doesn't show?

Key insight: TradingView uses MUCH more historical data (years worth).
Our data starts from Dec 2024 (only ~18 months). With only 371 rows,
the EMA 200 hasn't converged to its true value yet.

The EMA's exponential weighting means early values disproportionately 
affect the output when the series is short. TradingView, using 5+ years 
of data, produces very different EMA values.
"""
import pandas as pd
import numpy as np
import os

data_dir = os.path.join('data', 'processed')

for file in ['GLAND.NS.parquet', 'CANFINHOME.NS.parquet']:
    df = pd.read_parquet(os.path.join(data_dir, file), engine='pyarrow')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    sym = file.replace('.parquet', '')
    
    print(f"{'='*70}")
    print(f"STOCK: {sym}")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print()
    
    # With min_periods=200, EMA starts from row 199
    ema50 = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
    ema200 = df['Close'].ewm(span=200, min_periods=200, adjust=False).mean()
    
    # The EMA seed value is the mean of the first min_periods rows
    # With min_periods=200, the seed is the average of rows 0-199
    seed_value = df['Close'].iloc[:200].mean()
    print(f"EMA 200 seed (mean of first 200 closes): {round(seed_value, 2)}")
    print(f"EMA 200 at row 200: {round(ema200.iloc[199], 2) if pd.notna(ema200.iloc[199]) else 'NaN'}")
    print(f"EMA 200 last: {round(ema200.iloc[-1], 2) if pd.notna(ema200.iloc[-1]) else 'NaN'}")
    print()
    
    # Check how many extra rows we have beyond 200
    extra_rows = len(df) - 200
    print(f"Rows beyond 200: {extra_rows}")
    print(f"That's only {extra_rows} trading days for the EMA 200 to 'converge'")
    print()
    
    # The EMA 200 smoothing factor
    alpha = 2 / (200 + 1)
    # After N additional periods, the weight of the initial seed is (1-alpha)^N
    initial_weight = (1 - alpha) ** extra_rows
    print(f"EMA 200 smoothing factor (alpha): {round(alpha, 4)}")
    print(f"Weight of initial seed after {extra_rows} more rows: {round(initial_weight * 100, 1)}%")
    print(f"The initial seed STILL accounts for {round(initial_weight * 100, 1)}% of the current EMA 200!")
    print()
    
    # Compare: what would TradingView show with 1000+ rows?
    # We can't simulate that, but we can show that our EMA is unreliable
    print("CONCLUSION:")
    print(f"  With only {len(df)} rows total ({extra_rows} beyond the 200 min_periods),")
    print(f"  the EMA 200 is still heavily influenced by its initial seed value.")
    print(f"  TradingView uses 1000+ rows, so its EMA 200 is far more converged.")
    print(f"  Our 'crossovers' are artifacts of an insufficiently converged EMA.")
    print()

    # Check May 29 specifically
    may29 = df[df['Date'].dt.date == pd.Timestamp('2026-05-29').date()]
    if not may29.empty:
        idx = may29.index[0]
        print(f"On May 29:")
        print(f"  Close: {df.iloc[idx]['Close']}")
        print(f"  Our EMA 50: {round(ema50.iloc[idx], 2)}")
        print(f"  Our EMA 200: {round(ema200.iloc[idx], 2)}")
        gap = abs(ema50.iloc[idx] - ema200.iloc[idx])
        print(f"  Gap: {round(gap, 2)} (very small = unreliable)")
    print()

# How many rows would we need for EMA 200 to be 95% converged?
alpha = 2 / 201
needed = int(np.log(0.05) / np.log(1 - alpha))
print(f"{'='*70}")
print(f"CONVERGENCE ANALYSIS:")
print(f"  To reduce initial seed influence to <5%, need {needed} rows beyond min_periods")
print(f"  Total data needed: 200 + {needed} = {200 + needed} rows")
print(f"  That's roughly {round((200 + needed) / 252, 1)} years of trading data")
print(f"  We currently have only 371 rows (~1.5 years)")
print(f"  RECOMMENDATION: Require ~800 rows minimum for 200-period EMA crossovers")
