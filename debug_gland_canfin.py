import pandas as pd
import numpy as np
import os

# Search for GLAND and CANFIN files
data_dir = os.path.join('data', 'processed')
print("Looking for GLAND and CANFIN files...")
matches = [f for f in os.listdir(data_dir) if 'GLAND' in f.upper() or 'CANFIN' in f.upper()]
print(f"Found: {matches}")
print()

for file in matches:
    p = os.path.join(data_dir, file)
    if file.endswith('.parquet'):
        df = pd.read_parquet(p, engine='pyarrow')
    else:
        df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    sym = file.rsplit('.', 1)[0] if file.endswith('.parquet') else file.rsplit('.', 1)[0]
    print(f"{'='*60}")
    print(f"STOCK: {sym}")
    print(f"Rows: {len(df)}, Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Last close: {df['Close'].iloc[-1]}")
    print()
    
    # EMA WITH min_periods (fixed behavior)
    ema50 = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
    ema200 = df['Close'].ewm(span=200, min_periods=200, adjust=False).mean()
    print(f"EMA (WITH min_periods fix):")
    print(f"  EMA 50 valid values: {ema50.notna().sum()}")
    print(f"  EMA 200 valid values: {ema200.notna().sum()}")
    
    if ema200.notna().sum() > 0:
        # Check for crossovers
        signal = pd.Series(np.nan, index=df.index)
        valid = ema50.notna() & ema200.notna()
        signal[valid] = np.where(ema50[valid] > ema200[valid], 1.0, -1.0)
        cr = signal.diff().fillna(0)
        cr[~valid] = 0
        cross_idx = df.index[cr.abs() == 2]
        print(f"  Crossovers: {len(cross_idx)}")
        for i in cross_idx:
            ctype = "Bullish" if cr.iloc[i] == 2 else "Bearish"
            print(f"    {df.iloc[i]['Date'].date()}: {ctype}, EMA50={round(ema50.iloc[i],2)}, EMA200={round(ema200.iloc[i],2)}")
    else:
        print(f"  Crossovers: BLOCKED (not enough data for EMA 200)")
    print()
    
    # EMA WITHOUT min_periods (old buggy behavior)
    ema50_old = df['Close'].ewm(span=50, adjust=False).mean()
    ema200_old = df['Close'].ewm(span=200, adjust=False).mean()
    print(f"EMA (WITHOUT min_periods - old bug):")
    print(f"  EMA 50 last: {round(ema50_old.iloc[-1], 2)}")
    print(f"  EMA 200 last: {round(ema200_old.iloc[-1], 2)}")
    signal_old = np.where(ema50_old > ema200_old, 1, -1)
    cross_old = np.diff(signal_old, prepend=0)
    cross_dates = df.loc[np.abs(cross_old) == 2]
    print(f"  FALSE crossovers: {len(cross_dates)}")
    for _, row in cross_dates.iterrows():
        ctype = "Bullish" if cross_old[row.name] == 2 else "Bearish"
        print(f"    {row['Date'].date()}: {ctype}")
    print()
    
    # SMA
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    print(f"SMA:")
    print(f"  SMA 50 valid: {sma50.notna().sum()}, SMA 200 valid: {sma200.notna().sum()}")
    if sma200.notna().sum() > 0:
        print(f"  SMA 50 last: {round(sma50.iloc[-1], 2)}, SMA 200 last: {round(sma200.iloc[-1], 2)}")
    else:
        print(f"  SMA 200: NaN (not enough data)")
    print()
