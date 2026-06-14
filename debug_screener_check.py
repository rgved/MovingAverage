import pandas as pd, numpy as np, os

start = pd.Timestamp('2026-05-29').date()
end = pd.Timestamp('2026-06-01').date()

for stock_file in ['GLAND.NS.parquet', 'CANFINHOME.NS.parquet']:
    df = pd.read_parquet(os.path.join('data', 'processed', stock_file), engine='pyarrow')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    sym = stock_file.replace('.parquet', '')
    
    print(f"--- {sym} ---")
    for m in ['EMA', 'SMA']:
        if m == 'EMA':
            f = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
            s = df['Close'].ewm(span=200, min_periods=200, adjust=False).mean()
        else:
            f = df['Close'].rolling(50).mean()
            s = df['Close'].rolling(200).mean()
        sig = pd.Series(np.nan, index=df.index)
        v = f.notna() & s.notna()
        sig[v] = np.where(f[v] > s[v], 1.0, -1.0)
        cr = sig.diff().fillna(0)
        cr[~v] = 0
        temp = df.copy()
        temp['Cross'] = cr
        in_range = (temp['Date'].dt.date >= start) & (temp['Date'].dt.date <= end)
        events = temp[(temp['Cross'].abs() == 2) & in_range]
        if len(events) > 0:
            for _, row in events.iterrows():
                t = 'Bull' if row['Cross'] == 2 else 'Bear'
                idx = row.name
                dt = row['Date'].date()
                fv = round(f.iloc[idx], 2)
                sv = round(s.iloc[idx], 2)
                print(f"  {m} 50/200: {dt} {t} fast={fv} slow={sv} gap={round(abs(fv-sv),2)}")
        else:
            print(f"  {m} 50/200: No crossover in {start} to {end}")
    print()
