import sys
import pandas as pd
import yfinance as yf

print("Testing EMA Convergence for INDIACEM using yfinance (Close vs Adj Close)")

df_max = yf.download("INDIACEM.NS", period="max", progress=False)

if df_max.empty:
    print("Failed to fetch data from yfinance.")
    sys.exit(1)

# Handle multiindex columns if they exist
if isinstance(df_max.columns, pd.MultiIndex):
    df_max.columns = ['_'.join(col).strip() for col in df_max.columns.values]
    close_col = [c for c in df_max.columns if 'Close' in c and 'Adj' not in c][0]
    adj_close_col = [c for c in df_max.columns if 'Adj Close' in c][0]
else:
    close_col = 'Close'
    adj_close_col = 'Adj Close'

df_max.reset_index(inplace=True)
if df_max['Date'].dt.tz is not None:
    df_max['Date'] = df_max['Date'].dt.tz_localize(None)

df_max = df_max.sort_values("Date").reset_index(drop=True)

# Calculate EMA on Close
df_max['EMA50_Close'] = df_max[close_col].ewm(span=50, adjust=False).mean()
df_max['EMA200_Close'] = df_max[close_col].ewm(span=200, adjust=False).mean()

# Calculate EMA on Adj Close
df_max['EMA50_Adj'] = df_max[adj_close_col].ewm(span=50, adjust=False).mean()
df_max['EMA200_Adj'] = df_max[adj_close_col].ewm(span=200, adjust=False).mean()

target_date = pd.to_datetime('2026-06-10')
target_row = df_max[df_max['Date'].dt.date == target_date.date()]

if not target_row.empty:
    print("\n--- YFinance Results (Max History: {} rows) ---".format(len(df_max)))
    print(f"Close     -> EMA50: {target_row.iloc[0]['EMA50_Close']:.4f}, EMA200: {target_row.iloc[0]['EMA200_Close']:.4f}")
    print(f"Adj Close -> EMA50: {target_row.iloc[0]['EMA50_Adj']:.4f}, EMA200: {target_row.iloc[0]['EMA200_Adj']:.4f}")
else:
    print("Target date not found in yfinance data.")

print("\n--- Upstox Target Reference ---")
print("Upstox EMA 50 (June 10): 394.01")
print("Upstox EMA 200 (June 10): 388.89")
