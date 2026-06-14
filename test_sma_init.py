import os
import sys
import pandas as pd
from datetime import datetime, timedelta, date
import requests
from dotenv import load_dotenv

load_dotenv('.env')
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

instrument_key = "NSE_EQ|INE383A01012" # INDIACEM.NS

print("Testing SMA Initialization for EMA Convergence...")

def fetch_upstox_history(instrument_key, total_days):
    all_dfs = []
    current_to_date = date(2026, 6, 12)
    end_date = current_to_date - timedelta(days=total_days)
    
    while current_to_date > end_date:
        chunk_days = min((current_to_date - end_date).days, 365)
        current_from_date = current_to_date - timedelta(days=chunk_days)
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{current_to_date}/{current_from_date}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            candles = response.json().get("data", {}).get("candles", [])
            if candles:
                df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"])
                all_dfs.append(df)
        current_to_date = current_from_date
        
    if not all_dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df["Date"] = pd.to_datetime(final_df["Date"])
    final_df = final_df.sort_values("Date").reset_index(drop=True)
    return final_df

df_max = fetch_upstox_history(instrument_key, 2000)

if df_max.empty:
    print("Failed to fetch data.")
    sys.exit(1)

# Method 1: Standard EMA (starts with Price[0])
df_max['EMA200_Standard'] = df_max['Close'].ewm(span=200, adjust=False).mean()

# Method 2: SMA Initialization (starts with SMA of first 200 days)
sma200_val = df_max['Close'].iloc[:200].mean()
ema_sma_init = [pd.NA] * 199 + [sma200_val]

multiplier = 2 / (200 + 1)
for i in range(200, len(df_max)):
    prev_ema = ema_sma_init[-1]
    current_price = df_max['Close'].iloc[i]
    new_ema = (current_price - prev_ema) * multiplier + prev_ema
    ema_sma_init.append(new_ema)

df_max['EMA200_SMA_Init'] = ema_sma_init

target_date = pd.to_datetime('2026-06-10')
target_row = df_max[df_max['Date'].dt.date == target_date.date()]

if not target_row.empty:
    print("\n--- Results on target date ---")
    print(f"Standard EMA 200: {target_row.iloc[0]['EMA200_Standard']:.4f}")
    print(f"SMA-Init EMA 200: {target_row.iloc[0]['EMA200_SMA_Init']:.4f}")
else:
    print("Target date not found.")

print("\n--- Upstox Target Reference ---")
print("Upstox EMA 200 (June 10): 388.89")
