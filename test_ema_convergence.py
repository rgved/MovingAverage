import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
from dotenv import load_dotenv
import time

load_dotenv('.env')
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

instrument_key = "NSE_EQ|INE383A01012" # INDIACEM.NS

def fetch_upstox_history_batched(instrument_key, total_days):
    print(f"Fetching {total_days} days of data for INDIACEM from Upstox...")
    all_dfs = []
    
    current_to_date = date(2026, 6, 12)
    end_date = current_to_date - timedelta(days=total_days)
    
    while current_to_date > end_date:
        # Fetch in 365 day chunks
        chunk_days = min((current_to_date - end_date).days, 365)
        current_from_date = current_to_date - timedelta(days=chunk_days)
        
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{current_to_date}/{current_from_date}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"API Error: {response.text}")
            break
        
        candles = response.json().get("data", {}).get("candles", [])
        if candles:
            df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"])
            all_dfs.append(df)
            
        current_to_date = current_from_date
        time.sleep(0.5) # respect rate limits
        
    if not all_dfs:
        return pd.DataFrame()
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df["Date"] = pd.to_datetime(final_df["Date"])
    if final_df["Date"].dt.tz is not None:
        final_df["Date"] = final_df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    final_df = final_df.sort_values("Date").reset_index(drop=True)
    return final_df

df_max = fetch_upstox_history_batched(instrument_key, 3650)

if df_max.empty:
    print("Failed to fetch batched data.")
    sys.exit(1)

print(f"Total rows fetched: {len(df_max)}")

history_lengths = [500, 1000, 1500, 2000, 2500, 3000, 3650]
target_date = pd.to_datetime('2026-06-10')

results = []

for days in history_lengths:
    start_date = target_date.date() - timedelta(days=days)
    # Slice the max dataframe to simulate downloading `days` of history
    df_slice = df_max[df_max["Date"].dt.date >= start_date].copy()
    
    if df_slice.empty:
        continue
        
    # Calculate EMA
    df_slice['EMA50'] = df_slice['Close'].ewm(span=50, adjust=False).mean()
    df_slice['EMA200'] = df_slice['Close'].ewm(span=200, adjust=False).mean()
    
    target_row = df_slice[df_slice['Date'].dt.date == target_date.date()]
    if not target_row.empty:
        ema50 = target_row.iloc[0]['EMA50']
        ema200 = target_row.iloc[0]['EMA200']
        
        results.append({
            "History_Days": days,
            "EMA50": round(ema50, 4),
            "EMA200": round(ema200, 4),
            "Rows": len(df_slice)
        })

df_results = pd.DataFrame(results)
print("\n--- Test Results ---")
print(df_results.to_string(index=False))

print("\n--- Upstox Target Reference ---")
print("Upstox EMA 50 (June 10): 394.01")
print("Upstox EMA 200 (June 10): 388.89")
