import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import requests
from dotenv import load_dotenv
import glob

load_dotenv('.env')
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")

print("--- 1 & 3: Verifying Raw Data vs Fresh API Data ---")
symbol = "CUB.NS"
instrument_key = "NSE_EQ|INE491A01021" # I will need to get the actual key if this is wrong. Let's just use yfinance to check corporate actions first.

import json
with open(os.path.join(BASE_DIR, "upstox_symbol_map.json")) as f:
    sym_map = json.load(f)
if symbol in sym_map:
    instrument_key = sym_map[symbol]
else:
    print(f"Key for {symbol} not found.")

def fetch_upstox_fresh(instrument_key, days=60):
    to_date = date(2026, 6, 12)
    from_date = to_date - timedelta(days=days)
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        candles = response.json().get("data", {}).get("candles", [])
        if candles:
            df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"])
            df["Date"] = pd.to_datetime(df["Date"])
            if df["Date"].dt.tz is not None:
                df["Date"] = df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df = df.sort_values("Date").reset_index(drop=True)
            return df
    return pd.DataFrame()

df_fresh = fetch_upstox_fresh(instrument_key, 60)
df_existing = pd.read_csv(os.path.join(DATA_RAW, f"{symbol}.csv"))
df_existing["Date"] = pd.to_datetime(df_existing["Date"])

start_dt = pd.to_datetime("2026-05-15")
df_exist_slice = df_existing[(df_existing["Date"] >= start_dt)].copy()
df_fresh_slice = df_fresh[(df_fresh["Date"] >= start_dt)].copy()

merged = pd.merge(df_exist_slice[["Date", "Close"]], df_fresh_slice[["Date", "Close"]], on="Date", how="outer", suffixes=('_Stored', '_Fresh'))
print("\nPrice Comparison (Stored vs Fresh):")
print(merged.dropna().tail(15).to_string(index=False))

print("\n--- 2: Corporate Actions Validation ---")
ticker = yf.Ticker(symbol)
actions = ticker.actions
if not actions.empty:
    actions_recent = actions[actions.index >= "2026-01-01"]
    if not actions_recent.empty:
        print("Recent Corporate Actions found via yfinance:")
        print(actions_recent)
    else:
        print("No recent corporate actions found in yfinance for 2026.")
else:
    print("No corporate actions found in yfinance history.")

print("\n--- 4: Measure EMA Impact ---")
# EMA on Stored
df_exist_ema = df_existing.copy()
df_exist_ema['EMA50'] = df_exist_ema['Close'].ewm(span=50, adjust=False).mean()
df_exist_ema['EMA200'] = df_exist_ema['Close'].ewm(span=200, adjust=False).mean()

# To do EMA on fresh properly, we'd need fresh history > 200 days. 
# We can simulate by replacing existing close prices with fresh close prices where they differ, simulating a fully adjusted dataset.
df_fixed = df_existing.copy()
# Assume the fresh data is the "correct" post-split data. 
# We need to find the split ratio. If around June it dropped from 250 to 190, ratio might be something like a dividend or bonus. 
# Let's just fetch 1000 days of fresh data to see.
df_full_fresh = fetch_upstox_fresh(instrument_key, 1000)
if not df_full_fresh.empty:
    df_full_fresh['EMA50'] = df_full_fresh['Close'].ewm(span=50, adjust=False).mean()
    df_full_fresh['EMA200'] = df_full_fresh['Close'].ewm(span=200, adjust=False).mean()
    
    target_dt = pd.to_datetime('2026-06-10')
    stored_row = df_exist_ema[df_exist_ema['Date'] == target_dt]
    fresh_row = df_full_fresh[df_full_fresh['Date'] == target_dt]
    
    if not stored_row.empty and not fresh_row.empty:
        print("\nEMA Impact on 2026-06-10:")
        print(f"Stored Data -> EMA50: {stored_row.iloc[0]['EMA50']:.2f}, EMA200: {stored_row.iloc[0]['EMA200']:.2f}")
        print(f"Fresh Data  -> EMA50: {fresh_row.iloc[0]['EMA50']:.2f}, EMA200: {fresh_row.iloc[0]['EMA200']:.2f}")

print("\n--- 5: Determine Scope of Problem (Scan all CSVs) ---")
files = glob.glob(os.path.join(DATA_RAW, "*.csv"))
anomalies = []
for f in files:
    sym = os.path.basename(f).replace(".csv", "")
    try:
        df = pd.read_csv(f)
        df['pct_change'] = df['Close'].pct_change().abs()
        # Look for jumps > 20% in the last 6 months
        df['Date'] = pd.to_datetime(df['Date'])
        recent = df[df['Date'] >= pd.to_datetime('2026-01-01')]
        jumps = recent[recent['pct_change'] > 0.20]
        if not jumps.empty:
            anomalies.append((sym, len(jumps)))
    except:
        pass

if anomalies:
    print(f"\nFound {len(anomalies)} symbols with >20% daily price jumps in 2026:")
    for sym, count in anomalies:
        print(f"  {sym}: {count} occurrences")
else:
    print("\nNo other symbols with >20% daily jumps found.")
