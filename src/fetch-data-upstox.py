import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load API token
load_dotenv()
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise RuntimeError("UPSTOX_ACCESS_TOKEN missing in .env")

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(DATA_DIR, exist_ok=True)

# Load symbol map
with open(os.path.join(BASE_DIR, "upstox_symbol_map.json")) as f:
    SYMBOL_MAP = json.load(f)

def fetch_history(symbol, instrument_key, days=365):
    os.makedirs(DATA_DIR, exist_ok=True)

    to_date = (datetime.today() + timedelta(days=1)).date()
    from_date = to_date - timedelta(days=days)

    print(f"Fetching {symbol}: {from_date} -> {to_date}")

    url = (
        "https://api.upstox.com/v2/historical-candle/"
        f"{instrument_key}/day/{to_date}/{from_date}"
    )

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"X {symbol} | {response.status_code} | {response.text}")
        return

    candles = response.json().get("data", {}).get("candles", [])

    if not candles:
        print(f"! No data for {symbol}")
        return

    df = pd.DataFrame(
        candles,
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"]
    )

    df = df.drop(columns=["OI"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df.to_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), index=False)
    print(f"OK Saved {symbol} ({len(df)} rows)")


if __name__ == "__main__":
    for sym, key in SYMBOL_MAP.items():
        fetch_history(sym, key)

