import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # project root
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise PermissionError("UPSTOX_ACCESS_TOKEN missing or empty in .env")

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(DATA_DIR, exist_ok=True)

# Load symbol map
with open(os.path.join(BASE_DIR, "upstox_symbol_map.json")) as f:
    SYMBOL_MAP = json.load(f)

# ── Constants ────────────────────────────────────────────────────────────────
BOOTSTRAP_DAYS = 1000         # ~2.7 years — EMA 200 needs ~660 trading days to converge (<1% seed influence)
STALE_THRESHOLD_DAYS = 30    # if last date is older than this, do a full re-fetch
INCREMENTAL_BUFFER_DAYS = 5  # fetch a few extra days as buffer for weekends/holidays


def _fetch_candles(instrument_key: str, from_date: date, to_date: date) -> list:
    """Raw API call — returns list of candles or empty list on failure."""
    url = (
        "https://api.upstox.com/v2/historical-candle/"
        f"{instrument_key}/day/{to_date}/{from_date}"
    )
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 401 or response.status_code == 403:
        raise PermissionError(f"Authentication failed with status {response.status_code}: {response.text[:120]}")
    if response.status_code != 200:
        print(f"  API error {response.status_code}: {response.text[:120]}")
        return []
    return response.json().get("data", {}).get("candles", [])


def _candles_to_df(candles: list) -> pd.DataFrame:
    """Convert raw candle list to a clean DataFrame."""
    df = pd.DataFrame(
        candles,
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"]
    )
    df = df.drop(columns=["OI"])
    # Strip timezone info but preserve the exact IST date instead of shifting to UTC
    # Upstox returns "2026-05-18T00:00:00+05:30". Converting this to UTC would shift it to 
    # the previous day (2026-05-17 18:30:00).
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _load_existing(symbol: str) -> pd.DataFrame | None:
    """Load the existing raw CSV for a symbol, or return None if it doesn't exist."""
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  Warning: could not read existing file for {symbol}: {e}")
        return None


def fetch_history(symbol: str, instrument_key: str, force_bootstrap: bool = False):
    """
    Smart fetch with two modes:

    BOOTSTRAP  — No raw file exists (or file is stale > STALE_THRESHOLD_DAYS):
                  Fetch the last BOOTSTRAP_DAYS (~3 months) from the API.
                  This is the first-time setup for a symbol.

    INCREMENTAL — Raw file exists and is fresh:
                  Fetch only the days missing since the last stored date,
                  merge with existing data, deduplicate, and save.
                  Skips entirely if the file is already up-to-date.
    """
    today = datetime.today().date()
    to_date = today + timedelta(days=1)   # Upstox to_date is exclusive
    
    existing_df = _load_existing(symbol)

    # ── Decide fetch mode ────────────────────────────────────────────────────
    if force_bootstrap or existing_df is None or existing_df.empty:
        # BOOTSTRAP: no data at all or forced rebuild
        mode = "bootstrap"
        existing_df = None  # Ensure we don't merge with corrupted data
        from_date = today - timedelta(days=BOOTSTRAP_DAYS)
        print(f"[BOOTSTRAP ] {symbol}: no existing data or forced rebuild -> fetching {BOOTSTRAP_DAYS}d ({from_date} -> {today})")

    else:
        last_stored_date = existing_df["Date"].max().date()
        days_since_update = (today - last_stored_date).days

        if days_since_update <= 0:
            # Already up-to-date — nothing to do
            print(f"[UP-TO-DATE] {symbol}: last date={last_stored_date}, skipping.")
            return

        if days_since_update > STALE_THRESHOLD_DAYS:
            # File exists but is too old — re-fetch full 3 months to be safe
            mode = "bootstrap"
            from_date = today - timedelta(days=BOOTSTRAP_DAYS)
            print(
                f"[STALE     ] {symbol}: last date={last_stored_date} "
                f"({days_since_update}d ago) -> full re-fetch ({from_date} -> {today})"
            )
        else:
            # INCREMENTAL: fetch only the missing window
            mode = "incremental"
            # Start from last stored date (inclusive) with a small buffer
            from_date = last_stored_date - timedelta(days=INCREMENTAL_BUFFER_DAYS)
            print(
                f"[INCREMENTAL] {symbol}: last date={last_stored_date} "
                f"-> fetching {from_date} -> {today}"
            )

    # ── Fetch from API ───────────────────────────────────────────────────────
    candles = _fetch_candles(instrument_key, from_date, to_date)

    if not candles:
        print(f"  ! No new candles returned for {symbol}.")
        return

    new_df = _candles_to_df(candles)

    # ── Merge & deduplicate ──────────────────────────────────────────────────
    if mode == "incremental" and existing_df is not None and not existing_df.empty:
        # Ensure both sides are tz-naive before concat
        existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.tz_localize(None)
        new_df["Date"] = new_df["Date"].dt.tz_localize(None) if new_df["Date"].dt.tz is not None else new_df["Date"]
        
        # ── Self-Healing Overlap Check ───────────────────────────────────────
        overlap = pd.merge(existing_df[["Date", "Close"]], new_df[["Date", "Close"]], on="Date", suffixes=("_stored", "_new"))
        if not overlap.empty:
            first_overlap = overlap.iloc[0]
            stored_price = first_overlap["Close_stored"]
            new_price = first_overlap["Close_new"]
            
            # If price differs by > 5%, Upstox applied a corporate action adjustment
            if stored_price > 0 and abs(new_price - stored_price) / stored_price > 0.05:
                print(f"[CORRUPTION] {symbol}: Corporate action detected! Stored: {stored_price}, New: {new_price}.")
                print(f"             Discarding corrupted CSV and triggering full rebuild...")
                return fetch_history(symbol, instrument_key, force_bootstrap=True)
                
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Normalise to date-only for dedup (Upstox candles are start-of-day anyway)
        combined["Date"] = combined["Date"].dt.normalize()
        # Keep the latest record per date in case of overlap
        combined = (
            combined
            .sort_values("Date")
            .drop_duplicates(subset="Date", keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = new_df
        combined["Date"] = combined["Date"].dt.normalize()

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    combined.to_csv(out_path, index=False)
    new_rows = len(new_df)
    total_rows = len(combined)
    print(f"  OK Saved {symbol}: +{new_rows} new rows | {total_rows} total rows")


import concurrent.futures

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_symbols = len(SYMBOL_MAP)
    
    def process_symbol(args):
        sym, key = args
        try:
            fetch_history(sym, key)
        except Exception as e:
            print(f"  Error processing {sym}: {e}")
            raise
            
    items = list(SYMBOL_MAP.items())
    
    # Use ThreadPoolExecutor to run up to 3 fetches concurrently (avoid 429 rate limit)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        futures = {executor.submit(process_symbol, item): item for item in items}
        
        # Track completion and propagate errors
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except PermissionError as pe:
                # Critical auth error – abort all processing
                print(f"Authentication error during processing: {pe}")
                raise
            except Exception as e:
                print(f"Error processing {future}: {e}")
            finally:
                print(f"PROGRESS:{idx}/{total_symbols}", flush=True)
