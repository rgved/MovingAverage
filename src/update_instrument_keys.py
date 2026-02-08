import os
import requests
import pandas as pd
import json
import gzip
import io

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUMENTS_URL = "https://assets.upstox.com/feed/instruments/cse-instruments.csv.gz" # Wait, checking Upstox doc on my own memory?
# Actually Upstox provides a JSON/CSV file.
# URL: https://assets.upstox.com/feed/instruments/nse-eq.csv.gz (for NSE Equity)
# Let's try downloading the full instruments file or specific NSE EQ.

# Upstox Documentation says: https://assets.upstox.com/feed/instruments/nse-eq.csv.gz
NSE_EQ_URL = "https://assets.upstox.com/feed/instruments/nse-eq.csv.gz"

def update_symbol_map():
    print("Downloading NSE Equity instruments...")
    try:
        response = requests.get(NSE_EQ_URL)
        response.raise_for_status()
        
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            df = pd.read_csv(f)
            
        print(f"Downloaded {len(df)} instruments.")
        
        # We need a map: Symbol -> Instrument Key
        # Upstox CSV cols: instrument_key, exchange_token, tradingsymbol, name, last_price, expiry, strike, tick_size, lot_size, instrument_type, isin, exchange
        
        # Filter for EQ
        # It's already nse-eq file.
        
        # Create map: Trading Symbol -> Instrument Key
        # Example: 'RELIANCE' -> 'NSE_EQ|INE002A01018' (Wait, key format?)
        # instrument_key column usually has "NSE_EQ|..."
        
        symbol_map = {}
        for _, row in df.iterrows():
            sym = row['tradingsymbol']
            key = row['instrument_key']
            symbol_map[sym] = key
            
        # Save to file
        map_file = os.path.join(BASE_DIR, "upstox_symbol_map.json")
        
        # Load existing if any? 
        # Actually we want to OVERWRITE or MERGE.
        # Let's MERGE to be safe, but prioritize new keys.
        if os.path.exists(map_file):
            with open(map_file, 'r') as f:
                existing = json.load(f)
            existing.update(symbol_map)
            symbol_map = existing
            
        with open(map_file, 'w') as f:
            json.dump(symbol_map, f, indent=4)
            
        print(f"Updated symbol map with {len(symbol_map)} keys.")
        return symbol_map
        
    except Exception as e:
        print(f"Error updating symbol map: {e}")
        return {}

if __name__ == "__main__":
    update_symbol_map()
