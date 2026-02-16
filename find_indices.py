
import pandas as pd
import requests
import gzip
import io

URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
print("Downloading...")
response = requests.get(URL)
response.raise_for_status()

print("Reading JSON...")
with gzip.open(io.BytesIO(response.content), "rt", encoding="utf-8") as f:
    instruments = pd.read_json(f)

print("Columns:", instruments.columns)
print("Segments:", instruments['segment'].unique())

indices_to_find = [
    "Nifty 50", "Nifty 100", "Nifty 100 ESG Sector Leaders", "Nifty 500", "Nifty Alpha 50",
    "Nifty Alpha Low Volatility 30", "Nifty Auto", "Nifty Bank", "Nifty Commodities",
    "Nifty Dividend Opportunities 50", "Nifty EV & New Age Automotive", "Nifty Financial Services",
    "Nifty Financial Services Ex-Bank", "Nifty FMCG", "Nifty Growth Sectors 15", "Nifty Healthcare",
    "Nifty India Consumption", "Nifty India Defence", "Nifty India Digital", "Nifty India Manufacturing",
    "Nifty India New Age Consumption", "Nifty Infrastructure", "Nifty IT", "Nifty Metal",
    "Nifty Midcap 100", "Nifty Midcap 150", "Nifty Midcap 50", "Nifty Midcap150 Quality 50",
    "Nifty MidSmallcap400 Momentum Quality 100", "Nifty MNC", "Nifty Next 50", "Nifty Oil & Gas",
    "Nifty Pharma", "Nifty Private Bank", "Nifty PSE", "Nifty PSU Bank", "Nifty Realty",
    "Nifty Smallcap 250", "Nifty Smallcap250 Momentum Quality 100", "Nifty Top 10 Equal Weight",
    "Nifty100 Low Volatility 30", "Nifty100 Quality 30", "Nifty200 Alpha 30", "Nifty200 Momentum 30",
    "Nifty200 Quality 30", "Nifty200 Value 30", "Nifty50 Equal Weight", "Nifty50 Shariah",
    "Nifty50 Value 20", "Nifty500 Momentum 50", "Nifty500 Multicap 50:25:25",
    "Nifty500 Multicap Momentum Quality 50", "Nifty 10 yr Benchmark G-Sec", "Nifty 1D Rate",
    "Nifty 5 yr Benchmark G-Sec", "Nifty 8–13 yr G-Sec", "Nifty AAA Bond Plus SDL Apr 2026 50:50",
    "Nifty BHARAT Bond – April 2025", "Nifty BHARAT Bond – April 2030", "Nifty BHARAT Bond – April 2031",
    "Nifty BHARAT Bond – April 2032", "Nifty BHARAT Bond – April 2033", "Nifty SDL Apr 2026 Top 20 Equal Weight"
]

# Normalize names for fuzzy matching if needed, or just case-insensitive contains
print("\nSearching for indices...")
found_indices = {}

# Try to find segment for indices
# It appears usually segment is 'NSE_INDEX' if it exists in this file, or we need to check another file.
# Based on Upstox docs, indices are often in user readable name or trading_symbol

# Let's simple print first 5 rows to see structure
print(instruments.head())

for index_name in indices_to_find:
    # Search in name or tradingsymbol
    # Standardize search
    search_term = index_name.replace(" ", "").upper() # e.g. NIFTY50
    
    # Try exact match first or contains
    # Columns to search: name, trading_symbol, short_name
    
    mask = instruments['name'].str.contains(index_name, case=False, regex=False) | \
           instruments['trading_symbol'].str.contains(index_name.replace(" ", ""), case=False, regex=False)
    
    if 'short_name' in instruments.columns:
         mask = mask | instruments['short_name'].str.contains(index_name, case=False, regex=False)

           
    matches = instruments[mask]
    
    if not matches.empty:
        # Prefer NSE_INDEX segment if multiple
        index_matches = matches[matches['segment'] == 'NSE_INDEX']
        if not index_matches.empty:
            best_match = index_matches.iloc[0]
            found_indices[index_name] = {
                "symbol": best_match.get('tradingsymbol', best_match.get('symbol')),
                "key": best_match['instrument_key']
            }
        else:
            # Fallback
            best_match = matches.iloc[0]
            found_indices[index_name] = {
                "symbol": best_match.get('tradingsymbol', best_match.get('symbol')),
                "key": best_match['instrument_key']
            }
            # print(f"Found non-index segment match for {index_name} in {best_match['segment']}")

print("\n--- RESULTS ---")
for name, data in found_indices.items():
    print(f"'{name}': '{data['key']}',") # Output in python dict format for copy paste
    
print(f"\nTotal requested: {len(indices_to_find)}")
print(f"Total found: {len(found_indices)}")
