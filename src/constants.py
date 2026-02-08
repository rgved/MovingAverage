
NIFTY_50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "LTIM", "M&M", "MARUTI", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SUNPHARMA", "SHRIRAMFIN", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TCS", "TECHM", "TITAN", "ULTRACEMCO",
    "WIPRO"
]

FNO_STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT",
    "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN",
    "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG",
    "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT",
    "CANBK", "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB",
    "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DIVISLAB", "DIXON",
    "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP",
    "GRANULES", "GRASIM", "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH",
    "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDCOPPER",
    "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA",
    "IDFC", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM", "INDIAMART",
    "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC", "IPCALAB",
    "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL", "JUBLFOOD",
    "KOTAKBANK", "L&TFH", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LT",
    "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO",
    "MARUTI", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON",
    "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR",
    "NESTLEIND", "NMDC", "NTPC", "OBERIRLTY", "OFSS", "ONGC", "PAGEIND",
    "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND",
    "PNB", "POLYCAB", "POWERGRID", "PVRINOX", "RAMCOCEM", "RBLBANK",
    "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN",
    "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SRF", "SUNPHARMA", "SUNTV",
    "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAMOTORS",
    "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS",
    "WHIRLPOOL", "WIPRO", "ZEEL", "ZYDUSLIFE", "ETERNAL", "NYKAA", "PAYTM",
    "ZOMATO", "ADANIGREEN", "ADANIENSOL", "NHPC", "JSWENERGY", "ANGELONE", 
    "BANKINDIA", "BSE", "CDSL", "JIOFIN", "DMART", "VBL", "CESC", "DELHIVERY", 
    "TATAELXSI", "HUDCO", "YESBANK", "CASTROLIND", "GLAND", "NBCC", "PHOENIXLTD",
    "SOLARINDS", "TORNTPOWER"
]

INDICES = [
    "Nifty 50", "Nifty 100", "Nifty 500", "Nifty Alpha 50", "Nifty Auto",
    "Nifty Bank", "Nifty Commodities", "Nifty FMCG", "Nifty Healthcare",
    "Nifty IT", "Nifty Metal", "Nifty Midcap 100", "Nifty Midcap 150",
    "Nifty Midcap 50", "Nifty MNC", "Nifty Next 50", "Nifty Pharma",
    "Nifty PSE", "Nifty PSU Bank", "Nifty Realty", "Nifty200 Alpha 30",
    "Nifty200 Value 30", "Nifty50 Shariah", "Nifty50 Value 20"
]

import os
import pandas as pd

def load_custom_universe(file_path):
    """Loads symbols from a CSV file."""
    try:
        if not os.path.exists(file_path):
            return []
        
        df = pd.read_csv(file_path)
        # Look for SYMBOL column (case insensitive)
        col_map = {c.upper(): c for c in df.columns}
        if "SYMBOL" in col_map:
            symbols = df[col_map["SYMBOL"]].dropna().unique().tolist()
            # Clean symbols and add .NS if missing (assuming NSE)
            # But upstox might need specific format. 
            # The existing code appends .NS in fetch-data-upstox.py Line 45: f"{symbol}.NS"
            # So here we just return the raw symbols like "RELIANCE".
            return [str(s).strip() for s in symbols]
        else:
            return []
    except Exception as e:
        print(f"Error loading custom universe: {e}")
        return []

# Dynamic load
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSE_ALL_FILE = os.path.join(BASE_DIR, "all_nse_stocks.csv")
ALL_NSE_SYMBOLS = load_custom_universe(NSE_ALL_FILE)
