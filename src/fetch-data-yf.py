import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# ---------- LOAD ENV ----------
load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise RuntimeError("❌ UPSTOX_ACCESS_TOKEN not found in .env")

# ---------- CONSTANTS ----------
BASE_URL = "https://api.upstox.com/v2"
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

# ---------- SYMBOL MAP (IMPORTANT) ----------
# Upstox does NOT use Yahoo-style symbols
SYMBOL_MAP = {
     
    "TORNTPOWER.NS": "NSE_EQ|INE813H01021",
    "TARIL.NS": "NSE_EQ|INE763I01026",
    "TRENT.NS": "NSE_EQ|INE849A01020",
    "TRIDENT.NS": "NSE_EQ|INE064C01022",
    "TRIVENI.NS": "NSE_EQ|INE256C01024",
    "TRITURBINE.NS": "NSE_EQ|INE152M01016",
    "TIINDIA.NS": "NSE_EQ|INE974X01010",
    "UCOBANK.NS": "NSE_EQ|INE691A01018",
    "UNOMINDA.NS": "NSE_EQ|INE405E01023",
    "UPL.NS": "NSE_EQ|INE628A01036",
    "UTIAMC.NS": "NSE_EQ|INE094J01016",
    "ULTRACEMCO.NS": "NSE_EQ|INE481G01011",
    "UNIONBANK.NS": "NSE_EQ|INE692A01016",
    "UBL.NS": "NSE_EQ|INE686F01025",
    "UNITDSPR.NS": "NSE_EQ|INE854D01024",
    "USHAMART.NS": "NSE_EQ|INE228A01035",
    "VGUARD.NS": "NSE_EQ|INE951I01027",
    "DBREALTY.NS": "NSE_EQ|INE879I01012",
    "VTL.NS": "NSE_EQ|INE825A01020",
    "VBL.NS": "NSE_EQ|INE200M01039",
    "MANYAVAR.NS": "NSE_EQ|INE825V01034",
    "VEDL.NS": "NSE_EQ|INE205A01025",
    "VENTIVE.NS": "NSE_EQ|INE781S01027",
    "VIJAYA.NS": "NSE_EQ|INE043W01024",
    "VMM.NS": "NSE_EQ|INE01EA01019",
    "IDEA.NS": "NSE_EQ|INE669E01016",
    "VOLTAS.NS": "NSE_EQ|INE226A01021",
    "WAAREEENER.NS": "NSE_EQ|INE377N01017",
    "WELCORP.NS": "NSE_EQ|INE191B01025",
    "WELSPUNLIV.NS": "NSE_EQ|INE192B01031",
    "WHIRLPOOL.NS": "NSE_EQ|INE716A01013",
    "WIPRO.NS": "NSE_EQ|INE075A01022",
    "WOCKPHARMA.NS": "NSE_EQ|INE049B01025",
    "YESBANK.NS": "NSE_EQ|INE528G01035",
    "ZFCVINDIA.NS": "NSE_EQ|INE342J01019",
    "ZEEL.NS": "NSE_EQ|INE256A01028",
    "ZENTEC.NS": "NSE_EQ|INE251B01027",
    "ZENSARTECH.NS": "NSE_EQ|INE520A01027",
    "ZYDUSLIFE.NS": "NSE_EQ|INE010B01027",
    "ECLERX.NS": "NSE_EQ|INE738I01010",
}

# ---------- FETCH FUNCTION ----------
def get_upstox_data(symbol, out_dir="data/raw", days=365):
    if symbol not in SYMBOL_MAP:
        print(f"⚠️ No Upstox instrument mapping for {symbol}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    instrument_key = SYMBOL_MAP[symbol]
    to_date = datetime.today().date()
    from_date = to_date - timedelta(days=days)

    url = f"{BASE_URL}/historical-candle/{instrument_key}/day/{to_date}/{from_date}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"❌ API error for {symbol}: {response.text}")
        return None

    data = response.json()

    candles = data.get("data", {}).get("candles", [])

    if not candles:
        print(f"⚠️ No data returned for {symbol}")
        return None

    # Upstox candle format:
    # [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(
        candles,
        columns=["Date", "Open", "High", "Low", "Close", "Volume"]
    )

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date")

    path = os.path.join(out_dir, f"{symbol}.csv")
    df.to_csv(path, index=False)

    print(f"✅ Saved {symbol}: {len(df)} rows")
    return df
if __name__ == "__main__":
    symbols = [
       
    "RRKABEL.NS",
    "RBLBANK.NS",
    "RECLTD.NS",
    "RHIM.NS",
    "RITES.NS",
    "RADICO.NS",
    "RVNL.NS",
    "RAILTEL.NS",
    "RAINBOW.NS",
    "RKFORGE.NS",
    "RCF.NS",
    "REDINGTON.NS",
    "RELIANCE.NS",
    "RELINFRA.NS",
    "RPOWER.NS",
    "SBFC.NS",
    "SBICARD.NS",
    "SBILIFE.NS",
    "SJVN.NS",
    "SKFINDIA.NS",
    "SRF.NS",
    "SAGILITY.NS",
    "SAILIFE.NS",
    "SAMMAANCAP.NS",
    "MOTHERSON.NS",
    "SAPPHIRE.NS",
    "SARDAEN.NS",
    "SAREGAMA.NS",
    "SCHAEFFLER.NS",
    "SCHNEIDER.NS",
]


