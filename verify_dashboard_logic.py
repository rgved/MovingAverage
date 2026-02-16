import pandas as pd
import numpy as np
import os

# Mock paths
BASE_DIR = r"c:\Users\rgved\Downloads\MOVINGAVERAGE\MovingAverage"
reports_dir = os.path.join(BASE_DIR, "reports")
processed_dir = os.path.join(BASE_DIR, "data", "processed")

# Test Stock
symbol = "3MINDIA.NS"
report_file = os.path.join(reports_dir, "3MINDIA_NS_1y_dynamic_trend_noise_optimization.csv")
price_file = os.path.join(processed_dir, "3MINDIA.NS.csv")

print(f"Testing with Symbol: {symbol}")

# 1. Simulate Reading Report (Selected Row in Dashboard)
try:
    rep = pd.read_csv(report_file)
    best = rep.iloc[0]
    best_ma_type = best["MA_Type"]
    best_ma_pair = best["MA_Pair"]
    print(f"Report Best Strategy: {best_ma_type} {best_ma_pair}")
except Exception as e:
    print(f"Error reading report: {e}")
    exit()

# 2. Simulate Logic in App.py
# Mock Selected Row Data
selected_rows = pd.DataFrame([best])

# --- LOGIC START (From App.py) ---
if isinstance(selected_rows, pd.DataFrame):
    best_ma_type_viz = selected_rows.iloc[0]["MA_Type"] # In app it uses "Best MA Type" col name which comes from `rows` construction
    best_ma_pair_viz = selected_rows.iloc[0]["MA_Pair"] # In app "Best MA Pair"
else:
    best_ma_type_viz = selected_rows[0]["MA_Type"]
    best_ma_pair_viz = selected_rows[0]["MA_Pair"]

# Parse Pair
try:
    fast_viz, slow_viz = map(int, best_ma_pair_viz.split("/"))
    ma_type_viz = best_ma_type_viz
    print(f"Parsed Visualization Params: Type={ma_type_viz}, Fast={fast_viz}, Slow={slow_viz}")
except Exception as e:
    print(f"Parsing failed: {e}")
    exit()

# 3. Load Price Data & Calc MAs
df = pd.read_csv(price_file)
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date")

if ma_type_viz == "EMA":
    df["MA_Fast"] = df["Close"].ewm(span=fast_viz, adjust=False).mean()
    df["MA_Slow"] = df["Close"].ewm(span=slow_viz, adjust=False).mean()
else:
    df["MA_Fast"] = df["Close"].rolling(fast_viz).mean()
    df["MA_Slow"] = df["Close"].rolling(slow_viz).mean()

df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
df["Crossover"] = df["Signal"].diff()

# 4. Check for Crossovers
crossovers = df[df["Crossover"].abs() == 2]
last_crossover = crossovers.iloc[-1]["Date"] if not crossovers.empty else "None"
print(f"Last Crossover Found: {last_crossover}")

print("Verification Successful: Logic parses strategy and calculates crossovers.")
