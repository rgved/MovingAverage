import os
import shutil
import pandas as pd
import numpy as np
import sys

# Ensure src is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from features import process_all
from constants import FNO_STOCKS

def verify():
    print("Beginning Verification...")
    
    # Setup Test Dirs
    TEST_RAW = os.path.join(CURRENT_DIR, "..", "data", "test_raw")
    TEST_PROCESSED = os.path.join(CURRENT_DIR, "..", "data", "test_processed")
    
    if os.path.exists(TEST_RAW): shutil.rmtree(TEST_RAW)
    if os.path.exists(TEST_PROCESSED): shutil.rmtree(TEST_PROCESSED)
    
    os.makedirs(TEST_RAW)
    
    # 1. Create Non-F&O Stock
    non_fno_symbol = "DUMMYNONFNO"
    df_non = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=50),
        "Close": np.random.rand(50) * 100
    })
    df_non.to_csv(os.path.join(TEST_RAW, f"{non_fno_symbol}.csv"), index=False)
    
    # 2. Create F&O Stock with Crossover
    # Use a real F&O symbol from constants to pass filter
    fno_symbol = FNO_STOCKS[0] # e.g. AARTIIND or similar
    if "HDFCBANK" in FNO_STOCKS: fno_symbol = "HDFCBANK"
    
    # Synthesize Crossover: V-Shape
    # Downtrend then Uptrend to force -1 -> 1 transition
    dates = pd.date_range("2024-01-01", periods=100)
    # Price drops from 100 to 50 over 50 days, then rises to 100
    prices = list(np.linspace(100, 50, 50)) + list(np.linspace(50, 100, 50))
    
    df_fno = pd.DataFrame({
        "Date": dates,
        "Close": prices
    })
    df_fno.to_csv(os.path.join(TEST_RAW, f"{fno_symbol}.csv"), index=False)
    
    print(f"Created mocked data in {TEST_RAW}")
    
    # 3. Run Pipeline
    print("Running features.py process_all...")
    process_all(data_dir=TEST_RAW, out_dir=TEST_PROCESSED, ma_type="SMA", fast=10, slow=20)
    
    # 4. Verify F&O Filter
    if os.path.exists(os.path.join(TEST_PROCESSED, f"{non_fno_symbol}.csv")):
        print(f"FAILED: {non_fno_symbol} was processed but should have been skipped.")
    else:
        print(f"PASSED: {non_fno_symbol} was correctly skipped.")
        
    # 5. Verify Output Content
    out_file = os.path.join(TEST_PROCESSED, f"{fno_symbol}.csv")
    if not os.path.exists(out_file):
        print(f"FAILED: {fno_symbol} was not processed.")
        return
        
    df_out = pd.read_csv(out_file)
    print(f"PASSED: {fno_symbol} processed. Columns: {list(df_out.columns)}")
    
    # Check Date Column
    if "Date" not in df_out.columns:
        print("FAILED: Date column missing.")
    else:
        print("PASSED: Date column present.")
        
    # Check Crossover Logic
    # Find where Crossover == 2
    crossovers = df_out[df_out["Crossover"] == 2]
    
    print("\nDEBUG DATA STATS:")
    print(df_out["Signal"].value_counts())
    print("Non-zero Crossovers:")
    print(df_out[df_out["Crossover"] != 0])
    
    if crossovers.empty:
        print("WARNING: No bullish crossover found in synthetic data.")
    else:
        # Check alignment
        idx = crossovers.index[0]
        row = df_out.loc[idx]
        prev = df_out.loc[idx-1]
        
        print(f"Crossover at Index {idx}, Date {row['Date']}")
        print(f"  Prev: Fast {prev['MA_Fast']:.2f} vs Slow {prev['MA_Slow']:.2f} (Diff {prev['MA_Fast']-prev['MA_Slow']:.2f})")
        print(f"  Curr: Fast {row['MA_Fast']:.2f} vs Slow {row['MA_Slow']:.2f} (Diff {row['MA_Fast']-row['MA_Slow']:.2f})")
        
        if row["MA_Fast"] > row["MA_Slow"] and prev["MA_Fast"] <= prev["MA_Slow"]:
            print("PASSED: Crossover 2 aligned with Fast crossing above Slow.")
        else:
            print("FAILED: Crossover alignment mismatch.")

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        print(f"Verification Failed with Error: {e}")
        import traceback
        traceback.print_exc()
