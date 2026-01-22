import os
import pandas as pd
from optimize_ma import optimize_ma_windows

def run_all_optimizations(
    processed_dir="data/processed",
    ma_types=["EMA", "SMA"],
    ma_pairs=None
):
    if ma_pairs is None:
        ma_pairs = [(10, 20), (12, 26), (20, 50), (50, 100), (50, 200)]

    all_results = []

    files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]
    for file in files:
        symbol = file.replace(".csv", "")
        for ma_type in ma_types:
            try:
                df_res = optimize_ma_windows(symbol, ma_pairs=ma_pairs, ma_type=ma_type)
                all_results.append(df_res)
            except Exception as e:
                print(f"⚠️ Error optimizing {symbol} ({ma_type}): {e}")

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Find best config per stock
    best_per_stock = combined.sort_values(["Symbol", "Return"], ascending=[True, False]).groupby("Symbol").head(1)
    
    # Save all + best results
    os.makedirs("reports", exist_ok=True)
    combined.to_csv("reports/all_optimization_results.csv", index=False)
    best_per_stock.to_csv("reports/best_per_stock.csv", index=False)

    print("✅ All optimization results saved to reports/all_optimization_results.csv")
    print("🏆 Best-performing setups per stock saved to reports/best_per_stock.csv")

if __name__ == "__main__":
    run_all_optimizations()
