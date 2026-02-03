import pandas as pd
import numpy as np
import os
from backtest import backtest_strategy
from features import add_moving_averages, generate_signals

# ---------- PATH SETUP ----------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

def optimize_ma_pairs(symbol, fast_range=range(5, 51, 5), slow_range=range(10, 201, 10), min_trades=5):
    """
    Optimizes MA pairs for a given symbol based on Sharpe Ratio.
    Filters out pairs with fewer than min_trades.
    """
    print(f"\n Optimizing MA Pairs for {symbol}...")
    
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return None

    # Load Data
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Use only recent data if needed (e.g., last 2 years) to keep it relevant
    # df = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(years=2))]

    results = []

    for slow in slow_range:
        for fast in fast_range:
            if fast >= slow:
                continue
            
            # 1. Calculate Indicators
            # Using EMA as default as per user preference in previous context, or generic MA
            # The user prompt said "Calculate EMAs", so we use EMA.
            try:
                df_test = add_moving_averages(df, ma_type="EMA", fast=fast, slow=slow)
                df_test = generate_signals(df_test)
                
                # 2. Backtest
                metrics, trades = backtest_strategy(
                    df_test,
                    exit_mode="opposite", 
                    cost_bps=15
                )

                # 3. Store Results
                results.append({
                    "symbol": symbol,
                    "fast": fast,
                    "slow": slow,
                    "sharpe": metrics["sharpe"],
                    "win_rate": metrics["win_rate"],
                    "trades": metrics["trades"],
                    "total_return": metrics["total_return"] * 100,
                    "max_drawdown": metrics["max_drawdown"] * 100
                })
            except Exception as e:
                print(f" Error for {fast}/{slow}: {e}")
                continue

    # 4. Analyze Results
    if not results:
        print("No results generated.")
        return None

    results_df = pd.DataFrame(results)

    # Filter by minimum trades
    qualified_df = results_df[results_df["Trades"] >= min_trades]

    if qualified_df.empty:
        print(f" No pairs met the minimum trade threshold ({min_trades}). Returning stats for all pairs (top 5 by Sharpe).")
        qualified_df = results_df # Fallback to showing something

    # Select best by Sharpe Ratio
    best_result = qualified_df.sort_values("Sharpe", ascending=False).iloc[0]
    
    print("\n BEST MA PAIR FOUND:")
    print(best_result)

    # Save details
    out_path = os.path.join(REPORTS_DIR, f"{symbol}_ma_optimization.csv")
    qualified_df.sort_values("Sharpe", ascending=False).to_csv(out_path, index=False)
    print(f" Full results saved to: {out_path}")

    return best_result

if __name__ == "__main__":
    # Example usage
    target_symbol = "HDFCBANK.NS" 
    optimize_ma_pairs(target_symbol, 
                      fast_range=range(5, 60, 5), 
                      slow_range=range(20, 201, 10),
                      min_trades=5)
