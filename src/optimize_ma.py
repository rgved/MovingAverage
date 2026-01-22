import pandas as pd
from backtest import backtest_strategy

# ---------- Function to Recompute MAs ----------
def add_moving_averages(df, ma_type="EMA", fast=10, slow=20):
    df = df.copy()

    # Calculate moving averages
    if ma_type.upper() == "SMA":
        df["MA_Fast"] = df["Close"].rolling(window=fast).mean()
        df["MA_Slow"] = df["Close"].rolling(window=slow).mean()
    elif ma_type.upper() == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    else:
        raise ValueError("ma_type must be SMA or EMA")

    # Generate signals and crossovers
    df["Signal"] = 0
    df.loc[df["MA_Fast"] > df["MA_Slow"], "Signal"] = 1
    df.loc[df["MA_Fast"] < df["MA_Slow"], "Signal"] = -1
    df["Crossover"] = df["Signal"].diff()

    return df

# ---------- Optimizer Function ----------
def optimize_ma_windows(symbol="INFY.NS", ma_pairs=None, ma_type="EMA"):
    if ma_pairs is None:
        ma_pairs = [(10, 20), (12, 26), (20, 50), (50, 100), (50, 200)]

    # Load data
    df = pd.read_csv(f"data/processed/{symbol}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df_recent = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(months=3))]

    results = []

    # Loop over MA pairs
    for fast, slow in ma_pairs:
        df_pair = add_moving_averages(df_recent, ma_type=ma_type, fast=fast, slow=slow)

        metrics, _ = backtest_strategy(
            df_pair,
            exit_mode="time",
            hold_days=7,
            stop_loss=0.03,
            take_profit=0.05,
            cost_bps=15
        )

        results.append({
            "Symbol": symbol,
            "MA_Type": ma_type,
            "MA_Pair": f"{fast}/{slow}",
            "Return": metrics["Total Return"],
            "WinRate": metrics["Win Rate"],
            "Sharpe": metrics["Sharpe Ratio"],
            "MaxDD": metrics["Max Drawdown"],
            "Trades": metrics["Trades"]
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Return", ascending=False).reset_index(drop=True)

    print(f"\n📊 Optimization Results for {symbol} ({ma_type})")
    print(results_df)

    # Save to reports/
    out_path = f"reports/{symbol.replace('.','_')}_{ma_type}_optimization.csv"
    results_df.to_csv(out_path, index=False)
    print(f"✅ Saved results → {out_path}")

    return results_df

# ---------- Run for Single Symbol ----------
if __name__ == "__main__":
    # Try both EMA and SMA for comparison
    for ma_type in ["EMA", "SMA"]:
        optimize_ma_windows("HDFCBANK.NS", ma_type=ma_type)
