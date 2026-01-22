import pandas as pd
import numpy as np
from backtest import backtest_strategy

# ---------- Helper: Compute Volatility ----------
def compute_volatility(df, window=20):
    """Rolling volatility (standard deviation of daily returns)."""
    return df["Close"].pct_change().rolling(window).std().iloc[-1]

# ---------- Helper: Compute Trend Strength ----------
def compute_trend_strength(df, window=20):
    """
    Measures % change in price over last N days.
    Approximation of trend direction & strength (like a simple ADX proxy).
    """
    if len(df) < window:
        return 0
    start_price = df["Close"].iloc[-window]
    end_price = df["Close"].iloc[-1]
    return abs(end_price - start_price) / start_price

# ---------- Add Moving Averages ----------
def add_moving_averages(df, ma_type="EMA", fast=10, slow=20):
    df = df.copy()
    if ma_type.upper() == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    elif ma_type.upper() == "SMA":
        df["MA_Fast"] = df["Close"].rolling(window=fast).mean()
        df["MA_Slow"] = df["Close"].rolling(window=slow).mean()
    else:
        raise ValueError("ma_type must be EMA or SMA")

    # Signals
    df["Signal"] = 0
    df.loc[df["MA_Fast"] > df["MA_Slow"], "Signal"] = 1
    df.loc[df["MA_Fast"] < df["MA_Slow"], "Signal"] = -1
    df["Crossover"] = df["Signal"].diff()
    return df

# ---------- Smart MA Selector ----------
def select_ma_type(vol, trend, vol_threshold=0.01, trend_threshold=0.05):
    """
    Dynamically decide whether to use EMA or SMA based on:
    - volatility (> 1%)
    - trend strength (> 5%)
    """
    if vol > vol_threshold and trend > trend_threshold:
        return "EMA"  # volatile + trending → react faster
    else:
        return "SMA"  # calm or sideways → stay smooth

# ---------- Dynamic Optimizer ----------
def optimize_dynamic_trend(symbol, ma_pairs=None, vol_threshold=0.01, trend_threshold=0.05):
    if ma_pairs is None:
        ma_pairs = [(10, 20), (12, 26), (20, 50), (50, 100), (50, 200)]

    print(f"\n🔍 Running dynamic + trend-aware optimization for {symbol}...")

    df = pd.read_csv(f"data/processed/{symbol}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df_recent = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(months=3))]

    # Compute regime stats
    vol = compute_volatility(df_recent)
    trend = compute_trend_strength(df_recent)
    ma_type = select_ma_type(vol, trend, vol_threshold, trend_threshold)

    print(f"📈 Volatility = {vol:.2%}, Trend = {trend:.2%} → Using {ma_type}")

    results = []

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
            "Volatility": round(vol * 100, 2),
            "TrendStrength": round(trend * 100, 2),
            "MA_Type": ma_type,
            "MA_Pair": f"{fast}/{slow}",
            "Return": metrics["Total Return"],
            "WinRate": metrics["Win Rate"],
            "Sharpe": metrics["Sharpe Ratio"],
            "MaxDD": metrics["Max Drawdown"],
            "Trades": metrics["Trades"]
        })

    results_df = pd.DataFrame(results).sort_values("Return", ascending=False).reset_index(drop=True)
    out_path = f"reports/{symbol.replace('.', '_')}_dynamic_trend_optimization.csv"
    results_df.to_csv(out_path, index=False)

    print(results_df.head(3))
    print(f"✅ Saved → {out_path}")
    return results_df

# ---------- Batch Runner ----------
def run_all_dynamic_trend(symbols):
    all_results = []
    for sym in symbols:
        try:
            df_res = optimize_dynamic_trend(sym)
            all_results.append(df_res.head(1))  # best pair per stock
        except Exception as e:
            print(f"⚠️ Error for {sym}: {e}")
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv("reports/best_dynamic_trend_summary.csv", index=False)
        print("\n🏆 Saved final summary → reports/best_dynamic_trend_summary.csv")

if __name__ == "__main__":
    symbols = ["ADANIENT.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "RELIANCE.NS","ICICIBANK.NS", "ITC.NS", "MARUTI.NS","TATASTEEL.NS","LT.NS","SBI.NS"]
    run_all_dynamic_trend(symbols)
