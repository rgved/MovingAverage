#In this file we perform backtesting [applying a trading strategy to our past data]

import pandas as pd
import numpy as np

# ---------- Helper Metrics ----------

def max_drawdown(equity):
    """Calculate maximum drawdown (worst loss from peak)."""
    peak = equity.cummax()
    drawdown = (equity / peak) - 1
    return drawdown.min()

def sharpe_ratio(daily_returns, periods_per_year=252):
    """Annualized Sharpe ratio based on daily returns."""
    mean = daily_returns.mean()
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (mean / std) * np.sqrt(periods_per_year)

# ---------- Backtest Function ----------
def backtest_strategy(
    df,
    entry_col="Crossover",
    cost_bps=15,
    exit_mode="opposite",
    hold_days=10,
    stop_loss=None,       # e.g., 0.03 = 3%
    take_profit=None      # e.g., 0.05 = 5%
):
    """
    Runs a long-only MA crossover backtest with optimization levers.
    Entry: Bullish cross (next day's open)
    Exit: Opposite cross, time-based, stop-loss, or take-profit.
    """

    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    in_position = False
    entry_price = 0.0
    entry_date = None
    trades = []
    equity = [1.0]  # start with 1 unit capital

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # ENTRY condition: bullish crossover yesterday + price confirmation
        if not in_position and prev[entry_col] == 2:
            # Confirm price is above the slow MA → true uptrend
            if prev["Close"] > prev["MA_Slow"]:
                entry_price = curr["Open"]
                entry_date = curr["Date"]
                in_position = True
                continue

        # EXIT condition: bearish crossover yesterday + price confirmation
        if in_position:
            exit_condition = False
            # Only exit if confirmed downtrend (Close below slow MA)
            if exit_mode == "opposite" and prev[entry_col] == -2 and prev["Close"] < prev["MA_Slow"]:
                exit_condition = True
            elif exit_mode == "time" and (curr["Date"] - entry_date).days >= hold_days:
                exit_condition = True


            # Opposite crossover
            if exit_mode == "opposite" and prev[entry_col] == -2:
                exit_condition = True
                exit_reason = "Opposite crossover"

            # Time-based exit
            elif exit_mode == "time" and (curr["Date"] - entry_date).days >= hold_days:
                exit_condition = True
                exit_reason = f"{hold_days}-day exit"

            # Stop-loss condition
            elif stop_loss and curr["Low"] <= entry_price * (1 - stop_loss):
                exit_condition = True
                exit_reason = f"Stop loss ({stop_loss*100:.1f}%)"

            # Take-profit condition
            elif take_profit and curr["High"] >= entry_price * (1 + take_profit):
                exit_condition = True
                exit_reason = f"Take profit ({take_profit*100:.1f}%)"

            if exit_condition:
                exit_price = curr["Open"]
                gross_return = (exit_price / entry_price) - 1
                cost = 2 * (cost_bps / 10000)  # entry + exit
                net_return = gross_return - cost
                trades.append({
                    "EntryDate": entry_date,
                    "ExitDate": curr["Date"],
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "NetReturn": net_return,
                    "ExitReason": exit_reason
                })
                in_position = False

        # Track equity over time (approximate)
        equity.append(equity[-1] * (1 + df["Close"].pct_change().fillna(0).iloc[i]))

    # ---------- Metrics ----------
    n_trades = len(trades)
    if n_trades > 0:
        win_rate = len([t for t in trades if t["NetReturn"] > 0]) / n_trades
        total_return = np.prod([1 + t["NetReturn"] for t in trades]) - 1
    else:
        win_rate = 0
        total_return = 0

    equity_series = pd.Series(equity)
    daily_returns = equity_series.pct_change().fillna(0)

    metrics = {
        "Total Return": round(total_return * 100, 2),
        "Max Drawdown": round(max_drawdown(equity_series) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(daily_returns), 2),
        "Win Rate": round(win_rate * 100, 2),
        "Trades": n_trades
    }

    return metrics, trades


if __name__ == "__main__":


# Load processed file
    df = pd.read_csv("data/processed/HDFCBANK.NS.csv")

    # Run backtest for 3 months
    df["Date"] = pd.to_datetime(df["Date"])
    df_recent = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(months=3))]

    metrics, trades = backtest_strategy(df_recent, cost_bps=15, exit_mode="opposite")

    print("📊 Backtest Results for INFY.NS (3 months):")
    for k, v in metrics.items():
        print(f"{k:15s}: {v}")

    print(f"\nNumber of trades: {len(trades)}")
    if trades:
        print("First trade sample:", trades[0])











































#===============OLD STRATEGY BELOW===================
        
# def backtest_strategy(
#     df,
#     entry_col="Crossover",
#     cost_bps=15,
#     exit_mode="opposite",
#     hold_days=10
# ):
#     """
#     Runs a long-only MA crossover backtest.
#     Entry on bullish cross (next day's open).
#     Exit on opposite crossover or time-based.
#     """

#     df = df.copy().sort_values("Date").reset_index(drop=True)
#     df["Date"] = pd.to_datetime(df["Date"])

#     in_position = False
#     entry_price = 0.0
#     entry_date = None
#     trades = []
#     equity = [1.0]  # start with 1 unit capital

#     for i in range(1, len(df)):
#         prev = df.iloc[i - 1]
#         curr = df.iloc[i]

#         # ENTRY condition: bullish crossover yesterday
#         if not in_position and prev[entry_col] == 2:
#             entry_price = curr["Open"]
#             entry_date = curr["Date"]
#             in_position = True
#             continue

#         # EXIT condition
#         if in_position:
#             exit_condition = False
#             if exit_mode == "opposite" and prev[entry_col] == -2:
#                 exit_condition = True
#             elif exit_mode == "time":
#                 if (curr["Date"] - entry_date).days >= hold_days:
#                     exit_condition = True

#             if exit_condition:
#                 exit_price = curr["Open"]
#                 gross_return = (exit_price / entry_price) - 1
#                 cost = 2 * (cost_bps / 10000)  # entry + exit
#                 net_return = gross_return - cost
#                 trades.append({
#                     "EntryDate": entry_date,
#                     "ExitDate": curr["Date"],
#                     "EntryPrice": entry_price,
#                     "ExitPrice": exit_price,
#                     "NetReturn": net_return
#                 })
#                 in_position = False

#         # For cumulative equity curve (approximation)
#         equity.append(equity[-1] * (1 + df["Close"].pct_change().fillna(0).iloc[i]))

#     # Compute metrics
#     n_trades = len(trades)
#     if n_trades > 0:
#         win_rate = len([t for t in trades if t["NetReturn"] > 0]) / n_trades
#         total_return = np.prod([1 + t["NetReturn"] for t in trades]) - 1
#     else:
#         win_rate = 0
#         total_return = 0

#     equity_series = pd.Series(equity)
#     daily_returns = equity_series.pct_change().fillna(0)

#     metrics = {
#         "Total Return": round(total_return * 100, 2),
#         "Max Drawdown": round(max_drawdown(equity_series) * 100, 2),
#         "Sharpe Ratio": round(sharpe_ratio(daily_returns), 2),
#         "Win Rate": round(win_rate * 100, 2),
#         "Trades": n_trades
#     }

#     return metrics, trades
