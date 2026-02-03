import pandas as pd
import numpy as np

# ---------- Helper Metrics ----------

def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = (equity / peak) - 1
    return drawdown.min()

def sharpe_ratio(daily_returns, periods_per_year=252):
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
    stop_loss=None,
    take_profit=None
):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    in_position = False
    entry_price = None
    entry_date = None

    trades = []
    equity = [1.0]
    position = 0  # 1 = long, 0 = flat

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # ---------- ENTRY ----------
        if not in_position and prev[entry_col] == 2 and prev["Close"] > prev["MA_Slow"]:
            entry_price = curr["Open"]
            entry_date = curr["Date"]
            in_position = True
            position = 1

        # ---------- EXIT ----------
        elif in_position:
            exit_condition = False
            exit_reason = None

            if exit_mode == "opposite" and prev[entry_col] == -2:
                exit_condition = True
                exit_reason = "Opposite crossover"

            elif exit_mode == "time" and (curr["Date"] - entry_date).days >= hold_days:
                exit_condition = True
                exit_reason = f"{hold_days}-day exit"

            elif stop_loss and curr["Low"] <= entry_price * (1 - stop_loss):
                exit_condition = True
                exit_reason = "Stop loss"

            elif take_profit and curr["High"] >= entry_price * (1 + take_profit):
                exit_condition = True
                exit_reason = "Take profit"

            if exit_condition:
                exit_price = curr["Open"]
                gross_return = (exit_price / entry_price) - 1
                cost = 2 * (cost_bps / 10000)
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
                position = 0
                entry_price = None
                entry_date = None

        # ---------- Equity Curve ----------
        daily_ret = position * df["Close"].pct_change().iloc[i]
        equity.append(equity[-1] * (1 + (daily_ret if not np.isnan(daily_ret) else 0)))

    # ---------- METRICS ----------
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["NetReturn"] > 0)
    losses = n_trades - wins

    win_rate = wins / n_trades if n_trades > 0 else 0.0
    total_return = np.prod([1 + t["NetReturn"] for t in trades]) - 1 if n_trades > 0 else 0.0

    equity_series = pd.Series(equity)
    daily_returns = equity_series.pct_change().fillna(0)

    metrics = {
        "total_return": total_return,   # fraction
        "max_drawdown": max_drawdown(equity_series),
        "sharpe": sharpe_ratio(daily_returns),
        "win_rate": win_rate,           # fraction
        "wins": wins,
        "losses": losses,
        "trades": n_trades
    }

    return metrics, trades


if __name__ == "__main__":

    df = pd.read_csv("data/processed/HDFCBANK.NS.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Last 3 months
    df_recent = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=3)]

    metrics, trades = backtest_strategy(df_recent)

    print("\n📊 Backtest Results (3 Months)\n" + "-" * 35)
    print(f"Total Return  : {metrics['total_return']*100:.2f}%")
    print(f"Max Drawdown  : {metrics['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio  : {metrics['sharpe']:.2f}")
    print(f"Win Rate     : {metrics['wins']}/{metrics['trades']} "
          f"= {metrics['win_rate']*100:.2f}%")
    print(f"Trades       : {metrics['trades']}")

    if trades:
        print("\nFirst Trade Sample:")
        print(trades[0])









































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
