# Adaptive Moving Average Strategy – Finance Hackathon

## Our Approach

We developed an **Adaptive Moving Average Strategy** that dynamically decides whether to use a **Simple Moving Average (SMA)** or an **Exponential Moving Average (EMA)** based on market behavior parameters.

The model uses three key indicators:

- **Volatility** – measures how much the stock price fluctuates.  
- **Trend Strength** – measures how strongly the market is moving in one direction.  
- **Noise** – measures randomness or choppiness in price movements.

### Decision Logic

- **EMA** is used in volatile and trending markets where a faster response is beneficial.  
- **SMA** is used in calmer or sideways markets to smooth short-term fluctuations.

This makes the algorithm adaptive and capable of handling different market regimes.

---

## Libraries Used

| Library | Purpose |
|----------|----------|
| pandas | Data handling and manipulation |
| numpy | Numerical operations |
| matplotlib | Data visualization |
| streamlit | Building an interactive dashboard |
| yfinance | Fetching updated stock data |
| fastapi, uvicorn | Optional: for API endpoints |

---

## File Overview

### 1. `src/backtest.py`
This file implements a basic **Moving Average Crossover Backtest**.

**Core logic:**
- Entry: When the fast MA crosses above the slow MA (Bullish crossover → Buy next day).
- Exit: When the fast MA crosses below the slow MA (Bearish crossover) or when one of the following triggers occur:
  - Stop Loss: 3% below entry price
  - Take Profit: 5% above entry price
  - Maximum Holding Period: 7 trading days

**Outputs:**
- Total Return (%)
- Maximum Drawdown (%)
- Sharpe Ratio
- Win Rate (%)
- Number of Trades

---

### 2. `src/optimize_dynamic_trend_noise.py`
This file builds on `backtest.py` and **optimizes the logic** by repeatedly calling the backtest for different moving average configurations.

It calculates:
- **Volatility** using the standard deviation of returns.
- **Trend Strength** using directional percentage movement.
- **Noise Ratio** using the difference between cumulative and total absolute moves.

Based on these, it dynamically selects whether to use **SMA** or **EMA** and identifies the **best MA window pair** (e.g., 10/20, 12/26, 20/50, etc.) for each stock.

All results and performance metrics are stored in the `/reports/` folder.

---

### 3. `dashboard/app.py`
An interactive **Streamlit dashboard** that displays:
- The selected stock’s price, moving averages, and crossover signals.
- Key metrics (Return, Win Rate, Sharpe Ratio, Max Drawdown, Trades).
- Market condition parameters (Volatility, Trend Strength, Noise).
- The final decision of whether **SMA or EMA** was chosen and why.

The dashboard uses only **three months of data (August 1, 2025 – November 7, 2025)** as required by the hackathon.

---

## Data Source

All stock data was fetched using the **Yahoo Finance (`yfinance`)** library, which provides reliable and frequently updated price data.

**Data folders:**
- `data/raw/` – Unprocessed data from Yahoo Finance  
- `data/processed/` – Data after moving averages and signal computation  
- `data/trimmed/` – Final filtered data (Aug–Nov 2025) used for backtesting and dashboard  

---

## Stocks Analyzed

| Symbol | Company |
|---------|----------|
| RELIANCE.NS | Reliance Industries |
| TCS.NS | Tata Consultancy Services |
| INFY.NS | Infosys Ltd |
| HDFCBANK.NS | HDFC Bank |
| ICICIBANK.NS | ICICI Bank |
| ADANIENT.NS | Adani Enterprises |
| ITC.NS | ITC Limited |
| MARUTI.NS | Maruti Suzuki |
| TATASTEEL.NS | Tata Steel |
| LT.NS | Larsen & Toubro |
| SBI.NS | State Bank of India |

Each stock’s optimized strategy and performance metrics are saved in `/reports/`.

---

## Output Summary

Each report (e.g., `reports/RELIANCE_NS_dynamic_trend_noise_optimization.csv`) includes:

| Symbol | Volatility | TrendStrength | Noise | MA_Type | MA_Pair | Return | WinRate | Sharpe | MaxDD | Trades |
|--------|-------------|----------------|--------|----------|---------|---------|----------|---------|---------|---------|
| RELIANCE.NS | 1.22 | 6.97 | 60.10 | EMA | 10/20 | -1.93 | 33.3 | 0.97 | -4.98 | 3 |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
