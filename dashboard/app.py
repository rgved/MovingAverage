
# dashboard/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv, set_key
import subprocess
import sys

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
# Import constants
try:
    from constants import NIFTY_50_SYMBOLS, FNO_STOCKS
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from constants import NIFTY_50_SYMBOLS, FNO_STOCKS


# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reports_dir = os.path.join(BASE_DIR, "reports")
data_dir = os.path.join(BASE_DIR, "data", "trimmed")
src_dir = os.path.join(BASE_DIR, "src")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Adaptive MA Strategy Dashboard", layout="wide")
st.title("Adaptive Moving Average Strategy Dashboard")

if not os.path.exists(reports_dir):
    st.error(f"Reports directory not found: {reports_dir}")
    st.stop()

# ---------- SIDEBAR CONFIGURATION ----------
st.sidebar.title("Configuration")
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

current_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# Hide existing token, only allow updates
new_token = st.sidebar.text_input(
    "Upstox Access Token", 
    value="", 
    type="password", 
    placeholder="Enter new token to update",
    help="Enter a new token only if you want to update the existing one."
)

if new_token:
    set_key(env_path, "UPSTOX_ACCESS_TOKEN", new_token)
    # Reload environment variable to ensure consistency
    os.environ["UPSTOX_ACCESS_TOKEN"] = new_token
    st.sidebar.success("Token updated!")

st.sidebar.markdown("---")
st.sidebar.header("Data Management")

def run_script(script_name, status_text):
    script_path = os.path.join(src_dir, script_name)
    status_text.text(f"Running {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        # print(result.stdout) # Optional: Log output
    except subprocess.CalledProcessError as e:
        st.error(f"Error running {script_name}:\n{e.stderr}")
        raise e

if st.sidebar.button("Update Data & Run Optimization"):
    # Check if token exists
    if not current_token:
        st.sidebar.error("⚠ Access Token Missing! Please update the token above first.")
    else:
        status_placeholder = st.sidebar.empty()
        try:
            with st.spinner("Running data pipeline... This may take a while."):
                # 1. Fetch Data
                run_script("fetch-data-upstox.py", status_placeholder)
                
                # 2. Features
                run_script("features.py", status_placeholder)
                
                # 3. Trim Data
                run_script("trim_data.py", status_placeholder)
                
                # 4. Optimize
                run_script("optimize_on_dynamic_noise.py", status_placeholder)
                
            status_placeholder.success("Pipeline completed successfully! ✅")
            st.rerun() # Refresh app to show new data
            
        except Exception as e:
            st.sidebar.error("Pipeline failed!")

st.sidebar.markdown("---")
st.sidebar.header("Filters & Sorting")

# Timeframe Selection
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly"], index=0)

# Universe Selection
universe = st.sidebar.radio("Universe", ["NSE 500", "Nifty 50"], index=0)

# Sorting Selection
sort_by = st.sidebar.selectbox(
    "Sort By", 
    ["Return (%)", "Win Rate (%)", "Recent Bullish Crossover", "Recent Bearish Crossover"],
    index=0
)

# Min Trades Filter
min_trades = st.sidebar.slider("Minimum Trades", 0, 50, 5)

# ---------- CACHED CROSSOVER CALCULATION ----------
@st.cache_data
def calculate_crossovers(stock_list, tf):
    crossover_data = {}
    
    for symbol in stock_list:
        price_file = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(price_file):
            continue
            
        try:
            df = pd.read_csv(price_file)
            df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
            df = df.sort_values("Date")
            
            # Resample if Weekly
            if tf == "Weekly":
                df.set_index("Date", inplace=True)
                df = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                df.dropna(inplace=True)
                df.reset_index(inplace=True)

            # Optimization results (to get MA params)
            report_file = os.path.join(reports_dir, f"{symbol.replace('.', '_')}_dynamic_trend_noise_optimization.csv")
            if not os.path.exists(report_file):
                continue
                
            rep = pd.read_csv(report_file)
            best = rep.iloc[0]
            ma_type = best["MA_Type"]
            fast, slow = map(int, best["MA_Pair"].split("/"))

            # Calculate MAs
            if ma_type == "EMA":
                df["MA_Fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
                df["MA_Slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
            else:
                df["MA_Fast"] = df["Close"].rolling(fast).mean()
                df["MA_Slow"] = df["Close"].rolling(slow).mean()

            df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
            df["Crossover"] = df["Signal"].diff() # 2=Bullish, -2=Bearish

            # Find last crossover
            last_bullish_idx = df[df["Crossover"] == 2].index.max()
            last_bearish_idx = df[df["Crossover"] == -2].index.max()
            
            last_bullish_date = df.loc[last_bullish_idx, "Date"] if pd.notna(last_bullish_idx) else pd.Timestamp.min
            last_bearish_date = df.loc[last_bearish_idx, "Date"] if pd.notna(last_bearish_idx) else pd.Timestamp.min
            
            crossover_data[symbol] = {
                "Recent Bullish Crossover": last_bullish_date,
                "Recent Bearish Crossover": last_bearish_date
            }
            
        except Exception:
            continue
            
    return crossover_data


# ---------- BUILD SUMMARY TABLE (BEST STRATEGY PER STOCK) ----------
rows = []
symbols_to_process = []

# First pass: Collect data and filter by universe/trades
for file in os.listdir(reports_dir):
    if file.endswith("_dynamic_trend_noise_optimization.csv"):
        symbol = file.replace("_dynamic_trend_noise_optimization.csv", "").replace("_", ".")
        
        # Filter by Universe
        if universe == "Nifty 50":
            # Using simple check if symbol starts with Nifty 50 name (basic matching)
            # A more robust way is to check exact symbol match
            base_symbol = symbol.split(".")[0]
            if base_symbol not in NIFTY_50_SYMBOLS:
                continue

        rep = pd.read_csv(os.path.join(reports_dir, file))
        best = rep.iloc[0]
        
        trades = int(best["Trades"])
        
        # Filter by Min Trades
        if trades < min_trades:
            continue

        symbols_to_process.append(symbol)
        
        # Win Rate formatting
        win_rate = best["WinRate"]
        if win_rate > 1:
            win_rate /= 100
        wins = int(round(win_rate * trades))
        win_rate_str = f"{round(win_rate * 100, 1)}% ({wins}/{trades})"

        # F&O Indicator
        base_symbol_clean = symbol.split(".")[0]
        display_symbol = f"{symbol} *" if base_symbol_clean in FNO_STOCKS else symbol

        rows.append({
            "Symbol": display_symbol,
            "RawSymbol": symbol,
            "Best MA Type": best["MA_Type"],
            "Best MA Pair": best["MA_Pair"],
            "Return (%)": round(best["Return"], 2),
            "Win Rate (%)": win_rate_str,
            "RawWinRate": win_rate, # For sorting
            "Sharpe": round(best["Sharpe"], 2),
            "Trades": trades
        })

# Compute crossovers if needed for sorting
crossover_map = {}
if "Crossover" in sort_by:
    with st.spinner("Calculating crossovers..."):
        crossover_map = calculate_crossovers(symbols_to_process, timeframe)

# Add crossover data to rows
for row in rows:
    sym = row["RawSymbol"]
    if sym in crossover_map:
        row["Recent Bullish Crossover"] = crossover_map[sym]["Recent Bullish Crossover"]
        row["Recent Bearish Crossover"] = crossover_map[sym]["Recent Bearish Crossover"]
    else:
        # Default for sorting if calculation failed or didn't run
        row["Recent Bullish Crossover"] = pd.Timestamp.min
        row["Recent Bearish Crossover"] = pd.Timestamp.min

summary_df = pd.DataFrame(rows)

if not summary_df.empty:
    if sort_by == "Return (%)":
        summary_df = summary_df.sort_values(by="Return (%)", ascending=False)
    elif sort_by == "Win Rate (%)":
        summary_df = summary_df.sort_values(by="RawWinRate", ascending=False)
    elif sort_by == "Recent Bullish Crossover":
        summary_df = summary_df.sort_values(by="Recent Bullish Crossover", ascending=False)
    elif sort_by == "Recent Bearish Crossover":
        summary_df = summary_df.sort_values(by="Recent Bearish Crossover", ascending=False)
    
    # Drop raw cols used for sorting
    summary_df = summary_df.drop(columns=["RawSymbol", "RawWinRate", "Recent Bullish Crossover", "Recent Bearish Crossover"])
    summary_df = summary_df.reset_index(drop=True)


# ---------- SCENARIO CONTROLS (WHAT-IF MODE) ----------
st.markdown("### 🔎 Scenario Analysis (What-If MA Strategy)")

col1, col2, col3 = st.columns(3)

with col1:
    scenario_ma_type = st.selectbox("MA Type", ["EMA", "SMA"])

with col2:
    fast_ma = st.selectbox("Fast MA (MA1)", [5, 10, 12, 20, 50, 100])

with col3:
    slow_ma = st.selectbox("Slow MA (MA2)", [20, 50, 100, 200])

# ---------- VALIDATION ----------
if fast_ma >= slow_ma:
    st.warning("Fast MA must be smaller than Slow MA")
    st.stop()

st.caption(
    f"📌 Showing *what-if scenario* for **{scenario_ma_type} {fast_ma}/{slow_ma}** "
    f"on **{timeframe}** data (independent of historical optimization)"
)

# ---------- TABLE ----------
st.subheader("📊 Stock Performance Summary (Best Historical Strategy)")

gb = GridOptionsBuilder.from_dataframe(summary_df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_grid_options(domLayout="normal")

grid_response = AgGrid(
    summary_df,
    gridOptions=gb.build(),
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=350,
    theme="streamlit",
)

# ---------- HANDLE ROW SELECTION ----------
selected_rows = grid_response.get("selected_rows", None)

has_selection = (
    selected_rows is not None
    and isinstance(selected_rows, pd.DataFrame)
    and not selected_rows.empty
)

if has_selection:
    display_symbol = selected_rows.iloc[0]["Symbol"]
    # Remove asterisk if present
    selected_symbol = display_symbol.replace(" *", "")

    st.markdown("---")
    st.subheader(f"📈 Price Chart ({timeframe}) + Scenario MA Overlay")

    price_file = os.path.join(data_dir, f"{selected_symbol}.csv")

    if not os.path.exists(price_file):
        st.error("Price data not found for this stock.")
        st.stop()

    # ---------- LOAD PRICE DATA ----------
    df = pd.read_csv(price_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values("Date")
    
    # ---------- TIME FRAME RESAMPLING ----------
    if timeframe == "Weekly":
        df.set_index("Date", inplace=True)
        # Resample logic
        df = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

    # ---------- APPLY SCENARIO MOVING AVERAGES ----------
    if scenario_ma_type == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast_ma, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow_ma, adjust=False).mean()
    else:
        df["MA_Fast"] = df["Close"].rolling(fast_ma).mean()
        df["MA_Slow"] = df["Close"].rolling(slow_ma).mean()

    df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
    df["Crossover"] = df["Signal"].diff()

    # ---------- PLOT ----------
# ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["Date"], df["Close"], label="Close", color="gray", alpha=0.6)
    ax.plot(df["Date"], df["MA_Fast"], label=f"{scenario_ma_type} {fast_ma}", color="green")
    ax.plot(df["Date"], df["MA_Slow"], label=f"{scenario_ma_type} {slow_ma}", color="orange")

    buys = df[df["Crossover"] == 2]
    sells = df[df["Crossover"] == -2]

    ax.scatter(buys["Date"], buys["Close"], marker="^", color="lime", s=80, label="Buy")
    ax.scatter(sells["Date"], sells["Close"], marker="v", color="red", s=80, label="Sell")

    # ✅ FORCE X-AXIS TO SHOW LATEST DATE
    import matplotlib.dates as mdates
    latest_date = df["Date"].max()
    
    # Add padding to right side so last label fits (reduced from 5 to 1 day)
    # Adjust padding for Weekly vs Daily
    padding_days = 7 if timeframe == "Weekly" else 1
    padding = pd.Timedelta(days=padding_days) 
    ax.set_xlim(df["Date"].min(), latest_date + padding)

    # Get default ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ticks = list(ax.get_xticks())
    
    # Convert latest_date to matplotlib date number
    latest_num = mdates.date2num(latest_date)
    
    # Filter out ticks too close to the latest date
    threshold = 10.0 if timeframe == "Weekly" else 2.0
    ticks = [t for t in ticks if t <= latest_num and abs(t - latest_num) > threshold]
    
    # Add latest date tick
    ticks.append(latest_num)
    ticks.sort()

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30, ha='right')


    # ✅ Highlight latest price (MUST be before st.pyplot)
    latest_row = df.iloc[-1]
    ax.scatter(
        latest_row["Date"],
        latest_row["Close"],
        color="black",
        s=90,
        zorder=5,
        label="Latest"
    )
    ax.annotate(
    latest_date.strftime("%Y-%m-%d"),
    xy=(latest_date, latest_row["Close"]),
    xytext=(10, -15),
    textcoords="offset points",
    fontsize=9,
    color="black",
    arrowprops=dict(arrowstyle="->", alpha=0.4)
    )


    # ✅ Regime label
    current_signal = df.iloc[-1]["Signal"]
    regime = "Bullish 🟢" if current_signal == 1 else "Bearish 🔴"

    ax.text(
        0.01, 0.95,
        f"Current Regime: {regime}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    ax.set_title(
        f"{selected_symbol} — Scenario: {scenario_ma_type} ({fast_ma}/{slow_ma}) — {timeframe}"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # ✅ Render LAST
    st.pyplot(fig)

else:
    st.info("⬆ Select a stock from the table to run a scenario analysis")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 Adaptive Finance | Strategy Discovery + Scenario Analysis Dashboard")