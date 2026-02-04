
# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, set_key
import subprocess
import sys

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
# Import constants
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
try:
    from constants import NIFTY_50_SYMBOLS, FNO_STOCKS
except ImportError:
    # Fallback if running from root or implicit path issues
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    try:
        from constants import NIFTY_50_SYMBOLS, FNO_STOCKS
    except ImportError:
        # Last resort fallback if src/constants.py is meant to be the source
        pass


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
universe = st.sidebar.radio("Universe", ["NSE 500", "Nifty 50", "F&O"], index=0)

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
        elif universe == "F&O":
            base_symbol = symbol.split(".")[0]
            if base_symbol not in FNO_STOCKS:
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

# Compute crossovers (ALWAYS calc for sorting)
with st.spinner("Calculating crossovers..."):
    crossover_map = calculate_crossovers(symbols_to_process, timeframe)

# Add crossover data to rows
for row in rows:
    sym = row["RawSymbol"]
    if sym in crossover_map:
        row["Crossover Date"] = str(crossover_map[sym]["Recent Bullish Crossover"].date())
        row["Recent Bullish Crossover"] = crossover_map[sym]["Recent Bullish Crossover"]
    else:
        row["Crossover Date"] = "-"
        row["Recent Bullish Crossover"] = pd.Timestamp.min

summary_df = pd.DataFrame(rows)

if not summary_df.empty:
    # Always sort by Recent Bullish Crossover
    summary_df = summary_df.sort_values(by="Recent Bullish Crossover", ascending=False)
    
    # Drop raw cols used for sorting but keep Crossover Date for display
    summary_df = summary_df.drop(columns=["RawSymbol", "RawWinRate", "Recent Bullish Crossover"])
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
    # ---------- PLOT (PLOTLY) ----------
    import plotly.graph_objects as go

    fig = go.Figure()

    # 1. Price Line
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["Close"], 
        mode='lines', 
        name='Close', 
        line=dict(color='gray', width=1),
        opacity=0.6
    ))

    # 2. MA Lines
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["MA_Fast"], 
        mode='lines', 
        name=f'{scenario_ma_type} {fast_ma}', 
        line=dict(color='green', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["MA_Slow"], 
        mode='lines', 
        name=f'{scenario_ma_type} {slow_ma}', 
        line=dict(color='orange', width=1.5)
    ))

    # 3. Buy/Sell Arrows (Shifted Left by 1 candle as requested)
    # Ensure straightforward integer indexing
    df = df.reset_index(drop=True)
    
    # Get indices of crossovers
    buy_indices = df.index[df["Crossover"] == 2].to_numpy()
    sell_indices = df.index[df["Crossover"] == -2].to_numpy()

    # Shift left (i - 1), ensuring we don't go below 0
    buy_indices_shifted = buy_indices - 1
    buy_indices_shifted = buy_indices_shifted[buy_indices_shifted >= 0]
    
    sell_indices_shifted = sell_indices - 1
    sell_indices_shifted = sell_indices_shifted[sell_indices_shifted >= 0]

    buys = df.iloc[buy_indices_shifted].copy()
    sells = df.iloc[sell_indices_shifted].copy()

    # Calculate Offsets for Visibility (User Request: Buy Above, Sell Below)
    # Using a 2% offset from Close to ensure clearance
    buys["Arrow_Y"] = buys["High"] * 1.02
    sells["Arrow_Y"] = sells["Low"] * 0.98

    # Buy Arrows (Above, pointing up)
    fig.add_trace(go.Scatter(
        x=buys["Date"], 
        y=buys["Arrow_Y"],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=1.5, color='black'))
    ))

    # Sell Arrows (Below, pointing down)
    fig.add_trace(go.Scatter(
        x=sells["Date"], 
        y=sells["Arrow_Y"],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=1.5, color='black'))
    ))
    
    # Connector Lines (Dotted) - From Arrow Y to MA
    shapes = []
    
    def add_connector(row, arrow_y, color):
        # Determine which MA to connect to (usually the cross point, roughly mean of fast/slow or just fast)
        target_y = row["MA_Fast"] 
        return dict(
            type="line",
            x0=row["Date"], y0=arrow_y,
            x1=row["Date"], y1=target_y, 
            line=dict(color=color, width=1, dash="dot")
        )

    for _, row in buys.iterrows():
        shapes.append(add_connector(row, row["Arrow_Y"], "lime"))
    
    for _, row in sells.iterrows():
        shapes.append(add_connector(row, row["Arrow_Y"], "red"))

    # 4. Latest Price Annotation
    latest_row = df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[latest_row["Date"]],
        y=[latest_row["Close"]],
        mode='markers',
        name='Latest',
        marker=dict(color='black', size=10),
        showlegend=False
    ))

    fig.add_annotation(
        x=latest_row["Date"],
        y=latest_row["Close"],
        text=f"<b>{latest_row['Close']:.2f}</b>", # Bold text
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30,
        font=dict(color="black", size=12),
        bgcolor="#ffffff",      # White background
        bordercolor="#333333",  # Dark border
        borderwidth=1,
        opacity=0.9
    )

    # 5. Layout Config
    current_signal = df.iloc[-1]["Signal"]
    regime = "Bullish 🟢" if current_signal == 1 else "Bearish 🔴"

    fig.update_layout(
        title=dict(
            text=f"{selected_symbol} — {scenario_ma_type} ({fast_ma}/{slow_ma}) — {timeframe}<br>Regime: {regime}",
            y=0.95
        ),
        xaxis_title="Date",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            dtick="M1" if timeframe == "Daily" else "M3",
            showgrid=True, # Restored grid
            gridcolor="#f0f0f0" # Subtle gray
        ),
        yaxis_title="Price",
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"), # Restored grid
        shapes=shapes, # Restored shapes

        template="plotly_white",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )

    # Range Slider for zooming (Common in financial charts)
    fig.update_xaxes(rangeslider_visible=False)

    # ✅ Render
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬆ Select a stock from the table to run a scenario analysis")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 Adaptive Finance | Strategy Discovery + Scenario Analysis Dashboard")