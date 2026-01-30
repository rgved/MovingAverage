# dashboard/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv, set_key

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ---------- PATH SETUP ----------
import subprocess
import sys

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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
        st.sidebar.error("⚠️ Access Token Missing! Please update the token above first.")
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



# ---------- BUILD SUMMARY TABLE (BEST STRATEGY PER STOCK) ----------
rows = []

for file in os.listdir(reports_dir):
    if file.endswith("_dynamic_trend_noise_optimization.csv"):
        symbol = file.replace("_dynamic_trend_noise_optimization.csv", "").replace("_", ".")
        rep = pd.read_csv(os.path.join(reports_dir, file))
        best = rep.iloc[0]

        rows.append({
            "Symbol": symbol,
            "Best MA Type": best["MA_Type"],
            "Best MA Pair": best["MA_Pair"],
            "Return (%)": round(best["Return"], 2),
            "Win Rate (%)": round(best["WinRate"], 1),
            "Sharpe": round(best["Sharpe"], 2),
            "Trades": int(best["Trades"])
        })

summary_df = (
    pd.DataFrame(rows)
    .sort_values(by="Return (%)", ascending=False)
    .reset_index(drop=True)
)

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
    "(independent of historical optimization)"
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
    selected_symbol = selected_rows.iloc[0]["Symbol"]

    st.markdown("---")
    st.subheader("📈 Price Chart + Scenario MA Overlay")

    price_file = os.path.join(data_dir, f"{selected_symbol}.csv")

    if not os.path.exists(price_file):
        st.error("Price data not found for this stock.")
        st.stop()

    # ---------- LOAD PRICE DATA ----------
    df = pd.read_csv(price_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values("Date")

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
    padding = pd.Timedelta(days=1) 
    ax.set_xlim(df["Date"].min(), latest_date + padding)

    # Get default ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ticks = list(ax.get_xticks())
    
    # Convert latest_date to matplotlib date number
    latest_num = mdates.date2num(latest_date)
    
    # Filter out ticks too close to the latest date (within 2 days) to prevent overlap
    # AND filter out any ticks that are AFTER the latest date (future dates)
    ticks = [t for t in ticks if t <= latest_num and abs(t - latest_num) > 2.0]
    
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
        f"{selected_symbol} — Scenario: {scenario_ma_type} ({fast_ma}/{slow_ma})"
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
 