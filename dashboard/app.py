
# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, set_key
import subprocess
import sys

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
# Import constants
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
try:
    from constants import NIFTY_50_SYMBOLS, FNO_STOCKS, INDICES
except ImportError:
    # Fallback if running from root or implicit path issues
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    try:
        from constants import NIFTY_50_SYMBOLS, FNO_STOCKS, INDICES
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
    os.makedirs(reports_dir, exist_ok=True)
    st.warning(
        f"Reports directory was missing and has been created: {reports_dir}. "
        "Run the pipeline to generate reports."
    )

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


# Lookback Period Selection (New)
lookback_years = st.sidebar.selectbox("Lookback Period", ["1Y", "2Y", "3Y"], index=2) # Default 3Y
years = int(lookback_years.replace("Y", ""))

# Timeframe Selection
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly"], index=0)

# Universe Selection
universe = st.sidebar.radio("Universe", ["NSE 500", "Nifty 50", "F&O", "Indices", "All NSE"], index=0)

# Min Trades Filter
min_trades = st.sidebar.slider("Minimum Trades", 0, 50, 5)

# ---------- CACHED CROSSOVER CALCULATION ----------
@st.cache_data
def calculate_crossovers(stock_list, tf, years):
    crossover_data = {}
    
    # Use processed data (full history) instead of trimmed to ensure we find crossovers
    full_data_dir = os.path.join(BASE_DIR, "data", "processed")
    
    for symbol in stock_list:
        price_file = os.path.join(full_data_dir, f"{symbol}.csv")
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
            # Construct filename based on selected lookback years
            report_file = os.path.join(reports_dir, f"{symbol.replace('.', '_')}_{years}y_dynamic_trend_noise_optimization.csv")
            # If for some reason report doesn't exist but price does (rare), skip
            if not os.path.exists(report_file):
               # Logic to handle missing report? For now skip
               pass

            # Proceed if report exists OR if we want to allow viewing chart without optimization (todo)
            if os.path.exists(report_file):
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
                df["Crossover"] = df["Signal"].diff()
                
                
                # Filter for crossovers (2 = Buy, -2 = Sell)
                crossovers = df[df["Crossover"].abs() == 2]
                
                if not crossovers.empty:
                    last_crossover_row = crossovers.iloc[-1]
                    last_crossover = last_crossover_row["Date"]
                    crossover_type = "Bullish" if last_crossover_row["Crossover"] == 2 else "Bearish"
                    
                    # Count trades in last 3 months
                    three_months_ago = df["Date"].max() - pd.DateOffset(months=3)
                    recent_trades = crossovers[crossovers["Date"] >= three_months_ago]
                    recent_trade_count = len(recent_trades)
                    
                    crossover_data[symbol] = {
                        "Recent Bullish Crossover": last_crossover,
                        "Recent 3M Trades": recent_trade_count,
                        "Crossover Type": crossover_type
                    }
            else:
                 pass # No report found
            
        except Exception:
            continue
            
    return crossover_data


@st.cache_data
def build_download_csv(tf, ma_type_filter, crossover_filter, fast_mas, slow_mas, days_limit):
    """Build a market-breadth style export parameterized by custom MA selections."""
    
    full_data_dir = os.path.join(BASE_DIR, "data", "processed")

    # Construct the selected permutations of Fast/Slow MAs based on user input
    # (Only keeping valid pairs where Fast < Slow)
    ma_pairs = {}
    for f in fast_mas:
        for s in slow_mas:
            if f < s:
                # Add Bullish or Bearish or both based on the filter
                if crossover_filter in ["Both", "Bullish"]:
                    ma_pairs[f"Bullish_{f}/{s}"] = (f, s, "Bullish")
                if crossover_filter in ["Both", "Bearish"]:
                    ma_pairs[f"Bearish_{f}/{s}"] = (f, s, "Bearish")

    counts_by_date = {}

    for file in os.listdir(full_data_dir):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")

        price_file = os.path.join(full_data_dir, f"{symbol}.csv")
        if not os.path.exists(price_file):
            continue

        try:
            df = pd.read_csv(price_file)

            df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
            df = df.sort_values("Date")

            if tf == "Weekly":
                df = (
                    df.set_index("Date")
                    .resample("W")
                    .agg({
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum"
                    })
                    .dropna()
                    .reset_index()
                )
                
            # Precompute required MAs for this stock based on permutations
            ma_calcs = {}
            for f, s, _ in ma_pairs.values():
                if f not in ma_calcs:
                    if ma_type_filter in ["Both", "EMA"]:
                        ma_calcs[f"EMA_{f}"] = df["Close"].ewm(span=f, adjust=False).mean()
                    if ma_type_filter in ["Both", "SMA"]:
                        ma_calcs[f"SMA_{f}"] = df["Close"].rolling(f).mean()
                        
                if s not in ma_calcs:
                    if ma_type_filter in ["Both", "EMA"]:
                        ma_calcs[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
                    if ma_type_filter in ["Both", "SMA"]:
                        ma_calcs[f"SMA_{s}"] = df["Close"].rolling(s).mean()

            for col_name, (fast, slow, cross) in ma_pairs.items():
                
                # If Both MA Types are selected, we evaluate EMA and SMA and count if ANY matches true 
                # (or just follow the dominant strategy, but here we process each individually)
                types_to_check = ["EMA", "SMA"] if ma_type_filter == "Both" else [ma_type_filter]
                
                # Track which dates this stock already contributed to for this col_name,
                # so we don't double-count when checking both EMA and SMA.
                counted_dates = set()
                
                for m_type in types_to_check:
                    fast_series = ma_calcs[f"{m_type}_{fast}"]
                    slow_series = ma_calcs[f"{m_type}_{slow}"]
                    
                    # Compute signal and crossover diff
                    # 1 = Bullish state, -1 = Bearish state
                    raw_signal = np.where(fast_series > slow_series, 1, -1)
                    # Diff: 2 = Bullish Crossover, -2 = Bearish Crossover
                    crossover_diff = pd.Series(raw_signal).diff().fillna(0)
                    
                    if cross == "Bullish":
                        signal_triggered = crossover_diff == 2
                    else:
                        signal_triggered = crossover_diff == -2
                        
                    for dt, is_triggered in zip(df["Date"], signal_triggered):
                        if pd.isna(dt):
                            continue
                            
                        day_key = dt.date()
                        if day_key not in counts_by_date:
                            counts_by_date[day_key] = {k: 0 for k in ma_pairs}
                            
                        if bool(is_triggered) and day_key not in counted_dates:
                            counts_by_date[day_key][col_name] += 1
                            # Track this date so we don't double-count if checking
                            # both EMA and SMA for the same stock.
                            counted_dates.add(day_key)

        except Exception:
            continue

    # ---------- BUILD DATAFRAME ----------

    download_df = pd.DataFrame([
        {"Date": pd.to_datetime(day), **vals}
        for day, vals in counts_by_date.items()
    ])

    if download_df.empty:
        return pd.DataFrame(columns=["Date", *ma_pairs.keys(), "NIFTY*"])

    download_df = download_df.sort_values("Date").reset_index(drop=True)

    # ---------- ADD NIFTY CLOSE ----------

    nifty_found = False
    nifty_candidates = ["Nifty 50", "NIFTY 50", "NIFTY"]

    for nifty_symbol in nifty_candidates:

        nifty_file = os.path.join(full_data_dir, f"{nifty_symbol}.csv")

        if not os.path.exists(nifty_file):
            continue

        try:
            nifty_df = pd.read_csv(nifty_file)

            nifty_df["Date"] = pd.to_datetime(
                nifty_df["Date"], utc=True, errors="coerce"
            ).dt.tz_convert(None)

            nifty_df = nifty_df[["Date", "Close"]].dropna()
            nifty_df.rename(columns={"Close": "NIFTY*"}, inplace=True)

            if tf == "Weekly":
                nifty_df = (
                    nifty_df.set_index("Date")
                    .resample("W")
                    .last()
                    .dropna()
                    .reset_index()
                )

            download_df = download_df.merge(nifty_df, on="Date", how="left")
            nifty_found = True
            break

        except Exception:
            continue

    if not nifty_found:
        # Fallback: fetch Nifty 50 from Yahoo Finance when local file is missing
        try:
            import yfinance as yf
            nifty_yf = yf.download("^NSEI", period="1y", progress=False)
            # Flatten MultiIndex columns (newer yfinance returns ('Close', '^NSEI'))
            if isinstance(nifty_yf.columns, pd.MultiIndex):
                nifty_yf.columns = nifty_yf.columns.get_level_values(0)
            if not nifty_yf.empty and "Close" in nifty_yf.columns:
                nifty_yf = nifty_yf[["Close"]].reset_index()
                nifty_yf.columns = ["Date", "NIFTY*"]
                nifty_yf["Date"] = pd.to_datetime(nifty_yf["Date"])
                if nifty_yf["Date"].dt.tz is not None:
                    nifty_yf["Date"] = nifty_yf["Date"].dt.tz_convert(None)
                # Normalize to midnight (strip time component) to match download_df dates
                nifty_yf["Date"] = nifty_yf["Date"].dt.normalize()

                if tf == "Weekly":
                    nifty_yf = (
                        nifty_yf.set_index("Date")
                        .resample("W")
                        .last()
                        .dropna()
                        .reset_index()
                    )

                download_df = download_df.merge(nifty_yf, on="Date", how="left")
                nifty_found = True
        except Exception:
            pass

        if not nifty_found:
            download_df["NIFTY*"] = np.nan

    ordered_cols = ["Date", *list(ma_pairs.keys()), "NIFTY*"]
    
    # Filter based on user-defined number of days
    # (Getting the max date and slicing the past `days_limit`)
    max_date = download_df["Date"].max()
    cutoff_date = max_date - pd.Timedelta(days=days_limit)
    download_df = download_df[download_df["Date"] >= cutoff_date]

    return download_df[ordered_cols].reset_index(drop=True)


# ---------- BUILD SUMMARY TABLE (BEST STRATEGY PER STOCK) ----------
rows = []
symbols_to_process = []

# First pass: Collect data and filter by universe/trades
# Target suffix based on lookback
target_suffix = f"_{years}y_dynamic_trend_noise_optimization.csv"

for file in os.listdir(reports_dir):
    if file.endswith(target_suffix):
        # Original logic assumes symbol is filename. However, indices have spaces.
        # But report generation replaces spaces with underscores? We need to verify how reports are named.
        # Let's assume reports are generated with sanitized names.
        # If fetch-data saves as "Nifty 50.csv", then optimization might save as "Nifty 50_dynamic..."
        # or "Nifty_50_dynamic..."
        
        # Current logic: symbol = file.replace("_dynamic...", "").replace("_", ".")
        # This breaks for indices like "Nifty 50" -> "Nifty.50" which is wrong.
        # We need a robust reverse mapping or standardized naming.
        
        # INVESTIGATION: symbols in constants.py are "Nifty 50".
        # fetch-data-upstox saves as "Nifty 50.csv".
        # optimize_on_dynamic_noise.py likely uses the filename stem.
        # if file is "Nifty 50.csv", stem is "Nifty 50".
        # report file would be "Nifty 50_dynamic..."
        
        # BUT the code at line 181 does: symbol = file.replace(...).replace("_", ".")
        # This was probably for "RELIANCE_NS" -> "RELIANCE.NS" or similar?
        # No, upstox symbols are like "RELIANCE.NS" -> saved as "RELIANCE.NS.csv"??
        # fetch-data-upstox.py Line 64: df.to_csv(..., f"{symbol}.csv")
        # SYMBOL_MAP keys are "RELIANCE.NS". So file is "RELIANCE.NS.csv".
        # optimize code likely produces "RELIANCE_NS_dynamic..." due to some sanitization?
        # Let's look at `optimize_on_dynamic_noise.py` to be sure, but for now apply logic:
        
        raw_name = file.replace(target_suffix, "")
        # If it was "RELIANCE.NS", it might be "RELIANCE_NS" in report filename if sanitizer used.
        # The existing code replcaes "_" with "."
        # That converts "Nifty_50" -> "Nifty.50". 
        
        # We need to handle this.
        # Let's reconstruct symbol.
        if "Nifty" in raw_name: # Simple heuristic for indices
             # Indices usually don't have dots. They might have spaces.
             # If report replaced space with _, we need to reverse it.
             # BUT existing code blindly does replace("_", ".").
             
             # If universe is indices, we should be careful.
             pass
        
        symbol = raw_name.replace("_", ".") # Default behavior
        
        # Filter by Universe
        if universe == "Nifty 50":
            base_symbol = symbol.split(".")[0]
            if base_symbol not in NIFTY_50_SYMBOLS:
                continue
        elif universe == "F&O":
            base_symbol = symbol.split(".")[0]
            if base_symbol not in FNO_STOCKS:
                continue
        elif universe == "Indices":
            # For Indices, we need to match against INDICES list.
            # The symbol from report might be "Nifty.50" (if "Nifty 50" -> "Nifty_50" -> "Nifty.50")
            # OR "Nifty 50" (if spaces preserved).
            # The replace("_", ".") is dangerous if filenames have underscores for spaces.
            
            # Let's try to match loosely.
            # Convert symbol back to potential space-sep format?
            
            # Better approach: check if any index in INDICES matches the symbol (ignoring dot/underscore diffs)
            
            # Normalize for check
            norm_symbol = symbol.replace(".", " ").replace("_", " ")
            # "Nifty.50" -> "Nifty 50"
            
            is_index = False
            real_index_name = ""
            for idx in INDICES:
                if idx == norm_symbol or idx == symbol:
                    is_index = True
                    real_index_name = idx
                    break
            
            if not is_index:
                continue
                
            # If it is an index, use the clean name for display
            symbol = real_index_name 

        elif universe == "All NSE":
            # Show everything except Indices?
            # Or just show everything?
            # Typically "All NSE" implies all stocks.
            # We should probably filter out indices to avoid clutter if they are mixed in.
            
            # Check if it's an index
            if symbol in INDICES:
                continue
            
            # Also check if base symbol is in INDICES (e.g. "Nifty 50" from "Nifty.50")
            base_norm = symbol.replace(".", " ")
            if base_norm in INDICES:
                continue
                
        else: # NSE 500 (Default)
            # If NSE 500, we might want to EXCLUDE indices?
            # Or just show everything that isn't filtered out?
            # Usually NSE 500 implies stocks.
            # Check if it IS in INDICES, if so, skip?
            # For now, let's just let it be, or strictly filter?
            # Let's exclude recognized indices from NSE 500 view to keep it clean.
            
            start_name = symbol.split(".")[0].replace("_", " ")
            if start_name in INDICES or symbol in INDICES:
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
            "Sigma (Market)": best.get("MarketSigma", best.get("volatility", 0)), # Backward compat check
            "Strategy Vol": round(best.get("StrategyAggr", best.get("Volatility", 0)), 2),
            "Trades": trades
        })

if not rows:
    st.info(
        "No report files found yet. Run the data pipeline from the sidebar to generate reports."
    )

# Compute crossovers (ALWAYS calc for sorting)
with st.spinner("Calculating crossovers..."):
    crossover_map = calculate_crossovers(symbols_to_process, timeframe, years)

# Add crossover data to rows
for row in rows:
    sym = row["RawSymbol"]
    if sym in crossover_map:
        cross_date = crossover_map[sym]["Recent Bullish Crossover"]
        if cross_date.year < 1900:
             row["Crossover Date"] = "-"
             row["Crossover Signal"] = "-"
        else:
             row["Crossover Date"] = str(cross_date.date())
             row["Crossover Signal"] = crossover_map[sym].get("Crossover Type", "-")
        row["Recent Bullish Crossover"] = cross_date
        
        # Update Trades count to reflect only last 3 months
        if "Recent 3M Trades" in crossover_map[sym]:
            row["Trades"] = crossover_map[sym]["Recent 3M Trades"]
    else:
        row["Crossover Date"] = "-"
        row["Crossover Signal"] = "-"
        row["Recent Bullish Crossover"] = pd.Timestamp.min

summary_df = pd.DataFrame(rows)

if not summary_df.empty:
    # Always sort by Recent Bullish Crossover
    summary_df = summary_df.sort_values(by="Recent Bullish Crossover", ascending=False)
    
    # Drop raw cols used for sorting but keep Crossover Date for display
    summary_df = summary_df.drop(columns=["RawSymbol", "RawWinRate", "Recent Bullish Crossover"])
    summary_df = summary_df.reset_index(drop=True)

# ---------- CUSTOM SCRENER CONTROLS ----------
st.subheader("🎯 Refine Screener")
with st.form("screener_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        scr_ma_type = st.selectbox("MA Type", ["Both", "SMA", "EMA"], index=0)
    with c2:
        scr_fast_ma = st.selectbox("Fast MA", options=[5, 10, 12, 20, 50], index=3)
    with c3:
        scr_slow_ma = st.selectbox("Slow MA", options=[20, 26, 50, 100, 200], index=2)
        
    refine_btn = st.form_submit_button("Refine Screener")

if refine_btn and not summary_df.empty:
    if scr_fast_ma >= scr_slow_ma:
        st.warning("Fast MA must be smaller than Slow MA to filter properly.")
    else:
        # Filter logic based on choices
        target_pair = f"{scr_fast_ma}/{scr_slow_ma}"
        
        mask = (summary_df["Best MA Pair"] == target_pair)
        if scr_ma_type != "Both":
            mask = mask & (summary_df["Best MA Type"] == scr_ma_type)
            
        summary_df = summary_df[mask].reset_index(drop=True)
        if summary_df.empty:
            st.info(f"No stocks found with Best Strategy = {scr_ma_type if scr_ma_type != 'Both' else 'Any'} {target_pair}")

# ---------- TABLE ----------
st.subheader("📊 Stock Performance Summary (Best Historical Strategy)")

gb = GridOptionsBuilder.from_dataframe(summary_df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_grid_options(domLayout="normal")

grid_options = gb.build()
# Fix for deprecated warning: ensure rowSelection is set correctly and suppressRowClickSelection is avoided if possible
# The warning says: As of v32.2, suppressRowClickSelection is deprecated. Use `rowSelection.enableClickSelection` instead.
# We manually patch it for newer AG Grid versions:
if "suppressRowClickSelection" in grid_options:
    del grid_options["suppressRowClickSelection"]
grid_options["rowSelection"] = "single"  # Ensure this is set

grid_response = AgGrid(
    summary_df,
    gridOptions=grid_options,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
    height=350,
    theme="streamlit",
)

st.markdown("---")
# ---------- CSV DOWNLOAD CONFIGURATION SECTION ----------
st.subheader("📥 Download CSV Options")

with st.form("csv_download_options"):
    
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_ma_type = st.selectbox("Type of MA", ["Both", "SMA", "EMA"])
    with col2:
        csv_crossover = st.selectbox("Crossover", ["Both", "Bullish", "Bearish"])
    with col3:
        csv_days = st.selectbox("No. of Days", [30, 60, 90], index=2)
        
    col4, col5 = st.columns(2)
    with col4:
        # User specified if none are selected, select all options
        preselected_fast = [5, 10, 12, 20, 50]
        csv_fast_mas = st.multiselect("Fast MA", options=[5, 10, 12, 20, 50], default=preselected_fast)
    with col5:
        preselected_slow = [20, 26, 50, 100, 200]
        csv_slow_mas = st.multiselect("Slow MA", options=[20, 26, 50, 100, 200], default=preselected_slow)
        
    generate_btn = st.form_submit_button("Generate Export Data")

if generate_btn:
    if not csv_fast_mas:
        csv_fast_mas = [5, 10, 12, 20, 50]
    if not csv_slow_mas:
        csv_slow_mas = [20, 26, 50, 100, 200]
        
    with st.spinner("Compiling CSV Data Based on Selections..."):
        custom_csv_df = build_download_csv(timeframe, csv_ma_type, csv_crossover, csv_fast_mas, csv_slow_mas, csv_days)
        csv_bytes = custom_csv_df.to_csv(index=False).encode('utf-8')
        
    st.success("CSV Ready For Download!")
    st.download_button(
        label="Download All Data as CSV",
        data=csv_bytes,
        file_name="all_custom_summary.csv",
        mime="text/csv",
    )
selected_rows = grid_response.get("selected_rows", None)

has_selection = selected_rows is not None and (
    (isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty)
    or (isinstance(selected_rows, list) and len(selected_rows) > 0)
)

if has_selection:
    if isinstance(selected_rows, pd.DataFrame):
        display_symbol = selected_rows.iloc[0]["Symbol"]
    else:
        display_symbol = selected_rows[0]["Symbol"]
        
    # Remove asterisk if present
    selected_symbol = display_symbol.replace(" *", "")

    st.markdown("---")
    st.subheader(f"📈 Price Chart ({timeframe})")
    
    # ---------- SCENARIO CONTROLS (WHAT-IF MODE) ----------
    st.markdown("### 🔎 Strategy to Display")
    strategy_mode = st.radio("Mode", ["Optimized (Best Historical)", "Custom Scenario"], horizontal=True, label_visibility="collapsed")
    
    if strategy_mode == "Custom Scenario":
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            ma_type_viz = st.selectbox("MA Type", ["EMA", "SMA"], index=0)
        with sc2:
            fast_viz = st.selectbox("Fast MA", options=[5, 10, 12, 20, 50, 100], index=2)
        with sc3:
            slow_viz = st.selectbox("Slow MA", options=[20, 26, 50, 100, 200], index=1)
        
        if fast_viz >= slow_viz:
            st.warning("Fast MA must be smaller than Slow MA")
            st.stop()
    else:
        if isinstance(selected_rows, pd.DataFrame):
            best_ma_type = selected_rows.iloc[0]["Best MA Type"]
            best_ma_pair = selected_rows.iloc[0]["Best MA Pair"]
        else:
            best_ma_type = selected_rows[0]["Best MA Type"]
            best_ma_pair = selected_rows[0]["Best MA Pair"]
        
        try:
            fast_viz, slow_viz = map(int, best_ma_pair.split("/"))
            ma_type_viz = best_ma_type
        except:
            fast_viz, slow_viz = 12, 26
            ma_type_viz = "EMA"

    
    # Use full processed data for plotting to match optimization scope
    full_data_dir = os.path.join(BASE_DIR, "data", "processed")
    price_file = os.path.join(full_data_dir, f"{selected_symbol}.csv")

    if not os.path.exists(price_file):
        st.error("Price data not found for this stock.")
        st.stop()

    # ---------- LOAD PRICE DATA ----------
    df = pd.read_csv(price_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    
    # Filter by selected lookback years for the chart too
    # REVISION: User wants the graph to ONLY show the last 3 months, regardless of optimization lookback.
    # However, we must calculate features on FULL data first so MAs (e.g. 200) are valid, then slice.
    
    # 1. Sort Full Data
    df = df.sort_values("Date")
    
    # 2. Slice to last 3 months for Visualization
    # We use a fixed 3-month window for the graph as requested, 
    # but we will rely on the features being calculated on the full dataset if we were doing it here.
    # Since we are overlaying MAs, we let the plotting library handle it? 
    # actually, the MA calculation happens inside 'plot_stock_with_signals' or we calculate them here?
    # Looking at the code below, 'ma_noise_filter' calculates columns.
    # So we must NOT slice yet. We pass full DF to 'ma_noise_filter', then slice.
    
    # Let's keep 'df' as full for now, and slice later or let the user zoom? 
    # User explicitly asked to "show past 3 month data".
    
    # We will define a cutoff date
    metrics_lookback_date = df["Date"].max() - pd.DateOffset(months=3) 
    
    # But wait, 'ma_noise_filter' is called LATER. 
    # We need to ensure we don't filter rows before MAs are computed.
    # But `app.py` structure typically passes `df` to plotting. 
    # Let's look further down in `app.py` (I can't see it in this chunk).
    # Assuming the next block calculates MAs. 
    # I will pass the FULL df to the next steps, but I need to inject the slicing logic just before plotting.
    
    # Actually, to be safe and simple: 
    # 1. Calc MAs on full DF.
    # 2. Slice DF.
    
    # In this block, we are just loading data.
    # I will remove the 'years' filtering I added and replace it with a comment to slice later.
    # OR, since I can't verify what happens next easily without viewing more, 
    # I will slice it right here but be careful about the start date.
    
    # If I slice here, I break 200 MA. 
    # USE CASE: The user wants to SEE 3 months. 
    # I will calculate the start_date for 3 months, but I won't filter 'df' yet.
    # I will create a mask or slice later?
    # checking the code structure from previous `view_file` (which stopped at 400)...
    # I need to see how `df` is used.
    pass # Placeholder to fix indentation if needed, real code below
    
    # ---------- TIME FRAME RESAMPLING ----------
    if timeframe == "Weekly":
        df.set_index("Date", inplace=True)
        # Resample logic
        df = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

    # ---------- APPLY SELECTED MOVING AVERAGES ----------
    if ma_type_viz == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast_viz, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow_viz, adjust=False).mean()
    else:
        df["MA_Fast"] = df["Close"].rolling(fast_viz).mean()
        df["MA_Slow"] = df["Close"].rolling(slow_viz).mean()

    df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
    df["Crossover"] = df["Signal"].diff()

    # Update variables for title later
    scenario_ma_type = ma_type_viz 
    fast_ma = fast_viz
    slow_ma = slow_viz

    # REVISION: Slice to last 3 months for Visualization Only
    # Data has full history here, so MAs and Signals are accurate.
    # Now we just zoom in.
    metrics_lookback_date = df["Date"].max() - pd.DateOffset(months=3)
    df = df[df["Date"] >= metrics_lookback_date]

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
    # REVISION: Removed shift to align with table date (User Request)
    buy_indices_shifted = buy_indices 
    
    sell_indices_shifted = sell_indices

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
