
# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, set_key
import subprocess
import sys

# NOTE: st_aggrid removed — it crashes silently on Streamlit Cloud.
# Using native st.dataframe with selection instead.
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

def run_script(script_name, status_text, progress_bar, progress_text):
    import time
    script_path = os.path.join(src_dir, script_name)
    status_text.text(f"Running {script_name}...")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Line buffered
        )
        
        for line in iter(process.stdout.readline, ''):
            if line.startswith("PROGRESS:"):
                try:
                    parts = line.strip().split("PROGRESS:")[1].split("/")
                    curr = int(parts[0])
                    total = int(parts[1])
                    if total > 0:
                        pct = curr / total
                        progress_bar.progress(pct)
                        
                        elapsed = time.time() - start_time
                        if curr > 0:
                            total_estimated = elapsed / pct
                            remaining = max(0, total_estimated - elapsed)
                            mins, secs = divmod(int(remaining), 60)
                            progress_text.text(f"Estimated time left: {mins}m {secs}s ({curr}/{total})")
                except Exception:
                    pass
            # Optional: print(line.strip()) if debugging

        process.wait()
        if process.returncode != 0:
            stderr_output = process.stderr.read()
            st.error(f"Error running {script_name}:\n{stderr_output}")
            raise subprocess.CalledProcessError(process.returncode, process.args)
            
    except Exception as e:
        raise e

if st.sidebar.button("Update Data & Run Optimization"):
    # Check if token exists
    if not current_token:
        st.sidebar.error("⚠ Access Token Missing! Please update the token above first.")
    else:
        status_placeholder = st.sidebar.empty()
        # Initialize progress tracking elements inline
        progress_bar = st.sidebar.progress(0.0)
        progress_text = st.sidebar.empty()
        
        try:
            with st.spinner("Running data pipeline... This may take a while."):
                # 1. Fetch Data
                run_script("fetch-data-upstox.py", status_placeholder, progress_bar, progress_text)
                
                # 2. Features
                run_script("features.py", status_placeholder, progress_bar, progress_text)
                
                # 3. Trim Data
                run_script("trim_data.py", status_placeholder, progress_bar, progress_text)
                
                # 4. Optimize
                run_script("optimize_on_dynamic_noise.py", status_placeholder, progress_bar, progress_text)
                
            progress_bar.empty()
            progress_text.empty()
            status_placeholder.success("Pipeline completed successfully! ✅")
            st.rerun() # Refresh app to show new data
            
        except Exception as e:
            st.sidebar.error("Pipeline failed!")

st.sidebar.markdown("---")
st.sidebar.header("Filters & Sorting")


# Lookback Period Selection
lookback_years = st.sidebar.selectbox("Lookback Period", ["1Y", "2Y", "3Y"], index=2) # Default 3Y
years = int(lookback_years.replace("Y", ""))

# Timeframe Selection
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly"], index=0)

# Universe Selection
universe = st.sidebar.radio("Universe", ["NSE 500", "Nifty 50", "F&O", "Indices", "All NSE"], index=0)

# Min Trades Filter
min_trades = st.sidebar.slider("Minimum Trades", 0, 50, 5)

# ---------- HELPERS ----------
def get_dir_mtime(dir_path: str, ext: str = ".parquet") -> float:
    """Return the max file-mtime for `ext` files in dir_path.
    Passed as a @st.cache_data key — changes automatically when the
    pipeline writes new data, triggering auto-invalidation."""
    try:
        files = [os.path.join(dir_path, f)
                 for f in os.listdir(dir_path) if f.endswith(ext)]
        return max((os.path.getmtime(f) for f in files), default=0.0)
    except Exception:
        return 0.0


# ---------- REPORT CACHE — loaded once per session / lookback change ----------
@st.cache_data
def load_all_reports(years: int, reports_mtime: float) -> dict:
    """
    Read the first (best) row of every optimization report CSV and cache the result
    in memory.  Returns {raw_filename_stem: pd.Series}, e.g. {"GRASIM_NS": <row>}.

    Cache key includes reports_mtime — auto-invalidates when reports are updated
    by the pipeline without needing st.cache_data.clear().
    """
    suffix = f"_{years}y_dynamic_trend_noise_optimization.csv"
    cache = {}
    for file in os.listdir(reports_dir):
        if not file.endswith(suffix):
            continue
        try:
            rep = pd.read_csv(os.path.join(reports_dir, file))
            if rep.empty:
                continue
            raw_name = file.replace(suffix, "")
            cache[raw_name] = rep.iloc[0]   # store only the top (best) row
        except Exception:
            pass
    return cache

# ---------- PRICE DATA CACHE — loaded once per timeframe / data update ----------
@st.cache_data
def load_all_price_data(tf: str, data_mtime: float) -> dict:
    """
    Load all price files (Parquet preferred, CSV fallback) into memory once.
    ~20 MB total — fits comfortably in RAM for a single-user local setup.

    Cache key includes data_mtime — auto-invalidates after pipeline runs.
    Cached data must never be mutated; callers must call .copy() first.
    """
    full_data_dir = os.path.join(BASE_DIR, "data", "processed")
    price_data = {}
    needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

    for file in os.listdir(full_data_dir):
        is_parquet = file.endswith(".parquet")
        is_csv     = file.endswith(".csv")
        if not (is_parquet or is_csv):
            continue
        symbol = file.rsplit(".", 1)[0]   # strip extension
        try:
            if is_parquet:
                df = pd.read_parquet(
                    os.path.join(full_data_dir, file),
                    engine="pyarrow",
                    columns=needed_cols,
                )
            else:
                df = pd.read_csv(
                    os.path.join(full_data_dir, file),
                    usecols=needed_cols,
                )
            df["Date"] = pd.to_datetime(
                df["Date"], utc=True, errors="coerce"
            ).dt.tz_convert(None)
            df = df.sort_values("Date").reset_index(drop=True)
            if tf == "Weekly":
                df = (
                    df.set_index("Date")
                    .resample("W")
                    .agg({"Open": "first", "High": "max",
                          "Low": "min",  "Close": "last", "Volume": "sum"})
                    .dropna()
                    .reset_index()
                )
            price_data[symbol] = df
        except Exception:
            pass
    return price_data


# ---------- CACHED CROSSOVER CALCULATION ----------
# IMPORTANT: This function ALWAYS uses the report's best MA strategy to compute crossovers.
# Used in default (non-screener) mode.
@st.cache_data
def calculate_crossovers(stock_list: tuple, tf: str, years: int,
                         data_mtime: float) -> dict:
    """
    Calculate the most recent MA crossover for each stock using ONLY the stock's
    own optimized MA pair from its report file.

    stock_list MUST be a tuple so Streamlit can hash it as a cache key.
    Stocks are processed in parallel via ThreadPoolExecutor.
    data_mtime auto-invalidates the cache when the pipeline writes new data.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    reports_cache = load_all_reports(years, data_mtime)
    price_data    = load_all_price_data(tf, data_mtime)

    def process_one(symbol):
        df_cached = price_data.get(symbol)
        if df_cached is None:
            return symbol, None
        df = df_cached.copy()
        try:
            raw_name = symbol.replace(".", "_")
            best = reports_cache.get(raw_name)
            if best is None:
                return symbol, None

            fast, slow = map(int, best["MA_Pair"].split("/"))
            ma_types_to_check = [best["MA_Type"]]

            best_crossover_date = pd.Timestamp.min
            best_crossover_data = None

            for m_type in ma_types_to_check:
                if m_type == "EMA":
                    ma_fast = df["Close"].ewm(span=fast, adjust=False).mean()
                    ma_slow = df["Close"].ewm(span=slow, adjust=False).mean()
                else:
                    ma_fast = df["Close"].rolling(fast).mean()
                    ma_slow = df["Close"].rolling(slow).mean()

                raw_signal     = np.where(ma_fast > ma_slow, 1, -1)
                crossover_diff = pd.Series(raw_signal).diff().fillna(0)

                temp_df = df.copy()
                temp_df["Crossover"] = crossover_diff.values
                crossovers = temp_df[temp_df["Crossover"].abs() == 2]

                if not crossovers.empty:
                    last_row       = crossovers.iloc[-1]
                    last_date      = last_row["Date"]
                    crossover_type = "Bullish" if last_row["Crossover"] == 2 else "Bearish"
                    three_months_ago = temp_df["Date"].max() - pd.DateOffset(months=3)
                    recent_trades  = len(crossovers[crossovers["Date"] >= three_months_ago])

                    if last_date > best_crossover_date:
                        best_crossover_date = last_date
                        best_crossover_data = {
                            "Recent Bullish Crossover": last_date,
                            "Recent 3M Trades":         recent_trades,
                            "Crossover Type":           crossover_type,
                        }

            return symbol, best_crossover_data
        except Exception:
            return symbol, None

    crossover_data = {}
    workers = min(8, max(1, len(stock_list)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, s): s for s in stock_list}
        for future in as_completed(futures):
            try:
                symbol, data = future.result()
                if data:
                    crossover_data[symbol] = data
            except Exception:
                pass

    print(f"[CROSSOVER] Done: {len(crossover_data)}/{len(stock_list)} stocks.")
    return crossover_data


@st.cache_data
def calculate_crossovers_with_pair(
    stock_list: tuple,
    tf: str,
    fast: int,
    slow: int,
    ma_type: str,
    data_mtime: float,
) -> dict:
    """
    Calculate the most recent MA crossover for each stock using an EXPLICIT
    MA pair (fast/slow/ma_type) instead of each stock's best historical pair.

    This is used when the Refine Screener is active so that the table matches
    the Excel export — every stock in the universe is evaluated on the same
    user-selected pair, regardless of what its optimal MA happened to be.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    price_data = load_all_price_data(tf, data_mtime)

    # If "Both" MA types are selected, we check EMA first then SMA and keep
    # whichever produced the more recent crossover (same logic as the CSV export).
    types_to_check = ["EMA", "SMA"] if ma_type == "Both" else [ma_type]

    def process_one(symbol):
        df_cached = price_data.get(symbol)
        if df_cached is None:
            return symbol, None
        df = df_cached.copy()
        try:
            best_crossover_date = pd.Timestamp.min
            best_crossover_data = None

            for m_type in types_to_check:
                if m_type == "EMA":
                    ma_fast = df["Close"].ewm(span=fast, adjust=False).mean()
                    ma_slow = df["Close"].ewm(span=slow, adjust=False).mean()
                else:
                    ma_fast = df["Close"].rolling(fast).mean()
                    ma_slow = df["Close"].rolling(slow).mean()

                raw_signal     = np.where(ma_fast > ma_slow, 1, -1)
                crossover_diff = pd.Series(raw_signal).diff().fillna(0)

                temp_df = df.copy()
                temp_df["Crossover"] = crossover_diff.values
                crossovers = temp_df[temp_df["Crossover"].abs() == 2]

                if not crossovers.empty:
                    last_row       = crossovers.iloc[-1]
                    last_date      = last_row["Date"]
                    crossover_type = "Bullish" if last_row["Crossover"] == 2 else "Bearish"
                    three_months_ago = temp_df["Date"].max() - pd.DateOffset(months=3)
                    recent_trades  = len(crossovers[crossovers["Date"] >= three_months_ago])

                    if last_date > best_crossover_date:
                        best_crossover_date = last_date
                        best_crossover_data = {
                            "Recent Bullish Crossover": last_date,
                            "Recent 3M Trades":         recent_trades,
                            "Crossover Type":           crossover_type,
                        }

            return symbol, best_crossover_data
        except Exception:
            return symbol, None

    crossover_data = {}
    workers = min(8, max(1, len(stock_list)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, s): s for s in stock_list}
        for future in as_completed(futures):
            try:
                symbol, data = future.result()
                if data:
                    crossover_data[symbol] = data
            except Exception:
                pass

    print(f"[CROSSOVER PAIR {fast}/{slow} {ma_type}] Done: {len(crossover_data)}/{len(stock_list)} stocks.")
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
    names_by_date = {}  

    for file in os.listdir(full_data_dir):
        is_parquet = file.endswith(".parquet")
        is_csv = file.endswith(".csv")
        if not (is_parquet or is_csv):
            continue
        symbol = file.rsplit(".", 1)[0]

        price_file = os.path.join(full_data_dir, file)
        if not os.path.exists(price_file):
            continue

        try:
            if is_parquet:
                df = pd.read_parquet(price_file, engine="pyarrow")
            else:
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

            # Clean symbol name for display (remove .NS suffix)
            display_name = symbol.split(".")[0] if "." in symbol else symbol

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
                            names_by_date[day_key] = {k: [] for k in ma_pairs}
                            
                        if bool(is_triggered) and day_key not in counted_dates:
                            counts_by_date[day_key][col_name] += 1
                            names_by_date[day_key][col_name].append(display_name)
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

    # Build stock names columns (comma-separated)
    names_rows = []
    for day in download_df["Date"]:
        day_key = day.date()
        if day_key in names_by_date:
            names_rows.append({
                f"Stocks_{k}": ",".join(v) if v else ""
                for k, v in names_by_date[day_key].items()
            })
        else:
            names_rows.append({f"Stocks_{k}": "" for k in ma_pairs})
    
    names_df = pd.DataFrame(names_rows)
    download_df = pd.concat([download_df, names_df], axis=1)

    # Normalize download_df dates to midnight for consistent merging
    download_df["Date"] = download_df["Date"].dt.normalize()

    # ---------- ADD NIFTY CLOSE ----------

    nifty_found = False
    nifty_candidates = ["Nifty 50", "NIFTY 50", "NIFTY"]

    for nifty_symbol in nifty_candidates:

        nifty_file_pq = os.path.join(full_data_dir, f"{nifty_symbol}.parquet")
        nifty_file_csv = os.path.join(full_data_dir, f"{nifty_symbol}.csv")

        try:
            if os.path.exists(nifty_file_pq):
                nifty_df = pd.read_parquet(nifty_file_pq, engine="pyarrow")
            elif os.path.exists(nifty_file_csv):
                nifty_df = pd.read_csv(nifty_file_csv)
            else:
                continue

            nifty_df["Date"] = pd.to_datetime(
                nifty_df["Date"], utc=True, errors="coerce"
            ).dt.tz_convert(None)

            nifty_df = nifty_df[["Date", "Close"]].dropna()
            nifty_df.rename(columns={"Close": "NIFTY*"}, inplace=True)
            # Normalize to midnight for consistent merging
            nifty_df["Date"] = nifty_df["Date"].dt.normalize()

            if tf == "Weekly":
                nifty_df = (
                    nifty_df.set_index("Date")
                    .resample("W")
                    .last()
                    .dropna()
                    .reset_index()
                )

            download_df = download_df.merge(nifty_df, on="Date", how="left")
            # Forward-fill missing NIFTY values (handles trading holidays)
            download_df["NIFTY*"] = download_df["NIFTY*"].ffill()
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
                # Forward-fill missing NIFTY values (handles trading holidays)
                download_df["NIFTY*"] = download_df["NIFTY*"].ffill()
                nifty_found = True
        except Exception:
            pass

        if not nifty_found:
            download_df["NIFTY*"] = np.nan

    # Build ordered columns: count columns interleaved with their stock name columns
    ordered_cols = ["Date"]
    for k in ma_pairs.keys():
        ordered_cols.append(k)
        ordered_cols.append(f"Stocks_{k}")
    ordered_cols.append("NIFTY*")
    
    # Filter based on user-defined number of days
    # (Getting the max date and slicing the past `days_limit`)
    max_date = download_df["Date"].max()
    cutoff_date = max_date - pd.Timedelta(days=days_limit)
    download_df = download_df[download_df["Date"] >= cutoff_date]

    return download_df[ordered_cols].reset_index(drop=True)


# ---------- CUSTOM SCRENER CONTROLS ----------
if "screener_active" not in st.session_state:
    st.session_state.screener_active = False
if "scr_ma_type" not in st.session_state:
    st.session_state.scr_ma_type = "Both"
if "scr_fast_ma" not in st.session_state:
    st.session_state.scr_fast_ma = 20
if "scr_slow_ma" not in st.session_state:
    st.session_state.scr_slow_ma = 50

st.subheader("🎯 Refine Screener")
with st.form("screener_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        scr_ma_type_input = st.selectbox("MA Type", ["Both", "SMA", "EMA"], index=["Both", "SMA", "EMA"].index(st.session_state.scr_ma_type))
    with c2:
        scr_fast_ma_input = st.selectbox("Fast MA", options=[5, 10, 12, 20, 50], index=[5, 10, 12, 20, 50].index(st.session_state.scr_fast_ma))
    with c3:
        scr_slow_ma_input = st.selectbox("Slow MA", options=[20, 26, 50, 100, 200], index=[20, 26, 50, 100, 200].index(st.session_state.scr_slow_ma))
        
    colA, colB, _ = st.columns([1, 1.5, 3])
    with colA:
        refine_btn = st.form_submit_button("Refine Screener")
    with colB:
        refresh_btn = st.form_submit_button("Refresh")

if refine_btn:
    if scr_fast_ma_input >= scr_slow_ma_input:
        st.warning("Fast MA must be smaller than Slow MA to filter properly.")
    else:
        st.session_state.screener_active = True
        st.session_state.scr_ma_type = scr_ma_type_input
        st.session_state.scr_fast_ma = scr_fast_ma_input
        st.session_state.scr_slow_ma = scr_slow_ma_input
        st.rerun()

if refresh_btn:
    st.session_state.screener_active = False
    st.rerun()

# ================================================================
# PASS 1 — UNIVERSE PREPARATION
# Load cached datasets, apply universe filter, collect all eligible
# stocks. No screener / min_trades filtering yet.
# ================================================================
import time as _time
_t0 = _time.perf_counter()

# Compute directory mtimes — used as cache keys for auto-invalidation
_processed_dir = os.path.join(BASE_DIR, "data", "processed")
data_mtime    = get_dir_mtime(_processed_dir, ".parquet") or get_dir_mtime(_processed_dir, ".csv")
reports_mtime = get_dir_mtime(reports_dir, ".csv")

# Load report + price caches (disk only on first call or after data update)
reports_cache = load_all_reports(years, reports_mtime)
_t_reports = _time.perf_counter()

price_data = load_all_price_data(timeframe, data_mtime)
_t_prices = _time.perf_counter()

target_suffix        = f"_{years}y_dynamic_trend_noise_optimization.csv"
all_universe_rows    = []   # {raw_name, symbol, best} for every universe-eligible stock
all_universe_symbols = []   # symbol strings for the crossover batch

for file in os.listdir(reports_dir):
    if not file.endswith(target_suffix):
        continue

    raw_name = file.replace(target_suffix, "")
    if "Nifty" in raw_name:   # heuristic: index names may have underscore-for-space
        pass
    symbol = raw_name.replace("_", ".")   # e.g. RELIANCE_NS → RELIANCE.NS

    # ── Universe filter ─────────────────────────────────────────────
    if universe == "Nifty 50":
        if symbol.split(".")[0] not in NIFTY_50_SYMBOLS:
            continue
    elif universe == "F&O":
        if symbol.split(".")[0] not in FNO_STOCKS:
            continue
    elif universe == "Indices":
        norm_symbol = symbol.replace(".", " ").replace("_", " ")
        is_index, real_index_name = False, ""
        for idx in INDICES:
            if idx == norm_symbol or idx == symbol:
                is_index, real_index_name = True, idx
                break
        if not is_index:
            continue
        symbol = real_index_name
    elif universe == "All NSE":
        if symbol in INDICES or symbol.replace(".", " ") in INDICES:
            continue
    else:   # NSE 500 (default)
        start_name = symbol.split(".")[0].replace("_", " ")
        if start_name in INDICES or symbol in INDICES:
            continue
    # ─────────────────────────────────────────────────────────────────

    best = reports_cache.get(raw_name)
    if best is None:
        continue

    all_universe_rows.append({"raw_name": raw_name, "symbol": symbol, "best": best})
    all_universe_symbols.append(symbol)

# Compute crossovers for the FULL universe.
# - Default mode: use each stock's own best historical MA pair.
# - Screener mode: use the user-selected pair for ALL stocks (matches Excel logic).
with st.spinner("Calculating crossovers..."):
    if st.session_state.screener_active:
        crossover_map_all = calculate_crossovers_with_pair(
            tuple(all_universe_symbols),
            timeframe,
            st.session_state.scr_fast_ma,
            st.session_state.scr_slow_ma,
            st.session_state.scr_ma_type,
            data_mtime,
        )
    else:
        crossover_map_all = calculate_crossovers(
            tuple(all_universe_symbols),
            timeframe,
            years,
            data_mtime,
        )
_t_cross = _time.perf_counter()

# ⏱ Sidebar profiling — shows timing for last run
st.sidebar.markdown("---")
st.sidebar.caption(
    f"⏱ Reports: {_t_reports - _t0:.2f}s | "
    f"Prices: {_t_prices - _t_reports:.2f}s | "
    f"Crossovers: {_t_cross - _t_prices:.2f}s"
)

# ================================================================
# PASS 2 — SCREENER EVALUATION
# Apply min_trades filter. In screener mode, ALL universe stocks are
# included (crossovers already computed on the selected pair above).
# ================================================================
rows               = []
symbols_to_process = []

for item in all_universe_rows:
    symbol   = item["symbol"]
    best     = item["best"]

    # In screener mode we no longer drop stocks whose Best MA != selected pair.
    # Crossovers were already computed using the selected pair for every stock.

    # ── Min Trades filter ──────────────────────────────────────────
    trades = int(best["Trades"])
    if trades < min_trades:
        continue

    symbols_to_process.append(symbol)

    win_rate = best["WinRate"]
    if win_rate > 1:
        win_rate /= 100
    wins = int(round(win_rate * trades))
    win_rate_str = f"{round(win_rate * 100, 1)}% ({wins}/{trades})"

    base_symbol_clean = symbol.split(".")[0]
    display_symbol = f"{symbol} *" if base_symbol_clean in FNO_STOCKS else symbol

    # In screener mode, show the user-selected pair that generated the crossover.
    # In default mode, show the stock's own optimised pair (same pair used for crossover calc).
    if st.session_state.screener_active:
        crossover_ma_type = st.session_state.scr_ma_type
        crossover_ma_pair = f"{st.session_state.scr_fast_ma}/{st.session_state.scr_slow_ma}"
    else:
        crossover_ma_type = best["MA_Type"]
        crossover_ma_pair = best["MA_Pair"]

    rows.append({
        "Symbol":            display_symbol,
        "RawSymbol":         symbol,
        "Crossover MA Type": crossover_ma_type,
        "Crossover MA Pair": crossover_ma_pair,
        "Return (%)":        round(best["Return"], 2),
        "Win Rate (%)": win_rate_str,
        "RawWinRate":        win_rate,
        "Sharpe":            round(best["Sharpe"], 2),
        "Sigma (Market)":   best.get("MarketSigma", best.get("volatility", 0)),
        "Strategy Vol":      round(best.get("StrategyAggr", best.get("Volatility", 0)), 2),
        "Trades":            trades,
    })

# Subset crossovers — O(1) dict lookup, zero recomputation
crossover_map = {
    s: crossover_map_all[s]
    for s in symbols_to_process
    if s in crossover_map_all
}

if not rows:
    if st.session_state.screener_active:
        st.info("No stocks found matching the Refine Screener criteria.")
    else:
        st.info("No report files found yet. Run the data pipeline from the sidebar to generate reports.")


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
    
    # Fill any NaN values to prevent st_aggrid from crashing during JSON serialization
    summary_df = summary_df.fillna(0)
    
    summary_df = summary_df.reset_index(drop=True)

# ---------- TABLE ----------
table_title = "📊 Stock Performance Summary "
if st.session_state.screener_active:
    table_title += f"(Selected Strategy: {st.session_state.scr_ma_type if st.session_state.scr_ma_type != 'Both' else 'Any MA Type'} {st.session_state.scr_fast_ma}/{st.session_state.scr_slow_ma})"
else:
    table_title += "(Best Historical Strategy)"

st.subheader(table_title)

# Use native st.dataframe with selection (works reliably on Streamlit Cloud)
event = st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
    height=400,
    on_select="rerun",
    selection_mode="single-row",
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
# Handle row selection from native st.dataframe
selected_indices = event.selection.rows if event and event.selection else []
has_selection = len(selected_indices) > 0

if has_selection:
    selected_row_idx = selected_indices[0]
    display_symbol = summary_df.iloc[selected_row_idx]["Symbol"]
        
    # Remove asterisk if present
    selected_symbol = display_symbol.replace(" *", "")

    st.markdown("---")
    
    col_chart_title, col_chart_lookback = st.columns([3, 1])
    with col_chart_title:
        st.subheader(f"📈 Price Chart ({timeframe})")
    with col_chart_lookback:
        graph_lookback = st.selectbox("Graph Lookback", ["1 Month", "3 Months", "1 Year", "5 Years"], index=1)
    
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
        best_ma_type = summary_df.iloc[selected_row_idx]["Crossover MA Type"]
        best_ma_pair = summary_df.iloc[selected_row_idx]["Crossover MA Pair"]
        
        try:
            fast_viz, slow_viz = map(int, best_ma_pair.split("/"))
            # "Both" is not a valid MA type for plotting — default to EMA
            ma_type_viz = best_ma_type if best_ma_type in ("EMA", "SMA") else "EMA"
        except:
            fast_viz, slow_viz = 12, 26
            ma_type_viz = "EMA"

    
    # ---------- LOAD PRICE DATA (from preloaded cache) ----------
    # price_data is already in memory — this is a cache lookup, not a disk read.
    df = price_data.get(selected_symbol)
    if df is None:
        st.error("Price data not found for this stock.")
        st.stop()
    # Always copy before adding computed columns — never mutate cached objects
    df = df.copy()
    # Data is already sorted and (if Weekly) resampled by load_all_price_data

    
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

    # REVISION: Slice for Visualization Only
    # Data has full history here, so MAs and Signals are accurate.
    # Now we just zoom in based on selected lookback.
    if graph_lookback == "1 Month":
        metrics_lookback_date = df["Date"].max() - pd.DateOffset(months=1)
    elif graph_lookback == "3 Months":
        metrics_lookback_date = df["Date"].max() - pd.DateOffset(months=3)
    elif graph_lookback == "1 Year":
        metrics_lookback_date = df["Date"].max() - pd.DateOffset(years=1)
    elif graph_lookback == "5 Years":
        metrics_lookback_date = df["Date"].max() - pd.DateOffset(years=5)
        
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
