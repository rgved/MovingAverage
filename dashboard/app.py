
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
    current_token = new_token
    st.sidebar.success("Token updated!")

st.sidebar.markdown("---")
st.sidebar.header("Data Management")

def run_script(script_name, status_text, progress_bar, progress_text):
    import time
    script_path = os.path.join(src_dir, script_name)
    status_text.text(f"Running {script_name}...")
    start_time = time.time()
    
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1 # Line buffered
    )
    
    logs = st.session_state.get("pipeline_logs_list", [])
    for line in iter(process.stdout.readline, ''):
        logs.append(line.rstrip())
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
    
    st.session_state.pipeline_logs_list = logs
    st.session_state.pipeline_logs = "\n".join(logs[-100:])
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

if st.sidebar.button("Update Data & Run Optimization"):
    # Check if token exists
    if not current_token:
        st.sidebar.error("⚠ Access Token Missing! Please update the token above first.")
    else:
        st.session_state.pipeline_logs_list = []
        st.session_state.pipeline_logs = ""
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
            
        except subprocess.CalledProcessError as cpe:
            st.sidebar.error(f"Pipeline failed with exit code {cpe.returncode}. Check Pipeline Logs below.")
        except Exception as e:
            st.sidebar.error(f"Pipeline failed: {e}")

# Pipeline Logs expander
with st.sidebar.expander("Pipeline Logs", expanded=False):
    log_content = st.session_state.get("pipeline_logs", "No logs yet.")
    st.code(log_content, language="text")

# Cache-clear button — forces Streamlit to recompute all crossovers from fresh data
if st.sidebar.button("🔄 Clear Cache & Refresh", help="Use this if crossover dates look stale after a data update."):
    st.cache_data.clear()
    st.session_state.scr_start_date = None
    st.session_state.scr_end_date = None
    st.rerun()

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
min_trades = st.sidebar.slider("Minimum Trades", 0, 50, 0)

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
# Scans ALL MA pair combinations to find the most recent crossover per stock.
# Used in default (non-screener) mode.

# Increment this when crossover logic changes to force Streamlit cache bust.
_CROSSOVER_LOGIC_VERSION = 3   # v3: convergence-safe EMA + direct diff sign-change

# All valid MA pairs to scan (fast < slow)
_FAST_MAS = [5, 10, 12, 20, 50]
_SLOW_MAS = [20, 26, 50, 100, 200]
_ALL_MA_PAIRS = [(f, s) for f in _FAST_MAS for s in _SLOW_MAS if f < s]
_ALL_MA_TYPES = ["EMA", "SMA"]


def _ema_min_periods(span: int) -> int:
    """Minimum bars needed before EWM is considered converged.

    EWM with alpha=2/(span+1) retains (1-alpha)^k weight from bar k bars ago.
    We wait until the oldest bar contributes <1% of total weight — i.e.
    (1-alpha)^k < 0.01  =>  k > log(0.01)/log(1-alpha).
    This is roughly 2.3*span for small alpha.  We cap at 3*span for safety.
    The key effect: for EMA200, min_periods jumps from 200 to ~461, so stocks
    with fewer than 461 rows will show NaN for EMA200 instead of a fake
    unconverged value that can produce spurious crossovers.
    """
    import math
    alpha = 2.0 / (span + 1)
    if alpha >= 1:
        return span
    k = math.ceil(math.log(0.01) / math.log(1.0 - alpha))
    return max(span, min(k, span * 3))  # at least span, at most 3*span


def _compute_ma(close: pd.Series, span: int, ma_type: str) -> pd.Series:
    """Compute a moving average.
    EMA uses a convergence-safe min_periods (see _ema_min_periods) so that
    unconverged early values are NaN instead of producing fake crossovers.
    SMA uses the window size as min_periods (standard behaviour).
    """
    if ma_type == "EMA":
        return close.ewm(span=span, min_periods=_ema_min_periods(span), adjust=False).mean()
    else:
        return close.rolling(span, min_periods=span).mean()


def compute_crossover_series(
    ma_fast: pd.Series, ma_slow: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Detect MA crossovers on the EXACT bar they occur with no off-by-one error.

    Method: work directly on the difference (fast - slow) instead of discretising
    to a ±1 signal first.  A crossover is marked on bar i when:
      - both MAs are valid on bar i AND bar i-1, AND
      - (fast[i] - slow[i]) has a different sign from (fast[i-1] - slow[i-1]).

    This avoids the off-by-one that arises from diff()-ing a ±1 signal when the
    previous bar's signal value is NaN (first valid bar edge case).

    Returns ``(signal, crossover)`` where:
      - ``signal``    is the regime: 1.0 = bullish, -1.0 = bearish, NaN = no data.
      - ``crossover`` is ±2 on the exact bar the regime flips, 0 elsewhere.
    """
    both_valid  = ma_fast.notna() & ma_slow.notna()
    prev_valid  = both_valid.shift(1).fillna(False)

    # Regime signal (NaN where data is missing)
    signal = pd.Series(np.nan, index=ma_fast.index, dtype=float)
    signal.loc[both_valid] = np.where(
        ma_fast[both_valid] > ma_slow[both_valid], 1.0, -1.0
    )

    # Difference series — sign change = crossover
    diff     = (ma_fast - ma_slow).astype(float)
    prev_diff = diff.shift(1)

    crossover = pd.Series(0.0, index=ma_fast.index)
    # Bullish: fast crossed ABOVE slow (diff went from negative to positive)
    bull = both_valid & prev_valid & (prev_diff < 0) & (diff > 0)
    # Bearish: fast crossed BELOW slow (diff went from positive to negative)
    bear = both_valid & prev_valid & (prev_diff > 0) & (diff < 0)
    crossover[bull] =  2.0
    crossover[bear] = -2.0

    return signal, crossover


@st.cache_data
def calculate_crossovers(stock_list: tuple, tf: str, years: int,
                         data_mtime: float, reports_mtime: float,
                         logic_version: int = _CROSSOVER_LOGIC_VERSION) -> dict:
    """
    Calculate the most recent MA crossover for each stock by scanning ALL
    MA pair combinations (fast/slow x EMA/SMA).

    Returns the most recent crossover from ANY pair, along with which
    MA type and pair produced it.

    stock_list MUST be a tuple so Streamlit can hash it as a cache key.
    data_mtime + reports_mtime auto-invalidate the cache when the pipeline
    writes new data.  logic_version busts the cache when crossover logic changes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    price_data = load_all_price_data(tf, data_mtime)

    def process_one(symbol):
        df_cached = price_data.get(symbol)
        if df_cached is None:
            return symbol, None
        df = df_cached.copy()
        try:
            best_crossover_date = pd.Timestamp.min
            best_crossover_data = None

            for fast, slow in _ALL_MA_PAIRS:
                for m_type in _ALL_MA_TYPES:
                    ma_fast = _compute_ma(df["Close"], fast, m_type)
                    ma_slow = _compute_ma(df["Close"], slow, m_type)

                    _, crossover_diff = compute_crossover_series(ma_fast, ma_slow)

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
                                "Crossover MA Type":        m_type,
                                "Crossover MA Pair":        f"{fast}/{slow}",
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

    print(f"[CROSSOVER ALL PAIRS] Done: {len(crossover_data)}/{len(stock_list)} stocks.")
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
                ma_fast = _compute_ma(df["Close"], fast, m_type)
                ma_slow = _compute_ma(df["Close"], slow, m_type)

                _, crossover_diff = compute_crossover_series(ma_fast, ma_slow)

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
                            "Crossover MA Type":        m_type,
                            "Crossover MA Pair":        f"{fast}/{slow}",
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
def build_crossover_event_rows(
    stock_list: tuple,
    tf: str,
    start_date_iso: str,
    end_date_iso: str,
    ma_type: str,
    fast_filter,
    slow_filter,
    signal_filter: str,
    data_mtime: float,
) -> list:
    """
    Build event-level crossover rows for a specific screening date range.

    Unlike the stock-summary helpers above, this returns every crossover event
    that matches the selected date range and filters, so the screener can start from
    the complete event set and only then refine it.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    price_data = load_all_price_data(tf, data_mtime)
    start_date = pd.to_datetime(start_date_iso).date()
    end_date = pd.to_datetime(end_date_iso).date()

    if fast_filter is not None and slow_filter is not None:
        ma_pairs = [(fast_filter, slow_filter)] if fast_filter < slow_filter else []
    else:
        ma_pairs = []
        for fast, slow in _ALL_MA_PAIRS:
            if fast_filter is not None and fast != fast_filter:
                continue
            if slow_filter is not None and slow != slow_filter:
                continue
            ma_pairs.append((fast, slow))

    types_to_check = ["EMA", "SMA"] if ma_type == "Both" else [ma_type]

    def process_one(symbol):
        df_cached = price_data.get(symbol)
        if df_cached is None or df_cached.empty:
            return []

        df = df_cached.copy()
        rows = []

        try:
            latest_date = df["Date"].max()
            three_months_ago = latest_date - pd.DateOffset(months=3)

            for fast, slow in ma_pairs:
                for m_type in types_to_check:
                    ma_fast = _compute_ma(df["Close"], fast, m_type)
                    ma_slow = _compute_ma(df["Close"], slow, m_type)

                    _, crossover_diff = compute_crossover_series(ma_fast, ma_slow)

                    temp_df = df.copy()
                    temp_df["Crossover"] = crossover_diff.values
                    crossovers = temp_df[temp_df["Crossover"].abs() == 2].copy()

                    if crossovers.empty:
                        continue

                    recent_trades = int((crossovers["Date"] >= three_months_ago).sum())

                    for _, cross_row in crossovers.iterrows():
                        cross_date = cross_row["Date"]
                        if pd.isna(cross_date):
                            continue
                        cross_day = cross_date.date()
                        if cross_day < start_date or cross_day > end_date:
                            continue

                        crossover_type = "Bullish" if cross_row["Crossover"] == 2 else "Bearish"
                        if signal_filter != "Both" and crossover_type != signal_filter:
                            continue

                        rows.append({
                            "RawSymbol": symbol,
                            "Crossover Date": str(cross_date.date()),
                            "Crossover Signal": crossover_type,
                            "Crossover MA Type": m_type,
                            "Crossover MA Pair": f"{fast}/{slow}",
                            "Recent Bullish Crossover": cross_date,
                            "Recent 3M Trades": recent_trades,
                        })
        except Exception:
            return []

        return rows

    event_rows = []
    workers = min(8, max(1, len(stock_list)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, s): s for s in stock_list}
        for future in as_completed(futures):
            try:
                event_rows.extend(future.result())
            except Exception:
                pass

    print(
        f"[CROSSOVER EVENTS {start_date_iso} -> {end_date_iso} {ma_type} "
        f"{fast_filter or 'ANY'}/{slow_filter or 'ANY'} {signal_filter}] "
        f"Done: {len(event_rows)} events."
    )
    return event_rows


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
                    if ma_type_filter in ["Both", "EMA"]:
                        ma_pairs[f"Bullish_EMA_{f}_{s}"] = (f, s, "EMA", "Bullish")
                    if ma_type_filter in ["Both", "SMA"]:
                        ma_pairs[f"Bullish_SMA_{f}_{s}"] = (f, s, "SMA", "Bullish")
                if crossover_filter in ["Both", "Bearish"]:
                    if ma_type_filter in ["Both", "EMA"]:
                        ma_pairs[f"Bearish_EMA_{f}_{s}"] = (f, s, "EMA", "Bearish")
                    if ma_type_filter in ["Both", "SMA"]:
                        ma_pairs[f"Bearish_SMA_{f}_{s}"] = (f, s, "SMA", "Bearish")

    counts_by_date = {}
    names_by_date = {}  

    files = [f for f in os.listdir(full_data_dir) if f.endswith(".parquet") or f.endswith(".csv")]

    for file in files:
        # Skip indices here — they are handled separately for NIFTY*
        start_name = file.split(".")[0]
        if start_name in INDICES or file.replace(".parquet", "") in INDICES or file.replace(".csv", "") in INDICES:
            continue

        is_parquet = file.endswith(".parquet")
        is_csv = file.endswith(".csv")
        if not (is_parquet or is_csv):
            continue

        symbol = file.rsplit(".", 1)[0]
        display_name = f"{symbol} *" if symbol.split(".")[0] in FNO_STOCKS else symbol
        price_file = os.path.join(full_data_dir, file)

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
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna()
                    .reset_index()
                )

            for col_name, (fast, slow, ma_type, direction) in ma_pairs.items():
                temp_df = df.copy()

                temp_df["MA_Fast"] = _compute_ma(temp_df["Close"], fast, ma_type)
                temp_df["MA_Slow"] = _compute_ma(temp_df["Close"], slow, ma_type)

                temp_df["Signal"], temp_df["Crossover"] = compute_crossover_series(
                    temp_df["MA_Fast"], temp_df["MA_Slow"]
                )

                if direction == "Bullish":
                    signal_triggered = temp_df["Crossover"] == 2
                else:
                    signal_triggered = temp_df["Crossover"] == -2

                # Track which dates this stock already contributed to for this col_name,
                # so we only count each stock once per date per crossover bucket.
                counted_dates = set()

                for dt, is_triggered in zip(temp_df["Date"], signal_triggered):
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
                        # multiple rows for the same signal/day.
                        counted_dates.add(day_key)
        except Exception:
            continue

    download_rows = [
        {"Date": pd.to_datetime(day), **vals}
        for day, vals in counts_by_date.items()
    ]

    download_df = pd.DataFrame(download_rows)
    if download_df.empty:
        return pd.DataFrame(columns=["Date", *ma_pairs.keys(), "NIFTY*"])

    download_df = download_df.sort_values("Date").reset_index(drop=True)

    # Build a parallel notes/name map so the CSV consumer can inspect which names fired.
    names_rows = []
    for day in download_df["Date"]:
        day_key = day.date()
        if day_key in names_by_date:
            names_rows.append({
                "Date": day,
                **{
                    f"{k}_Names": ", ".join(v)
                    for k, v in names_by_date[day_key].items()
                }
            })

    names_df = pd.DataFrame(names_rows) if names_rows else pd.DataFrame(columns=["Date"])

    # Normalize download_df dates to midnight for consistent merging
    download_df["Date"] = download_df["Date"].dt.normalize()

    # ---------- NIFTY* COLUMN ----------
    # First try local processed/index files; if not found, fallback to yfinance for ^NSEI.
    nifty_df = None
    nifty_candidates = ["Nifty 50", "NIFTY 50", "NIFTY"]

    for nifty_symbol in nifty_candidates:
        nifty_file_pq = os.path.join(full_data_dir, f"{nifty_symbol}.parquet")
        nifty_file_csv = os.path.join(full_data_dir, f"{nifty_symbol}.csv")

        try:
            if os.path.exists(nifty_file_pq):
                nifty_df = pd.read_parquet(nifty_file_pq, engine="pyarrow")
            elif os.path.exists(nifty_file_csv):
                nifty_df = pd.read_csv(nifty_file_csv)

            if nifty_df is not None and not nifty_df.empty:
                nifty_df["Date"] = pd.to_datetime(
                    nifty_df["Date"], utc=True, errors="coerce"
                ).dt.tz_convert(None)

                nifty_df = nifty_df[["Date", "Close"]].dropna()
                nifty_df = nifty_df.rename(columns={"Close": "NIFTY*"})

                # Normalize to midnight (strip time component) to match download_df dates
                nifty_df["Date"] = nifty_df["Date"].dt.normalize()

                if tf == "Weekly":
                    nifty_df = (
                        nifty_df.set_index("Date")
                        .resample("W")
                        .last()
                        .dropna()
                        .reset_index()
                    )
                    nifty_df["Date"] = nifty_df["Date"].dt.normalize()

                download_df = download_df.merge(nifty_df, on="Date", how="left")
                break
        except Exception:
            nifty_df = None
            continue

    # Fallback: fetch NIFTY from yfinance if local index file missing
    if "NIFTY*" not in download_df.columns or download_df["NIFTY*"].isna().all():
        try:
            import yfinance as yf
            max_days = int(days_limit) + 40  # pad to handle weekends/holidays
            nifty_yf = yf.download("^NSEI", period=f"{max_days}d", interval="1d", auto_adjust=False, progress=False)
            if not nifty_yf.empty:
                nifty_yf = nifty_yf.reset_index()[["Date", "Close"]]
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
                    nifty_yf["Date"] = nifty_yf["Date"].dt.normalize()

                download_df = download_df.merge(nifty_yf, on="Date", how="left")
        except Exception:
            # Leave NIFTY* missing if even fallback fails
            pass

    if not names_df.empty:
        download_df = download_df.merge(names_df, on="Date", how="left")

    ordered_cols = ["Date"]
    ordered_cols.extend(ma_pairs.keys())
    if "NIFTY*" in download_df.columns:
        ordered_cols.append("NIFTY*")
    ordered_cols.extend([c for c in download_df.columns if c.endswith("_Names")])
    download_df = download_df[[c for c in ordered_cols if c in download_df.columns]]

    # Filter to the requested recent window by actual rows present
    # (Getting the max date and slicing the past `days_limit`)
    max_date = download_df["Date"].max()
    cutoff_date = max_date - pd.Timedelta(days=days_limit)
    download_df = download_df[download_df["Date"] >= cutoff_date]

    return download_df.reset_index(drop=True)

if "screener_active" not in st.session_state:
    st.session_state.screener_active = False
if "scr_ma_type" not in st.session_state:
    st.session_state.scr_ma_type = "Both"
if "scr_fast_ma" not in st.session_state:
    st.session_state.scr_fast_ma = "Any"
if "scr_slow_ma" not in st.session_state:
    st.session_state.scr_slow_ma = "Any"
if "scr_signal" not in st.session_state:
    st.session_state.scr_signal = "Both"
if "scr_start_date" not in st.session_state:
    st.session_state.scr_start_date = None
if "scr_end_date" not in st.session_state:
    st.session_state.scr_end_date = None


# Load cached datasets, apply universe filter, collect all eligible
# report files, and build the final DataFrame.
# This avoids repeated disk scans and recomputation on every small UI change.

_processed_dir = os.path.join(BASE_DIR, "data", "processed")

# Compute directory mtimes — used as cache keys for auto-invalidation
data_mtime    = get_dir_mtime(_processed_dir, ".parquet") or get_dir_mtime(_processed_dir, ".csv")
reports_mtime = get_dir_mtime(reports_dir, ".csv")

# Load report + price caches (disk only on first call or after data update)
reports_cache = load_all_reports(years, reports_mtime)
price_data = load_all_price_data(timeframe, data_mtime)

# collect available report files once for current lookback
report_files = set()
target_suffix        = f"_{years}y_dynamic_trend_noise_optimization.csv"
non_yearly_suffix    = f"_dynamic_trend_noise_optimization.csv"
for file in os.listdir(reports_dir):
    if file.endswith(target_suffix):
        report_files.add(file)
    # Fallback if only old non-year-specific report exists
    elif file.endswith(non_yearly_suffix) and target_suffix.replace(f"_{years}y", "") == non_yearly_suffix:
        report_files.add(file)

# build crossover metadata only for symbols that *might* appear in the table
# based on current report files and universe selection
candidate_symbols = []
for raw_name in reports_cache.keys():
    symbol = raw_name.replace("_", ".")

    if universe == "Nifty 50":
        if symbol.split(".")[0] not in NIFTY_50_SYMBOLS:
            continue
    elif universe == "F&O":
        if symbol.split(".")[0] not in FNO_STOCKS:
            continue
    elif universe == "Indices":
        norm_symbol = symbol.replace(".", " ").replace("_", " ")
        if not any(idx == norm_symbol or idx == symbol for idx in INDICES):
            continue
    elif universe == "All NSE":
        if symbol in INDICES or symbol.replace(".", " ") in INDICES:
            continue
    else:  # NSE 500
        # exclude known index names
        start_name = symbol.split(".")[0].replace("_", " ")
        if start_name in INDICES or symbol in INDICES:
            continue

    candidate_symbols.append(symbol)

# compute crossover info once, using the cached price_data
if st.session_state.screener_active:
    if st.session_state.scr_fast_ma == "Any" or st.session_state.scr_slow_ma == "Any":
        crossover_data = calculate_crossovers(
            tuple(candidate_symbols),
            timeframe,
            years,
            data_mtime,
            reports_mtime,
        )
    else:
        crossover_data = calculate_crossovers_with_pair(
            tuple(candidate_symbols),
            timeframe,
            int(st.session_state.scr_fast_ma),
            int(st.session_state.scr_slow_ma),
            st.session_state.scr_ma_type,
            data_mtime,
        )
else:
    crossover_data = calculate_crossovers(
        tuple(candidate_symbols),
        timeframe,
        years,
        data_mtime,
        reports_mtime,
    )

# ---------- BUILD SUMMARY TABLE ----------
rows = []
for raw_name in candidate_symbols:
    # raw_name is already in symbol form, convert back to lookup format for reports_cache
    best = reports_cache.get(raw_name.replace(".", "_"))
    if best is None:
        continue

    # Min Trades filter
    trades = int(best.get("Trades", 0)) if not pd.isna(best.get("Trades", np.nan)) else 0
    if trades < min_trades:
        continue

    cross_data = crossover_data.get(raw_name)

    # Clean symbol label, add * if F&O
    base_symbol_clean = raw_name.split(".")[0]
    display_symbol = f"{raw_name} *" if base_symbol_clean in FNO_STOCKS else raw_name

    win_rate = best.get("WinRate", np.nan)
    if pd.notna(win_rate) and win_rate > 1:
        win_rate /= 100
    if pd.notna(win_rate):
        wins = int(round(win_rate * trades))
        win_rate_str = f"{round(win_rate * 100, 1)}% ({wins}/{trades})"
    else:
        win_rate_str = "-"

    row = {
        "Symbol": display_symbol,
        "Return (%)": round(best["Return"], 2),
        "Win Rate (%)": win_rate_str,
        "Sharpe": round(best["Sharpe"], 2),
        "Sigma (Market)": best.get("MarketSigma", best.get("volatility", np.nan)),
        "RawSymbol": raw_name,
        "RawWinRate": win_rate if pd.notna(win_rate) else -1,
    }

    if cross_data is not None:
        cross_date = cross_data["Recent Bullish Crossover"]
        if cross_date.year < 1900:
             row["Crossover Date"] = "-"
        else:
             row["Crossover Date"] = str(cross_date.date())
        row["Crossover Type"] = cross_data["Crossover Type"]
        row["Crossover MA Type"] = cross_data["Crossover MA Type"]
        row["Crossover MA Pair"] = cross_data["Crossover MA Pair"]
        row["Recent Bullish Crossover"] = cross_date
        # Update Trades count to reflect only last 3 months
        row["Recent 3M Trades"] = cross_data["Recent 3M Trades"]
    else:
        row["Crossover Date"] = "-"
        row["Crossover Type"] = "-"
        row["Crossover MA Type"] = "-"
        row["Crossover MA Pair"] = "-"
        row["Recent Bullish Crossover"] = pd.Timestamp.min
        row["Recent 3M Trades"] = 0

    rows.append(row)

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
    table_title += f"(Filtered: {st.session_state.scr_ma_type if st.session_state.scr_ma_type != 'Both' else 'Any MA Type'} {st.session_state.scr_fast_ma}/{st.session_state.scr_slow_ma})"
else:
    table_title += "(All MA Pairs)"

# Override the stock-summary table with an event-driven screener dataset.
def symbol_matches_event_universe(symbol: str) -> bool:
    if universe == "Nifty 50":
        return symbol.split(".")[0] in NIFTY_50_SYMBOLS
    if universe == "F&O":
        return symbol.split(".")[0] in FNO_STOCKS
    if universe == "Indices":
        norm_symbol = symbol.replace(".", " ").replace("_", " ")
        return any(idx == norm_symbol or idx == symbol for idx in INDICES)
    if universe == "All NSE":
        return symbol not in INDICES and symbol.replace(".", " ") not in INDICES

    start_name = symbol.split(".")[0].replace("_", " ")
    return start_name not in INDICES and symbol not in INDICES


event_universe_symbols = sorted(
    [symbol for symbol in price_data.keys() if symbol_matches_event_universe(symbol)]
)
latest_screen_date = max(
    (df["Date"].max() for symbol, df in price_data.items() if symbol in event_universe_symbols and not df.empty),
    default=pd.Timestamp.today().normalize(),
)
latest_screen_date = pd.Timestamp(latest_screen_date).normalize()

if st.session_state.scr_start_date is None:
    # Default: show last 7 calendar days so recent crossovers are visible without manual date picking
    default_start = (latest_screen_date - pd.DateOffset(days=6)).normalize()
    st.session_state.scr_start_date = default_start.date().isoformat()
if st.session_state.scr_end_date is None:
    st.session_state.scr_end_date = latest_screen_date.date().isoformat()

st.subheader("🎯 Refine Screener")
# Screener controls are rendered closer to the event table below.
with st.form("screener_form_v2"):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        scr_date_range_input = st.date_input(
            "Crossover Date Range",
            value=(
                pd.to_datetime(st.session_state.scr_start_date).date(),
                pd.to_datetime(st.session_state.scr_end_date).date(),
            ),
            max_value=latest_screen_date.date(),
        )
    with c2:
        scr_signal_input = st.selectbox(
            "Signal",
            ["Both", "Bullish", "Bearish"],
            index=["Both", "Bullish", "Bearish"].index(st.session_state.scr_signal),
        )
    with c3:
        scr_ma_type_input = st.selectbox(
            "MA Type",
            ["Both", "SMA", "EMA"],
            index=["Both", "SMA", "EMA"].index(st.session_state.scr_ma_type),
        )
    with c4:
        scr_fast_ma_input = st.selectbox(
            "Fast MA",
            options=["Any", 5, 10, 12, 20, 50],
            index=["Any", 5, 10, 12, 20, 50].index(st.session_state.scr_fast_ma),
        )
    with c5:
        scr_slow_ma_input = st.selectbox(
            "Slow MA",
            options=["Any", 20, 26, 50, 100, 200],
            index=["Any", 20, 26, 50, 100, 200].index(st.session_state.scr_slow_ma),
        )

    colA, colB, _ = st.columns([1, 1.2, 3])
    with colA:
        refine_btn = st.form_submit_button("Refine Screener")
    with colB:
        refresh_btn = st.form_submit_button("Reset Filters")

if refine_btn:
    if (
        not isinstance(scr_date_range_input, tuple)
        or len(scr_date_range_input) != 2
        or scr_date_range_input[0] is None
        or scr_date_range_input[1] is None
    ):
        st.warning("Select a start and end date for the screener range.")
    else:
        scr_start_date_input, scr_end_date_input = scr_date_range_input
        selected_days = (scr_end_date_input - scr_start_date_input).days + 1
        if selected_days < 1 or selected_days > 30:
            st.warning("Date range must be between 1 and 30 days.")
        elif (
            scr_fast_ma_input != "Any"
            and scr_slow_ma_input != "Any"
            and scr_fast_ma_input >= scr_slow_ma_input
        ):
            st.warning("Fast MA must be smaller than Slow MA to filter properly.")
        else:
            st.session_state.screener_active = True
            st.session_state.scr_start_date = pd.to_datetime(scr_start_date_input).date().isoformat()
            st.session_state.scr_end_date = pd.to_datetime(scr_end_date_input).date().isoformat()
            st.session_state.scr_signal = scr_signal_input
            st.session_state.scr_ma_type = scr_ma_type_input
            st.session_state.scr_fast_ma = scr_fast_ma_input
            st.session_state.scr_slow_ma = scr_slow_ma_input
            st.rerun()

if refresh_btn:
    st.session_state.screener_active = False
    # Reset to default 7-day window
    default_start = (latest_screen_date - pd.DateOffset(days=6)).normalize()
    st.session_state.scr_start_date = default_start.date().isoformat()
    st.session_state.scr_end_date = latest_screen_date.date().isoformat()
    st.session_state.scr_signal = "Both"
    st.session_state.scr_ma_type = "Both"
    st.session_state.scr_fast_ma = "Any"
    st.session_state.scr_slow_ma = "Any"
    st.rerun()

active_scr_start_date = (
    st.session_state.scr_start_date
    if st.session_state.screener_active and st.session_state.scr_start_date
    else latest_screen_date.date().isoformat()
)
active_scr_end_date = (
    st.session_state.scr_end_date
    if st.session_state.screener_active and st.session_state.scr_end_date
    else latest_screen_date.date().isoformat()
)
active_scr_signal = st.session_state.scr_signal if st.session_state.screener_active else "Both"
active_scr_ma_type = st.session_state.scr_ma_type if st.session_state.screener_active else "Both"
active_scr_fast = (
    None if not st.session_state.screener_active or st.session_state.scr_fast_ma == "Any"
    else int(st.session_state.scr_fast_ma)
)
active_scr_slow = (
    None if not st.session_state.screener_active or st.session_state.scr_slow_ma == "Any"
    else int(st.session_state.scr_slow_ma)
)

report_by_symbol = {
    raw_name.replace("_", "."): best
    for raw_name, best in reports_cache.items()
}

with st.spinner("Calculating crossover events..."):
    crossover_events = build_crossover_event_rows(
        tuple(event_universe_symbols),
        timeframe,
        active_scr_start_date,
        active_scr_end_date,
        active_scr_ma_type,
        active_scr_fast,
        active_scr_slow,
        active_scr_signal,
        data_mtime,
    )

event_rows = []
for cross_data in crossover_events:
    symbol = cross_data["RawSymbol"]
    best = report_by_symbol.get(symbol)

    trades = None
    if best is not None and not pd.isna(best.get("Trades", np.nan)):
        trades = int(best["Trades"])
        if trades < min_trades:
            continue

    win_rate = best.get("WinRate", np.nan) if best is not None else np.nan
    if pd.notna(win_rate) and win_rate > 1:
        win_rate /= 100
    if pd.notna(win_rate) and trades is not None:
        wins = int(round(win_rate * trades))
        win_rate_str = f"{round(win_rate * 100, 1)}% ({wins}/{trades})"
    else:
        win_rate_str = "-"

    base_symbol_clean = symbol.split(".")[0]
    display_symbol = f"{symbol} *" if base_symbol_clean in FNO_STOCKS else symbol

    event_rows.append({
        "Symbol": display_symbol,
        "Crossover Date": cross_data["Crossover Date"],
        "Crossover Signal": cross_data["Crossover Signal"],
        "Crossover MA Type": cross_data["Crossover MA Type"],
        "Crossover MA Pair": cross_data["Crossover MA Pair"],
        "Recent Bullish Crossover": cross_data["Recent Bullish Crossover"],
        "Return (%)": round(best["Return"], 2) if best is not None else np.nan,
        "Win Rate (%)": win_rate_str,
        "Sharpe": round(best["Sharpe"], 2) if best is not None else np.nan,
        "Sigma (Market)": best.get("MarketSigma", best.get("volatility", np.nan)) if best is not None else np.nan,
    })

summary_df = pd.DataFrame(event_rows)
if not summary_df.empty:
    summary_df = summary_df.sort_values(
        by=["Recent Bullish Crossover", "Symbol", "Crossover MA Type", "Crossover MA Pair"],
        ascending=[False, True, True, True],
    )
    summary_df = summary_df.drop(columns=["Recent Bullish Crossover"])
    summary_df = summary_df.fillna(0).reset_index(drop=True)
else:
    summary_df = pd.DataFrame(
        columns=[
            "Symbol",
            "Crossover Date",
            "Crossover Signal",
            "Crossover MA Type",
            "Crossover MA Pair",
            "Return (%)",
            "Win Rate (%)",
            "Sharpe",
            "Sigma (Market)",
        ]
    )
    st.info("No crossover events found for the selected screener filters.")

table_title = "📊 Crossover Screener "
if st.session_state.screener_active:
    table_title += (
        f"({active_scr_start_date} to {active_scr_end_date} | {active_scr_signal} | "
        f"{active_scr_ma_type if active_scr_ma_type != 'Both' else 'Any MA Type'} | "
        f"{active_scr_fast if active_scr_fast is not None else 'Any'}/"
        f"{active_scr_slow if active_scr_slow is not None else 'Any'})"
    )
else:
    table_title += f"(All Crossovers On {active_scr_start_date})"

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
    strategy_mode = st.radio(
        "Mode",
        ["Match Screener", "Backtest Optimal", "Custom Scenario"],
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Match Screener: chart uses the same MA pair the table reported — crossovers will always match.\n"
            "Backtest Optimal: chart uses the historically best pair from the optimization report.\n"
            "Custom Scenario: pick any MA type and pair."
        ),
    )

    if strategy_mode == "Custom Scenario":
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            ma_type_viz = st.selectbox("MA Type", ["EMA", "SMA"], index=0)
        with sc2:
            fast_viz = st.selectbox("Fast MA", options=[5, 10, 12, 20, 50, 100], index=2)
        with sc3:
            # Include 200 so user can replicate TradingView EMA 50/200 setup
            slow_viz = st.selectbox("Slow MA", options=[20, 26, 50, 100, 200], index=4)

        if fast_viz >= slow_viz:
            st.warning("Fast MA must be smaller than Slow MA")
            st.stop()

    elif strategy_mode == "Backtest Optimal":
        # ---------------------------------------------------------------
        # BACKTEST OPTIMAL MODE: Read the MA pair from the BACKTEST REPORT
        # (reports_cache top row = best historical pair).
        # This pair is NOT guaranteed to match the screener's crossover — use
        # "Match Screener" if you want the chart and table to agree.
        # ---------------------------------------------------------------
        report_raw_name = selected_symbol.replace(".", "_")
        report_row = reports_cache.get(report_raw_name)

        fast_viz, slow_viz, ma_type_viz = 12, 26, "EMA"  # safe defaults

        if report_row is not None:
            try:
                rpt_ma_type = str(report_row.get("MA_Type", "EMA")).strip()
                rpt_ma_pair = str(report_row.get("MA_Pair", "12/26")).strip()
                rpt_fast, rpt_slow = map(int, rpt_ma_pair.split("/"))
                if rpt_fast < rpt_slow and rpt_ma_type in ("EMA", "SMA"):
                    fast_viz    = rpt_fast
                    slow_viz    = rpt_slow
                    ma_type_viz = rpt_ma_type
            except Exception:
                pass  # keep safe defaults

        st.caption(
            f"📊 Showing **{ma_type_viz} {fast_viz}/{slow_viz}** "
            f"(backtest-optimal pair from report — may differ from screener pair)."
        )

    else:
        # ---------------------------------------------------------------
        # MATCH SCREENER MODE (default): Use the EXACT same MA pair that
        # the screener table used to detect the crossover shown in the row.
        # This guarantees the chart arrow always aligns with the table date.
        # ---------------------------------------------------------------
        fast_viz, slow_viz, ma_type_viz = 12, 26, "EMA"  # safe defaults
        try:
            screener_pair = str(summary_df.iloc[selected_row_idx].get("Crossover MA Pair", "12/26"))
            screener_type = str(summary_df.iloc[selected_row_idx].get("Crossover MA Type", "EMA"))
            s_fast, s_slow = map(int, screener_pair.split("/"))
            if s_fast < s_slow and screener_type in ("EMA", "SMA"):
                fast_viz    = s_fast
                slow_viz    = s_slow
                ma_type_viz = screener_type
        except Exception:
            pass  # keep safe defaults

        st.caption(
            f"📊 Showing **{ma_type_viz} {fast_viz}/{slow_viz}** "
            f"(same pair as screener table — crossover markers will match the table date)."
        )

    
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
    df["MA_Fast"] = _compute_ma(df["Close"], fast_viz, ma_type_viz)
    df["MA_Slow"] = _compute_ma(df["Close"], slow_viz, ma_type_viz)

    df["Signal"], df["Crossover"] = compute_crossover_series(
        df["MA_Fast"], df["MA_Slow"]
    )

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
