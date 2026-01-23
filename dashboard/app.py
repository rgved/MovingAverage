# dashboard/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reports_dir = os.path.join(BASE_DIR, "reports")
data_dir = os.path.join(BASE_DIR, "src","data", "trimmed")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Adaptive MA Strategy Dashboard", layout="wide")
st.title("Adaptive Moving Average Strategy Dashboard")

if not os.path.exists(reports_dir):
    st.error(f"Reports directory not found: {reports_dir}")
    st.stop()

# ---------- BUILD SUMMARY TABLE ----------
rows = []

for file in os.listdir(reports_dir):
    if file.endswith("_dynamic_trend_noise_optimization.csv"):
        symbol = file.replace("_dynamic_trend_noise_optimization.csv", "").replace("_", ".")
        rep = pd.read_csv(os.path.join(reports_dir, file))
        best = rep.iloc[0]

        rows.append({
            "Symbol": symbol,
            "MA Type": best["MA_Type"],
            "MA Pair": best["MA_Pair"],
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

# ---------- STRATEGY FILTERS ----------
st.markdown("### Strategy Filters")

col1, col2, col3 = st.columns(3)

with col1:
    ma_type_filter = st.selectbox("MA Type", ["EMA", "SMA"])

with col2:
    fast_ma = st.selectbox("Fast MA (MA1)", list(range(5, 51, 5)))

with col3:
    slow_ma = st.selectbox("Slow MA (MA2)", list(range(10, 101, 10)))

# ---------- VALIDATION ----------
if fast_ma >= slow_ma:
    st.warning("Fast MA must be smaller than Slow MA")
    st.stop()

selected_pair = f"{fast_ma}/{slow_ma}"

# ---------- FILTER DATA ----------
filtered_df = summary_df[
    (summary_df["MA Type"] == ma_type_filter) &
    (summary_df["MA Pair"] == selected_pair)
].reset_index(drop=True)

if filtered_df.empty:
    st.warning(
        f"No stocks found for {ma_type_filter} {selected_pair}. "
        "Try a different MA combination."
    )

st.subheader("Stock Performance Summary (Click a row)")

# ---------- AGGRID CONFIG ----------
gb = GridOptionsBuilder.from_dataframe(filtered_df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_grid_options(domLayout="normal")

grid_response = AgGrid(
    filtered_df,
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
    clicked_symbol = selected_rows.iloc[0]["Symbol"]

    st.markdown("---")
    st.subheader("📈 Stock Visualization")

    selected_symbol = st.selectbox(
        "Selected stock",
        filtered_df["Symbol"].tolist(),
        index=filtered_df["Symbol"].tolist().index(clicked_symbol)
    )

    price_file = os.path.join(data_dir, f"{selected_symbol}.csv")
    report_file = os.path.join(
        reports_dir,
        f"{selected_symbol.replace('.', '_')}_dynamic_trend_noise_optimization.csv"
    )

    if not os.path.exists(price_file) or not os.path.exists(report_file):
        st.error("Required data files not found for this stock.")
        st.stop()

    # ---------- LOAD PRICE DATA ----------
    df = pd.read_csv(price_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values("Date")

    # ---------- MOVING AVERAGES ----------
    if ma_type_filter == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast_ma, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow_ma, adjust=False).mean()
    else:
        df["MA_Fast"] = df["Close"].rolling(fast_ma).mean()
        df["MA_Slow"] = df["Close"].rolling(slow_ma).mean()

    df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
    df["Crossover"] = df["Signal"].diff()

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["Date"], df["Close"], label="Close", color="gray", alpha=0.6)
    ax.plot(df["Date"], df["MA_Fast"], label=f"{ma_type_filter} {fast_ma}", color="green")
    ax.plot(df["Date"], df["MA_Slow"], label=f"{ma_type_filter} {slow_ma}", color="orange")

    buys = df[df["Crossover"] == 2]
    sells = df[df["Crossover"] == -2]

    ax.scatter(buys["Date"], buys["Close"], marker="^", color="lime", s=80, label="Buy")
    ax.scatter(sells["Date"], sells["Close"], marker="v", color="red", s=80, label="Sell")

    ax.set_title(f"{selected_symbol} | {ma_type_filter} ({fast_ma}/{slow_ma})")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

else:
    st.info("⬆ Select a stock from the table to view its chart")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 Adaptive Finance | Professional Trading Dashboard")
