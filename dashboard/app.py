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
data_dir = os.path.join(BASE_DIR, "data", "trimmed")

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

# ---------- MA TYPE FILTER ----------
st.markdown("### Filter Stocks by MA Type")

ma_filter = st.selectbox(
    "Show stocks using",
    ["All", "EMA", "SMA"],
    index=0
)

if ma_filter != "All":
    filtered_df = summary_df[summary_df["MA Type"] == ma_filter].reset_index(drop=True)
else:
    filtered_df = summary_df.copy()

st.subheader("📊 Stock Performance Summary (Click a row)")

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
        "Selected stock (from table)",
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
    df = df[(df["Date"] >= "2025-08-01") & (df["Date"] <= "2025-11-07")]
    df = df.sort_values("Date")

    report = pd.read_csv(report_file)
    best = report.iloc[0]

    fast, slow = map(int, best["MA_Pair"].split("/"))
    ma_type = best["MA_Type"]

    # ---------- MOVING AVERAGES ----------
    if ma_type == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    else:
        df["MA_Fast"] = df["Close"].rolling(fast).mean()
        df["MA_Slow"] = df["Close"].rolling(slow).mean()

    df["Signal"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, -1)
    df["Crossover"] = df["Signal"].diff()

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["Date"], df["Close"], label="Close", color="gray", alpha=0.6)
    ax.plot(df["Date"], df["MA_Fast"], label=f"{ma_type} {fast}", color="green")
    ax.plot(df["Date"], df["MA_Slow"], label=f"{ma_type} {slow}", color="orange")

    buys = df[df["Crossover"] == 2]
    sells = df[df["Crossover"] == -2]

    ax.scatter(buys["Date"], buys["Close"], marker="^", color="lime", s=80, label="Buy")
    ax.scatter(sells["Date"], sells["Close"], marker="v", color="red", s=80, label="Sell")

    ax.set_title(f"{selected_symbol} | {ma_type} ({fast}/{slow})")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

else:
    st.info("⬆ Click a stock row in the table to view its chart")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 Adaptive Finance | Professional Trading Dashboard")
