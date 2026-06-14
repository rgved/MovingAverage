"""
Diagnose why the screener shows 50/200 SMA crossover on June 11th
instead of June 4th/5th.

Run from the MovingAverage directory:
    python diagnose_crossover_date.py [SYMBOL]

If no symbol given, it will scan all processed files and report any that
have a crossover on or near June 11th for SMA 50/200.
"""
import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ---- Exact same logic as app.py ----
def _compute_ma(close, span, ma_type):
    if ma_type == "EMA":
        import math
        alpha = 2.0 / (span + 1)
        k = math.ceil(math.log(0.01) / math.log(1.0 - alpha))
        min_p = max(span, min(k, span * 3))
        return close.ewm(span=span, min_periods=min_p, adjust=False).mean()
    else:
        return close.rolling(span, min_periods=span).mean()


def compute_crossover_series(ma_fast, ma_slow):
    both_valid = ma_fast.notna() & ma_slow.notna()
    prev_valid  = both_valid.shift(1).fillna(False)

    signal = pd.Series(np.nan, index=ma_fast.index, dtype=float)
    signal.loc[both_valid] = np.where(
        ma_fast[both_valid] > ma_slow[both_valid], 1.0, -1.0
    )

    diff      = (ma_fast - ma_slow).astype(float)
    prev_diff = diff.shift(1)

    crossover = pd.Series(0.0, index=ma_fast.index)
    bull = both_valid & prev_valid & (prev_diff < 0) & (diff > 0)
    bear = both_valid & prev_valid & (prev_diff > 0) & (diff < 0)
    crossover[bull] =  2.0
    crossover[bear] = -2.0
    return signal, crossover


def load_df(filepath):
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath, engine="pyarrow")
    else:
        df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def analyse_symbol(symbol, df, fast=50, slow=200, ma_type="SMA"):
    sma_f = _compute_ma(df["Close"], fast, ma_type)
    sma_s = _compute_ma(df["Close"], slow, ma_type)
    _, crossover = compute_crossover_series(sma_f, sma_s)

    df = df.copy()
    df["MA_Fast"]   = sma_f
    df["MA_Slow"]   = sma_s
    df["Crossover"] = crossover.values

    crosses = df[df["Crossover"].abs() == 2].copy()
    if crosses.empty:
        return None, df

    # Print ALL crossovers found
    print(f"\n{'='*60}")
    print(f"  {symbol}  |  {ma_type} {fast}/{slow}")
    print(f"{'='*60}")
    for _, row in crosses.iterrows():
        ctype = "BUY " if row["Crossover"] == 2 else "SELL"
        print(f"  [{ctype}] {row['Date'].date()}  "
              f"Close={row['Close']:.2f}  "
              f"MA{fast}={row['MA_Fast']:.4f}  "
              f"MA{slow}={row['MA_Slow']:.4f}  "
              f"diff={row['MA_Fast']-row['MA_Slow']:.4f}")

    # Print the 5 rows around the most recent crossover to spot off-by-one issues
    last_cross_idx = crosses.index[-1]
    window = df.loc[max(0, last_cross_idx-3): last_cross_idx+3]
    print(f"\n  Context rows around most-recent crossover:")
    print(f"  {'Date':<12} {'Close':>8} {'MA'+str(fast):>10} {'MA'+str(slow):>10} {'diff':>10} {'Cross':>6}")
    for _, r in window.iterrows():
        cstr = " <-- CROSS" if r["Crossover"] != 0 else ""
        print(f"  {str(r['Date'].date()):<12} {r['Close']:>8.2f} "
              f"{r['MA_Fast']:>10.4f} {r['MA_Slow']:>10.4f} "
              f"{r['MA_Fast']-r['MA_Slow']:>10.4f} {cstr}")

    last = crosses.iloc[-1]
    return last["Date"].date(), df


def scan_all(fast=50, slow=200, ma_type="SMA", target_month=6, target_year=2026):
    """Scan all files and print those that have their most-recent crossover in June 2026."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") or f.endswith(".csv")]
    hits = []
    for f in sorted(files):
        sym = f.rsplit(".", 1)[0]
        try:
            df = load_df(os.path.join(DATA_DIR, f))
            sma_f = _compute_ma(df["Close"], fast, ma_type)
            sma_s = _compute_ma(df["Close"], slow, ma_type)
            _, crossover = compute_crossover_series(sma_f, sma_s)
            df["Crossover"] = crossover.values
            crosses = df[df["Crossover"].abs() == 2]
            if crosses.empty:
                continue
            last_date = crosses.iloc[-1]["Date"].date()
            if last_date.month == target_month and last_date.year == target_year:
                hits.append((sym, last_date, "BUY" if crosses.iloc[-1]["Crossover"] == 2 else "SELL"))
        except Exception as e:
            print(f"  ERROR {sym}: {e}")

    print(f"\n{'='*60}")
    print(f"  Stocks with {ma_type} {fast}/{slow} crossover in {target_month}/{target_year}")
    print(f"{'='*60}")
    for sym, dt, ctype in sorted(hits, key=lambda x: x[1]):
        print(f"  {sym:<30} {dt}  [{ctype}]")
    return hits


# ─────────────────────────── MAIN ───────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol_arg = sys.argv[1]
        # Try to find the file
        candidates = [
            os.path.join(DATA_DIR, f"{symbol_arg}.parquet"),
            os.path.join(DATA_DIR, f"{symbol_arg}.csv"),
            os.path.join(DATA_DIR, f"{symbol_arg}.NS.parquet"),
            os.path.join(DATA_DIR, f"{symbol_arg}.NS.csv"),
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if not found:
            print(f"File not found for symbol '{symbol_arg}'. Files in {DATA_DIR}:")
            print([f for f in os.listdir(DATA_DIR) if symbol_arg.upper() in f.upper()][:10])
            sys.exit(1)

        df = load_df(found)
        print(f"\nData range: {df['Date'].min().date()} -> {df['Date'].max().date()} ({len(df)} rows)")
        print(f"Latest 5 trading days:")
        print(df[["Date","Close"]].tail(5).to_string(index=False))

        # Test SMA and EMA
        for mt in ["SMA", "EMA"]:
            analyse_symbol(symbol_arg, df, 50, 200, mt)

    else:
        print("No symbol given — scanning ALL processed files for June 2026 crossovers...")
        print("(This may take a minute)\n")
        hits = scan_all(50, 200, "SMA", 6, 2026)
        if not hits:
            print("  (none found)")
        print()
        hits2 = scan_all(50, 200, "EMA", 6, 2026)
        if not hits2:
            print("  (none found)")

        print("\n\nNow checking specifically for crossovers on June 11, 2026...")
        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") or f.endswith(".csv")]
        june11 = []
        for f in sorted(all_files):
            sym = f.rsplit(".", 1)[0]
            try:
                df = load_df(os.path.join(DATA_DIR, f))
                for mt in ["SMA", "EMA"]:
                    sma_f = _compute_ma(df["Close"], 50, mt)
                    sma_s = _compute_ma(df["Close"], 200, mt)
                    _, crossover = compute_crossover_series(sma_f, sma_s)
                    df2 = df.copy()
                    df2["Crossover"] = crossover.values
                    crosses = df2[df2["Crossover"].abs() == 2]
                    if not crosses.empty:
                        last_date = crosses.iloc[-1]["Date"].date()
                        import datetime
                        if last_date == datetime.date(2026, 6, 11):
                            ctype = "BUY" if crosses.iloc[-1]["Crossover"] == 2 else "SELL"
                            june11.append((sym, mt, ctype))
            except Exception:
                pass

        print(f"\nStocks with most-recent {50}/{200} crossover exactly on June 11, 2026:")
        for sym, mt, ctype in june11:
            print(f"  {sym:<30} {mt}  [{ctype}]")
        if not june11:
            print("  (none)")
