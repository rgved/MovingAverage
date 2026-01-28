# src/trim_data.py
import pandas as pd
import os
from datetime import datetime

# ---------- PATH SETUP ----------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                # project root

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "trimmed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Rolling 3-Month Window ----------
END_DATE = pd.Timestamp(datetime.today().date())
LOOKBACK_DAYS = 92   # ~3 months
START_DATE = END_DATE - pd.Timedelta(days=LOOKBACK_DAYS)

print(
    f"📅 Trimming all datasets to rolling window:\n"
    f"    {START_DATE.date()} → {END_DATE.date()} (Last ~3 months)\n"
)

summary = []

# ---------- Trim Loop ----------
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        file_path = os.path.join(INPUT_DIR, file)
        df = pd.read_csv(file_path)

        # Parse date safely and remove timezone if present
        df["Date"] = pd.to_datetime(
            df["Date"], utc=True, errors="coerce"
        ).dt.tz_convert(None)

        df = df.sort_values("Date")

        # Filter rolling window
        df_trimmed = df[
            (df["Date"] >= START_DATE) &
            (df["Date"] <= END_DATE)
        ]

        out_path = os.path.join(OUTPUT_DIR, file)
        df_trimmed.to_csv(out_path, index=False)

        row_count = len(df_trimmed)
        status = "✅" if row_count > 0 else "⚠️ EMPTY"
        print(f"{status} {file:<25} | {row_count:>4} rows retained")

        summary.append((file, row_count))

# ---------- Summary ----------
print("\n📊 Summary Report")
for f, count in summary:
    if count < 30:
        print(f"⚠️ {f:<25} — Only {count} rows")
    else:
        print(f"✅ {f:<25} — OK ({count} rows)")

print(f"\n🎯 Rolling 3-month datasets saved to:\n{OUTPUT_DIR}")
