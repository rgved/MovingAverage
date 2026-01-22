# src/trim_data.py
import pandas as pd
import os
from datetime import datetime, timedelta

# ---------- Paths ----------
input_dir = "data/processed"
output_dir = "data/trimmed"
os.makedirs(output_dir, exist_ok=True)

# ---------- Rolling 3-Month Window ----------
END_DATE = pd.Timestamp(datetime.today().date())
LOOKBACK_DAYS = 92   # ~3 months buffer
START_DATE = END_DATE - pd.Timedelta(days=LOOKBACK_DAYS)

print(
    f"📅 Trimming all datasets to rolling window:\n"
    f"    {START_DATE.date()} → {END_DATE.date()} (Last ~3 months)\n"
)

summary = []

# ---------- Trim Loop ----------
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        # Parse date safely and remove timezone
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
        df = df.sort_values("Date")

        # Filter rolling window
        df_trimmed = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]

        # Save trimmed data
        out_path = os.path.join(output_dir, file)
        df_trimmed.to_csv(out_path, index=False)

        row_count = len(df_trimmed)
        status = "✅" if row_count > 0 else "⚠️ EMPTY"
        print(f"{status} {file:<22} | {row_count:>4} rows retained")

        summary.append((file, row_count))

# ---------- Summary ----------
print("\n📊 Summary Report")
for f, count in summary:
    if count < 30:
        print(f"⚠️ {f:<22} — Only {count} rows (low trading days)")
    else:
        print(f"✅ {f:<22} — OK ({count} rows)")

print("\n🎯 Rolling 3-month datasets saved to: data/trimmed/")
