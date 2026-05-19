"""
convert_to_parquet.py
─────────────────────
One-time migration script: converts all existing CSV files in
data/processed/ and data/trimmed/ to Parquet format (pyarrow engine).

Original CSVs are moved to archive_csv/{subdir}/ for rollback safety.
Only delete the archive after the dashboard and pipeline are verified.

Usage:
    python src/convert_to_parquet.py
"""

import os
import shutil
import pandas as pd

# ── Path Setup ────────────────────────────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DIRS_TO_CONVERT = {
    "processed": os.path.join(PROJECT_ROOT, "data", "processed"),
    "trimmed":   os.path.join(PROJECT_ROOT, "data", "trimmed"),
}

ARCHIVE_ROOT = os.path.join(PROJECT_ROOT, "archive_csv")

# ── Conversion ────────────────────────────────────────────────────────────────
total_ok    = 0
total_skip  = 0
total_err   = 0

for label, dir_path in DIRS_TO_CONVERT.items():
    if not os.path.exists(dir_path):
        print(f"[SKIP] Directory not found: {dir_path}")
        continue

    archive_dir = os.path.join(ARCHIVE_ROOT, label)
    os.makedirs(archive_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
    print(f"\n[{label.upper()}]  {len(csv_files)} CSV files found -> {dir_path}")

    for file in csv_files:
        csv_path     = os.path.join(dir_path, file)
        parquet_name = file.replace(".csv", ".parquet")
        parquet_path = os.path.join(dir_path, parquet_name)
        archive_path = os.path.join(archive_dir, file)

        # Skip if parquet already exists (idempotent)
        if os.path.exists(parquet_path):
            print(f"  [SKIP] {parquet_name} already exists.")
            total_skip += 1
            continue

        try:
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path, index=False, engine="pyarrow")

            # Validate: re-read and compare row count
            df_check = pd.read_parquet(parquet_path, engine="pyarrow")
            if len(df_check) != len(df):
                raise ValueError(
                    f"Row count mismatch: CSV={len(df)}, Parquet={len(df_check)}"
                )

            # Move original CSV to archive (not delete)
            shutil.move(csv_path, archive_path)
            print(f"  [OK]   {file:<35} -> {parquet_name}  ({len(df)} rows, archived)")
            total_ok += 1

        except Exception as e:
            print(f"  [ERR]  {file}: {e}")
            # Remove partially written parquet if it exists
            if os.path.exists(parquet_path):
                os.remove(parquet_path)
            total_err += 1

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Converted : {total_ok}")
print(f"  Skipped   : {total_skip}  (parquet already existed)")
print(f"  Errors    : {total_err}")
print(f"  Archives  : {ARCHIVE_ROOT}")
print(f"{'='*55}")

if total_err == 0:
    print("\nMigration complete. Original CSVs archived.")
    print("   Run the dashboard and verify, then delete archive_csv/ when satisfied.")
else:
    print(f"\n{total_err} file(s) failed. Check errors above.")
    print("   Originals that failed were NOT archived -- they remain in place.")
