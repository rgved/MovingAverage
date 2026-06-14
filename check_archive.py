import pandas as pd
# Check archive data date range
df = pd.read_csv('archive_csv/processed/RELIANCE.NS.csv')
print(f"Archive RELIANCE rows: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Columns: {list(df.columns)}")
