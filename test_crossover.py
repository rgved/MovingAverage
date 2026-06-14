import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dashboard.app import compute_crossover_series, _compute_ma

# Create dummy data
df = pd.DataFrame({
    'Close': [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98, 100, 102, 104, 106, 108, 110]
})

ma_fast = _compute_ma(df['Close'], 3, 'SMA')
ma_slow = _compute_ma(df['Close'], 5, 'SMA')

df['MA_Fast'] = ma_fast
df['MA_Slow'] = ma_slow

signal, crossover = compute_crossover_series(ma_fast, ma_slow)
df['Signal'] = signal
df['Crossover'] = crossover

print(df)
