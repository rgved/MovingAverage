
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.optimize_on_dynamic_noise import run_all_dynamic_trend_noise
from src.constants import INDICES

if __name__ == "__main__":
    print(f"Optimizing {len(INDICES)} indices...")
    run_all_dynamic_trend_noise(INDICES)
