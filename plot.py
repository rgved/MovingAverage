import pandas as pd
import matplotlib.pyplot as plt

def plot_processed_csv(filepath):
    # Load file
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # Make sure moving averages exist
    if not {"MA_Fast", "MA_Slow", "Crossover"}.issubset(df.columns):
        print("⚠️ File does not contain MA columns. Run Phase 2 first.")
        return

    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Close"], label="Close Price", linewidth=1.2)
    plt.plot(df["Date"], df["MA_Fast"], label="Fast MA", linewidth=1.1)
    plt.plot(df["Date"], df["MA_Slow"], label="Slow MA", linewidth=1.1)

    # Bullish crossovers (+2) → BUY signals
    buys = df[df["Crossover"] == 2]
    plt.scatter(buys["Date"], buys["Close"], marker="^", color="green", label="Buy Signal", s=100, zorder=5)

    # Bearish crossovers (–2) → SELL signals
    sells = df[df["Crossover"] == -2]
    plt.scatter(sells["Date"], sells["Close"], marker="v", color="red", label="Sell Signal", s=100, zorder=5)

    plt.title(f"Moving Average Crossover Signals – {filepath.split('/')[-1]}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_processed_csv("data/processed/ADANIENT.NS.csv")
