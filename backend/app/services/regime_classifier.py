
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "regimes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_regimes(ticker):

    df = pd.read_csv(DATA_DIR / f"{ticker}.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    # Rolling volatility over 20 trading sessions
    df["Rolling_Volatility"] = (
        df["Return"].rolling(20).std() * np.sqrt(252)
    )

    # Rolling average volume
    df["Rolling_Volume"] = (
        df["Volume"].rolling(20).mean()
    )

    # Rolling mean return
    df["Rolling_Return"] = (
        df["Return"].rolling(20).mean()
    )

    # Expanding historical median thresholds
    volatility_threshold = (
        df["Rolling_Volatility"].expanding().median()
    )

    volume_threshold = (
        df["Rolling_Volume"].expanding().median()
    )

    # Volatility regime
    df["Volatility_Regime"] = np.where(
        df["Rolling_Volatility"] > volatility_threshold,
        "High Volatility",
        "Low Volatility"
    )

    # Volume proxy for liquidity regime
    df["Liquidity_Regime"] = np.where(
        df["Rolling_Volume"] > volume_threshold,
        "High Volume",
        "Low Volume"
    )

    # Return trend regime
    df["Trend_Regime"] = np.where(
        df["Rolling_Return"] > 0,
        "Positive Trend",
        "Negative Trend"
    )

    df["Market_Regime"] = (
        df["Volatility_Regime"] + " | " +
        df["Liquidity_Regime"] + " | " +
        df["Trend_Regime"]
    )

    # Save
    output_file = OUTPUT_DIR / f"{ticker}_regimes.csv"

    df.to_csv(output_file, index=False)

    print(f"Saved regime data: {output_file}")

    return df


if __name__ == "__main__":

    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN",
        "NVDA", "META", "JPM", "GS",
        "BLK", "SPY", "QQQ", "IWM"
    ]

    for ticker in tickers:

        print(f"\nClassifying {ticker}")

        classify_regimes(ticker)

    print("\nALL REGIMES GENERATED!")
