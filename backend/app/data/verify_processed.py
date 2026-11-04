
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "JPM", "GS",
    "BLK", "SPY", "QQQ", "IWM"
]


def main():

    for ticker in TICKERS:

        file_path = DATA_DIR / f"{ticker}.csv"

        print(f"\nChecking {ticker}...")

        df = pd.read_csv(file_path)

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Missing values:")
        print(df.isnull().sum())

        print("Duplicate dates:", df["Date"].duplicated().sum())

        print("First 5 rows:")
        print(df.head())

        assert len(df) > 0, f"{ticker} has no data"

        assert df["Date"].duplicated().sum() == 0, (
            f"{ticker} contains duplicate dates"
        )

        assert df["Close"].isnull().sum() == 0, (
            f"{ticker} has missing closing prices"
        )

        assert (df["Close"] > 0).all(), (
            f"{ticker} has invalid closing prices"
        )

    print("\nALL DATASETS VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
