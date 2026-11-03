
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "JPM", "GS",
    "BLK", "SPY", "QQQ", "IWM"
]


def main():
    print("\n========== DATA VERIFICATION ==========\n")

    for ticker in TICKERS:
        file_path = DATA_DIR / f"{ticker}.csv"

        if not file_path.exists():
            print(f"{ticker}: FILE MISSING")
            continue

        df = pd.read_csv(file_path)

        print(f"\nTicker: {ticker}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate dates: {df['Date'].duplicated().sum()}")

        if len(df) > 0:
            print("Status: DATA PRESENT")


if __name__ == "__main__":
    main()
