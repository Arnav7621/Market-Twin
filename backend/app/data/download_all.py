
from pathlib import Path

from market_data import download_stock_data


# Project data directory
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# Initial MarketTwin asset universe
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "JPM",
    "GS",
    "BLK",
    "SPY",
    "QQQ",
    "IWM"
]

# Historical data range
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for ticker in TICKERS:
        try:
            df = download_stock_data(
                ticker=ticker,
                start=START_DATE,
                end=END_DATE
            )

            # Save the data as a CSV file
            file_path = DATA_DIR / f"{ticker}.csv"

            df.to_csv(file_path)

            successful.append(ticker)

            print(f"Saved: {file_path}")

        except Exception as error:
            failed.append(ticker)

            print(f"Failed: {ticker}")
            print(f"Error: {error}")

    print("\n========== DOWNLOAD SUMMARY ==========")

    print(f"Successful downloads: {len(successful)}")
    print(f"Failed downloads: {len(failed)}")

    print("\nSuccessful tickers:")
    print(successful)

    print("\nFailed tickers:")
    print(failed)


if __name__ == "__main__":
    main()
