
from pathlib import Path
import pandas as pd
import numpy as np


# Paths
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Assets
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "JPM", "GS",
    "BLK", "SPY", "QQQ", "IWM"
]

# Required market data columns
REQUIRED_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume"
]


def clean_data(df):
    """
    Clean historical stock data and calculate returns.
    """

    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove rows with invalid dates
    df = df.dropna(subset=["Date"])

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["Date"], keep="first")

    # Sort chronologically
    df = df.sort_values("Date")

    # Convert market data columns to numeric
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows with missing or invalid closing prices
    df = df.dropna(subset=["Close"])

    # Remove rows with non-positive closing prices
    df = df[df["Close"] > 0]

    # Remove invalid OHLC observations
    df = df.dropna(
        subset=["Open", "High", "Low"]
    )

    # Check price consistency
    df = df[
        (df["High"] >= df["Low"]) &
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close"]) &
        (df["Low"] <= df["Open"]) &
        (df["Low"] <= df["Close"])
    ]

    # Calculate simple daily returns
    df["Return"] = df["Close"].pct_change(
        fill_method=None
    )

    # Calculate log returns
    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # Calculate cumulative returns
    df["Cumulative_Return"] = (
        (1 + df["Return"]).cumprod() - 1
    )

    # Remove infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main():

    # Create processed directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []

    for ticker in TICKERS:

        print(f"\nProcessing {ticker}...")

        raw_file = RAW_DIR / f"{ticker}.csv"

        if not raw_file.exists():
            print(f"Raw file missing: {ticker}")
            continue

        # Load raw data
        df = pd.read_csv(raw_file)

        raw_rows = len(df)

        # Clean and calculate returns
        df = clean_data(df)

        # Save processed data
        output_file = PROCESSED_DIR / f"{ticker}.csv"

        df.to_csv(
            output_file,
            index=False
        )

        # Store data quality summary
        summary.append({
            "Ticker": ticker,
            "Raw_Rows": raw_rows,
            "Processed_Rows": len(df),
            "Missing_Values": int(
                df.isnull().sum().sum()
            ),
            "Duplicate_Dates": int(
                df["Date"].duplicated().sum()
            ),
            "Output_File": str(output_file)
        })

        print(f"Raw rows: {raw_rows}")
        print(f"Processed rows: {len(df)}")
        print(f"Saved: {output_file}")

    # Save summary
    summary_df = pd.DataFrame(summary)

    summary_file = PROCESSED_DIR / "data_quality_summary.csv"

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n========== PROCESSING COMPLETE ==========")
    print(summary_df)
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()
