
import yfinance as yf
import pandas as pd


def download_stock_data(ticker, start, end):
    """
    Download historical stock data from Yahoo Finance.

    Parameters:
        ticker: Stock ticker symbol
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format

    Returns:
        DataFrame containing historical OHLCV data
    """

    print(f"Downloading data for {ticker}...")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the required market data columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    df = df[required_columns]

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="first")]

    # Sort data chronologically
    df = df.sort_index()

    print(f"Successfully downloaded {len(df)} rows for {ticker}")

    return df
