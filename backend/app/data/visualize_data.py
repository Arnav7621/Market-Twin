
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "plots"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def visualize_stock(ticker):

    file_path = DATA_DIR / f"{ticker}.csv"

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    # Plot closing price
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        df["Close"]
    )

    plt.title(f"{ticker} Historical Closing Price")

    plt.xlabel("Date")

    plt.ylabel("Adjusted Closing Price")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{ticker}_price.png"
    )

    plt.show()

    plt.close()

    # Plot cumulative returns
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        df["Cumulative_Return"]
    )

    plt.title(f"{ticker} Cumulative Returns")

    plt.xlabel("Date")

    plt.ylabel("Cumulative Return")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{ticker}_cumulative_return.png"
    )

    plt.show()

    plt.close()

    # Plot daily return distribution
    plt.figure(figsize=(10, 5))

    df["Return"].dropna().hist(bins=50)

    plt.title(f"{ticker} Daily Return Distribution")

    plt.xlabel("Daily Return")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{ticker}_return_distribution.png"
    )

    plt.show()

    plt.close()

    print(f"Plots created for {ticker}")


if __name__ == "__main__":

    visualize_stock("AAPL")
