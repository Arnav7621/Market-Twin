
from pathlib import Path
import pandas as pd

from risk_metrics import calculate_all_risk_metrics


# MarketTwin project paths
PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_FILE = DATA_DIR / "risk_metrics.csv"


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "META", "JPM", "GS",
    "BLK", "SPY", "QQQ", "IWM"
]


def main():

    results = []

    print("\n========== MARKETTWIN RISK ANALYSIS ==========\n")

    for ticker in TICKERS:

        print(f"Analyzing {ticker}...")

        file_path = DATA_DIR / f"{ticker}.csv"

        if not file_path.exists():
            print(f"File missing: {ticker}")
            continue

        # Load processed stock data
        df = pd.read_csv(file_path)

        # Extract daily returns
        returns = pd.to_numeric(
            df["Return"],
            errors="coerce"
        )

        # Calculate risk metrics
        metrics = calculate_all_risk_metrics(
            returns
        )

        # Add asset information
        metrics["Ticker"] = ticker

        metrics["Observations"] = int(
            returns.dropna().shape[0]
        )

        metrics["Start_Date"] = df["Date"].min()

        metrics["End_Date"] = df["Date"].max()

        # Store results
        results.append(metrics)

        print("Completed:", ticker)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Reorder columns
    columns = [
        "Ticker",
        "Start_Date",
        "End_Date",
        "Observations",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Sortino_Ratio",
        "Maximum_Drawdown",
        "VaR_95",
        "CVaR_95"
    ]

    results_df = results_df[columns]

    # Save results
    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== FINAL RISK REPORT ==========\n")

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nRisk report saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
