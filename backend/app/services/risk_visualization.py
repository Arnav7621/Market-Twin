
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

PLOT_DIR = DATA_DIR / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    df = pd.read_csv(
        DATA_DIR / "risk_metrics.csv"
    )

    # Annualized volatility chart
    plt.figure(figsize=(12, 6))

    plt.bar(
        df["Ticker"],
        df["Annualized_Volatility"]
    )

    plt.title("MarketTwin Annualized Volatility")

    plt.xlabel("Ticker")

    plt.ylabel("Annualized Volatility")

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "annualized_volatility.png"
    )

    plt.show()

    plt.close()

    # Maximum drawdown chart
    plt.figure(figsize=(12, 6))

    plt.bar(
        df["Ticker"],
        df["Maximum_Drawdown"]
    )

    plt.title("MarketTwin Maximum Drawdown")

    plt.xlabel("Ticker")

    plt.ylabel("Maximum Drawdown")

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "maximum_drawdown.png"
    )

    plt.show()

    plt.close()

    # Sharpe ratio chart
    plt.figure(figsize=(12, 6))

    plt.bar(
        df["Ticker"],
        df["Sharpe_Ratio"]
    )

    plt.title("MarketTwin Sharpe Ratio")

    plt.xlabel("Ticker")

    plt.ylabel("Sharpe Ratio")

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "sharpe_ratio.png"
    )

    plt.show()

    plt.close()

    print("Risk visualizations created successfully!")


if __name__ == "__main__":
    main()
