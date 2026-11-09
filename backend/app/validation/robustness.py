
from pathlib import Path

import numpy as np

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

OUTPUT_DIR = (
    DATA_DIR /
    "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def bootstrap_robustness(
    returns,
    n_bootstrap=2000,
    block_size=5,
    seed=42
):

    returns = np.asarray(
        returns,
        dtype=float
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) < 10:

        raise ValueError(
            "At least 10 returns are required"
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(returns)

    bootstrap_means = []

    bootstrap_sharpes = []

    for _ in range(
        n_bootstrap
    ):

        sampled = []

        while len(sampled) < n:

            start = rng.integers(
                0,
                n
            )

            indices = (
                start +
                np.arange(
                    block_size
                )
            ) % n

            sampled.extend(
                returns[indices]
            )

        sampled = np.asarray(
            sampled[:n]
        )

        mean_return = (
            sampled.mean()
        )

        std_return = (
            sampled.std(ddof=1)
        )

        sharpe = (

            mean_return /
            std_return *
            np.sqrt(252)

            if std_return > 0

            else np.nan

        )

        bootstrap_means.append(
            mean_return
        )

        bootstrap_sharpes.append(
            sharpe
        )

    bootstrap_means = np.asarray(
        bootstrap_means
    )

    bootstrap_sharpes = np.asarray(
        bootstrap_sharpes
    )

    results = {

        "Mean_Return": returns.mean(),

        "Mean_Return_Lower_CI": np.quantile(
            bootstrap_means,
            0.025
        ),

        "Mean_Return_Upper_CI": np.quantile(
            bootstrap_means,
            0.975
        ),

        "Median_Return": np.median(
            bootstrap_means
        ),

        "Probability_Positive_Mean": np.mean(
            bootstrap_means > 0
        ),

        "Sharpe_Lower_CI": np.nanquantile(
            bootstrap_sharpes,
            0.025
        ),

        "Sharpe_Upper_CI": np.nanquantile(
            bootstrap_sharpes,
            0.975
        )

    }

    return results



def run_robustness(
    ticker="AAPL",
    strategy="momentum"
):

    ticker = ticker.upper()

    backtest_file = (
        DATA_DIR /
        "backtests" /
        f"{ticker}_{strategy}_backtest.csv"
    )

    if not backtest_file.exists():

        raise FileNotFoundError(
            f"Backtest file not found: {backtest_file}\n"
            "Run the backtest first."
        )

    df = pd.read_csv(
        backtest_file
    )

    returns = df[
        "Net_Return"
    ].to_numpy()

    results = bootstrap_robustness(
        returns
    )

    print(
        "\nROBUSTNESS RESULTS"
    )

    print(
        results
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_{strategy}_robustness.csv"
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved:",
        output_file
    )


if __name__ == "__main__":

    run_robustness(
        "AAPL"
    )
