
from pathlib import Path

import numpy as np
import pandas as pd

from performance_metrics import (
    calculate_sharpe,
    calculate_max_drawdown
)


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


def prepare_strategy_returns(
    ticker="AAPL"
):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Alpha_Factor"] = (
        df["Momentum_5D"] /
        (
            df["Volatility_20D"] + 1e-8
        )
    )

    # Signal must use only previous day's information
    df["Signal"] = np.sign(
        df["Alpha_Factor"].shift(1)
    )

    # Current day's return is earned using previous signal
    df["Strategy_Return"] = (
        df["Signal"] *
        df["Return"]
    )

    # Trading cost when the signal changes
    df["Turnover"] = (
        df["Signal"].diff().abs()
    )

    transaction_cost_rate = 0.001

    df["Strategy_Return"] -= (
        df["Turnover"] *
        transaction_cost_rate
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Strategy_Return",
            "Signal"
        ]
    )

    return df


def walk_forward_test(
    ticker="AAPL",
    train_fraction=0.60,
    test_fraction=0.10
):

    df = prepare_strategy_returns(
        ticker
    )

    n = len(df)

    train_size = int(
        n * train_fraction
    )

    test_size = int(
        n * test_fraction
    )

    results = []

    start = train_size

    fold = 1

    while (
        start + test_size <= n
    ):

        test_data = df.iloc[
            start:start + test_size
        ]

        returns = test_data[
            "Strategy_Return"
        ].to_numpy()

        sharpe = calculate_sharpe(
            returns
        )

        drawdown = calculate_max_drawdown(
            returns
        )

        total_return = (
            np.prod(
                1 + returns
            ) - 1
        )

        results.append({

            "Fold": fold,

            "Train_End": df["Date"].iloc[
                start - 1
            ],

            "Test_Start": test_data[
                "Date"
            ].iloc[0],

            "Test_End": test_data[
                "Date"
            ].iloc[-1],

            "Sharpe_Ratio": sharpe,

            "Maximum_Drawdown": drawdown,

            "Total_Return": total_return

        })

        start += test_size

        fold += 1

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_walk_forward.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nWALK-FORWARD RESULTS")

    print(
        results_df
    )

    print("\nSaved:")

    print(
        output_file
    )

    return results_df


def bootstrap_confidence_interval(
    returns,
    n_bootstrap=1000,
    confidence_level=0.95,
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

    if len(returns) == 0:

        raise ValueError(
            "Returns cannot be empty"
        )

    if block_size <= 0:

        raise ValueError(
            "Block size must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(returns)

    n_blocks = int(
        np.ceil(
            n / block_size
        )
    )

    bootstrap_means = []

    for _ in range(
        n_bootstrap
    ):

        sampled = []

        for _ in range(
            n_blocks
        ):

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

        bootstrap_means.append(
            sampled.mean()
        )

    alpha = (
        1 - confidence_level
    )

    lower = np.quantile(
        bootstrap_means,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2
    )

    return {
        "Mean_Return": returns.mean(),
        "Confidence_Level": confidence_level,
        "Lower_CI": lower,
        "Upper_CI": upper
    }


if __name__ == "__main__":

    ticker = "AAPL"

    print(
        "\nRUNNING WALK-FORWARD TEST"
    )

    walk_forward_test(
        ticker=ticker
    )

    print(
        "\nRUNNING BOOTSTRAP"
    )

    df = prepare_strategy_returns(
        ticker
    )

    returns = df[
        "Strategy_Return"
    ].to_numpy()

    ci = bootstrap_confidence_interval(
        returns,
        n_bootstrap=1000,
        confidence_level=0.95,
        block_size=5,
        seed=42
    )

    print(
        "\nBOOTSTRAP CONFIDENCE INTERVAL"
    )

    print(
        ci
    )

    pd.DataFrame(
        [ci]
    ).to_csv(
        OUTPUT_DIR /
        f"{ticker}_bootstrap_ci.csv",
        index=False
    )

    print(
        "\nBOOTSTRAP COMPLETE"
    )
