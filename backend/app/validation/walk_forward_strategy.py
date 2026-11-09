
from pathlib import Path

import numpy as np
import pandas as pd

from app.strategies.strategy_engine import (
    run_strategy
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


def walk_forward_strategy(
    ticker="AAPL",
    strategy="momentum",
    train_size=500,
    test_size=100
):

    df = pd.read_csv(
        DATA_DIR /
        f"{ticker}.csv"
    )

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    results = []

    start = train_size

    fold = 1

    while (
        start + test_size <= len(df)
    ):

        test_data = df.iloc[
            start:start + test_size
        ].copy()

        # Give the strategy enough preceding history
        # to compute rolling features.
        historical = df.iloc[
            :start + test_size
        ].copy()

        signals = run_strategy(
            historical,
            strategy
        )

        test_signals = signals.iloc[
            start:start + test_size
        ].copy()

        returns = (
            test_signals["Position"] *
            test_signals["Close"].pct_change()
        )

        returns = returns.dropna()

        if len(returns) > 1:

            sharpe = (
                returns.mean() /
                returns.std(ddof=1) *
                np.sqrt(252)
                if returns.std(ddof=1) > 0
                else np.nan
            )

            equity = (
                1 + returns
            ).cumprod()

            drawdown = (
                equity /
                equity.cummax() - 1
            )

            total_return = (
                equity.iloc[-1] - 1
            )

        else:

            sharpe = np.nan

            drawdown = pd.Series(
                [np.nan]
            )

            total_return = np.nan

        results.append({

            "Fold": fold,

            "Train_Start": df["Date"].iloc[0],

            "Train_End": df["Date"].iloc[start - 1],

            "Test_Start": test_data["Date"].iloc[0],

            "Test_End": test_data["Date"].iloc[-1],

            "Sharpe": sharpe,

            "Max_Drawdown": drawdown.min(),

            "Total_Return": total_return

        })

        fold += 1

        start += test_size

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_{strategy}_walk_forward.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nWALK-FORWARD RESULTS"
    )

    print(
        results_df
    )

    print(
        "\nSaved:",
        output_file
    )

    return results_df


if __name__ == "__main__":

    walk_forward_strategy(
        ticker="AAPL",
        strategy="momentum",
        train_size=500,
        test_size=100
    )
