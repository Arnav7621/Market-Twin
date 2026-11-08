
from pathlib import Path

import pandas as pd

from performance_metrics import (
    calculate_metrics
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

BACKTEST_DIR = (
    PROJECT_DIR /
    "data" /
    "processed" /
    "backtests"
)

OUTPUT_DIR = (
    PROJECT_DIR /
    "data" /
    "processed" /
    "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def validate_backtest(
    ticker="AAPL"
):

    backtest = pd.read_csv(
        BACKTEST_DIR /
        f"{ticker}_backtest.csv"
    )

    pnl = backtest[
        "Net_PnL"
    ].to_numpy()

    metrics = calculate_metrics(
        pnl,
        capital=100000
    )

    print("\nVALIDATION RESULTS")

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_file,
        index=False
    )

    print("\nMetrics saved!")

    print(output_file)


if __name__ == "__main__":

    validate_backtest(
        "AAPL"
    )
