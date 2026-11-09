
from pathlib import Path

import pandas as pd

from app.strategies.backtest_engine import (
    run_backtest,
    calculate_summary
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

OUTPUT_DIR = (
    DATA_DIR /
    "integrated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def run_full_pipeline(
    ticker="AAPL",
    strategy="momentum"
):

    print(
        "\n1. LOADING MARKET DATA"
    )

    df = pd.read_csv(
        DATA_DIR /
        f"{ticker}.csv"
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "\n2. RUNNING STRATEGY ENGINE"
    )

    result = run_backtest(
        df,
        strategy=strategy
    )

    print(
        "\n3. CALCULATING SUMMARY"
    )

    summary = calculate_summary(
        result
    )

    print(
        summary
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_full_pipeline.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT_DIR /
        f"{ticker}_summary.csv",
        index=False
    )

    print(
        "\nFULL PIPELINE COMPLETED"
    )

    print(
        "Saved:",
        output_file
    )

    return result, summary


if __name__ == "__main__":

    run_full_pipeline(
        ticker="AAPL",
        strategy="momentum"
    )
