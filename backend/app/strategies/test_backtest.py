
from pathlib import Path

import pandas as pd

from backtest_engine import (
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
    PROJECT_DIR /
    "data" /
    "processed" /
    "backtests"
)


def test_backtest():

    print("\nSTEP 1: Loading historical data")

    df = pd.read_csv(
        DATA_DIR / "AAPL.csv"
    )

    print("Historical data shape:", df.shape)

    print("\nSTEP 2: Running momentum backtest")

    result = run_backtest(
        df,
        strategy="momentum"
    )

    print("\nSTEP 3: Calculating summary")

    summary = calculate_summary(
        result
    )

    assert "Net_Return" in result.columns
    assert "Equity" in result.columns
    assert "Sharpe_Ratio" in summary
    assert result["Equity"].notna().any()

    print("\nBACKTEST SUMMARY")
    print(summary)

    print("\nSTEP 4: Saving backtest output")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR /
        "AAPL_backtest.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print("\nBACKTEST SUCCESSFUL!")
    print("Saved:", output_file)

    return result


if __name__ == "__main__":

    test_backtest()
