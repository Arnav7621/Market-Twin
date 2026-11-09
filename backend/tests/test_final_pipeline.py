
from pathlib import Path

import pandas as pd

from app.strategies.backtest_engine import (
    run_backtest,
    calculate_summary
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)


def test_market_data():

    df = pd.read_csv(
        DATA_DIR /
        "AAPL.csv"
    )

    assert len(df) > 0

    assert "Close" in df.columns

    assert "Return" in df.columns


def test_strategy_backtest():

    df = pd.read_csv(
        DATA_DIR /
        "AAPL.csv"
    )

    result = run_backtest(
        df,
        strategy="momentum"
    )

    assert len(result) > 0

    assert "Net_Return" in result.columns

    assert "Equity" in result.columns


def test_summary_metrics():

    df = pd.read_csv(
        DATA_DIR /
        "AAPL.csv"
    )

    result = run_backtest(
        df,
        strategy="momentum"
    )

    summary = calculate_summary(
        result
    )

    assert "Sharpe_Ratio" in summary

    assert "Maximum_Drawdown" in summary

    assert "Total_Return" in summary
