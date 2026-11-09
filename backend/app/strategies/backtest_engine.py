
import numpy as np
import pandas as pd

from app.strategies.strategy_engine import run_strategy


def run_backtest(
    df,
    strategy="momentum",
    initial_capital=100000,
    transaction_cost=0.001
):

    df = df.copy()

    df = run_strategy(
        df,
        strategy
    )

    df["Return"] = (
        df["Close"].pct_change()
    )

    df["Position"] = (
        df["Position"].fillna(0)
    )

    df["Gross_Return"] = (
        df["Position"] *
        df["Return"]
    )

    df["Turnover"] = (
        df["Position"].diff().abs()
    )

    df["Turnover"] = (
        df["Turnover"].fillna(
            df["Position"].abs()
        )
    )

    df["Transaction_Cost"] = (
        df["Turnover"] *
        transaction_cost
    )

    df["Net_Return"] = (
        df["Gross_Return"] -
        df["Transaction_Cost"]
    )

    df["Net_PnL"] = (
        df["Net_Return"] *
        initial_capital
    )

    df["Equity"] = (
        initial_capital *
        (
            1 + df["Net_Return"]
        ).cumprod()
    )

    df["Peak_Equity"] = (
        df["Equity"].cummax()
    )

    df["Drawdown"] = (
        df["Equity"] /
        df["Peak_Equity"] - 1
    )

    return df


def calculate_summary(
    df
):

    returns = (
        df["Net_Return"].dropna()
    )

    if len(returns) < 2:

        raise ValueError(
            "Not enough returns to calculate metrics"
        )

    std = returns.std(
        ddof=1
    )

    sharpe = (
        returns.mean() /
        std *
        np.sqrt(252)
        if std > 0
        else np.nan
    )

    return {

        "Total_PnL": df["Net_PnL"].sum(),

        "Total_Return": (
            df["Equity"].iloc[-1] /
            df["Equity"].iloc[0] - 1
        ),

        "Sharpe_Ratio": sharpe,

        "Maximum_Drawdown": (
            df["Drawdown"].min()
        ),

        "Number_Of_Observations": len(df)

    }
