
import numpy as np
import pandas as pd

from scipy.stats import spearmanr


def calculate_sharpe(
    returns,
    risk_free_rate=0.0,
    periods_per_year=252
):

    returns = np.asarray(
        returns,
        dtype=float
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) < 2:

        return np.nan

    excess_returns = (
        returns - risk_free_rate / periods_per_year
    )

    std = np.std(
        excess_returns,
        ddof=1
    )

    if std == 0:

        return np.nan

    sharpe = (
        np.mean(excess_returns) /
        std
    ) * np.sqrt(
        periods_per_year
    )

    return sharpe


def calculate_max_drawdown(
    returns
):

    returns = np.asarray(
        returns,
        dtype=float
    )

    returns = np.nan_to_num(
        returns
    )

    cumulative = np.cumprod(
        1 + returns
    )

    running_max = np.maximum.accumulate(
        cumulative
    )

    drawdown = (
        cumulative /
        running_max - 1
    )

    return drawdown.min()


def calculate_information_coefficient(
    factor,
    future_returns
):

    factor = pd.Series(
        factor
    )

    future_returns = pd.Series(
        future_returns
    )

    valid = (
        factor.notna() &
        future_returns.notna()
    )

    factor = factor[valid]

    future_returns = future_returns[valid]

    if len(factor) < 3:

        return np.nan

    return spearmanr(
        factor,
        future_returns
    ).statistic


def calculate_turnover(
    traded_notional,
    capital
):

    if capital <= 0:

        raise ValueError(
            "Capital must be positive"
        )

    return (
        traded_notional /
        capital
    )


def calculate_win_rate(
    pnl
):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) == 0:

        return np.nan

    return np.mean(
        pnl > 0
    )


def calculate_profit_factor(
    pnl
):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    profits = pnl[
        pnl > 0
    ].sum()

    losses = abs(
        pnl[pnl < 0].sum()
    )

    if losses == 0:

        return np.inf if profits > 0 else np.nan

    return profits / losses


def calculate_metrics(
    pnl,
    capital=100000
):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = np.nan_to_num(
        pnl
    )

    returns = (
        pnl / capital
    )

    metrics = {

        "Total_PnL": pnl.sum(),

        "Sharpe_Ratio": calculate_sharpe(
            returns
        ),

        "Maximum_Drawdown": calculate_max_drawdown(
            returns
        ),

        "Win_Rate": calculate_win_rate(
            pnl
        ),

        "Profit_Factor": calculate_profit_factor(
            pnl
        ),

        "Average_PnL": pnl.mean(),

        "Number_Of_Trades": len(pnl)

    }

    return metrics


if __name__ == "__main__":

    pnl = np.array([
        100,
        -50,
        200,
        -30,
        150,
        -20,
        80,
        -40
    ])

    metrics = calculate_metrics(
        pnl,
        capital=100000
    )

    print("\nPERFORMANCE METRICS")

    for name, value in metrics.items():

        print(
            name,
            ":",
            value
        )
