
import numpy as np
import pandas as pd


TRADING_DAYS = 252
ANNUAL_RISK_FREE_RATE = 0.04


def calculate_volatility(returns):
    """
    Calculate annualized volatility.
    """

    daily_volatility = returns.std(ddof=1)

    annualized_volatility = (
        daily_volatility * np.sqrt(TRADING_DAYS)
    )

    return annualized_volatility


def calculate_sharpe_ratio(returns):
    """
    Calculate annualized Sharpe ratio.
    """

    daily_rf = (
        (1 + ANNUAL_RISK_FREE_RATE) ** (1 / TRADING_DAYS)
        - 1
    )

    daily_excess_returns = returns - daily_rf

    annualized_excess_return = (
        daily_excess_returns.mean() * TRADING_DAYS
    )

    annualized_volatility = calculate_volatility(
        returns
    )

    if annualized_volatility == 0:
        return np.nan

    sharpe = (
        annualized_excess_return /
        annualized_volatility
    )

    return sharpe


def calculate_sortino_ratio(returns):
    """
    Calculate annualized Sortino ratio.
    """

    daily_rf = (
        (1 + ANNUAL_RISK_FREE_RATE) ** (1 / TRADING_DAYS)
        - 1
    )

    daily_excess_returns = returns - daily_rf

    downside_returns = np.minimum(
        daily_excess_returns,
        0
    )

    downside_deviation = np.sqrt(
        np.mean(downside_returns ** 2)
    )

    annualized_downside_deviation = (
        downside_deviation * np.sqrt(TRADING_DAYS)
    )

    annualized_excess_return = (
        daily_excess_returns.mean() * TRADING_DAYS
    )

    if annualized_downside_deviation == 0:
        return np.nan

    sortino = (
        annualized_excess_return /
        annualized_downside_deviation
    )

    return sortino


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown.
    """

    wealth = (1 + returns).cumprod()

    # Include initial wealth of 1.0
    wealth = pd.concat([
        pd.Series([1.0]),
        wealth.reset_index(drop=True)
    ], ignore_index=True)

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max - 1
    )

    max_drawdown = drawdown.min()

    return max_drawdown


def calculate_var(returns, confidence=0.95):
    """
    Calculate historical Value at Risk.
    Returns a positive loss value.
    """

    percentile = 1 - confidence

    var = -returns.quantile(percentile)

    return var


def calculate_cvar(returns, confidence=0.95):
    """
    Calculate historical Conditional Value at Risk.
    Returns a positive loss value.
    """

    percentile = 1 - confidence

    threshold = returns.quantile(percentile)

    tail_returns = returns[
        returns <= threshold
    ]

    cvar = -tail_returns.mean()

    return cvar


def calculate_all_risk_metrics(returns):
    """
    Calculate all MarketTwin risk metrics.
    """

    # Remove missing and infinite returns
    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(returns) < 2:
        raise ValueError(
            "At least two valid return observations are required."
        )

    metrics = {
        "Annualized_Volatility":
            calculate_volatility(returns),

        "Sharpe_Ratio":
            calculate_sharpe_ratio(returns),

        "Sortino_Ratio":
            calculate_sortino_ratio(returns),

        "Maximum_Drawdown":
            calculate_max_drawdown(returns),

        "VaR_95":
            calculate_var(returns, confidence=0.95),

        "CVaR_95":
            calculate_cvar(returns, confidence=0.95)
    }

    return metrics
