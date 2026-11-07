
import numpy as np
import pandas as pd

from scipy.stats import spearmanr


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

    result = spearmanr(
        factor,
        future_returns
    )

    return result.statistic


def calculate_complexity(expression):

    expression = str(
        expression
    )

    return len(
        expression.split()
    ) + expression.count("(")


def calculate_turnover(factor):

    factor = pd.Series(
        factor
    ).dropna()

    if len(factor) < 2:

        return 0.0

    factor = factor / (
        factor.abs().mean() + 1e-8
    )

    turnover = np.mean(
        np.abs(
            np.diff(factor)
        )
    )

    return turnover


def calculate_fitness(
    factor,
    future_returns,
    expression,
    lambda_complexity=0.001,
    gamma_turnover=0.01
):

    ic = calculate_information_coefficient(
        factor,
        future_returns
    )

    complexity = calculate_complexity(
        expression
    )

    turnover = calculate_turnover(
        factor
    )

    fitness = (
        ic
        - lambda_complexity * complexity
        - gamma_turnover * turnover
    )

    return {
        "IC": ic,
        "Complexity": complexity,
        "Turnover": turnover,
        "Fitness": fitness
    }


if __name__ == "__main__":

    factor = pd.Series([
        0.10,
        0.20,
        0.15,
        0.30,
        0.25,
        0.40,
        0.35,
        0.50
    ])

    future_returns = pd.Series([
        0.01,
        0.02,
        0.015,
        0.03,
        0.025,
        0.04,
        0.035,
        0.05
    ])

    expression = "add(mul(X0, X1), X2)"

    result = calculate_fitness(
        factor,
        future_returns,
        expression
    )

    print("\nALPHA FITNESS RESULT")

    print(result)
