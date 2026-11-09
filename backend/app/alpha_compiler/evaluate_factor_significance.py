
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_ind,
    mannwhitneyu
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed" /
    "alpha_compiler"
)

INPUT_PATH = (
    DATA_DIR /
    "AAPL_factor_predictions.csv"
)


def evaluate_significance():

    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(
        subset=[
            "Factor_Value",
            "Actual_Return"
        ]
    ).copy()

    df["Quantile"] = pd.qcut(
        df["Factor_Value"].rank(method="first"),
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    q1 = df[
        df["Quantile"] == "Q1"
    ]["Actual_Return"].values

    q5 = df[
        df["Quantile"] == "Q5"
    ]["Actual_Return"].values

    spread = q5.mean() - q1.mean()

    # Welch's independent two-sample t-test.
    t_stat, p_value = ttest_ind(
        q5,
        q1,
        equal_var=False
    )

    # Non-parametric distribution comparison.
    u_stat, u_pvalue = mannwhitneyu(
        q5,
        q1,
        alternative="two-sided"
    )

    print("\nFACTOR SIGNIFICANCE ANALYSIS")
    print("=" * 50)

    print(f"Q5 Mean Return: {q5.mean():.6f}")
    print(f"Q1 Mean Return: {q1.mean():.6f}")

    print(f"Q5-Q1 Spread: {spread:.6f}")

    print(f"Welch t-statistic: {t_stat:.6f}")
    print(f"Welch p-value: {p_value:.6f}")

    print(f"Mann-Whitney U: {u_stat:.6f}")
    print(f"Mann-Whitney p-value: {u_pvalue:.6f}")

    print("\nINTERPRETATION")

    if p_value < 0.05:

        print(
            "The mean return difference is statistically "
            "significant at the 5% level under the "
            "Welch test assumptions."
        )

    else:

        print(
            "The mean return difference is not statistically "
            "significant at the 5% level under the "
            "Welch test assumptions."
        )


if __name__ == "__main__":

    evaluate_significance()
