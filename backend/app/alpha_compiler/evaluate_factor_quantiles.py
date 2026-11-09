
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed" / "alpha_compiler"

TICKER = "AAPL"

INPUT_PATH = (
    DATA_DIR /
    f"{TICKER}_factor_predictions.csv"
)

OUTPUT_PATH = (
    DATA_DIR /
    f"{TICKER}_factor_quantile_results.csv"
)


def evaluate_quantiles():

    print("\nALPHACOMPILER QUANTILE BACKTEST")
    print("=" * 50)

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "Date",
        "Factor_Value",
        "Actual_Return"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=[
            "Factor_Value",
            "Actual_Return"
        ]
    ).copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values("Date")

    # Assign observations to five factor quantiles.
    df["Quantile"] = pd.qcut(
        df["Factor_Value"].rank(method="first"),
        q=5,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5"
        ]
    )

    # =====================================
    # QUANTILE PERFORMANCE
    # =====================================

    results = []

    for quantile in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5"
    ]:

        subset = df[
            df["Quantile"] == quantile
        ]

        returns = subset[
            "Actual_Return"
        ]

        mean_return = returns.mean()

        volatility = returns.std()

        cumulative_return = (
            np.prod(1 + returns) - 1
        )

        results.append({

            "Quantile": quantile,

            "Observations": len(subset),

            "Mean_Return": mean_return,

            "Return_Std": volatility,

            "Cumulative_Return": cumulative_return,

            "Positive_Return_Rate": (
                returns > 0
            ).mean()

        })

    results_df = pd.DataFrame(
        results
    )

    print("\nQUANTILE RESULTS")

    print(
        results_df.to_string(
            index=False
        )
    )

    # =====================================
    # TOP VS BOTTOM QUANTILE
    # =====================================

    q5_returns = df[
        df["Quantile"] == "Q5"
    ]["Actual_Return"]

    q1_returns = df[
        df["Quantile"] == "Q1"
    ]["Actual_Return"]

    q5_mean = q5_returns.mean()

    q1_mean = q1_returns.mean()

    spread = q5_mean - q1_mean

    print("\nFACTOR SPREAD")

    print(
        f"Q5 Mean Return: {q5_mean:.6f}"
    )

    print(
        f"Q1 Mean Return: {q1_mean:.6f}"
    )

    print(
        f"Q5-Q1 Mean Return Spread: {spread:.6f}"
    )

    # =====================================
    # SAVE RESULTS
    # =====================================

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nRESULTS SAVED")

    print(OUTPUT_PATH)


if __name__ == "__main__":

    evaluate_quantiles()
