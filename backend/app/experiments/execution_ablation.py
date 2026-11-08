
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = (
    DATA_DIR / "experiments"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_data(
    ticker="AAPL"
):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Alpha_Factor"] = (
        df["Momentum_5D"] /
        (
            df["Volatility_20D"] + 1e-8
        )
    )

    df["Target"] = (
        df["Return"].shift(-1)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Alpha_Factor",
            "Target"
        ]
    )

    return df


def prediction_only(
    df
):

    return (
        df["Alpha_Factor"] *
        df["Target"]
    )


def execution_aware(
    df,
    fill_probability=0.80
):

    spread_cost = (
        5 / 10000
    )

    slippage_cost = (
        2 / 10000
    )

    transaction_cost = (
        0.001
    )

    total_cost = (
        spread_cost / 2 +
        slippage_cost +
        transaction_cost
    )

    expected_return = (
        df["Alpha_Factor"] *
        df["Target"]
    )

    return (
        expected_return *
        fill_probability -
        total_cost
    )


def statistical_filter(
    df,
    threshold=0.001
):

    factor = (
        df["Alpha_Factor"]
    )

    filtered = factor.where(
        factor.abs() > threshold,
        0
    )

    return filtered


def run_experiments(
    ticker="AAPL"
):

    df = load_data(
        ticker
    )

    # Experiment C
    prediction_score = (
        prediction_only(df)
    )

    execution_score = (
        execution_aware(df)
    )

    # Experiment D
    fill_low = (
        execution_aware(
            df,
            fill_probability=0.30
        )
    )

    fill_high = (
        execution_aware(
            df,
            fill_probability=0.80
        )
    )

    # Experiment E
    original_factor = (
        df["Alpha_Factor"]
    )

    filtered_factor = (
        statistical_filter(df)
    )

    results = pd.DataFrame({

        "Experiment": [
            "Prediction_Only",
            "Execution_Aware",
            "Fill_Probability_30",
            "Fill_Probability_80",
            "Unfiltered_Factor",
            "Statistically_Filtered"
        ],

        "Average_Score": [

            prediction_score.mean(),

            execution_score.mean(),

            fill_low.mean(),

            fill_high.mean(),

            original_factor.mean(),

            filtered_factor.mean()

        ],

        "IC": [

            spearmanr(
                prediction_score,
                df["Target"]
            ).statistic,

            spearmanr(
                execution_score,
                df["Target"]
            ).statistic,

            spearmanr(
                fill_low,
                df["Target"]
            ).statistic,

            spearmanr(
                fill_high,
                df["Target"]
            ).statistic,

            spearmanr(
                original_factor,
                df["Target"]
            ).statistic,

            spearmanr(
                filtered_factor,
                df["Target"]
            ).statistic

        ]

    })

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_execution_ablation.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        "\nEXECUTION ABLATION RESULTS"
    )

    print(
        results
    )

    print(
        "\nSaved:",
        output_file
    )

    return results


if __name__ == "__main__":

    run_experiments(
        ticker="AAPL"
    )
