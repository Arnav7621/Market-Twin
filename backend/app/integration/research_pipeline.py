from pathlib import Path
import sys

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from market_twin.scenario_generator import generate_scenario

PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "integration"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def generate_alpha_factor(df):

    df = df.copy()

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Alpha_Factor"] = (
        df["Momentum_5D"] /
        (df["Volatility_20D"] + 1e-8)
    )

    return df


def run_pipeline(
    ticker="AAPL",
    duration=1000,
    seed=42
):

    print("\nSTEP 1: Loading historical data")

    historical = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    print(
        "Historical shape:",
        historical.shape
    )

    print("\nSTEP 2: Generating alpha factor")

    historical = generate_alpha_factor(
        historical
    )

    print(
        historical[
            [
                "Date",
                "Return",
                "Alpha_Factor"
            ]
        ].tail()
    )

    print("\nSTEP 3: Generating MarketTwin scenario")

    scenario = generate_scenario(
        ticker=ticker,
        volatility="high",
        liquidity="low",
        imbalance="negative",
        duration=duration,
        seed=seed
    )

    print(
        "Scenario shape:",
        scenario.shape
    )

    print("\nSTEP 4: Computing synthetic alpha factor")

    scenario["Momentum_5D"] = (
        scenario["Price"].pct_change(5)
    )

    scenario["Volatility_20D"] = (
        scenario["Return"].rolling(20).std()
    )

    scenario["Alpha_Factor"] = (
        scenario["Momentum_5D"] /
        (
            scenario["Volatility_20D"] + 1e-8
        )
    )

    print("\nSTEP 5: Saving integration output")

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_integrated_scenario.csv"
    )

    scenario.to_csv(
        output_file,
        index=False
    )

    print("\nINTEGRATION SUCCESSFUL!")

    print(
        "Saved:",
        output_file
    )

    return scenario


if __name__ == "__main__":

    run_pipeline(
        ticker="AAPL",
        duration=1000,
        seed=42
    )
