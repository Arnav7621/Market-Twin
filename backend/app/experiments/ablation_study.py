
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

SCENARIO_DIR = (
    DATA_DIR / "market_twin"
)

OUTPUT_DIR = (
    DATA_DIR / "experiments"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURES = [
    "Momentum_5D",
    "Volatility_20D",
    "Volume"
]


def load_real_data(ticker="AAPL"):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    # Predict next-day return
    df["Target"] = (
        df["Return"].shift(-1)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target"]
    )

    return df


def load_synthetic_data():

    scenario_file = (
        SCENARIO_DIR /
        "high_vol_low_liq_negative.csv"
    )

    df = pd.read_csv(
        scenario_file
    )

    df["Momentum_5D"] = (
        df["Price"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    # Target is next-step simulated return
    df["Target"] = (
        df["Return"].shift(-1)
    )

    df["Volume"] = df["Volume"]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target"]
    )

    return df


def calculate_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    ic = spearmanr(
        y_true,
        y_pred
    ).statistic

    return {
        "MSE": mse,
        "IC": ic
    }


def run_ablation(
    ticker="AAPL",
    seed=42
):

    print("\nLOADING REAL DATA")

    real = load_real_data(
        ticker
    )

    print(
        "Real data:",
        real.shape
    )

    # Chronological split
    split = int(
        len(real) * 0.70
    )

    train_real = real.iloc[
        :split
    ]

    test_real = real.iloc[
        split:
    ]

    X_train_real = train_real[
        FEATURES
    ]

    y_train_real = train_real[
        "Target"
    ]

    X_test = test_real[
        FEATURES
    ]

    y_test = test_real[
        "Target"
    ]

    # Experiment A
    print(
        "\nEXPERIMENT A: REAL ONLY"
    )

    model_real = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=seed
    )

    model_real.fit(
        X_train_real,
        y_train_real
    )

    pred_real = model_real.predict(
        X_test
    )

    metrics_real = calculate_metrics(
        y_test,
        pred_real
    )

    print(
        metrics_real
    )

    # Experiment B
    print(
        "\nEXPERIMENT B: REAL + SYNTHETIC"
    )

    synthetic = load_synthetic_data()

    # Synthetic augmentation uses training data only
    X_synthetic = synthetic[
        FEATURES
    ]

    y_synthetic = synthetic[
        "Target"
    ]

    X_augmented = pd.concat(
        [
            X_train_real,
            X_synthetic
        ],
        ignore_index=True
    )

    y_augmented = pd.concat(
        [
            y_train_real,
            y_synthetic
        ],
        ignore_index=True
    )

    model_augmented = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=seed
    )

    model_augmented.fit(
        X_augmented,
        y_augmented
    )

    pred_augmented = (
        model_augmented.predict(
            X_test
        )
    )

    metrics_augmented = calculate_metrics(
        y_test,
        pred_augmented
    )

    print(
        metrics_augmented
    )

    results = pd.DataFrame({

        "Experiment": [
            "Real_Only",
            "Real_Plus_Synthetic"
        ],

        "MSE": [
            metrics_real["MSE"],
            metrics_augmented["MSE"]
        ],

        "IC": [
            metrics_real["IC"],
            metrics_augmented["IC"]
        ]

    })

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_ablation.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        "\nABLATION COMPLETED"
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

    run_ablation(
        ticker="AAPL",
        seed=42
    )
