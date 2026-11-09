
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

SCENARIO_DIR = DATA_DIR / "market_twin"

OUTPUT_DIR = DATA_DIR / "experiments"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BASE_FEATURES = [
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

    df["Alpha_Factor"] = (
        df["Momentum_5D"] /
        (df["Volatility_20D"] + 1e-8)
    )

    # QueueMind proxy:
    # Higher volume and lower volatility imply
    # a higher estimated fill probability.
    df["Fill_Probability"] = (
        0.5
        + 0.3 * np.tanh(
            np.log1p(df["Volume"]) -
            np.log1p(df["Volume"].median())
        )
        - 0.2 * np.tanh(
            df["Volatility_20D"] /
            (df["Volatility_20D"].median() + 1e-8)
            - 1
        )
    )

    df["Fill_Probability"] = (
        df["Fill_Probability"].clip(0.05, 0.95)
    )

    # Proxy execution cost
    df["Execution_Cost"] = (
        0.001
        + 0.0005 *
        df["Volatility_20D"].fillna(0)
        + 0.0001 /
        (np.log1p(df["Volume"]) + 1e-8)
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
        subset=BASE_FEATURES + [
            "Alpha_Factor",
            "Fill_Probability",
            "Execution_Cost",
            "Target"
        ]
    )

    return df


def load_synthetic_data():

    scenario_file = (
        SCENARIO_DIR /
        "high_vol_low_liq_negative.csv"
    )

    if not scenario_file.exists():

        raise FileNotFoundError(
            f"Synthetic scenario not found: {scenario_file}"
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

    df["Alpha_Factor"] = (
        df["Momentum_5D"] /
        (df["Volatility_20D"] + 1e-8)
    )

    df["Fill_Probability"] = (
        0.5
        + 0.3 * np.tanh(
            np.log1p(df["Volume"]) -
            np.log1p(df["Volume"].median())
        )
        - 0.2 * np.tanh(
            df["Volatility_20D"] /
            (df["Volatility_20D"].median() + 1e-8)
            - 1
        )
    )

    df["Fill_Probability"] = (
        df["Fill_Probability"].clip(0.05, 0.95)
    )

    df["Execution_Cost"] = (
        0.001
        + 0.0005 *
        df["Volatility_20D"].fillna(0)
        + 0.0001 /
        (np.log1p(df["Volume"]) + 1e-8)
    )

    df["Target"] = (
        df["Return"].shift(-1)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=BASE_FEATURES + [
            "Alpha_Factor",
            "Fill_Probability",
            "Execution_Cost",
            "Target"
        ]
    )

    return df


def calculate_ic(
    actual,
    predicted
):

    result = spearmanr(
        actual,
        predicted
    )

    return float(result.statistic)


def train_model(
    train_df,
    test_df,
    features,
    synthetic_df=None,
    seed=42
):

    X_train = train_df[
        features
    ]

    y_train = train_df[
        "Target"
    ]

    # Synthetic augmentation for MarketTwin proxy
    if synthetic_df is not None:

        X_synthetic = synthetic_df[
            features
        ]

        y_synthetic = synthetic_df[
            "Target"
        ]

        X_train = pd.concat(
            [
                X_train,
                X_synthetic
            ],
            ignore_index=True
        )

        y_train = pd.concat(
            [
                y_train,
                y_synthetic
            ],
            ignore_index=True
        )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        test_df[features]
    )

    return predictions


def run_full_ablation(
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

    # Chronological train-test split
    split = int(
        len(real) * 0.70
    )

    train_df = real.iloc[
        :split
    ].copy()

    test_df = real.iloc[
        split:
    ].copy()

    actual = test_df[
        "Target"
    ].to_numpy()

    print(
        "\nLoading synthetic MarketTwin data"
    )

    synthetic = load_synthetic_data()

    print(
        "Synthetic data:",
        synthetic.shape
    )

    results = []

    # A: Baseline
    baseline_pred = train_model(
        train_df,
        test_df,
        BASE_FEATURES,
        seed=seed
    )

    baseline_ic = calculate_ic(
        actual,
        baseline_pred
    )

    results.append({
        "Experiment": "A_Baseline",
        "IC": baseline_ic
    })

    # B: AlphaCompiler proxy
    alpha_features = BASE_FEATURES + [
        "Alpha_Factor"
    ]

    alpha_pred = train_model(
        train_df,
        test_df,
        alpha_features,
        seed=seed
    )

    alpha_ic = calculate_ic(
        actual,
        alpha_pred
    )

    results.append({
        "Experiment": "B_AlphaCompiler_Proxy",
        "IC": alpha_ic
    })

    # C: QueueMind proxy
    queue_features = BASE_FEATURES + [
        "Fill_Probability"
    ]

    queue_pred = train_model(
        train_df,
        test_df,
        queue_features,
        seed=seed
    )

    queue_ic = calculate_ic(
        actual,
        queue_pred
    )

    results.append({
        "Experiment": "C_QueueMind_Proxy",
        "IC": queue_ic
    })

    # D: MarketTwin proxy
    market_features = BASE_FEATURES

    market_pred = train_model(
        train_df,
        test_df,
        market_features,
        synthetic_df=synthetic,
        seed=seed
    )

    market_ic = calculate_ic(
        actual,
        market_pred
    )

    results.append({
        "Experiment": "D_MarketTwin_Proxy",
        "IC": market_ic
    })

    # E: Alpha + Execution proxy
    alpha_execution_pred = (
        alpha_pred *
        test_df["Fill_Probability"].to_numpy()
        - test_df["Execution_Cost"].to_numpy()
    )

    alpha_execution_ic = calculate_ic(
        actual,
        alpha_execution_pred
    )

    results.append({
        "Experiment": "E_Alpha_Execution_Proxy",
        "IC": alpha_execution_ic
    })

    # F: Full pipeline proxy
    full_features = BASE_FEATURES + [
        "Alpha_Factor",
        "Fill_Probability"
    ]

    full_pred = train_model(
        train_df,
        test_df,
        full_features,
        synthetic_df=synthetic,
        seed=seed
    )

    full_execution_pred = (
        full_pred *
        test_df["Fill_Probability"].to_numpy()
        - test_df["Execution_Cost"].to_numpy()
    )

    full_ic = calculate_ic(
        actual,
        full_execution_pred
    )

    results.append({
        "Experiment": "F_Full_Pipeline_Proxy",
        "IC": full_ic
    })

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_full_ablation.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nFULL PIPELINE ABLATION"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        output_file
    )

    return results_df


if __name__ == "__main__":

    run_full_ablation(
        ticker="AAPL",
        seed=42
    )
