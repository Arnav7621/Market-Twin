
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BACKEND_DIR.parent
    / "data"
    / "processed"
    / "alpha_compiler"
    / "AAPL_features.csv"
)

OUTPUT_DIR = (
    BACKEND_DIR.parent
    / "data"
    / "processed"
    / "alpha_compiler"
    / "prepared"
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

HORIZONS = {
    1: "Target_Return_1D",
    5: "Target_Return_5D",
    10: "Target_Return_10D"
}


# --------------------------------------------------
# APPROVED FEATURE WHITELIST
# --------------------------------------------------

FEATURE_COLUMNS = [
    "Return",
    "Log_Return",

    "Momentum_5D",
    "Momentum_10D",
    "Momentum_20D",
    "Momentum_60D",

    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "Volatility_Ratio",

    "RSI_14",

    "Distance_From_SMA_20",
    "Distance_From_SMA_50",
    "Distance_From_SMA_100",

    "ATR_Percent",

    "Volume_Change",
    "Volume_ZScore_20D",

    "Price_Range",
    "Open_Close_Return"
]


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(df):

    df = df.copy()

    # Normalize MACD features by current closing price.
    # This reduces dependence on absolute stock price.

    df["MACD_Normalized"] = df["MACD"] / df["Close"]

    df["MACD_Signal_Normalized"] = (
        df["MACD_Signal"] / df["Close"]
    )

    df["MACD_Histogram_Normalized"] = (
        df["MACD_Histogram"] / df["Close"]
    )

    features = FEATURE_COLUMNS + [
        "MACD_Normalized",
        "MACD_Signal_Normalized",
        "MACD_Histogram_Normalized"
    ]

    return df, features


# --------------------------------------------------
# CHRONOLOGICAL SPLIT WITH PURGING
# --------------------------------------------------

def create_splits(df, horizon):

    df = df.copy().reset_index(drop=True)

    total_rows = len(df)

    train_end = int(total_rows * TRAIN_RATIO)

    validation_end = int(
        total_rows * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    # Purge the final `horizon` observations of each
    # preceding split to prevent overlapping target
    # windows across split boundaries.

    train = df.iloc[:max(0, train_end - horizon)].copy()

    validation = df.iloc[
        train_end:max(train_end, validation_end - horizon)
    ].copy()

    test = df.iloc[validation_end:].copy()

    train["Split"] = "train"
    validation["Split"] = "validation"
    test["Split"] = "test"

    return train, validation, test


# --------------------------------------------------
# PROCESS ONE HORIZON
# --------------------------------------------------

def process_horizon(df, horizon, target_column, features):

    print("\n" + "=" * 65)
    print(f"PREPARING {horizon}-DAY PREDICTION DATASET")
    print("=" * 65)

    required = [
        "Date",
        target_column
    ] + features

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows where any required feature or target
    # is unavailable.

    before = len(data)

    data = data.dropna(
        subset=required
    ).sort_values("Date").reset_index(drop=True)

    print("\nRows before cleaning:", before)
    print("Rows after cleaning:", len(data))
    print("Rows removed:", before - len(data))

    if len(data) < 100:
        raise ValueError(
            "Not enough observations for chronological splitting."
        )

    # --------------------------------------------------
    # CREATE PURGED SPLITS
    # --------------------------------------------------

    train, validation, test = create_splits(
        data,
        horizon
    )

    splits = {
        "train": train,
        "validation": validation,
        "test": test
    }

    # --------------------------------------------------
    # DISPLAY SPLIT INFORMATION
    # --------------------------------------------------

    print("\nSPLIT SUMMARY")

    for name, part in splits.items():

        print(f"\n{name.upper()}")

        print("Observations:", len(part))

        if len(part) > 0:
            print(
                "Date range:",
                part["Date"].min(),
                "to",
                part["Date"].max()
            )

    # --------------------------------------------------
    # VALIDATE SPLIT BOUNDARIES
    # --------------------------------------------------

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "One or more splits are empty."
        )

    train_last = train.index.max()
    validation_first = validation.index.min()

    validation_last = validation.index.max()
    test_first = test.index.min()

    print("\nPURGE VALIDATION")

    print(
        "Train-to-validation row gap:",
        validation_first - train_last - 1
    )

    print(
        "Validation-to-test row gap:",
        test_first - validation_last - 1
    )

    # --------------------------------------------------
    # SAVE PREPARED DATASET
    # --------------------------------------------------

    combined = pd.concat(
        [train, validation, test],
        ignore_index=True
    )

    output_path = (
        OUTPUT_DIR
        / f"AAPL_{horizon}d_prepared.csv"
    )

    combined.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:", output_path)

    print("\nFeature count:", len(features))
    print("Target column:", target_column)

    print(
        "Missing feature values:",
        combined[features].isna().sum().sum()
    )

    print(
        "Missing target values:",
        combined[target_column].isna().sum()
    )

    print(
        "Infinite feature values:",
        np.isinf(
            combined[features].to_numpy(dtype=float)
        ).sum()
    )

    return combined


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("ALPHACOMPILER LEAKAGE-SAFE DATA PREPARATION")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    df, features = prepare_features(df)

    print("\nInput observations:", len(df))
    print("Selected feature count:", len(features))

    print("\nSelected features:")

    for feature in features:
        print("-", feature)

    print("\nExcluded from model inputs:")
    print("- Raw OHLC price levels")
    print("- Raw volume")
    print("- Cumulative return")
    print("- Raw moving averages")
    print("- Raw MACD values")
    print("- All future target columns")

    for horizon, target_column in HORIZONS.items():

        process_horizon(
            df,
            horizon,
            target_column,
            features
        )

    print("\n" + "=" * 65)
    print("ALL PREPARED DATASETS GENERATED")
    print("=" * 65)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
