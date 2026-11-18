
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BACKEND_DIR.parent
    / "data"
    / "processed"
    / "alpha_compiler"
    / "prepared"
)

OUTPUT_DIR = (
    BACKEND_DIR.parent
    / "data"
    / "processed"
    / "alpha_compiler"
    / "walk_forward"
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

N_FOLDS = 4

INITIAL_TRAIN_RATIO = 0.50

VALIDATION_WINDOW_RATIO = 0.10

HORIZONS = {
    1: "Target_Return_1D",
    5: "Target_Return_5D",
    10: "Target_Return_10D"
}

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
    "Open_Close_Return",
    "MACD_Normalized",
    "MACD_Signal_Normalized",
    "MACD_Histogram_Normalized"
]


# --------------------------------------------------
# CREATE ONE WALK-FORWARD FOLD
# --------------------------------------------------

def create_fold(data, fold_number, horizon):

    total_rows = len(data)

    initial_train_end = int(
        total_rows * INITIAL_TRAIN_RATIO
    )

    validation_size = int(
        total_rows * VALIDATION_WINDOW_RATIO
    )

    # Expanding training window:
    # Each fold moves forward by one validation window.

    validation_start = (
        initial_train_end
        + (fold_number - 1) * validation_size
    )

    validation_end = (
        validation_start + validation_size
    )

    # Purge the observations immediately before
    # validation to prevent overlapping forward targets.

    train_end = validation_start - horizon

    if train_end <= 0 or validation_end > total_rows:
        raise ValueError(
            f"Insufficient observations for fold {fold_number}"
        )

    train = data.iloc[:train_end].copy()

    validation = data.iloc[
        validation_start:validation_end
    ].copy()

    return train, validation


# --------------------------------------------------
# PROCESS ONE HORIZON
# --------------------------------------------------

def process_horizon(horizon, target_column):

    print("\n" + "=" * 70)
    print(f"WALK-FORWARD VALIDATION: {horizon}-DAY HORIZON")
    print("=" * 70)

    input_path = (
        INPUT_DIR / f"AAPL_{horizon}d_prepared.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    df["Date"] = pd.to_datetime(df["Date"])

    # Only use development data.
    # The original final test period remains untouched.

    development = df[
        df["Split"].isin(["train", "validation"])
    ].copy()

    development = development.sort_values(
        "Date"
    ).reset_index(drop=True)

    required_columns = [
        "Date",
        target_column
    ] + FEATURE_COLUMNS

    missing = [
        col for col in required_columns
        if col not in development.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    development = development[
        required_columns
    ].copy()

    development = development.replace(
        [np.inf, -np.inf],
        np.nan
    )

    development = development.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    print("\nDevelopment observations:", len(development))

    horizon_output = OUTPUT_DIR / f"{horizon}d"

    horizon_output.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []

    # --------------------------------------------------
    # GENERATE FOLDS
    # --------------------------------------------------

    for fold_number in range(1, N_FOLDS + 1):

        train, validation = create_fold(
            development,
            fold_number,
            horizon
        )

        print(f"\nFOLD {fold_number}")

        print("Training observations:", len(train))
        print("Validation observations:", len(validation))

        print(
            "Training dates:",
            train["Date"].min(),
            "to",
            train["Date"].max()
        )

        print(
            "Validation dates:",
            validation["Date"].min(),
            "to",
            validation["Date"].max()
        )

        # Confirm chronological ordering and separation.

        if train["Date"].max() >= validation["Date"].min():
            raise ValueError(
                f"Date overlap in fold {fold_number}"
            )

        # Save fold datasets.

        train_path = (
            horizon_output
            / f"fold_{fold_number}_train.csv"
        )

        validation_path = (
            horizon_output
            / f"fold_{fold_number}_validation.csv"
        )

        train.to_csv(
            train_path,
            index=False
        )

        validation.to_csv(
            validation_path,
            index=False
        )

        summary.append({
            "Horizon": horizon,
            "Fold": fold_number,
            "Train_Observations": len(train),
            "Validation_Observations": len(validation),
            "Purge_Rows": horizon,
            "Train_Start": train["Date"].min(),
            "Train_End": train["Date"].max(),
            "Validation_Start": validation["Date"].min(),
            "Validation_End": validation["Date"].max()
        })

        print("Chronological ordering: PASS")
        print("Training file:", train_path.name)
        print("Validation file:", validation_path.name)

    # --------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------

    summary_df = pd.DataFrame(summary)

    summary_path = (
        horizon_output / "fold_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nFOLD SUMMARY")

    print(
        summary_df.to_string(index=False)
    )

    print("\nSaved summary:", summary_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("ALPHACOMPILER WALK-FORWARD VALIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for horizon, target_column in HORIZONS.items():

        process_horizon(
            horizon,
            target_column
        )

    print("\n" + "=" * 70)
    print("ALL WALK-FORWARD FOLDS GENERATED")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nIMPORTANT:")
    print("These folds are for development and model selection.")
    print("The original final test periods remain untouched.")
    print("No model has been trained or evaluated yet.")


if __name__ == "__main__":
    main()
