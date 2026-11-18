
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BACKEND_DIR.parent / "data" / "processed" / "AAPL.csv"

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def main():

    section("ALPHACOMPILER DATA QUALITY CHECK")

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    print(f"\nDataset path: {DATA_PATH}")

    if not DATA_PATH.exists():
        print("\nERROR: AAPL.csv was not found.")
        print("Check the dataset path before continuing.")
        return

    df = pd.read_csv(DATA_PATH)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nAvailable columns:")
    print(df.columns.tolist())

    # --------------------------------------------------
    # 2. NORMALIZE COLUMN NAMES
    # --------------------------------------------------

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    required_lower = [
        col.lower() for col in REQUIRED_COLUMNS
    ]

    missing_columns = [
        col for col in required_lower
        if col not in column_map
    ]

    if missing_columns:
        print("\nERROR: Required OHLCV columns are missing:")
        print(missing_columns)
        print("\nAvailable columns:", df.columns.tolist())
        return

    # Rename required columns consistently
    rename_map = {
        column_map[col.lower()]: col
        for col in REQUIRED_COLUMNS
    }

    df = df.rename(columns=rename_map)

    # --------------------------------------------------
    # 3. DATE COLUMN CHECK
    # --------------------------------------------------

    section("DATE VALIDATION")

    date_candidates = [
        col for col in df.columns
        if col.strip().lower() in [
            "date", "datetime", "timestamp"
        ]
    ]

    if not date_candidates:
        print("WARNING: No recognized date column found.")
        date_col = None
    else:
        date_col = date_candidates[0]

        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        print("Date column:", date_col)
        print("Invalid dates:", df[date_col].isna().sum())

        df = df.dropna(subset=[date_col])

        duplicate_dates = df[date_col].duplicated().sum()

        print("Duplicate dates:", duplicate_dates)
        print("Date range:", df[date_col].min(), "to", df[date_col].max())

        if not df[date_col].is_monotonic_increasing:
            print("WARNING: Dates are not in ascending order.")
            df = df.sort_values(date_col).reset_index(drop=True)
        else:
            print("Date ordering: PASS")

    # --------------------------------------------------
    # 4. DUPLICATE ROW CHECK
    # --------------------------------------------------

    section("DUPLICATE CHECK")

    duplicate_rows = df.duplicated().sum()

    print("Duplicate rows:", duplicate_rows)

    # --------------------------------------------------
    # 5. MISSING VALUES
    # --------------------------------------------------

    section("MISSING VALUE CHECK")

    print(df[REQUIRED_COLUMNS].isnull().sum())

    # --------------------------------------------------
    # 6. NUMERIC AND INFINITE VALUE CHECK
    # --------------------------------------------------

    section("NUMERIC VALIDATION")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print("\nNon-numeric or missing values:")
    print(df[REQUIRED_COLUMNS].isna().sum())

    numeric_data = df[REQUIRED_COLUMNS].to_numpy(
        dtype=float
    )

    infinite_count = np.isinf(numeric_data).sum()

    print("\nInfinite values:", infinite_count)

    # --------------------------------------------------
    # 7. PRICE VALIDATION
    # --------------------------------------------------

    section("PRICE VALIDATION")

    for col in ["Open", "High", "Low", "Close"]:
        invalid = (df[col] <= 0).sum()
        print(f"{col} values <= 0:", invalid)

    invalid_high_low = (
        (df["High"] < df["Low"])
    ).sum()

    print("Rows where High < Low:", invalid_high_low)

    invalid_high = (
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"])
    ).sum()

    invalid_low = (
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"])
    ).sum()

    print("High inconsistent with Open/Close:", invalid_high)
    print("Low inconsistent with Open/Close:", invalid_low)

    print("Negative volume:", (df["Volume"] < 0).sum())

    # --------------------------------------------------
    # 8. RETURN AND TARGET ALIGNMENT
    # --------------------------------------------------

    section("TARGET ALIGNMENT CHECK")

    df["Return"] = df["Close"].pct_change()

    # Next trading day's close-to-close return
    df["Target_Return_1D"] = df["Return"].shift(-1)

    print(
        "\nCurrent Return at t:",
        df["Return"].iloc[1:4].round(6).tolist()
    )

    print(
        "Next-day Target at t:",
        df["Target_Return_1D"].iloc[1:4].round(6).tolist()
    )

    print(
        "\nMissing Return observations:",
        df["Return"].isna().sum()
    )

    print(
        "Missing Target observations:",
        df["Target_Return_1D"].isna().sum()
    )

    # Verify that the target is the next row's return
    expected_target = df["Return"].shift(-1)

    target_match = np.allclose(
        df["Target_Return_1D"].iloc[:-1].to_numpy(),
        expected_target.iloc[:-1].to_numpy(),
        equal_nan=True
    )

    print("Target shift alignment:", "PASS" if target_match else "FAIL")

    # --------------------------------------------------
    # 9. DATASET SUMMARY
    # --------------------------------------------------

    section("DATASET SUMMARY")

    print("Final rows:", len(df))
    print("Final columns:", len(df.columns))

    print("\nOHLCV statistical summary:")
    print(df[REQUIRED_COLUMNS].describe().round(4))

    # --------------------------------------------------
    # 10. AUTOMATIC WARNINGS
    # --------------------------------------------------

    section("QUALITY WARNINGS")

    warnings = []

    if duplicate_rows > 0:
        warnings.append("Duplicate rows detected.")

    if df[REQUIRED_COLUMNS].isna().any().any():
        warnings.append("Missing OHLCV values detected.")

    if infinite_count > 0:
        warnings.append("Infinite numeric values detected.")

    if invalid_high_low > 0:
        warnings.append("Invalid High/Low relationship detected.")

    if invalid_high > 0 or invalid_low > 0:
        warnings.append("OHLC price consistency issue detected.")

    if (df["Volume"] < 0).any():
        warnings.append("Negative volume detected.")

    if date_col is None:
        warnings.append("No recognized date column found.")

    if warnings:
        print("\nIssues requiring investigation:")
        for warning in warnings:
            print("-", warning)
    else:
        print("\nNo basic data-quality warnings detected.")

    print("\nIMPORTANT:")
    print("This script performs basic data checks.")
    print("It does not prove that every feature is free from look-ahead leakage.")
    print("Feature-generation code must be audited separately.")

    print("\nDATA QUALITY CHECK COMPLETED")


if __name__ == "__main__":
    main()
