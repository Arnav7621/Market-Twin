
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BACKEND_DIR.parent
    / "data"
    / "processed"
    / "AAPL.csv"
)

OUTPUT_DIR = (
    BACKEND_DIR.parent
    / "data"
    / "processed"
    / "alpha_compiler"
)

OUTPUT_PATH = OUTPUT_DIR / "AAPL_features.csv"


# --------------------------------------------------
# RSI
# --------------------------------------------------

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # Handle periods with gains but no losses
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    return rsi


# --------------------------------------------------
# MACD
# --------------------------------------------------

def calculate_macd(series):

    ema_12 = series.ewm(
        span=12,
        adjust=False,
        min_periods=12
    ).mean()

    ema_26 = series.ewm(
        span=26,
        adjust=False,
        min_periods=26
    ).mean()

    macd = ema_12 - ema_26

    signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def create_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------
    # 1. RETURNS
    # --------------------------------------------------

    df["Return"] = df["Close"].pct_change()

    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # --------------------------------------------------
    # 2. MOMENTUM
    # --------------------------------------------------

    for period in [5, 10, 20, 60]:

        df[f"Momentum_{period}D"] = (
            df["Close"].pct_change(period)
        )

    # --------------------------------------------------
    # 3. VOLATILITY
    # --------------------------------------------------

    df["Volatility_5D"] = (
        df["Return"].rolling(5).std()
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Volatility_60D"] = (
        df["Return"].rolling(60).std()
    )

    df["Volatility_Ratio"] = (
        df["Volatility_5D"]
        / df["Volatility_20D"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # 4. RSI
    # --------------------------------------------------

    df["RSI_14"] = calculate_rsi(
        df["Close"],
        period=14
    )

    # --------------------------------------------------
    # 5. MACD
    # --------------------------------------------------

    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Histogram"]
    ) = calculate_macd(df["Close"])

    # --------------------------------------------------
    # 6. MOVING AVERAGES
    # --------------------------------------------------

    df["SMA_20"] = df["Close"].rolling(20).mean()

    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["SMA_100"] = df["Close"].rolling(100).mean()

    df["Distance_From_SMA_20"] = (
        df["Close"] / df["SMA_20"] - 1
    )

    df["Distance_From_SMA_50"] = (
        df["Close"] / df["SMA_50"] - 1
    )

    df["Distance_From_SMA_100"] = (
        df["Close"] / df["SMA_100"] - 1
    )

    # --------------------------------------------------
    # 7. ATR - AVERAGE TRUE RANGE
    # --------------------------------------------------

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = true_range.rolling(14).mean()

    df["ATR_Percent"] = (
        df["ATR_14"] / df["Close"]
    )

    # --------------------------------------------------
    # 8. VOLUME FEATURES
    # --------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    volume_mean = df["Volume"].rolling(20).mean()

    volume_std = df["Volume"].rolling(20).std()

    df["Volume_ZScore_20D"] = (
        (df["Volume"] - volume_mean)
        / volume_std.replace(0, np.nan)
    )

    # --------------------------------------------------
    # 9. PRICE RANGE
    # --------------------------------------------------

    df["Price_Range"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    df["Open_Close_Return"] = (
        df["Close"] / df["Open"] - 1
    )

    # --------------------------------------------------
    # 10. FUTURE TARGETS
    # --------------------------------------------------

    # Close-to-close future returns.
    # These are labels, NOT input features.

    for horizon in [1, 5, 10]:

        df[f"Target_Return_{horizon}D"] = (
            df["Close"].shift(-horizon)
            / df["Close"] - 1
        )

    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("ALPHACOMPILER ADVANCED FEATURE ENGINEERING")
    print("=" * 65)

    if not INPUT_PATH.exists():

        print("\nERROR: Input dataset not found:")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        print("\nMissing required columns:", missing)
        return

    for col in required[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(subset=required)

    df = df.sort_values("Date").reset_index(drop=True)

    print("\nInput observations:", len(df))

    df = create_features(df)

    # --------------------------------------------------
    # FEATURE SUMMARY
    # --------------------------------------------------

    target_columns = [
        "Target_Return_1D",
        "Target_Return_5D",
        "Target_Return_10D"
    ]

    excluded_columns = [
        "Date",
        "Target_Return_1D",
        "Target_Return_5D",
        "Target_Return_10D"
    ]

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    print("\nNumber of features:", len(feature_columns))

    print("\nFeature columns:")

    for feature in feature_columns:
        print("-", feature)

    # --------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------

    print("\nMissing values per feature:")

    print(df[feature_columns].isna().sum())

    print("\nInfinite values per feature:")

    print(
        pd.Series(
            np.isinf(
                df[feature_columns].to_numpy(dtype=float)
            ).sum(axis=0),
            index=feature_columns
        )
    )

    # --------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 65)

    print("FEATURE ENGINEERING COMPLETED")

    print("\nOutput file:")
    print(OUTPUT_PATH)

    print("\nTotal observations:", len(df))

    print("Total columns:", len(df.columns))

    print("\nTarget columns:")

    for target in target_columns:
        print("-", target)

    print("\nNOTE:")
    print("Initial rolling-window NaNs are expected.")
    print("Final target NaNs are expected.")
    print("Do not fill future targets with zero.")

    print("=" * 65)


if __name__ == "__main__":
    main()
