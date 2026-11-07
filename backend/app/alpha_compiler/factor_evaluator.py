
import numpy as np
import pandas as pd


def add(a, b):

    return a + b


def subtract(a, b):

    return a - b


def multiply(a, b):

    return a * b


def protected_divide(a, b):

    denominator = np.where(
        np.abs(b) < 1e-8,
        np.nan,
        b
    )

    return a / denominator


def rolling_mean(series, window=20):

    return series.rolling(
        window=window
    ).mean()


def rolling_std(series, window=20):

    return series.rolling(
        window=window
    ).std()


def rank(series, window=20):

    return series.rolling(
        window=window
    ).apply(
        lambda x: pd.Series(x).rank(
            pct=True
        ).iloc[-1],
        raw=False
    )


def lag(series, periods=1):

    return series.shift(periods)


def calculate_example_factor(df):

    momentum = df["Close"].pct_change(20)

    imbalance = (
        df["Volume"].pct_change()
    )

    volatility = (
        df["Return"].rolling(20).std()
    )

    factor = protected_divide(
        add(
            momentum,
            imbalance
        ),
        volatility
    )

    return factor


if __name__ == "__main__":

    df = pd.read_csv(
        "../../../data/processed/AAPL.csv"
    )

    factor = calculate_example_factor(df)

    print("\nExample Alpha Factor:")

    print(factor.tail(10))
