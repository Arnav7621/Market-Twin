
import pandas as pd


def generate_signal(
    df,
    window=20,
    threshold=2
):

    df = df.copy()

    df["Mean"] = (
        df["Close"].rolling(
            window
        ).mean()
    )

    df["Std"] = (
        df["Close"].rolling(
            window
        ).std()
    )

    df["Z_Score"] = (
        df["Close"] - df["Mean"]
    ) / (
        df["Std"] + 1e-8
    )

    df["Signal"] = 0

    df.loc[
        df["Z_Score"] < -threshold,
        "Signal"
    ] = 1

    df.loc[
        df["Z_Score"] > threshold,
        "Signal"
    ] = -1

    df["Position"] = (
        df["Signal"].shift(1)
    )

    return df
