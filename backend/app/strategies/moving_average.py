
import pandas as pd


def generate_signal(
    df,
    short_window=5,
    long_window=20
):

    df = df.copy()

    df["Short_MA"] = (
        df["Close"].rolling(
            short_window
        ).mean()
    )

    df["Long_MA"] = (
        df["Close"].rolling(
            long_window
        ).mean()
    )

    df["Signal"] = 0

    df.loc[
        df["Short_MA"] > df["Long_MA"],
        "Signal"
    ] = 1

    df.loc[
        df["Short_MA"] < df["Long_MA"],
        "Signal"
    ] = -1

    # Trade using the previous day's signal
    df["Position"] = (
        df["Signal"].shift(1)
    )

    return df
