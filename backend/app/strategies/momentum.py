
import pandas as pd


def generate_signal(
    df,
    window=5
):

    df = df.copy()

    df["Momentum"] = (
        df["Close"].pct_change(
            window
        )
    )

    df["Signal"] = 0

    df.loc[
        df["Momentum"] > 0,
        "Signal"
    ] = 1

    df.loc[
        df["Momentum"] < 0,
        "Signal"
    ] = -1

    df["Position"] = (
        df["Signal"].shift(1)
    )

    return df
