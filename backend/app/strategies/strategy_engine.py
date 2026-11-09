
from app.strategies.moving_average import generate_signal as ma_signal

from app.strategies.momentum import generate_signal as momentum_signal

from app.strategies.mean_reversion import generate_signal as mr_signal


def run_strategy(
    df,
    strategy="momentum"
):

    if strategy == "moving_average":

        result = ma_signal(df)

    elif strategy == "momentum":

        result = momentum_signal(df)

    elif strategy == "mean_reversion":

        result = mr_signal(df)

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    return result
