
from pathlib import Path
import numpy as np
import pandas as pd

from fitness import calculate_fitness


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "alpha_compiler"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def evaluate_factor(ticker="AAPL"):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Target"] = df["Return"].shift(-1)

    df["Example_Factor"] = (
        df["Momentum_5D"] /
        (df["Volatility_20D"] + 1e-8)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Example_Factor",
            "Target"
        ]
    )

    n = len(df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    # Evaluate only on validation observations
    validation = df.iloc[
        train_end:val_end
    ]

    result = calculate_fitness(
        factor=validation["Example_Factor"],
        future_returns=validation["Target"],
        expression="protected_division(Momentum_5D, Volatility_20D)"
    )

    print("\nVALIDATION FITNESS")

    print(result)

    pd.DataFrame(
        [result]
    ).to_csv(
        OUTPUT_DIR / f"{ticker}_fitness.csv",
        index=False
    )


if __name__ == "__main__":

    evaluate_factor("AAPL")
