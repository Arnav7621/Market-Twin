
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[4]

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "queue_mind"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def generate_dataset(n=50000, seed=42):

    rng = np.random.default_rng(seed)

    spread = rng.uniform(0.01, 0.10, n)

    bid_depth = rng.uniform(100, 2000, n)

    ask_depth = rng.uniform(100, 2000, n)

    imbalance = (
        bid_depth - ask_depth
    ) / (
        bid_depth + ask_depth
    )

    volatility = rng.uniform(0.001, 0.05, n)

    order_size = rng.uniform(10, 500, n)

    distance_from_mid = rng.uniform(
        0,
        0.10,
        n
    )

    # Synthetic queue difficulty
    difficulty = (
        0.5 * distance_from_mid +
        0.3 * order_size / (bid_depth + 1) +
        0.2 * spread
    )

    # Relative fill scores
    score = (
        2.0 * imbalance -
        10.0 * difficulty -
        5.0 * volatility
    )

    # Horizon-dependent fill probabilities
    p1 = 1 / (
        1 + np.exp(-score)
    )

    p5 = 1 / (
        1 + np.exp(-(score + 0.8))
    )

    p10 = 1 / (
        1 + np.exp(-(score + 1.4))
    )

    # Generate synthetic outcomes
    fill_1s = rng.binomial(1, p1)

    fill_5s = rng.binomial(1, p5)

    fill_10s = rng.binomial(1, p10)

    df = pd.DataFrame({
        "Spread": spread,
        "Bid_Depth": bid_depth,
        "Ask_Depth": ask_depth,
        "Imbalance": imbalance,
        "Volatility": volatility,
        "Order_Size": order_size,
        "Distance_From_Mid": distance_from_mid,
        "Fill_1s": fill_1s,
        "Fill_5s": fill_5s,
        "Fill_10s": fill_10s
    })

    return df


if __name__ == "__main__":

    df = generate_dataset()

    output_file = (
        OUTPUT_DIR / "synthetic_fill_dataset.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("Dataset generated!")

    print("Shape:", df.shape)

    print("Saved:", output_file)

    print(df.head())
