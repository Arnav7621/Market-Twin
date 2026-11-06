
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from lifelines import CoxPHFitter


# Project root directory
PROJECT_DIR = Path(__file__).resolve().parents[4]

# Dataset path
DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "queue_mind"
    / "synthetic_fill_dataset.csv"
)

# Model saving directory
MODEL_DIR = (
    PROJECT_DIR
    / "backend"
    / "app"
    / "queue_mind"
    / "models"
    / "saved"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def train_time_to_fill():

    df = pd.read_csv(DATA_FILE)

    rng = np.random.default_rng(42)

    # Synthetic duration model
    difficulty = (
        df["Distance_From_Mid"] * 20
        + df["Order_Size"] / (df["Bid_Depth"] + 1)
        + df["Volatility"] * 10
    )

    durations = rng.exponential(
        scale=np.maximum(
            1,
            1 + difficulty * 20
        )
    )

    # Cap duration at 10 seconds
    observed = durations <= 10

    df["Duration"] = np.minimum(
        durations,
        10
    )

    df["Event"] = observed.astype(int)

    # Cox model input
    columns = [
        "Duration",
        "Event",
        "Spread",
        "Bid_Depth",
        "Ask_Depth",
        "Imbalance",
        "Volatility",
        "Order_Size",
        "Distance_From_Mid"
    ]

    training_data = df[columns]

    # Train Cox model
    cph = CoxPHFitter()

    cph.fit(
        training_data,
        duration_col="Duration",
        event_col="Event"
    )

    print("\nCOX MODEL SUMMARY")
    print(cph.summary)

    # Save trained model
    model_file = (
        MODEL_DIR / "time_to_fill_model.pkl"
    )

    joblib.dump(cph, model_file)

    print("\nSaved model:", model_file)


if __name__ == "__main__":
    train_time_to_fill()
