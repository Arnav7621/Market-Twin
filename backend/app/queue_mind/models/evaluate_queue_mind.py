
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix
)


PROJECT_DIR = Path(__file__).resolve().parents[5]

DATA_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "queue_mind" / "synthetic_fill_dataset.csv"
)

MODEL_DIR = (
    PROJECT_DIR / "backend" / "app" /
    "queue_mind" / "models" / "saved"
)


FEATURES = [
    "Spread",
    "Bid_Depth",
    "Ask_Depth",
    "Imbalance",
    "Volatility",
    "Order_Size",
    "Distance_From_Mid"
]


def evaluate(horizon):

    df = pd.read_csv(DATA_FILE)

    n = len(df)

    test_start = int(n * 0.85)

    X_test = df[FEATURES].iloc[test_start:]

    y_test = df[f"Fill_{horizon}s"].iloc[test_start:]

    model = joblib.load(
        MODEL_DIR / f"fill_model_{horizon}s.joblib"
    )

    calibrator = joblib.load(
        MODEL_DIR / f"calibrator_{horizon}s.joblib"
    )

    raw_prob = model.predict_proba(X_test)[:, 1]

    import numpy as np

    logits = np.log(
        np.clip(raw_prob, 1e-6, 1 - 1e-6) /
        np.clip(1 - raw_prob, 1e-6, 1)
    ).reshape(-1, 1)

    probabilities = calibrator.predict_proba(
        logits
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(f"\n========== {horizon}s EVALUATION ==========")

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_test,
            probabilities
        )
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "Brier Score:",
        brier_score_loss(
            y_test,
            probabilities
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":

    evaluate(1)

    evaluate(5)

    evaluate(10)
