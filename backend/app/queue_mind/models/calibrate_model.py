
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


PROJECT_DIR = Path(__file__).resolve().parents[4]

DATA_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "queue_mind" / "synthetic_fill_dataset.csv"
)

MODEL_DIR = (
    PROJECT_DIR / "backend" / "app" /
    "queue_mind" / "models" / "saved"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
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


def calibrate(horizon):

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]

    y = df[f"Fill_{horizon}s"]

    n = len(df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end]

    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]

    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]

    y_test = y.iloc[val_end:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # Validation probabilities
    val_prob = model.predict_proba(X_val)[:, 1]

    # Convert probabilities to logits
    val_logits = np.log(
        np.clip(val_prob, 1e-6, 1 - 1e-6) /
        np.clip(1 - val_prob, 1e-6, 1)
    ).reshape(-1, 1)

    calibrator = LogisticRegression()

    calibrator.fit(
        val_logits,
        y_val
    )

    # Test probabilities
    test_prob = model.predict_proba(X_test)[:, 1]

    test_logits = np.log(
        np.clip(test_prob, 1e-6, 1 - 1e-6) /
        np.clip(1 - test_prob, 1e-6, 1)
    ).reshape(-1, 1)

    calibrated_prob = calibrator.predict_proba(
        test_logits
    )[:, 1]

    print(f"\n{horizon}s Calibration")

    print(
        "Uncalibrated Brier Score:",
        brier_score_loss(
            y_test,
            test_prob
        )
    )

    print(
        "Calibrated Brier Score:",
        brier_score_loss(
            y_test,
            calibrated_prob
        )
    )

    model_file = (
        MODEL_DIR / f"fill_model_{horizon}s.joblib"
    )

    calibrator_file = (
        MODEL_DIR / f"calibrator_{horizon}s.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        calibrator,
        calibrator_file
    )

    print("Calibration complete!")


if __name__ == "__main__":

    calibrate(1)

    calibrate(5)

    calibrate(10)
