
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss
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


def train_model(horizon):

    df = pd.read_csv(DATA_FILE)

    target = f"Fill_{horizon}s"

    X = df[FEATURES]

    y = df[target]

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

    val_prob = model.predict_proba(X_val)[:, 1]

    test_prob = model.predict_proba(X_test)[:, 1]

    test_pred = model.predict(X_test)

    print(f"\n========== {horizon}s MODEL ==========")

    print(
        "Validation ROC-AUC:",
        roc_auc_score(y_val, val_prob)
    )

    print(
        "Test ROC-AUC:",
        roc_auc_score(y_test, test_prob)
    )

    print(
        "Test Accuracy:",
        accuracy_score(y_test, test_pred)
    )

    print(
        "Test Brier Score:",
        brier_score_loss(y_test, test_prob)
    )

    # Save model
    model_file = (
        MODEL_DIR / f"fill_model_{horizon}s.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print("Saved:", model_file)


if __name__ == "__main__":

    train_model(1)

    train_model(5)

    train_model(10)

    print("\nALL FILL MODELS TRAINED!")
