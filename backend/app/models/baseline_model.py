
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline(ticker):

    df = pd.read_csv(DATA_DIR / f"{ticker}.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    # Features known at time t
    df["Return_5D"] = df["Close"].pct_change(5)

    df["Return_20D"] = df["Close"].pct_change(20)

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Volume_Change"] = df["Volume"].pct_change()

    # Target: next day's direction
    df["Target"] = (
        df["Return"].shift(-1) > 0
    ).astype(int)

    features = [
        "Return",
        "Return_5D",
        "Return_20D",
        "Volatility_20D",
        "Volume_Change"
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=features + ["Target"]
    )

    X = df[features]

    y = df["Target"]

    n = len(df)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end]

    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]

    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]

    y_test = y.iloc[val_end:]

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    # Validation performance
    val_predictions = model.predict(X_val)

    print("\nVALIDATION ACCURACY:")

    print(
        accuracy_score(
            y_val,
            val_predictions
        )
    )

    # Test performance
    test_predictions = model.predict(X_test)

    print("\nTEST ACCURACY:")

    print(
        accuracy_score(
            y_test,
            test_predictions
        )
    )

    print("\nCLASSIFICATION REPORT:")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print("\nCONFUSION MATRIX:")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    # Save test predictions
    results = df.iloc[val_end:].copy()

    results["Predicted_Direction"] = test_predictions

    results["Predicted_Probability"] = (
        model.predict_proba(X_test)[:, 1]
    )

    results.to_csv(
        OUTPUT_DIR / f"{ticker}_predictions.csv",
        index=False
    )

    return model


if __name__ == "__main__":

    model = train_baseline("AAPL")
