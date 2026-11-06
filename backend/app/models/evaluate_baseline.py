
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed" / "baseline"


def evaluate(ticker):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}_predictions.csv"
    )

    y_true = df["Target"]

    y_pred = df["Predicted_Direction"]

    y_prob = df["Predicted_Probability"]

    metrics = {
        "Ticker": ticker,

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            y_prob
        )
    }

    print("\nEVALUATION:")

    print(metrics)

    print("\nCONFUSION MATRIX:")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    return metrics


if __name__ == "__main__":

    result = evaluate("AAPL")

    pd.DataFrame([result]).to_csv(
        DATA_DIR / "evaluation.csv",
        index=False
    )
