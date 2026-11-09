from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/queue-mind",
    tags=["QueueMind"],
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "queue_mind"
    / "synthetic_fill_dataset.csv"
)

MODEL_DIR = (
    PROJECT_DIR
    / "backend"
    / "app"
    / "queue_mind"
    / "models"
    / "saved"
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

FEATURES = [
    "Spread",
    "Bid_Depth",
    "Ask_Depth",
    "Imbalance",
    "Volatility",
    "Order_Size",
    "Distance_From_Mid",
]

HORIZONS = [1, 5, 10]


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class QueueMindRequest(BaseModel):
    ticker: str = Field(default="AAPL", min_length=1, max_length=10)
    order_side: Literal["buy", "sell"] = "buy"
    order_quantity: int = Field(default=10, ge=1, le=500)
    seed: int = Field(default=42, ge=0)


# ---------------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------------

def predict_fill_probability(
    model,
    calibrator,
    features: pd.DataFrame,
) -> float:

    raw_probability = model.predict_proba(features)[0][1]

    # Convert probability into log-odds for calibration.
    raw_probability = np.clip(
        raw_probability,
        1e-6,
        1 - 1e-6,
    )

    logit = np.log(
        raw_probability / (1 - raw_probability)
    )

    calibrated_probability = calibrator.predict_proba(
        np.array([[logit]])
    )[0][1]

    return float(
        np.clip(calibrated_probability, 0, 1)
    )


# ---------------------------------------------------------
# API ENDPOINT
# ---------------------------------------------------------

@router.post("/predict")
def predict_fill(request: QueueMindRequest):

    try:
        # Load synthetic market scenarios.
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Dataset not found: {DATA_FILE}",
            )

        df = pd.read_csv(DATA_FILE)

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Synthetic fill dataset is empty.",
            )

        # Choose a deterministic scenario using the seed.
        rng = np.random.default_rng(request.seed)

        selected_index = int(
            rng.integers(0, len(df))
        )

        scenario = df.iloc[selected_index][FEATURES].copy()

        # Use the requested order quantity.
        scenario["Order_Size"] = float(
            request.order_quantity
        )

        # Convert selected scenario to model input.
        feature_df = pd.DataFrame(
            [scenario],
            columns=FEATURES,
        )

        predictions = {}

        for horizon in HORIZONS:

            model_path = (
                MODEL_DIR
                / f"fill_model_{horizon}s.joblib"
            )

            calibrator_path = (
                MODEL_DIR
                / f"calibrator_{horizon}s.joblib"
            )

            if not model_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"Missing model: {model_path.name}",
                )

            if not calibrator_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"Missing calibrator: {calibrator_path.name}",
                )

            model = joblib.load(model_path)
            calibrator = joblib.load(calibrator_path)

            probability = predict_fill_probability(
                model,
                calibrator,
                feature_df,
            )

            predictions[f"{horizon}s"] = round(
                probability,
                4,
            )

        return {
            "status": "success",
            "ticker": request.ticker.upper(),
            "order_side": request.order_side,
            "order_quantity": request.order_quantity,
            "seed": request.seed,
            "scenario": {
                "Spread": round(float(scenario["Spread"]), 6),
                "Bid_Depth": round(float(scenario["Bid_Depth"]), 2),
                "Ask_Depth": round(float(scenario["Ask_Depth"]), 2),
                "Imbalance": round(float(scenario["Imbalance"]), 4),
                "Volatility": round(float(scenario["Volatility"]), 6),
                "Order_Size": request.order_quantity,
                "Distance_From_Mid": round(
                    float(scenario["Distance_From_Mid"]),
                    6,
                ),
            },
            "fill_probabilities": predictions,
            "model": "RandomForest + Logistic Calibration",
            "data_source": "Synthetic market scenario",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"QueueMind prediction failed: {str(e)}",
        )