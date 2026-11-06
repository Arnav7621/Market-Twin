
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


PROJECT_DIR = Path(__file__).resolve().parents[4]

MODEL_DIR = (
    PROJECT_DIR
    / "backend"
    / "app"
    / "queue_mind"
    / "models"
    / "saved"
)

router = APIRouter(
    prefix="/queue-mind",
    tags=["QueueMind"]
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


class FillRequest(BaseModel):

    horizon: Literal[1, 5, 10] = 1

    Spread: float = Field(gt=0)

    Bid_Depth: float = Field(gt=0)

    Ask_Depth: float = Field(gt=0)

    Imbalance: float = Field(ge=-1, le=1)

    Volatility: float = Field(ge=0)

    Order_Size: float = Field(gt=0)

    Distance_From_Mid: float = Field(ge=0)


@router.post("/predict-fill")
def predict_fill(request: FillRequest):

    horizon = request.horizon

    model_file = (
        MODEL_DIR / f"fill_model_{horizon}s.joblib"
    )

    calibrator_file = (
        MODEL_DIR / f"calibrator_{horizon}s.joblib"
    )

    if not model_file.exists() or not calibrator_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Fill models are not trained yet."
        )

    model = joblib.load(model_file)

    calibrator = joblib.load(calibrator_file)

    data = request.model_dump(
        exclude={"horizon"}
    )

    X = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    probability = model.predict_proba(X)[0, 1]

    logit = np.log(
        np.clip(probability, 1e-6, 1 - 1e-6)
        /
        np.clip(1 - probability, 1e-6, 1)
    )

    calibrated_probability = calibrator.predict_proba(
        [[logit]]
    )[0, 1]

    return {
        "horizon_seconds": horizon,
        "fill_probability": round(
            float(calibrated_probability),
            4
        ),
        "model": "RandomForest + Logistic Calibration"
    }
