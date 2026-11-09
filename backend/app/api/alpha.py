
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/alpha",
    tags=["AlphaCompiler"]
)


# =========================================
# PROJECT PATHS
# =========================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

ALPHA_COMPILER_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "alpha_compiler"
)


# =========================================
# REQUEST MODEL
# =========================================

class AlphaRequest(BaseModel):

    ticker: str = "AAPL"

    population_size: int = Field(
        default=50,
        ge=2,
        le=500
    )

    generations: int = Field(
        default=10,
        ge=1,
        le=100
    )

    seed: int = 42


# =========================================
# ALPHACOMPILER API
# =========================================

@router.post("/discover")
def discover_alpha(request: AlphaRequest):

    ticker = request.ticker.upper().strip()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker cannot be empty"
        )

    market_data_path = (
        PROJECT_DIR
        / "data"
        / "processed"
        / f"{ticker}.csv"
    )

    if not market_data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Processed market data not found for {ticker}"
        )

    try:

        # Import actual AlphaCompiler discovery function
        from app.alpha_compiler.discover_factors import discover

        # Run actual factor discovery
        discover(ticker=ticker)

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"AlphaCompiler failed: {str(exc)}"
        )

    # =====================================
    # LOAD GENERATED FILES
    # =====================================

    best_factor_path = (
        ALPHA_COMPILER_DIR
        / f"{ticker}_best_factor.txt"
    )

    predictions_path = (
        ALPHA_COMPILER_DIR
        / f"{ticker}_factor_predictions.csv"
    )

    if not best_factor_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Best factor file was not generated"
        )

    if not predictions_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Factor predictions CSV was not generated"
        )

    try:

        best_factor = best_factor_path.read_text(
            encoding="utf-8"
        ).strip()

        predictions_df = pd.read_csv(
            predictions_path
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read AlphaCompiler outputs: {str(exc)}"
        )

    # =====================================
    # VALIDATE PREDICTION COLUMNS
    # =====================================

    required_columns = [
        "Actual_Return",
        "Predicted_Return"
    ]

    for column in required_columns:

        if column not in predictions_df.columns:

            raise HTTPException(
                status_code=500,
                detail=f"Missing prediction column: {column}"
            )

    # =====================================
    # CALCULATE METRICS
    # =====================================

    valid_df = predictions_df[
        ["Actual_Return", "Predicted_Return"]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(valid_df) == 0:

        raise HTTPException(
            status_code=500,
            detail="No valid predictions available for evaluation"
        )

    actual = valid_df["Actual_Return"].to_numpy()

    predicted = valid_df["Predicted_Return"].to_numpy()

    errors = actual - predicted

    mae = float(
        np.mean(np.abs(errors))
    )

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    # Pearson IC
    if (
        np.std(actual) > 0
        and np.std(predicted) > 0
    ):

        information_coefficient = float(
            np.corrcoef(actual, predicted)[0, 1]
        )

    else:

        information_coefficient = None

    # =====================================
    # PREVIEW ONLY 10 PREDICTIONS
    # =====================================

    preview_df = predictions_df.head(10).copy()

    # Convert Date column into JSON-safe strings
    if "Date" in preview_df.columns:

        preview_df["Date"] = (
            preview_df["Date"].astype(str)
        )

    preview = preview_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).to_dict(
        orient="records"
    )

    # =====================================
    # API RESPONSE
    # =====================================

    return {

        "status": "success",

        "ticker": ticker,

        "message": (
            "AlphaCompiler factor discovery completed successfully"
        ),

        "factor_expression": best_factor,

        "metrics": {

            "MAE": mae,

            "RMSE": rmse,

            "Information_Coefficient": (
                information_coefficient
            ),

            "Total_Predictions": len(valid_df)

        },

        "prediction_preview": preview,

        "output_files": {

            "best_factor": str(
                best_factor_path.relative_to(
                    PROJECT_DIR
                )
            ),

            "predictions": str(
                predictions_path.relative_to(
                    PROJECT_DIR
                )
            )

        }

    }
