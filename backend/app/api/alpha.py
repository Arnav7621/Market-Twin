
from pathlib import Path

import json
import pandas as pd

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel, Field

from app.alpha_compiler.discover_factors import discover


# =========================================
# ROUTER CONFIGURATION
# =========================================

router = APIRouter(
    prefix="/api/alpha",
    tags=["AlphaCompiler"]
)


# =========================================
# PROJECT PATHS
# =========================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

ALPHA_COMPILER_DIR = (
    PROJECT_DIR /
    "data" /
    "processed" /
    "alpha_compiler"
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
# ALPHACOMPILER DISCOVERY API
# =========================================

@router.post("/discover")
def discover_alpha(
    request: AlphaRequest
):

    ticker = request.ticker.upper().strip()

    if not ticker:

        raise HTTPException(
            status_code=400,
            detail="Ticker cannot be empty"
        )

    # -------------------------------------
    # STEP 1: VERIFY PROCESSED MARKET DATA
    # -------------------------------------

    market_data_path = (
        PROJECT_DIR /
        "data" /
        "processed" /
        f"{ticker}.csv"
    )

    if not market_data_path.exists():

        raise HTTPException(
            status_code=404,
            detail=f"Processed market data not found for {ticker}"
        )

    # -------------------------------------
    # STEP 2: RUN ACTUAL ALPHACOMPILER
    # -------------------------------------

    try:

        discover(
            ticker=ticker
        )

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
            detail=f"AlphaCompiler discovery failed: {str(exc)}"
        )

    # -------------------------------------
    # STEP 3: LOCATE GENERATED OUTPUT FILES
    # -------------------------------------

    best_factor_path = (
        ALPHA_COMPILER_DIR /
        f"{ticker}_best_factor.txt"
    )

    predictions_path = (
        ALPHA_COMPILER_DIR /
        f"{ticker}_factor_predictions.csv"
    )

    if not best_factor_path.exists():

        raise HTTPException(
            status_code=500,
            detail="AlphaCompiler completed but best factor file was not generated"
        )

    if not predictions_path.exists():

        raise HTTPException(
            status_code=500,
            detail="AlphaCompiler completed but predictions file was not generated"
        )

    # -------------------------------------
    # STEP 4: READ BEST FACTOR
    # -------------------------------------

    try:

        best_factor = best_factor_path.read_text(
            encoding="utf-8"
        ).strip()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read best factor: {str(exc)}"
        )

    # -------------------------------------
    # STEP 5: READ PREDICTIONS
    # -------------------------------------

    try:

        predictions_df = pd.read_csv(
            predictions_path
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read factor predictions: {str(exc)}"
        )

    # Convert predictions to JSON-safe records.
    # NaN values are converted to JSON null.

    predictions_json = json.loads(
        predictions_df.to_json(
            orient="records"
        )
    )

    # -------------------------------------
    # STEP 6: RETURN API RESPONSE
    # -------------------------------------

    return {

        "status": "success",

        "ticker": ticker,

        "message": (
            "AlphaCompiler factor discovery completed successfully"
        ),

        "factor_expression": best_factor,

        "predictions": predictions_json,

        "prediction_summary": {

            "total_predictions": len(
                predictions_df
            ),

            "columns": list(
                predictions_df.columns
            ),

            "preview": predictions_json[:10]

        },

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
