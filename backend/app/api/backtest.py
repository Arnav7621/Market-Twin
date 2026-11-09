
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException

from app.strategies.backtest_engine import (
    run_backtest,
    calculate_summary
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

OUTPUT_DIR = (
    DATA_DIR /
    "backtests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


router = APIRouter(
    prefix="/api/backtest",
    tags=["Backtesting"]
)


def clean_json(value):

    if isinstance(value, dict):

        return {
            key: clean_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):

        return [
            clean_json(item)
            for item in value
        ]

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (np.floating, float)):

        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(value, np.bool_):

        return bool(value)

    return value


@router.get("/{ticker}/{strategy}")
def backtest(
    ticker: str,
    strategy: str
):

    ticker = ticker.upper()

    file_path = (
        DATA_DIR /
        f"{ticker}.csv"
    )

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Ticker not found"
        )

    try:

        df = pd.read_csv(file_path)

        result = run_backtest(
            df,
            strategy=strategy
        )

        summary = calculate_summary(
            result
        )

        output_file = (
            OUTPUT_DIR /
            f"{ticker}_{strategy}_backtest.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        return clean_json({

            "ticker": ticker,

            "strategy": strategy,

            "summary": summary,

            "data": result.tail(
                200
            ).to_dict(
                orient="records"
            )

        })

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )