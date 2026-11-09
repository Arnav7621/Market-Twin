
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import BaseModel

from app.market_twin.scenario_generator import (
    generate_scenario
)
from app.api.backtest import router as backtest_router
from app.api.alpha import router as alpha_router

from app.api.queue_mind import router as queue_mind_router

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

OUTPUT_DIR = (
    DATA_DIR /
    "market_twin"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


app = FastAPI(
    title="MarketTwin API",
    description="Quantitative market research platform",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ScenarioRequest(BaseModel):

    ticker: str = "AAPL"

    volatility: str = "high"

    liquidity: str = "low"

    imbalance: str = "negative"

    duration: int = 1000

    seed: int = 42


@app.get("/")
def root():

    return {
        "message": "MarketTwin API running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/api/market-data/{ticker}")
def get_market_data(
    ticker: str
):

    ticker = ticker.upper()

    file_path = (
        DATA_DIR /
        f"{ticker}.csv"
    )

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Ticker data not found"
        )

    df = pd.read_csv(
        file_path
    )

    df = df.tail(
        200
    )

    return {
        "ticker": ticker,
        "data": df.to_dict(
            orient="records"
        )
    }


@app.post("/api/scenario")
def create_scenario(
    request: ScenarioRequest
):

    if request.duration > 5000:

        raise HTTPException(
            status_code=400,
            detail="Duration cannot exceed 5000"
        )

    try:

        scenario = generate_scenario(
            ticker=request.ticker,
            volatility=request.volatility,
            liquidity=request.liquidity,
            imbalance=request.imbalance,
            duration=request.duration,
            seed=request.seed
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    output_file = (
        OUTPUT_DIR /
        (
            f"{request.ticker}_dashboard_scenario.csv"
        )
    )

    scenario.to_csv(
        output_file,
        index=False
    )

    return {
        "message": "Scenario generated",
        "ticker": request.ticker,
        "rows": len(scenario),
        "data": scenario.to_dict(
            orient="records"
        )
    }


@app.get("/api/experiments")
def get_experiments():

    file_path = (
        DATA_DIR /
        "experiments" /
        "AAPL_ablation.csv"
    )

    if not file_path.exists():

        return {
            "message": "No experiments found",
            "data": []
        }

    df = pd.read_csv(
        file_path
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


@app.get("/api/metrics")
def get_metrics():

    file_path = (
        DATA_DIR /
        "processed" /
        "validation" /
        "AAPL_metrics.csv"
    )

    # Correct path: DATA_DIR already points
    # to data/processed.
    file_path = (
        DATA_DIR /
        "validation" /
        "AAPL_metrics.csv"
    )

    if not file_path.exists():

        return {
            "message": "Metrics not found",
            "data": []
        }

    df = pd.read_csv(
        file_path
    )

    return {
        "data": df.to_dict(
            orient="records"
        )
    }


app.include_router(backtest_router)
app.include_router(
    alpha_router
)

app.include_router(
    queue_mind_router
)