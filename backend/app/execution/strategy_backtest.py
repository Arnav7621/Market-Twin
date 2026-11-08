
from pathlib import Path

import numpy as np
import pandas as pd

from execution_simulator import (
    simulate_execution
)


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "backtests"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_factor(df):

    df = df.copy()

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Alpha_Factor"] = (
        df["Momentum_5D"] /
        (
            df["Volatility_20D"] + 1e-8
        )
    )

    return df


def generate_signal(factor):

    if factor > 0:

        return "buy"

    elif factor < 0:

        return "sell"

    return "hold"


def run_backtest(
    ticker="AAPL",
    order_quantity=10,
    fill_probability=0.80,
    seed=42
):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    df = calculate_factor(
        df
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Alpha_Factor",
            "Return",
            "Close"
        ]
    )

    rng = np.random.default_rng(
        seed
    )

    results = []

    for i in range(
        len(df) - 1
    ):

        row = df.iloc[i]

        next_row = df.iloc[i + 1]

        signal = generate_signal(
            row["Alpha_Factor"]
        )

        if signal == "hold":

            continue

        entry_price = float(
            row["Close"]
        )

        next_price = float(
            next_row["Close"]
        )

        execution = simulate_execution(
            market_price=entry_price,
            side=signal,
            order_quantity=order_quantity,
            fill_probability=fill_probability,
            spread_bps=5,
            slippage_bps=2,
            fee_rate=0.001,
            rng=rng
        )

        if not execution["filled"]:

            results.append({

                "Date": row["Date"],

                "Signal": signal,

                "Filled": False,

                "Quantity": 0,

                "Execution_Price": np.nan,

                "Gross_PnL": 0.0,

                "Transaction_Cost": 0.0,

                "Slippage_Cost": 0.0,

                "Net_PnL": 0.0

            })

            continue

        quantity = (
            execution["filled_quantity"]
        )

        execution_price = (
            execution["execution_price"]
        )

        if signal == "buy":

            gross_pnl = (
                next_price -
                execution_price
            ) * quantity

        else:

            gross_pnl = (
                execution_price -
                next_price
            ) * quantity

        net_pnl = (
            gross_pnl -
            execution["transaction_cost"]
        )

        results.append({

            "Date": row["Date"],

            "Signal": signal,

            "Filled": True,

            "Quantity": quantity,

            "Execution_Price": execution_price,

            "Gross_PnL": gross_pnl,

            "Transaction_Cost": (
                execution["transaction_cost"]
            ),

            "Slippage_Cost": (
                execution["slippage_cost"]
            ),

            "Net_PnL": net_pnl

        })

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR /
        f"{ticker}_backtest.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nBACKTEST COMPLETED")

    print(
        "Total trades:",
        len(results_df)
    )

    print(
        "Net PnL:",
        results_df["Net_PnL"].sum()
    )

    print(
        "Output:",
        output_file
    )

    return results_df


if __name__ == "__main__":

    run_backtest(
        ticker="AAPL",
        order_quantity=10,
        fill_probability=0.80,
        seed=42
    )
