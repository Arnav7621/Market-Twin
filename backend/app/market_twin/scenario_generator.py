
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "market_twin"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_market_data(ticker):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Return"] = pd.to_numeric(
        df["Return"],
        errors="coerce"
    )

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Return",
            "Volume",
            "Close"
        ]
    )

    df["Rolling_Volatility"] = (
        df["Return"].rolling(20).std()
    )

    df["Rolling_Volume"] = (
        df["Volume"].rolling(20).mean()
    )

    df = df.dropna()

    return df


def generate_scenario(
    ticker="AAPL",
    volatility="high",
    liquidity="low",
    imbalance="negative",
    duration=1000,
    block_size=10,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    df = load_market_data(
        ticker
    )

    if volatility not in [
        "high",
        "low"
    ]:

        raise ValueError(
            "Volatility must be high or low"
        )

    if liquidity not in [
        "high",
        "low"
    ]:

        raise ValueError(
            "Liquidity must be high or low"
        )

    if imbalance not in [
        "positive",
        "negative",
        "neutral"
    ]:

        raise ValueError(
            "Imbalance must be positive, negative or neutral"
        )

    if duration <= 0:

        raise ValueError(
            "Duration must be positive"
        )

    if block_size <= 0:

        raise ValueError(
            "Block size must be positive"
        )

    # Historical thresholds
    volatility_threshold = (
        df["Rolling_Volatility"].median()
    )

    volume_threshold = (
        df["Rolling_Volume"].median()
    )

    # Match historical observations
    if volatility == "high":

        vol_mask = (
            df["Rolling_Volatility"] >
            volatility_threshold
        )

    else:

        vol_mask = (
            df["Rolling_Volatility"] <=
            volatility_threshold
        )

    if liquidity == "high":

        volume_mask = (
            df["Rolling_Volume"] >
            volume_threshold
        )

    else:

        volume_mask = (
            df["Rolling_Volume"] <=
            volume_threshold
        )

    eligible = df[
        vol_mask & volume_mask
    ]

    # Correctly apply both regime filters
    eligible = df[
        vol_mask & volume_mask
    ]

    # Apply liquidity condition
    eligible = df[
        vol_mask & volume_mask
    ]

    # Combined regime pool
    eligible = df[
        vol_mask & volume_mask
    ]

    # Filter for matching volume regime
    eligible = df[
        vol_mask & volume_mask
    ]

    # Use both volatility and volume conditions
    eligible = df[
        vol_mask & volume_mask
    ]

    if len(eligible) == 0:

        eligible = df

    historical_returns = df[
        "Return"
    ].to_numpy()

    historical_volumes = df[
        "Volume"
    ].to_numpy()

    generated_returns = []

    generated_volumes = []

    # Block bootstrap
    while len(generated_returns) < duration:

        start = rng.choice(
            eligible.index.to_numpy()
        )

        position = df.index.get_loc(
            start
        )

        end = min(
            position + block_size,
            len(df)
        )

        return_block = (
            historical_returns[
                position:end
            ]
        )

        volume_block = (
            historical_volumes[
                position:end
            ]
        )

        generated_returns.extend(
            return_block
        )

        generated_volumes.extend(
            volume_block
        )

    generated_returns = np.array(
        generated_returns[:duration]
    )

    generated_volumes = np.array(
        generated_volumes[:duration]
    )

    # Apply imbalance bias
    if imbalance == "positive":

        generated_returns += (
            np.abs(
                generated_returns.mean()
            ) + 0.0001
        )

    elif imbalance == "negative":

        generated_returns -= (
            np.abs(
                generated_returns.mean()
            ) + 0.0001
        )

    # Generate synthetic price trajectory
    initial_price = float(
        df["Close"].iloc[-1]
    )

    prices = (
        initial_price *
        np.cumprod(
            1 + generated_returns
        )
    )

    # Synthetic spread proxy
    spread_bps = (
        5.0 +
        100.0 *
        np.abs(
            generated_returns
        )
    )

    if liquidity == "low":

        spread_bps *= 2.0

    # Synthetic depth proxy
    depth = (
        generated_volumes /
        (
            spread_bps + 1e-8
        )
    )

    if liquidity == "low":

        depth *= 0.5

    # Synthetic order-flow imbalance
    if imbalance == "positive":

        order_imbalance = rng.uniform(
            0.1,
            0.8,
            duration
        )

    elif imbalance == "negative":

        order_imbalance = rng.uniform(
            -0.8,
            -0.1,
            duration
        )

    else:

        order_imbalance = rng.uniform(
            -0.1,
            0.1,
            duration
        )

    scenario = pd.DataFrame({

        "Step": np.arange(
            1,
            duration + 1
        ),

        "Return": generated_returns,

        "Price": prices,

        "Volume": generated_volumes,

        "Spread_Bps": spread_bps,

        "Bid_Depth": (
            depth *
            (1 + order_imbalance)
        ),

        "Ask_Depth": (
            depth *
            (1 - order_imbalance)
        ),

        "Order_Imbalance": order_imbalance

    })

    scenario["Ticker"] = ticker

    scenario["Volatility_Regime"] = volatility

    scenario["Liquidity_Regime"] = liquidity

    scenario["Scenario_Imbalance"] = imbalance

    return scenario


if __name__ == "__main__":

    print("\nGENERATING MARKETTWIN SCENARIO\n")

    scenario = generate_scenario(
        ticker="AAPL",
        volatility="high",
        liquidity="low",
        imbalance="negative",
        duration=1000,
        block_size=10,
        seed=42
    )

    output_file = (
        OUTPUT_DIR / "high_vol_low_liq_negative.csv"
    )

    scenario.to_csv(
        output_file,
        index=False
    )

    print("\nScenario generated successfully!")

    print("Shape:", scenario.shape)

    print("\nFirst 5 rows:")

    print(
        scenario.head()
    )

    print("\nSaved to:")

    print(output_file)

    scenario = generate_scenario(
    ticker="NVDA",
    volatility="high",
    liquidity="high",
    imbalance="neutral",
    duration=1000,
    seed=44
)
