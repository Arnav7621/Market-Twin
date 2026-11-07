
# MarketTwin — Synthetic Market Scenario Generator

## Objective

Generate statistical market scenarios conditioned on
specified volatility, liquidity, and imbalance regimes.

## Inputs

- Ticker
- Volatility regime
- Liquidity regime
- Order-flow imbalance
- Scenario duration
- Bootstrap block size
- Random seed

## Outputs

- Simulated returns
- Simulated price trajectory
- Simulated trading volume
- Synthetic spread proxy
- Synthetic bid depth proxy
- Synthetic ask depth proxy
- Synthetic order-flow imbalance

## Methodology

MarketTwin uses historical return and volume data
with conditional block bootstrap sampling.

The generator selects historical blocks matching
the requested volatility and volume regimes.

Spread, depth, and order-flow imbalance are generated
using explicit simulation assumptions.

## Limitations

The current generator is based on daily OHLCV data.

It does not contain historical order IDs, queue positions,
or actual bid/ask depth observations.

The generated spread and depth variables are proxies.

The scenarios are statistical research simulations
and must not be interpreted as exchange-grade
counterfactual order-book reconstructions.

## Reproducibility

The generator accepts a random seed.

Identical inputs and seeds should produce
reproducible outputs.
