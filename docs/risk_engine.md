
# MarketTwin Risk Engine

## Overview

The MarketTwin Risk Engine calculates historical risk metrics
using daily adjusted stock returns.

## Data Source

Historical OHLCV data is retrieved using yfinance.

## Metrics

### Annualized Volatility

Measures annualized variability of daily returns.

Formula:

sigma_annual = sigma_daily * sqrt(252)

### Sharpe Ratio

Measures annualized excess return relative to volatility.

### Sortino Ratio

Measures annualized excess return relative to downside deviation.

### Maximum Drawdown

Measures the maximum peak-to-trough decline in cumulative wealth.

### Value at Risk

Historical VaR estimates a loss threshold at a selected
confidence level.

MarketTwin currently uses 95% confidence.

### Conditional Value at Risk

CVaR estimates the average loss in the worst 5% of
historical return observations.

## Assumptions

- Daily return observations are used.
- 252 trading days are assumed for annualization.
- Annual risk-free rate is assumed to be 4%.
- Historical VaR and CVaR are calculated using the empirical
  distribution of daily returns.
- Risk metrics are historical estimates and are not forecasts.
