
# MarketTwin — Week 4 Implementation Report

## 1. Execution Simulator

Implemented a simplified execution simulator supporting:

- Buy and sell orders
- Fill probability
- Execution price calculation
- Spread cost
- Slippage cost
- Transaction fees

## 2. Strategy Backtesting

Implemented a baseline factor strategy with:

- Momentum-based alpha factor
- Buy/sell signal generation
- Simulated execution
- Gross PnL
- Net PnL

## 3. Statistical Validation

Implemented:

- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Walk-forward evaluation
- Bootstrap confidence intervals

## 4. Data

The initial experiments use processed historical
daily OHLCV data.

The current implementation is a prototype and does
not use a full raw order-level market event stream.

## 5. Limitations

- Fill probability is currently a placeholder.
- Partial fills are not implemented.
- Queue position is not directly observed.
- Synthetic depth and spread are modeled proxies.
- Capital constraints are not fully modeled.
- The strategy has not been validated for live trading.

## 6. Next Steps

- Connect QueueMind fill probability.
- Implement partial fills.
- Improve portfolio accounting.
- Conduct ablation studies.
- Add statistical multiple-testing controls.
- Build the research dashboard.
