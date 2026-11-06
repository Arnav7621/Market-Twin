
# MarketTwin Week 1 Report

## 1. Dataset

Historical daily OHLCV data for 12 US stocks and ETFs.

## 2. Data Pipeline

- Historical data downloaded.
- Dates parsed and validated.
- Duplicate observations removed.
- Invalid prices removed.
- Simple returns calculated.
- Log returns calculated.
- Cumulative returns calculated.

## 3. Risk Metrics

- Annualized volatility.
- Sharpe ratio.
- Sortino ratio.
- Maximum drawdown.
- Historical VaR.
- Historical CVaR.

## 4. Market Regime Classification

Rule-based regime classification based on:

- Rolling volatility.
- Rolling trading volume.
- Rolling returns.

## 5. Baseline Model

Logistic Regression predicts the next session's return direction.

## 6. Evaluation

Metrics:

- Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.

## 7. Limitations

Daily OHLCV data does not contain order book depth,
queue position, or actual execution events.

The baseline is a prediction model and not a validated
profitable trading strategy.
