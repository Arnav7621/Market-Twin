
# MarketTwin Week 2 Report — QueueMind

## Objective

Estimate limit order fill probability and time-to-fill.

## Models

- Random Forest fill classifier.
- Cox Proportional Hazards time-to-fill model.
- Sigmoid calibration model.

## Horizons

- 1 second.
- 5 seconds.
- 10 seconds.

## Input Features

- Spread.
- Bid depth.
- Ask depth.
- Order book imbalance.
- Volatility.
- Order size.
- Distance from midpoint.

## API

POST /predict-fill

## Evaluation

- ROC-AUC.
- Precision.
- Recall.
- Brier Score.
- Confusion Matrix.

## Limitations

Fill labels are synthetically generated.

The model is a prototype and has not been validated
against actual order execution events.

The FI-2010 benchmark provides normalized order-book
snapshots but does not provide complete order-level
queue histories.
