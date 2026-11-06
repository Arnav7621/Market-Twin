
# QueueMind Fill Prediction Problem

## Objective

Estimate the probability of a limit order filling within
a specified time horizon.

## Horizons

- 1 second
- 5 seconds
- 10 seconds

## Input Variables

- Spread
- Bid depth
- Ask depth
- Order book imbalance
- Volatility
- Order size
- Distance from midpoint

## Output

Fill probability between 0 and 1.

## Data Limitations

The FI-2010 dataset is a normalized limit-order-book
benchmark. It does not provide complete order IDs,
queue position, and every actual execution event.

Therefore, simulated fill labels must be identified
as synthetic rather than observed historical fills.
