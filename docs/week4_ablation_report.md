
# MarketTwin — Ablation Study Report

## Experiment A — Real-Only

Objective:
Measure the predictive performance of the baseline
factor using real historical data.

## Experiment B — Real + Synthetic

Objective:
Determine whether synthetic MarketTwin data
improves out-of-sample predictive performance.

## Experiment C — Prediction-Only vs Execution-Aware

Objective:
Compare prediction scores against scores
incorporating simplified execution costs.

## Experiment D — Fill Probability

Objective:
Evaluate how different fill-probability assumptions
affect expected execution outcomes.

## Experiment E — Statistical Filter

Objective:
Compare filtered and unfiltered alpha factors.

## Metrics

- Mean Squared Error
- Information Coefficient
- Average score
- Net PnL (to be integrated in future experiments)

## Limitations

The current experiments use a baseline momentum factor.

Execution costs are simplified.

Fill probabilities are fixed assumptions.

Results require walk-forward evaluation
and statistical significance testing.

No claim of improved strategy performance
should be made without measured evidence.
