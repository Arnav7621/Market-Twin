
# MarketTwin — Full Pipeline Ablation

## Objective

Evaluate the contribution of individual components
to the quantitative research workflow.

## Experiments

A. Baseline momentum strategy

B. AlphaCompiler factor

C. QueueMind execution assumptions

D. MarketTwin scenario augmentation

E. AlphaCompiler + QueueMind + Execution

F. Full integrated research pipeline

## Metrics

- Information Coefficient
- Net PnL
- Sharpe Ratio
- Maximum Drawdown
- Execution cost
- Fill rate

## Research Requirements

Experiments must use:

- Chronological train/test separation
- Identical evaluation periods
- Consistent transaction cost assumptions
- Reproducible random seeds
- Actual module outputs

## Conclusion

The contribution of each component must be
evaluated using the actual experiment results.

No component should be described as improving
performance without measured evidence.
