
import pandas as pd

from risk_metrics import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_all_risk_metrics
)


# Artificial daily returns
returns = pd.Series([
    0.01,
    0.02,
    -0.03,
    0.015,
    -0.01,
    0.025,
    -0.04,
    0.01,
    0.005,
    -0.02
])


print("\n========== RISK ENGINE TEST ==========")

print(
    "Annualized Volatility:",
    calculate_volatility(returns)
)

print(
    "Sharpe Ratio:",
    calculate_sharpe_ratio(returns)
)

print(
    "Sortino Ratio:",
    calculate_sortino_ratio(returns)
)

print(
    "Maximum Drawdown:",
    calculate_max_drawdown(returns)
)

print(
    "VaR 95%:",
    calculate_var(returns)
)

print(
    "CVaR 95%:",
    calculate_cvar(returns)
)

print("\nAll Metrics:")

print(
    calculate_all_risk_metrics(returns)
)
