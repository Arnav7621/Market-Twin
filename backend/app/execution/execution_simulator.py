
import numpy as np


def simulate_execution(
    market_price,
    side,
    order_quantity,
    fill_probability,
    spread_bps=5.0,
    slippage_bps=2.0,
    fee_rate=0.001,
    rng=None
):

    if market_price <= 0:

        raise ValueError(
            "Market price must be positive"
        )

    if side not in [
        "buy",
        "sell"
    ]:

        raise ValueError(
            "Side must be buy or sell"
        )

    if order_quantity <= 0:

        raise ValueError(
            "Order quantity must be positive"
        )

    if not 0 <= fill_probability <= 1:

        raise ValueError(
            "Fill probability must be between 0 and 1"
        )

    if spread_bps < 0 or slippage_bps < 0:

        raise ValueError(
            "Spread and slippage cannot be negative"
        )

    if fee_rate < 0:

        raise ValueError(
            "Fee rate cannot be negative"
        )

    if rng is None:

        rng = np.random.default_rng(42)

    # Simulate whether order fills
    is_filled = (
        rng.random() < fill_probability
    )

    if not is_filled:

        return {
            "filled": False,
            "filled_quantity": 0,
            "execution_price": None,
            "transaction_cost": 0.0,
            "slippage_cost": 0.0
        }

    filled_quantity = order_quantity

    # Convert basis points into decimal
    spread = spread_bps / 10000

    slippage = slippage_bps / 10000

    if side == "buy":

        execution_price = (
            market_price *
            (
                1 +
                spread / 2 +
                slippage
            )
        )

    else:

        execution_price = (
            market_price *
            (
                1 -
                spread / 2 -
                slippage
            )
        )

    transaction_cost = (
        execution_price *
        filled_quantity *
        fee_rate
    )

    slippage_cost = (
        market_price *
        filled_quantity *
        slippage
    )

    return {
        "filled": True,
        "filled_quantity": filled_quantity,
        "execution_price": execution_price,
        "transaction_cost": transaction_cost,
        "slippage_cost": slippage_cost
    }


if __name__ == "__main__":

    result = simulate_execution(
        market_price=100,
        side="buy",
        order_quantity=100,
        fill_probability=0.80,
        spread_bps=10,
        slippage_bps=5,
        fee_rate=0.001
    )

    print("\nEXECUTION RESULT")

    print(result)
