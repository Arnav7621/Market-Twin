
import numpy as np

from execution_simulator import (
    simulate_execution
)


def test_full_fill():

    result = simulate_execution(
        market_price=100,
        side="buy",
        order_quantity=10,
        fill_probability=1.0,
        spread_bps=0,
        slippage_bps=0,
        fee_rate=0,
        rng=np.random.default_rng(1)
    )

    assert result["filled"] is True

    assert result["filled_quantity"] == 10

    assert result["execution_price"] == 100


def test_no_fill():

    result = simulate_execution(
        market_price=100,
        side="buy",
        order_quantity=10,
        fill_probability=0.0,
        rng=np.random.default_rng(1)
    )

    assert result["filled"] is False

    assert result["filled_quantity"] == 0


def test_buy_execution_cost():

    result = simulate_execution(
        market_price=100,
        side="buy",
        order_quantity=10,
        fill_probability=1.0,
        spread_bps=10,
        slippage_bps=5,
        fee_rate=0,
        rng=np.random.default_rng(1)
    )

    assert result["execution_price"] > 100


def test_sell_execution_cost():

    result = simulate_execution(
        market_price=100,
        side="sell",
        order_quantity=10,
        fill_probability=1.0,
        spread_bps=10,
        slippage_bps=5,
        fee_rate=0,
        rng=np.random.default_rng(1)
    )

    assert result["execution_price"] < 100


if __name__ == "__main__":

    test_full_fill()

    test_no_fill()

    test_buy_execution_cost()

    test_sell_execution_cost()

    print("ALL EXECUTION TESTS PASSED!")
