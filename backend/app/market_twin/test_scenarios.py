
from scenario_generator import generate_scenario


def test_scenario_generation():

    scenario = generate_scenario(
        ticker="AAPL",
        volatility="high",
        liquidity="low",
        imbalance="negative",
        duration=1000,
        seed=42
    )

    assert len(scenario) == 1000

    assert scenario["Price"].notna().all()

    assert scenario["Return"].notna().all()

    assert scenario["Spread_Bps"].gt(0).all()

    assert scenario["Bid_Depth"].ge(0).all()

    assert scenario["Ask_Depth"].ge(0).all()

    assert scenario["Order_Imbalance"].lt(0).all()

    assert (
        scenario["Volatility_Regime"].eq("high").all()
    )

    assert (
        scenario["Liquidity_Regime"].eq("low").all()
    )

    print("ALL MARKETTWIN SCENARIO TESTS PASSED!")


if __name__ == "__main__":

    test_scenario_generation()
