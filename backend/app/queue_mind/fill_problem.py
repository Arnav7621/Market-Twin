
from dataclasses import dataclass


@dataclass
class FillPredictionRequest:

    spread: float

    bid_depth: float

    ask_depth: float

    imbalance: float

    volatility: float

    order_size: float

    distance_from_mid: float

    horizon_seconds: int


def validate_request(request):

    if request.spread <= 0:
        raise ValueError("Spread must be positive")

    if request.bid_depth < 0:
        raise ValueError("Bid depth cannot be negative")

    if request.ask_depth < 0:
        raise ValueError("Ask depth cannot be negative")

    if not -1 <= request.imbalance <= 1:
        raise ValueError("Imbalance must be between -1 and 1")

    if request.volatility < 0:
        raise ValueError("Volatility cannot be negative")

    if request.order_size <= 0:
        raise ValueError("Order size must be positive")

    if request.horizon_seconds not in [1, 5, 10]:
        raise ValueError("Horizon must be 1, 5 or 10 seconds")

    return True


if __name__ == "__main__":

    request = FillPredictionRequest(
        spread=0.02,
        bid_depth=500,
        ask_depth=400,
        imbalance=0.11,
        volatility=0.02,
        order_size=100,
        distance_from_mid=0.01,
        horizon_seconds=5
    )

    print(validate_request(request))
