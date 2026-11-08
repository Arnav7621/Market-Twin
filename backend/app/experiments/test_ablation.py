
from ablation_study import (
    load_real_data,
    load_synthetic_data,
    calculate_metrics
)


def test_real_data():

    df = load_real_data(
        "AAPL"
    )

    assert len(df) > 0

    assert df["Target"].notna().all()


def test_synthetic_data():

    df = load_synthetic_data()

    assert len(df) > 0

    assert df["Target"].notna().all()


def test_metrics():

    y_true = [0.01, 0.02, -0.01, 0.03]

    y_pred = [0.01, 0.01, -0.01, 0.02]

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    assert metrics["MSE"] >= 0

    assert -1 <= metrics["IC"] <= 1


if __name__ == "__main__":

    test_real_data()

    test_synthetic_data()

    test_metrics()

    print(
        "ALL ABLATION TESTS PASSED!"
    )
