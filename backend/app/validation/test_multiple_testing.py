
from app.validation.multiple_testing import (
    benjamini_hochberg,
    adjust_p_values
)


def test_bh():

    p_values = [
        0.001,
        0.005,
        0.010,
        0.030,
        0.040,
        0.200,
        0.500
    ]

    significant = benjamini_hochberg(
        p_values
    )

    assert len(
        significant
    ) == len(
        p_values
    )


def test_adjusted_p_values():

    p_values = [
        0.01,
        0.05,
        0.20
    ]

    adjusted = adjust_p_values(
        p_values
    )

    assert (
        adjusted >= p_values
    ).all()


if __name__ == "__main__":

    test_bh()

    test_adjusted_p_values()

    print(
        "MULTIPLE TESTING TESTS PASSED"
    )
