import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)
import numpy as np

from factor_evaluator import (
    add,
    subtract,
    multiply,
    protected_divide
)


def test_add():

    assert add(2, 3) == 5


def test_subtract():

    assert subtract(5, 3) == 2


def test_multiply():

    assert multiply(2, 3) == 6


def test_protected_divide():

    result = protected_divide(
        10,
        2
    )

    assert result == 5


def test_division_by_zero():

    result = protected_divide(
        10,
        0
    )

    assert np.isnan(result)


if __name__ == "__main__":

    test_add()

    test_subtract()

    test_multiply()

    test_protected_divide()

    test_division_by_zero()

    print("ALL FACTOR TESTS PASSED!")
