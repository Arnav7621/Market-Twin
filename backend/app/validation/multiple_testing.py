
import numpy as np

import pandas as pd


def benjamini_hochberg(
    p_values,
    alpha=0.05
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(
        p_values
    )

    if n == 0:

        return np.array(
            [],
            dtype=bool
        )

    if (
        np.any(p_values < 0) or
        np.any(p_values > 1) or
        np.any(~np.isfinite(p_values))
    ):

        raise ValueError(
            "P-values must be finite and between 0 and 1"
        )

    sorted_indices = np.argsort(
        p_values
    )

    sorted_p = p_values[
        sorted_indices
    ]

    thresholds = (
        np.arange(1, n + 1) /
        n *
        alpha
    )

    passed = (
        sorted_p <= thresholds
    )

    significant = np.zeros(
        n,
        dtype=bool
    )

    if passed.any():

        max_index = np.where(
            passed
        )[0].max()

        significant[
            sorted_indices[
                :max_index + 1
            ]
        ] = True

    return significant


def adjust_p_values(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    sorted_p = p_values[
        order
    ]

    adjusted = (
        sorted_p *
        n /
        np.arange(
            1,
            n + 1
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    result = np.empty(
        n
    )

    result[
        order
    ] = adjusted

    return result


if __name__ == "__main__":

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
        p_values,
        alpha=0.05
    )

    adjusted = adjust_p_values(
        p_values
    )

    results = pd.DataFrame({

        "P_Value": p_values,

        "Adjusted_P_Value": adjusted,

        "Significant": significant

    })

    print(
        "\nMULTIPLE TESTING RESULTS"
    )

    print(
        results
    )
