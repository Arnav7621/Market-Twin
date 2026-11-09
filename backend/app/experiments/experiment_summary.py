
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed"
)

EXPERIMENT_DIR = (
    DATA_DIR /
    "experiments"
)

VALIDATION_DIR = (
    DATA_DIR /
    "validation"
)

OUTPUT_DIR = (
    DATA_DIR /
    "final_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def read_csv_if_exists(
    path
):

    if path.exists():

        return pd.read_csv(
            path
        )

    return pd.DataFrame()


def build_summary():

    ablation = read_csv_if_exists(
        EXPERIMENT_DIR /
        "AAPL_ablation.csv"
    )

    execution = read_csv_if_exists(
        EXPERIMENT_DIR /
        "AAPL_execution_ablation.csv"
    )

    full_ablation = read_csv_if_exists(
        EXPERIMENT_DIR /
        "AAPL_full_ablation.csv"
    )

    metrics = read_csv_if_exists(
        VALIDATION_DIR /
        "AAPL_metrics.csv"
    )

    walk_forward = read_csv_if_exists(
        VALIDATION_DIR /
        "AAPL_momentum_walk_forward.csv"
    )

    robustness = read_csv_if_exists(
        VALIDATION_DIR /
        "AAPL_robustness.csv"
    )

    bootstrap = read_csv_if_exists(
        VALIDATION_DIR /
        "AAPL_bootstrap_ci.csv"
    )

    tables = {

        "real_vs_synthetic": ablation,

        "execution_ablation": execution,

        "full_pipeline_ablation": full_ablation,

        "performance_metrics": metrics,

        "walk_forward": walk_forward,

        "robustness": robustness,

        "bootstrap": bootstrap

    }

    for name, table in (
        tables.items()
    ):

        print(
            f"\n{name.upper()}"
        )

        print(
            table.head()
        )

        table.to_csv(
            OUTPUT_DIR /
            f"{name}.csv",
            index=False
        )

    print(
        "\nEXPERIMENT SUMMARY COMPLETE"
    )


if __name__ == "__main__":

    build_summary()
