
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_DIR /
    "data" /
    "processed" /
    "final_results"
)


def plot_ablation():

    file_path = (
        DATA_DIR /
        "real_vs_synthetic.csv"
    )

    df = pd.read_csv(
        file_path
    )

    if df.empty:

        print(
            "No ablation data available"
        )

        return

    plt.figure()

    plt.bar(
        df["Experiment"],
        df["IC"]
    )

    plt.title(
        "Real vs Synthetic Augmentation"
    )

    plt.xlabel(
        "Experiment"
    )

    plt.ylabel(
        "Information Coefficient"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    output_file = (
        DATA_DIR /
        "ablation_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":

    plot_ablation()
