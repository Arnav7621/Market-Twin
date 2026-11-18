
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "alpha_compiler"
    / "nested_discovery"
)


# ============================================================
# LOAD OUTER PREDICTIONS
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "AAPL_1d_fold_*_outer_predictions.csv"
    )
)

if not files:
    raise FileNotFoundError(
        f"No outer prediction files found in {INPUT_DIR}"
    )


all_results = []

print("\n========================================")
print("OUTER FACTOR QUINTILE ANALYSIS")
print("========================================")


# ============================================================
# ANALYZE EACH FOLD
# ============================================================

for file_path in files:

    df = pd.read_csv(file_path)

    required = [
        "Date",
        "Factor_Value",
        "Actual_Return",
        "Factor_Quantile"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_path.name} missing columns: {missing}"
        )

    print(f"\nFILE: {file_path.name}")

    summary = (
        df.groupby("Factor_Quantile")
        .agg(
            Observations=("Actual_Return", "size"),
            Mean_Return=("Actual_Return", "mean"),
            Median_Return=("Actual_Return", "median"),
            Positive_Rate=(
                "Actual_Return",
                lambda x: (x > 0).mean()
            )
        )
        .reset_index()
    )

    summary["Fold"] = file_path.stem.split("_")[-2]

    print(
        summary.to_string(index=False)
    )

    if len(summary) == 5:

        q1 = summary.iloc[0]["Mean_Return"]
        q5 = summary.iloc[4]["Mean_Return"]

        spread = q5 - q1

        print(
            f"\nQ5-Q1 Mean Return Spread: {spread:.6f}"
        )

    all_results.append(summary)


# ============================================================
# COMBINED QUINTILE RESULTS
# ============================================================

combined = pd.concat(
    all_results,
    ignore_index=True
)

combined_summary = (
    combined.groupby("Factor_Quantile")
    .agg(
        Total_Observations=("Observations", "sum"),
        Average_Fold_Mean_Return=("Mean_Return", "mean"),
        Average_Fold_Median_Return=("Median_Return", "mean"),
        Average_Positive_Rate=("Positive_Rate", "mean")
    )
    .round(6)
)

print("\n========================================")
print("COMBINED QUINTILE RESULTS")
print("========================================")

print(combined_summary.to_string())


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    INPUT_DIR / "AAPL_1d_outer_quintile_analysis.csv"
)

combined.to_csv(
    output_path,
    index=False
)

print("\nAnalysis saved to:")
print(output_path)

print("\nQuintile analysis completed.")
