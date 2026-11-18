from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("../data/processed/alpha_compiler/nested_discovery")
OUTPUT_DIR = DATA_DIR

FILE_PATTERN = "AAPL_1d_fold_*_outer_predictions.csv"
HAC_MAX_LAGS = 5

REQUIRED_COLUMNS = [
    "Date",
    "Factor_Value",
    "Actual_Return",
    "Factor_Quantile",
]


# --------------------------------------------------
# LOAD OUTER-FOLD PREDICTIONS
# --------------------------------------------------

files = sorted(DATA_DIR.glob(FILE_PATTERN))

if len(files) != 4:
    raise ValueError(
        f"Expected 4 outer-fold prediction files, found {len(files)}."
    )

frames = []

for file_path in files:

    match = re.search(r"fold_(\d+)_outer_predictions", file_path.stem)

    if not match:
        raise ValueError(f"Could not identify fold number: {file_path.name}")

    fold_number = int(match.group(1))

    df = pd.read_csv(file_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Factor_Value"] = pd.to_numeric(
        df["Factor_Value"], errors="coerce"
    )
    df["Actual_Return"] = pd.to_numeric(
        df["Actual_Return"], errors="coerce"
    )
    df["Factor_Quantile"] = pd.to_numeric(
        df["Factor_Quantile"], errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=REQUIRED_COLUMNS)

    df["Fold"] = fold_number

    if len(df) < 10:
        raise ValueError(
            f"Fold {fold_number} has fewer than 10 valid observations."
        )

    # Standardize factor values within each fold.
    # This makes factor coefficients comparable across folds.
    factor_std = df["Factor_Value"].std(ddof=0)

    if factor_std == 0:
        raise ValueError(
            f"Factor_Value has zero variance in fold {fold_number}."
        )

    df["Factor_ZScore"] = (
        df["Factor_Value"] - df["Factor_Value"].mean()
    ) / factor_std

    frames.append(df)

    print(
        f"Loaded Fold {fold_number}: "
        f"{len(df)} observations "
        f"({df['Date'].min().date()} to {df['Date'].max().date()})"
    )


# --------------------------------------------------
# COMBINE AND VALIDATE DATA
# --------------------------------------------------

data = pd.concat(frames, ignore_index=True)

data = data.sort_values(["Date", "Fold"]).reset_index(drop=True)

if data["Date"].duplicated().any():
    print(
        "\nWARNING: Duplicate dates found across outer folds. "
        "Check whether evaluation observations overlap."
    )

data.to_csv(
    OUTPUT_DIR / "AAPL_1d_outer_significance_dataset.csv",
    index=False
)

print("\nTotal observations:", len(data))
print("Number of folds:", data["Fold"].nunique())


# --------------------------------------------------
# HELPER: NEWey-WEST HAC REGRESSION
# --------------------------------------------------

def run_hac_regression(df, include_fold_controls=False):

    y = df["Actual_Return"].astype(float)

    X = pd.DataFrame(
        {"Factor_ZScore": df["Factor_ZScore"].astype(float)},
        index=df.index
    )

    if include_fold_controls:
        fold_dummies = pd.get_dummies(
            df["Fold"].astype(str),
            prefix="Fold",
            drop_first=True,
            dtype=float
        )

        X = pd.concat([X, fold_dummies], axis=1)

    X = sm.add_constant(X)

    maxlags = min(HAC_MAX_LAGS, len(df) - 1)

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags}
    )

    confidence_interval = model.conf_int().loc["Factor_ZScore"]

    return {
        "Observations": len(df),
        "Factor_Coefficient": model.params["Factor_ZScore"],
        "HAC_Std_Error": model.bse["Factor_ZScore"],
        "HAC_t_Statistic": model.tvalues["Factor_ZScore"],
        "HAC_p_Value": model.pvalues["Factor_ZScore"],
        "CI_95_Lower": confidence_interval.iloc[0],
        "CI_95_Upper": confidence_interval.iloc[1],
        "R_Squared": model.rsquared,
        "HAC_Max_Lags": maxlags,
    }


# --------------------------------------------------
# FOLD-BY-FOLD ANALYSIS
# --------------------------------------------------

fold_results = []

for fold_number in sorted(data["Fold"].unique()):

    fold_df = (
        data[data["Fold"] == fold_number]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    correlation, spearman_p = spearmanr(
        fold_df["Factor_Value"],
        fold_df["Actual_Return"]
    )

    hac_results = run_hac_regression(fold_df)

    result = {
        "Fold": int(fold_number),
        "Spearman_Correlation": correlation,
        "Spearman_p_Value": spearman_p,
        **hac_results,
    }

    fold_results.append(result)

fold_summary = pd.DataFrame(fold_results)

fold_summary.to_csv(
    OUTPUT_DIR / "AAPL_1d_hac_fold_results.csv",
    index=False
)


# --------------------------------------------------
# COMBINED SPEARMAN CORRELATION
# --------------------------------------------------

combined_spearman, combined_spearman_p = spearmanr(
    data["Factor_Value"],
    data["Actual_Return"]
)


# --------------------------------------------------
# COMBINED HAC REGRESSION
# --------------------------------------------------

# Controls for fold-specific average return differences.
# Factor_ZScore is standardized separately within each fold.

combined_hac = run_hac_regression(
    data,
    include_fold_controls=True
)

combined_result = pd.DataFrame([{
    "Spearman_Correlation": combined_spearman,
    "Spearman_p_Value": combined_spearman_p,
    **combined_hac,
}])

combined_result.to_csv(
    OUTPUT_DIR / "AAPL_1d_hac_combined_results.csv",
    index=False
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FOLD-BY-FOLD STATISTICAL RESULTS")
print("=" * 65)

print(
    fold_summary[
        [
            "Fold",
            "Observations",
            "Spearman_Correlation",
            "Spearman_p_Value",
            "Factor_Coefficient",
            "HAC_Std_Error",
            "HAC_p_Value",
            "CI_95_Lower",
            "CI_95_Upper",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 65)
print("COMBINED OUTER-FOLD RESULTS")
print("=" * 65)

print("Total observations:", len(data))
print("Combined Spearman correlation:", combined_spearman)
print("Combined Spearman p-value:", combined_spearman_p)

for key, value in combined_hac.items():
    print(f"{key}: {value}")

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)

positive_folds = (
    fold_summary["Spearman_Correlation"] > 0
).sum()

significant_folds = (
    fold_summary["HAC_p_Value"] < 0.05
).sum()

print(f"Positive Spearman folds: {positive_folds}/4")
print(f"Folds with HAC p-value < 0.05: {significant_folds}/4")

if (
    combined_hac["HAC_p_Value"] < 0.05
    and combined_hac["Factor_Coefficient"] > 0
):
    print(
        "Combined HAC regression indicates a statistically significant "
        "positive relationship at the 5% level."
    )
else:
    print(
        "Combined HAC regression does not establish a statistically "
        "significant positive relationship at the 5% level."
    )

print(
    "\nIMPORTANT: These are exploratory diagnostics. "
    "Four outer folds provide limited evidence. "
    "The combined regression assumes observations are appropriately "
    "ordered in time and does not correct for every possible form "
    "of data dependence or multiple testing."
)

print("\nSaved results:")
print("1. AAPL_1d_outer_significance_dataset.csv")
print("2. AAPL_1d_hac_fold_results.csv")
print("3. AAPL_1d_hac_combined_results.csv")
