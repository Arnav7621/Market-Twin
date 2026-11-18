
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data" / "processed"

WALK_FORWARD_DIR = (
    DATA_DIR / "alpha_compiler" / "walk_forward"
)

OUTPUT_DIR = (
    DATA_DIR / "alpha_compiler" / "walk_forward_discovery"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZON = "1d"

N_FOLDS = 4

RANDOM_SEEDS = [42, 101, 202, 303]

POPULATION_SIZE = 300
GENERATIONS = 12
TOURNAMENT_SIZE = 20

COMPLEXITY_PENALTY = 0.001
MIN_VALIDATION_IC = 0.01
MAX_COMPLEXITY = 25

FEATURES = [
    "Return",
    "Log_Return",
    "Momentum_5D",
    "Momentum_10D",
    "Momentum_20D",
    "Momentum_60D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "Volatility_Ratio",
    "RSI_14",
    "Distance_From_SMA_20",
    "Distance_From_SMA_50",
    "Distance_From_SMA_100",
    "ATR_Percent",
    "Volume_Change",
    "Volume_ZScore_20D",
    "Price_Range",
    "Open_Close_Return",
    "MACD_Normalized",
    "MACD_Signal_Normalized",
    "MACD_Histogram_Normalized",
]

TARGET = "Target_Return_1D"


# ============================================================
# CUSTOM FUNCTIONS
# ============================================================

def protected_division(x1, x2):

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore"
    ):

        result = np.divide(x1, x2)

    result = np.where(
        np.isfinite(result),
        result,
        0.0
    )

    return result


def correlation_fitness(y, y_pred, sample_weight):

    if np.std(y_pred) < 1e-12:
        return -1.0

    if np.std(y) < 1e-12:
        return -1.0

    result = spearmanr(y, y_pred).statistic

    if not np.isfinite(result):
        return -1.0

    return float(result)


correlation_metric = make_fitness(
    function=correlation_fitness,
    greater_is_better=True,
    wrap=False
)


div_function = make_function(
    function=protected_division,
    name="protected_division",
    arity=2
)


# ============================================================
# MODEL
# ============================================================

def create_model(seed):

    return SymbolicRegressor(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        tournament_size=TOURNAMENT_SIZE,
        stopping_criteria=0.99,
        const_range=(-1, 1),
        init_depth=(2, 5),
        function_set=[
            "add",
            "sub",
            "mul",
            div_function
        ],
        metric=correlation_metric,
        parsimony_coefficient=0.001,
        max_samples=0.8,
        verbose=0,
        random_state=seed,
        n_jobs=1
    )


# ============================================================
# METRICS
# ============================================================

def calculate_spearman(x, y):

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    value = spearmanr(x, y).statistic

    return float(value) if np.isfinite(value) else np.nan


def calculate_pearson(x, y):

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    value = pearsonr(x, y).statistic

    return float(value) if np.isfinite(value) else np.nan


# ============================================================
# DATA VALIDATION
# ============================================================

def load_fold(fold_number):

    fold_dir = WALK_FORWARD_DIR / HORIZON

    train_path = (
        fold_dir / f"fold_{fold_number}_train.csv"
    )

    validation_path = (
        fold_dir / f"fold_{fold_number}_validation.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    required_columns = (
        ["Date", TARGET] + FEATURES
    )

    for name, df in [
        ("Training", train_df),
        ("Validation", validation_df)
    ]:

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} fold {fold_number} "
                f"is missing columns: {missing}"
            )

        if df[required_columns].isnull().any().any():
            raise ValueError(
                f"Missing values found in {name} "
                f"fold {fold_number}"
            )

        values = df[FEATURES + [TARGET]].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {name} "
                f"fold {fold_number}"
            )

    train_dates = pd.to_datetime(train_df["Date"])
    validation_dates = pd.to_datetime(validation_df["Date"])

    if train_dates.max() >= validation_dates.min():
        raise ValueError(
            f"Chronological ordering failed in fold {fold_number}"
        )

    return train_df, validation_df


# ============================================================
# WALK-FORWARD DISCOVERY
# ============================================================

def discover():

    print("\n========================================")
    print("ALPHACOMPILER WALK-FORWARD DISCOVERY")
    print("========================================")

    print(f"Horizon: {HORIZON}")
    print(f"Features: {len(FEATURES)}")
    print(f"Folds: {N_FOLDS}")
    print(f"Seeds per fold: {len(RANDOM_SEEDS)}")

    all_candidate_rows = []
    fold_summary_rows = []
    selected_expression_rows = []

    for fold_number in range(1, N_FOLDS + 1):

        print("\n----------------------------------------")
        print(f"FOLD {fold_number}/{N_FOLDS}")
        print("----------------------------------------")

        train_df, validation_df = load_fold(fold_number)

        X_train = train_df[FEATURES].to_numpy()
        y_train = train_df[TARGET].to_numpy()

        X_validation = validation_df[FEATURES].to_numpy()
        y_validation = validation_df[TARGET].to_numpy()

        # Fit scaler only on training observations.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_validation_scaled = scaler.transform(
            X_validation
        )

        print(f"Training observations: {len(train_df)}")
        print(f"Validation observations: {len(validation_df)}")

        print(
            f"Training dates: {train_df['Date'].iloc[0]} "
            f"to {train_df['Date'].iloc[-1]}"
        )

        print(
            f"Validation dates: {validation_df['Date'].iloc[0]} "
            f"to {validation_df['Date'].iloc[-1]}"
        )

        fold_candidates = []

        # Train multiple symbolic-regression candidates.
        for candidate_number, seed in enumerate(
            RANDOM_SEEDS,
            start=1
        ):

            print(
                f"\nCandidate {candidate_number}/"
                f"{len(RANDOM_SEEDS)} | Seed={seed}"
            )

            model = create_model(seed)

            model.fit(
                X_train_scaled,
                y_train
            )

            factor_validation = model.predict(
                X_validation_scaled
            )

            if not np.isfinite(factor_validation).all():
                print("Rejected: non-finite factor values")
                continue

            validation_ic = calculate_spearman(
                factor_validation,
                y_validation
            )

            validation_pearson = calculate_pearson(
                factor_validation,
                y_validation
            )

            complexity = int(model._program.length_)

            if not np.isfinite(validation_ic):
                validation_ic = -1.0

            selection_score = (
                validation_ic
                - COMPLEXITY_PENALTY * complexity
            )

            eligible = (
                validation_ic >= MIN_VALIDATION_IC
                and complexity <= MAX_COMPLEXITY
            )

            expression = str(model._program)

            candidate = {
                "Fold": fold_number,
                "Candidate": candidate_number,
                "Seed": seed,
                "Validation_IC": validation_ic,
                "Validation_Pearson": validation_pearson,
                "Complexity": complexity,
                "Selection_Score": selection_score,
                "Eligible": eligible,
                "Expression": expression
            }

            fold_candidates.append(candidate)
            all_candidate_rows.append(candidate)

            print(f"Validation IC: {validation_ic:.6f}")
            print(f"Validation Pearson: {validation_pearson:.6f}")
            print(f"Complexity: {complexity}")
            print(f"Selection score: {selection_score:.6f}")

        if not fold_candidates:
            raise RuntimeError(
                f"No valid candidates produced in fold {fold_number}"
            )

        # Prefer candidates passing both constraints.
        eligible_candidates = [
            c for c in fold_candidates
            if c["Eligible"]
        ]

        if eligible_candidates:
            selection_pool = eligible_candidates
            selection_status = "PASSED"
        else:
            # Retain the best candidate for diagnostics.
            # It is NOT treated as passing the constraints.
            selection_pool = fold_candidates
            selection_status = "NO_CANDIDATE_PASSED"

        best_candidate = max(
            selection_pool,
            key=lambda c: (
                c["Selection_Score"],
                c["Validation_IC"],
                -c["Complexity"]
            )
        )

        print("\nSELECTED CANDIDATE FOR THIS FOLD")
        print(f"Seed: {best_candidate['Seed']}")
        print(f"Validation IC: {best_candidate['Validation_IC']:.6f}")
        print(f"Complexity: {best_candidate['Complexity']}")
        print(f"Status: {selection_status}")
        print(f"Expression: {best_candidate['Expression']}")

        fold_summary_rows.append({
            "Fold": fold_number,
            "Training_Size": len(train_df),
            "Validation_Size": len(validation_df),
            "Best_Seed": best_candidate["Seed"],
            "Validation_IC": best_candidate["Validation_IC"],
            "Validation_Pearson": best_candidate["Validation_Pearson"],
            "Complexity": best_candidate["Complexity"],
            "Selection_Score": best_candidate["Selection_Score"],
            "Selection_Status": selection_status,
            "Training_Start": train_df["Date"].iloc[0],
            "Training_End": train_df["Date"].iloc[-1],
            "Validation_Start": validation_df["Date"].iloc[0],
            "Validation_End": validation_df["Date"].iloc[-1]
        })

        selected_expression_rows.append({
            "Fold": fold_number,
            "Seed": best_candidate["Seed"],
            "Expression": best_candidate["Expression"]
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    candidates_df = pd.DataFrame(all_candidate_rows)

    summary_df = pd.DataFrame(fold_summary_rows)

    expressions_df = pd.DataFrame(selected_expression_rows)

    candidates_path = (
        OUTPUT_DIR / f"AAPL_{HORIZON}_candidate_leaderboard.csv"
    )

    summary_path = (
        OUTPUT_DIR / f"AAPL_{HORIZON}_fold_summary.csv"
    )

    expressions_path = (
        OUTPUT_DIR / f"AAPL_{HORIZON}_selected_expressions.csv"
    )

    candidates_df.to_csv(candidates_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    expressions_df.to_csv(expressions_path, index=False)

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print("\n========================================")
    print("WALK-FORWARD DISCOVERY SUMMARY")
    print("========================================")

    print(
        summary_df[
            [
                "Fold",
                "Best_Seed",
                "Validation_IC",
                "Complexity",
                "Selection_Status"
            ]
        ].to_string(index=False)
    )

    mean_ic = summary_df["Validation_IC"].mean()
    median_ic = summary_df["Validation_IC"].median()

    positive_folds = int(
        (summary_df["Validation_IC"] > 0).sum()
    )

    passed_folds = int(
        (summary_df["Selection_Status"] == "PASSED").sum()
    )

    print("\nAGGREGATE RESULTS")
    print(f"Mean validation IC: {mean_ic:.6f}")
    print(f"Median validation IC: {median_ic:.6f}")
    print(f"Positive IC folds: {positive_folds}/{N_FOLDS}")
    print(f"Folds passing constraints: {passed_folds}/{N_FOLDS}")

    print("\nSAVED ARTIFACTS")
    print(candidates_path)
    print(summary_path)
    print(expressions_path)

    print("\nWalk-forward discovery completed.")


if __name__ == "__main__":
    discover()
