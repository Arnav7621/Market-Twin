
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
    DATA_DIR / "alpha_compiler" / "nested_discovery"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "AAPL"
HORIZON = "1d"
TARGET = "Target_Return_1D"

N_FOLDS = 4

RANDOM_SEEDS = [42, 101, 202, 303]

POPULATION_SIZE = 300
GENERATIONS = 12
TOURNAMENT_SIZE = 20

COMPLEXITY_PENALTY = 0.001
MIN_INNER_IC = 0.01
MAX_COMPLEXITY = 25

INNER_TRAIN_FRACTION = 0.80
PURGE_GAP = 1

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


# ============================================================
# CUSTOM GP FUNCTIONS
# ============================================================

def protected_division(x1, x2):

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore"
    ):

        result = np.divide(x1, x2)

    return np.where(
        np.isfinite(result),
        result,
        0.0
    )


def correlation_fitness(y, y_pred, sample_weight):

    if np.std(y_pred) < 1e-12:
        return -1.0

    if np.std(y) < 1e-12:
        return -1.0

    value = spearmanr(y, y_pred).statistic

    if not np.isfinite(value):
        return -1.0

    return float(value)


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
# LOAD AND VALIDATE OUTER FOLD
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
    outer_validation_df = pd.read_csv(validation_path)

    required = ["Date", TARGET] + FEATURES

    for name, df in [
        ("Outer training", train_df),
        ("Outer validation", outer_validation_df)
    ]:

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} fold {fold_number} "
                f"missing columns: {missing}"
            )

        if df[required].isnull().any().any():
            raise ValueError(
                f"Missing values in {name}, fold {fold_number}"
            )

        values = df[FEATURES + [TARGET]].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in {name}, fold {fold_number}"
            )

    train_dates = pd.to_datetime(train_df["Date"])
    outer_dates = pd.to_datetime(outer_validation_df["Date"])

    if train_dates.max() >= outer_dates.min():
        raise ValueError(
            f"Outer chronological ordering failed in fold {fold_number}"
        )

    return train_df, outer_validation_df


# ============================================================
# INNER CHRONOLOGICAL SPLIT
# ============================================================

def create_inner_split(outer_train_df):

    n = len(outer_train_df)

    split_index = int(n * INNER_TRAIN_FRACTION)

    inner_train_end = split_index

    inner_validation_start = (
        inner_train_end + PURGE_GAP
    )

    if inner_validation_start >= n:
        raise ValueError("Insufficient observations for inner split.")

    inner_train_df = outer_train_df.iloc[
        :inner_train_end
    ].copy()

    inner_validation_df = outer_train_df.iloc[
        inner_validation_start:
    ].copy()

    if len(inner_train_df) < 100:
        raise ValueError("Inner training set is too small.")

    if len(inner_validation_df) < 30:
        raise ValueError("Inner validation set is too small.")

    train_end = pd.to_datetime(
        inner_train_df["Date"].iloc[-1]
    )

    validation_start = pd.to_datetime(
        inner_validation_df["Date"].iloc[0]
    )

    if train_end >= validation_start:
        raise ValueError("Inner chronological ordering failed.")

    return inner_train_df, inner_validation_df


# ============================================================
# INNER CANDIDATE SELECTION
# ============================================================

def select_candidate(inner_train_df, inner_validation_df, fold_number):

    X_train = inner_train_df[FEATURES].to_numpy()
    y_train = inner_train_df[TARGET].to_numpy()

    X_validation = inner_validation_df[FEATURES].to_numpy()
    y_validation = inner_validation_df[TARGET].to_numpy()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_validation_scaled = scaler.transform(
        X_validation
    )

    candidates = []

    print("\nINNER CANDIDATE SELECTION")

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

        factor = model.predict(
            X_validation_scaled
        )

        if not np.isfinite(factor).all():
            print("Rejected: non-finite factor.")
            continue

        inner_ic = calculate_spearman(
            factor,
            y_validation
        )

        inner_pearson = calculate_pearson(
            factor,
            y_validation
        )

        if not np.isfinite(inner_ic):
            inner_ic = -1.0

        complexity = int(model._program.length_)

        selection_score = (
            inner_ic
            - COMPLEXITY_PENALTY * complexity
        )

        eligible = (
            inner_ic >= MIN_INNER_IC
            and complexity <= MAX_COMPLEXITY
        )

        candidate = {
            "Fold": fold_number,
            "Seed": seed,
            "Inner_IC": inner_ic,
            "Inner_Pearson": inner_pearson,
            "Complexity": complexity,
            "Selection_Score": selection_score,
            "Eligible": eligible,
            "Expression": str(model._program)
        }

        candidates.append(candidate)

        print(f"Inner IC: {inner_ic:.6f}")
        print(f"Complexity: {complexity}")
        print(f"Selection score: {selection_score:.6f}")
        print(f"Eligible: {eligible}")

    if not candidates:
        raise RuntimeError(
            f"No valid candidate in fold {fold_number}"
        )

    eligible_candidates = [
        c for c in candidates
        if c["Eligible"]
    ]

    if eligible_candidates:
        selection_pool = eligible_candidates
        status = "PASSED"
    else:
        selection_pool = candidates
        status = "NO_CANDIDATE_PASSED"

    best = max(
        selection_pool,
        key=lambda c: (
            c["Selection_Score"],
            c["Inner_IC"],
            -c["Complexity"]
        )
    )

    return best, candidates, status


# ============================================================
# REFIT AND OUTER EVALUATION
# ============================================================

def evaluate_outer_fold(
    outer_train_df,
    outer_validation_df,
    selected_candidate,
    fold_number
):

    X_train = outer_train_df[FEATURES].to_numpy()
    y_train = outer_train_df[TARGET].to_numpy()

    X_outer = outer_validation_df[FEATURES].to_numpy()
    y_outer = outer_validation_df[TARGET].to_numpy()

    # Fit scaler using the complete outer training set.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_outer_scaled = scaler.transform(X_outer)

    # Refit using the seed selected by inner validation.
    selected_seed = selected_candidate["Seed"]

    final_model = create_model(selected_seed)

    final_model.fit(
        X_train_scaled,
        y_train
    )

    outer_factor = final_model.predict(
        X_outer_scaled
    )

    if not np.isfinite(outer_factor).all():
        raise RuntimeError(
            f"Non-finite outer factor in fold {fold_number}"
        )

    outer_ic = calculate_spearman(
        outer_factor,
        y_outer
    )

    outer_pearson = calculate_pearson(
        outer_factor,
        y_outer
    )

    outer_complexity = int(
        final_model._program.length_
    )

    # The factor's rank correlation is the primary evaluation.
    # This is not a trading return or profitability estimate.

    predictions = pd.DataFrame({
        "Date": outer_validation_df["Date"],
        "Factor_Value": outer_factor,
        "Actual_Return": y_outer
    })

    predictions["Factor_Quantile"] = pd.qcut(
        predictions["Factor_Value"].rank(method="first"),
        q=5,
        labels=False
    ) + 1

    return {
        "Fold": fold_number,
        "Selected_Seed": selected_seed,
        "Inner_IC": selected_candidate["Inner_IC"],
        "Inner_Complexity": selected_candidate["Complexity"],
        "Selection_Status": (
            "PASSED"
            if selected_candidate["Eligible"]
            else "NO_CANDIDATE_PASSED"
        ),
        "Outer_IC": outer_ic,
        "Outer_Pearson": outer_pearson,
        "Refitted_Complexity": outer_complexity,
        "Outer_Training_Size": len(outer_train_df),
        "Outer_Validation_Size": len(outer_validation_df),
        "Outer_Validation_Start": outer_validation_df["Date"].iloc[0],
        "Outer_Validation_End": outer_validation_df["Date"].iloc[-1]
    }, predictions


# ============================================================
# MAIN NESTED DISCOVERY PIPELINE
# ============================================================

def discover():

    print("\n========================================")
    print("ALPHACOMPILER NESTED WALK-FORWARD")
    print("========================================")

    print(f"Ticker: {TICKER}")
    print(f"Horizon: {HORIZON}")
    print(f"Features: {len(FEATURES)}")
    print(f"Outer folds: {N_FOLDS}")
    print(f"Candidate seeds: {RANDOM_SEEDS}")

    all_inner_candidates = []
    outer_results = []

    for fold_number in range(1, N_FOLDS + 1):

        print("\n========================================")
        print(f"OUTER FOLD {fold_number}/{N_FOLDS}")
        print("========================================")

        outer_train_df, outer_validation_df = load_fold(
            fold_number
        )

        inner_train_df, inner_validation_df = create_inner_split(
            outer_train_df
        )

        print("\nINNER SPLIT")
        print(f"Inner training: {len(inner_train_df)}")
        print(f"Inner validation: {len(inner_validation_df)}")
        print(f"Purged observations: {PURGE_GAP}")

        print(
            f"Inner training dates: "
            f"{inner_train_df['Date'].iloc[0]} "
            f"to {inner_train_df['Date'].iloc[-1]}"
        )

        print(
            f"Inner validation dates: "
            f"{inner_validation_df['Date'].iloc[0]} "
            f"to {inner_validation_df['Date'].iloc[-1]}"
        )

        best, candidates, selection_status = select_candidate(
            inner_train_df,
            inner_validation_df,
            fold_number
        )

        all_inner_candidates.extend(candidates)

        print("\nINNER SELECTED CANDIDATE")
        print(f"Seed: {best['Seed']}")
        print(f"Inner IC: {best['Inner_IC']:.6f}")
        print(f"Complexity: {best['Complexity']}")
        print(f"Status: {selection_status}")
        print(f"Expression: {best['Expression']}")

        # Outer validation is used only after selection is complete.
        outer_result, predictions = evaluate_outer_fold(
            outer_train_df,
            outer_validation_df,
            best,
            fold_number
        )

        outer_results.append(outer_result)

        prediction_path = (
            OUTPUT_DIR
            / f"{TICKER}_{HORIZON}_fold_{fold_number}_outer_predictions.csv"
        )

        predictions.to_csv(
            prediction_path,
            index=False
        )

        print("\nOUTER EVALUATION")
        print(f"Outer IC: {outer_result['Outer_IC']:.6f}")
        print(f"Outer Pearson: {outer_result['Outer_Pearson']:.6f}")
        print(
            f"Refitted complexity: "
            f"{outer_result['Refitted_Complexity']}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    candidates_df = pd.DataFrame(all_inner_candidates)
    results_df = pd.DataFrame(outer_results)

    candidates_path = (
        OUTPUT_DIR
        / f"{TICKER}_{HORIZON}_inner_candidates.csv"
    )

    results_path = (
        OUTPUT_DIR
        / f"{TICKER}_{HORIZON}_nested_outer_results.csv"
    )

    candidates_df.to_csv(
        candidates_path,
        index=False
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("NESTED WALK-FORWARD SUMMARY")
    print("========================================")

    print(
        results_df[
            [
                "Fold",
                "Selected_Seed",
                "Inner_IC",
                "Outer_IC",
                "Outer_Pearson",
                "Refitted_Complexity",
                "Selection_Status"
            ]
        ].to_string(index=False)
    )

    outer_ic_values = results_df["Outer_IC"].dropna()

    mean_outer_ic = (
        outer_ic_values.mean()
        if len(outer_ic_values)
        else np.nan
    )

    median_outer_ic = (
        outer_ic_values.median()
        if len(outer_ic_values)
        else np.nan
    )

    positive_outer_folds = int(
        (outer_ic_values > 0).sum()
    )

    print("\nAGGREGATE OUTER RESULTS")
    print(f"Mean outer IC: {mean_outer_ic:.6f}")
    print(f"Median outer IC: {median_outer_ic:.6f}")
    print(
        f"Positive outer folds: "
        f"{positive_outer_folds}/{len(outer_ic_values)}"
    )

    print("\nSAVED FILES")
    print(candidates_path)
    print(results_path)

    print("\nNested discovery completed.")


if __name__ == "__main__":
    discover()
