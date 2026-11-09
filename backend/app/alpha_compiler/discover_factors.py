from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "alpha_compiler"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "Return",
    "Momentum_5D",
    "Momentum_20D",
    "Volatility_20D",
    "Volume_Change"
]


# =========================================
# CONFIGURATION
# =========================================

N_CANDIDATES = 8

POPULATION_SIZE = 300

GENERATIONS = 12

RANDOM_SEEDS = [
    42,
    101,
    202,
    303,
    404,
    505,
    606,
    707
]


# =========================================
# DATA LOADING
# =========================================

def load_data(ticker):

    file_path = DATA_DIR / f"{ticker}.csv"

    if not file_path.exists():

        raise FileNotFoundError(
            f"Market data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "Date",
        "Close",
        "Volume",
        "Return"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Momentum_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Momentum_20D"] = (
        df["Close"].pct_change(20)
    )

    df["Volatility_20D"] = (
        df["Return"].rolling(20).std()
    )

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    # Predict the next trading session's return.
    df["Target"] = df["Return"].shift(-1)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target"]
    )

    df = df.reset_index(drop=True)

    if len(df) < 100:

        raise ValueError(
            "Not enough observations for reliable "
            "train-validation-test evaluation."
        )

    return df


# =========================================
# PROTECTED DIVISION
# =========================================

def protected_division(x1, x2):

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore"
    ):

        result = np.divide(
            x1,
            x2
        )

    result = np.where(
        np.isfinite(result),
        result,
        0.0
    )

    return result


# =========================================
# SPEARMAN FITNESS
# =========================================

def correlation_fitness(
    y,
    y_pred,
    sample_weight
):

    if np.std(y_pred) < 1e-12:

        return -1.0

    if np.std(y) < 1e-12:

        return -1.0

    correlation = spearmanr(
        y,
        y_pred
    ).statistic

    if not np.isfinite(correlation):

        return -1.0

    return float(correlation)


correlation_metric = make_fitness(
    function=correlation_fitness,
    greater_is_better=True,
    wrap=False
)


# =========================================
# IC HELPER
# =========================================

def calculate_ic(
    factor,
    returns
):

    factor = np.asarray(
        factor,
        dtype=float
    )

    returns = np.asarray(
        returns,
        dtype=float
    )

    valid = (
        np.isfinite(factor) &
        np.isfinite(returns)
    )

    factor = factor[valid]

    returns = returns[valid]

    if len(factor) < 3:

        return np.nan

    if np.std(factor) < 1e-12:

        return np.nan

    if np.std(returns) < 1e-12:

        return np.nan

    result = spearmanr(
        factor,
        returns
    ).statistic

    if not np.isfinite(result):

        return np.nan

    return float(result)


# =========================================
# MODEL CREATION
# =========================================

def create_model(
    random_state,
    div_function
):

    return SymbolicRegressor(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        tournament_size=20,
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
        random_state=random_state,
        n_jobs=1
    )


# =========================================
# ALPHACOMPILER
# =========================================

def discover(ticker="AAPL"):

    print("\n=========================================")

    print("ALPHACOMPILER")

    print("=========================================")

    df = load_data(ticker)

    X = df[FEATURES].to_numpy()

    y = df["Target"].to_numpy()

    dates = df["Date"].to_numpy()

    n = len(df)

    # =====================================
    # CHRONOLOGICAL SPLIT
    # =====================================

    train_end = int(n * 0.60)

    validation_end = int(n * 0.80)

    X_train = X[:train_end]

    y_train = y[:train_end]

    X_validation = X[
        train_end:validation_end
    ]

    y_validation = y[
        train_end:validation_end
    ]

    X_test = X[
        validation_end:
    ]

    y_test = y[
        validation_end:
    ]

    validation_dates = dates[
        train_end:validation_end
    ]

    test_dates = dates[
        validation_end:
    ]

    print("\nDATA SPLIT")

    print(f"Total observations: {n}")

    print(f"Training: {len(X_train)}")

    print(f"Validation: {len(X_validation)}")

    print(f"Test: {len(X_test)}")

    # =====================================
    # STANDARDIZATION
    # =====================================

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
    )

    X_validation_scaled = feature_scaler.transform(
        X_validation
    )

    X_test_scaled = feature_scaler.transform(
        X_test
    )

    # =====================================
    # SYMBOLIC REGRESSION
    # =====================================

    div_function = make_function(
        function=protected_division,
        name="protected_division",
        arity=2
    )

    candidates = []

    leaderboard_rows = []

    print("\nTRAINING FACTOR CANDIDATES")

    for candidate_number, seed in enumerate(
        RANDOM_SEEDS[:N_CANDIDATES],
        start=1
    ):

        print(
            f"\nCandidate {candidate_number}/"
            f"{N_CANDIDATES} | Seed={seed}"
        )

        model = create_model(
            random_state=seed,
            div_function=div_function
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        factor_train = model.predict(
            X_train_scaled
        )

        factor_validation = model.predict(
            X_validation_scaled
        )

        expression = str(
            model._program
        )

        validation_ic = calculate_ic(
            factor_validation,
            y_validation
        )

        complexity = int(
            model._program.length_
        )

        if not np.isfinite(validation_ic):

            validation_ic = -1.0

        print(
            f"Validation IC: "
            f"{validation_ic:.6f}"
        )

        print(
            f"Expression length: "
            f"{complexity}"
        )

        print(
            f"Expression: "
            f"{expression}"
        )

        candidates.append({
            "model": model,
            "expression": expression,
            "validation_ic": validation_ic,
            "complexity": complexity,
            "factor_train": factor_train,
            "factor_validation": factor_validation,
            "seed": seed
        })

        leaderboard_rows.append({
            "Candidate": candidate_number,
            "Seed": seed,
            "Validation_IC": validation_ic,
            "Complexity": complexity,
            "Expression": expression
        })

   
    # =====================================
    # COMPLEXITY-AWARE CANDIDATE SELECTION
    # =====================================

    leaderboard = pd.DataFrame(
        leaderboard_rows
    )

    # Selection constraints
    MIN_VALIDATION_IC = 0.01
    MAX_COMPLEXITY = 25

    # Penalize unnecessarily complex expressions.
    COMPLEXITY_PENALTY = 0.001

    leaderboard["Selection_Score"] = (
        leaderboard["Validation_IC"]
        - COMPLEXITY_PENALTY
        * leaderboard["Complexity"]
    )

    # Reject weak and excessively complex candidates.
    eligible_candidates = leaderboard[
        (leaderboard["Validation_IC"] >= MIN_VALIDATION_IC)
        & (leaderboard["Complexity"] <= MAX_COMPLEXITY)
    ].copy()

    if eligible_candidates.empty:
        raise RuntimeError(
            "No candidate passed the validation IC "
            "and complexity requirements. "
            "Review the candidate leaderboard."
        )

    eligible_candidates = eligible_candidates.sort_values(
        by=[
            "Selection_Score",
            "Validation_IC",
            "Complexity"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    leaderboard = leaderboard.sort_values(
        by=[
            "Selection_Score",
            "Validation_IC",
            "Complexity"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    leaderboard.to_csv(
        OUTPUT_DIR /
        f"{ticker}_candidate_leaderboard.csv",
        index=False
    )

    print("\nCOMPLEXITY-AWARE CANDIDATE LEADERBOARD")

    print(
        leaderboard[
            [
                "Candidate",
                "Seed",
                "Validation_IC",
                "Complexity",
                "Selection_Score"
            ]
        ].to_string(index=False)
    )

    # =====================================
    # SELECT BEST ELIGIBLE CANDIDATE
    # =====================================

    best_seed = int(
        eligible_candidates.iloc[0]["Seed"]
    )

    best_candidate = next(
        candidate
        for candidate in candidates
        if candidate["seed"] == best_seed
    )

    selected_model = best_candidate["model"]

    selected_expression = (
        best_candidate["expression"]
    )

    selected_validation_factor = (
        best_candidate["factor_validation"]
    )

    selected_validation_ic = (
        best_candidate["validation_ic"]
    )

    print("\nSELECTED FACTOR")

    print(selected_expression)

    print(
        f"Validation IC: "
        f"{selected_validation_ic:.6f}"
    )

    print(
        f"Complexity: "
        f"{best_candidate['complexity']}"
    )

    print(
        f"Selection Score: "
        f"{eligible_candidates.iloc[0]['Selection_Score']:.6f}"
    )


        # =====================================
    # VALIDATION CALIBRATION
    # =====================================

    if np.std(
        selected_validation_factor
    ) < 1e-12:

        raise RuntimeError(
            "Selected factor is constant on validation data."
        )

    calibrator = LinearRegression()

    calibrator.fit(
        selected_validation_factor.reshape(-1, 1),
        y_validation
    )

    validation_predictions = calibrator.predict(
        selected_validation_factor.reshape(-1, 1)
    )

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions
        )
    )

    print("\nVALIDATION CALIBRATION")

    print(
        f"Validation MAE: {validation_mae:.6f}"
    )

    print(
        f"Validation RMSE: {validation_rmse:.6f}"
    )

    # =====================================
    # TEST PREDICTIONS
    # =====================================

    factor_test = selected_model.predict(
        X_test_scaled
    )

    if not np.all(
        np.isfinite(factor_test)
    ):

        raise RuntimeError(
            "Test factor contains non-finite values."
        )

    if np.std(
        factor_test
    ) < 1e-12:

        raise RuntimeError(
            "Selected factor is constant on test data."
        )

    predicted_returns = calibrator.predict(
        factor_test.reshape(-1, 1)
    )

    # =====================================
    # BASELINE
    # =====================================

    baseline_return = float(
        np.mean(y_train)
    )

    baseline_predictions = np.full(
        len(y_test),
        baseline_return
    )

    # =====================================
    # TEST EVALUATION
    # =====================================

    test_mae = mean_absolute_error(
        y_test,
        predicted_returns
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predicted_returns
        )
    )

    test_ic = calculate_ic(
        factor_test,
        y_test
    )

    test_pearson = np.corrcoef(
        factor_test,
        y_test
    )[0, 1]

    directional_accuracy = np.mean(
        np.sign(predicted_returns) ==
        np.sign(y_test)
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )

    print("\n=========================================")

    print("ALPHACOMPILER TEST RESULTS")

    print("=========================================")

    print(
        f"Test Spearman IC: {test_ic:.6f}"
    )

    print(
        f"Test Pearson IC: {test_pearson:.6f}"
    )

    print(
        f"Test MAE: {test_mae:.6f}"
    )

    print(
        f"Test RMSE: {test_rmse:.6f}"
    )

    print(
        f"Directional Accuracy: "
        f"{directional_accuracy:.4%}"
    )

    print("\nBASELINE RESULTS")

    print(
        f"Baseline MAE: {baseline_mae:.6f}"
    )

    print(
        f"Baseline RMSE: {baseline_rmse:.6f}"
    )

    # =====================================
    # SAVE PREDICTIONS
    # =====================================

    predictions_df = pd.DataFrame({

        "Date": test_dates,

        "Factor_Value": factor_test,

        "Actual_Return": y_test,

        "Predicted_Return": predicted_returns,

        "Baseline_Return": baseline_predictions

    })

    predictions_path = (
        OUTPUT_DIR /
        f"{ticker}_factor_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # =====================================
    # SAVE SELECTED FACTOR
    # =====================================

    factor_path = (
        OUTPUT_DIR /
        f"{ticker}_best_factor.txt"
    )

    factor_path.write_text(
        selected_expression,
        encoding="utf-8"
    )

    # =====================================
    # SAVE SELECTED MODEL
    # =====================================

    model_path = (
        OUTPUT_DIR /
        f"{ticker}_symbolic_model.pkl"
    )

    joblib.dump(
        selected_model,
        model_path
    )

    # =====================================
    # SAVE FEATURE SCALER
    # =====================================

    scaler_path = (
        OUTPUT_DIR /
        f"{ticker}_feature_scaler.pkl"
    )

    joblib.dump(
        feature_scaler,
        scaler_path
    )

    # =====================================
    # SAVE CALIBRATOR
    # =====================================

    calibrator_path = (
        OUTPUT_DIR /
        f"{ticker}_factor_calibrator.pkl"
    )

    joblib.dump(
        calibrator,
        calibrator_path
    )

    # =====================================
    # FINAL STATUS
    # =====================================

    print("\n=========================================")

    print("ARTIFACTS SAVED")

    print("=========================================")

    print(f"Factor: {factor_path}")

    print(f"Model: {model_path}")

    print(f"Feature scaler: {scaler_path}")

    print(f"Calibrator: {calibrator_path}")

    print(f"Predictions: {predictions_path}")

    print("\nAlphaCompiler discovery completed.")
    
if __name__ == "__main__":
    discover("AAPL")
