from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
# DATA
# =========================================

def load_data(ticker):

    df = pd.read_csv(DATA_DIR / f"{ticker}.csv")

    df["Momentum_5D"] = df["Close"].pct_change(5)

    df["Momentum_20D"] = df["Close"].pct_change(20)

    df["Volatility_20D"] = df["Return"].rolling(20).std()

    df["Volume_Change"] = df["Volume"].pct_change()

    # Next trading session return.
    df["Target"] = df["Return"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=FEATURES + ["Target"])

    return df


# =========================================
# PROTECTED DIVISION
# =========================================

def protected_division(x1, x2):

    with np.errstate(divide="ignore", invalid="ignore"):

        result = np.divide(x1, x2)

    result = np.where(
        np.isfinite(result),
        result,
        0.0
    )

    return result


# =========================================
# CORRELATION FITNESS
# =========================================

def correlation_fitness(y, y_pred, sample_weight):

    if np.std(y_pred) < 1e-12:

        return -1.0

    correlation = np.corrcoef(y, y_pred)[0, 1]

    if not np.isfinite(correlation):

        return -1.0

    return float(correlation)


correlation_metric = make_fitness(
    function=correlation_fitness,
    greater_is_better=True,
    wrap=False
)


# =========================================
# ALPHACOMPILER
# =========================================

def discover(ticker="AAPL"):

    df = load_data(ticker)

    X = df[FEATURES].to_numpy()

    y = df["Target"].to_numpy()

    n = len(df)

    train_end = int(n * 0.70)

    X_train = X[:train_end]

    X_test = X[train_end:]

    y_train = y[:train_end]

    y_test = y[train_end:]

    # =====================================
    # FEATURE STANDARDIZATION
    # =====================================

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
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

    model = SymbolicRegressor(
        population_size=500,
        generations=15,
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
        parsimony_coefficient=0.0005,
        max_samples=0.8,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    print("\nTraining AlphaCompiler using correlation fitness...")

    model.fit(
        X_train_scaled,
        y_train
    )

    # =====================================
    # DISCOVERED FACTOR
    # =====================================

    factor_train = model.predict(
        X_train_scaled
    ).reshape(-1, 1)

    factor_test = model.predict(
        X_test_scaled
    ).reshape(-1, 1)

    expression = str(model._program)

    print("\nDISCOVERED FACTOR:")

    print(expression)

    if np.std(factor_train) < 1e-12:

        raise RuntimeError(
            "AlphaCompiler discovered a constant factor. "
            "Do not accept this result."
        )

    # =====================================
    # LINEAR FACTOR CALIBRATION
    # =====================================

    calibrator = LinearRegression()

    calibrator.fit(
        factor_train,
        y_train
    )

    predictions = calibrator.predict(
        factor_test
    )

    # =====================================
    # HISTORICAL MEAN BASELINE
    # =====================================

    baseline_predictions = np.full(
        len(y_test),
        np.mean(y_train)
    )

    # =====================================
    # METRICS
    # =====================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
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

    ic = np.corrcoef(
        y_test,
        predictions
    )[0, 1]

    print("\nTEST RESULTS")

    print(f"MAE: {mae:.6f}")

    print(f"RMSE: {rmse:.6f}")

    print(f"Information Coefficient: {ic:.6f}")

    print(f"Baseline MAE: {baseline_mae:.6f}")

    print(f"Baseline RMSE: {baseline_rmse:.6f}")

    print(f"Total Predictions: {len(predictions)}")

    # =====================================
    # SAVE PREDICTIONS
    # =====================================

    result = pd.DataFrame({
        "Date": df["Date"].iloc[train_end:].values,
        "Actual_Return": y_test,
        "Predicted_Return": predictions,
        "Baseline_Return": baseline_predictions
    })

    result.to_csv(
        OUTPUT_DIR / f"{ticker}_factor_predictions.csv",
        index=False
    )

    # =====================================
    # SAVE FACTOR
    # =====================================

    with open(
        OUTPUT_DIR / f"{ticker}_best_factor.txt",
        "w"
    ) as file:

        file.write(expression)

    # =====================================
    # SAVE MODEL ARTIFACTS
    # =====================================

    joblib.dump(
        feature_scaler,
        OUTPUT_DIR / f"{ticker}_feature_scaler.pkl"
    )

    joblib.dump(
        calibrator,
        OUTPUT_DIR / f"{ticker}_factor_calibrator.pkl"
    )

    joblib.dump(
        model,
        OUTPUT_DIR / f"{ticker}_symbolic_model.pkl"
    )

    print("\nAlphaCompiler completed successfully!")


if __name__ == "__main__":

    discover("AAPL")