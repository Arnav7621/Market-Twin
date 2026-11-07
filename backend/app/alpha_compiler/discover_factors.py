
from pathlib import Path

import numpy as np
import pandas as pd

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.metrics import mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = DATA_DIR / "alpha_compiler"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURES = [
    "Return",
    "Momentum_5D",
    "Momentum_20D",
    "Volatility_20D",
    "Volume_Change"
]


def load_data(ticker):

    df = pd.read_csv(
        DATA_DIR / f"{ticker}.csv"
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

    # Target is next trading session return
    df["Target"] = df["Return"].shift(-1)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target"]
    )

    return df


def protected_division(x1, x2):

    with np.errstate(
        divide="ignore",
        invalid="ignore"
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


def discover(ticker="AAPL"):

    df = load_data(ticker)

    X = df[FEATURES].to_numpy()

    y = df["Target"].to_numpy()

    n = len(df)

    train_end = int(n * 0.70)

    X_train = X[:train_end]

    y_train = y[:train_end]

    X_test = X[train_end:]

    y_test = y[train_end:]

    div_function = make_function(
        function=protected_division,
        name="protected_division",
        arity=2
    )

    model = SymbolicRegressor(
        population_size=300,
        generations=10,
        tournament_size=20,
        stopping_criteria=0.01,
        const_range=(-1, 1),
        init_depth=(2, 4),
        function_set=[
            "add",
            "sub",
            "mul",
            div_function
        ],
        metric="mean absolute error",
        parsimony_coefficient=0.001,
        max_samples=0.8,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    print("\nTraining AlphaCompiler...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    expression = str(
        model._program
    )

    print("\nDISCOVERED FACTOR:")

    print(expression)

    print("\nTEST MSE:")

    print(mse)

    result = pd.DataFrame({
        "Date": df["Date"].iloc[train_end:],
        "Actual_Return": y_test,
        "Predicted_Return": predictions
    })

    result.to_csv(
        OUTPUT_DIR / f"{ticker}_factor_predictions.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / f"{ticker}_best_factor.txt",
        "w"
    ) as file:

        file.write(expression)

    print("\nFactor saved successfully!")


if __name__ == "__main__":

    discover("AAPL")
