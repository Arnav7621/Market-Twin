
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

df = pd.read_csv(DATA_DIR / "AAPL.csv")

# Calculate simple daily returns
df["Return"] = df["Close"].pct_change()

# Calculate log returns
df["Log_Return"] = (
    df["Close"] / df["Close"].shift(1)
).apply(lambda x: __import__("math").log(x) if pd.notna(x) and x > 0 else float("nan"))

print("\nFirst 10 observations:")
print(df[["Date", "Close", "Return", "Log_Return"]].head(10))

print("\nMean daily return:")
print(df["Return"].mean())

print("\nDaily volatility:")
print(df["Return"].std())

print("\nData loaded and tested successfully!")
