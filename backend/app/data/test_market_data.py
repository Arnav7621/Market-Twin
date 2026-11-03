
from market_data import download_stock_data

df = download_stock_data(
    ticker="AAPL",
    start="2020-01-01",
    end="2025-01-01"
)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)
