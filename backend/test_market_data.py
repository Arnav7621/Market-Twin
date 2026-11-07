from app.services.market_data import get_stock_data

df = get_stock_data("AAPL")

print(df.head())
print(df.shape)