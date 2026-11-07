import yfinance as yf


def get_stock_data(symbol: str, period="2y"):
    df = yf.Ticker(symbol).history(period=period)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    return df.reset_index()


def get_multiple_stocks(symbols, period="2y"):
    result = {}

    for symbol in symbols:
        result[symbol] = get_stock_data(symbol, period)

    return result