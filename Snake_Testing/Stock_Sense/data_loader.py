import pandas as pd
import yfinance as yf
import numpy as np


def load_data(ticker, start_date, end_date):

    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for ticker {ticker} from {start_date} to {end_date}.")
            return None
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def preprocess_data(data):
    if data is None or data.empty:
        print("No data to preprocess.")
        return None

    data = data.dropna()

    data['Daily_Return'] = data['Adj Close'].pct_change()
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
    data['Volatility_20'] = data['Adj Close'].rolling(window=20).std()
    data = data.dropna()

    print("Data preprocessed successfully")
    return data


def calculate_technical_indicators(data):
    # Calculate Relative Strength Index (RSI)
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Moving Average Convergence Divergence (MACD)
    exp1 = data['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    print("Technical indicators calculated successfully")
    return data