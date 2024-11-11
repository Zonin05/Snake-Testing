# feature_engineering.py
import pandas as pd
import numpy as np


def add_moving_averages(data, windows=[10, 20, 50]):
    for window in windows:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
        print(f"Moving average with window {window} added")
    return data


def add_technical_indicators(data):
    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    print("RSI indicator added")

    # Calculate MACD (Moving Average Convergence Divergence)
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    print("MACD and Signal Line added")

    # Calculate Bollinger Bands
    data['20_SMA'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['20_SMA'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['20_SMA'] - (data['Close'].rolling(window=20).std() * 2)
    print("Bollinger Bands added")

    return data


def add_volatility(data, window=10):
    data['Volatility'] = data['Close'].pct_change().rolling(window=window).std()
    print(f"Volatility with window {window} added")
    return data


# Example usage
if __name__ == "__main__":
    sample_data = {
        'Close': [150, 152, 153, 150, 149, 152, 153, 155, 156, 154, 155, 157, 160, 158, 157]
    }
    df = pd.DataFrame(sample_data)
    df = add_moving_averages(df)
    df = add_technical_indicators(df)
    df = add_volatility(df)

    print("\nSample of processed data with new features:")
    print(df)
