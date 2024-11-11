from data_loader import load_data, preprocess_data, calculate_technical_indicators
from visualizer import plot_stock_price, display_data


def main():
    # Load stock data
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    data = load_data(ticker, start_date, end_date)

    if data is not None:
        # Preprocess data
        data = preprocess_data(data)

        # Calculate technical indicators
        data = calculate_technical_indicators(data)

        # Display processed data in a GUI window
        display_data(data)

        print("Sample of processed data:")
        print(data.head())

        # You can also plot stock price and predictions
        plot_stock_price(data, ticker)
        # Add more plotting and analysis as needed

if __name__ == "__main__":
    main()
