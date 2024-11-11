import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk, Canvas, Text, Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_stock_price(data, ticker):
    """
    Plots the historical stock prices (Close) for a given ticker.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label=f'{ticker} Close Price', color='blue')
    ax.set_title(f'{ticker} Stock Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create Tkinter window for plotting
    root = Tk()
    root.title(f'{ticker} Stock Price Plot')

    # Add Matplotlib plot to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    root.mainloop()  # Show the Tkinter window


def display_data(data):
    """
    Displays the processed stock data in a scrollable Tkinter window.

    Parameters:
        - data: DataFrame containing stock price data
    """
    root = Tk()
    root.title("Processed Stock Data")

    # Create a Text widget with a Scrollbar
    scrollbar = Scrollbar(root)
    scrollbar.pack(side="right", fill="y")

    text_widget = Text(root, wrap="none", yscrollcommand=scrollbar.set, height=20, width=80)
    text_widget.pack(side="left", fill="both", expand=True)

    # Insert the DataFrame content into the Text widget
    text_widget.insert("1.0", data.to_string())

    # Configure scrollbar
    scrollbar.config(command=text_widget.yview)

    # Start the Tkinter event loop
    root.mainloop()


def plot_moving_average(data, ticker, window=10):
    """
    Plots the stock price with a moving average.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label=f'{ticker} Close Price', color='blue')
    ax.plot(data[f'MA_{window}'], label=f'{ticker} {window}-Day Moving Average', color='orange')
    ax.set_title(f'{ticker} Stock Price with {window}-Day Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create Tkinter window for plotting
    root = Tk()
    root.title(f'{ticker} Moving Average Plot')

    # Add Matplotlib plot to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    root.mainloop()  # Show the Tkinter window


def plot_rsi(data):
    """
    Plots the Relative Strength Index (RSI) indicator.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data['RSI'], label='RSI', color='purple')
    ax.axhline(70, linestyle='--', color='red', label='Overbought (70)')
    ax.axhline(30, linestyle='--', color='green', label='Oversold (30)')
    ax.set_title('Relative Strength Index (RSI)')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create Tkinter window for plotting
    root = Tk()
    root.title('RSI Plot')

    # Add Matplotlib plot to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    root.mainloop()  # Show the Tkinter window


def plot_predictions(actual, predicted):
    """
    Plots actual vs predicted stock prices for model evaluation.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual, label='Actual Prices', color='blue')
    ax.plot(predicted, label='Predicted Prices', color='red', linestyle='--')
    ax.set_title('Actual vs Predicted Stock Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create Tkinter window for plotting
    root = Tk()
    root.title('Actual vs Predicted Stock Prices')

    # Add Matplotlib plot to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    root.mainloop()  # Show the Tkinter window


def plot_error_distribution(errors):
    """
    Plots the distribution of prediction errors.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=20, color='gray', edgecolor='black')
    ax.set_title('Distribution of Prediction Errors')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create Tkinter window for plotting
    root = Tk()
    root.title('Error Distribution')

    # Add Matplotlib plot to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    root.mainloop()  # Show the Tkinter window
