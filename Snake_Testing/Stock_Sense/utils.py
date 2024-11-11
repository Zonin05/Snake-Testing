# utils.py
import pandas as pd
import matplotlib.pyplot as plt
import time


def print_section_header(title):
    """
    Utility to print a section header for better readability.
    """
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50 + "\n")


def print_data_info(data, title="Data Information"):
    """
    Utility to print basic information about the dataset.
    """
    print_section_header(title)
    print("Data Shape:", data.shape)
    print("Data Columns:", data.columns)
    print("First 5 rows of the data:\n", data.head(), "\n")
    print("Missing Values:", data.isnull().sum(), "\n")


def log_time_taken(start_time, end_time):
    """
    Logs the time taken for a specific task or process.
    """
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")


def plot_multiple_graphs(data, graphs, title="Multiple Graphs"):
    """
    Plots multiple graphs for comparison.
    Parameters:
        data (DataFrame): The dataset to be plotted.
        graphs (list): A list of tuples where each tuple contains (column name, label for the plot).
    """
    plt.figure(figsize=(12, 8))
    for column, label in graphs:
        plt.plot(data[column], label=label)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def print_model_metrics(model, X_test, y_test):
    """
    Utility to print out key performance metrics of the trained model.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print_section_header("Model Performance Metrics")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")


def print_prediction_info(actual, predicted, start_date, end_date):
    """
    Utility to print prediction-related information and comparison.
    """
    print_section_header(f"Prediction Info ({start_date} to {end_date})")
    comparison_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    print(comparison_df.tail(), "\n")

