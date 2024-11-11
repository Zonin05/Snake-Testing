# predictor.py
import joblib
import pandas as pd
import numpy as np


def load_model(model_path='stock_predictor_model.pkl'):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
    except FileNotFoundError:
        print(f"Model file not found at path: {model_path}")
        return None


def prepare_input_data(data):
    if isinstance(data, pd.DataFrame):
        print("Data is already a DataFrame.")
        return data
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
        print("Data converted from dictionary to DataFrame.")
        return df
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        print("Data converted from NumPy array to DataFrame.")
        return df
    else:
        raise ValueError("Unsupported data format. Please provide a DataFrame, dictionary, or NumPy array.")


def make_prediction(model, data):
    if model is None:
        print("Model is not loaded. Please load a valid model.")
        return None

    data = prepare_input_data(data)  # Ensure data is in DataFrame format
    prediction = model.predict(data)
    print("Prediction made successfully")
    return prediction


def save_predictions(predictions, output_path='predictions.csv'):
    if isinstance(predictions, np.ndarray):
        predictions = pd.DataFrame(predictions, columns=['Predicted_Price'])

    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# Example usage
if __name__ == "__main__":
    model = load_model()

    example_data = {
        'Open': 135.67,
        'High': 137.45,
        'Low': 134.65,
        'Volume': 1000000,
    }

    input_data = prepare_input_data(example_data)
    prediction = make_prediction(model, input_data)

    if prediction is not None:
        save_predictions(prediction, 'output_predictions.csv')
        print(f"Predicted stock price: {prediction[0]}")
