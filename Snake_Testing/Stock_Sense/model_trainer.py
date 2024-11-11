# model_trainer.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pandas as pd


def train_model(data, target_column='Close'):
    # Splitting features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and test sets")

    # Initial model training
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    print("Model trained on training set")

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")

    # Evaluation on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluation Metrics:\n - Mean Squared Error: {mse}\n - Mean Absolute Error: {mae}\n - R-squared: {r2}")

    # Save the trained model
    joblib.dump(best_model, 'stock_predictor_model.pkl')
    print("Best model saved as stock_predictor_model.pkl")

    return best_model


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = {
        'Close': [150, 152, 153, 150, 149, 152, 153, 155, 156, 154, 155, 157, 160, 158, 157],
        'MA_10': [None, None, None, None, None, 150, 151, 152, 153, 153, 154, 155, 156, 157, 156],
        'RSI': [45, 50, 55, 60, 62, 58, 53, 49, 47, 55, 52, 50, 48, 51, 53]
    }
    df = pd.DataFrame(sample_data).dropna()  # Drop rows with missing values for this example

    # Train model on sample data
    model = train_model(df)
