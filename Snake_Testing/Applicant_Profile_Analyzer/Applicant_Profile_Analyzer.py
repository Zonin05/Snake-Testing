import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from prophet import Prophet

def get_num_applicants():
    while True:
        try:
            num_applicants = int(input("Enter the number of applicants: "))
            if num_applicants > 0:
                return num_applicants
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

NUM_APPLICANTS = get_num_applicants()  # Get user input for the number of applicants

def generate_applicant_data(num_applicants):
    data = {
        "Applicant_ID": [f"A{i + 1}" for i in range(num_applicants)],
        "Age": [random.randint(18, 70) for _ in range(num_applicants)],
        "Income": [random.randint(30000, 150000) for _ in range(num_applicants)],
        "Loan_Amount": [random.randint(5000, 50000) for _ in range(num_applicants)],
        "Employment_Status": [random.choice(["Employed", "Unemployed", "Self-Employed"]) for _ in
                              range(num_applicants)],
        "Credit_History_Score": [random.randint(300, 850) for _ in range(num_applicants)],
    }
    return pd.DataFrame(data)


def generate_applicant_data(num_applicants):
    data = {
        "Applicant_ID": [f"A{i + 1}" for i in range(num_applicants)],
        "Age": [random.randint(18, 70) for _ in range(num_applicants)],
        "Income": [random.randint(30000, 150000) for _ in range(num_applicants)],
        "Loan_Amount": [random.randint(5000, 50000) for _ in range(num_applicants)],
        "Employment_Status": [random.choice(["Employed", "Unemployed", "Self-Employed"]) for _ in
                              range(num_applicants)],
        "Credit_History_Score": [random.randint(300, 850) for _ in range(num_applicants)],
    }
    return pd.DataFrame(data)


def calculate_summary_stats(df):
    summary = df.describe().transpose()
    print("Summary Statistics:\n", summary)
    return summary


def employment_status_distribution(df):
    status_counts = df["Employment_Status"].value_counts()
    print("\nEmployment Status Distribution:")
    print(status_counts)
    return status_counts


def categorize_income(df):
    income_bins = [0, 40000, 80000, 120000, float('inf')]
    income_labels = ["Low", "Medium", "High", "Very High"]
    df["Income_Category"] = pd.cut(df["Income"], bins=income_bins, labels=income_labels)
    print("\nIncome Category Distribution:")
    print(df["Income_Category"].value_counts())
    return df


def calculate_dti(df):
    df["Debt_to_Income_Ratio"] = (df["Loan_Amount"] / df["Income"]).round(2)
    print("\nDebt-to-Income Ratio Summary:")
    print(df["Debt_to_Income_Ratio"].describe())
    return df


def categorize_credit_score(df):
    credit_score_bins = [0, 500, 600, 700, 850]
    credit_score_labels = ["Poor", "Fair", "Good", "Excellent"]
    df["Credit_Score_Category"] = pd.cut(df["Credit_History_Score"], bins=credit_score_bins, labels=credit_score_labels)
    print("\nCredit Score Category Distribution:")
    print(df["Credit_Score_Category"].value_counts())
    return df


def categorize_age(df):
    age_bins = [18, 30, 45, 60, 70]
    age_labels = ["Young", "Adult", "Middle-Aged", "Senior"]
    df["Age_Category"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    print("\nAge Category Distribution:")
    print(df["Age_Category"].value_counts())
    return df


def save_report_to_csv(df, filename="applicant_data_report.csv"):
    df.to_csv(filename, index=False)
    print(f"\nOverview report saved as {filename}.")


def generate_overview_report(df):
    print("\n--- Overview Report ---")
    calculate_summary_stats(df)
    employment_status_distribution(df)
    categorize_income(df)
    calculate_dti(df)
    categorize_credit_score(df)
    categorize_age(df)
    save_report_to_csv(df)
    print("\nOverview Report Completed.")


# Visualization Functions
def plot_income_vs_credit_score(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Income', y='Credit_History_Score', hue='Income_Category', palette='viridis')
    plt.title('Income vs Credit Score')
    plt.xlabel('Income')
    plt.ylabel('Credit History Score')
    plt.show()


def plot_loan_amount_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Loan_Amount"], kde=True)
    plt.title('Loan Amount Distribution')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')
    plt.show()


# Machine Learning Model for Loan Approval Prediction
def loan_approval_predictor(df):
    # Create a binary loan approval column based on income, credit score, and loan amount
    df['Loan_Approved'] = np.where((df['Credit_History_Score'] >= 650) & (df['Income'] > 40000), 1, 0)

    # Features and labels
    features = ['Income', 'Loan_Amount', 'Credit_History_Score']
    X = df[features]
    y = df['Loan_Approved']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def loan_amount_forecast(df):
    # Simulate monthly data over a year for the loan amounts
    df['Month'] = pd.to_datetime(pd.date_range(start='1/1/2023', periods=NUM_APPLICANTS, freq='ME'))
    df_monthly = df.groupby(df['Month'].dt.to_period('M')).agg({'Loan_Amount': 'mean'}).reset_index()

    # Convert 'Month' to datetime before passing it to Prophet
    df_monthly['Month'] = df_monthly['Month'].dt.to_timestamp()  # Convert to datetime

    # Prepare data for Prophet
    df_prophet = df_monthly.rename(columns={'Month': 'ds', 'Loan_Amount': 'y'})

    # Initialize and fit the model
    model = Prophet()
    model.fit(df_prophet)

    # Make a future dataframe
    future = model.make_future_dataframe(df_prophet, periods=12, freq='M')

    # Predict
    forecast = model.predict(future)

    # Plot the forecast
    model.plot(forecast)
    plt.title('Loan Amount Forecast for the Next Year')
    plt.show()



# Statistical Analysis (e.g., Correlation Analysis)
def correlation_analysis(df):
    correlation_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


# Main Execution
applicant_data = generate_applicant_data(NUM_APPLICANTS)
generate_overview_report(applicant_data)

# Run ML Model
loan_approval_predictor(applicant_data)

# Time Series Forecasting
loan_amount_forecast(applicant_data)

# Statistical Analysis
correlation_analysis(applicant_data)

# Visualizations
plot_income_vs_credit_score(applicant_data)
plot_loan_amount_distribution(applicant_data)