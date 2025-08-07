# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load dataset
data = pd.read_csv('data/housing.csv')

# Preprocessing (assuming 'target' is the target variable)
X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow tracking
mlflow.set_experiment('california-housing')

# Train and evaluate multiple models

# 1. Linear Regression
with mlflow.start_run(run_name="Linear Regression"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    
    # Log parameters, metrics, and model
    mlflow.log_param('model', 'Linear Regression')
    mlflow.log_metric('mse', mse)
    mlflow.sklearn.log_model(lr_model, 'model')
    print(f"Linear Regression MSE: {mse}")

# 2. Decision Tree
with mlflow.start_run(run_name="Decision Tree"):
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    
    # Log parameters, metrics, and model
    mlflow.log_param('model', 'Decision Tree')
    mlflow.log_metric('mse', mse)
    mlflow.sklearn.log_model(dt_model, 'model')
    print(f"Decision Tree MSE: {mse}")

