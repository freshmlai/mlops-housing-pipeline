import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import json
from pathlib import Path

# ------------------------
# Load dataset
# ------------------------
data = pd.read_csv("../data/housing.csv")

# ------------------------
# Preprocessing (target is MedHouseVal)
# ------------------------
X = data.drop(columns=["MedHouseVal"])
y = data["MedHouseVal"]

# Save feature names for inference
feature_names_path = Path(__file__).parent / "feature_names.json"
with open(feature_names_path, "w") as f:
    json.dump(list(X.columns), f)

# ------------------------
# Split into training/testing
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# MLflow setup
# ------------------------
mlflow.set_experiment("california-housing")

# ------------------------
# 1. Linear Regression
# ------------------------
with mlflow.start_run(run_name="Linear Regression"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model", "Linear Regression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(lr_model, "model")

    print(f"Linear Regression MSE: {mse}")

# ------------------------
# 2. Decision Tree
# ------------------------
with mlflow.start_run(run_name="Decision Tree"):
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model", "Decision Tree")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(dt_model, "model")

    print(f"Decision Tree MSE: {mse}")
