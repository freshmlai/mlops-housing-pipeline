import os
import sqlite3
import pandas as pd
import json
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from prometheus_fastapi_instrumentator import Instrumentator


# ========================
# Config & Paths
# ========================
DB_PATH = "logs.db"
# Use a more robust path relative to this script's location
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXPERIMENT_NAME = "california-housing"
REGISTERED_MODEL_NAME = "CaliforniaHousingModel"

os.makedirs(DATA_DIR, exist_ok=True)


# ========================
# SQLite Logging Setup
# ========================
def init_db():
    """Initializes the SQLite database and logs table."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )'''
        )
        conn.commit()


def log_event(event_type: str, details: str):
    """Logs an event to the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (event_type, details) VALUES (?, ?)",
            (event_type, details),
        )
        conn.commit()


init_db()


# ========================
# Load Model & Features from MLflow
# ========================
def load_production_model_and_features():
    """
    Loads the model and feature list from the MLflow Model Registry
    using the '@production' alias.
    """
    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
        model = mlflow.sklearn.load_model(model_uri)

        client = MlflowClient()
        prod_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, "production"
        )
        run_id = prod_version.run_id

        downloaded_features_path = client.download_artifacts(
            run_id, "features.json"
        )

        with open(downloaded_features_path, 'r') as f:
            features = json.load(f)["features"]

        details = (
            f"Loaded model '{REGISTERED_MODEL_NAME}' version "
            f"{prod_version.version} from run {run_id}"
        )
        log_event("model_load", details)
        return model, features

    except MlflowException:
        details = (
            "Could not find a model with alias 'production'. "
            "Please set the alias in the MLflow UI."
        )
        log_event("error", details)
        return None, None
    except Exception as e:
        log_event("error", f"An unexpected error occurred loading model: {e}")
        return None, None


model, features = load_production_model_and_features()


# ========================
# FastAPI App
# ========================
app = FastAPI(title="California Housing Model API")
Instrumentator().instrument(app).expose(app)


# ========================
# Input & Output Schemas
# ========================
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class PredictionResponse(BaseModel):
    prediction: float


# ========================
# Prediction Endpoint
# ========================
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HousingInput) -> PredictionResponse:
    """Receives housing data and returns a prediction."""
    if model is None or features is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please check server logs."
        )

    try:
        input_df = pd.DataFrame([input_data.dict()], columns=features)
        prediction_value = float(model.predict(input_df)[0])

        # FIX: Reformat long line
        details = f"Input: {input_data.dict()}, Prediction: {prediction_value}"
        log_event("prediction", details)

        return PredictionResponse(prediction=prediction_value)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        )


# ========================
# Retraining Logic
# ========================
def train_model_from_csv(csv_path: str) -> float:
    """
    Trains a new model from a CSV and updates the in-memory model.
    NOTE: This does NOT update the model in the MLflow Registry.
    """
    global model, features
    df = pd.read_csv(csv_path)

    if 'MedHouseVal' not in df.columns:
        raise ValueError(
            "CSV must contain 'MedHouseVal' column for retraining."
        )

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    new_features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    new_model = LinearRegression()
    new_model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, new_model.predict(X_test))

    model, features = new_model, new_features
    print(f"✅ In-memory model updated. New MSE: {mse:.4f}")

    return mse


# ========================
# API Endpoints for Retraining
# ========================
@app.post("/upload")
async def upload_csv_and_retrain(file: UploadFile = File(...)):
    """Accepts a CSV file upload and triggers in-memory retraining."""
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file. Please upload a .csv file."
        )

    save_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        mse = train_model_from_csv(save_path)
        details = (
            f"Triggered by file upload: {file.filename}, New MSE: {mse:.4f}"
        )
        log_event("retrain", details)

        return {
            "message": f"Model retrained in-memory from {file.filename}",
            "new_mse": mse,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {e}"
        )


@app.post("/retrain")
async def manual_retrain(request: Request):
    """Triggers in-memory retraining from a server-side CSV file path."""
    data = await request.json()
    csv_path = data.get("csv_path")

    if not csv_path:
        raise HTTPException(
            status_code=400, detail="Missing 'csv_path' in request body"
        )
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404, detail=f"CSV not found at path: {csv_path}"
        )

    try:
        mse = train_model_from_csv(csv_path)
        details = f"Manual retrain from {csv_path}, New MSE: {mse:.4f}"
        log_event("retrain", details)

        return {
            "message": f"Model retrained in-memory from {csv_path}",
            "new_mse": mse,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {e}"
        )
