import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict

import mlflow
from mlflow.exceptions import MlflowException
from prometheus_fastapi_instrumentator import Instrumentator

# ========================
# Config & Paths
# ========================

DB_PATH = "logs.db"
DATA_DIR = "data"
EXPERIMENT_NAME = "california-housing"
REGISTERED_MODEL_NAME = "CaliforniaHousingModel"

os.makedirs(DATA_DIR, exist_ok=True)

# ========================
# SQLite Logging Setup
# ========================

def init_db():
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
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (event_type, details) VALUES (?, ?)",
            (event_type, details),
        )
        conn.commit()

init_db()

# ========================
# Load model and features from MLflow Production stage
# ========================

def load_production_model_and_features():
    try:
        # Load production model URI
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)

        # Features are stored as artifact "feature_names.json" in the same run
        client = mlflow.tracking.MlflowClient()
        # Get latest production version info
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if not versions:
            raise MlflowException(f"No production model version found for {REGISTERED_MODEL_NAME}")
        prod_version = versions[0]
        run_id = prod_version.run_id

        # Download feature_names.json artifact
        tmp_dir = os.path.join(DATA_DIR, "tmp_features")
        os.makedirs(tmp_dir, exist_ok=True)
        client.download_artifacts(run_id, "features.json", tmp_dir)
        features_path = os.path.join(os.path.dirname(__file__), "feature_names.json")
        features = pd.read_json(features_path)["features"].tolist() if os.path.exists(features_path) else None
        if features is None:
            raise Exception("Feature names artifact not found or empty")

        return model, features

    except Exception as e:
        print(f"Error loading production model and features: {e}")
        return None, None

# Load once at startup
model, features = load_production_model_and_features()

# ========================
# FastAPI app
# ========================

app = FastAPI(title="California Housing Model API")

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# ========================
# Input Schema
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

# ========================
# Prediction Endpoint
# ========================

@app.post("/predict")
async def predict(input_data: HousingInput) -> Dict[str, float]:
    global model, features
    # Reload model and features each time, or cache and reload periodically
    if model is None or features is None:
        model, features = load_production_model_and_features()
        if model is None or features is None:
            raise HTTPException(status_code=400, detail="Production model not available")

    try:
        input_dict = input_data.dict()
        # Ensure features order
        input_df = pd.DataFrame([input_dict], columns=features)

        prediction = float(model.predict(input_df)[0])
        log_event(
            "prediction",
            f"Input: {input_dict}, Prediction: {prediction}",
        )
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ========================
# Upload CSV & Auto Retrain (reuse your existing train_model_from_csv logic)
# ========================

def train_model_from_csv(csv_path: str) -> float:
    import joblib
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    global model, features
    df = pd.read_csv(csv_path)

    if 'MedHouseVal' not in df.columns:
        raise ValueError("CSV must contain 'MedHouseVal' column")

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    new_model = LinearRegression()
    new_model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, new_model.predict(X_test))

    # Save model & features locally for fallback if needed
    joblib.dump(new_model, "../model.pkl")
    joblib.dump(list(X.columns), "../features.pkl")

    # Update in-memory model and features
    model, features = new_model, list(X.columns)

    return mse

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    save_path = os.path.join(DATA_DIR, filename)
    try:
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())

        mse = train_model_from_csv(save_path)
        log_event(
            "retrain",
            (
                f"Triggered by file upload: {filename}, MSE: {mse}"
            ),
        )

        return {
            "message": f"Model retrained from {filename}",
            "mse": mse,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain(request: Request):
    data = await request.json()
    csv_path = data.get("csv_path")

    if not csv_path:
        raise HTTPException(status_code=400, detail="Missing 'csv_path'")
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=400,
            detail=f"CSV not found at {csv_path}",
        )

    try:
        mse = train_model_from_csv(csv_path)
        log_event(
            "retrain",
            f"Manual retrain from {csv_path}, MSE: {mse}",
        )

        return {
            "message": f"Model retrained from {csv_path}",
            "mse": mse,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
