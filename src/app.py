import os
import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict

# ========================
# Config & Paths
# ========================
MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"
DB_PATH = "logs.db"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# ========================
# Prometheus Metrics
# ========================
PREDICTION_COUNTER = Counter('prediction_requests_total', 'Total prediction requests')
RETRAIN_COUNTER = Counter('model_retrain_total', 'Total model retraining events')

# ========================
# SQLite Logging Setup
# ========================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_type TEXT,
                            details TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )''')
        conn.commit()

def log_event(event_type: str, details: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO logs (event_type, details) VALUES (?, ?)", (event_type, details))
        conn.commit()

init_db()

# ========================
# Load Model & Features
# ========================
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    return None, None

model, features = load_model()

# ========================
# Train Model Function
# ========================
def train_model_from_csv(csv_path: str) -> float:
    global model, features
    df = pd.read_csv(csv_path)

    if 'MedHouseValue' not in df.columns:
        raise ValueError("CSV must contain 'MedHouseValue' column")

    X = df.drop('MedHouseValue', axis=1)
    y = df['MedHouseValue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_model = LinearRegression()
    new_model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, new_model.predict(X_test))

    # Save model & features
    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    # Reload into memory
    model, features = load_model()

    return mse

# ========================
# FastAPI app
# ========================
app = FastAPI(title="California Housing Model API")

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
    if model is None or features is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")

    try:
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict], columns=features)

        prediction = float(model.predict(input_df)[0])
        PREDICTION_COUNTER.inc()
        log_event("prediction", f"Input: {input_dict}, Prediction: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ========================
# Upload CSV & Auto Retrain
# ========================
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
        RETRAIN_COUNTER.inc()
        log_event("retrain", f"Triggered by file upload: {filename}, MSE: {mse}")

        return {"message": f"Model retrained from {filename}", "mse": mse}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========================
# Manual Retrain Endpoint
# ========================
@app.post("/retrain")
async def retrain(request: Request):
    data = await request.json()
    csv_path = data.get("csv_path")

    if not csv_path:
        raise HTTPException(status_code=400, detail="Missing 'csv_path'")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail=f"CSV not found at {csv_path}")

    try:
        mse = train_model_from_csv(csv_path)
        RETRAIN_COUNTER.inc()
        log_event("retrain", f"Manual retrain from {csv_path}, MSE: {mse}")

        return {"message": f"Model retrained from {csv_path}", "mse": mse}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ========================
# Metrics Endpoint
# ========================
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
