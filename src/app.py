from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import logging
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from prometheus_client import Counter, generate_latest
from fastapi.responses import PlainTextResponse

# ------------------------
# Config
# ------------------------
MODEL_URI = "runs:/73f83bfae79c4d56a240fa34998366ac/model"  # replace if needed
LOG_FILE = Path(__file__).parent / "prediction_logs.log"
LOG_DB = Path(__file__).parent / "logs.db"

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------
# Load model from MLflow
# ------------------------
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    logging.exception("Failed to load model from MLflow")
    raise e

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="California Housing Model API")

# ------------------------
# Metrics
# ------------------------
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")


# ------------------------
# Input schema
# ------------------------
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# ------------------------
# Helper: Save to SQLite
# ------------------------
def save_request_to_db(features: dict, prediction: float):
    conn = sqlite3.connect(LOG_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            features TEXT,
            prediction REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO predictions (ts, features, prediction) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), json.dumps(features), prediction),
    )
    conn.commit()
    conn.close()


# ------------------------
# Routes
# ------------------------
@app.get("/")
def read_root():
    return {"message": "California Housing Model API is up"}


@app.post("/predict")
def predict(input_data: HousingInput):
    REQUEST_COUNT.inc()
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.dict()])
        prediction = float(model.predict(data)[0])

        # Log to file
        logging.info(f"Features={input_data.dict()} => Prediction={prediction}")

        # Save to SQLite
        save_request_to_db(input_data.dict(), prediction)

        return {"prediction": prediction}
    except Exception as e:
        ERROR_COUNT.inc()
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# Prometheus format (for monitoring tools)
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()


# JSON format (for humans / Swagger)
@app.get("/metrics-json")
def metrics_json():
    return {
        "prediction_requests_total": REQUEST_COUNT._value.get(),
        "prediction_errors_total": ERROR_COUNT._value.get(),
    }
