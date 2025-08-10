from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import mlflow.sklearn
import logging
from pathlib import Path
import sqlite3
import pandas as pd
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import json
from typing import Dict

# ------------------------
# Config
# ------------------------
MODEL_URI = (
    "runs:/9c84758eb3c74a9788827a9039093866/model"
)
LOG_FILE = Path(__file__).parent / "prediction_logs.log"
DB_FILE = Path(__file__).parent / "prediction_logs.db"
FEATURE_NAMES_FILE = Path(__file__).parent / "feature_names.json"

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------
# Load feature names
# ------------------------
try:
    with open(FEATURE_NAMES_FILE, "r") as fh:
        FEATURE_NAMES = json.load(fh)
except Exception:
    logging.exception("Failed to load feature names from JSON")
    raise

# ------------------------
# Prometheus metrics
# ------------------------
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

# ------------------------
# Database helpers
# ------------------------
def init_db() -> None:
    """Create the prediction_logs table if it does not exist."""
    sql = (
        "CREATE TABLE IF NOT EXISTS prediction_logs ("
        "timestamp TEXT DEFAULT CURRENT_TIMESTAMP, "
        "input TEXT, prediction REAL)"
    )
    # Use a short-lived connection to avoid cross-thread issues.
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()


def log_to_db(input_data: Dict, prediction: float) -> None:
    """Insert one prediction record into the SQLite DB."""
    insert_sql = (
        "INSERT INTO prediction_logs (input, prediction) VALUES (?, ?)"
    )
    # New connection per invocation — safe with FastAPI threads/workers.
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute(insert_sql, (str(input_data), float(prediction)))
        conn.commit()


# ------------------------
# Load model from MLflow
# ------------------------
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception:
    logging.exception("Failed to load model from MLflow")
    raise

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="California Housing Model API")

# Ensure DB exists when the app starts
@app.on_event("startup")
def on_startup():
    init_db()


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
# Prediction endpoint
# ------------------------
@app.post("/predict")
def predict(input_data: HousingInput):
    try:
        REQUEST_COUNT.inc()

        # Build DataFrame with the exact columns used in training
        input_df = pd.DataFrame(
            [[
                input_data.MedInc,
                input_data.HouseAge,
                input_data.AveRooms,
                input_data.AveBedrms,
                input_data.Population,
                input_data.AveOccup,
                input_data.Latitude,
                input_data.Longitude,
            ]],
            columns=FEATURE_NAMES,
        )

        pred_value = float(model.predict(input_df)[0])

        # Log to file and DB
        logging.info(
            f"Input: {input_data.dict()} | "
            f"Prediction: {pred_value:.4f}"
        )
        log_to_db(input_data.dict(), pred_value)

        return {"prediction": pred_value}

    except Exception as exc:
        logging.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


# ------------------------
# Metrics endpoint for Prometheus
# ------------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
