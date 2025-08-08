from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import mlflow.sklearn
import logging
from pathlib import Path
import sqlite3
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST


# ------------------------
# Config
# ------------------------
MODEL_URI = "runs:/73f83bfae79c4d56a240fa34998366ac/model"  # Replace with your actual model URI
LOG_FILE = Path(__file__).parent / "prediction_logs.log"
DB_FILE = Path(__file__).parent / "prediction_logs.db"


# ------------------------
# Logging setup (to file)
# ------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(message)s"
    ),
)


# ------------------------
# SQLite setup for logging prediction requests
# ------------------------
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS prediction_logs (
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        input TEXT,
        prediction REAL
    )
    '''
)
conn.commit()


def log_to_db(input_data: dict, prediction: float):
    cursor.execute(
        "INSERT INTO prediction_logs (input, prediction) VALUES (?, ?)",
        (str(input_data), prediction),
    )
    conn.commit()


# ------------------------
# Prometheus metrics setup
# ------------------------
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")


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
        REQUEST_COUNT.inc()  # Increment Prometheus metric

        features = [
            [
                input_data.MedInc,
                input_data.HouseAge,
                input_data.AveRooms,
                input_data.AveBedrms,
                input_data.Population,
                input_data.AveOccup,
                input_data.Latitude,
                input_data.Longitude,
            ]
        ]

        prediction = model.predict(features)[0]

        # Log to file
        logging.info(
            f"Input: {input_data.dict()} | "
            f"Prediction: {prediction:.4f}"
        )

        # Log to SQLite DB
        log_to_db(input_data.dict(), prediction)

        return {"prediction": prediction}

    except Exception:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")


# ------------------------
# Metrics endpoint for Prometheus
# ------------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
