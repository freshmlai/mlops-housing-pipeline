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
    model = mlflow.sklearn.load_model(
        MODEL_URI
    )
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
