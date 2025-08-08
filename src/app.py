from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import logging
from pathlib import Path

# ------------------------
# Config
# ------------------------
MODEL_URI = "runs:/73f83bfae79c4d56a240fa34998366ac/model"
LOG_FILE = Path(__file__).parent / "prediction_logs.log"

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