from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import logging
from pathlib import Path

# ------------------------
# Config
# ------------------------
MODEL_URI = "runs:/73f83bfae79c4d56a240fa34998366ac/model"  # Replace with your actual model URI
LOG_FILE = Path(__file__).parent / "prediction_logs.log"


# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(message)s"
    ),
)

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
        # Prepare input for model (as 2D array)
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

        # Log the input and prediction
        logging.info(
            f"Input: {input_data.dict()} | "
            f"Prediction: {prediction:.4f}"
        )


        return {"prediction": prediction}

    except Exception:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
