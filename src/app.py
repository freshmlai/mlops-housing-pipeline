from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

# Load model from MLflow
model_uri = "runs:/73f83bfae79c4d56a240fa34998366ac/model"  # Replace with actual run ID or use "models:/ModelName/1"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI()

# Define the input schema
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {"message": "California Housing Model API is up"}

@app.post("/predict")
def predict(input_data: HousingInput):
    # Convert input to DataFrame
    data = pd.DataFrame([input_data.dict()])
    prediction = model.predict(data)[0]
    return {"prediction": prediction}
