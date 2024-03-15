from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.training import initializing_model, fitting_model, save_model
from src.preprocessing import preprocess, load_saved_minmax
from src.eda import perform_eda
from src.inference import _load_model, inference_load_model_dev, preprocessing_inference
import pandas as pd
import os

app = FastAPI()

# Define request body model
class InputData(BaseModel):
    # Define fields based on your input data
    field1: float
    field2: float
    # Add more fields as needed

# Initialize and load model
model = initializing_model()
model_path = "assets/my_model"
if os.path.exists(model_path):
    model.load_weights(model_path)
else:
    raise FileNotFoundError(f"Model weights not found at {model_path}")

# Load saved minmax scaler
min_max_scaler = load_saved_minmax("assets")

# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert input data to pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Preprocess data
    preprocessed_data = preprocessing_inference(min_max_scaler, input_df)

    # Perform inference
    prediction = inference_load_model_dev(model, preprocessed_data)

    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=1000)
