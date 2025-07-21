from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
import uvicorn
from prometheus_client import start_http_server, Counter, Summary, Histogram

# Prometheus metrics
REQUEST_COUNT = Counter('fastapi_requests_total', 'Total number of requests')
PREDICTION_COUNT = Counter('fastapi_predictions_total', 'Total number of predictions')
REQUEST_LATENCY = Histogram('fastapi_request_latency_seconds', 'Request latency in seconds')

app = FastAPI(title="Customer Churn Prediction API")

# Load MLflow model (assuming it's registered in the MLflow Model Registry)
# In production, you'd likely specify a stage like "Production"
# Example: model_uri = "models:/ProjetAIModel/Production"
# For local testing, you might load from a run_id or local path
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ProjetAIModel")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production") # Or latest version: "models:/ProjetAIModel/latest"

# Set MLflow tracking URI to connect to the MLflow server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

try:
    # Load the model from MLflow Model Registry
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    print(f"Model '{MODEL_NAME}' in stage '{MODEL_STAGE}' loaded successfully.")
except Exception as e:
    print(f"Error loading MLflow model: {e}")
    model = None # Handle case where model might not be available yet

class ChurnFeatures(BaseModel):
    # Define your model input features here based on your dataset
    # Example features for churn prediction:
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Gender: str # 'Male', 'Female'
    SeniorCitizen: int # 0 or 1
    Partner: str # 'Yes', 'No'
    Dependents: str # 'Yes', 'No'
    # ... add all other features your model expects

@app.on_event("startup")
async def startup_event():
    # Start Prometheus metrics server
    start_http_server(8001) # Metrics will be available on port 8001
    print("Prometheus metrics server started on port 8001.")

@app.post("/predict")
async def predict_churn(features: ChurnFeatures):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

        try:
            # Convert Pydantic model to Pandas DataFrame for prediction
            input_df = pd.DataFrame([features.model_dump()])
            prediction = model.predict(input_df)
            PREDICTION_COUNT.inc()
            return {"prediction": int(prediction[0]), "probability": float(model.predict_proba(input_df)[:, 1][0])}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)