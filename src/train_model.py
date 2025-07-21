import pandas as pd
from pycaret.classification import *
import mlflow
import mlflow.pyfunc # Needed for logging PyCaret models with MLflow

# Set MLflow tracking URI (will be 'http://mlflow:5000' in Docker Compose)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Churn Prediction")

def train_and_log_model(data_path: str, target_column: str):
    df = pd.read_csv(data_path)

    # Initialize PyCaret environment, logging to MLflow
    exp = setup(data=df,
                target=target_column,
                session_id=123,
                log_experiment=True, # Crucial for MLflow integration
                experiment_name='Churn_Prediction_PyCaret',
                silent=True,
                n_jobs=-1)

    # Compare and select the best model
    best_model = compare_models(exclude=['svm', 'ridge']) # Exclude some models for speed

    # Evaluate the best model (metrics are automatically logged by PyCaret)
    # evaluate_model(best_model) # Opens UI, not ideal for CI/CD

    # Log the best model with MLflow
    # PyCaret's `setup` already logs the run, but we can specifically register the model
    print(f"Best Model: {best_model}")
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # Register the model to MLflow Model Registry
    # This will create a new version of the model in the registry
    save_model(best_model, 'churn_prediction_model', model_name='ProjetAIModel',
               serialization_format='mlflow') # Saves as MLflow artifact, registers in registry

    print("Model trained, logged, and registered to MLflow.")

if __name__ == "__main__":
    # Use DVC-tracked processed data
    train_and_log_model("data/processed/features.csv", "Churn")