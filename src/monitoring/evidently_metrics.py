import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import json
import os
import requests # To fetch production data/predictions if available via API

# Define paths (adjust as needed, e.g., if pulling from database)
REFERENCE_DATA_PATH = "data/processed/features.csv" # Or a snapshot of production data
CURRENT_DATA_PATH = "data/new_production_data.csv" # Simulate new production data
REPORT_OUTPUT_PATH = "src/monitoring/evidently_report.html"

# Assume your model's target column and prediction column names
TARGET_COLUMN = "Churn"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"

def generate_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    # Ensure 'prediction' and 'probability' columns exist in current_df for ClassificationPreset
    if PREDICTION_COLUMN not in current_df.columns:
        # Example: If you're only getting raw features from production,
        # you might need to make predictions using your loaded model here.
        # For simplicity, let's assume `current_df` already has these.
        print(f"Warning: '{PREDICTION_COLUMN}' not found in current data. ClassificationPreset might be incomplete.")

    # Create a combined report with Data Drift and Classification Performance
    churn_report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(target_name=TARGET_COLUMN, prediction_name=PREDICTION_COLUMN,
                             probas_name=PROBABILITY_COLUMN)
    ])

    churn_report.run(reference_data=reference_df, current_data=current_df)
    churn_report.save_html(REPORT_OUTPUT_PATH)
    print(f"EvidentlyAI report saved to {REPORT_OUTPUT_PATH}")

    # Optional: Extract key metrics and log to MLflow/Prometheus
    # This part requires parsing the Evidently report's JSON output
    report_json = churn_report.as_dict()
    # You'd extract relevant metrics like:
    # data_drift_detected = report_json['metrics'][0]['result']['dataset_drift']
    # accuracy = report_json['metrics'][1]['result']['current']['accuracy']
    # Log these to MLflow using `mlflow.log_metric()` or expose via Prometheus endpoint.

if __name__ == "__main__":
    # Load reference data (e.g., your validation set or initial production data)
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    # Simulate loading new production data (in a real scenario, this would come from a database, data lake, etc.)
    # For a live API, you might have a data collector service that periodically logs inferences.
    try:
        # For demonstration, let's create a dummy current_data
        current_data = reference_data.sample(frac=0.2, random_state=42).copy()
        # Introduce some synthetic drift for demonstration
        current_data['MonthlyCharges'] = current_data['MonthlyCharges'] * 1.1 + np.random.normal(0, 5, len(current_data))
        current_data['prediction'] = (current_data['MonthlyCharges'] > 70).astype(int) # Simple dummy prediction
        current_data['probability'] = current_data['prediction'].apply(lambda x: 0.8 if x == 1 else 0.2)
        print("Generated dummy current data for EvidentlyAI.")
        generate_evidently_report(reference_data, current_data)
    except Exception as e:
        print(f"Error generating EvidentlyAI report: {e}")
        print("Ensure you have a 'prediction' and 'probability' column in your current data for full report.")