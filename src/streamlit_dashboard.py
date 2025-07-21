import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

FASTAPI_URL = "http://localhost:8000/predict" # Will be 'http://api:8000/predict' in Docker Compose

st.set_page_config(layout="wide")
st.title("Customer Churn Prediction Dashboard")

st.header("Make a Prediction")

with st.form("churn_prediction_form"):
    # Input fields for features (match your ChurnFeatures Pydantic model)
    tenure = st.slider("Tenure (months)", 1, 72, 24)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 50.0, step=0.1)
    total_charges = st.number_input("Total Charges", 0.0, 6000.0, 1200.0, step=0.1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    # ... add more feature inputs

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_data = {
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents
            # ... include all features
        }
        st.write("Input Data:", input_data)

        try:
            response = requests.post(FASTAPI_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()
                prediction = "Churn" if result["prediction"] == 1 else "No Churn"
                probability = result["probability"]
                st.success(f"Prediction: **{prediction}** with probability **{probability:.2f}**")
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the prediction API. Make sure the FastAPI service is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.header("Model Monitoring (via Grafana)")
st.write("Access the Grafana dashboard to view real-time model performance and data drift metrics.")
st.info("Grafana is typically accessible at `http://localhost:3000`")

st.markdown("---")
st.header("MLflow UI")
st.write("Access the MLflow UI to track experiments, manage models, and view artifacts.")
st.info("MLflow UI is typically accessible at `http://localhost:5000`")