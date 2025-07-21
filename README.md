# Project Structure

projet-ia/
├── .dvc/                             # DVC internal files
├── data/
│   ├── raw/
│   │   └── customer_data.csv         # Raw, initial dataset
│   └── processed/
│       └── features.csv              # Processed and DVC-versioned data
├── models/                           # Store MLflow model artifacts (optional, MLflow can store)
├── notebooks/
│   └── EDA_and_initial_training.ipynb
├── src/
│   ├── data_prep.py                  # Data cleaning and feature engineering
│   ├── train_model.py                # PyCaret model training, MLflow logging
│   ├── predict_api.py                # FastAPI endpoint for inference
│   ├── streamlit_dashboard.py        # Streamlit app for UI
│   ├── monitoring/
│   │   ├── evidently_metrics.py      # Script to generate EvidentlyAI metrics
│   │   └── evidently_report.html     # Generated EvidentlyAI reports
│   └── utils.py                      # Common utility functions
├── tests/
│   ├── test_data_prep.py
│   └── test_model_inference.py
├── .gitignore
├── requirements.txt                  # Python dependencies
├── DVCfile                           # DVC pipeline stages (similar to dvc.yaml)
├── params.yaml                       # DVC parameters for reproducibility
├── Dockerfile.api                    # Dockerfile for FastAPI service
├── Dockerfile.streamlit              # Dockerfile for Streamlit app
├── docker-compose.yml                # Docker Compose orchestration
├── Jenkinsfile                       # Jenkins CI/CD pipeline definition
├── pipeline.groovy                   # Jenkins file to deploy all the project within params
├── prometheus.yml                    # Prometheus configuration
├── grafana/
│   └── dashboards/
│       └── churn_dashboard.json      # Grafana dashboard definition
└── README.md