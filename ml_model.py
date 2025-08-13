import joblib
from threading import Lock
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# Path to your saved model file
MODEL_PATH = "triage_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Thread lock to ensure safe concurrent access
model_lock = Lock()

CSV_PATH = "balanced_triage.csv"
MODEL_PATH = "triage_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None
feature_columns = None

def load_model_and_scaler():
    global model, scaler, feature_columns
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found. Train first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(CSV_PATH)
    feature_columns = [c for c in df.columns if c != "severity_score"]


def predict_single(input_dict: dict) -> float:
    """Predict severity score for a single input dictionary."""
    global model, scaler, feature_columns
    with model_lock:
        # Create a copy to avoid modifying the input
        input_dict = input_dict.copy()
        # Map respiratory_rate to resp_rate if present
        if "respiratory_rate" in input_dict:
            input_dict["resp_rate"] = input_dict.pop("respiratory_rate")
        # Ensure input has all required feature columns, default to 0 if missing
        input_data = {col: input_dict.get(col, 0) for col in feature_columns}
        try:
            X = pd.DataFrame([input_data])[feature_columns]
            X_scaled = scaler.transform(X)
            return float(model.predict(X_scaled)[0])
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

def retrain_model():
    """Retrain the model using the latest CSV data."""
    global model, scaler, feature_columns
    df = pd.read_csv(CSV_PATH).dropna()
    X = df[feature_columns]
    y = df["severity_score"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    mse = mean_squared_error(y, model.predict(X_scaled))
    r2 = r2_score(y, model.predict(X_scaled))
    print(f"[Retrain] MSE: {mse:.4f}, R²: {r2:.4f}")

def append_feedback_row(features: dict, correct_score: float):
    """Append feedback row to CSV."""
    row = {**features, "severity_score": correct_score}
    df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
