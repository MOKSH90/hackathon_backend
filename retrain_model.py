#!/usr/bin/env python3
"""
continuous_retrain.py

Retrains the triage model based on new feedback stored in a CSV file.

Expected feedback CSV format:
--------------------------------
Each row must contain:
    - All model features (same columns as balanced_triage.csv except severity_score)
    - severity_score: float (the corrected value from the doctor)

Process:
    1. Load base dataset (balanced_triage.csv).
    2. Append new feedback rows from feedback.csv.
    3. Retrain model on updated dataset.
    4. Save new model and scaler (timestamped and "latest" versions).
"""

import os
import json
from datetime import datetime
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- CONFIG ----------
CSV_PATH = "balanced_triage.csv"       # Main dataset
FEEDBACK_CSV = "feedback.csv"          # New feedback to append
MODEL_DIR = "models"                   # Where to save models
MODEL_LATEST = "triage_model.pkl"      # For API to load
SCALER_LATEST = "scaler.pkl"           # For API to load
FEATURES_JSON = "feature_columns.json" # Store feature order
LOG_CSV = "training_log.csv"

RFR_PARAMS = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1
}
# ----------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_csv(path, required_columns=None):
    """Load CSV and drop NaN rows only for required columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    if required_columns:
        df = df.dropna(subset=required_columns)
    return df

def get_feature_columns(df):
    """Return feature columns (exclude severity_score)."""
    if "severity_score" not in df.columns:
        raise ValueError("CSV must contain 'severity_score' column")
    return [c for c in df.columns if c != "severity_score"]

def append_feedback():
    """Append rows from feedback.csv into balanced_triage.csv, then clear feedback.csv."""
    if not os.path.exists(FEEDBACK_CSV):
        print("[INFO] No feedback.csv found — skipping append.")
        return

    feedback_df = load_csv(FEEDBACK_CSV)
    if feedback_df.empty:
        print("[INFO] feedback.csv is empty — skipping append.")
        return

    main_df = load_csv(CSV_PATH)
    expected_features = get_feature_columns(main_df)
    numeric_features = main_df[expected_features].select_dtypes(include=["number"]).columns.tolist()

    # Validate required columns
    missing = [c for c in numeric_features + ["severity_score"] if c not in feedback_df.columns]
    if missing:
        raise ValueError(f"Feedback CSV missing required numeric columns: {missing}")

    # Append feedback to main dataset
    updated_df = pd.concat([main_df, feedback_df], ignore_index=True)
    updated_df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Appended {len(feedback_df)} feedback rows to {CSV_PATH}")

    # Clear feedback.csv but keep headers
    pd.DataFrame(columns=feedback_df.columns).to_csv(FEEDBACK_CSV, index=False)
    print("[INFO] Cleared feedback.csv")

def train_and_save():
    """Train RandomForest model on updated dataset and save artifacts."""
    df = load_csv(CSV_PATH)
    features = get_feature_columns(df)

    # Keep numeric features only
    X = df[features].select_dtypes(include=["number"])
    y = df["severity_score"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(**RFR_PARAMS)
    model.fit(X_scaled, y)

    mse = mean_squared_error(y, model.predict(X_scaled))
    r2 = r2_score(y, model.predict(X_scaled))

    ts = timestamp_str()
    model_path = os.path.join(MODEL_DIR, f"triage_model_{ts}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ts}.pkl")

    # Save timestamped versions
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save latest versions for API
    joblib.dump(model, MODEL_LATEST)
    joblib.dump(scaler, SCALER_LATEST)

    # Save feature column order for API use
    with open(FEATURES_JSON, "w") as f:
        json.dump(features, f)

    # Log training
    log_row = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "num_rows": len(df),
        "mse_train": float(mse),
        "r2_train": float(r2),
        "params": json.dumps(RFR_PARAMS)
    }
    if os.path.exists(LOG_CSV):
        log_df = pd.read_csv(LOG_CSV)
        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_row])
    log_df.to_csv(LOG_CSV, index=False)

    print(f"[INFO] Saved model -> {model_path}")
    print(f"[INFO] Saved scaler -> {scaler_path}")
    print(f"[INFO] Train MSE: {mse:.4f}, R²: {r2:.4f}")

def retrain_from_feedback():
    """Main callable function to append feedback and retrain."""
    append_feedback()
    train_and_save()

if __name__ == "__main__":
    retrain_from_feedback()
