# retrain_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import shutil
import tempfile

def retrain_model():
    """Retrains model with feedback data and replaces old model when done."""
    print("🔄 Starting model retraining...")

    # Load original dataset
    original = pd.read_csv("RANDOM.csv")

    # Merge with feedback if exists
    if os.path.exists("feedback.csv"):
        feedback = pd.read_csv("feedback.csv")
        feedback.columns = list(original.columns[:-2]) + ["severity_score"]
        data = pd.concat([original.drop(columns=["condition"]), feedback], ignore_index=True)
    else:
        data = original.drop(columns=["condition"])

    X = data.drop(columns=["severity_score"])
    y = data["severity_score"]

    # Train a new model
    new_model = RandomForestRegressor(n_estimators=100, random_state=42)
    new_model.fit(X, y)

    # Save new model to a temp file, then replace old one atomically
    temp_path = tempfile.mktemp(suffix=".pkl")
    joblib.dump(new_model, temp_path)

    shutil.move(temp_path, "triage_model.pkl")
    print("✅ Model retrained and replaced successfully.")

if __name__ == "__main__":
    retrain_model()
