from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, conint
from enum import Enum
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from models import PredictionInput

# ============================================================
# ML MODEL LOGIC
# ============================================================
# Define feature columns in the order expected by the model
feature_columns = [
    "age", "chest_pain", "shortness_of_breath", "fever", "cough", "fatigue",
    "dizziness", "nausea", "confusion", "abdominal_pain", "headache",
    "heart_rate", "systolic_bp", "resp_rate", "temperature", "mews_score"
]

# Global variables to store model, scaler, and feedback data
model = None
scaler = None
feedback_file = Path("feedback_data.csv")

def load_model_and_scaler():
    """Load the trained model and scaler at startup."""
    global model, scaler
    try:
        model = joblib.load("triage_model.pkl")  # Ensure model.pkl is in the project directory
        scaler = joblib.load("scaler.pkl")  # Ensure scaler.pkl is in the project directory
        print("[INFO] Model and scaler loaded successfully")
    except Exception as e:  # Fixed from ExceptionA to Exception
        print(f"[ERROR] Failed to load model or scaler: {e}")
        raise

def predict_single(data):
    """Make a prediction for a single input using the loaded model."""
    global model, scaler
    if model is None or scaler is None:
        raise ValueError("Model or scaler not loaded")

    # Convert input Pydantic model to a DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # Ensure features are in the correct order
    input_data = input_df[feature_columns].values

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    return float(prediction)  # Convert to float for JSON serialization

def append_feedback_row(features: dict, correct_severity_score: float):
    """Append feedback data to a CSV file for retraining."""
    # Ensure all required features are present
    feedback_data = {col: features.get(col, 0) for col in feature_columns}
    feedback_data["severity_score"] = correct_severity_score

    # Convert to DataFrame and append to CSV
    feedback_df = pd.DataFrame([feedback_data])
    
    # Check if feedback file exists, append without header if it does
    if feedback_file.exists():
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, mode='w', header=True, index=False)
    print("[INFO] Feedback data appended successfully")

def retrain_model():
    """Retrain the model using feedback data."""
    global model, scaler
    if not feedback_file.exists():
        print("[INFO] No feedback data available for retraining")
        return

    try:
        # Load feedback data
        feedback_data = pd.read_csv(feedback_file)
        if feedback_data.empty:
            print("[INFO] Feedback data is empty, skipping retraining")
            return

        # Prepare features and target
        X = feedback_data[feature_columns]
        y = feedback_data["severity_score"]

        # Scale features
        X_scaled = scaler.transform(X)

        # Retrain model
        model.fit(X_scaled, y)
        print("[INFO] Model retrained successfully")

        # Save the updated model
        joblib.dump(model, "model.pkl")
        print("[INFO] Updated model saved")
    except Exception as e:
        print(f"[ERROR] Failed to retrain model: {e}")
        raise

# ============================================================
# ENUMS
# ============================================================
class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"

class BloodGroup(str, Enum):
    A_positive = "A+"
    A_negative = "A-"
    B_positive = "B+"
    B_negative = "B-"
    AB_positive = "AB+"
    AB_negative = "AB-"
    O_positive = "O+"
    O_negative = "O-"

class Department(str, Enum):
    ICU = "ICU"
    Emergency = "Emergency"
    General = "General"

class Role(str, Enum):
    doctor = "doctor"
    nurse = "nurse"
    admin = "admin"

# ============================================================
# AUTH MODELS
# ============================================================
class SignupData(BaseModel):
    username: str
    password: str
    role: Role

class LoginData(BaseModel):
    username: str
    password: str

# ============================================================
# PATIENT MODELS
# ============================================================
class PatientSymptoms(BaseModel):
    symptoms: List[str]

def calculate_mews(hr, sbp, rr, temp):
    """Calculate Modified Early Warning Score (MEWS)."""
    score = 0
    if hr <= 40 or hr >= 130:
        score += 2
    elif 41 <= hr <= 50 or 111 <= hr <= 129:
        score += 1
    if sbp <= 70:
        score += 3
    elif 71 <= sbp <= 80:
        score += 2
    elif 81 <= sbp <= 100 or sbp >= 200:
        score += 1
    if rr <= 8 or rr >= 30:
        score += 2
    elif 21 <= rr <= 29:
        score += 1
    if temp < 35 or temp > 38.5:
        score += 2
    return score


class PatientSymptomsDetailed(BaseModel):
    chest_pain: bool = False
    shortness_of_breath: bool = False
    fever: bool = False
    cough: bool = False
    fatigue: bool = False
    dizziness: bool = False
    nausea: bool = False
    Confusion: bool = False
    abdominal_pain: bool = False
    headache: bool = False

# ============================================================
# BED MODELS
# ============================================================
class BedSetup(BaseModel):
    department: Department
    capacity: int

class BedAllocation(BaseModel):
    bed_id: str
    patient_id: str

# ============================================================
# FEEDBACK MODELS
# ============================================================
class FeedbackInput(BaseModel):
    name: str
    email: EmailStr
    message: str

class Feedback(FeedbackInput):
    id: Optional[str] = None
    created_at: Optional[str] = None

class FeedbackData(BaseModel):
    features: dict
    correct_severity_score: float
    doctor_id: str
    notes: Optional[str] = None

# ============================================================
# PREDICTION MODEL
# ============================================================


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(
    prefix="/triage",
    tags=["Triage System"]
)

@router.on_event("startup")
def startup_event():
    load_model_and_scaler()

@router.post("/predict")
def predict(data: PredictionInput | PatientSymptomsDetailed):
    print(f"[INFO] Received prediction request: {data}")
    try:
        # Convert to PredictionInput format if PatientSymptomsDetailed is received
        if isinstance(data, PatientSymptomsDetailed):
            data_dict = data.dict()
            # Convert booleans to integers and handle case-sensitive field names
            data_dict = {
                k.lower(): int(v) if isinstance(v, bool) else v
                for k, v in data_dict.items()
            }
            # Add missing fields (e.g., from Patient model or calculate_mews)
            data_dict.update({
                "age": data_dict.get("age", 0),  # Replace with actual value
                "heart_rate": data_dict.get("heart_rate", 0),
                "systolic_bp": data_dict.get("systolic_bp", 0),
                "resp_rate": data_dict.get("resp_rate", 0),
                "temperature": data_dict.get("temperature", 0.0),
                "mews_score": calculate_mews(
                    data_dict.get("heart_rate", 0),
                    data_dict.get("systolic_bp", 0),
                    data_dict.get("resp_rate", 0),
                    data_dict.get("temperature", 0.0)
                )
            })
            data = PredictionInput(**data_dict)
        
        prediction = predict_single(data)
        print(f"[INFO] Prediction result: {prediction}")
        return {"severity_score": prediction}
    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/feedback")
def submit_feedback(feedback: FeedbackData):
    missing = [f for f in feature_columns if f not in feedback.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    append_feedback_row(feedback.features, feedback.correct_severity_score)
    retrain_model()
    return {"message": "Feedback processed, model retrained"}

@router.get("/health")
def health_check():
    return {"status": "ok", "model_features": feature_columns}

# Register routes
app.include_router(router)
# Placeholder for other routers (ensure these exist)
try:
    from routes import auth, patients, beds, feedback, doctor
    app.include_router(auth.router)
    app.include_router(patients.router)
    app.include_router(beds.router)
    app.include_router(feedback.router)
    app.include_router(doctor.router)
except ImportError as e:
    print(f"[WARNING] Failed to import routes: {e}. Ensure routes modules exist.")