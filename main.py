from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, patients, beds, feedback, doctor
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import ml_model

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

class PredictionInput(BaseModel):
    age: int
    heart_rate: int
    systolic_bp: int
    respiratory_rate: int
    oxygen_sat: float
    temperature: float
    bleeding: int
    # Add other required model features here

class FeedbackData(BaseModel):
    features: dict
    correct_severity_score: float
    doctor_id: str
    notes: Optional[str] = None

@router.on_event("startup")
def startup_event():
    ml_model.load_model_and_scaler()

@router.post("/predict")
def predict(data: PredictionInput):
    try:
        prediction = ml_model.predict_single(data.dict())
        return {"severity_score": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/feedback")
def submit_feedback(feedback: FeedbackData):
    missing = [f for f in ml_model.feature_columns if f not in feedback.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    ml_model.append_feedback_row(feedback.features, feedback.correct_severity_score)
    ml_model.retrain_model()

    return {"message": "Feedback processed, model retrained"}

@router.get("/health")
def health_check():
    return {"status": "ok", "model_features": ml_model.feature_columns}

# Register routes
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(beds.router)
app.include_router(feedback.router)
app.include_router(doctor.router)
