from fastapi import APIRouter, HTTPException
from models import PredictionInput, FeedbackData
import ml_model

router = APIRouter(
    prefix="/triage",
    tags=["Triage System"]
)

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