from fastapi import APIRouter, HTTPException, BackgroundTasks
from config import patient_collection, feedback_collection
from models import Feedback, FeedbackInput
from utils import retrain_model
from datetime import datetime
import pandas as pd
import os

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/doctor")
async def doctor_feedback(feedback: Feedback):
    patient = await patient_collection.find_one({"patient_id": feedback.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    await feedback_collection.insert_one(feedback.model_dump())
    await patient_collection.update_one(
        {"patient_id": feedback.patient_id},
        {"$set": {"maws": feedback.adjusted_score, "alert": feedback.adjusted_score >= 6}}
    )
    return {"message": "Feedback recorded and patient updated"}

@router.post("/feedback")
async def feedback_input(input_data: FeedbackInput, background_tasks: BackgroundTasks):
    patient = await patient_collection.find_one({"patient_id": input_data.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    vitals = patient["vitals"]
    maws = patient["maws"]

    row = [
        vitals["bp_systolic"], vitals["bp_diastolic"], vitals["spo2"],
        vitals["temperature"], vitals["blood_sugar"], vitals["bpm"],
        vitals["respiratory_rate"], maws, input_data.actual_score
    ]

    df = pd.DataFrame([row])
    file_exists = os.path.isfile("RANDOM.csv")
    df.to_csv("RANDOM.csv", mode="a", header=not file_exists, index=False)

    background_tasks.add_task(retrain_model)
    return {"message": "Feedback saved and model retraining started in background."}
