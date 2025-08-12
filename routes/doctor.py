from fastapi import APIRouter
from config import patient_collection

router = APIRouter(prefix="/doctor", tags=["Doctor"])

@router.get("/dashboard")
async def doctor_dashboard():
    patients = []
    async for patient in patient_collection.find().sort("maws", -1):
        patient["_id"] = str(patient["_id"])
        patients.append(patient)
    return {"patients": patients}
