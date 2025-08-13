from fastapi import APIRouter, HTTPException
from datetime import datetime
from pymongo import ReturnDocument
from config import patient_collection, counter_collection, bed_collection, archive_collection
from models import Patient, PatientSymptoms, Department
from utils import calculate_mews, convert_symptoms
from ml_model import model, model_lock
import logging
from ml_model import load_model_and_scaler
import joblib 
import pandas as pd

router = APIRouter(prefix="/patients", tags=["Patients"])

logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def get_next_patient_id():
    if not await counter_collection.find_one({"_id": "patient_id"}):
        await counter_collection.insert_one({"_id": "patient_id", "sequence_value": 0})

    result = await counter_collection.find_one_and_update(
        {"_id": "patient_id"},
        {"$inc": {"sequence_value": 1}},
        return_document=ReturnDocument.AFTER
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate patient ID")
    return f"PT-{result['sequence_value']:03}"

async def get_next_bed_id(department: Department):
    dept_name = department.value
    bed = await bed_collection.find_one(
        {"department": dept_name, "occupied": False},
        sort=[("bed_id", 1)]
    )
    if not bed:
        raise HTTPException(status_code=400, detail=f"No beds available in {dept_name}")
    await bed_collection.update_one({"bed_id": bed["bed_id"]}, {"$set": {"occupied": True}})
    return bed["bed_id"]

@router.post("/add-patient")
async def add_patient(patient: dict):
    """
    Add a new patient to the system, assign a bed, calculate MEWS and severity.
    Accepts flat JSON with vitals and symptoms.
    """
    try:
        # Load model & scaler fresh (no globals)
        local_model = joblib.load("triage_model.pkl")
        local_scaler = joblib.load("scaler.pkl")

        # Load feature column order
        df = pd.read_csv("balanced_triage.csv")
        feature_columns = [c for c in df.columns if c != "severity_score"]
        logger.info(f"Feature columns: {feature_columns}")

        # Extract vitals
        name = patient.get("name")
        age = patient.get("age")
        gender = patient.get("gender")
        heart_rate = patient.get("heart_rate")
        systolic_bp = patient.get("systolic_bp")
        diastolic_bp = patient.get("diastolic_bp")
        resp_rate = patient.get("resp_rate")
        temperature = patient.get("temperature")
        spo2 = patient.get("spo2")

        # Symptoms as integers
        chest_pain = int(patient.get("chest_pain", 0))
        shortness_of_breath = int(patient.get("shortness_of_breath", 0))
        fever = int(patient.get("fever", 0))
        cough = int(patient.get("cough", 0))
        fatigue = int(patient.get("fatigue", 0))
        dizziness = int(patient.get("dizziness", 0))
        nausea = int(patient.get("nausea", 0))
        confusion = int(patient.get("confusion", 0))
        abdominal_pain = int(patient.get("abdominal_pain", 0))
        headache = int(patient.get("headache", 0))
        logger.info(f"Vitals before MEWS: HR={heart_rate} ({type(heart_rate)}), "
            f"SBP={systolic_bp} ({type(systolic_bp)}), "
            f"RR={resp_rate} ({type(resp_rate)}), "
            f"TEMP={temperature} ({type(temperature)})")
        # Calculate MEWS
        mews_score = calculate_mews(heart_rate, systolic_bp, resp_rate, temperature)
        logger.info(f"Calculated MEWS score: {mews_score}")

        # Build feature dict in correct order
        feature_dict = {
            "age": age,
            "chest_pain": chest_pain,
            "shortness_of_breath": shortness_of_breath,
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "dizziness": dizziness,
            "nausea": nausea,
            "confusion": confusion,
            "abdominal_pain": abdominal_pain,
            "headache": headache,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "spo2": spo2,
            "resp_rate": resp_rate,
            "temperature": temperature,
            "mews_score": mews_score
        }

        # Create ordered feature list
        features = [feature_dict[col] for col in feature_columns]
        logger.info(f"Features: {features}")
        # Scale and predict
        features_scaled = local_scaler.transform([features])
        severity = float(local_model.predict(features_scaled)[0])
        logger.info(f"Predicted severity score: {severity}")

        # Prepare patient dict for DB
        patient_dict = {
            "patient_id": await get_next_patient_id(),
            "name": name,
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "resp_rate": resp_rate,
            "temperature": temperature,
            "spo2": spo2,
            "chest_pain": chest_pain,
            "shortness_of_breath": shortness_of_breath,
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "dizziness": dizziness,
            "nausea": nausea,
            "confusion": confusion,
            "abdominal_pain": abdominal_pain,
            "headache": headache,
            "severity_score": severity,
            "mews_score": mews_score,
            "created_at": datetime.utcnow().isoformat()
        }
        await patient_collection.insert_one(patient_dict)

        # Assign bed
        # department = patient.get("department", "General")
        # bed_id = await get_next_bed_id(department)
        # patient_dict["department"] = department
        # patient_dict["bed_id"] = bed_id

        # Save to DB

        return {
            "message": "Patient added successfully",
            "patient_id": patient_dict["patient_id"],
            "severity_score": severity,
            "mews_score": mews_score
        }

    except Exception as e:
        print(f"Error in add_patient: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add patient: {str(e)}")

@router.post("/symptoms")
async def patient_symptoms(data: PatientSymptoms):
    patient = await patient_collection.find_one({"patient_id": data.patient_id})
    if not patient: 
        raise HTTPException(status_code=404, detail="Patient not found")
    await patient_collection.update_one(
        {"patient_id": data.patient_id},
        {"$set": {"symptoms": [symptom.value for symptom in data.symptoms]}}
    )
    return {"message": "Symptoms added successfully"}

@router.get("")
async def get_patients():
    patients_cursor = patient_collection.find()
    patients = []
    async for patient in patients_cursor:
        patient["_id"] = str(patient["_id"])
        patient["symptoms"] = convert_symptoms(patient.get("symptoms", []))
        patients.append(patient)
    return patients

@router.get("/{patient_id}")
async def get_patient_by_id(patient_id: str):
    patient = await patient_collection.find_one({"patient_id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient["_id"] = str(patient["_id"])
    patient["symptoms"] = convert_symptoms(patient.get("symptoms", []))
    return patient

@router.put("/{patient_id}")
async def update_patient(patient_id: str, updated_patient: Patient):
    maws_score = calculate_maws(updated_patient.vitals)
    update_dict = updated_patient.model_dump()
    update_dict["maws"] = maws_score
    update_dict["alert"] = maws_score >= 6
    update_dict["$push"] = {"vital_history": {"timestamp": datetime.utcnow(), "vitals": updated_patient.vitals.model_dump()}}

    result = await patient_collection.update_one({"patient_id": patient_id}, {"$set": update_dict, "$push": update_dict["$push"]})
    if result.modified_count:
        updated = await patient_collection.find_one({"patient_id": patient_id})
        updated["_id"] = str(updated["_id"])
        updated["symptoms"] = convert_symptoms(updated.get("symptoms", []))
        return {"message": "Patient updated successfully"}
    raise HTTPException(status_code=404, detail="Patient not found")

@router.post("/discharge/{patient_id}")
async def discharge_patient(patient_id: str):
    patient = await patient_collection.find_one({"patient_id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    await bed_collection.update_one({"bed_id": patient["bed_id"]}, {"$set": {"occupied": False}})
    patient["status"] = "Discharged"
    patient["discharged_at"] = datetime.utcnow()
    await archive_collection.insert_one(patient)
    await patient_collection.delete_one({"patient_id": patient_id})
    return {"message": "Patient discharged and archived"}
