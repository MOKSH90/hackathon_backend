from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
from pymongo import ReturnDocument
from config import patient_collection, counter_collection, bed_collection, archive_collection
from models import Patient, PatientSymptoms, Department
from utils import calculate_mews, convert_symptoms
from ml_model import model, model_lock
from bson import ObjectId
 
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
    """
    Generate the next patient ID in the format PT-XXX.
    Initializes the counter if it doesn't exist.
    """
    try:
        # Ensure counter document exists
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
    except Exception as e:
        print(f"Error generating patient ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate patient ID: {str(e)}")

async def get_next_bed_id(department: str):
    """
    Assign the next available bed in the specified department.
    """
    try:
        bed = await bed_collection.find_one(
            {"department": department, "occupied": False},
            sort=[("bed_id", 1)]
        )
        if not bed:
            raise HTTPException(status_code=400, detail=f"No beds available in {department}")
        await bed_collection.update_one(
            {"bed_id": bed["bed_id"]},
            {"$set": {"occupied": True}}
        )
        return bed["bed_id"]
    except Exception as e:
        print(f"Error assigning bed in {department}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign bed: {str(e)}")

@router.post("/add-patient")
async def add_patient(patient: dict):
    """
    Add a new patient to the system, assign a bed, calculate MEWS and severity.
    Accepts flat JSON with vitals and symptoms.
    """
    try:
        data = await request.json()
        # Extract vitals and symptoms from incoming data
        name = data.get("name")
        age = data.get("age")
        gender = data.get("gender")
        heart_rate = data.get("heart_rate")
        systolic_bp = data.get("systolic_bp")
        diastolic_bp = data.get("diastolic_bp")
        resp_rate = data.get("resp_rate")
        temperature = data.get("temperature")
        spo2 = data.get("spo2")
        # Symptoms (as int 0/1)
        # Convert integer symptoms to boolean
        chest_pain = bool(data.get("chest_pain", 0))
        shortness_of_breath = bool(data.get("shortness_of_breath", 0))
        fever = bool(data.get("fever", 0))
        cough = bool(data.get("cough", 0))
        fatigue = bool(data.get("fatigue", 0))
        dizziness = bool(data.get("dizziness", 0))
        nausea = bool(data.get("nausea", 0))
        confusion = bool(data.get("confusion", 0))
        abdominal_pain = bool(data.get("abdominal_pain", 0))
        headache = bool(data.get("headache", 0))
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

def convert_object_ids(data):
    if isinstance(data, dict):
        return {k: convert_object_ids(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_object_ids(i) for i in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

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

@router.get("/all-patients")
async def get_patients():
    try:
        patients_cursor = patient_collection.find()
        patients = []
        async for patient in patients_cursor:
            patients.append(convert_object_ids(patient))
        return patients
    except Exception as e:
        print("Error retrieving patients:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patients: {str(e)}")

@router.get("/{patient_id}")
async def get_patient_by_id(patient_id: str):
    """
    Retrieve a patient by their patient ID.
    """
    try:
        patient = await patient_collection.find_one({"patient_id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        patient["_id"] = str(patient["_id"])
        return patient
    except Exception as e:
        print(f"Error retrieving patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve patient: {str(e)}")

@router.put("/{patient_id}")
async def update_patient(patient_id: str, updated_patient: dict):
    """
    Update a patient's information and append new vitals to history.
    """
    try:
        # Calculate MEWS using extended function
        mews_score = calc_mews_ext(
            updated_patient.get("heart_rate"), updated_patient.get("systolic_bp"), updated_patient.get("diastolic_bp"),
            updated_patient.get("resp_rate"), updated_patient.get("temperature"), updated_patient.get("spo2")
        )

        # Prepare ML features (you may need to adjust order as per your model)
        features = [
            updated_patient.get("age"), updated_patient.get("heart_rate"), updated_patient.get("systolic_bp"),
            updated_patient.get("diastolic_bp"), updated_patient.get("resp_rate"), updated_patient.get("spo2"),
            updated_patient.get("temperature"),
            updated_patient.get("chest_pain", 0), updated_patient.get("shortness_of_breath", 0),
            updated_patient.get("fever", 0), updated_patient.get("cough", 0), updated_patient.get("fatigue", 0),
            updated_patient.get("dizziness", 0), updated_patient.get("nausea", 0), updated_patient.get("confusion", 0),
            updated_patient.get("abdominal_pain", 0), updated_patient.get("headache", 0),
            mews_score
        ]

        # Predict severity
        try:
            with model_lock:
                severity = model.predict([features])[0]
        except Exception as e:
            print(f"ML model prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"ML model prediction failed: {str(e)}")

        # Update patient data
        update_dict = {
            "heart_rate": updated_patient.get("heart_rate"),
            "systolic_bp": updated_patient.get("systolic_bp"),
            "diastolic_bp": updated_patient.get("diastolic_bp"),
            "resp_rate": updated_patient.get("resp_rate"),
            "temperature": updated_patient.get("temperature"),
            "spo2": updated_patient.get("spo2"),
            "chest_pain": updated_patient.get("chest_pain", 0),
            "shortness_of_breath": updated_patient.get("shortness_of_breath", 0),
            "fever": updated_patient.get("fever", 0),
            "cough": updated_patient.get("cough", 0),
            "fatigue": updated_patient.get("fatigue", 0),
            "dizziness": updated_patient.get("dizziness", 0),
            "nausea": updated_patient.get("nausea", 0),
            "confusion": updated_patient.get("confusion", 0),
            "abdominal_pain": updated_patient.get("abdominal_pain", 0),
            "headache": updated_patient.get("headache", 0),
            "severity_score": severity,
            "mews_score": mews_score,
        }

        # Append new vitals to history
        result = await patient_collection.update_one(
            {"patient_id": patient_id},
            {
                "$set": update_dict,
                "$push": {
                    "vital_history": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "vitals": updated_patient
                    }
                }
            }
        )

        if result.modified_count:
            updated = await patient_collection.find_one({"patient_id": patient_id})
            updated["_id"] = str(updated["_id"])
            return {"message": "Patient updated successfully", "patient": updated}
        raise HTTPException(status_code=404, detail="Patient not found")
    except Exception as e:
        print(f"Error updating patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update patient: {str(e)}")

@router.post("/discharge/{patient_id}")
async def discharge_patient(patient_id: str):
    """
    Discharge a patient, free their bed, and archive their record.
    """
    try:
        patient = await patient_collection.find_one({"patient_id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Free the assigned bed
        await bed_collection.update_one(
            {"bed_id": patient["bed_id"]},
            {"$set": {"occupied": False}}
        )

        # Update patient status and discharge time
        patient["status"] = "Discharged"
        patient["discharged_at"] = datetime.utcnow().isoformat()

        # Archive patient record
        await archive_collection.insert_one(patient)

        # Remove patient from active collection
        await patient_collection.delete_one({"patient_id": patient_id})

        return {"message": "Patient discharged and archived"}
    except Exception as e:
        print(f"Error discharging patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to discharge patient: {str(e)}")