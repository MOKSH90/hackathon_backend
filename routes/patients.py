from fastapi import APIRouter, HTTPException
from datetime import datetime
from pymongo import ReturnDocument
from config import patient_collection, counter_collection, bed_collection, archive_collection
from models import Patient, PatientSymptoms, Department
from utils import calculate_maws, convert_symptoms
from ml_model import model, model_lock

router = APIRouter(prefix="/patients", tags=["Patients"])

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

@router.post("")
async def add_patient(patient: Patient):
    try:
        patient_dict = patient.model_dump()
        patient_id = await get_next_patient_id()
        maws_score = calculate_maws(patient.vitals)

        features = [
            patient.age,
            patient.vitals.bp_systolic,
            patient.vitals.bp_diastolic,
            patient.vitals.spo2,
            patient.vitals.temperature,
            patient.vitals.blood_sugar,
            patient.vitals.bpm,
            patient.vitals.respiratory_rate,
            maws_score
        ]

        with model_lock:
            severity = model.predict([features])[0]

        if severity > 80:
            department = Department.icu
        elif severity > 50:
            department = Department.emergency
        else:
            department = Department.general_ward

        bed_id = await get_next_bed_id(department)

        patient_dict.update({
            "patient_id": patient_id,
            "maws": maws_score,
            "severity": severity,
            "department_assigned": department,
            "bed_id": bed_id,
            "alert": maws_score >= 6,
            "vital_history": [{"timestamp": datetime.utcnow(), "vitals": patient.vitals.dict()}]
        })

        result = await patient_collection.insert_one(patient_dict)
        patient_dict["_id"] = str(result.inserted_id)
        patient_dict["symptoms"] = convert_symptoms(patient_dict.get("symptoms", []))

        return {"message": "Patient added successfully", "patient": patient_dict}

    except Exception as e:
        print("Error while adding patient:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
