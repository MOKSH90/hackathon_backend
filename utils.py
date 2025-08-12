from datetime import datetime
from typing import Dict

# ------------------------
# Common helper functions
# ------------------------

def get_timestamp() -> str:
    """Returns the current timestamp in ISO format."""
    return datetime.now().isoformat()

def generate_custom_id(prefix: str, last_id: str) -> str:
    """
    Generates a new custom ID with prefix and padded number.
    Example: ("PT", "PT-005") -> "PT-006"
    """
    if not last_id:
        return f"{prefix}-001"
    num = int(last_id.split("-")[1]) + 1
    return f"{prefix}-{str(num).zfill(3)}"

def calculate_maws(vitals: Dict) -> int:
    """
    Calculates Modified Early Warning Score (MAWS) based on vitals.
    vitals example: {"spo2": 95, "bpm": 110, "bp": "140/90", "temp": 38}
    """
    score = 0

    # Oxygen saturation
    if vitals.get("spo2") < 90:
        score += 3
    elif 90 <= vitals.get("spo2") <= 94:
        score += 2
    elif 95 <= vitals.get("spo2") <= 96:
        score += 1

    # Heart rate
    bpm = vitals.get("bpm")
    if bpm < 40 or bpm > 130:
        score += 3
    elif 40 <= bpm <= 50 or 110 <= bpm <= 130:
        score += 2
    elif 51 <= bpm <= 60 or 100 <= bpm <= 109:
        score += 1

    # Blood pressure (systolic)
    try:
        systolic = int(vitals.get("bp", "0/0").split("/")[0])
        if systolic < 70 or systolic > 200:
            score += 3
        elif 70 <= systolic <= 80 or 180 <= systolic <= 200:
            score += 2
        elif 81 <= systolic <= 100 or 150 <= systolic <= 179:
            score += 1
    except ValueError:
        pass

    # Temperature
    temp = vitals.get("temp")
    if temp < 35 or temp > 39:
        score += 3
    elif 35 <= temp <= 36 or 38 <= temp <= 39:
        score += 1

    return score

def convert_symptoms(symptoms):
    """
    Convert PatientSymptoms boolean values into numeric (0/1) for ML model.
    """
    if isinstance(symptoms, dict):
        # If already a dict (e.g., from request body)
        return [1 if symptoms.get(key) else 0 for key in symptoms]
    else:
        # If it's a Pydantic model instance
        return [1 if getattr(symptoms, field) else 0 for field in symptoms.model_fields]

def retrain_model(new_data=None):
    """
    Placeholder function for retraining your ML model.
    Replace this logic with actual retraining code.
    """
    print("Retraining model...")
    if new_data:
        print(f"Using {len(new_data)} new data samples")
    # TODO: add your real retraining logic here
    return {"status": "Model retrained successfully"}