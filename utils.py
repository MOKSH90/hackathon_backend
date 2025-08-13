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

def calculate_mews(hr, sbp, rr, temp):
    """
    Calculate Modified Early Warning Score (MEWS)
    """
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