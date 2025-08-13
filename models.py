from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr, conint
from enum import Enum
from typing import Optional, List


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
    name: str

class LoginData(BaseModel):
    username: str
    password: str

# ============================================================
# PATIENT MODELS
# ============================================================


def calculate_mews(hr, sbp, dbp, rr, temp, spo2):
    """
    Extended MEWS calculation with diastolic BP and SpO₂ included.
    Worst cases can yield very high scores.
    """
    score = 0

    # Heart rate
    if hr < 30 or hr > 170:
        score += 4  # extreme danger
    elif hr <= 40 or hr >= 130:
        score += 3
    elif 41 <= hr <= 50 or 111 <= hr <= 129:
        score += 1

    # Systolic BP
    if sbp < 60 or sbp > 220:
        score += 4
    elif sbp <= 70:
        score += 3
    elif 71 <= sbp <= 80:
        score += 2
    elif 81 <= sbp <= 100 or sbp >= 200:
        score += 1

    # Diastolic BP (non-standard but useful for worst cases)
    if dbp < 40 or dbp > 120:
        score += 3
    elif 40 <= dbp <= 50 or 100 <= dbp <= 110:
        score += 2
    elif 51 <= dbp <= 60 or 90 <= dbp <= 99:
        score += 1

    # Respiratory rate
    if rr < 6 or rr > 40:
        score += 4
    elif rr <= 8 or rr >= 30:
        score += 3
    elif 21 <= rr <= 29:
        score += 1

    # Temperature
    if temp < 32 or temp > 41:
        score += 4
    elif temp < 35 or temp > 38.5:
        score += 2

    # SpO₂ scoring (NEWS2 style)
    if spo2 < 80:
        score += 4
    elif spo2 < 85:
        score += 3
    elif 85 <= spo2 <= 89:
        score += 2
    elif 90 <= spo2 <= 94:
        score += 1

    return score

# class PatientSymptomsDetailed(BaseModel):
#     chest_pain: bool = False
#     shortness_of_breath: bool = False
#     fever: bool = False
#     cough: bool = False
#     fatigue: bool = False
#     dizziness: bool = False
#     nausea: bool = False
#     confusion: bool = False
#     abdominal_pain: bool = False
#     headache: bool = False

class Patient(BaseModel):
    patient_id: Optional[str] = None
    name: str
    age: conint(gt=0)
    gender: Gender
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    resp_rate: int
    temperature: float
    spo2: int
    chest_pain: bool = False
    shortness_of_breath: bool = False
    fever: bool = False
    cough: bool = False
    fatigue: bool = False
    dizziness: bool = False
    nausea: bool = False
    confusion: bool = False
    abdominal_pain: bool = False
    headache: bool = False
    department: Optional[Department] = None
    bed_id: Optional[str] = None
    severity_score: Optional[float] = None
    mews_score: Optional[int] = None
    created_at: Optional[str] = None

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


class PredictionInput(BaseModel):
    age: int
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    resp_rate: int
    spo2: float
    temperature: float

class FeedbackData(BaseModel):
    features: dict
    correct_severity_score: float
    doctor_id: str
    notes: Optional[str] = None

