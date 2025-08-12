from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr, conint
from enum import Enum
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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

class LoginData(BaseModel):
    username: str
    password: str

# ============================================================
# PATIENT MODELS
# ============================================================
class PatientSymptoms(BaseModel):
    symptoms: List[str]

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

class Patient(BaseModel):
    patient_id: Optional[str] = None
    name: str
    age: conint(gt=0)
    gender: Gender
    blood_group: BloodGroup
    spo2: float
    temperature: float
    bpm: conint(gt=0)
    blood_pressure_systolic: conint(gt=0)
    blood_pressure_diastolic: conint(gt=0)
    blood_sugar: float
    symptoms: List[str]
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

class PatientSymptomsDetailed(BaseModel):
    chest_pain: bool = False
    shortness_of_breath: bool = False
    fever: bool = False
    cough: bool = False
    fatigue: bool = False
    dizziness: bool = False
    nausea: bool = False
    Confusion: bool = False
    abdominal_pain: bool = False
    headache: bool = False

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

