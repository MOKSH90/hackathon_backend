# models.py
from pydantic import BaseModel, Field, EmailStr, conint
from enum import Enum
from typing import Optional, List

# -------------------------
# ENUMS
# -------------------------
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

# -------------------------
# AUTH MODELS
# -------------------------
class SignupData(BaseModel):
    username: str
    email: EmailStr
    password: str
    hospital_name: str
    role: Role

class LoginData(BaseModel):
    email: str
    password: str

# -------------------------
# PATIENT MODELS
# -------------------------
class PatientSymptoms(BaseModel):
    symptoms: List[str]

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
    maws_score: Optional[int] = None
    created_at: Optional[str] = None

# -------------------------
# BED MODELS
# -------------------------
class BedSetup(BaseModel):
    department: Department
    capacity: int

class BedAllocation(BaseModel):
    bed_id: str
    patient_id: str

class PatientSymptoms(BaseModel):
    chest_pain: bool = False
    headache: bool = False
    dizziness: bool = False
    back_pain: bool = False
    shortness_of_breath: bool = False
    nausea: bool = False
    fatigue: bool = False
    cough: bool = False
    fever: bool = False
    vomiting: bool = False
    abdominal_pain: bool = False
    difficulty_swallowing: bool = False

# -------------------------
# FEEDBACK MODELS
# -------------------------
class FeedbackInput(BaseModel):
    name: str
    email: EmailStr
    message: str

class Feedback(FeedbackInput):
    id: Optional[str] = None
    created_at: Optional[str] = None
