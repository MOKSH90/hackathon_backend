from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

Mongo_url = "mongodb+srv://mokshgumber:FHs8OceINXPHJNZF@cluster0.imaptd9.mongodb.net/"
client = AsyncIOMotorClient(Mongo_url)
db = client["hospital_db"]

user_collection = db["users"]
patient_collection = db["patients"]
counter_collection = db.patient_id_counter
feedback_collection = db.feedback
bed_collection = db.beds
archive_collection = db.archived_patients
bed_capacity_collection = db.bed_capacity

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

def create_hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expire_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expire_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict, expire_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expire_delta or timedelta(days=7))
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
